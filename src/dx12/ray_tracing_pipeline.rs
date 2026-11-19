use std::ffi::c_void;
use std::mem::ManuallyDrop;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_DXIL_LIBRARY_DESC, D3D12_GLOBAL_ROOT_SIGNATURE, D3D12_GPU_VIRTUAL_ADDRESS_RANGE,
    D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE, D3D12_HIT_GROUP_DESC,
    D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE, D3D12_RAYTRACING_PIPELINE_CONFIG,
    D3D12_RAYTRACING_SHADER_CONFIG, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT,
    D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT, D3D12_SHADER_BYTECODE,
    D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_STATE_OBJECT_DESC,
    D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, D3D12_STATE_SUBOBJECT,
    D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE,
    D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG,
    D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, ID3D12StateObject,
    ID3D12StateObjectProperties,
};
use windows::core::{Interface, PCWSTR};

use super::{Dx12Buffer, Dx12Device, Dx12PipelineLayout, Dx12ShaderModule};
use crate::ray_tracing_pipeline::{RayTracingPipelineSettings, compute_sbt_layout};
use crate::{BufferDescriptor, BufferMemory, BufferUsages, RhiError};

/// One owned DXR state object with its upload-heap shader binding table and
/// precomputed dispatch ranges.
pub(crate) struct Dx12RayTracingPipeline {
    pub(crate) state_object: ID3D12StateObject,
    #[allow(dead_code)] // Keeps the SBT resource alive for the ranges below.
    sbt: Dx12Buffer,
    pub(crate) raygen_range: D3D12_GPU_VIRTUAL_ADDRESS_RANGE,
    pub(crate) miss_range: D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE,
    pub(crate) hit_range: D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE,
}

/// NUL-terminated UTF-16 storage that outlives every PCWSTR into it.
fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

impl Dx12Device {
    pub(crate) fn create_ray_tracing_pipeline(
        &self,
        descriptor: &RayTracingPipelineSettings<'_, Dx12ShaderModule>,
        layout: &Dx12PipelineLayout,
    ) -> Result<Dx12RayTracingPipeline, RhiError> {
        // Every name buffer must stay alive through CreateStateObject.
        let raygen_name = wide(descriptor.ray_generation.entry);
        let miss_names: Vec<Vec<u16>> = descriptor
            .miss_entries
            .iter()
            .map(|stage| wide(stage.entry))
            .collect();
        let group_names: Vec<Vec<u16>> = descriptor
            .hit_groups
            .iter()
            .map(|group| wide(group.name))
            .collect();
        let intersection_names: Vec<Option<Vec<u16>>> = descriptor
            .hit_groups
            .iter()
            .map(|group| group.intersection.as_ref().map(|stage| wide(stage.entry)))
            .collect();
        let any_hit_names: Vec<Option<Vec<u16>>> = descriptor
            .hit_groups
            .iter()
            .map(|group| group.any_hit.as_ref().map(|stage| wide(stage.entry)))
            .collect();
        let closest_hit_names: Vec<Option<Vec<u16>>> = descriptor
            .hit_groups
            .iter()
            .map(|group| group.closest_hit.as_ref().map(|stage| wide(stage.entry)))
            .collect();

        // Each module the stages come from, once: a library of the state
        // object.
        let mut modules: Vec<&Dx12ShaderModule> = Vec::new();
        let stages = std::iter::once(&descriptor.ray_generation)
            .chain(&descriptor.miss_entries)
            .chain(descriptor.hit_groups.iter().flat_map(|group| {
                group
                    .intersection
                    .iter()
                    .chain(&group.closest_hit)
                    .chain(&group.any_hit)
            }));
        for stage in stages {
            if !modules.iter().any(|held| std::ptr::eq(*held, stage.module)) {
                modules.push(stage.module);
            }
        }
        let libraries: Vec<D3D12_DXIL_LIBRARY_DESC> = modules
            .iter()
            .map(|module| D3D12_DXIL_LIBRARY_DESC {
                DXILLibrary: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: module.bytes.as_ptr().cast::<c_void>(),
                    BytecodeLength: module.bytes.len(),
                },
                // Zero exports means every entry in the library is exported.
                NumExports: 0,
                pExports: std::ptr::null_mut(),
            })
            .collect();
        // The name of an entry a group has, or no name at all.
        let import = |name: &Option<Vec<u16>>| {
            name.as_ref()
                .map_or(PCWSTR::null(), |name| PCWSTR::from_raw(name.as_ptr()))
        };
        let hit_groups: Vec<D3D12_HIT_GROUP_DESC> = (0..descriptor.hit_groups.len())
            .map(|index| D3D12_HIT_GROUP_DESC {
                HitGroupExport: PCWSTR::from_raw(group_names[index].as_ptr()),
                // A group with no intersection shader is for triangles.
                Type: if intersection_names[index].is_some() {
                    D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE
                } else {
                    windows::Win32::Graphics::Direct3D12::D3D12_HIT_GROUP_TYPE_TRIANGLES
                },
                AnyHitShaderImport: import(&any_hit_names[index]),
                ClosestHitShaderImport: import(&closest_hit_names[index]),
                IntersectionShaderImport: import(&intersection_names[index]),
            })
            .collect();
        let shader_config = D3D12_RAYTRACING_SHADER_CONFIG {
            MaxPayloadSizeInBytes: descriptor.max_payload_size,
            MaxAttributeSizeInBytes: descriptor.max_attribute_size,
        };
        let global_root_signature = D3D12_GLOBAL_ROOT_SIGNATURE {
            pGlobalRootSignature: ManuallyDrop::new(Some(layout.root_signature().clone())),
        };
        let pipeline_config = D3D12_RAYTRACING_PIPELINE_CONFIG {
            MaxTraceRecursionDepth: descriptor.max_recursion_depth,
        };

        let mut subobjects = Vec::with_capacity(3 + libraries.len() + hit_groups.len());
        for library in &libraries {
            subobjects.push(D3D12_STATE_SUBOBJECT {
                Type: D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY,
                pDesc: std::ptr::from_ref(library).cast::<c_void>(),
            });
        }
        for hit_group in &hit_groups {
            subobjects.push(D3D12_STATE_SUBOBJECT {
                Type: D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP,
                pDesc: std::ptr::from_ref(hit_group).cast::<c_void>(),
            });
        }
        subobjects.push(D3D12_STATE_SUBOBJECT {
            Type: D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG,
            pDesc: (&raw const shader_config).cast::<c_void>(),
        });
        subobjects.push(D3D12_STATE_SUBOBJECT {
            Type: D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE,
            pDesc: (&raw const global_root_signature).cast::<c_void>(),
        });
        subobjects.push(D3D12_STATE_SUBOBJECT {
            Type: D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG,
            pDesc: (&raw const pipeline_config).cast::<c_void>(),
        });
        let state_object_description = D3D12_STATE_OBJECT_DESC {
            Type: D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
            NumSubobjects: subobjects.len() as u32,
            pSubobjects: subobjects.as_ptr(),
        };
        // SAFETY: every subobject pointer targets a local that lives through
        // this call, including all wide-string name buffers.
        let created: Result<ID3D12StateObject, _> = unsafe {
            self.native
                .CreateStateObject(&raw const state_object_description)
        };
        // Release the root-signature reference cloned into the subobject.
        let _ = ManuallyDrop::into_inner(global_root_signature.pGlobalRootSignature);
        let state_object =
            created.map_err(|error| RhiError::dx12("ID3D12Device5::CreateStateObject", error))?;

        let properties: ID3D12StateObjectProperties = state_object
            .cast()
            .map_err(|error| RhiError::dx12("ID3D12StateObjectProperties cast", error))?;
        let identifier = |name: &[u16], entry: &str| -> Result<[u8; 32], RhiError> {
            // SAFETY: the wide name is NUL-terminated and the properties
            // interface is live; a null result means the export is missing.
            let pointer =
                unsafe { properties.GetShaderIdentifier(PCWSTR::from_raw(name.as_ptr())) };
            if pointer.is_null() {
                return Err(RhiError::MissingShaderExport {
                    entry: entry.to_owned(),
                });
            }
            let mut bytes = [0_u8; D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES as usize];
            // SAFETY: DXR guarantees identifier blobs are exactly 32 bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pointer.cast::<u8>(),
                    bytes.as_mut_ptr(),
                    bytes.len(),
                );
            }
            Ok(bytes)
        };

        let raygen_identifier = identifier(&raygen_name, descriptor.ray_generation.entry)?;
        let miss_identifiers: Vec<[u8; 32]> = miss_names
            .iter()
            .zip(&descriptor.miss_entries)
            .map(|(name, stage)| identifier(name, stage.entry))
            .collect::<Result<_, _>>()?;
        let hit_identifiers: Vec<[u8; 32]> = group_names
            .iter()
            .zip(&descriptor.hit_groups)
            .map(|(name, group)| identifier(name, group.name))
            .collect::<Result<_, _>>()?;

        let table = compute_sbt_layout(
            D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES,
            D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT,
            D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT,
            descriptor.miss_entries.len() as u64,
            descriptor.hit_groups.len() as u64,
        );
        let sbt = self.create_buffer(BufferDescriptor {
            size: table.total_size,
            usages: BufferUsages::COPY_SOURCE,
            memory: BufferMemory::Upload,
        })?;
        sbt.write(table.raygen_offset, &raygen_identifier)?;
        for (index, identifier_bytes) in miss_identifiers.iter().enumerate() {
            sbt.write(
                table.miss_offset + index as u64 * table.miss_stride,
                identifier_bytes,
            )?;
        }
        for (index, identifier_bytes) in hit_identifiers.iter().enumerate() {
            sbt.write(
                table.hit_offset + index as u64 * table.hit_stride,
                identifier_bytes,
            )?;
        }
        let address = sbt.device_address()?;
        if !address.is_multiple_of(u64::from(D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT)) {
            return Err(RhiError::ShaderBindingTableAlignment {
                address,
                required: u64::from(D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT),
            });
        }

        super::name_object(&state_object, descriptor.label);
        Ok(Dx12RayTracingPipeline {
            state_object,
            raygen_range: D3D12_GPU_VIRTUAL_ADDRESS_RANGE {
                StartAddress: address + table.raygen_offset,
                SizeInBytes: table.raygen_size,
            },
            miss_range: D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE {
                StartAddress: address + table.miss_offset,
                SizeInBytes: table.miss_size,
                StrideInBytes: table.miss_stride,
            },
            hit_range: D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE {
                StartAddress: address + table.hit_offset,
                SizeInBytes: table.hit_size,
                StrideInBytes: table.hit_stride,
            },
            sbt,
        })
    }
}
