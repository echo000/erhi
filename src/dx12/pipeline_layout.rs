use std::slice;

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION_1_0, D3D12_COMPARISON_FUNC_NEVER, D3D12_DESCRIPTOR_RANGE,
    D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND, D3D12_DESCRIPTOR_RANGE_TYPE,
    D3D12_DESCRIPTOR_RANGE_TYPE_CBV, D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
    D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
    D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_ROOT_CONSTANTS,
    D3D12_ROOT_DESCRIPTOR_TABLE, D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0,
    D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
    D3D12_ROOT_SIGNATURE_DESC, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    D3D12_SHADER_VISIBILITY_ALL, D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
    D3D12_STATIC_SAMPLER_DESC, D3D12SerializeRootSignature, ID3D12RootSignature,
};

use super::Dx12Device;
use crate::pipeline_layout::{FilterMode, StaticSampler};
use crate::{DescriptorType, PipelineLayoutDescriptor, RhiError};

pub(crate) struct Dx12PipelineLayout {
    root_signature: ID3D12RootSignature,
    /// Root-parameter index of each set's sequential resource descriptor
    /// table: none for a set that declares no resource.
    resource_table_parameters: Vec<Option<u32>>,
    /// Root-parameter index of each set's table of `Sampler` bindings.
    sampler_table_parameters: Vec<Option<u32>>,
    /// Root-parameter index of the 32-bit root-constant block, with its
    /// declared size in 32-bit words.
    push_constant_parameter: Option<(u32, u32)>,
}

impl Dx12PipelineLayout {
    pub(crate) fn root_signature(&self) -> &ID3D12RootSignature {
        &self.root_signature
    }

    pub(crate) fn resource_table_parameter(&self, set: usize) -> Option<u32> {
        self.resource_table_parameters.get(set).copied().flatten()
    }

    pub(crate) fn sampler_table_parameter(&self, set: usize) -> Option<u32> {
        self.sampler_table_parameters.get(set).copied().flatten()
    }

    pub(crate) fn push_constant_parameter(&self) -> Option<(u32, u32)> {
        self.push_constant_parameter
    }
}

impl Dx12Device {
    pub(crate) fn create_pipeline_layout(
        &self,
        descriptor: PipelineLayoutDescriptor<'_>,
    ) -> Result<Dx12PipelineLayout, RhiError> {
        // The ranges of each set: its resources, and its samplers.
        let mut ranges: Vec<(Vec<D3D12_DESCRIPTOR_RANGE>, Vec<D3D12_DESCRIPTOR_RANGE>)> =
            Vec::new();
        let mut static_samplers = Vec::new();
        for set in descriptor.sets {
            let mut resource_ranges = Vec::new();
            let mut sampler_ranges = Vec::new();
            // Combined image samplers surface as SRVs at their `t` register
            // plus a static sampler whose `s` register is the binding's
            // ordinal among the set's combined bindings in declaration
            // order, which matches how Slang lowers `Sampler2D` to DXIL.
            let mut combined_ordinal = 0;
            for binding in set.bindings {
                let range = D3D12_DESCRIPTOR_RANGE {
                    RangeType: dx12_range_type(binding.descriptor_type),
                    NumDescriptors: binding.count,
                    // Validated: no binding is below its offset.
                    BaseShaderRegister: set
                        .register_offsets
                        .register(binding)
                        .unwrap_or(binding.binding),
                    RegisterSpace: set.set,
                    OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                };
                match binding.descriptor_type {
                    DescriptorType::Sampler => sampler_ranges.push(range),
                    DescriptorType::CombinedImageSampler(sampler) => {
                        resource_ranges.push(range);
                        static_samplers.push(static_sampler_description(
                            sampler,
                            combined_ordinal,
                            set.set,
                        ));
                        combined_ordinal += 1;
                    }
                    _ => resource_ranges.push(range),
                }
            }
            ranges.push((resource_ranges, sampler_ranges));
        }

        // A table for each set's resources and one for its samplers, in the
        // order of the sets.
        let mut parameters = Vec::with_capacity(ranges.len() * 2 + 1);
        let mut resource_table_parameters = Vec::with_capacity(ranges.len());
        let mut sampler_table_parameters = Vec::with_capacity(ranges.len());
        for (resource_ranges, sampler_ranges) in &ranges {
            let mut table = |ranges: &Vec<D3D12_DESCRIPTOR_RANGE>| {
                if ranges.is_empty() {
                    return None;
                }
                parameters.push(descriptor_table_parameter(ranges));
                Some(parameters.len() as u32 - 1)
            };
            resource_table_parameters.push(table(resource_ranges));
            sampler_table_parameters.push(table(sampler_ranges));
        }
        let mut push_constant_parameter = None;
        if let Some(push) = descriptor.push_constants {
            let words = push.size / 4;
            push_constant_parameter = Some((parameters.len() as u32, words));
            parameters.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: push.register,
                        RegisterSpace: push.space,
                        Num32BitValues: words,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
        }
        let root = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: parameters.len() as u32,
            pParameters: parameters.as_ptr(),
            NumStaticSamplers: static_samplers.len() as u32,
            pStaticSamplers: if static_samplers.is_empty() {
                std::ptr::null()
            } else {
                static_samplers.as_ptr()
            },
            // Graphics pipelines built on this layout declare vertex input
            // layouts; the flag is meaningless to compute and DXR pipelines.
            Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        };
        let mut signature = None;
        let mut errors = None;
        // SAFETY: all root parameters point into range vectors that remain
        // live and unchanged throughout this immediate serialization call.
        if let Err(error) = unsafe {
            D3D12SerializeRootSignature(
                &raw const root,
                D3D_ROOT_SIGNATURE_VERSION_1_0,
                &mut signature,
                Some(&mut errors),
            )
        } {
            let message = errors
                .as_ref()
                .map(blob_text)
                .filter(|message| !message.is_empty())
                .unwrap_or_else(|| error.to_string());
            return Err(RhiError::Dx12RootSignatureSerialization {
                code: error.code().0,
                message,
            });
        }
        let signature = signature.ok_or_else(|| RhiError::Dx12 {
            operation: "D3D12SerializeRootSignature returned no blob",
            code: 0,
            message: "successful call returned a null blob".to_owned(),
        })?;
        let bytes = blob_bytes(&signature);
        // SAFETY: the serialized blob is live and contains a root signature
        // accepted by the serializer on this machine.
        let root_signature: ID3D12RootSignature = unsafe {
            self.native
                .CreateRootSignature(0, bytes)
                .map_err(|error| RhiError::dx12("ID3D12Device::CreateRootSignature", error))?
        };
        Ok(Dx12PipelineLayout {
            root_signature,
            resource_table_parameters,
            sampler_table_parameters,
            push_constant_parameter,
        })
    }
}

fn static_sampler_description(
    sampler: StaticSampler,
    register: u32,
    space: u32,
) -> D3D12_STATIC_SAMPLER_DESC {
    let address = super::sampler::dx12_address_mode(sampler.address_mode);
    D3D12_STATIC_SAMPLER_DESC {
        Filter: match sampler.filter {
            FilterMode::Nearest => D3D12_FILTER_MIN_MAG_MIP_POINT,
            FilterMode::Linear => D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        },
        AddressU: address,
        AddressV: address,
        AddressW: address,
        MipLODBias: 0.0,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        BorderColor: D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ShaderRegister: register,
        RegisterSpace: space,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

fn descriptor_table_parameter(ranges: &[D3D12_DESCRIPTOR_RANGE]) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: ranges.len() as u32,
                pDescriptorRanges: ranges.as_ptr(),
            },
        },
        // DXR root signatures have no per-ray-stage visibility enum. The
        // common visibility remains useful to Vulkan and shader validation;
        // DX12 correctly exposes these global tables to every DXR stage.
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

fn dx12_range_type(descriptor_type: DescriptorType) -> D3D12_DESCRIPTOR_RANGE_TYPE {
    match descriptor_type {
        DescriptorType::AccelerationStructure
        | DescriptorType::ReadOnlyStorageBuffer
        | DescriptorType::SampledImage
        | DescriptorType::CombinedImageSampler(_) => D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        DescriptorType::StorageImage | DescriptorType::StorageBuffer => {
            D3D12_DESCRIPTOR_RANGE_TYPE_UAV
        }
        DescriptorType::UniformBuffer => D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
        DescriptorType::Sampler => D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
    }
}

pub(crate) fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: the returned pointer and size belong to `blob` and remain valid
    // for the borrow of that COM object.
    unsafe { slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()) }
}

pub(crate) fn blob_text(blob: &ID3DBlob) -> String {
    String::from_utf8_lossy(blob_bytes(blob))
        .trim_end_matches('\0')
        .trim()
        .to_owned()
}
