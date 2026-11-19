use std::sync::Arc;
use std::{mem, slice};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC,
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0, D3D12_ELEMENTS_LAYOUT_ARRAY,
    D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE, D3D12_RAYTRACING_AABB,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL, D3D12_RAYTRACING_GEOMETRY_AABBS_DESC,
    D3D12_RAYTRACING_GEOMETRY_DESC, D3D12_RAYTRACING_GEOMETRY_DESC_0,
    D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
    D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS, D3D12_RAYTRACING_INSTANCE_DESC,
    D3D12_RAYTRACING_INSTANCE_DESCS_BYTE_ALIGNMENT,
    D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE,
};

use super::{Dx12Buffer, Dx12Device};
use crate::acceleration_structure::{
    AccelerationStructureUsage, NativeAccelerationStructureInstance,
};
use crate::{
    Aabb, AccelerationStructureBuildReport, BufferDescriptor, BufferMemory, BufferUsages, RhiError,
    TriangleGeometry,
};

/// The build inputs of one procedural BLAS: alive until the command list
/// that builds it has completed. The storage belongs to the
/// `Dx12AccelerationStructure` returned beside it.
pub(crate) struct Dx12BlasBuild {
    /// The boxes, or the positions and indices, the geometry points into.
    _inputs: Vec<Dx12Buffer>,
    _scratch: Dx12Buffer,
    geometry: D3D12_RAYTRACING_GEOMETRY_DESC,
    storage_address: u64,
    scratch_address: u64,
}

impl Dx12BlasBuild {
    pub(crate) fn description(&self) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: self.storage_address,
            Inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                NumDescs: 1,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: &raw const self.geometry,
                },
            },
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: self.scratch_address,
        }
    }
}

pub(crate) struct Dx12AccelerationStructure {
    _storage: Dx12Buffer,
    /// Present only on top-level structures built `Updatable`.
    refit: Option<Dx12TlasRefit>,
}

/// What a top-level refit needs beyond the structure itself.
struct Dx12TlasRefit {
    /// Shared GPU-only workspace, ordered by the queued build barriers.
    scratch: Arc<Dx12Buffer>,
    /// Source and destination of the in-place refit.
    storage_address: u64,
    /// Pinned at build time: a refit must present the same instance count.
    instance_count: u32,
    /// The build's flags, which the refit repeats with `PERFORM_UPDATE`
    /// added.
    flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS,
}

/// Build flags for a top-level structure of the requested usage.
///
/// `ALLOW_UPDATE` combines with the fast-trace preference; the mutually
/// exclusive pair is fast-trace against fast-build. The prebuild query and
/// the build must pass identical flags, because `ALLOW_UPDATE` changes the
/// sizes the driver reports.
fn tlas_flags(
    usage: AccelerationStructureUsage,
) -> D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS {
    match usage {
        AccelerationStructureUsage::Static => {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
        }
        AccelerationStructureUsage::Updatable => {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
                | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE
        }
    }
}

/// Lower the neutral instance records to DXR's packed form.
fn instance_descriptors(
    instances: &[NativeAccelerationStructureInstance],
) -> Vec<D3D12_RAYTRACING_INSTANCE_DESC> {
    let flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE.0 as u32;
    instances
        .iter()
        .map(|instance| D3D12_RAYTRACING_INSTANCE_DESC {
            Transform: instance.transform,
            _bitfield1: instance.custom_index | (u32::from(instance.mask) << 24),
            _bitfield2: instance.shader_binding_table_record_offset | (flags << 24),
            AccelerationStructure: instance.blas_address,
        })
        .collect()
}

impl Dx12Device {
    /// Build every set's BLAS on one command list with one wait, in input
    /// order. Chunk streaming builds many small BLASes per frame; one at a
    /// time each paid a full queue wait.
    /// Allocate and describe every set's BLAS without building it.
    pub(crate) fn prepare_procedural_aabb_blas_batch(
        &self,
        sets: &[Vec<Aabb>],
    ) -> Result<
        Vec<(
            Dx12AccelerationStructure,
            AccelerationStructureBuildReport,
            Dx12BlasBuild,
        )>,
        RhiError,
    > {
        let mut prepared = Vec::with_capacity(sets.len());

        for aabbs in sets {
            prepared.push(self.prepare_procedural_aabb_blas(aabbs)?);
        }
        Ok(prepared)
    }

    /// Allocate and describe one procedural BLAS without building it.
    fn prepare_procedural_aabb_blas(
        &self,
        aabbs: &[Aabb],
    ) -> Result<
        (
            Dx12AccelerationStructure,
            AccelerationStructureBuildReport,
            Dx12BlasBuild,
        ),
        RhiError,
    > {
        let dx_aabbs: Vec<D3D12_RAYTRACING_AABB> = aabbs
            .iter()
            .map(|bounds| D3D12_RAYTRACING_AABB {
                MinX: bounds.min[0],
                MinY: bounds.min[1],
                MinZ: bounds.min[2],
                MaxX: bounds.max[0],
                MaxY: bounds.max[1],
                MaxZ: bounds.max[2],
            })
            .collect();
        let aabb_size = mem::size_of::<D3D12_RAYTRACING_AABB>() as u64;
        let input = self.create_buffer(BufferDescriptor {
            size: aabb_size * dx_aabbs.len() as u64,
            usages: BufferUsages::ACCELERATION_STRUCTURE_INPUT
                | BufferUsages::SHADER_DEVICE_ADDRESS,
            memory: BufferMemory::Upload,
        })?;
        // SAFETY: the D3D12 AABBs are repr(C), Copy records with no
        // references; this byte view lasts only for the following upload.
        let bytes = unsafe {
            slice::from_raw_parts(
                dx_aabbs.as_ptr().cast::<u8>(),
                mem::size_of::<D3D12_RAYTRACING_AABB>() * dx_aabbs.len(),
            )
        };
        input.write(0, bytes)?;
        let input_address = input.device_address()?;

        let aabb_geometry = D3D12_RAYTRACING_GEOMETRY_AABBS_DESC {
            AABBCount: dx_aabbs.len() as u64,
            AABBs: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                StartAddress: input_address,
                StrideInBytes: aabb_size,
            },
        };
        let geometry = D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS,
            Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                AABBs: aabb_geometry,
            },
        };
        self.prepare_blas(geometry, vec![input])
    }

    /// Allocate and describe one triangle BLAS without building it.
    pub(crate) fn prepare_triangle_blas(
        &self,
        triangles: &TriangleGeometry<'_>,
        _triangle_count: u32,
    ) -> Result<
        (
            Dx12AccelerationStructure,
            AccelerationStructureBuildReport,
            Dx12BlasBuild,
        ),
        RhiError,
    > {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RAYTRACING_GEOMETRY_FLAG_NONE, D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC,
            D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        };
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_UNKNOWN,
        };
        let input = |bytes: &[u8]| -> Result<Dx12Buffer, RhiError> {
            let buffer = self.create_buffer(BufferDescriptor {
                size: bytes.len() as u64,
                usages: BufferUsages::ACCELERATION_STRUCTURE_INPUT
                    | BufferUsages::SHADER_DEVICE_ADDRESS,
                memory: BufferMemory::Upload,
            })?;
            buffer.write(0, bytes)?;
            Ok(buffer)
        };
        // SAFETY: positions and indices are plain numbers with no padding;
        // each byte view lasts only for the upload beside it.
        let (position_bytes, index_bytes) = unsafe {
            (
                slice::from_raw_parts(
                    triangles.positions.as_ptr().cast::<u8>(),
                    mem::size_of_val(triangles.positions),
                ),
                triangles.indices.map(|indices| {
                    slice::from_raw_parts(indices.as_ptr().cast::<u8>(), mem::size_of_val(indices))
                }),
            )
        };
        let positions = input(position_bytes)?;
        let mut description = D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
            Transform3x4: 0,
            IndexFormat: DXGI_FORMAT_UNKNOWN,
            VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
            IndexCount: 0,
            VertexCount: triangles.positions.len() as u32,
            IndexBuffer: 0,
            VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                StartAddress: positions.device_address()?,
                StrideInBytes: 12,
            },
        };
        let mut inputs = vec![positions];
        if let (Some(indices), Some(bytes)) = (triangles.indices, index_bytes) {
            let buffer = input(bytes)?;
            description.IndexFormat = DXGI_FORMAT_R32_UINT;
            description.IndexCount = indices.len() as u32;
            description.IndexBuffer = buffer.device_address()?;
            inputs.push(buffer);
        }
        let geometry = D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
            Flags: if triangles.opaque {
                D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
            } else {
                D3D12_RAYTRACING_GEOMETRY_FLAG_NONE
            },
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                Triangles: description,
            },
        };
        self.prepare_blas(geometry, inputs)
    }

    /// Allocate the structure and the scratch of one BLAS of `geometry`.
    /// `inputs` are the buffers the geometry points into.
    fn prepare_blas(
        &self,
        geometry: D3D12_RAYTRACING_GEOMETRY_DESC,
        inputs: Vec<Dx12Buffer>,
    ) -> Result<
        (
            Dx12AccelerationStructure,
            AccelerationStructureBuildReport,
            Dx12BlasBuild,
        ),
        RhiError,
    > {
        let description = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                pGeometryDescs: &raw const geometry,
            },
        };
        let mut sizes = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: `description` points at one live geometry and `sizes` is a
        // writable output record of the exact required type.
        unsafe {
            self.native
                .GetRaytracingAccelerationStructurePrebuildInfo(&raw const description, &mut sizes);
        }
        if sizes.ResultDataMaxSizeInBytes == 0 || sizes.ScratchDataSizeInBytes == 0 {
            return Err(RhiError::InvalidAccelerationStructureSizes {
                result: sizes.ResultDataMaxSizeInBytes,
                scratch: sizes.ScratchDataSizeInBytes,
            });
        }

        let storage = self.create_buffer(BufferDescriptor {
            size: sizes.ResultDataMaxSizeInBytes,
            usages: BufferUsages::ACCELERATION_STRUCTURE_STORAGE
                | BufferUsages::SHADER_DEVICE_ADDRESS,
            memory: BufferMemory::Device,
        })?;
        let scratch = self.create_buffer(BufferDescriptor {
            size: sizes.ScratchDataSizeInBytes,
            usages: BufferUsages::STORAGE | BufferUsages::SHADER_DEVICE_ADDRESS,
            memory: BufferMemory::Device,
        })?;
        let storage_address = storage.device_address()?;
        let scratch_address = scratch.device_address()?;
        self.validate_address_alignment(
            "DXR acceleration structure",
            storage_address,
            self.capabilities.scratch_alignment,
        )?;
        self.validate_address_alignment(
            "DXR build scratch",
            scratch_address,
            self.capabilities.scratch_alignment,
        )?;
        Ok((
            Dx12AccelerationStructure {
                _storage: storage,
                refit: None,
            },
            AccelerationStructureBuildReport {
                acceleration_structure_size: sizes.ResultDataMaxSizeInBytes,
                build_scratch_size: sizes.ScratchDataSizeInBytes,
                device_address: storage_address,
            },
            Dx12BlasBuild {
                _inputs: inputs,
                _scratch: scratch,
                geometry,
                storage_address,
                scratch_address,
            },
        ))
    }

    pub(crate) fn prepare_tlas(
        &self,
        usage: AccelerationStructureUsage,
        instances: &[NativeAccelerationStructureInstance],
    ) -> Result<
        (
            Dx12AccelerationStructure,
            AccelerationStructureBuildReport,
            Dx12TlasBuild,
        ),
        RhiError,
    > {
        let build_flags = tlas_flags(usage);
        let descriptors = instance_descriptors(instances);
        let byte_size = mem::size_of_val(descriptors.as_slice()) as u64;
        let input = self.create_buffer(BufferDescriptor {
            size: byte_size,
            usages: BufferUsages::ACCELERATION_STRUCTURE_INPUT
                | BufferUsages::SHADER_DEVICE_ADDRESS,
            memory: BufferMemory::Upload,
        })?;
        // SAFETY: DXR instance descriptors are repr(C), Copy records with no
        // references, and the byte view lasts only for the upload call.
        let bytes =
            unsafe { slice::from_raw_parts(descriptors.as_ptr().cast::<u8>(), byte_size as usize) };
        input.write(0, bytes)?;
        let input_address = input.device_address()?;
        self.validate_address_alignment(
            "DXR instance descriptions",
            input_address,
            D3D12_RAYTRACING_INSTANCE_DESCS_BYTE_ALIGNMENT,
        )?;

        let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: build_flags,
            NumDescs: instances.len() as u32,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                InstanceDescs: input_address,
            },
        };
        let mut sizes = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: the input address names a live, aligned array containing
        // `NumDescs` initialized DXR instance records.
        unsafe {
            self.native
                .GetRaytracingAccelerationStructurePrebuildInfo(&raw const inputs, &mut sizes);
        }
        if sizes.ResultDataMaxSizeInBytes == 0 || sizes.ScratchDataSizeInBytes == 0 {
            return Err(RhiError::InvalidAccelerationStructureSizes {
                result: sizes.ResultDataMaxSizeInBytes,
                scratch: sizes.ScratchDataSizeInBytes,
            });
        }

        let storage = self.create_buffer(BufferDescriptor {
            size: sizes.ResultDataMaxSizeInBytes,
            usages: BufferUsages::ACCELERATION_STRUCTURE_STORAGE
                | BufferUsages::SHADER_DEVICE_ADDRESS,
            memory: BufferMemory::Device,
        })?;
        let scratch = Arc::new(
            self.create_buffer(BufferDescriptor {
                size: sizes
                    .ScratchDataSizeInBytes
                    .max(sizes.UpdateScratchDataSizeInBytes),
                usages: BufferUsages::STORAGE | BufferUsages::SHADER_DEVICE_ADDRESS,
                memory: BufferMemory::Device,
            })?,
        );
        let storage_address = storage.device_address()?;
        let scratch_address = scratch.device_address()?;
        self.validate_address_alignment(
            "DXR top-level acceleration structure",
            storage_address,
            self.capabilities.scratch_alignment,
        )?;
        self.validate_address_alignment(
            "DXR TLAS build scratch",
            scratch_address,
            self.capabilities.scratch_alignment,
        )?;

        let build = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: storage_address,
            Inputs: inputs,
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: scratch_address,
        };
        let refit = match usage {
            AccelerationStructureUsage::Static => None,
            AccelerationStructureUsage::Updatable => Some(Dx12TlasRefit {
                scratch: scratch.clone(),
                storage_address,
                instance_count: instances.len() as u32,
                flags: build_flags,
            }),
        };
        if let Some(refit) = &refit {
            self.validate_address_alignment(
                "DXR TLAS refit scratch",
                refit.scratch.device_address()?,
                self.capabilities.scratch_alignment,
            )?;
        }

        let report = AccelerationStructureBuildReport {
            acceleration_structure_size: sizes.ResultDataMaxSizeInBytes,
            build_scratch_size: sizes.ScratchDataSizeInBytes,
            device_address: storage_address,
        };
        Ok((
            Dx12AccelerationStructure {
                _storage: storage,
                refit,
            },
            report,
            Dx12TlasBuild {
                destination: build.DestAccelerationStructureData,
                source: build.SourceAccelerationStructureData,
                scratch_address: build.ScratchAccelerationStructureData,
                instance_count: build.Inputs.NumDescs,
                flags: build.Inputs.Flags,
                _input: input,
                _scratch: scratch,
            },
        ))
    }

    pub(crate) fn prepare_tlas_update(
        &self,
        acceleration: &Dx12AccelerationStructure,
        instances: &[NativeAccelerationStructureInstance],
    ) -> Result<Dx12TlasBuild, RhiError> {
        let refit = acceleration
            .refit
            .as_ref()
            .ok_or(RhiError::AccelerationStructureNotUpdatable)?;
        if instances.len() as u32 != refit.instance_count {
            return Err(RhiError::AccelerationStructureShapeChanged(
                "the number of instances",
            ));
        }
        let descriptors = instance_descriptors(instances);
        let byte_size = mem::size_of_val(descriptors.as_slice());
        // SAFETY: DXR instance descriptors are repr(C), Copy records with no
        // references, and the byte view lasts only for the upload call.
        let bytes = unsafe { slice::from_raw_parts(descriptors.as_ptr().cast::<u8>(), byte_size) };
        let input = self.create_buffer(BufferDescriptor {
            size: byte_size as u64,
            usages: BufferUsages::ACCELERATION_STRUCTURE_INPUT
                | BufferUsages::SHADER_DEVICE_ADDRESS,
            memory: BufferMemory::Upload,
        })?;
        input.write(0, bytes)?;
        self.validate_address_alignment(
            "DXR instance descriptions",
            input.device_address()?,
            D3D12_RAYTRACING_INSTANCE_DESCS_BYTE_ALIGNMENT,
        )?;

        let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            // The build's flags plus `PERFORM_UPDATE`, which selects the
            // refit; dropping `ALLOW_UPDATE` here would make the result
            // un-refittable next frame.
            Flags: refit.flags | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE,
            NumDescs: refit.instance_count,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                InstanceDescs: input.device_address()?,
            },
        };
        let update = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: refit.storage_address,
            Inputs: inputs,
            SourceAccelerationStructureData: refit.storage_address,
            ScratchAccelerationStructureData: refit.scratch.device_address()?,
        };
        Ok(Dx12TlasBuild {
            destination: update.DestAccelerationStructureData,
            source: update.SourceAccelerationStructureData,
            scratch_address: update.ScratchAccelerationStructureData,
            instance_count: update.Inputs.NumDescs,
            flags: update.Inputs.Flags,
            _input: input,
            _scratch: refit.scratch.clone(),
        })
    }

    fn validate_address_alignment(
        &self,
        resource: &'static str,
        address: u64,
        required: u32,
    ) -> Result<(), RhiError> {
        if address.is_multiple_of(u64::from(required)) {
            Ok(())
        } else {
            Err(RhiError::DeviceAddressAlignment {
                resource,
                address,
                required,
            })
        }
    }
}

pub(crate) struct Dx12TlasBuild {
    destination: u64,
    source: u64,
    scratch_address: u64,
    instance_count: u32,
    flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS,
    _input: Dx12Buffer,
    _scratch: Arc<Dx12Buffer>,
}

impl Dx12TlasBuild {
    pub(crate) fn description(
        &self,
    ) -> Result<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC, RhiError> {
        Ok(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: self.destination,
            SourceAccelerationStructureData: self.source,
            ScratchAccelerationStructureData: self.scratch_address,
            Inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                Flags: self.flags,
                NumDescs: self.instance_count,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    InstanceDescs: self._input.device_address()?,
                },
            },
        })
    }
}
