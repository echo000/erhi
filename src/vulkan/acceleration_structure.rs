//! Vulkan acceleration-structure builds: the procedural AABB BLAS and the

//! instance TLAS behind the common vocabulary.

use std::sync::Arc;

use ash::vk;

use super::{BufferAllocation, VulkanDevice};
use crate::acceleration_structure::{
    AccelerationStructureUsage, NativeAccelerationStructureInstance,
};
use crate::{Aabb, AccelerationStructureBuildReport, RhiError, TriangleGeometry};

/// Build flags for a top-level structure of the requested usage.
///
/// `ALLOW_UPDATE` rides alongside the fast-trace preference rather than
/// replacing it; the pair that cannot coexist is fast-trace against
/// fast-build. The size query and every later build must pass the identical
/// flags, which is why this is one function and not a literal at each site:
/// `ALLOW_UPDATE` changes the sizes the driver reports.
fn tlas_flags(usage: AccelerationStructureUsage) -> vk::BuildAccelerationStructureFlagsKHR {
    match usage {
        AccelerationStructureUsage::Static => {
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
        }
        AccelerationStructureUsage::Updatable => {
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE
        }
    }
}

/// Lower the neutral instance records to Vulkan's packed form.
fn vk_instances(
    instances: &[NativeAccelerationStructureInstance],
) -> Vec<vk::AccelerationStructureInstanceKHR> {
    // (The instance flags are the low eight bits.)
    let flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8;
    instances
        .iter()
        .map(|instance| vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: instance.transform,
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(
                instance.custom_index,
                instance.mask,
            ),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                instance.shader_binding_table_record_offset,
                flags,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: instance.blas_address,
            },
        })
        .collect()
}

impl VulkanDevice {
    /// Allocate and describe every set's BLAS without building it. The
    /// device address is known at creation, so the structure can be handed
    /// out before the recorded build has run.
    pub(crate) fn prepare_procedural_aabb_blas_batch(
        &self,
        sets: &[Vec<Aabb>],
    ) -> Result<
        Vec<(
            BuiltAcceleration,
            AccelerationStructureBuildReport,
            VulkanBlasBuild,
        )>,
        RhiError,
    > {
        let mut prepared = Vec::with_capacity(sets.len());

        for aabbs in sets {
            prepared.push(self.prepare_procedural_aabb_blas(aabbs)?);
        }
        Ok(prepared)
    }

    /// Record the builds of `builds` into `command`, in order, with the
    /// same barriers as a queued TLAS build so later traces and TLAS builds
    /// in the same submission see the finished structures.
    pub(crate) fn record_blas_batch(
        &self,
        command: vk::CommandBuffer,
        builds: &[&VulkanBlasBuild],
    ) {
        if builds.is_empty() {
            return;
        }
        let ranges: Vec<[vk::AccelerationStructureBuildRangeInfoKHR; 1]> = builds
            .iter()
            .map(|build| {
                [vk::AccelerationStructureBuildRangeInfoKHR::default()
                    .primitive_count(build.primitive_count)]
            })
            .collect();
        let range_slices: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> =
            ranges.iter().map(|range| &range[..]).collect();
        let infos: Vec<_> = builds
            .iter()
            .map(|build| {
                vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .dst_acceleration_structure(build.destination)
                    .geometries(&build.geometries)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: build.scratch_address,
                    })
            })
            .collect();
        let accesses = vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
            | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR;
        let stages = vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
            | vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR;
        let before = vk::MemoryBarrier::default()
            .src_access_mask(accesses)
            .dst_access_mask(accesses);
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_access_mask(accesses);
        // SAFETY: the caller keeps every build's buffers and destination
        // alive until the command buffer has executed; the barriers order
        // the builds after prior traces and before later builds or traces.
        unsafe {
            self.logical.handle.cmd_pipeline_barrier(
                command,
                stages,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            self.acceleration_structure
                .cmd_build_acceleration_structures(command, &infos, &range_slices);
            self.logical.handle.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                stages,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
    }

    /// Allocate and describe one procedural BLAS without building it.
    fn prepare_procedural_aabb_blas(
        &self,
        aabbs: &[Aabb],
    ) -> Result<
        (
            BuiltAcceleration,
            AccelerationStructureBuildReport,
            VulkanBlasBuild,
        ),
        RhiError,
    > {
        // Revalidate values constructed outside `Aabb::new` (notably constants).
        let mut vk_bounds = Vec::with_capacity(aabbs.len());
        for bounds in aabbs {
            let bounds = Aabb::new(bounds.min, bounds.max)?;
            vk_bounds.push(vk::AabbPositionsKHR {
                min_x: bounds.min[0],
                min_y: bounds.min[1],
                min_z: bounds.min[2],
                max_x: bounds.max[0],
                max_y: bounds.max[1],
                max_z: bounds.max[2],
            });
        }
        let primitive_count = vk_bounds.len() as u32;
        let aabb_input = self.allocate_buffer(
            (std::mem::size_of::<vk::AabbPositionsKHR>() * vk_bounds.len()) as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            true,
        )?;
        aabb_input.write_slice(&vk_bounds)?;
        let aabb_address = aabb_input.device_address()?;

        let aabb_data = vk::AccelerationStructureGeometryAabbsDataKHR::default()
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: aabb_address,
            })
            .stride(std::mem::size_of::<vk::AabbPositionsKHR>() as u64);
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::AABBS)
            .geometry(vk::AccelerationStructureGeometryDataKHR { aabbs: aabb_data })
            .flags(vk::GeometryFlagsKHR::OPAQUE);
        self.prepare_blas(geometry, primitive_count, vec![aabb_input])
    }

    /// Allocate and describe one triangle BLAS without building it.
    pub(crate) fn prepare_triangle_blas(
        &self,
        triangles: &TriangleGeometry<'_>,
        triangle_count: u32,
    ) -> Result<
        (
            BuiltAcceleration,
            AccelerationStructureBuildReport,
            VulkanBlasBuild,
        ),
        RhiError,
    > {
        let input = |bytes: u64| {
            self.allocate_buffer(
                bytes,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                true,
            )
        };
        let positions = input(std::mem::size_of_val(triangles.positions) as u64)?;
        positions.write_slice(triangles.positions)?;
        let mut data = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: positions.device_address()?,
            })
            .vertex_stride(12)
            .max_vertex(triangles.positions.len() as u32 - 1)
            .index_type(vk::IndexType::NONE_KHR);
        let mut inputs = vec![positions];
        if let Some(indices) = triangles.indices {
            let buffer = input(std::mem::size_of_val(indices) as u64)?;
            buffer.write_slice(indices)?;
            data = data.index_type(vk::IndexType::UINT32).index_data(
                vk::DeviceOrHostAddressConstKHR {
                    device_address: buffer.device_address()?,
                },
            );
            inputs.push(buffer);
        }
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: data })
            .flags(if triangles.opaque {
                vk::GeometryFlagsKHR::OPAQUE
            } else {
                vk::GeometryFlagsKHR::empty()
            });
        self.prepare_blas(geometry, triangle_count, inputs)
    }

    /// Allocate the structure and the scratch of one BLAS of `geometry`.
    /// `inputs` are the buffers the geometry points into.
    fn prepare_blas(
        &self,
        geometry: vk::AccelerationStructureGeometryKHR<'static>,
        primitive_count: u32,
        inputs: Vec<BufferAllocation>,
    ) -> Result<
        (
            BuiltAcceleration,
            AccelerationStructureBuildReport,
            VulkanBlasBuild,
        ),
        RhiError,
    > {
        let geometries = [geometry];
        let size_query = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: `size_query` names one valid geometry and `sizes` is a
        // writable output structure for the duration of the call.
        unsafe {
            self.acceleration_structure
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &size_query,
                    &[primitive_count],
                    &mut sizes,
                );
        }

        let storage = self.allocate_buffer(
            sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?;
        let acceleration_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(storage.handle)
            .size(sizes.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
        // SAFETY: `storage` is large enough according to the immediately
        // preceding device query and remains alive inside `BuiltAcceleration`.
        let acceleration_handle = unsafe {
            self.acceleration_structure
                .create_acceleration_structure(&acceleration_info, None)
        }
        .map_err(|result| RhiError::vk("vkCreateAccelerationStructureKHR", result))?;
        let acceleration = BuiltAcceleration {
            loader: self.acceleration_structure.clone(),
            handle: acceleration_handle,
            _storage: storage,
            refit: None,
        };
        let scratch = self.allocate_buffer(
            sizes.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?;
        let scratch_address = scratch.device_address()?;
        let address_info = vk::AccelerationStructureDeviceAddressInfoKHR::default()
            .acceleration_structure(acceleration.handle);
        // SAFETY: the acceleration structure is live; its address does not
        // depend on the build having run.
        let device_address = unsafe {
            self.acceleration_structure
                .get_acceleration_structure_device_address(&address_info)
        };
        if device_address == 0 {
            return Err(RhiError::ZeroDeviceAddress(
                "bottom-level acceleration structure",
            ));
        }
        Ok((
            acceleration,
            AccelerationStructureBuildReport {
                acceleration_structure_size: sizes.acceleration_structure_size,
                build_scratch_size: sizes.build_scratch_size,
                device_address,
            },
            VulkanBlasBuild {
                _inputs: inputs,
                _scratch: scratch,
                geometries,
                primitive_count,
                scratch_address,
                destination: acceleration_handle,
            },
        ))
    }

    pub(crate) fn prepare_tlas(
        &self,
        usage: AccelerationStructureUsage,
        instances: &[NativeAccelerationStructureInstance],
    ) -> Result<
        (
            BuiltAcceleration,
            AccelerationStructureBuildReport,
            VulkanTlasBuild,
        ),
        RhiError,
    > {
        let flags = tlas_flags(usage);
        let vk_instances = vk_instances(instances);
        let input = self.allocate_buffer(
            std::mem::size_of_val(vk_instances.as_slice()) as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            true,
        )?;
        input.write_slice(&vk_instances)?;

        let instance_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: input.device_address()?,
            });
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: instance_data,
            });
        let geometries = [geometry];
        let instance_count = instances.len() as u32;
        let query = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: the query references one live instance-array geometry.
        unsafe {
            self.acceleration_structure
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &query,
                    &[instance_count],
                    &mut sizes,
                );
        }
        if sizes.acceleration_structure_size == 0 || sizes.build_scratch_size == 0 {
            return Err(RhiError::InvalidAccelerationStructureSizes {
                result: sizes.acceleration_structure_size,
                scratch: sizes.build_scratch_size,
            });
        }

        let storage = self.allocate_buffer(
            sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?;
        let create = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(storage.handle)
            .size(sizes.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
        // SAFETY: storage was allocated from this device for the queried size.
        let handle = unsafe {
            self.acceleration_structure
                .create_acceleration_structure(&create, None)
        }
        .map_err(|result| RhiError::vk("vkCreateAccelerationStructureKHR(TLAS)", result))?;
        // Establish the handle owner before any further fallible allocation.
        let mut acceleration = BuiltAcceleration {
            loader: self.acceleration_structure.clone(),
            handle,
            _storage: storage,
            refit: None,
        };
        let scratch = Arc::new(self.allocate_buffer(
            sizes.build_scratch_size.max(sizes.update_scratch_size),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?);
        if usage == AccelerationStructureUsage::Updatable {
            acceleration.refit = Some(TlasRefit {
                scratch: scratch.clone(),
                instance_count,
                flags,
            });
        }

        let address_info = vk::AccelerationStructureDeviceAddressInfoKHR::default()
            .acceleration_structure(acceleration.handle);
        // SAFETY: the handle and its storage exist. Querying its address does not
        // read built contents; recording/submission establish that dependency.
        let device_address = unsafe {
            self.acceleration_structure
                .get_acceleration_structure_device_address(&address_info)
        };
        if device_address == 0 {
            return Err(RhiError::ZeroDeviceAddress(
                "top-level acceleration structure",
            ));
        }
        let report = AccelerationStructureBuildReport {
            acceleration_structure_size: sizes.acceleration_structure_size,
            build_scratch_size: sizes.build_scratch_size,
            device_address,
        };
        let build = VulkanTlasBuild {
            input,
            scratch,
            destination: acceleration.handle,
            instance_count,
            flags,
            update: false,
        };
        Ok((acceleration, report, build))
    }

    pub(crate) fn prepare_tlas_update(
        &self,
        acceleration: &BuiltAcceleration,
        instances: &[NativeAccelerationStructureInstance],
    ) -> Result<VulkanTlasBuild, RhiError> {
        let refit = acceleration
            .refit
            .as_ref()
            .ok_or(RhiError::AccelerationStructureNotUpdatable)?;
        if instances.len() as u32 != refit.instance_count {
            return Err(RhiError::AccelerationStructureShapeChanged(
                "the number of instances",
            ));
        }
        let records = vk_instances(instances);
        let input = self.allocate_buffer(
            std::mem::size_of_val(records.as_slice()) as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            true,
        )?;
        input.write_slice(&records)?;
        Ok(VulkanTlasBuild {
            input,
            scratch: refit.scratch.clone(),
            destination: acceleration.handle,
            instance_count: refit.instance_count,
            flags: refit.flags,
            update: true,
        })
    }

    pub(crate) fn record_tlas(
        &self,
        command: vk::CommandBuffer,
        work: &VulkanTlasBuild,
    ) -> Result<(), RhiError> {
        let instance_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: work.input.device_address()?,
            });
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: instance_data,
            });
        let geometries = [geometry];
        let build = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(work.flags)
            .mode(if work.update {
                vk::BuildAccelerationStructureModeKHR::UPDATE
            } else {
                vk::BuildAccelerationStructureModeKHR::BUILD
            })
            .src_acceleration_structure(if work.update {
                work.destination
            } else {
                vk::AccelerationStructureKHR::null()
            })
            .dst_acceleration_structure(work.destination)
            .geometries(&geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: work.scratch.device_address()?,
            });
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(work.instance_count)];
        let range_slices = [&ranges[..]];
        let accesses = vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
            | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR;
        let stages = vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
            | vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR;
        let before = vk::MemoryBarrier::default()
            .src_access_mask(accesses)
            .dst_access_mask(accesses);
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_access_mask(accesses);
        // SAFETY: the frame retains the input snapshot, scratch, TLAS and BLAS
        // dependencies. Barriers order previous traces/refits and future reads,
        // including reuse of shared GPU-only scratch on this graphics queue.
        unsafe {
            self.logical.handle.cmd_pipeline_barrier(
                command,
                stages,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            self.acceleration_structure
                .cmd_build_acceleration_structures(command, &[build], &range_slices);
            self.logical.handle.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                stages,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        Ok(())
    }
}

/// The build inputs of one BLAS: alive until the recording that builds it
/// has completed on the queue. The destination handle belongs to the
/// `BuiltAcceleration` returned beside it.
pub(crate) struct VulkanBlasBuild {
    /// The boxes, or the positions and indices, the geometry points into.
    _inputs: Vec<BufferAllocation>,
    _scratch: BufferAllocation,
    geometries: [vk::AccelerationStructureGeometryKHR<'static>; 1],
    primitive_count: u32,
    scratch_address: u64,
    destination: vk::AccelerationStructureKHR,
}

pub(crate) struct BuiltAcceleration {
    loader: ash::khr::acceleration_structure::Device,
    pub(crate) handle: vk::AccelerationStructureKHR,
    _storage: BufferAllocation,
    /// Present only on top-level structures built `Updatable`.
    refit: Option<TlasRefit>,
}

/// What a top-level refit needs beyond the structure itself.
struct TlasRefit {
    /// GPU-only scratch can be reused with queue barriers; CPU input cannot.
    scratch: Arc<BufferAllocation>,
    /// Pinned at build time: a refit must present the same primitive count.
    instance_count: u32,
    /// The build's flags, which a refit is required to repeat exactly.
    flags: vk::BuildAccelerationStructureFlagsKHR,
}

pub(crate) struct VulkanTlasBuild {
    input: BufferAllocation,
    scratch: Arc<BufferAllocation>,
    destination: vk::AccelerationStructureKHR,
    instance_count: u32,
    flags: vk::BuildAccelerationStructureFlagsKHR,
    update: bool,
}

impl Drop for BuiltAcceleration {
    fn drop(&mut self) {
        // SAFETY: synchronous calls await completion and queued frames retain the
        // owning acceleration structure through their submit fence.
        unsafe {
            self.loader
                .destroy_acceleration_structure(self.handle, None)
        };
    }
}
