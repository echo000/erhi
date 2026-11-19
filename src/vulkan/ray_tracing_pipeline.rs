use std::ffi::CString;

use ash::vk;

use super::{BufferAllocation, VulkanDevice, VulkanPipelineLayout, VulkanShaderModule};
use crate::RhiError;
use crate::ray_tracing_pipeline::{RayTracingPipelineSettings, StageOf, compute_sbt_layout};

/// One owned Vulkan ray-tracing pipeline with its host-visible shader
/// binding table and precomputed strided regions.
pub(crate) struct VulkanRayTracingPipeline {
    device: ash::Device,
    pub(crate) pipeline: vk::Pipeline,
    #[allow(dead_code)] // Keeps the SBT storage alive for the regions below.
    sbt: BufferAllocation,
    pub(crate) raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub(crate) miss_region: vk::StridedDeviceAddressRegionKHR,
    pub(crate) hit_region: vk::StridedDeviceAddressRegionKHR,
}

impl VulkanDevice {
    pub(crate) fn create_ray_tracing_pipeline(
        &self,
        descriptor: &RayTracingPipelineSettings<'_, VulkanShaderModule>,
        layout: &VulkanPipelineLayout,
    ) -> Result<VulkanRayTracingPipeline, RhiError> {
        let to_cstring = |entry: &str| {
            CString::new(entry).map_err(|_| RhiError::InvalidRayTracingPipeline {
                reason: "entry point names must not contain NUL bytes",
            })
        };
        // Stage order: ray generation, every miss entry, then per hit group
        // whichever it has of its intersection, closest hit and any hit.
        let mut names = vec![to_cstring(descriptor.ray_generation.entry)?];
        let mut modules = vec![descriptor.ray_generation.module.handle];
        for miss in &descriptor.miss_entries {
            names.push(to_cstring(miss.entry)?);
            modules.push(miss.module.handle);
        }
        let mut stage_kinds = vec![vk::ShaderStageFlags::RAYGEN_KHR];
        stage_kinds.extend(std::iter::repeat_n(
            vk::ShaderStageFlags::MISS_KHR,
            descriptor.miss_entries.len(),
        ));
        let mut groups = Vec::new();
        groups.push(
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
        );
        for index in 0..descriptor.miss_entries.len() {
            groups.push(
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                    .general_shader(1 + index as u32)
                    .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                    .any_hit_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
            );
        }
        for group in &descriptor.hit_groups {
            let mut stage =
                |entry: &Option<StageOf<'_, VulkanShaderModule>>, kind| -> Result<u32, RhiError> {
                    let Some(entry) = entry else {
                        return Ok(vk::SHADER_UNUSED_KHR);
                    };
                    let index = names.len() as u32;
                    names.push(to_cstring(entry.entry)?);
                    modules.push(entry.module.handle);
                    stage_kinds.push(kind);
                    Ok(index)
                };
            let intersection = stage(&group.intersection, vk::ShaderStageFlags::INTERSECTION_KHR)?;
            let closest_hit = stage(&group.closest_hit, vk::ShaderStageFlags::CLOSEST_HIT_KHR)?;
            let any_hit = stage(&group.any_hit, vk::ShaderStageFlags::ANY_HIT_KHR)?;
            groups.push(
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    // A group with no intersection shader is for triangles.
                    .ty(if group.intersection.is_some() {
                        vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP
                    } else {
                        vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
                    })
                    .general_shader(vk::SHADER_UNUSED_KHR)
                    .closest_hit_shader(closest_hit)
                    .any_hit_shader(any_hit)
                    .intersection_shader(intersection),
            );
        }

        let stages: Vec<_> = names
            .iter()
            .zip(&stage_kinds)
            .zip(&modules)
            .map(|((name, kind), module)| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(*kind)
                    .module(*module)
                    .name(name)
            })
            .collect();
        let create_info = [vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(descriptor.max_recursion_depth)
            .layout(layout.handle())];
        let (_held, cache) = self.pipeline_cache();
        // SAFETY: the module, layout, stage names, and group arrays are live
        // for this call.
        let pipeline = unsafe {
            self.ray_tracing_pipeline.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                cache,
                &create_info,
                None,
            )
        }
        .map_err(|(_, result)| RhiError::vk("vkCreateRayTracingPipelinesKHR", result))?[0];

        let destroy_pipeline = || {
            // SAFETY: the fresh pipeline has no pending GPU work.
            unsafe { self.logical.handle.destroy_pipeline(pipeline, None) };
        };

        let capabilities = self.capabilities();
        let handle_size = capabilities.shader_group_handle_size;
        let group_count = groups.len() as u32;
        // SAFETY: the query copies `group_count` handles from the live
        // pipeline into an exactly-sized output vector.
        let handles = unsafe {
            self.ray_tracing_pipeline
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    group_count,
                    (handle_size * group_count) as usize,
                )
        };
        let handles = match handles {
            Ok(handles) => handles,
            Err(result) => {
                destroy_pipeline();
                return Err(RhiError::vk("vkGetRayTracingShaderGroupHandlesKHR", result));
            }
        };
        for group in 0..group_count as usize {
            let handle = &handles[group * handle_size as usize..(group + 1) * handle_size as usize];
            if handle.iter().all(|byte| *byte == 0) {
                destroy_pipeline();
                return Err(RhiError::ZeroDeviceAddress(
                    "ray-tracing shader group handle",
                ));
            }
        }

        let table = compute_sbt_layout(
            handle_size,
            capabilities.shader_group_handle_alignment,
            capabilities.shader_group_base_alignment,
            descriptor.miss_entries.len() as u64,
            descriptor.hit_groups.len() as u64,
        );
        let sbt = match self.allocate_buffer(
            table.total_size,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            true,
        ) {
            Ok(sbt) => sbt,
            Err(error) => {
                destroy_pipeline();
                return Err(error);
            }
        };
        let write_group = |group: usize, offset: u64| {
            let handle = &handles[group * handle_size as usize..(group + 1) * handle_size as usize];
            sbt.write_bytes_at(offset, handle)
        };
        let mut written = write_group(0, table.raygen_offset);
        for miss in 0..descriptor.miss_entries.len() {
            written = written.and_then(|()| {
                write_group(
                    1 + miss,
                    table.miss_offset + miss as u64 * table.miss_stride,
                )
            });
        }
        for hit in 0..descriptor.hit_groups.len() {
            written = written.and_then(|()| {
                write_group(
                    1 + descriptor.miss_entries.len() + hit,
                    table.hit_offset + hit as u64 * table.hit_stride,
                )
            });
        }
        let address = written.and_then(|()| sbt.device_address());
        let address = match address {
            Ok(address) => address,
            Err(error) => {
                destroy_pipeline();
                return Err(error);
            }
        };
        if !address.is_multiple_of(u64::from(capabilities.shader_group_base_alignment)) {
            destroy_pipeline();
            return Err(RhiError::ShaderBindingTableAlignment {
                address,
                required: u64::from(capabilities.shader_group_base_alignment),
            });
        }

        self.name_object(pipeline, descriptor.label);
        Ok(VulkanRayTracingPipeline {
            device: self.logical.handle.clone(),
            pipeline,
            sbt,
            raygen_region: vk::StridedDeviceAddressRegionKHR::default()
                .device_address(address + table.raygen_offset)
                .stride(table.raygen_size)
                .size(table.raygen_size),
            miss_region: vk::StridedDeviceAddressRegionKHR::default()
                .device_address(address + table.miss_offset)
                .stride(table.miss_stride)
                .size(table.miss_size),
            hit_region: vk::StridedDeviceAddressRegionKHR::default()
                .device_address(address + table.hit_offset)
                .stride(table.hit_stride)
                .size(table.hit_size),
        })
    }
}

impl Drop for VulkanRayTracingPipeline {
    fn drop(&mut self) {
        // SAFETY: the synchronous encoder waits for queue completion before
        // owners drop, and this owner holds the only raw pipeline handle.
        unsafe { self.device.destroy_pipeline(self.pipeline, None) };
    }
}
