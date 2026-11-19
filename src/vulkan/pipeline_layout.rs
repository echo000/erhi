use ash::vk;

use super::VulkanDevice;
use crate::pipeline_layout::{FilterMode, StaticSampler};
use crate::{DescriptorType, PipelineLayoutDescriptor, RhiError, ShaderStages};

pub(crate) struct VulkanPipelineLayout {
    device: ash::Device,
    handle: vk::PipelineLayout,
    set_layouts: Vec<vk::DescriptorSetLayout>,
    /// Immutable samplers referenced by the set layouts; destroyed with them.
    immutable_samplers: Vec<vk::Sampler>,
    /// Stage flags of the declared push-constant range, for `cmd_push_constants`.
    push_constant_stages: vk::ShaderStageFlags,
}

impl VulkanPipelineLayout {
    pub(crate) fn set_layout_handle(&self, set: u32) -> Option<vk::DescriptorSetLayout> {
        self.set_layouts.get(set as usize).copied()
    }

    pub(crate) fn handle(&self) -> vk::PipelineLayout {
        self.handle
    }

    pub(crate) fn push_constant_stages(&self) -> vk::ShaderStageFlags {
        self.push_constant_stages
    }
}

impl VulkanDevice {
    pub(crate) fn create_pipeline_layout(
        &self,
        descriptor: PipelineLayoutDescriptor<'_>,
    ) -> Result<VulkanPipelineLayout, RhiError> {
        let mut set_layouts = Vec::with_capacity(descriptor.sets.len());
        let mut immutable_samplers: Vec<vk::Sampler> = Vec::new();
        let destroy_all = |layouts: &[vk::DescriptorSetLayout], samplers: &[vk::Sampler]| {
            destroy_set_layouts(&self.logical.handle, layouts);
            destroy_samplers(&self.logical.handle, samplers);
        };
        for set in descriptor.sets {
            // Immutable samplers must outlive layout creation, so create the
            // set's samplers first and reference them from the bindings.
            let mut set_samplers = Vec::new();
            for binding in set.bindings {
                if let DescriptorType::CombinedImageSampler(sampler) = binding.descriptor_type {
                    let handle = match create_immutable_sampler(&self.logical.handle, sampler) {
                        Ok(handle) => handle,
                        Err(error) => {
                            destroy_all(&set_layouts, &immutable_samplers);
                            destroy_samplers(&self.logical.handle, &set_samplers);
                            return Err(error);
                        }
                    };
                    set_samplers.push(handle);
                }
            }
            // A binding of several takes its sampler once for each.
            let repeated: Vec<Vec<vk::Sampler>> = set
                .bindings
                .iter()
                .filter(|binding| {
                    matches!(
                        binding.descriptor_type,
                        DescriptorType::CombinedImageSampler(_)
                    )
                })
                .zip(&set_samplers)
                .map(|(binding, sampler)| vec![*sampler; binding.count as usize])
                .collect();
            let mut sampler_cursor = 0;
            let bindings: Vec<_> = set
                .bindings
                .iter()
                .map(|binding| {
                    let entry = vk::DescriptorSetLayoutBinding::default()
                        .binding(binding.binding)
                        .descriptor_type(vulkan_descriptor_type(binding.descriptor_type))
                        .descriptor_count(binding.count)
                        .stage_flags(vulkan_shader_stages(binding.visibility));
                    if matches!(
                        binding.descriptor_type,
                        DescriptorType::CombinedImageSampler(_)
                    ) {
                        let entry = entry.immutable_samplers(&repeated[sampler_cursor]);
                        sampler_cursor += 1;
                        entry
                    } else {
                        entry
                    }
                })
                .collect();
            let create = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            // SAFETY: binding storage and the referenced immutable samplers
            // remain live for this immediate call.
            let layout = match unsafe {
                self.logical
                    .handle
                    .create_descriptor_set_layout(&create, None)
            } {
                Ok(layout) => layout,
                Err(result) => {
                    destroy_all(&set_layouts, &immutable_samplers);
                    destroy_samplers(&self.logical.handle, &set_samplers);
                    return Err(RhiError::vk("vkCreateDescriptorSetLayout", result));
                }
            };
            set_layouts.push(layout);
            immutable_samplers.append(&mut set_samplers);
        }

        let push_constant_stages = descriptor
            .push_constants
            .map(|push| vulkan_shader_stages(push.stages))
            .unwrap_or_default();
        let push_ranges: Vec<vk::PushConstantRange> = descriptor
            .push_constants
            .map(|push| {
                vec![
                    vk::PushConstantRange::default()
                        .stage_flags(push_constant_stages)
                        .offset(0)
                        .size(push.size),
                ]
            })
            .unwrap_or_default();
        let create = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        // SAFETY: every set layout belongs to this device and remains live in
        // the returned owner.
        let handle = match unsafe { self.logical.handle.create_pipeline_layout(&create, None) } {
            Ok(handle) => handle,
            Err(result) => {
                destroy_all(&set_layouts, &immutable_samplers);
                return Err(RhiError::vk("vkCreatePipelineLayout", result));
            }
        };
        Ok(VulkanPipelineLayout {
            device: self.logical.handle.clone(),
            handle,
            set_layouts,
            immutable_samplers,
            push_constant_stages,
        })
    }
}

fn create_immutable_sampler(
    device: &ash::Device,
    sampler: StaticSampler,
) -> Result<vk::Sampler, RhiError> {
    let filter = super::sampler::vk_filter(sampler.filter);
    let address = super::sampler::vk_address_mode(sampler.address_mode);
    let create = vk::SamplerCreateInfo::default()
        .mag_filter(filter)
        .min_filter(filter)
        .mipmap_mode(match sampler.filter {
            FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
            FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
        })
        .address_mode_u(address)
        .address_mode_v(address)
        .address_mode_w(address)
        .min_lod(0.0)
        .max_lod(vk::LOD_CLAMP_NONE);
    // SAFETY: the create info contains no borrowed data.
    unsafe { device.create_sampler(&create, None) }
        .map_err(|result| RhiError::vk("vkCreateSampler(immutable)", result))
}

pub(crate) fn vulkan_descriptor_type(descriptor_type: DescriptorType) -> vk::DescriptorType {
    match descriptor_type {
        DescriptorType::AccelerationStructure => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        DescriptorType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        DescriptorType::ReadOnlyStorageBuffer | DescriptorType::StorageBuffer => {
            vk::DescriptorType::STORAGE_BUFFER
        }
        DescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        DescriptorType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
        DescriptorType::Sampler => vk::DescriptorType::SAMPLER,
        DescriptorType::CombinedImageSampler(_) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
    }
}

pub(crate) fn vulkan_shader_stages(stages: ShaderStages) -> vk::ShaderStageFlags {
    let mut flags = vk::ShaderStageFlags::empty();
    if stages.contains(ShaderStages::RAY_GENERATION) {
        flags |= vk::ShaderStageFlags::RAYGEN_KHR;
    }
    if stages.contains(ShaderStages::MISS) {
        flags |= vk::ShaderStageFlags::MISS_KHR;
    }
    if stages.contains(ShaderStages::CLOSEST_HIT) {
        flags |= vk::ShaderStageFlags::CLOSEST_HIT_KHR;
    }
    if stages.contains(ShaderStages::ANY_HIT) {
        flags |= vk::ShaderStageFlags::ANY_HIT_KHR;
    }
    if stages.contains(ShaderStages::INTERSECTION) {
        flags |= vk::ShaderStageFlags::INTERSECTION_KHR;
    }
    if stages.contains(ShaderStages::COMPUTE) {
        flags |= vk::ShaderStageFlags::COMPUTE;
    }
    if stages.contains(ShaderStages::VERTEX) {
        flags |= vk::ShaderStageFlags::VERTEX;
    }
    if stages.contains(ShaderStages::FRAGMENT) {
        flags |= vk::ShaderStageFlags::FRAGMENT;
    }
    flags
}

fn destroy_set_layouts(device: &ash::Device, layouts: &[vk::DescriptorSetLayout]) {
    for layout in layouts.iter().rev() {
        // SAFETY: each handle was created by this device and is not retained
        // after a failed construction.
        unsafe { device.destroy_descriptor_set_layout(*layout, None) };
    }
}

fn destroy_samplers(device: &ash::Device, samplers: &[vk::Sampler]) {
    for sampler in samplers.iter().rev() {
        // SAFETY: each sampler was created by this device; set layouts that
        // referenced it are destroyed first by every call path.
        unsafe { device.destroy_sampler(*sampler, None) };
    }
}

impl Drop for VulkanPipelineLayout {
    fn drop(&mut self) {
        // SAFETY: the pipeline layout is destroyed before the set layouts it
        // references, and the set layouts before their immutable samplers;
        // all handles belong to this live device.
        unsafe { self.device.destroy_pipeline_layout(self.handle, None) };
        destroy_set_layouts(&self.device, &self.set_layouts);
        destroy_samplers(&self.device, &self.immutable_samplers);
    }
}
