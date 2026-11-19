use std::sync::{Arc, Mutex};

use ash::vk;

use super::pipeline_layout::vulkan_descriptor_type;
use super::{BuiltAcceleration, VulkanBuffer, VulkanDevice, VulkanImageView, VulkanPipelineLayout};
use crate::{DescriptorBinding, DescriptorType, RhiError};

pub(super) struct DescriptorPage {
    device: ash::Device,
    pool: vk::DescriptorPool,
    sizes: Vec<vk::DescriptorPoolSize>,
    available: Mutex<u32>,
}

/// Sets share a pool but own their descriptors until their frame fence retires.
pub(crate) struct VulkanDescriptorSet {
    device: ash::Device,
    page: Arc<DescriptorPage>,
    pub(crate) handle: vk::DescriptorSet,
}

impl VulkanDevice {
    pub(crate) fn create_descriptor_set(
        &self,
        layout: &VulkanPipelineLayout,
        set: u32,
        bindings: &[DescriptorBinding],
    ) -> Result<VulkanDescriptorSet, RhiError> {
        let set_layout =
            layout
                .set_layout_handle(set)
                .ok_or(RhiError::InvalidDescriptorSetOrder {
                    expected: set,
                    actual: u32::MAX,
                })?;
        let mut sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
        for binding in bindings {
            let descriptor_type = vulkan_descriptor_type(binding.descriptor_type);
            match sizes.iter_mut().find(|size| size.ty == descriptor_type) {
                Some(size) => size.descriptor_count += binding.count,
                None => sizes.push(vk::DescriptorPoolSize {
                    ty: descriptor_type,
                    descriptor_count: binding.count,
                }),
            }
        }
        sizes.sort_by_key(|size| size.ty);
        let mut pages = crate::sync::lock(&self.descriptor_pages, "the descriptor pools")?;
        pages.retain(|page| page.strong_count() != 0);
        for page in pages.iter().filter_map(std::sync::Weak::upgrade) {
            if page.sizes.len() == sizes.len()
                && page
                    .sizes
                    .iter()
                    .zip(&sizes)
                    .all(|(a, b)| a.ty == b.ty && a.descriptor_count == b.descriptor_count)
                && let Some(handle) = page.allocate(set_layout)?
            {
                return Ok(VulkanDescriptorSet {
                    device: self.logical.handle.clone(),
                    page,
                    handle,
                });
            }
        }
        let largest = sizes
            .iter()
            .map(|size| size.descriptor_count)
            .max()
            .unwrap_or(1);
        let capacity = (16384 / largest).clamp(1, 256);
        let pool_sizes: Vec<_> = sizes
            .iter()
            .map(|size| vk::DescriptorPoolSize {
                ty: size.ty,
                descriptor_count: size.descriptor_count * capacity,
            })
            .collect();
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(capacity)
            .pool_sizes(&pool_sizes);
        // SAFETY: the pool sizes are fully initialized and live for this call.
        let pool = unsafe { self.logical.handle.create_descriptor_pool(&pool_info, None) }
            .map_err(|result| RhiError::vk("vkCreateDescriptorPool", result))?;

        let page = Arc::new(DescriptorPage {
            device: self.logical.handle.clone(),
            pool,
            sizes,
            available: Mutex::new(capacity),
        });
        let handle = page.allocate(set_layout)?.ok_or_else(|| {
            RhiError::vk(
                "vkAllocateDescriptorSets",
                vk::Result::ERROR_OUT_OF_POOL_MEMORY,
            )
        })?;
        pages.push(Arc::downgrade(&page));
        Ok(VulkanDescriptorSet {
            device: self.logical.handle.clone(),
            page,
            handle,
        })
    }
}

impl DescriptorPage {
    fn allocate(
        &self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<Option<vk::DescriptorSet>, RhiError> {
        let mut available = crate::sync::lock(&self.available, "the descriptor pools")?;
        if *available == 0 {
            return Ok(None);
        }
        let set_layouts = [layout];
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&set_layouts);
        // SAFETY: matching descriptor counts have capacity; the lock serializes pool access.
        let handle = match unsafe { self.device.allocate_descriptor_sets(&allocate) } {
            Ok(mut sets) => sets.remove(0),
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                *available = 0;
                return Ok(None);
            }
            Err(result) => {
                return Err(RhiError::vk("vkAllocateDescriptorSets", result));
            }
        };
        *available -= 1;
        Ok(Some(handle))
    }
}

impl VulkanDescriptorSet {
    pub(crate) fn write_acceleration_structure(
        &self,
        binding: u32,
        element: u32,
        acceleration: &BuiltAcceleration,
    ) {
        let handles = [acceleration.handle];
        let mut acceleration_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&handles);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.handle)
            .dst_binding(binding)
            .dst_array_element(element)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut acceleration_write);
        // SAFETY: the acceleration structure and set are live; the common
        // layer validated the binding's declared type.
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
    }

    pub(crate) fn write_image(
        &self,
        binding: u32,
        element: u32,
        view: &VulkanImageView,
        descriptor_type: DescriptorType,
    ) {
        // Storage and sampled images both use the GENERAL layout: the common
        // resource states map every shader image access to GENERAL.
        let info = [vk::DescriptorImageInfo::default()
            .image_view(view.handle)
            .image_layout(vk::ImageLayout::GENERAL)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.handle)
            .dst_binding(binding)
            .dst_array_element(element)
            .descriptor_count(1)
            .descriptor_type(vulkan_descriptor_type(descriptor_type))
            .image_info(&info);
        // SAFETY: the view and set are live; the descriptor type was
        // validated against the layout by the common layer.
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
    }

    pub(crate) fn write_sampler(&self, binding: u32, element: u32, sampler: &super::VulkanSampler) {
        let info = [vk::DescriptorImageInfo::default().sampler(sampler.handle)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.handle)
            .dst_binding(binding)
            .dst_array_element(element)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(&info);
        // SAFETY: the sampler and set are live; the binding's type was
        // validated against the layout by the common layer.
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
    }

    pub(crate) fn write_storage_buffer(
        &self,
        binding: u32,
        element: u32,
        buffer: &VulkanBuffer,
        offset: u64,
        size: u64,
    ) {
        self.write_buffer(
            binding,
            element,
            buffer,
            offset,
            size,
            vk::DescriptorType::STORAGE_BUFFER,
        );
    }

    pub(crate) fn write_uniform_buffer(
        &self,
        binding: u32,
        element: u32,
        buffer: &VulkanBuffer,
        offset: u64,
        size: u64,
    ) {
        self.write_buffer(
            binding,
            element,
            buffer,
            offset,
            size,
            vk::DescriptorType::UNIFORM_BUFFER,
        );
    }

    fn write_buffer(
        &self,
        binding: u32,
        element: u32,
        buffer: &VulkanBuffer,
        offset: u64,
        size: u64,
        descriptor_type: vk::DescriptorType,
    ) {
        let info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer.handle)
            .offset(offset)
            .range(size)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.handle)
            .dst_binding(binding)
            .dst_array_element(element)
            .descriptor_count(1)
            .descriptor_type(descriptor_type)
            .buffer_info(&info);
        // SAFETY: the buffer and set are live; usage, range, and type were
        // validated by the common layer.
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
    }
}

impl Drop for VulkanDescriptorSet {
    fn drop(&mut self) {
        let mut available = self
            .page
            .available
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // SAFETY: the owning frame has retired and pool access is serialized.
        if unsafe {
            self.device
                .free_descriptor_sets(self.page.pool, &[self.handle])
        }
        .is_ok()
        {
            *available += 1;
        }
    }
}

impl Drop for DescriptorPage {
    fn drop(&mut self) {
        // SAFETY: the final set has retired; no allocation can still use this pool.
        unsafe { self.device.destroy_descriptor_pool(self.pool, None) };
    }
}
