use ash::vk;

use super::VulkanDevice;
use crate::RhiError;

/// One owned Vulkan query pool: of timestamps, or of occlusion counts.
pub(crate) struct VulkanTimestampQuerySet {
    device: ash::Device,
    pub(crate) handle: vk::QueryPool,
}

impl VulkanDevice {
    pub(crate) fn timestamp_valid_bits(&self) -> Result<u32, RhiError> {
        // SAFETY: the physical device and queue family belong to this instance.
        let queue_properties = unsafe {
            self._instance
                .handle
                .get_physical_device_queue_family_properties(self.physical_device)
        };
        let queue = queue_properties
            .get(self.queue_family_index as usize)
            .ok_or(RhiError::TimestampQueriesUnavailable)?;
        Ok(queue.timestamp_valid_bits)
    }

    pub(crate) fn create_timestamp_query_set(
        &self,
        count: u32,
    ) -> Result<VulkanTimestampQuerySet, RhiError> {
        if self.timestamp_valid_bits()? == 0 {
            return Err(RhiError::TimestampQueriesUnavailable);
        }
        let create = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(count);
        // SAFETY: the create info has no borrowed storage.
        let handle = unsafe { self.logical.handle.create_query_pool(&create, None) }
            .map_err(|result| RhiError::vk("vkCreateQueryPool", result))?;
        Ok(VulkanTimestampQuerySet {
            device: self.logical.handle.clone(),
            handle,
        })
    }

    pub(crate) fn create_occlusion_query_set(
        &self,
        count: u32,
    ) -> Result<VulkanTimestampQuerySet, RhiError> {
        let create = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::OCCLUSION)
            .query_count(count);
        // SAFETY: the create info has no borrowed storage.
        let handle = unsafe { self.logical.handle.create_query_pool(&create, None) }
            .map_err(|result| RhiError::vk("vkCreateQueryPool(occlusion)", result))?;
        Ok(VulkanTimestampQuerySet {
            device: self.logical.handle.clone(),
            handle,
        })
    }

    pub(crate) fn timestamp_period_nanoseconds(&self) -> Result<f64, RhiError> {
        // SAFETY: the physical device belongs to this live instance.
        let properties = unsafe {
            self._instance
                .handle
                .get_physical_device_properties(self.physical_device)
        };
        Ok(f64::from(properties.limits.timestamp_period))
    }
}

impl Drop for VulkanTimestampQuerySet {
    fn drop(&mut self) {
        // SAFETY: query submissions complete before this owner drops.
        unsafe { self.device.destroy_query_pool(self.handle, None) };
    }
}
