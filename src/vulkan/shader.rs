use ash::vk;

use super::VulkanDevice;
use crate::RhiError;

/// One owned `VkShaderModule` created from validated SPIR-V bytes.
pub(crate) struct VulkanShaderModule {
    device: ash::Device,
    pub(crate) handle: vk::ShaderModule,
}

impl VulkanDevice {
    pub(crate) fn create_shader_module(
        &self,
        bytes: &[u8],
    ) -> Result<VulkanShaderModule, RhiError> {
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let create = vk::ShaderModuleCreateInfo::default().code(&words);
        // SAFETY: the word vector is live for this call and starts with the
        // SPIR-V magic validated by the common layer.
        let handle = unsafe { self.logical.handle.create_shader_module(&create, None) }
            .map_err(|result| RhiError::vk("vkCreateShaderModule", result))?;
        Ok(VulkanShaderModule {
            device: self.logical.handle.clone(),
            handle,
        })
    }
}

impl Drop for VulkanShaderModule {
    fn drop(&mut self) {
        // SAFETY: pipelines retain their own reference; this owner holds the
        // only raw handle by the time it drops.
        unsafe { self.device.destroy_shader_module(self.handle, None) };
    }
}
