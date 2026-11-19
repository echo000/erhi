use std::ffi::CString;

use ash::vk;

use super::VulkanDevice;
use super::pipeline_layout::VulkanPipelineLayout;
use super::shader::VulkanShaderModule;
use crate::RhiError;

/// One owned `VkPipeline` for a single compute entry point.
pub(crate) struct VulkanComputePipeline {
    device: ash::Device,
    pub(crate) pipeline: vk::Pipeline,
}

impl VulkanDevice {
    pub(crate) fn create_compute_pipeline(
        &self,
        label: &str,
        module: &VulkanShaderModule,
        entry: &str,
        layout: &VulkanPipelineLayout,
    ) -> Result<VulkanComputePipeline, RhiError> {
        let entry_name = CString::new(entry).map_err(|_| RhiError::InvalidComputePipeline {
            reason: "the entry-point name contains an interior NUL byte",
        })?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module.handle)
            .name(&entry_name);
        let create = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout.handle());
        let (_held, cache) = self.pipeline_cache();
        // SAFETY: the shader module, entry name, and pipeline layout are live
        // for this immediate call.
        let pipeline = unsafe {
            self.logical
                .handle
                .create_compute_pipelines(cache, &[create], None)
        }
        .map_err(|(_, result)| RhiError::vk("vkCreateComputePipelines", result))?[0];
        self.name_object(pipeline, label);
        Ok(VulkanComputePipeline {
            device: self.logical.handle.clone(),
            pipeline,
        })
    }
}

impl Drop for VulkanComputePipeline {
    fn drop(&mut self) {
        // SAFETY: dispatch completion is awaited before pipeline owners drop.
        unsafe { self.device.destroy_pipeline(self.pipeline, None) };
    }
}
