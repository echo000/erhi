use ash::vk;

use super::VulkanDevice;
use super::render_pipeline::compare_op;
use crate::sampler::{BorderColor, SamplerDescriptor};
use crate::{AddressMode, FilterMode, RhiError};

/// One owned `VkSampler`.
pub(crate) struct VulkanSampler {
    device: ash::Device,
    pub(crate) handle: vk::Sampler,
}

pub(crate) fn vk_filter(filter: FilterMode) -> vk::Filter {
    match filter {
        FilterMode::Nearest => vk::Filter::NEAREST,
        FilterMode::Linear => vk::Filter::LINEAR,
    }
}

pub(crate) fn vk_address_mode(mode: AddressMode) -> vk::SamplerAddressMode {
    match mode {
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        AddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
    }
}

impl VulkanDevice {
    pub(crate) fn create_sampler(
        &self,
        descriptor: &SamplerDescriptor,
    ) -> Result<VulkanSampler, RhiError> {
        // A device without the feature samples plainly.
        let anisotropy = if self.sampler_anisotropy {
            descriptor.anisotropy
        } else {
            1
        };
        let create = vk::SamplerCreateInfo::default()
            .mag_filter(vk_filter(descriptor.mag_filter))
            .min_filter(vk_filter(descriptor.min_filter))
            .mipmap_mode(match descriptor.mip_filter {
                FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
                FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
            })
            .address_mode_u(vk_address_mode(descriptor.address_u))
            .address_mode_v(vk_address_mode(descriptor.address_v))
            .address_mode_w(vk_address_mode(descriptor.address_w))
            .mip_lod_bias(descriptor.lod_bias)
            .min_lod(descriptor.lod_min)
            .max_lod(descriptor.lod_max)
            .anisotropy_enable(anisotropy > 1)
            .max_anisotropy(f32::from(anisotropy))
            .compare_enable(descriptor.compare.is_some())
            .compare_op(descriptor.compare.map_or(vk::CompareOp::NEVER, compare_op))
            .border_color(match descriptor.border {
                BorderColor::TransparentBlack => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
                BorderColor::OpaqueBlack => vk::BorderColor::FLOAT_OPAQUE_BLACK,
                BorderColor::OpaqueWhite => vk::BorderColor::FLOAT_OPAQUE_WHITE,
            });
        // SAFETY: the create info contains no borrowed data.
        let handle = unsafe { self.logical.handle.create_sampler(&create, None) }
            .map_err(|result| RhiError::vk("vkCreateSampler", result))?;
        Ok(VulkanSampler {
            device: self.logical.handle.clone(),
            handle,
        })
    }
}

impl Drop for VulkanSampler {
    fn drop(&mut self) {
        // SAFETY: sets that hold the sampler retain it until their frame
        // retires, so nothing in flight still reads it.
        unsafe { self.device.destroy_sampler(self.handle, None) };
    }
}
