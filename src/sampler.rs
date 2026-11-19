//! Sampler objects: how a shader reads between the texels of an image.
//!
//! A sampler baked into a pipeline layout ([`crate::StaticSampler`]) costs
//! nothing to bind and covers the common cases. This is the general kind,
//! written into a `Sampler` binding of a descriptor set like any other
//! resource: separate mip filtering, a level-of-detail range, anisotropy,
//! comparison (shadow maps) and the wrap modes.

use std::fmt;
use std::sync::Arc;

use crate::device::BackendDevice;
use crate::{AddressMode, Backend, CompareFunction, Device, FilterMode, RhiError};

/// What a sampler reads beyond the edge of an image whose address mode is
/// [`AddressMode::ClampToBorder`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BorderColor {
    /// Black with no alpha.
    #[default]
    TransparentBlack,
    /// Black with full alpha.
    OpaqueBlack,
    /// White with full alpha.
    OpaqueWhite,
}

/// The most texels an anisotropic sampler takes along the stretched axis.
pub const MAX_ANISOTROPY: u8 = 16;

/// How a sampler reads an image. The default filters linearly between
/// texels and between mips, over every mip, and clamps at the edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerDescriptor {
    /// Between texels when the image is magnified.
    pub mag_filter: FilterMode,
    /// Between texels when it is minified.
    pub min_filter: FilterMode,
    /// Between mip levels.
    pub mip_filter: FilterMode,
    /// What coordinates outside 0 to 1 read along the width.
    pub address_u: AddressMode,
    /// What coordinates outside 0 to 1 read along the height.
    pub address_v: AddressMode,
    /// What coordinates outside 0 to 1 read along the depth.
    pub address_w: AddressMode,
    /// The least and greatest mip level sampled (0 is the largest mip).
    pub lod_min: f32,
    /// The smallest mip sampled, counted from the largest as zero.
    pub lod_max: f32,
    /// Added to the level the hardware works out.
    pub lod_bias: f32,
    /// 1 for none, up to [`MAX_ANISOTROPY`]. Needs linear filtering
    /// throughout; a device without the feature samples as with 1.
    pub anisotropy: u8,
    /// A comparison sampler: each texel is compared against a reference
    /// (`SamplerComparisonState`) and the results filtered. Shadow maps.
    pub compare: Option<CompareFunction>,
    /// The colour [`AddressMode::ClampToBorder`] reads.
    pub border: BorderColor,
}

impl Default for SamplerDescriptor {
    fn default() -> Self {
        Self {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            address_u: AddressMode::ClampToEdge,
            address_v: AddressMode::ClampToEdge,
            address_w: AddressMode::ClampToEdge,
            lod_min: 0.0,
            lod_max: 1000.0,
            lod_bias: 0.0,
            anisotropy: 1,
            compare: None,
            border: BorderColor::TransparentBlack,
        }
    }
}

impl SamplerDescriptor {
    pub(crate) fn validate(&self) -> Result<(), RhiError> {
        let invalid = |reason| Err(RhiError::InvalidSampler { reason });
        if ![self.lod_min, self.lod_max, self.lod_bias]
            .into_iter()
            .all(f32::is_finite)
            || self.lod_min < 0.0
            || self.lod_max < self.lod_min
        {
            return invalid("the level-of-detail range must be finite, from 0, and in order");
        }
        if self.anisotropy == 0 || self.anisotropy > MAX_ANISOTROPY {
            return invalid("anisotropy is from 1 to 16");
        }
        if self.anisotropy > 1
            && [self.mag_filter, self.min_filter, self.mip_filter] != [FilterMode::Linear; 3]
        {
            return invalid("anisotropy needs linear filtering between texels and mips");
        }
        Ok(())
    }
}

/// One owned sampler. Clones share the native object.
#[derive(Clone)]
pub struct Sampler {
    pub(crate) inner: Arc<SamplerInner>,
}

pub(crate) struct SamplerInner {
    device_id: u64,
    descriptor: SamplerDescriptor,
    pub(crate) backend: SamplerBackend,
    _device: BackendDevice,
}

pub(crate) enum SamplerBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::vulkan::VulkanSampler>),
    /// Direct3D 12 has no sampler object: a sampler is its description,
    /// written into the device's sampler heap for the sets that hold it.
    /// That heap has 2048 slots; sets that hold the same samplers in the
    /// same order share theirs.
    #[cfg(all(feature = "dx12", windows))]
    Dx12(crate::dx12::Dx12Sampler),
}

impl fmt::Debug for Sampler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Sampler")
            .field("backend", &self.backend())
            .field("descriptor", &self.inner.descriptor)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Create a sampler, to write into `Sampler` bindings of descriptor sets.
    pub fn create_sampler(&self, descriptor: &SamplerDescriptor) -> Result<Sampler, RhiError> {
        descriptor.validate()?;
        let backend = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => {
                SamplerBackend::Vulkan(Box::new(device.create_sampler(descriptor)?))
            }
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(_) => {
                SamplerBackend::Dx12(crate::dx12::Dx12Sampler::new(descriptor))
            }
        };
        Ok(Sampler {
            inner: Arc::new(SamplerInner {
                device_id: self.id(),
                descriptor: *descriptor,
                backend,
                _device: self.backend.clone(),
            }),
        })
    }
}

impl Sampler {
    /// The backend of the device that made this.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match &self.inner.backend {
            #[cfg(feature = "vulkan")]
            SamplerBackend::Vulkan(_) => Backend::Vulkan,
            #[cfg(all(feature = "dx12", windows))]
            SamplerBackend::Dx12(_) => Backend::Dx12,
        }
    }

    /// The descriptor the sampler was made from.
    #[must_use]
    pub fn descriptor(&self) -> SamplerDescriptor {
        self.inner.descriptor
    }

    pub(crate) fn device_id(&self) -> u64 {
        self.inner.device_id
    }
}

#[cfg(test)]
#[path = "tests/sampler.rs"]
mod tests;
