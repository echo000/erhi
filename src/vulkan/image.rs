use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use super::{VulkanDevice, lock_allocator};
use crate::{
    ImageAspect, ImageDescriptor, ImageDimension, ImageFormat, ImageUsages, ImageViewDimension,
    ImageViewRange, RhiError,
};

pub(crate) fn vk_format(format: ImageFormat) -> vk::Format {
    match format {
        ImageFormat::Bc1Unorm => vk::Format::BC1_RGBA_UNORM_BLOCK,
        ImageFormat::Bc1UnormSrgb => vk::Format::BC1_RGBA_SRGB_BLOCK,
        ImageFormat::Bc2Unorm => vk::Format::BC2_UNORM_BLOCK,
        ImageFormat::Bc2UnormSrgb => vk::Format::BC2_SRGB_BLOCK,
        ImageFormat::Bc3Unorm => vk::Format::BC3_UNORM_BLOCK,
        ImageFormat::Bc3UnormSrgb => vk::Format::BC3_SRGB_BLOCK,
        ImageFormat::Bc4Unorm => vk::Format::BC4_UNORM_BLOCK,
        ImageFormat::Bc4Snorm => vk::Format::BC4_SNORM_BLOCK,
        ImageFormat::Bc5Unorm => vk::Format::BC5_UNORM_BLOCK,
        ImageFormat::Bc5Snorm => vk::Format::BC5_SNORM_BLOCK,
        ImageFormat::Bc6hUfloat => vk::Format::BC6H_UFLOAT_BLOCK,
        ImageFormat::Bc6hSfloat => vk::Format::BC6H_SFLOAT_BLOCK,
        ImageFormat::Bc7Unorm => vk::Format::BC7_UNORM_BLOCK,
        ImageFormat::Bc7UnormSrgb => vk::Format::BC7_SRGB_BLOCK,
        ImageFormat::R8Unorm => vk::Format::R8_UNORM,
        ImageFormat::R8Snorm => vk::Format::R8_SNORM,
        ImageFormat::R8Uint => vk::Format::R8_UINT,
        ImageFormat::R8Sint => vk::Format::R8_SINT,
        ImageFormat::Rg8Unorm => vk::Format::R8G8_UNORM,
        ImageFormat::Rg8Snorm => vk::Format::R8G8_SNORM,
        ImageFormat::Rg8Uint => vk::Format::R8G8_UINT,
        ImageFormat::Rg8Sint => vk::Format::R8G8_SINT,
        ImageFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        ImageFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        ImageFormat::Rgba8Snorm => vk::Format::R8G8B8A8_SNORM,
        ImageFormat::Rgba8Uint => vk::Format::R8G8B8A8_UINT,
        ImageFormat::Rgba8Sint => vk::Format::R8G8B8A8_SINT,
        ImageFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        ImageFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        ImageFormat::R16Unorm => vk::Format::R16_UNORM,
        ImageFormat::R16Snorm => vk::Format::R16_SNORM,
        ImageFormat::R16Uint => vk::Format::R16_UINT,
        ImageFormat::R16Sint => vk::Format::R16_SINT,
        ImageFormat::R16Float => vk::Format::R16_SFLOAT,
        ImageFormat::Rg16Unorm => vk::Format::R16G16_UNORM,
        ImageFormat::Rg16Snorm => vk::Format::R16G16_SNORM,
        ImageFormat::Rg16Uint => vk::Format::R16G16_UINT,
        ImageFormat::Rg16Sint => vk::Format::R16G16_SINT,
        ImageFormat::Rg16Float => vk::Format::R16G16_SFLOAT,
        ImageFormat::Rgba16Unorm => vk::Format::R16G16B16A16_UNORM,
        ImageFormat::Rgba16Snorm => vk::Format::R16G16B16A16_SNORM,
        ImageFormat::Rgba16Uint => vk::Format::R16G16B16A16_UINT,
        ImageFormat::Rgba16Sint => vk::Format::R16G16B16A16_SINT,
        ImageFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        ImageFormat::R32Uint => vk::Format::R32_UINT,
        ImageFormat::R32Sint => vk::Format::R32_SINT,
        ImageFormat::R32Float => vk::Format::R32_SFLOAT,
        ImageFormat::Rg32Uint => vk::Format::R32G32_UINT,
        ImageFormat::Rg32Sint => vk::Format::R32G32_SINT,
        ImageFormat::Rg32Float => vk::Format::R32G32_SFLOAT,
        ImageFormat::Rgba32Uint => vk::Format::R32G32B32A32_UINT,
        ImageFormat::Rgba32Sint => vk::Format::R32G32B32A32_SINT,
        ImageFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        ImageFormat::Rgb10a2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        ImageFormat::Rg11b10Ufloat => vk::Format::B10G11R11_UFLOAT_PACK32,
        ImageFormat::Depth16Unorm => vk::Format::D16_UNORM,
        ImageFormat::Depth32Float => vk::Format::D32_SFLOAT,
        ImageFormat::Depth24UnormStencil8 => vk::Format::D24_UNORM_S8_UINT,
        ImageFormat::Depth32FloatStencil8 => vk::Format::D32_SFLOAT_S8_UINT,
    }
}

/// Every plane of an image of `format`: what a transition or an
/// image-to-image copy covers.
/// `count` samples a texel (a validated power of two) as Vulkan's flag.
pub(crate) fn vk_samples(count: u32) -> vk::SampleCountFlags {
    vk::SampleCountFlags::from_raw(count)
}

pub(crate) fn vk_aspects(format: ImageFormat) -> vk::ImageAspectFlags {
    if format.has_stencil() {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else if format.is_depth() {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

impl VulkanDevice {
    /// Whether images of `format` can be created for every one of `usages`.
    pub(crate) fn supports_sample_count(&self, format: ImageFormat, sample_count: u32) -> bool {
        // SAFETY: the physical device belongs to this live instance.
        unsafe {
            self._instance
                .handle
                .get_physical_device_image_format_properties(
                    self.physical_device,
                    vk_format(format),
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    if format.is_depth() {
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    } else {
                        vk::ImageUsageFlags::COLOR_ATTACHMENT
                    },
                    vk::ImageCreateFlags::empty(),
                )
        }
        .is_ok_and(|properties| properties.sample_counts.contains(vk_samples(sample_count)))
    }

    pub(crate) fn format_supports(&self, format: ImageFormat, usages: ImageUsages) -> bool {
        // SAFETY: the physical device belongs to this live instance.
        let features = unsafe {
            self._instance
                .handle
                .get_physical_device_format_properties(self.physical_device, vk_format(format))
        }
        .optimal_tiling_features;
        let mut needed = vk::FormatFeatureFlags::empty();
        for (usage, feature) in [
            (ImageUsages::STORAGE, vk::FormatFeatureFlags::STORAGE_IMAGE),
            (ImageUsages::SAMPLED, vk::FormatFeatureFlags::SAMPLED_IMAGE),
            (
                ImageUsages::COPY_SOURCE,
                vk::FormatFeatureFlags::TRANSFER_SRC,
            ),
            (
                ImageUsages::COPY_DESTINATION,
                vk::FormatFeatureFlags::TRANSFER_DST,
            ),
            (
                ImageUsages::RENDER_TARGET,
                vk::FormatFeatureFlags::COLOR_ATTACHMENT,
            ),
            (
                ImageUsages::DEPTH_TARGET,
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
            ),
        ] {
            if usages.contains(usage) {
                needed |= feature;
            }
        }
        features.contains(needed)
    }
}

fn vk_usage(usages: ImageUsages) -> vk::ImageUsageFlags {
    let mut usage = vk::ImageUsageFlags::empty();
    if usages.contains(ImageUsages::DEPTH_TARGET) {
        usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    }
    if usages.contains(ImageUsages::STORAGE) {
        usage |= vk::ImageUsageFlags::STORAGE;
    }
    if usages.contains(ImageUsages::SAMPLED) {
        usage |= vk::ImageUsageFlags::SAMPLED;
    }
    if usages.contains(ImageUsages::COPY_SOURCE) {
        usage |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if usages.contains(ImageUsages::COPY_DESTINATION) {
        usage |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    if usages.contains(ImageUsages::RENDER_TARGET) {
        usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    usage
}

/// One owned `VkImage` whose memory comes from the shared device allocator.
pub(crate) struct VulkanImage {
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,
    pub(crate) handle: vk::Image,
    allocation: std::mem::ManuallyDrop<Allocation>,
}

impl VulkanImage {
    pub(crate) fn native(&self, descriptor: ImageDescriptor) -> crate::NativeImage {
        use ash::vk::Handle;
        crate::NativeImage::Vulkan {
            image: self.handle.as_raw(),
            format: vk_format(descriptor.format).as_raw(),
            usage: vk_usage(descriptor.usages).as_raw(),
        }
    }
}

impl VulkanDevice {
    pub(crate) fn create_image(
        &self,
        descriptor: ImageDescriptor,
    ) -> Result<VulkanImage, RhiError> {
        let (image_type, extent, array_layers) = match descriptor.dimension {
            ImageDimension::D2 => (
                vk::ImageType::TYPE_2D,
                vk::Extent3D {
                    width: descriptor.width,
                    height: descriptor.height,
                    depth: 1,
                },
                descriptor.depth_or_layers,
            ),
            ImageDimension::D3 => (
                vk::ImageType::TYPE_3D,
                vk::Extent3D {
                    width: descriptor.width,
                    height: descriptor.height,
                    depth: descriptor.depth_or_layers,
                },
                1,
            ),
        };
        // Six or more square layers may be viewed as a cube.
        let flags = if descriptor.dimension == ImageDimension::D2
            && array_layers >= 6
            && descriptor.width == descriptor.height
        {
            vk::ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            vk::ImageCreateFlags::empty()
        };
        let create_info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(image_type)
            .format(vk_format(descriptor.format))
            .extent(extent)
            .mip_levels(descriptor.mip_count)
            .array_layers(array_layers)
            .samples(vk_samples(descriptor.sample_count))
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk_usage(descriptor.usages))
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: the create info contains no dangling pointers.
        let handle = unsafe { self.logical.handle.create_image(&create_info, None) }
            .map_err(|result| RhiError::vk("vkCreateImage", result))?;
        // SAFETY: `handle` belongs to this live device.
        let requirements = unsafe { self.logical.handle.get_image_memory_requirements(handle) };
        let allocation = self.lock_allocator().allocate(&AllocationCreateDesc {
            name: "erhi image",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });
        let allocation = match allocation {
            Ok(allocation) => allocation,
            Err(error) => {
                // SAFETY: no memory was bound, so only the image exists.
                unsafe { self.logical.handle.destroy_image(handle, None) };
                return Err(self.allocation_error("allocate image memory", error.to_string()));
            }
        };
        // SAFETY: the allocation satisfies this image's exact requirements and
        // both objects belong to this device.
        let bound = unsafe {
            self.logical
                .handle
                .bind_image_memory(handle, allocation.memory(), allocation.offset())
        };
        if let Err(result) = bound {
            // SAFETY: binding failed, so the image and allocation are unused.
            unsafe { self.logical.handle.destroy_image(handle, None) };
            let _ = self.lock_allocator().free(allocation);
            return Err(RhiError::vk("vkBindImageMemory", result));
        }
        Ok(VulkanImage {
            device: self.logical.handle.clone(),
            allocator: self.allocator(),
            handle,
            allocation: std::mem::ManuallyDrop::new(allocation),
        })
    }
}

impl VulkanImage {
    pub(crate) fn allocation_size(&self) -> u64 {
        self.allocation.size()
    }

    pub(crate) fn create_view(
        &self,
        descriptor: ImageDescriptor,
        range: ImageViewRange,
    ) -> Result<VulkanImageView, RhiError> {
        let view_type = match range.dimension {
            ImageViewDimension::D2 => vk::ImageViewType::TYPE_2D,
            ImageViewDimension::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
            ImageViewDimension::Cube => vk::ImageViewType::CUBE,
            ImageViewDimension::CubeArray => vk::ImageViewType::CUBE_ARRAY,
            ImageViewDimension::D3 => vk::ImageViewType::TYPE_3D,
        };
        let aspects = match range.aspect {
            ImageAspect::All => vk_aspects(descriptor.format),
            ImageAspect::Depth => vk::ImageAspectFlags::DEPTH,
            ImageAspect::Stencil => vk::ImageAspectFlags::STENCIL,
        };
        let create_info = vk::ImageViewCreateInfo::default()
            .image(self.handle)
            .view_type(view_type)
            .format(vk_format(descriptor.format))
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspects)
                    .base_mip_level(range.base_mip)
                    .level_count(range.mip_count)
                    .base_array_layer(range.base_layer)
                    .layer_count(range.layer_count),
            );
        // SAFETY: the image is live and the range covers existing subresources.
        let handle = unsafe { self.device.create_image_view(&create_info, None) }
            .map_err(|result| RhiError::vk("vkCreateImageView", result))?;
        Ok(VulkanImageView {
            device: self.device.clone(),
            handle,
        })
    }
}

impl Drop for VulkanImage {
    fn drop(&mut self) {
        // SAFETY: callers wait for queue completion before dropping images, and
        // this owner holds the only raw image handle.
        unsafe { self.device.destroy_image(self.handle, None) };
        // SAFETY: this is the drop, so `allocation` is taken once and never
        // used again.
        let allocation = unsafe { std::mem::ManuallyDrop::take(&mut self.allocation) };
        let _ = lock_allocator(&self.allocator).free(allocation);
    }
}

/// One owned `VkImageView`. The common layer keeps the parent image alive.
pub(crate) struct VulkanImageView {
    device: ash::Device,
    pub(crate) handle: vk::ImageView,
}

impl Drop for VulkanImageView {
    fn drop(&mut self) {
        // SAFETY: the parent image outlives this view via the common wrapper,
        // and queue completion is awaited before destruction.
        unsafe { self.device.destroy_image_view(self.handle, None) };
    }
}
