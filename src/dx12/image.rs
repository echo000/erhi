use std::sync::{Arc, Mutex, OnceLock};

use gpu_allocator::MemoryLocation;
use gpu_allocator::d3d12::{
    Allocator, Resource, ResourceCategory, ResourceCreateDesc, ResourceStateOrBarrierLayout,
    ResourceType,
};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DEPTH_STENCIL_VIEW_DESC, D3D12_DEPTH_STENCIL_VIEW_DESC_0,
    D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
    D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_DSV_DIMENSION_TEXTURE2D,
    D3D12_DSV_DIMENSION_TEXTURE2DARRAY, D3D12_FEATURE_DATA_FORMAT_SUPPORT,
    D3D12_FEATURE_FORMAT_SUPPORT, D3D12_FORMAT_SUPPORT1, D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL,
    D3D12_FORMAT_SUPPORT1_RENDER_TARGET, D3D12_FORMAT_SUPPORT1_SHADER_LOAD,
    D3D12_FORMAT_SUPPORT1_TEXTURE2D, D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW,
    D3D12_RENDER_TARGET_VIEW_DESC, D3D12_RENDER_TARGET_VIEW_DESC_0, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_DIMENSION_TEXTURE3D,
    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE,
    D3D12_RESOURCE_STATE_COMMON, D3D12_RTV_DIMENSION_TEXTURE2D, D3D12_RTV_DIMENSION_TEXTURE2DARRAY,
    D3D12_RTV_DIMENSION_TEXTURE3D, D3D12_TEX2D_ARRAY_DSV, D3D12_TEX2D_ARRAY_RTV, D3D12_TEX2D_DSV,
    D3D12_TEX2D_RTV, D3D12_TEX3D_RTV, D3D12_TEXTURE_LAYOUT_UNKNOWN, ID3D12DescriptorHeap,
    ID3D12Device5, ID3D12Resource,
};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_DSV_DIMENSION_TEXTURE2DMS, D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS,
    D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, D3D12_RTV_DIMENSION_TEXTURE2DMS,
};
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{Dx12Device, lock_allocator};
use crate::{
    ImageAspect, ImageDescriptor, ImageDimension, ImageFormat, ImageUsages, ImageViewRange,
    RhiError,
};

/// The format an image of `format` is rendered to and copied as.
/// The format a buffer holds one plane of `format` in, for copies: the
/// depth of a depth-stencil format as floats, its stencil as bytes.
pub(crate) fn copy_format(format: ImageFormat, aspect: crate::ImageAspect) -> DXGI_FORMAT {
    match (format, aspect) {
        (_, crate::ImageAspect::Stencil) => DXGI_FORMAT_R8_UINT,
        (ImageFormat::Depth32FloatStencil8, crate::ImageAspect::Depth) => DXGI_FORMAT_R32_FLOAT,
        _ => dxgi_format(format),
    }
}

pub(crate) fn dxgi_format(format: ImageFormat) -> DXGI_FORMAT {
    match format {
        ImageFormat::Bc1Unorm => DXGI_FORMAT_BC1_UNORM,
        ImageFormat::Bc1UnormSrgb => DXGI_FORMAT_BC1_UNORM_SRGB,
        ImageFormat::Bc2Unorm => DXGI_FORMAT_BC2_UNORM,
        ImageFormat::Bc2UnormSrgb => DXGI_FORMAT_BC2_UNORM_SRGB,
        ImageFormat::Bc3Unorm => DXGI_FORMAT_BC3_UNORM,
        ImageFormat::Bc3UnormSrgb => DXGI_FORMAT_BC3_UNORM_SRGB,
        ImageFormat::Bc4Unorm => DXGI_FORMAT_BC4_UNORM,
        ImageFormat::Bc4Snorm => DXGI_FORMAT_BC4_SNORM,
        ImageFormat::Bc5Unorm => DXGI_FORMAT_BC5_UNORM,
        ImageFormat::Bc5Snorm => DXGI_FORMAT_BC5_SNORM,
        ImageFormat::Bc6hUfloat => DXGI_FORMAT_BC6H_UF16,
        ImageFormat::Bc6hSfloat => DXGI_FORMAT_BC6H_SF16,
        ImageFormat::Bc7Unorm => DXGI_FORMAT_BC7_UNORM,
        ImageFormat::Bc7UnormSrgb => DXGI_FORMAT_BC7_UNORM_SRGB,
        ImageFormat::R8Unorm => DXGI_FORMAT_R8_UNORM,
        ImageFormat::R8Snorm => DXGI_FORMAT_R8_SNORM,
        ImageFormat::R8Uint => DXGI_FORMAT_R8_UINT,
        ImageFormat::R8Sint => DXGI_FORMAT_R8_SINT,
        ImageFormat::Rg8Unorm => DXGI_FORMAT_R8G8_UNORM,
        ImageFormat::Rg8Snorm => DXGI_FORMAT_R8G8_SNORM,
        ImageFormat::Rg8Uint => DXGI_FORMAT_R8G8_UINT,
        ImageFormat::Rg8Sint => DXGI_FORMAT_R8G8_SINT,
        ImageFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        ImageFormat::Rgba8UnormSrgb => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        ImageFormat::Rgba8Snorm => DXGI_FORMAT_R8G8B8A8_SNORM,
        ImageFormat::Rgba8Uint => DXGI_FORMAT_R8G8B8A8_UINT,
        ImageFormat::Rgba8Sint => DXGI_FORMAT_R8G8B8A8_SINT,
        ImageFormat::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        ImageFormat::Bgra8UnormSrgb => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        ImageFormat::R16Unorm => DXGI_FORMAT_R16_UNORM,
        ImageFormat::R16Snorm => DXGI_FORMAT_R16_SNORM,
        ImageFormat::R16Uint => DXGI_FORMAT_R16_UINT,
        ImageFormat::R16Sint => DXGI_FORMAT_R16_SINT,
        ImageFormat::R16Float => DXGI_FORMAT_R16_FLOAT,
        ImageFormat::Rg16Unorm => DXGI_FORMAT_R16G16_UNORM,
        ImageFormat::Rg16Snorm => DXGI_FORMAT_R16G16_SNORM,
        ImageFormat::Rg16Uint => DXGI_FORMAT_R16G16_UINT,
        ImageFormat::Rg16Sint => DXGI_FORMAT_R16G16_SINT,
        ImageFormat::Rg16Float => DXGI_FORMAT_R16G16_FLOAT,
        ImageFormat::Rgba16Unorm => DXGI_FORMAT_R16G16B16A16_UNORM,
        ImageFormat::Rgba16Snorm => DXGI_FORMAT_R16G16B16A16_SNORM,
        ImageFormat::Rgba16Uint => DXGI_FORMAT_R16G16B16A16_UINT,
        ImageFormat::Rgba16Sint => DXGI_FORMAT_R16G16B16A16_SINT,
        ImageFormat::Rgba16Float => DXGI_FORMAT_R16G16B16A16_FLOAT,
        ImageFormat::R32Uint => DXGI_FORMAT_R32_UINT,
        ImageFormat::R32Sint => DXGI_FORMAT_R32_SINT,
        ImageFormat::R32Float => DXGI_FORMAT_R32_FLOAT,
        ImageFormat::Rg32Uint => DXGI_FORMAT_R32G32_UINT,
        ImageFormat::Rg32Sint => DXGI_FORMAT_R32G32_SINT,
        ImageFormat::Rg32Float => DXGI_FORMAT_R32G32_FLOAT,
        ImageFormat::Rgba32Uint => DXGI_FORMAT_R32G32B32A32_UINT,
        ImageFormat::Rgba32Sint => DXGI_FORMAT_R32G32B32A32_SINT,
        ImageFormat::Rgba32Float => DXGI_FORMAT_R32G32B32A32_FLOAT,
        ImageFormat::Rgb10a2Unorm => DXGI_FORMAT_R10G10B10A2_UNORM,
        ImageFormat::Rg11b10Ufloat => DXGI_FORMAT_R11G11B10_FLOAT,
        ImageFormat::Depth16Unorm => DXGI_FORMAT_D16_UNORM,
        ImageFormat::Depth32Float => DXGI_FORMAT_D32_FLOAT,
        ImageFormat::Depth24UnormStencil8 => DXGI_FORMAT_D24_UNORM_S8_UINT,
        ImageFormat::Depth32FloatStencil8 => DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
    }
}

/// The format a depth image's resource is created with when shaders sample
/// it too: a depth format itself has no shader-resource view.
fn typeless_format(format: ImageFormat) -> DXGI_FORMAT {
    match format {
        ImageFormat::Depth16Unorm => DXGI_FORMAT_R16_TYPELESS,
        ImageFormat::Depth32Float => DXGI_FORMAT_R32_TYPELESS,
        ImageFormat::Depth24UnormStencil8 => DXGI_FORMAT_R24G8_TYPELESS,
        ImageFormat::Depth32FloatStencil8 => DXGI_FORMAT_R32G8X24_TYPELESS,
        other => dxgi_format(other),
    }
}

/// The format shaders read one plane of an image through, and which plane
/// of the resource that is.
fn shader_format(format: ImageFormat, aspect: ImageAspect) -> (DXGI_FORMAT, u32) {
    match (format, aspect) {
        (ImageFormat::Depth16Unorm, _) => (DXGI_FORMAT_R16_UNORM, 0),
        (ImageFormat::Depth32Float, _) => (DXGI_FORMAT_R32_FLOAT, 0),
        (ImageFormat::Depth24UnormStencil8, ImageAspect::Stencil) => {
            (DXGI_FORMAT_X24_TYPELESS_G8_UINT, 1)
        }
        (ImageFormat::Depth24UnormStencil8, _) => (DXGI_FORMAT_R24_UNORM_X8_TYPELESS, 0),
        (ImageFormat::Depth32FloatStencil8, ImageAspect::Stencil) => {
            (DXGI_FORMAT_X32_TYPELESS_G8X24_UINT, 1)
        }
        (ImageFormat::Depth32FloatStencil8, _) => (DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, 0),
        (other, _) => (dxgi_format(other), 0),
    }
}

/// One owned placed texture whose heap comes from the shared device allocator.
pub(crate) struct Dx12Image {
    allocator: Arc<Mutex<Allocator>>,
    resource: std::mem::ManuallyDrop<Resource>,
}

/// One single-descriptor RTV or DSV heap kept alive with its view.
struct Dx12TargetView {
    _heap: ID3D12DescriptorHeap,
    handle_pointer: usize,
}

impl Dx12Device {
    /// Whether images of `format` can be created for every one of `usages`.
    pub(crate) fn supports_sample_count(&self, format: ImageFormat, sample_count: u32) -> bool {
        let mut data = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
            Format: dxgi_format(format),
            SampleCount: sample_count,
            ..Default::default()
        };
        // SAFETY: the pointer and size describe `data`, which matches the
        // queried feature.
        let answered = unsafe {
            self.native.CheckFeatureSupport(
                D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                (&raw mut data).cast(),
                std::mem::size_of::<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS>() as u32,
            )
        };
        answered.is_ok() && data.NumQualityLevels > 0
    }

    pub(crate) fn format_supports(&self, format: ImageFormat, usages: ImageUsages) -> bool {
        let support = |format: DXGI_FORMAT| {
            let mut data = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
                Format: format,
                ..Default::default()
            };
            // SAFETY: the pointer and size describe `data`, which matches
            // the queried feature.
            let answered = unsafe {
                self.native.CheckFeatureSupport(
                    D3D12_FEATURE_FORMAT_SUPPORT,
                    (&raw mut data).cast(),
                    std::mem::size_of::<D3D12_FEATURE_DATA_FORMAT_SUPPORT>() as u32,
                )
            };
            if answered.is_ok() {
                data.Support1
            } else {
                D3D12_FORMAT_SUPPORT1(0)
            }
        };
        let has = |held: D3D12_FORMAT_SUPPORT1, flag: D3D12_FORMAT_SUPPORT1| held.0 & flag.0 != 0;
        let own = support(dxgi_format(format));
        if !has(own, D3D12_FORMAT_SUPPORT1_TEXTURE2D) {
            return false;
        }
        // A depth image is read by shaders through another format.
        let read = support(shader_format(format, ImageAspect::Depth).0);
        (!usages.contains(ImageUsages::SAMPLED) || has(read, D3D12_FORMAT_SUPPORT1_SHADER_LOAD))
            && (!usages.contains(ImageUsages::STORAGE)
                || has(own, D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW))
            && (!usages.contains(ImageUsages::RENDER_TARGET)
                || has(own, D3D12_FORMAT_SUPPORT1_RENDER_TARGET))
            && (!usages.contains(ImageUsages::DEPTH_TARGET)
                || has(own, D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL))
    }

    pub(crate) fn create_image(&self, descriptor: ImageDescriptor) -> Result<Dx12Image, RhiError> {
        let dimension = match descriptor.dimension {
            ImageDimension::D2 => D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            ImageDimension::D3 => D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        };
        let depth_or_layers = u16::try_from(descriptor.depth_or_layers).map_err(|_| {
            RhiError::InvalidImageDimensions {
                width: descriptor.width,
                height: descriptor.height,
                depth_or_layers: descriptor.depth_or_layers,
            }
        })?;
        // (Validated: an image has at most fifteen mips.)
        let mip_count = descriptor.mip_count as u16;
        let resource_description = D3D12_RESOURCE_DESC {
            Dimension: dimension,
            Alignment: 0,
            Width: u64::from(descriptor.width),
            Height: descriptor.height,
            DepthOrArraySize: depth_or_layers,
            MipLevels: mip_count,
            Format: if descriptor.usages.contains(ImageUsages::SAMPLED) {
                typeless_format(descriptor.format)
            } else {
                dxgi_format(descriptor.format)
            },
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: descriptor.sample_count,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: {
                let mut flags = D3D12_RESOURCE_FLAG_NONE;
                if descriptor.usages.contains(ImageUsages::DEPTH_TARGET) {
                    flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
                }
                if descriptor.usages.contains(ImageUsages::STORAGE) {
                    flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                }
                if descriptor.usages.contains(ImageUsages::RENDER_TARGET) {
                    flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
                }
                flags
            },
        };
        let resource = self
            .lock_allocator()
            .create_resource(&ResourceCreateDesc {
                name: "erhi image",
                memory_location: MemoryLocation::GpuOnly,
                resource_category: ResourceCategory::OtherTexture,
                resource_desc: &resource_description,
                castable_formats: &[],
                clear_value: None,
                initial_state_or_layout: ResourceStateOrBarrierLayout::ResourceState(
                    D3D12_RESOURCE_STATE_COMMON,
                ),
                resource_type: &ResourceType::Placed,
            })
            .map_err(|error| self.allocation_error("create placed texture", error.to_string()))?;
        Ok(Dx12Image {
            allocator: self.allocator(),
            resource: std::mem::ManuallyDrop::new(resource),
        })
    }
}

impl Dx12Image {
    pub(crate) fn allocation_size(&self) -> u64 {
        self.resource.size
    }

    pub(crate) fn native(&self, descriptor: ImageDescriptor) -> crate::NativeImage {
        use windows::core::Interface;
        // SAFETY: reading the description of a live resource.
        let format = unsafe { self.resource().GetDesc() }.Format.0 as u32;
        let _ = descriptor;
        crate::NativeImage::Dx12 {
            resource: self.resource().as_raw() as usize,
            format,
        }
    }

    pub(crate) fn resource(&self) -> &ID3D12Resource {
        self.resource.resource()
    }

    pub(crate) fn create_view(
        &self,
        descriptor: ImageDescriptor,
        range: ImageViewRange,
    ) -> Dx12ImageView {
        // Direct3D 12 has no standalone view object: descriptors are written
        // into heaps when a set is bound or a pass begun. The view records
        // what those writes need.
        let (format, plane) = shader_format(descriptor.format, range.aspect);
        Dx12ImageView {
            format,
            plane,
            range,
            target: OnceLock::new(),
        }
    }
}

impl Drop for Dx12Image {
    fn drop(&mut self) {
        // SAFETY: this is the drop, so `resource` is taken once and never
        // used again.
        let resource = unsafe { std::mem::ManuallyDrop::take(&mut self.resource) };
        let _ = lock_allocator(&self.allocator).free_resource(resource);
    }
}

/// What descriptor writes of a view need, and the view's render-target or
/// depth-stencil descriptor once a pass has drawn into it.
pub(crate) struct Dx12ImageView {
    /// The format shaders read the view through.
    pub(crate) format: DXGI_FORMAT,
    /// The plane of the resource they read (1 for stencil).
    pub(crate) plane: u32,
    pub(crate) range: ImageViewRange,
    /// Created on first use and kept for the view's lifetime, so no
    /// descriptor heap is created per frame.
    target: OnceLock<Dx12TargetView>,
}

impl Dx12ImageView {
    /// The descriptor a render pass binds this view's first mip through: a
    /// render-target view, or a depth-stencil view for a depth image.
    pub(crate) fn target_handle(
        &self,
        device: &ID3D12Device5,
        image: &Dx12Image,
        descriptor: ImageDescriptor,
    ) -> Result<D3D12_CPU_DESCRIPTOR_HANDLE, RhiError> {
        if let Some(view) = self.target.get() {
            return Ok(D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: view.handle_pointer,
            });
        }
        let depth = descriptor.format.is_depth();
        let description = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: if depth {
                D3D12_DESCRIPTOR_HEAP_TYPE_DSV
            } else {
                D3D12_DESCRIPTOR_HEAP_TYPE_RTV
            },
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        // SAFETY: the descriptor is fully initialized and the device is live.
        let heap: ID3D12DescriptorHeap = unsafe {
            device
                .CreateDescriptorHeap(&raw const description)
                .map_err(|error| {
                    RhiError::dx12("ID3D12Device::CreateDescriptorHeap(target)", error)
                })?
        };
        // SAFETY: reading the heap start has no side effects.
        let handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let range = self.range;
        // A view of one layer of an array is still an array view of one.
        let array = descriptor.layer_count() > 1;
        // (A multisampled image is one layer of one mip: its view names no
        // subresource at all.)
        let multisampled = descriptor.sample_count > 1;
        if depth {
            let view = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: dxgi_format(descriptor.format),
                ViewDimension: if multisampled {
                    D3D12_DSV_DIMENSION_TEXTURE2DMS
                } else if array {
                    D3D12_DSV_DIMENSION_TEXTURE2DARRAY
                } else {
                    D3D12_DSV_DIMENSION_TEXTURE2D
                },
                Flags: Default::default(),
                Anonymous: if multisampled {
                    D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                        Texture2DMS: Default::default(),
                    }
                } else if array {
                    D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                        Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                            MipSlice: range.base_mip,
                            FirstArraySlice: range.base_layer,
                            ArraySize: range.layer_count,
                        },
                    }
                } else {
                    D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_DSV {
                            MipSlice: range.base_mip,
                        },
                    }
                },
            };
            // SAFETY: the view names the resource's own depth format and
            // subresources inside it; the one-entry heap is live.
            unsafe { device.CreateDepthStencilView(image.resource(), Some(&view), handle) };
        } else {
            let view = D3D12_RENDER_TARGET_VIEW_DESC {
                Format: dxgi_format(descriptor.format),
                ViewDimension: match descriptor.dimension {
                    ImageDimension::D3 => D3D12_RTV_DIMENSION_TEXTURE3D,
                    ImageDimension::D2 if multisampled => D3D12_RTV_DIMENSION_TEXTURE2DMS,
                    ImageDimension::D2 if array => D3D12_RTV_DIMENSION_TEXTURE2DARRAY,
                    ImageDimension::D2 => D3D12_RTV_DIMENSION_TEXTURE2D,
                },
                Anonymous: match descriptor.dimension {
                    ImageDimension::D3 => D3D12_RENDER_TARGET_VIEW_DESC_0 {
                        Texture3D: D3D12_TEX3D_RTV {
                            MipSlice: range.base_mip,
                            FirstWSlice: 0,
                            WSize: u32::MAX,
                        },
                    },
                    ImageDimension::D2 if multisampled => D3D12_RENDER_TARGET_VIEW_DESC_0 {
                        Texture2DMS: Default::default(),
                    },
                    ImageDimension::D2 if array => D3D12_RENDER_TARGET_VIEW_DESC_0 {
                        Texture2DArray: D3D12_TEX2D_ARRAY_RTV {
                            MipSlice: range.base_mip,
                            FirstArraySlice: range.base_layer,
                            ArraySize: range.layer_count,
                            PlaneSlice: 0,
                        },
                    },
                    ImageDimension::D2 => D3D12_RENDER_TARGET_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_RTV {
                            MipSlice: range.base_mip,
                            PlaneSlice: 0,
                        },
                    },
                },
            };
            // SAFETY: the view names the resource's own format and
            // subresources inside it; the one-entry heap is live.
            unsafe { device.CreateRenderTargetView(image.resource(), Some(&view), handle) };
        }
        let view = self.target.get_or_init(|| Dx12TargetView {
            _heap: heap,
            handle_pointer: handle.ptr,
        });
        Ok(D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: view.handle_pointer,
        })
    }
}
