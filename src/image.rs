use std::fmt;
use std::ops::{BitOr, BitOrAssign};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::device::BackendDevice;
#[cfg(all(feature = "dx12", windows))]
use crate::dx12::{Dx12Image, Dx12ImageView};
#[cfg(feature = "vulkan")]
use crate::vulkan::{VulkanImage, VulkanImageView};
use crate::{Backend, Device, ResourceState, RhiError};

/// Buffer row alignment shared by both backends for image copies.
///
/// Direct3D 12 requires 256-byte row pitch for buffer/texture copies; the
/// Vulkan backend adopts the same pitch so readback bytes are identical on
/// both APIs.
pub const IMAGE_COPY_ROW_ALIGNMENT: u64 = 256;

/// Buffer offset alignment shared by both backends for image copies.
pub const IMAGE_COPY_OFFSET_ALIGNMENT: u64 = 512;

/// Texel formats, the same list on both backends.
///
/// Names give the channels, the bits of each and how they are read: `Unorm`
/// and `Snorm` as fractions, `Uint` and `Sint` as whole numbers, `Float` as
/// floats, `Srgb` as `Unorm` whose colour channels are decoded from sRGB when
/// sampled and encoded when rendered to. Whether a device can use a format
/// for a purpose is asked of [`Device::format_supports`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ImageFormat {
    /// BC1 block compression: colour with one bit of alpha, eight bytes to a
    /// block of four by four texels.
    Bc1Unorm,
    /// BC1 block compression: colour with one bit of alpha, eight bytes to a
    /// block of four by four texels, decoded from sRGB when sampled.
    Bc1UnormSrgb,
    /// BC2 block compression: colour with four bits of alpha a texel, sixteen
    /// bytes to a block of four by four texels.
    Bc2Unorm,
    /// BC2 block compression: colour with four bits of alpha a texel, sixteen
    /// bytes to a block of four by four texels, decoded from sRGB when
    /// sampled.
    Bc2UnormSrgb,
    /// 4x4 RGBA blocks, 16 bytes per block.
    Bc3Unorm,
    /// BC3 block compression: colour with smooth alpha, sixteen bytes to a
    /// block of four by four texels, decoded from sRGB when sampled.
    Bc3UnormSrgb,
    /// BC4 block compression: one channel, eight bytes to a block of four by
    /// four texels.
    Bc4Unorm,
    /// BC4 block compression: one channel, eight bytes to a block of four by
    /// four texels, read as numbers from -1 to 1.
    Bc4Snorm,
    /// BC5 block compression: two channels (normal maps), sixteen bytes to a
    /// block of four by four texels.
    Bc5Unorm,
    /// BC5 block compression: two channels (normal maps), sixteen bytes to a
    /// block of four by four texels, read as numbers from -1 to 1.
    Bc5Snorm,
    /// BC6H block compression: high-dynamic-range colour without alpha,
    /// sixteen bytes to a block of four by four texels, of unsigned floats.
    Bc6hUfloat,
    /// BC6H block compression: high-dynamic-range colour without alpha,
    /// sixteen bytes to a block of four by four texels, of signed floats.
    Bc6hSfloat,
    /// BC7 block compression: colour with alpha at high quality, sixteen bytes
    /// to a block of four by four texels.
    Bc7Unorm,
    /// BC7 block compression: colour with alpha at high quality, sixteen bytes
    /// to a block of four by four texels, decoded from sRGB when sampled.
    Bc7UnormSrgb,
    /// Single-channel coverage, used by the UI glyph atlas.
    R8Unorm,
    /// One channel of 8-bit signed integers read as numbers from -1 to 1.
    R8Snorm,
    /// One channel of 8-bit unsigned integers, read as integers.
    R8Uint,
    /// One channel of 8-bit signed integers, read as integers.
    R8Sint,
    /// Two channels of 8-bit unsigned integers read as numbers from 0 to 1.
    Rg8Unorm,
    /// Two channels of 8-bit signed integers read as numbers from -1 to 1.
    Rg8Snorm,
    /// Two channels of 8-bit unsigned integers, read as integers.
    Rg8Uint,
    /// Two channels of 8-bit signed integers, read as integers.
    Rg8Sint,
    /// Four channels of 8-bit unsigned integers read as numbers from 0 to 1.
    Rgba8Unorm,
    /// Four channels of 8-bit unsigned integers read as numbers from 0 to 1,
    /// the colour ones decoded from sRGB when sampled and encoded when drawn.
    Rgba8UnormSrgb,
    /// Four channels of 8-bit signed integers read as numbers from -1 to 1.
    Rgba8Snorm,
    /// Four channels of 8-bit unsigned integers, read as integers.
    Rgba8Uint,
    /// Four channels of 8-bit signed integers, read as integers.
    Rgba8Sint,
    /// Four channels, blue first in memory, of 8-bit unsigned integers read as
    /// numbers from 0 to 1.
    Bgra8Unorm,
    /// Four channels, blue first in memory, of 8-bit unsigned integers read as
    /// numbers from 0 to 1, the colour ones decoded from sRGB when sampled and
    /// encoded when drawn.
    Bgra8UnormSrgb,
    /// One channel of 16-bit unsigned integers read as numbers from 0 to 1.
    R16Unorm,
    /// One channel of 16-bit signed integers read as numbers from -1 to 1.
    R16Snorm,
    /// One channel of 16-bit unsigned integers, read as integers.
    R16Uint,
    /// One channel of 16-bit signed integers, read as integers.
    R16Sint,
    /// One channel of 16-bit floats.
    R16Float,
    /// Two channels of 16-bit unsigned integers read as numbers from 0 to 1.
    Rg16Unorm,
    /// Two channels of 16-bit signed integers read as numbers from -1 to 1.
    Rg16Snorm,
    /// Two channels of 16-bit unsigned integers, read as integers.
    Rg16Uint,
    /// Two channels of 16-bit signed integers, read as integers.
    Rg16Sint,
    /// Two channels of 16-bit floats.
    Rg16Float,
    /// Four channels of 16-bit unsigned integers read as numbers from 0 to 1.
    Rgba16Unorm,
    /// Four channels of 16-bit signed integers read as numbers from -1 to 1.
    Rgba16Snorm,
    /// Four channels of 16-bit unsigned integers, read as integers.
    Rgba16Uint,
    /// Four channels of 16-bit signed integers, read as integers.
    Rgba16Sint,
    /// Four channels of 16-bit floats.
    Rgba16Float,
    /// One channel of 32-bit unsigned integers, read as integers.
    R32Uint,
    /// One channel of 32-bit signed integers, read as integers.
    R32Sint,
    /// Full precision scalar data, including linear depth pyramids.
    R32Float,
    /// Two channels of 32-bit unsigned integers, read as integers.
    Rg32Uint,
    /// Two channels of 32-bit signed integers, read as integers.
    Rg32Sint,
    /// Two channels of 32-bit floats.
    Rg32Float,
    /// Four whole 32-bit words a texel: packed records a shader reads and
    /// writes as they are.
    Rgba32Uint,
    /// Four channels of 32-bit signed integers, read as integers.
    Rgba32Sint,
    /// Four channels of 32-bit floats.
    Rgba32Float,
    /// Ten bits a colour channel and two of alpha, in one word.
    Rgb10a2Unorm,
    /// Unsigned floats of 11, 11 and 10 bits in one word: HDR colour.
    Rg11b10Ufloat,
    /// Depth alone, as 16-bit unsigned integers read as numbers from 0 to 1.
    Depth16Unorm,
    /// Depth alone, as 32-bit floats.
    Depth32Float,
    /// 24 bits of depth and 8 of stencil. Not on every Vulkan device: ask
    /// [`Device::format_supports`], and use [`Self::Depth32FloatStencil8`] where
    /// it is refused.
    Depth24UnormStencil8,
    /// Depth as 32-bit floats, with eight bits of stencil.
    Depth32FloatStencil8,
}

impl ImageFormat {
    /// Bytes of one block: one texel, or one 4x4 block of a compressed
    /// format. For the two formats with stencil, the bytes of the depth
    /// plane and its stencil together as a copy would carry them (copies of
    /// those to and from buffers are not offered).
    #[must_use]
    pub const fn bytes_per_block(self) -> u32 {
        match self {
            Self::R8Unorm | Self::R8Snorm | Self::R8Uint | Self::R8Sint => 1,
            Self::Rg8Unorm
            | Self::Rg8Snorm
            | Self::Rg8Uint
            | Self::Rg8Sint
            | Self::R16Unorm
            | Self::R16Snorm
            | Self::R16Uint
            | Self::R16Sint
            | Self::R16Float
            | Self::Depth16Unorm => 2,
            Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Rgba8Snorm
            | Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            | Self::Rg16Unorm
            | Self::Rg16Snorm
            | Self::Rg16Uint
            | Self::Rg16Sint
            | Self::Rg16Float
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float
            | Self::Rgb10a2Unorm
            | Self::Rg11b10Ufloat
            | Self::Depth32Float
            | Self::Depth24UnormStencil8 => 4,
            Self::Bc1Unorm
            | Self::Bc1UnormSrgb
            | Self::Bc4Unorm
            | Self::Bc4Snorm
            | Self::Rgba16Unorm
            | Self::Rgba16Snorm
            | Self::Rgba16Uint
            | Self::Rgba16Sint
            | Self::Rgba16Float
            | Self::Rg32Uint
            | Self::Rg32Sint
            | Self::Rg32Float
            | Self::Depth32FloatStencil8 => 8,
            Self::Bc2Unorm
            | Self::Bc2UnormSrgb
            | Self::Bc3Unorm
            | Self::Bc3UnormSrgb
            | Self::Bc5Unorm
            | Self::Bc5Snorm
            | Self::Bc6hUfloat
            | Self::Bc6hSfloat
            | Self::Bc7Unorm
            | Self::Bc7UnormSrgb
            | Self::Rgba32Uint
            | Self::Rgba32Sint
            | Self::Rgba32Float => 16,
        }
    }

    /// Texels along one side of a block: 4 for the compressed formats.
    #[must_use]
    pub const fn block_extent(self) -> u32 {
        if self.is_compressed() { 4 } else { 1 }
    }

    /// Whether the format is block compressed (the BC family).
    #[must_use]
    pub const fn is_compressed(self) -> bool {
        matches!(
            self,
            Self::Bc1Unorm
                | Self::Bc1UnormSrgb
                | Self::Bc2Unorm
                | Self::Bc2UnormSrgb
                | Self::Bc3Unorm
                | Self::Bc3UnormSrgb
                | Self::Bc4Unorm
                | Self::Bc4Snorm
                | Self::Bc5Unorm
                | Self::Bc5Snorm
                | Self::Bc6hUfloat
                | Self::Bc6hSfloat
                | Self::Bc7Unorm
                | Self::Bc7UnormSrgb
        )
    }

    /// Whether the format holds depth (with or without stencil).
    #[must_use]
    pub const fn is_depth(self) -> bool {
        matches!(
            self,
            Self::Depth16Unorm
                | Self::Depth32Float
                | Self::Depth24UnormStencil8
                | Self::Depth32FloatStencil8
        )
    }

    /// Whether the format holds a stencil plane beside its depth.
    #[must_use]
    pub const fn has_stencil(self) -> bool {
        matches!(
            self,
            Self::Depth24UnormStencil8 | Self::Depth32FloatStencil8
        )
    }

    /// Bytes of one block of `aspect` as a buffer holds it in a copy, or
    /// `None` where that is not copied to or from buffers: both planes of a
    /// depth-stencil format at once (name one), and the depth of
    /// [`Self::Depth24UnormStencil8`], whose bits differ between drivers.
    /// Depth is copied as it is stored (`f32` or `u16`), stencil as bytes.
    #[must_use]
    pub const fn copy_bytes(self, aspect: ImageAspect) -> Option<u32> {
        match aspect {
            ImageAspect::All if self.has_stencil() => None,
            ImageAspect::All => Some(self.bytes_per_block()),
            ImageAspect::Depth => match self {
                Self::Depth16Unorm => Some(2),
                Self::Depth32Float | Self::Depth32FloatStencil8 => Some(4),
                _ => None,
            },
            ImageAspect::Stencil if self.has_stencil() => Some(1),
            ImageAspect::Stencil => None,
        }
    }

    /// Whether colour is stored sRGB-encoded.
    #[must_use]
    pub const fn is_srgb(self) -> bool {
        matches!(
            self,
            Self::Bc1UnormSrgb
                | Self::Bc2UnormSrgb
                | Self::Bc3UnormSrgb
                | Self::Bc7UnormSrgb
                | Self::Rgba8UnormSrgb
                | Self::Bgra8UnormSrgb
        )
    }

    /// Whether a shader reads the format as whole numbers (`Uint`, `Sint`):
    /// such an image is loaded, never filtered.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::R8Uint
                | Self::R8Sint
                | Self::Rg8Uint
                | Self::Rg8Sint
                | Self::Rgba8Uint
                | Self::Rgba8Sint
                | Self::R16Uint
                | Self::R16Sint
                | Self::Rg16Uint
                | Self::Rg16Sint
                | Self::Rgba16Uint
                | Self::Rgba16Sint
                | Self::R32Uint
                | Self::R32Sint
                | Self::Rg32Uint
                | Self::Rg32Sint
                | Self::Rgba32Uint
                | Self::Rgba32Sint
        )
    }

    /// Every format, for tables and tests.
    pub const ALL: [Self; 59] = [
        Self::Bc1Unorm,
        Self::Bc1UnormSrgb,
        Self::Bc2Unorm,
        Self::Bc2UnormSrgb,
        Self::Bc3Unorm,
        Self::Bc3UnormSrgb,
        Self::Bc4Unorm,
        Self::Bc4Snorm,
        Self::Bc5Unorm,
        Self::Bc5Snorm,
        Self::Bc6hUfloat,
        Self::Bc6hSfloat,
        Self::Bc7Unorm,
        Self::Bc7UnormSrgb,
        Self::R8Unorm,
        Self::R8Snorm,
        Self::R8Uint,
        Self::R8Sint,
        Self::Rg8Unorm,
        Self::Rg8Snorm,
        Self::Rg8Uint,
        Self::Rg8Sint,
        Self::Rgba8Unorm,
        Self::Rgba8UnormSrgb,
        Self::Rgba8Snorm,
        Self::Rgba8Uint,
        Self::Rgba8Sint,
        Self::Bgra8Unorm,
        Self::Bgra8UnormSrgb,
        Self::R16Unorm,
        Self::R16Snorm,
        Self::R16Uint,
        Self::R16Sint,
        Self::R16Float,
        Self::Rg16Unorm,
        Self::Rg16Snorm,
        Self::Rg16Uint,
        Self::Rg16Sint,
        Self::Rg16Float,
        Self::Rgba16Unorm,
        Self::Rgba16Snorm,
        Self::Rgba16Uint,
        Self::Rgba16Sint,
        Self::Rgba16Float,
        Self::R32Uint,
        Self::R32Sint,
        Self::R32Float,
        Self::Rg32Uint,
        Self::Rg32Sint,
        Self::Rg32Float,
        Self::Rgba32Uint,
        Self::Rgba32Sint,
        Self::Rgba32Float,
        Self::Rgb10a2Unorm,
        Self::Rg11b10Ufloat,
        Self::Depth16Unorm,
        Self::Depth32Float,
        Self::Depth24UnormStencil8,
        Self::Depth32FloatStencil8,
    ];
}

/// Whether an image's texels lie in a plane or fill a volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageDimension {
    /// Texels in a plane, in one or more layers:
    /// [`ImageDescriptor::depth_or_layers`] is the number of layers. Six
    /// layers can be shown as a cube.
    D2,
    /// Texels filling a volume: [`ImageDescriptor::depth_or_layers`] is its
    /// depth.
    D3,
}

/// Usage flags for an image.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageUsages(u32);

impl ImageUsages {
    /// Written, or read and written, by a shader through a storage-image
    /// binding.
    pub const STORAGE: Self = Self(1 << 0);

    /// Read by a shader through a sampled-image binding.
    pub const SAMPLED: Self = Self(1 << 1);

    /// Copied from: to a buffer, to another image, or to a window to be
    /// presented.
    pub const COPY_SOURCE: Self = Self(1 << 2);

    /// Copied into, from a buffer or another image.
    pub const COPY_DESTINATION: Self = Self(1 << 3);

    /// Drawn into as a colour attachment of a render pass.
    pub const RENDER_TARGET: Self = Self(1 << 4);

    /// Drawn into as the depth attachment of a render pass.
    pub const DEPTH_TARGET: Self = Self(1 << 5);

    /// Whether every usage of `other` is among these.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether there are no usages at all.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for ImageUsages {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for ImageUsages {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Explicit dimensions, format, and usage of one owned GPU image.
///
/// For `D2`, `depth_or_layers` is the array layer count; for `D3` it is the
/// depth extent.
/// The longest side of a 2D image, of a 3D image, and the most layers:
/// the Direct3D 12 limit, which Vulkan devices of this class match or pass.
pub const MAX_IMAGE_EXTENT: u32 = 16_384;

/// The longest side of a 3D image, in texels.
pub const MAX_IMAGE_EXTENT_3D: u32 = 2_048;

/// The most layers a 2D image has.
pub const MAX_IMAGE_LAYERS: u32 = 2_048;

/// Describes an image: its shape, its format, and everything it will be used
/// for. [`Device::create_image`] makes one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImageDescriptor {
    /// Whether the texels lie in a plane or fill a volume.
    pub dimension: ImageDimension,
    /// What a texel is.
    pub format: ImageFormat,
    /// The width of the largest mip, in texels.
    pub width: u32,
    /// The height of the largest mip, in texels.
    pub height: u32,
    /// The depth of a 3D image, or the number of layers of a 2D one (six a
    /// cube).
    pub depth_or_layers: u32,
    /// How many mips there are, each half the size of the one before: at least
    /// one.
    pub mip_count: u32,
    /// Samples a texel: 1, or 2, 4, 8 or 16 for a multisampled render or
    /// depth target (ask [`Device::supports_sample_count`]). A multisampled
    /// image is one 2D layer of one mip, drawn into and then resolved into
    /// an ordinary image by the pass ([`crate::ColorAttachment::resolve`]).
    pub sample_count: u32,
    /// Everything the image will be used for. A use not named here is refused
    /// later.
    pub usages: ImageUsages,
}

impl ImageDescriptor {
    /// An image of one mip and one sample a texel. `depth_or_layers` is the
    /// depth of a 3D image and the layer count of a 2D one.
    #[must_use]
    pub const fn new(
        dimension: ImageDimension,
        format: ImageFormat,
        width: u32,
        height: u32,
        depth_or_layers: u32,
        usages: ImageUsages,
    ) -> Self {
        Self {
            dimension,
            format,
            width,
            height,
            depth_or_layers,
            mip_count: 1,
            sample_count: 1,
            usages,
        }
    }

    /// A 2D image of one layer, one mip and one sample a texel.
    #[must_use]
    pub const fn d2(format: ImageFormat, width: u32, height: u32, usages: ImageUsages) -> Self {
        Self::new(ImageDimension::D2, format, width, height, 1, usages)
    }

    /// A 3D image of one mip.
    #[must_use]
    pub const fn d3(
        format: ImageFormat,
        width: u32,
        height: u32,
        depth: u32,
        usages: ImageUsages,
    ) -> Self {
        Self::new(ImageDimension::D3, format, width, height, depth, usages)
    }

    /// The same 2D image with `layers` layers: six make a cube.
    #[must_use]
    pub const fn with_layers(mut self, layers: u32) -> Self {
        self.depth_or_layers = layers;
        self
    }

    /// The same image with `mip_count` mips.
    #[must_use]
    pub const fn with_mip_count(mut self, mip_count: u32) -> Self {
        self.mip_count = mip_count;
        self
    }

    /// The same image with `sample_count` samples a texel.
    #[must_use]
    pub const fn with_sample_count(mut self, sample_count: u32) -> Self {
        self.sample_count = sample_count;
        self
    }
}

impl ImageDescriptor {
    pub(crate) fn validate(&self) -> Result<(), RhiError> {
        if !matches!(self.sample_count, 1 | 2 | 4 | 8 | 16) {
            return Err(RhiError::InvalidImageSampleCount {
                reason: "an image has 1, 2, 4, 8 or 16 samples a texel",
            });
        }
        if self.sample_count > 1
            && (self.dimension != ImageDimension::D2
                || self.depth_or_layers != 1
                || self.mip_count != 1
                || self.format.is_compressed()
                || self.usages.contains(ImageUsages::STORAGE)
                || !(self.usages.contains(ImageUsages::RENDER_TARGET)
                    || self.usages.contains(ImageUsages::DEPTH_TARGET)))
        {
            return Err(RhiError::InvalidImageSampleCount {
                reason: "a multisampled image is one uncompressed 2D layer of one mip,                          a render or depth target, and not written as storage",
            });
        }
        if self.width == 0 || self.height == 0 || self.depth_or_layers == 0 {
            return Err(RhiError::InvalidImageDimensions {
                width: self.width,
                height: self.height,
                depth_or_layers: self.depth_or_layers,
            });
        }
        if self.usages.is_empty() {
            return Err(RhiError::EmptyImageUsages);
        }
        // No larger than both APIs take.
        let too_large = self.width.max(self.height) > MAX_IMAGE_EXTENT
            || match self.dimension {
                ImageDimension::D2 => self.depth_or_layers > MAX_IMAGE_LAYERS,
                ImageDimension::D3 => {
                    self.width.max(self.height).max(self.depth_or_layers) > MAX_IMAGE_EXTENT_3D
                }
            };
        if too_large {
            return Err(RhiError::InvalidImageDimensions {
                width: self.width,
                height: self.height,
                depth_or_layers: self.depth_or_layers,
            });
        }
        // What its format can be used for at all.
        let invalid = |reason| Err(RhiError::InvalidImageUsage { reason });
        let (depth, compressed) = (self.format.is_depth(), self.format.is_compressed());
        if self.usages.contains(ImageUsages::DEPTH_TARGET) && !depth {
            return invalid("a depth target needs a depth format");
        }
        if self.usages.contains(ImageUsages::RENDER_TARGET) && (depth || compressed) {
            return invalid("a render target needs an uncompressed color format");
        }
        if depth && self.dimension == ImageDimension::D3 {
            return invalid("a depth format is for 2D images");
        }
        if depth && self.usages.contains(ImageUsages::STORAGE) {
            return invalid("a depth format is not written as a storage image");
        }
        if compressed && self.usages.contains(ImageUsages::STORAGE) {
            return invalid("a block-compressed format is not written as a storage image");
        }
        let block = self.format.block_extent();
        if compressed && !(self.width.is_multiple_of(block) && self.height.is_multiple_of(block)) {
            return invalid("a block-compressed image is a whole number of blocks wide and high");
        }
        let maximum = self.max_mip_count();
        if self.mip_count == 0 || self.mip_count > maximum {
            return Err(RhiError::InvalidImageMipCount {
                mip_count: self.mip_count,
                maximum,
            });
        }
        Ok(())
    }

    /// Number of mip levels until the largest extent reaches one texel.
    #[must_use]
    pub fn max_mip_count(&self) -> u32 {
        let mut largest = self.width.max(self.height);
        if self.dimension == ImageDimension::D3 {
            largest = largest.max(self.depth_or_layers);
        }
        32 - largest.leading_zeros()
    }

    /// The (width, height, depth) extent of one mip level.
    #[must_use]
    pub fn mip_extent(&self, mip: u32) -> (u32, u32, u32) {
        let depth = match self.dimension {
            ImageDimension::D2 => 1,
            ImageDimension::D3 => (self.depth_or_layers >> mip).max(1),
        };
        (
            (self.width >> mip).max(1),
            (self.height >> mip).max(1),
            depth,
        )
    }

    /// Array layer count seen by copies: layers for `D2`, one for `D3`.
    #[must_use]
    pub const fn layer_count(&self) -> u32 {
        match self.dimension {
            ImageDimension::D2 => self.depth_or_layers,
            ImageDimension::D3 => 1,
        }
    }
}

/// How a view presents its image to a shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageViewDimension {
    /// One layer, as a plain 2D image.
    D2,
    /// Several layers, as an array of 2D images.
    D2Array,
    /// Six layers of a square `D2` image as the faces of a cube, in the
    /// order +x, -x, +y, -y, +z, -z.
    Cube,
    /// Whole cubes of six layers each.
    CubeArray,
    /// A volume.
    D3,
}

/// Which planes of an image a view covers.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageAspect {
    /// The colour of a colour image, the depth of a depth image, and both
    /// planes of a depth-stencil image (what a render pass attaches).
    #[default]
    All,
    /// The depth plane alone: what a shader samples of a depth-stencil image.
    Depth,
    /// The stencil plane alone.
    Stencil,
}

/// The part of an image a view covers. The default is all of it, presented
/// as the image is: [`Image::create_full_view`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageViewDescriptor {
    /// `None`: `D3` for a 3D image, `D2Array` for a 2D image of several
    /// layers, `D2` for one of a single layer.
    pub dimension: Option<ImageViewDimension>,
    /// The first mip the view shows.
    pub base_mip: u32,
    /// `None`: every mip from `base_mip`.
    pub mip_count: Option<u32>,
    /// First array layer (always 0 for a 3D image).
    pub base_layer: u32,
    /// `None`: every layer from `base_layer`.
    pub layer_count: Option<u32>,
    /// Which planes of the image the view covers.
    pub aspect: ImageAspect,
}

/// A view's descriptor with nothing left to default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageViewRange {
    /// How the view presents its image to a shader.
    pub dimension: ImageViewDimension,
    /// The first mip the view shows.
    pub base_mip: u32,
    /// How many mips it shows.
    pub mip_count: u32,
    /// The first layer the view shows.
    pub base_layer: u32,
    /// How many layers it shows.
    pub layer_count: u32,
    /// Which planes of the image the view covers.
    pub aspect: ImageAspect,
}

impl ImageViewDescriptor {
    /// One mip of one layer: what a render pass draws into.
    #[must_use]
    pub const fn subresource(mip: u32, layer: u32) -> Self {
        Self {
            dimension: Some(ImageViewDimension::D2),
            base_mip: mip,
            mip_count: Some(1),
            base_layer: layer,
            layer_count: Some(1),
            aspect: ImageAspect::All,
        }
    }

    pub(crate) fn resolve(&self, image: &ImageDescriptor) -> Result<ImageViewRange, RhiError> {
        let invalid = |reason| RhiError::InvalidImageView { reason };
        let layers = image.layer_count();
        let mip_count = self
            .mip_count
            .unwrap_or_else(|| image.mip_count.saturating_sub(self.base_mip));
        let layer_count = self
            .layer_count
            .unwrap_or_else(|| layers.saturating_sub(self.base_layer));
        if mip_count == 0
            || self
                .base_mip
                .checked_add(mip_count)
                .is_none_or(|end| end > image.mip_count)
        {
            return Err(invalid("the mip range lies outside the image"));
        }
        if layer_count == 0
            || self
                .base_layer
                .checked_add(layer_count)
                .is_none_or(|end| end > layers)
        {
            return Err(invalid("the layer range lies outside the image"));
        }
        let dimension = self.dimension.unwrap_or(match image.dimension {
            ImageDimension::D3 => ImageViewDimension::D3,
            ImageDimension::D2 if layers > 1 => ImageViewDimension::D2Array,
            ImageDimension::D2 => ImageViewDimension::D2,
        });
        let fits = match (image.dimension, dimension) {
            (ImageDimension::D3, ImageViewDimension::D3) => true,
            (ImageDimension::D3, _) | (ImageDimension::D2, ImageViewDimension::D3) => false,
            (ImageDimension::D2, ImageViewDimension::D2) => layer_count == 1,
            (ImageDimension::D2, ImageViewDimension::D2Array) => true,
            (ImageDimension::D2, ImageViewDimension::Cube) => {
                layer_count == 6 && image.width == image.height
            }
            (ImageDimension::D2, ImageViewDimension::CubeArray) => {
                layer_count.is_multiple_of(6) && image.width == image.height
            }
        };
        if !fits {
            return Err(invalid(
                "the view dimension does not fit the image and its layer range",
            ));
        }
        let planes = match self.aspect {
            ImageAspect::All => true,
            ImageAspect::Depth => image.format.is_depth(),
            ImageAspect::Stencil => image.format.has_stencil(),
        };
        if !planes {
            return Err(invalid("the image's format has no such plane"));
        }
        Ok(ImageViewRange {
            dimension,
            base_mip: self.base_mip,
            mip_count,
            base_layer: self.base_layer,
            layer_count,
            aspect: self.aspect,
        })
    }
}

/// Byte layout of one mip level inside a copy-linked buffer.
///
/// `total_size` covers every slice and array layer of the mip level using the
/// shared 256-byte row pitch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageCopyLayout {
    /// From the start of one row of texels to the start of the next, in bytes:
    /// the row itself, padded.
    pub row_pitch: u64,
    /// How many rows one slice or layer has (rows of blocks, for a compressed
    /// format).
    pub rows_per_slice: u32,
    /// How many slices or layers the mip has.
    pub slice_count: u32,
    /// The size of a buffer that holds the whole mip, in bytes.
    pub total_size: u64,
}

/// A GPU image. Clones share the native object.
#[derive(Clone)]
pub struct Image {
    pub(crate) inner: Arc<ImageInner>,
}

/// Some mips of some layers of an image: what a view shows, a copy touches
/// or a pass draws into. Each mip of each layer is in its own state,
/// so that one may be read while another of the same image is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Subresources {
    pub(crate) base_mip: u32,
    pub(crate) mip_count: u32,
    pub(crate) base_layer: u32,
    pub(crate) layer_count: u32,
}

impl Subresources {
    pub(crate) const fn whole(image: &ImageDescriptor) -> Self {
        Self {
            base_mip: 0,
            mip_count: image.mip_count,
            base_layer: 0,
            layer_count: image.layer_count(),
        }
    }

    /// One mip, of every layer.
    pub(crate) const fn mip(image: &ImageDescriptor, mip: u32) -> Self {
        Self {
            base_mip: mip,
            mip_count: 1,
            base_layer: 0,
            layer_count: image.layer_count(),
        }
    }

    pub(crate) const fn one(mip: u32, layer: u32) -> Self {
        Self {
            base_mip: mip,
            mip_count: 1,
            base_layer: layer,
            layer_count: 1,
        }
    }

    /// What `view` shows.
    pub(crate) fn of(view: &ImageView) -> Self {
        let range = view.range();
        Self {
            base_mip: range.base_mip,
            mip_count: range.mip_count,
            base_layer: range.base_layer,
            layer_count: range.layer_count,
        }
    }

    pub(crate) const fn overlaps(self, other: Self) -> bool {
        self.base_mip < other.base_mip + other.mip_count
            && other.base_mip < self.base_mip + self.mip_count
            && self.base_layer < other.base_layer + other.layer_count
            && other.base_layer < self.base_layer + self.layer_count
    }

    /// Where the state of each mip of each layer is kept: mips of the first
    /// layer, then those of the second.
    pub(crate) fn indices(self, image: &ImageDescriptor) -> impl Iterator<Item = u32> + use<> {
        let mips = image.mip_count;
        (self.base_layer..self.base_layer + self.layer_count).flat_map(move |layer| {
            (self.base_mip..self.base_mip + self.mip_count).map(move |mip| mip + layer * mips)
        })
    }
}

pub(crate) struct ImageInner {
    device_id: u64,
    descriptor: ImageDescriptor,
    /// The state of each mip of each layer: see [`Subresources::indices`].
    states: Box<[AtomicU8]>,
    pub(crate) backend: ImageBackend,
    pub(crate) memory: crate::memory::MemoryTicket,
    // Worker results may outlive the renderer during device replacement.
    _device: BackendDevice,
}

pub(crate) enum ImageBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<VulkanImage>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Box<Dx12Image>),
}

impl fmt::Debug for Image {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Image")
            .field("backend", &self.backend())
            .field("descriptor", &self.inner.descriptor)
            .field("state", &self.current_state())
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Whether this device can create images of `format` for every one of
    /// `usages`. Most formats are everywhere; the ones to ask about are
    /// [`ImageFormat::Depth24UnormStencil8`], the compressed formats, and
    /// storage or render-target use of the less common ones.
    #[must_use]
    pub fn format_supports(&self, format: ImageFormat, usages: ImageUsages) -> bool {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => device.format_supports(format, usages),
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => device.format_supports(format, usages),
        }
    }

    /// Whether this device makes render targets (depth targets, for a
    /// depth format) of `format` with `sample_count` samples a texel.
    #[must_use]
    pub fn supports_sample_count(&self, format: ImageFormat, sample_count: u32) -> bool {
        if !matches!(sample_count, 1 | 2 | 4 | 8 | 16) || format.is_compressed() {
            return false;
        }
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => device.supports_sample_count(format, sample_count),
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => device.supports_sample_count(format, sample_count),
        }
    }

    /// Create one owned GPU image in the `Undefined` resource state.
    pub fn create_image(&self, descriptor: ImageDescriptor) -> Result<Image, RhiError> {
        descriptor.validate()?;
        // Asked of the device before it is asked to make it: made with a
        // format it cannot use so, an image is undefined in Vulkan.
        if !self.format_supports(descriptor.format, descriptor.usages) {
            return Err(RhiError::UnsupportedByDevice {
                reason: "the image's format does not serve these usages on this device",
            });
        }
        if descriptor.sample_count > 1
            && !self.supports_sample_count(descriptor.format, descriptor.sample_count)
        {
            return Err(RhiError::UnsupportedByDevice {
                reason: "the image's format is not multisampled so on this device",
            });
        }
        let backend = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => {
                ImageBackend::Vulkan(Box::new(device.create_image(descriptor)?))
            }
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => {
                ImageBackend::Dx12(Box::new(device.create_image(descriptor)?))
            }
        };
        let bytes = match &backend {
            #[cfg(feature = "vulkan")]
            ImageBackend::Vulkan(native) => native.allocation_size(),
            #[cfg(all(feature = "dx12", windows))]
            ImageBackend::Dx12(native) => native.allocation_size(),
        };
        let memory =
            crate::memory::MemoryTicket::new(&self.memory, crate::MemoryKind::Image, bytes);
        Ok(Image {
            inner: Arc::new(ImageInner {
                device_id: self.id(),
                memory,
                descriptor,
                states: (0..descriptor.mip_count * descriptor.layer_count())
                    .map(|_| AtomicU8::new(ResourceState::Undefined.bits()))
                    .collect(),
                backend,
                _device: self.backend.clone(),
            }),
        })
    }
}

impl Image {
    /// The descriptor the image was made from.
    #[must_use]
    pub fn descriptor(&self) -> ImageDescriptor {
        self.inner.descriptor
    }

    /// Name the image for capture tools (RenderDoc, PIX) and validation
    /// messages. Of no effect on what is drawn.
    pub fn set_label(&self, label: &str) {
        match (&self.inner._device, &self.inner.backend) {
            #[cfg(feature = "vulkan")]
            (BackendDevice::Vulkan(device), ImageBackend::Vulkan(image)) => {
                device.name_object(image.handle, label);
            }
            #[cfg(all(feature = "dx12", windows))]
            (BackendDevice::Dx12(_), ImageBackend::Dx12(image)) => {
                crate::dx12::name_object(image.resource(), label);
            }
            #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
            _ => {}
        }
    }

    /// The native resource and format, for third-party SDKs (see
    /// [`crate::NativeImage`]). Valid while this image lives.
    #[must_use]
    pub fn native(&self) -> crate::NativeImage {
        match &self.inner.backend {
            #[cfg(feature = "vulkan")]
            ImageBackend::Vulkan(image) => image.native(self.inner.descriptor),
            #[cfg(all(feature = "dx12", windows))]
            ImageBackend::Dx12(image) => image.native(self.inner.descriptor),
        }
    }

    /// The backend of the device that made this image.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match &self.inner.backend {
            #[cfg(feature = "vulkan")]
            ImageBackend::Vulkan(_) => Backend::Vulkan,
            #[cfg(all(feature = "dx12", windows))]
            ImageBackend::Dx12(_) => Backend::Dx12,
        }
    }

    /// How many mips of layers this image is: each has its own state.
    pub(crate) fn subresource_count(&self) -> u32 {
        self.inner.states.len() as u32
    }

    /// The state one mip of one layer was left in by the commands that have
    /// run, by index in [`Subresources::indices`].
    pub(crate) fn state_of(&self, index: u32) -> ResourceState {
        ResourceState::from_bits(self.inner.states[index as usize].load(Ordering::Acquire))
    }

    pub(crate) fn set_state_of(&self, index: u32, state: ResourceState) {
        self.inner.states[index as usize].store(state.bits(), Ordering::Release);
    }

    /// The state the whole image is in, where every mip of every layer is
    /// in the same one.
    #[must_use]
    pub(crate) fn current_state(&self) -> Option<ResourceState> {
        let first = self.state_of(0);
        (1..self.subresource_count())
            .all(|index| self.state_of(index) == first)
            .then_some(first)
    }

    pub(crate) fn device_id(&self) -> u64 {
        self.inner.device_id
    }

    /// Buffer layout used by copies of one mip level on either backend.
    pub fn copy_layout(&self, mip: u32) -> Result<ImageCopyLayout, RhiError> {
        self.layout_of_blocks(mip, self.inner.descriptor.format.bytes_per_block())
    }

    /// Buffer layout of one mip level of one plane: what
    /// [`crate::CommandEncoder::copy_image_plane_to_buffer`] writes and
    /// [`crate::CommandEncoder::copy_buffer_to_image_plane`] reads. Refused
    /// for the planes [`ImageFormat::copy_bytes`] has no size for.
    pub fn copy_layout_of(
        &self,
        mip: u32,
        aspect: ImageAspect,
    ) -> Result<ImageCopyLayout, RhiError> {
        let bytes = self.inner.descriptor.format.copy_bytes(aspect).ok_or(
            RhiError::UnsupportedImageCopy {
                reason: "that plane of the format is not copied to or from buffers",
            },
        )?;
        self.layout_of_blocks(mip, bytes)
    }

    fn layout_of_blocks(&self, mip: u32, block_bytes: u32) -> Result<ImageCopyLayout, RhiError> {
        let descriptor = self.inner.descriptor;
        if descriptor.sample_count > 1 {
            return Err(RhiError::UnsupportedImageCopy {
                reason: "a multisampled image is resolved, not copied to or from buffers",
            });
        }
        if mip >= descriptor.mip_count {
            return Err(RhiError::ImageMipOutOfRange {
                mip,
                mip_count: descriptor.mip_count,
            });
        }
        let (width, height, depth) = descriptor.mip_extent(mip);
        let unaligned =
            u64::from(width.div_ceil(descriptor.format.block_extent())) * u64::from(block_bytes);
        let height = height.div_ceil(descriptor.format.block_extent());
        let row_pitch = unaligned.div_ceil(IMAGE_COPY_ROW_ALIGNMENT) * IMAGE_COPY_ROW_ALIGNMENT;
        let slice_count = depth * descriptor.layer_count();
        Ok(ImageCopyLayout {
            row_pitch,
            rows_per_slice: height,
            slice_count,
            total_size: row_pitch * u64::from(height) * u64::from(slice_count),
        })
    }

    pub(crate) fn require_usage(
        &self,
        usage: ImageUsages,
        name: &'static str,
    ) -> Result<(), RhiError> {
        if self.inner.descriptor.usages.contains(usage) {
            Ok(())
        } else {
            Err(RhiError::MissingImageUsage { required: name })
        }
    }

    pub(crate) fn require_state(&self, expected: ResourceState) -> Result<(), RhiError> {
        match (0..self.subresource_count())
            .map(|index| self.state_of(index))
            .find(|actual| *actual != expected)
        {
            None => Ok(()),
            Some(actual) => Err(RhiError::ImageStateUnexpected { expected, actual }),
        }
    }

    /// Create a view covering every mip level and layer of this image.
    pub fn create_full_view(&self) -> Result<ImageView, RhiError> {
        self.create_view(&ImageViewDescriptor::default())
    }

    /// Create a view of part of this image: some of its mips and layers,
    /// presented as a 2D image, an array, a cube or a volume.
    pub fn create_view(&self, descriptor: &ImageViewDescriptor) -> Result<ImageView, RhiError> {
        let range = descriptor.resolve(&self.inner.descriptor)?;
        let backend = match &self.inner.backend {
            #[cfg(feature = "vulkan")]
            ImageBackend::Vulkan(image) => {
                ImageViewBackend::Vulkan(Box::new(image.create_view(self.inner.descriptor, range)?))
            }
            #[cfg(all(feature = "dx12", windows))]
            ImageBackend::Dx12(image) => {
                ImageViewBackend::Dx12(image.create_view(self.inner.descriptor, range))
            }
        };
        Ok(ImageView {
            inner: Arc::new(ImageViewInner {
                image: self.clone(),
                range,
                backend,
            }),
        })
    }
}

/// A view of an image. The view retains its parent image, and
/// clones share the native view so descriptor sets can retain it.
#[derive(Clone)]
pub struct ImageView {
    pub(crate) inner: Arc<ImageViewInner>,
}

pub(crate) struct ImageViewInner {
    pub(crate) backend: ImageViewBackend,
    range: ImageViewRange,
    image: Image,
}

pub(crate) enum ImageViewBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<VulkanImageView>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Dx12ImageView),
}

impl fmt::Debug for ImageView {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ImageView")
            .field("image", &self.inner.image)
            .finish_non_exhaustive()
    }
}

impl ImageView {
    /// The image this is a view of.
    #[must_use]
    pub fn image(&self) -> &Image {
        &self.inner.image
    }

    /// The mips, layers and planes this view covers.
    #[must_use]
    pub fn range(&self) -> ImageViewRange {
        self.inner.range
    }

    /// Width and height of the view's first mip: the extent of a render
    /// pass that draws into it.
    #[must_use]
    pub fn extent(&self) -> (u32, u32) {
        let (width, height, _) = self
            .inner
            .image
            .descriptor()
            .mip_extent(self.inner.range.base_mip);
        (width, height)
    }
}

#[cfg(test)]
#[path = "tests/image.rs"]
mod tests;
