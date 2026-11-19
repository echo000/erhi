use std::fmt;

/// One concrete failure retained while `Backend::Auto` tries available
/// implementations. No trait object or erased source error is involved.
#[derive(Debug)]
pub struct BackendAttempt {
    /// The backend that was tried: `vulkan` or `dx12`.
    pub backend: &'static str,
    /// Why it did not open.
    pub error: RhiError,
}

/// Every error this crate returns. Validation errors are detected before
/// any native call is made. Native failures carry the driver's error code
/// or message.
#[derive(Debug)]
#[non_exhaustive]
pub enum RhiError {
    /// A frame ring or one of its frames was used wrongly: a frame submitted
    /// after its slot was taken again, or a ring whose device is gone.
    InvalidFrame {
        /// A description of the problem.
        reason: &'static str,
    },
    /// A callback given to [`crate::CommandEncoder::record_native`] returned
    /// an error; its text is carried.
    NativeCallback(String),
    /// A ray dispatch of no rays, or of more than the device launches at
    /// once ([`crate::Capabilities::max_ray_dispatch_invocations`]).
    InvalidRayDispatch {
        /// How many rays across were asked for.
        width: u32,
        /// How many rays down were asked for.
        height: u32,
    },
    /// A backend name that is neither `auto`, `vulkan` nor `dx12` (parsing a
    /// [`crate::Backend`] from text).
    InvalidBackendName(String),
    /// The backend asked for was left out of this build: its Cargo feature is
    /// off, or it is Direct3D 12 off Windows.
    BackendNotCompiled(&'static str),
    /// [`crate::Backend::Auto`] tried every backend in the build and none
    /// opened.
    NoBackendAvailable {
        /// Each backend tried, with why it failed, in the order tried.
        attempts: Vec<BackendAttempt>,
    },
    /// A Direct3D 12 or DXGI call failed for a reason other than the device
    /// being lost.
    Dx12 {
        /// The native call that failed.
        operation: &'static str,
        /// The `HRESULT` it returned.
        code: i32,
        /// The system's text for that code.
        message: String,
    },
    /// Ray tracing was required and no Direct3D 12 adapter has it.
    NoDx12RayTracingDevice,
    /// Waiting for the device to finish its work failed.
    BackendWaitFailed {
        /// The backend it happened on: `vulkan` or `dx12`.
        backend: &'static str,
        /// The native wait result.
        result: u32,
    },
    /// A buffer of zero bytes was asked for, or bytes were given whose length
    /// is not whole rows of what they are for; the size is carried.
    InvalidBufferSize(u64),
    /// A buffer was described with no usages at all.
    EmptyBufferUsages,
    /// A buffer was used for something its descriptor did not declare.
    MissingBufferUsage {
        /// The usage that was needed.
        required: &'static str,
    },
    /// A range reached past the end of its buffer.
    BufferRangeOutOfBounds {
        /// How large the buffer is, in bytes.
        buffer_size: u64,
        /// Where the range began.
        offset: u64,
        /// How long the range was.
        size: u64,
    },
    /// A buffer was written or read from the CPU though its memory does not
    /// allow that: only upload buffers are written and only readback buffers
    /// read.
    BufferMemoryUnavailable {
        /// The kind of memory the buffer has.
        memory: &'static str,
        /// What was attempted: `write` or `read`.
        operation: &'static str,
    },
    /// Two objects of different backends were used together.
    ResourceBackendMismatch,
    /// An object was used with a device other than the one that made it.
    ResourceDeviceMismatch,
    /// A top-level acceleration structure was asked for with no instances.
    EmptyAccelerationStructureInstances,
    /// An acceleration structure's input was wrong: geometry that is not whole
    /// or not finite, an instance of the wrong kind of structure, a value out
    /// of range. The text says which.
    InvalidAccelerationStructureInstance(&'static str),
    /// A refit was asked of a structure not built to accept one.
    AccelerationStructureNotUpdatable,
    /// A refit named a different instance array than the build did. A refit
    /// reuses the tree the build produced, so it can move instances but not
    /// add, drop, or repoint them; the caller must rebuild instead.
    AccelerationStructureShapeChanged(&'static str),
    /// A pipeline layout's sets were not numbered from zero upwards without
    /// gaps, or there were more than [`crate::MAX_DESCRIPTOR_SETS`].
    InvalidDescriptorSetOrder {
        /// The set number that belonged at that place.
        expected: u32,
        /// The set number found there.
        actual: u32,
    },
    /// A binding was declared, written or bound wrongly.
    InvalidDescriptorBinding {
        /// The descriptor set of the binding.
        set: u32,
        /// The binding's number.
        binding: u32,
        /// A description of the problem.
        reason: &'static str,
    },
    /// Direct3D 12 refused the root signature made from a pipeline layout.
    Dx12RootSignatureSerialization {
        /// The `HRESULT` it returned.
        code: i32,
        /// The serializer's error text.
        message: String,
    },
    /// The device reported sizes for an acceleration-structure build that
    /// cannot be right (a zero).
    InvalidAccelerationStructureSizes {
        /// The size reported for the structure itself, in bytes.
        result: u64,
        /// The size reported for the build's scratch space, in bytes.
        scratch: u64,
    },
    /// A device address was not aligned as its use requires.
    DeviceAddressAlignment {
        /// Which resource the address belongs to.
        resource: &'static str,
        /// The address.
        address: u64,
        /// The alignment it must have, in bytes.
        required: u32,
    },
    /// The Vulkan library could not be loaded: no driver is installed, or the
    /// loader is missing.
    VulkanLoader(String),
    /// A Vulkan call failed for a reason other than the device being lost.
    Vulkan {
        /// The native call that failed.
        operation: &'static str,
        /// The `VkResult` it returned.
        result: i32,
    },
    /// Ray tracing was required and no Vulkan device has it.
    NoRayTracingDevice,
    /// The device was opened without hardware ray tracing
    /// (`require_ray_tracing: false` on an adapter without it); an RT entry
    /// point cannot be used.
    RayTracingUnsupported,
    /// The Vulkan device lacks extensions this crate cannot do without.
    MissingDeviceExtensions {
        /// The device's name.
        device: String,
        /// The names of the extensions it lacks.
        extensions: Vec<String>,
    },
    /// The Vulkan device has the ray-tracing extensions but not every feature
    /// of them that is needed.
    MissingRayTracingFeatures {
        /// The device's name.
        device: String,
        /// The names of the features it lacks.
        features: Vec<&'static str>,
    },
    /// The Vulkan device has no queue that takes graphics and compute work.
    NoComputeQueue {
        /// The device's name.
        device: String,
    },
    /// No Vulkan memory type suits a resource.
    NoMemoryType {
        /// The memory types the resource can live in, as a bit mask.
        required_bits: u32,
        /// The `VkMemoryPropertyFlags` that were requested.
        required_flags: u32,
    },
    /// A box whose corners are not finite, or whose minimum is not below its
    /// maximum on every axis.
    InvalidAabb {
        /// The corner given as the least.
        min: [f32; 3],
        /// The corner given as the greatest.
        max: [f32; 3],
    },
    /// A width or height of zero, or beyond what the device draws or presents.
    InvalidRenderDimensions {
        /// The width asked for.
        width: u32,
        /// The height asked for.
        height: u32,
    },
    /// Bytes written to device memory did not fit the allocation they were
    /// for.
    InvalidUpload(&'static str),
    /// A top-level acceleration structure of more instances than the device
    /// builds.
    TooManyInstances {
        /// How many instances were given.
        count: usize,
        /// The most the device accepts.
        maximum: u64,
    },
    /// The device's queue cannot write timestamps, so no timestamp query set
    /// can be made.
    TimestampQueriesUnavailable,
    /// [`crate::DeviceOptions::frames_in_flight`] was outside one to eight; the
    /// number is carried.
    InvalidFramesInFlight(u32),
    /// A shader binding table landed at an address the device does not accept.
    ShaderBindingTableAlignment {
        /// The address.
        address: u64,
        /// The alignment it must have, in bytes.
        required: u64,
    },
    /// Vulkan gave no device address for a resource that must have one; the
    /// text says which.
    ZeroDeviceAddress(&'static str),
    /// Device memory could not be allocated or mapped. Usually the device is
    /// out of memory.
    Allocation {
        /// The backend it happened on: `vulkan` or `dx12`.
        backend: &'static str,
        /// The native call that failed.
        operation: &'static str,
        /// The allocator's error text.
        message: String,
    },
    /// An image with a dimension of zero, beyond the device's limits, or of a
    /// shape its kind cannot have (a cube needs square faces, six to a cube).
    InvalidImageDimensions {
        /// The width asked for.
        width: u32,
        /// The height asked for.
        height: u32,
        /// The depth or layer count asked for.
        depth_or_layers: u32,
    },
    /// An image with no mips, or with more than its size has.
    InvalidImageMipCount {
        /// The mip count asked for.
        mip_count: u32,
        /// The most an image of that size has.
        maximum: u32,
    },
    /// An image was described with no usages at all.
    EmptyImageUsages,
    /// An image was used for something its descriptor did not declare.
    MissingImageUsage {
        /// The usage that was needed.
        required: &'static str,
    },
    /// A mip was named that the image does not have.
    ImageMipOutOfRange {
        /// The mip named.
        mip: u32,
        /// How many mips the image has.
        mip_count: u32,
    },
    /// An image was not in the state a command needs it in, where the command
    /// does not move it there itself.
    ImageStateUnexpected {
        /// The state the command needs.
        expected: crate::ResourceState,
        /// The state the image is in.
        actual: crate::ResourceState,
    },
    /// A thread panicked while it held one of this crate's locks, and what
    /// the lock guards may have been left half changed. The device that
    /// owns it should be dropped.
    LockPoisoned {
        /// The data the lock protects.
        what: &'static str,
    },
    /// A copy that neither API defines: onto itself.
    InvalidCopy {
        /// A description of the problem.
        reason: &'static str,
    },
    /// A buffer is bound for a combination of uses that no single state allows.
    InvalidBufferUsage {
        /// A description of the problem.
        reason: &'static str,
    },
    /// An image whose usages its format cannot serve.
    InvalidImageUsage {
        /// A description of the problem.
        reason: &'static str,
    },
    /// A valid request that this device does not support.
    UnsupportedByDevice {
        /// A description of the problem.
        reason: &'static str,
    },
    /// A descriptor set used with a pipeline whose layout declares that set
    /// otherwise.
    DescriptorSetLayoutMismatch {
        /// The number of the set that does not match.
        set: u32,
    },
    /// A buffer offset in a copy to or from an image was not aligned
    /// ([`crate::IMAGE_COPY_OFFSET_ALIGNMENT`]).
    UnalignedCopyOffset {
        /// The offset given.
        offset: u64,
        /// The alignment it must have, in bytes.
        required: u64,
    },
    /// A copy between two images that do not match in size, format or sample
    /// count.
    MismatchedImageCopy {
        /// A description of the problem.
        reason: &'static str,
    },
    /// Shader bytes for the other backend: DXIL given to Vulkan, or SPIR-V to
    /// Direct3D 12.
    ShaderSourceUnsupported {
        /// The backend it happened on: `vulkan` or `dx12`.
        backend: &'static str,
        /// What was given.
        source: &'static str,
    },
    /// Shader bytes that are not what they were given as: too short, or
    /// without the format's header.
    InvalidShaderBlob {
        /// The format they were given as.
        expected: &'static str,
    },
    /// A ray-tracing pipeline described wrongly.
    InvalidRayTracingPipeline {
        /// A description of the problem.
        reason: &'static str,
    },
    /// A pipeline named an entry point its shader module does not have.
    MissingShaderExport {
        /// The name that was not found.
        entry: String,
    },
    /// Push constants declared wrongly in a layout, or given to a draw or
    /// dispatch in a size the layout does not declare.
    InvalidPushConstants {
        /// A description of the problem.
        reason: &'static str,
    },
    /// A compute pipeline described wrongly, or a dispatch of no work.
    InvalidComputePipeline {
        /// A description of the problem.
        reason: &'static str,
    },
    /// A timestamp query used out of order or out of range.
    InvalidTimestampQuery {
        /// A description of the problem.
        reason: &'static str,
    },
    /// A render pipeline described wrongly.
    InvalidRenderPipeline {
        /// A description of the problem.
        reason: &'static str,
    },
    /// A render pass begun, used or ended wrongly: attachments that do not
    /// agree, a draw outside a pass, a command a pass does not allow.
    InvalidRenderPass {
        /// A description of the problem.
        reason: &'static str,
    },
    /// A view that does not fit its image (`Image::create_view`).
    InvalidImageView {
        /// A description of the problem.
        reason: &'static str,
    },
    /// A sample count no image has, or a multisampled image of a shape
    /// that cannot be one.
    InvalidImageSampleCount {
        /// A description of the problem.
        reason: &'static str,
    },
    /// An occlusion query used out of order or out of range.
    InvalidOcclusionQuery {
        /// A description of the problem.
        reason: &'static str,
    },
    /// A sampler description no device accepts (`Device::create_sampler`).
    InvalidSampler {
        /// A description of the problem.
        reason: &'static str,
    },
    /// A copy this format does not take part in.
    UnsupportedImageCopy {
        /// A description of the problem.
        reason: &'static str,
    },
    /// A surface was asked for a kind of window this crate does not present
    /// to.
    UnsupportedWindowHandle,
    /// The device cannot present to this window.
    PresentationUnsupported {
        /// A description of the problem.
        reason: &'static str,
    },
    /// The device stopped working: the driver reset or crashed, or the
    /// adapter went away. Nothing made from this device works again. Drop
    /// every object made from it, then ask for a new one with
    /// [`crate::Device::open`]; see [`Self::is_device_lost`].
    DeviceLost {
        /// The backend it happened on: `vulkan` or `dx12`.
        backend: &'static str,
        /// The call that reported it.
        operation: &'static str,
        /// The driver's message, if it gave one.
        reason: String,
    },
}

impl RhiError {
    #[cfg(feature = "vulkan")]
    pub(crate) fn vk(operation: &'static str, result: ash::vk::Result) -> Self {
        if result == ash::vk::Result::ERROR_DEVICE_LOST {
            return Self::DeviceLost {
                backend: "vulkan",
                operation,
                reason: "VK_ERROR_DEVICE_LOST".to_owned(),
            };
        }
        Self::Vulkan {
            operation,
            result: result.as_raw(),
        }
    }

    #[cfg(all(feature = "dx12", windows))]
    pub(crate) fn dx12(operation: &'static str, error: windows::core::Error) -> Self {
        Self::dx12_with_message(operation, error.code().0, error.to_string())
    }

    /// A Direct3D 12 failure: [`Self::DeviceLost`] for the codes that mean
    /// the device is gone (removed, hung, reset, or the driver failed).
    #[cfg(all(feature = "dx12", windows))]
    pub(crate) fn dx12_with_message(operation: &'static str, code: i32, message: String) -> Self {
        use windows::Win32::Graphics::Dxgi::{
            DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET,
            DXGI_ERROR_DRIVER_INTERNAL_ERROR,
        };
        let lost = [
            DXGI_ERROR_DEVICE_REMOVED,
            DXGI_ERROR_DEVICE_HUNG,
            DXGI_ERROR_DEVICE_RESET,
            DXGI_ERROR_DRIVER_INTERNAL_ERROR,
        ]
        .iter()
        .any(|lost| lost.0 == code);
        if lost {
            Self::DeviceLost {
                backend: "dx12",
                operation,
                reason: message,
            }
        } else {
            Self::Dx12 {
                operation,
                code,
                message,
            }
        }
    }

    /// Whether this is the device having stopped working
    /// ([`Self::DeviceLost`]), as against a call that was wrong or a
    /// resource that ran out: the one failure a new device recovers from.
    #[must_use]
    pub const fn is_device_lost(&self) -> bool {
        matches!(self, Self::DeviceLost { .. })
    }
}

impl fmt::Display for RhiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFrame { reason } => write!(formatter, "invalid frame: {reason}"),
            Self::NativeCallback(message) => {
                write!(formatter, "a native callback failed: {message}")
            }
            Self::InvalidRayDispatch { width, height } => {
                write!(formatter, "invalid ray dispatch of {width}x{height} rays")
            }
            Self::LockPoisoned { what } => write!(
                formatter,
                "a thread panicked while holding the lock on {what}"
            ),
            Self::InvalidImageSampleCount { reason } => {
                write!(formatter, "invalid image sample count: {reason}")
            }
            Self::InvalidOcclusionQuery { reason } => {
                write!(formatter, "invalid occlusion query: {reason}")
            }
            Self::DeviceLost {
                backend,
                operation,
                reason,
            } => write!(
                formatter,
                "the {backend} device was lost (reported by {operation}): {reason}"
            ),
            Self::InvalidBackendName(name) => write!(
                formatter,
                "unknown backend `{name}`; expected auto, vulkan, or dx12"
            ),
            Self::BackendNotCompiled(backend) => {
                write!(
                    formatter,
                    "backend `{backend}` is not compiled into this binary"
                )
            }
            Self::NoBackendAvailable { attempts } => {
                if attempts.is_empty() {
                    formatter.write_str("this binary contains no usable backends")
                } else {
                    formatter.write_str("no backend could create a device (")?;
                    for (index, attempt) in attempts.iter().enumerate() {
                        if index != 0 {
                            formatter.write_str("; ")?;
                        }
                        write!(formatter, "{}: {}", attempt.backend, attempt.error)?;
                    }
                    formatter.write_str(")")
                }
            }
            Self::Dx12 {
                operation,
                code,
                message,
            } => write!(
                formatter,
                "Direct3D 12 operation `{operation}` failed with {code:#010x}: {message}"
            ),
            Self::NoDx12RayTracingDevice => {
                formatter.write_str("no Direct3D 12 adapter supports the required DXR tier")
            }
            Self::BackendWaitFailed { backend, result } => write!(
                formatter,
                "backend `{backend}` wait failed with result {result:#010x}"
            ),
            Self::InvalidBufferSize(size) => {
                write!(
                    formatter,
                    "invalid GPU buffer size {size}; size must be non-zero"
                )
            }
            Self::EmptyBufferUsages => {
                formatter.write_str("GPU buffer must declare at least one usage")
            }
            Self::MissingBufferUsage { required } => {
                write!(
                    formatter,
                    "GPU buffer is missing required `{required}` usage"
                )
            }
            Self::BufferRangeOutOfBounds {
                buffer_size,
                offset,
                size,
            } => write!(
                formatter,
                "GPU buffer range offset {offset} size {size} exceeds buffer size {buffer_size}"
            ),
            Self::BufferMemoryUnavailable { memory, operation } => write!(
                formatter,
                "GPU buffer memory `{memory}` does not support CPU {operation}"
            ),
            Self::ResourceBackendMismatch => {
                formatter.write_str("GPU resources belong to different backends")
            }
            Self::ResourceDeviceMismatch => {
                formatter.write_str("GPU resources belong to different devices")
            }
            Self::EmptyAccelerationStructureInstances => formatter
                .write_str("a top-level acceleration structure requires at least one instance"),
            Self::InvalidAccelerationStructureInstance(reason) => {
                write!(
                    formatter,
                    "invalid acceleration-structure instance: {reason}"
                )
            }
            Self::AccelerationStructureNotUpdatable => formatter.write_str(
                "this acceleration structure was not built to allow refits; rebuild instead",
            ),
            Self::AccelerationStructureShapeChanged(reason) => write!(
                formatter,
                "an acceleration-structure refit cannot change {reason}; rebuild instead"
            ),
            Self::InvalidDescriptorSetOrder { expected, actual } => write!(
                formatter,
                "descriptor sets must be contiguous and ordered: expected set {expected}, got {actual}"
            ),
            Self::InvalidDescriptorBinding {
                set,
                binding,
                reason,
            } => write!(
                formatter,
                "invalid descriptor binding set {set}, binding {binding}: {reason}"
            ),
            Self::Dx12RootSignatureSerialization { code, message } => write!(
                formatter,
                "Direct3D 12 root-signature serialization failed with {code:#010x}: {message}"
            ),
            Self::InvalidAccelerationStructureSizes { result, scratch } => write!(
                formatter,
                "invalid acceleration-structure prebuild sizes: result {result}, scratch {scratch}"
            ),
            Self::DeviceAddressAlignment {
                resource,
                address,
                required,
            } => write!(
                formatter,
                "{resource} address {address:#x} is not aligned to {required} bytes"
            ),
            Self::VulkanLoader(error) => write!(formatter, "could not load Vulkan: {error}"),
            Self::Vulkan { operation, result } => {
                write!(
                    formatter,
                    "Vulkan operation `{operation}` failed with {result}"
                )
            }
            Self::NoRayTracingDevice => {
                formatter.write_str("no Vulkan device supports the required ray-tracing profile")
            }
            Self::RayTracingUnsupported => formatter.write_str(
                "this device was opened without hardware ray tracing; use the compute tracer",
            ),
            Self::MissingDeviceExtensions { device, extensions } => write!(
                formatter,
                "Vulkan device `{device}` is missing extensions: {}",
                extensions.join(", ")
            ),
            Self::MissingRayTracingFeatures { device, features } => write!(
                formatter,
                "Vulkan device `{device}` is missing features: {}",
                features.join(", ")
            ),
            Self::NoComputeQueue { device } => {
                write!(
                    formatter,
                    "Vulkan device `{device}` has no compute-capable queue"
                )
            }
            Self::NoMemoryType {
                required_bits,
                required_flags,
            } => write!(
                formatter,
                "no Vulkan memory type matches bits {required_bits:#010x} and flags {required_flags:#010x}"
            ),
            Self::InvalidAabb { min, max } => {
                write!(formatter, "invalid AABB: min {min:?}, max {max:?}")
            }
            Self::InvalidRenderDimensions { width, height } => {
                write!(formatter, "invalid render dimensions {width}x{height}")
            }
            Self::InvalidUpload(message) => write!(formatter, "invalid upload: {message}"),
            Self::TooManyInstances { count, maximum } => write!(
                formatter,
                "{count} instances in one top-level structure, but the device limit is {maximum}"
            ),
            Self::TimestampQueriesUnavailable => {
                formatter.write_str("the selected Vulkan queue does not support timestamp queries")
            }
            Self::InvalidFramesInFlight(count) => {
                write!(formatter, "{count} frames in flight: one to eight are allowed")
            }
            Self::ShaderBindingTableAlignment { address, required } => write!(
                formatter,
                "shader binding table address {address:#x} is not aligned to {required} bytes"
            ),
            Self::ZeroDeviceAddress(resource) => {
                write!(
                    formatter,
                    "Vulkan returned a zero device address for {resource}"
                )
            }
            Self::Allocation {
                backend,
                operation,
                message,
            } => write!(
                formatter,
                "GPU memory allocation `{operation}` failed on backend `{backend}`: {message}"
            ),
            Self::InvalidImageDimensions {
                width,
                height,
                depth_or_layers,
            } => write!(
                formatter,
                "invalid image dimensions {width}x{height}x{depth_or_layers}; every extent must be non-zero"
            ),
            Self::InvalidImageMipCount { mip_count, maximum } => write!(
                formatter,
                "invalid image mip count {mip_count}; expected 1 through {maximum} for these dimensions"
            ),
            Self::EmptyImageUsages => {
                formatter.write_str("GPU image must declare at least one usage")
            }
            Self::MissingImageUsage { required } => {
                write!(
                    formatter,
                    "GPU image is missing required `{required}` usage"
                )
            }
            Self::ImageMipOutOfRange { mip, mip_count } => write!(
                formatter,
                "image mip {mip} is out of range for an image with {mip_count} mip levels"
            ),
            Self::InvalidCopy { reason } => write!(formatter, "invalid copy: {reason}"),
            Self::InvalidBufferUsage { reason } => write!(formatter, "invalid buffer: {reason}"),
            Self::InvalidImageUsage { reason } => write!(formatter, "invalid image: {reason}"),
            Self::UnsupportedByDevice { reason } => {
                write!(formatter, "not supported by this device: {reason}")
            }
            Self::DescriptorSetLayoutMismatch { set } => write!(
                formatter,
                "descriptor set {set} was made for a layout that declares it differently from the \
                 pipeline's"
            ),
            Self::ImageStateUnexpected { expected, actual } => write!(
                formatter,
                "image is in state `{}` but the operation requires `{}`; record the transition first",
                actual.name(),
                expected.name()
            ),
            Self::UnalignedCopyOffset { offset, required } => write!(
                formatter,
                "buffer offset {offset} for an image copy must be aligned to {required} bytes"
            ),
            Self::MismatchedImageCopy { reason } => {
                write!(formatter, "image-to-image copy is invalid: {reason}")
            }
            Self::ShaderSourceUnsupported { backend, source } => {
                write!(formatter, "backend `{backend}` cannot consume {source}")
            }
            Self::InvalidShaderBlob { expected } => {
                write!(formatter, "shader blob is not valid {expected}")
            }
            Self::InvalidRayTracingPipeline { reason } => {
                write!(formatter, "invalid ray-tracing pipeline: {reason}")
            }
            Self::MissingShaderExport { entry } => {
                write!(
                    formatter,
                    "shader module does not export entry point `{entry}`"
                )
            }
            Self::InvalidPushConstants { reason } => {
                write!(formatter, "invalid push constants: {reason}")
            }
            Self::InvalidComputePipeline { reason } => {
                write!(formatter, "invalid compute pipeline: {reason}")
            }
            Self::InvalidTimestampQuery { reason } => {
                write!(formatter, "invalid timestamp query: {reason}")
            }
            Self::InvalidRenderPipeline { reason } => {
                write!(formatter, "invalid render pipeline: {reason}")
            }
            Self::InvalidRenderPass { reason } => {
                write!(formatter, "invalid render pass: {reason}")
            }
            Self::InvalidImageView { reason } => {
                write!(formatter, "invalid image view: {reason}")
            }
            Self::InvalidSampler { reason } => {
                write!(formatter, "invalid sampler: {reason}")
            }
            Self::UnsupportedImageCopy { reason } => {
                write!(formatter, "unsupported image copy: {reason}")
            }
            Self::UnsupportedWindowHandle => formatter.write_str(
                "the window handle is not one this platform's backends present to                  (Win32; Xlib, Xcb or Wayland on Linux)",
            ),
            Self::PresentationUnsupported { reason } => {
                write!(formatter, "presentation is unavailable: {reason}")
            }
        }
    }
}

impl std::error::Error for RhiError {}
