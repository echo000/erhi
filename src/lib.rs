//! A rendering hardware interface over Vulkan and Direct3D 12.
//!
//! erhi exposes one API over two backends. The backend is chosen when a
//! device is opened ([`Device::open`]) and does not change afterwards. The
//! common layer validates and records commands ([`CommandEncoder`],
//! [`Frame`]); the backend replays them into a native command list. Native
//! types do not appear in the public API. Code that needs the underlying
//! objects, such as a vendor upscaler SDK, gets them as plain integers
//! ([`NativeDevice`]).
//!
//! # Descriptors
//!
//! Every object is created from a descriptor. Descriptors with required
//! fields have a constructor and `with_*` methods for the optional fields.
//! They are `#[non_exhaustive]`, so adding a field later is not a breaking
//! change:
//!
//! ```text
//! let buffer = device.create_buffer(BufferDescriptor::new(
//!     1024,
//!     BufferUsages::STORAGE,
//!     BufferMemory::Device,
//! ))?;
//! let image = device.create_image(
//!     ImageDescriptor::d2(ImageFormat::Rgba8Unorm, 512, 512, ImageUsages::SAMPLED)
//!         .with_mip_count(4),
//! )?;
//! ```
//!
//! Descriptors whose fields all have defaults are plain structs. Construct
//! them with `..Default::default()` ([`DeviceOptions`],
//! [`SamplerDescriptor`], [`ImageViewDescriptor`]). Recorded commands use
//! descriptors too: [`RenderPassDescriptor`], [`DrawDescriptor`],
//! [`DispatchDescriptor`], [`TraceRaysDescriptor`].
//!
//! # Errors and logging
//!
//! Validation errors and native failures are returned as [`RhiError`]. The
//! crate does not print or log. Messages from the native debug layers are
//! stored and returned by [`Device::debug_messages`].
//!
//! # Shaders
//!
//! Shader modules are created from compiled bytecode ([`ShaderSource`]):
//! SPIR-V for Vulkan, DXIL for Direct3D 12. The crate does not compile
//! shaders or run external tools, and it does not matter which language the
//! bytecode was compiled from (Slang, HLSL and GLSL are all tested). The
//! bytecode and the descriptors must agree on the following:
//!
//! - **Bindings.** Each binding has one number, unique within its set. On
//!   Vulkan it is the binding number, in the descriptor set with the set's
//!   number. On Direct3D 12 it is the register number, in the register
//!   class for the binding's type (`t`, `u`, `b` or `s`) and the register
//!   space with the set's number. HLSL that numbers each register class
//!   from zero is compiled to SPIR-V with a binding shift per class; give
//!   the same shifts in [`RegisterOffsets`].
//! - **Vertex inputs.** Vulkan identifies an input by location, Direct3D 12
//!   by semantic ([`VertexSemantic`]).
//! - **Push constants.** One block per layout. On Vulkan it is a
//!   push-constant block. On Direct3D 12 it is a constant buffer at the
//!   register given in [`PushConstantRange`].
//! - **Entry points.** Entry points are matched by name. A SPIR-V module
//!   whose only entry point is `main` (all GLSL, and single-entry Slang
//!   modules) also matches the name declared in its [`ShaderKind`]. Each
//!   pipeline stage is a [`ShaderEntry`] ([`ShaderModule::entry`]), so the
//!   stages of a ray-tracing pipeline can come from one library or from
//!   separate modules.
//! - **Combined image samplers.** [`DescriptorType::CombinedImageSampler`]
//!   is a Vulkan combined image sampler with an immutable sampler. On
//!   Direct3D 12 it is an image at the binding's `t` register plus a static
//!   sampler at register `s<n>`, where `n` is the binding's index among
//!   the set's combined bindings. HLSL shaders should declare a separate
//!   `Texture2D` and `SamplerState` instead
//!   ([`DescriptorType::SampledImage`], [`DescriptorType::Sampler`]).
//!
//! # Resource states
//!
//! Resource state transitions are automatic. Each command transitions the
//! images and buffers it uses to the state it needs: the destination of a
//! copy, the resources a descriptor set binds for a draw or dispatch, the
//! attachments of a render pass. Image state is tracked per mip level and
//! array layer, so one mip can be sampled while another is rendered to.
//! No transition can happen inside a render pass, so all uses of a
//! subresource within one pass must be compatible with a single state.
//! Sampling an attachment of the current pass is an error.
//! [`CommandEncoder::transition_image`] and
//! [`CommandEncoder::transition_buffer`] are available for explicit
//! transitions.
//!
//! # Descriptor sets
//!
//! A [`PipelineLayout`] declares up to [`MAX_DESCRIPTOR_SETS`] sets. A draw,
//! dispatch or ray dispatch binds one [`DescriptorSet`] per declared set,
//! in order. This allows per-frame data in one set and per-object data in
//! another. A binding can be an array of descriptors. Each element is
//! written by index (`write_sampled_image_at` and similar), and every
//! element must be written before the set is used.
//!
//! # Threads
//!
//! All handles ([`Device`], [`Buffer`], [`Image`], pipelines,
//! [`DescriptorSet`], [`AccelerationStructure`]) are `Send` and `Sync`.
//! Resources can be created, upload buffers written, readback buffers read
//! and commands submitted from any thread concurrently. Types that record
//! commands ([`CommandEncoder`], [`Frame`]) take `&mut self`, so only one
//! thread can record into one at a time; they can be moved between threads.
//! Submissions from several threads execute in the order they reach the
//! queue. If two encoders that use the same resource are submitted
//! concurrently, the caller must order them.

mod acceleration_structure;
mod buffer;
mod command_encoder;
mod compute_pipeline;
mod descriptor_set;
mod device;
mod error;
mod frame;
mod image;
mod memory;
mod native;
mod occlusion_query;
mod pipeline_layout;
mod profiling;
mod ray_tracing_pipeline;
mod render_commands;
mod render_pipeline;
mod resource_state;
mod sampler;
mod shader;
mod submission;
mod surface;
mod sync;
mod timestamp_query;

pub use acceleration_structure::{
    AccelerationStructure, AccelerationStructureInstance, AccelerationStructureKind,
    TriangleGeometry,
};
pub use buffer::{Buffer, BufferDescriptor, BufferMemory, BufferUsages};
pub use command_encoder::{
    ColorAttachment, CommandEncoder, DepthAttachment, DepthLoadOp, DispatchDescriptor,
    DrawDescriptor, IndexBufferBinding, IndexFormat, LoadOp, MAX_DISPATCH_GROUPS,
    RenderPassDescriptor, StencilLoadOp, TraceRaysDescriptor,
};
pub use compute_pipeline::{ComputePipeline, ComputePipelineDescriptor};
pub use descriptor_set::{
    BufferBinding, DescriptorSet, STORAGE_BUFFER_ALIGNMENT, UNIFORM_BUFFER_OFFSET_ALIGNMENT,
};
pub use error::{BackendAttempt, RhiError};
pub use frame::{Frame, FrameRing, FrameSubmissionTimings, RecordedFrame};
pub use image::{
    IMAGE_COPY_OFFSET_ALIGNMENT, IMAGE_COPY_ROW_ALIGNMENT, Image, ImageAspect, ImageCopyLayout,
    ImageDescriptor, ImageDimension, ImageFormat, ImageUsages, ImageView, ImageViewDescriptor,
    ImageViewDimension, ImageViewRange, MAX_IMAGE_EXTENT, MAX_IMAGE_EXTENT_3D, MAX_IMAGE_LAYERS,
};
pub use memory::{
    MemoryKind, MemoryOwner, MemoryRow, MemoryScope, MemorySnapshot, memory_owner_id, memory_scope,
    memory_scope_for,
};
pub use pipeline_layout::{
    AddressMode, DescriptorBinding, DescriptorSetLayoutDescriptor, DescriptorType, FilterMode,
    MAX_DESCRIPTOR_SETS, MAX_PUSH_CONSTANT_SIZE, PipelineLayout, PipelineLayoutDescriptor,
    PushConstantRange, RegisterOffsets, ShaderStages, StaticSampler,
};

pub use device::{
    AdapterPreference, Backend, DebugMessage, DebugSeverity, Device, DeviceOptions, VulkanOptions,
};
pub use native::{NativeCommandList, NativeDevice, NativeImage, NativeRecord};
pub use occlusion_query::{OCCLUSION_QUERY_SIZE, OcclusionQuerySet};
pub use profiling::TimestampDomain;
pub use ray_tracing_pipeline::{
    HitGroup, MAX_HIT_ATTRIBUTE_SIZE, RayTracingPipeline, RayTracingPipelineDescriptor, ShaderEntry,
};
pub use render_pipeline::{
    BlendComponent, BlendFactor, BlendMode, BlendOperation, BlendState, ColorTargetState,
    ColorWrites, CompareFunction, CullMode, DepthBias, DepthStencilState, FrontFace,
    MAX_COLOR_TARGETS, MAX_VERTEX_BUFFERS, MultisampleState, PrimitiveState, PrimitiveTopology,
    RenderPipeline, RenderPipelineDescriptor, StencilFaceState, StencilOperation, StencilState,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexSemantic, VertexStep,
};
pub use resource_state::ResourceState;
pub(crate) use resource_state::StateSet;
pub use sampler::{BorderColor, MAX_ANISOTROPY, Sampler, SamplerDescriptor};
pub use shader::{
    ShaderBinaryFormat, ShaderKind, ShaderModule, ShaderModuleDescriptor, ShaderSource,
};
pub use surface::{
    PresentMode, PresentStatus, Presentation, Surface, SurfaceCapabilities, SurfaceColorSpace,
    SurfaceConfiguration, SurfaceFormat,
};
pub use timestamp_query::{TIMESTAMP_SIZE, TimestampQuerySet};

#[cfg(feature = "vulkan")]
mod vulkan;

#[cfg(all(feature = "dx12", windows))]
mod dx12;

/// A device's capabilities: its name and API version, whether it
/// traces rays, and the limits ray tracing code has to keep to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// Whether this device supports hardware ray tracing at all. False on
    /// a device opened with `require_ray_tracing: false` when the adapter
    /// lacks the extensions/tier; every RT entry point then returns
    /// [`RhiError::RayTracingUnsupported`] and rendering uses the compute
    /// tracer.
    pub ray_tracing: bool,
    /// Whether compute and fragment shaders may trace rays themselves
    /// (`RayQuery`) against a top-level structure bound to them, with no
    /// ray-tracing pipeline.
    pub ray_query: bool,
    /// The adapter name reported by the driver.
    pub device_name: String,
    /// The version of the native API the device speaks, as that API encodes
    /// it.
    pub api_version: u32,
    /// The size of one shader record handle in a shader binding table, in
    /// bytes.
    pub(crate) shader_group_handle_size: u32,
    /// The alignment of each record in a shader binding table, in bytes.
    pub(crate) shader_group_handle_alignment: u32,
    /// The alignment of each region of a shader binding table, in bytes.
    pub(crate) shader_group_base_alignment: u32,
    /// The deepest a ray-tracing pipeline may trace rays from within hit and
    /// miss shaders.
    pub max_ray_recursion_depth: u32,
    /// The most rays one [`CommandEncoder::trace_rays`] may launch: width
    /// times height.
    pub max_ray_dispatch_invocations: u32,
    /// The most geometries a bottom-level acceleration structure holds.
    pub max_geometry_count: u64,
    /// The most instances a top-level acceleration structure holds.
    pub max_instance_count: u64,
    /// The alignment of the scratch space an acceleration-structure build
    /// uses, in bytes.
    pub(crate) scratch_alignment: u32,
}

/// A procedural axis-aligned box accepted by a ray-tracing acceleration
/// structure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// The corner with the least coordinate on every axis.
    pub min: [f32; 3],
    /// The corner with the greatest coordinate on every axis.
    pub max: [f32; 3],
}

impl Aabb {
    /// Construct a finite, positive-volume AABB.
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Result<Self, RhiError> {
        let finite = min.into_iter().chain(max).all(f32::is_finite);
        let ordered = min
            .into_iter()
            .zip(max)
            .all(|(minimum, maximum)| minimum < maximum);
        if !finite || !ordered {
            return Err(RhiError::InvalidAabb { min, max });
        }
        Ok(Self { min, max })
    }
}

/// Native allocation measurements produced by an acceleration-structure build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccelerationStructureBuildReport {
    /// The size of the structure itself, in bytes.
    pub acceleration_structure_size: u64,
    /// The scratch space its build used, in bytes.
    pub build_scratch_size: u64,
    /// The structure's address on the device.
    pub device_address: u64,
}

/// Held by each test that reads a device's validation messages. Direct3D 12
/// hands every device of one adapter in a process the same object, and so
/// the same message queue: without this, a message one test puts there on
/// purpose is counted by another running beside it.
#[cfg(test)]
pub(crate) static VALIDATION_MESSAGES: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
#[path = "tests/crate_root.rs"]
mod tests;
