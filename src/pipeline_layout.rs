use std::fmt;
use std::ops::{BitOr, BitOrAssign};

use crate::device::BackendDevice;
#[cfg(all(feature = "dx12", windows))]
use crate::dx12::Dx12PipelineLayout;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanPipelineLayout;
use crate::{Backend, Device, RhiError};

/// The type of resource a binding holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DescriptorType {
    /// A top-level acceleration structure, for rays to be traced against
    /// (`RaytracingAccelerationStructure`).
    AccelerationStructure,
    /// An image a shader writes, or reads and writes (`RWTexture2D`).
    StorageImage,
    /// A buffer of records a shader reads (`StructuredBuffer`).
    ReadOnlyStorageBuffer,
    /// A buffer of records a shader reads and writes (`RWStructuredBuffer`).
    StorageBuffer,
    /// A small block of constants (`ConstantBuffer`, `cbuffer`).
    UniformBuffer,
    /// An image a shader samples or loads from (`Texture2D` and related types).
    SampledImage,
    /// A sampler made by [`crate::Device::create_sampler`] (`SamplerState`).
    Sampler,
    /// A sampled image paired with an immutable sampler baked into the
    /// layout. This is the type Slang's `Sampler2D` lowers to: a Vulkan
    /// combined image sampler with an immutable sampler, and on DX12 an SRV
    /// plus a static sampler whose `s` register is the binding's ordinal
    /// among the set's combined bindings in declaration order.
    CombinedImageSampler(StaticSampler),
}

/// Minification/magnification filtering for an immutable sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilterMode {
    /// The nearest texel, unblended.
    Nearest,
    /// The texels around the point, blended by how near each is.
    Linear,
}

/// Texture addressing outside [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressMode {
    /// Coordinates outside read the texel at the nearest edge.
    ClampToEdge,
    /// The image repeats: only the fraction of a coordinate counts.
    Repeat,
    /// The image repeats, each repeat the mirror image of the one before.
    MirrorRepeat,
    /// Reads the sampler's border colour (transparent black for an
    /// immutable sampler).
    ClampToBorder,
}

/// An immutable sampler baked into a pipeline layout: one filter between
/// texels and between mips alike, over every mip, and one address mode.
/// Anything else is a [`crate::Sampler`] written into a `Sampler` binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct StaticSampler {
    /// How texels are blended, between texels and between mips alike.
    pub filter: FilterMode,
    /// What coordinates outside 0 to 1 read, on every axis.
    pub address_mode: AddressMode,
}

impl StaticSampler {
    /// A sampler that blends as `filter` says and reads outside the image
    /// as `address_mode` says.
    #[must_use]
    pub const fn new(filter: FilterMode, address_mode: AddressMode) -> Self {
        Self {
            filter,
            address_mode,
        }
    }
}

impl StaticSampler {
    /// The linear-filter, clamp-to-edge sampler used by every post pass.
    #[must_use]
    pub const fn linear_clamp() -> Self {
        Self {
            filter: FilterMode::Linear,
            address_mode: AddressMode::ClampToEdge,
        }
    }
}

/// One push-constant block starting at offset zero. Vulkan consumes it as a
/// push-constant range; Direct3D 12 as 32-bit root constants in the
/// constant buffer at `register` of `space`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct PushConstantRange {
    /// The shader stages that read the block.
    pub stages: ShaderStages,
    /// Size in bytes: non-zero, a multiple of four, at most 128.
    pub size: u32,
    /// Direct3D 12 only: the `b` register the shader declares the block at.
    /// Zero is where Slang puts a `[[vk::push_constant]]` block that says
    /// no register; HLSL says its own (`register(b0, space0)`).
    pub register: u32,
    /// Direct3D 12 only: the register space of `register`.
    pub space: u32,
}

impl PushConstantRange {
    /// A block of `size` bytes for `stages`, at `b0, space0` on Direct3D 12.
    #[must_use]
    pub const fn new(stages: ShaderStages, size: u32) -> Self {
        Self {
            stages,
            size,
            register: 0,
            space: 0,
        }
    }

    /// The same block, declared by the shader at `register` of `space` on
    /// Direct3D 12.
    #[must_use]
    pub const fn at_register(self, register: u32, space: u32) -> Self {
        Self {
            register,
            space,
            ..self
        }
    }
}

/// What a set's binding numbers are above its Direct3D 12 register
/// numbers, for each kind of register.
///
/// A binding has one number, unique in its set: Vulkan's binding number. On
/// Direct3D 12 its register is of the kind its type has (`t` for what is
/// read, `u` for what is written, `b` for a uniform buffer, `s` for a
/// sampler) and its number is the binding number less the offset here for
/// that kind. With no offsets, the default, the two numbers are the same,
/// which is how Slang numbers them when a shader gives both
/// (`[[vk::binding(1, 0)]]`, `register(u1, space0)`).
///
/// Offsets are for HLSL written the Direct3D way, each kind of register
/// counted from zero, and compiled to SPIR-V with each kind shifted
/// (`dxc -fvk-t-shift 0 0 -fvk-s-shift 128 0 ...`): give the same shifts
/// here and number the bindings as SPIR-V has them.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegisterOffsets {
    /// `t` registers: sampled images, read-only storage buffers,
    /// acceleration structures.
    pub shader_resource: u32,
    /// `s` registers: samplers.
    pub sampler: u32,
    /// `b` registers: uniform buffers.
    pub constant_buffer: u32,
    /// `u` registers: storage images, storage buffers.
    pub unordered_access: u32,
}

impl RegisterOffsets {
    /// No offsets: a binding's register number is its binding number.
    pub const NONE: Self = Self {
        shader_resource: 0,
        sampler: 0,
        constant_buffer: 0,
        unordered_access: 0,
    };

    /// The Direct3D 12 register number of `binding`, or `None` when its
    /// number is below the offset of its kind.
    #[must_use]
    pub(crate) fn register(self, binding: &DescriptorBinding) -> Option<u32> {
        let offset = match binding.descriptor_type {
            DescriptorType::AccelerationStructure
            | DescriptorType::ReadOnlyStorageBuffer
            | DescriptorType::SampledImage
            | DescriptorType::CombinedImageSampler(_) => self.shader_resource,
            DescriptorType::StorageImage | DescriptorType::StorageBuffer => self.unordered_access,
            DescriptorType::UniformBuffer => self.constant_buffer,
            DescriptorType::Sampler => self.sampler,
        };
        binding.binding.checked_sub(offset)
    }
}

/// The largest guaranteed-portable push-constant block.
pub const MAX_PUSH_CONSTANT_SIZE: u32 = 128;

/// A set of shader stages: which of them see a binding or read the push
/// constants.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderStages(u32);

impl ShaderStages {
    /// Ray-generation shaders.
    pub const RAY_GENERATION: Self = Self(1 << 0);

    /// Miss shaders.
    pub const MISS: Self = Self(1 << 1);

    /// Closest-hit shaders.
    pub const CLOSEST_HIT: Self = Self(1 << 2);

    /// Intersection shaders.
    pub const INTERSECTION: Self = Self(1 << 3);

    /// Compute shaders.
    pub const COMPUTE: Self = Self(1 << 4);

    /// Vertex shaders.
    pub const VERTEX: Self = Self(1 << 5);

    /// Fragment (pixel) shaders.
    pub const FRAGMENT: Self = Self(1 << 6);

    /// Any-hit shaders.
    pub const ANY_HIT: Self = Self(1 << 7);

    /// Every stage of a ray-tracing pipeline.
    pub const ALL_RAY_TRACING: Self = Self(
        Self::RAY_GENERATION.0
            | Self::MISS.0
            | Self::CLOSEST_HIT.0
            | Self::INTERSECTION.0
            | Self::ANY_HIT.0,
    );

    /// Whether every stage of `other` is among these.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether there are no stages at all.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// These stages and those of `other`; `|` does the same.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOr for ShaderStages {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for ShaderStages {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// One binding of a descriptor set: its number, what it holds, how many, and
/// which stages see it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct DescriptorBinding {
    /// The binding's number, unique in its set: the Vulkan binding number, and
    /// on Direct3D 12 the register number (less the set's
    /// [`RegisterOffsets`]).
    pub binding: u32,
    /// The type of resource the binding holds.
    pub descriptor_type: DescriptorType,
    /// How many descriptors the binding is: one, or the length of the array
    /// the shader declares. Each is written by index (`write_*_at`).
    pub count: u32,
    /// The shader stages that use the binding.
    pub visibility: ShaderStages,
}

impl DescriptorBinding {
    /// One descriptor of `descriptor_type` at `binding`, seen by
    /// `visibility`.
    #[must_use]
    pub const fn new(
        binding: u32,
        descriptor_type: DescriptorType,
        visibility: ShaderStages,
    ) -> Self {
        Self {
            binding,
            descriptor_type,
            count: 1,
            visibility,
        }
    }

    /// The same binding as an array of `count` descriptors.
    #[must_use]
    pub const fn with_count(mut self, count: u32) -> Self {
        self.count = count;
        self
    }
}

/// The bindings of one descriptor set of a pipeline layout.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct DescriptorSetLayoutDescriptor<'a> {
    /// The set's number, which must be its place in
    /// [`PipelineLayoutDescriptor::sets`]: the Vulkan set number and the
    /// Direct3D 12 register space.
    pub set: u32,
    /// The set's bindings, each with its own number.
    pub bindings: &'a [DescriptorBinding],
    /// Direct3D 12 only: what the binding numbers are above the register
    /// numbers. [`RegisterOffsets::NONE`] makes them the same.
    pub register_offsets: RegisterOffsets,
}

impl<'a> DescriptorSetLayoutDescriptor<'a> {
    /// Set number `set` with `bindings`, each register number the same as
    /// its binding number.
    #[must_use]
    pub const fn new(set: u32, bindings: &'a [DescriptorBinding]) -> Self {
        Self {
            set,
            bindings,
            register_offsets: RegisterOffsets::NONE,
        }
    }

    /// The same set, its binding numbers above its Direct3D 12 register
    /// numbers by `offsets`.
    #[must_use]
    pub const fn with_register_offsets(mut self, offsets: RegisterOffsets) -> Self {
        self.register_offsets = offsets;
        self
    }
}

/// Everything the shaders of a pipeline are given from outside: their
/// descriptor sets and their push constants. [`Device::create_pipeline_layout`]
/// makes a layout of it, which pipelines and descriptor sets are then made
/// against.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct PipelineLayoutDescriptor<'a> {
    /// The descriptor sets, numbered from zero in order: at most
    /// [`MAX_DESCRIPTOR_SETS`].
    pub sets: &'a [DescriptorSetLayoutDescriptor<'a>],
    /// Optional push-constant block shared by every pipeline on this layout.
    pub push_constants: Option<PushConstantRange>,
}

impl<'a> PipelineLayoutDescriptor<'a> {
    /// A layout of `sets` with no push constants.
    #[must_use]
    pub const fn new(sets: &'a [DescriptorSetLayoutDescriptor<'a>]) -> Self {
        Self {
            sets,
            push_constants: None,
        }
    }

    /// The same layout with a push-constant block.
    #[must_use]
    pub const fn with_push_constants(mut self, range: PushConstantRange) -> Self {
        self.push_constants = Some(range);
        self
    }
}

/// A pipeline layout. Clones share the native objects so
/// pipelines and descriptor sets can retain the layout they were built from.
#[derive(Clone)]
pub struct PipelineLayout {
    inner: std::sync::Arc<PipelineLayoutInner>,
}

struct PipelineLayoutInner {
    device_id: u64,
    backend: PipelineLayoutBackend,
    sets: Vec<OwnedDescriptorSetLayout>,
    resource_descriptor_count: u32,
    sampler_descriptor_count: u32,
    push_constants: Option<PushConstantRange>,
    _device: BackendDevice,
}

/// Owned copy of one set's binding declarations, retained for validating
/// descriptor-set allocation and writes without borrowing the descriptor.
#[derive(Debug, Clone)]
pub(crate) struct OwnedDescriptorSetLayout {
    pub set: u32,
    pub bindings: Vec<DescriptorBinding>,
}

pub(crate) enum PipelineLayoutBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<VulkanPipelineLayout>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Box<Dx12PipelineLayout>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DescriptorCounts {
    pub resources: u32,
    pub samplers: u32,
}

impl fmt::Debug for PipelineLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PipelineLayout")
            .field("backend", &self.backend())
            .field(
                "resource_descriptor_count",
                &self.inner.resource_descriptor_count,
            )
            .field(
                "sampler_descriptor_count",
                &self.inner.sampler_descriptor_count,
            )
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Make a pipeline layout. Every pipeline made with it, and every
    /// descriptor set made for it, agree on what is bound where.
    pub fn create_pipeline_layout(
        &self,
        descriptor: PipelineLayoutDescriptor<'_>,
    ) -> Result<PipelineLayout, RhiError> {
        let counts = validate_pipeline_layout(descriptor)?;
        let backend = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => {
                PipelineLayoutBackend::Vulkan(Box::new(device.create_pipeline_layout(descriptor)?))
            }
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => {
                PipelineLayoutBackend::Dx12(Box::new(device.create_pipeline_layout(descriptor)?))
            }
        };
        Ok(PipelineLayout {
            inner: std::sync::Arc::new(PipelineLayoutInner {
                device_id: self.id(),
                backend,
                _device: self.backend.clone(),
                sets: descriptor
                    .sets
                    .iter()
                    .map(|set| OwnedDescriptorSetLayout {
                        set: set.set,
                        bindings: set.bindings.to_vec(),
                    })
                    .collect(),
                resource_descriptor_count: counts.resources,
                sampler_descriptor_count: counts.samplers,
                push_constants: descriptor.push_constants,
            }),
        })
    }
}

impl PipelineLayout {
    /// The backend of the device that made this.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match &self.inner.backend {
            #[cfg(feature = "vulkan")]
            PipelineLayoutBackend::Vulkan(layout) => {
                let _ = layout;
                Backend::Vulkan
            }
            #[cfg(all(feature = "dx12", windows))]
            PipelineLayoutBackend::Dx12(layout) => {
                let _ = layout;
                Backend::Dx12
            }
        }
    }

    /// How many descriptors that are not samplers every set of this layout has
    /// between them.
    #[must_use]
    pub fn resource_descriptor_count(&self) -> u32 {
        self.inner.resource_descriptor_count
    }

    /// How many sampler descriptors every set of this layout has between them.
    #[must_use]
    pub fn sampler_descriptor_count(&self) -> u32 {
        self.inner.sampler_descriptor_count
    }

    /// The push-constant block declared at creation, if any.
    #[must_use]
    pub fn push_constants(&self) -> Option<PushConstantRange> {
        self.inner.push_constants
    }

    pub(crate) fn device_id(&self) -> u64 {
        self.inner.device_id
    }

    /// How many sets the layout declares.
    pub(crate) fn set_count(&self) -> usize {
        self.inner.sets.len()
    }

    /// What tells this layout from every other alive.
    pub(crate) fn id(&self) -> usize {
        std::sync::Arc::as_ptr(&self.inner) as usize
    }

    pub(crate) fn set_layout(&self, set: u32) -> Option<&OwnedDescriptorSetLayout> {
        self.inner.sets.iter().find(|layout| layout.set == set)
    }

    pub(crate) fn native(&self) -> &PipelineLayoutBackend {
        &self.inner.backend
    }
}

/// The most descriptor sets a pipeline layout declares, and so the most a
/// draw binds at once.
pub const MAX_DESCRIPTOR_SETS: usize = 8;

pub(crate) fn validate_pipeline_layout(
    descriptor: PipelineLayoutDescriptor<'_>,
) -> Result<DescriptorCounts, RhiError> {
    let mut counts = DescriptorCounts {
        resources: 0,
        samplers: 0,
    };
    if let Some(push) = descriptor.push_constants {
        if push.size == 0 || !push.size.is_multiple_of(4) || push.size > MAX_PUSH_CONSTANT_SIZE {
            return Err(RhiError::InvalidPushConstants {
                reason: "push-constant size must be a non-zero multiple of four, at most 128 bytes",
            });
        }
        if push.stages.is_empty() {
            return Err(RhiError::InvalidPushConstants {
                reason: "push-constant visibility must be non-empty",
            });
        }
        // On DX12 the block is root constants at a `b` register, which a
        // uniform buffer of that space cannot also be at.
        if descriptor.sets.get(push.space as usize).is_some_and(|set| {
            set.bindings.iter().any(|binding| {
                binding.descriptor_type == DescriptorType::UniformBuffer
                    && set.register_offsets.register(binding) == Some(push.register)
            })
        }) {
            return Err(RhiError::InvalidPushConstants {
                reason: "a uniform buffer is at the register the push constants have: move one of them",
            });
        }
    }
    if descriptor.sets.len() > MAX_DESCRIPTOR_SETS {
        return Err(RhiError::InvalidDescriptorSetOrder {
            expected: MAX_DESCRIPTOR_SETS as u32 - 1,
            actual: descriptor.sets.len() as u32 - 1,
        });
    }
    for (expected_set, set) in descriptor.sets.iter().enumerate() {
        if set.set != expected_set as u32 {
            return Err(RhiError::InvalidDescriptorSetOrder {
                expected: expected_set as u32,
                actual: set.set,
            });
        }
        for (index, binding) in set.bindings.iter().enumerate() {
            if binding.count == 0 {
                return Err(RhiError::InvalidDescriptorBinding {
                    set: set.set,
                    binding: binding.binding,
                    reason: "descriptor count must be non-zero",
                });
            }
            if set.register_offsets.register(binding).is_none() {
                return Err(RhiError::InvalidDescriptorBinding {
                    set: set.set,
                    binding: binding.binding,
                    reason: "the binding number is below the set's register offset for its type",
                });
            }
            // (In DXIL each texture of such an array has its own sampler
            // register, which one static sampler does not cover.)
            if binding.count != 1
                && matches!(
                    binding.descriptor_type,
                    DescriptorType::CombinedImageSampler(_)
                )
            {
                return Err(RhiError::InvalidDescriptorBinding {
                    set: set.set,
                    binding: binding.binding,
                    reason: "a combined image sampler is one descriptor: for several,                              declare sampled images and a sampler",
                });
            }
            if binding.visibility.is_empty() {
                return Err(RhiError::InvalidDescriptorBinding {
                    set: set.set,
                    binding: binding.binding,
                    reason: "shader visibility must be non-empty",
                });
            }
            if set.bindings[..index]
                .iter()
                .any(|previous| previous.binding == binding.binding)
            {
                return Err(RhiError::InvalidDescriptorBinding {
                    set: set.set,
                    binding: binding.binding,
                    reason: "binding number is duplicated within the set",
                });
            }
            let count = match binding.descriptor_type {
                DescriptorType::Sampler => &mut counts.samplers,
                _ => &mut counts.resources,
            };
            *count =
                count
                    .checked_add(binding.count)
                    .ok_or(RhiError::InvalidDescriptorBinding {
                        set: set.set,
                        binding: binding.binding,
                        reason: "total descriptor count overflows u32",
                    })?;
        }
    }
    Ok(counts)
}

#[cfg(test)]
#[path = "tests/pipeline_layout.rs"]
mod tests;
