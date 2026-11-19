use std::fmt;
use std::sync::Arc;

use crate::device::BackendDevice;
#[cfg(all(feature = "dx12", windows))]
use crate::dx12::Dx12RayTracingPipeline;
#[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
use crate::pipeline_layout::PipelineLayoutBackend;
#[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
use crate::shader::ShaderModuleBackend;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanRayTracingPipeline;
use crate::{Backend, Device, PipelineLayout, RhiError, ShaderModule};

/// Largest hit-attribute size both APIs guarantee.
pub const MAX_HIT_ATTRIBUTE_SIZE: u32 = 32;

/// One entry point of one shader module: what a stage of a pipeline runs,
/// whether a render, a compute or a ray-tracing one.
/// [`ShaderModule::entry`] makes one.
///
/// The stages of a ray-tracing pipeline may all come from one library (what Slang and
/// HLSL compile to) or each from its own module (what GLSL compiles
/// to, every entry point named `main`). On Direct3D 12 the modules of one
/// pipeline are linked together, so no two of them may export one name.
#[derive(Debug, Clone, Copy)]
pub struct ShaderEntry<'a> {
    /// The module the entry point is in.
    pub module: &'a ShaderModule,
    /// The entry point's name in that module.
    pub entry: &'a str,
}

impl ShaderModule {
    /// The entry point of this module with the name `entry`.
    #[must_use]
    pub fn entry<'a>(&'a self, entry: &'a str) -> ShaderEntry<'a> {
        ShaderEntry {
            module: self,
            entry,
        }
    }
}

/// One hit group: the shaders run for the geometry of the instances that
/// select it.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct HitGroup<'a> {
    /// Stable group name, used as the DX12 hit-group export.
    pub name: &'a str,
    /// The shader that finds where a ray meets procedural geometry (boxes).
    /// `None` makes this a group for triangles, which the device intersects
    /// itself.
    pub intersection: Option<ShaderEntry<'a>>,
    /// Run for the nearest hit a ray ends with.
    pub closest_hit: Option<ShaderEntry<'a>>,
    /// Asked of every hit of geometry that is not opaque, to take it or
    /// let the ray pass (cut-outs).
    pub any_hit: Option<ShaderEntry<'a>>,
}

impl<'a> HitGroup<'a> {
    /// A group called `name` with no shaders yet: give it a closest-hit or
    /// an any-hit one, and an intersection one if its geometry is boxes.
    #[must_use]
    pub const fn new(name: &'a str) -> Self {
        Self {
            name,
            intersection: None,
            closest_hit: None,
            any_hit: None,
        }
    }

    /// The same group for procedural geometry, which `shader` finds the
    /// hits of.
    #[must_use]
    pub const fn with_intersection(mut self, shader: ShaderEntry<'a>) -> Self {
        self.intersection = Some(shader);
        self
    }

    /// The same group, running `shader` for the nearest hit.
    #[must_use]
    pub const fn with_closest_hit(mut self, shader: ShaderEntry<'a>) -> Self {
        self.closest_hit = Some(shader);
        self
    }

    /// The same group, asking `shader` of every hit of geometry that is
    /// not opaque.
    #[must_use]
    pub const fn with_any_hit(mut self, shader: ShaderEntry<'a>) -> Self {
        self.any_hit = Some(shader);
        self
    }
}

/// A ray-tracing pipeline description.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RayTracingPipelineDescriptor<'a> {
    /// A name for the pipeline, shown by debuggers and in errors.
    pub label: &'a str,
    /// The descriptor sets and push constants the shaders use.
    pub layout: &'a PipelineLayout,
    /// The shader run once for each ray launched by
    /// [`crate::CommandEncoder::trace_rays`].
    pub ray_generation: ShaderEntry<'a>,
    /// Ordered miss entries; the trace-call miss index selects into this list.
    pub miss_entries: &'a [ShaderEntry<'a>],
    /// Ordered hit groups; an instance's record offset and the trace call's
    /// select into this list.
    pub hit_groups: &'a [HitGroup<'a>],
    /// The deepest rays are traced from within hit and miss shaders: one where
    /// only the ray-generation shader traces.
    pub max_recursion_depth: u32,
    /// The size of the largest payload any ray carries, in bytes.
    pub max_payload_size: u32,
    /// The size of the largest hit attributes, in bytes: eight for triangles
    /// (two barycentric floats).
    pub max_attribute_size: u32,
}

impl<'a> RayTracingPipelineDescriptor<'a> {
    /// A pipeline that launches rays with `ray_generation`, whose largest
    /// payload is `max_payload_size` bytes. It has no miss shaders and no
    /// hit groups yet, traces from the ray-generation shader alone, and
    /// takes the hit attributes of triangles (eight bytes).
    #[must_use]
    pub const fn new(
        label: &'a str,
        layout: &'a PipelineLayout,
        ray_generation: ShaderEntry<'a>,
        max_payload_size: u32,
    ) -> Self {
        Self {
            label,
            layout,
            ray_generation,
            miss_entries: &[],
            hit_groups: &[],
            max_recursion_depth: 1,
            max_payload_size,
            max_attribute_size: 8,
        }
    }

    /// The miss shaders, which a trace call picks among by index.
    #[must_use]
    pub const fn with_miss_entries(mut self, entries: &'a [ShaderEntry<'a>]) -> Self {
        self.miss_entries = entries;
        self
    }

    /// The hit groups, which instances pick among by index.
    #[must_use]
    pub const fn with_hit_groups(mut self, groups: &'a [HitGroup<'a>]) -> Self {
        self.hit_groups = groups;
        self
    }

    /// The deepest rays are traced from within hit and miss shaders.
    #[must_use]
    pub const fn with_max_recursion_depth(mut self, depth: u32) -> Self {
        self.max_recursion_depth = depth;
        self
    }

    /// The size of the largest hit attributes, in bytes.
    #[must_use]
    pub const fn with_max_attribute_size(mut self, size: u32) -> Self {
        self.max_attribute_size = size;
        self
    }
}

/// One stage as a backend accepts it: its own module, and the entry in it.
#[derive(Debug)]
#[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
pub(crate) struct StageOf<'a, M> {
    pub module: &'a M,
    pub entry: &'a str,
}

/// One hit group as a backend accepts it.
#[derive(Debug)]
#[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
pub(crate) struct HitGroupOf<'a, M> {
    // Direct3D 12 exports a group by its name; Vulkan knows it by index.
    #[cfg_attr(not(all(feature = "dx12", windows)), allow(dead_code))]
    pub name: &'a str,
    pub intersection: Option<StageOf<'a, M>>,
    pub closest_hit: Option<StageOf<'a, M>>,
    pub any_hit: Option<StageOf<'a, M>>,
}

/// The descriptor as a backend accepts it: every stage with the backend's
/// own module, and no layout.
#[derive(Debug)]
#[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
pub(crate) struct RayTracingPipelineSettings<'a, M> {
    pub label: &'a str,
    pub ray_generation: StageOf<'a, M>,
    pub miss_entries: Vec<StageOf<'a, M>>,
    pub hit_groups: Vec<HitGroupOf<'a, M>>,
    pub max_recursion_depth: u32,
    // Payload/attribute sizes configure the DXR shader-config subobject;
    // Vulkan carries them inside the SPIR-V itself.
    #[cfg_attr(not(all(feature = "dx12", windows)), allow(dead_code))]
    pub max_payload_size: u32,
    #[cfg_attr(not(all(feature = "dx12", windows)), allow(dead_code))]
    pub max_attribute_size: u32,
}

impl<'a> RayTracingPipelineDescriptor<'a> {
    /// The descriptor for one backend: `native` gives that backend's module
    /// of a common one, or refuses a module of the other backend.
    #[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
    pub(crate) fn settings<M>(
        &self,
        native: impl Fn(&'a ShaderModule) -> Result<&'a M, RhiError>,
    ) -> Result<RayTracingPipelineSettings<'a, M>, RhiError> {
        let stage = |entry: ShaderEntry<'a>| {
            Ok::<_, RhiError>(StageOf {
                module: native(entry.module)?,
                entry: entry.entry,
            })
        };
        Ok(RayTracingPipelineSettings {
            label: self.label,
            ray_generation: stage(self.ray_generation)?,
            miss_entries: self
                .miss_entries
                .iter()
                .map(|entry| stage(*entry))
                .collect::<Result<_, _>>()?,
            hit_groups: self
                .hit_groups
                .iter()
                .map(|group| {
                    Ok(HitGroupOf {
                        name: group.name,
                        intersection: group.intersection.map(&stage).transpose()?,
                        closest_hit: group.closest_hit.map(&stage).transpose()?,
                        any_hit: group.any_hit.map(&stage).transpose()?,
                    })
                })
                .collect::<Result<_, RhiError>>()?,
            max_recursion_depth: self.max_recursion_depth,
            max_payload_size: self.max_payload_size,
            max_attribute_size: self.max_attribute_size,
        })
    }
}

/// A ray-tracing pipeline with its shader binding table.
#[derive(Clone)]
pub struct RayTracingPipeline {
    device_id: u64,
    // The native pipeline references the layout at bind time.
    layout: PipelineLayout,
    pub(crate) backend: RayTracingPipelineBackend,
    _device: BackendDevice,
}

#[derive(Clone)]
pub(crate) enum RayTracingPipelineBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Arc<VulkanRayTracingPipeline>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Arc<Dx12RayTracingPipeline>),
}

impl fmt::Debug for RayTracingPipeline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RayTracingPipeline")
            .field("backend", &self.backend())
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Create a ray-tracing pipeline and its shader binding table.
    pub fn create_ray_tracing_pipeline(
        &self,
        descriptor: &RayTracingPipelineDescriptor<'_>,
    ) -> Result<RayTracingPipeline, RhiError> {
        if !self.capabilities().ray_tracing {
            return Err(RhiError::RayTracingUnsupported);
        }
        if descriptor.layout.device_id() != self.id()
            || pipeline_stages(descriptor).any(|stage| stage.module.device_id() != self.id())
        {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        validate_ray_tracing_pipeline(descriptor, self.capabilities().max_ray_recursion_depth)?;
        for stage in pipeline_stages(descriptor) {
            if stage.module.has_entry_point(stage.entry) == Some(false) {
                return Err(RhiError::MissingShaderExport {
                    entry: stage.entry.to_owned(),
                });
            }
        }

        let backend: RayTracingPipelineBackend = match (&self.backend, descriptor.layout.native()) {
            #[cfg(feature = "vulkan")]
            (BackendDevice::Vulkan(device), PipelineLayoutBackend::Vulkan(layout)) => {
                let settings = descriptor.settings(|module| match module.native() {
                    ShaderModuleBackend::Vulkan(module) => Ok(&**module),
                    #[cfg(all(feature = "dx12", windows))]
                    ShaderModuleBackend::Dx12(_) => Err(RhiError::ResourceBackendMismatch),
                })?;
                RayTracingPipelineBackend::Vulkan(Arc::new(
                    device.create_ray_tracing_pipeline(&settings, layout)?,
                ))
            }
            #[cfg(all(feature = "dx12", windows))]
            (BackendDevice::Dx12(device), PipelineLayoutBackend::Dx12(layout)) => {
                let settings = descriptor.settings(|module| match module.native() {
                    ShaderModuleBackend::Dx12(module) => Ok(&**module),
                    #[cfg(feature = "vulkan")]
                    ShaderModuleBackend::Vulkan(_) => Err(RhiError::ResourceBackendMismatch),
                })?;
                RayTracingPipelineBackend::Dx12(Arc::new(
                    device.create_ray_tracing_pipeline(&settings, layout)?,
                ))
            }
            #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
            _ => return Err(RhiError::ResourceBackendMismatch),
        };
        Ok(RayTracingPipeline {
            device_id: self.id(),
            layout: descriptor.layout.clone(),
            backend,
            _device: self.backend.clone(),
        })
    }
}

impl RayTracingPipeline {
    /// The backend of the device that made this.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match self.backend {
            #[cfg(feature = "vulkan")]
            RayTracingPipelineBackend::Vulkan(_) => Backend::Vulkan,
            #[cfg(all(feature = "dx12", windows))]
            RayTracingPipelineBackend::Dx12(_) => Backend::Dx12,
        }
    }

    pub(crate) fn device_id(&self) -> u64 {
        self.device_id
    }

    /// The common layout this pipeline binds with.
    #[must_use]
    pub fn layout(&self) -> &PipelineLayout {
        &self.layout
    }
}

/// Every stage the descriptor has, for validation.
fn pipeline_stages<'a>(
    descriptor: &'a RayTracingPipelineDescriptor<'a>,
) -> impl Iterator<Item = ShaderEntry<'a>> {
    std::iter::once(descriptor.ray_generation)
        .chain(descriptor.miss_entries.iter().copied())
        .chain(descriptor.hit_groups.iter().flat_map(|group| {
            group
                .intersection
                .into_iter()
                .chain(group.closest_hit)
                .chain(group.any_hit)
        }))
}

pub(crate) fn validate_ray_tracing_pipeline(
    descriptor: &RayTracingPipelineDescriptor<'_>,
    device_max_recursion: u32,
) -> Result<(), RhiError> {
    if descriptor.ray_generation.entry.is_empty() {
        return Err(RhiError::InvalidRayTracingPipeline {
            reason: "a ray-generation entry point is required",
        });
    }
    if descriptor.max_recursion_depth == 0 || descriptor.max_recursion_depth > device_max_recursion
    {
        return Err(RhiError::InvalidRayTracingPipeline {
            reason: "recursion depth must be at least one and within the device limit",
        });
    }
    if descriptor.max_payload_size == 0 {
        return Err(RhiError::InvalidRayTracingPipeline {
            reason: "the maximum payload size must be non-zero",
        });
    }
    if descriptor.max_attribute_size == 0 || descriptor.max_attribute_size > MAX_HIT_ATTRIBUTE_SIZE
    {
        return Err(RhiError::InvalidRayTracingPipeline {
            reason: "the maximum attribute size must be 1 through 32 bytes",
        });
    }
    for (index, group) in descriptor.hit_groups.iter().enumerate() {
        if group.name.is_empty() {
            return Err(RhiError::InvalidRayTracingPipeline {
                reason: "hit groups need a non-empty name",
            });
        }
        if [group.intersection, group.closest_hit, group.any_hit]
            .into_iter()
            .flatten()
            .any(|stage| stage.entry.is_empty())
        {
            return Err(RhiError::InvalidRayTracingPipeline {
                reason: "a hit group's entry-point names must be non-empty",
            });
        }
        if group.intersection.is_none() && group.closest_hit.is_none() && group.any_hit.is_none() {
            return Err(RhiError::InvalidRayTracingPipeline {
                reason: "a hit group for triangles needs a closest-hit or an any-hit entry point",
            });
        }
        if descriptor.hit_groups[..index]
            .iter()
            .any(|previous| previous.name == group.name)
        {
            return Err(RhiError::InvalidRayTracingPipeline {
                reason: "hit group names must be unique",
            });
        }
        // (Direct3D 12 exports a group under its name beside the shaders
        // under theirs: one name cannot be both.)
        if pipeline_stages(descriptor).any(|stage| stage.entry == group.name) {
            return Err(RhiError::InvalidRayTracingPipeline {
                reason: "a hit group cannot have the name of a shader entry point",
            });
        }
    }
    Ok(())
}

/// Shader-binding-table byte layout shared by both backends.
///
/// Records are handle-size values padded to the handle alignment; each region
/// starts at the base alignment. On DX12 those constants are 32/32/64; Vulkan
/// substitutes the device's reported alignments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(any(test, feature = "vulkan", all(feature = "dx12", windows)))]
pub(crate) struct ShaderBindingTableLayout {
    pub handle_size: u32,
    pub raygen_offset: u64,
    pub raygen_size: u64,
    pub miss_offset: u64,
    pub miss_stride: u64,
    pub miss_size: u64,
    pub hit_offset: u64,
    pub hit_stride: u64,
    pub hit_size: u64,
    pub total_size: u64,
}

#[cfg(any(test, feature = "vulkan", all(feature = "dx12", windows)))]
pub(crate) fn compute_sbt_layout(
    handle_size: u32,
    handle_alignment: u32,
    base_alignment: u32,
    miss_count: u64,
    hit_count: u64,
) -> ShaderBindingTableLayout {
    let align = |value: u64, alignment: u64| value.div_ceil(alignment) * alignment;
    let record_stride = align(u64::from(handle_size), u64::from(handle_alignment));
    let raygen_offset = 0;
    let raygen_size = record_stride;
    let miss_offset = align(raygen_offset + raygen_size, u64::from(base_alignment));
    let miss_size = record_stride * miss_count;
    let hit_offset = align(miss_offset + miss_size, u64::from(base_alignment));
    let hit_size = record_stride * hit_count;
    ShaderBindingTableLayout {
        handle_size,
        raygen_offset,
        raygen_size,
        miss_offset,
        miss_stride: record_stride,
        miss_size,
        hit_offset,
        hit_stride: record_stride,
        hit_size,
        total_size: hit_offset + hit_size.max(record_stride),
    }
}

#[cfg(test)]
#[path = "tests/ray_tracing_pipeline.rs"]
mod tests;
