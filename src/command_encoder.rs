use crate::image::Subresources;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::render_commands::Command;
use crate::{
    Backend, Buffer, BufferUsages, ComputePipeline, DescriptorSet, Device,
    IMAGE_COPY_OFFSET_ALIGNMENT, Image, ImageAspect, ImageUsages, ImageView, RayTracingPipeline,
    RenderPipeline, ResourceState, RhiError, VertexStep,
};

#[cfg(all(feature = "dx12", windows))]
use crate::dx12::Dx12CommandEncoder;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanCommandEncoder;

/// How the color attachment's existing contents are treated at pass begin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadOp {
    /// Keep what the image already contains (UI drawn over a scene).
    Load,
    /// Clear to the given linear RGBA color.
    Clear([f32; 4]),
}

/// How a depth attachment's existing depth is treated at pass begin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DepthLoadOp {
    /// The depth already in the attachment is kept.
    Load,
    /// Clear to this depth (1.0 is the far plane).
    Clear(f32),
}

/// How a depth attachment's existing stencil is treated at pass begin. Of
/// no effect on a format without stencil.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StencilLoadOp {
    /// The stencil already in the attachment is kept.
    Load,
    /// Every stencil value is set to this.
    Clear(u8),
}

/// One colour target of a pass: the first mip of `view`.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ColorAttachment<'a> {
    /// The mip and layer drawn into.
    pub view: &'a ImageView,
    /// What becomes of the colour already there when the pass begins.
    pub load: LoadOp,
    /// Where the samples of a multisampled `view` are averaged into when
    /// the pass ends: a view of one sample a texel, of the same format and
    /// extent, its image a render target.
    /// `None` keeps the samples as they are.
    pub resolve: Option<&'a ImageView>,
}

impl<'a> ColorAttachment<'a> {
    /// `view` as a colour target, with what becomes of what is in it.
    #[must_use]
    pub const fn new(view: &'a ImageView, load: LoadOp) -> Self {
        Self {
            view,
            load,
            resolve: None,
        }
    }

    /// The same target, its samples averaged into `resolve` when the pass
    /// ends.
    #[must_use]
    pub const fn with_resolve(mut self, resolve: &'a ImageView) -> Self {
        self.resolve = Some(resolve);
        self
    }
}

/// The depth-stencil target of a pass: the first mip of `view`.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct DepthAttachment<'a> {
    /// The mip and layer of the depth image used.
    pub view: &'a ImageView,
    /// What becomes of the depth already there when the pass begins.
    pub depth: DepthLoadOp,
    /// What becomes of the stencil already there when the pass begins.
    pub stencil: StencilLoadOp,
}

impl<'a> DepthAttachment<'a> {
    /// `view` as the depth-stencil target, cleared when the pass begins:
    /// the depth to 1 (the far plane) and the stencil to 0.
    #[must_use]
    pub const fn new(view: &'a ImageView) -> Self {
        Self {
            view,
            depth: DepthLoadOp::Clear(1.0),
            stencil: StencilLoadOp::Clear(0),
        }
    }

    /// The same target, with this done to the depth already in it.
    #[must_use]
    pub const fn with_depth_load(mut self, depth: DepthLoadOp) -> Self {
        self.depth = depth;
        self
    }

    /// The same target, with this done to the stencil already in it.
    #[must_use]
    pub const fn with_stencil_load(mut self, stencil: StencilLoadOp) -> Self {
        self.stencil = stencil;
        self
    }
}

/// What a raster pass draws into: up to [`crate::MAX_COLOR_TARGETS`] colour
/// targets and an optional depth-stencil target, all of one extent.
///
/// ```text
/// // One colour target:
/// encoder.begin_render_pass(&RenderPassDescriptor::color(&view, LoadOp::Clear([0.0; 4])))?;
/// // Two, and depth:
/// encoder.begin_render_pass(
///     &RenderPassDescriptor::new()
///         .with_color(ColorAttachment::new(&albedo, LoadOp::Load))
///         .with_color(ColorAttachment::new(&normals, LoadOp::Load))
///         .with_depth(DepthAttachment::new(&depth)),
/// )?;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RenderPassDescriptor<'a> {
    color: [Option<ColorAttachment<'a>>; crate::MAX_COLOR_TARGETS],
    /// How many colour targets were given, which may be more than are held.
    color_count: usize,
    depth: Option<DepthAttachment<'a>>,
}

impl Default for RenderPassDescriptor<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> RenderPassDescriptor<'a> {
    /// A pass with no targets yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            color: [None; crate::MAX_COLOR_TARGETS],
            color_count: 0,
            depth: None,
        }
    }

    /// A pass that draws into `view` alone.
    #[must_use]
    pub const fn color(view: &'a ImageView, load: LoadOp) -> Self {
        Self::new().with_color(ColorAttachment::new(view, load))
    }

    /// The same pass with one more colour target, after those it has: the
    /// targets are in the order of the pipeline's.
    #[must_use]
    pub const fn with_color(mut self, attachment: ColorAttachment<'a>) -> Self {
        if self.color_count < crate::MAX_COLOR_TARGETS {
            self.color[self.color_count] = Some(attachment);
        }
        self.color_count += 1;
        self
    }

    /// The same pass with a depth-stencil target (or, given `None`, with
    /// none).
    #[must_use]
    pub fn with_depth(mut self, attachment: impl Into<Option<DepthAttachment<'a>>>) -> Self {
        self.depth = attachment.into();
        self
    }

    /// The colour targets, in their order.
    pub fn color_attachments(&self) -> impl Iterator<Item = ColorAttachment<'a>> + '_ {
        self.color.iter().flatten().copied()
    }

    /// The depth-stencil target, if the pass has one.
    #[must_use]
    pub const fn depth_attachment(&self) -> Option<DepthAttachment<'a>> {
        self.depth
    }
}

/// The width of the indices of an indexed draw.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexFormat {
    /// Indices of sixteen bits.
    Uint16,
    /// Indices of thirty-two bits.
    #[default]
    Uint32,
}

impl IndexFormat {
    /// Bytes of one index.
    #[must_use]
    pub const fn size(self) -> u64 {
        match self {
            Self::Uint16 => 2,
            Self::Uint32 => 4,
        }
    }
}

/// The indices of an indexed draw.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct IndexBufferBinding<'a> {
    /// The buffer the indices are in.
    pub buffer: &'a Buffer,
    /// How wide each index is.
    pub format: IndexFormat,
}

impl<'a> IndexBufferBinding<'a> {
    /// The indices in `buffer`, each as wide as `format` says.
    #[must_use]
    pub const fn new(buffer: &'a Buffer, format: IndexFormat) -> Self {
        Self { buffer, format }
    }
}

/// One draw: the pipeline, what it reads, and how much of it.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct DrawDescriptor<'a> {
    /// The pipeline that draws.
    pub pipeline: &'a RenderPipeline,
    /// One for each set the pipeline's layout declares, in its order.
    pub descriptor_sets: &'a [&'a DescriptorSet],
    /// One buffer for each of the pipeline's vertex streams, in their order.
    pub vertex_buffers: &'a [&'a Buffer],
    /// `None` draws the vertices in the order they are stored.
    pub indices: Option<IndexBufferBinding<'a>>,
    /// The first index of an indexed draw, otherwise the first vertex.
    pub first: u32,
    /// How many indices, otherwise how many vertices.
    pub count: u32,
    /// Added to every index before it picks its vertex.
    pub base_vertex: i32,
    /// The first record read of the per-instance streams.
    pub first_instance: u32,
    /// How many instances are drawn: one for a draw that is not instanced.
    pub instance_count: u32,
    /// The same size contract as [`CommandEncoder::dispatch`].
    pub push_constants: &'a [u8],
}

impl<'a> DrawDescriptor<'a> {
    /// A draw with `pipeline` that as yet reads nothing and draws nothing:
    /// one instance, no sets, no vertex streams, no push constants.
    ///
    /// ```text
    /// encoder.draw(
    ///     &DrawDescriptor::new(&pipeline)
    ///         .with_sets(&[&set])
    ///         .with_vertex_buffers(&[&vertices])
    ///         .with_range(0, 3),
    /// )?;
    /// ```
    #[must_use]
    pub const fn new(pipeline: &'a RenderPipeline) -> Self {
        Self {
            pipeline,
            descriptor_sets: &[],
            vertex_buffers: &[],
            indices: None,
            first: 0,
            count: 0,
            base_vertex: 0,
            first_instance: 0,
            instance_count: 1,
            push_constants: &[],
        }
    }

    /// One set for each the pipeline's layout declares, in its order.
    #[must_use]
    pub const fn with_sets(mut self, sets: &'a [&'a DescriptorSet]) -> Self {
        self.descriptor_sets = sets;
        self
    }

    /// One buffer for each of the pipeline's vertex streams, in their order.
    #[must_use]
    pub const fn with_vertex_buffers(mut self, buffers: &'a [&'a Buffer]) -> Self {
        self.vertex_buffers = buffers;
        self
    }

    /// Draw by the indices in `buffer`, each as wide as `format` says.
    #[must_use]
    pub const fn with_indices(mut self, buffer: &'a Buffer, format: IndexFormat) -> Self {
        self.indices = Some(IndexBufferBinding::new(buffer, format));
        self
    }

    /// How much is drawn: `count` indices from `first` where there are
    /// indices, vertices otherwise.
    #[must_use]
    pub const fn with_range(mut self, first: u32, count: u32) -> Self {
        self.first = first;
        self.count = count;
        self
    }

    /// Added to every index before it picks its vertex.
    #[must_use]
    pub const fn with_base_vertex(mut self, base_vertex: i32) -> Self {
        self.base_vertex = base_vertex;
        self
    }

    /// Draw `count` instances, the per-instance streams read from record
    /// `first`.
    #[must_use]
    pub const fn with_instances(mut self, first: u32, count: u32) -> Self {
        self.first_instance = first;
        self.instance_count = count;
        self
    }

    /// The push constants of the draw. A draw of the same pass that gives
    /// none is given those of the draw before it.
    #[must_use]
    pub const fn with_push_constants(mut self, bytes: &'a [u8]) -> Self {
        self.push_constants = bytes;
        self
    }
}

/// One compute dispatch: the pipeline, what it reads, and how many groups.
///
/// ```text
/// encoder.dispatch(
///     &DispatchDescriptor::new(&pipeline)
///         .with_sets(&[&set])
///         .with_groups([width.div_ceil(8), height.div_ceil(8), 1]),
/// )?;
/// ```
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct DispatchDescriptor<'a> {
    /// The pipeline that runs.
    pub pipeline: &'a ComputePipeline,
    /// One for each set the pipeline's layout declares, in its order.
    pub descriptor_sets: &'a [&'a DescriptorSet],
    /// As long as the layout's push constants are, or empty where it
    /// declares none.
    pub push_constants: &'a [u8],
    /// How many thread groups along each axis. An indirect dispatch reads
    /// them from its buffer instead.
    pub groups: [u32; 3],
}

impl<'a> DispatchDescriptor<'a> {
    /// One group of `pipeline`, with no sets and no push constants yet.
    #[must_use]
    pub const fn new(pipeline: &'a ComputePipeline) -> Self {
        Self {
            pipeline,
            descriptor_sets: &[],
            push_constants: &[],
            groups: [1, 1, 1],
        }
    }

    /// One set for each the pipeline's layout declares, in its order.
    #[must_use]
    pub const fn with_sets(mut self, sets: &'a [&'a DescriptorSet]) -> Self {
        self.descriptor_sets = sets;
        self
    }

    /// The push constants of the dispatch.
    #[must_use]
    pub const fn with_push_constants(mut self, bytes: &'a [u8]) -> Self {
        self.push_constants = bytes;
        self
    }

    /// How many thread groups along each axis.
    #[must_use]
    pub const fn with_groups(mut self, groups: [u32; 3]) -> Self {
        self.groups = groups;
        self
    }
}

/// One ray dispatch: the pipeline, what it reads, and how many rays.
///
/// ```text
/// encoder.trace_rays(
///     &TraceRaysDescriptor::new(&pipeline, width, height).with_sets(&[&set]),
/// )?;
/// ```
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TraceRaysDescriptor<'a> {
    /// The pipeline whose ray-generation shader runs.
    pub pipeline: &'a RayTracingPipeline,
    /// One for each set the pipeline's layout declares, in its order.
    pub descriptor_sets: &'a [&'a DescriptorSet],
    /// How many rays across: `DispatchRaysDimensions().x`.
    pub width: u32,
    /// How many rays down: `DispatchRaysDimensions().y`.
    pub height: u32,
}

impl<'a> TraceRaysDescriptor<'a> {
    /// `width` by `height` rays of `pipeline`, with no sets yet.
    #[must_use]
    pub const fn new(pipeline: &'a RayTracingPipeline, width: u32, height: u32) -> Self {
        Self {
            pipeline,
            descriptor_sets: &[],
            width,
            height,
        }
    }

    /// One set for each the pipeline's layout declares, in its order.
    #[must_use]
    pub const fn with_sets(mut self, sets: &'a [&'a DescriptorSet]) -> Self {
        self.descriptor_sets = sets;
        self
    }
}

/// One owned graphics command packet.
///
/// Validation runs while building the packet. Native recording and submission run
/// on the calling thread. Standalone `finish` waits for GPU completion; frame-ring
/// submission waits only for native recording, then returns with the GPU in flight.
pub struct CommandEncoder<'device> {
    device_id: u64,
    /// Extent of the open render pass, `None` outside a pass. Compute,
    /// copies, transitions, and `finish` are rejected while a pass is open.
    render_pass_extent: Option<(u32, u32)>,
    render_pass_depth: Option<crate::ImageFormat>,
    render_pass_color: Vec<crate::ImageFormat>,
    /// Samples a texel of the open pass's attachments.
    render_pass_samples: u32,
    /// Whether an occlusion query is open in the open pass.
    pub(crate) occlusion_query_open: bool,
    pub(crate) commands: Vec<Command>,
    /// What is known of the state each mip of each layer of the images used
    /// here will be in when the commands so far have run: by the image and
    /// the mip's place in it.
    image_states: HashMap<(usize, u32), Known>,
    /// Those written through a storage binding since their last barrier.
    images_written: std::collections::HashSet<(usize, u32)>,
    /// While a pass is open: where its begin command is. What a draw needs
    /// of an image is recorded before that, since nothing can be
    /// transitioned inside a pass.
    pass_begin: Option<usize>,
    /// While a pass is open: what each mip of each layer is needed for in
    /// it.
    pass_uses: HashMap<(usize, u32), PassUse>,
    /// The images those are of, and the order they were first used in.
    pass_images: HashMap<usize, Image>,
    pass_order: Vec<usize>,
    /// The state each device buffer used here is in as the commands so far
    /// leave it. A buffer not here has not been used by this encoder yet:
    /// Direct3D 12 hands every buffer back in its common state when a
    /// command list has run, from which its first use takes it wherever
    /// that use needs, and Vulkan has no state of a buffer to keep.
    buffer_states: HashMap<usize, ResourceState>,
    /// Buffers written through a storage binding since their last barrier.
    buffers_written: std::collections::HashSet<usize>,
    /// While a pass is open: the push constants its last draw was given,
    /// and the layout they were for. A draw that gives none is given these.
    pass_push: Option<(usize, Vec<u8>)>,
    /// While a pass is open: what each buffer is needed for in it.
    pass_buffers: HashMap<usize, PassBuffer>,
    /// Descriptor sets that bind nothing writable and whose images are, as
    /// the commands stand, already as their use needs them: by the set's
    /// version and what used it. Kept so that a thousand draws with one set
    /// look at its images once.
    sets_served: HashMap<usize, DescriptorSet>,
    device: &'device Device,
}

/// What is known, as commands are recorded, of the state an image will be
/// in when they run.
#[derive(Debug, Clone, Copy)]
enum Known {
    /// In this state: an explicit transition put it there.
    State(ResourceState),
    /// In one of these: a use of it was served by whichever it was in.
    Within(crate::StateSet),
}

/// What one mip of one layer of an image is needed for within the open
/// pass.
#[derive(Debug, Clone, Copy)]
struct PassUse {
    need: ImageNeed,
    /// It was written through a storage binding before the pass and not
    /// waited for since.
    barrier: bool,
    /// It is an attachment: put right as the pass began.
    placed: bool,
}

/// What a command needs an image for. Needs of one image within one render
/// pass add up (a set may bind an image for the vertex shader and another
/// for the fragment shader); what they add up to is served by one state or
/// by none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ImageNeed(u16);

impl ImageNeed {
    /// Read by shaders: sampled.
    const SAMPLED: Self = Self(1 << 0);
    /// Read and written through a storage binding.
    const STORAGE: Self = Self(1 << 1);
    pub(crate) const COLOR_TARGET: Self = Self(1 << 2);
    pub(crate) const DEPTH_TARGET: Self = Self(1 << 3);
    pub(crate) const COPY_SOURCE: Self = Self(1 << 4);
    pub(crate) const COPY_DESTINATION: Self = Self(1 << 5);

    const fn written(self) -> bool {
        self.0 & Self::STORAGE.0 != 0
    }

    /// The states that serve this, and the one an image in none of them is
    /// moved to. `None`: no one state serves all of it. An image is used
    /// for one thing at a time, so only a single need is served.
    fn states(self) -> Option<(crate::StateSet, ResourceState)> {
        use ResourceState as S;

        let state = match self {
            Self::SAMPLED => S::ShaderResource,
            Self::STORAGE => S::UnorderedAccess,
            Self::COLOR_TARGET => S::RenderTarget,
            Self::DEPTH_TARGET => S::DepthWrite,
            Self::COPY_SOURCE => S::CopySource,
            Self::COPY_DESTINATION => S::CopyDest,
            _ => return None,
        };
        Some((crate::StateSet::of(&[state]), state))
    }
}

/// What a command needs a buffer for. As with an image, the needs of one
/// buffer within one pass add up, to something one state serves or to
/// nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BufferNeed(u16);

impl BufferNeed {
    pub(crate) const VERTEX: Self = Self(1 << 0);
    pub(crate) const INDEX: Self = Self(1 << 1);
    pub(crate) const INDIRECT: Self = Self(1 << 2);
    /// Read through a uniform-buffer binding.
    const UNIFORM: Self = Self(1 << 3);
    /// Read through a read-only storage binding.
    const READ: Self = Self(1 << 4);
    /// Read and written through a storage binding.
    const STORAGE: Self = Self(1 << 5);
    pub(crate) const COPY_SOURCE: Self = Self(1 << 6);
    pub(crate) const COPY_DESTINATION: Self = Self(1 << 7);

    const fn written(self) -> bool {
        self.0 & Self::STORAGE.0 != 0
    }

    /// The states that serve this, and the one a buffer in none of them is
    /// moved to. `None`: no one state serves all of it.
    fn states(self) -> Option<(crate::StateSet, ResourceState)> {
        use ResourceState as S;

        let writes = Self::STORAGE.0 | Self::COPY_DESTINATION.0;
        // A read of one kind has its own state, and the state that is
        // every read serves it as well; reads of several kinds are served
        // by that one alone.
        let read = |state| Some((crate::StateSet::of(&[state, S::GenericRead]), state));
        match self {
            Self(0) => None,
            Self::STORAGE => Some((
                crate::StateSet::of(&[S::UnorderedAccess]),
                S::UnorderedAccess,
            )),
            Self::COPY_DESTINATION => Some((crate::StateSet::of(&[S::CopyDest]), S::CopyDest)),
            _ if self.0 & writes != 0 => None,
            Self::VERTEX => read(S::VertexBuffer),
            Self::INDEX => read(S::IndexBuffer),
            Self::INDIRECT => read(S::IndirectArgument),
            Self::UNIFORM => read(S::ConstantBuffer),
            Self::READ => read(S::ShaderResource),
            Self::COPY_SOURCE => read(S::CopySource),
            _ => Some((crate::StateSet::of(&[S::GenericRead]), S::GenericRead)),
        }
    }
}

/// What a buffer is needed for within the open pass.
#[derive(Debug, Clone, Copy)]
struct PassBuffer {
    need: BufferNeed,
    /// The command, before the pass's begin, that puts it right.
    slot: Option<usize>,
    /// The state it was in when the pass first used it (`None`: not yet
    /// used by this encoder).
    at_entry: Option<ResourceState>,
}

/// The images and buffers a command writes through storage bindings.
#[derive(Debug, Default)]
struct Written {
    images: Vec<(usize, u32)>,
    buffers: Vec<usize>,
}

impl fmt::Debug for CommandEncoder<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CommandEncoder")
            .field("backend", &self.backend())
            .finish_non_exhaustive()
    }
}

pub(crate) enum CommandEncoderBackend<'device> {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<VulkanCommandEncoder<'device>>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Box<Dx12CommandEncoder<'device>>),
}

impl<'device> CommandEncoder<'device> {
    pub(crate) fn new(device: &'device Device) -> Self {
        Self {
            device_id: device.id(),
            render_pass_extent: None,
            render_pass_depth: None,
            render_pass_color: Vec::new(),
            render_pass_samples: 1,
            occlusion_query_open: false,
            commands: Vec::new(),
            image_states: HashMap::new(),
            images_written: std::collections::HashSet::new(),
            pass_begin: None,
            pass_uses: HashMap::new(),
            pass_images: HashMap::new(),
            pass_order: Vec::new(),
            buffer_states: HashMap::new(),
            buffers_written: std::collections::HashSet::new(),
            pass_push: None,
            pass_buffers: HashMap::new(),
            sets_served: HashMap::new(),
            device,
        }
    }

    /// Record whatever puts the mips and layers `range` of `image` in a
    /// state that serves `need`, for the command about to be recorded.
    ///
    /// This is what makes transitions automatic. A caller can still set an
    /// image's state explicitly ([`Self::transition_image`]); that state is
    /// kept for as long as it is compatible with each later use.
    ///
    /// Outside a pass the record goes where the command will. Inside one
    /// nothing can be transitioned, so what the pass needs is gathered as
    /// its draws are recorded and put before its begin when it ends; what
    /// one mip of one layer is needed for by every draw of the pass has to
    /// be something one state serves.
    pub(crate) fn use_image(
        &mut self,
        image: &Image,
        range: Subresources,
        need: ImageNeed,
    ) -> Result<(), RhiError> {
        let key = Arc::as_ptr(&image.inner) as usize;
        let descriptor = image.descriptor();
        if range.base_mip + range.mip_count > descriptor.mip_count
            || range.base_layer + range.layer_count > descriptor.layer_count()
        {
            // No such mip: the command that names it is about to be refused.
            return Ok(());
        }
        if self.pass_begin.is_none() {
            let (accepted, preferred) = need.states().ok_or(RhiError::InvalidImageUsage {
                reason: "one mip of an image is bound in ways no one state serves: to be \
                         sampled and to be written by the same command",
            })?;
            let mut served = true;
            let mut written = false;
            for index in range.indices(&descriptor) {
                written |= self.images_written.remove(&(key, index));
                served &= match self.image_states.get(&(key, index)) {
                    Some(Known::State(state)) => accepted.contains(*state),
                    Some(Known::Within(states)) => accepted.holds(*states),
                    None => false,
                };
            }
            if served && !written {
                return Ok(());
            }
            self.commands.push(Command::UseImage {
                image: image.clone(),
                range,
                accepted,
                preferred,
                barrier: written,
            });
            for index in range.indices(&descriptor) {
                let known = match self.image_states.get(&(key, index)).copied() {
                    Some(Known::State(state)) if accepted.contains(state) => Known::State(state),
                    Some(Known::State(_)) => Known::State(preferred),
                    _ => Known::Within(accepted),
                };
                self.image_states.insert((key, index), known);
            }
            self.sets_served.clear();
            return Ok(());
        }
        for index in range.indices(&descriptor) {
            let had = self.pass_uses.get(&(key, index)).copied();
            let all = ImageNeed(had.map_or(0, |had| had.need.0) | need.0);
            if had.is_some_and(|had| had.need == all) {
                continue;
            }
            if all.states().is_none() {
                return Err(RhiError::InvalidRenderPass {
                    reason: "an image is used within one pass in ways no one state serves: an \
                             attachment that is also read, or a sampled image that is also \
                             written",
                });
            }
            let barrier =
                had.is_some_and(|had| had.barrier) | self.images_written.remove(&(key, index));
            self.pass_uses.insert(
                (key, index),
                PassUse {
                    need: all,
                    barrier,
                    placed: false,
                },
            );
        }
        if let std::collections::hash_map::Entry::Vacant(entry) = self.pass_images.entry(key) {
            entry.insert(image.clone());
            self.pass_order.push(key);
        }
        Ok(())
    }

    /// The pass is ending: before its begin, record what puts every image
    /// its draws used as they need it.
    fn place_pass_uses(&mut self) {
        let Some(begin) = self.pass_begin.take() else {
            return;
        };
        let mut commands = Vec::new();
        for key in std::mem::take(&mut self.pass_order) {
            let Some(image) = self.pass_images.remove(&key) else {
                continue;
            };
            let descriptor = image.descriptor();
            let uses: Vec<(u32, PassUse)> = (0..image.subresource_count())
                .filter_map(|index| {
                    let used = self.pass_uses.get(&(key, index))?;
                    (!used.placed).then_some((index, *used))
                })
                .collect();
            let Some(&(_, first)) = uses.first() else {
                continue;
            };
            // All of it for one thing is one record; otherwise one for each
            // mip of each layer.
            let whole = uses.len() as u32 == image.subresource_count()
                && uses.iter().all(|(_, used)| used.need == first.need);
            let records: Vec<(Subresources, ImageNeed, bool)> = if whole {
                vec![(
                    Subresources::whole(&descriptor),
                    first.need,
                    uses.iter().any(|(_, used)| used.barrier),
                )]
            } else {
                uses.iter()
                    .map(|(index, used)| {
                        (
                            Subresources::one(
                                index % descriptor.mip_count,
                                index / descriptor.mip_count,
                            ),
                            used.need,
                            used.barrier,
                        )
                    })
                    .collect()
            };
            for (range, need, barrier) in records {
                // (A use no one state serves was refused as it was gathered.)
                let Some((accepted, preferred)) = need.states() else {
                    continue;
                };
                for index in range.indices(&descriptor) {
                    self.image_states
                        .insert((key, index), Known::Within(accepted));
                }
                commands.push(Command::UseImage {
                    image: image.clone(),
                    range,
                    accepted,
                    preferred,
                    barrier,
                });
            }
        }
        self.commands.splice(begin..begin, commands);
        self.pass_uses.clear();
        self.pass_images.clear();
    }

    /// As [`Self::use_image`], for a buffer.
    ///
    /// Only a device buffer has a state to change: an upload buffer is
    /// always readable and a readback buffer always a copy's destination.
    /// And only within this encoder: a buffer it has not used yet is taken
    /// by its first use wherever that use needs (Direct3D 12 does this of
    /// itself, every buffer being handed back in its common state when a
    /// command list has run; Vulkan has no state of a buffer). What is
    /// recorded is the change from one use to the next.
    pub(crate) fn use_buffer(&mut self, buffer: &Buffer, need: BufferNeed) -> Result<(), RhiError> {
        if buffer.descriptor().memory != crate::BufferMemory::Device {
            return Ok(());
        }
        let key = Arc::as_ptr(&buffer.inner) as usize;
        // From one read to another the state that is every read serves, and
        // stays; anything else is the state the need names.
        let toward = |state: ResourceState, preferred: ResourceState| {
            if state.writes() || preferred.writes() {
                preferred
            } else {
                ResourceState::GenericRead
            }
        };
        let Some(begin) = self.pass_begin else {
            let (accepted, preferred) = need.states().ok_or(RhiError::InvalidBufferUsage {
                reason: "a buffer is bound in ways no one state serves: to be read and to be \
                         written by the same command",
            })?;
            let written = self.buffers_written.remove(&key);
            let (before, after) = match self.buffer_states.get(&key).copied() {
                None => {
                    self.commands.push(Command::SyncBuffer {
                        buffer: buffer.clone(),
                        after: preferred,
                    });
                    self.buffer_states.insert(key, preferred);
                    return Ok(());
                }
                Some(state) if accepted.contains(state) => {
                    if !(written && state.writes()) {
                        return Ok(());
                    }
                    (state, state)
                }
                Some(state) => (state, toward(state, preferred)),
            };
            self.commands.push(Command::TransitionBuffer {
                buffer: buffer.clone(),
                before,
                after,
            });
            self.buffer_states.insert(key, after);
            self.sets_served.clear();
            return Ok(());
        };
        let had = self.pass_buffers.get(&key).copied();
        let (all, slot, at_entry) = match had {
            Some(had) => (BufferNeed(had.need.0 | need.0), had.slot, had.at_entry),
            None => (need, None, self.buffer_states.get(&key).copied()),
        };
        if had.is_some_and(|had| had.need == all) {
            return Ok(());
        }
        let (accepted, preferred) = all.states().ok_or(RhiError::InvalidRenderPass {
            reason: "a buffer is used within one pass in ways no one state serves: read by one \
                     draw and written by another",
        })?;
        // The state it has to be in when the pass begins. One the encoder
        // had not used is taken there by the pass's first use of it, unless
        // the pass needs it for more than one thing.
        let target = match at_entry {
            Some(state) if accepted.contains(state) => state,
            Some(state) => toward(state, preferred),
            None => preferred,
        };
        let recorded = at_entry.is_none_or(|state| state != target);
        let mut slot = slot;
        if recorded {
            let command = match (at_entry, had) {
                // Its first use by this encoder, and for one thing.
                (None, None) => Command::SyncBuffer {
                    buffer: buffer.clone(),
                    after: target,
                },
                _ => Command::TransitionBuffer {
                    buffer: buffer.clone(),
                    before: at_entry.unwrap_or(ResourceState::Undefined),
                    after: target,
                },
            };
            match slot {
                Some(at) => self.commands[at] = command,
                None => {
                    self.commands.insert(begin, command);
                    self.pass_begin = Some(begin + 1);
                    slot = Some(begin);
                }
            }
            self.sets_served.clear();
        }
        self.pass_buffers.insert(
            key,
            PassBuffer {
                need: all,
                slot,
                at_entry,
            },
        );
        self.buffer_states.insert(key, target);
        Ok(())
    }

    /// What was written by the command just recorded: what uses any of it
    /// next waits for those writes.
    fn mark_written(&mut self, written: Written) {
        self.images_written.extend(written.images);
        self.buffers_written.extend(written.buffers);
    }

    /// Put every image and buffer `sets` bind as their bindings need. Returns
    /// those they bind for writing, to be marked once the command is
    /// recorded.
    fn use_sets(&mut self, sets: &[&DescriptorSet]) -> Result<Written, RhiError> {
        // (Several sets are looked at together every time: what one binds
        // may be bound otherwise by another.)
        let served = match sets {
            [set] => Some(set.version()),
            _ => None,
        };
        if served.is_some_and(|served| self.sets_served.contains_key(&served)) {
            return Ok(Written::default());
        }
        let mut written = Written::default();
        let bound: Vec<(Image, Subresources, ImageNeed)> = sets
            .iter()
            .flat_map(|set| set.images())
            .map(|(view, kind, _)| {
                let need = match kind {
                    crate::DescriptorType::StorageImage => ImageNeed::STORAGE,
                    _ => ImageNeed::SAMPLED,
                };
                (view.image().clone(), Subresources::of(view), need)
            })
            .collect();
        // One image may be bound more than once, and differently: what a
        // mip of it is needed for is all that any binding of that mip needs.
        let mut of_image: HashMap<usize, Vec<usize>> = HashMap::new();
        for (at, (image, ..)) in bound.iter().enumerate() {
            of_image
                .entry(Arc::as_ptr(&image.inner) as usize)
                .or_default()
                .push(at);
        }
        for (image, range, _) in &bound {
            let key = Arc::as_ptr(&image.inner) as usize;
            let need = ImageNeed(
                of_image[&key]
                    .iter()
                    .map(|at| &bound[*at])
                    .filter(|(_, covered, _)| covered.overlaps(*range))
                    .fold(0, |all, (.., need)| all | need.0),
            );
            self.use_image(image, *range, need)?;
            if need.written() {
                written
                    .images
                    .extend(range.indices(&image.descriptor()).map(|index| (key, index)));
            }
        }
        let buffers: Vec<(Buffer, BufferNeed)> = sets
            .iter()
            .flat_map(|set| set.buffers())
            .map(|(buffer, kind, _)| {
                let need = match kind {
                    crate::DescriptorType::UniformBuffer => BufferNeed::UNIFORM,
                    crate::DescriptorType::StorageBuffer => BufferNeed::STORAGE,
                    _ => BufferNeed::READ,
                };
                (buffer.clone(), need)
            })
            .collect();
        // One buffer may be bound more than once, and differently: it is
        // needed for all of that at once.
        let mut merged: Vec<(Buffer, BufferNeed)> = Vec::with_capacity(buffers.len());
        let mut places: HashMap<usize, usize> = HashMap::new();
        for (buffer, need) in buffers {
            match places.entry(Arc::as_ptr(&buffer.inner) as usize) {
                std::collections::hash_map::Entry::Occupied(at) => {
                    merged[*at.get()].1.0 |= need.0;
                }
                std::collections::hash_map::Entry::Vacant(at) => {
                    at.insert(merged.len());
                    merged.push((buffer, need));
                }
            }
        }
        for (buffer, need) in &merged {
            self.use_buffer(buffer, *need)?;
            if need.written() && buffer.descriptor().memory == crate::BufferMemory::Device {
                written.buffers.push(Arc::as_ptr(&buffer.inner) as usize);
            }
        }
        if let Some(served) = served
            && written.images.is_empty()
            && written.buffers.is_empty()
        {
            self.sets_served.insert(served, sets[0].clone());
        }
        Ok(written)
    }

    pub(crate) fn device_id(&self) -> u64 {
        self.device_id
    }

    pub(crate) fn in_render_pass(&self) -> bool {
        self.render_pass_extent.is_some()
    }

    /// A descriptor set is used only with a pipeline whose layout declares
    /// that set exactly as the layout the set was made for: bound to any
    /// other, its bindings would land on whatever that layout has there.
    /// And only once every binding it declares has been written: a shader
    /// that reads one that was not reads nothing that is defined.
    fn require_set_layout(
        &self,
        set: &DescriptorSet,
        layout: &crate::PipelineLayout,
    ) -> Result<(), RhiError> {
        if !set.declared_as_in(layout) {
            return Err(RhiError::DescriptorSetLayoutMismatch { set: set.set() });
        }
        match set.unwritten() {
            Some(binding) => Err(RhiError::InvalidDescriptorBinding {
                set: set.set(),
                binding,
                reason: "the layout declares this binding and nothing was written to it",
            }),
            None => Ok(()),
        }
    }

    /// A command binds one set for each the layout of its pipeline
    /// declares, in the layout's order, each fit for its place.
    fn require_sets(
        &self,
        sets: &[&DescriptorSet],
        layout: &crate::PipelineLayout,
    ) -> Result<(), RhiError> {
        if sets.len() != layout.set_count() {
            return Err(RhiError::DescriptorSetLayoutMismatch {
                set: sets.len().min(layout.set_count()) as u32,
            });
        }
        for (index, set) in sets.iter().enumerate() {
            if set.set() != index as u32 {
                return Err(RhiError::DescriptorSetLayoutMismatch { set: index as u32 });
            }
            self.require_set_layout(set, layout)?;
        }
        Ok(())
    }

    pub(crate) fn require_no_render_pass(&self, operation: &'static str) -> Result<(), RhiError> {
        if self.render_pass_extent.is_some() {
            return Err(RhiError::InvalidRenderPass { reason: operation });
        }
        Ok(())
    }
}

/// The most groups a dispatch has along one axis: the Direct3D 12 limit,
/// and the least Vulkan promises.
pub const MAX_DISPATCH_GROUPS: u32 = 65_535;

impl Device {
    /// Begin a command packet for direct submission on the calling thread.
    pub fn begin_commands(&self) -> Result<CommandEncoder<'_>, RhiError> {
        Ok(CommandEncoder::new(self))
    }
}

impl CommandEncoder<'_> {
    /// The already-selected backend that owns this recording.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.device.backend()
    }

    /// Record a byte-for-byte buffer copy.
    pub fn copy_buffer(
        &mut self,
        source: &Buffer,
        source_offset: u64,
        destination: &Buffer,
        destination_offset: u64,
        size: u64,
    ) -> Result<(), RhiError> {
        self.require_no_render_pass("buffer copies are invalid inside a render pass")?;
        if source.device_id() != self.device_id || destination.device_id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        source.require_usage(BufferUsages::COPY_SOURCE, "copy source")?;
        destination.require_usage(BufferUsages::COPY_DESTINATION, "copy destination")?;
        source.validate_range(source_offset, size)?;
        destination.validate_range(destination_offset, size)?;
        // Not within one buffer, whether or not the ranges meet: Direct3D 12
        // holds a resource in one state, and it cannot be at once what is
        // copied from and what is copied to. (Where they meet, Vulkan does
        // not define the result either.)
        if Arc::ptr_eq(&source.inner, &destination.inner) {
            return Err(RhiError::InvalidCopy {
                reason: "a buffer cannot be copied within itself",
            });
        }
        self.use_buffer(source, BufferNeed::COPY_SOURCE)?;
        self.use_buffer(destination, BufferNeed::COPY_DESTINATION)?;
        if size == 0 {
            return Ok(());
        }

        self.commands.push(Command::CopyBuffer {
            source: source.clone(),
            source_offset,
            destination: destination.clone(),
            destination_offset,
            size,
        });
        Ok(())
    }

    /// Transition `buffer` to `state` explicitly.
    ///
    /// Seldom needed: every command puts the buffers it uses in the state
    /// it needs them in. What the encoder records after this takes the
    /// buffer to be in `state`, and whatever wrote to it before has landed
    /// by then: also the way to make one dispatch's writes to a buffer
    /// wait for another's. An upload or a readback buffer has one state
    /// for life and is left alone.
    pub fn transition_buffer(
        &mut self,
        buffer: &Buffer,
        state: ResourceState,
    ) -> Result<(), RhiError> {
        self.require_no_render_pass("buffer transitions are invalid inside a render pass")?;
        if buffer.device_id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        if state == ResourceState::Undefined {
            return Err(RhiError::InvalidBufferUsage {
                reason: "a buffer is not put in the undefined state: that is where it starts",
            });
        }
        if buffer.descriptor().memory != crate::BufferMemory::Device {
            return Ok(());
        }
        let key = Arc::as_ptr(&buffer.inner) as usize;
        // What it was in is what this encoder last left it in; one the
        // encoder has not used yet is as earlier submissions left it.
        let before = self
            .buffer_states
            .insert(key, state)
            .unwrap_or(ResourceState::Undefined);
        self.commands.push(Command::TransitionBuffer {
            buffer: buffer.clone(),
            before,
            after: state,
        });
        self.buffers_written.remove(&key);
        self.sets_served.clear();
        Ok(())
    }

    /// Record an explicit image state transition, of every mip and layer.
    ///
    /// Seldom needed: every command puts the images it uses in the state it
    /// needs them in, and leaves alone one that a transition recorded here
    /// has already put in a state that serves.
    ///
    /// Transitions are always recorded, even between identical states, so
    /// write-after-write and read-after-write hazards stay ordered.
    pub fn transition_image(
        &mut self,
        image: &Image,
        state: ResourceState,
    ) -> Result<(), RhiError> {
        self.require_no_render_pass("image transitions are invalid inside a render pass")?;
        if image.device_id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        self.commands.push(Command::TransitionImage {
            image: image.clone(),
            state,
        });
        let key = Arc::as_ptr(&image.inner) as usize;
        for index in 0..image.subresource_count() {
            self.image_states.insert((key, index), Known::State(state));
            self.images_written.remove(&(key, index));
        }
        self.sets_served.clear();
        Ok(())
    }

    /// Record one compute dispatch of `groups` thread groups.
    ///
    /// The descriptor's sets are one for each the pipeline's layout declares,
    /// in its order. Its push constants must match the layout's declared
    /// push-constant size exactly, or be empty when the layout declares none.
    pub fn dispatch(&mut self, dispatch: &DispatchDescriptor<'_>) -> Result<(), RhiError> {
        let DispatchDescriptor {
            pipeline,
            descriptor_sets,
            push_constants,
            groups,
        } = *dispatch;
        if groups.contains(&0) {
            return Err(RhiError::InvalidComputePipeline {
                reason: "dispatch group counts must be non-zero",
            });
        }
        if groups.iter().any(|count| *count > MAX_DISPATCH_GROUPS) {
            return Err(RhiError::InvalidComputePipeline {
                reason: "a dispatch has at most 65,535 groups along each axis",
            });
        }
        self.dispatch_impl(pipeline, descriptor_sets, push_constants, groups, None)
    }

    /// Record one compute dispatch whose thread-group counts the GPU wrote:
    /// `arguments` holds, at `offset`, the three words `[x, y, z]`. A count
    /// of zero dispatches nothing.
    pub fn dispatch_indirect(
        &mut self,
        dispatch: &DispatchDescriptor<'_>,
        arguments: &Buffer,
        offset: u64,
    ) -> Result<(), RhiError> {
        let DispatchDescriptor {
            pipeline,
            descriptor_sets,
            push_constants,
            ..
        } = *dispatch;
        if arguments.device_id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        arguments.require_usage(BufferUsages::INDIRECT, "indirect dispatch")?;
        arguments.validate_range(offset, 12)?;
        if !offset.is_multiple_of(4) {
            return Err(RhiError::InvalidBufferSize(offset));
        }
        self.dispatch_impl(
            pipeline,
            descriptor_sets,
            push_constants,
            [0; 3],
            Some((arguments, offset)),
        )
    }

    fn dispatch_impl(
        &mut self,
        pipeline: &ComputePipeline,
        descriptor_sets: &[&DescriptorSet],
        push_constants: &[u8],
        groups: [u32; 3],
        indirect: Option<(&Buffer, u64)>,
    ) -> Result<(), RhiError> {
        self.require_no_render_pass("compute dispatches are invalid inside a render pass")?;
        if pipeline.device_id() != self.device_id
            || descriptor_sets
                .iter()
                .any(|set| set.device_id() != self.device_id)
        {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        self.require_sets(descriptor_sets, pipeline.layout())?;
        let declared = pipeline
            .layout()
            .push_constants()
            .map_or(0, |push| push.size);
        if push_constants.len() as u32 != declared && !push_constants.is_empty() {
            return Err(RhiError::InvalidPushConstants {
                reason: "push-constant bytes must match the layout's declared size",
            });
        }
        if declared != 0 && push_constants.is_empty() {
            return Err(RhiError::InvalidPushConstants {
                reason: "the layout declares push constants but none were provided",
            });
        }
        if let Some((arguments, _)) = indirect {
            self.use_buffer(arguments, BufferNeed::INDIRECT)?;
        }
        let written = self.use_sets(descriptor_sets)?;
        self.commands.push(Command::Dispatch {
            pipeline: pipeline.clone(),
            descriptor_sets: descriptor_sets.iter().map(|set| (*set).clone()).collect(),
            push_constants: push_constants.to_vec(),
            groups,
            indirect: indirect.map(|(buffer, offset)| (buffer.clone(), offset)),
        });
        self.mark_written(written);
        Ok(())
    }

    /// Record a third-party SDK's commands at this point.
    ///
    /// `record` runs when the encoder replays into the backend command list,
    /// with `data`. Every image the SDK reads or writes must be listed in
    /// `images` (kept alive until the submission retires) and already be
    /// transitioned to the state the SDK is told it is in.
    pub fn record_native(
        &mut self,
        record: crate::NativeRecord,
        data: Vec<u8>,
        images: &[&Image],
    ) -> Result<(), RhiError> {
        self.require_no_render_pass("native recordings are invalid inside a render pass")?;
        if images
            .iter()
            .any(|image| image.device_id() != self.device_id)
        {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        self.commands.push(Command::Native {
            record,
            data,
            _images: images.iter().map(|image| (*image).clone()).collect(),
        });
        Ok(())
    }

    /// Record a copy of one mip level from a buffer into an image.
    ///
    /// The buffer bytes must use the shared 256-byte row pitch reported by
    /// [`Image::copy_layout`].
    pub fn copy_buffer_to_image(
        &mut self,
        source: &Buffer,
        source_offset: u64,
        destination: &Image,
        mip: u32,
    ) -> Result<(), RhiError> {
        self.copy_buffer_to_image_impl(
            source,
            source_offset,
            destination,
            mip,
            ImageAspect::All,
            None,
        )
    }

    /// Record a copy from a buffer into one plane of one mip level: the
    /// depth or the stencil of a depth-stencil image. The bytes are laid out
    /// as [`Image::copy_layout_of`] reports, the whole mip at once.
    pub fn copy_buffer_to_image_plane(
        &mut self,
        source: &Buffer,
        source_offset: u64,
        destination: &Image,
        mip: u32,
        aspect: ImageAspect,
    ) -> Result<(), RhiError> {
        self.copy_buffer_to_image_impl(source, source_offset, destination, mip, aspect, None)
    }

    /// Upload complete block rows of a single-layer 2D mip. `rows` is in texels;
    /// its boundaries must be block aligned, except at the bottom of the mip.
    pub fn copy_buffer_to_image_rows(
        &mut self,
        source: &Buffer,
        source_offset: u64,
        destination: &Image,
        mip: u32,
        rows: std::ops::Range<u32>,
    ) -> Result<(), RhiError> {
        self.copy_buffer_to_image_impl(
            source,
            source_offset,
            destination,
            mip,
            ImageAspect::All,
            Some(rows),
        )
    }

    fn copy_buffer_to_image_impl(
        &mut self,
        source: &Buffer,
        source_offset: u64,
        destination: &Image,
        mip: u32,
        aspect: ImageAspect,
        rows: Option<std::ops::Range<u32>>,
    ) -> Result<(), RhiError> {
        self.require_no_render_pass("image copies are invalid inside a render pass")?;
        if destination.device_id() != self.device_id || source.device_id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        source.require_usage(BufferUsages::COPY_SOURCE, "copy source")?;
        destination.require_usage(ImageUsages::COPY_DESTINATION, "copy destination")?;
        self.use_image(
            destination,
            Subresources::mip(&destination.descriptor(), mip),
            ImageNeed::COPY_DESTINATION,
        )?;
        self.use_buffer(source, BufferNeed::COPY_SOURCE)?;
        if !source_offset.is_multiple_of(IMAGE_COPY_OFFSET_ALIGNMENT) {
            return Err(RhiError::UnalignedCopyOffset {
                offset: source_offset,
                required: IMAGE_COPY_OFFSET_ALIGNMENT,
            });
        }
        let mut layout = destination.copy_layout_of(mip, aspect)?;
        if let Some(rows) = &rows {
            let desc = destination.descriptor();
            let (_, height, _) = desc.mip_extent(mip);
            let block = desc.format.block_extent();
            if desc.dimension != crate::ImageDimension::D2
                || desc.depth_or_layers != 1
                || rows.start >= rows.end
                || rows.end > height
                || !rows.start.is_multiple_of(block)
                || (rows.end != height && !rows.end.is_multiple_of(block))
            {
                return Err(RhiError::MismatchedImageCopy {
                    reason: "invalid 2D texture upload rows",
                });
            }
            layout.rows_per_slice = (rows.end - rows.start).div_ceil(block);
            layout.total_size = layout.row_pitch * u64::from(layout.rows_per_slice);
        }
        source.validate_range(source_offset, layout.total_size)?;

        self.commands.push(Command::CopyBufferToImageImpl {
            source: source.clone(),
            source_offset,
            destination: destination.clone(),
            mip,
            aspect,
            rows,
            layout,
        });
        Ok(())
    }

    /// Record a copy of one mip level from an image into a readback buffer.
    ///
    /// The buffer
    /// receives rows padded to the shared 256-byte pitch reported by
    /// [`Image::copy_layout`].
    pub fn copy_image_to_buffer(
        &mut self,
        source: &Image,
        mip: u32,
        destination: &Buffer,
        destination_offset: u64,
    ) -> Result<(), RhiError> {
        self.copy_image_plane_to_buffer(
            source,
            mip,
            ImageAspect::All,
            destination,
            destination_offset,
        )
    }

    /// Record a copy of one plane of one mip level into a readback buffer:
    /// the depth or the stencil of a depth-stencil image, laid out as
    /// [`Image::copy_layout_of`] reports. Depth arrives as it is stored
    /// (`f32` or `u16`), stencil as bytes.
    pub fn copy_image_plane_to_buffer(
        &mut self,
        source: &Image,
        mip: u32,
        aspect: ImageAspect,
        destination: &Buffer,
        destination_offset: u64,
    ) -> Result<(), RhiError> {
        self.require_no_render_pass("image copies are invalid inside a render pass")?;
        if source.device_id() != self.device_id || destination.device_id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        source.require_usage(ImageUsages::COPY_SOURCE, "copy source")?;
        destination.require_usage(BufferUsages::COPY_DESTINATION, "copy destination")?;
        self.use_image(
            source,
            Subresources::mip(&source.descriptor(), mip),
            ImageNeed::COPY_SOURCE,
        )?;
        self.use_buffer(destination, BufferNeed::COPY_DESTINATION)?;
        if !destination_offset.is_multiple_of(IMAGE_COPY_OFFSET_ALIGNMENT) {
            return Err(RhiError::UnalignedCopyOffset {
                offset: destination_offset,
                required: IMAGE_COPY_OFFSET_ALIGNMENT,
            });
        }
        let layout = source.copy_layout_of(mip, aspect)?;
        destination.validate_range(destination_offset, layout.total_size)?;

        self.commands.push(Command::CopyImageToBuffer {
            source: source.clone(),
            mip,
            aspect,
            destination: destination.clone(),
            destination_offset,
            layout,
        });
        Ok(())
    }

    /// Record a whole-image copy between two images of identical dimension,
    /// format, and extent (every mip level and array layer).
    pub fn copy_image(&mut self, source: &Image, destination: &Image) -> Result<(), RhiError> {
        self.require_no_render_pass("image copies are invalid inside a render pass")?;
        if source.device_id() != self.device_id || destination.device_id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        source.require_usage(ImageUsages::COPY_SOURCE, "copy source")?;
        destination.require_usage(ImageUsages::COPY_DESTINATION, "copy destination")?;
        if Arc::ptr_eq(&source.inner, &destination.inner) {
            return Err(RhiError::InvalidCopy {
                reason: "an image cannot be copied onto itself",
            });
        }
        self.use_image(
            source,
            Subresources::whole(&source.descriptor()),
            ImageNeed::COPY_SOURCE,
        )?;
        self.use_image(
            destination,
            Subresources::whole(&destination.descriptor()),
            ImageNeed::COPY_DESTINATION,
        )?;
        let from = source.descriptor();
        let to = destination.descriptor();
        if from.sample_count != to.sample_count {
            return Err(RhiError::MismatchedImageCopy {
                reason: "the images have different sample counts",
            });
        }
        if from.dimension != to.dimension {
            return Err(RhiError::MismatchedImageCopy {
                reason: "the images have different dimensions",
            });
        }
        if from.format != to.format {
            return Err(RhiError::MismatchedImageCopy {
                reason: "the images have different formats",
            });
        }
        if (from.width, from.height, from.depth_or_layers)
            != (to.width, to.height, to.depth_or_layers)
        {
            return Err(RhiError::MismatchedImageCopy {
                reason: "the images have different extents",
            });
        }
        if from.mip_count != to.mip_count {
            return Err(RhiError::MismatchedImageCopy {
                reason: "the images have different mip counts",
            });
        }

        self.commands.push(Command::CopyImage {
            source: source.clone(),
            destination: destination.clone(),
        });
        Ok(())
    }

    /// Record one ray dispatch with the pipeline's shader binding table.
    ///
    /// The descriptor's sets are one for each the pipeline's layout declares,
    /// in its order.
    pub fn trace_rays(&mut self, trace: &TraceRaysDescriptor<'_>) -> Result<(), RhiError> {
        let TraceRaysDescriptor {
            pipeline,
            descriptor_sets,
            width,
            height,
        } = *trace;
        self.require_no_render_pass("ray dispatches are invalid inside a render pass")?;
        if pipeline.device_id() != self.device_id
            || descriptor_sets
                .iter()
                .any(|set| set.device_id() != self.device_id)
        {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        if width == 0 || height == 0 {
            return Err(RhiError::InvalidRayDispatch { width, height });
        }
        // (A device that reports no limit has none to check against.)
        let limit = u64::from(self.device.capabilities().max_ray_dispatch_invocations);
        if limit != 0 && u64::from(width) * u64::from(height) > limit {
            return Err(RhiError::InvalidRayDispatch { width, height });
        }
        self.require_sets(descriptor_sets, pipeline.layout())?;
        let written = self.use_sets(descriptor_sets)?;
        self.commands.push(Command::TraceRays {
            pipeline: pipeline.clone(),
            descriptor_sets: descriptor_sets.iter().map(|set| (*set).clone()).collect(),
            width,
            height,
        });
        self.mark_written(written);
        Ok(())
    }

    /// Begin a raster pass into the first mip of each attachment's view.
    ///
    /// Every colour image needs the `RENDER_TARGET` usage; the depth image
    /// `DEPTH_TARGET`. All views must have one extent and one sample
    /// count, and each must cover a single mip. The viewport and scissor
    /// start as the whole of it. Multisampled colour attachments that name
    /// a resolve target are averaged into it when the pass ends.
    pub fn begin_render_pass(
        &mut self,
        descriptor: &RenderPassDescriptor<'_>,
    ) -> Result<(), RhiError> {
        let color: Vec<ColorAttachment<'_>> = descriptor.color_attachments().collect();
        self.require_no_render_pass("a render pass is already open")?;
        let invalid = |reason| RhiError::InvalidRenderPass { reason };
        if color.is_empty() && descriptor.depth.is_none() {
            return Err(invalid("a render pass needs a color or depth attachment"));
        }
        if descriptor.color_count > crate::MAX_COLOR_TARGETS {
            return Err(invalid(
                "a render pass draws into at most eight color targets",
            ));
        }
        let first = color
            .first()
            .map(|attachment| attachment.view)
            .or(descriptor.depth.map(|attachment| attachment.view))
            .ok_or(RhiError::InvalidRenderPass {
                reason: "a render pass needs an attachment",
            })?;
        let extent = first.extent();
        let layers = first.range().layer_count;
        let mut formats = Vec::with_capacity(color.len());
        for attachment in &color {
            let image = attachment.view.image();
            if image.device_id() != self.device_id {
                return Err(RhiError::ResourceDeviceMismatch);
            }
            image.require_usage(ImageUsages::RENDER_TARGET, "render target")?;
            let format = image.descriptor().format;
            if format.is_depth() {
                return Err(invalid("a color attachment needs a color format"));
            }
            formats.push(format);
        }
        if let Some(attachment) = descriptor.depth {
            let image = attachment.view.image();
            if image.device_id() != self.device_id {
                return Err(RhiError::ResourceDeviceMismatch);
            }
            image.require_usage(ImageUsages::DEPTH_TARGET, "depth target")?;
            if !image.descriptor().format.is_depth()
                || attachment.view.range().aspect != crate::ImageAspect::All
            {
                return Err(invalid(
                    "the depth attachment needs a view of every plane of a depth image",
                ));
            }
        }
        for view in color
            .iter()
            .map(|attachment| attachment.view)
            .chain(descriptor.depth.map(|attachment| attachment.view))
        {
            let range = view.range();
            if view.extent() != extent || range.layer_count != layers {
                return Err(invalid("every attachment of a pass has the same extent"));
            }
            if range.mip_count != 1 {
                return Err(invalid("an attachment's view covers one mip"));
            }
        }
        let samples = first.image().descriptor().sample_count;
        if color
            .iter()
            .map(|attachment| attachment.view)
            .chain(descriptor.depth.map(|attachment| attachment.view))
            .any(|view| view.image().descriptor().sample_count != samples)
        {
            return Err(invalid(
                "every attachment of a pass has the same sample count",
            ));
        }
        for attachment in &color {
            let Some(resolve) = attachment.resolve else {
                continue;
            };
            let target = resolve.image();
            if target.device_id() != self.device_id {
                return Err(RhiError::ResourceDeviceMismatch);
            }
            target.require_usage(ImageUsages::RENDER_TARGET, "resolve target")?;
            let (from, to) = (attachment.view.image().descriptor(), target.descriptor());
            let range = resolve.range();
            if samples == 1
                || to.sample_count != 1
                || from.format != to.format
                || from.format.is_integer()
                || resolve.extent() != extent
                || range.mip_count != 1
                || range.layer_count != 1
            {
                return Err(invalid(
                    "a resolve target is one mip and layer, of one sample a texel, with the format and extent of its multisampled attachment, and not of whole numbers",
                ));
            }
        }
        let mut targets: Vec<(usize, u32, std::ops::Range<u32>)> = Vec::new();
        for view in color
            .iter()
            .flat_map(|attachment| std::iter::once(attachment.view).chain(attachment.resolve))
            .chain(descriptor.depth.map(|attachment| attachment.view))
        {
            let range = view.range();
            let image = Arc::as_ptr(&view.image().inner) as usize;
            let layers = range.base_layer..range.base_layer + range.layer_count;
            if targets.iter().any(|(other, mip, taken)| {
                *other == image
                    && *mip == range.base_mip
                    && taken.start < layers.end
                    && layers.start < taken.end
            }) {
                return Err(invalid(
                    "one mip and layer of an image is one attachment of a pass at the most",
                ));
            }
            targets.push((image, range.base_mip, layers));
        }
        let depth_format = descriptor
            .depth
            .map(|attachment| attachment.view.image().descriptor().format);
        // Every attachment in the state it is drawn into in.
        for attachment in &color {
            for view in std::iter::once(attachment.view).chain(attachment.resolve) {
                self.use_image(
                    view.image(),
                    Subresources::of(view),
                    ImageNeed::COLOR_TARGET,
                )?;
            }
        }
        if let Some(attachment) = descriptor.depth {
            let view = attachment.view;
            self.use_image(
                view.image(),
                Subresources::of(view),
                ImageNeed::DEPTH_TARGET,
            )?;
        }
        self.pass_uses.clear();
        self.pass_images.clear();
        self.pass_order.clear();
        self.pass_buffers.clear();
        self.pass_push = None;
        for (view, need) in color
            .iter()
            .flat_map(|attachment| std::iter::once(attachment.view).chain(attachment.resolve))
            .map(|view| (view, ImageNeed::COLOR_TARGET))
            .chain(
                descriptor
                    .depth
                    .map(|attachment| (attachment.view, ImageNeed::DEPTH_TARGET)),
            )
        {
            let image = view.image();
            for index in Subresources::of(view).indices(&image.descriptor()) {
                self.pass_uses.insert(
                    (Arc::as_ptr(&image.inner) as usize, index),
                    PassUse {
                        need,
                        barrier: false,
                        placed: true,
                    },
                );
            }
        }
        self.sets_served.clear();
        self.pass_begin = Some(self.commands.len());
        self.commands.push(Command::BeginRenderPass {
            color: color
                .iter()
                .map(|attachment| {
                    (
                        attachment.view.clone(),
                        match attachment.load {
                            LoadOp::Load => None,
                            LoadOp::Clear(color) => Some(color),
                        },
                        attachment.resolve.cloned(),
                    )
                })
                .collect(),
            depth: descriptor.depth.map(|attachment| {
                (
                    attachment.view.clone(),
                    match attachment.depth {
                        DepthLoadOp::Load => None,
                        DepthLoadOp::Clear(depth) => Some(depth),
                    },
                    match attachment.stencil {
                        StencilLoadOp::Load => None,
                        StencilLoadOp::Clear(stencil) => Some(stencil),
                    },
                )
            }),
            extent,
        });
        self.render_pass_extent = Some(extent);
        self.render_pass_depth = depth_format;
        self.render_pass_color = formats;
        self.render_pass_samples = samples;
        Ok(())
    }

    /// Whether `pipeline` draws into attachments of the open pass's formats.
    fn require_pass_formats(&self, pipeline: &RenderPipeline) -> Result<(), RhiError> {
        if pipeline.depth_format() != self.render_pass_depth
            || pipeline.color_formats() != self.render_pass_color
        {
            return Err(RhiError::InvalidRenderPass {
                reason: "the pipeline's target formats do not match the pass attachments",
            });
        }
        if self.in_render_pass() && pipeline.sample_count() != self.render_pass_samples {
            return Err(RhiError::InvalidRenderPass {
                reason: "the pipeline's sample count does not match the pass attachments",
            });
        }
        Ok(())
    }

    /// Map clip space onto a rectangle of the open pass, and depth onto
    /// `min_depth..max_depth`. Persists until the next `set_viewport` or
    /// pass. `y` is measured from the top on both backends.
    pub fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Result<(), RhiError> {
        if self.render_pass_extent.is_none() {
            return Err(RhiError::InvalidRenderPass {
                reason: "set_viewport requires an open render pass",
            });
        }
        let finite = [x, y, width, height, min_depth, max_depth]
            .into_iter()
            .all(f32::is_finite);
        if !finite
            || width <= 0.0
            || height <= 0.0
            || !(0.0..=1.0).contains(&min_depth)
            || !(0.0..=1.0).contains(&max_depth)
        {
            return Err(RhiError::InvalidRenderPass {
                reason: "a viewport needs a positive size and depths within 0 to 1",
            });
        }
        self.commands.push(Command::SetViewport {
            rectangle: [x, y, width, height],
            depth: [min_depth, max_depth],
        });
        Ok(())
    }

    /// Restrict raster output to a rectangle of the open pass. Persists
    /// until the next `set_scissor` or `begin_render_pass` (which resets to
    /// the full attachment).
    pub fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) -> Result<(), RhiError> {
        let Some(extent) = self.render_pass_extent else {
            return Err(RhiError::InvalidRenderPass {
                reason: "set_scissor requires an open render pass",
            });
        };
        // Clamp into the attachment: egui-style clip rectangles routinely
        // overshoot by a pixel of rounding.
        let x0 = x.clamp(0, extent.0 as i32);
        let y0 = y.clamp(0, extent.1 as i32);
        let x1 = (x.saturating_add(width as i32)).clamp(x0, extent.0 as i32);
        let y1 = (y.saturating_add(height as i32)).clamp(y0, extent.1 as i32);
        self.commands.push(Command::SetScissor { x0, y0, x1, y1 });
        Ok(())
    }

    /// Record one draw in full: any number of vertex streams, indices of
    /// either width or none, and instances. The draws above are this with
    /// the common cases filled in.
    pub fn draw(&mut self, draw: &DrawDescriptor<'_>) -> Result<(), RhiError> {
        if draw.indices.is_none() && draw.count == 0 {
            return Err(RhiError::InvalidRenderPass {
                reason: "draw needs a non-zero vertex count",
            });
        }
        self.draw_impl(draw, None)
    }

    fn draw_impl(
        &mut self,
        draw: &DrawDescriptor<'_>,
        indirect: Option<(&Buffer, u64, u32)>,
    ) -> Result<(), RhiError> {
        let pipeline = draw.pipeline;
        if self.render_pass_extent.is_none() {
            return Err(RhiError::InvalidRenderPass {
                reason: "a draw requires an open render pass",
            });
        }
        self.require_pass_formats(pipeline)?;
        if pipeline.device_id() != self.device_id
            || draw
                .descriptor_sets
                .iter()
                .any(|set| set.device_id() != self.device_id)
            || draw
                .vertex_buffers
                .iter()
                .any(|buffer| buffer.device_id() != self.device_id)
            || draw
                .indices
                .is_some_and(|indices| indices.buffer.device_id() != self.device_id)
        {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        self.require_sets(draw.descriptor_sets, pipeline.layout())?;
        let streams = pipeline.vertex_streams();
        if draw.vertex_buffers.len() != streams.len() {
            return Err(RhiError::InvalidRenderPipeline {
                reason: "a draw supplies one buffer for each of the pipeline's vertex streams",
            });
        }
        for (buffer, &(stride, step)) in draw.vertex_buffers.iter().zip(streams) {
            buffer.require_usage(BufferUsages::VERTEX, "vertex")?;
            // What the draw reads of this stream, where that is known here:
            // the vertices of an indexed draw are whatever its indices name,
            // and an indirect draw's counts are the GPU's own.
            let records = match step {
                _ if indirect.is_some() => None,
                VertexStep::Instance => Some((draw.first_instance, draw.instance_count)),
                VertexStep::Vertex if draw.indices.is_none() => Some((draw.first, draw.count)),
                VertexStep::Vertex => None,
            };
            if let Some((first, count)) = records {
                let end = (u64::from(first) + u64::from(count))
                    .checked_mul(u64::from(stride))
                    .ok_or(RhiError::InvalidRenderPass {
                        reason: "the draw range overflows a vertex buffer",
                    })?;
                buffer.validate_range(0, end)?;
            }
        }
        if let Some(indices) = draw.indices {
            indices
                .buffer
                .require_usage(BufferUsages::INDEX, "indexed draw indices")?;
            if indirect.is_none() {
                let size = indices.format.size();
                indices
                    .buffer
                    .validate_range(u64::from(draw.first) * size, u64::from(draw.count) * size)?;
            }
        }
        let declared = pipeline
            .layout()
            .push_constants()
            .map_or(0, |range| range.size);
        let given = draw.push_constants.len() as u32;
        if given != declared && given != 0 {
            return Err(RhiError::InvalidPushConstants {
                reason: "push-constant bytes must match the layout's declared size",
            });
        }
        // A draw that supplies no push constants where its layout declares
        // them reuses those of the previous draw, if that draw used the same
        // layout. Both native APIs leave this case undefined.
        let layout = pipeline.layout().id();
        let push_constants = match &self.pass_push {
            _ if given != 0 || declared == 0 => draw.push_constants.to_vec(),
            Some((last, bytes)) if *last == layout => bytes.clone(),
            _ => {
                return Err(RhiError::InvalidPushConstants {
                    reason: "the layout declares push constants: a draw gives them, or follows \
                             one of the same layout in its pass that did",
                });
            }
        };
        if given != 0 {
            self.pass_push = Some((layout, push_constants.clone()));
        }
        for buffer in draw.vertex_buffers {
            self.use_buffer(buffer, BufferNeed::VERTEX)?;
        }
        if let Some(indices) = draw.indices {
            self.use_buffer(indices.buffer, BufferNeed::INDEX)?;
        }
        if let Some((arguments, ..)) = indirect {
            self.use_buffer(arguments, BufferNeed::INDIRECT)?;
        }
        let written = self.use_sets(draw.descriptor_sets)?;
        self.commands.push(Command::Draw {
            pipeline: pipeline.clone(),
            descriptor_sets: draw
                .descriptor_sets
                .iter()
                .map(|set| (*set).clone())
                .collect(),
            vertex_buffers: draw
                .vertex_buffers
                .iter()
                .map(|buffer| (*buffer).clone())
                .collect(),
            indices: draw
                .indices
                .map(|indices| (indices.buffer.clone(), indices.format)),
            indirect: indirect.map(|(buffer, offset, draws)| (buffer.clone(), offset, draws)),
            first: draw.first,
            count: draw.count,
            base_vertex: draw.base_vertex,
            first_instance: draw.first_instance,
            instance_count: draw.instance_count,
            push_constants,
        });
        self.mark_written(written);
        Ok(())
    }

    /// Record a draw whose counts the GPU wrote, and `draw`'s own counts are
    /// not read. `arguments` holds, at `offset`, for an indexed draw the
    /// five words `[index_count, instance_count, first_index, base_vertex,
    /// first_instance]`, and for one without indices the four words
    /// `[vertex_count, instance_count, first_vertex, first_instance]`. The
    /// producer must bound its counts to the supplied streams.
    pub fn draw_indirect(
        &mut self,
        draw: &DrawDescriptor<'_>,
        arguments: &Buffer,
        offset: u64,
    ) -> Result<(), RhiError> {
        self.multi_draw_indirect(draw, arguments, offset, 1)
    }

    /// Record `draw_count` draws of one pipeline and one set of streams,
    /// each with counts the GPU wrote: the records of
    /// [`Self::draw_indirect`], one after the other from `offset`
    /// (20 bytes each for indexed draws, 16 without indices).
    pub fn multi_draw_indirect(
        &mut self,
        draw: &DrawDescriptor<'_>,
        arguments: &Buffer,
        offset: u64,
        draw_count: u32,
    ) -> Result<(), RhiError> {
        if arguments.device_id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        if draw_count == 0 {
            return Err(RhiError::InvalidRenderPass {
                reason: "an indirect draw needs at least one record",
            });
        }
        let stride = if draw.indices.is_some() { 20 } else { 16 };
        arguments.require_usage(BufferUsages::INDIRECT, "indirect draw")?;
        arguments.validate_range(offset, u64::from(draw_count) * stride)?;
        if !offset.is_multiple_of(4) {
            return Err(RhiError::InvalidBufferSize(offset));
        }
        self.draw_impl(draw, Some((arguments, offset, draw_count)))
    }

    /// End the open raster pass.
    pub fn end_render_pass(&mut self) -> Result<(), RhiError> {
        if self.render_pass_extent.is_none() {
            return Err(RhiError::InvalidRenderPass {
                reason: "no render pass is open",
            });
        }
        if self.occlusion_query_open {
            return Err(RhiError::InvalidRenderPass {
                reason: "an occlusion query is still open",
            });
        }
        self.commands.push(Command::EndRenderPass {});
        self.place_pass_uses();
        self.pass_buffers.clear();
        self.pass_push = None;
        self.sets_served.clear();
        self.render_pass_extent = None;
        self.render_pass_depth = None;
        self.render_pass_color.clear();
        Ok(())
    }

    /// End, submit, and synchronously wait for this recording.
    pub fn finish(self) -> Result<(), RhiError> {
        self.require_no_render_pass("finish requires the render pass to be ended first")?;
        crate::submission::execute(&self.device.backend, &self.commands)
    }
}

#[cfg(test)]
#[path = "tests/command_encoder.rs"]
mod tests;
