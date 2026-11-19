use std::fmt;
use std::sync::Arc;

use crate::device::BackendDevice;
#[cfg(all(feature = "dx12", windows))]
use crate::dx12::Dx12RenderPipeline;
use crate::pipeline_layout::PipelineLayoutBackend;
use crate::shader::ShaderModuleBackend;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanRenderPipeline;
use crate::{Backend, Device, ImageFormat, PipelineLayout, RhiError, ShaderModule};

/// Vertex attribute formats: what each attribute's bytes are, and so what
/// the shader's input reads (`Unorm` and `Snorm` as fractions, `Uint` and
/// `Sint` as whole numbers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VertexFormat {
    /// One 32-bit float.
    Float32,
    /// Two 32-bit floats.
    Float32x2,
    /// Three 32-bit floats.
    Float32x3,
    /// Four 32-bit floats.
    Float32x4,
    /// Two 16-bit floats.
    Float16x2,
    /// Four 16-bit floats.
    Float16x4,
    /// One 32-bit unsigned integer, as integers.
    Uint32,
    /// Two 32-bit unsigned integers, as integers.
    Uint32x2,
    /// Three 32-bit unsigned integers, as integers.
    Uint32x3,
    /// Four 32-bit unsigned integers, as integers.
    Uint32x4,
    /// One 32-bit signed integer, as integers.
    Sint32,
    /// Two 32-bit signed integers, as integers.
    Sint32x2,
    /// Three 32-bit signed integers, as integers.
    Sint32x3,
    /// Four 32-bit signed integers, as integers.
    Sint32x4,
    /// Two 16-bit unsigned integers, as integers.
    Uint16x2,
    /// Four 16-bit unsigned integers, as integers.
    Uint16x4,
    /// Two 16-bit signed integers, as integers.
    Sint16x2,
    /// Four 16-bit signed integers, as integers.
    Sint16x4,
    /// Two 16-bit unsigned integers, read as numbers from 0 to 1.
    Unorm16x2,
    /// Four 16-bit unsigned integers, read as numbers from 0 to 1.
    Unorm16x4,
    /// Two 16-bit signed integers, read as numbers from -1 to 1.
    Snorm16x2,
    /// Four 16-bit signed integers, read as numbers from -1 to 1.
    Snorm16x4,
    /// Two 8-bit unsigned integers, as integers.
    Uint8x2,
    /// Four 8-bit unsigned integers, as integers.
    Uint8x4,
    /// Two 8-bit signed integers, as integers.
    Sint8x2,
    /// Four 8-bit signed integers, as integers.
    Sint8x4,
    /// Two 8-bit unsigned integers, read as numbers from 0 to 1.
    Unorm8x2,
    /// Packed colours, as egui-style UIs emit them.
    Unorm8x4,
    /// Two 8-bit signed integers, read as numbers from -1 to 1.
    Snorm8x2,
    /// Four 8-bit signed integers, read as numbers from -1 to 1.
    Snorm8x4,
}

impl VertexFormat {
    /// How many bytes one attribute of this format takes in a vertex.
    #[must_use]
    pub const fn size(self) -> u32 {
        match self {
            Self::Uint8x2 | Self::Sint8x2 | Self::Unorm8x2 | Self::Snorm8x2 => 2,
            Self::Float32
            | Self::Uint32
            | Self::Sint32
            | Self::Float16x2
            | Self::Uint16x2
            | Self::Sint16x2
            | Self::Unorm16x2
            | Self::Snorm16x2
            | Self::Uint8x4
            | Self::Sint8x4
            | Self::Unorm8x4
            | Self::Snorm8x4 => 4,
            Self::Float32x2
            | Self::Uint32x2
            | Self::Sint32x2
            | Self::Float16x4
            | Self::Uint16x4
            | Self::Sint16x4
            | Self::Unorm16x4
            | Self::Snorm16x4 => 8,
            Self::Float32x3 | Self::Uint32x3 | Self::Sint32x3 => 12,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 => 16,
        }
    }
}

/// What Direct3D 12 knows a vertex input by: the semantic the shader gives
/// it. Vulkan knows an input by its location alone.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexSemantic {
    /// `ATTRIB<location>`: a Slang shader that declares its inputs
    /// `ATTRIB0`, `ATTRIB1` and on gets those locations in SPIR-V and
    /// these semantics in DXIL, so one declaration serves both backends.
    #[default]
    Location,
    /// The semantic an HLSL shader gives the input: `POSITION`, or
    /// `TEXCOORD` with index 1 for `TEXCOORD1`.
    Named {
        /// The semantic's name, without its number.
        name: &'static str,
        /// The semantic's number: zero where it has none.
        index: u32,
    },
}

/// One vertex attribute at `location`, read from byte `offset` of each vertex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct VertexAttribute {
    /// The input's location in the vertex shader, which is how Vulkan knows
    /// it.
    pub location: u32,
    /// Where the attribute begins in each record of its stream, in bytes.
    pub offset: u32,
    /// The data format of the attribute.
    pub format: VertexFormat,
    /// Direct3D 12 only: the input's semantic in the vertex shader.
    pub semantic: VertexSemantic,
}

impl VertexAttribute {
    /// The attribute at `location`, by the `ATTRIB<location>` semantic on
    /// Direct3D 12.
    #[must_use]
    pub const fn new(location: u32, offset: u32, format: VertexFormat) -> Self {
        Self {
            location,
            offset,
            format,
            semantic: VertexSemantic::Location,
        }
    }

    /// The same attribute, by the semantic `name` with `index` on
    /// Direct3D 12.
    #[must_use]
    pub const fn with_semantic(self, name: &'static str, index: u32) -> Self {
        Self {
            semantic: VertexSemantic::Named { name, index },
            ..self
        }
    }
}

/// When a vertex stream moves on to its next record.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexStep {
    /// Once per vertex.
    #[default]
    Vertex,
    /// Once per instance.
    Instance,
}

/// The most vertex streams one pipeline reads.
pub const MAX_VERTEX_BUFFERS: usize = 8;

/// Attribute layout of one vertex stream: the records of one buffer.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct VertexBufferLayout<'a> {
    /// The size of one record, in bytes: how far the stream moves on at each
    /// step.
    pub stride: u32,
    /// Whether the stream moves on for each vertex or for each instance.
    pub step: VertexStep,
    /// The attributes read from each record.
    pub attributes: &'a [VertexAttribute],
}

impl<'a> VertexBufferLayout<'a> {
    /// A stream of records `stride` bytes long, one a vertex, holding
    /// `attributes`.
    #[must_use]
    pub const fn new(stride: u32, attributes: &'a [VertexAttribute]) -> Self {
        Self {
            stride,
            step: VertexStep::Vertex,
            attributes,
        }
    }

    /// The same stream, moving on as `step` says.
    #[must_use]
    pub const fn with_step(mut self, step: VertexStep) -> Self {
        self.step = step;
        self
    }
}

/// How the vertices of a draw are assembled into primitives.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveTopology {
    /// Every three vertices are a triangle.
    #[default]
    TriangleList,
    /// Each vertex after the second makes a triangle with the two before it.
    TriangleStrip,
    /// Every two vertices are a line.
    LineList,
    /// Each vertex after the first makes a line with the one before it.
    LineStrip,
    /// Every vertex is a point.
    PointList,
}

/// Which side of a triangle is not drawn.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CullMode {
    /// Every triangle is drawn, whichever way it faces.
    #[default]
    None,
    /// Triangles facing the viewer are dropped.
    Front,
    /// Triangles facing away from the viewer are dropped.
    Back,
}

/// The winding of a triangle's front, as seen on the screen (y up, the
/// same on both backends).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrontFace {
    /// A triangle faces the viewer when its vertices go round counter-
    /// clockwise on screen.
    #[default]
    CounterClockwise,
    /// A triangle faces the viewer when its vertices go round clockwise on
    /// screen.
    Clockwise,
}

/// How primitives are assembled and which are kept. The default draws both
/// sides of a triangle list.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrimitiveState {
    /// How vertices are assembled into primitives.
    pub topology: PrimitiveTopology,
    /// Which triangles are dropped for the way they face.
    pub cull_mode: CullMode,
    /// Which winding counts as facing the viewer.
    pub front_face: FrontFace,
}

/// The four common ways of blending into a colour target. Each is a
/// [`BlendState`] (`None` for `Opaque`): see [`ColorTargetState::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendMode {
    /// Source overwrites the destination.
    Opaque,
    /// Standard alpha blending:
    /// `source.rgb * source.a + destination.rgb * (1 - source.a)` for color,
    /// `source.a + destination.a * (1 - source.a)` for alpha.
    Alpha,
    /// Premultiplied-alpha compositing (`ONE, ONE_MINUS_SRC_ALPHA` for both
    /// colour and alpha), which is what UI libraries such as egui emit.
    PremultipliedAlpha,
    /// Emissive effects: source.rgb * source.a + destination.rgb; preserve target alpha.
    Additive,
}

/// What a source or destination value is multiplied by before blending.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    /// Nought: the term is dropped.
    Zero,
    /// One: the term is kept whole.
    One,
    /// The colour the fragment shader wrote, channel by channel.
    Src,
    /// One less the colour the fragment shader wrote, channel by channel.
    OneMinusSrc,
    /// The alpha the fragment shader wrote.
    SrcAlpha,
    /// One less the alpha the fragment shader wrote.
    OneMinusSrcAlpha,
    /// The colour already in the target, channel by channel.
    Dst,
    /// One less the colour already in the target, channel by channel.
    OneMinusDst,
    /// The alpha already in the target.
    DstAlpha,
    /// One less the alpha already in the target.
    OneMinusDstAlpha,
    /// The lesser of the written alpha and one less the target's alpha; one
    /// for the alpha channel.
    SrcAlphaSaturated,
}

/// How the two multiplied values are combined.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendOperation {
    /// The source term plus the destination term.
    #[default]
    Add,
    /// The source term less the destination term.
    Subtract,
    /// The destination term less the source term.
    ReverseSubtract,
    /// The lesser of the two colours; the factors are not applied.
    Min,
    /// The greater of the two colours; the factors are not applied.
    Max,
}

/// Blending of the colour channels or of alpha:
/// `operation(source * src_factor, destination * dst_factor)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendComponent {
    /// The factor applied to the colour the fragment shader wrote.
    pub src_factor: BlendFactor,
    /// The factor applied to the colour already in the target.
    pub dst_factor: BlendFactor,
    /// How the two products are combined.
    pub operation: BlendOperation,
}

impl BlendComponent {
    const fn add(src_factor: BlendFactor, dst_factor: BlendFactor) -> Self {
        Self {
            src_factor,
            dst_factor,
            operation: BlendOperation::Add,
        }
    }
}

/// Blending into one colour target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendState {
    /// The blend of the red, green and blue channels.
    pub color: BlendComponent,
    /// The blend of the alpha channel.
    pub alpha: BlendComponent,
}

impl BlendMode {
    /// The mode as blend factors; `None` for `Opaque`, which does not blend.
    #[must_use]
    pub const fn state(self) -> Option<BlendState> {
        use BlendFactor::{One, OneMinusSrcAlpha, SrcAlpha, Zero};
        match self {
            Self::Opaque => None,
            // Alpha accumulates at full weight, as in the usual
            // `ALPHA_BLENDING` so all paths agree byte for byte.
            Self::Alpha => Some(BlendState {
                color: BlendComponent::add(SrcAlpha, OneMinusSrcAlpha),
                alpha: BlendComponent::add(One, OneMinusSrcAlpha),
            }),
            Self::PremultipliedAlpha => Some(BlendState {
                color: BlendComponent::add(One, OneMinusSrcAlpha),
                alpha: BlendComponent::add(One, OneMinusSrcAlpha),
            }),
            Self::Additive => Some(BlendState {
                color: BlendComponent::add(SrcAlpha, One),
                alpha: BlendComponent::add(Zero, One),
            }),
        }
    }
}

/// Which channels of a colour target a pipeline writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorWrites(pub(crate) u8);

impl ColorWrites {
    /// No channel is written.
    pub const NONE: Self = Self(0);

    /// The red channel.
    pub const RED: Self = Self(1);

    /// The green channel.
    pub const GREEN: Self = Self(2);

    /// The blue channel.
    pub const BLUE: Self = Self(4);

    /// The alpha channel.
    pub const ALPHA: Self = Self(8);

    /// Every channel: the default.
    pub const ALL: Self = Self(15);

    /// Whether every channel of `other` is among these.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// These channels and those of `other`.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl Default for ColorWrites {
    fn default() -> Self {
        Self::ALL
    }
}

/// The most colour targets one pipeline, and so one pass, draws into.
pub const MAX_COLOR_TARGETS: usize = 8;

/// One colour target of a pipeline: the fragment shader's output of the
/// same index (`SV_Target<n>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorTargetState {
    /// The format of the image drawn into, which the pass's attachment must
    /// have.
    pub format: ImageFormat,
    /// `None` overwrites the target.
    pub blend: Option<BlendState>,
    /// Which channels a draw writes.
    pub write_mask: ColorWrites,
}

impl ColorTargetState {
    /// A target of `format` blended in one of the common ways, every
    /// channel written.
    #[must_use]
    pub const fn new(format: ImageFormat, blend: BlendMode) -> Self {
        Self {
            format,
            blend: blend.state(),
            write_mask: ColorWrites::ALL,
        }
    }
}

/// A test of a new value against the one already there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareFunction {
    /// No value passes.
    Never,
    /// Passes where the new value is less than the one held.
    Less,
    /// Passes where the two are equal.
    Equal,
    /// Passes where the new value is less than or equal to the one held.
    LessEqual,
    /// Passes where the new value is greater than the one held.
    Greater,
    /// Passes where the two differ.
    NotEqual,
    /// Passes where the new value is greater than or equal to the one held.
    GreaterEqual,
    /// Every value passes.
    Always,
}

/// What becomes of a stencil value.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StencilOperation {
    /// The stencil value stays as it is.
    #[default]
    Keep,
    /// The stencil value becomes zero.
    Zero,
    /// Set to the pipeline's reference value.
    Replace,
    /// One is added, stopping at 255.
    IncrementClamp,
    /// One is taken away, stopping at zero.
    DecrementClamp,
    /// Every bit is flipped.
    Invert,
    /// One is added, 255 going round to zero.
    IncrementWrap,
    /// One is taken away, zero going round to 255.
    DecrementWrap,
}

/// The stencil test of one side of a triangle, and what follows from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StencilFaceState {
    /// The reference value against the stored one.
    pub compare: CompareFunction,
    /// When the stencil test fails.
    pub fail: StencilOperation,
    /// When it passes and the depth test fails.
    pub depth_fail: StencilOperation,
    /// When both pass.
    pub pass: StencilOperation,
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self {
            compare: CompareFunction::Always,
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::Keep,
            pass: StencilOperation::Keep,
        }
    }
}

/// Stencil testing and writing, for a depth target whose format has a
/// stencil plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StencilState {
    /// The test and its outcomes for triangles facing the viewer.
    pub front: StencilFaceState,
    /// The test and its outcomes for triangles facing away.
    pub back: StencilFaceState,
    /// The bits of the stencil value and of the reference that the test
    /// compares.
    pub read_mask: u8,
    /// The bits of the stencil value an operation may change.
    pub write_mask: u8,
    /// The value tests compare against and `Replace` writes.
    pub reference: u8,
}

impl Default for StencilState {
    fn default() -> Self {
        Self {
            front: StencilFaceState::default(),
            back: StencilFaceState::default(),
            read_mask: 0xff,
            write_mask: 0xff,
            reference: 0,
        }
    }
}

/// The depth target of a pipeline, and how it is tested and written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthStencilState {
    /// The format of the depth image, which the pass's depth attachment must
    /// have.
    pub format: ImageFormat,
    /// How a fragment's depth is tested against the depth held.
    pub depth_compare: CompareFunction,
    /// Whether a fragment that passes writes its depth.
    pub depth_write: bool,
    /// The stencil test, if the format has a stencil plane and stencil testing is needed.
    pub stencil: Option<StencilState>,
}

impl DepthStencilState {
    /// A 32-bit float depth target tested nearer-or-equal, written or only
    /// tested (blended surfaces test without writing, so that sorted
    /// transparent ones compose).
    #[must_use]
    pub const fn depth32(write: bool) -> Self {
        Self {
            format: ImageFormat::Depth32Float,
            depth_compare: CompareFunction::LessEqual,
            depth_write: write,
            stencil: None,
        }
    }
}

/// How many samples a texel the pipeline draws: the sample count of every
/// attachment of the passes it draws in. The default is one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultisampleState {
    /// 1, 2, 4, 8 or 16.
    pub count: u32,
    /// Turn the alpha of the first colour target into how many of a
    /// texel's samples are covered: cut-outs with soft edges, unsorted.
    pub alpha_to_coverage: bool,
}

impl Default for MultisampleState {
    fn default() -> Self {
        Self {
            count: 1,
            alpha_to_coverage: false,
        }
    }
}

/// Fixed raster depth offset, in depth units plus a slope-scaled term.
#[derive(Debug, Clone, Copy)]
pub struct DepthBias {
    /// Added to every fragment's depth, in the smallest steps the depth format
    /// tells apart.
    pub constant: i32,
    /// Added in proportion to how steeply the triangle's depth changes across
    /// the screen.
    pub slope: f32,
}

impl DepthBias {
    /// Bring coplanar painted layers toward the camera without changing their geometry.
    pub const DECAL: Self = Self {
        constant: -2,
        slope: -1.,
    };
}

/// A render pipeline description: up to [`MAX_VERTEX_BUFFERS`] vertex
/// streams, up to [`MAX_COLOR_TARGETS`] colour targets and an optional
/// depth-stencil target.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RenderPipelineDescriptor<'a> {
    /// A name for the pipeline, shown by debuggers and in errors.
    pub label: &'a str,
    /// The module the vertex shader is in.
    pub vertex_module: &'a ShaderModule,
    /// The vertex shader's entry point.
    pub vertex_entry: &'a str,
    /// May reference the same module as the vertex stage on Vulkan; DX12
    /// modules carry one DXIL entry each, so the library supplies two blobs.
    pub fragment_module: &'a ShaderModule,
    /// The fragment shader's entry point.
    pub fragment_entry: &'a str,
    /// The descriptor sets and push constants the shaders use.
    pub layout: &'a PipelineLayout,
    /// The vertex streams, in the order a draw supplies their buffers
    /// (binding/slot 0, 1, ...). Empty for a shader that works its vertices
    /// out from their index alone.
    pub vertex_buffers: &'a [VertexBufferLayout<'a>],
    /// What the vertices are joined into, and which triangles are dropped.
    pub primitive: PrimitiveState,
    /// How many samples a pixel the attachments have.
    pub multisample: MultisampleState,
    /// The colour targets, in the order of the fragment shader's outputs.
    /// Empty for a depth-only pipeline.
    pub color_targets: &'a [ColorTargetState],
    /// `None` draws without a depth target.
    pub depth_stencil: Option<DepthStencilState>,
    /// An offset to every fragment's depth, for surfaces that lie on others.
    pub depth_bias: Option<DepthBias>,
}

impl<'a> RenderPipelineDescriptor<'a> {
    /// A pipeline of the two shaders that as yet reads no vertex streams
    /// and draws into nothing: triangles, none of them dropped, one sample
    /// a texel.
    ///
    /// ```text
    /// let pipeline = device.create_render_pipeline(
    ///     &RenderPipelineDescriptor::new(
    ///         "triangle",
    ///         &layout,
    ///         vertex_module.entry("vertex_main"),
    ///         fragment_module.entry("fragment_main"),
    ///     )
    ///     .with_vertex_buffers(&[VertexBufferLayout::new(20, &attributes)])
    ///     .with_color_targets(&[ColorTargetState::new(
    ///         ImageFormat::Rgba8Unorm,
    ///         BlendMode::Opaque,
    ///     )]),
    /// )?;
    /// ```
    #[must_use]
    pub fn new(
        label: &'a str,
        layout: &'a PipelineLayout,
        vertex: crate::ShaderEntry<'a>,
        fragment: crate::ShaderEntry<'a>,
    ) -> Self {
        Self {
            label,
            vertex_module: vertex.module,
            vertex_entry: vertex.entry,
            fragment_module: fragment.module,
            fragment_entry: fragment.entry,
            layout,
            vertex_buffers: &[],
            primitive: PrimitiveState::default(),
            multisample: MultisampleState::default(),
            color_targets: &[],
            depth_stencil: None,
            depth_bias: None,
        }
    }

    /// The vertex streams the pipeline reads, in the order a draw gives
    /// their buffers.
    #[must_use]
    pub const fn with_vertex_buffers(mut self, layouts: &'a [VertexBufferLayout<'a>]) -> Self {
        self.vertex_buffers = layouts;
        self
    }

    /// What the vertices are joined into, and which triangles are dropped.
    #[must_use]
    pub const fn with_primitive(mut self, primitive: PrimitiveState) -> Self {
        self.primitive = primitive;
        self
    }

    /// How many samples a texel the attachments have.
    #[must_use]
    pub const fn with_multisample(mut self, multisample: MultisampleState) -> Self {
        self.multisample = multisample;
        self
    }

    /// The colour targets drawn into, in the order a pass gives them.
    #[must_use]
    pub const fn with_color_targets(mut self, targets: &'a [ColorTargetState]) -> Self {
        self.color_targets = targets;
        self
    }

    /// The depth-stencil target and its tests (or, given `None`, no such
    /// target).
    #[must_use]
    pub fn with_depth_stencil(mut self, state: impl Into<Option<DepthStencilState>>) -> Self {
        self.depth_stencil = state.into();
        self
    }

    /// An offset to every fragment's depth.
    #[must_use]
    pub const fn with_depth_bias(mut self, bias: DepthBias) -> Self {
        self.depth_bias = Some(bias);
        self
    }
}

/// A render pipeline.
#[derive(Clone)]
pub struct RenderPipeline {
    device_id: u64,
    layout: PipelineLayout,
    /// Stride and step of each vertex stream, in order.
    vertex_streams: Arc<[(u32, VertexStep)]>,
    color_formats: Arc<[ImageFormat]>,
    depth_format: Option<ImageFormat>,
    sample_count: u32,
    pub(crate) backend: RenderPipelineBackend,
    _device: BackendDevice,
}

#[derive(Clone)]
pub(crate) enum RenderPipelineBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Arc<VulkanRenderPipeline>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Arc<Dx12RenderPipeline>),
}

impl fmt::Debug for RenderPipeline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RenderPipeline")
            .field("backend", &self.backend())
            .field("vertex_streams", &self.vertex_streams)
            .finish_non_exhaustive()
    }
}

fn validate_render_pipeline(descriptor: &RenderPipelineDescriptor<'_>) -> Result<(), RhiError> {
    if descriptor.color_targets.is_empty() && descriptor.depth_stencil.is_none() {
        return Err(RhiError::InvalidRenderPipeline {
            reason: "a render pipeline needs a color or depth attachment",
        });
    }
    if descriptor.color_targets.len() > MAX_COLOR_TARGETS {
        return Err(RhiError::InvalidRenderPipeline {
            reason: "a render pipeline draws into at most eight color targets",
        });
    }
    if descriptor
        .color_targets
        .iter()
        .any(|target| target.format.is_depth() || target.format.is_compressed())
    {
        return Err(RhiError::InvalidRenderPipeline {
            reason: "a color target needs an uncompressed color format",
        });
    }
    if descriptor
        .color_targets
        .iter()
        .any(|target| target.blend.is_some() && target.format.is_integer())
    {
        return Err(RhiError::InvalidRenderPipeline {
            reason: "a color target of whole numbers cannot blend",
        });
    }
    if let Some(depth) = descriptor.depth_stencil {
        if !depth.format.is_depth() {
            return Err(RhiError::InvalidRenderPipeline {
                reason: "the depth target needs a depth format",
            });
        }
        if depth.stencil.is_some() && !depth.format.has_stencil() {
            return Err(RhiError::InvalidRenderPipeline {
                reason: "stencil state needs a depth format with a stencil plane",
            });
        }
    }
    if !matches!(descriptor.multisample.count, 1 | 2 | 4 | 8 | 16) {
        return Err(RhiError::InvalidRenderPipeline {
            reason: "a render pipeline draws 1, 2, 4, 8 or 16 samples a texel",
        });
    }
    if descriptor.vertex_entry.is_empty() || descriptor.fragment_entry.is_empty() {
        return Err(RhiError::InvalidRenderPipeline {
            reason: "vertex and fragment entry-point names must be non-empty",
        });
    }
    if descriptor
        .depth_bias
        .is_some_and(|bias| !bias.slope.is_finite())
    {
        return Err(RhiError::InvalidRenderPipeline {
            reason: "Depth bias must be finite",
        });
    }
    validate_vertex_layouts(descriptor.vertex_buffers)
}

fn validate_vertex_layouts(layouts: &[VertexBufferLayout<'_>]) -> Result<(), RhiError> {
    if layouts.len() > MAX_VERTEX_BUFFERS {
        return Err(RhiError::InvalidRenderPipeline {
            reason: "a render pipeline reads at most eight vertex streams",
        });
    }
    let mut locations = std::collections::HashSet::new();
    for layout in layouts {
        if layout.stride == 0 || layout.attributes.is_empty() {
            return Err(RhiError::InvalidRenderPipeline {
                reason: "the vertex layout needs a non-zero stride and at least one attribute",
            });
        }
        for attribute in layout.attributes {
            if attribute
                .offset
                .checked_add(attribute.format.size())
                .is_none_or(|end| end > layout.stride)
            {
                return Err(RhiError::InvalidRenderPipeline {
                    reason: "a vertex attribute extends past the vertex stride",
                });
            }
            if let VertexSemantic::Named { name, .. } = attribute.semantic
                && (name.is_empty() || name.contains('\0'))
            {
                return Err(RhiError::InvalidRenderPipeline {
                    reason: "a vertex attribute's semantic name must be non-empty text",
                });
            }
            if !locations.insert(attribute.location) {
                return Err(RhiError::InvalidRenderPipeline {
                    reason: "vertex attribute locations must be unique",
                });
            }
        }
    }
    Ok(())
}

impl Device {
    /// Create a graphics pipeline with a color target, depth target, or both.
    pub fn create_render_pipeline(
        &self,
        descriptor: &RenderPipelineDescriptor<'_>,
    ) -> Result<RenderPipeline, RhiError> {
        if descriptor.vertex_module.device_id() != self.id()
            || descriptor.fragment_module.device_id() != self.id()
            || descriptor.layout.device_id() != self.id()
        {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        validate_render_pipeline(descriptor)?;
        // What it draws into, the device must be able to draw into.
        let samples = descriptor.multisample.count;
        for (format, usage) in descriptor
            .color_targets
            .iter()
            .map(|target| (target.format, crate::ImageUsages::RENDER_TARGET))
            .chain(
                descriptor
                    .depth_stencil
                    .iter()
                    .map(|depth| (depth.format, crate::ImageUsages::DEPTH_TARGET)),
            )
        {
            if !self.format_supports(format, usage)
                || (samples > 1 && !self.supports_sample_count(format, samples))
            {
                return Err(RhiError::UnsupportedByDevice {
                    reason: "a target format of the pipeline is not drawn into, or not with this \
                             many samples, on this device",
                });
            }
        }
        for (module, entry) in [
            (descriptor.vertex_module, descriptor.vertex_entry),
            (descriptor.fragment_module, descriptor.fragment_entry),
        ] {
            if module.has_entry_point(entry) == Some(false) {
                return Err(RhiError::MissingShaderExport {
                    entry: entry.to_owned(),
                });
            }
        }
        let backend = match (
            &self.backend,
            descriptor.vertex_module.native(),
            descriptor.fragment_module.native(),
        ) {
            #[cfg(feature = "vulkan")]
            (
                BackendDevice::Vulkan(device),
                ShaderModuleBackend::Vulkan(vertex),
                ShaderModuleBackend::Vulkan(fragment),
            ) => match descriptor.layout.native() {
                PipelineLayoutBackend::Vulkan(layout) => {
                    let vertex_entry = descriptor
                        .vertex_module
                        .resolve_entry(descriptor.vertex_entry);
                    let fragment_entry = descriptor
                        .fragment_module
                        .resolve_entry(descriptor.fragment_entry);
                    RenderPipelineBackend::Vulkan(Arc::new(device.create_render_pipeline(
                        descriptor,
                        vertex_entry,
                        fragment_entry,
                        vertex,
                        fragment,
                        layout,
                    )?))
                }
                #[cfg(all(feature = "dx12", windows))]
                PipelineLayoutBackend::Dx12(_) => {
                    return Err(RhiError::ResourceBackendMismatch);
                }
            },
            #[cfg(all(feature = "dx12", windows))]
            (
                BackendDevice::Dx12(device),
                ShaderModuleBackend::Dx12(vertex),
                ShaderModuleBackend::Dx12(fragment),
            ) => match descriptor.layout.native() {
                PipelineLayoutBackend::Dx12(layout) => RenderPipelineBackend::Dx12(Arc::new(
                    device.create_render_pipeline(descriptor, vertex, fragment, layout)?,
                )),
                #[cfg(feature = "vulkan")]
                PipelineLayoutBackend::Vulkan(_) => {
                    return Err(RhiError::ResourceBackendMismatch);
                }
            },
            #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
            _ => return Err(RhiError::ResourceBackendMismatch),
        };
        Ok(RenderPipeline {
            depth_format: descriptor.depth_stencil.map(|depth| depth.format),
            sample_count: descriptor.multisample.count,
            device_id: self.id(),
            layout: descriptor.layout.clone(),
            vertex_streams: descriptor
                .vertex_buffers
                .iter()
                .map(|layout| (layout.stride, layout.step))
                .collect(),
            color_formats: descriptor
                .color_targets
                .iter()
                .map(|target| target.format)
                .collect(),
            backend,
            _device: self.backend.clone(),
        })
    }
}

impl RenderPipeline {
    /// The format of the depth target this pipeline draws with, if any.
    #[must_use]
    pub const fn depth_format(&self) -> Option<ImageFormat> {
        self.depth_format
    }

    /// Samples a texel of the attachments this pipeline draws into.
    #[must_use]
    pub const fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// The formats of its colour targets, in order.
    #[must_use]
    pub fn color_formats(&self) -> &[ImageFormat] {
        &self.color_formats
    }

    /// The backend of the device that made this pipeline.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            RenderPipelineBackend::Vulkan(_) => Backend::Vulkan,
            #[cfg(all(feature = "dx12", windows))]
            RenderPipelineBackend::Dx12(_) => Backend::Dx12,
        }
    }

    /// The common layout this pipeline binds with.
    #[must_use]
    pub fn layout(&self) -> &PipelineLayout {
        &self.layout
    }

    /// Stride and step of each vertex stream, in the order a draw supplies
    /// their buffers.
    #[must_use]
    pub fn vertex_streams(&self) -> &[(u32, VertexStep)] {
        &self.vertex_streams
    }

    pub(crate) fn device_id(&self) -> u64 {
        self.device_id
    }
}

#[cfg(test)]
#[path = "tests/render_pipeline.rs"]
mod tests;
