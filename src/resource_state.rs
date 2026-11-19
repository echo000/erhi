/// The state of a buffer, or of one mip of one layer of an image: what the
/// resource is currently used for.
///
/// Commands transition the resources they use automatically, so callers
/// rarely name a state. They are needed only for explicit transitions
/// ([`crate::CommandEncoder::transition_image`],
/// [`crate::CommandEncoder::transition_buffer`]). The names are those of
/// Direct3D 12 and of NVRHI; each backend translates them to its own
/// barriers.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum ResourceState {
    /// Nothing is known of it and nothing in it need be kept: what an image
    /// is in when it is made.
    #[default]
    Undefined,
    /// Read as the vertices of a draw.
    VertexBuffer,
    /// Read as the indices of a draw.
    IndexBuffer,
    /// Read through a uniform-buffer binding.
    ConstantBuffer,
    /// Read as the arguments of an indirect draw or dispatch.
    IndirectArgument,
    /// Read by shaders of any stage: a sampled image, a read-only storage
    /// buffer.
    ShaderResource,
    /// Read and written by shaders of any stage: a storage image, a
    /// storage buffer.
    UnorderedAccess,
    /// Drawn into as a colour attachment.
    RenderTarget,
    /// Drawn into as the depth and stencil attachment.
    DepthWrite,
    /// Copied from.
    CopySource,
    /// Copied into.
    CopyDest,
    /// Every way of reading a buffer at once: one that is vertices to one
    /// draw and a shader resource to another.
    GenericRead,
}

impl ResourceState {
    /// Whether what is in the resource may change while it is in this
    /// state.
    #[must_use]
    pub const fn writes(self) -> bool {
        matches!(
            self,
            Self::UnorderedAccess | Self::CopyDest | Self::RenderTarget | Self::DepthWrite
        )
    }

    /// The state's name, for messages.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Undefined => "undefined",
            Self::VertexBuffer => "vertex buffer",
            Self::IndexBuffer => "index buffer",
            Self::ConstantBuffer => "constant buffer",
            Self::IndirectArgument => "indirect argument",
            Self::ShaderResource => "shader resource",
            Self::UnorderedAccess => "unordered access",
            Self::RenderTarget => "render target",
            Self::DepthWrite => "depth write",
            Self::CopySource => "copy source",
            Self::CopyDest => "copy destination",
            Self::GenericRead => "generic read",
        }
    }

    /// The state as one byte, to keep where threads share it.
    pub(crate) const fn bits(self) -> u8 {
        self as u8
    }

    pub(crate) const fn from_bits(bits: u8) -> Self {
        match bits {
            1 => Self::VertexBuffer,
            2 => Self::IndexBuffer,
            3 => Self::ConstantBuffer,
            4 => Self::IndirectArgument,
            5 => Self::ShaderResource,
            6 => Self::UnorderedAccess,
            7 => Self::RenderTarget,
            8 => Self::DepthWrite,
            9 => Self::CopySource,
            10 => Self::CopyDest,
            11 => Self::GenericRead,
            _ => Self::Undefined,
        }
    }
}

/// Some of the states: those a resource may be in and serve a use as it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StateSet(u16);

impl StateSet {
    const fn bit(state: ResourceState) -> u16 {
        1 << state.bits()
    }

    pub(crate) const fn of(states: &[ResourceState]) -> Self {
        let mut bits = 0;
        let mut index = 0;
        while index < states.len() {
            bits |= Self::bit(states[index]);
            index += 1;
        }
        Self(bits)
    }

    pub(crate) const fn contains(self, state: ResourceState) -> bool {
        self.0 & Self::bit(state) != 0
    }

    /// Whether every state of `other` is one of these.
    pub(crate) const fn holds(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}
