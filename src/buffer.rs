use std::fmt;
use std::ops::{BitOr, BitOrAssign};

use crate::device::BackendDevice;
#[cfg(all(feature = "dx12", windows))]
use crate::dx12::Dx12Buffer;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanBuffer;
use crate::{Backend, Device, RhiError};

/// CPU/GPU visibility contract for a buffer allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferMemory {
    /// Device memory, which only the device reads and writes: fastest for it,
    /// and reached from the CPU only by copying.
    Device,
    /// Memory the CPU writes ([`Buffer::write`]) and the device reads.
    Upload,
    /// Memory the device writes and the CPU reads ([`Buffer::read`]).
    Readback,
}

impl BufferMemory {
    /// The kind's name in lower case, for messages.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::Upload => "upload",
            Self::Readback => "readback",
        }
    }
}

/// Usage flags for a buffer.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferUsages(u32);

impl BufferUsages {
    /// Copied from.
    pub const COPY_SOURCE: Self = Self(1 << 0);

    /// Copied into.
    pub const COPY_DESTINATION: Self = Self(1 << 1);

    /// Bound as a storage buffer, read-only or not.
    pub const STORAGE: Self = Self(1 << 2);

    /// Bound as a uniform buffer.
    pub const UNIFORM: Self = Self(1 << 3);

    /// Reached by its address on the device rather than through a binding.
    pub const SHADER_DEVICE_ADDRESS: Self = Self(1 << 4);

    /// Read by an acceleration-structure build: geometry, or instances.
    pub const ACCELERATION_STRUCTURE_INPUT: Self = Self(1 << 5);

    /// Holds an acceleration structure.
    pub const ACCELERATION_STRUCTURE_STORAGE: Self = Self(1 << 6);

    /// Holds the shader binding table of a ray-tracing pipeline.
    pub const SHADER_BINDING_TABLE: Self = Self(1 << 7);

    /// Bound as a vertex stream of a draw.
    pub const VERTEX: Self = Self(1 << 8);

    /// Bindable as an index buffer (32-bit indices).
    pub const INDEX: Self = Self(1 << 9);

    /// GPU-generated indexed draw arguments.
    pub const INDIRECT: Self = Self(1 << 10);

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

impl BitOr for BufferUsages {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for BufferUsages {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Describes a buffer: its size, everything it will be used for, and who reaches
/// its memory. [`Device::create_buffer`] makes one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct BufferDescriptor {
    /// The size in bytes: not zero.
    pub size: u64,
    /// Everything the buffer will be used for. A use not named here is refused
    /// later.
    pub usages: BufferUsages,
    /// Whether the CPU writes it, reads it, or neither.
    pub memory: BufferMemory,
}

impl BufferDescriptor {
    /// A buffer of `size` bytes, used as `usages` say, in `memory`.
    #[must_use]
    pub const fn new(size: u64, usages: BufferUsages, memory: BufferMemory) -> Self {
        Self {
            size,
            usages,
            memory,
        }
    }
}

/// A GPU buffer. Clones share the native allocation, so
/// descriptor sets can retain the buffers written into them.
#[derive(Clone)]
pub struct Buffer {
    pub(crate) inner: std::sync::Arc<BufferInner>,
}

pub(crate) struct BufferInner {
    device_id: u64,
    descriptor: BufferDescriptor,
    pub(crate) backend: BufferBackend,
    pub(crate) memory: crate::memory::MemoryTicket,
    // Drop the allocation before releasing its device owner.
    _device: BackendDevice,
}

pub(crate) enum BufferBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<VulkanBuffer>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Box<Dx12Buffer>),
}

impl fmt::Debug for Buffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Buffer")
            .field("backend", &self.backend())
            .field("descriptor", &self.inner.descriptor)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Make a buffer. Its contents are not defined until written.
    pub fn create_buffer(&self, descriptor: BufferDescriptor) -> Result<Buffer, RhiError> {
        if descriptor.size == 0 {
            return Err(RhiError::InvalidBufferSize(descriptor.size));
        }
        if descriptor.usages.is_empty() {
            return Err(RhiError::EmptyBufferUsages);
        }
        let backend = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => {
                BufferBackend::Vulkan(Box::new(device.create_buffer(descriptor)?))
            }
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => {
                BufferBackend::Dx12(Box::new(device.create_buffer(descriptor)?))
            }
        };
        let bytes = match &backend {
            #[cfg(feature = "vulkan")]
            BufferBackend::Vulkan(native) => native.allocation_size(),
            #[cfg(all(feature = "dx12", windows))]
            BufferBackend::Dx12(native) => native.allocation_size(),
        };
        let memory = crate::memory::MemoryTicket::new(
            &self.memory,
            match descriptor.memory {
                BufferMemory::Device => crate::MemoryKind::DeviceBuffer,
                BufferMemory::Upload => crate::MemoryKind::Upload,
                BufferMemory::Readback => crate::MemoryKind::Readback,
            },
            bytes,
        );
        let buffer = Buffer {
            inner: std::sync::Arc::new(BufferInner {
                device_id: self.id(),
                memory,
                descriptor,
                backend,
                _device: self.backend.clone(),
            }),
        };
        buffer.inner.memory.observe_buffer(&buffer.inner);
        Ok(buffer)
    }
}

impl Buffer {
    /// The descriptor the buffer was made from.
    #[must_use]
    pub fn descriptor(&self) -> BufferDescriptor {
        self.inner.descriptor
    }

    /// Name the buffer for capture tools (RenderDoc, PIX) and validation
    /// messages. Of no effect on what is drawn.
    pub fn set_label(&self, label: &str) {
        match (&self.inner._device, &self.inner.backend) {
            #[cfg(feature = "vulkan")]
            (BackendDevice::Vulkan(device), BufferBackend::Vulkan(buffer)) => {
                device.name_object(buffer.handle, label);
            }
            #[cfg(all(feature = "dx12", windows))]
            (BackendDevice::Dx12(_), BufferBackend::Dx12(buffer)) => {
                crate::dx12::name_object(buffer.resource(), label);
            }
            #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
            _ => {}
        }
    }

    pub(crate) fn device_id(&self) -> u64 {
        self.inner.device_id
    }

    /// The backend of the device that made this.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match &self.inner.backend {
            #[cfg(feature = "vulkan")]
            BufferBackend::Vulkan(_) => Backend::Vulkan,
            #[cfg(all(feature = "dx12", windows))]
            BufferBackend::Dx12(_) => Backend::Dx12,
        }
    }

    /// Copy `bytes` into the buffer at `offset`. Only an upload buffer takes
    /// this ([`BufferMemory::Upload`]); the device sees the bytes in whatever
    /// is submitted afterwards.
    pub fn write(&self, offset: u64, bytes: &[u8]) -> Result<(), RhiError> {
        if self.inner.descriptor.memory != BufferMemory::Upload {
            return Err(RhiError::BufferMemoryUnavailable {
                memory: self.inner.descriptor.memory.name(),
                operation: "writes",
            });
        }
        self.validate_range(offset, bytes.len() as u64)?;
        if bytes.is_empty() {
            return Ok(());
        }
        match &self.inner.backend {
            #[cfg(feature = "vulkan")]
            BufferBackend::Vulkan(buffer) => buffer.write(offset, bytes),
            #[cfg(all(feature = "dx12", windows))]
            BufferBackend::Dx12(buffer) => buffer.write(offset, bytes),
        }
    }

    /// The `size` bytes at `offset`. Only a readback buffer gives them
    /// ([`BufferMemory::Readback`]), and they are what the device had written
    /// by the last finished submission.
    pub fn read(&self, offset: u64, size: u64) -> Result<Vec<u8>, RhiError> {
        if self.inner.descriptor.memory != BufferMemory::Readback {
            return Err(RhiError::BufferMemoryUnavailable {
                memory: self.inner.descriptor.memory.name(),
                operation: "reads",
            });
        }
        self.validate_range(offset, size)?;
        if size == 0 {
            return Ok(Vec::new());
        }
        match &self.inner.backend {
            #[cfg(feature = "vulkan")]
            BufferBackend::Vulkan(buffer) => buffer.read(offset, size),
            #[cfg(all(feature = "dx12", windows))]
            BufferBackend::Dx12(buffer) => buffer.read(offset, size),
        }
    }

    pub(crate) fn write_rows(
        &self,
        offset: u64,
        bytes: &[u8],
        row_bytes: usize,
        row_pitch: usize,
    ) -> Result<(), RhiError> {
        let size = row_write_size(bytes, row_bytes, row_pitch)?;
        if self.inner.descriptor.memory != BufferMemory::Upload {
            return Err(RhiError::BufferMemoryUnavailable {
                memory: self.inner.descriptor.memory.name(),
                operation: "writes",
            });
        }
        self.validate_range(offset, size)?;
        match &self.inner.backend {
            #[cfg(feature = "vulkan")]
            BufferBackend::Vulkan(buffer) => buffer.write_rows(offset, bytes, row_bytes, row_pitch),
            #[cfg(all(feature = "dx12", windows))]
            BufferBackend::Dx12(buffer) => buffer.write_rows(offset, bytes, row_bytes, row_pitch),
        }
    }

    pub(crate) fn require_usage(
        &self,
        usage: BufferUsages,
        name: &'static str,
    ) -> Result<(), RhiError> {
        if self.inner.descriptor.usages.contains(usage) {
            Ok(())
        } else {
            Err(RhiError::MissingBufferUsage { required: name })
        }
    }

    pub(crate) fn validate_range(&self, offset: u64, size: u64) -> Result<(), RhiError> {
        if offset
            .checked_add(size)
            .is_some_and(|end| end <= self.inner.descriptor.size)
        {
            Ok(())
        } else {
            Err(RhiError::BufferRangeOutOfBounds {
                buffer_size: self.inner.descriptor.size,
                offset,
                size,
            })
        }
    }
}

pub(crate) fn row_write_size(
    bytes: &[u8],
    row_bytes: usize,
    row_pitch: usize,
) -> Result<u64, RhiError> {
    if row_bytes == 0
        || row_pitch < row_bytes
        || bytes.is_empty()
        || !bytes.len().is_multiple_of(row_bytes)
    {
        return Err(RhiError::InvalidBufferSize(bytes.len() as u64));
    }
    (bytes.len() / row_bytes)
        .checked_mul(row_pitch)
        .map(|size| size as u64)
        .ok_or(RhiError::InvalidBufferSize(u64::MAX))
}

#[cfg(test)]
#[path = "tests/buffer.rs"]
mod tests;
