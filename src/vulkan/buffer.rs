use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use super::{VulkanDevice, lock_allocator};
use crate::{BufferDescriptor, BufferMemory, BufferUsages, RhiError};

/// One owned `VkBuffer` whose memory comes from the shared device allocator.
/// Host-visible locations are persistently mapped by the allocator.
pub(crate) struct VulkanBuffer {
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,
    pub(crate) handle: vk::Buffer,
    allocation: std::mem::ManuallyDrop<Allocation>,
}

impl VulkanDevice {
    pub(crate) fn create_buffer(
        &self,
        descriptor: BufferDescriptor,
    ) -> Result<VulkanBuffer, RhiError> {
        let mut usage = vk::BufferUsageFlags::empty();
        if descriptor.usages.contains(BufferUsages::COPY_SOURCE) {
            usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if descriptor.usages.contains(BufferUsages::COPY_DESTINATION) {
            usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        if descriptor.usages.contains(BufferUsages::STORAGE) {
            usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if descriptor.usages.contains(BufferUsages::UNIFORM) {
            usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if descriptor.usages.contains(BufferUsages::VERTEX) {
            usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if descriptor.usages.contains(BufferUsages::INDIRECT) {
            usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        }
        if descriptor.usages.contains(BufferUsages::INDEX) {
            usage |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if descriptor
            .usages
            .contains(BufferUsages::ACCELERATION_STRUCTURE_INPUT)
        {
            usage |= vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
        }
        if descriptor
            .usages
            .contains(BufferUsages::ACCELERATION_STRUCTURE_STORAGE)
        {
            usage |= vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR;
        }
        if descriptor
            .usages
            .contains(BufferUsages::SHADER_BINDING_TABLE)
        {
            usage |= vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR;
        }
        let device_address = descriptor
            .usages
            .contains(BufferUsages::SHADER_DEVICE_ADDRESS)
            || descriptor
                .usages
                .contains(BufferUsages::ACCELERATION_STRUCTURE_INPUT)
            || descriptor
                .usages
                .contains(BufferUsages::ACCELERATION_STRUCTURE_STORAGE)
            || descriptor
                .usages
                .contains(BufferUsages::SHADER_BINDING_TABLE);
        if device_address {
            usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        }
        let location = match descriptor.memory {
            BufferMemory::Device => MemoryLocation::GpuOnly,
            BufferMemory::Upload => MemoryLocation::CpuToGpu,
            BufferMemory::Readback => MemoryLocation::GpuToCpu,
        };

        let create_info = vk::BufferCreateInfo::default()
            .size(descriptor.size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the create info contains no dangling pointers.
        let handle = unsafe { self.logical.handle.create_buffer(&create_info, None) }
            .map_err(|result| RhiError::vk("vkCreateBuffer", result))?;
        // SAFETY: `handle` belongs to this live device.
        let requirements = unsafe { self.logical.handle.get_buffer_memory_requirements(handle) };
        let allocation = self.lock_allocator().allocate(&AllocationCreateDesc {
            name: "erhi buffer",
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });
        let allocation = match allocation {
            Ok(allocation) => allocation,
            Err(error) => {
                // SAFETY: no memory was bound, so only the buffer exists.
                unsafe { self.logical.handle.destroy_buffer(handle, None) };
                return Err(self.allocation_error("allocate buffer memory", error.to_string()));
            }
        };
        // SAFETY: the allocation satisfies this buffer's requirements and both
        // objects belong to this device.
        let bound = unsafe {
            self.logical
                .handle
                .bind_buffer_memory(handle, allocation.memory(), allocation.offset())
        };
        if let Err(result) = bound {
            // SAFETY: binding failed, so the buffer and allocation are unused.
            unsafe { self.logical.handle.destroy_buffer(handle, None) };
            let _ = self.lock_allocator().free(allocation);
            return Err(RhiError::vk("vkBindBufferMemory", result));
        }
        Ok(VulkanBuffer {
            device: self.logical.handle.clone(),
            allocator: self.allocator(),
            handle,
            allocation: std::mem::ManuallyDrop::new(allocation),
        })
    }
}

impl VulkanBuffer {
    pub(crate) fn allocation_size(&self) -> u64 {
        self.allocation.size()
    }

    fn mapped_pointer(&self, operation: &'static str) -> Result<*mut u8, RhiError> {
        self.allocation
            .mapped_ptr()
            .map(|pointer| pointer.as_ptr().cast::<u8>())
            .ok_or(RhiError::BufferMemoryUnavailable {
                memory: "device",
                operation,
            })
    }

    pub(crate) fn write(&self, offset: u64, bytes: &[u8]) -> Result<(), RhiError> {
        let mapped = self.mapped_pointer("writes")?;
        // SAFETY: the common layer validated the range against the logical
        // buffer size, and the persistent mapping covers the allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.add(offset as usize), bytes.len());
        }
        Ok(())
    }

    pub(crate) fn read(&self, offset: u64, size: u64) -> Result<Vec<u8>, RhiError> {
        let mapped = self.mapped_pointer("reads")?;
        let mut bytes = vec![0_u8; size as usize];
        // SAFETY: the common layer validated the range, and the persistent
        // mapping covers the allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped.add(offset as usize),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        Ok(bytes)
    }

    pub(crate) fn write_rows(
        &self,
        offset: u64,
        bytes: &[u8],
        row_bytes: usize,
        row_pitch: usize,
    ) -> Result<(), RhiError> {
        let mapped = self.mapped_pointer("writes")?;
        // SAFETY: the common layer validates the entire strided destination and upload memory.
        unsafe {
            for (row, source) in bytes.chunks_exact(row_bytes).enumerate() {
                std::ptr::copy_nonoverlapping(
                    source.as_ptr(),
                    mapped.add(offset as usize + row * row_pitch),
                    row_bytes,
                );
            }
        }
        Ok(())
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        // SAFETY: owners defer drops until the queue completed their frames,
        // and this owner holds the only raw buffer handle.
        unsafe { self.device.destroy_buffer(self.handle, None) };
        // SAFETY: this is the drop, so `allocation` is taken once and never
        // used again.
        let allocation = unsafe { std::mem::ManuallyDrop::take(&mut self.allocation) };
        let _ = lock_allocator(&self.allocator).free(allocation);
    }
}
