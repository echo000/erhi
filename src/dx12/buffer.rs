use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex};

use gpu_allocator::MemoryLocation;
use gpu_allocator::d3d12::{
    Allocator, Resource, ResourceCategory, ResourceCreateDesc, ResourceStateOrBarrierLayout,
    ResourceType,
};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_RANGE, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE,
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
    ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

use super::{Dx12Device, lock_allocator};
use crate::{BufferDescriptor, BufferMemory, BufferUsages, RhiError};

/// One owned placed buffer whose heap comes from the shared device allocator.
pub(crate) struct Dx12Buffer {
    allocator: Arc<Mutex<Allocator>>,
    resource: std::mem::ManuallyDrop<Resource>,
}

impl Dx12Device {
    pub(crate) fn create_buffer(
        &self,
        descriptor: BufferDescriptor,
    ) -> Result<Dx12Buffer, RhiError> {
        let memory_location = match descriptor.memory {
            BufferMemory::Device => MemoryLocation::GpuOnly,
            BufferMemory::Upload => MemoryLocation::CpuToGpu,
            BufferMemory::Readback => MemoryLocation::GpuToCpu,
        };
        let unordered_access = descriptor.memory == BufferMemory::Device
            && (descriptor.usages.contains(BufferUsages::STORAGE)
                || descriptor
                    .usages
                    .contains(BufferUsages::ACCELERATION_STRUCTURE_STORAGE));
        let resource_description = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            // CBV extents are rounded to 256 bytes, even for smaller logical buffers.
            Width: if descriptor.usages.contains(BufferUsages::UNIFORM) {
                descriptor
                    .size
                    .checked_next_multiple_of(256)
                    .ok_or(RhiError::InvalidBufferSize(descriptor.size))?
            } else {
                descriptor.size
            },
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: if unordered_access {
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_FLAG_NONE
            },
        };
        let initial_state = match descriptor.memory {
            BufferMemory::Device
                if descriptor
                    .usages
                    .contains(BufferUsages::ACCELERATION_STRUCTURE_STORAGE) =>
            {
                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE
            }
            // Ordinary device buffers stay in COMMON: Direct3D 12 implicitly
            // promotes buffers from COMMON to the accessed state and decays
            // them back after each completed submission.
            BufferMemory::Device => D3D12_RESOURCE_STATE_COMMON,
            BufferMemory::Upload => D3D12_RESOURCE_STATE_GENERIC_READ,
            BufferMemory::Readback => D3D12_RESOURCE_STATE_COPY_DEST,
        };
        let resource = self
            .lock_allocator()
            .create_resource(&ResourceCreateDesc {
                name: "erhi buffer",
                memory_location,
                resource_category: ResourceCategory::Buffer,
                resource_desc: &resource_description,
                castable_formats: &[],
                clear_value: None,
                initial_state_or_layout: ResourceStateOrBarrierLayout::ResourceState(initial_state),
                resource_type: &ResourceType::Placed,
            })
            .map_err(|error| self.allocation_error("create placed buffer", error.to_string()))?;
        Ok(Dx12Buffer {
            allocator: self.allocator(),
            resource: std::mem::ManuallyDrop::new(resource),
        })
    }
}

impl Dx12Buffer {
    pub(crate) fn allocation_size(&self) -> u64 {
        self.resource.size
    }

    pub(crate) fn resource(&self) -> &ID3D12Resource {
        self.resource.resource()
    }

    pub(crate) fn device_address(&self) -> Result<u64, RhiError> {
        // SAFETY: the placed buffer resource is live.
        let address = unsafe { self.resource().GetGPUVirtualAddress() };
        if address == 0 {
            Err(RhiError::ZeroDeviceAddress("DX12 buffer"))
        } else {
            Ok(address)
        }
    }

    pub(crate) fn write(&self, offset: u64, bytes: &[u8]) -> Result<(), RhiError> {
        let mut mapped = ptr::null_mut::<c_void>();
        let no_cpu_reads = D3D12_RANGE { Begin: 0, End: 0 };
        // SAFETY: the common layer permits writes only to upload heaps and has
        // validated the range. The resource remains mapped only for this copy.
        unsafe {
            self.resource()
                .Map(0, Some(&raw const no_cpu_reads), Some(&mut mapped))
                .map_err(|error| RhiError::dx12("ID3D12Resource::Map(upload)", error))?;
            ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapped.cast::<u8>().add(offset as usize),
                bytes.len(),
            );
            let written = D3D12_RANGE {
                Begin: offset as usize,
                End: offset as usize + bytes.len(),
            };
            self.resource().Unmap(0, Some(&raw const written));
        }
        Ok(())
    }

    pub(crate) fn read(&self, offset: u64, size: u64) -> Result<Vec<u8>, RhiError> {
        let mut bytes = vec![0_u8; size as usize];
        let read = D3D12_RANGE {
            Begin: offset as usize,
            End: (offset + size) as usize,
        };
        let mut mapped = ptr::null_mut::<c_void>();
        // SAFETY: the common layer permits reads only from readback heaps and
        // validated the range. Queue completion is required before this call.
        unsafe {
            self.resource()
                .Map(0, Some(&raw const read), Some(&mut mapped))
                .map_err(|error| RhiError::dx12("ID3D12Resource::Map(readback)", error))?;
            ptr::copy_nonoverlapping(
                mapped.cast::<u8>().add(offset as usize),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
            self.resource().Unmap(0, None);
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
        let mut mapped = ptr::null_mut::<c_void>();
        let no_cpu_reads = D3D12_RANGE { Begin: 0, End: 0 };
        // SAFETY: the common layer validates the entire strided destination and upload memory.
        unsafe {
            self.resource()
                .Map(0, Some(&raw const no_cpu_reads), Some(&mut mapped))
                .map_err(|error| RhiError::dx12("ID3D12Resource::Map(upload)", error))?;
            for (row, source) in bytes.chunks_exact(row_bytes).enumerate() {
                ptr::copy_nonoverlapping(
                    source.as_ptr(),
                    mapped.cast::<u8>().add(offset as usize + row * row_pitch),
                    row_bytes,
                );
            }
            let written = D3D12_RANGE {
                Begin: offset as usize,
                End: offset as usize + bytes.len() / row_bytes * row_pitch,
            };
            self.resource().Unmap(0, Some(&raw const written));
        }
        Ok(())
    }
}

impl Drop for Dx12Buffer {
    fn drop(&mut self) {
        // SAFETY: this is the drop, so `resource` is taken once and never
        // used again.
        let resource = unsafe { std::mem::ManuallyDrop::take(&mut self.resource) };
        let _ = lock_allocator(&self.allocator).free_resource(resource);
    }
}
