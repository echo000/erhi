use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, ID3D12CommandAllocator, ID3D12CommandList, ID3D12Fence,
    ID3D12GraphicsCommandList4,
};
use windows::core::Interface;

use super::Dx12Device;
use crate::RhiError;

struct Dx12FrameContext {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList4,
    fence_value: u64,
    submitted: bool,
}

/// Reusable per-frame command allocators tracked by shared-fence values:
/// recording a new frame in a slot waits only for that slot's previous
/// submission, never the queue.
pub(crate) struct Dx12FrameRing {
    contexts: Vec<Dx12FrameContext>,
    // A clone of the device's shared fence so the drop path can confirm
    // completion without borrowing the device.
    fence: ID3D12Fence,
}

impl Dx12Device {
    pub(crate) fn create_frame_ring(
        &self,
        frames_in_flight: u32,
    ) -> Result<Dx12FrameRing, RhiError> {
        let mut contexts = Vec::with_capacity(frames_in_flight as usize);
        for _ in 0..frames_in_flight {
            // SAFETY: both objects come from the live device with the direct
            // list type; the fresh list is closed so `begin` can reset it.
            let (allocator, list) = unsafe {
                let allocator: ID3D12CommandAllocator = self
                    .native
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .map_err(|error| {
                        RhiError::dx12("ID3D12Device::CreateCommandAllocator(frame)", error)
                    })?;
                let list: ID3D12GraphicsCommandList4 = self
                    .native
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .map_err(|error| {
                        RhiError::dx12("ID3D12Device::CreateCommandList(frame)", error)
                    })?;
                list.Close()
                    .map_err(|error| RhiError::dx12("ID3D12CommandList::Close(frame)", error))?;
                (allocator, list)
            };
            contexts.push(Dx12FrameContext {
                allocator,
                list,
                fence_value: 0,
                submitted: false,
            });
        }
        Ok(Dx12FrameRing {
            contexts,
            fence: self.direct_fence.clone(),
        })
    }
}

impl Dx12FrameRing {
    pub(crate) fn wait_slot(&mut self, device: &Dx12Device, slot: usize) -> Result<(), RhiError> {
        let context = &mut self.contexts[slot];
        if context.submitted {
            device.wait_for_fence_value(context.fence_value)?;
            context.submitted = false;
        }
        Ok(())
    }

    /// Wait for the slot's previous frame and begin a fresh recording.
    pub(crate) fn begin(
        &mut self,
        device: &Dx12Device,
        slot: usize,
    ) -> Result<ID3D12GraphicsCommandList4, RhiError> {
        self.wait_slot(device, slot)?;
        let context = &self.contexts[slot];
        // SAFETY: the previous submission from this allocator has completed,
        // so resetting the allocator and reopening the list is valid.
        unsafe {
            context
                .allocator
                .Reset()
                .map_err(|error| RhiError::dx12("ID3D12CommandAllocator::Reset", error))?;
            context
                .list
                .Reset(&context.allocator, None)
                .map_err(|error| RhiError::dx12("ID3D12GraphicsCommandList::Reset", error))?;
        }
        Ok(context.list.clone())
    }

    /// Close the slot's recording, execute it, and signal the shared fence
    /// without waiting for the GPU.
    pub(crate) fn submit(&mut self, device: &Dx12Device, slot: usize) -> Result<(), RhiError> {
        let context = &mut self.contexts[slot];
        // SAFETY: the list is open with recorded work; the queue executes it
        // and the monotonically increasing signal marks its completion.
        let value = unsafe {
            context
                .list
                .Close()
                .map_err(|error| RhiError::dx12("ID3D12GraphicsCommandList::Close", error))?;
            let list: ID3D12CommandList = context
                .list
                .cast()
                .map_err(|error| RhiError::dx12("ID3D12CommandList cast", error))?;
            device.direct_queue.ExecuteCommandLists(&[Some(list)]);
            device.signal_fence()?
        };
        context.fence_value = value;
        context.submitted = true;
        Ok(())
    }

    /// Wait for every in-flight frame; used at shutdown and by readbacks.
    pub(crate) fn wait_idle(&mut self, device: &Dx12Device) -> Result<(), RhiError> {
        for slot in 0..self.contexts.len() {
            self.wait_slot(device, slot)?;
        }
        Ok(())
    }
}

impl Drop for Dx12FrameRing {
    fn drop(&mut self) {
        // Best effort: dropping the ring with frames in flight must not
        // release allocators the GPU is still reading. Bounded poll so a dead
        // device cannot hang destruction forever.
        for context in &self.contexts {
            if !context.submitted {
                continue;
            }
            for _ in 0..10_000 {
                // SAFETY: the fence clone is live independent of the device.
                if unsafe { self.fence.GetCompletedValue() } >= context.fence_value {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}
