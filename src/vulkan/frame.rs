use ash::vk;

use super::{CommandPool, VulkanDevice};
use crate::RhiError;

/// One bounded fence-wait slice; a healthy frame completes far sooner, and a
/// dead device surfaces as an error instead of an infinite stall.
const FENCE_WAIT_NANOSECONDS: u64 = 10_000_000_000;

struct FenceOwner {
    device: ash::Device,
    handle: vk::Fence,
}

impl Drop for FenceOwner {
    fn drop(&mut self) {
        // SAFETY: the ring waits for the fence before its owner drops.
        unsafe { self.device.destroy_fence(self.handle, None) };
    }
}

struct VulkanFrameContext {
    pool: CommandPool,
    command: vk::CommandBuffer,
    fence: FenceOwner,
    submitted: bool,
}

/// Reusable per-frame command pools and fences: recording a new frame in a
/// slot waits only for that slot's previous submission, never the queue.
pub(crate) struct VulkanFrameRing {
    contexts: Vec<VulkanFrameContext>,
}

impl VulkanDevice {
    pub(crate) fn create_frame_ring(
        &self,
        frames_in_flight: u32,
    ) -> Result<VulkanFrameRing, RhiError> {
        let mut contexts = Vec::with_capacity(frames_in_flight as usize);
        for _ in 0..frames_in_flight {
            let pool = CommandPool::new(&self.logical.handle, self.queue_family_index)?;
            let command = pool.allocate_primary()?;
            let create = vk::FenceCreateInfo::default();
            // SAFETY: the create info is complete; the fence starts unsignaled
            // because nothing has been submitted from this slot yet.
            let fence = unsafe { self.logical.handle.create_fence(&create, None) }
                .map_err(|result| RhiError::vk("vkCreateFence(frame)", result))?;
            contexts.push(VulkanFrameContext {
                pool,
                command,
                fence: FenceOwner {
                    device: self.logical.handle.clone(),
                    handle: fence,
                },
                submitted: false,
            });
        }
        Ok(VulkanFrameRing { contexts })
    }
}

impl VulkanFrameRing {
    pub(crate) fn wait_slot(&mut self, device: &VulkanDevice, slot: usize) -> Result<(), RhiError> {
        let context = &mut self.contexts[slot];
        if !context.submitted {
            return Ok(());
        }
        loop {
            // SAFETY: the fence belongs to this live device and was submitted.
            let result = unsafe {
                device.logical.handle.wait_for_fences(
                    &[context.fence.handle],
                    true,
                    FENCE_WAIT_NANOSECONDS,
                )
            };
            match result {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) => continue,
                Err(result) => return Err(RhiError::vk("vkWaitForFences(frame)", result)),
            }
        }
        // SAFETY: the fence completed and can be reset for the next frame.
        unsafe { device.logical.handle.reset_fences(&[context.fence.handle]) }
            .map_err(|result| RhiError::vk("vkResetFences(frame)", result))?;
        context.submitted = false;
        Ok(())
    }

    /// Wait for the slot's previous frame and begin a fresh recording.
    pub(crate) fn begin(
        &mut self,
        device: &VulkanDevice,
        slot: usize,
    ) -> Result<vk::CommandBuffer, RhiError> {
        self.wait_slot(device, slot)?;
        let context = &self.contexts[slot];
        // SAFETY: the previous submission from this pool has completed, so
        // resetting the pool and re-recording its buffer is valid.
        unsafe {
            device
                .logical
                .handle
                .reset_command_pool(context.pool.handle, vk::CommandPoolResetFlags::empty())
                .map_err(|result| RhiError::vk("vkResetCommandPool(frame)", result))?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .logical
                .handle
                .begin_command_buffer(context.command, &begin)
                .map_err(|result| RhiError::vk("vkBeginCommandBuffer(frame)", result))?;
        }
        Ok(context.command)
    }

    /// End the slot's recording and submit it with the slot fence, without
    /// waiting for the GPU.
    pub(crate) fn submit(&mut self, device: &VulkanDevice, slot: usize) -> Result<(), RhiError> {
        self.submit_with_semaphores(device, slot, None)
    }

    /// Like [`Self::submit`], optionally waiting a swapchain-acquire
    /// semaphore at the transfer stage and signalling a render-complete
    /// semaphore for `vkQueuePresentKHR`.
    pub(crate) fn submit_with_semaphores(
        &mut self,
        device: &VulkanDevice,
        slot: usize,
        present: Option<(vk::Semaphore, vk::Semaphore)>,
    ) -> Result<(), RhiError> {
        let context = &mut self.contexts[slot];
        // SAFETY: the slot's command buffer is in the recording state and its
        // fence was reset by `begin`; any semaphores are live surface objects.
        unsafe {
            device
                .logical
                .handle
                .end_command_buffer(context.command)
                .map_err(|result| RhiError::vk("vkEndCommandBuffer(frame)", result))?;
            let commands = [context.command];
            let mut submit = vk::SubmitInfo::default().command_buffers(&commands);
            let waits;
            let stages;
            let signals;
            if let Some((acquire, render)) = present {
                waits = [acquire];
                // The frame's present blit is the first swapchain access.
                stages = [vk::PipelineStageFlags::TRANSFER];
                signals = [render];
                submit = submit
                    .wait_semaphores(&waits)
                    .wait_dst_stage_mask(&stages)
                    .signal_semaphores(&signals);
            }
            let _queue = crate::sync::lock_anyway(&device.queue_lock);
            device
                .logical
                .handle
                .queue_submit(device.queue, &[submit], context.fence.handle)
                .map_err(|result| RhiError::vk("vkQueueSubmit(frame)", result))?;
        }
        context.submitted = true;
        Ok(())
    }

    /// Wait for every in-flight frame; used at shutdown and by readbacks.
    pub(crate) fn wait_idle(&mut self, device: &VulkanDevice) -> Result<(), RhiError> {
        for slot in 0..self.contexts.len() {
            self.wait_slot(device, slot)?;
        }
        Ok(())
    }
}

impl Drop for VulkanFrameRing {
    fn drop(&mut self) {
        // Best effort: dropping the ring with frames in flight must not
        // destroy pools the GPU is still reading.
        for context in &self.contexts {
            if context.submitted {
                // SAFETY: the fence is live; an error here means the device
                // is already lost and destruction proceeds regardless.
                let _ = unsafe {
                    context.fence.device.wait_for_fences(
                        &[context.fence.handle],
                        true,
                        FENCE_WAIT_NANOSECONDS,
                    )
                };
            }
        }
    }
}
