use std::fmt;
use std::sync::Arc;

use crate::acceleration_structure::{AccelerationBuild, AccelerationStructureUsage};
use crate::command_encoder::CommandEncoder;
use crate::device::BackendDevice;
#[cfg(all(feature = "dx12", windows))]
use crate::dx12::Dx12FrameRing;
use crate::render_commands::Command;
use crate::surface::Presentation;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanFrameRing;
use crate::{
    AccelerationStructure, AccelerationStructureInstance, Buffer, Device, Image, ImageUsages,
    ImageView, PresentStatus, RhiError, Surface,
};

/// CPU wall times on the submitting thread, not GPU timestamp durations.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameSubmissionTimings {
    /// Replaying the frame's recorded commands into a native command list.
    pub encode_ms: f64,
    /// Waiting for a swapchain image to draw the frame into.
    pub acquire_ms: f64,
    /// Handing the command list to the device's queue.
    pub submit_ms: f64,
    /// The present call itself.
    pub present_ms: f64,
}

/// Resources whose native objects must outlive an in-flight frame. Clones
/// share the native allocation, so holding one here defers destruction until
/// the frame's fence has completed.
enum RetiredResource {
    AccelerationBuild(#[allow(dead_code)] AccelerationBuild),
    BlasBatch(#[allow(dead_code)] crate::acceleration_structure::BlasBatchBuild),
    SharedDescriptorSet(#[allow(dead_code)] std::sync::Arc<crate::DescriptorSet>),
    Buffer(#[allow(dead_code)] Buffer),
    Image(#[allow(dead_code)] Image),
    ImageView(#[allow(dead_code)] ImageView),
    AccelerationStructure(#[allow(dead_code)] AccelerationStructure),
    DescriptorSet(#[allow(dead_code)] crate::DescriptorSet),
}

/// A ring of reusable frame contexts (command pool/allocator plus fence per
/// slot). Beginning a frame waits only for that slot's previous submission;
/// finishing submits without waiting, so the CPU is no longer serialized
/// behind every GPU frame. Retired resources are dropped when their frame's
/// fence completes.
pub struct FrameRing {
    timings: FrameSubmissionTimings,
    device_id: u64,
    identity: Arc<()>,
    frames_in_flight: usize,
    next_slot: usize,
    generations: Vec<u64>,
    // The native ring waits in Drop before retained resources are released.
    backend: Option<FrameRingBackend>,
    packets: Vec<Arc<Vec<Command>>>,
    retired: Vec<Vec<RetiredResource>>,
    retained_memory: Vec<Vec<crate::memory::MemoryRetention>>,
    uploads: Vec<Vec<(Buffer, u64)>>,
    // The device owner drops after every captured native resource.
    device: BackendDevice,
}

pub(crate) enum FrameRingBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<VulkanFrameRing>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Box<Dx12FrameRing>),
}

impl fmt::Debug for FrameRing {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrameRing")
            .field("frames_in_flight", &self.frames_in_flight)
            .field("next_slot", &self.next_slot)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Create a ring of reusable frame contexts, as many as the device was
    /// opened with ([`crate::DeviceOptions::frames_in_flight`]).
    pub fn create_frame_ring(&self) -> Result<FrameRing, RhiError> {
        let frames_in_flight = self.frames_in_flight();
        let backend = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => {
                FrameRingBackend::Vulkan(Box::new(device.create_frame_ring(frames_in_flight)?))
            }
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => {
                FrameRingBackend::Dx12(Box::new(device.create_frame_ring(frames_in_flight)?))
            }
        };
        Ok(FrameRing {
            timings: FrameSubmissionTimings::default(),
            device_id: self.id(),
            identity: Arc::new(()),
            frames_in_flight: frames_in_flight as usize,
            next_slot: 0,
            generations: vec![0; frames_in_flight as usize],
            retired: (0..frames_in_flight).map(|_| Vec::new()).collect(),
            retained_memory: (0..frames_in_flight).map(|_| Vec::new()).collect(),
            uploads: (0..frames_in_flight).map(|_| Vec::new()).collect(),
            backend: Some(backend),
            device: self.backend.clone(),
            packets: (0..frames_in_flight)
                .map(|_| Arc::new(Vec::new()))
                .collect(),
        })
    }
}

impl FrameRing {
    /// How long each part of the last submission took on the CPU, in
    /// milliseconds.
    pub fn submission_timings(&self) -> FrameSubmissionTimings {
        self.timings
    }

    /// Wait for this slot's previous frame, drop its retired resources, and
    /// begin recording a new frame.
    pub fn begin_frame<'r>(&'r mut self, device: &'r Device) -> Result<Frame<'r>, RhiError> {
        if device.id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        let slot = self.next_slot;
        self.next_slot = (slot + 1) % self.frames_in_flight;
        // Host fence waits do not enter the command queue: acquiring another
        // viewport must not drain unrelated recording or uploads first.
        match (self.backend.as_mut(), &device.backend) {
            #[cfg(feature = "vulkan")]
            (Some(FrameRingBackend::Vulkan(ring)), BackendDevice::Vulkan(native)) => {
                ring.wait_slot(native, slot)?
            }
            #[cfg(all(feature = "dx12", windows))]
            (Some(FrameRingBackend::Dx12(ring)), BackendDevice::Dx12(native)) => {
                ring.wait_slot(native, slot)?
            }
            _ => {
                return Err(RhiError::InvalidFrame {
                    reason: "the frame ring is unavailable",
                });
            }
        }
        // The slot's fence completed, so its previous frame's
        // retirements are now safe to drop.
        self.generations[slot] =
            self.generations[slot]
                .checked_add(1)
                .ok_or(RhiError::InvalidFrame {
                    reason: "the frame slot has been used too many times to count",
                })?;
        self.packets[slot] = Arc::new(Vec::new());
        self.retired[slot].clear();
        self.retained_memory[slot].clear();
        for (_, used) in &mut self.uploads[slot] {
            *used = 0;
        }
        Ok(Frame {
            encoder: CommandEncoder::new(device),
            ring: self,
            device,
            slot,
        })
    }

    /// Wait for every in-flight frame and drop all retired resources.
    pub fn wait_idle(&mut self, device: &Device) -> Result<(), RhiError> {
        if device.id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        self.wait()?;
        for packet in &mut self.packets {
            *packet = Arc::new(Vec::new());
        }
        for slot in &mut self.retained_memory {
            slot.clear();
        }
        for slot in &mut self.retired {
            slot.clear();
        }
        Ok(())
    }

    fn wait(&mut self) -> Result<(), RhiError> {
        let backend = self.backend.as_mut().ok_or(RhiError::InvalidFrame {
            reason: "the frame ring is unavailable",
        })?;
        crate::submission::wait_ring(&self.device, backend)
    }

    /// Record and submit an owned frame on the calling thread. Scheduling is external.
    pub fn submit(
        &mut self,
        device: &Device,
        frame: RecordedFrame,
        presentation: Option<Presentation<'_>>,
    ) -> Result<(), RhiError> {
        if device.id() != self.device_id
            || frame.device_id != self.device_id
            || !Arc::ptr_eq(&frame.identity, &self.identity)
            || presentation
                .as_ref()
                .is_some_and(|p| p.image.device_id() != self.device_id)
        {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        if frame.generation != self.generations[frame.slot] {
            return Err(RhiError::InvalidFrame {
                reason: "the frame's slot has already been taken again",
            });
        }
        self.packets[frame.slot] = frame.commands.clone();
        let backend = self.backend.as_mut().ok_or(RhiError::InvalidFrame {
            reason: "the frame ring is unavailable",
        })?;
        self.timings = crate::submission::frame(
            &self.device,
            backend,
            frame.slot,
            &frame.commands,
            presentation,
        )?;
        Ok(())
    }
}

/// One frame being recorded. Record through [`Frame::encoder`], hand any
/// frame-local resources to the `retire_*` methods, then [`Frame::finish`]
/// to submit without waiting for the GPU.
pub struct Frame<'r> {
    encoder: CommandEncoder<'r>,
    ring: &'r mut FrameRing,
    device: &'r Device,
    slot: usize,
}

impl fmt::Debug for Frame<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Frame")
            .field("slot", &self.slot)
            .finish_non_exhaustive()
    }
}

impl<'r> Frame<'r> {
    /// Suballocate staging memory. A slot is reused only after its submission fence completes.
    pub fn stage_rows(
        &mut self,
        bytes: &[u8],
        row_bytes: usize,
        row_pitch: usize,
    ) -> Result<(Buffer, u64), RhiError> {
        let size = crate::buffer::row_write_size(bytes, row_bytes, row_pitch)?;
        let chunks = &mut self.ring.uploads[self.slot];
        let alignment = crate::IMAGE_COPY_OFFSET_ALIGNMENT;
        let index = chunks.iter().position(|(buffer, used)| {
            used.next_multiple_of(alignment) + size <= buffer.descriptor().size
        });
        let index = match index {
            Some(index) => index,
            None => {
                chunks.push((
                    self.device.create_buffer(crate::BufferDescriptor {
                        size: size.max(4 * 1024 * 1024),
                        usages: crate::BufferUsages::COPY_SOURCE,
                        memory: crate::BufferMemory::Upload,
                    })?,
                    0,
                ));
                chunks.len() - 1
            }
        };
        let (buffer, used) = &mut chunks[index];
        let offset = used.next_multiple_of(alignment);
        buffer.write_rows(offset, bytes, row_bytes, row_pitch)?;
        *used = offset + size;
        Ok((buffer.clone(), offset))
    }

    /// The command encoder recording this frame.
    pub fn encoder(&mut self) -> &mut CommandEncoder<'r> {
        &mut self.encoder
    }

    /// Record a static TLAS build. It can be traced by subsequent commands in
    /// this frame or by later submissions on this device's graphics queue. The
    /// frame owns its inputs and all BLAS dependencies through the submit fence.
    /// Do not use the returned structure on the GPU before submitting this frame.
    pub fn build_tlas(
        &mut self,
        instances: &[AccelerationStructureInstance<'_>],
    ) -> Result<AccelerationStructure, RhiError> {
        self.build_tlas_for(AccelerationStructureUsage::Static, instances)
    }

    /// Record a TLAS build that later queued refits may update in place.
    pub fn build_updatable_tlas(
        &mut self,
        instances: &[AccelerationStructureInstance<'_>],
    ) -> Result<AccelerationStructure, RhiError> {
        self.build_tlas_for(AccelerationStructureUsage::Updatable, instances)
    }

    fn build_tlas_for(
        &mut self,
        usage: AccelerationStructureUsage,
        instances: &[AccelerationStructureInstance<'_>],
    ) -> Result<AccelerationStructure, RhiError> {
        let build = self.device.prepare_tlas(usage, instances)?;
        self.device.record_tlas(&mut self.encoder, &build)?;
        let acceleration = build.acceleration.clone();
        self.ring.retired[self.slot].push(RetiredResource::AccelerationBuild(build));
        Ok(acceleration)
    }

    /// Record an in-place refit, ordered after prior traces and before following
    /// traces on the graphics queue. Every call owns a private CPU input snapshot;
    /// no upload memory still read by an earlier submission is overwritten.
    /// Instance count and referenced BLAS order must match the original build.
    pub fn update_tlas(
        &mut self,
        tlas: &AccelerationStructure,
        instances: &[AccelerationStructureInstance<'_>],
    ) -> Result<(), RhiError> {
        let build = self.device.prepare_tlas_update(tlas, instances)?;
        self.device.record_tlas(&mut self.encoder, &build)?;
        self.ring.retired[self.slot].push(RetiredResource::AccelerationBuild(build));
        Ok(())
    }

    /// Record one procedural BLAS build per AABB set into this frame, in
    /// input order, without waiting: TLAS builds and traces recorded later
    /// in the frame see the finished structures. The frame owns the build
    /// inputs through its submit fence. Do not use the structures on the
    /// GPU before submitting this frame.
    pub fn build_procedural_aabb_blas_batch(
        &mut self,
        sets: &[&[crate::Aabb]],
    ) -> Result<Vec<AccelerationStructure>, RhiError> {
        let build = self.device.prepare_blas_batch(sets)?;
        if build.accelerations.is_empty() {
            return Ok(Vec::new());
        }
        self.device.record_blas_batch(&mut self.encoder, &build)?;
        let accelerations = build.accelerations.clone();
        self.ring.retired[self.slot].push(RetiredResource::BlasBatch(build));
        Ok(accelerations)
    }

    /// Record one triangle BLAS build per geometry into this frame, in
    /// input order, without waiting: as
    /// [`Self::build_procedural_aabb_blas_batch`] does for boxes.
    pub fn build_triangle_blas_batch(
        &mut self,
        geometries: &[crate::TriangleGeometry<'_>],
    ) -> Result<Vec<AccelerationStructure>, RhiError> {
        let build = self.device.prepare_triangle_blas_batch(geometries)?;
        if build.accelerations.is_empty() {
            return Ok(Vec::new());
        }
        self.device.record_blas_batch(&mut self.encoder, &build)?;
        let accelerations = build.accelerations.clone();
        self.ring.retired[self.slot].push(RetiredResource::BlasBatch(build));
        Ok(accelerations)
    }

    /// Retain immutable shared descriptors until this frame's fence completes.
    pub fn retire_shared_descriptor_set(&mut self, set: std::sync::Arc<crate::DescriptorSet>) {
        self.ring.retired[self.slot].push(RetiredResource::SharedDescriptorSet(set));
    }

    /// Keep a buffer alive until this frame's fence completes.
    pub fn retire_buffer(&mut self, buffer: Buffer) {
        self.ring.retained_memory[self.slot].push(buffer.inner.memory.retain());
        self.ring.retired[self.slot].push(RetiredResource::Buffer(buffer));
    }

    /// Keep an image alive until this frame's fence completes.
    pub fn retire_image(&mut self, image: Image) {
        self.ring.retained_memory[self.slot].push(image.inner.memory.retain());
        self.ring.retired[self.slot].push(RetiredResource::Image(image));
    }

    /// Keep an image view alive until this frame's fence completes.
    pub fn retire_image_view(&mut self, view: ImageView) {
        self.ring.retained_memory[self.slot].push(view.image().inner.memory.retain());
        self.ring.retired[self.slot].push(RetiredResource::ImageView(view));
    }

    /// Keep an acceleration structure alive until this frame's fence
    /// completes.
    pub fn retire_acceleration_structure(&mut self, acceleration: AccelerationStructure) {
        self.ring.retired[self.slot].push(RetiredResource::AccelerationStructure(acceleration));
    }

    /// Keep a descriptor set (and on DX12 the shader-visible heap it owns)
    /// alive until this frame's fence completes. Dropping a replaced set
    /// while an in-flight command list still points into its heap is a GPU
    /// hang, not a validation error.
    pub fn retire_descriptor_set(&mut self, set: crate::DescriptorSet) {
        self.ring.retired[self.slot].push(RetiredResource::DescriptorSet(set));
    }

    /// The ring slot this frame occupies, for per-slot CPU-side resources
    /// (per-frame uniform buffers and the like).
    #[must_use]
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// Present `source` through `surface` as part of this frame's own
    /// submission. Acquires a backbuffer,
    /// records the copy/blit into this frame's command buffer, submits with
    /// the slot fence (waiting the acquire and signalling the present on
    /// Vulkan), and presents.
    ///
    /// The source image must carry `COPY_SOURCE` and match the surface extent. When the
    /// swapchain is out of date the frame is still submitted. Returns
    /// the presentation status; `OutOfDate` requests a reconfigure before continuing.
    pub fn present(
        mut self,
        surface: &mut Surface<'_>,
        source: &Image,
    ) -> Result<PresentStatus, RhiError> {
        if source.device_id() != self.device.id() || surface.device_id != self.device.id() {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        source.require_usage(ImageUsages::COPY_SOURCE, "copy source")?;
        self.encoder.use_image(
            source,
            crate::image::Subresources::whole(&source.descriptor()),
            crate::command_encoder::ImageNeed::COPY_SOURCE,
        )?;
        let descriptor = source.descriptor();
        let configuration = surface.configuration();
        if descriptor.width != configuration.width || descriptor.height != configuration.height {
            return Err(RhiError::InvalidRenderDimensions {
                width: descriptor.width,
                height: descriptor.height,
            });
        }
        let presentation = surface.presentation(source)?;
        let device = self.device;
        let ring = self.ring;
        self.encoder
            .require_no_render_pass("finish requires the render pass to be ended first")?;
        let recorded = RecordedFrame {
            identity: ring.identity.clone(),
            generation: ring.generations[self.slot],
            device_id: device.id(),
            slot: self.slot,
            commands: Arc::new(self.encoder.commands),
        };
        ring.submit(device, recorded, Some(presentation))?;
        surface.presentation(source)?.completed_status()
    }

    /// Finish recording without scheduling or submitting native work.
    pub fn record(self) -> Result<RecordedFrame, RhiError> {
        self.encoder
            .require_no_render_pass("finish requires the render pass to be ended first")?;
        Ok(RecordedFrame {
            identity: self.ring.identity.clone(),
            generation: self.ring.generations[self.slot],
            device_id: self.device.id(),
            slot: self.slot,
            commands: Arc::new(self.encoder.commands),
        })
    }

    /// Record and submit natively on this thread, leaving GPU work in flight.
    pub fn finish(self) -> Result<(), RhiError> {
        self.encoder
            .require_no_render_pass("finish requires the render pass to be ended first")?;
        let recorded = RecordedFrame {
            identity: self.ring.identity.clone(),
            generation: self.ring.generations[self.slot],
            device_id: self.device.id(),
            slot: self.slot,
            commands: Arc::new(self.encoder.commands),
        };
        self.ring.submit(self.device, recorded, None)
    }
}

impl Drop for FrameRing {
    fn drop(&mut self) {
        // Retirement must wait for native submissions to complete on the GPU.
        let _ = self.wait();
    }
}

/// A captured frame; owns all command resources until submitted to its frame ring.
pub struct RecordedFrame {
    device_id: u64,
    identity: Arc<()>,
    generation: u64,
    slot: usize,
    commands: Arc<Vec<Command>>,
}

impl fmt::Debug for RecordedFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecordedFrame")
            .field("slot", &self.slot)
            .field("commands", &self.commands.len())
            .finish_non_exhaustive()
    }
}
