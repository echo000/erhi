//! Submits recorded commands to the device. The caller decides which thread
//! submits and when.

use std::sync::Mutex;
use std::time::Instant;

use crate::command_encoder::CommandEncoderBackend;
use crate::device::BackendDevice;
use crate::frame::FrameRingBackend;
use crate::render_commands::Command;
use crate::surface::{Presentation, SurfaceBackend};
use crate::{FrameSubmissionTimings, Image, PresentStatus, RhiError, SurfaceConfiguration};

fn disconnected() -> RhiError {
    RhiError::LockPoisoned {
        what: "the surface",
    }
}

pub(crate) fn configure(
    device: &BackendDevice,
    surface: &Mutex<SurfaceBackend>,
    configuration: SurfaceConfiguration,
    frames: u32,
) -> Result<(), RhiError> {
    wait_idle(device)?;
    let mut surface = surface.lock().map_err(|_| disconnected())?;
    match (&mut *surface, device) {
        #[cfg(feature = "vulkan")]
        (SurfaceBackend::Vulkan(surface), BackendDevice::Vulkan(native)) => {
            surface.configure(native, configuration, frames)
        }
        #[cfg(all(feature = "dx12", windows))]
        (SurfaceBackend::Dx12(surface), BackendDevice::Dx12(native)) => {
            let _ = frames;
            surface.configure(native, configuration)
        }
        #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
        _ => Err(RhiError::ResourceBackendMismatch),
    }
}

pub(crate) fn present_image(
    device: &BackendDevice,
    presentation: &Presentation<'_>,
) -> Result<PresentStatus, RhiError> {
    use crate::image::ImageBackend;

    let mut surface = presentation.surface.lock().map_err(|_| disconnected())?;
    match (&mut *surface, device, &presentation.image.inner.backend) {
        #[cfg(feature = "vulkan")]
        (
            SurfaceBackend::Vulkan(surface),
            BackendDevice::Vulkan(native),
            ImageBackend::Vulkan(image),
        ) => surface.present_image(native, image, presentation.image.descriptor()),
        #[cfg(all(feature = "dx12", windows))]
        (SurfaceBackend::Dx12(surface), BackendDevice::Dx12(native), ImageBackend::Dx12(image)) => {
            surface.present_image(native, image, presentation.image.descriptor())
        }
        #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
        _ => Err(RhiError::ResourceBackendMismatch),
    }
}

pub(crate) fn wait_idle(device: &BackendDevice) -> Result<(), RhiError> {
    match device {
        #[cfg(feature = "vulkan")]
        BackendDevice::Vulkan(native) => native.wait_idle(),
        #[cfg(all(feature = "dx12", windows))]
        BackendDevice::Dx12(native) => native.wait_idle(),
    }
}

pub(crate) fn wait_ring(
    device: &BackendDevice,
    ring: &mut FrameRingBackend,
) -> Result<(), RhiError> {
    match (ring, device) {
        #[cfg(feature = "vulkan")]
        (FrameRingBackend::Vulkan(ring), BackendDevice::Vulkan(native)) => ring.wait_idle(native),
        #[cfg(all(feature = "dx12", windows))]
        (FrameRingBackend::Dx12(ring), BackendDevice::Dx12(native)) => ring.wait_idle(native),
        #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
        _ => Err(RhiError::ResourceBackendMismatch),
    }
}

pub(crate) fn execute(device: &BackendDevice, commands: &[Command]) -> Result<(), RhiError> {
    let mut encoder = match device {
        #[cfg(feature = "vulkan")]
        BackendDevice::Vulkan(native) => {
            CommandEncoderBackend::Vulkan(Box::new(native.begin_commands()?))
        }
        #[cfg(all(feature = "dx12", windows))]
        BackendDevice::Dx12(native) => {
            CommandEncoderBackend::Dx12(Box::new(native.begin_commands()?))
        }
    };
    for command in commands {
        command.record(device, &mut encoder)?;
    }
    match encoder {
        #[cfg(feature = "vulkan")]
        CommandEncoderBackend::Vulkan(encoder) => encoder.finish(),
        #[cfg(all(feature = "dx12", windows))]
        CommandEncoderBackend::Dx12(encoder) => encoder.finish(),
    }
}

pub(crate) fn frame(
    device: &BackendDevice,
    ring: &mut FrameRingBackend,
    slot: usize,
    commands: &[Command],
    presentation: Option<Presentation<'_>>,
) -> Result<FrameSubmissionTimings, RhiError> {
    let started = Instant::now();
    let mut encoder = match (&mut *ring, device) {
        #[cfg(feature = "vulkan")]
        (FrameRingBackend::Vulkan(ring), BackendDevice::Vulkan(native)) => {
            let command = ring.begin(native, slot)?;
            CommandEncoderBackend::Vulkan(Box::new(native.encoder_from_frame(command)))
        }
        #[cfg(all(feature = "dx12", windows))]
        (FrameRingBackend::Dx12(ring), BackendDevice::Dx12(native)) => {
            let command = ring.begin(native, slot)?;
            CommandEncoderBackend::Dx12(Box::new(native.encoder_from_frame(command)))
        }
        #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
        _ => return Err(RhiError::ResourceBackendMismatch),
    };
    for command in commands {
        command.record(device, &mut encoder)?;
    }
    let mut timings = FrameSubmissionTimings {
        encode_ms: started.elapsed().as_secs_f64() * 1000.,
        ..Default::default()
    };
    if let Some(presentation) = presentation {
        presentation
            .image
            .require_state(crate::ResourceState::CopySource)?;
        let mut surface = presentation.surface.lock().map_err(|_| disconnected())?;
        let status = present(
            device,
            ring,
            slot,
            encoder,
            &mut surface,
            &presentation.image,
            &mut timings,
        )?;
        *presentation.status.lock().map_err(|_| disconnected())? = status;
        return Ok(timings);
    }
    drop(encoder);
    let started = Instant::now();
    submit(device, ring, slot)?;
    timings.submit_ms = started.elapsed().as_secs_f64() * 1000.;
    Ok(timings)
}

pub(crate) fn submit(
    device: &BackendDevice,
    ring: &mut FrameRingBackend,
    slot: usize,
) -> Result<(), RhiError> {
    match (ring, device) {
        #[cfg(feature = "vulkan")]
        (FrameRingBackend::Vulkan(ring), BackendDevice::Vulkan(native)) => {
            ring.submit(native, slot)
        }
        #[cfg(all(feature = "dx12", windows))]
        (FrameRingBackend::Dx12(ring), BackendDevice::Dx12(native)) => ring.submit(native, slot),
        #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
        _ => Err(RhiError::ResourceBackendMismatch),
    }
}

pub(crate) fn present(
    device: &BackendDevice,
    ring: &mut FrameRingBackend,
    slot: usize,
    encoder: CommandEncoderBackend<'_>,
    surface: &mut SurfaceBackend,
    source: &Image,
    timings: &mut FrameSubmissionTimings,
) -> Result<PresentStatus, RhiError> {
    use crate::image::ImageBackend;

    match (surface, ring, device, &source.inner.backend) {
        #[cfg(feature = "vulkan")]
        (
            SurfaceBackend::Vulkan(surface),
            FrameRingBackend::Vulkan(ring),
            BackendDevice::Vulkan(native),
            ImageBackend::Vulkan(image),
        ) => {
            let started = Instant::now();
            let Some(index) = surface.acquire_frame(slot)? else {
                drop(encoder);
                ring.submit(native, slot)?;
                return Ok(PresentStatus::OutOfDate);
            };
            timings.acquire_ms = started.elapsed().as_secs_f64() * 1000.;
            let command = match &encoder {
                CommandEncoderBackend::Vulkan(encoder) => encoder.raw_command(),
                #[cfg(all(feature = "dx12", windows))]
                _ => return Err(RhiError::ResourceBackendMismatch),
            };
            surface.record_frame_blit(command, image, source.descriptor(), index);
            drop(encoder);
            let semaphores = surface.frame_semaphores(slot, index);
            let started = Instant::now();
            ring.submit_with_semaphores(native, slot, Some(semaphores))?;
            timings.submit_ms = started.elapsed().as_secs_f64() * 1000.;
            let started = Instant::now();
            let status = surface.present_frame(native, slot, index);
            timings.present_ms = started.elapsed().as_secs_f64() * 1000.;
            status
        }
        #[cfg(all(feature = "dx12", windows))]
        (
            SurfaceBackend::Dx12(surface),
            FrameRingBackend::Dx12(ring),
            BackendDevice::Dx12(native),
            ImageBackend::Dx12(image),
        ) => {
            let started = Instant::now();
            let (_, backbuffer) = surface.current_backbuffer()?;
            timings.acquire_ms = started.elapsed().as_secs_f64() * 1000.;
            let list = match &encoder {
                CommandEncoderBackend::Dx12(encoder) => encoder.raw_list().clone(),
                #[cfg(feature = "vulkan")]
                _ => return Err(RhiError::ResourceBackendMismatch),
            };
            surface.record_frame_copy(&list, &backbuffer, image);
            drop(encoder);
            let started = Instant::now();
            ring.submit(native, slot)?;
            timings.submit_ms = started.elapsed().as_secs_f64() * 1000.;
            let started = Instant::now();
            let status = surface.present_frame();
            timings.present_ms = started.elapsed().as_secs_f64() * 1000.;
            status
        }
        #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
        _ => Err(RhiError::ResourceBackendMismatch),
    }
}
