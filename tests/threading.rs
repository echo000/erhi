//! Thread-safety guarantees, checked at compile time: if a type stops being
//! `Send` or `Sync` stops this file compiling.
//!
//! Every handle (the device, a buffer, an image, a pipeline, a descriptor
//! set, an acceleration structure) may be sent to and shared between
//! threads: creating resources, writing upload buffers and reading
//! readback buffers from any thread at once is defined. What records
//! commands (an encoder, a frame) belongs to one thread at a time and is
//! borrowed mutably to be used, so the compiler already holds callers to
//! that; it may still be moved to another thread between uses.

use erhi::{
    AccelerationStructure, Backend, Buffer, BufferDescriptor, BufferMemory, BufferUsages,
    CommandEncoder, ComputePipeline, DescriptorSet, Device, DeviceOptions, Frame, FrameRing, Image,
    ImageView, OcclusionQuerySet, PipelineLayout, RayTracingPipeline, RecordedFrame,
    RenderPipeline, RhiError, Sampler, ShaderModule, Surface, TimestampQuerySet,
};

fn shared<T: Send + Sync>() {}

fn sent<T: Send>() {}

#[test]
fn handles_cross_threads() {
    shared::<Device>();
    shared::<Buffer>();
    shared::<Image>();
    shared::<ImageView>();
    shared::<Sampler>();
    shared::<ShaderModule>();
    shared::<PipelineLayout>();
    shared::<ComputePipeline>();
    shared::<RenderPipeline>();
    shared::<RayTracingPipeline>();
    shared::<DescriptorSet>();
    shared::<AccelerationStructure>();
    shared::<TimestampQuerySet>();
    shared::<OcclusionQuerySet>();
    shared::<RhiError>();
    sent::<Surface<'static>>();
    sent::<FrameRing>();
    sent::<RecordedFrame>();
    sent::<CommandEncoder<'static>>();
    sent::<Frame<'static>>();
}

const THREADS: u32 = 8;

const ROUNDS: u32 = 40;

const SIZE: u64 = 256;

/// Every thread makes buffers, records a copy through the device and back,
/// submits it and reads what came back, all at once with the others.
fn stress(backend: Backend) -> Result<(), RhiError> {
    let device = Device::open(DeviceOptions {
        backend,
        debug: true,
        require_ray_tracing: false,
        ..DeviceOptions::default()
    })?;
    let buffer = |usages, memory| device.create_buffer(BufferDescriptor::new(SIZE, usages, memory));
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..THREADS)
            .map(|thread| {
                let (device, buffer) = (&device, &buffer);
                scope.spawn(move || -> Result<(), RhiError> {
                    for round in 0..ROUNDS {
                        let bytes = (thread * 1000 + round)
                            .to_le_bytes()
                            .repeat(SIZE as usize / 4);
                        let upload = buffer(BufferUsages::COPY_SOURCE, BufferMemory::Upload)?;
                        let held = buffer(
                            BufferUsages::COPY_SOURCE | BufferUsages::COPY_DESTINATION,
                            BufferMemory::Device,
                        )?;
                        let back = buffer(BufferUsages::COPY_DESTINATION, BufferMemory::Readback)?;
                        upload.write(0, &bytes)?;
                        let mut encoder = device.begin_commands()?;
                        encoder.copy_buffer(&upload, 0, &held, 0, SIZE)?;
                        encoder.copy_buffer(&held, 0, &back, 0, SIZE)?;
                        encoder.finish()?;
                        assert_eq!(back.read(0, SIZE)?, bytes, "thread {thread}, round {round}");
                    }
                    Ok(())
                })
            })
            .collect();
        workers
            .into_iter()
            .try_for_each(|worker| worker.join().expect("a worker panicked"))
    })?;
    device.wait_idle()?;
    eprintln!("{:#?}", device.debug_messages());
    assert_eq!(device.debug_error_count().unwrap_or(0), 0, "{backend:?}");
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan device (and a DX12 adapter when dx12 is enabled)"]
fn threads_create_record_and_submit_at_once_on_every_backend() {
    stress(Backend::Vulkan).expect("Vulkan threads");
    #[cfg(all(feature = "dx12", windows))]
    stress(Backend::Dx12).expect("DX12 threads");
}
