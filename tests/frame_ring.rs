//! Frame-lifetime acceptance: thousands of frames through the reusable frame
//! ring without memory growth, with deferred destruction proven safe by

//! byte-exact results from buffers retired while their frames were in flight.

#![cfg(feature = "vulkan")]

use erhi::{
    Backend, BufferDescriptor, BufferMemory, BufferUsages, Device, DeviceOptions, RhiError,
};

const FRAMES: u32 = 1000;

const PAYLOAD: u64 = 32;

fn pattern(frame: u32) -> Vec<u8> {
    (0..PAYLOAD as u32)
        .map(|byte| (frame.wrapping_mul(31).wrapping_add(byte) % 251) as u8)
        .collect()
}

fn run_ring(backend: Backend) -> Result<(), RhiError> {
    // Frames in flight are fixed when the device is opened.
    let device = Device::open(DeviceOptions {
        backend,
        frames_in_flight: 2,
        ..DeviceOptions::default()
    })?;
    assert_eq!(device.frames_in_flight(), 2);
    let mut ring = device.create_frame_ring()?;

    // One persistent readback per ring slot; frame N copies into slot N % 2.
    let readbacks = [
        device.create_buffer(BufferDescriptor::new(
            PAYLOAD,
            BufferUsages::COPY_DESTINATION,
            BufferMemory::Readback,
        ))?,
        device.create_buffer(BufferDescriptor::new(
            PAYLOAD,
            BufferUsages::COPY_DESTINATION,
            BufferMemory::Readback,
        ))?,
    ];

    let mut baseline_bytes = 0_u64;
    for frame_index in 0..FRAMES {
        // A frame-local upload buffer, retired while its copy is in flight:
        // if deferred destruction freed it early, the copy would read freed
        // memory and the final patterns below would be wrong.
        let upload = device.create_buffer(BufferDescriptor::new(
            PAYLOAD,
            BufferUsages::COPY_SOURCE,
            BufferMemory::Upload,
        ))?;
        upload.write(0, &pattern(frame_index))?;

        let mut frame = ring.begin_frame(&device)?;
        frame.encoder().copy_buffer(
            &upload,
            0,
            &readbacks[(frame_index % 2) as usize],
            0,
            PAYLOAD,
        )?;
        frame.retire_buffer(upload);
        frame.finish()?;

        if frame_index == 100 {
            baseline_bytes = device.allocated_bytes();
        }
    }
    ring.wait_idle(&device)?;

    // Steady-state memory must not have grown after warm-up.
    let final_bytes = device.allocated_bytes();
    assert!(
        final_bytes <= baseline_bytes,
        "allocator grew from {baseline_bytes} to {final_bytes} bytes over {FRAMES} frames"
    );

    // The last frame into each slot must have landed byte-exactly.
    assert_eq!(
        readbacks[((FRAMES - 1) % 2) as usize].read(0, PAYLOAD)?,
        pattern(FRAMES - 1),
        "the final frame's copy must be intact"
    );
    assert_eq!(
        readbacks[((FRAMES - 2) % 2) as usize].read(0, PAYLOAD)?,
        pattern(FRAMES - 2),
        "the previous frame's copy must be intact"
    );
    Ok(())
}

/// Sequential on purpose: parallel multi-device work wedges the driver.
#[test]
#[ignore = "requires an RT-capable Vulkan device (and a DXR adapter when dx12 is enabled)"]
fn frame_ring_runs_a_thousand_frames_without_growth() -> Result<(), RhiError> {
    run_ring(Backend::Vulkan)?;
    #[cfg(all(feature = "dx12", windows))]
    run_ring(Backend::Dx12)?;

    // A ring rejects a foreign device before any native call.
    let rhi_a = Device::open(DeviceOptions {
        backend: Backend::Vulkan,
        ..DeviceOptions::default()
    })?;
    let rhi_b = Device::open(DeviceOptions {
        backend: Backend::Vulkan,
        ..DeviceOptions::default()
    })?;
    let mut ring = rhi_a.create_frame_ring()?;
    assert!(matches!(
        ring.begin_frame(&rhi_b),
        Err(RhiError::ResourceDeviceMismatch)
    ));

    // An out-of-range startup configuration is rejected at device creation.
    assert!(matches!(
        Device::open(DeviceOptions {
            backend: Backend::Vulkan,
            frames_in_flight: 0,
            ..DeviceOptions::default()
        }),
        Err(RhiError::InvalidFramesInFlight(0))
    ));
    Ok(())
}
