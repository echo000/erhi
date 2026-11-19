//! A lost device reports `DeviceLost`, and a new device can then be opened.
//!
//! Alone in its test binary: every Direct3D 12 device this process holds on
//! one adapter is the same device, so losing it here would lose it under
//! any test running beside this one.

use erhi::{
    Backend, BufferDescriptor, BufferMemory, BufferUsages, Device, DeviceOptions, RhiError,
};

fn copy_round_trip(device: &Device) -> Result<Vec<u8>, RhiError> {
    let payload = b"still here";
    let upload = device.create_buffer(BufferDescriptor::new(
        payload.len() as u64,
        BufferUsages::COPY_SOURCE,
        BufferMemory::Upload,
    ))?;
    upload.write(0, payload)?;
    let readback = device.create_buffer(BufferDescriptor::new(
        payload.len() as u64,
        BufferUsages::COPY_DESTINATION,
        BufferMemory::Readback,
    ))?;
    let mut encoder = device.begin_commands()?;
    encoder.copy_buffer(&upload, 0, &readback, 0, payload.len() as u64)?;
    encoder.finish()?;
    readback.read(0, payload.len() as u64)
}

#[test]
#[ignore = "requires a Vulkan device (and a DX12 adapter when dx12 is enabled)"]
fn a_lost_device_reports_it_and_is_replaced() -> Result<(), RhiError> {
    // Vulkan has no way to lose a device on purpose: it stays whole.
    let vulkan = Device::open(DeviceOptions {
        backend: Backend::Vulkan,
        ..DeviceOptions::default()
    })?;
    assert!(!vulkan.simulate_device_loss());
    vulkan.check_device()?;
    assert_eq!(copy_round_trip(&vulkan)?, b"still here");
    drop(vulkan);

    #[cfg(all(feature = "dx12", windows))]
    {
        let options = DeviceOptions {
            backend: Backend::Dx12,
            ..DeviceOptions::default()
        };
        let device = Device::open(options)?;
        let frames = device.create_frame_ring()?;
        device.check_device()?;
        assert_eq!(copy_round_trip(&device)?, b"still here");

        assert!(device.simulate_device_loss());
        let lost = device.check_device().expect_err("the device was removed");
        assert!(lost.is_device_lost(), "{lost}");
        // Work asked of it fails the same way, without hanging or panicking.
        let failed = copy_round_trip(&device).expect_err("a removed device does no work");
        assert!(failed.is_device_lost(), "{failed}");
        // (Waiting for it returns: with nothing left to wait for, or lost.)
        if let Err(error) = device.wait_idle() {
            assert!(error.is_device_lost(), "{error}");
        }

        // Everything made from it is dropped, and then there is a new one.
        drop(frames);
        drop(device);
        let device = Device::open(options)?;
        device.check_device()?;
        assert_eq!(copy_round_trip(&device)?, b"still here");
    }
    Ok(())
}
