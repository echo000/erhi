//! Presentation acceptance: both backends open a real window, present an

//! animated image, toggle VSync, resize, and tear down without errors.

#![cfg(all(feature = "vulkan", windows))]

use std::time::Duration;

use erhi::{
    Backend, BufferDescriptor, BufferMemory, BufferUsages, Device, DeviceOptions, ImageDescriptor,
    ImageFormat, ImageUsages, PresentMode, PresentStatus, ResourceState, RhiError,
    SurfaceConfiguration,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::EventLoopExtPumpEvents;
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::{Window, WindowId};

#[derive(Default)]
struct App {
    window: Option<Window>,
    created: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !self.created {
            self.created = true;
            let attributes = Window::default_attributes()
                .with_title("erhi present probe")
                .with_inner_size(winit::dpi::PhysicalSize::new(512, 288));
            self.window = Some(
                event_loop
                    .create_window(attributes)
                    .expect("create the probe window"),
            );
        }
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

/// Fill an image with an animated gradient through the copy path and leave
/// it in `CopySource`, ready to present.
fn upload_frame(
    device: &Device,
    image: &erhi::Image,
    width: u32,
    height: u32,
    frame: u32,
) -> Result<(), RhiError> {
    let layout = image.copy_layout(0)?;
    let mut bytes = vec![0_u8; layout.total_size as usize];
    for y in 0..height as usize {
        let row = y * layout.row_pitch as usize;
        for x in 0..width as usize {
            let texel = row + x * 4;
            bytes[texel] = ((x as u32 * 255 / width) as u8).wrapping_add(frame as u8);
            bytes[texel + 1] = (y as u32 * 255 / height) as u8;
            bytes[texel + 2] = (frame % 255) as u8;
            bytes[texel + 3] = 255;
        }
    }
    let staging = device.create_buffer(BufferDescriptor::new(
        layout.total_size,
        BufferUsages::COPY_SOURCE,
        BufferMemory::Upload,
    ))?;
    staging.write(0, &bytes)?;
    let mut encoder = device.begin_commands()?;
    encoder.transition_image(image, ResourceState::CopyDest)?;
    encoder.copy_buffer_to_image(&staging, 0, image, 0)?;
    encoder.transition_image(image, ResourceState::CopySource)?;
    encoder.finish()
}

fn make_presentable(device: &Device, width: u32, height: u32) -> Result<erhi::Image, RhiError> {
    device.create_image(ImageDescriptor::d2(
        ImageFormat::Rgba8Unorm,
        width,
        height,
        ImageUsages::COPY_SOURCE | ImageUsages::COPY_DESTINATION,
    ))
}

fn run_backend(
    backend: Backend,
    event_loop: &mut EventLoop<()>,
    app: &mut App,
    window: &Window,
) -> Result<(), RhiError> {
    let device = Device::open(DeviceOptions {
        backend,
        ..DeviceOptions::default()
    })?;
    let size = window.inner_size();
    let mut configuration = SurfaceConfiguration {
        present_mode: PresentMode::Fifo,
        ..SurfaceConfiguration::new(size.width, size.height)
    };
    let mut surface = device.create_surface(window, configuration)?;
    // Every surface takes the SDR default, paced by VSync.
    let capabilities = surface.capabilities(&device)?;
    assert!(
        capabilities.formats.contains(&(
            erhi::SurfaceFormat::Rgba8Unorm,
            erhi::SurfaceColorSpace::Srgb
        )),
        "{capabilities:?}"
    );
    assert!(capabilities.present_modes.contains(&PresentMode::Fifo));
    assert_eq!(surface.backend(), backend);
    assert!(
        surface.image_count() >= 2,
        "swapchains must double-buffer at minimum"
    );

    let mut image = make_presentable(&device, configuration.width, configuration.height)?;
    let mut presented = 0_u32;
    for frame in 0..90_u32 {
        // Keep the window responsive while presenting.
        let _ = event_loop.pump_app_events(Some(Duration::ZERO), app);

        // Toggle VSync a third of the way through, resize two thirds in.
        if frame == 30 {
            configuration.present_mode = PresentMode::Immediate;
            surface.configure(&device, configuration)?;
        }
        if frame == 60 {
            let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(384, 384));
            let _ = event_loop.pump_app_events(Some(Duration::ZERO), app);
            let new_size = window.inner_size();
            configuration.width = new_size.width.max(1);
            configuration.height = new_size.height.max(1);
            surface.configure(&device, configuration)?;
            image = make_presentable(&device, configuration.width, configuration.height)?;
        }

        upload_frame(
            &device,
            &image,
            configuration.width,
            configuration.height,
            frame,
        )?;
        match surface.present_image(&device, &image)? {
            PresentStatus::Presented => presented += 1,
            PresentStatus::OutOfDate => {
                let size = window.inner_size();
                configuration.width = size.width.max(1);
                configuration.height = size.height.max(1);
                surface.configure(&device, configuration)?;
                image = make_presentable(&device, configuration.width, configuration.height)?;
            }
        }
    }
    assert!(
        presented >= 80,
        "{backend:?} presented only {presented} of 90 frames"
    );

    // A stale-extent image is rejected before touching the swapchain.
    let wrong = make_presentable(&device, 64, 64)?;
    upload_frame(&device, &wrong, 64, 64, 0)?;
    assert!(matches!(
        surface.present_image(&device, &wrong),
        Err(RhiError::InvalidRenderDimensions { .. })
    ));
    Ok(())
}

/// Sequential on purpose: parallel multi-device work wedges the driver, and
/// both backends share one probe window.
#[test]
#[ignore = "requires a display plus an RT-capable Vulkan device (and a DXR adapter when dx12 is enabled)"]
fn presents_to_a_window_on_every_backend() -> Result<(), RhiError> {
    let mut event_loop = EventLoop::builder()
        .with_any_thread(true)
        .build()
        .expect("build the windowed event loop");
    let mut app = App::default();
    while app.window.is_none() {
        let _ = event_loop.pump_app_events(Some(Duration::from_millis(10)), &mut app);
    }
    // Own the window outside the handler so surfaces can borrow it while the
    // event loop keeps pumping.
    let window = app.window.take().expect("window created");

    run_backend(Backend::Vulkan, &mut event_loop, &mut app, &window)?;
    #[cfg(feature = "dx12")]
    run_backend(Backend::Dx12, &mut event_loop, &mut app, &window)?;
    Ok(())
}
