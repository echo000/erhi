//! A window showing a moving gradient: a device, a surface, pixels copied
//! into an image, the image presented. No shader and no pipeline; for
//! those, see the `triangle` example.
//!
//! ```text
//! cargo run --example window
//! cargo run --example window -- dx12
//! ```

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
use winit::window::{Window, WindowId};

#[derive(Default)]
struct App {
    window: Option<Window>,
    created: bool,
    close_requested: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !self.created {
            self.created = true;
            self.window = Some(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("erhi window")
                            .with_inner_size(winit::dpi::PhysicalSize::new(960, 540)),
                    )
                    .expect("create the probe window"),
            );
        }
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if matches!(event, WindowEvent::CloseRequested) {
            self.close_requested = true;
        }
    }
}

fn main() -> Result<(), RhiError> {
    let backend = std::env::args()
        .nth(1)
        .map(|name| name.parse::<Backend>())
        .transpose()?
        .unwrap_or(Backend::Auto);
    let device = Device::open(DeviceOptions {
        backend,
        ..DeviceOptions::default()
    })?;
    println!("backend: {:?}", device.backend());

    let mut event_loop = EventLoop::new().expect("build the event loop");
    let mut app = App::default();
    while app.window.is_none() {
        let _ = event_loop.pump_app_events(Some(Duration::from_millis(10)), &mut app);
    }
    let window = app.window.take().expect("window created");
    let size = window.inner_size();
    let mut configuration = SurfaceConfiguration {
        present_mode: PresentMode::Fifo,
        ..SurfaceConfiguration::new(size.width, size.height)
    };
    let mut surface = device.create_surface(&window, configuration)?;
    println!("swapchain images: {}", surface.image_count());

    let make_image = |width: u32, height: u32| {
        device.create_image(ImageDescriptor::d2(
            ImageFormat::Rgba8Unorm,
            width,
            height,
            ImageUsages::COPY_SOURCE | ImageUsages::COPY_DESTINATION,
        ))
    };
    let mut image = make_image(configuration.width, configuration.height)?;

    let mut frame = 0_u32;
    while !app.close_requested && frame < 600 {
        let _ = event_loop.pump_app_events(Some(Duration::ZERO), &mut app);
        let current = window.inner_size();
        if current.width.max(1) != configuration.width
            || current.height.max(1) != configuration.height
        {
            configuration.width = current.width.max(1);
            configuration.height = current.height.max(1);
            surface.configure(&device, configuration)?;
            image = make_image(configuration.width, configuration.height)?;
        }

        let layout = image.copy_layout(0)?;
        let mut bytes = vec![0_u8; layout.total_size as usize];
        for y in 0..configuration.height as usize {
            let row = y * layout.row_pitch as usize;
            for x in 0..configuration.width as usize {
                let texel = row + x * 4;
                bytes[texel] =
                    ((x as u32 * 255 / configuration.width) as u8).wrapping_add(frame as u8);
                bytes[texel + 1] = (y as u32 * 255 / configuration.height) as u8;
                bytes[texel + 2] = 128_u8.wrapping_add((frame * 2) as u8);
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
        encoder.transition_image(&image, ResourceState::CopyDest)?;
        encoder.copy_buffer_to_image(&staging, 0, &image, 0)?;
        encoder.transition_image(&image, ResourceState::CopySource)?;
        encoder.finish()?;

        if surface.present_image(&device, &image)? == PresentStatus::OutOfDate {
            let size = window.inner_size();
            configuration.width = size.width.max(1);
            configuration.height = size.height.max(1);
            surface.configure(&device, configuration)?;
            image = make_image(configuration.width, configuration.height)?;
        }
        frame += 1;
    }
    println!("presented {frame} frames");
    Ok(())
}
