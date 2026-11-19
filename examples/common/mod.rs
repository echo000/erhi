//! Code shared by the windowed examples: argument parsing, window creation,
//! and the main loop, which gives each frame an image to render into and then
//! presents it.

#![allow(dead_code)]

use std::time::{Duration, Instant};

use erhi::{
    Backend, Device, Image, ImageDescriptor, ImageFormat, ImageUsages, PresentMode, PresentStatus,
    RhiError, ShaderBinaryFormat, ShaderKind, ShaderModule, ShaderModuleDescriptor, ShaderSource,
    SurfaceConfiguration,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::platform::pump_events::EventLoopExtPumpEvents;
use winit::window::{Window, WindowId};

/// The backend an example was asked for (`vulkan`, `dx12`; whichever opens
/// when left out) and the number of frames to stop after, if one was given.
pub fn arguments() -> Result<(Backend, Option<u32>), RhiError> {
    let mut arguments = std::env::args().skip(1);
    let backend = arguments
        .next()
        .map(|name| name.parse::<Backend>())
        .transpose()?
        .unwrap_or(Backend::Auto);
    Ok((
        backend,
        arguments.next().and_then(|count| count.parse().ok()),
    ))
}

/// A shader module from whichever of the two compiled forms the device
/// takes. [`shader!`] gives both from the files beside the source.
pub fn module(
    device: &Device,
    label: &str,
    kind: ShaderKind<'_>,
    spirv: &[u8],
    dxil: &[u8],
) -> Result<ShaderModule, RhiError> {
    device.create_shader_module(&ShaderModuleDescriptor::new(
        label,
        kind,
        match device.shader_binary_format() {
            ShaderBinaryFormat::SpirV => ShaderSource::SpirV(spirv),
            ShaderBinaryFormat::Dxil => ShaderSource::Dxil(dxil),
        },
    ))
}

/// The module `shaders/<stem>.spv` or `shaders/<stem>.dxil`, whichever the
/// device takes: `compile_shaders.ps1` built both from the source beside
/// them.
macro_rules! shader {
    ($device:expr, $kind:expr, $stem:literal) => {
        common::module(
            $device,
            $stem,
            $kind,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/examples/shaders/",
                $stem,
                ".spv"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/examples/shaders/",
                $stem,
                ".dxil"
            )),
        )
    };
}

pub(crate) use shader;

/// `values` as the bytes a buffer or a push-constant block holds.
pub fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// One frame to fill.
pub struct Frame<'a> {
    /// What is presented once the frame is filled: it must be left in
    /// [`erhi::ResourceState::CopySource`].
    pub target: &'a Image,
    pub width: u32,
    pub height: u32,
    /// How many frames came before this one.
    pub index: u32,
    /// Total seconds elapsed since the first frame. For motion that is a
    /// function of time, such as `angle = time * speed`.
    pub time: f32,
    /// Seconds elapsed since the previous frame (delta time). For anything
    /// that accumulates per frame, such as a fixed-rate simulation step.
    pub delta: f32,
    /// Whether `target` is a new image: on the first frame, and after the
    /// window changes size.
    pub fresh: bool,
    /// Whether space or a mouse button went down since the last frame.
    pub pressed: bool,
}

#[derive(Default)]
struct App {
    title: String,
    window: Option<Window>,
    created: bool,
    close_requested: bool,
    pressed: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !self.created {
            self.created = true;
            self.window = Some(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title(self.title.as_str())
                            .with_inner_size(winit::dpi::PhysicalSize::new(960, 540)),
                    )
                    .expect("create the window"),
            );
        }
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.close_requested = true,
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                ..
            }
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Named(NamedKey::Space),
                        ..
                    },
                ..
            } => self.pressed = true,
            _ => {}
        }
    }
}

/// Open a window and call `draw` for each frame until the window closes or
/// `frames` have been drawn. The image a frame fills has `usages` (and may
/// be copied from, to be presented) and the size of the window.
pub fn run(
    title: &str,
    device: &Device,
    usages: ImageUsages,
    frames: Option<u32>,
    mut draw: impl FnMut(Frame<'_>) -> Result<(), RhiError>,
) -> Result<(), RhiError> {
    println!(
        "{:?} on {}",
        device.backend(),
        device.capabilities().device_name
    );
    let mut event_loop = EventLoop::new().expect("build the event loop");
    let mut app = App {
        title: title.to_owned(),
        ..App::default()
    };
    while app.window.is_none() {
        let _ = event_loop.pump_app_events(Some(Duration::from_millis(10)), &mut app);
    }
    let window = app.window.take().expect("window created");
    let size = window.inner_size();
    let mut configuration = SurfaceConfiguration {
        present_mode: PresentMode::Fifo,
        ..SurfaceConfiguration::new(size.width.max(1), size.height.max(1))
    };
    let mut surface = device.create_surface(&window, configuration)?;
    // Frames are shown as fast as they are drawn where the window allows
    // it, so the rate in the title is what the device can do and not the
    // display's refresh.
    if surface
        .capabilities(device)?
        .present_modes
        .contains(&PresentMode::Immediate)
    {
        configuration.present_mode = PresentMode::Immediate;
        surface.configure(device, configuration)?;
    }
    let make_target = |width: u32, height: u32| {
        device.create_image(ImageDescriptor::d2(
            ImageFormat::Rgba8Unorm,
            width,
            height,
            usages | ImageUsages::COPY_SOURCE,
        ))
    };
    let mut target = make_target(configuration.width, configuration.height)?;
    let mut fresh = true;

    let started = Instant::now();
    let mut last = 0.0_f32;

    // How fast frames are being drawn, shown in the title twice a second.
    let mut counted_from = Instant::now();
    let mut counted = 0_u32;

    let mut index = 0_u32;
    while !app.close_requested && frames.is_none_or(|count| index < count) {
        let _ = event_loop.pump_app_events(Some(Duration::ZERO), &mut app);
        let current = window.inner_size();
        let (width, height) = (current.width.max(1), current.height.max(1));
        if width != configuration.width || height != configuration.height {
            // Nothing made for the old size may still be in use.
            device.wait_idle()?;
            configuration.width = width;
            configuration.height = height;
            surface.configure(device, configuration)?;
            target = make_target(width, height)?;
            fresh = true;
        }

        let time = started.elapsed().as_secs_f32();
        draw(Frame {
            target: &target,
            width: configuration.width,
            height: configuration.height,
            index,
            time,
            delta: time - last,
            fresh,
            pressed: std::mem::take(&mut app.pressed),
        })?;
        fresh = false;
        last = time;

        if surface.present_image(device, &target)? == PresentStatus::OutOfDate {
            device.wait_idle()?;
            let size = window.inner_size();
            configuration.width = size.width.max(1);
            configuration.height = size.height.max(1);
            surface.configure(device, configuration)?;
            target = make_target(configuration.width, configuration.height)?;
            fresh = true;
        }
        index += 1;

        counted += 1;
        let elapsed = counted_from.elapsed();
        if elapsed >= Duration::from_millis(500) {
            let rate = f64::from(counted) / elapsed.as_secs_f64();
            window.set_title(&format!(
                "{title}: {rate:.0} fps, {:.2} ms, {:?}",
                1000.0 / rate,
                device.backend()
            ));
            counted_from = Instant::now();
            counted = 0;
        }
    }
    device.wait_idle()?;
    println!("drew {index} frames");
    Ok(())
}
