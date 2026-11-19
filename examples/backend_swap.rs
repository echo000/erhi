//! One window, two backends, swapped while it runs: F12 or space closes the
//! device and opens the other backend's on the same window.
//!
//! ```text
//! cargo run --example backend_swap
//! cargo run --example backend_swap -- vulkan
//! cargo run --example backend_swap -- dx12 240
//! ```
//!
//! So that it is plain which one is drawing, Direct3D 12 draws a red
//! triangle on blue and Vulkan a blue triangle on red. Nothing else
//! differs: the same code makes the pipeline and draws the frame on both.
//!
//! The first argument is the backend to begin on (`vulkan`, `dx12`;
//! whichever opens when left out). The second is a number of frames to stop
//! after, and then the backend is swapped by itself every sixty frames.
//! The shaders are the triangle example's, `shaders/triangle.slang`.

mod common;

use std::time::{Duration, Instant};

use common::shader;
use erhi::{
    Backend, BlendMode, Buffer, BufferDescriptor, BufferMemory, BufferUsages, ColorTargetState,
    Device, DeviceOptions, DrawDescriptor, Image, ImageDescriptor, ImageFormat, ImageUsages,
    LoadOp, PipelineLayoutDescriptor, PresentMode, PresentStatus, PushConstantRange,
    RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, ResourceState, RhiError,
    ShaderKind, ShaderStages, Surface, SurfaceConfiguration, VertexAttribute, VertexBufferLayout,
    VertexFormat,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::platform::pump_events::EventLoopExtPumpEvents;
use winit::window::{Window, WindowId};

const RED: [f32; 3] = [0.85, 0.12, 0.10];

const BLUE: [f32; 3] = [0.10, 0.25, 0.85];

/// Everything made from one device, and the device. A swap drops all of it
/// and makes it again on the other backend.
///
/// The fields are dropped in this order, so the device goes last: after
/// everything that was made from it.
struct Renderer<'window> {
    surface: Surface<'window>,
    configuration: SurfaceConfiguration,
    target: Image,
    pipeline: RenderPipeline,
    vertices: Buffer,
    background: [f32; 3],
    device: Device,
}

impl<'window> Renderer<'window> {
    /// Open `backend` on `window` and make what a frame needs.
    fn open(backend: Backend, window: &'window Window) -> Result<Self, RhiError> {
        let device = Device::open(DeviceOptions {
            backend,
            require_ray_tracing: false,
            ..DeviceOptions::default()
        })?;
        println!(
            "{:?} on {}",
            device.backend(),
            device.capabilities().device_name
        );

        let size = window.inner_size();
        let configuration = SurfaceConfiguration {
            present_mode: PresentMode::Fifo,
            ..SurfaceConfiguration::new(size.width.max(1), size.height.max(1))
        };
        let mut configuration = configuration;
        let mut surface = device.create_surface(window, configuration)?;
        // Frames are shown as fast as they are drawn where the window
        // allows it. (The surface comes up waiting for the display first,
        // which a window another backend has just let go of needs.)
        if surface
            .capabilities(&device)?
            .present_modes
            .contains(&PresentMode::Immediate)
        {
            configuration.present_mode = PresentMode::Immediate;
            surface.configure(&device, configuration)?;
        }

        let vertex_module = shader!(
            &device,
            ShaderKind::Vertex {
                entry: "vertex_main",
            },
            "triangle.vertex_main"
        )?;
        let fragment_module = shader!(
            &device,
            ShaderKind::Fragment {
                entry: "fragment_main",
            },
            "triangle.fragment_main"
        )?;
        let layout = device.create_pipeline_layout(
            PipelineLayoutDescriptor::new(&[])
                .with_push_constants(PushConstantRange::new(ShaderStages::VERTEX, 16)),
        )?;
        let attributes = [
            VertexAttribute::new(0, 0, VertexFormat::Float32x2),
            VertexAttribute::new(1, 8, VertexFormat::Float32x3),
        ];
        let pipeline = device.create_render_pipeline(
            &RenderPipelineDescriptor::new(
                "backend_swap",
                &layout,
                vertex_module.entry("vertex_main"),
                fragment_module.entry("fragment_main"),
            )
            .with_vertex_buffers(&[VertexBufferLayout::new(20, &attributes)])
            .with_color_targets(&[ColorTargetState::new(
                ImageFormat::Rgba8Unorm,
                BlendMode::Opaque,
            )]),
        )?;

        // The one thing that differs: which colour is the triangle's and
        // which the background's.
        let (triangle, background) = match device.backend() {
            Backend::Dx12 => (RED, BLUE),
            _ => (BLUE, RED),
        };
        let [r, g, b] = triangle;
        let bytes = common::bytes(&[
            0.0, 0.7, r, g, b, //
            -0.6, -0.5, r, g, b, //
            0.6, -0.5, r, g, b,
        ]);
        let vertices = device.create_buffer(BufferDescriptor::new(
            bytes.len() as u64,
            BufferUsages::VERTEX,
            BufferMemory::Upload,
        ))?;
        vertices.write(0, &bytes)?;

        let target = Self::target(&device, configuration)?;
        Ok(Self {
            surface,
            configuration,
            target,
            pipeline,
            vertices,
            background,
            device,
        })
    }

    /// The image a frame is drawn into before it is presented.
    fn target(device: &Device, configuration: SurfaceConfiguration) -> Result<Image, RhiError> {
        device.create_image(ImageDescriptor::d2(
            ImageFormat::Rgba8Unorm,
            configuration.width,
            configuration.height,
            ImageUsages::RENDER_TARGET | ImageUsages::COPY_SOURCE,
        ))
    }

    /// Fit the surface and the target to a window of `width` by `height`.
    fn resize(&mut self, width: u32, height: u32) -> Result<(), RhiError> {
        // Nothing made for the old size may still be in use.
        self.device.wait_idle()?;
        self.configuration.width = width.max(1);
        self.configuration.height = height.max(1);
        self.surface.configure(&self.device, self.configuration)?;
        self.target = Self::target(&self.device, self.configuration)?;
        Ok(())
    }

    /// Draw the frame `time` seconds in, and present it.
    fn draw(&mut self, time: f32, window: &Window) -> Result<(), RhiError> {
        let size = window.inner_size();
        if size.width.max(1) != self.configuration.width
            || size.height.max(1) != self.configuration.height
        {
            self.resize(size.width, size.height)?;
        }

        let angle = time * 1.2;
        let aspect = self.configuration.height as f32 / self.configuration.width as f32;
        let constants = common::bytes(&[angle.cos(), angle.sin(), aspect, 0.0]);
        let [r, g, b] = self.background;

        let mut encoder = self.device.begin_commands()?;
        encoder.begin_render_pass(&RenderPassDescriptor::color(
            &self.target.create_full_view()?,
            LoadOp::Clear([r, g, b, 1.0]),
        ))?;
        encoder.draw(
            &DrawDescriptor::new(&self.pipeline)
                .with_vertex_buffers(&[&self.vertices])
                .with_range(0, 3)
                .with_push_constants(&constants),
        )?;
        encoder.end_render_pass()?;
        encoder.transition_image(&self.target, ResourceState::CopySource)?;
        encoder.finish()?;

        if self.surface.present_image(&self.device, &self.target)? == PresentStatus::OutOfDate {
            let size = window.inner_size();
            self.resize(size.width, size.height)?;
        }
        Ok(())
    }

    /// The backend that is not this one.
    fn other(&self) -> Backend {
        match self.device.backend() {
            Backend::Dx12 => Backend::Vulkan,
            _ => Backend::Dx12,
        }
    }
}

#[derive(Default)]
struct App {
    window: Option<Window>,
    created: bool,
    close_requested: bool,
    swap_requested: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !self.created {
            self.created = true;
            self.window = Some(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("erhi backend swap: F12 or space")
                            .with_inner_size(winit::dpi::PhysicalSize::new(960, 540)),
                    )
                    .expect("create the window"),
            );
        }
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.close_requested = true,
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Named(NamedKey::F12 | NamedKey::Space),
                        repeat: false,
                        ..
                    },
                ..
            } => self.swap_requested = true,
            _ => {}
        }
    }
}

fn main() -> Result<(), RhiError> {
    let (backend, frames) = common::arguments()?;

    let mut event_loop = EventLoop::new().expect("build the event loop");
    let mut app = App::default();
    while app.window.is_none() {
        let _ = event_loop.pump_app_events(Some(Duration::from_millis(10)), &mut app);
    }
    let window = app.window.take().expect("window created");

    let mut renderer = Renderer::open(backend, &window)?;
    let started = Instant::now();
    let mut index = 0_u32;
    let mut swaps = 0_u32;
    // How fast frames are being drawn, shown in the title twice a second.
    let mut counted_from = Instant::now();
    let mut counted = 0_u32;
    while !app.close_requested && frames.is_none_or(|count| index < count) {
        let _ = event_loop.pump_app_events(Some(Duration::ZERO), &mut app);

        // Left to run for a set number of frames, it swaps by itself.
        let timed = frames.is_some() && index > 0 && index.is_multiple_of(60);
        if std::mem::take(&mut app.swap_requested) || timed {
            let other = renderer.other();
            // The old device finishes its work and is dropped, with all
            // that was made from it, before the new one takes the window.
            renderer.device.wait_idle()?;
            drop(renderer);
            renderer = match Renderer::open(other, &window) {
                Ok(opened) => {
                    swaps += 1;
                    opened
                }
                Err(error) => {
                    // A machine with one backend stays on it.
                    println!("{other:?} did not open ({error})");
                    let current = match other {
                        Backend::Dx12 => Backend::Vulkan,
                        _ => Backend::Dx12,
                    };
                    Renderer::open(current, &window)?
                }
            };
        }

        renderer.draw(started.elapsed().as_secs_f32(), &window)?;
        index += 1;

        counted += 1;
        let elapsed = counted_from.elapsed();
        if elapsed >= Duration::from_millis(500) {
            let rate = f64::from(counted) / elapsed.as_secs_f64();
            window.set_title(&format!(
                "erhi backend swap (F12 or space): {rate:.0} fps, {:?}",
                renderer.device.backend()
            ));
            counted_from = Instant::now();
            counted = 0;
        }
    }
    renderer.device.wait_idle()?;
    println!("drew {index} frames, swapped {swaps} times");
    Ok(())
}
