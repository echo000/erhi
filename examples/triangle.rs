//! A window with one coloured triangle turning in it: a device, two shader
//! modules, a render pipeline, a draw, a present.
//!
//! ```text
//! cargo run --example triangle
//! cargo run --example triangle -- dx12
//! cargo run --example triangle -- vulkan 120
//! ```
//!
//! The first argument is the backend (`vulkan`, `dx12`; whichever opens
//! when left out), the second a number of frames to stop after. The shader
//! is `shaders/triangle.slang`, compiled ahead to the SPIR-V and DXIL
//! beside it (`compile_shaders.ps1` rebuilds them): a device takes
//! compiled bytes and runs no compiler.

mod common;

use common::shader;
use erhi::{
    BlendMode, BufferDescriptor, BufferMemory, BufferUsages, ColorTargetState, Device,
    DeviceOptions, DrawDescriptor, ImageFormat, ImageUsages, LoadOp, PipelineLayoutDescriptor,
    PushConstantRange, RenderPassDescriptor, RenderPipelineDescriptor, ResourceState, RhiError,
    ShaderKind, ShaderStages, VertexAttribute, VertexBufferLayout, VertexFormat,
};

fn main() -> Result<(), RhiError> {
    let (backend, frames) = common::arguments()?;
    // A triangle needs no ray tracing: open on any adapter.
    let device = Device::open(DeviceOptions {
        backend,
        require_ray_tracing: false,
        ..DeviceOptions::default()
    })?;

    // The pipeline: one stream of position and colour, sixteen bytes of
    // push constants for the vertex shader, no descriptor sets.
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
            "triangle",
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

    // Three vertices: where, and what colour.
    let vertices = common::bytes(&[
        0.0, 0.7, 1.0, 0.25, 0.2, //
        -0.6, -0.5, 0.2, 0.85, 0.35, //
        0.6, -0.5, 0.25, 0.4, 1.0,
    ]);
    let vertex_buffer = device.create_buffer(BufferDescriptor::new(
        vertices.len() as u64,
        BufferUsages::VERTEX,
        BufferMemory::Upload,
    ))?;
    vertex_buffer.write(0, &vertices)?;

    common::run(
        "erhi triangle",
        &device,
        ImageUsages::RENDER_TARGET,
        frames,
        |frame| {
            let angle = frame.time * 1.2;
            let aspect = frame.height as f32 / frame.width as f32;
            let constants = common::bytes(&[angle.cos(), angle.sin(), aspect, 0.0]);

            let mut encoder = device.begin_commands()?;
            encoder.begin_render_pass(&RenderPassDescriptor::color(
                &frame.target.create_full_view()?,
                LoadOp::Clear([0.05, 0.06, 0.09, 1.0]),
            ))?;
            encoder.draw(
                &DrawDescriptor::new(&pipeline)
                    .with_vertex_buffers(&[&vertex_buffer])
                    .with_range(0, 3)
                    .with_push_constants(&constants),
            )?;
            encoder.end_render_pass()?;
            encoder.transition_image(frame.target, ResourceState::CopySource)?;
            encoder.finish()
        },
    )
}
