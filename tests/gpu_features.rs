//! Hardware checks of the features a renderer grows into, each on Vulkan
//! and on DX12:
//!
//! - A compute dispatch whose group counts the GPU reads from a buffer.
//! - Many indirect draws in one call, with and without indices.
//! - Occlusion queries around draws that are seen and that are not.
//! - Names on objects, and pipelines kept from one device to the next.
//! - A pass of four samples a texel, averaged into an ordinary image.

mod fixtures;

use erhi::{
    Backend, BlendMode, Buffer, BufferBinding, BufferDescriptor, BufferMemory, BufferUsages,
    ColorAttachment, ColorTargetState, ComputePipelineDescriptor, DepthAttachment,
    DepthStencilState, DescriptorBinding, DescriptorSetLayoutDescriptor, DescriptorType, Device,
    DeviceOptions, DispatchDescriptor, DrawDescriptor, Image, ImageDescriptor, ImageFormat,
    ImageUsages, IndexBufferBinding, IndexFormat, LoadOp, MultisampleState,
    PipelineLayoutDescriptor, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ResourceState, RhiError, ShaderKind, ShaderStages, VertexAttribute, VertexBufferLayout,
    VertexFormat,
};

const SIZE: u32 = 64;

fn device(backend: Backend) -> Result<Device, RhiError> {
    Device::open(DeviceOptions {
        backend,
        debug: true,
        ..DeviceOptions::default()
    })
}

/// A device-local buffer holding `bytes`, left in `state`.
fn filled(
    device: &Device,
    usages: BufferUsages,
    bytes: &[u8],
    state: ResourceState,
) -> Result<Buffer, RhiError> {
    let staging = device.create_buffer(BufferDescriptor::new(
        bytes.len() as u64,
        BufferUsages::COPY_SOURCE,
        BufferMemory::Upload,
    ))?;
    staging.write(0, bytes)?;
    let buffer = device.create_buffer(BufferDescriptor::new(
        bytes.len() as u64,
        usages | BufferUsages::COPY_DESTINATION,
        BufferMemory::Device,
    ))?;
    let mut encoder = device.begin_commands()?;
    encoder.transition_buffer(&buffer, ResourceState::CopyDest)?;
    encoder.copy_buffer(&staging, 0, &buffer, 0, bytes.len() as u64)?;
    encoder.transition_buffer(&buffer, state)?;
    encoder.finish()?;
    Ok(buffer)
}

fn words(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn readback(device: &Device, size: u64) -> Result<Buffer, RhiError> {
    device.create_buffer(BufferDescriptor::new(
        size,
        BufferUsages::COPY_DESTINATION,
        BufferMemory::Readback,
    ))
}

/// The group counts of a dispatch come from a buffer: 3 by 2 by 1 groups of
/// one thread each add six to a counter.
fn indirect_dispatch(device: &Device) -> Result<(), RhiError> {
    let bindings = [DescriptorBinding::new(
        0,
        DescriptorType::StorageBuffer,
        ShaderStages::COMPUTE,
    )];
    let layout = device.create_pipeline_layout(PipelineLayoutDescriptor::new(&[
        DescriptorSetLayoutDescriptor::new(0, &bindings),
    ]))?;
    let module = fixtures::module(
        device,
        "count",
        "count",
        ShaderKind::Compute { entry: "main" },
    )?;
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor::new(
        "count",
        &layout,
        module.entry("main"),
    ))?;
    let counter = filled(
        device,
        BufferUsages::STORAGE | BufferUsages::COPY_SOURCE,
        &words(&[0; 4]),
        ResourceState::UnorderedAccess,
    )?;
    // The counts sit past a word that is not theirs.
    let arguments = filled(
        device,
        BufferUsages::INDIRECT,
        &words(&[99, 3, 2, 1]),
        ResourceState::IndirectArgument,
    )?;
    let mut set = device.create_descriptor_set(&layout, 0)?;
    set.write_storage_buffer(0, BufferBinding::range(&counter, 0, 16).with_stride(4))?;
    let result = readback(device, 16)?;
    let mut encoder = device.begin_commands()?;
    // Counts that would run off the end of the buffer are refused.
    assert!(
        encoder
            .dispatch_indirect(
                &DispatchDescriptor::new(&pipeline).with_sets(&[&set]),
                &arguments,
                8
            )
            .is_err()
    );
    encoder.dispatch_indirect(
        &DispatchDescriptor::new(&pipeline).with_sets(&[&set]),
        &arguments,
        4,
    )?;
    encoder.transition_buffer(&counter, ResourceState::CopySource)?;
    encoder.copy_buffer(&counter, 0, &result, 0, 16)?;
    encoder.finish()?;
    assert_eq!(
        result.read(0, 4)?,
        6_u32.to_le_bytes(),
        "{:?}",
        device.backend()
    );
    Ok(())
}

struct Quads {
    pipeline: RenderPipeline,
    /// Three quads of six vertices each, an eighth of the clip square to
    /// each side of pixels (16, 32), (32, 32) and (48, 32).
    vertices: Buffer,
    /// 0 to 17: each index its own vertex.
    indices: Buffer,
}

impl Quads {
    fn new(device: &Device) -> Result<Self, RhiError> {
        Self::with(device, MultisampleState::default(), None)
    }

    fn with(
        device: &Device,
        multisample: MultisampleState,
        depth_stencil: Option<DepthStencilState>,
    ) -> Result<Self, RhiError> {
        let layout = device.create_pipeline_layout(PipelineLayoutDescriptor::new(&[]))?;
        let module = |kind, label| fixtures::module(device, label, "gpu_features_quad", kind);
        let vertex = module(
            ShaderKind::Vertex {
                entry: "vertex_main",
            },
            "quad vertex",
        )?;
        let fragment = module(
            ShaderKind::Fragment {
                entry: "fragment_main",
            },
            "quad fragment",
        )?;
        let pipeline = device.create_render_pipeline(
            &RenderPipelineDescriptor::new(
                "quads",
                &layout,
                vertex.entry("vertex_main"),
                fragment.entry("fragment_main"),
            )
            .with_vertex_buffers(&[VertexBufferLayout::new(
                8,
                &[VertexAttribute::new(0, 0, VertexFormat::Float32x2)],
            )])
            .with_multisample(multisample)
            .with_color_targets(&[ColorTargetState::new(
                ImageFormat::Rgba8Unorm,
                BlendMode::Opaque,
            )])
            .with_depth_stencil(depth_stencil),
        )?;
        let mut corners = Vec::new();
        for centre in [-0.5_f32, 0.0, 0.5] {
            for (x, y) in [(-1, -1), (1, -1), (-1, 1), (-1, 1), (1, -1), (1, 1)] {
                corners.extend((centre + 0.125 * x as f32).to_le_bytes());
                corners.extend((0.125 * y as f32).to_le_bytes());
            }
        }
        Ok(Self {
            pipeline,
            vertices: filled(
                device,
                BufferUsages::VERTEX,
                &corners,
                ResourceState::VertexBuffer,
            )?,
            indices: filled(
                device,
                BufferUsages::INDEX,
                &words(&(0..18).collect::<Vec<_>>()),
                ResourceState::GenericRead,
            )?,
        })
    }

    /// A draw of the first quad; `streams` is `[&self.vertices]`.
    fn draw<'a>(&'a self, streams: &'a [&'a Buffer], indexed: bool) -> DrawDescriptor<'a> {
        let mut draw = DrawDescriptor::new(&self.pipeline)
            .with_vertex_buffers(streams)
            .with_range(0, 6);
        draw.indices =
            indexed.then_some(IndexBufferBinding::new(&self.indices, IndexFormat::Uint32));
        draw
    }
}

fn target(device: &Device) -> Result<Image, RhiError> {
    device.create_image(ImageDescriptor::d2(
        ImageFormat::Rgba8Unorm,
        SIZE,
        SIZE,
        ImageUsages::RENDER_TARGET | ImageUsages::COPY_SOURCE,
    ))
}

/// Whether the pixels at x = 16, 32 and 48 of the middle row were drawn.
fn drawn(device: &Device, image: &Image) -> Result<[bool; 3], RhiError> {
    let layout = image.copy_layout(0)?;
    let result = readback(device, layout.total_size)?;
    let mut encoder = device.begin_commands()?;
    encoder.transition_image(image, ResourceState::CopySource)?;
    encoder.copy_image_to_buffer(image, 0, &result, 0)?;
    encoder.finish()?;
    let bytes = result.read(0, layout.total_size)?;
    Ok([16, 32, 48].map(|x| bytes[32 * layout.row_pitch as usize + x * 4 + 1] > 128))
}

/// Records of GPU-written counts, several to a call: without indices (four
/// words each) and with them (five).
fn multi_draw(device: &Device) -> Result<(), RhiError> {
    let quads = Quads::new(device)?;
    // Without indices: the first and the third quad. A record between them
    // that draws nothing, and one after them that is not asked for.
    let plain = filled(
        device,
        BufferUsages::INDIRECT,
        &words(&[6, 1, 0, 0, 0, 1, 6, 0, 6, 1, 12, 0, 6, 1, 6, 0]),
        ResourceState::IndirectArgument,
    )?;
    // With them: the second and the third, the third by way of a base vertex.
    let indexed = filled(
        device,
        BufferUsages::INDIRECT,
        &words(&[6, 1, 6, 0, 0, 6, 1, 6, 6, 0]),
        ResourceState::IndirectArgument,
    )?;
    for (arguments, with_indices, draws, expected) in [
        (&plain, false, 3, [true, false, true]),
        (&plain, false, 1, [true, false, false]),
        (&indexed, true, 2, [false, true, true]),
    ] {
        let image = target(device)?;
        let view = image.create_full_view()?;
        let mut encoder = device.begin_commands()?;
        encoder.transition_image(&image, ResourceState::RenderTarget)?;
        encoder.begin_render_pass(&RenderPassDescriptor::color(
            &view,
            LoadOp::Clear([0.0, 0.0, 1.0, 1.0]),
        ))?;
        let streams = [&quads.vertices];
        let draw = quads.draw(&streams, with_indices);
        // More records than the buffer holds are refused.
        assert!(encoder.multi_draw_indirect(&draw, arguments, 0, 5).is_err());
        encoder.multi_draw_indirect(&draw, arguments, 0, draws)?;
        encoder.end_render_pass()?;
        encoder.finish()?;
        assert_eq!(
            drawn(device, &image)?,
            expected,
            "{:?}: {draws} draws, indices {with_indices}",
            device.backend()
        );
    }
    Ok(())
}

/// One query around a quad that is drawn, one around a quad the scissor
/// keeps out, and one around nothing.
fn occlusion(device: &Device) -> Result<(), RhiError> {
    let quads = Quads::new(device)?;
    let queries = device.create_occlusion_query_set(3)?;
    let image = target(device)?;
    let view = image.create_full_view()?;
    let result = readback(device, 24)?;
    let mut encoder = device.begin_commands()?;
    encoder.reset_occlusion_queries(&queries)?;
    // (A query counts inside a pass.)
    assert!(encoder.begin_occlusion_query(&queries, 0).is_err());
    encoder.transition_image(&image, ResourceState::RenderTarget)?;
    encoder.begin_render_pass(&RenderPassDescriptor::color(
        &view,
        LoadOp::Clear([0.0, 0.0, 1.0, 1.0]),
    ))?;
    let streams = [&quads.vertices];
    let draw = quads.draw(&streams, false);
    encoder.begin_occlusion_query(&queries, 0)?;
    // One at a time, each ended as it was begun, and before the pass is.
    assert!(encoder.begin_occlusion_query(&queries, 1).is_err());
    assert!(encoder.end_occlusion_query(&queries, 1).is_err());
    assert!(encoder.end_render_pass().is_err());
    encoder.draw(&draw)?;
    encoder.end_occlusion_query(&queries, 0)?;
    encoder.set_scissor(40, 0, 8, 8)?;
    encoder.begin_occlusion_query(&queries, 1)?;
    encoder.draw(&draw)?;
    encoder.end_occlusion_query(&queries, 1)?;
    encoder.begin_occlusion_query(&queries, 2)?;
    encoder.end_occlusion_query(&queries, 2)?;
    assert!(encoder.begin_occlusion_query(&queries, 3).is_err());
    encoder.end_render_pass()?;
    encoder.resolve_occlusion_queries(&queries, 0, 3, &result, 0)?;
    encoder.finish()?;
    let counts: Vec<u64> = result
        .read(0, 24)?
        .chunks_exact(8)
        .map(|count| u64::from_le_bytes(count.try_into().unwrap()))
        .collect();
    // The quad is 8 by 8 pixels.
    if device.occlusion_queries_are_precise() {
        assert_eq!(counts, [64, 0, 0], "{:?}", device.backend());
    } else {
        assert!(
            counts[0] > 0 && counts[1..] == [0, 0],
            "{:?}",
            device.backend()
        );
    }
    Ok(())
}

/// A triangle whose long edge runs corner to corner of a target of four
/// samples a texel: inside it is drawn, outside it is not, and the texels
/// the edge crosses come out part of each once the samples are averaged.
fn multisampling(device: &Device) -> Result<(), RhiError> {
    let backend = device.backend();
    assert!(device.supports_sample_count(ImageFormat::Rgba8Unorm, 1));
    assert!(!device.supports_sample_count(ImageFormat::Rgba8Unorm, 3));
    if !device.supports_sample_count(ImageFormat::Rgba8Unorm, 4)
        || !device.supports_sample_count(ImageFormat::Depth32Float, 4)
    {
        println!("{backend:?}: no targets of four samples a texel");
        return Ok(());
    }
    let image = |format, sample_count, usages| {
        device.create_image(
            ImageDescriptor::d2(format, SIZE, SIZE, usages).with_sample_count(sample_count),
        )
    };
    let samples = image(ImageFormat::Rgba8Unorm, 4, ImageUsages::RENDER_TARGET)?;
    let depth = image(ImageFormat::Depth32Float, 4, ImageUsages::DEPTH_TARGET)?;
    let resolved = target(device)?;
    // What a multisampled image cannot be: mipped, or copied to a buffer.
    assert!(matches!(
        device.create_image(samples.descriptor().with_mip_count(2)),
        Err(RhiError::InvalidImageSampleCount { .. })
    ));
    assert!(matches!(
        device.create_image(samples.descriptor().with_sample_count(3)),
        Err(RhiError::InvalidImageSampleCount { .. })
    ));
    assert!(samples.copy_layout(0).is_err());

    let four = Quads::with(
        device,
        MultisampleState {
            count: 4,
            alpha_to_coverage: false,
        },
        Some(DepthStencilState::depth32(true)),
    )?;
    let one = Quads::new(device)?;
    // The lower left half of the target (y is up).
    let mut corners = Vec::new();
    for value in [-1.0_f32, -1.0, 1.0, -1.0, -1.0, 1.0] {
        corners.extend(value.to_le_bytes());
    }
    let triangle = filled(
        device,
        BufferUsages::VERTEX,
        &corners,
        ResourceState::VertexBuffer,
    )?;
    let (samples_view, depth_view, resolved_view) = (
        samples.create_full_view()?,
        depth.create_full_view()?,
        resolved.create_full_view()?,
    );
    let mut encoder = device.begin_commands()?;
    encoder.transition_image(&samples, ResourceState::RenderTarget)?;
    encoder.transition_image(&resolved, ResourceState::RenderTarget)?;
    encoder.transition_image(&depth, ResourceState::DepthWrite)?;
    let depth_attachment = Some(DepthAttachment::new(&depth_view));
    // A resolve target is not itself multisampled, and an attachment of
    // one sample has nothing to resolve.
    for (view, resolve) in [
        (&samples_view, &samples_view),
        (&resolved_view, &resolved_view),
    ] {
        assert!(
            encoder
                .begin_render_pass(
                    &RenderPassDescriptor::new()
                        .with_color(ColorAttachment::new(view, LoadOp::Load).with_resolve(resolve))
                )
                .is_err()
        );
    }
    // Nor do attachments of different sample counts make one pass.
    assert!(
        encoder
            .begin_render_pass(
                &RenderPassDescriptor::color(&resolved_view, LoadOp::Load)
                    .with_depth(depth_attachment)
            )
            .is_err()
    );
    encoder.begin_render_pass(
        &RenderPassDescriptor::new()
            .with_color(
                ColorAttachment::new(&samples_view, LoadOp::Clear([0.0, 0.0, 1.0, 1.0]))
                    .with_resolve(&resolved_view),
            )
            .with_depth(depth_attachment),
    )?;
    let streams = [&triangle];
    let draw = |quads| {
        let mut made = Quads::draw(quads, &streams, false);
        made.count = 3;
        made
    };
    // A pipeline of one sample a texel does not draw into this pass.
    assert!(encoder.draw(&draw(&one)).is_err());
    encoder.draw(&draw(&four))?;
    encoder.end_render_pass()?;
    encoder.transition_image(&resolved, ResourceState::CopySource)?;
    let layout = resolved.copy_layout(0)?;
    let result = readback(device, layout.total_size)?;
    encoder.copy_image_to_buffer(&resolved, 0, &result, 0)?;
    encoder.finish()?;
    let bytes = result.read(0, layout.total_size)?;
    // Green of the pixel at (x, y), y from the top.
    let green = |x: usize, y: usize| bytes[y * layout.row_pitch as usize + x * 4 + 1];
    assert_eq!(green(8, 48), 255, "{backend:?}: inside the triangle");
    assert_eq!(green(56, 8), 0, "{backend:?}: outside the triangle");
    // The edge runs from the top left to the bottom right, through the
    // middle of every texel on that diagonal.
    let partial = (4..60).filter(|at| (32..224).contains(&green(*at, *at)));
    assert!(
        partial.count() > 40,
        "{backend:?}: the edge was not averaged from its samples"
    );
    Ok(())
}

/// Objects take names, and the pipelines one device compiled are handed to
/// the next.
fn names_and_pipeline_cache(backend: Backend) -> Result<(), RhiError> {
    let kept = {
        let device = device(backend)?;
        let quads = Quads::new(&device)?;
        quads.vertices.set_label("quad vertices");
        quads.vertices.set_label("");
        let image = target(&device)?;
        image.set_label("a target with a name");
        indirect_dispatch(&device)?;
        eprintln!("{:#?}", device.debug_messages());
        assert_eq!(device.debug_error_count().unwrap_or(0), 0, "{backend:?}");
        device.pipeline_cache_data()?
    };
    assert!(!kept.is_empty(), "{backend:?} kept no pipelines");
    let device = device(backend)?;
    // Bytes that are no cache, and a cache cut short, are left out.
    assert!(!device.load_pipeline_cache(b"not a pipeline cache at all, only bytes")?);
    assert!(!device.load_pipeline_cache(&[])?);
    assert!(
        device.load_pipeline_cache(&kept)?,
        "{backend:?} refused its own pipelines"
    );
    // The same pipelines again, now with the kept ones to draw on.
    Quads::new(&device)?;
    indirect_dispatch(&device)?;
    assert!(device.pipeline_cache_data()?.len() >= kept.len() / 2);
    eprintln!("{:#?}", device.debug_messages());
    assert_eq!(device.debug_error_count().unwrap_or(0), 0, "{backend:?}");
    Ok(())
}

fn run(backend: Backend) -> Result<(), RhiError> {
    let device = device(backend)?;
    device.check_device()?;
    indirect_dispatch(&device)?;
    multi_draw(&device)?;
    occlusion(&device)?;
    multisampling(&device)?;
    eprintln!("{:#?}", device.debug_messages());
    assert_eq!(device.debug_error_count().unwrap_or(0), 0, "{backend:?}");
    drop(device);
    names_and_pipeline_cache(backend)
}

#[test]
#[ignore = "requires a Vulkan device (and a DX12 adapter when dx12 is enabled)"]
fn renderer_features_agree_on_every_backend() {
    run(Backend::Vulkan).expect("Vulkan renderer features");

    #[cfg(all(feature = "dx12", windows))]
    run(Backend::Dx12).expect("DX12 renderer features");
}
