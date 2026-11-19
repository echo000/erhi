//! Compute pipelines with push constants and immutable samplers, timestamp
//! queries, and the render pipeline path, on each backend.
//!
//! Both backends run sequentially in one test body: parallel multi-device

//! work from test threads has wedged this machine's driver before.

#![cfg(feature = "vulkan")]

mod fixtures;

use erhi::{
    Backend, BlendMode, BufferDescriptor, BufferMemory, BufferUsages, ColorTargetState,
    ComputePipelineDescriptor, DepthAttachment, DepthStencilState, DescriptorBinding,
    DescriptorSetLayoutDescriptor, DescriptorType, Device, DeviceOptions, DispatchDescriptor,
    DrawDescriptor, IMAGE_COPY_OFFSET_ALIGNMENT, ImageDescriptor, ImageFormat, ImageUsages, LoadOp,
    PipelineLayoutDescriptor, PushConstantRange, RenderPassDescriptor, RenderPipelineDescriptor,
    ResourceState, RhiError, ShaderKind, ShaderStages, StaticSampler, VertexAttribute,
    VertexBufferLayout, VertexFormat,
};

const SIZE: u32 = 8;

fn upload_row_padded(
    device: &Device,
    pixels: &[[u8; 4]],
    width: u32,
    height: u32,
) -> Result<erhi::Buffer, RhiError> {
    let row_pitch = 256_usize;
    let mut bytes = vec![0_u8; row_pitch * height as usize];
    for row in 0..height as usize {
        for column in 0..width as usize {
            let pixel = pixels[row * width as usize + column];
            let start = row * row_pitch + column * 4;
            bytes[start..start + 4].copy_from_slice(&pixel);
        }
    }
    let staging = device.create_buffer(BufferDescriptor::new(
        bytes.len() as u64,
        BufferUsages::COPY_SOURCE,
        BufferMemory::Upload,
    ))?;
    staging.write(0, &bytes)?;
    Ok(staging)
}

fn read_row_padded(bytes: &[u8], width: u32, height: u32) -> Vec<[u8; 4]> {
    let row_pitch = 256_usize;
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for row in 0..height as usize {
        for column in 0..width as usize {
            let start = row * row_pitch + column * 4;
            pixels.push([
                bytes[start],
                bytes[start + 1],
                bytes[start + 2],
                bytes[start + 3],
            ]);
        }
    }
    pixels
}

fn assert_close(actual: [u8; 4], expected: [u8; 4], context: &str) {
    for channel in 0..4 {
        let delta = i16::from(actual[channel]).abs_diff(i16::from(expected[channel]));
        assert!(
            delta <= 2,
            "{context}: channel {channel} expected {} got {} (pixel {actual:?} vs {expected:?})",
            expected[channel],
            actual[channel]
        );
    }
}

/// Compute pipeline + push constants + immutable sampler + timestamps.
fn run_compute_features(backend: Backend) -> Result<(), RhiError> {
    let device = Device::open(DeviceOptions {
        backend,
        ..DeviceOptions::default()
    })?;

    let bindings = [
        DescriptorBinding::new(
            0,
            DescriptorType::CombinedImageSampler(StaticSampler::linear_clamp()),
            ShaderStages::COMPUTE,
        ),
        DescriptorBinding::new(1, DescriptorType::StorageImage, ShaderStages::COMPUTE),
    ];
    let sets = [DescriptorSetLayoutDescriptor::new(0, &bindings)];
    let layout = device.create_pipeline_layout(
        PipelineLayoutDescriptor::new(&sets)
            .with_push_constants(PushConstantRange::new(ShaderStages::COMPUTE, 16)),
    )?;
    let module = fixtures::module(
        &device,
        "pipelines_test_compute",
        "pipelines_compute",
        ShaderKind::Compute { entry: "main" },
    )?;
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor::new(
        "pipelines_test_compute",
        &layout,
        module.entry("main"),
    ))?;

    // Source: a deterministic gradient. Destination starts undefined.
    let source_pixels: Vec<[u8; 4]> = (0..SIZE * SIZE)
        .map(|index| {
            let x = (index % SIZE) as u8;
            let y = (index / SIZE) as u8;
            [x * 16, y * 16, 64, 255]
        })
        .collect();
    let staging = upload_row_padded(&device, &source_pixels, SIZE, SIZE)?;
    let source = device.create_image(ImageDescriptor::d2(
        ImageFormat::Rgba8Unorm,
        SIZE,
        SIZE,
        ImageUsages::SAMPLED | ImageUsages::COPY_DESTINATION,
    ))?;
    let destination = device.create_image(ImageDescriptor::d2(
        ImageFormat::Rgba8Unorm,
        SIZE,
        SIZE,
        ImageUsages::STORAGE | ImageUsages::COPY_SOURCE,
    ))?;
    let source_view = source.create_full_view()?;
    let destination_view = destination.create_full_view()?;

    let mut set = device.create_descriptor_set(&layout, 0)?;
    set.write_combined_image_sampler(0, &source_view)?;
    set.write_storage_image(1, &destination_view)?;

    let readback = device.create_buffer(BufferDescriptor::new(
        256 * u64::from(SIZE),
        BufferUsages::COPY_DESTINATION,
        BufferMemory::Readback,
    ))?;
    let timestamps = device.create_timestamp_query_set(2)?;
    let timestamp_readback = device.create_buffer(BufferDescriptor::new(
        16,
        BufferUsages::COPY_DESTINATION,
        BufferMemory::Readback,
    ))?;

    // Push adds a quarter to red and an eighth to green.
    let add: [f32; 4] = [0.25, 0.125, 0.0, 0.0];
    let push_bytes: Vec<u8> = add.iter().flat_map(|value| value.to_le_bytes()).collect();

    let mut encoder = device.begin_commands()?;
    encoder.reset_timestamps(&timestamps)?;
    encoder.transition_image(&source, ResourceState::CopyDest)?;
    encoder.copy_buffer_to_image(&staging, 0, &source, 0)?;
    encoder.transition_image(&source, ResourceState::ShaderResource)?;
    encoder.transition_image(&destination, ResourceState::UnorderedAccess)?;
    encoder.write_timestamp(&timestamps, 0)?;
    encoder.dispatch(
        &DispatchDescriptor::new(&pipeline)
            .with_sets(&[&set])
            .with_push_constants(&push_bytes)
            .with_groups([1, 1, 1]),
    )?;
    encoder.write_timestamp(&timestamps, 1)?;
    encoder.transition_image(&destination, ResourceState::CopySource)?;
    encoder.copy_image_to_buffer(&destination, 0, &readback, 0)?;
    encoder.resolve_timestamps(&timestamps, 0, 2, &timestamp_readback, 0)?;
    encoder.finish()?;

    let pixels = read_row_padded(&readback.read(0, 256 * u64::from(SIZE))?, SIZE, SIZE);
    for (index, pixel) in pixels.iter().enumerate() {
        let expected_source = source_pixels[index];
        let expected = [
            expected_source[0].saturating_add(64),
            expected_source[1].saturating_add(32),
            expected_source[2],
            expected_source[3],
        ];
        assert_close(
            *pixel,
            expected,
            &format!("{backend:?} compute texel {index}"),
        );
    }

    let ticks = timestamp_readback.read(0, 16)?;
    let start = u64::from_le_bytes(ticks[0..8].try_into().expect("eight bytes"));
    let end = u64::from_le_bytes(ticks[8..16].try_into().expect("eight bytes"));
    assert!(
        end > start,
        "{backend:?}: timestamp {end} must follow {start}"
    );
    let period = device.timestamp_period_nanoseconds()?;
    assert!(
        period > 0.0 && period < 1_000.0,
        "{backend:?}: implausible timestamp period {period} ns/tick"
    );
    let span_nanoseconds = (end - start) as f64 * period;
    assert!(
        span_nanoseconds < 1.0e9,
        "{backend:?}: an 8x8 dispatch took {span_nanoseconds} ns"
    );
    Ok(())
}

/// Raster pipeline: full-screen alpha-blended triangle over a cleared target.
fn run_raster_features(backend: Backend) -> Result<(), RhiError> {
    let device = Device::open(DeviceOptions {
        backend,
        debug: true,
        ..DeviceOptions::default()
    })?;

    let layout = device.create_pipeline_layout(
        PipelineLayoutDescriptor::new(&[])
            .with_push_constants(PushConstantRange::new(ShaderStages::VERTEX, 16)),
    )?;
    let vertex_module = fixtures::module(
        &device,
        "pipelines_test_vertex",
        "pipelines_raster",
        ShaderKind::Vertex {
            entry: "vertex_main",
        },
    )?;
    let fragment_module = fixtures::module(
        &device,
        "pipelines_test_fragment",
        "pipelines_raster",
        ShaderKind::Fragment {
            entry: "fragment_main",
        },
    )?;
    let attributes = [
        VertexAttribute::new(0, 0, VertexFormat::Float32x2),
        VertexAttribute::new(1, 8, VertexFormat::Float32x4),
    ];
    let vertex_buffers = [VertexBufferLayout::new(24, &attributes)];
    let color_targets = [ColorTargetState::new(
        ImageFormat::Rgba8Unorm,
        BlendMode::Alpha,
    )];
    let descriptor = RenderPipelineDescriptor::new(
        "pipelines_test_raster",
        &layout,
        vertex_module.entry("vertex_main"),
        fragment_module.entry("fragment_main"),
    )
    .with_vertex_buffers(&vertex_buffers)
    .with_color_targets(&color_targets);
    let pipeline = device.create_render_pipeline(&descriptor)?;

    // One triangle covering the whole viewport: half-transparent red.
    let vertices: [[f32; 6]; 3] = [
        [-1.0, -1.0, 1.0, 0.0, 0.0, 0.5],
        [3.0, -1.0, 1.0, 0.0, 0.0, 0.5],
        [-1.0, 3.0, 1.0, 0.0, 0.0, 0.5],
    ];
    let vertex_bytes: Vec<u8> = vertices
        .iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let vertex_buffer = device.create_buffer(BufferDescriptor::new(
        vertex_bytes.len() as u64,
        BufferUsages::VERTEX,
        BufferMemory::Upload,
    ))?;
    vertex_buffer.write(0, &vertex_bytes)?;

    let target = device.create_image(ImageDescriptor::d2(
        ImageFormat::Rgba8Unorm,
        64,
        64,
        ImageUsages::RENDER_TARGET | ImageUsages::COPY_SOURCE,
    ))?;
    let target_view = target.create_full_view()?;
    let readback = device.create_buffer(BufferDescriptor::new(
        256 * 64,
        BufferUsages::COPY_DESTINATION,
        BufferMemory::Readback,
    ))?;

    // Identity transform through the push constants.
    let scale_offset: [f32; 4] = [1.0, 1.0, 0.0, 0.0];
    let push_bytes: Vec<u8> = scale_offset
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();

    let mut encoder = device.begin_commands()?;
    encoder.transition_image(&target, ResourceState::RenderTarget)?;
    encoder.begin_render_pass(&RenderPassDescriptor::color(
        &target_view,
        LoadOp::Clear([0.0, 0.0, 1.0, 1.0]),
    ))?;
    encoder.draw(
        &DrawDescriptor::new(&pipeline)
            .with_vertex_buffers(&[&vertex_buffer])
            .with_range(0, 3)
            .with_push_constants(&push_bytes),
    )?;
    encoder.end_render_pass()?;
    encoder.transition_image(&target, ResourceState::CopySource)?;
    encoder.copy_image_to_buffer(&target, 0, &readback, 0)?;
    encoder.finish()?;

    let depth_fragment = fixtures::module(
        &device,
        "depth_only",
        "pipelines_raster",
        ShaderKind::Fragment {
            entry: "depth_main",
        },
    )?;
    let depth_pipeline = device.create_render_pipeline(&{
        let mut made = descriptor;
        made.fragment_module = &depth_fragment;
        made.fragment_entry = "depth_main";
        made.color_targets = &[];
        made.depth_stencil = Some(DepthStencilState::depth32(true));
        made
    })?;
    let depth = device.create_image({
        let mut made = target.descriptor();
        made.format = ImageFormat::Depth32Float;
        made.usages = ImageUsages::DEPTH_TARGET | ImageUsages::COPY_SOURCE;
        made
    })?;
    let depth_readback = device.create_buffer(readback.descriptor())?;
    let mut encoder = device.begin_commands()?;
    encoder.transition_image(&depth, ResourceState::DepthWrite)?;
    encoder.begin_render_pass(
        &RenderPassDescriptor::new().with_depth(DepthAttachment::new(&depth.create_full_view()?)),
    )?;
    assert!(
        encoder
            .draw(
                &DrawDescriptor::new(&pipeline)
                    .with_vertex_buffers(&[&vertex_buffer])
                    .with_range(0, 3)
                    .with_push_constants(&push_bytes)
            )
            .is_err()
    );
    encoder.draw(
        &DrawDescriptor::new(&depth_pipeline)
            .with_vertex_buffers(&[&vertex_buffer])
            .with_range(0, 3)
            .with_push_constants(&push_bytes),
    )?;
    encoder.end_render_pass()?;
    encoder.transition_image(&depth, ResourceState::CopySource)?;
    let depth_copy = device.create_image({
        let mut made = depth.descriptor();
        made.usages = ImageUsages::COPY_SOURCE | ImageUsages::COPY_DESTINATION;
        made
    })?;
    encoder.transition_image(&depth_copy, ResourceState::CopyDest)?;
    encoder.copy_image(&depth, &depth_copy)?;
    encoder.transition_image(&depth_copy, ResourceState::CopySource)?;
    encoder.copy_image_to_buffer(&depth_copy, 0, &depth_readback, 0)?;
    encoder.finish()?;
    let depth_bytes = depth_readback.read(0, 256 * 64)?;
    assert!(
        depth_bytes
            .chunks_exact(4)
            .all(|p| f32::from_le_bytes(p.try_into().unwrap()) == 0.)
    );

    // Alpha 0.5: red over blue leaves (128, 0, 127); alpha accumulates at
    // full weight: 0.5 + 0.5 * 1.0 = 1.0.
    let pixels = read_row_padded(&readback.read(0, 256 * 64)?, 64, 64);
    let expected = [128, 0, 127, 255];
    for (name, index) in [
        ("top-left", 0_usize),
        ("top-right", 63),
        ("center", 32 * 64 + 32),
        ("bottom-left", 63 * 64),
        ("bottom-right", 63 * 64 + 63),
    ] {
        assert_close(
            pixels[index],
            expected,
            &format!("{backend:?} raster {name}"),
        );
    }
    eprintln!("{:#?}", device.debug_messages());
    assert_eq!(device.debug_error_count().unwrap_or(0), 0);
    Ok(())
}

#[test]
#[ignore = "requires an RT-capable Vulkan 1.2+ device (and a DXR adapter when dx12 is enabled)"]
fn compute_timestamps_and_raster_run_on_every_backend() {
    // Offsets must respect the shared copy alignment for the readbacks above.
    assert_eq!(IMAGE_COPY_OFFSET_ALIGNMENT, 512);

    run_compute_features(Backend::Vulkan).expect("Vulkan compute features");
    run_raster_features(Backend::Vulkan).expect("Vulkan raster features");

    #[cfg(all(feature = "dx12", windows))]
    {
        run_compute_features(Backend::Dx12).expect("DX12 compute features");
        run_raster_features(Backend::Dx12).expect("DX12 raster features");
    }
}
