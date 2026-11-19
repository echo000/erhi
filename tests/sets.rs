//! Several descriptor sets bound at once, bindings of several descriptors,
//! and what a pass may do with the mips of one image: on each backend, with
//! the debug layers silent.
//!
//! Both backends run sequentially in one test body: parallel multi-device
//! work from test threads has wedged this machine's driver before.

mod fixtures;

use erhi::{
    Backend, BlendMode, Buffer, BufferBinding, BufferDescriptor, BufferMemory, BufferUsages,
    ColorTargetState, ComputePipelineDescriptor, DescriptorBinding, DescriptorSetLayoutDescriptor,
    DescriptorType, Device, DeviceOptions, DispatchDescriptor, DrawDescriptor, Image,
    ImageDescriptor, ImageFormat, ImageUsages, ImageViewDescriptor, LoadOp, MultisampleState,
    PipelineLayoutDescriptor, PushConstantRange, RegisterOffsets, RenderPassDescriptor,
    RenderPipelineDescriptor, RhiError, SamplerDescriptor, ShaderKind, ShaderStages,
    VertexAttribute, VertexBufferLayout, VertexFormat,
};

const SIZE: u64 = 256;

fn buffer(device: &Device, usages: BufferUsages, memory: BufferMemory) -> Result<Buffer, RhiError> {
    device.create_buffer(BufferDescriptor::new(SIZE, usages, memory))
}

/// A device buffer whose first number is `value`.
fn number(device: &Device, value: u32) -> Result<Buffer, RhiError> {
    let upload = buffer(device, BufferUsages::COPY_SOURCE, BufferMemory::Upload)?;
    upload.write(0, &[0; SIZE as usize])?;
    upload.write(0, &value.to_le_bytes())?;
    let held = buffer(
        device,
        BufferUsages::STORAGE | BufferUsages::COPY_DESTINATION,
        BufferMemory::Device,
    )?;
    let mut encoder = device.begin_commands()?;
    encoder.copy_buffer(&upload, 0, &held, 0, SIZE)?;
    encoder.finish()?;
    Ok(held)
}

fn first_number(device: &Device, from: &Buffer) -> Result<u32, RhiError> {
    let back = buffer(
        device,
        BufferUsages::COPY_DESTINATION,
        BufferMemory::Readback,
    )?;
    let mut encoder = device.begin_commands()?;
    encoder.copy_buffer(from, 0, &back, 0, SIZE)?;
    encoder.finish()?;
    let bytes = back.read(0, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn several_sets_and_several_descriptors(device: &Device) -> Result<(), RhiError> {
    let binding = |descriptor_type, count| {
        DescriptorBinding::new(0, descriptor_type, ShaderStages::COMPUTE).with_count(count)
    };
    let layout = device.create_pipeline_layout(PipelineLayoutDescriptor::new(&[
        DescriptorSetLayoutDescriptor::new(0, &[binding(DescriptorType::ReadOnlyStorageBuffer, 3)]),
        DescriptorSetLayoutDescriptor::new(1, &[binding(DescriptorType::ReadOnlyStorageBuffer, 1)]),
        DescriptorSetLayoutDescriptor::new(2, &[binding(DescriptorType::StorageBuffer, 1)]),
    ]))?;
    let module = fixtures::module(
        device,
        "sets_sum",
        "sets_sum",
        ShaderKind::Compute { entry: "main" },
    )?;
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor::new(
        "sets_sum",
        &layout,
        module.entry("main"),
    ))?;

    let digits = [number(device, 7)?, number(device, 4)?, number(device, 2)?];
    let (once, twice) = (number(device, 1)?, number(device, 2)?);
    let result = || {
        buffer(
            device,
            BufferUsages::STORAGE | BufferUsages::COPY_SOURCE,
            BufferMemory::Device,
        )
    };
    let (first, second) = (result()?, result()?);

    let mut numbers = device.create_descriptor_set(&layout, 0)?;
    for (element, digit) in digits.iter().enumerate().take(2) {
        numbers.write_storage_buffer_at(
            0,
            element as u32,
            BufferBinding::range(digit, 0, SIZE).with_stride(4),
        )?;
    }
    // No such element.
    assert!(matches!(
        numbers.write_storage_buffer_at(
            0,
            3,
            BufferBinding::range(&digits[2], 0, SIZE).with_stride(4)
        ),
        Err(RhiError::InvalidDescriptorBinding { binding: 0, .. })
    ));
    let scale = |by: &Buffer| {
        let mut set = device.create_descriptor_set(&layout, 1)?;
        set.write_storage_buffer(0, BufferBinding::range(by, 0, SIZE).with_stride(4))?;
        Ok::<_, RhiError>(set)
    };
    let (by_one, by_two) = (scale(&once)?, scale(&twice)?);
    let into = |to: &Buffer| {
        let mut set = device.create_descriptor_set(&layout, 2)?;
        set.write_storage_buffer(0, BufferBinding::range(to, 0, SIZE).with_stride(4))?;
        Ok::<_, RhiError>(set)
    };
    let (into_first, into_second) = (into(&first)?, into(&second)?);

    let mut encoder = device.begin_commands()?;
    // One element of the three was never written.
    assert!(matches!(
        encoder.dispatch(
            &DispatchDescriptor::new(&pipeline)
                .with_sets(&[&numbers, &by_one, &into_first])
                .with_groups([1, 1, 1])
        ),
        Err(RhiError::InvalidDescriptorBinding {
            set: 0,
            binding: 0,
            ..
        })
    ));
    numbers.write_storage_buffer_at(
        0,
        2,
        BufferBinding::range(&digits[2], 0, SIZE).with_stride(4),
    )?;
    // Fewer sets than the layout declares, and sets out of their order.
    assert!(matches!(
        encoder.dispatch(
            &DispatchDescriptor::new(&pipeline)
                .with_sets(&[&numbers, &by_one])
                .with_groups([1, 1, 1])
        ),
        Err(RhiError::DescriptorSetLayoutMismatch { set: 2 })
    ));
    assert!(matches!(
        encoder.dispatch(
            &DispatchDescriptor::new(&pipeline)
                .with_sets(&[&numbers, &into_first, &by_one])
                .with_groups([1, 1, 1])
        ),
        Err(RhiError::DescriptorSetLayoutMismatch { set: 1 })
    ));
    // The same first set with one pair of the others, then with another.
    encoder.dispatch(
        &DispatchDescriptor::new(&pipeline)
            .with_sets(&[&numbers, &by_one, &into_first])
            .with_groups([1, 1, 1]),
    )?;
    encoder.dispatch(
        &DispatchDescriptor::new(&pipeline)
            .with_sets(&[&numbers, &by_two, &into_second][..])
            .with_groups([1, 1, 1]),
    )?;
    encoder.finish()?;
    assert_eq!(first_number(device, &first)?, 247);
    assert_eq!(first_number(device, &second)?, 494);
    Ok(())
}

/// An image of two mips whose larger is `colour` throughout.
fn chain(device: &Device, colour: [u8; 4]) -> Result<Image, RhiError> {
    let image = device.create_image(
        ImageDescriptor::d2(
            ImageFormat::Rgba8Unorm,
            16,
            16,
            ImageUsages::COPY_DESTINATION
                | ImageUsages::COPY_SOURCE
                | ImageUsages::SAMPLED
                | ImageUsages::RENDER_TARGET,
        )
        .with_mip_count(2),
    )?;
    let copy = image.copy_layout(0)?;
    let mut texels = vec![0_u8; copy.total_size as usize];
    for row in 0..16 {
        let start = row * copy.row_pitch as usize;
        for texel in texels[start..start + 16 * 4].chunks_exact_mut(4) {
            texel.copy_from_slice(&colour);
        }
    }
    let upload = device.create_buffer(BufferDescriptor::new(
        copy.total_size,
        BufferUsages::COPY_SOURCE,
        BufferMemory::Upload,
    ))?;
    upload.write(0, &texels)?;
    let mut encoder = device.begin_commands()?;
    encoder.copy_buffer_to_image(&upload, 0, &image, 0)?;
    encoder.finish()?;
    Ok(image)
}

fn a_pass_reads_one_mip_and_draws_into_the_next(device: &Device) -> Result<(), RhiError> {
    let layout = device.create_pipeline_layout(
        PipelineLayoutDescriptor::new(&[
            DescriptorSetLayoutDescriptor::new(
                0,
                &[DescriptorBinding::new(
                    1,
                    DescriptorType::UniformBuffer,
                    ShaderStages::FRAGMENT,
                )],
            ),
            DescriptorSetLayoutDescriptor::new(
                1,
                &[
                    DescriptorBinding::new(0, DescriptorType::SampledImage, ShaderStages::FRAGMENT),
                    DescriptorBinding::new(1, DescriptorType::Sampler, ShaderStages::FRAGMENT),
                ],
            ),
        ])
        .with_push_constants(PushConstantRange::new(ShaderStages::VERTEX, 16)),
    )?;
    let module = |label, kind| fixtures::module(device, label, "sets_tint", kind);
    let vertex = module(
        "sets_tint_vertex",
        ShaderKind::Vertex {
            entry: "vertex_main",
        },
    )?;
    let fragment = module(
        "sets_tint_fragment",
        ShaderKind::Fragment {
            entry: "fragment_main",
        },
    )?;
    let attributes = [VertexAttribute::new(0, 0, VertexFormat::Float32x2)];
    let vertex_buffers = [VertexBufferLayout::new(8, &attributes)];
    let color_targets = [ColorTargetState::new(
        ImageFormat::Rgba8Unorm,
        BlendMode::Opaque,
    )];
    let descriptor = RenderPipelineDescriptor::new(
        "sets_tint",
        &layout,
        vertex.entry("vertex_main"),
        fragment.entry("fragment_main"),
    )
    .with_vertex_buffers(&vertex_buffers)
    .with_color_targets(&color_targets);
    let pipeline = device.create_render_pipeline(&descriptor)?;
    // A format nothing is drawn into.
    assert!(matches!(
        device.create_render_pipeline(&descriptor.with_color_targets(&[ColorTargetState::new(
            ImageFormat::Bc1Unorm,
            BlendMode::Opaque,
        )])),
        Err(RhiError::InvalidRenderPipeline { .. })
    ));
    // More samples than most devices draw: made where the device does,
    // refused by name where it does not, and never left to the driver.
    match device.create_render_pipeline(&descriptor.with_multisample(MultisampleState {
        count: 16,
        ..MultisampleState::default()
    })) {
        Ok(_) | Err(RhiError::UnsupportedByDevice { .. }) => {}
        Err(other) => panic!("sixteen samples: {other}"),
    }

    // The triangle and both tints go up to device buffers in the encoder
    // that then draws with them.
    let staged = |bytes: &[u8], usages: BufferUsages| {
        let upload = buffer(device, BufferUsages::COPY_SOURCE, BufferMemory::Upload)?;
        upload.write(0, &[0; SIZE as usize])?;
        upload.write(0, bytes)?;
        let held = buffer(
            device,
            usages | BufferUsages::COPY_DESTINATION,
            BufferMemory::Device,
        )?;
        Ok::<_, RhiError>((upload, held))
    };
    let floats = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    let (triangle_upload, triangle) = staged(
        &floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]),
        BufferUsages::VERTEX,
    )?;
    let (white_upload, white) = staged(&floats(&[1.0; 4]), BufferUsages::UNIFORM)?;
    let (amber_upload, amber) = staged(&floats(&[1.0, 0.5, 0.0, 1.0]), BufferUsages::UNIFORM)?;

    let image = chain(device, [51, 102, 153, 255])?;
    let (large, small) = (
        image.create_view(&ImageViewDescriptor::subresource(0, 0))?,
        image.create_view(&ImageViewDescriptor::subresource(1, 0))?,
    );
    let sampler = device.create_sampler(&SamplerDescriptor::default())?;
    let tinted = |with: &Buffer| {
        let mut set = device.create_descriptor_set(&layout, 0)?;
        set.write_uniform_buffer(1, BufferBinding::range(with, 0, SIZE))?;
        Ok::<_, RhiError>(set)
    };
    let (in_white, in_amber) = (tinted(&white)?, tinted(&amber)?);
    let reading = |view| {
        let mut set = device.create_descriptor_set(&layout, 1)?;
        set.write_sampled_image(0, view)?;
        set.write_sampler(1, &sampler)?;
        Ok::<_, RhiError>(set)
    };
    let (from_large, from_small) = (reading(&large)?, reading(&small)?);
    let whole = floats(&[1.0, 1.0, 0.0, 0.0]);
    let back = device.create_buffer(BufferDescriptor::new(
        image.copy_layout(1)?.total_size,
        BufferUsages::COPY_DESTINATION,
        BufferMemory::Readback,
    ))?;
    let timestamps = device.create_timestamp_query_set(2)?;
    let times = buffer(
        device,
        BufferUsages::COPY_DESTINATION,
        BufferMemory::Readback,
    )?;

    let mut encoder = device.begin_commands()?;
    for (upload, held) in [
        (&triangle_upload, &triangle),
        (&white_upload, &white),
        (&amber_upload, &amber),
    ] {
        encoder.copy_buffer(upload, 0, held, 0, SIZE)?;
    }
    encoder.begin_render_pass(&RenderPassDescriptor::color(
        &small,
        LoadOp::Clear([0.0; 4]),
    ))?;
    // No draw in this pass has supplied push constants yet.
    assert!(matches!(
        encoder.draw(
            &DrawDescriptor::new(&pipeline)
                .with_sets(&[&in_white, &from_large])
                .with_vertex_buffers(&[&triangle])
                .with_range(0, 3)
        ),
        Err(RhiError::InvalidPushConstants { .. })
    ));
    // The mip being drawn into is not also read.
    assert!(matches!(
        encoder.draw(
            &DrawDescriptor::new(&pipeline)
                .with_sets(&[&in_white, &from_small])
                .with_vertex_buffers(&[&triangle])
                .with_range(0, 3)
                .with_push_constants(&whole)
        ),
        Err(RhiError::InvalidRenderPass { .. })
    ));
    // Nor are queries resolved while a pass is open.
    assert!(matches!(
        encoder.resolve_timestamps(&timestamps, 0, 2, &times, 0),
        Err(RhiError::InvalidRenderPass { .. })
    ));
    encoder.draw(
        &DrawDescriptor::new(&pipeline)
            .with_sets(&[&in_white, &from_large])
            .with_vertex_buffers(&[&triangle])
            .with_range(0, 3)
            .with_push_constants(&whole),
    )?;
    // The second draw supplies no push constants, so it reuses the first's.
    encoder.draw(
        &DrawDescriptor::new(&pipeline)
            .with_sets(&[&in_amber, &from_large])
            .with_vertex_buffers(&[&triangle])
            .with_range(0, 3),
    )?;
    encoder.end_render_pass()?;
    encoder.copy_image_to_buffer(&image, 1, &back, 0)?;
    encoder.finish()?;

    let texel = back.read(0, 4)?;
    for (actual, expected) in texel.iter().zip([51_u8, 51, 0, 255]) {
        assert!(
            actual.abs_diff(expected) <= 2,
            "the smaller mip is {texel:?}"
        );
    }
    Ok(())
}

/// Bytecode from HLSL written in the usual Direct3D style and compiled by
/// `dxc`: each register class numbered from zero, a named vertex semantic,
/// and push constants at an explicit register.
fn hlsl_numbered_the_direct3d_way(device: &Device) -> Result<(), RhiError> {
    let binding = |binding, descriptor_type| {
        DescriptorBinding::new(binding, descriptor_type, ShaderStages::FRAGMENT)
    };
    // `t0`, `s0` and `b0`, as SPIR-V has them shifted.
    let layout = device.create_pipeline_layout(
        PipelineLayoutDescriptor::new(&[DescriptorSetLayoutDescriptor::new(
            0,
            &[
                binding(0, DescriptorType::SampledImage),
                binding(16, DescriptorType::Sampler),
                binding(32, DescriptorType::UniformBuffer),
            ],
        )
        .with_register_offsets(RegisterOffsets {
            shader_resource: 0,
            sampler: 16,
            constant_buffer: 32,
            unordered_access: 48,
        })])
        .with_push_constants(PushConstantRange::new(ShaderStages::VERTEX, 16).at_register(1, 0)),
    )?;
    // A binding below the offset of its kind has no register.
    assert!(matches!(
        device.create_pipeline_layout(PipelineLayoutDescriptor::new(&[
            DescriptorSetLayoutDescriptor::new(0, &[binding(3, DescriptorType::Sampler)])
                .with_register_offsets(RegisterOffsets {
                    sampler: 16,
                    ..RegisterOffsets::NONE
                })
        ])),
        Err(RhiError::InvalidDescriptorBinding { binding: 3, .. })
    ));
    // Nor may a uniform buffer be at the register the push constants have.
    assert!(matches!(
        device.create_pipeline_layout(
            PipelineLayoutDescriptor::new(&[DescriptorSetLayoutDescriptor::new(
                0,
                &[binding(33, DescriptorType::UniformBuffer)]
            )
            .with_register_offsets(RegisterOffsets {
                constant_buffer: 32,
                ..RegisterOffsets::NONE
            })])
            .with_push_constants(
                PushConstantRange::new(ShaderStages::VERTEX, 16).at_register(1, 0)
            )
        ),
        Err(RhiError::InvalidPushConstants { .. })
    ));
    let vertex = fixtures::module(
        device,
        "sets_hlsl_vertex",
        "sets_hlsl",
        ShaderKind::Vertex {
            entry: "vertex_main",
        },
    )?;
    let fragment = fixtures::module(
        device,
        "sets_hlsl_pixel",
        "sets_hlsl",
        ShaderKind::Fragment {
            entry: "pixel_main",
        },
    )?;
    let attributes =
        [VertexAttribute::new(0, 0, VertexFormat::Float32x2).with_semantic("POSITION", 0)];
    let pipeline = device.create_render_pipeline(
        &RenderPipelineDescriptor::new(
            "sets_hlsl",
            &layout,
            vertex.entry("vertex_main"),
            fragment.entry("pixel_main"),
        )
        .with_vertex_buffers(&[VertexBufferLayout::new(8, &attributes)])
        .with_color_targets(&[ColorTargetState::new(
            ImageFormat::Rgba8Unorm,
            BlendMode::Opaque,
        )]),
    )?;

    let floats = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    let staged = |bytes: &[u8], usages: BufferUsages| {
        let upload = buffer(device, BufferUsages::COPY_SOURCE, BufferMemory::Upload)?;
        upload.write(0, &[0; SIZE as usize])?;
        upload.write(0, bytes)?;
        let held = buffer(
            device,
            usages | BufferUsages::COPY_DESTINATION,
            BufferMemory::Device,
        )?;
        Ok::<_, RhiError>((upload, held))
    };
    let (triangle_upload, triangle) = staged(
        &floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]),
        BufferUsages::VERTEX,
    )?;
    let (amber_upload, amber) = staged(&floats(&[1.0, 0.5, 0.0, 1.0]), BufferUsages::UNIFORM)?;
    let image = chain(device, [51, 102, 153, 255])?;
    let (large, small) = (
        image.create_view(&ImageViewDescriptor::subresource(0, 0))?,
        image.create_view(&ImageViewDescriptor::subresource(1, 0))?,
    );
    let sampler = device.create_sampler(&SamplerDescriptor::default())?;
    let mut set = device.create_descriptor_set(&layout, 0)?;
    set.write_sampled_image(0, &large)?;
    set.write_sampler(16, &sampler)?;
    set.write_uniform_buffer(32, BufferBinding::range(&amber, 0, SIZE))?;
    let back = device.create_buffer(BufferDescriptor::new(
        image.copy_layout(1)?.total_size,
        BufferUsages::COPY_DESTINATION,
        BufferMemory::Readback,
    ))?;

    let mut encoder = device.begin_commands()?;
    encoder.copy_buffer(&triangle_upload, 0, &triangle, 0, SIZE)?;
    encoder.copy_buffer(&amber_upload, 0, &amber, 0, SIZE)?;
    encoder.begin_render_pass(&RenderPassDescriptor::color(
        &small,
        LoadOp::Clear([0.0; 4]),
    ))?;
    encoder.draw(
        &DrawDescriptor::new(&pipeline)
            .with_sets(&[&set])
            .with_vertex_buffers(&[&triangle])
            .with_range(0, 3)
            .with_push_constants(&floats(&[1.0, 1.0, 0.0, 0.0])),
    )?;
    encoder.end_render_pass()?;
    encoder.copy_image_to_buffer(&image, 1, &back, 0)?;
    encoder.finish()?;

    let texel = back.read(0, 4)?;
    for (actual, expected) in texel.iter().zip([51_u8, 51, 0, 255]) {
        assert!(
            actual.abs_diff(expected) <= 2,
            "{:?} from HLSL: the smaller mip is {texel:?}",
            device.backend()
        );
    }
    Ok(())
}

/// Bytecode from GLSL, compiled by `glslc`: one entry point a module, and
/// its name is `main`.
fn glsl_computes(device: &Device) -> Result<(), RhiError> {
    let binding = |binding, descriptor_type| {
        DescriptorBinding::new(binding, descriptor_type, ShaderStages::COMPUTE)
    };
    let layout = device.create_pipeline_layout(PipelineLayoutDescriptor::new(&[
        DescriptorSetLayoutDescriptor::new(
            0,
            &[
                binding(0, DescriptorType::ReadOnlyStorageBuffer),
                binding(1, DescriptorType::StorageBuffer),
            ],
        ),
    ]))?;
    let module = fixtures::glsl(
        device,
        "sets_glsl",
        "sets_glsl.comp",
        ShaderKind::Compute { entry: "main" },
    )?;
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor::new(
        "sets_glsl",
        &layout,
        module.entry("main"),
    ))?;
    let source = number(device, 14)?;
    let destination = buffer(
        device,
        BufferUsages::STORAGE | BufferUsages::COPY_SOURCE,
        BufferMemory::Device,
    )?;
    let mut set = device.create_descriptor_set(&layout, 0)?;
    set.write_storage_buffer(0, BufferBinding::range(&source, 0, SIZE).with_stride(4))?;
    set.write_storage_buffer(
        1,
        BufferBinding::range(&destination, 0, SIZE).with_stride(4),
    )?;
    let mut encoder = device.begin_commands()?;
    encoder.dispatch(
        &DispatchDescriptor::new(&pipeline)
            .with_sets(&[&set])
            .with_groups([1, 1, 1]),
    )?;
    encoder.finish()?;
    assert_eq!(first_number(device, &destination)?, 42);
    Ok(())
}

fn run(backend: Backend) -> Result<(), RhiError> {
    let device = Device::open(DeviceOptions {
        backend,
        debug: true,
        require_ray_tracing: false,
        ..DeviceOptions::default()
    })?;
    several_sets_and_several_descriptors(&device)?;
    a_pass_reads_one_mip_and_draws_into_the_next(&device)?;
    hlsl_numbered_the_direct3d_way(&device)?;
    if backend == Backend::Vulkan {
        glsl_computes(&device)?;
    }
    device.wait_idle()?;
    eprintln!("{:#?}", device.debug_messages());
    assert_eq!(device.debug_error_count().unwrap_or(0), 0, "{backend:?}");
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan device (and a DX12 adapter when dx12 is enabled)"]
fn sets_arrays_and_mips_in_a_pass_work_on_every_backend() {
    run(Backend::Vulkan).expect("Vulkan sets");
    #[cfg(all(feature = "dx12", windows))]
    run(Backend::Dx12).expect("DX12 sets");
}
