//! Hardware checks of ray tracing against triangles, each on Vulkan and on
//! DX12:
//!
//! - A bottom-level structure of indexed triangles, and one of triangles
//!   taken three positions at a time, placed by an instance transform.
//! - Hit groups for triangles: a closest hit, and an any hit that lets
//!   rays through geometry that is not opaque.
//! - The same scene traced from a compute shader with inline ray queries.

mod fixtures;

use erhi::{
    AccelerationStructure, AccelerationStructureInstance, Backend, BufferDescriptor, BufferMemory,
    BufferUsages, ComputePipelineDescriptor, DescriptorBinding, DescriptorSetLayoutDescriptor,
    DescriptorType, Device, DeviceOptions, DispatchDescriptor, HitGroup, Image, ImageDescriptor,
    ImageFormat, ImageUsages, PipelineLayout, PipelineLayoutDescriptor,
    RayTracingPipelineDescriptor, ResourceState, RhiError, ShaderEntry, ShaderKind, ShaderModule,
    ShaderStages, TraceRaysDescriptor, TriangleGeometry,
};

const SIZE: u32 = 16;

const BLUE: [u8; 4] = [0, 0, 255, 255];

fn layout(device: &Device, stage: ShaderStages) -> Result<PipelineLayout, RhiError> {
    let bindings = [
        (0, DescriptorType::AccelerationStructure),
        (1, DescriptorType::StorageImage),
    ]
    .map(|(binding, descriptor_type)| DescriptorBinding::new(binding, descriptor_type, stage));
    device.create_pipeline_layout(PipelineLayoutDescriptor::new(&[
        DescriptorSetLayoutDescriptor::new(0, &bindings),
    ]))
}

fn target(device: &Device) -> Result<Image, RhiError> {
    device.create_image(ImageDescriptor::d2(
        ImageFormat::Rgba8Unorm,
        SIZE,
        SIZE,
        ImageUsages::STORAGE | ImageUsages::COPY_SOURCE,
    ))
}

/// The pixels of `image` (in `CopySource`), row by row.
fn pixels(device: &Device, image: &Image) -> Result<Vec<[u8; 4]>, RhiError> {
    let layout = image.copy_layout(0)?;
    let readback = device.create_buffer(BufferDescriptor::new(
        layout.total_size,
        BufferUsages::COPY_DESTINATION,
        BufferMemory::Readback,
    ))?;
    let mut encoder = device.begin_commands()?;
    encoder.copy_image_to_buffer(image, 0, &readback, 0)?;
    encoder.finish()?;
    let bytes = readback.read(0, layout.total_size)?;
    Ok((0..SIZE as usize)
        .flat_map(|row| {
            let start = row * layout.row_pitch as usize;
            bytes[start..start + SIZE as usize * 4]
                .chunks_exact(4)
                .map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]])
                .collect::<Vec<_>>()
        })
        .collect())
}

fn at(x: u32, y: u32) -> usize {
    (y * SIZE + x) as usize
}

/// A quad of two indexed triangles on the left, and one triangle that is
/// not opaque, made about the origin and moved to the right by its instance.
fn scene(device: &Device) -> Result<AccelerationStructure, RhiError> {
    let quad = device.build_triangle_blas(
        &TriangleGeometry::new(&[
            [-0.75, -0.5, 0.0],
            [-0.25, -0.5, 0.0],
            [-0.75, 0.5, 0.0],
            [-0.25, 0.5, 0.0],
        ])
        .with_indices(&[0, 1, 2, 2, 1, 3]),
    )?;
    let cut_out =
        TriangleGeometry::new(&[[-0.25, -0.5, 0.0], [0.25, -0.5, 0.0], [-0.25, 0.5, 0.0]])
            .with_opaque(false);
    // Triangles that are not whole, or that name positions not there.
    for wrong in [
        {
            let mut made = cut_out;
            made.positions = &cut_out.positions[..2];
            made
        },
        cut_out.with_indices(&[0, 1, 3]),
        cut_out.with_indices(&[0, 1]),
        {
            let mut made = cut_out;
            made.positions = &[[f32::NAN; 3]; 3];
            made
        },
    ] {
        assert!(device.build_triangle_blas(&wrong).is_err());
    }
    let cut_out = device.build_triangle_blas(&cut_out)?;
    let mut moved = AccelerationStructureInstance::new(&cut_out);
    moved.transform[3] = 0.5;
    moved.shader_binding_table_record_offset = 1;
    device.build_tlas(&[AccelerationStructureInstance::new(&quad), moved])
}

/// The quad's two triangles each in their colour, nothing outside it.
fn assert_quad(pixels: &[[u8; 4]], context: &str) {
    for (x, y, expected, what) in [
        (2, 4, [255, 255, 0, 255], "the quad's first triangle"),
        (5, 11, [0, 255, 0, 255], "the quad's second triangle"),
        (6, 8, BLUE, "beside the quad"),
        (3, 2, BLUE, "above the quad"),
        (14, 14, BLUE, "far from everything"),
    ] {
        assert_eq!(pixels[at(x, y)], expected, "{context}: {what}");
    }
}

/// The four shaders of the pipeline, wherever each was compiled from.
#[derive(Clone, Copy)]
struct Stages<'a> {
    ray_gen: ShaderEntry<'a>,
    miss: ShaderEntry<'a>,
    closest_hit: ShaderEntry<'a>,
    let_through: ShaderEntry<'a>,
}

impl<'a> Stages<'a> {
    /// All four by their names in one library.
    fn of(library: &'a ShaderModule) -> Self {
        Self {
            ray_gen: library.entry("ray_gen"),
            miss: library.entry("miss"),
            closest_hit: library.entry("closest_hit"),
            let_through: library.entry("let_through"),
        }
    }
}

fn pipeline_trace(
    device: &Device,
    scene: &AccelerationStructure,
    stages: Stages<'_>,
    from: &str,
) -> Result<(), RhiError> {
    let layout = layout(device, ShaderStages::RAY_GENERATION)?;
    let miss_entries = [stages.miss];
    let hit_groups = [
        HitGroup::new("solid").with_closest_hit(stages.closest_hit),
        HitGroup::new("cut_out").with_any_hit(stages.let_through),
    ];
    let descriptor = RayTracingPipelineDescriptor::new("triangles", &layout, stages.ray_gen, 16)
        .with_miss_entries(&miss_entries)
        .with_hit_groups(&hit_groups);
    // A group cannot go by the name of a shader, on either backend.
    assert!(
        device
            .create_ray_tracing_pipeline(&descriptor.with_hit_groups(&[
                HitGroup::new(stages.closest_hit.entry).with_closest_hit(stages.closest_hit)
            ]))
            .is_err()
    );
    // A group for triangles with nothing to run is no group.
    assert!(
        device
            .create_ray_tracing_pipeline(&descriptor.with_hit_groups(&[HitGroup::new("empty")]))
            .is_err()
    );
    let pipeline = device.create_ray_tracing_pipeline(&descriptor)?;
    let image = target(device)?;
    let mut set = device.create_descriptor_set(&layout, 0)?;
    set.write_acceleration_structure(0, scene)?;
    set.write_storage_image(1, &image.create_full_view()?)?;
    let mut encoder = device.begin_commands()?;
    encoder.transition_image(&image, ResourceState::UnorderedAccess)?;
    encoder.trace_rays(&TraceRaysDescriptor::new(&pipeline, SIZE, SIZE).with_sets(&[&set]))?;
    encoder.transition_image(&image, ResourceState::CopySource)?;
    encoder.finish()?;
    let pixels = pixels(device, &image)?;
    let context = format!("{:?} pipeline {from}", device.backend());
    assert_quad(&pixels, &context);
    // Every hit of the triangle on the right is let through: a miss.
    assert_eq!(pixels[at(10, 5)], BLUE, "{context}: the cut-out");
    Ok(())
}

fn query_trace(device: &Device, scene: &AccelerationStructure) -> Result<(), RhiError> {
    if !device.capabilities().ray_query {
        println!("{:?}: no inline ray queries", device.backend());
        return Ok(());
    }
    let layout = layout(device, ShaderStages::COMPUTE)?;
    let module = fixtures::module(
        device,
        "ray queries",
        "triangles_query",
        ShaderKind::Compute { entry: "main" },
    )?;
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor::new(
        "ray queries",
        &layout,
        module.entry("main"),
    ))?;
    let image = target(device)?;
    let mut set = device.create_descriptor_set(&layout, 0)?;
    set.write_acceleration_structure(0, scene)?;
    set.write_storage_image(1, &image.create_full_view()?)?;
    let mut encoder = device.begin_commands()?;
    encoder.transition_image(&image, ResourceState::UnorderedAccess)?;
    encoder.dispatch(
        &DispatchDescriptor::new(&pipeline)
            .with_sets(&[&set])
            .with_groups([SIZE / 8, SIZE / 8, 1]),
    )?;
    encoder.transition_image(&image, ResourceState::CopySource)?;
    encoder.finish()?;
    let pixels = pixels(device, &image)?;
    let context = format!("{:?} query", device.backend());
    assert_quad(&pixels, &context);
    // The triangle on the right is where its instance put it, and its hit
    // was the query's to take.
    assert_eq!(pixels[at(10, 5)], [255; 4], "{context}: the cut-out");
    assert_eq!(pixels[at(13, 10)], BLUE, "{context}: past the cut-out");
    Ok(())
}

fn run(backend: Backend) -> Result<(), RhiError> {
    let device = Device::open(DeviceOptions {
        backend,
        debug: true,
        ..DeviceOptions::default()
    })?;
    let scene = scene(&device)?;
    let library = |file| fixtures::module(&device, file, file, ShaderKind::RayTracingLibrary);
    pipeline_trace(
        &device,
        &scene,
        Stages::of(&library("triangles_pipeline")?),
        "from Slang",
    )?;
    pipeline_trace(
        &device,
        &scene,
        Stages::of(&library("triangles_hlsl")?),
        "from HLSL",
    )?;
    if backend == Backend::Vulkan {
        // A module a stage, each with the one entry point `main`.
        let stage = |extension: &str| {
            fixtures::glsl(
                &device,
                "triangles_glsl",
                &format!("triangles_glsl.{extension}"),
                ShaderKind::RayTracingLibrary,
            )
        };
        let (ray_gen, miss, closest_hit, let_through) = (
            stage("rgen")?,
            stage("rmiss")?,
            stage("rchit")?,
            stage("rahit")?,
        );
        pipeline_trace(
            &device,
            &scene,
            Stages {
                ray_gen: ray_gen.entry("main"),
                miss: miss.entry("main"),
                closest_hit: closest_hit.entry("main"),
                let_through: let_through.entry("main"),
            },
            "from GLSL",
        )?;
    }
    query_trace(&device, &scene)?;
    eprintln!("{:#?}", device.debug_messages());
    assert_eq!(device.debug_error_count().unwrap_or(0), 0, "{backend:?}");
    Ok(())
}

#[test]
#[ignore = "requires an RT-capable Vulkan device (and a DXR adapter when dx12 is enabled)"]
fn triangles_are_traced_on_every_backend() {
    run(Backend::Vulkan).expect("Vulkan triangle ray tracing");

    #[cfg(all(feature = "dx12", windows))]
    run(Backend::Dx12).expect("DX12 triangle ray tracing");
}
