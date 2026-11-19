//! One triangle, ray traced: a bottom-level acceleration structure of it, a
//! top-level one that turns it, a ray-tracing pipeline of three shaders,
//! and a ray for each pixel of the window.
//!
//! ```text
//! cargo run --example rt_triangle
//! cargo run --example rt_triangle -- dx12
//! ```
//!
//! Needs a device with hardware ray tracing. The shaders are
//! `shaders/rt_triangle.slang`: one library, compiled ahead to the SPIR-V
//! and DXIL beside it.

mod common;

use common::shader;
use erhi::{
    AccelerationStructureInstance, DescriptorBinding, DescriptorSetLayoutDescriptor,
    DescriptorType, Device, DeviceOptions, HitGroup, ImageUsages, PipelineLayoutDescriptor,
    RayTracingPipelineDescriptor, ResourceState, RhiError, ShaderKind, ShaderStages,
    TraceRaysDescriptor, TriangleGeometry,
};

fn main() -> Result<(), RhiError> {
    let (backend, frames) = common::arguments()?;
    let device = Device::open(DeviceOptions {
        backend,
        ..DeviceOptions::default()
    })?;

    // The triangle itself, and the scene: one instance of it, built so its
    // transform can change each frame.
    let triangle = device.build_triangle_blas(&TriangleGeometry::new(&[
        [0.0, 0.8, 0.0],
        [-0.8, -0.6, 0.0],
        [0.8, -0.6, 0.0],
    ]))?;
    let scene = device.build_updatable_tlas(&[AccelerationStructureInstance::new(&triangle)])?;

    // What the ray-generation shader reads and writes: the scene, and the
    // image it gives each pixel its colour in.
    let binding = |binding, descriptor_type| {
        DescriptorBinding::new(binding, descriptor_type, ShaderStages::RAY_GENERATION)
    };
    let layout = device.create_pipeline_layout(PipelineLayoutDescriptor::new(&[
        DescriptorSetLayoutDescriptor::new(
            0,
            &[
                binding(0, DescriptorType::AccelerationStructure),
                binding(1, DescriptorType::StorageImage),
            ],
        ),
    ]))?;
    let library = shader!(&device, ShaderKind::RayTracingLibrary, "rt_triangle")?;
    let pipeline = device.create_ray_tracing_pipeline(
        &RayTracingPipelineDescriptor::new("rt_triangle", &layout, library.entry("ray_gen"), 16)
            .with_miss_entries(&[library.entry("miss")])
            .with_hit_groups(&[
                HitGroup::new("triangle").with_closest_hit(library.entry("closest_hit"))
            ]),
    )?;

    let mut set = None;
    common::run(
        "erhi ray-traced triangle",
        &device,
        ImageUsages::STORAGE,
        frames,
        |frame| {
            // A new image to write means a new set that binds it.
            if frame.fresh {
                let mut bound = device.create_descriptor_set(&layout, 0)?;
                bound.write_acceleration_structure(0, &scene)?;
                bound.write_storage_image(1, &frame.target.create_full_view()?)?;
                set = Some(bound);
            }
            let set = set.as_ref().expect("bound on the first frame");

            // Turn the triangle about the upright axis: the rows of a
            // three-by-four transform.
            let (sin, cos) = (frame.time * 1.2).sin_cos();
            let mut turned = AccelerationStructureInstance::new(&triangle);
            turned.transform = [
                cos, 0.0, sin, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                -sin, 0.0, cos, 0.0,
            ];
            device.update_tlas(&scene, &[turned])?;

            let mut encoder = device.begin_commands()?;
            encoder.transition_image(frame.target, ResourceState::UnorderedAccess)?;
            encoder.trace_rays(
                &TraceRaysDescriptor::new(&pipeline, frame.width, frame.height).with_sets(&[set]),
            )?;
            encoder.transition_image(frame.target, ResourceState::CopySource)?;
            encoder.finish()
        },
    )
}
