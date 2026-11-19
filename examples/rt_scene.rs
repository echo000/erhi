//! A knot over a chequered floor, lit by ray tracing: two lights, shadows
//! that are rays towards each, and reflections that are rays onwards.
//!
//! ```text
//! cargo run --example rt_scene
//! cargo run --example rt_scene -- dx12
//! ```
//!
//! Needs a device with hardware ray tracing. The mesh is made here, in
//! code; the hit shader reads its normals from the same vertices the
//! acceleration structure was built of. The shaders are
//! `shaders/rt_scene.slang`.

mod common;

use common::shader;
use erhi::{
    AccelerationStructureInstance, Buffer, BufferBinding, BufferDescriptor, BufferMemory,
    BufferUsages, DescriptorBinding, DescriptorSetLayoutDescriptor, DescriptorType, Device,
    DeviceOptions, HitGroup, ImageUsages, PipelineLayoutDescriptor, RayTracingPipelineDescriptor,
    ResourceState, RhiError, ShaderKind, ShaderStages, TraceRaysDescriptor, TriangleGeometry,
};

type Vector = [f32; 3];

fn add(a: Vector, b: Vector) -> Vector {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn subtract(a: Vector, b: Vector) -> Vector {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: Vector, by: f32) -> Vector {
    [a[0] * by, a[1] * by, a[2] * by]
}

fn cross(a: Vector, b: Vector) -> Vector {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: Vector) -> Vector {
    let length = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    scale(a, 1.0 / length)
}

/// Where the knot's centre line is at `t`, once round for `t` from 0 to 1.
fn knot(t: f32) -> Vector {
    let angle = t * std::f32::consts::TAU;
    let radius = 1.0 + 0.45 * (3.0 * angle).cos();
    [
        radius * (2.0 * angle).cos(),
        1.55 + 0.45 * (3.0 * angle).sin(),
        radius * (2.0 * angle).sin(),
    ]
}

/// The floor's two triangles, then a tube about the knot: positions,
/// normals, and the indices of the triangles.
fn mesh() -> (Vec<Vector>, Vec<Vector>, Vec<u32>) {
    const ALONG: u32 = 256;
    const AROUND: u32 = 20;
    const THICKNESS: f32 = 0.22;
    const FLOOR: f32 = 12.0;

    let mut positions = vec![
        [-FLOOR, 0.0, -FLOOR],
        [FLOOR, 0.0, -FLOOR],
        [-FLOOR, 0.0, FLOOR],
        [FLOOR, 0.0, FLOOR],
    ];
    let mut normals = vec![[0.0, 1.0, 0.0]; 4];
    let mut indices = vec![0, 2, 1, 1, 2, 3];

    let first = positions.len() as u32;
    for step in 0..ALONG {
        let t = step as f32 / ALONG as f32;
        let centre = knot(t);
        let along = normalize(subtract(knot(t + 0.001), knot(t - 0.001)));
        // Two directions across the tube: away from the knot's own axis,
        // and square to that.
        let side = normalize(cross(along, [0.0, 1.0, 0.0]));
        let over = cross(side, along);
        for ring in 0..AROUND {
            let angle = ring as f32 / AROUND as f32 * std::f32::consts::TAU;
            let normal = add(scale(side, angle.cos()), scale(over, angle.sin()));
            positions.push(add(centre, scale(normal, THICKNESS)));
            normals.push(normal);
        }
    }
    for step in 0..ALONG {
        for ring in 0..AROUND {
            let at = |step: u32, ring: u32| first + (step % ALONG) * AROUND + ring % AROUND;
            let (a, b) = (at(step, ring), at(step, ring + 1));
            let (c, d) = (at(step + 1, ring), at(step + 1, ring + 1));
            indices.extend([a, c, b, b, c, d]);
        }
    }
    (positions, normals, indices)
}

/// A device buffer holding `bytes`, for a shader to read.
fn held(device: &Device, bytes: &[u8]) -> Result<Buffer, RhiError> {
    let upload = device.create_buffer(BufferDescriptor::new(
        bytes.len() as u64,
        BufferUsages::COPY_SOURCE,
        BufferMemory::Upload,
    ))?;
    upload.write(0, bytes)?;
    let buffer = device.create_buffer(BufferDescriptor::new(
        bytes.len() as u64,
        BufferUsages::STORAGE | BufferUsages::COPY_DESTINATION,
        BufferMemory::Device,
    ))?;
    let mut encoder = device.begin_commands()?;
    encoder.copy_buffer(&upload, 0, &buffer, 0, bytes.len() as u64)?;
    encoder.finish()?;
    Ok(buffer)
}

/// The size of the block the shaders call `View`, as a uniform buffer.
const VIEW_SIZE: u64 = 256;

fn main() -> Result<(), RhiError> {
    let (backend, frames) = common::arguments()?;
    let device = Device::open(DeviceOptions {
        backend,
        ..DeviceOptions::default()
    })?;

    // The mesh twice over: as what rays are traced against, and as the
    // buffers the hit shader looks its normals up in.
    let (positions, normals, indices) = mesh();
    let geometry =
        device.build_triangle_blas(&TriangleGeometry::new(&positions).with_indices(&indices))?;
    let scene = device.build_tlas(&[AccelerationStructureInstance::new(&geometry)])?;
    let vertex_values: Vec<f32> = positions
        .iter()
        .zip(&normals)
        .flat_map(|(position, normal)| {
            [
                position[0],
                position[1],
                position[2],
                1.0,
                normal[0],
                normal[1],
                normal[2],
                0.0,
            ]
        })
        .collect();
    let vertices = held(&device, &common::bytes(&vertex_values))?;
    let index_bytes: Vec<u8> = indices
        .iter()
        .flat_map(|index| index.to_le_bytes())
        .collect();
    let index_buffer = held(&device, &index_bytes)?;

    // The eye and the light change each frame: written to an upload buffer
    // and copied into the uniform buffer the shaders read.
    let view_upload = device.create_buffer(BufferDescriptor::new(
        VIEW_SIZE,
        BufferUsages::COPY_SOURCE,
        BufferMemory::Upload,
    ))?;
    view_upload.write(0, &[0; VIEW_SIZE as usize])?;
    let view = device.create_buffer(BufferDescriptor::new(
        VIEW_SIZE,
        BufferUsages::UNIFORM | BufferUsages::COPY_DESTINATION,
        BufferMemory::Device,
    ))?;

    let binding = |binding, descriptor_type, visibility| {
        DescriptorBinding::new(binding, descriptor_type, visibility)
    };
    let layout = device.create_pipeline_layout(PipelineLayoutDescriptor::new(&[
        DescriptorSetLayoutDescriptor::new(
            0,
            &[
                binding(
                    0,
                    DescriptorType::AccelerationStructure,
                    ShaderStages::RAY_GENERATION,
                ),
                binding(
                    1,
                    DescriptorType::StorageImage,
                    ShaderStages::RAY_GENERATION,
                ),
                binding(
                    2,
                    DescriptorType::UniformBuffer,
                    ShaderStages::RAY_GENERATION,
                ),
                binding(
                    3,
                    DescriptorType::ReadOnlyStorageBuffer,
                    ShaderStages::CLOSEST_HIT,
                ),
                binding(
                    4,
                    DescriptorType::ReadOnlyStorageBuffer,
                    ShaderStages::CLOSEST_HIT,
                ),
            ],
        ),
    ]))?;
    let library = shader!(&device, ShaderKind::RayTracingLibrary, "rt_scene")?;
    // Two miss shaders: the trace call says which one a ray that lands on
    // nothing runs. Rays towards the light skip the hit shader altogether.
    let pipeline = device.create_ray_tracing_pipeline(
        &RayTracingPipelineDescriptor::new("rt_scene", &layout, library.entry("ray_gen"), 32)
            .with_miss_entries(&[library.entry("miss"), library.entry("shadow_miss")])
            .with_hit_groups(&[
                HitGroup::new("mesh").with_closest_hit(library.entry("closest_hit"))
            ]),
    )?;

    let mut set = None;
    common::run(
        "erhi ray-traced scene",
        &device,
        ImageUsages::STORAGE,
        frames,
        |frame| {
            if frame.fresh {
                let mut bound = device.create_descriptor_set(&layout, 0)?;
                bound.write_acceleration_structure(0, &scene)?;
                bound.write_storage_image(1, &frame.target.create_full_view()?)?;
                bound.write_uniform_buffer(2, BufferBinding::range(&view, 0, VIEW_SIZE))?;
                bound.write_storage_buffer(3, BufferBinding::whole(&vertices).with_stride(32))?;
                bound
                    .write_storage_buffer(4, BufferBinding::whole(&index_buffer).with_stride(4))?;
                set = Some(bound);
            }
            let set = set.as_ref().expect("bound on the first frame");

            // The camera orbits the knot slowly. Each light orbits at its own
            // speed.
            let time = frame.time;
            let eye = [6.5 * (time * 0.25).cos(), 3.2, 6.5 * (time * 0.25).sin()];
            let forward = normalize(subtract([0.0, 1.3, 0.0], eye));
            let right = normalize(cross([0.0, 1.0, 0.0], forward));
            let up = cross(forward, right);
            // A warm light high up and a cooler, dimmer one lower down, each
            // with its place and in the fourth number how bright it is.
            let warm = [4.0 * (-time * 0.7).cos(), 5.5, 4.0 * (-time * 0.7).sin()];
            let cool = [3.0 * (time * 0.4).sin(), 2.5, 3.0 * (time * 0.4).cos()];
            let mut values = Vec::with_capacity(32);
            for vector in [eye, forward, right, up] {
                values.extend([vector[0], vector[1], vector[2], 0.0]);
            }
            values.extend([warm[0], warm[1], warm[2], 30.0]);
            values.extend([cool[0], cool[1], cool[2], 7.0]);
            values.extend([1.0, 0.80, 0.55, 0.0]);
            values.extend([0.30, 0.50, 1.0, 0.0]);
            view_upload.write(0, &common::bytes(&values))?;

            let mut encoder = device.begin_commands()?;
            encoder.copy_buffer(&view_upload, 0, &view, 0, VIEW_SIZE)?;
            encoder.transition_image(frame.target, ResourceState::UnorderedAccess)?;
            encoder.trace_rays(
                &TraceRaysDescriptor::new(&pipeline, frame.width, frame.height).with_sets(&[set]),
            )?;
            encoder.transition_image(frame.target, ResourceState::CopySource)?;
            encoder.finish()
        },
    )
}
