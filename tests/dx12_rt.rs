//! Acceleration structures, descriptor sets and a procedural ray-tracing
//! pipeline on Direct3D 12.

#![cfg(all(feature = "dx12", windows))]

use erhi::{
    Aabb, AccelerationStructureInstance, AccelerationStructureKind, AdapterPreference, Backend,
    BufferBinding, BufferDescriptor, BufferMemory, BufferUsages, DescriptorBinding,
    DescriptorSetLayoutDescriptor, DescriptorType, Device, DeviceOptions, HitGroup,
    ImageDescriptor, ImageFormat, ImageUsages, PipelineLayoutDescriptor,
    RayTracingPipelineDescriptor, ResourceState, RhiError, ShaderKind, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

/// Nine bindings of mixed types, to exercise descriptor sets.
const TEST_BINDINGS: [DescriptorBinding; 9] = [
    DescriptorBinding::new(
        0,
        DescriptorType::AccelerationStructure,
        ShaderStages::RAY_GENERATION,
    ),
    DescriptorBinding::new(
        1,
        DescriptorType::StorageImage,
        ShaderStages::RAY_GENERATION,
    ),
    DescriptorBinding::new(
        2,
        DescriptorType::ReadOnlyStorageBuffer,
        ShaderStages::INTERSECTION
            .union(ShaderStages::CLOSEST_HIT)
            .union(ShaderStages::RAY_GENERATION),
    ),
    DescriptorBinding::new(
        3,
        DescriptorType::ReadOnlyStorageBuffer,
        ShaderStages::INTERSECTION.union(ShaderStages::RAY_GENERATION),
    ),
    DescriptorBinding::new(
        4,
        DescriptorType::ReadOnlyStorageBuffer,
        ShaderStages::INTERSECTION
            .union(ShaderStages::CLOSEST_HIT)
            .union(ShaderStages::RAY_GENERATION),
    ),
    DescriptorBinding::new(
        5,
        DescriptorType::ReadOnlyStorageBuffer,
        ShaderStages::RAY_GENERATION.union(ShaderStages::INTERSECTION),
    ),
    DescriptorBinding::new(
        6,
        DescriptorType::UniformBuffer,
        ShaderStages::RAY_GENERATION,
    ),
    // Emissive radiance target: the production bloom source.
    DescriptorBinding::new(
        7,
        DescriptorType::StorageImage,
        ShaderStages::RAY_GENERATION,
    ),
    // A buffer two stages read.
    DescriptorBinding::new(
        8,
        DescriptorType::ReadOnlyStorageBuffer,
        ShaderStages::RAY_GENERATION.union(ShaderStages::CLOSEST_HIT),
    ),
];

const TEST_SETS: [DescriptorSetLayoutDescriptor<'static>; 1] =
    [DescriptorSetLayoutDescriptor::new(0, &TEST_BINDINGS)];

const fn test_ray_tracing_pipeline_layout() -> PipelineLayoutDescriptor<'static> {
    PipelineLayoutDescriptor::new(&TEST_SETS)
}

#[test]
#[ignore = "requires a DXR-capable Direct3D 12 adapter"]
fn creates_a_dxr_device_and_direct_queue() {
    let device = Device::open(DeviceOptions {
        backend: Backend::Dx12,
        adapter_preference: AdapterPreference::HighPerformance,
        ..DeviceOptions::default()
    })
    .expect("create a Direct3D 12 DXR device");
    assert_eq!(device.backend(), Backend::Dx12);
    assert!(!device.capabilities().device_name.is_empty());
    println!("adapter: {}", device.capabilities().device_name);
    println!("capabilities: {:#?}", device.capabilities());

    let pipeline_layout = device
        .create_pipeline_layout(test_ray_tracing_pipeline_layout())
        .expect("create DX12 procedural global root signature");
    assert_eq!(pipeline_layout.backend(), Backend::Dx12);
    assert_eq!(pipeline_layout.resource_descriptor_count(), 9);
    assert_eq!(pipeline_layout.sampler_descriptor_count(), 0);

    let encoder = device.begin_commands().expect("begin DX12 commands");
    assert_eq!(encoder.backend(), Backend::Dx12);
    encoder.finish().expect("submit empty DX12 command list");

    let payload = b"erhi native DX12 transfer";
    let upload = device
        .create_buffer(BufferDescriptor::new(
            payload.len() as u64,
            BufferUsages::COPY_SOURCE,
            BufferMemory::Upload,
        ))
        .expect("create DX12 upload buffer");
    let readback = device
        .create_buffer(BufferDescriptor::new(
            payload.len() as u64,
            BufferUsages::COPY_DESTINATION,
            BufferMemory::Readback,
        ))
        .expect("create DX12 readback buffer");
    upload.write(0, payload).expect("write DX12 upload buffer");
    let mut encoder = device.begin_commands().expect("begin DX12 transfer");
    encoder
        .copy_buffer(&upload, 0, &readback, 0, payload.len() as u64)
        .expect("record DX12 buffer copy");
    encoder.finish().expect("submit DX12 buffer copy");
    assert_eq!(
        readback
            .read(0, payload.len() as u64)
            .expect("read DX12 buffer"),
        payload
    );

    let blas = device
        .build_procedural_aabb_blas(&[Aabb::new([0.0; 3], [64.0; 3]).unwrap()])
        .expect("build DX12 procedural AABB BLAS");
    assert_eq!(blas.backend(), Backend::Dx12);
    assert_eq!(blas.kind(), AccelerationStructureKind::BottomLevel);
    assert!(blas.report().acceleration_structure_size > 0);
    assert!(blas.report().build_scratch_size > 0);
    assert_ne!(blas.device_address(), 0);
    println!("BLAS: {:#?}", blas.report());

    let mut instance = AccelerationStructureInstance::new(&blas);
    instance.transform[3] = 64.0;
    instance.custom_index = 7;
    let tlas = device
        .build_tlas(&[instance])
        .expect("build DX12 one-instance TLAS");
    assert_eq!(tlas.backend(), Backend::Dx12);
    assert_eq!(tlas.kind(), AccelerationStructureKind::TopLevel);
    assert!(tlas.report().acceleration_structure_size > 0);
    assert!(tlas.report().build_scratch_size > 0);
    assert_ne!(tlas.device_address(), 0);
    println!("TLAS: {:#?}", tlas.report());
    drop(blas);
    assert_ne!(tlas.device_address(), 0, "TLAS retains its BLAS dependency");

    let image = device
        .create_image(ImageDescriptor::d2(
            ImageFormat::Rgba16Float,
            64,
            64,
            ImageUsages::STORAGE | ImageUsages::COPY_SOURCE | ImageUsages::COPY_DESTINATION,
        ))
        .expect("create DX12 storage image");
    assert_eq!(image.backend(), Backend::Dx12);
    let _view = image.create_full_view().expect("create DX12 image view");

    let layout = image.copy_layout(0).expect("query DX12 image copy layout");
    assert_eq!(layout.row_pitch, 64 * 8, "64 texels of RGBA16F per row");
    let pattern: Vec<u8> = (0..layout.total_size).map(|i| (i % 251) as u8).collect();
    let image_upload = device
        .create_buffer(BufferDescriptor::new(
            layout.total_size,
            BufferUsages::COPY_SOURCE,
            BufferMemory::Upload,
        ))
        .expect("create DX12 image upload buffer");
    let image_readback = device
        .create_buffer(BufferDescriptor::new(
            layout.total_size,
            BufferUsages::COPY_DESTINATION,
            BufferMemory::Readback,
        ))
        .expect("create DX12 image readback buffer");
    image_upload
        .write(0, &pattern)
        .expect("write DX12 image pattern");

    let mut encoder = device.begin_commands().expect("begin DX12 image commands");
    encoder
        .transition_image(&image, ResourceState::CopyDest)
        .expect("transition DX12 image to transfer write");
    // A misaligned buffer offset never reaches the driver.
    assert!(matches!(
        encoder.copy_buffer_to_image(&image_upload, 4, &image, 0),
        Err(RhiError::UnalignedCopyOffset { required: 512, .. })
    ));
    encoder
        .copy_buffer_to_image(&image_upload, 0, &image, 0)
        .expect("record DX12 buffer-to-image copy");
    encoder
        .transition_image(&image, ResourceState::CopySource)
        .expect("transition DX12 image to transfer read");
    encoder
        .copy_image_to_buffer(&image, 0, &image_readback, 0)
        .expect("record DX12 image-to-buffer copy");
    encoder.finish().expect("submit DX12 image round trip");
    assert_eq!(
        image_readback
            .read(0, layout.total_size)
            .expect("read DX12 image bytes"),
        pattern,
        "DX12 image round trip must preserve the deterministic pattern"
    );

    let mut set = device
        .create_descriptor_set(&pipeline_layout, 0)
        .expect("create DX12 descriptor set");
    assert_eq!(set.backend(), Backend::Dx12);
    let view = image.create_full_view().expect("recreate DX12 image view");
    let storage = |size: u64| {
        device
            .create_buffer(BufferDescriptor::new(
                size,
                BufferUsages::STORAGE | BufferUsages::COPY_DESTINATION,
                BufferMemory::Device,
            ))
            .expect("create DX12 storage buffer")
    };
    let nodes = storage(4096);
    let samples = storage(8192);
    let metadata = storage(256);
    let materials = storage(512);
    let camera = device
        .create_buffer(BufferDescriptor::new(
            256,
            BufferUsages::UNIFORM,
            BufferMemory::Upload,
        ))
        .expect("create DX12 camera buffer");

    // The full seven-binding procedural contract, written with real resources.
    set.write_acceleration_structure(0, &tlas)
        .expect("write DX12 TLAS descriptor");
    set.write_storage_image(1, &view)
        .expect("write DX12 storage-image descriptor");
    set.write_storage_buffer(2, BufferBinding::range(&nodes, 0, 4096).with_stride(16))
        .expect("write DX12 node buffer descriptor");
    set.write_storage_buffer(3, BufferBinding::range(&samples, 0, 8192).with_stride(4))
        .expect("write DX12 procedural buffer descriptor");
    set.write_storage_buffer(4, BufferBinding::range(&metadata, 0, 256).with_stride(16))
        .expect("write DX12 metadata buffer descriptor");
    set.write_storage_buffer(5, BufferBinding::range(&materials, 0, 480).with_stride(48))
        .expect("write DX12 material buffer descriptor");
    set.write_uniform_buffer(6, BufferBinding::range(&camera, 0, 256))
        .expect("write DX12 camera descriptor");

    // Every failure below must reject before any native descriptor update.
    assert!(matches!(
        set.write_storage_buffer(9, BufferBinding::range(&nodes, 0, 4096).with_stride(16)),
        Err(RhiError::InvalidDescriptorBinding {
            reason: "binding does not exist in this set",
            ..
        })
    ));
    assert!(matches!(
        set.write_uniform_buffer(2, BufferBinding::range(&camera, 0, 256)),
        Err(RhiError::InvalidDescriptorBinding {
            reason: "binding type does not accept this write",
            ..
        })
    ));
    assert!(matches!(
        set.write_uniform_buffer(6, BufferBinding::range(&camera, 128, 64)),
        Err(RhiError::InvalidDescriptorBinding {
            reason: "uniform-buffer ranges must be non-empty with a 256-byte-aligned offset",
            ..
        })
    ));
    assert!(matches!(
        set.write_storage_buffer(2, BufferBinding::range(&nodes, 0, 8192).with_stride(16)),
        Err(RhiError::BufferRangeOutOfBounds { .. })
    ));
    assert!(matches!(
        set.write_storage_buffer(2, BufferBinding::range(&camera, 0, 256).with_stride(16)),
        Err(RhiError::MissingBufferUsage { .. })
    ));

    // A ray-tracing library, from the checked-in DXIL.
    let module = device
        .create_shader_module(&ShaderModuleDescriptor::new(
            "procedural_rt",
            ShaderKind::RayTracingLibrary,
            ShaderSource::Dxil(include_bytes!("fixtures/procedural.dxil")),
        ))
        .expect("create DX12 shader module from checked-in DXIL");
    assert_eq!(module.backend(), Backend::Dx12);

    // One compute entry point.
    let compute = device
        .create_shader_module(&ShaderModuleDescriptor::new(
            "tone_map",
            ShaderKind::Compute { entry: "main" },
            ShaderSource::Dxil(include_bytes!("fixtures/compute.main.dxil")),
        ))
        .expect("create DX12 shader module from checked-in compute DXIL");
    assert_eq!(compute.backend(), Backend::Dx12);

    // The wrong blob kind is rejected before any native call.
    assert!(matches!(
        device.create_shader_module(&ShaderModuleDescriptor::new(
            "wrong_blob",
            ShaderKind::RayTracingLibrary,
            ShaderSource::SpirV(include_bytes!("fixtures/procedural.spv"))
        )),
        Err(RhiError::ShaderSourceUnsupported {
            backend: "dx12",
            ..
        })
    ));

    // Full DXR state object from the checked-in Slang-built DXIL library,
    // with shader identifiers proving every export resolved.
    let miss_entries = [module.entry("primary_miss"), module.entry("shadow_miss")];
    let hit_groups = [HitGroup::new("procedural_hit")
        .with_intersection(module.entry("intersection"))
        .with_closest_hit(module.entry("closest_hit"))];
    let rt_descriptor = RayTracingPipelineDescriptor::new(
        "procedural_rt",
        &pipeline_layout,
        module.entry("ray_gen"),
        28,
    )
    .with_miss_entries(&miss_entries)
    .with_hit_groups(&hit_groups)
    .with_max_attribute_size(20);
    let rt_pipeline = device
        .create_ray_tracing_pipeline(&rt_descriptor)
        .expect("create DX12 DXR state object and shader binding table");
    assert_eq!(rt_pipeline.backend(), Backend::Dx12);

    // A missing export surfaces as a concrete error from identifier lookup.
    assert!(matches!(
        device.create_ray_tracing_pipeline(&{
            let mut made = rt_descriptor;
            made.ray_generation = module.entry("missing_entry");
            made
        }),
        Err(RhiError::MissingShaderExport { .. }) | Err(RhiError::Dx12 { .. })
    ));
    // Zero attribute size never reaches native creation.
    assert!(matches!(
        device.create_ray_tracing_pipeline(&rt_descriptor.with_max_attribute_size(0)),
        Err(RhiError::InvalidRayTracingPipeline { .. })
    ));
}
