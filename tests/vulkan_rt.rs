//! Acceleration structures, descriptor sets and a procedural ray-tracing
//! pipeline on Vulkan.

#![cfg(feature = "vulkan")]
use erhi::{
    Aabb, AccelerationStructureInstance, AccelerationStructureKind, Backend, BufferBinding,
    BufferDescriptor, BufferMemory, BufferUsages, DescriptorBinding, DescriptorSetLayoutDescriptor,
    DescriptorType, Device, DeviceOptions, HitGroup, ImageDescriptor, ImageFormat, ImageUsages,
    PipelineLayoutDescriptor, RayTracingPipelineDescriptor, ResourceState, RhiError, ShaderKind,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
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

/// Explicit hardware test: ordinary workspace tests must remain runnable on
/// machines without a Vulkan RT adapter.
#[test]
#[ignore = "requires an RT-capable Vulkan 1.2+ device"]
fn builds_a_procedural_blas() -> Result<(), erhi::RhiError> {
    let device = Device::open(DeviceOptions {
        backend: Backend::Vulkan,
        ..DeviceOptions::default()
    })?;
    let pipeline_layout = device.create_pipeline_layout(test_ray_tracing_pipeline_layout())?;
    assert_eq!(pipeline_layout.backend(), Backend::Vulkan);
    assert_eq!(pipeline_layout.resource_descriptor_count(), 9);
    assert_eq!(pipeline_layout.sampler_descriptor_count(), 0);
    let blas = device.build_procedural_aabb_blas(&[Aabb::new([0.0; 3], [64.0; 3]).unwrap()])?;
    eprintln!("adapter: {}", device.capabilities().device_name);
    eprintln!("capabilities: {:#?}", device.capabilities());
    eprintln!("BLAS: {:#?}", blas.report());
    assert_eq!(blas.backend(), Backend::Vulkan);
    assert_eq!(blas.kind(), AccelerationStructureKind::BottomLevel);
    assert!(blas.report().acceleration_structure_size > 0);
    assert!(blas.report().build_scratch_size > 0);
    assert_ne!(blas.device_address(), 0);

    let mut instance = AccelerationStructureInstance::new(&blas);
    instance.transform[3] = 64.0;
    instance.custom_index = 7;
    let tlas = device.build_tlas(&[instance])?;
    eprintln!("TLAS: {:#?}", tlas.report());
    assert_eq!(tlas.backend(), Backend::Vulkan);
    assert_eq!(tlas.kind(), AccelerationStructureKind::TopLevel);
    assert!(tlas.report().acceleration_structure_size > 0);
    assert!(tlas.report().build_scratch_size > 0);
    assert_ne!(tlas.device_address(), 0);
    drop(blas);
    assert_ne!(tlas.device_address(), 0, "TLAS retains its BLAS dependency");
    Ok(())
}

#[test]
#[ignore = "requires an RT-capable Vulkan 1.2+ device"]
fn a_device_copies_a_vulkan_buffer() -> Result<(), erhi::RhiError> {
    let device = Device::open(DeviceOptions {
        backend: Backend::Vulkan,
        ..DeviceOptions::default()
    })?;
    let payload = b"erhi native Vulkan transfer";
    let upload = device.create_buffer(BufferDescriptor::new(
        payload.len() as u64,
        BufferUsages::COPY_SOURCE,
        BufferMemory::Upload,
    ))?;
    upload.write(0, payload)?;
    let readback = device.create_buffer(BufferDescriptor::new(
        payload.len() as u64,
        BufferUsages::COPY_DESTINATION,
        BufferMemory::Readback,
    ))?;
    let mut encoder = device.begin_commands()?;
    encoder.copy_buffer(&upload, 0, &readback, 0, payload.len() as u64)?;
    encoder.finish()?;
    assert_eq!(readback.read(0, payload.len() as u64)?, payload);
    Ok(())
}

#[test]
#[ignore = "requires an RT-capable Vulkan 1.2+ device"]
fn vulkan_image_copies_reject_bad_states_offsets_and_devices() -> Result<(), erhi::RhiError> {
    let device = Device::open(DeviceOptions {
        backend: Backend::Vulkan,
        ..DeviceOptions::default()
    })?;
    let image = device.create_image(ImageDescriptor::d2(
        ImageFormat::Rgba16Float,
        64,
        64,
        ImageUsages::COPY_SOURCE | ImageUsages::COPY_DESTINATION,
    ))?;
    let layout = image.copy_layout(0)?;
    let upload = device.create_buffer(BufferDescriptor::new(
        layout.total_size + 1024,
        BufferUsages::COPY_SOURCE,
        BufferMemory::Upload,
    ))?;

    let mut encoder = device.begin_commands()?;
    encoder.transition_image(&image, ResourceState::CopyDest)?;
    // A misaligned buffer offset never reaches the driver.
    assert!(matches!(
        encoder.copy_buffer_to_image(&upload, 4, &image, 0),
        Err(RhiError::UnalignedCopyOffset { required: 512, .. })
    ));
    // An out-of-range mip never reaches the driver.
    assert!(matches!(
        encoder.copy_buffer_to_image(&upload, 0, &image, 1),
        Err(RhiError::ImageMipOutOfRange { .. })
    ));
    drop(encoder);

    // A second device rejects the first device's image before recording.
    let other = Device::open(DeviceOptions {
        backend: Backend::Vulkan,
        ..DeviceOptions::default()
    })?;
    let mut foreign = other.begin_commands()?;
    assert!(matches!(
        foreign.transition_image(&image, ResourceState::CopyDest),
        Err(RhiError::ResourceDeviceMismatch)
    ));
    drop(foreign);
    Ok(())
}

#[test]
#[ignore = "requires an RT-capable Vulkan 1.2+ device"]
fn writes_the_test_descriptor_set_on_vulkan() -> Result<(), erhi::RhiError> {
    let device = Device::open(DeviceOptions {
        backend: Backend::Vulkan,
        ..DeviceOptions::default()
    })?;
    let layout = device.create_pipeline_layout(test_ray_tracing_pipeline_layout())?;
    let mut set = device.create_descriptor_set(&layout, 0)?;
    assert_eq!(set.backend(), Backend::Vulkan);

    let blas = device.build_procedural_aabb_blas(&[Aabb::new([0.0; 3], [64.0; 3]).unwrap()])?;
    let tlas = device.build_tlas(&[AccelerationStructureInstance::new(&blas)])?;
    let image = device.create_image(ImageDescriptor::d2(
        ImageFormat::Rgba16Float,
        64,
        64,
        ImageUsages::STORAGE | ImageUsages::COPY_SOURCE,
    ))?;
    let view = image.create_full_view()?;
    let storage = |size: u64| {
        device.create_buffer(BufferDescriptor::new(
            size,
            BufferUsages::STORAGE | BufferUsages::COPY_DESTINATION,
            BufferMemory::Device,
        ))
    };
    let nodes = storage(4096)?;
    let samples = storage(8192)?;
    let chunks = storage(256)?;
    let materials = storage(512)?;
    let camera = device.create_buffer(BufferDescriptor::new(
        256,
        BufferUsages::UNIFORM,
        BufferMemory::Upload,
    ))?;
    let emissive = device.create_image(ImageDescriptor::d2(
        ImageFormat::Rgba16Float,
        64,
        64,
        ImageUsages::STORAGE | ImageUsages::COPY_SOURCE,
    ))?;
    let emissive_view = emissive.create_full_view()?;
    let lookup = storage(64)?;

    // The full nine-binding procedural contract, written with real resources.
    set.write_acceleration_structure(0, &tlas)?;
    set.write_storage_image(1, &view)?;
    set.write_storage_buffer(2, BufferBinding::range(&nodes, 0, 4096).with_stride(16))?;
    set.write_storage_buffer(3, BufferBinding::range(&samples, 0, 8192).with_stride(4))?;
    set.write_storage_buffer(4, BufferBinding::range(&chunks, 0, 256).with_stride(16))?;
    set.write_storage_buffer(5, BufferBinding::range(&materials, 0, 480).with_stride(48))?;
    set.write_uniform_buffer(6, BufferBinding::range(&camera, 0, 256))?;
    set.write_storage_image(7, &emissive_view)?;
    set.write_storage_buffer(8, BufferBinding::range(&lookup, 0, 64).with_stride(4))?;

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
        set.write_acceleration_structure(0, &blas),
        Err(RhiError::InvalidDescriptorBinding {
            reason: "acceleration-structure descriptors require a top-level structure",
            ..
        })
    ));
    assert!(matches!(
        set.write_storage_buffer(2, BufferBinding::range(&nodes, 2, 8).with_stride(4)),
        Err(RhiError::InvalidDescriptorBinding {
            reason: "storage-buffer ranges must be non-empty multiples of a four-byte-aligned element stride",
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

    let other = Device::open(DeviceOptions {
        backend: Backend::Vulkan,
        ..DeviceOptions::default()
    })?;
    let foreign = other.create_buffer(BufferDescriptor::new(
        4096,
        BufferUsages::STORAGE,
        BufferMemory::Device,
    ))?;
    assert!(matches!(
        set.write_storage_buffer(2, BufferBinding::range(&foreign, 0, 4096).with_stride(16)),
        Err(RhiError::ResourceDeviceMismatch)
    ));
    Ok(())
}

#[test]
#[ignore = "requires an RT-capable Vulkan 1.2+ device"]
fn creates_vulkan_shader_modules_from_source_and_blobs() -> Result<(), erhi::RhiError> {
    let device = Device::open(DeviceOptions {
        backend: Backend::Vulkan,
        ..DeviceOptions::default()
    })?;

    // A ray-tracing library, from the checked-in SPIR-V.
    let module = device.create_shader_module(&ShaderModuleDescriptor::new(
        "procedural_rt",
        ShaderKind::RayTracingLibrary,
        ShaderSource::SpirV(include_bytes!("fixtures/procedural.spv")),
    ))?;
    assert_eq!(module.backend(), Backend::Vulkan);

    // One compute entry point.
    let compute = device.create_shader_module(&ShaderModuleDescriptor::new(
        "tone_map",
        ShaderKind::Compute { entry: "main" },
        ShaderSource::SpirV(include_bytes!("fixtures/compute.main.spv")),
    ))?;
    assert_eq!(compute.backend(), Backend::Vulkan);

    // Bytes that are no SPIR-V, and the other backend's bytes, are refused
    // before any native call.
    let error = device
        .create_shader_module(&ShaderModuleDescriptor::new(
            "broken",
            ShaderKind::Compute { entry: "main" },
            ShaderSource::SpirV(b"this is not SPIR-V"),
        ))
        .expect_err("bytes that are no SPIR-V must be refused");
    assert!(matches!(error, RhiError::InvalidShaderBlob { .. }));
    assert!(matches!(
        device.create_shader_module(&ShaderModuleDescriptor::new(
            "wrong_blob",
            ShaderKind::RayTracingLibrary,
            ShaderSource::Dxil(include_bytes!("fixtures/procedural.dxil"))
        )),
        Err(RhiError::ShaderSourceUnsupported {
            backend: "vulkan",
            ..
        })
    ));
    Ok(())
}

#[test]
#[ignore = "requires an RT-capable Vulkan 1.2+ device"]
fn creates_the_procedural_ray_tracing_pipeline_on_vulkan() -> Result<(), erhi::RhiError> {
    let device = Device::open(DeviceOptions {
        backend: Backend::Vulkan,
        ..DeviceOptions::default()
    })?;
    let layout = device.create_pipeline_layout(test_ray_tracing_pipeline_layout())?;
    let module = device.create_shader_module(&ShaderModuleDescriptor::new(
        "procedural_rt",
        ShaderKind::RayTracingLibrary,
        ShaderSource::SpirV(include_bytes!("fixtures/procedural.spv")),
    ))?;
    let miss_entries = [module.entry("primary_miss"), module.entry("shadow_miss")];
    let hit_groups = [HitGroup::new("procedural_hit")
        .with_intersection(module.entry("intersection"))
        .with_closest_hit(module.entry("closest_hit"))];
    let descriptor =
        RayTracingPipelineDescriptor::new("procedural_rt", &layout, module.entry("ray_gen"), 20)
            .with_miss_entries(&miss_entries)
            .with_hit_groups(&hit_groups)
            .with_max_attribute_size(16);
    let pipeline = device.create_ray_tracing_pipeline(&descriptor)?;
    assert_eq!(pipeline.backend(), Backend::Vulkan);

    // Rejections must occur before native creation.
    assert!(matches!(
        device.create_ray_tracing_pipeline(&{
            let mut made = descriptor;
            made.max_payload_size = 0;
            made
        }),
        Err(RhiError::InvalidRayTracingPipeline { .. })
    ));
    assert!(matches!(
        device.create_ray_tracing_pipeline(&descriptor.with_max_recursion_depth(0)),
        Err(RhiError::InvalidRayTracingPipeline { .. })
    ));
    assert!(matches!(
        device.create_ray_tracing_pipeline(
            &descriptor.with_hit_groups(&[
                HitGroup::new("procedural_hit")
                    .with_intersection(module.entry("intersection"))
                    .with_closest_hit(module.entry("closest_hit")),
                HitGroup::new("procedural_hit").with_intersection(module.entry("intersection")),
            ])
        ),
        Err(RhiError::InvalidRayTracingPipeline {
            reason: "hit group names must be unique",
        })
    ));
    // The SPIR-V entry list is known, so a bogus export fails immediately.
    assert!(matches!(
        device.create_ray_tracing_pipeline(&{
            let mut made = descriptor;
            made.ray_generation = module.entry("missing_entry");
            made
        }),
        Err(RhiError::MissingShaderExport { .. })
    ));
    Ok(())
}
