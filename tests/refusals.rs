//! Validation errors: each misuse here is one a caller could make and
//! both graphics APIs leave undefined. It must come back as an error that
//! names the mistake, before anything reaches a driver; the debug layers
//! then see nothing.

mod fixtures;

use erhi::{
    Backend, BufferBinding, BufferDescriptor, BufferMemory, BufferUsages, ColorAttachment,
    ComputePipelineDescriptor, DescriptorBinding, DescriptorSetLayoutDescriptor, DescriptorType,
    Device, DeviceOptions, DispatchDescriptor, ImageDescriptor, ImageDimension, ImageFormat,
    ImageUsages, LoadOp, MAX_DISPATCH_GROUPS, MAX_IMAGE_EXTENT, PipelineLayout,
    PipelineLayoutDescriptor, RenderPassDescriptor, ResourceState, RhiError, ShaderKind,
    ShaderStages,
};

fn image(usages: ImageUsages) -> ImageDescriptor {
    ImageDescriptor::d2(ImageFormat::Rgba8Unorm, 64, 64, usages)
}

/// A layout of one set with one binding of `kind` at `binding`.
fn layout(device: &Device, binding: u32, kind: DescriptorType) -> Result<PipelineLayout, RhiError> {
    device.create_pipeline_layout(PipelineLayoutDescriptor::new(&[
        DescriptorSetLayoutDescriptor::new(
            0,
            &[DescriptorBinding::new(binding, kind, ShaderStages::COMPUTE)],
        ),
    ]))
}

fn images_that_cannot_be(device: &Device) {
    let refused = |descriptor: ImageDescriptor| device.create_image(descriptor).err();
    // Larger than either API accepts.
    assert!(matches!(
        refused({
            let mut made = image(ImageUsages::SAMPLED);
            made.width = MAX_IMAGE_EXTENT + 1;
            made
        }),
        Some(RhiError::InvalidImageDimensions { .. })
    ));
    // Usages its format cannot serve.
    for descriptor in [
        image(ImageUsages::DEPTH_TARGET),
        {
            let mut made = image(ImageUsages::RENDER_TARGET);
            made.format = ImageFormat::Depth32Float;
            made
        },
        {
            let mut made = image(ImageUsages::DEPTH_TARGET | ImageUsages::STORAGE);
            made.format = ImageFormat::Depth32Float;
            made
        },
        {
            let mut made = image(ImageUsages::DEPTH_TARGET);
            made.format = ImageFormat::Depth32Float;
            made.dimension = ImageDimension::D3;
            made
        },
    ] {
        assert!(
            matches!(
                refused(descriptor),
                Some(RhiError::InvalidImageUsage { .. })
            ),
            "{descriptor:?}"
        );
    }
}

fn copies_that_are_not_defined(device: &Device) -> Result<(), RhiError> {
    let buffer = device.create_buffer(BufferDescriptor::new(
        4096,
        BufferUsages::COPY_SOURCE | BufferUsages::COPY_DESTINATION,
        BufferMemory::Device,
    ))?;
    let twin = device.create_buffer(buffer.descriptor())?;
    let target = device.create_image(image(
        ImageUsages::RENDER_TARGET | ImageUsages::COPY_SOURCE | ImageUsages::COPY_DESTINATION,
    ))?;
    let other = device.create_image(target.descriptor())?;
    let view = target.create_full_view()?;

    let mut encoder = device.begin_commands()?;
    // Within one buffer, whether the ranges meet or stand apart.
    for destination in [512, 2048] {
        assert!(matches!(
            encoder.copy_buffer(&buffer, 0, &buffer, destination, 1024),
            Err(RhiError::InvalidCopy { .. })
        ));
    }
    encoder.copy_buffer(&buffer, 0, &twin, 0, 1024)?;
    // An image onto itself.
    encoder.transition_image(&target, ResourceState::CopySource)?;
    assert!(matches!(
        encoder.copy_image(&target, &target),
        Err(RhiError::InvalidCopy { .. })
    ));
    // No copy of an image while a pass is open.
    encoder.transition_image(&target, ResourceState::RenderTarget)?;
    encoder.transition_image(&other, ResourceState::CopySource)?;
    encoder.begin_render_pass(&RenderPassDescriptor::color(&view, LoadOp::Clear([0.0; 4])))?;
    assert!(matches!(
        encoder.copy_image_to_buffer(&other, 0, &buffer, 0),
        Err(RhiError::InvalidRenderPass { .. })
    ));
    assert!(matches!(
        encoder.copy_buffer_to_image(&buffer, 0, &other, 0),
        Err(RhiError::InvalidRenderPass { .. })
    ));
    assert!(matches!(
        encoder.copy_image(&other, &target),
        Err(RhiError::InvalidRenderPass { .. })
    ));
    encoder.end_render_pass()?;
    // One image is one attachment of a pass at the most.
    assert!(matches!(
        encoder.begin_render_pass(
            &RenderPassDescriptor::new()
                .with_color(ColorAttachment::new(&view, LoadOp::Load))
                .with_color(ColorAttachment::new(&view, LoadOp::Load))
        ),
        Err(RhiError::InvalidRenderPass { .. })
    ));
    encoder.finish()
}

fn sets_that_do_not_fit(device: &Device) -> Result<(), RhiError> {
    // A combined image sampler is one descriptor, never several.
    assert!(matches!(
        device.create_pipeline_layout(PipelineLayoutDescriptor::new(&[
            DescriptorSetLayoutDescriptor::new(
                0,
                &[DescriptorBinding::new(
                    0,
                    DescriptorType::CombinedImageSampler(erhi::StaticSampler::linear_clamp(),),
                    ShaderStages::COMPUTE
                )
                .with_count(4)]
            )
        ])),
        Err(RhiError::InvalidDescriptorBinding { set: 0, .. })
    ));

    let counting = layout(device, 0, DescriptorType::StorageBuffer)?;
    let module = fixtures::module(
        device,
        "count",
        "count",
        ShaderKind::Compute { entry: "main" },
    )?;
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor::new(
        "count",
        &counting,
        module.entry("main"),
    ))?;
    let counter = device.create_buffer(BufferDescriptor::new(
        256,
        BufferUsages::STORAGE,
        BufferMemory::Device,
    ))?;

    let mut encoder = device.begin_commands()?;
    // A set whose binding was never written.
    let mut set = device.create_descriptor_set(&counting, 0)?;
    assert!(matches!(
        encoder.dispatch(
            &DispatchDescriptor::new(&pipeline)
                .with_sets(&[&set])
                .with_groups([1, 1, 1])
        ),
        Err(RhiError::InvalidDescriptorBinding { binding: 0, .. })
    ));
    set.write_storage_buffer(0, BufferBinding::range(&counter, 0, 256).with_stride(4))?;
    // More groups than either API accepts along an axis.
    assert!(matches!(
        encoder.dispatch(
            &DispatchDescriptor::new(&pipeline)
                .with_sets(&[&set])
                .with_groups([MAX_DISPATCH_GROUPS + 1, 1, 1])
        ),
        Err(RhiError::InvalidComputePipeline { .. })
    ));
    // A set made for a layout that declares its bindings otherwise.
    let elsewhere = layout(device, 3, DescriptorType::StorageBuffer)?;
    let mut stranger = device.create_descriptor_set(&elsewhere, 0)?;
    stranger.write_storage_buffer(3, BufferBinding::range(&counter, 0, 256).with_stride(4))?;
    assert!(matches!(
        encoder.dispatch(
            &DispatchDescriptor::new(&pipeline)
                .with_sets(&[&stranger])
                .with_groups([1, 1, 1])
        ),
        Err(RhiError::DescriptorSetLayoutMismatch { set: 0 })
    ));
    // One made for another layout that declares the same is as good.
    let twin = layout(device, 0, DescriptorType::StorageBuffer)?;
    let mut alike = device.create_descriptor_set(&twin, 0)?;
    alike.write_storage_buffer(0, BufferBinding::range(&counter, 0, 256).with_stride(4))?;
    encoder.dispatch(
        &DispatchDescriptor::new(&pipeline)
            .with_sets(&[&alike])
            .with_groups([1, 1, 1]),
    )?;
    encoder.dispatch(
        &DispatchDescriptor::new(&pipeline)
            .with_sets(&[&set])
            .with_groups([1, 1, 1]),
    )?;
    encoder.finish()
}

fn run(backend: Backend) -> Result<(), RhiError> {
    let device = Device::open(DeviceOptions {
        backend,
        debug: true,
        require_ray_tracing: false,
        ..DeviceOptions::default()
    })?;
    images_that_cannot_be(&device);
    copies_that_are_not_defined(&device)?;
    sets_that_do_not_fit(&device)?;
    device.wait_idle()?;
    // Nothing that was refused reached the driver, and what was let
    // through was right.
    eprintln!("{:#?}", device.debug_messages());
    assert_eq!(device.debug_error_count().unwrap_or(0), 0, "{backend:?}");
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan device (and a DX12 adapter when dx12 is enabled)"]
fn misuse_is_refused_by_name_on_every_backend() {
    run(Backend::Vulkan).expect("Vulkan refusals");
    #[cfg(all(feature = "dx12", windows))]
    run(Backend::Dx12).expect("DX12 refusals");
}
