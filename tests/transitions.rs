//! No test here sets an image's state explicitly: the encoder transitions each
//! where the command that uses it needs it. Copied into, sampled, written
//! as a storage image, sampled again, drawn into and copied out of, with
//! not one `transition_image`, and the debug layers have nothing to say.
//! Buffers likewise: copied into, written by one dispatch and another, read
//! by a third, copied out of, with not one `transition_buffer`. And one mip
//! of an image is sampled while the next is written.

mod fixtures;

use erhi::{
    Backend, Buffer, BufferBinding, BufferDescriptor, BufferMemory, BufferUsages, ComputePipeline,
    ComputePipelineDescriptor, DescriptorBinding, DescriptorSetLayoutDescriptor, DescriptorType,
    Device, DeviceOptions, DispatchDescriptor, Image, ImageDescriptor, ImageFormat, ImageUsages,
    ImageViewDescriptor, LoadOp, PipelineLayout, PipelineLayoutDescriptor, RenderPassDescriptor,
    ResourceState, RhiError, ShaderKind, ShaderStages, StaticSampler,
};

const EDGE: u32 = 16;

fn image(device: &Device, usages: ImageUsages) -> Result<Image, RhiError> {
    device.create_image(ImageDescriptor::d2(
        ImageFormat::Rgba8Unorm,
        EDGE,
        EDGE,
        usages,
    ))
}

/// A buffer the whole of `image` is read back into.
fn readback(device: &Device, image: &Image) -> Result<Buffer, RhiError> {
    device.create_buffer(BufferDescriptor::new(
        image.copy_layout(0)?.total_size,
        BufferUsages::COPY_DESTINATION,
        BufferMemory::Readback,
    ))
}

/// The first texel of what was read back.
fn first_texel(buffer: &Buffer) -> Result<[u8; 4], RhiError> {
    let bytes = buffer.read(0, 4)?;
    Ok([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn close(actual: [u8; 4], expected: [u8; 4], what: &str) {
    for (a, e) in actual.into_iter().zip(expected) {
        assert!(a.abs_diff(e) <= 2, "{what}: {actual:?}, not {expected:?}");
    }
}

fn first_number(buffer: &Buffer) -> Result<u32, RhiError> {
    let bytes = buffer.read(0, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn buffers(device: &Device) -> Result<(), RhiError> {
    const SIZE: u64 = 256;
    let binding = |binding: u32, descriptor_type: DescriptorType| {
        DescriptorBinding::new(binding, descriptor_type, ShaderStages::COMPUTE)
    };
    let pipeline = |label: &'static str, file: &'static str, bindings: &[DescriptorBinding]| {
        let layout = device.create_pipeline_layout(PipelineLayoutDescriptor::new(&[
            DescriptorSetLayoutDescriptor::new(0, bindings),
        ]))?;
        let module = fixtures::module(device, label, file, ShaderKind::Compute { entry: "main" })?;
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor::new(
            label,
            &layout,
            module.entry("main"),
        ))?;
        Ok::<_, RhiError>((layout, pipeline))
    };
    let (count_layout, count) = pipeline(
        "transitions_count",
        "transitions_count",
        &[binding(0, DescriptorType::StorageBuffer)],
    )?;
    let (sum_layout, sum) = pipeline(
        "transitions_sum",
        "transitions_sum",
        &[
            binding(0, DescriptorType::ReadOnlyStorageBuffer),
            binding(1, DescriptorType::StorageBuffer),
        ],
    )?;
    let held =
        |usages| device.create_buffer(BufferDescriptor::new(SIZE, usages, BufferMemory::Device));
    let counter =
        held(BufferUsages::STORAGE | BufferUsages::COPY_SOURCE | BufferUsages::COPY_DESTINATION)?;
    let total = held(BufferUsages::STORAGE | BufferUsages::COPY_SOURCE)?;
    let upload = device.create_buffer(BufferDescriptor::new(
        SIZE,
        BufferUsages::COPY_SOURCE,
        BufferMemory::Upload,
    ))?;
    upload.write(0, &[0_u8; SIZE as usize])?;
    upload.write(0, &7_u32.to_le_bytes())?;
    let back = || {
        device.create_buffer(BufferDescriptor::new(
            SIZE,
            BufferUsages::COPY_DESTINATION,
            BufferMemory::Readback,
        ))
    };
    let (counted, summed, again) = (back()?, back()?, back()?);
    let mut counting = device.create_descriptor_set(&count_layout, 0)?;
    counting.write_storage_buffer(0, BufferBinding::range(&counter, 0, SIZE).with_stride(4))?;
    let mut summing = device.create_descriptor_set(&sum_layout, 0)?;
    summing.write_storage_buffer(0, BufferBinding::range(&counter, 0, SIZE).with_stride(4))?;
    summing.write_storage_buffer(1, BufferBinding::range(&total, 0, SIZE).with_stride(4))?;

    // In one encoder: copied into, written twice, read, copied out of.
    let mut encoder = device.begin_commands()?;
    encoder.copy_buffer(&upload, 0, &counter, 0, SIZE)?;
    encoder.dispatch(
        &DispatchDescriptor::new(&count)
            .with_sets(&[&counting])
            .with_groups([1, 1, 1]),
    )?;
    encoder.dispatch(
        &DispatchDescriptor::new(&count)
            .with_sets(&[&counting])
            .with_groups([1, 1, 1]),
    )?;
    encoder.dispatch(
        &DispatchDescriptor::new(&sum)
            .with_sets(&[&summing])
            .with_groups([1, 1, 1]),
    )?;
    encoder.copy_buffer(&counter, 0, &counted, 0, SIZE)?;
    encoder.copy_buffer(&total, 0, &summed, 0, SIZE)?;
    encoder.finish()?;
    assert_eq!(first_number(&counted)?, 9, "counted twice from seven");
    assert_eq!(first_number(&summed)?, 109, "read after it was written");

    // Across encoders: read first here, having last been copied from, then
    // written, then read again.
    let mut encoder = device.begin_commands()?;
    encoder.dispatch(
        &DispatchDescriptor::new(&sum)
            .with_sets(&[&summing])
            .with_groups([1, 1, 1]),
    )?;
    encoder.dispatch(
        &DispatchDescriptor::new(&count)
            .with_sets(&[&counting])
            .with_groups([1, 1, 1]),
    )?;
    encoder.dispatch(
        &DispatchDescriptor::new(&sum)
            .with_sets(&[&summing])
            .with_groups([1, 1, 1]),
    )?;
    encoder.copy_buffer(&total, 0, &again, 0, SIZE)?;
    encoder.finish()?;
    assert_eq!(first_number(&again)?, 110, "across encoders");
    Ok(())
}

/// One mip of an image is read while another of the same image is written:
/// each is in its own state.
fn mips(
    device: &Device,
    layout: &PipelineLayout,
    invert: &ComputePipeline,
    upload: &Buffer,
) -> Result<(), RhiError> {
    let chain = device.create_image(
        ImageDescriptor::d2(
            ImageFormat::Rgba8Unorm,
            EDGE,
            EDGE,
            ImageUsages::COPY_DESTINATION
                | ImageUsages::COPY_SOURCE
                | ImageUsages::SAMPLED
                | ImageUsages::STORAGE,
        )
        .with_mip_count(2),
    )?;
    let (large, small) = (
        chain.create_view(&ImageViewDescriptor::subresource(0, 0))?,
        chain.create_view(&ImageViewDescriptor::subresource(1, 0))?,
    );
    let mut down = device.create_descriptor_set(layout, 0)?;
    down.write_combined_image_sampler(0, &large)?;
    down.write_storage_image(1, &small)?;
    // The same mip to be sampled and to be written: nothing serves that.
    let mut both = device.create_descriptor_set(layout, 0)?;
    both.write_combined_image_sampler(0, &large)?;
    both.write_storage_image(1, &large)?;
    let read = || {
        device.create_buffer(BufferDescriptor::new(
            chain.copy_layout(1)?.total_size,
            BufferUsages::COPY_DESTINATION,
            BufferMemory::Readback,
        ))
    };
    let (once, whole) = (read()?, read()?);

    let mut encoder = device.begin_commands()?;
    encoder.copy_buffer_to_image(upload, 0, &chain, 0)?;
    assert!(matches!(
        encoder.dispatch(
            &DispatchDescriptor::new(invert)
                .with_sets(&[&both])
                .with_groups([1, 1, 1])
        ),
        Err(RhiError::InvalidImageUsage { .. })
    ));
    encoder.dispatch(
        &DispatchDescriptor::new(invert)
            .with_sets(&[&down])
            .with_groups([1, 1, 1]),
    )?;
    encoder.copy_image_to_buffer(&chain, 1, &once, 0)?;
    // Written again, having been copied from, the larger still being read.
    encoder.dispatch(
        &DispatchDescriptor::new(invert)
            .with_sets(&[&down])
            .with_groups([1, 1, 1]),
    )?;
    encoder.finish()?;
    close(first_texel(&once)?, [204, 153, 102, 255], "the smaller mip");

    // And all of it brought to one state from the two it was left in.
    let mut encoder = device.begin_commands()?;
    encoder.transition_image(&chain, ResourceState::CopySource)?;
    encoder.copy_image_to_buffer(&chain, 1, &whole, 0)?;
    encoder.finish()?;
    close(
        first_texel(&whole)?,
        [204, 153, 102, 255],
        "the whole chain",
    );
    Ok(())
}

fn run(backend: Backend) -> Result<(), RhiError> {
    let device = Device::open(DeviceOptions {
        backend,
        debug: true,
        require_ray_tracing: false,
        ..DeviceOptions::default()
    })?;
    let first = image(
        &device,
        ImageUsages::COPY_DESTINATION
            | ImageUsages::COPY_SOURCE
            | ImageUsages::SAMPLED
            | ImageUsages::RENDER_TARGET,
    )?;
    let second = image(&device, ImageUsages::STORAGE | ImageUsages::SAMPLED)?;
    let third = image(&device, ImageUsages::STORAGE | ImageUsages::COPY_SOURCE)?;

    let layout = device.create_pipeline_layout(PipelineLayoutDescriptor::new(&[
        DescriptorSetLayoutDescriptor::new(
            0,
            &[
                DescriptorBinding::new(
                    0,
                    DescriptorType::CombinedImageSampler(StaticSampler::linear_clamp()),
                    ShaderStages::COMPUTE,
                ),
                DescriptorBinding::new(1, DescriptorType::StorageImage, ShaderStages::COMPUTE),
            ],
        ),
    ]))?;
    let module = fixtures::module(
        &device,
        "transitions_invert",
        "transitions_invert",
        ShaderKind::Compute { entry: "main" },
    )?;
    let invert = device.create_compute_pipeline(&ComputePipelineDescriptor::new(
        "transitions_invert",
        &layout,
        module.entry("main"),
    ))?;
    let mut first_to_second = device.create_descriptor_set(&layout, 0)?;
    first_to_second.write_combined_image_sampler(0, &first.create_full_view()?)?;
    first_to_second.write_storage_image(1, &second.create_full_view()?)?;
    let mut second_to_third = device.create_descriptor_set(&layout, 0)?;
    second_to_third.write_combined_image_sampler(0, &second.create_full_view()?)?;
    second_to_third.write_storage_image(1, &third.create_full_view()?)?;

    // One colour, in every texel of the first image.
    let colour = [51_u8, 102, 153, 255];
    let copy = first.copy_layout(0)?;
    let mut texels = vec![0_u8; copy.total_size as usize];
    for row in 0..EDGE as usize {
        let start = row * copy.row_pitch as usize;
        for texel in texels[start..start + EDGE as usize * 4].chunks_exact_mut(4) {
            texel.copy_from_slice(&colour);
        }
    }
    let upload = device.create_buffer(BufferDescriptor::new(
        copy.total_size,
        BufferUsages::COPY_SOURCE,
        BufferMemory::Upload,
    ))?;
    upload.write(0, &texels)?;
    let (inverted, restored, cleared) = (
        readback(&device, &third)?,
        readback(&device, &third)?,
        readback(&device, &first)?,
    );

    // In one encoder: fresh from creation into each state in turn.
    let mut encoder = device.begin_commands()?;
    encoder.copy_buffer_to_image(&upload, 0, &first, 0)?;
    encoder.dispatch(
        &DispatchDescriptor::new(&invert)
            .with_sets(&[&first_to_second])
            .with_groups([2, 2, 1]),
    )?;
    // The second image was written by that and is read by this.
    encoder.dispatch(
        &DispatchDescriptor::new(&invert)
            .with_sets(&[&second_to_third])
            .with_groups([2, 2, 1]),
    )?;
    encoder.copy_image_to_buffer(&third, 0, &restored, 0)?;
    encoder.finish()?;
    close(
        first_texel(&restored)?,
        colour,
        "twice inverted, in one encoder",
    );

    // Across encoders: each finds every image as the last left it.
    let mut encoder = device.begin_commands()?;
    encoder.dispatch(
        &DispatchDescriptor::new(&invert)
            .with_sets(&[&first_to_second])
            .with_groups([2, 2, 1]),
    )?;
    encoder.finish()?;
    let mut encoder = device.begin_commands()?;
    // The third is written a second time, straight after being copied from.
    let mut second_into_third = device.create_descriptor_set(&layout, 0)?;
    second_into_third.write_combined_image_sampler(0, &first.create_full_view()?)?;
    second_into_third.write_storage_image(1, &third.create_full_view()?)?;
    encoder.dispatch(
        &DispatchDescriptor::new(&invert)
            .with_sets(&[&second_into_third])
            .with_groups([2, 2, 1]),
    )?;
    encoder.copy_image_to_buffer(&third, 0, &inverted, 0)?;
    encoder.finish()?;
    close(
        first_texel(&inverted)?,
        [204, 153, 102, 255],
        "inverted once, across encoders",
    );

    // Drawn into (a clear is enough) and copied out of, having last been
    // sampled; and an explicitly set state is kept while it stays valid.
    let mut encoder = device.begin_commands()?;
    encoder.begin_render_pass(&RenderPassDescriptor::color(
        &first.create_full_view()?,
        LoadOp::Clear([1.0, 0.0, 0.0, 1.0]),
    ))?;
    encoder.end_render_pass()?;
    encoder.transition_image(&first, ResourceState::CopySource)?;
    encoder.copy_image_to_buffer(&first, 0, &cleared, 0)?;
    encoder.finish()?;
    close(first_texel(&cleared)?, [255, 0, 0, 255], "cleared");

    buffers(&device)?;
    mips(&device, &layout, &invert, &upload)?;

    device.wait_idle()?;
    eprintln!("{:#?}", device.debug_messages());
    assert_eq!(device.debug_error_count().unwrap_or(0), 0, "{backend:?}");
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan device (and a DX12 adapter when dx12 is enabled)"]
fn images_are_put_in_their_states_without_being_told_on_every_backend() {
    run(Backend::Vulkan).expect("Vulkan transitions");
    #[cfg(all(feature = "dx12", windows))]
    run(Backend::Dx12).expect("DX12 transitions");
}
