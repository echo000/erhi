//! Unit tests of `image.rs`.

use super::*;

#[test]
#[ignore = "requires Vulkan and DX12 hardware"]
fn worker_resources_keep_device_alive_after_renderer_is_dropped() {
    for backend in [Backend::Vulkan, Backend::Dx12]
        .into_iter()
        .filter(|backend| backend.is_compiled())
    {
        let device = Device::open(crate::DeviceOptions {
            backend,
            debug: true,
            ..Default::default()
        })
        .unwrap();
        let image = device.create_image(base_descriptor()).unwrap();
        let buffer = device
            .create_buffer(crate::BufferDescriptor {
                size: 16,
                usages: crate::BufferUsages::STORAGE,
                memory: crate::BufferMemory::Upload,
            })
            .unwrap();
        drop(device);
        buffer.write(0, &[1; 16]).unwrap();
        let view = image.create_full_view().unwrap();
        drop(image);
        drop(buffer);
        drop(view);
    }
}

fn base_descriptor() -> ImageDescriptor {
    ImageDescriptor {
        dimension: ImageDimension::D2,
        format: ImageFormat::Rgba16Float,
        width: 64,
        height: 64,
        depth_or_layers: 1,
        mip_count: 1,
        sample_count: 1,
        usages: ImageUsages::STORAGE | ImageUsages::COPY_SOURCE,
    }
}

#[test]
fn views_default_to_the_whole_image_and_reject_ranges_outside_it() {
    let image = ImageDescriptor {
        depth_or_layers: 12,
        mip_count: 4,
        ..base_descriptor()
    };
    let all = ImageViewDescriptor::default().resolve(&image).unwrap();
    assert_eq!(
        (all.dimension, all.mip_count, all.layer_count),
        (ImageViewDimension::D2Array, 4, 12)
    );
    let face = ImageViewDescriptor::subresource(2, 7)
        .resolve(&image)
        .unwrap();
    assert_eq!(
        (
            face.dimension,
            face.base_mip,
            face.mip_count,
            face.base_layer,
            face.layer_count
        ),
        (ImageViewDimension::D2, 2, 1, 7, 1)
    );
    let cube = ImageViewDescriptor {
        dimension: Some(ImageViewDimension::Cube),
        base_layer: 6,
        layer_count: Some(6),
        ..Default::default()
    };
    assert!(cube.resolve(&image).is_ok());
    let wrong = |descriptor: ImageViewDescriptor| {
        matches!(
            descriptor.resolve(&image),
            Err(RhiError::InvalidImageView { .. })
        )
    };
    // Five layers are no cube; mips and layers past the end; a colour
    // image has no depth plane; a 2D image is no volume.
    assert!(wrong(ImageViewDescriptor {
        layer_count: Some(5),
        ..cube
    }));
    assert!(wrong(ImageViewDescriptor {
        base_mip: 4,
        ..Default::default()
    }));
    assert!(wrong(ImageViewDescriptor {
        base_layer: 10,
        layer_count: Some(3),
        ..Default::default()
    }));
    assert!(wrong(ImageViewDescriptor {
        aspect: ImageAspect::Depth,
        ..Default::default()
    }));
    assert!(wrong(ImageViewDescriptor {
        dimension: Some(ImageViewDimension::D3),
        ..Default::default()
    }));
}

#[test]
fn every_format_is_listed_once_with_a_block_size() {
    let unique: std::collections::HashSet<_> = ImageFormat::ALL.into_iter().collect();
    assert_eq!(unique.len(), ImageFormat::ALL.len());
    for format in ImageFormat::ALL {
        assert!(format.bytes_per_block() > 0, "{format:?}");
        assert_eq!(
            format.block_extent() == 4,
            format.is_compressed(),
            "{format:?}"
        );
        assert!(!format.has_stencil() || format.is_depth(), "{format:?}");
    }
    assert_eq!(ImageFormat::Rgba8UnormSrgb.bytes_per_block(), 4);
    assert_eq!(ImageFormat::Bc1UnormSrgb.bytes_per_block(), 8);
    assert!(ImageFormat::Bgra8UnormSrgb.is_srgb() && !ImageFormat::Bgra8Unorm.is_srgb());
    assert!(ImageFormat::Rgba32Uint.is_integer() && !ImageFormat::Rgba32Float.is_integer());
}

#[test]
fn zero_dimensions_are_rejected() {
    let mut descriptor = base_descriptor();
    descriptor.width = 0;
    assert!(matches!(
        descriptor.validate(),
        Err(RhiError::InvalidImageDimensions { .. })
    ));
    let mut descriptor = base_descriptor();
    descriptor.depth_or_layers = 0;
    assert!(matches!(
        descriptor.validate(),
        Err(RhiError::InvalidImageDimensions { .. })
    ));
}

#[test]
fn empty_usages_are_rejected() {
    let mut descriptor = base_descriptor();
    descriptor.usages = ImageUsages::default();
    assert!(matches!(
        descriptor.validate(),
        Err(RhiError::EmptyImageUsages)
    ));
}

#[test]
fn mip_counts_are_bounded_by_the_largest_extent() {
    let mut descriptor = base_descriptor();
    assert_eq!(descriptor.max_mip_count(), 7);
    descriptor.mip_count = 7;
    assert!(descriptor.validate().is_ok());
    descriptor.mip_count = 8;
    assert!(matches!(
        descriptor.validate(),
        Err(RhiError::InvalidImageMipCount { .. })
    ));
    descriptor.mip_count = 0;
    assert!(matches!(
        descriptor.validate(),
        Err(RhiError::InvalidImageMipCount { .. })
    ));
}

#[test]
fn copy_rows_use_the_shared_alignment() {
    // 64 texels * 8 bytes = 512 bytes: already aligned.
    let descriptor = base_descriptor();
    let unaligned = u64::from(descriptor.width) * u64::from(descriptor.format.bytes_per_block());
    assert_eq!(unaligned % IMAGE_COPY_ROW_ALIGNMENT, 0);
    // 10 texels * 4 bytes = 40 bytes: rounds up to one 256-byte row.
    let narrow = 10_u64 * 4;
    assert_eq!(
        narrow.div_ceil(IMAGE_COPY_ROW_ALIGNMENT) * IMAGE_COPY_ROW_ALIGNMENT,
        256
    );
}

#[test]
fn mip_extents_never_reach_zero() {
    let descriptor = ImageDescriptor {
        width: 64,
        height: 4,
        mip_count: 7,
        ..base_descriptor()
    };
    assert_eq!(descriptor.mip_extent(0), (64, 4, 1));
    assert_eq!(descriptor.mip_extent(3), (8, 1, 1));
    assert_eq!(descriptor.mip_extent(6), (1, 1, 1));
}

#[test]
fn a_descriptor_is_refused_for_its_size_and_for_usages_its_format_cannot_serve() {
    let base = base_descriptor();
    assert!(base.validate().is_ok());
    // The largest an image may be, and one past it.
    let widest = ImageDescriptor {
        width: MAX_IMAGE_EXTENT,
        mip_count: 1,
        ..base
    };
    assert!(widest.validate().is_ok());
    for too_large in [
        ImageDescriptor {
            width: MAX_IMAGE_EXTENT + 1,
            ..widest
        },
        ImageDescriptor {
            depth_or_layers: MAX_IMAGE_LAYERS + 1,
            ..widest
        },
        ImageDescriptor {
            dimension: ImageDimension::D3,
            width: MAX_IMAGE_EXTENT_3D + 1,
            ..widest
        },
    ] {
        assert!(
            matches!(
                too_large.validate(),
                Err(RhiError::InvalidImageDimensions { .. })
            ),
            "{too_large:?}"
        );
    }
    let with = |format: ImageFormat, usages: ImageUsages| ImageDescriptor {
        format,
        usages,
        mip_count: 1,
        ..base
    };
    for refused in [
        with(ImageFormat::Rgba8Unorm, ImageUsages::DEPTH_TARGET),
        with(ImageFormat::Depth32Float, ImageUsages::RENDER_TARGET),
        with(
            ImageFormat::Depth32Float,
            ImageUsages::DEPTH_TARGET | ImageUsages::STORAGE,
        ),
        ImageDescriptor {
            dimension: ImageDimension::D3,
            ..with(ImageFormat::Depth32Float, ImageUsages::DEPTH_TARGET)
        },
    ] {
        assert!(
            matches!(refused.validate(), Err(RhiError::InvalidImageUsage { .. })),
            "{refused:?}"
        );
    }
    assert!(
        with(ImageFormat::Depth32Float, ImageUsages::DEPTH_TARGET)
            .validate()
            .is_ok()
    );
}

#[test]
fn each_mip_of_each_layer_has_a_place_of_its_own() {
    let image = ImageDescriptor {
        dimension: ImageDimension::D2,
        format: ImageFormat::Rgba8Unorm,
        width: 64,
        height: 64,
        depth_or_layers: 3,
        mip_count: 4,
        sample_count: 1,
        usages: ImageUsages::SAMPLED,
    };
    let places = |range: Subresources| range.indices(&image).collect::<Vec<_>>();
    assert_eq!(
        places(Subresources::whole(&image)),
        (0..12).collect::<Vec<_>>()
    );
    assert_eq!(places(Subresources::mip(&image, 2)), [2, 6, 10]);
    assert_eq!(places(Subresources::one(1, 2)), [9]);
    // Ranges meet where they share a mip of a layer, and only there.
    assert!(Subresources::mip(&image, 2).overlaps(Subresources::one(2, 1)));
    assert!(!Subresources::mip(&image, 2).overlaps(Subresources::one(1, 1)));
    assert!(!Subresources::one(1, 0).overlaps(Subresources::one(1, 2)));
    assert!(Subresources::whole(&image).overlaps(Subresources::one(3, 2)));
    // A 3D image is one layer however deep it is.
    let volume = ImageDescriptor {
        dimension: ImageDimension::D3,
        depth_or_layers: 32,
        mip_count: 2,
        ..image
    };
    assert_eq!(
        Subresources::whole(&volume)
            .indices(&volume)
            .collect::<Vec<_>>(),
        [0, 1]
    );
}
