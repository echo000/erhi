//! Hardware checks of the general raster and texture features: each asserts
//! the same pixels on Vulkan and on DX12.
//!
//! - Culling and winding agree between the backends.
//! - Two colour targets, each with its own blend and write mask.
//! - Stencil written by one pipeline and tested by the next.
//! - Line lists, and a viewport of part of the target.
//! - A pass into one mip of one layer of an image.
//! - Sampler objects (their level-of-detail range) and cube views, and more
//!   sets alive with samplers than a Direct3D 12 sampler heap has slots.
//! - Every format the device accepts, copied in and back out.
//! - Three vertex streams (two per vertex, one per instance) under 16-bit
//!   indices, and a pipeline with no stream at all.
//! - The depth and the stencil of one image, each copied out and back in.

mod fixtures;

use erhi::{
    Backend, BlendMode, Buffer, BufferDescriptor, BufferMemory, BufferUsages, ColorAttachment,
    ColorTargetState, ColorWrites, CompareFunction, ComputePipelineDescriptor, CullMode,
    DepthAttachment, DepthLoadOp, DepthStencilState, DescriptorBinding,
    DescriptorSetLayoutDescriptor, DescriptorType, Device, DeviceOptions, DispatchDescriptor,
    DrawDescriptor, FilterMode, FrontFace, Image, ImageAspect, ImageDescriptor, ImageFormat,
    ImageUsages, ImageViewDescriptor, ImageViewDimension, IndexFormat, LoadOp, PipelineLayout,
    PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, PushConstantRange,
    RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, ResourceState, RhiError,
    SamplerDescriptor, ShaderKind, ShaderModule, ShaderStages, StencilFaceState, StencilLoadOp,
    StencilOperation, StencilState, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStep,
};

const SIZE: u32 = 64;

struct Fixture {
    device: Device,
    layout: PipelineLayout,
    vertex: ShaderModule,
    fragment: ShaderModule,
    /// One triangle over the whole viewport, counter-clockwise on the screen.
    triangle: Buffer,
    push: Vec<u8>,
    backend: Backend,
}

const ATTRIBUTES: [VertexAttribute; 2] = [
    VertexAttribute::new(0, 0, VertexFormat::Float32x2),
    VertexAttribute::new(1, 8, VertexFormat::Float32x4),
];

fn vertices(device: &Device, vertices: &[[f32; 6]]) -> Result<Buffer, RhiError> {
    let bytes: Vec<u8> = vertices
        .iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    filled(device, BufferUsages::VERTEX, &bytes)
}

fn floats(device: &Device, values: &[f32]) -> Result<Buffer, RhiError> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    filled(device, BufferUsages::VERTEX, &bytes)
}

fn filled(device: &Device, usages: BufferUsages, bytes: &[u8]) -> Result<Buffer, RhiError> {
    let buffer = device.create_buffer(BufferDescriptor::new(
        bytes.len() as u64,
        usages,
        BufferMemory::Upload,
    ))?;
    buffer.write(0, bytes)?;
    Ok(buffer)
}

impl Fixture {
    fn new(backend: Backend) -> Result<Self, RhiError> {
        let device = Device::open(DeviceOptions {
            backend,
            debug: true,
            ..DeviceOptions::default()
        })?;
        let layout = device.create_pipeline_layout(
            PipelineLayoutDescriptor::new(&[])
                .with_push_constants(PushConstantRange::new(ShaderStages::VERTEX, 16)),
        )?;
        let vertex = fixtures::module(
            &device,
            "raster_state_vertex",
            "raster_state_raster",
            ShaderKind::Vertex {
                entry: "vertex_main",
            },
        )?;
        let fragment = Self::fragment_module(&device, "fragment_main")?;
        // Counter-clockwise with y up: right along the bottom, then up.
        let triangle = self::vertices(
            &device,
            &[
                [-1.0, -1.0, 0.0, 1.0, 0.0, 1.0],
                [3.0, -1.0, 0.0, 1.0, 0.0, 1.0],
                [-1.0, 3.0, 0.0, 1.0, 0.0, 1.0],
            ],
        )?;
        let push = [1.0_f32, 1.0, 0.0, 0.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        Ok(Self {
            device,
            layout,
            vertex,
            fragment,
            triangle,
            push,
            backend,
        })
    }

    fn fragment_module(device: &Device, entry: &'static str) -> Result<ShaderModule, RhiError> {
        fixtures::module(
            device,
            entry,
            "raster_state_raster",
            ShaderKind::Fragment { entry },
        )
    }

    fn pipeline(
        &self,
        primitive: PrimitiveState,
        color_targets: &[ColorTargetState],
        depth_stencil: Option<DepthStencilState>,
    ) -> Result<RenderPipeline, RhiError> {
        self.device.create_render_pipeline(
            &RenderPipelineDescriptor::new(
                "raster_state",
                &self.layout,
                self.vertex.entry("vertex_main"),
                self.fragment.entry("fragment_main"),
            )
            .with_vertex_buffers(&[VertexBufferLayout::new(24, &ATTRIBUTES)])
            .with_primitive(primitive)
            .with_color_targets(color_targets)
            .with_depth_stencil(depth_stencil),
        )
    }

    fn target(&self, format: ImageFormat) -> Result<Image, RhiError> {
        self.device.create_image(ImageDescriptor::d2(
            format,
            SIZE,
            SIZE,
            ImageUsages::RENDER_TARGET | ImageUsages::COPY_SOURCE,
        ))
    }

    /// The RGBA8 pixels of mip `mip` of `image` (in `CopySource`), every
    /// layer one after the other.
    fn pixels(&self, image: &Image, mip: u32) -> Result<Vec<[u8; 4]>, RhiError> {
        let layout = image.copy_layout(mip)?;
        let readback = self.device.create_buffer(BufferDescriptor::new(
            layout.total_size,
            BufferUsages::COPY_DESTINATION,
            BufferMemory::Readback,
        ))?;
        let mut encoder = self.device.begin_commands()?;
        encoder.copy_image_to_buffer(image, mip, &readback, 0)?;
        encoder.finish()?;
        let bytes = readback.read(0, layout.total_size)?;
        let (width, _, _) = image.descriptor().mip_extent(mip);
        let rows = layout.rows_per_slice * layout.slice_count;
        Ok((0..rows as usize)
            .flat_map(|row| {
                let start = row * layout.row_pitch as usize;
                bytes[start..start + width as usize * 4]
                    .chunks_exact(4)
                    .map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]])
                    .collect::<Vec<_>>()
            })
            .collect())
    }

    fn context(&self, what: &str) -> String {
        format!("{:?} {what}", self.backend)
    }
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

const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

const GREEN_PIXEL: [u8; 4] = [0, 255, 0, 255];

const BLUE_PIXEL: [u8; 4] = [0, 0, 255, 255];

fn at(x: u32, y: u32) -> usize {
    (y * SIZE + x) as usize
}

/// A counter-clockwise triangle is kept by a pipeline whose front is
/// counter-clockwise and that culls backs, and dropped by one whose front
/// is clockwise: on both backends alike.
fn culling(fixture: &Fixture) -> Result<(), RhiError> {
    let targets = [ColorTargetState::new(
        ImageFormat::Rgba8Unorm,
        BlendMode::Opaque,
    )];
    for (front_face, cull_mode, expected) in [
        (FrontFace::CounterClockwise, CullMode::Back, GREEN_PIXEL),
        (FrontFace::Clockwise, CullMode::Back, BLUE_PIXEL),
        (FrontFace::CounterClockwise, CullMode::Front, BLUE_PIXEL),
        (FrontFace::Clockwise, CullMode::Front, GREEN_PIXEL),
        (FrontFace::Clockwise, CullMode::None, GREEN_PIXEL),
    ] {
        let pipeline = fixture.pipeline(
            PrimitiveState {
                cull_mode,
                front_face,
                ..Default::default()
            },
            &targets,
            None,
        )?;
        let target = fixture.target(ImageFormat::Rgba8Unorm)?;
        let view = target.create_full_view()?;
        let mut encoder = fixture.device.begin_commands()?;
        encoder.transition_image(&target, ResourceState::RenderTarget)?;
        encoder.begin_render_pass(&RenderPassDescriptor::color(&view, LoadOp::Clear(BLUE)))?;
        encoder.draw(
            &DrawDescriptor::new(&pipeline)
                .with_vertex_buffers(&[&fixture.triangle])
                .with_range(0, 3)
                .with_push_constants(&fixture.push),
        )?;
        encoder.end_render_pass()?;
        encoder.transition_image(&target, ResourceState::CopySource)?;
        encoder.finish()?;
        assert_close(
            fixture.pixels(&target, 0)?[at(32, 32)],
            expected,
            &fixture.context(&format!("{front_face:?} front, cull {cull_mode:?}")),
        );
    }
    Ok(())
}

/// Two targets from one fragment shader: the first overwritten, the second
/// blended additively through a mask that lets only red through.
fn two_targets(fixture: &Fixture) -> Result<(), RhiError> {
    let fragment = Fixture::fragment_module(&fixture.device, "fragment_two")?;
    let pipeline = fixture.device.create_render_pipeline(
        &RenderPipelineDescriptor::new(
            "two_targets",
            &fixture.layout,
            fixture.vertex.entry("vertex_main"),
            fragment.entry("fragment_two"),
        )
        .with_vertex_buffers(&[VertexBufferLayout::new(24, &ATTRIBUTES)])
        .with_color_targets(&[
            ColorTargetState::new(ImageFormat::Rgba8Unorm, BlendMode::Opaque),
            ColorTargetState {
                write_mask: ColorWrites::RED,
                ..ColorTargetState::new(ImageFormat::Rgba8Unorm, BlendMode::Additive)
            },
        ]),
    )?;
    let first = fixture.target(ImageFormat::Rgba8Unorm)?;
    let second = fixture.target(ImageFormat::Rgba8Unorm)?;
    let (first_view, second_view) = (first.create_full_view()?, second.create_full_view()?);
    let mut encoder = fixture.device.begin_commands()?;
    encoder.transition_image(&first, ResourceState::RenderTarget)?;
    encoder.transition_image(&second, ResourceState::RenderTarget)?;
    encoder.begin_render_pass(
        &RenderPassDescriptor::new()
            .with_color(ColorAttachment::new(&first_view, LoadOp::Clear(BLUE)))
            .with_color(ColorAttachment::new(
                &second_view,
                LoadOp::Clear([0.5, 0.5, 0.5, 1.0]),
            )),
    )?;
    // A pipeline of one target does not draw into a pass of two.
    let single = fixture.pipeline(
        PrimitiveState::default(),
        &[ColorTargetState::new(
            ImageFormat::Rgba8Unorm,
            BlendMode::Opaque,
        )],
        None,
    )?;
    assert!(
        encoder
            .draw(
                &DrawDescriptor::new(&single)
                    .with_vertex_buffers(&[&fixture.triangle])
                    .with_range(0, 3)
                    .with_push_constants(&fixture.push)
            )
            .is_err()
    );
    encoder.draw(
        &DrawDescriptor::new(&pipeline)
            .with_vertex_buffers(&[&fixture.triangle])
            .with_range(0, 3)
            .with_push_constants(&fixture.push),
    )?;
    encoder.end_render_pass()?;
    encoder.transition_image(&first, ResourceState::CopySource)?;
    encoder.transition_image(&second, ResourceState::CopySource)?;
    encoder.finish()?;
    assert_close(
        fixture.pixels(&first, 0)?[at(32, 32)],
        GREEN_PIXEL,
        &fixture.context("first target"),
    );
    // 0.25 * 1 + 0.5 in red; green, blue and alpha as they were cleared.
    assert_close(
        fixture.pixels(&second, 0)?[at(32, 32)],
        [191, 128, 128, 255],
        &fixture.context("second target"),
    );
    Ok(())
}

/// One pipeline writes a stencil value over the left half and no colour;
/// the next draws only where that value is.
fn stencil(fixture: &Fixture) -> Result<(), RhiError> {
    let format = [
        ImageFormat::Depth32FloatStencil8,
        ImageFormat::Depth24UnormStencil8,
    ]
    .into_iter()
    .find(|format| {
        fixture
            .device
            .format_supports(*format, ImageUsages::DEPTH_TARGET)
    })
    .expect("a device has a depth-stencil format");
    let state = |compare, pass| DepthStencilState {
        format,
        depth_compare: CompareFunction::Always,
        depth_write: false,
        stencil: Some(StencilState {
            front: StencilFaceState {
                compare,
                pass,
                ..Default::default()
            },
            back: StencilFaceState {
                compare,
                pass,
                ..Default::default()
            },
            reference: 1,
            ..Default::default()
        }),
    };
    let mark = fixture.pipeline(
        PrimitiveState::default(),
        &[ColorTargetState {
            write_mask: ColorWrites::NONE,
            ..ColorTargetState::new(ImageFormat::Rgba8Unorm, BlendMode::Opaque)
        }],
        Some(state(CompareFunction::Always, StencilOperation::Replace)),
    )?;
    let test = fixture.pipeline(
        PrimitiveState::default(),
        &[ColorTargetState::new(
            ImageFormat::Rgba8Unorm,
            BlendMode::Opaque,
        )],
        Some(state(CompareFunction::Equal, StencilOperation::Keep)),
    )?;
    let target = fixture.target(ImageFormat::Rgba8Unorm)?;
    let depth = fixture.device.create_image({
        let mut made = target.descriptor();
        made.format = format;
        made.usages = ImageUsages::DEPTH_TARGET;
        made
    })?;
    let (view, depth_view) = (target.create_full_view()?, depth.create_full_view()?);
    let mut encoder = fixture.device.begin_commands()?;
    encoder.transition_image(&target, ResourceState::RenderTarget)?;
    encoder.transition_image(&depth, ResourceState::DepthWrite)?;
    encoder.begin_render_pass(
        &RenderPassDescriptor::new()
            .with_color(ColorAttachment::new(&view, LoadOp::Clear(BLUE)))
            .with_depth(DepthAttachment::new(&depth_view)),
    )?;
    encoder.set_scissor(0, 0, SIZE / 2, SIZE)?;
    encoder.draw(
        &DrawDescriptor::new(&mark)
            .with_vertex_buffers(&[&fixture.triangle])
            .with_range(0, 3)
            .with_push_constants(&fixture.push),
    )?;
    encoder.set_scissor(0, 0, SIZE, SIZE)?;
    encoder.draw(
        &DrawDescriptor::new(&test)
            .with_vertex_buffers(&[&fixture.triangle])
            .with_range(0, 3)
            .with_push_constants(&fixture.push),
    )?;
    encoder.end_render_pass()?;
    encoder.transition_image(&target, ResourceState::CopySource)?;
    encoder.finish()?;
    let pixels = fixture.pixels(&target, 0)?;
    assert_close(
        pixels[at(10, 32)],
        GREEN_PIXEL,
        &fixture.context("stencil marked"),
    );
    assert_close(
        pixels[at(50, 32)],
        BLUE_PIXEL,
        &fixture.context("stencil unmarked"),
    );
    Ok(())
}

/// A line list draws lines, and a viewport of the right half keeps a
/// full-screen triangle out of the left.
fn lines_and_viewport(fixture: &Fixture) -> Result<(), RhiError> {
    let targets = [ColorTargetState::new(
        ImageFormat::Rgba8Unorm,
        BlendMode::Opaque,
    )];
    // Through the middle of pixel row 20 (y up: row 20 from the top).
    let y = 1.0 - 2.0 * 20.5 / SIZE as f32;
    let line = vertices(
        &fixture.device,
        &[[-1.0, y, 0.0, 1.0, 0.0, 1.0], [1.0, y, 0.0, 1.0, 0.0, 1.0]],
    )?;
    let lines = fixture.pipeline(
        PrimitiveState {
            topology: PrimitiveTopology::LineList,
            ..Default::default()
        },
        &targets,
        None,
    )?;
    let triangles = fixture.pipeline(PrimitiveState::default(), &targets, None)?;
    let (first, second) = (
        fixture.target(ImageFormat::Rgba8Unorm)?,
        fixture.target(ImageFormat::Rgba8Unorm)?,
    );
    let mut encoder = fixture.device.begin_commands()?;
    encoder.transition_image(&first, ResourceState::RenderTarget)?;
    encoder.begin_render_pass(&RenderPassDescriptor::color(
        &first.create_full_view()?,
        LoadOp::Clear(BLUE),
    ))?;
    encoder.draw(
        &DrawDescriptor::new(&lines)
            .with_vertex_buffers(&[&line])
            .with_range(0, 2)
            .with_push_constants(&fixture.push),
    )?;
    encoder.end_render_pass()?;
    encoder.transition_image(&first, ResourceState::CopySource)?;
    encoder.transition_image(&second, ResourceState::RenderTarget)?;
    encoder.begin_render_pass(&RenderPassDescriptor::color(
        &second.create_full_view()?,
        LoadOp::Clear(BLUE),
    ))?;
    encoder.set_viewport(
        SIZE as f32 / 2.0,
        0.0,
        SIZE as f32 / 2.0,
        SIZE as f32,
        0.0,
        1.0,
    )?;
    encoder.draw(
        &DrawDescriptor::new(&triangles)
            .with_vertex_buffers(&[&fixture.triangle])
            .with_range(0, 3)
            .with_push_constants(&fixture.push),
    )?;
    encoder.end_render_pass()?;
    encoder.transition_image(&second, ResourceState::CopySource)?;
    encoder.finish()?;
    let pixels = fixture.pixels(&first, 0)?;
    assert_close(
        pixels[at(32, 20)],
        GREEN_PIXEL,
        &fixture.context("on the line"),
    );
    assert_close(
        pixels[at(32, 40)],
        BLUE_PIXEL,
        &fixture.context("off the line"),
    );
    let pixels = fixture.pixels(&second, 0)?;
    assert_close(
        pixels[at(48, 32)],
        GREEN_PIXEL,
        &fixture.context("inside the viewport"),
    );
    assert_close(
        pixels[at(16, 32)],
        BLUE_PIXEL,
        &fixture.context("outside the viewport"),
    );
    Ok(())
}

/// A pass into the second mip of the second layer of an image draws there
/// and nowhere else.
fn mip_and_layer(fixture: &Fixture) -> Result<(), RhiError> {
    let image = fixture.device.create_image(
        ImageDescriptor::d2(
            ImageFormat::Rgba8Unorm,
            SIZE,
            SIZE,
            ImageUsages::RENDER_TARGET | ImageUsages::COPY_SOURCE,
        )
        .with_layers(2)
        .with_mip_count(2),
    )?;
    let pipeline = fixture.pipeline(
        PrimitiveState::default(),
        &[ColorTargetState::new(
            ImageFormat::Rgba8Unorm,
            BlendMode::Opaque,
        )],
        None,
    )?;
    let views = [
        image.create_view(&ImageViewDescriptor::subresource(1, 0))?,
        image.create_view(&ImageViewDescriptor::subresource(1, 1))?,
    ];
    assert_eq!(views[1].extent(), (SIZE / 2, SIZE / 2));
    let mut encoder = fixture.device.begin_commands()?;
    encoder.transition_image(&image, ResourceState::RenderTarget)?;
    encoder.begin_render_pass(&RenderPassDescriptor::color(&views[0], LoadOp::Clear(BLUE)))?;
    encoder.end_render_pass()?;
    encoder.begin_render_pass(&RenderPassDescriptor::color(&views[1], LoadOp::Clear(BLUE)))?;
    encoder.draw(
        &DrawDescriptor::new(&pipeline)
            .with_vertex_buffers(&[&fixture.triangle])
            .with_range(0, 3)
            .with_push_constants(&fixture.push),
    )?;
    encoder.end_render_pass()?;
    // A view of every mip is no attachment.
    assert!(
        encoder
            .begin_render_pass(&RenderPassDescriptor::color(
                &image.create_full_view()?,
                LoadOp::Load
            ))
            .is_err()
    );
    encoder.transition_image(&image, ResourceState::CopySource)?;
    encoder.finish()?;
    let pixels = fixture.pixels(&image, 1)?;
    let half = (SIZE / 2) as usize;
    let centre = half * half / 2 + half / 2;
    assert_close(
        pixels[centre],
        BLUE_PIXEL,
        &fixture.context("layer 0, mip 1"),
    );
    assert_close(
        pixels[half * half + centre],
        GREEN_PIXEL,
        &fixture.context("layer 1, mip 1"),
    );
    Ok(())
}

/// A staging buffer of `layout`'s size whose every texel of slice `s` is
/// `colours[s]`.
fn solid_slices(
    device: &Device,
    image: &Image,
    mip: u32,
    colours: &[[u8; 4]],
) -> Result<Buffer, RhiError> {
    let layout = image.copy_layout(mip)?;
    let mut bytes = vec![0_u8; layout.total_size as usize];
    let slice = layout.row_pitch as usize * layout.rows_per_slice as usize;
    for (index, colour) in colours.iter().enumerate() {
        for texel in bytes[index * slice..(index + 1) * slice].chunks_exact_mut(4) {
            texel.copy_from_slice(colour);
        }
    }
    let staging = device.create_buffer(BufferDescriptor::new(
        layout.total_size,
        BufferUsages::COPY_SOURCE,
        BufferMemory::Upload,
    ))?;
    staging.write(0, &bytes)?;
    Ok(staging)
}

/// A cube view reads each face along its axis, and a sampler object's
/// level-of-detail range decides which mip a sample reads.
fn samplers_and_cubes(fixture: &Fixture) -> Result<(), RhiError> {
    let device = &fixture.device;
    let bindings = [
        (0, DescriptorType::SampledImage),
        (1, DescriptorType::Sampler),
        (2, DescriptorType::SampledImage),
        (3, DescriptorType::StorageImage),
    ]
    .map(|(binding, descriptor_type)| {
        DescriptorBinding::new(binding, descriptor_type, ShaderStages::COMPUTE)
    });
    let layout = device.create_pipeline_layout(PipelineLayoutDescriptor::new(&[
        DescriptorSetLayoutDescriptor::new(0, &bindings),
    ]))?;
    let module = fixtures::module(
        device,
        "samplers_and_cubes",
        "raster_state_sample",
        ShaderKind::Compute { entry: "main" },
    )?;
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor::new(
        "samplers_and_cubes",
        &layout,
        module.entry("main"),
    ))?;
    let sampled = ImageUsages::SAMPLED | ImageUsages::COPY_DESTINATION;
    let mips = device.create_image(
        ImageDescriptor::d2(ImageFormat::Rgba8Unorm, 8, 8, sampled).with_mip_count(2),
    )?;
    let cube = device.create_image({
        let mut made = mips.descriptor();
        made.depth_or_layers = 6;
        made.mip_count = 1;
        made
    })?;
    let destination = device.create_image({
        let mut made = mips.descriptor();
        made.width = 8;
        made.height = 1;
        made.mip_count = 1;
        made.usages = ImageUsages::STORAGE | ImageUsages::COPY_SOURCE;
        made
    })?;
    let large = [200, 0, 0, 255];
    let small = [0, 200, 0, 255];
    let faces: [[u8; 4]; 6] = std::array::from_fn(|face| [40 * face as u8 + 20, 0, 255, 255]);
    let mut encoder = device.begin_commands()?;
    encoder.transition_image(&mips, ResourceState::CopyDest)?;
    encoder.transition_image(&cube, ResourceState::CopyDest)?;
    encoder.copy_buffer_to_image(&solid_slices(device, &mips, 0, &[large])?, 0, &mips, 0)?;
    encoder.copy_buffer_to_image(&solid_slices(device, &mips, 1, &[small])?, 0, &mips, 1)?;
    encoder.copy_buffer_to_image(&solid_slices(device, &cube, 0, &faces)?, 0, &cube, 0)?;
    encoder.transition_image(&mips, ResourceState::ShaderResource)?;
    encoder.transition_image(&cube, ResourceState::ShaderResource)?;
    encoder.finish()?;

    let cube_view = cube.create_view(&ImageViewDescriptor {
        dimension: Some(ImageViewDimension::Cube),
        ..Default::default()
    })?;
    let nearest = SamplerDescriptor {
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mip_filter: FilterMode::Nearest,
        ..Default::default()
    };
    // The second sampler never reads the largest mip.
    for (sampler, first_mip) in [
        (nearest, large),
        (
            SamplerDescriptor {
                lod_min: 1.0,
                ..nearest
            },
            small,
        ),
    ] {
        let sampler = device.create_sampler(&sampler)?;
        let mut set = device.create_descriptor_set(&layout, 0)?;
        set.write_sampled_image(0, &mips.create_full_view()?)?;
        set.write_sampler(1, &sampler)?;
        set.write_sampled_image(2, &cube_view)?;
        set.write_storage_image(3, &destination.create_full_view()?)?;
        let mut encoder = device.begin_commands()?;
        encoder.transition_image(&destination, ResourceState::UnorderedAccess)?;
        encoder.dispatch(
            &DispatchDescriptor::new(&pipeline)
                .with_sets(&[&set])
                .with_groups([1, 1, 1]),
        )?;
        encoder.transition_image(&destination, ResourceState::CopySource)?;
        encoder.finish()?;
        let pixels = fixture.pixels(&destination, 0)?;
        for (face, expected) in faces.iter().enumerate() {
            assert_close(
                pixels[face],
                *expected,
                &fixture.context(&format!("cube face {face}")),
            );
        }
        assert_close(pixels[6], first_mip, &fixture.context("level 0 asked"));
        assert_close(pixels[7], small, &fixture.context("level 5 asked"));
    }
    // More sets alive at once than a Direct3D 12 sampler heap has slots,
    // each with its own sampler object. Those that describe the same
    // sampler share a slot, and the two kinds still read as themselves.
    let views = (mips.create_full_view()?, destination.create_full_view()?);
    let mut sets = Vec::new();
    let mut encoder = device.begin_commands()?;
    encoder.transition_image(&destination, ResourceState::UnorderedAccess)?;
    for index in 0..2100 {
        let sampler = device.create_sampler(&SamplerDescriptor {
            lod_min: (index % 2) as f32,
            ..nearest
        })?;
        let mut set = device.create_descriptor_set(&layout, 0)?;
        set.write_sampled_image(0, &views.0)?;
        set.write_sampler(1, &sampler)?;
        set.write_sampled_image(2, &cube_view)?;
        set.write_storage_image(3, &views.1)?;
        encoder.dispatch(
            &DispatchDescriptor::new(&pipeline)
                .with_sets(&[&set])
                .with_groups([1, 1, 1]),
        )?;
        sets.push(set);
    }
    encoder.finish()?;
    for (set, first_mip) in [(&sets[0], large), (&sets[1], small)] {
        let mut encoder = device.begin_commands()?;
        encoder.transition_image(&destination, ResourceState::UnorderedAccess)?;
        encoder.dispatch(
            &DispatchDescriptor::new(&pipeline)
                .with_sets(&[set])
                .with_groups([1, 1, 1]),
        )?;
        encoder.transition_image(&destination, ResourceState::CopySource)?;
        encoder.finish()?;
        assert_close(
            fixture.pixels(&destination, 0)?[6],
            first_mip,
            &fixture.context("one of many sets with samplers"),
        );
    }
    drop(sets);
    // Anisotropy with a nearest filter is refused before any device sees it.
    assert!(
        device
            .create_sampler(&SamplerDescriptor {
                anisotropy: 8,
                ..nearest
            })
            .is_err()
    );
    device.create_sampler(&SamplerDescriptor {
        anisotropy: 16,
        compare: Some(CompareFunction::LessEqual),
        ..Default::default()
    })?;
    Ok(())
}

/// Every format the device samples and copies comes back from an image as
/// the bytes that went in. Returns how many were tried.
fn format_round_trips(fixture: &Fixture) -> Result<usize, RhiError> {
    let device = &fixture.device;
    let usages = ImageUsages::SAMPLED | ImageUsages::COPY_SOURCE | ImageUsages::COPY_DESTINATION;
    let mut tried = 0;
    for format in ImageFormat::ALL {
        if format.is_depth() {
            continue;
        }
        if !device.format_supports(format, usages) {
            println!("{:?}: {format:?} is not offered", fixture.backend);
            continue;
        }
        let image = device.create_image(ImageDescriptor::d2(format, 8, 8, usages))?;
        let layout = image.copy_layout(0)?;
        let row_bytes = (8 / format.block_extent() * format.bytes_per_block()) as usize;
        let mut bytes = vec![0_u8; layout.total_size as usize];
        for row in 0..layout.rows_per_slice as usize {
            let start = row * layout.row_pitch as usize;
            for (index, byte) in bytes[start..start + row_bytes].iter_mut().enumerate() {
                // (Not every bit pattern is a float that survives a copy
                // through some drivers; these are plain small values.)
                *byte = ((row * 7 + index * 3) % 120) as u8;
            }
        }
        let staging = device.create_buffer(BufferDescriptor::new(
            layout.total_size,
            BufferUsages::COPY_SOURCE,
            BufferMemory::Upload,
        ))?;
        staging.write(0, &bytes)?;
        let readback = device.create_buffer(BufferDescriptor::new(
            layout.total_size,
            BufferUsages::COPY_DESTINATION,
            BufferMemory::Readback,
        ))?;
        let mut encoder = device.begin_commands()?;
        encoder.transition_image(&image, ResourceState::CopyDest)?;
        encoder.copy_buffer_to_image(&staging, 0, &image, 0)?;
        encoder.transition_image(&image, ResourceState::CopySource)?;
        encoder.copy_image_to_buffer(&image, 0, &readback, 0)?;
        encoder.finish()?;
        let back = readback.read(0, layout.total_size)?;
        for row in 0..layout.rows_per_slice as usize {
            let start = row * layout.row_pitch as usize;
            assert_eq!(
                back[start..start + row_bytes],
                bytes[start..start + row_bytes],
                "{:?} {format:?} row {row}",
                fixture.backend
            );
        }
        tried += 1;
    }
    // The ones a renderer leans on are everywhere.
    for format in [
        ImageFormat::Rgba8UnormSrgb,
        ImageFormat::Bgra8Unorm,
        ImageFormat::Rgb10a2Unorm,
        ImageFormat::Rg11b10Ufloat,
        ImageFormat::Rgba16Float,
        ImageFormat::R32Uint,
        ImageFormat::Bc7UnormSrgb,
    ] {
        assert!(
            device.format_supports(format, ImageUsages::SAMPLED),
            "{:?} {format:?}",
            fixture.backend
        );
    }
    assert!(device.format_supports(ImageFormat::Depth32Float, ImageUsages::DEPTH_TARGET));
    assert!(!device.format_supports(ImageFormat::Bc1Unorm, ImageUsages::RENDER_TARGET));
    Ok(tried)
}

/// A quad of two per-vertex streams (positions, colours) under 16-bit
/// indices, placed by a per-instance stream; then the same streams without
/// indices; then a pipeline with no stream at all.
fn vertex_streams(fixture: &Fixture) -> Result<(), RhiError> {
    let device = &fixture.device;
    let shifted = fixtures::module(
        device,
        "vertex_shifted",
        "raster_state_raster",
        ShaderKind::Vertex {
            entry: "vertex_shifted",
        },
    )?;
    let bare = fixtures::module(
        device,
        "vertex_bare",
        "raster_state_raster",
        ShaderKind::Vertex {
            entry: "vertex_bare",
        },
    )?;
    let targets = [ColorTargetState::new(
        ImageFormat::Rgba8Unorm,
        BlendMode::Opaque,
    )];
    let attribute = |location, format| [VertexAttribute::new(location, 0, format)];
    let (positions, colours, shifts) = (
        attribute(0, VertexFormat::Float32x2),
        attribute(1, VertexFormat::Float32x4),
        attribute(2, VertexFormat::Float32x2),
    );
    let pipeline = device.create_render_pipeline(
        &RenderPipelineDescriptor::new(
            "vertex_streams",
            &fixture.layout,
            shifted.entry("vertex_shifted"),
            fixture.fragment.entry("fragment_main"),
        )
        .with_vertex_buffers(&[
            VertexBufferLayout::new(8, &positions),
            VertexBufferLayout::new(16, &colours),
            VertexBufferLayout::new(8, &shifts).with_step(VertexStep::Instance),
        ])
        .with_color_targets(&targets),
    )?;
    // A quad an eighth of the clip square to each side of the origin. The
    // first record of both per-vertex streams is never drawn: the indexed
    // draw's base vertex steps over it.
    const QUAD: [[f32; 2]; 4] = [
        [-0.125, -0.125],
        [0.125, -0.125],
        [-0.125, 0.125],
        [0.125, 0.125],
    ];
    const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
    let mut corners = vec![9.0, 9.0];
    corners.extend(QUAD.iter().flatten());
    let mut paints = vec![1.0, 0.0, 0.0, 1.0];
    paints.extend([GREEN; 4].iter().flatten());
    let (corners, paints) = (floats(device, &corners)?, floats(device, &paints)?);
    // Where each instance goes: pixels (16, 32), (48, 16) and (48, 48).
    let places = floats(device, &[-0.5, 0.0, 0.5, 0.5, 0.5, -0.5])?;
    let order: Vec<u8> = [0_u16, 1, 2, 2, 1, 3]
        .iter()
        .flat_map(|index| index.to_le_bytes())
        .collect();
    let order = filled(device, BufferUsages::INDEX, &order)?;
    let vertex_buffers = [&corners, &paints, &places];
    let indexed = DrawDescriptor::new(&pipeline)
        .with_vertex_buffers(&vertex_buffers)
        .with_indices(&order, IndexFormat::Uint16)
        .with_range(0, 6)
        .with_base_vertex(1)
        .with_instances(1, 2)
        .with_push_constants(&fixture.push);
    let draw_and_read = |draw: &DrawDescriptor<'_>| -> Result<Vec<[u8; 4]>, RhiError> {
        let target = fixture.target(ImageFormat::Rgba8Unorm)?;
        let view = target.create_full_view()?;
        let mut encoder = device.begin_commands()?;
        encoder.transition_image(&target, ResourceState::RenderTarget)?;
        encoder.begin_render_pass(&RenderPassDescriptor::color(&view, LoadOp::Clear(BLUE)))?;
        encoder.draw(draw)?;
        encoder.end_render_pass()?;
        encoder.transition_image(&target, ResourceState::CopySource)?;
        encoder.finish()?;
        fixture.pixels(&target, 0)
    };
    let pixels = draw_and_read(&indexed)?;
    for (x, y, expected, what) in [
        (16, 32, BLUE_PIXEL, "the instance stepped over"),
        (48, 16, GREEN_PIXEL, "the second instance"),
        (48, 48, GREEN_PIXEL, "the third instance"),
        (32, 32, BLUE_PIXEL, "between the instances"),
    ] {
        assert_close(pixels[at(x, y)], expected, &fixture.context(what));
    }

    // Without indices: the quad as six vertices, the first instance alone.
    let mut corners = Vec::new();
    for corner in [0, 1, 2, 2, 1, 3] {
        corners.extend(QUAD[corner]);
    }
    let corners = floats(device, &corners)?;
    let paints = floats(device, &[GREEN; 6].concat())?;
    let plain_streams = [&corners, &paints, &places];
    let mut plain = indexed
        .with_vertex_buffers(&plain_streams)
        .with_base_vertex(0)
        .with_instances(0, 1);
    plain.indices = None;
    let pixels = draw_and_read(&plain)?;
    for (x, y, expected, what) in [
        (16, 32, GREEN_PIXEL, "the first instance"),
        (48, 16, BLUE_PIXEL, "an instance not drawn"),
    ] {
        assert_close(pixels[at(x, y)], expected, &fixture.context(what));
    }

    // What no device should see: a stream short of the draw, a buffer for
    // every stream but one, and more instances than there are places.
    let target = fixture.target(ImageFormat::Rgba8Unorm)?;
    let view = target.create_full_view()?;
    let mut encoder = device.begin_commands()?;
    encoder.transition_image(&target, ResourceState::RenderTarget)?;
    encoder.begin_render_pass(&RenderPassDescriptor::color(&view, LoadOp::Clear(BLUE)))?;
    for wrong in [
        {
            let mut made = plain;
            made.count = 7;
            made
        },
        plain.with_vertex_buffers(&[&corners, &paints]),
        {
            let mut made = plain;
            made.instance_count = 4;
            made
        },
        {
            let mut made = indexed;
            made.count = 7;
            made
        },
    ] {
        assert!(encoder.draw(&wrong).is_err());
    }
    encoder.end_render_pass()?;
    encoder.finish()?;

    // No streams: the shader makes its triangle from the vertex index.
    let bare = device.create_render_pipeline(
        &RenderPipelineDescriptor::new(
            "vertex_bare",
            &fixture.layout,
            bare.entry("vertex_bare"),
            fixture.fragment.entry("fragment_main"),
        )
        .with_color_targets(&targets),
    )?;
    let pixels = draw_and_read(&{
        let mut made = plain;
        made.pipeline = &bare;
        made.vertex_buffers = &[];
        made.count = 3;
        made
    })?;
    for (x, y) in [(1, 1), (62, 1), (1, 62), (62, 62)] {
        assert_close(
            pixels[at(x, y)],
            GREEN_PIXEL,
            &fixture.context("a triangle from no stream"),
        );
    }
    Ok(())
}

/// The depth and the stencil of one image are copied out as a pass left
/// them, and copied in from buffers: one plane at a time, never both.
fn plane_copies(fixture: &Fixture) -> Result<(), RhiError> {
    let device = &fixture.device;
    // (The depth of the 24-bit format is in no fixed layout: only its
    // stencil is copied.)
    use ImageAspect::{All, Depth, Stencil};
    assert_eq!(ImageFormat::Depth24UnormStencil8.copy_bytes(Depth), None);
    assert_eq!(
        ImageFormat::Depth24UnormStencil8.copy_bytes(Stencil),
        Some(1)
    );
    assert_eq!(ImageFormat::Depth32FloatStencil8.copy_bytes(All), None);
    assert_eq!(ImageFormat::Rgba8Unorm.copy_bytes(Stencil), None);
    let format = ImageFormat::Depth32FloatStencil8;
    let usages =
        ImageUsages::DEPTH_TARGET | ImageUsages::COPY_SOURCE | ImageUsages::COPY_DESTINATION;
    if !device.format_supports(format, usages) {
        println!("{:?}: {format:?} is not copied", fixture.backend);
        return Ok(());
    }
    let image = device.create_image(ImageDescriptor::d2(format, SIZE, SIZE, usages))?;
    let view = image.create_full_view()?;
    // The texels of one plane, row by row, each of `size` bytes.
    let read = |aspect, size: usize| -> Result<Vec<u8>, RhiError> {
        let layout = image.copy_layout_of(0, aspect)?;
        let readback = device.create_buffer(BufferDescriptor::new(
            layout.total_size,
            BufferUsages::COPY_DESTINATION,
            BufferMemory::Readback,
        ))?;
        let mut encoder = device.begin_commands()?;
        encoder.copy_image_plane_to_buffer(&image, 0, aspect, &readback, 0)?;
        encoder.finish()?;
        let bytes = readback.read(0, layout.total_size)?;
        Ok((0..SIZE as usize)
            .flat_map(|row| {
                let start = row * layout.row_pitch as usize;
                bytes[start..start + SIZE as usize * size].to_vec()
            })
            .collect())
    };
    let depths = |bytes: Vec<u8>| -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|texel| f32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]))
            .collect()
    };

    let mut encoder = device.begin_commands()?;
    encoder.transition_image(&image, ResourceState::DepthWrite)?;
    encoder.begin_render_pass(
        &RenderPassDescriptor::new().with_depth(
            DepthAttachment::new(&view)
                .with_depth_load(DepthLoadOp::Clear(0.25))
                .with_stencil_load(StencilLoadOp::Clear(7)),
        ),
    )?;
    encoder.end_render_pass()?;
    encoder.transition_image(&image, ResourceState::CopySource)?;
    // Both planes at once have no layout in a buffer.
    let scratch = device.create_buffer(BufferDescriptor::new(
        1 << 16,
        BufferUsages::COPY_DESTINATION,
        BufferMemory::Readback,
    ))?;
    assert!(matches!(
        encoder.copy_image_to_buffer(&image, 0, &scratch, 0),
        Err(RhiError::UnsupportedImageCopy { .. })
    ));
    encoder.finish()?;
    assert!(
        depths(read(Depth, 4)?).iter().all(|depth| *depth == 0.25),
        "{}",
        fixture.context("cleared depth")
    );
    assert!(
        read(Stencil, 1)?.iter().all(|stencil| *stencil == 7),
        "{}",
        fixture.context("cleared stencil")
    );

    // In from buffers: a depth and a stencil that differ at every texel.
    let depth_at = |x: u32, y: u32| (x + y * SIZE) as f32 / (SIZE * SIZE) as f32;
    let stencil_at = |x: u32, y: u32| (x * 3 + y * 5) as u8;
    // `texels` is the plane row by row, each texel its own bytes.
    let staged = |aspect, texels: Vec<Vec<u8>>| -> Result<Buffer, RhiError> {
        let layout = image.copy_layout_of(0, aspect)?;
        let mut bytes = vec![0_u8; layout.total_size as usize];
        for (y, row) in texels.chunks_exact(SIZE as usize).enumerate() {
            let row = row.concat();
            let start = y * layout.row_pitch as usize;
            bytes[start..start + row.len()].copy_from_slice(&row);
        }
        filled(device, BufferUsages::COPY_SOURCE, &bytes)
    };
    let texels = (0..SIZE * SIZE).map(|index| (index % SIZE, index / SIZE));
    let new_depth = staged(
        Depth,
        texels
            .clone()
            .map(|(x, y)| depth_at(x, y).to_le_bytes().to_vec())
            .collect(),
    )?;
    let new_stencil = staged(
        Stencil,
        texels.map(|(x, y)| vec![stencil_at(x, y)]).collect(),
    )?;
    let mut encoder = device.begin_commands()?;
    encoder.transition_image(&image, ResourceState::CopyDest)?;
    encoder.copy_buffer_to_image_plane(&new_depth, 0, &image, 0, Depth)?;
    encoder.copy_buffer_to_image_plane(&new_stencil, 0, &image, 0, Stencil)?;
    encoder.transition_image(&image, ResourceState::CopySource)?;
    encoder.finish()?;
    let (depth, stencil) = (depths(read(Depth, 4)?), read(Stencil, 1)?);
    for y in 0..SIZE {
        for x in 0..SIZE {
            assert_eq!(
                (depth[at(x, y)], stencil[at(x, y)]),
                (depth_at(x, y), stencil_at(x, y)),
                "{} at {x}, {y}",
                fixture.context("planes copied in")
            );
        }
    }
    Ok(())
}

fn run(backend: Backend) -> Result<(), RhiError> {
    let fixture = Fixture::new(backend)?;
    culling(&fixture)?;
    two_targets(&fixture)?;
    stencil(&fixture)?;
    lines_and_viewport(&fixture)?;
    mip_and_layer(&fixture)?;
    samplers_and_cubes(&fixture)?;
    vertex_streams(&fixture)?;
    plane_copies(&fixture)?;
    let formats = format_round_trips(&fixture)?;
    println!("{backend:?}: {formats} formats copied in and out");
    assert!(formats >= 40, "{backend:?}: only {formats} formats offered");
    eprintln!("{:#?}", fixture.device.debug_messages());
    assert_eq!(fixture.device.debug_error_count().unwrap_or(0), 0);
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan device (and a DX12 adapter when dx12 is enabled)"]
fn raster_and_texture_features_agree_on_every_backend() {
    run(Backend::Vulkan).expect("Vulkan raster and texture features");

    #[cfg(all(feature = "dx12", windows))]
    run(Backend::Dx12).expect("DX12 raster and texture features");
}
