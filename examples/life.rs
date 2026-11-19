//! Conway's Game of Life on the compute pipeline: two images that take
//! turns being the board and the next board, and three compute shaders
//! that seed one, step from one to the other, and draw one to the window.
//!
//! ```text
//! cargo run --example life
//! cargo run --example life -- dx12
//! ```
//!
//! Space or a click starts a new board. No image is ever told what state to
//! be in: a dispatch reads one and writes the other, and the encoder moves
//! each to where that needs it. The shaders are `shaders/life.slang`.

mod common;

use common::shader;
use erhi::{
    ComputePipelineDescriptor, DescriptorBinding, DescriptorSet, DescriptorSetLayoutDescriptor,
    DescriptorType, Device, DeviceOptions, DispatchDescriptor, Image, ImageDescriptor, ImageFormat,
    ImageUsages, PipelineLayoutDescriptor, PushConstantRange, ResourceState, RhiError, ShaderKind,
    ShaderModule, ShaderStages,
};

/// The board, in cells.
const BOARD: [u32; 2] = [320, 180];

/// How many generations a second.
const GENERATIONS_A_SECOND: f32 = 15.0;

/// How many groups of eight cover `size`.
fn groups(size: [u32; 2]) -> [u32; 3] {
    [size[0].div_ceil(8), size[1].div_ceil(8), 1]
}

fn main() -> Result<(), RhiError> {
    let (backend, frames) = common::arguments()?;
    let device = Device::open(DeviceOptions {
        backend,
        require_ray_tracing: false,
        ..DeviceOptions::default()
    })?;

    // One layout for all three shaders: an image to read, an image to
    // write, and a seed.
    let binding = |binding, descriptor_type| {
        DescriptorBinding::new(binding, descriptor_type, ShaderStages::COMPUTE)
    };
    let layout = device.create_pipeline_layout(
        PipelineLayoutDescriptor::new(&[DescriptorSetLayoutDescriptor::new(
            0,
            &[
                binding(0, DescriptorType::SampledImage),
                binding(1, DescriptorType::StorageImage),
            ],
        )])
        .with_push_constants(PushConstantRange::new(ShaderStages::COMPUTE, 16)),
    )?;
    let pipeline = |entry: &'static str, module: ShaderModule| {
        device.create_compute_pipeline(&ComputePipelineDescriptor::new(
            entry,
            &layout,
            module.entry(entry),
        ))
    };
    let seed = pipeline(
        "seed",
        shader!(&device, ShaderKind::Compute { entry: "seed" }, "life.seed")?,
    )?;
    let step = pipeline(
        "advance",
        shader!(
            &device,
            ShaderKind::Compute { entry: "advance" },
            "life.advance"
        )?,
    )?;
    let show = pipeline(
        "show",
        shader!(&device, ShaderKind::Compute { entry: "show" }, "life.show")?,
    )?;

    // The two boards, and a set for each way round: read one, write the
    // other.
    let board = || {
        device.create_image(ImageDescriptor::d2(
            ImageFormat::Rgba8Unorm,
            BOARD[0],
            BOARD[1],
            ImageUsages::SAMPLED | ImageUsages::STORAGE,
        ))
    };
    let boards = [board()?, board()?];
    let from_to = |from: &Image, to: &Image| -> Result<DescriptorSet, RhiError> {
        let mut set = device.create_descriptor_set(&layout, 0)?;
        set.write_sampled_image(0, &from.create_full_view()?)?;
        set.write_storage_image(1, &to.create_full_view()?)?;
        Ok(set)
    };
    let steps = [
        from_to(&boards[0], &boards[1])?,
        from_to(&boards[1], &boards[0])?,
    ];

    // Which board is the current one, and the sets that draw each to the
    // window's image.
    let mut current = 0;
    let mut shows: Option<[DescriptorSet; 2]> = None;
    let no_seed = [0_u8; 16];
    // How much time has passed that no generation has yet been made for.
    let mut owed = 0.0_f32;
    common::run(
        "erhi life (space or click: new board)",
        &device,
        ImageUsages::STORAGE,
        frames,
        |frame| {
            if frame.fresh {
                shows = Some([
                    from_to(&boards[0], frame.target)?,
                    from_to(&boards[1], frame.target)?,
                ]);
            }
            let shows = shows.as_ref().expect("bound on the first frame");

            owed += frame.delta;
            let mut encoder = device.begin_commands()?;
            if frame.index == 0 || frame.pressed {
                // A new board is written into the one that is not current,
                // which then is.
                let mut constants = no_seed;
                constants[..4].copy_from_slice(&(frame.index + 1).to_le_bytes());
                encoder.dispatch(
                    &DispatchDescriptor::new(&seed)
                        .with_sets(&[&steps[current]])
                        .with_push_constants(&constants)
                        .with_groups(groups(BOARD)),
                )?;
                current = 1 - current;
            } else if owed >= 1.0 / GENERATIONS_A_SECOND {
                // One generation a frame at most: a slow frame falls behind
                // rather than skipping ahead.
                owed = (owed - 1.0 / GENERATIONS_A_SECOND).min(1.0 / GENERATIONS_A_SECOND);
                encoder.dispatch(
                    &DispatchDescriptor::new(&step)
                        .with_sets(&[&steps[current]])
                        .with_push_constants(&no_seed)
                        .with_groups(groups(BOARD)),
                )?;
                current = 1 - current;
            }
            encoder.dispatch(
                &DispatchDescriptor::new(&show)
                    .with_sets(&[&shows[current]])
                    .with_push_constants(&no_seed)
                    .with_groups(groups([frame.width, frame.height])),
            )?;
            encoder.transition_image(frame.target, ResourceState::CopySource)?;
            encoder.finish()
        },
    )
}
