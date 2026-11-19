use crate::buffer::BufferBackend;
use crate::compute_pipeline::ComputePipelineBackend;
use crate::descriptor_set::DescriptorSetBackend;
use crate::image::ImageBackend;
#[cfg(feature = "vulkan")]
use crate::image::ImageViewBackend;
use crate::occlusion_query::OcclusionQuerySetBackend;
use crate::pipeline_layout::PipelineLayoutBackend;
#[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
use crate::ray_tracing_pipeline::RayTracingPipelineBackend;
use crate::render_pipeline::RenderPipelineBackend;
use crate::{
    Buffer, ComputePipeline, DescriptorSet, Image, ImageView, RayTracingPipeline, RenderPipeline,
    ResourceState, RhiError,
};

use crate::TimestampQuerySet;
use crate::acceleration_structure::AccelerationBuildBackend;
use crate::command_encoder::CommandEncoderBackend;
use crate::device::BackendDevice;
use crate::timestamp_query::TimestampQuerySetBackend;

/// Record the barriers that take mips of layers of `image` from one state
/// to another: each of `moves` is one of them by its index, the state it
/// is in and the one it is to be in. Where they are all of the image and
/// all alike, one barrier does it.
fn move_image(
    image: &Image,
    moves: &[(u32, ResourceState, ResourceState)],
    backend: &mut CommandEncoderBackend<'_>,
) -> Result<(), RhiError> {
    let Some(&(_, before, after)) = moves.first() else {
        return Ok(());
    };
    let descriptor = image.descriptor();
    let whole = moves.len() as u32 == image.subresource_count()
        && moves.iter().all(|step| step.1 == before && step.2 == after);
    let single = [(0, before, after)];
    let steps = if whole { &single[..] } else { moves };

    match (backend, &image.inner.backend) {
        #[cfg(feature = "vulkan")]
        (CommandEncoderBackend::Vulkan(encoder), ImageBackend::Vulkan(native)) => {
            for &(index, before, after) in steps {
                let at = (!whole)
                    .then_some((index % descriptor.mip_count, index / descriptor.mip_count));
                encoder.transition_image(native, descriptor, at, before, after)?;
            }
        }
        #[cfg(all(feature = "dx12", windows))]
        (CommandEncoderBackend::Dx12(encoder), ImageBackend::Dx12(native)) => {
            // A DX12 transition barrier between identical states is
            // invalid; a global UAV barrier provides the same
            // write-after-write ordering instead.
            let mut ordered = false;
            for &(index, before, after) in steps {
                if before == after {
                    if !ordered {
                        encoder.uav_barrier();
                        ordered = true;
                    }
                    continue;
                }
                // A stencil plane is its own subresource, after every
                // mip of every layer of the depth plane.
                let planes = if descriptor.format.has_stencil() {
                    2
                } else {
                    1
                };
                let subresources: Vec<u32> = if whole {
                    Vec::new()
                } else {
                    (0..planes)
                        .map(|plane| index + plane * image.subresource_count())
                        .collect()
                };
                encoder.transition_image(native, &subresources, before, after)?;
            }
        }
        #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
        _ => return Err(RhiError::ResourceBackendMismatch),
    }
    for &(index, _, after) in moves {
        image.set_state_of(index, after);
    }
    Ok(())
}

/// `buffer` as the Vulkan backend holds it.
#[cfg(feature = "vulkan")]
fn vulkan_buffer(buffer: &Buffer) -> Result<&crate::vulkan::VulkanBuffer, RhiError> {
    match &buffer.inner.backend {
        BufferBackend::Vulkan(native) => Ok(&**native),
        #[cfg(all(feature = "dx12", windows))]
        BufferBackend::Dx12(_) => Err(RhiError::ResourceBackendMismatch),
    }
}

/// `sets` as the Vulkan backend holds them.
#[cfg(feature = "vulkan")]
fn vulkan_sets(
    sets: &[DescriptorSet],
) -> Result<Vec<&crate::vulkan::VulkanDescriptorSet>, RhiError> {
    sets.iter()
        .map(|set| match set.native()? {
            DescriptorSetBackend::Vulkan(native) => Ok(&**native),
            #[cfg(all(feature = "dx12", windows))]
            DescriptorSetBackend::Dx12(_) => Err(RhiError::ResourceBackendMismatch),
        })
        .collect()
}

/// `sets` as the Direct3D 12 backend holds them.
#[cfg(all(feature = "dx12", windows))]
fn dx12_sets(sets: &[DescriptorSet]) -> Result<Vec<&crate::dx12::Dx12DescriptorSet>, RhiError> {
    sets.iter()
        .map(|set| match set.native()? {
            DescriptorSetBackend::Dx12(native) => Ok(&**native),
            #[cfg(feature = "vulkan")]
            DescriptorSetBackend::Vulkan(_) => Err(RhiError::ResourceBackendMismatch),
        })
        .collect()
}

/// `buffer` as the Direct3D 12 backend holds it.
#[cfg(all(feature = "dx12", windows))]
fn dx12_buffer(buffer: &Buffer) -> Result<&crate::dx12::Dx12Buffer, RhiError> {
    match &buffer.inner.backend {
        BufferBackend::Dx12(native) => Ok(&**native),
        #[cfg(feature = "vulkan")]
        BufferBackend::Vulkan(_) => Err(RhiError::ResourceBackendMismatch),
    }
}

/// An immutable packet retains every resource until its submission fence retires.
pub(crate) enum Command {
    CopyBuffer {
        source: Buffer,
        source_offset: u64,
        destination: Buffer,
        destination_offset: u64,
        size: u64,
    },
    TransitionBuffer {
        buffer: Buffer,
        before: ResourceState,
        after: ResourceState,
    },
    /// The first use of a buffer by an encoder: what an earlier submission
    /// wrote to it has landed before it is used as `after`. Vulkan needs
    /// telling; Direct3D 12 accepts a buffer from its common state to its
    /// first use of itself.
    SyncBuffer {
        buffer: Buffer,
        after: ResourceState,
    },
    TransitionImage {
        image: Image,
        state: ResourceState,
    },
    /// What the encoder records of its own accord before a command that
    /// uses an image: left alone if the image is in a state that serves
    /// (`accepted`), moved to `preferred` if not. Decided when the commands
    /// run, by the state the image is then in: when they were recorded,
    /// commands recorded earlier might not have run yet.
    UseImage {
        image: Image,
        range: crate::image::Subresources,
        accepted: crate::StateSet,
        preferred: ResourceState,
        /// The image was written through a storage binding since its last
        /// barrier: what reads or writes it next waits for that, though its
        /// state does not change.
        barrier: bool,
    },
    Dispatch {
        pipeline: ComputePipeline,
        descriptor_sets: Vec<DescriptorSet>,
        push_constants: Vec<u8>,
        groups: [u32; 3],
        /// The GPU-written group counts, and where they start.
        indirect: Option<(Buffer, u64)>,
    },
    CopyBufferToImageImpl {
        source: Buffer,
        source_offset: u64,
        destination: Image,
        mip: u32,
        aspect: crate::ImageAspect,
        rows: Option<std::ops::Range<u32>>,
        layout: crate::ImageCopyLayout,
    },
    CopyImageToBuffer {
        source: Image,
        mip: u32,
        aspect: crate::ImageAspect,
        destination: Buffer,
        destination_offset: u64,
        layout: crate::ImageCopyLayout,
    },
    CopyImage {
        source: Image,
        destination: Image,
    },
    TraceRays {
        pipeline: RayTracingPipeline,
        descriptor_sets: Vec<DescriptorSet>,
        width: u32,
        height: u32,
    },
    BeginRenderPass {
        /// Each colour attachment, the colour it is cleared to (loaded
        /// when `None`), and where its samples are resolved into.
        color: Vec<(ImageView, Option<[f32; 4]>, Option<ImageView>)>,
        /// The depth attachment, and what its depth and its stencil are
        /// cleared to (each loaded when `None`).
        depth: Option<(ImageView, Option<f32>, Option<u8>)>,
        extent: (u32, u32),
    },
    SetViewport {
        /// x, y, width, height, y from the top.
        rectangle: [f32; 4],
        /// Least and greatest depth.
        depth: [f32; 2],
    },
    Draw {
        pipeline: RenderPipeline,
        descriptor_sets: Vec<DescriptorSet>,
        /// One for each of the pipeline's vertex streams, in order.
        vertex_buffers: Vec<Buffer>,
        indices: Option<(Buffer, crate::IndexFormat)>,
        /// The GPU-written counts: where the first record starts, and how
        /// many records (draws) there are.
        indirect: Option<(Buffer, u64, u32)>,
        /// The first index, or the first vertex of a draw without indices.
        first: u32,
        count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
        push_constants: Vec<u8>,
    },
    SetScissor {
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
    },
    EndRenderPass {},
    ResetOcclusionQueries {
        query_set: crate::OcclusionQuerySet,
    },
    BeginOcclusionQuery {
        query_set: crate::OcclusionQuerySet,
        index: u32,
    },
    EndOcclusionQuery {
        query_set: crate::OcclusionQuerySet,
        index: u32,
    },
    ResolveOcclusionQueries {
        query_set: crate::OcclusionQuerySet,
        first: u32,
        count: u32,
        destination: Buffer,
        destination_offset: u64,
    },
    ResetTimestamps {
        query_set: TimestampQuerySet,
    },
    WriteTimestamp {
        query_set: TimestampQuerySet,
        index: u32,
    },
    ResolveTimestamps {
        query_set: TimestampQuerySet,
        first: u32,
        count: u32,
        destination: Buffer,
        destination_offset: u64,
    },
    Tlas {
        build: crate::acceleration_structure::AccelerationBuild,
    },
    BlasBatch {
        build: crate::acceleration_structure::BlasBatchBuild,
    },
    /// A third-party SDK's recording (`CommandEncoder::record_native`).
    Native {
        record: crate::NativeRecord,
        data: Vec<u8>,
        /// Kept alive until the submission retires.
        _images: Vec<Image>,
    },
}

impl Command {
    pub(crate) fn record(
        &self,
        device: &BackendDevice,
        backend: &mut CommandEncoderBackend<'_>,
    ) -> Result<(), RhiError> {
        match self {
            Self::CopyBuffer {
                source,
                source_offset,
                destination,
                destination_offset,
                size,
            } => {
                let source_offset = *source_offset;
                let destination_offset = *destination_offset;
                let size = *size;

                match (backend, &source.inner.backend, &destination.inner.backend) {
                    #[cfg(feature = "vulkan")]
                    (
                        CommandEncoderBackend::Vulkan(encoder),
                        BufferBackend::Vulkan(source),
                        BufferBackend::Vulkan(destination),
                    ) => encoder.copy_buffer(
                        source,
                        source_offset,
                        destination,
                        destination_offset,
                        size,
                    ),
                    #[cfg(all(feature = "dx12", windows))]
                    (
                        CommandEncoderBackend::Dx12(encoder),
                        BufferBackend::Dx12(source),
                        BufferBackend::Dx12(destination),
                    ) => encoder.copy_buffer(
                        source,
                        source_offset,
                        destination,
                        destination_offset,
                        size,
                    ),
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => Err(RhiError::ResourceBackendMismatch),
                }
            }
            Self::TransitionBuffer {
                buffer,
                before,
                after,
            } => {
                let before = *before;
                let after = *after;

                match (backend, &buffer.inner.backend) {
                    #[cfg(feature = "vulkan")]
                    (CommandEncoderBackend::Vulkan(encoder), BufferBackend::Vulkan(buffer)) => {
                        encoder.transition_buffer(buffer, before, after)
                    }
                    #[cfg(all(feature = "dx12", windows))]
                    (CommandEncoderBackend::Dx12(encoder), BufferBackend::Dx12(buffer)) => {
                        encoder.transition_buffer(buffer, before, after)
                    }
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => Err(RhiError::ResourceBackendMismatch),
                }
            }
            Self::SyncBuffer { buffer, after } => match (backend, &buffer.inner.backend) {
                #[cfg(feature = "vulkan")]
                (CommandEncoderBackend::Vulkan(encoder), BufferBackend::Vulkan(buffer)) => {
                    encoder.transition_buffer(buffer, ResourceState::Undefined, *after)
                }
                #[cfg(all(feature = "dx12", windows))]
                (CommandEncoderBackend::Dx12(_), BufferBackend::Dx12(_)) => {
                    // DX12 orders work on a buffer by its own state barriers.
                    let _ = after;
                    Ok(())
                }
                #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                _ => Err(RhiError::ResourceBackendMismatch),
            },
            Self::Native { record, data, .. } => {
                let list = match backend {
                    #[cfg(feature = "vulkan")]
                    CommandEncoderBackend::Vulkan(encoder) => {
                        use ash::vk::Handle;
                        crate::NativeCommandList::Vulkan(encoder.raw_command().as_raw())
                    }
                    #[cfg(all(feature = "dx12", windows))]
                    CommandEncoderBackend::Dx12(encoder) => {
                        use windows::core::Interface;
                        crate::NativeCommandList::Dx12(encoder.raw_list().as_raw() as usize)
                    }
                };
                record(list, data).map_err(|error| RhiError::NativeCallback(error.to_string()))
            }
            Self::UseImage {
                image,
                range,
                accepted,
                preferred,
                barrier,
            } => {
                let moves: Vec<_> = range
                    .indices(&image.descriptor())
                    .filter_map(|index| {
                        let before = image.state_of(index);
                        if !accepted.contains(before) {
                            Some((index, before, *preferred))
                        } else if *barrier && before.writes() {
                            // As it is, once what was written to it has
                            // landed.
                            Some((index, before, before))
                        } else {
                            None
                        }
                    })
                    .collect();
                move_image(image, &moves, backend)
            }
            Self::TransitionImage { image, state } => {
                let moves: Vec<_> = (0..image.subresource_count())
                    .map(|index| (index, image.state_of(index), *state))
                    .collect();
                move_image(image, &moves, backend)
            }
            Self::Dispatch {
                pipeline,
                descriptor_sets,
                push_constants,
                groups,
                indirect,
            } => {
                let push_constants = push_constants.as_slice();
                let groups = *groups;

                match (backend, &pipeline.backend) {
                    #[cfg(feature = "vulkan")]
                    (
                        CommandEncoderBackend::Vulkan(encoder),
                        ComputePipelineBackend::Vulkan(native),
                    ) => match pipeline.layout().native() {
                        PipelineLayoutBackend::Vulkan(layout) => {
                            let sets = vulkan_sets(descriptor_sets)?;
                            let arguments = match indirect {
                                None => None,
                                Some((buffer, offset)) => Some((vulkan_buffer(buffer)?, *offset)),
                            };
                            encoder.dispatch(
                                native,
                                layout,
                                &sets,
                                push_constants,
                                groups,
                                arguments,
                            )
                        }
                        #[cfg(all(feature = "dx12", windows))]
                        PipelineLayoutBackend::Dx12(_) => Err(RhiError::ResourceBackendMismatch),
                    },
                    #[cfg(all(feature = "dx12", windows))]
                    (
                        CommandEncoderBackend::Dx12(encoder),
                        ComputePipelineBackend::Dx12(native),
                    ) => match pipeline.layout().native() {
                        PipelineLayoutBackend::Dx12(layout) => {
                            let sets = dx12_sets(descriptor_sets)?;
                            let arguments = match indirect {
                                None => None,
                                Some((buffer, offset)) => Some((dx12_buffer(buffer)?, *offset)),
                            };
                            encoder.dispatch(
                                native,
                                layout,
                                &sets,
                                push_constants,
                                groups,
                                arguments,
                            )
                        }
                        #[cfg(feature = "vulkan")]
                        PipelineLayoutBackend::Vulkan(_) => Err(RhiError::ResourceBackendMismatch),
                    },
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => Err(RhiError::ResourceBackendMismatch),
                }
            }
            Self::CopyBufferToImageImpl {
                source,
                source_offset,
                destination,
                mip,
                aspect,
                rows,
                layout,
            } => {
                let source_offset = *source_offset;
                let mip = *mip;
                let aspect = *aspect;
                let rows = rows.clone();
                let layout = *layout;

                match (backend, &source.inner.backend, &destination.inner.backend) {
                    #[cfg(feature = "vulkan")]
                    (
                        CommandEncoderBackend::Vulkan(encoder),
                        BufferBackend::Vulkan(source),
                        ImageBackend::Vulkan(image),
                    ) => encoder.copy_buffer_to_image(
                        source,
                        source_offset,
                        image,
                        destination.descriptor(),
                        mip,
                        aspect,
                        layout,
                        rows,
                    ),
                    #[cfg(all(feature = "dx12", windows))]
                    (
                        CommandEncoderBackend::Dx12(encoder),
                        BufferBackend::Dx12(source),
                        ImageBackend::Dx12(image),
                    ) => encoder.copy_buffer_to_image(
                        source,
                        source_offset,
                        image,
                        destination.descriptor(),
                        mip,
                        aspect,
                        layout,
                        rows,
                    ),
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => Err(RhiError::ResourceBackendMismatch),
                }
            }
            Self::CopyImageToBuffer {
                source,
                mip,
                aspect,
                destination,
                destination_offset,
                layout,
            } => {
                let mip = *mip;
                let aspect = *aspect;
                let destination_offset = *destination_offset;
                let layout = *layout;

                match (backend, &source.inner.backend, &destination.inner.backend) {
                    #[cfg(feature = "vulkan")]
                    (
                        CommandEncoderBackend::Vulkan(encoder),
                        ImageBackend::Vulkan(image),
                        BufferBackend::Vulkan(destination_buffer),
                    ) => encoder.copy_image_to_buffer(
                        image,
                        source.descriptor(),
                        mip,
                        aspect,
                        destination_buffer,
                        destination_offset,
                        layout,
                    ),
                    #[cfg(all(feature = "dx12", windows))]
                    (
                        CommandEncoderBackend::Dx12(encoder),
                        ImageBackend::Dx12(image),
                        BufferBackend::Dx12(destination_buffer),
                    ) => encoder.copy_image_to_buffer(
                        image,
                        source.descriptor(),
                        mip,
                        aspect,
                        destination_buffer,
                        destination_offset,
                        layout,
                    ),
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => Err(RhiError::ResourceBackendMismatch),
                }
            }
            Self::CopyImage {
                source,
                destination,
            } => match (backend, &source.inner.backend, &destination.inner.backend) {
                #[cfg(feature = "vulkan")]
                (
                    CommandEncoderBackend::Vulkan(encoder),
                    ImageBackend::Vulkan(source_image),
                    ImageBackend::Vulkan(destination_image),
                ) => encoder.copy_image(source_image, destination_image, source.descriptor()),
                #[cfg(all(feature = "dx12", windows))]
                (
                    CommandEncoderBackend::Dx12(encoder),
                    ImageBackend::Dx12(source_image),
                    ImageBackend::Dx12(destination_image),
                ) => encoder.copy_image(source_image, destination_image),
                #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                _ => Err(RhiError::ResourceBackendMismatch),
            },
            Self::TraceRays {
                pipeline,
                descriptor_sets,
                width,
                height,
            } => {
                let width = *width;
                let height = *height;

                match (backend, &pipeline.backend) {
                    #[cfg(feature = "vulkan")]
                    (
                        CommandEncoderBackend::Vulkan(encoder),
                        RayTracingPipelineBackend::Vulkan(native),
                    ) => match pipeline.layout().native() {
                        PipelineLayoutBackend::Vulkan(layout) => {
                            let sets = vulkan_sets(descriptor_sets)?;
                            encoder.trace_rays(native, layout.handle(), &sets, width, height)
                        }
                        #[cfg(all(feature = "dx12", windows))]
                        PipelineLayoutBackend::Dx12(_) => Err(RhiError::ResourceBackendMismatch),
                    },
                    #[cfg(all(feature = "dx12", windows))]
                    (
                        CommandEncoderBackend::Dx12(encoder),
                        RayTracingPipelineBackend::Dx12(native),
                    ) => match pipeline.layout().native() {
                        PipelineLayoutBackend::Dx12(layout) => {
                            let sets = dx12_sets(descriptor_sets)?;
                            encoder.trace_rays(native, layout, &sets, width, height)
                        }
                        #[cfg(feature = "vulkan")]
                        PipelineLayoutBackend::Vulkan(_) => Err(RhiError::ResourceBackendMismatch),
                    },
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => Err(RhiError::ResourceBackendMismatch),
                }
            }
            Self::BeginRenderPass {
                color,
                depth,
                extent,
            } => {
                let extent = *extent;
                match backend {
                    #[cfg(feature = "vulkan")]
                    CommandEncoderBackend::Vulkan(encoder) => {
                        let mut colors = Vec::with_capacity(color.len());
                        for (view, clear, resolve) in color {
                            let resolve = match resolve.as_ref().map(|view| &view.inner.backend) {
                                None => None,
                                Some(ImageViewBackend::Vulkan(native)) => Some(&**native),
                                #[cfg(all(feature = "dx12", windows))]
                                Some(_) => return Err(RhiError::ResourceBackendMismatch),
                            };
                            match &view.inner.backend {
                                ImageViewBackend::Vulkan(native) => {
                                    colors.push((&**native, *clear, resolve))
                                }
                                #[cfg(all(feature = "dx12", windows))]
                                _ => return Err(RhiError::ResourceBackendMismatch),
                            }
                        }
                        let depth = match depth {
                            None => None,
                            Some((view, clear, stencil)) => match &view.inner.backend {
                                ImageViewBackend::Vulkan(native) => Some((
                                    &**native,
                                    *clear,
                                    *stencil,
                                    view.image().descriptor().format.has_stencil(),
                                )),
                                #[cfg(all(feature = "dx12", windows))]
                                _ => return Err(RhiError::ResourceBackendMismatch),
                            },
                        };
                        encoder.begin_render_pass(&colors, extent, depth);
                    }
                    #[cfg(all(feature = "dx12", windows))]
                    CommandEncoderBackend::Dx12(encoder) => {
                        /// A view's image, its own native record and the
                        /// image's descriptor.
                        fn native(
                            view: &ImageView,
                        ) -> Result<
                            (
                                &crate::dx12::Dx12Image,
                                &crate::dx12::Dx12ImageView,
                                crate::ImageDescriptor,
                            ),
                            RhiError,
                        > {
                            match (&view.image().inner.backend, &view.inner.backend) {
                                (
                                    ImageBackend::Dx12(image),
                                    crate::image::ImageViewBackend::Dx12(native),
                                ) => Ok((&**image, native, view.image().descriptor())),
                                #[cfg(feature = "vulkan")]
                                _ => Err(RhiError::ResourceBackendMismatch),
                            }
                        }
                        let mut colors = Vec::with_capacity(color.len());
                        for (view, clear, resolve) in color {
                            colors.push((
                                native(view)?,
                                *clear,
                                resolve.as_ref().map(native).transpose()?,
                            ));
                        }
                        let depth = match depth {
                            None => None,
                            Some((view, clear, stencil)) => Some((native(view)?, *clear, *stencil)),
                        };
                        encoder.begin_render_pass(&colors, extent, depth)?;
                    }
                }
                Ok(())
            }
            Self::SetViewport { rectangle, depth } => {
                match backend {
                    #[cfg(feature = "vulkan")]
                    CommandEncoderBackend::Vulkan(encoder) => {
                        encoder.set_viewport(*rectangle, *depth)
                    }
                    #[cfg(all(feature = "dx12", windows))]
                    CommandEncoderBackend::Dx12(encoder) => {
                        encoder.set_viewport(*rectangle, *depth)
                    }
                }
                Ok(())
            }
            Self::Draw {
                pipeline,
                descriptor_sets,
                vertex_buffers,
                indices,
                indirect,
                first,
                count,
                base_vertex,
                first_instance,
                instance_count,
                push_constants,
            } => {
                let push_constants = push_constants.as_slice();

                match (backend, &pipeline.backend) {
                    #[cfg(feature = "vulkan")]
                    (
                        CommandEncoderBackend::Vulkan(encoder),
                        RenderPipelineBackend::Vulkan(native),
                    ) => match pipeline.layout().native() {
                        PipelineLayoutBackend::Vulkan(layout) => {
                            let sets = vulkan_sets(descriptor_sets)?;
                            let vertices = vertex_buffers
                                .iter()
                                .map(vulkan_buffer)
                                .collect::<Result<Vec<_>, _>>()?;
                            let indices = match indices {
                                None => None,
                                Some((buffer, format)) => Some((vulkan_buffer(buffer)?, *format)),
                            };
                            let arguments = match indirect {
                                None => None,
                                Some((buffer, offset, draws)) => {
                                    Some((vulkan_buffer(buffer)?, *offset, *draws))
                                }
                            };
                            encoder.draw(
                                native,
                                layout,
                                &sets,
                                push_constants,
                                &vertices,
                                indices,
                                arguments,
                                *first,
                                *count,
                                *base_vertex,
                                *first_instance,
                                *instance_count,
                            );
                            Ok(())
                        }
                        #[cfg(all(feature = "dx12", windows))]
                        PipelineLayoutBackend::Dx12(_) => Err(RhiError::ResourceBackendMismatch),
                    },
                    #[cfg(all(feature = "dx12", windows))]
                    (CommandEncoderBackend::Dx12(encoder), RenderPipelineBackend::Dx12(native)) => {
                        match pipeline.layout().native() {
                            PipelineLayoutBackend::Dx12(layout) => {
                                let sets = dx12_sets(descriptor_sets)?;
                                // Each stream: where it is, its size and stride.
                                let mut vertices = Vec::with_capacity(vertex_buffers.len());
                                for (buffer, (stride, _)) in
                                    vertex_buffers.iter().zip(pipeline.vertex_streams())
                                {
                                    vertices.push((
                                        dx12_buffer(buffer)?.device_address()?,
                                        buffer.descriptor().size,
                                        *stride,
                                    ));
                                }
                                let indices = match indices {
                                    None => None,
                                    Some((buffer, format)) => Some((
                                        dx12_buffer(buffer)?.device_address()?,
                                        buffer.descriptor().size,
                                        *format,
                                    )),
                                };
                                let arguments = match indirect {
                                    None => None,
                                    Some((buffer, offset, draws)) => {
                                        Some((dx12_buffer(buffer)?, *offset, *draws))
                                    }
                                };
                                encoder.draw(
                                    native,
                                    layout,
                                    &sets,
                                    push_constants,
                                    &vertices,
                                    indices,
                                    arguments,
                                    *first,
                                    *count,
                                    *base_vertex,
                                    *first_instance,
                                    *instance_count,
                                );
                                Ok(())
                            }
                            #[cfg(feature = "vulkan")]
                            PipelineLayoutBackend::Vulkan(_) => {
                                Err(RhiError::ResourceBackendMismatch)
                            }
                        }
                    }
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => Err(RhiError::ResourceBackendMismatch),
                }
            }
            Self::SetScissor { x0, y0, x1, y1 } => {
                let x0 = *x0;
                let y0 = *y0;
                let x1 = *x1;
                let y1 = *y1;

                match backend {
                    #[cfg(feature = "vulkan")]
                    CommandEncoderBackend::Vulkan(encoder) => {
                        encoder.set_scissor(x0, y0, (x1 - x0) as u32, (y1 - y0) as u32);
                    }
                    #[cfg(all(feature = "dx12", windows))]
                    CommandEncoderBackend::Dx12(encoder) => {
                        encoder.set_scissor(x0, y0, (x1 - x0) as u32, (y1 - y0) as u32);
                    }
                }
                Ok(())
            }
            Self::EndRenderPass {} => {
                match backend {
                    #[cfg(feature = "vulkan")]
                    CommandEncoderBackend::Vulkan(encoder) => encoder.end_render_pass(),
                    #[cfg(all(feature = "dx12", windows))]
                    CommandEncoderBackend::Dx12(encoder) => encoder.end_render_pass(),
                }
                Ok(())
            }
            Self::ResetOcclusionQueries { query_set } => match (backend, &query_set.backend) {
                #[cfg(feature = "vulkan")]
                (CommandEncoderBackend::Vulkan(encoder), OcclusionQuerySetBackend::Vulkan(set)) => {
                    encoder.reset_timestamps(set, query_set.count());
                    Ok(())
                }
                #[cfg(all(feature = "dx12", windows))]
                (CommandEncoderBackend::Dx12(_), OcclusionQuerySetBackend::Dx12(_)) => Ok(()),
                #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                _ => Err(RhiError::ResourceBackendMismatch),
            },
            Self::BeginOcclusionQuery { query_set, index } => {
                match (backend, &query_set.backend) {
                    #[cfg(feature = "vulkan")]
                    (
                        CommandEncoderBackend::Vulkan(encoder),
                        OcclusionQuerySetBackend::Vulkan(set),
                    ) => encoder.begin_occlusion_query(set, *index),
                    #[cfg(all(feature = "dx12", windows))]
                    (CommandEncoderBackend::Dx12(encoder), OcclusionQuerySetBackend::Dx12(set)) => {
                        encoder.begin_occlusion_query(set, *index);
                    }
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => return Err(RhiError::ResourceBackendMismatch),
                }
                Ok(())
            }
            Self::EndOcclusionQuery { query_set, index } => {
                match (backend, &query_set.backend) {
                    #[cfg(feature = "vulkan")]
                    (
                        CommandEncoderBackend::Vulkan(encoder),
                        OcclusionQuerySetBackend::Vulkan(set),
                    ) => encoder.end_occlusion_query(set, *index),
                    #[cfg(all(feature = "dx12", windows))]
                    (CommandEncoderBackend::Dx12(encoder), OcclusionQuerySetBackend::Dx12(set)) => {
                        encoder.end_occlusion_query(set, *index);
                    }
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => return Err(RhiError::ResourceBackendMismatch),
                }
                Ok(())
            }
            Self::ResolveOcclusionQueries {
                query_set,
                first,
                count,
                destination,
                destination_offset,
            } => {
                match (backend, &query_set.backend) {
                    #[cfg(feature = "vulkan")]
                    (
                        CommandEncoderBackend::Vulkan(encoder),
                        OcclusionQuerySetBackend::Vulkan(set),
                    ) => encoder.resolve_timestamps(
                        set,
                        *first,
                        *count,
                        vulkan_buffer(destination)?,
                        *destination_offset,
                    ),
                    #[cfg(all(feature = "dx12", windows))]
                    (CommandEncoderBackend::Dx12(encoder), OcclusionQuerySetBackend::Dx12(set)) => {
                        encoder.resolve_occlusion_queries(
                            set,
                            *first,
                            *count,
                            dx12_buffer(destination)?,
                            *destination_offset,
                        );
                    }
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => return Err(RhiError::ResourceBackendMismatch),
                }
                Ok(())
            }
            Self::ResetTimestamps { query_set } => match (backend, &query_set.backend) {
                #[cfg(feature = "vulkan")]
                (CommandEncoderBackend::Vulkan(encoder), TimestampQuerySetBackend::Vulkan(set)) => {
                    encoder.reset_timestamps(set, query_set.count());
                    Ok(())
                }
                #[cfg(all(feature = "dx12", windows))]
                (CommandEncoderBackend::Dx12(encoder), TimestampQuerySetBackend::Dx12(set)) => {
                    let _ = (encoder, set);
                    Ok(())
                }
                #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                _ => Err(RhiError::ResourceBackendMismatch),
            },
            Self::WriteTimestamp { query_set, index } => {
                let index = *index;

                match (backend, &query_set.backend) {
                    #[cfg(feature = "vulkan")]
                    (
                        CommandEncoderBackend::Vulkan(encoder),
                        TimestampQuerySetBackend::Vulkan(set),
                    ) => {
                        encoder.write_timestamp(set, index);
                        Ok(())
                    }
                    #[cfg(all(feature = "dx12", windows))]
                    (CommandEncoderBackend::Dx12(encoder), TimestampQuerySetBackend::Dx12(set)) => {
                        encoder.write_timestamp(set, index);
                        Ok(())
                    }
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => Err(RhiError::ResourceBackendMismatch),
                }
            }
            Self::ResolveTimestamps {
                query_set,
                first,
                count,
                destination,
                destination_offset,
            } => {
                let first = *first;
                let count = *count;
                let destination_offset = *destination_offset;

                match (backend, &query_set.backend, &destination.inner.backend) {
                    #[cfg(feature = "vulkan")]
                    (
                        CommandEncoderBackend::Vulkan(encoder),
                        TimestampQuerySetBackend::Vulkan(set),
                        BufferBackend::Vulkan(buffer),
                    ) => {
                        encoder.resolve_timestamps(set, first, count, buffer, destination_offset);
                        Ok(())
                    }
                    #[cfg(all(feature = "dx12", windows))]
                    (
                        CommandEncoderBackend::Dx12(encoder),
                        TimestampQuerySetBackend::Dx12(set),
                        BufferBackend::Dx12(buffer),
                    ) => {
                        encoder.resolve_timestamps(set, first, count, buffer, destination_offset);
                        Ok(())
                    }
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => Err(RhiError::ResourceBackendMismatch),
                }
            }
            Self::BlasBatch { build } => match (device, backend, &build.backend) {
                #[cfg(feature = "vulkan")]
                (
                    BackendDevice::Vulkan(device),
                    CommandEncoderBackend::Vulkan(encoder),
                    crate::acceleration_structure::BlasBatchBackend::Vulkan(builds),
                ) => {
                    let builds: Vec<_> = builds.iter().collect();
                    device.record_blas_batch(encoder.raw_command(), &builds);
                    Ok(())
                }
                #[cfg(all(feature = "dx12", windows))]
                (
                    BackendDevice::Dx12(_),
                    CommandEncoderBackend::Dx12(encoder),
                    crate::acceleration_structure::BlasBatchBackend::Dx12(builds),
                ) => {
                    for build in builds.iter() {
                        encoder.uav_barrier();
                        encoder.build_acceleration_structure(&build.description())?;
                    }
                    Ok(())
                }
                #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                _ => Err(RhiError::ResourceBackendMismatch),
            },
            Self::Tlas { build } => match (device, backend, &build.backend) {
                #[cfg(feature = "vulkan")]
                (
                    BackendDevice::Vulkan(device),
                    CommandEncoderBackend::Vulkan(encoder),
                    AccelerationBuildBackend::Vulkan(build),
                ) => device.record_tlas(encoder.raw_command(), build),
                #[cfg(all(feature = "dx12", windows))]
                (
                    BackendDevice::Dx12(_),
                    CommandEncoderBackend::Dx12(encoder),
                    AccelerationBuildBackend::Dx12(build),
                ) => {
                    encoder.uav_barrier();
                    encoder.build_acceleration_structure(&build.description()?)
                }
                #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                _ => Err(RhiError::ResourceBackendMismatch),
            },
        }
    }
}
