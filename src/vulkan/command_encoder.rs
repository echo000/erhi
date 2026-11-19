use ash::vk;

use super::compute_pipeline::VulkanComputePipeline;
use super::image::VulkanImageView;
use super::pipeline_layout::VulkanPipelineLayout;
use super::ray_tracing_pipeline::VulkanRayTracingPipeline;
use super::render_pipeline::VulkanRenderPipeline;
use super::{CommandPool, VulkanBuffer, VulkanDescriptorSet, VulkanDevice, VulkanImage};
use crate::{
    ImageAspect, ImageCopyLayout, ImageDescriptor, ImageDimension, IndexFormat, ResourceState,
    RhiError,
};

pub(crate) struct VulkanCommandEncoder<'device> {
    device: &'device VulkanDevice,
    // `None` when a frame ring owns the pool; the synchronous path owns a
    // dedicated transient pool per encoder.
    _pool: Option<CommandPool>,
    command: vk::CommandBuffer,
}

impl VulkanDevice {
    pub(crate) fn begin_commands(&self) -> Result<VulkanCommandEncoder<'_>, RhiError> {
        let pool = CommandPool::new(&self.logical.handle, self.queue_family_index)?;
        let command = pool.allocate_primary()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer is fresh and owned by `pool`, which is
        // retained by the returned encoder until submission completes.
        unsafe {
            self.logical
                .handle
                .begin_command_buffer(command, &begin)
                .map_err(|result| RhiError::vk("vkBeginCommandBuffer", result))?;
        }
        Ok(VulkanCommandEncoder {
            device: self,
            _pool: Some(pool),
            command,
        })
    }

    /// Wrap a frame ring's already-recording command buffer in an encoder.
    pub(crate) fn encoder_from_frame(
        &self,
        command: vk::CommandBuffer,
    ) -> VulkanCommandEncoder<'_> {
        VulkanCommandEncoder {
            device: self,
            _pool: None,
            command,
        }
    }
}

impl VulkanCommandEncoder<'_> {
    /// The raw recording command buffer, for backend-internal recording that
    /// spans layers (the surface's frame-integrated present blit).
    pub(crate) fn raw_command(&self) -> vk::CommandBuffer {
        self.command
    }

    pub(crate) fn copy_buffer(
        &mut self,
        source: &VulkanBuffer,
        source_offset: u64,
        destination: &VulkanBuffer,
        destination_offset: u64,
        size: u64,
    ) -> Result<(), RhiError> {
        let region = vk::BufferCopy::default()
            .src_offset(source_offset)
            .dst_offset(destination_offset)
            .size(size);
        // SAFETY: the common layer validated usages and bounds. Both buffers
        // and this command buffer belong to the same Vulkan device.
        unsafe {
            self.device.logical.handle.cmd_copy_buffer(
                self.command,
                source.handle,
                destination.handle,
                &[region],
            );
        }
        Ok(())
    }

    pub(crate) fn transition_buffer(
        &mut self,
        buffer: &VulkanBuffer,
        before: ResourceState,
        after: ResourceState,
    ) -> Result<(), RhiError> {
        // A buffer whose state is not known was left however an earlier
        // submission left it: wait for whatever wrote to it.
        let (source_stage, source_access) = match before {
            ResourceState::Undefined => (
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::MEMORY_WRITE,
            ),
            _ => {
                let (stage, access, _) =
                    state_mapping(before, self.device.capabilities.ray_tracing);
                (stage, access)
            }
        };
        let (destination_stage, destination_access, _) =
            state_mapping(after, self.device.capabilities.ray_tracing);
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(source_access)
            .dst_access_mask(destination_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer.handle)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        // SAFETY: the buffer is live for this recording; states are the
        // caller's explicit statement (buffers carry no tracked state).
        unsafe {
            self.device.logical.handle.cmd_pipeline_barrier(
                self.command,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }
        Ok(())
    }

    pub(crate) fn transition_image(
        &mut self,
        image: &VulkanImage,
        descriptor: ImageDescriptor,
        at: Option<(u32, u32)>,
        before: ResourceState,
        after: ResourceState,
    ) -> Result<(), RhiError> {
        // One mip of one layer, or all of them.
        let (base_mip, mips, base_layer, layers) = match at {
            Some((mip, layer)) => (mip, 1, layer, 1),
            None => (0, descriptor.mip_count, 0, descriptor.layer_count()),
        };
        let (source_stage, source_access, old_layout) =
            state_mapping(before, self.device.capabilities.ray_tracing);
        let (destination_stage, destination_access, new_layout) =
            state_mapping(after, self.device.capabilities.ray_tracing);
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(source_access)
            .dst_access_mask(destination_access)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image.handle)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(super::image::vk_aspects(descriptor.format))
                    .base_mip_level(base_mip)
                    .level_count(mips)
                    .base_array_layer(base_layer)
                    .layer_count(layers),
            );
        // SAFETY: the image is live for this recording and the transition was
        // derived from the common layer's tracked state.
        unsafe {
            self.device.logical.handle.cmd_pipeline_barrier(
                self.command,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy_buffer_to_image(
        &mut self,
        source: &VulkanBuffer,
        source_offset: u64,
        destination: &VulkanImage,
        descriptor: ImageDescriptor,
        mip: u32,
        aspect: ImageAspect,
        layout: ImageCopyLayout,
        rows: Option<std::ops::Range<u32>>,
    ) -> Result<(), RhiError> {
        let mut region = buffer_image_region(source_offset, descriptor, mip, aspect, layout);
        if let Some(rows) = rows {
            region.image_offset.y = rows.start as i32;
            region.image_extent.height = rows.end - rows.start;
        }
        // SAFETY: the common layer validated usages, states, and bounds. Both
        // resources and this command buffer belong to the same Vulkan device.
        unsafe {
            self.device.logical.handle.cmd_copy_buffer_to_image(
                self.command,
                source.handle,
                destination.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy_image_to_buffer(
        &mut self,
        source: &VulkanImage,
        descriptor: ImageDescriptor,
        mip: u32,
        aspect: ImageAspect,
        destination: &VulkanBuffer,
        destination_offset: u64,
        layout: ImageCopyLayout,
    ) -> Result<(), RhiError> {
        let region = buffer_image_region(destination_offset, descriptor, mip, aspect, layout);
        // SAFETY: the common layer validated usages, states, and bounds. Both
        // resources and this command buffer belong to the same Vulkan device.
        unsafe {
            self.device.logical.handle.cmd_copy_image_to_buffer(
                self.command,
                source.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                destination.handle,
                &[region],
            );
        }
        Ok(())
    }

    pub(crate) fn copy_image(
        &mut self,
        source: &VulkanImage,
        destination: &VulkanImage,
        descriptor: ImageDescriptor,
    ) -> Result<(), RhiError> {
        let regions: Vec<vk::ImageCopy> = (0..descriptor.mip_count)
            .map(|mip| {
                let (width, height, depth) = descriptor.mip_extent(mip);
                let subresource = vk::ImageSubresourceLayers {
                    aspect_mask: super::image::vk_aspects(descriptor.format),
                    mip_level: mip,
                    base_array_layer: 0,
                    layer_count: descriptor.layer_count(),
                };
                vk::ImageCopy {
                    src_subresource: subresource,
                    src_offset: vk::Offset3D::default(),
                    dst_subresource: subresource,
                    dst_offset: vk::Offset3D::default(),
                    extent: vk::Extent3D {
                        width,
                        height,
                        depth,
                    },
                }
            })
            .collect();
        // SAFETY: the common layer validated usages, states, and matching
        // descriptors. Both images and this command buffer belong to the
        // same Vulkan device.
        unsafe {
            self.device.logical.handle.cmd_copy_image(
                self.command,
                source.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                destination.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }
        Ok(())
    }

    /// Bind `sets` as the sets of `layout`, from its first.
    ///
    /// # Safety
    /// The sets were made for a layout that declares them as `layout`
    /// does, and outlive the recording.
    unsafe fn bind_sets(
        &self,
        point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        sets: &[&VulkanDescriptorSet],
    ) {
        if sets.is_empty() {
            return;
        }
        let mut handles = [vk::DescriptorSet::null(); crate::MAX_DESCRIPTOR_SETS];
        for (handle, set) in handles.iter_mut().zip(sets) {
            *handle = set.handle;
        }
        // SAFETY: per this function's contract.
        unsafe {
            self.device.logical.handle.cmd_bind_descriptor_sets(
                self.command,
                point,
                layout,
                0,
                &handles[..sets.len()],
                &[],
            );
        }
    }

    pub(crate) fn trace_rays(
        &mut self,
        pipeline: &VulkanRayTracingPipeline,
        layout: vk::PipelineLayout,
        descriptor_sets: &[&VulkanDescriptorSet],
        width: u32,
        height: u32,
    ) -> Result<(), RhiError> {
        // SAFETY: the pipeline, layout, and descriptor set are live and were
        // validated against this device by the common layer; the SBT regions
        // were computed at pipeline creation.
        unsafe {
            self.device.logical.handle.cmd_bind_pipeline(
                self.command,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.pipeline,
            );
            self.bind_sets(
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                layout,
                descriptor_sets,
            );
            self.device.ray_tracing_pipeline.cmd_trace_rays(
                self.command,
                &pipeline.raygen_region,
                &pipeline.miss_region,
                &pipeline.hit_region,
                &vk::StridedDeviceAddressRegionKHR::default(),
                width,
                height,
                1,
            );
        }
        Ok(())
    }

    pub(crate) fn dispatch(
        &mut self,
        pipeline: &VulkanComputePipeline,
        layout: &VulkanPipelineLayout,
        descriptor_sets: &[&VulkanDescriptorSet],
        push_constants: &[u8],
        groups: [u32; 3],
        indirect: Option<(&VulkanBuffer, u64)>,
    ) -> Result<(), RhiError> {
        // SAFETY: the pipeline, layout, and descriptor set are live and were
        // validated against this device by the common layer.
        unsafe {
            self.device.logical.handle.cmd_bind_pipeline(
                self.command,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );
            self.bind_sets(
                vk::PipelineBindPoint::COMPUTE,
                layout.handle(),
                descriptor_sets,
            );
            if !push_constants.is_empty() {
                self.device.logical.handle.cmd_push_constants(
                    self.command,
                    layout.handle(),
                    layout.push_constant_stages(),
                    0,
                    push_constants,
                );
            }
            match indirect {
                Some((buffer, offset)) => {
                    self.device.logical.handle.cmd_dispatch_indirect(
                        self.command,
                        buffer.handle,
                        offset,
                    );
                }
                None => {
                    self.device.logical.handle.cmd_dispatch(
                        self.command,
                        groups[0],
                        groups[1],
                        groups[2],
                    );
                }
            }
        }
        Ok(())
    }

    /// Begin a dynamic-rendering pass: each colour view with the colour it
    /// is cleared to and the view its samples are resolved into, and the
    /// depth view with what its depth and stencil are cleared to and whether
    /// its format has a stencil plane.
    #[allow(clippy::type_complexity)]
    pub(crate) fn begin_render_pass(
        &mut self,
        color: &[(&VulkanImageView, Option<[f32; 4]>, Option<&VulkanImageView>)],
        extent: (u32, u32),
        depth: Option<(&VulkanImageView, Option<f32>, Option<u8>, bool)>,
    ) {
        let load = |clear: bool| {
            if clear {
                vk::AttachmentLoadOp::CLEAR
            } else {
                vk::AttachmentLoadOp::LOAD
            }
        };
        let attachments: Vec<_> = color
            .iter()
            .map(|(view, clear, resolve)| {
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(view.handle)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(load(clear.is_some()))
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: clear.unwrap_or_default(),
                        },
                    });
                match resolve {
                    Some(resolve) => attachment
                        .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                        .resolve_image_view(resolve.handle)
                        .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                    None => attachment,
                }
            })
            .collect();
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: extent.0,
                height: extent.1,
            },
        };
        let plane = |view: &VulkanImageView, clear: bool, depth: f32, stencil: u8| {
            vk::RenderingAttachmentInfo::default()
                .image_view(view.handle)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(load(clear))
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth,
                        stencil: u32::from(stencil),
                    },
                })
        };
        // One view serves both planes; each has its own load operation.
        let depth_attachment = depth.map(|(view, clear, stencil, _)| {
            plane(
                view,
                clear.is_some(),
                clear.unwrap_or(1.0),
                stencil.unwrap_or(0),
            )
        });
        let stencil_attachment =
            depth
                .filter(|(_, _, _, has_stencil)| *has_stencil)
                .map(|(view, clear, stencil, _)| {
                    plane(
                        view,
                        stencil.is_some(),
                        clear.unwrap_or(1.0),
                        stencil.unwrap_or(0),
                    )
                });
        let mut rendering = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(&attachments);
        if let Some(attachment) = &depth_attachment {
            rendering = rendering.depth_attachment(attachment);
        }
        if let Some(attachment) = &stencil_attachment {
            rendering = rendering.stencil_attachment(attachment);
        }
        // SAFETY: the views are live, the images were transitioned to their
        // attachment states by the common layer, and dynamic rendering is an
        // enabled device feature.
        unsafe {
            self.device
                .dynamic_rendering
                .cmd_begin_rendering(self.command, &rendering);
            self.device
                .logical
                .handle
                .cmd_set_scissor(self.command, 0, &[area]);
        }
        self.set_viewport([0.0, 0.0, extent.0 as f32, extent.1 as f32], [0.0, 1.0]);
    }

    /// `rectangle` is x, y, width, height with y from the top.
    pub(crate) fn set_viewport(&mut self, rectangle: [f32; 4], depth: [f32; 2]) {
        let [x, y, width, height] = rectangle;
        // SAFETY: a dynamic-rendering pass is open; the viewport is a
        // declared dynamic state on every raster pipeline.
        unsafe {
            // Negative viewport height (maintenance1, core 1.1): Vulkan clip
            // space then matches D3D's y-up NDC, so one shader and one set of
            // vertex data draw identically on both backends.
            self.device.logical.handle.cmd_set_viewport(
                self.command,
                0,
                &[vk::Viewport {
                    x,
                    y: y + height,
                    width,
                    height: -height,
                    min_depth: depth[0],
                    max_depth: depth[1],
                }],
            );
        }
    }

    /// One draw: `vertex_buffers` bound from binding 0 up, then indexed
    /// (with the GPU's own counts where `indirect` names them) or not.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw(
        &mut self,
        pipeline: &VulkanRenderPipeline,
        layout: &VulkanPipelineLayout,
        descriptor_sets: &[&VulkanDescriptorSet],
        push_constants: &[u8],
        vertex_buffers: &[&VulkanBuffer],
        indices: Option<(&VulkanBuffer, IndexFormat)>,
        indirect: Option<(&VulkanBuffer, u64, u32)>,
        first: u32,
        count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
    ) {
        let mut handles = [vk::Buffer::null(); crate::MAX_VERTEX_BUFFERS];
        for (handle, buffer) in handles.iter_mut().zip(vertex_buffers) {
            *handle = buffer.handle;
        }
        let streams = vertex_buffers.len();
        // SAFETY: every resource is live and validated by the common layer;
        // a render pass is open per the common encoder state.
        unsafe {
            self.device.logical.handle.cmd_bind_pipeline(
                self.command,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            );
            self.bind_sets(
                vk::PipelineBindPoint::GRAPHICS,
                layout.handle(),
                descriptor_sets,
            );
            if !push_constants.is_empty() {
                self.device.logical.handle.cmd_push_constants(
                    self.command,
                    layout.handle(),
                    layout.push_constant_stages(),
                    0,
                    push_constants,
                );
            }
            if streams != 0 {
                self.device.logical.handle.cmd_bind_vertex_buffers(
                    self.command,
                    0,
                    &handles[..streams],
                    &[0; crate::MAX_VERTEX_BUFFERS][..streams],
                );
            }
            // Many records in one call where the device accepts them, else
            // one call for each: the same draws either way.
            let calls = |draws: u32, stride: u64| {
                let together = self.device.multi_draw_indirect || draws == 1;
                (0..if together { 1 } else { draws })
                    .map(move |call| (u64::from(call) * stride, if together { draws } else { 1 }))
            };
            let Some((index_buffer, format)) = indices else {
                if let Some((buffer, offset, draws)) = indirect {
                    for (skip, records) in calls(draws, 16) {
                        self.device.logical.handle.cmd_draw_indirect(
                            self.command,
                            buffer.handle,
                            offset + skip,
                            records,
                            16,
                        );
                    }
                } else {
                    self.device.logical.handle.cmd_draw(
                        self.command,
                        count,
                        instance_count,
                        first,
                        first_instance,
                    );
                }
                return;
            };
            self.device.logical.handle.cmd_bind_index_buffer(
                self.command,
                index_buffer.handle,
                0,
                match format {
                    IndexFormat::Uint16 => vk::IndexType::UINT16,
                    IndexFormat::Uint32 => vk::IndexType::UINT32,
                },
            );
            if let Some((buffer, offset, draws)) = indirect {
                for (skip, records) in calls(draws, 20) {
                    self.device.logical.handle.cmd_draw_indexed_indirect(
                        self.command,
                        buffer.handle,
                        offset + skip,
                        records,
                        20,
                    );
                }
            } else {
                self.device.logical.handle.cmd_draw_indexed(
                    self.command,
                    count,
                    instance_count,
                    first,
                    base_vertex,
                    first_instance,
                );
            }
        }
    }

    pub(crate) fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        // SAFETY: a dynamic-rendering pass is open; scissor is a declared
        // dynamic state on every raster pipeline.
        unsafe {
            self.device.logical.handle.cmd_set_scissor(
                self.command,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x, y },
                    extent: vk::Extent2D { width, height },
                }],
            );
        }
    }

    pub(crate) fn end_render_pass(&mut self) {
        // SAFETY: a dynamic rendering pass is open per the common encoder
        // state.
        unsafe {
            self.device
                .dynamic_rendering
                .cmd_end_rendering(self.command);
        }
    }

    pub(crate) fn reset_timestamps(&mut self, set: &super::VulkanTimestampQuerySet, count: u32) {
        // SAFETY: the query pool is live and the reset covers its full range.
        unsafe {
            self.device
                .logical
                .handle
                .cmd_reset_query_pool(self.command, set.handle, 0, count);
        }
    }

    pub(crate) fn begin_occlusion_query(
        &mut self,
        set: &super::VulkanTimestampQuerySet,
        index: u32,
    ) {
        let flags = if self.device.occlusion_query_precise {
            vk::QueryControlFlags::PRECISE
        } else {
            vk::QueryControlFlags::empty()
        };
        // SAFETY: the pool is live, the slot was validated by the common
        // layer and reset earlier in the recording, and a pass is open.
        unsafe {
            self.device
                .logical
                .handle
                .cmd_begin_query(self.command, set.handle, index, flags);
        }
    }

    pub(crate) fn end_occlusion_query(&mut self, set: &super::VulkanTimestampQuerySet, index: u32) {
        // SAFETY: this slot's query was begun in the same pass.
        unsafe {
            self.device
                .logical
                .handle
                .cmd_end_query(self.command, set.handle, index);
        }
    }

    pub(crate) fn write_timestamp(&mut self, set: &super::VulkanTimestampQuerySet, index: u32) {
        // SAFETY: the query pool is live and the slot index was validated by
        // the common layer. BOTTOM_OF_PIPE matches DX12's end-of-pipe
        // timestamp semantics.
        unsafe {
            self.device.logical.handle.cmd_write_timestamp(
                self.command,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                set.handle,
                index,
            );
        }
    }

    pub(crate) fn resolve_timestamps(
        &mut self,
        set: &super::VulkanTimestampQuerySet,
        first: u32,
        count: u32,
        destination: &VulkanBuffer,
        destination_offset: u64,
    ) {
        // SAFETY: the pool and buffer are live; the range and destination
        // were validated by the common layer. WAIT makes the copy block on
        // slots written earlier in this recording, matching DX12's resolve.
        unsafe {
            self.device.logical.handle.cmd_copy_query_pool_results(
                self.command,
                set.handle,
                first,
                count,
                destination.handle,
                destination_offset,
                crate::timestamp_query::TIMESTAMP_SIZE,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            );
        }
    }

    pub(crate) fn finish(self) -> Result<(), RhiError> {
        // SAFETY: recording is active, and the command pool remains alive
        // through the synchronous queue wait.
        unsafe {
            self.device
                .logical
                .handle
                .end_command_buffer(self.command)
                .map_err(|result| RhiError::vk("vkEndCommandBuffer", result))?;
            let commands = [self.command];
            let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
            let _queue = crate::sync::lock_anyway(&self.device.queue_lock);
            self.device
                .logical
                .handle
                .queue_submit(self.device.queue, &submits, vk::Fence::null())
                .map_err(|result| RhiError::vk("vkQueueSubmit", result))?;
            self.device
                .logical
                .handle
                .queue_wait_idle(self.device.queue)
                .map_err(|result| RhiError::vk("vkQueueWaitIdle", result))?;
        }
        Ok(())
    }
}

/// Stage, access, and layout for each resource state.
/// The stages, the accesses and the image layout of `state`. `ray_tracing`:
/// whether the device has ray tracing shaders, which are then among the
/// shaders that read and write.
pub(crate) fn state_mapping(
    state: ResourceState,
    ray_tracing: bool,
) -> (vk::PipelineStageFlags, vk::AccessFlags, vk::ImageLayout) {
    let shaders = vk::PipelineStageFlags::VERTEX_SHADER
        | vk::PipelineStageFlags::FRAGMENT_SHADER
        | vk::PipelineStageFlags::COMPUTE_SHADER
        | if ray_tracing {
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR
        } else {
            vk::PipelineStageFlags::empty()
        };
    match state {
        ResourceState::Undefined => (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::empty(),
            vk::ImageLayout::UNDEFINED,
        ),
        ResourceState::VertexBuffer => (
            vk::PipelineStageFlags::VERTEX_INPUT,
            vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            vk::ImageLayout::GENERAL,
        ),
        ResourceState::IndexBuffer => (
            vk::PipelineStageFlags::VERTEX_INPUT,
            vk::AccessFlags::INDEX_READ,
            vk::ImageLayout::GENERAL,
        ),
        ResourceState::ConstantBuffer => (
            shaders,
            vk::AccessFlags::UNIFORM_READ,
            vk::ImageLayout::GENERAL,
        ),
        ResourceState::IndirectArgument => (
            vk::PipelineStageFlags::DRAW_INDIRECT,
            vk::AccessFlags::INDIRECT_COMMAND_READ,
            vk::ImageLayout::GENERAL,
        ),
        // Shader image reads use the GENERAL layout everywhere in this device,
        // matching the descriptor writes.
        ResourceState::ShaderResource => (
            shaders,
            vk::AccessFlags::SHADER_READ,
            vk::ImageLayout::GENERAL,
        ),
        ResourceState::UnorderedAccess => (
            shaders,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            vk::ImageLayout::GENERAL,
        ),
        ResourceState::RenderTarget => (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ),
        ResourceState::DepthWrite => (
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ),
        ResourceState::CopySource => (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ),
        ResourceState::CopyDest => (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ),
        ResourceState::GenericRead => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ,
            vk::ImageLayout::GENERAL,
        ),
    }
}

/// One full-mip buffer/image copy region using the shared 256-byte row pitch.
fn buffer_image_region(
    buffer_offset: u64,
    descriptor: ImageDescriptor,
    mip: u32,
    aspect: ImageAspect,
    layout: ImageCopyLayout,
) -> vk::BufferImageCopy {
    let (width, height, depth) = descriptor.mip_extent(mip);
    // (The common layer refuses the planes that are not copied, so there is
    // a size; and a row of the widest image is far short of 4 GiB.)
    let block_bytes = descriptor.format.copy_bytes(aspect).unwrap_or(1).max(1);
    let row_length_texels = (layout.row_pitch / u64::from(block_bytes)
        * u64::from(descriptor.format.block_extent())) as u32;
    let layer_count = match descriptor.dimension {
        ImageDimension::D2 => descriptor.depth_or_layers,
        ImageDimension::D3 => 1,
    };
    vk::BufferImageCopy::default()
        .buffer_offset(buffer_offset)
        .buffer_row_length(row_length_texels)
        .buffer_image_height(layout.rows_per_slice * descriptor.format.block_extent())
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                // One plane at a time: `All` is only ever a format of one.
                .aspect_mask(match aspect {
                    ImageAspect::All => super::image::vk_aspects(descriptor.format),
                    ImageAspect::Depth => vk::ImageAspectFlags::DEPTH,
                    ImageAspect::Stencil => vk::ImageAspectFlags::STENCIL,
                })
                .mip_level(mip)
                .base_array_layer(0)
                .layer_count(layer_count),
        )
        .image_extent(vk::Extent3D {
            width,
            height,
            depth,
        })
}
