use std::mem::ManuallyDrop;
use std::sync::atomic::Ordering;

use windows::Win32::Foundation::{CloseHandle, HANDLE, RECT, WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC, D3D12_COMMAND_LIST_TYPE_DIRECT,
    D3D12_DISPATCH_RAYS_DESC, D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE,
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_QUERY_TYPE_TIMESTAMP, D3D12_RESOURCE_BARRIER,
    D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_BARRIER_TYPE_UAV, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    D3D12_RESOURCE_STATES, D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_RESOURCE_UAV_BARRIER,
    D3D12_SUBRESOURCE_FOOTPRINT, D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    D3D12_VERTEX_BUFFER_VIEW, D3D12_VIEWPORT, ID3D12CommandAllocator, ID3D12CommandList,
    ID3D12DeviceRemovedExtendedData, ID3D12GraphicsCommandList4, ID3D12Resource,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

/// Bounded queue wait: long enough for any real submission, short enough to
/// surface a hung or removed device as an error.
const QUEUE_WAIT_MILLISECONDS: u32 = 10_000;

use windows::core::Interface;

use super::compute_pipeline::Dx12ComputePipeline;
use super::ray_tracing_pipeline::Dx12RayTracingPipeline;
use super::{Dx12Buffer, Dx12DescriptorSet, Dx12Device, Dx12Image, Dx12PipelineLayout};
use crate::{ImageAspect, ImageCopyLayout, ImageDescriptor, IndexFormat, ResourceState, RhiError};

pub(crate) struct Dx12CommandEncoder<'device> {
    device: &'device Dx12Device,
    // `None` when a frame ring owns the allocator; the synchronous path owns
    // a dedicated allocator per encoder.
    _allocator: Option<ID3D12CommandAllocator>,
    command_list: ID3D12GraphicsCommandList4,
    /// The multisampled attachments of the open pass to average into their
    /// resolve targets when it ends.
    resolves: Vec<PendingResolve>,
}

struct PendingResolve {
    source: ID3D12Resource,
    target: ID3D12Resource,
    target_subresource: u32,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
}

#[derive(Clone, Copy)]
enum CopyDirection {
    BufferToImage,
    ImageToBuffer,
}

/// The Direct3D 12 resource state for each `ResourceState`.
fn dx12_state(state: ResourceState) -> D3D12_RESOURCE_STATES {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_INDEX_BUFFER,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
    };

    match state {
        ResourceState::Undefined => D3D12_RESOURCE_STATE_COMMON,
        ResourceState::VertexBuffer | ResourceState::ConstantBuffer => {
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
        }
        ResourceState::IndexBuffer => D3D12_RESOURCE_STATE_INDEX_BUFFER,
        ResourceState::IndirectArgument => D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        // Read by pixel shaders and by every other kind: Direct3D 12 tells
        // the two apart, and a shader resource is both.
        ResourceState::ShaderResource => {
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        }
        ResourceState::UnorderedAccess => D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ResourceState::RenderTarget => D3D12_RESOURCE_STATE_RENDER_TARGET,
        ResourceState::DepthWrite => D3D12_RESOURCE_STATE_DEPTH_WRITE,
        ResourceState::CopySource => D3D12_RESOURCE_STATE_COPY_SOURCE,
        ResourceState::CopyDest => D3D12_RESOURCE_STATE_COPY_DEST,
        ResourceState::GenericRead => D3D12_RESOURCE_STATE_GENERIC_READ,
    }
}

struct OwnedEvent(HANDLE);

impl OwnedEvent {
    fn new() -> Result<Self, RhiError> {
        // SAFETY: no security descriptor or name is supplied; Windows returns
        // an independently owned auto-reset event handle.
        let handle = unsafe { CreateEventW(None, false, false, None) }
            .map_err(|error| RhiError::dx12("CreateEventW", error))?;
        Ok(Self(handle))
    }
}

impl Drop for OwnedEvent {
    fn drop(&mut self) {
        // SAFETY: this owner holds the only raw event handle and drops after
        // every wait has completed.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

impl Dx12Device {
    pub(crate) fn begin_commands(&self) -> Result<Dx12CommandEncoder<'_>, RhiError> {
        // SAFETY: both objects are created from the live device and use the
        // same direct command-list type as the queue.
        let allocator: ID3D12CommandAllocator = unsafe {
            self.native
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(|error| RhiError::dx12("ID3D12Device::CreateCommandAllocator", error))?
        };
        // SAFETY: the allocator is live, the initial pipeline state is null,
        // and the requested interface derives from a direct graphics list.
        let command_list: ID3D12GraphicsCommandList4 = unsafe {
            self.native
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .map_err(|error| RhiError::dx12("ID3D12Device::CreateCommandList", error))?
        };
        Ok(Dx12CommandEncoder {
            device: self,
            _allocator: Some(allocator),
            command_list,
            resolves: Vec::new(),
        })
    }

    /// Wrap a frame ring's already-open command list in an encoder.
    pub(crate) fn encoder_from_frame(
        &self,
        command_list: ID3D12GraphicsCommandList4,
    ) -> Dx12CommandEncoder<'_> {
        Dx12CommandEncoder {
            device: self,
            _allocator: None,
            command_list,
            resolves: Vec::new(),
        }
    }

    /// Signal the shared direct-queue fence with the next monotonic value.
    pub(crate) fn signal_fence(&self) -> Result<u64, RhiError> {
        // Allocate and signal in the same order across render and upload threads.
        let _signal = crate::sync::lock_anyway(&self.fence_signal_lock);
        let value = self.next_fence_value.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the fence and queue belong to the same live device.
        unsafe {
            self.direct_queue
                .Signal(&self.direct_fence, value)
                .map_err(|error| RhiError::dx12("ID3D12CommandQueue::Signal", error))?;
        }
        Ok(value)
    }

    /// Block until the shared fence reaches `value`, waiting in bounded
    /// slices so a removed device surfaces as an error, never a hang.
    pub(crate) fn wait_for_fence_value(&self, value: u64) -> Result<(), RhiError> {
        // SAFETY: reading the completed value has no side effects.
        if unsafe { self.direct_fence.GetCompletedValue() } >= value {
            return Ok(());
        }
        let event = OwnedEvent::new()?;
        // SAFETY: the event remains alive until the wait returns, and the fence
        // remains owned by the device for the entire call.
        unsafe {
            self.direct_fence
                .SetEventOnCompletion(value, event.0)
                .map_err(|error| RhiError::dx12("ID3D12Fence::SetEventOnCompletion", error))?;
            loop {
                let result = WaitForSingleObject(event.0, QUEUE_WAIT_MILLISECONDS);
                if result == WAIT_OBJECT_0 {
                    break;
                }
                if result != WAIT_TIMEOUT {
                    return Err(RhiError::BackendWaitFailed {
                        backend: "dx12",
                        result: result.0,
                    });
                }
                if let Err(removed) = self.native.GetDeviceRemovedReason() {
                    return Err(self.device_removed_error(removed));
                }
            }
        }
        Ok(())
    }

    /// Build a device-removed error, enriched with DRED auto-breadcrumb
    /// context when the debug tooling captured it.
    fn device_removed_error(&self, removed: windows::core::Error) -> RhiError {
        device_removed_report(&self.native, "device removed during queue wait", removed)
    }

    fn wait_for_direct_queue(&self) -> Result<(), RhiError> {
        let value = self.signal_fence()?;
        self.wait_for_fence_value(value)
    }
}

/// A device-removed `RhiError` with the fullest diagnosis available: the
/// removal reason, and â€” when DRED was enabled at device creation
/// (`DeviceOptions::debug`) â€” the auto-breadcrumb summary and the names of
/// the deepest incomplete operations, which is the closest DX12 gets to
/// naming the faulting command.
pub(crate) fn device_removed_report(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device5,
    operation: &'static str,
    removed: windows::core::Error,
) -> RhiError {
    let mut message = removed.to_string();
    if let Ok(dred) = device.cast::<ID3D12DeviceRemovedExtendedData>() {
        // SAFETY: the node chain lives as long as the removed device.
        unsafe {
            if let Ok(breadcrumbs) = dred.GetAutoBreadcrumbsOutput() {
                let mut nodes = 0_u32;
                let mut incomplete = 0_u32;
                let mut last_operations = String::new();
                let mut node = breadcrumbs.pHeadAutoBreadcrumbNode;
                while !node.is_null() && nodes < 1024 {
                    nodes += 1;
                    let reference = &*node;
                    let completed = reference
                        .pLastBreadcrumbValue
                        .as_ref()
                        .copied()
                        .unwrap_or_default();
                    if completed < reference.BreadcrumbCount {
                        incomplete += 1;
                        // The op the list stopped inside is the breadcrumb at
                        // the completed index.
                        if !reference.pCommandHistory.is_null()
                            && completed < reference.BreadcrumbCount
                            && incomplete <= 4
                        {
                            let history = std::slice::from_raw_parts(
                                reference.pCommandHistory,
                                reference.BreadcrumbCount as usize,
                            );
                            last_operations.push_str(&format!(
                                " [stopped at {:?} ({}/{})]",
                                history[completed as usize], completed, reference.BreadcrumbCount
                            ));
                        }
                    }
                    node = reference.pNext;
                }
                message.push_str(&format!(
                    " (DRED: {nodes} breadcrumb nodes, {incomplete} incomplete{last_operations})"
                ));
            }
            if let Ok(fault) = dred.GetPageFaultAllocationOutput() {
                message.push_str(&format!(
                    " (DRED page fault at GPU VA {:#x})",
                    fault.PageFaultVA
                ));
            }
        }
    }
    // Retained validation messages (debug devices): the drain names the
    // invalid call the way a debugger break would have, without needing one
    // attached.
    if let Ok(info_queue) = device.cast::<windows::Win32::Graphics::Direct3D12::ID3D12InfoQueue>() {
        // SAFETY: reading stored messages has no device state requirements.
        unsafe {
            let stored = info_queue.GetNumStoredMessages();
            let start = stored.saturating_sub(8);
            for index in start..stored {
                let mut length = 0_usize;
                if info_queue.GetMessage(index, None, &mut length).is_ok() && length > 0 {
                    let mut bytes = vec![0_u8; length];
                    let pointer = bytes.as_mut_ptr()
                        as *mut windows::Win32::Graphics::Direct3D12::D3D12_MESSAGE;
                    if info_queue
                        .GetMessage(index, Some(pointer), &mut length)
                        .is_ok()
                    {
                        let stored_message = &*pointer;
                        let text = std::slice::from_raw_parts(
                            stored_message.pDescription,
                            stored_message.DescriptionByteLength.saturating_sub(1),
                        );
                        message
                            .push_str(&format!(" [validation: {}]", String::from_utf8_lossy(text)));
                    }
                }
            }
        }
    }
    RhiError::dx12_with_message(operation, removed.code().0, message)
}

impl Dx12CommandEncoder<'_> {
    /// The raw recording command list, for backend-internal recording that
    /// spans layers (the surface's frame-integrated present copy).
    pub(crate) fn raw_list(&self) -> &ID3D12GraphicsCommandList4 {
        &self.command_list
    }

    /// A global UAV barrier ordering same-state unordered-access work.
    pub(crate) fn uav_barrier(&mut self) {
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: ManuallyDrop::new(None),
                }),
            },
        };
        // SAFETY: a null-resource UAV barrier is always valid on an open list.
        unsafe { self.command_list.ResourceBarrier(&[barrier]) };
    }

    pub(crate) fn build_acceleration_structure(
        &mut self,
        build: &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC,
    ) -> Result<(), RhiError> {
        // A global UAV barrier avoids cloning a COM resource into the union and
        // makes the completed build visible to subsequent ray-tracing work.
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: ManuallyDrop::new(None),
                }),
            },
        };
        // SAFETY: all GPU addresses in `build` reference live resources kept
        // by the caller through `finish`; the barrier orders the build writes.
        unsafe {
            self.command_list
                .BuildRaytracingAccelerationStructure(build, None);
            self.command_list.ResourceBarrier(&[barrier]);
        }
        Ok(())
    }

    pub(crate) fn copy_buffer(
        &mut self,
        source: &Dx12Buffer,
        source_offset: u64,
        destination: &Dx12Buffer,
        destination_offset: u64,
        size: u64,
    ) -> Result<(), RhiError> {
        // SAFETY: the common layer validated usages and bounds. Both resources
        // and this command list belong to the same DX12 device.
        unsafe {
            self.command_list.CopyBufferRegion(
                destination.resource(),
                destination_offset,
                source.resource(),
                source_offset,
                size,
            );
        }
        Ok(())
    }

    pub(crate) fn transition_buffer(
        &mut self,
        buffer: &Dx12Buffer,
        before: ResourceState,
        after: ResourceState,
    ) -> Result<(), RhiError> {
        let state_before = dx12_state(before);
        let state_after = dx12_state(after);
        if state_before == state_after {
            // DX12 rejects same-state transitions; the hazard this expresses
            // (write-then-read in the same states) is a UAV ordering, which
            // the global UAV barrier covers.
            self.uav_barrier();
            return Ok(());
        }
        let mut barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: ManuallyDrop::new(Some(buffer.resource().clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: state_before,
                    StateAfter: state_after,
                }),
            },
        };
        // SAFETY: the buffer is live for this recording; states are the
        // caller's explicit statement (buffers carry no tracked state, and
        // common-state promotion counts as being in the promoted state).
        unsafe {
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&barrier));
        }
        // SAFETY: the transition variant was constructed above; taking it
        // back releases the COM reference cloned into the barrier union.
        unsafe {
            let transition = ManuallyDrop::take(&mut barrier.Anonymous.Transition);
            let _ = ManuallyDrop::into_inner(transition.pResource);
        }
        Ok(())
    }

    pub(crate) fn transition_image(
        &mut self,
        image: &Dx12Image,
        subresources: &[u32],
        before: ResourceState,
        after: ResourceState,
    ) -> Result<(), RhiError> {
        // None named: all of them.
        let all = [D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES];
        let subresources = if subresources.is_empty() {
            &all[..]
        } else {
            subresources
        };
        let mut barriers: Vec<D3D12_RESOURCE_BARRIER> = subresources
            .iter()
            .map(|subresource| D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: ManuallyDrop::new(Some(image.resource().clone())),
                        Subresource: *subresource,
                        StateBefore: dx12_state(before),
                        StateAfter: dx12_state(after),
                    }),
                },
            })
            .collect();
        // SAFETY: the image is live for this recording and the transition was
        // derived from the common layer's tracked state of each subresource.
        unsafe {
            self.command_list.ResourceBarrier(&barriers);
        }
        for barrier in &mut barriers {
            // SAFETY: the transition variant was constructed above; taking it
            // back releases the COM reference cloned into the barrier union.
            unsafe {
                let transition = ManuallyDrop::take(&mut barrier.Anonymous.Transition);
                let _ = ManuallyDrop::into_inner(transition.pResource);
            }
        }
        Ok(())
    }

    pub(crate) fn copy_image(
        &mut self,
        source: &Dx12Image,
        destination: &Dx12Image,
    ) -> Result<(), RhiError> {
        // SAFETY: both images are live for this recording; the common layer
        // validated matching descriptors and copy states, which is exactly
        // what `CopyResource` requires for a whole-resource copy.
        unsafe {
            self.command_list
                .CopyResource(destination.resource(), source.resource());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy_buffer_to_image(
        &mut self,
        source: &Dx12Buffer,
        source_offset: u64,
        destination: &Dx12Image,
        descriptor: ImageDescriptor,
        mip: u32,
        aspect: ImageAspect,
        layout: ImageCopyLayout,
        rows: Option<std::ops::Range<u32>>,
    ) -> Result<(), RhiError> {
        self.copy_between_buffer_and_image(
            source.resource(),
            source_offset,
            destination,
            descriptor,
            mip,
            aspect,
            layout,
            CopyDirection::BufferToImage,
            rows,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy_image_to_buffer(
        &mut self,
        source: &Dx12Image,
        descriptor: ImageDescriptor,
        mip: u32,
        aspect: ImageAspect,
        destination: &Dx12Buffer,
        destination_offset: u64,
        layout: ImageCopyLayout,
    ) -> Result<(), RhiError> {
        self.copy_between_buffer_and_image(
            destination.resource(),
            destination_offset,
            source,
            descriptor,
            mip,
            aspect,
            layout,
            CopyDirection::ImageToBuffer,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_between_buffer_and_image(
        &mut self,
        buffer: &ID3D12Resource,
        buffer_offset: u64,
        image: &Dx12Image,
        descriptor: ImageDescriptor,
        mip: u32,
        aspect: ImageAspect,
        layout: ImageCopyLayout,
        direction: CopyDirection,
        rows: Option<std::ops::Range<u32>>,
    ) -> Result<(), RhiError> {
        let (width, height, depth) = descriptor.mip_extent(mip);
        // (Validated: a row of the widest image is far short of 4 GiB.)
        let row_pitch = layout.row_pitch as u32;
        let slice_size = layout.row_pitch * u64::from(layout.rows_per_slice);
        // One texture-copy location per array layer: Direct3D 12 addresses
        // array layers as separate subresources, while depth slices share one.
        // The planes of a depth-stencil image are further subresources: all
        // of the depth's mips and layers, then all of the stencil's.
        let plane = u32::from(aspect == ImageAspect::Stencil);
        for layer in 0..descriptor.layer_count() {
            let subresource = mip
                + layer * descriptor.mip_count
                + plane * descriptor.mip_count * descriptor.layer_count();
            // BC readback copies a complete block even for a 1x1 or 2x2 tail mip.
            let block = descriptor.format.block_extent();
            let (footprint_width, footprint_height) =
                if matches!(direction, CopyDirection::ImageToBuffer) {
                    (
                        width.div_ceil(block) * block,
                        height.div_ceil(block) * block,
                    )
                } else {
                    (width, rows.as_ref().map_or(height, |r| r.end - r.start))
                };
            let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: buffer_offset + u64::from(layer) * u64::from(depth) * slice_size,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: super::image::copy_format(descriptor.format, aspect),
                    Width: footprint_width,
                    Height: footprint_height,
                    Depth: depth,
                    RowPitch: row_pitch,
                },
            };
            let mut buffer_location = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(buffer.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: footprint,
                },
            };
            let mut image_location = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(image.resource().clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: subresource,
                },
            };
            // SAFETY: the common layer validated usages, states, and bounds.
            // Both resources and this command list belong to the same DX12 device.
            unsafe {
                match direction {
                    CopyDirection::BufferToImage => self.command_list.CopyTextureRegion(
                        &image_location,
                        0,
                        rows.as_ref().map_or(0, |r| r.start),
                        0,
                        &buffer_location,
                        None,
                    ),
                    CopyDirection::ImageToBuffer => self.command_list.CopyTextureRegion(
                        &buffer_location,
                        0,
                        0,
                        0,
                        &image_location,
                        None,
                    ),
                }
            }
            // SAFETY: both locations own one COM clone each; taking the
            // resources back releases those references after recording.
            unsafe {
                let _ = ManuallyDrop::take(&mut buffer_location.pResource);
                let _ = ManuallyDrop::take(&mut image_location.pResource);
            }
        }
        Ok(())
    }

    pub(crate) fn trace_rays(
        &mut self,
        pipeline: &Dx12RayTracingPipeline,
        layout: &Dx12PipelineLayout,
        descriptor_sets: &[&Dx12DescriptorSet],
        width: u32,
        height: u32,
    ) -> Result<(), RhiError> {
        let dispatch = D3D12_DISPATCH_RAYS_DESC {
            RayGenerationShaderRecord: pipeline.raygen_range,
            MissShaderTable: pipeline.miss_range,
            HitGroupTable: pipeline.hit_range,
            CallableShaderTable: D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE::default(),
            Width: width,
            Height: height,
            Depth: 1,
        };
        // SAFETY: the state object, root signature, heap, and shader tables
        // are live and were validated against this device by the common
        // layer. Root parameter 0 is the sequential resource table.
        unsafe {
            self.command_list
                .SetComputeRootSignature(layout.root_signature());
            self.bind_tables(layout, descriptor_sets, false);
            self.command_list.SetPipelineState1(&pipeline.state_object);
            self.command_list.DispatchRays(&raw const dispatch);
        }
        Ok(())
    }

    pub(crate) fn dispatch(
        &mut self,
        pipeline: &Dx12ComputePipeline,
        layout: &Dx12PipelineLayout,
        descriptor_sets: &[&Dx12DescriptorSet],
        push_constants: &[u8],
        groups: [u32; 3],
        indirect: Option<(&Dx12Buffer, u64)>,
    ) -> Result<(), RhiError> {
        // SAFETY: the pipeline state, root signature, and heap are live and
        // were validated against this device by the common layer. Root
        // bindings are re-established after every root-signature bind.
        unsafe {
            self.command_list
                .SetComputeRootSignature(layout.root_signature());
            self.bind_tables(layout, descriptor_sets, false);
            if let Some((parameter, words)) = layout.push_constant_parameter()
                && !push_constants.is_empty()
            {
                self.command_list.SetComputeRoot32BitConstants(
                    parameter,
                    words,
                    push_constants.as_ptr().cast::<std::ffi::c_void>(),
                    0,
                );
            }
            self.command_list.SetPipelineState(&pipeline.pipeline);
            match indirect {
                Some((buffer, offset)) => self.command_list.ExecuteIndirect(
                    &self.device.indirect.dispatch,
                    1,
                    buffer.resource(),
                    offset,
                    None,
                    0,
                ),
                None => self.command_list.Dispatch(groups[0], groups[1], groups[2]),
            }
        }
        Ok(())
    }

    /// Bind the tables of `sets` to the root signature just set, each as
    /// the set of the layout it is by index: its resources, and its
    /// samplers where it has any. (A list binds one heap of each kind,
    /// both named in the one call; every table is a run of the one heap of
    /// resources, and every run of samplers of the one heap of those.)
    ///
    /// # Safety
    /// The matching root signature (graphics or compute) is bound on this
    /// open list, and the sets outlive the recording.
    unsafe fn bind_tables(
        &self,
        layout: &Dx12PipelineLayout,
        sets: &[&Dx12DescriptorSet],
        graphics: bool,
    ) {
        let Some((heap, start, increment)) = self.device.table_heap() else {
            return;
        };
        if sets.is_empty() {
            return;
        }
        let table = |parameter: u32, handle: windows::Win32::Graphics::Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE| {
            // SAFETY: per this function's contract.
            unsafe {
                if graphics {
                    self.command_list
                        .SetGraphicsRootDescriptorTable(parameter, handle);
                } else {
                    self.command_list
                        .SetComputeRootDescriptorTable(parameter, handle);
                }
            }
        };
        // SAFETY: per this function's contract; the heaps are live with the
        // device and with the sets.
        unsafe {
            match sets.iter().find_map(|set| set.samplers.as_ref()) {
                Some(samplers) => self
                    .command_list
                    .SetDescriptorHeaps(&[Some(heap), Some(samplers.heap().clone())]),
                None => self.command_list.SetDescriptorHeaps(&[Some(heap)]),
            }
        }
        for (index, set) in sets.iter().enumerate() {
            if let Some(parameter) = layout.resource_table_parameter(index) {
                table(
                    parameter,
                    windows::Win32::Graphics::Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE {
                        ptr: start.ptr + u64::from(increment) * u64::from(set.offset),
                    },
                );
            }
            if let (Some(parameter), Some(samplers)) =
                (layout.sampler_table_parameter(index), &set.samplers)
            {
                table(parameter, samplers.gpu_handle());
            }
        }
    }

    /// Begin a pass: each colour view (with its image and the image's
    /// descriptor) and the colour it is cleared to, and the depth view with
    /// what its depth and stencil are cleared to.
    #[allow(clippy::type_complexity)]
    pub(crate) fn begin_render_pass(
        &mut self,
        color: &[(
            (&Dx12Image, &super::Dx12ImageView, crate::ImageDescriptor),
            Option<[f32; 4]>,
            Option<(&Dx12Image, &super::Dx12ImageView, crate::ImageDescriptor)>,
        )],
        extent: (u32, u32),
        depth: Option<(
            (&Dx12Image, &super::Dx12ImageView, crate::ImageDescriptor),
            Option<f32>,
            Option<u8>,
        )>,
    ) -> Result<(), RhiError> {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_CLEAR_FLAG_DEPTH, D3D12_CLEAR_FLAG_STENCIL, D3D12_CLEAR_FLAGS,
        };
        let mut targets = Vec::with_capacity(color.len());
        for ((image, view, descriptor), _, resolve) in color {
            targets.push(view.target_handle(&self.device.native, image, *descriptor)?);
            if let Some((target, target_view, target_descriptor)) = resolve {
                self.resolves.push(PendingResolve {
                    source: image.resource().clone(),
                    target: target.resource().clone(),
                    target_subresource: target_view.range.base_mip
                        + target_view.range.base_layer * target_descriptor.mip_count,
                    format: super::image::dxgi_format(descriptor.format),
                });
            }
        }
        let depth_target = match depth {
            None => None,
            Some(((image, view, descriptor), _, _)) => {
                Some(view.target_handle(&self.device.native, image, descriptor)?)
            }
        };
        // SAFETY: the images are live, in their attachment states per the
        // common layer, and each view's cached descriptor heap outlives this
        // recording.
        unsafe {
            self.command_list.OMSetRenderTargets(
                targets.len() as u32,
                (!targets.is_empty()).then_some(targets.as_ptr()),
                false,
                depth_target.as_ref().map(|handle| handle as *const _),
            );
            if let (Some(handle), Some(((_, _, descriptor), clear, stencil))) =
                (depth_target, depth)
            {
                let mut flags = D3D12_CLEAR_FLAGS(0);
                if clear.is_some() {
                    flags |= D3D12_CLEAR_FLAG_DEPTH;
                }
                if stencil.is_some() && descriptor.format.has_stencil() {
                    flags |= D3D12_CLEAR_FLAG_STENCIL;
                }
                if flags.0 != 0 {
                    self.command_list.ClearDepthStencilView(
                        handle,
                        flags,
                        clear.unwrap_or(1.0),
                        stencil.unwrap_or(0),
                        None,
                    );
                }
            }
            for (handle, (_, clear, _)) in targets.iter().zip(color) {
                if let Some(color) = clear {
                    self.command_list
                        .ClearRenderTargetView(*handle, color, None);
                }
            }
            self.command_list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: extent.0 as i32,
                bottom: extent.1 as i32,
            }]);
        }
        self.set_viewport([0.0, 0.0, extent.0 as f32, extent.1 as f32], [0.0, 1.0]);
        Ok(())
    }

    /// End the pass: average each multisampled attachment that named a
    /// resolve target into it. Both are render targets around the resolve,
    /// as the common layer's states have them.
    pub(crate) fn end_render_pass(&mut self) {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
        };
        for resolve in std::mem::take(&mut self.resolves) {
            let target = D3D12_RESOURCE_STATE_RENDER_TARGET;
            // The resolve writes the whole of its subresource, which may
            // never have been written before. Direct3D requires a discard
            // before the first use of a render target in that case.
            let whole = windows::Win32::Graphics::Direct3D12::D3D12_DISCARD_REGION {
                NumRects: 0,
                pRects: std::ptr::null(),
                FirstSubresource: resolve.target_subresource,
                NumSubresources: 1,
            };
            // SAFETY: the target is live and a render target here, and the
            // region names one of its subresources.
            unsafe {
                self.command_list
                    .DiscardResource(&resolve.target, Some(&whole));
            }
            self.raw_transition(&resolve.source, target, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
            self.raw_transition(&resolve.target, target, D3D12_RESOURCE_STATE_RESOLVE_DEST);
            // SAFETY: both resources are live and in their resolve states;
            // the common layer validated one format and extent.
            unsafe {
                self.command_list.ResolveSubresource(
                    &resolve.target,
                    resolve.target_subresource,
                    &resolve.source,
                    0,
                    resolve.format,
                );
            }
            self.raw_transition(&resolve.source, D3D12_RESOURCE_STATE_RESOLVE_SOURCE, target);
            self.raw_transition(&resolve.target, D3D12_RESOURCE_STATE_RESOLVE_DEST, target);
        }
    }

    /// Move every subresource of `resource` between two native states.
    fn raw_transition(
        &self,
        resource: &ID3D12Resource,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) {
        let mut barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: ManuallyDrop::new(Some(resource.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        };
        // SAFETY: the resource is live for this recording and in `before`;
        // taking the transition back releases the reference cloned into it.
        unsafe {
            self.command_list
                .ResourceBarrier(std::slice::from_ref(&barrier));
            let transition = ManuallyDrop::take(&mut barrier.Anonymous.Transition);
            let _ = ManuallyDrop::into_inner(transition.pResource);
        }
    }

    /// `rectangle` is x, y, width, height with y from the top.
    pub(crate) fn set_viewport(&mut self, rectangle: [f32; 4], depth: [f32; 2]) {
        // SAFETY: a raster pass is open on this list.
        unsafe {
            self.command_list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: rectangle[0],
                TopLeftY: rectangle[1],
                Width: rectangle[2],
                Height: rectangle[3],
                MinDepth: depth[0],
                MaxDepth: depth[1],
            }]);
        }
    }

    /// One draw. Each of `vertex_buffers` is where a stream is, its size and
    /// its stride, bound from slot 0 up; `indices` likewise with their
    /// format. `indirect` names the GPU's own counts for an indexed draw.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw(
        &mut self,
        pipeline: &super::Dx12RenderPipeline,
        layout: &Dx12PipelineLayout,
        descriptor_sets: &[&Dx12DescriptorSet],
        push_constants: &[u8],
        vertex_buffers: &[(u64, u64, u32)],
        indices: Option<(u64, u64, IndexFormat)>,
        indirect: Option<(&Dx12Buffer, u64, u32)>,
        first: u32,
        count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
    ) {
        // A view covers at most 4 GiB of its buffer.
        let view_size = |size: u64| u32::try_from(size).unwrap_or(u32::MAX);
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); crate::MAX_VERTEX_BUFFERS];
        for (view, &(address, size, stride)) in views.iter_mut().zip(vertex_buffers) {
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: address,
                SizeInBytes: view_size(size),
                StrideInBytes: stride,
            };
        }
        let streams = vertex_buffers.len();
        // SAFETY: the pipeline state, root signature, heap, and buffers are
        // live and were validated by the common layer. Root bindings are
        // re-established after every root-signature bind.
        unsafe {
            self.command_list
                .SetGraphicsRootSignature(layout.root_signature());
            self.bind_tables(layout, descriptor_sets, true);
            if let Some((parameter, words)) = layout.push_constant_parameter()
                && !push_constants.is_empty()
            {
                self.command_list.SetGraphicsRoot32BitConstants(
                    parameter,
                    words,
                    push_constants.as_ptr().cast::<std::ffi::c_void>(),
                    0,
                );
            }
            self.command_list.SetPipelineState(&pipeline.pipeline);
            self.command_list.IASetPrimitiveTopology(pipeline.topology);
            if let Some(reference) = pipeline.stencil_reference {
                self.command_list.OMSetStencilRef(reference);
            }
            if streams != 0 {
                self.command_list
                    .IASetVertexBuffers(0, Some(&views[..streams]));
            }
            let Some((address, size, format)) = indices else {
                if let Some((buffer, offset, draws)) = indirect {
                    self.command_list.ExecuteIndirect(
                        &self.device.indirect.draw,
                        draws,
                        buffer.resource(),
                        offset,
                        None,
                        0,
                    );
                } else {
                    self.command_list
                        .DrawInstanced(count, instance_count, first, first_instance);
                }
                return;
            };
            let index_view = windows::Win32::Graphics::Direct3D12::D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: address,
                SizeInBytes: view_size(size),
                Format: match format {
                    IndexFormat::Uint16 => {
                        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16_UINT
                    }
                    IndexFormat::Uint32 => {
                        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT
                    }
                },
            };
            self.command_list.IASetIndexBuffer(Some(&index_view));
            if let Some((buffer, offset, draws)) = indirect {
                self.command_list.ExecuteIndirect(
                    &self.device.indirect.draw_indexed,
                    draws,
                    buffer.resource(),
                    offset,
                    None,
                    0,
                );
            } else {
                self.command_list.DrawIndexedInstanced(
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
        // SAFETY: a raster pass is open on this list.
        unsafe {
            self.command_list
                .RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                    left: x,
                    top: y,
                    right: x + width as i32,
                    bottom: y + height as i32,
                }]);
        }
    }

    pub(crate) fn begin_occlusion_query(&mut self, set: &super::Dx12TimestampQuerySet, index: u32) {
        // SAFETY: the query heap is live and the slot index was validated by
        // the common layer.
        unsafe {
            self.command_list.BeginQuery(
                &set.heap,
                windows::Win32::Graphics::Direct3D12::D3D12_QUERY_TYPE_OCCLUSION,
                index,
            );
        }
    }

    pub(crate) fn end_occlusion_query(&mut self, set: &super::Dx12TimestampQuerySet, index: u32) {
        // SAFETY: this slot's query was begun earlier in the recording.
        unsafe {
            self.command_list.EndQuery(
                &set.heap,
                windows::Win32::Graphics::Direct3D12::D3D12_QUERY_TYPE_OCCLUSION,
                index,
            );
        }
    }

    pub(crate) fn resolve_occlusion_queries(
        &mut self,
        set: &super::Dx12TimestampQuerySet,
        first: u32,
        count: u32,
        destination: &Dx12Buffer,
        destination_offset: u64,
    ) {
        // SAFETY: the heap and buffer are live; every resolved slot counted
        // earlier in this recording per the common contract.
        unsafe {
            self.command_list.ResolveQueryData(
                &set.heap,
                windows::Win32::Graphics::Direct3D12::D3D12_QUERY_TYPE_OCCLUSION,
                first,
                count,
                destination.resource(),
                destination_offset,
            );
        }
    }

    pub(crate) fn write_timestamp(&mut self, set: &super::Dx12TimestampQuerySet, index: u32) {
        // SAFETY: the query heap is live and the slot index was validated by
        // the common layer.
        unsafe {
            self.command_list
                .EndQuery(&set.heap, D3D12_QUERY_TYPE_TIMESTAMP, index);
        }
    }

    pub(crate) fn resolve_timestamps(
        &mut self,
        set: &super::Dx12TimestampQuerySet,
        first: u32,
        count: u32,
        destination: &Dx12Buffer,
        destination_offset: u64,
    ) {
        // SAFETY: the heap and buffer are live; every resolved slot was
        // written earlier in this recording per the common contract.
        unsafe {
            self.command_list.ResolveQueryData(
                &set.heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                first,
                count,
                destination.resource(),
                destination_offset,
            );
        }
    }

    pub(crate) fn finish(self) -> Result<(), RhiError> {
        // SAFETY: the command list is open and its allocator remains alive in
        // `self` until the queue fence confirms completion.
        unsafe {
            self.command_list.Close().map_err(|error| {
                device_removed_report(
                    &self.device.native,
                    "ID3D12GraphicsCommandList::Close",
                    error,
                )
            })?;
            let command_list: ID3D12CommandList = self
                .command_list
                .cast()
                .map_err(|error| RhiError::dx12("ID3D12CommandList cast", error))?;
            self.device
                .direct_queue
                .ExecuteCommandLists(&[Some(command_list)]);
        }
        self.device.wait_for_direct_queue()
    }
}
