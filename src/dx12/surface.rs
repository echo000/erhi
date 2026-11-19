use std::ffi::c_void;
use std::mem::ManuallyDrop;

use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATES, D3D12_RESOURCE_TRANSITION_BARRIER,
    ID3D12CommandAllocator, ID3D12CommandList, ID3D12GraphicsCommandList4, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_ALPHA_MODE_IGNORE, DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,
    DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709, DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
    DXGI_COLOR_SPACE_TYPE, DXGI_FORMAT, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    DXGI_FEATURE_PRESENT_ALLOW_TEARING, DXGI_PRESENT_ALLOW_TEARING, DXGI_SWAP_CHAIN_DESC1,
    DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING, DXGI_SWAP_EFFECT_FLIP_DISCARD,
    DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGIFactory5, IDXGISwapChain3,
};
use windows::core::Interface;

use super::{Dx12Device, Dx12Image};
use crate::{
    ImageDescriptor, PresentMode, PresentStatus, RhiError, SurfaceCapabilities, SurfaceColorSpace,
    SurfaceConfiguration, SurfaceFormat,
};

fn dxgi_surface_format(format: SurfaceFormat) -> DXGI_FORMAT {
    super::image::dxgi_format(format.image_format())
}

fn dxgi_color_space(space: SurfaceColorSpace) -> DXGI_COLOR_SPACE_TYPE {
    match space {
        SurfaceColorSpace::Srgb => DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
        SurfaceColorSpace::Hdr10 => DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
        SurfaceColorSpace::ExtendedLinear => DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,
    }
}

/// The colour spaces each format may be presented in.
const FORMAT_SPACES: [(SurfaceFormat, SurfaceColorSpace); 5] = [
    (SurfaceFormat::Rgba8Unorm, SurfaceColorSpace::Srgb),
    (SurfaceFormat::Bgra8Unorm, SurfaceColorSpace::Srgb),
    (SurfaceFormat::Rgb10a2Unorm, SurfaceColorSpace::Srgb),
    (SurfaceFormat::Rgb10a2Unorm, SurfaceColorSpace::Hdr10),
    (
        SurfaceFormat::Rgba16Float,
        SurfaceColorSpace::ExtendedLinear,
    ),
];

/// One DXGI flip-model swapchain with its per-present command recording.
pub(crate) struct Dx12Surface {
    swapchain: IDXGISwapChain3,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList4,
    buffer_count: u32,
    tearing_supported: bool,
    present_mode: PresentMode,
    format: SurfaceFormat,
    /// For the device-removed diagnosis on a failed present.
    device: windows::Win32::Graphics::Direct3D12::ID3D12Device5,
}

impl Dx12Device {
    pub(crate) fn create_surface(
        &self,
        hwnd: isize,
        configuration: SurfaceConfiguration,
        frames_in_flight: u32,
    ) -> Result<Dx12Surface, RhiError> {
        if !FORMAT_SPACES.contains(&(configuration.format, configuration.color_space)) {
            return Err(RhiError::PresentationUnsupported {
                reason: "the surface format cannot be presented in this colour space",
            });
        }
        // Tearing support gates Immediate presentation on flip-model chains.
        let tearing_supported =
            self._factory
                .cast::<IDXGIFactory5>()
                .ok()
                .is_some_and(|factory5| {
                    let mut allowed = windows::core::BOOL(0);
                    // SAFETY: output pointer and size match the queried feature.
                    unsafe {
                        factory5
                            .CheckFeatureSupport(
                                DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                                (&raw mut allowed).cast::<c_void>(),
                                std::mem::size_of::<windows::core::BOOL>() as u32,
                            )
                            .is_ok()
                            && allowed.as_bool()
                    }
                });

        let buffer_count = (frames_in_flight + 1).max(2);
        let description = DXGI_SWAP_CHAIN_DESC1 {
            Width: configuration.width,
            Height: configuration.height,
            Format: dxgi_surface_format(configuration.format),
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: buffer_count,
            Scaling: Default::default(),
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_IGNORE,
            Flags: if tearing_supported {
                DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32
            } else {
                0
            },
        };
        // SAFETY: the queue and window handle are live; the caller's window
        // outlives the returned surface via the common wrapper's borrow.
        let swapchain = unsafe {
            self._factory
                .CreateSwapChainForHwnd(
                    &self.direct_queue,
                    HWND(hwnd as *mut c_void),
                    &raw const description,
                    None,
                    None,
                )
                .map_err(|error| RhiError::dx12("IDXGIFactory::CreateSwapChainForHwnd", error))?
        };
        let swapchain: IDXGISwapChain3 = swapchain
            .cast()
            .map_err(|error| RhiError::dx12("IDXGISwapChain3 cast", error))?;

        // SAFETY: both objects come from the live device with the direct
        // list type; the fresh list is closed so presents can reset it.
        let (allocator, list) = unsafe {
            let allocator: ID3D12CommandAllocator = self
                .native
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(|error| {
                    RhiError::dx12("ID3D12Device::CreateCommandAllocator(present)", error)
                })?;
            let list: ID3D12GraphicsCommandList4 = self
                .native
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .map_err(|error| {
                    RhiError::dx12("ID3D12Device::CreateCommandList(present)", error)
                })?;
            list.Close()
                .map_err(|error| RhiError::dx12("ID3D12CommandList::Close(present)", error))?;
            (allocator, list)
        };
        let surface = Dx12Surface {
            swapchain,
            allocator,
            list,
            buffer_count,
            tearing_supported,
            present_mode: configuration.present_mode,
            format: configuration.format,
            device: self.native.clone(),
        };
        surface.set_color_space(configuration.color_space)?;
        Ok(surface)
    }
}

fn transition(
    list: &ID3D12GraphicsCommandList4,
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
    // SAFETY: the transition variant was constructed above; the resource is
    // live for this recording.
    unsafe {
        list.ResourceBarrier(std::slice::from_ref(&barrier));
        let inner = ManuallyDrop::take(&mut barrier.Anonymous.Transition);
        let _ = ManuallyDrop::into_inner(inner.pResource);
    }
}

impl Dx12Surface {
    pub(crate) fn image_count(&self) -> u32 {
        self.buffer_count
    }

    /// Whether the swapchain, as it now is, can present in `space`.
    fn supports_color_space(&self, space: SurfaceColorSpace) -> bool {
        use windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT;
        // SAFETY: the swapchain is live; the query has no side effects.
        unsafe {
            self.swapchain
                .CheckColorSpaceSupport(dxgi_color_space(space))
        }
        .is_ok_and(|support| {
            support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT.0 as u32 != 0
        })
    }

    fn set_color_space(&self, space: SurfaceColorSpace) -> Result<(), RhiError> {
        // (sRGB is what a swapchain presents in unless told otherwise.)
        if space != SurfaceColorSpace::Srgb && !self.supports_color_space(space) {
            return Err(RhiError::PresentationUnsupported {
                reason: "the display does not take this colour space",
            });
        }
        // SAFETY: the swapchain is live and the colour space was checked
        // against its present format.
        unsafe { self.swapchain.SetColorSpace1(dxgi_color_space(space)) }
            .map_err(|error| RhiError::dx12("IDXGISwapChain3::SetColorSpace1", error))
    }

    /// The configurations this surface supports. HDR pairs are listed only
    /// for the swapchain's current format, because colour-space support is
    /// queried against the existing swapchain. SDR pairs are always listed.
    pub(crate) fn capabilities(&self) -> SurfaceCapabilities {
        let formats = FORMAT_SPACES
            .into_iter()
            .filter(|(format, space)| {
                *space == SurfaceColorSpace::Srgb
                    || (*format == self.format && self.supports_color_space(*space))
                    || *format != self.format
            })
            .collect();
        let mut present_modes = vec![PresentMode::Fifo, PresentMode::Mailbox];
        if self.tearing_supported {
            present_modes.push(PresentMode::Immediate);
        }
        SurfaceCapabilities {
            formats,
            present_modes,
        }
    }

    /// The sync interval and flags of a present in this surface's mode.
    fn present_arguments(&self) -> (u32, windows::Win32::Graphics::Dxgi::DXGI_PRESENT) {
        match self.present_mode {
            PresentMode::Fifo => (1, Default::default()),
            // The flip model shows the newest frame at the next vertical
            // blank when presents do not wait and may not tear.
            PresentMode::Mailbox => (0, Default::default()),
            PresentMode::Immediate => {
                if self.tearing_supported {
                    (0, DXGI_PRESENT_ALLOW_TEARING)
                } else {
                    (0, Default::default())
                }
            }
        }
    }

    pub(crate) fn configure(
        &mut self,
        device: &Dx12Device,
        configuration: SurfaceConfiguration,
    ) -> Result<(), RhiError> {
        if !FORMAT_SPACES.contains(&(configuration.format, configuration.color_space)) {
            return Err(RhiError::PresentationUnsupported {
                reason: "the surface format cannot be presented in this colour space",
            });
        }
        // Flush the queue: ResizeBuffers requires no outstanding backbuffer
        // references or in-flight presents.
        let value = device.signal_fence()?;
        device.wait_for_fence_value(value)?;
        // SAFETY: no backbuffer interfaces are retained between presents.
        unsafe {
            self.swapchain
                .ResizeBuffers(
                    self.buffer_count,
                    configuration.width,
                    configuration.height,
                    dxgi_surface_format(configuration.format),
                    if self.tearing_supported {
                        DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
                    } else {
                        Default::default()
                    },
                )
                .map_err(|error| RhiError::dx12("IDXGISwapChain::ResizeBuffers", error))?;
        }
        self.present_mode = configuration.present_mode;
        self.format = configuration.format;
        self.set_color_space(configuration.color_space)
    }

    /// The backbuffer the next present targets.
    pub(crate) fn current_backbuffer(&self) -> Result<(u32, ID3D12Resource), RhiError> {
        // SAFETY: the swapchain is live; the index selects the next buffer.
        let index = unsafe { self.swapchain.GetCurrentBackBufferIndex() };
        // SAFETY: flip-model buffers are valid resources for this index; the
        // caller drops the reference before any resize.
        let backbuffer: ID3D12Resource = unsafe {
            self.swapchain
                .GetBuffer(index)
                .map_err(|error| RhiError::dx12("IDXGISwapChain::GetBuffer(frame)", error))?
        };
        Ok((index, backbuffer))
    }

    /// Record the present copy into an already-open command list (the frame
    /// ring's): PRESENT -> COPY_DEST, copy, COPY_DEST -> PRESENT.
    pub(crate) fn record_frame_copy(
        &self,
        list: &ID3D12GraphicsCommandList4,
        backbuffer: &ID3D12Resource,
        source: &Dx12Image,
    ) {
        transition(
            list,
            backbuffer,
            D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: both resources are live for this recording; the common
        // layer validated the source's usage, state, and extent.
        unsafe {
            list.CopyResource(backbuffer, source.resource());
        }
        transition(
            list,
            backbuffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PRESENT,
        );
    }

    /// Present after the frame submission that filled the backbuffer.
    pub(crate) fn present_frame(&mut self) -> Result<PresentStatus, RhiError> {
        let (interval, flags) = self.present_arguments();
        // SAFETY: the swapchain is live and the copied backbuffer is in the
        // present state; the queue orders the flip after the frame's copy.
        let result = unsafe { self.swapchain.Present(interval, flags) };
        if result.is_err() {
            // A device removal surfacing here happened earlier, on the
            // queue. Always assemble the full diagnosis (removal reason
            // when there is one, DRED breadcrumbs, retained validation
            // messages) rather than reporting only where it surfaced.
            // SAFETY: querying the removed reason has no side effects.
            let removed = unsafe { self.device.GetDeviceRemovedReason() }
                .err()
                .unwrap_or_else(|| windows::core::Error::from_hresult(result));
            return Err(super::command_encoder::device_removed_report(
                &self.device,
                "IDXGISwapChain::Present(frame)",
                removed,
            ));
        }
        Ok(PresentStatus::Presented)
    }

    pub(crate) fn present_image(
        &mut self,
        device: &Dx12Device,
        source: &Dx12Image,
        _descriptor: ImageDescriptor,
    ) -> Result<PresentStatus, RhiError> {
        // SAFETY: the swapchain is live; the index selects the next buffer.
        let index = unsafe { self.swapchain.GetCurrentBackBufferIndex() };
        // SAFETY: flip-model buffers are valid resources for this index; the
        // reference is dropped before any resize.
        let backbuffer: ID3D12Resource = unsafe {
            self.swapchain
                .GetBuffer(index)
                .map_err(|error| RhiError::dx12("IDXGISwapChain::GetBuffer", error))?
        };
        // SAFETY: the previous present completed (synchronous wait below), so
        // the allocator can reset; recorded resources are live.
        unsafe {
            self.allocator
                .Reset()
                .map_err(|error| RhiError::dx12("ID3D12CommandAllocator::Reset(present)", error))?;
            self.list.Reset(&self.allocator, None).map_err(|error| {
                RhiError::dx12("ID3D12GraphicsCommandList::Reset(present)", error)
            })?;
            transition(
                &self.list,
                &backbuffer,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            self.list.CopyResource(&backbuffer, source.resource());
            transition(
                &self.list,
                &backbuffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PRESENT,
            );
            self.list.Close().map_err(|error| {
                RhiError::dx12("ID3D12GraphicsCommandList::Close(present)", error)
            })?;
            let list: ID3D12CommandList = self
                .list
                .cast()
                .map_err(|error| RhiError::dx12("ID3D12CommandList cast(present)", error))?;
            device.direct_queue.ExecuteCommandLists(&[Some(list)]);
        }
        let value = device.signal_fence()?;
        device.wait_for_fence_value(value)?;
        drop(backbuffer);

        let (interval, flags) = self.present_arguments();
        // SAFETY: the swapchain is live and the copied backbuffer is in the
        // present state.
        let result = unsafe { self.swapchain.Present(interval, flags) };
        if result.is_err() {
            return Err(RhiError::dx12(
                "IDXGISwapChain::Present",
                windows::core::Error::from_hresult(result),
            ));
        }
        Ok(PresentStatus::Presented)
    }
}
