use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

#[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
use raw_window_handle::RawWindowHandle;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::device::BackendDevice;
#[cfg(all(feature = "dx12", windows))]
use crate::dx12::Dx12Surface;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanSurface;
use crate::{Backend, Device, Image, ImageFormat, ImageUsages, ResourceState, RhiError};

/// Presentation pacing. `Fifo` is VSync and always available. `Immediate`
/// presents without waiting for vertical blank, and may tear. `Mailbox`
/// shows the newest finished frame at each vertical blank: no tearing, no
/// waiting. Either falls back to `Fifo` where the backend does not offer it
/// (see [`Surface::capabilities`]).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PresentMode {
    /// Frames wait their turn for the display's refresh: no tearing, and every
    /// device has it.
    #[default]
    Fifo,
    /// Frames are shown at once, tearing if the display is mid-refresh.
    Immediate,
    /// The newest frame waiting is shown at the next refresh, and older ones
    /// are dropped: no tearing, low delay.
    Mailbox,
}

/// The format of a surface's backbuffers, and so of the image presented to
/// it ([`Self::image_format`]).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SurfaceFormat {
    /// 8-bit RGBA holding sRGB-encoded values: what a tone-mapped frame is.
    #[default]
    Rgba8Unorm,
    /// Eight bits a channel, blue first in memory.
    Bgra8Unorm,
    /// Ten bits a channel: HDR10, or finer SDR.
    Rgb10a2Unorm,
    /// Half floats: linear scRGB, where 1.0 is SDR white and more is HDR.
    Rgba16Float,
}

impl SurfaceFormat {
    /// The format an image must have to be presented to such a surface.
    #[must_use]
    pub const fn image_format(self) -> ImageFormat {
        match self {
            Self::Rgba8Unorm => ImageFormat::Rgba8Unorm,
            Self::Bgra8Unorm => ImageFormat::Bgra8Unorm,
            Self::Rgb10a2Unorm => ImageFormat::Rgb10a2Unorm,
            Self::Rgba16Float => ImageFormat::Rgba16Float,
        }
    }
}

/// How the display interprets presented pixel values.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SurfaceColorSpace {
    /// sRGB-encoded values with the sRGB primaries: every SDR display.
    #[default]
    Srgb,
    /// PQ-encoded (ST 2084) values with the Rec. 2020 primaries: HDR10, for
    /// [`SurfaceFormat::Rgb10a2Unorm`].
    Hdr10,
    /// Linear values with the sRGB primaries, unbounded: scRGB, for
    /// [`SurfaceFormat::Rgba16Float`].
    ExtendedLinear,
}

/// Swapchain settings. Presentation copies the presented image into a
/// backbuffer and converts nothing: the image holds what the display is to
/// be given, in the surface's format and colour space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceConfiguration {
    /// The width of the swapchain's images, in pixels.
    pub width: u32,
    /// The height of the swapchain's images, in pixels.
    pub height: u32,
    /// How frames are timed against the display's refresh.
    pub present_mode: PresentMode,
    /// The format of the swapchain's images.
    pub format: SurfaceFormat,
    /// How the display is to read the values in them.
    pub color_space: SurfaceColorSpace,
}

impl SurfaceConfiguration {
    /// An SDR surface of 8-bit RGBA, paced by VSync.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            present_mode: PresentMode::Fifo,
            format: SurfaceFormat::Rgba8Unorm,
            color_space: SurfaceColorSpace::Srgb,
        }
    }
}

/// What a surface can be configured with on this device and display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceCapabilities {
    /// Each format with a colour space it may be given.
    pub formats: Vec<(SurfaceFormat, SurfaceColorSpace)>,
    /// The present modes it may be given.
    pub present_modes: Vec<PresentMode>,
}

/// The outcome of a present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentStatus {
    /// The frame was handed to the display.
    Presented,
    /// The swapchain no longer matches the window (typically a resize);
    /// call [`Surface::configure`] with the new size and present again.
    OutOfDate,
}

/// One presentable window surface. The caller owns the window and its
/// event loop; this type only borrows the raw handles, and the borrow keeps
/// the surface from outliving its window.
pub struct Surface<'window> {
    pub(crate) device_id: u64,
    configuration: SurfaceConfiguration,
    pub(crate) backend: Arc<Mutex<SurfaceBackend>>,
    pub(crate) status: Arc<Mutex<PresentStatus>>,
    device: BackendDevice,
    kind: Backend,
    marker: PhantomData<&'window ()>,
}

pub(crate) enum SurfaceBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<VulkanSurface>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Box<Dx12Surface>),
}

impl fmt::Debug for Surface<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Surface")
            .field("backend", &self.backend())
            .field("configuration", &self.configuration)
            .finish_non_exhaustive()
    }
}

/// The native handles of a window, as its window system names them.
#[derive(Debug, Clone, Copy)]
pub(crate) enum WindowHandles {
    /// A Win32 window and the module that owns it.
    #[cfg_attr(not(windows), allow(dead_code))]
    Win32 {
        hwnd: isize,
        // DXGI needs only the window; Vulkan's surface creation reads both.
        #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
        hinstance: isize,
    },
    /// An X11 window through Xlib: the `Display` and the window's id.
    #[cfg_attr(not(all(unix, feature = "vulkan")), allow(dead_code))]
    Xlib {
        display: *mut std::ffi::c_void,
        window: u64,
    },
    /// An X11 window through XCB: the connection and the window's id.
    #[cfg_attr(not(all(unix, feature = "vulkan")), allow(dead_code))]
    Xcb {
        connection: *mut std::ffi::c_void,
        window: u32,
    },
    /// A Wayland surface and the display it belongs to.
    #[cfg_attr(not(all(unix, feature = "vulkan")), allow(dead_code))]
    Wayland {
        display: *mut std::ffi::c_void,
        surface: *mut std::ffi::c_void,
    },
}

#[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
fn window_handles<W>(window: &W) -> Result<WindowHandles, RhiError>
where
    W: HasWindowHandle + HasDisplayHandle,
{
    use raw_window_handle::RawDisplayHandle;

    let handle = window
        .window_handle()
        .map_err(|_| RhiError::UnsupportedWindowHandle)?;
    // Only the X and Wayland handles need their display beside them.
    let display = window.display_handle().ok().map(|display| display.as_raw());
    let pointer = |pointer: Option<std::ptr::NonNull<std::ffi::c_void>>| {
        pointer
            .map(std::ptr::NonNull::as_ptr)
            .ok_or(RhiError::UnsupportedWindowHandle)
    };
    match (handle.as_raw(), display) {
        (RawWindowHandle::Win32(win32), _) => Ok(WindowHandles::Win32 {
            hwnd: win32.hwnd.get(),
            hinstance: win32.hinstance.map_or(0, std::num::NonZeroIsize::get),
        }),
        (RawWindowHandle::Xlib(window), Some(RawDisplayHandle::Xlib(display))) => {
            Ok(WindowHandles::Xlib {
                display: pointer(display.display)?,
                #[allow(clippy::unnecessary_cast)]
                window: window.window as u64,
            })
        }
        (RawWindowHandle::Xcb(window), Some(RawDisplayHandle::Xcb(display))) => {
            Ok(WindowHandles::Xcb {
                connection: pointer(display.connection)?,
                window: window.window.get(),
            })
        }
        (RawWindowHandle::Wayland(window), Some(RawDisplayHandle::Wayland(display))) => {
            Ok(WindowHandles::Wayland {
                display: display.display.as_ptr(),
                surface: window.surface.as_ptr(),
            })
        }
        _ => Err(RhiError::UnsupportedWindowHandle),
    }
}

impl Device {
    /// Create a surface for a window: any `raw-window-handle`
    /// window works, and the returned surface borrows it.
    pub fn create_surface<'window, W>(
        &self,
        window: &'window W,
        configuration: SurfaceConfiguration,
    ) -> Result<Surface<'window>, RhiError>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        self.create_surface_inner(window, configuration)
    }

    /// Create a surface whose window lifetime is managed by the caller.
    ///
    /// # Safety
    /// The window must remain alive until this surface and every captured
    /// presentation have been dropped and their native GPU work has completed.
    pub unsafe fn create_surface_unchecked<W>(
        &self,
        window: &W,
        configuration: SurfaceConfiguration,
    ) -> Result<Surface<'static>, RhiError>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        self.create_surface_inner(window, configuration)
    }

    fn create_surface_inner<'window, W>(
        &self,
        window: &W,
        configuration: SurfaceConfiguration,
    ) -> Result<Surface<'window>, RhiError>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        if configuration.width == 0 || configuration.height == 0 {
            return Err(RhiError::InvalidRenderDimensions {
                width: configuration.width,
                height: configuration.height,
            });
        }
        let backend = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => {
                SurfaceBackend::Vulkan(Box::new(device.create_surface(
                    window_handles(window)?,
                    configuration,
                    self.frames_in_flight(),
                )?))
            }
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => {
                let WindowHandles::Win32 { hwnd, .. } = window_handles(window)? else {
                    return Err(RhiError::UnsupportedWindowHandle);
                };
                SurfaceBackend::Dx12(Box::new(device.create_surface(
                    hwnd,
                    configuration,
                    self.frames_in_flight(),
                )?))
            }
        };
        Ok(Surface {
            device_id: self.id(),
            configuration,
            backend: Arc::new(Mutex::new(backend)),
            status: Arc::new(Mutex::new(PresentStatus::Presented)),
            device: self.backend.clone(),
            kind: self.backend(),
            marker: PhantomData,
        })
    }
}

impl<'window> Surface<'window> {
    /// The backend of the device that made this.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.kind
    }

    /// The configuration the surface now has.
    #[must_use]
    pub fn configuration(&self) -> SurfaceConfiguration {
        self.configuration
    }

    /// The number of backbuffer images. No more frames than this are in
    /// flight at once, regardless of the device's `frames_in_flight`.
    #[must_use]
    pub fn image_count(&self) -> u32 {
        match &*crate::sync::lock_anyway(&self.backend) {
            #[cfg(feature = "vulkan")]
            SurfaceBackend::Vulkan(surface) => surface.image_count(),
            #[cfg(all(feature = "dx12", windows))]
            SurfaceBackend::Dx12(surface) => surface.image_count(),
        }
    }

    /// The formats, colour spaces and present modes this surface may be
    /// configured with: what the device and the display offer between them.
    pub fn capabilities(&self, device: &Device) -> Result<SurfaceCapabilities, RhiError> {
        if device.id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        let surface = self.backend.lock().map_err(|_| RhiError::LockPoisoned {
            what: "the surface",
        })?;
        match (&*surface, &device.backend) {
            #[cfg(feature = "vulkan")]
            (SurfaceBackend::Vulkan(surface), BackendDevice::Vulkan(native)) => {
                surface.capabilities(native)
            }
            #[cfg(all(feature = "dx12", windows))]
            (SurfaceBackend::Dx12(surface), BackendDevice::Dx12(_)) => Ok(surface.capabilities()),
            #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
            _ => Err(RhiError::ResourceBackendMismatch),
        }
    }

    /// Recreate the swapchain for a new size, format or present mode. Call
    /// after a window resize or when a present reports
    /// [`PresentStatus::OutOfDate`]. A window with no area (minimised) has
    /// no swapchain to make: `InvalidRenderDimensions`, and the surface
    /// keeps what it had until it is configured with a size again.
    pub fn configure(
        &mut self,
        device: &Device,
        configuration: SurfaceConfiguration,
    ) -> Result<(), RhiError> {
        if device.id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        if configuration.width == 0 || configuration.height == 0 {
            return Err(RhiError::InvalidRenderDimensions {
                width: configuration.width,
                height: configuration.height,
            });
        }
        crate::submission::configure(
            &device.backend,
            &self.backend,
            configuration,
            device.frames_in_flight(),
        )?;
        *self.status.lock().map_err(|_| RhiError::LockPoisoned {
            what: "the present status",
        })? = PresentStatus::Presented;
        self.configuration = configuration;
        Ok(())
    }

    /// Present an image to the window.
    ///
    /// The image must carry `COPY_SOURCE` usage, have the surface's format
    /// ([`SurfaceFormat::image_format`]) and match the surface extent exactly
    /// (the viewport aspect ratio follows the actual render target). Returns
    /// [`PresentStatus::OutOfDate`] when the swapchain needs reconfiguring.
    pub fn present_image(
        &mut self,
        device: &Device,
        source: &Image,
    ) -> Result<PresentStatus, RhiError> {
        if device.id() != self.device_id || source.device_id() != device.id() {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        source.require_usage(ImageUsages::COPY_SOURCE, "copy source")?;
        // Put where it is copied from, if it is not there already.
        if source.current_state() != Some(ResourceState::CopySource) {
            let mut encoder = device.begin_commands()?;
            encoder.use_image(
                source,
                crate::image::Subresources::whole(&source.descriptor()),
                crate::command_encoder::ImageNeed::COPY_SOURCE,
            )?;
            encoder.finish()?;
        }
        source.require_state(ResourceState::CopySource)?;
        let descriptor = source.descriptor();
        if descriptor.width != self.configuration.width
            || descriptor.height != self.configuration.height
        {
            return Err(RhiError::InvalidRenderDimensions {
                width: descriptor.width,
                height: descriptor.height,
            });
        }
        crate::submission::present_image(&device.backend, &self.presentation(source)?)
    }

    /// Capture presentation resources while retaining the window lifetime.
    pub fn presentation(&self, source: &Image) -> Result<Presentation<'window>, RhiError> {
        if source.device_id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        source.require_usage(ImageUsages::COPY_SOURCE, "copy source")?;
        let descriptor = source.descriptor();
        if descriptor.width != self.configuration.width
            || descriptor.height != self.configuration.height
        {
            return Err(RhiError::InvalidRenderDimensions {
                width: descriptor.width,
                height: descriptor.height,
            });
        }
        if descriptor.format != self.configuration.format.image_format() {
            return Err(RhiError::PresentationUnsupported {
                reason: "the presented image does not have the surface's format",
            });
        }
        Ok(Presentation {
            surface: self.backend.clone(),
            status: self.status.clone(),
            image: source.clone(),
            marker: PhantomData,
        })
    }
}

impl Drop for Surface<'_> {
    fn drop(&mut self) {
        // Native submissions must retire before releasing the window surface.
        let _ = crate::submission::wait_idle(&self.device);
    }
}

/// One present that has been asked for: holds what it needs until the frame
/// it belongs to is submitted, and says afterwards how it went.
pub struct Presentation<'window> {
    pub(crate) surface: Arc<Mutex<SurfaceBackend>>,
    pub(crate) status: Arc<Mutex<PresentStatus>>,
    pub(crate) image: Image,
    marker: PhantomData<&'window ()>,
}

impl Presentation<'_> {
    /// How the present went, once the frame it belongs to has been submitted.
    pub fn completed_status(&self) -> Result<PresentStatus, RhiError> {
        self.status
            .lock()
            .map(|status| *status)
            .map_err(|_| RhiError::LockPoisoned {
                what: "the present status",
            })
    }
}

impl std::fmt::Debug for Presentation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Presentation")
            .field("image", &self.image)
            .finish_non_exhaustive()
    }
}
