use ash::vk;

use super::{CommandPool, VulkanDevice, VulkanImage};
use crate::surface::WindowHandles;
use crate::{
    ImageDescriptor, PresentMode, PresentStatus, RhiError, SurfaceCapabilities, SurfaceColorSpace,
    SurfaceConfiguration, SurfaceFormat,
};

/// One bounded acquire/fence wait slice; a dead device surfaces as an error.
const WAIT_NANOSECONDS: u64 = 10_000_000_000;

/// One Vulkan window surface with its swapchain, per-present command buffer,
/// and backend-owned synchronization (acquire/render semaphores + fence).
pub(crate) struct VulkanSurface {
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    device: ash::Device,
    queue_lock: std::sync::Arc<std::sync::Mutex<()>>,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    acquire_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,
    fence: vk::Fence,
    pool: CommandPool,
    command: vk::CommandBuffer,
    /// Per-ring-slot semaphore pairs for the frame-integrated present: the
    /// slot fence proves the pair's previous use completed before reuse.
    frame_acquire: Vec<vk::Semaphore>,
    /// Present-signal semaphores indexed by swapchain IMAGE, not ring slot:
    /// the presentation engine may still wait a slot's semaphore after the
    /// ring reuses the slot (validation: "may still be in use by
    /// VkSwapchainKHR"), but image N's semaphore is provably free once
    /// image N is reacquired.
    frame_render: Vec<vk::Semaphore>,
}

impl VulkanDevice {
    /// Whether the instance was made with extension `name` (every surface
    /// extension the loader offered was enabled at its creation).
    #[cfg(all(unix, not(target_vendor = "apple")))]
    fn instance_extension_enabled(&self, name: &std::ffi::CStr) -> bool {
        // SAFETY: enumeration returns owned property records.
        unsafe {
            self._instance
                ._entry
                .enumerate_instance_extension_properties(None)
        }
        .unwrap_or_default()
        .iter()
        .any(|property| {
            // SAFETY: extension names are fixed NUL-terminated arrays.
            (unsafe { std::ffi::CStr::from_ptr(property.extension_name.as_ptr()) }) == name
        })
    }

    /// The `VkSurfaceKHR` of a window, by its window system.
    fn create_native_surface(&self, handles: WindowHandles) -> Result<vk::SurfaceKHR, RhiError> {
        let entry = &self._instance._entry;
        let instance = &self._instance.handle;
        match handles {
            #[cfg(windows)]
            WindowHandles::Win32 { hwnd, hinstance } => {
                let loader = ash::khr::win32_surface::Instance::new(entry, instance);
                let create = vk::Win32SurfaceCreateInfoKHR::default()
                    .hinstance(hinstance)
                    .hwnd(hwnd);
                // SAFETY: the caller's window outlives the returned surface
                // via the common wrapper's borrow.
                unsafe { loader.create_win32_surface(&create, None) }
                    .map_err(|result| RhiError::vk("vkCreateWin32SurfaceKHR", result))
            }
            #[cfg(all(unix, not(target_vendor = "apple")))]
            WindowHandles::Xlib { display, window }
                if self.instance_extension_enabled(ash::khr::xlib_surface::NAME) =>
            {
                let loader = ash::khr::xlib_surface::Instance::new(entry, instance);
                let create = vk::XlibSurfaceCreateInfoKHR::default()
                    .dpy(display.cast())
                    .window(window as vk::Window);
                // SAFETY: the display and window are the caller's live ones,
                // and outlive the surface via the common wrapper's borrow.
                unsafe { loader.create_xlib_surface(&create, None) }
                    .map_err(|result| RhiError::vk("vkCreateXlibSurfaceKHR", result))
            }
            #[cfg(all(unix, not(target_vendor = "apple")))]
            WindowHandles::Xcb { connection, window }
                if self.instance_extension_enabled(ash::khr::xcb_surface::NAME) =>
            {
                let loader = ash::khr::xcb_surface::Instance::new(entry, instance);
                let create = vk::XcbSurfaceCreateInfoKHR::default()
                    .connection(connection.cast())
                    .window(window);
                // SAFETY: as for Xlib above.
                unsafe { loader.create_xcb_surface(&create, None) }
                    .map_err(|result| RhiError::vk("vkCreateXcbSurfaceKHR", result))
            }
            #[cfg(all(unix, not(target_vendor = "apple")))]
            WindowHandles::Wayland { display, surface }
                if self.instance_extension_enabled(ash::khr::wayland_surface::NAME) =>
            {
                let loader = ash::khr::wayland_surface::Instance::new(entry, instance);
                let create = vk::WaylandSurfaceCreateInfoKHR::default()
                    .display(display.cast())
                    .surface(surface.cast());
                // SAFETY: as for Xlib above.
                unsafe { loader.create_wayland_surface(&create, None) }
                    .map_err(|result| RhiError::vk("vkCreateWaylandSurfaceKHR", result))
            }
            // A window of a system this platform's Vulkan does not draw to.
            _ => Err(RhiError::UnsupportedWindowHandle),
        }
    }

    pub(crate) fn create_surface(
        &self,
        handles: WindowHandles,
        configuration: SurfaceConfiguration,
        frames_in_flight: u32,
    ) -> Result<VulkanSurface, RhiError> {
        let surface = self.create_native_surface(handles)?;
        let surface_loader =
            ash::khr::surface::Instance::new(&self._instance._entry, &self._instance.handle);

        // SAFETY: the queue family and surface belong to this instance.
        let supported = unsafe {
            surface_loader.get_physical_device_surface_support(
                self.physical_device,
                self.queue_family_index,
                surface,
            )
        }
        .map_err(|result| RhiError::vk("vkGetPhysicalDeviceSurfaceSupportKHR", result))?;
        if !supported {
            // SAFETY: the fresh surface has no swapchain yet.
            unsafe { surface_loader.destroy_surface(surface, None) };
            return Err(RhiError::PresentationUnsupported {
                reason: "the selected queue family cannot present to this window",
            });
        }

        let swapchain_loader =
            ash::khr::swapchain::Device::new(&self._instance.handle, &self.logical.handle);
        let semaphore = |label: &'static str| -> Result<vk::Semaphore, RhiError> {
            // SAFETY: default binary semaphore creation.
            unsafe {
                self.logical
                    .handle
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }
            .map_err(|result| RhiError::Vulkan {
                operation: label,
                result: result.as_raw(),
            })
        };
        let acquire_semaphore = semaphore("vkCreateSemaphore(acquire)")?;
        let render_semaphore = semaphore("vkCreateSemaphore(render)")?;
        let mut frame_acquire = Vec::with_capacity(frames_in_flight as usize);
        for _ in 0..frames_in_flight {
            frame_acquire.push(semaphore("vkCreateSemaphore(frame acquire)")?);
        }
        // Per-image render semaphores are (re)created by `configure`, which
        // is the first place the image count exists.
        let frame_render = Vec::new();
        // Signaled so the first present's fence wait passes immediately.
        let fence_create = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        // SAFETY: complete create info.
        let fence = unsafe { self.logical.handle.create_fence(&fence_create, None) }
            .map_err(|result| RhiError::vk("vkCreateFence(present)", result))?;
        let pool = CommandPool::new(&self.logical.handle, self.queue_family_index)?;
        let command = pool.allocate_primary()?;

        let mut created = VulkanSurface {
            surface_loader,
            swapchain_loader,
            device: self.logical.handle.clone(),
            queue_lock: self.queue_lock.clone(),
            surface,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            // Chosen by `configure`, which follows.
            format: vk::SurfaceFormatKHR::default(),
            extent: vk::Extent2D::default(),
            acquire_semaphore,
            render_semaphore,
            fence,
            pool,
            command,
            frame_acquire,
            frame_render,
        };
        // Windows/NVIDIA: a Vulkan swapchain created straight into a
        // non-FIFO present mode on an HWND that DXGI has previously owned
        // presents into a window the compositor never updates again. Every
        // present returns success and the acquire keeps cycling images, so a
        // live DX12-to-Vulkan swap looks like a hang with healthy counters.
        // FIFO is unaffected, and coming up on FIFO first re-establishes
        // composition for the mode actually wanted. DXGI reclaiming its own
        // window never needed this, which is why only one direction broke.
        if configuration.present_mode != PresentMode::Fifo {
            created.configure(
                self,
                SurfaceConfiguration {
                    present_mode: PresentMode::Fifo,
                    ..configuration
                },
                frames_in_flight,
            )?;
        }
        created.configure(self, configuration, frames_in_flight)?;
        Ok(created)
    }
}

/// The Vulkan names of a surface format and of a colour space.
fn vk_surface_format(format: SurfaceFormat) -> vk::Format {
    super::image::vk_format(format.image_format())
}

fn vk_color_space(space: SurfaceColorSpace) -> vk::ColorSpaceKHR {
    match space {
        SurfaceColorSpace::Srgb => vk::ColorSpaceKHR::SRGB_NONLINEAR,
        SurfaceColorSpace::Hdr10 => vk::ColorSpaceKHR::HDR10_ST2084_EXT,
        SurfaceColorSpace::ExtendedLinear => vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
    }
}

impl VulkanSurface {
    pub(crate) fn image_count(&self) -> u32 {
        self.images.len() as u32
    }

    fn surface_formats(
        &self,
        device: &VulkanDevice,
    ) -> Result<Vec<vk::SurfaceFormatKHR>, RhiError> {
        // SAFETY: the surface belongs to this instance.
        unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(device.physical_device, self.surface)
        }
        .map_err(|result| RhiError::vk("vkGetPhysicalDeviceSurfaceFormatsKHR", result))
    }

    fn surface_present_modes(
        &self,
        device: &VulkanDevice,
    ) -> Result<Vec<vk::PresentModeKHR>, RhiError> {
        // SAFETY: the surface belongs to this instance.
        unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(device.physical_device, self.surface)
        }
        .map_err(|result| RhiError::vk("vkGetPhysicalDeviceSurfacePresentModesKHR", result))
    }

    pub(crate) fn capabilities(
        &self,
        device: &VulkanDevice,
    ) -> Result<SurfaceCapabilities, RhiError> {
        let offered = self.surface_formats(device)?;
        let mut formats = Vec::new();
        for format in [
            SurfaceFormat::Rgba8Unorm,
            SurfaceFormat::Bgra8Unorm,
            SurfaceFormat::Rgb10a2Unorm,
            SurfaceFormat::Rgba16Float,
        ] {
            for space in [
                SurfaceColorSpace::Srgb,
                SurfaceColorSpace::Hdr10,
                SurfaceColorSpace::ExtendedLinear,
            ] {
                if offered.iter().any(|candidate| {
                    candidate.format == vk_surface_format(format)
                        && candidate.color_space == vk_color_space(space)
                }) {
                    formats.push((format, space));
                }
            }
        }
        // (An SDR surface of either 8-bit order is always taken: the blit
        // into the backbuffer reorders the channels where they differ.)
        for format in [SurfaceFormat::Rgba8Unorm, SurfaceFormat::Bgra8Unorm] {
            if !formats.contains(&(format, SurfaceColorSpace::Srgb)) {
                formats.push((format, SurfaceColorSpace::Srgb));
            }
        }
        let offered = self.surface_present_modes(device)?;
        let mut present_modes = vec![PresentMode::Fifo];
        if offered.contains(&vk::PresentModeKHR::IMMEDIATE) {
            present_modes.push(PresentMode::Immediate);
        }
        if offered.contains(&vk::PresentModeKHR::MAILBOX) {
            present_modes.push(PresentMode::Mailbox);
        }
        Ok(SurfaceCapabilities {
            formats,
            present_modes,
        })
    }

    /// The surface format a configuration is given: the one asked for where
    /// the surface offers it; for an SDR 8-bit request, whichever 8-bit
    /// format it does offer (the present blit converts between them).
    fn choose_format(
        &self,
        device: &VulkanDevice,
        configuration: SurfaceConfiguration,
    ) -> Result<vk::SurfaceFormatKHR, RhiError> {
        let offered = self.surface_formats(device)?;
        let wanted = vk_surface_format(configuration.format);
        let space = vk_color_space(configuration.color_space);
        if let Some(exact) = offered
            .iter()
            .copied()
            .find(|format| format.format == wanted && format.color_space == space)
        {
            return Ok(exact);
        }
        let plain = configuration.color_space == SurfaceColorSpace::Srgb
            && matches!(
                configuration.format,
                SurfaceFormat::Rgba8Unorm | SurfaceFormat::Bgra8Unorm
            );
        if !plain {
            return Err(RhiError::PresentationUnsupported {
                reason: "the surface does not offer this format and colour space",
            });
        }
        // Prefer 8-bit UNORM (the tone-mapped output is already sRGB-encoded
        // bytes, so a UNORM view presents it unchanged); blitting converts
        // RGBA to BGRA where needed.
        [vk::Format::R8G8B8A8_UNORM, vk::Format::B8G8R8A8_UNORM]
            .into_iter()
            .find_map(|format| {
                offered
                    .iter()
                    .copied()
                    .find(|candidate| candidate.format == format)
            })
            .or_else(|| offered.first().copied())
            .ok_or(RhiError::PresentationUnsupported {
                reason: "the surface reports no formats",
            })
    }

    fn wait_present_fence(&self, device: &VulkanDevice) -> Result<(), RhiError> {
        loop {
            // SAFETY: the fence belongs to this live device.
            match unsafe {
                device
                    .logical
                    .handle
                    .wait_for_fences(&[self.fence], true, WAIT_NANOSECONDS)
            } {
                Ok(()) => return Ok(()),
                Err(vk::Result::TIMEOUT) => continue,
                Err(result) => return Err(RhiError::vk("vkWaitForFences(present)", result)),
            }
        }
    }

    pub(crate) fn configure(
        &mut self,
        device: &VulkanDevice,
        configuration: SurfaceConfiguration,
        frames_in_flight: u32,
    ) -> Result<(), RhiError> {
        self.wait_present_fence(device)?;
        if self.swapchain != vk::SwapchainKHR::null() {
            // A frame-ring fence covers rendering, not the presentation queue's
            // wait on its signal semaphore. Complete presents before replacing
            // the swapchain and destroying its per-image semaphores.
            let _queue = crate::sync::lock_anyway(&self.queue_lock);
            // SAFETY: this live device owns the surface; queue access is locked.
            unsafe { self.device.device_wait_idle() }
                .map_err(|result| RhiError::vk("vkDeviceWaitIdle(swapchain)", result))?;
        }
        // SAFETY: the surface belongs to this instance.
        let capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(device.physical_device, self.surface)
        }
        .map_err(|result| RhiError::vk("vkGetPhysicalDeviceSurfaceCapabilitiesKHR", result))?;
        let format = self.choose_format(device, configuration)?;
        let extent = if capabilities.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: configuration.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: configuration.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        } else {
            capabilities.current_extent
        };
        // A minimised window has no area, and a swapchain of none cannot be
        // made: the old one is kept until the window has a size again.
        if extent.width == 0 || extent.height == 0 {
            return Err(RhiError::InvalidRenderDimensions {
                width: extent.width,
                height: extent.height,
            });
        }
        let max_images = if capabilities.max_image_count == 0 {
            u32::MAX
        } else {
            capabilities.max_image_count
        };
        let image_count = (frames_in_flight + 1).clamp(capabilities.min_image_count, max_images);
        let present_modes = self.surface_present_modes(device)?;
        let wanted = match configuration.present_mode {
            PresentMode::Fifo => vk::PresentModeKHR::FIFO,
            PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        };
        let present_mode = if present_modes.contains(&wanted) {
            wanted
        } else {
            vk::PresentModeKHR::FIFO
        };

        let create = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(self.swapchain);
        // SAFETY: the surface, formats, and old swapchain are live; the old
        // swapchain has no pending presents after the fence wait above.
        let swapchain = unsafe { self.swapchain_loader.create_swapchain(&create, None) }
            .map_err(|result| RhiError::vk("vkCreateSwapchainKHR", result))?;
        if self.swapchain != vk::SwapchainKHR::null() {
            // SAFETY: retired by the create above and no longer presented to.
            unsafe {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None)
            };
        }
        self.swapchain = swapchain;
        // SAFETY: the fresh swapchain is live.
        self.images = unsafe { self.swapchain_loader.get_swapchain_images(swapchain) }
            .map_err(|result| RhiError::vk("vkGetSwapchainImagesKHR", result))?;
        // One present-signal semaphore per image. The device-idle wait above
        // includes presentation, so replacing the old set is safe.
        for semaphore in self.frame_render.drain(..) {
            // SAFETY: no submission references it once the device idled.
            unsafe { self.device.destroy_semaphore(semaphore, None) };
        }
        for _ in 0..self.images.len() {
            // SAFETY: default binary semaphore creation.
            let semaphore = unsafe {
                self.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }
            .map_err(|result| RhiError::vk("vkCreateSemaphore(frame render)", result))?;
            self.frame_render.push(semaphore);
        }
        self.extent = extent;
        self.format = format;
        Ok(())
    }

    /// Record backbuffer transitions and the source blit into `command`.
    ///
    /// # Safety
    /// `command` must be in the recording state; the source image must be in
    /// `TRANSFER_SRC_OPTIMAL` and outlive the submission.
    unsafe fn record_blit(
        &self,
        command: vk::CommandBuffer,
        source: &VulkanImage,
        descriptor: ImageDescriptor,
        image_index: u32,
    ) {
        let backbuffer = self.images[image_index as usize];
        // SAFETY: per this function's contract.
        unsafe {
            let subresource = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let to_transfer = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(backbuffer)
                .subresource_range(subresource);
            self.device.cmd_pipeline_barrier(
                command,
                // Chain the layout transition after the acquire semaphore's
                // transfer-stage wait, even though old contents are discarded.
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer],
            );

            let layers = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1);
            let blit = vk::ImageBlit::default()
                .src_subresource(layers)
                .src_offsets([
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: descriptor.width as i32,
                        y: descriptor.height as i32,
                        z: 1,
                    },
                ])
                .dst_subresource(layers)
                .dst_offsets([
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: self.extent.width as i32,
                        y: self.extent.height as i32,
                        z: 1,
                    },
                ]);
            self.device.cmd_blit_image(
                command,
                source.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                backbuffer,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::NEAREST,
            );

            let to_present = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(backbuffer)
                .subresource_range(subresource);
            self.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present],
            );
        }
    }

    /// Acquire a backbuffer for the frame-integrated present, signalling the
    /// slot's acquire semaphore. `None` means the swapchain is out of date.
    pub(crate) fn acquire_frame(&mut self, slot: usize) -> Result<Option<u32>, RhiError> {
        // SAFETY: the swapchain and the slot's semaphore are live; the slot
        // fence completed in `FrameRing::begin`, so the semaphore is reusable.
        let acquired = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                WAIT_NANOSECONDS,
                self.frame_acquire[slot],
                vk::Fence::null(),
            )
        };
        match acquired {
            Ok((index, _suboptimal)) => Ok(Some(index)),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(None),
            Err(result) => Err(RhiError::vk("vkAcquireNextImageKHR(frame)", result)),
        }
    }

    /// Record the frame's present blit into an already-recording command
    /// buffer (the frame ring's), after a successful [`Self::acquire_frame`].
    pub(crate) fn record_frame_blit(
        &self,
        command: vk::CommandBuffer,
        source: &VulkanImage,
        descriptor: ImageDescriptor,
        image_index: u32,
    ) {
        // SAFETY: the frame encoder's command buffer is recording and the
        // common layer validated the source image's usage and state.
        unsafe { self.record_blit(command, source, descriptor, image_index) };
    }

    /// The (acquire, render) semaphore pair for the frame submission: the
    /// acquire follows the ring slot, the render-signal follows the
    /// acquired image (the presentation engine may outlive the slot).
    pub(crate) fn frame_semaphores(
        &self,
        slot: usize,
        image_index: u32,
    ) -> (vk::Semaphore, vk::Semaphore) {
        (
            self.frame_acquire[slot],
            self.frame_render[image_index as usize],
        )
    }

    /// Present the acquired backbuffer after the frame submission that
    /// signalled the slot's render semaphore.
    pub(crate) fn present_frame(
        &mut self,
        device: &VulkanDevice,
        _slot: usize,
        image_index: u32,
    ) -> Result<PresentStatus, RhiError> {
        let swapchains = [self.swapchain];
        let indices = [image_index];
        let waits = [self.frame_render[image_index as usize]];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&swapchains)
            .image_indices(&indices);
        let _queue = crate::sync::lock_anyway(&self.queue_lock);
        // SAFETY: the live swapchain's render semaphore was signalled by the frame;
        // the guard serializes host access to the presentation queue.
        match unsafe { self.swapchain_loader.queue_present(device.queue, &present) } {
            Ok(_suboptimal) => Ok(PresentStatus::Presented),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(PresentStatus::OutOfDate),
            Err(result) => Err(RhiError::vk("vkQueuePresentKHR(frame)", result)),
        }
    }

    pub(crate) fn present_image(
        &mut self,
        device: &VulkanDevice,
        source: &VulkanImage,
        descriptor: ImageDescriptor,
    ) -> Result<PresentStatus, RhiError> {
        self.wait_present_fence(device)?;
        // SAFETY: the swapchain and semaphore are live; a bounded timeout
        // keeps a dead device diagnosable.
        let acquired = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                WAIT_NANOSECONDS,
                self.acquire_semaphore,
                vk::Fence::null(),
            )
        };
        let image_index = match acquired {
            Ok((index, _suboptimal)) => index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(PresentStatus::OutOfDate),
            Err(result) => return Err(RhiError::vk("vkAcquireNextImageKHR", result)),
        };
        // SAFETY: the fence completed, so the pool's previous recording is
        // finished; the recorded blit references live images.
        unsafe {
            self.device
                .reset_fences(&[self.fence])
                .map_err(|result| RhiError::vk("vkResetFences(present)", result))?;
            self.device
                .reset_command_pool(self.pool.handle, vk::CommandPoolResetFlags::empty())
                .map_err(|result| RhiError::vk("vkResetCommandPool(present)", result))?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(self.command, &begin)
                .map_err(|result| RhiError::vk("vkBeginCommandBuffer(present)", result))?;
            self.record_blit(self.command, source, descriptor, image_index);
            self.device
                .end_command_buffer(self.command)
                .map_err(|result| RhiError::vk("vkEndCommandBuffer(present)", result))?;

            let waits = [self.acquire_semaphore];
            let stages = [vk::PipelineStageFlags::TRANSFER];
            let commands = [self.command];
            let signals = [self.render_semaphore];
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&waits)
                .wait_dst_stage_mask(&stages)
                .command_buffers(&commands)
                .signal_semaphores(&signals);
            let _queue = crate::sync::lock_anyway(&self.queue_lock);
            self.device
                .queue_submit(device.queue, &[submit], self.fence)
                .map_err(|result| RhiError::vk("vkQueueSubmit(present)", result))?;

            let swapchains = [self.swapchain];
            let indices = [image_index];
            let present_waits = [self.render_semaphore];
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&present_waits)
                .swapchains(&swapchains)
                .image_indices(&indices);
            match self.swapchain_loader.queue_present(device.queue, &present) {
                Ok(_suboptimal) => Ok(PresentStatus::Presented),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(PresentStatus::OutOfDate),
                Err(result) => Err(RhiError::vk("vkQueuePresentKHR", result)),
            }
        }
    }
}

impl Drop for VulkanSurface {
    fn drop(&mut self) {
        // SAFETY: a device-wide idle covers both the blit submission and the
        // presentation engine before teardown; an error means the device is
        // already lost and destruction proceeds regardless. Every handle
        // below is exclusively owned here.
        unsafe {
            let _queue = crate::sync::lock_anyway(&self.queue_lock);
            let _ = self.device.device_wait_idle();
            self.device.destroy_semaphore(self.acquire_semaphore, None);
            self.device.destroy_semaphore(self.render_semaphore, None);
            for semaphore in self
                .frame_acquire
                .drain(..)
                .chain(self.frame_render.drain(..))
            {
                self.device.destroy_semaphore(semaphore, None);
            }
            self.device.destroy_fence(self.fence, None);
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
