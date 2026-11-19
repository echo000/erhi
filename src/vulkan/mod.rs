//! The Vulkan backend.

mod acceleration_structure;
mod buffer;
mod command_encoder;
mod compute_pipeline;
mod descriptor_set;
mod frame;
mod image;
mod pipeline_layout;
mod ray_tracing_pipeline;
mod render_pipeline;
mod sampler;
mod shader;
mod surface;
mod timestamp_query;

use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex, MutexGuard};
use std::{fmt, ptr};

pub(crate) use acceleration_structure::{BuiltAcceleration, VulkanBlasBuild, VulkanTlasBuild};
use ash::{Entry, vk};
pub(crate) use buffer::VulkanBuffer;
pub(crate) use command_encoder::VulkanCommandEncoder;
pub(crate) use compute_pipeline::VulkanComputePipeline;
pub(crate) use descriptor_set::VulkanDescriptorSet;
pub(crate) use frame::VulkanFrameRing;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::{AllocationSizes, AllocatorDebugSettings};
pub(crate) use image::{VulkanImage, VulkanImageView};
pub(crate) use pipeline_layout::VulkanPipelineLayout;
pub(crate) use ray_tracing_pipeline::VulkanRayTracingPipeline;
pub(crate) use render_pipeline::VulkanRenderPipeline;
pub(crate) use sampler::VulkanSampler;
pub(crate) use shader::VulkanShaderModule;
pub(crate) use surface::VulkanSurface;
pub(crate) use timestamp_query::VulkanTimestampQuerySet;

use crate::{AdapterPreference, Capabilities, RhiError};

const MINIMUM_API_VERSION: u32 = vk::API_VERSION_1_2;

impl VulkanDevice {
    pub(crate) fn wait_idle(&self) -> Result<(), RhiError> {
        let _queue = crate::sync::lock_anyway(&self.queue_lock);
        // SAFETY: the device is live; queue access is serialized by this guard.
        unsafe { self.logical.handle.device_wait_idle() }
            .map_err(|result| RhiError::vk("vkDeviceWaitIdle", result))
    }
}

/// Extensions every device needs, ray tracing or not.
fn base_device_extensions() -> [&'static CStr; 2] {
    [
        ash::khr::swapchain::NAME,
        // The raster path renders with `vkCmdBeginRenderingKHR` instead of
        // render-pass and framebuffer objects.
        ash::khr::dynamic_rendering::NAME,
    ]
}

/// The hardware ray-tracing extension set, optional on a compute-only open.
fn ray_tracing_device_extensions() -> [&'static CStr; 3] {
    [
        ash::khr::acceleration_structure::NAME,
        ash::khr::ray_tracing_pipeline::NAME,
        ash::khr::deferred_host_operations::NAME,
    ]
}

struct InstanceOwner {
    _entry: Entry,
    handle: ash::Instance,
}

/// The most messages a device keeps: later ones are counted and dropped.
const MAX_DEBUG_MESSAGES: usize = 1024;

/// What the validation layer has said since the device was opened.
struct DebugSink {
    errors: std::sync::atomic::AtomicU64,
    messages: std::sync::Mutex<Vec<crate::DebugMessage>>,
}

/// Counts validation errors and keeps every message for the caller.
extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if user_data.is_null() {
        return vk::FALSE;
    }
    let error = severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR);
    // SAFETY: `data` is valid for the duration of the callback per the spec,
    // and `user_data` is the Arc'd sink kept alive by the messenger.
    unsafe {
        let sink = &*user_data.cast::<DebugSink>();
        if error {
            sink.errors
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if !data.is_null() && !(*data).p_message.is_null() {
            let mut kept = crate::sync::lock_anyway(&sink.messages);
            if kept.len() < MAX_DEBUG_MESSAGES {
                kept.push(crate::DebugMessage {
                    severity: if error {
                        crate::DebugSeverity::Error
                    } else {
                        crate::DebugSeverity::Warning
                    },
                    text: CStr::from_ptr((*data).p_message)
                        .to_string_lossy()
                        .into_owned(),
                });
            }
        }
    }
    vk::FALSE
}

/// The validation-layer messenger plus its error counter.
struct DebugMessenger {
    loader: ash::ext::debug_utils::Instance,
    handle: vk::DebugUtilsMessengerEXT,
    sink: Arc<DebugSink>,
}

impl DebugMessenger {
    fn new(instance: &InstanceOwner) -> Result<Self, RhiError> {
        let loader = ash::ext::debug_utils::Instance::new(&instance._entry, &instance.handle);
        let sink = Arc::new(DebugSink {
            errors: std::sync::atomic::AtomicU64::new(0),
            messages: std::sync::Mutex::new(Vec::new()),
        });
        let create = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
            )
            .pfn_user_callback(Some(debug_callback))
            .user_data(Arc::as_ptr(&sink).cast_mut().cast());
        // SAFETY: the callback and counter outlive the messenger, which is
        // destroyed before the instance.
        let handle = unsafe { loader.create_debug_utils_messenger(&create, None) }
            .map_err(|result| RhiError::vk("vkCreateDebugUtilsMessengerEXT", result))?;
        Ok(Self {
            loader,
            handle,
            sink,
        })
    }
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        // SAFETY: declared before the instance owner, so the instance is
        // still live here; this owner holds the only messenger handle.
        unsafe {
            self.loader.destroy_debug_utils_messenger(self.handle, None);
        }
    }
}

impl Drop for InstanceOwner {
    fn drop(&mut self) {
        // SAFETY: this owner is the final owner of the instance, and the device
        // owner is declared before it in `VulkanDevice`, so the device is gone.
        unsafe { self.handle.destroy_instance(None) };
    }
}

struct DeviceOwner {
    handle: ash::Device,
    /// A fence that is always signalled: asking its status asks the device
    /// whether it still works, without submitting anything.
    health: vk::Fence,
    /// What the driver keeps of the pipelines created on this device.
    pipeline_cache: vk::PipelineCache,
}

impl Drop for DeviceOwner {
    fn drop(&mut self) {
        // SAFETY: all resources created by this crate are scoped inside calls
        // and destroyed before the device. Waiting also makes early test failure
        // deterministic instead of racing destruction.
        unsafe {
            let _ = self.handle.device_wait_idle();
            self.handle.destroy_fence(self.health, None);
            self.handle
                .destroy_pipeline_cache(self.pipeline_cache, None);
            self.handle.destroy_device(None);
        }
    }
}

#[derive(Debug)]
struct Candidate {
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    device_type: vk::PhysicalDeviceType,
    capabilities: Capabilities,
}

/// A Vulkan device, with the KHR ray-tracing extensions enabled where the
/// adapter has them.
pub struct VulkanDevice {
    acceleration_structure: ash::khr::acceleration_structure::Device,
    ray_tracing_pipeline: ash::khr::ray_tracing_pipeline::Device,
    pub(crate) dynamic_rendering: ash::khr::dynamic_rendering::Device,
    // Declaration order matters: the allocator frees its device memory before
    // the device is destroyed, device resources die before the instance, and
    // the debug messenger is destroyed while the instance is still live.
    allocator: Arc<Mutex<Allocator>>,
    descriptor_pages: Mutex<Vec<std::sync::Weak<descriptor_set::DescriptorPage>>>,
    logical: DeviceOwner,
    /// Held to read by pipeline creation and to write while saved pipelines
    /// are merged in: a merge needs the cache to itself.
    pipeline_cache_lock: std::sync::RwLock<()>,
    /// Names objects for capture tools, where the loader has debug-utils.
    debug_names: Option<ash::ext::debug_utils::Device>,
    debug_messenger: Option<DebugMessenger>,
    _instance: InstanceOwner,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_lock: Arc<Mutex<()>>,
    queue_family_index: u32,
    /// Whether samplers may filter anisotropically on this device.
    pub(crate) sampler_anisotropy: bool,
    /// Whether one call draws many indirect records (else one call each).
    pub(crate) multi_draw_indirect: bool,
    /// Whether occlusion queries count samples.
    pub(crate) occlusion_query_precise: bool,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    capabilities: Capabilities,
}

impl fmt::Debug for VulkanDevice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VulkanDevice")
            .field("capabilities", &self.capabilities)
            .field("queue_family_index", &self.queue_family_index)
            .finish_non_exhaustive()
    }
}

impl VulkanDevice {
    /// Open a Vulkan adapter, optionally with the validation layer
    /// and a debug-utils messenger counting errors.
    pub fn open(
        adapter_preference: AdapterPreference,
        debug: bool,
        require_ray_tracing: bool,
        optional_extensions: (&[&str], &[&str]),
    ) -> Result<Self, RhiError> {
        // SAFETY: `ash` dynamically loads the system Vulkan loader. The entry
        // is retained by `InstanceOwner` until after instance destruction.
        let entry =
            unsafe { Entry::load() }.map_err(|error| RhiError::VulkanLoader(error.to_string()))?;
        // SAFETY: this only queries the loader and creates no borrowed state.
        let loader_version = unsafe { entry.try_enumerate_instance_version() }
            .map_err(|result| RhiError::vk("vkEnumerateInstanceVersion", result))?
            .unwrap_or(vk::API_VERSION_1_0);
        if loader_version < MINIMUM_API_VERSION {
            return Err(RhiError::NoRayTracingDevice);
        }

        let application_name = c"erhi";
        let application = vk::ApplicationInfo::default()
            .application_name(application_name)
            .application_version(1)
            .engine_name(application_name)
            .engine_version(1)
            .api_version(loader_version.min(vk::API_VERSION_1_3));
        // Surfaces are created later from raw window handles, so the surface
        // extensions are enabled here, at instance creation. They are
        // harmless when no window is ever used.
        let mut instance_extensions: Vec<*const std::ffi::c_char> = Vec::new();
        // The validation layer only participates when installed (Vulkan SDK
        // machines); a missing layer downgrades to a plain instance.
        let validation_layer = c"VK_LAYER_KHRONOS_validation";
        // SAFETY: enumeration returns owned property records.
        let validation_available = debug
            && unsafe { entry.enumerate_instance_layer_properties() }
                .map(|layers| {
                    layers.iter().any(|layer| {
                        // SAFETY: layer names are fixed NUL-terminated arrays.
                        let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                        name == validation_layer
                    })
                })
                .unwrap_or(false);
        let layer_names = [validation_layer.as_ptr()];
        // Extensions a caller asked for as well, enabled below where the
        // driver has them.
        let names = |list: &[&str]| -> Vec<std::ffi::CString> {
            list.iter()
                .filter_map(|name| std::ffi::CString::new(*name).ok())
                .collect()
        };
        let (optional_instance, optional_device) =
            (names(optional_extensions.0), names(optional_extensions.1));
        // SAFETY: enumeration returns owned property records.
        let available_instance =
            unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
        // The surface extensions of this platform's window systems, and the
        // one that names the HDR colour spaces: each where the loader has
        // it. A machine with none of them still renders without a window.
        let surface_extensions: &[&CStr] = &[
            ash::khr::surface::NAME,
            #[cfg(windows)]
            ash::khr::win32_surface::NAME,
            #[cfg(all(unix, not(target_vendor = "apple")))]
            ash::khr::xlib_surface::NAME,
            #[cfg(all(unix, not(target_vendor = "apple")))]
            ash::khr::xcb_surface::NAME,
            #[cfg(all(unix, not(target_vendor = "apple")))]
            ash::khr::wayland_surface::NAME,
            ash::ext::swapchain_colorspace::NAME,
        ];
        for name in surface_extensions {
            if available_instance.iter().any(|property| {
                // SAFETY: extension names are fixed NUL-terminated arrays.
                (unsafe { CStr::from_ptr(property.extension_name.as_ptr()) }) == *name
            }) {
                instance_extensions.push(name.as_ptr());
            }
        }
        for name in &optional_instance {
            let supported = available_instance.iter().any(|property| {
                // SAFETY: extension names are fixed NUL-terminated arrays.
                (unsafe { CStr::from_ptr(property.extension_name.as_ptr()) }) == name.as_c_str()
            });
            if supported
                && !instance_extensions
                    .iter()
                    // SAFETY: every entry is a NUL-terminated static name.
                    .any(|existing| unsafe { CStr::from_ptr(*existing) } == name.as_c_str())
            {
                instance_extensions.push(name.as_ptr());
            }
        }
        let mut create_info = vk::InstanceCreateInfo::default().application_info(&application);
        // Debug-utils names objects for capture tools (RenderDoc) with or
        // without validation; the messenger below is validation's alone.
        let debug_utils = available_instance.iter().any(|property| {
            // SAFETY: extension names are fixed NUL-terminated arrays.
            (unsafe { CStr::from_ptr(property.extension_name.as_ptr()) })
                == ash::ext::debug_utils::NAME
        });
        if debug_utils
            && !instance_extensions
                .iter()
                // SAFETY: every entry is a NUL-terminated static name.
                .any(|existing| unsafe { CStr::from_ptr(*existing) } == ash::ext::debug_utils::NAME)
        {
            instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        let validation_available = validation_available && debug_utils;
        if validation_available {
            create_info = create_info.enabled_layer_names(&layer_names);
        }
        let create_info = create_info.enabled_extension_names(&instance_extensions);
        // SAFETY: all pointers in `create_info` remain alive for this call.
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|result| RhiError::vk("vkCreateInstance", result))?;
        let instance = InstanceOwner {
            _entry: entry,
            handle: instance,
        };

        let debug_messenger = if validation_available {
            Some(DebugMessenger::new(&instance)?)
        } else {
            None
        };

        let candidate =
            select_candidate(&instance.handle, adapter_preference, require_ray_tracing)?;
        let with_ray_tracing = candidate.capabilities.ray_tracing;
        let queue_priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(candidate.queue_family_index)
            .queue_priorities(&queue_priorities)];
        let mut extension_pointers: Vec<*const i8> = base_device_extensions()
            .iter()
            .map(|name| name.as_ptr())
            .collect();
        if with_ray_tracing {
            extension_pointers.extend(
                ray_tracing_device_extensions()
                    .iter()
                    .map(|name| name.as_ptr()),
            );
        }
        if candidate.capabilities.ray_query {
            extension_pointers.push(ash::khr::ray_query::NAME.as_ptr());
        }
        // SAFETY: the selected physical device belongs to this live instance.
        let available_device = unsafe {
            instance
                .handle
                .enumerate_device_extension_properties(candidate.physical_device)
        }
        .unwrap_or_default();
        for name in &optional_device {
            let supported = available_device.iter().any(|property| {
                // SAFETY: extension names are fixed NUL-terminated arrays.
                (unsafe { CStr::from_ptr(property.extension_name.as_ptr()) }) == name.as_c_str()
            });
            if supported
                && !extension_pointers
                    .iter()
                    // SAFETY: every entry is a NUL-terminated name kept alive here.
                    .any(|existing| unsafe { CStr::from_ptr(*existing) } == name.as_c_str())
            {
                extension_pointers.push(name.as_ptr());
            }
        }

        let mut buffer_address =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut acceleration = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(true);
        let mut ray_pipeline =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
        let mut dynamic_rendering_feature =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        // Enable BC only when advertised; uncompressed assets remain usable everywhere.
        // SAFETY: the selected physical device belongs to this live instance.
        let supported = unsafe {
            instance
                .handle
                .get_physical_device_features(candidate.physical_device)
        };
        // R8 storage images use the extended formats. The rest are what a
        // general renderer asks of a desktop device, each taken where it is
        // offered: anisotropic samplers, arrays of cubes, and colour targets
        // that each blend their own way.
        let base_features = vk::PhysicalDeviceFeatures::default()
            .texture_compression_bc(supported.texture_compression_bc != 0)
            .shader_storage_image_extended_formats(
                supported.shader_storage_image_extended_formats != 0,
            )
            .sampler_anisotropy(supported.sampler_anisotropy != 0)
            .image_cube_array(supported.image_cube_array != 0)
            .independent_blend(supported.independent_blend != 0)
            // Many indirect draws in one call, and occlusion queries that
            // count samples rather than only say whether any passed.
            .multi_draw_indirect(supported.multi_draw_indirect != 0)
            .occlusion_query_precise(supported.occlusion_query_precise != 0);
        // `SV_VertexID` and `SV_InstanceID` compile to SPIR-V's draw
        // parameters (their Direct3D meaning leaves out the draw's base
        // vertex and first instance), so that is taken where offered too.
        let mut draw_parameters = vk::PhysicalDeviceShaderDrawParametersFeatures::default();
        {
            let mut offered =
                vk::PhysicalDeviceFeatures2::default().push_next(&mut draw_parameters);
            // SAFETY: the selected physical device belongs to this live
            // instance, and the chain is the one structure above.
            unsafe {
                instance
                    .handle
                    .get_physical_device_features2(candidate.physical_device, &mut offered);
            }
        }
        let mut device_info = vk::DeviceCreateInfo::default()
            .enabled_features(&base_features)
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&extension_pointers)
            .push_next(&mut buffer_address)
            .push_next(&mut dynamic_rendering_feature)
            .push_next(&mut draw_parameters);
        if with_ray_tracing {
            device_info = device_info
                .push_next(&mut acceleration)
                .push_next(&mut ray_pipeline);
        }
        if candidate.capabilities.ray_query {
            device_info = device_info.push_next(&mut ray_query);
        }
        // SAFETY: the selected adapter advertised every enabled extension and
        // feature, and all create-info backing arrays live through the call.
        let device = unsafe {
            instance
                .handle
                .create_device(candidate.physical_device, &device_info, None)
        }
        .map_err(|result| RhiError::vk("vkCreateDevice", result))?;
        let mut device = DeviceOwner {
            handle: device,
            health: vk::Fence::null(),
            pipeline_cache: vk::PipelineCache::null(),
        };
        // SAFETY: the device is live; neither create info borrows anything.
        unsafe {
            device.health = device
                .handle
                .create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
                .map_err(|result| RhiError::vk("vkCreateFence(health)", result))?;
            device.pipeline_cache = device
                .handle
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
                .map_err(|result| RhiError::vk("vkCreatePipelineCache", result))?;
        }
        let debug_names = debug_utils
            .then(|| ash::ext::debug_utils::Device::new(&instance.handle, &device.handle));
        // SAFETY: queue zero exists because one queue was requested above.
        let queue = unsafe {
            device
                .handle
                .get_device_queue(candidate.queue_family_index, 0)
        };
        // SAFETY: the physical device belongs to this live instance.
        let memory_properties = unsafe {
            instance
                .handle
                .get_physical_device_memory_properties(candidate.physical_device)
        };
        let acceleration_structure =
            ash::khr::acceleration_structure::Device::new(&instance.handle, &device.handle);
        let ray_tracing_pipeline =
            ash::khr::ray_tracing_pipeline::Device::new(&instance.handle, &device.handle);
        let dynamic_rendering =
            ash::khr::dynamic_rendering::Device::new(&instance.handle, &device.handle);
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.handle.clone(),
            device: device.handle.clone(),
            physical_device: candidate.physical_device,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::default(),
        })
        .map_err(|error| RhiError::Allocation {
            backend: "vulkan",
            operation: "create allocator",
            message: error.to_string(),
        })?;

        Ok(Self {
            acceleration_structure,
            ray_tracing_pipeline,
            dynamic_rendering,
            allocator: Arc::new(Mutex::new(allocator)),
            descriptor_pages: Mutex::new(Vec::new()),
            logical: device,
            pipeline_cache_lock: std::sync::RwLock::new(()),
            debug_names,
            debug_messenger,
            _instance: instance,
            physical_device: candidate.physical_device,
            queue,
            queue_lock: Arc::new(Mutex::new(())),
            queue_family_index: candidate.queue_family_index,
            sampler_anisotropy: supported.sampler_anisotropy != 0,
            multi_draw_indirect: supported.multi_draw_indirect != 0,
            occlusion_query_precise: supported.occlusion_query_precise != 0,
            memory_properties,
            capabilities: candidate.capabilities,
        })
    }

    /// Validation errors seen by the debug messenger, when debug tooling was
    /// requested and the validation layer is installed.
    #[must_use]
    pub fn debug_error_count(&self) -> Option<u64> {
        self.debug_messenger.as_ref().map(|messenger| {
            messenger
                .sink
                .errors
                .load(std::sync::atomic::Ordering::Relaxed)
        })
    }

    /// What the validation layer has said since the device was opened, oldest
    /// first: nothing when debug tooling was not asked for.
    #[must_use]
    pub fn debug_messages(&self) -> Vec<crate::DebugMessage> {
        self.debug_messenger
            .as_ref()
            .map(|messenger| crate::sync::lock_anyway(&messenger.sink.messages).clone())
            .unwrap_or_default()
    }

    #[must_use]
    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    /// [`RhiError::DeviceLost`] once the device has stopped working.
    pub(crate) fn check_device(&self) -> Result<(), RhiError> {
        // SAFETY: the fence belongs to this live device.
        unsafe { self.logical.handle.get_fence_status(self.logical.health) }
            .map(|_| ())
            .map_err(|result| RhiError::vk("vkGetFenceStatus(health)", result))
    }

    /// The failure of an allocation: the device having been lost, when
    /// that is why (the allocator reports it only as text).
    pub(crate) fn allocation_error(&self, operation: &'static str, message: String) -> RhiError {
        match self.check_device() {
            Err(lost) if lost.is_device_lost() => lost,
            _ => RhiError::Allocation {
                backend: "vulkan",
                operation,
                message,
            },
        }
    }

    /// Give `handle` a name capture tools and validation messages show.
    pub(crate) fn name_object<T: vk::Handle>(&self, handle: T, label: &str) {
        let (Some(names), Ok(label)) = (&self.debug_names, CString::new(label)) else {
            return;
        };
        if label.is_empty() {
            return;
        }
        let info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(handle)
            .object_name(&label);
        // SAFETY: the handle is a live object of this device, and the name
        // lives through the call. A failure only leaves the object unnamed.
        let _ = unsafe { names.set_debug_utils_object_name(&info) };
    }

    /// The cache every pipeline of this device is created through, held
    /// for as long as the returned guard lives.
    pub(crate) fn pipeline_cache(&self) -> (std::sync::RwLockReadGuard<'_, ()>, vk::PipelineCache) {
        (
            self.pipeline_cache_lock
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            self.logical.pipeline_cache,
        )
    }

    pub(crate) fn pipeline_cache_data(&self) -> Result<Vec<u8>, RhiError> {
        let _held = self
            .pipeline_cache_lock
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: the cache belongs to this live device and nothing else
        // uses it while the lock is held.
        unsafe {
            self.logical
                .handle
                .get_pipeline_cache_data(self.logical.pipeline_cache)
        }
        .map_err(|result| RhiError::vk("vkGetPipelineCacheData", result))
    }

    pub(crate) fn load_pipeline_cache(&self, data: &[u8]) -> Result<bool, RhiError> {
        // The header every cache starts with: its own length, the version,
        // then the vendor, device and cache identity of the driver that
        // wrote it. Data from anything else is left out here, rather than
        // trusting every driver to do so.
        // SAFETY: the physical device belongs to this live instance.
        let properties = unsafe {
            self._instance
                .handle
                .get_physical_device_properties(self.physical_device)
        };
        let word = |at: usize| {
            data.get(at..at + 4)
                .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        };
        let ours = word(0).is_some_and(|length| length >= 32)
            && word(4) == Some(vk::PipelineCacheHeaderVersion::ONE.as_raw() as u32)
            && word(8) == Some(properties.vendor_id)
            && word(12) == Some(properties.device_id)
            && data.get(16..32) == Some(&properties.pipeline_cache_uuid[..]);
        if !ours {
            return Ok(false);
        }
        let create = vk::PipelineCacheCreateInfo::default().initial_data(data);
        let _held = self
            .pipeline_cache_lock
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: the data lives through creation; the merge has the
        // device's cache to itself while the lock is held, and the loaded
        // cache is destroyed once merged.
        unsafe {
            let loaded = self
                .logical
                .handle
                .create_pipeline_cache(&create, None)
                .map_err(|result| RhiError::vk("vkCreatePipelineCache(loaded)", result))?;
            let merged = self
                .logical
                .handle
                .merge_pipeline_caches(self.logical.pipeline_cache, &[loaded]);
            self.logical.handle.destroy_pipeline_cache(loaded, None);
            merged.map_err(|result| RhiError::vk("vkMergePipelineCaches", result))?;
        }
        Ok(true)
    }

    #[must_use]
    pub(crate) fn native_device(&self) -> crate::NativeDevice {
        use ash::vk::Handle;
        crate::NativeDevice::Vulkan {
            instance: self._instance.handle.handle().as_raw(),
            physical_device: self.physical_device.as_raw(),
            device: self.logical.handle.handle().as_raw(),
            queue: self.queue.as_raw(),
            queue_family: self.queue_family_index,
            get_instance_proc_addr: self._instance._entry.static_fn().get_instance_proc_addr
                as usize,
            get_device_proc_addr: self._instance.handle.fp_v1_0().get_device_proc_addr as usize,
        }
    }

    pub(crate) fn allocator(&self) -> Arc<Mutex<Allocator>> {
        Arc::clone(&self.allocator)
    }

    pub(crate) fn lock_allocator(&self) -> MutexGuard<'_, Allocator> {
        lock_allocator(&self.allocator)
    }

    fn allocate_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_flags: vk::MemoryPropertyFlags,
        device_address: bool,
    ) -> Result<BufferAllocation, RhiError> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the create info contains no dangling pointers.
        let handle = unsafe { self.logical.handle.create_buffer(&create_info, None) }
            .map_err(|result| RhiError::vk("vkCreateBuffer", result))?;
        // SAFETY: `handle` belongs to this device and remains live.
        let requirements = unsafe { self.logical.handle.get_buffer_memory_requirements(handle) };
        let memory_type = find_memory_type(
            &self.memory_properties,
            requirements.memory_type_bits,
            memory_flags,
        )
        .ok_or(RhiError::NoMemoryType {
            required_bits: requirements.memory_type_bits,
            required_flags: memory_flags.as_raw(),
        });
        let memory_type = match memory_type {
            Ok(index) => index,
            Err(error) => {
                // SAFETY: allocation never happened, so only the buffer exists.
                unsafe { self.logical.handle.destroy_buffer(handle, None) };
                return Err(error);
            }
        };
        let mut address_flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        if device_address {
            allocate_info = allocate_info.push_next(&mut address_flags);
        }
        // SAFETY: the memory type came from this physical device and the size
        // comes from this buffer's requirements.
        let memory = match unsafe { self.logical.handle.allocate_memory(&allocate_info, None) } {
            Ok(memory) => memory,
            Err(result) => {
                // SAFETY: no memory was allocated, so only the buffer exists.
                unsafe { self.logical.handle.destroy_buffer(handle, None) };
                return Err(RhiError::vk("vkAllocateMemory", result));
            }
        };
        // SAFETY: memory was allocated for this buffer's exact requirements.
        if let Err(result) = unsafe { self.logical.handle.bind_buffer_memory(handle, memory, 0) } {
            // SAFETY: neither object has pending GPU use during construction.
            unsafe {
                self.logical.handle.free_memory(memory, None);
                self.logical.handle.destroy_buffer(handle, None);
            }
            return Err(RhiError::vk("vkBindBufferMemory", result));
        }

        Ok(BufferAllocation {
            device: self.logical.handle.clone(),
            handle,
            memory,
            allocation_size: requirements.size,
        })
    }
}

/// Lock a shared allocator, recovering the guard if a prior panic poisoned it.
pub(crate) fn lock_allocator(allocator: &Mutex<Allocator>) -> MutexGuard<'_, Allocator> {
    allocator
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn select_candidate(
    instance: &ash::Instance,
    adapter_preference: AdapterPreference,
    require_ray_tracing: bool,
) -> Result<Candidate, RhiError> {
    // SAFETY: `instance` is live and enumeration writes an owned handle list.
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|result| RhiError::vk("vkEnumeratePhysicalDevices", result))?;
    let mut candidates = Vec::new();
    for physical_device in devices {
        if let Ok(candidate) = inspect_candidate(instance, physical_device, require_ray_tracing) {
            candidates.push(candidate);
        }
    }
    candidates
        .into_iter()
        .max_by_key(|candidate| adapter_score(candidate.device_type, adapter_preference))
        .ok_or(RhiError::NoRayTracingDevice)
}

fn adapter_score(device_type: vk::PhysicalDeviceType, adapter_preference: AdapterPreference) -> u8 {
    match (adapter_preference, device_type) {
        (AdapterPreference::HighPerformance, vk::PhysicalDeviceType::DISCRETE_GPU)
        | (AdapterPreference::LowPower, vk::PhysicalDeviceType::INTEGRATED_GPU) => 4,
        (AdapterPreference::HighPerformance, vk::PhysicalDeviceType::INTEGRATED_GPU)
        | (AdapterPreference::LowPower, vk::PhysicalDeviceType::DISCRETE_GPU) => 3,
        (_, vk::PhysicalDeviceType::VIRTUAL_GPU) => 2,
        (_, vk::PhysicalDeviceType::CPU) => 1,
        _ => 0,
    }
}

fn inspect_candidate(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    require_ray_tracing: bool,
) -> Result<Candidate, RhiError> {
    // SAFETY: every query uses a physical-device handle from this instance.
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    // SAFETY: Vulkan guarantees a NUL-terminated fixed device-name array.
    let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    if properties.api_version < MINIMUM_API_VERSION {
        return Err(RhiError::NoRayTracingDevice);
    }

    // SAFETY: extension enumeration returns owned property records.
    let available = unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .map_err(|result| RhiError::vk("vkEnumerateDeviceExtensionProperties", result))?;
    let has_extension = |required: &CStr| {
        available.iter().any(|property| {
            // SAFETY: Vulkan extension names are fixed NUL-terminated arrays.
            let present = unsafe { CStr::from_ptr(property.extension_name.as_ptr()) };
            present == required
        })
    };
    let missing_base: Vec<String> = base_device_extensions()
        .iter()
        .filter(|required| !has_extension(required))
        .map(|extension| extension.to_string_lossy().into_owned())
        .collect();
    if !missing_base.is_empty() {
        return Err(RhiError::MissingDeviceExtensions {
            device: name,
            extensions: missing_base,
        });
    }
    let missing_rt: Vec<String> = ray_tracing_device_extensions()
        .iter()
        .filter(|required| !has_extension(required))
        .map(|extension| extension.to_string_lossy().into_owned())
        .collect();
    if require_ray_tracing && !missing_rt.is_empty() {
        return Err(RhiError::MissingDeviceExtensions {
            device: name,
            extensions: missing_rt,
        });
    }

    let mut buffer_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
    let mut acceleration = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
    let mut ray_pipeline = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
    let mut dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures::default();
    let mut ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut buffer_address)
        .push_next(&mut acceleration)
        .push_next(&mut ray_pipeline)
        .push_next(&mut dynamic_rendering)
        .push_next(&mut ray_query);
    // SAFETY: all output feature structures remain alive through this query.
    unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
    let mut missing_features = Vec::new();
    if buffer_address.buffer_device_address == vk::FALSE {
        missing_features.push("bufferDeviceAddress");
    }
    if dynamic_rendering.dynamic_rendering == vk::FALSE {
        missing_features.push("dynamicRendering");
    }
    let mut missing_rt_features = Vec::new();
    if acceleration.acceleration_structure == vk::FALSE {
        missing_rt_features.push("accelerationStructure");
    }
    if ray_pipeline.ray_tracing_pipeline == vk::FALSE {
        missing_rt_features.push("rayTracingPipeline");
    }
    if require_ray_tracing {
        missing_features.extend(missing_rt_features.iter().copied());
    }
    if !missing_features.is_empty() {
        return Err(RhiError::MissingRayTracingFeatures {
            device: name,
            features: missing_features,
        });
    }
    let ray_tracing = missing_rt.is_empty() && missing_rt_features.is_empty();

    // SAFETY: this returns an owned queue-family property list.
    let queue_family_index =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
            .iter()
            .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|index| index as u32)
            .ok_or_else(|| RhiError::NoComputeQueue {
                device: name.clone(),
            })?;

    let mut acceleration_properties =
        vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut ray_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    let mut properties2 = vk::PhysicalDeviceProperties2::default()
        .push_next(&mut acceleration_properties)
        .push_next(&mut ray_properties);
    // SAFETY: all output property structures remain alive through this query.
    unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };
    let capabilities = Capabilities {
        ray_tracing,
        // Rays traced from compute and fragment shaders: an extension of
        // its own beside the acceleration structures it reads.
        ray_query: ray_tracing
            && has_extension(ash::khr::ray_query::NAME)
            && ray_query.ray_query != vk::FALSE,
        device_name: name,
        api_version: properties.api_version,
        shader_group_handle_size: ray_properties.shader_group_handle_size,
        shader_group_handle_alignment: ray_properties.shader_group_handle_alignment,
        shader_group_base_alignment: ray_properties.shader_group_base_alignment,
        max_ray_recursion_depth: ray_properties.max_ray_recursion_depth,
        max_ray_dispatch_invocations: ray_properties.max_ray_dispatch_invocation_count,
        max_geometry_count: acceleration_properties.max_geometry_count,
        max_instance_count: acceleration_properties.max_instance_count,
        scratch_alignment: acceleration_properties
            .min_acceleration_structure_scratch_offset_alignment,
    };
    Ok(Candidate {
        physical_device,
        queue_family_index,
        device_type: properties.device_type,
        capabilities,
    })
}

fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    allowed_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    properties.memory_types[..properties.memory_type_count as usize]
        .iter()
        .enumerate()
        .find(|(index, memory)| {
            allowed_bits & (1_u32 << index) != 0 && memory.property_flags.contains(required)
        })
        .map(|(index, _)| index as u32)
}

struct BufferAllocation {
    device: ash::Device,
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    allocation_size: u64,
}

impl BufferAllocation {
    fn write_slice<T: Copy>(&self, values: &[T]) -> Result<(), RhiError> {
        let size = std::mem::size_of_val(values) as u64;
        if size > self.allocation_size {
            return Err(RhiError::InvalidUpload(
                "upload exceeds its Vulkan buffer allocation",
            ));
        }
        // SAFETY: callers only use HOST_VISIBLE | HOST_COHERENT allocations;
        // the mapped byte range is bounded by the check above.
        unsafe {
            let mapped = self
                .device
                .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(|result| RhiError::vk("vkMapMemory", result))?;
            ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                mapped.cast::<u8>(),
                size as usize,
            );
            self.device.unmap_memory(self.memory);
        }
        Ok(())
    }

    fn write_bytes_at(&self, offset: u64, bytes: &[u8]) -> Result<(), RhiError> {
        let size = bytes.len() as u64;
        if offset
            .checked_add(size)
            .is_none_or(|end| end > self.allocation_size)
        {
            return Err(RhiError::InvalidUpload(
                "shader binding table write exceeds its allocation",
            ));
        }
        // SAFETY: the SBT allocation is HOST_VISIBLE | HOST_COHERENT and the
        // checked range lies entirely inside it.
        unsafe {
            let mapped = self
                .device
                .map_memory(self.memory, offset, size, vk::MemoryMapFlags::empty())
                .map_err(|result| RhiError::vk("vkMapMemory", result))?;
            ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
            self.device.unmap_memory(self.memory);
        }
        Ok(())
    }

    fn device_address(&self) -> Result<u64, RhiError> {
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.handle);
        // SAFETY: this buffer was created and allocated with device-address use.
        let address = unsafe { self.device.get_buffer_device_address(&info) };
        if address == 0 {
            return Err(RhiError::ZeroDeviceAddress("buffer"));
        }
        Ok(address)
    }
}

impl Drop for BufferAllocation {
    fn drop(&mut self) {
        // SAFETY: the owning method waits for queue idle before local resources
        // can be dropped, and this allocation owns both handles exclusively.
        unsafe {
            self.device.destroy_buffer(self.handle, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

struct CommandPool {
    device: ash::Device,
    handle: vk::CommandPool,
}

impl CommandPool {
    fn new(device: &ash::Device, queue_family_index: u32) -> Result<Self, RhiError> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        // SAFETY: the queue family belongs to this logical device.
        let handle = unsafe { device.create_command_pool(&info, None) }
            .map_err(|result| RhiError::vk("vkCreateCommandPool", result))?;
        Ok(Self {
            device: device.clone(),
            handle,
        })
    }

    fn allocate_primary(&self) -> Result<vk::CommandBuffer, RhiError> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.handle)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool is live and belongs to this device.
        unsafe { self.device.allocate_command_buffers(&info) }
            .map_err(|result| RhiError::vk("vkAllocateCommandBuffers", result))?
            .into_iter()
            .next()
            .ok_or(RhiError::Vulkan {
                operation: "vkAllocateCommandBuffers returned no command buffer",
                result: vk::Result::ERROR_UNKNOWN.as_raw(),
            })
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        // SAFETY: queue idle is awaited before this local owner drops.
        unsafe { self.device.destroy_command_pool(self.handle, None) };
    }
}

#[cfg(test)]
#[path = "tests/vulkan_device.rs"]
mod tests;
