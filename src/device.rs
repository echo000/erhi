use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(all(feature = "dx12", windows))]
use crate::dx12::Dx12Device;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanDevice;
use crate::{BackendAttempt, Capabilities, RhiError};

/// Native backend preference used only while creating a device.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Whichever opens: Direct3D 12 first on Windows, then Vulkan.
    #[default]
    Auto,
    /// Vulkan.
    Vulkan,
    /// Direct3D 12. Windows only.
    Dx12,
}

impl Backend {
    /// The backend's name in lower case, as [`std::str::FromStr`] reads it
    /// back.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Vulkan => "vulkan",
            Self::Dx12 => "dx12",
        }
    }

    /// Whether this selection can be satisfied by the current binary.
    #[must_use]
    pub const fn is_compiled(self) -> bool {
        match self {
            Self::Auto => true,
            Self::Vulkan => cfg!(feature = "vulkan"),
            Self::Dx12 => cfg!(all(feature = "dx12", windows)),
        }
    }
}

impl FromStr for Backend {
    type Err = RhiError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "vulkan" | "vk" => Ok(Self::Vulkan),
            "dx12" | "d3d12" => Ok(Self::Dx12),
            _ => Err(RhiError::InvalidBackendName(name.to_owned())),
        }
    }
}

/// How serious a message of the native debug layer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DebugSeverity {
    /// Something that works and is likely not what was meant.
    Warning,
    /// A call that broke the native API's rules: its result is not defined.
    Error,
}

/// One message of the native debug layer ([`Device::debug_messages`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebugMessage {
    /// How serious it is.
    pub severity: DebugSeverity,
    /// The message text.
    pub text: String,
}

/// Adapter ranking used during the one-time backend/device selection.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdapterPreference {
    /// The discrete adapter, where there is one.
    #[default]
    HighPerformance,
    /// The integrated adapter, where there is one.
    LowPower,
}

/// Options that apply only to the Vulkan backend. Direct3D
/// 12 reads none of it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VulkanOptions<'a> {
    /// Instance extensions to enable as well, each only where the driver
    /// has it: a vendor SDK may need a few.
    pub instance_extensions: &'a [&'a str],
    /// Device extensions to enable as well, each only where the driver has
    /// it.
    pub device_extensions: &'a [&'a str],
}

/// Device creation settings. They are consumed by `Device::open`; no
/// backend selection remains in the frame or pass APIs afterward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceOptions<'a> {
    /// Which backend to open, or [`Backend::Auto`] for whichever does.
    pub backend: Backend,
    /// Which adapter to prefer on a machine with several.
    pub adapter_preference: AdapterPreference,
    /// Frames in flight, fixed when the device is opened. `FrameRing`s
    /// created from this device use this size; presentation clamps it to
    /// the swapchain's image count. Two by default.
    pub frames_in_flight: u32,
    /// Require hardware ray tracing. On by default. With it off, an adapter
    /// without the ray-tracing extensions (Vulkan) or tier (Direct3D 12)
    /// still opens; `capabilities().ray_tracing` reports what was found, and
    /// ray-tracing calls return an error.
    pub require_ray_tracing: bool,
    /// Enable native debug tooling at device creation: the Vulkan
    /// validation layer with a debug-utils messenger, or the DX12 debug
    /// layer with break-on-error and DRED breadcrumbs. Costs performance;
    /// intended for development and the validation soak tests.
    pub debug: bool,
    /// Vulkan instance and device extensions to enable as well, each only
    /// where the driver has it (a third-party SDK may need a few: DLSS
    /// does). Names the driver lacks are skipped; Direct3D 12 ignores both.
    pub vulkan: VulkanOptions<'a>,
}

impl Default for DeviceOptions<'_> {
    fn default() -> Self {
        Self {
            backend: Backend::default(),
            adapter_preference: AdapterPreference::default(),
            frames_in_flight: 2,
            require_ray_tracing: true,
            debug: false,
            vulkan: VulkanOptions {
                instance_extensions: &[],
                device_extensions: &[],
            },
        }
    }
}

/// Native device whose backend is selected exactly once during creation.
///
/// The closed backend enum is private, so frame and pass code cannot branch on
/// a requested backend or replace it later. Shared concrete devices keep this wrapper
/// small without introducing trait objects.
#[derive(Debug, Clone)]
pub struct Device {
    id: u64,
    pub(crate) memory: std::sync::Arc<std::sync::Mutex<crate::memory::MemoryTracker>>,
    frames_in_flight: u32,
    pub(crate) backend: BackendDevice,
}

static NEXT_RHI_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub(crate) enum BackendDevice {
    #[cfg(feature = "vulkan")]
    Vulkan(std::sync::Arc<VulkanDevice>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(std::sync::Arc<Dx12Device>),
}

impl Device {
    /// Select and open one native backend. The returned enum arm remains fixed
    /// for the lifetime of this device.
    pub fn open(options: DeviceOptions<'_>) -> Result<Self, RhiError> {
        if !(1..=8).contains(&options.frames_in_flight) {
            return Err(RhiError::InvalidFramesInFlight(options.frames_in_flight));
        }
        if options.backend != Backend::Auto {
            return Self::request_backend(options);
        }

        let mut attempts = Vec::new();
        for backend in Self::auto_backend_order() {
            match Self::request_backend(DeviceOptions {
                backend: *backend,
                ..options
            }) {
                Ok(device) => return Ok(device),
                Err(error) => attempts.push(BackendAttempt {
                    backend: backend.name(),
                    error,
                }),
            }
        }
        Err(RhiError::NoBackendAvailable { attempts })
    }

    fn request_backend(options: DeviceOptions<'_>) -> Result<Self, RhiError> {
        match options.backend {
            Backend::Auto => Self::open(options),
            #[cfg(feature = "vulkan")]
            Backend::Vulkan => VulkanDevice::open(
                options.adapter_preference,
                options.debug,
                options.require_ray_tracing,
                (
                    options.vulkan.instance_extensions,
                    options.vulkan.device_extensions,
                ),
            )
            .map(std::sync::Arc::new)
            .map(BackendDevice::Vulkan)
            .map(|backend| Self::from_backend(backend, options.frames_in_flight)),
            #[cfg(not(feature = "vulkan"))]
            Backend::Vulkan => Err(RhiError::BackendNotCompiled("vulkan")),
            #[cfg(all(feature = "dx12", windows))]
            Backend::Dx12 => Dx12Device::open(
                options.adapter_preference,
                options.debug,
                options.require_ray_tracing,
            )
            .map(std::sync::Arc::new)
            .map(BackendDevice::Dx12)
            .map(|backend| Self::from_backend(backend, options.frames_in_flight)),
            #[cfg(not(all(feature = "dx12", windows)))]
            Backend::Dx12 => Err(RhiError::BackendNotCompiled("dx12")),
        }
    }

    /// Validation errors reported by the native debug tooling since device
    /// creation, when [`DeviceOptions::debug`] was enabled. `None` when it was
    /// not, or when the debug layer's messages cannot be read: never a
    /// count of zero. [`Self::debug_messages`] returns the messages.
    #[must_use]
    pub fn debug_error_count(&self) -> Option<u64> {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => device.debug_error_count(),
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => device.debug_error_count(),
        }
    }

    /// The warnings and errors the native debug layer has given since the
    /// device was opened, oldest first, when [`DeviceOptions::debug`] was
    /// enabled; nothing otherwise. This crate prints nothing itself: print
    /// these, log them or assert on them as suits the program. A device
    /// keeps the first thousand or so.
    #[must_use]
    pub fn debug_messages(&self) -> Vec<DebugMessage> {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => device.debug_messages(),
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => device.debug_messages().unwrap_or_default(),
        }
    }

    /// Whether the device still works: [`RhiError::DeviceLost`] once the
    /// driver has reset or crashed or the adapter is gone. Cheap enough to
    /// ask once a frame. A lost device also fails the calls made of it with
    /// the same error; this asks without making one.
    pub fn check_device(&self) -> Result<(), RhiError> {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => device.check_device(),
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => device.check_device(),
        }
    }

    /// Make the device lost, as a driver crash would, to exercise recovery.
    /// False where the backend has no way to (Vulkan): the device is then
    /// untouched. Every device this process holds on the same Direct3D 12
    /// adapter is the same device, and is lost with it.
    pub fn simulate_device_loss(&self) -> bool {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(_) => false,
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => {
                device.remove_device();
                true
            }
        }
    }

    /// What the device has kept of the pipelines created so far, to save and
    /// hand to [`Self::load_pipeline_cache`] on a later run: creating the
    /// same pipelines then skips most of the driver's compiling. Vulkan
    /// keeps all three kinds of pipeline; Direct3D 12 keeps render and
    /// compute pipelines (its ray-tracing state objects have no cache).
    /// Empty where the device keeps none.
    pub fn pipeline_cache_data(&self) -> Result<Vec<u8>, RhiError> {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => device.pipeline_cache_data(),
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => device.pipeline_cache_data(),
        }
    }

    /// Take the pipelines an earlier run kept ([`Self::pipeline_cache_data`]).
    /// Call it before creating pipelines. False when the data is not from
    /// this backend, adapter and driver version, or is damaged: it is then
    /// left out, and pipelines are compiled as if nothing had been saved.
    pub fn load_pipeline_cache(&self, data: &[u8]) -> Result<bool, RhiError> {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => device.load_pipeline_cache(data),
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => device.load_pipeline_cache(data),
        }
    }

    /// Complete device queue work, including presentation submitted after a
    /// frame-ring fence. Use for device replacement, not normal frame pacing.
    pub fn wait_idle(&self) -> Result<(), RhiError> {
        crate::submission::wait_idle(&self.backend)
    }

    fn from_backend(backend: BackendDevice, frames_in_flight: u32) -> Self {
        Self {
            id: NEXT_RHI_ID.fetch_add(1, Ordering::Relaxed),
            memory: Default::default(),
            frames_in_flight,
            backend,
        }
    }

    /// A number no other device of this process has, to tell devices apart.
    /// Every resource created by this `Device` rejects use on another device.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// The startup-configured frames-in-flight count consumed by
    /// [`Device::create_frame_ring`].
    #[must_use]
    pub const fn frames_in_flight(&self) -> u32 {
        self.frames_in_flight
    }

    fn auto_backend_order() -> &'static [Backend] {
        // Prefer the native Windows API, retaining Vulkan as an automatic
        // fallback. Other platforms currently use Vulkan. Explicit backend
        // requests bypass this list and saved user preferences remain intact.
        const BACKENDS: &[Backend] = &[
            #[cfg(all(feature = "dx12", windows))]
            Backend::Dx12,
            #[cfg(feature = "vulkan")]
            Backend::Vulkan,
        ];
        BACKENDS
    }

    /// The backend this device opened on.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(_) => Backend::Vulkan,
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(_) => Backend::Dx12,
        }
    }

    /// Native device handles for third-party SDKs (see [`crate::NativeDevice`]).
    #[must_use]
    pub fn native_device(&self) -> crate::NativeDevice {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => device.native_device(),
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => device.native_device(),
        }
    }

    /// The device's capabilities and limits.
    #[must_use]
    pub fn capabilities(&self) -> &Capabilities {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => device.capabilities(),
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => device.capabilities(),
        }
    }

    /// Bytes of device memory currently reserved by the shared allocator.
    ///
    /// This is a stability diagnostic: steady-state rendering must not grow
    /// it frame over frame.
    #[must_use]
    pub fn allocated_bytes(&self) -> u64 {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => {
                device
                    .lock_allocator()
                    .generate_report()
                    .total_capacity_bytes
            }
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => {
                device
                    .lock_allocator()
                    .generate_report()
                    .total_capacity_bytes
            }
        }
    }
}

#[cfg(test)]
#[path = "tests/device.rs"]
mod tests;
