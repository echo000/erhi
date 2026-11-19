//! The Direct3D 12 backend.
//!
//! This backend is selectable explicitly for capability and resource probes,
//! but is withheld from `Backend::Auto` until the procedural scene passes have DX12

//! implementations.

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

use std::ffi::c_void;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, MutexGuard};
use std::{fmt, mem};

pub(crate) use acceleration_structure::{Dx12AccelerationStructure, Dx12BlasBuild, Dx12TlasBuild};
pub(crate) use buffer::Dx12Buffer;
pub(crate) use command_encoder::Dx12CommandEncoder;
pub(crate) use compute_pipeline::Dx12ComputePipeline;
pub(crate) use descriptor_set::Dx12DescriptorSet;
pub(crate) use frame::Dx12FrameRing;
use gpu_allocator::d3d12::{Allocator, AllocatorCreateDesc, ID3D12DeviceVersion};
use gpu_allocator::{AllocationSizes, AllocatorDebugSettings};
pub(crate) use image::{Dx12Image, Dx12ImageView};
pub(crate) use pipeline_layout::Dx12PipelineLayout;
pub(crate) use ray_tracing_pipeline::Dx12RayTracingPipeline;
pub(crate) use render_pipeline::Dx12RenderPipeline;
pub(crate) use sampler::Dx12Sampler;
pub(crate) use shader::Dx12ShaderModule;
pub(crate) use surface::Dx12Surface;
pub(crate) use timestamp_query::Dx12TimestampQuerySet;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_1;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
    D3D12_COMMAND_QUEUE_PRIORITY_NORMAL, D3D12_DRED_ENABLEMENT_FORCED_ON,
    D3D12_FEATURE_D3D12_OPTIONS5, D3D12_FEATURE_DATA_D3D12_OPTIONS5, D3D12_FENCE_FLAG_NONE,
    D3D12_MESSAGE, D3D12_MESSAGE_SEVERITY_CORRUPTION, D3D12_MESSAGE_SEVERITY_ERROR,
    D3D12_MESSAGE_SEVERITY_WARNING, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT,
    D3D12_RAYTRACING_MAX_DECLARABLE_TRACE_RECURSION_DEPTH,
    D3D12_RAYTRACING_MAX_GEOMETRIES_PER_BOTTOM_LEVEL_ACCELERATION_STRUCTURE,
    D3D12_RAYTRACING_MAX_INSTANCES_PER_TOP_LEVEL_ACCELERATION_STRUCTURE,
    D3D12_RAYTRACING_MAX_RAY_GENERATION_SHADER_THREADS,
    D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT,
    D3D12_RAYTRACING_TIER_NOT_SUPPORTED, D3D12CreateDevice, D3D12GetDebugInterface,
    ID3D12CommandQueue, ID3D12Debug, ID3D12Device, ID3D12Device5,
    ID3D12DeviceRemovedExtendedDataSettings, ID3D12Fence, ID3D12InfoQueue,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_ADAPTER_FLAG3_SOFTWARE, DXGI_CREATE_FACTORY_DEBUG,
    DXGI_CREATE_FACTORY_FLAGS, DXGI_ERROR_NOT_FOUND, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
    DXGI_GPU_PREFERENCE_MINIMUM_POWER, IDXGIAdapter4, IDXGIFactory6,
};
use windows::core::Interface;

use crate::{AdapterPreference, Capabilities, RhiError};

/// A Direct3D 12 device with a direct queue and DXR support.
pub struct Dx12Device {
    _factory: IDXGIFactory6,
    _adapter: IDXGIAdapter4,
    native: ID3D12Device5,
    direct_queue: ID3D12CommandQueue,
    direct_fence: ID3D12Fence,
    next_fence_value: AtomicU64,
    fence_signal_lock: Mutex<()>,
    allocator: Arc<Mutex<Allocator>>,
    /// The heap every descriptor table is a run of, made with the first.
    tables: Arc<Mutex<Option<descriptor_set::TableHeap>>>,
    /// The one sampler heap, made when a set first holds a sampler.
    sampler_heap: Mutex<Option<Arc<sampler::SamplerHeap>>>,
    /// The render and compute pipelines created so far, as the driver
    /// compiled them. None where the system has no pipeline libraries.
    pipeline_library: Mutex<Option<PipelineLibrary>>,
    pub(crate) indirect: IndirectSignatures,
    capabilities: Capabilities,
}

/// What a record of GPU-written arguments is, for each call that reads
/// one. None of them sets a root argument, so every pipeline shares them.
pub(crate) struct IndirectSignatures {
    /// Four words: vertex count, instance count, first vertex, first instance.
    pub(crate) draw: windows::Win32::Graphics::Direct3D12::ID3D12CommandSignature,
    /// Five words: index count, instance count, first index, base vertex,
    /// first instance.
    pub(crate) draw_indexed: windows::Win32::Graphics::Direct3D12::ID3D12CommandSignature,
    /// Three words: the thread-group counts.
    pub(crate) dispatch: windows::Win32::Graphics::Direct3D12::ID3D12CommandSignature,
}

fn indirect_signatures(device: &ID3D12Device5) -> Result<IndirectSignatures, RhiError> {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_COMMAND_SIGNATURE_DESC, D3D12_INDIRECT_ARGUMENT_DESC, D3D12_INDIRECT_ARGUMENT_TYPE,
        D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
        D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, ID3D12CommandSignature,
    };
    let signature = |kind: D3D12_INDIRECT_ARGUMENT_TYPE, stride: u32| {
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: kind,
            ..Default::default()
        };
        let description = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: stride,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            ..Default::default()
        };
        let mut created: Option<ID3D12CommandSignature> = None;
        // SAFETY: the description lives through creation and names no root
        // arguments, so it needs no root signature.
        unsafe { device.CreateCommandSignature(&description, None, &mut created) }
            .map_err(|error| RhiError::dx12("ID3D12Device::CreateCommandSignature", error))?;
        created.ok_or(RhiError::Dx12 {
            operation: "ID3D12Device::CreateCommandSignature returned no signature",
            code: 0,
            message: "successful call returned a null interface".to_owned(),
        })
    };
    Ok(IndirectSignatures {
        draw: signature(D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, 16)?,
        draw_indexed: signature(D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, 20)?,
        dispatch: signature(D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, 12)?,
    })
}

/// A pipeline library and the bytes it was opened from: the library reads
/// them for as long as it lives, without its own copy.
struct PipelineLibrary {
    library: windows::Win32::Graphics::Direct3D12::ID3D12PipelineLibrary,
    _opened_from: Vec<u8>,
}

/// Give a Direct3D 12 object a name capture tools (PIX, RenderDoc) and the
/// debug layer's messages show.
pub(crate) fn name_object(object: &impl Interface, label: &str) {
    if label.is_empty() {
        return;
    }
    if let Ok(object) = object.cast::<windows::Win32::Graphics::Direct3D12::ID3D12Object>() {
        // SAFETY: the object is live and the name lives through the call.
        // A failure only leaves the object unnamed.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(label)) };
    }
}

/// A name for a pipeline in the library: the same description gives the
/// same name on every run (64-bit FNV-1a over what describes it).
pub(crate) fn pipeline_key(parts: &[&[u8]]) -> windows::core::HSTRING {
    windows::core::HSTRING::from(format!("{:016x}", fnv(parts)))
}

fn fnv(parts: &[&[u8]]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for part in parts {
        for byte in part.iter().chain(&(part.len() as u64).to_le_bytes()) {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// What saved pipelines start with: these eight bytes, the sixteen of
/// [`Dx12Device::library_identity`], then the hash of the library that follows.
const LIBRARY_MAGIC: &[u8; 8] = b"ERHIPSO1";

const LIBRARY_HEADER: usize = 32;

impl fmt::Debug for Dx12Device {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Dx12Device")
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

impl Dx12Device {
    pub(crate) fn native_device(&self) -> crate::NativeDevice {
        use windows::core::Interface;
        crate::NativeDevice::Dx12 {
            device: self.native.as_raw() as usize,
            queue: self.direct_queue.as_raw() as usize,
        }
    }

    /// Open a DXR device, optionally with the debug layer
    /// (break on corruption, retain errors) and DRED breadcrumb/page-fault capture.
    pub fn open(
        adapter_preference: AdapterPreference,
        debug: bool,
        require_ray_tracing: bool,
    ) -> Result<Self, RhiError> {
        if debug {
            // SAFETY: querying the debug interfaces has no device state yet;
            // both must be enabled before device creation.
            unsafe {
                let mut controller: Option<ID3D12Debug> = None;
                if D3D12GetDebugInterface(&mut controller).is_ok()
                    && let Some(controller) = controller
                {
                    controller.EnableDebugLayer();
                }
                let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
                if D3D12GetDebugInterface(&mut dred).is_ok()
                    && let Some(dred) = dred
                {
                    dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                    dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                }
            }
        }
        // SAFETY: factory creation returns an owned COM interface.
        let factory: IDXGIFactory6 = unsafe {
            CreateDXGIFactory2(if debug {
                DXGI_CREATE_FACTORY_DEBUG
            } else {
                DXGI_CREATE_FACTORY_FLAGS(0)
            })
            .map_err(|error| RhiError::dx12("CreateDXGIFactory2", error))?
        };
        let preference = match adapter_preference {
            AdapterPreference::HighPerformance => DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            AdapterPreference::LowPower => DXGI_GPU_PREFERENCE_MINIMUM_POWER,
        };

        let mut adapter_index = 0;
        loop {
            // SAFETY: DXGI owns enumeration state; a successful call returns
            // an independently reference-counted adapter interface.
            let adapter = match unsafe {
                factory.EnumAdapterByGpuPreference::<IDXGIAdapter4>(adapter_index, preference)
            } {
                Ok(adapter) => adapter,
                Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => {
                    return Err(RhiError::NoDx12RayTracingDevice);
                }
                Err(error) => {
                    return Err(RhiError::dx12(
                        "IDXGIFactory6::EnumAdapterByGpuPreference",
                        error,
                    ));
                }
            };
            adapter_index += 1;
            // SAFETY: `adapter` is a live IDXGIAdapter4 interface.
            let description = unsafe { adapter.GetDesc3() }
                .map_err(|error| RhiError::dx12("IDXGIAdapter4::GetDesc3", error))?;
            if description.Flags.contains(DXGI_ADAPTER_FLAG3_SOFTWARE) {
                continue;
            }

            let mut device = None;
            // SAFETY: the adapter remains live and the output receives an
            // owned ID3D12Device5 interface on success.
            if unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_1, &mut device) }.is_err()
            {
                continue;
            }
            let Some(device): Option<ID3D12Device5> = device else {
                continue;
            };
            // Every device this process holds on one adapter is the same
            // object: while anything keeps a removed one alive, this is it
            // again, and nothing made from it would work.
            // SAFETY: asking the removal reason has no side effects.
            if let Err(removed) = unsafe { device.GetDeviceRemovedReason() } {
                return Err(RhiError::DeviceLost {
                    backend: "dx12",
                    operation: "D3D12CreateDevice",
                    reason: format!(
                        "the adapter's removed device is still held by this process; drop \
                         everything made from it before asking for a new one ({removed})"
                    ),
                });
            }
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
            // SAFETY: the output pointer and byte size exactly match OPTIONS5.
            if unsafe {
                device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS5,
                    (&raw mut options).cast::<c_void>(),
                    mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() as u32,
                )
            }
            .is_err()
                || (require_ray_tracing
                    && options.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED)
            {
                continue;
            }

            let queue_description = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
                Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                NodeMask: 0,
            };
            // SAFETY: the descriptor is fully initialized and the device is
            // live. The returned queue is independently reference counted.
            let direct_queue: ID3D12CommandQueue = unsafe {
                device
                    .CreateCommandQueue(&raw const queue_description)
                    .map_err(|error| RhiError::dx12("ID3D12Device::CreateCommandQueue", error))?
            };
            // SAFETY: the fence is created from the same live device as the
            // direct queue and starts in the unsignalled zero state.
            let direct_fence: ID3D12Fence = unsafe {
                device
                    .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                    .map_err(|error| RhiError::dx12("ID3D12Device::CreateFence", error))?
            };
            if debug {
                // Corruption still breaks; plain validation errors are
                // *retained* and drained into the device-removed report:
                // breaking on them without a debugger attached fast-fails
                // the process with no message at all, which is how the
                // first DRED hunt produced three-line logs.
                if let Ok(info_queue) = device.cast::<ID3D12InfoQueue>() {
                    // SAFETY: the info queue belongs to the live device.
                    unsafe {
                        let _ =
                            info_queue.SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
                    }
                }
            }
            let base_device: ID3D12Device = device
                .cast()
                .map_err(|error| RhiError::dx12("ID3D12Device5 cast to ID3D12Device", error))?;
            let allocator = Allocator::new(&AllocatorCreateDesc {
                device: ID3D12DeviceVersion::Device(base_device),
                debug_settings: AllocatorDebugSettings::default(),
                allocation_sizes: AllocationSizes::default(),
            })
            .map_err(|error| RhiError::Allocation {
                backend: "dx12",
                operation: "create allocator",
                message: error.to_string(),
            })?;
            let capabilities = Capabilities {
                ray_tracing: options.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED,
                // `RayQuery` came with the second tier of DXR.
                ray_query: options.RaytracingTier.0
                    >= windows::Win32::Graphics::Direct3D12::D3D12_RAYTRACING_TIER_1_1.0,
                device_name: adapter_name(&description.Description),
                api_version: D3D_FEATURE_LEVEL_12_1.0 as u32,
                shader_group_handle_size: 32,
                shader_group_handle_alignment: D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT,
                shader_group_base_alignment: D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT,
                max_ray_recursion_depth: D3D12_RAYTRACING_MAX_DECLARABLE_TRACE_RECURSION_DEPTH,
                max_ray_dispatch_invocations: D3D12_RAYTRACING_MAX_RAY_GENERATION_SHADER_THREADS,
                max_geometry_count: u64::from(
                    D3D12_RAYTRACING_MAX_GEOMETRIES_PER_BOTTOM_LEVEL_ACCELERATION_STRUCTURE,
                ),
                max_instance_count: u64::from(
                    D3D12_RAYTRACING_MAX_INSTANCES_PER_TOP_LEVEL_ACCELERATION_STRUCTURE,
                ),
                scratch_alignment: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT,
            };
            let pipeline_library = open_pipeline_library(&device, Vec::new()).ok();
            let indirect = indirect_signatures(&device)?;
            return Ok(Self {
                _factory: factory,
                _adapter: adapter,
                native: device,
                direct_queue,
                direct_fence,
                next_fence_value: AtomicU64::new(1),
                fence_signal_lock: Mutex::new(()),
                allocator: Arc::new(Mutex::new(allocator)),
                tables: Arc::new(Mutex::new(None)),
                sampler_heap: Mutex::new(None),
                pipeline_library: Mutex::new(pipeline_library),
                indirect,
                capabilities,
            });
        }
    }

    #[must_use]
    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    /// The failure of an allocation: the device having been removed, when
    /// that is why (the allocator reports it only as text).
    pub(crate) fn allocation_error(&self, operation: &'static str, message: String) -> RhiError {
        match self.check_device() {
            Err(lost) if lost.is_device_lost() => lost,
            _ => RhiError::Allocation {
                backend: "dx12",
                operation,
                message,
            },
        }
    }

    /// The adapter and the driver that compiled this device's pipelines:
    /// its vendor, its device, and the driver's version. A library saved
    /// under another identity is left out without the driver being asked.
    fn library_identity(&self) -> [u8; 16] {
        // SAFETY: both calls only read the live adapter's properties.
        let (description, driver) = unsafe {
            (
                self._adapter.GetDesc1().unwrap_or_default(),
                self._adapter
                    .CheckInterfaceSupport(&windows::Win32::Graphics::Dxgi::IDXGIDevice::IID)
                    .unwrap_or(0),
            )
        };
        let mut identity = [0; 16];
        identity[..4].copy_from_slice(&description.VendorId.to_le_bytes());
        identity[4..8].copy_from_slice(&description.DeviceId.to_le_bytes());
        identity[8..].copy_from_slice(&driver.to_le_bytes());
        identity
    }

    /// [`RhiError::DeviceLost`] once the device has been removed.
    pub(crate) fn check_device(&self) -> Result<(), RhiError> {
        // SAFETY: asking the removal reason has no side effects.
        unsafe { self.native.GetDeviceRemovedReason() }.map_err(|removed| {
            command_encoder::device_removed_report(&self.native, "device check", removed)
        })
    }

    /// Remove the device, as a driver crash would.
    pub(crate) fn remove_device(&self) {
        // SAFETY: removal is a supported call on a live device; every later
        // call reports the device removed.
        unsafe { self.native.RemoveDevice() };
    }

    /// A render pipeline from the library when it holds one of this name
    /// and description, else compiled and added to it.
    pub(crate) fn cached_graphics_pipeline(
        &self,
        key: &windows::core::HSTRING,
        description: &windows::Win32::Graphics::Direct3D12::D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    ) -> windows::core::Result<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState> {
        let held = self
            .pipeline_library
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: the description and what it points at live through each
        // call. A name the library holds under another description fails
        // to load and to store; the pipeline is then simply compiled.
        unsafe {
            if let Some(held) = &*held
                && let Ok(pipeline) = held.library.LoadGraphicsPipeline(key, description)
            {
                return Ok(pipeline);
            }
            let pipeline: windows::Win32::Graphics::Direct3D12::ID3D12PipelineState =
                self.native.CreateGraphicsPipelineState(description)?;
            if let Some(held) = &*held {
                let _ = held.library.StorePipeline(key, &pipeline);
            }
            Ok(pipeline)
        }
    }

    /// The compute counterpart of [`Self::cached_graphics_pipeline`].
    pub(crate) fn cached_compute_pipeline(
        &self,
        key: &windows::core::HSTRING,
        description: &windows::Win32::Graphics::Direct3D12::D3D12_COMPUTE_PIPELINE_STATE_DESC,
    ) -> windows::core::Result<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState> {
        let held = self
            .pipeline_library
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: as for the render pipelines above.
        unsafe {
            if let Some(held) = &*held
                && let Ok(pipeline) = held.library.LoadComputePipeline(key, description)
            {
                return Ok(pipeline);
            }
            let pipeline: windows::Win32::Graphics::Direct3D12::ID3D12PipelineState =
                self.native.CreateComputePipelineState(description)?;
            if let Some(held) = &*held {
                let _ = held.library.StorePipeline(key, &pipeline);
            }
            Ok(pipeline)
        }
    }

    pub(crate) fn pipeline_cache_data(&self) -> Result<Vec<u8>, RhiError> {
        let held = self
            .pipeline_library
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(held) = &*held else {
            return Ok(Vec::new());
        };
        // SAFETY: the library is live, and the buffer is the size it asked for.
        let library = unsafe {
            let mut library = vec![0_u8; held.library.GetSerializedSize()];
            held.library
                .Serialize(&mut library)
                .map_err(|error| RhiError::dx12("ID3D12PipelineLibrary::Serialize", error))?;
            library
        };
        let mut data = Vec::with_capacity(LIBRARY_HEADER + library.len());
        data.extend(LIBRARY_MAGIC);
        data.extend(self.library_identity());
        data.extend(fnv(&[&library]).to_le_bytes());
        data.extend(library);
        Ok(data)
    }

    pub(crate) fn load_pipeline_cache(&self, data: &[u8]) -> Result<bool, RhiError> {
        if data.len() <= LIBRARY_HEADER
            || data[..8] != LIBRARY_MAGIC[..]
            || data[8..24] != self.library_identity()
            || data[24..LIBRARY_HEADER] != fnv(&[&data[LIBRARY_HEADER..]]).to_le_bytes()
        {
            return Ok(false);
        }
        // Opening checks the adapter, the driver version and the layout of
        // the data once more; a library that fails any of them is left out.
        let Ok(library) = open_pipeline_library(&self.native, data[LIBRARY_HEADER..].to_vec())
        else {
            return Ok(false);
        };
        *self
            .pipeline_library
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(library);
        Ok(true)
    }

    /// Every warning and error the debug layer has kept, oldest first. `None`
    /// means the debug message queue is unavailable or could not be read,
    /// never that it was empty.
    #[must_use]
    pub fn debug_messages(&self) -> Option<Vec<crate::DebugMessage>> {
        let queue = self.native.cast::<ID3D12InfoQueue>().ok()?;
        let mut messages = Vec::new();
        // SAFETY: the queue belongs to this live device. GetMessage reports the
        // required storage size; MaybeUninit provides the message's alignment,
        // and the header is read only after a successful fill.
        unsafe {
            for index in 0..queue.GetNumStoredMessagesAllowedByRetrievalFilter() {
                let mut length = 0;
                queue.GetMessage(index, None, &mut length).ok()?;
                if length < mem::size_of::<D3D12_MESSAGE>() {
                    return None;
                }
                let mut storage = vec![
                    mem::MaybeUninit::<D3D12_MESSAGE>::zeroed();
                    length.div_ceil(mem::size_of::<D3D12_MESSAGE>())
                ];
                let pointer = storage.as_mut_ptr().cast::<D3D12_MESSAGE>();
                queue.GetMessage(index, Some(pointer), &mut length).ok()?;
                let message = &*pointer;
                let severity = match message.Severity {
                    D3D12_MESSAGE_SEVERITY_ERROR | D3D12_MESSAGE_SEVERITY_CORRUPTION => {
                        crate::DebugSeverity::Error
                    }
                    D3D12_MESSAGE_SEVERITY_WARNING => crate::DebugSeverity::Warning,
                    _ => continue,
                };
                let text = if message.pDescription.is_null() {
                    String::new()
                } else {
                    let bytes = std::slice::from_raw_parts(
                        message.pDescription.cast::<u8>(),
                        message.DescriptionByteLength,
                    );
                    String::from_utf8_lossy(bytes)
                        .trim_end_matches('\0')
                        .to_owned()
                };
                messages.push(crate::DebugMessage { severity, text });
            }
        }
        Some(messages)
    }

    /// Count retained validation errors/corruption. None means the debug message
    /// queue is unavailable or could not be read, never a successful zero count.
    #[must_use]
    pub fn debug_error_count(&self) -> Option<u64> {
        let errors = self
            .debug_messages()?
            .iter()
            .filter(|message| message.severity == crate::DebugSeverity::Error)
            .count();
        Some(errors as u64)
    }

    pub(crate) fn wait_idle(&self) -> Result<(), RhiError> {
        let value = self.signal_fence()?;
        self.wait_for_fence_value(value)
    }

    pub(crate) fn allocator(&self) -> Arc<Mutex<Allocator>> {
        Arc::clone(&self.allocator)
    }

    pub(crate) fn lock_allocator(&self) -> MutexGuard<'_, Allocator> {
        lock_allocator(&self.allocator)
    }
}

/// Lock a shared allocator, recovering the guard if a prior panic poisoned it.
pub(crate) fn lock_allocator(allocator: &Mutex<Allocator>) -> MutexGuard<'_, Allocator> {
    allocator
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn adapter_name(description: &[u16; 128]) -> String {
    let length = description
        .iter()
        .position(|character| *character == 0)
        .unwrap_or(description.len());
    String::from_utf16_lossy(&description[..length])
}

#[cfg(test)]
#[path = "tests/dx12_device.rs"]
mod tests;

/// Open a pipeline library from `data` as an earlier run serialized it, or
/// an empty one from no data.
fn open_pipeline_library(
    device: &ID3D12Device5,
    data: Vec<u8>,
) -> windows::core::Result<PipelineLibrary> {
    // SAFETY: the bytes outlive the library (they are kept beside it), and
    // no bytes at all ask for an empty library.
    let library = unsafe { device.CreatePipelineLibrary(&data) }?;
    Ok(PipelineLibrary {
        library,
        _opened_from: data,
    })
}
