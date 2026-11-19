//! Native handles for third-party SDKs that need the underlying API objects
//! (device, images, command list) and record into the device's command lists.
//!
//! Handles are plain integers so this API names no backend types. They are
//! valid only while the owning objects live; `CommandEncoder::record_native`
//! retains the images it is given until the submission retires.

/// The device an SDK attaches to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeDevice {
    /// A Vulkan device: each handle as the integer Vulkan defines it to be.
    Vulkan {
        /// `VkInstance`.
        instance: u64,
        /// `VkPhysicalDevice`.
        physical_device: u64,
        /// `VkDevice`.
        device: u64,
        /// `VkQueue`: the one queue everything is submitted to.
        queue: u64,
        /// The family of that queue.
        queue_family: u32,
        /// `PFN_vkGetInstanceProcAddr`.
        get_instance_proc_addr: usize,
        /// `PFN_vkGetDeviceProcAddr`.
        get_device_proc_addr: usize,
    },
    /// A Direct3D 12 device: each interface as a pointer, with no reference
    /// taken for the caller.
    Dx12 {
        /// `ID3D12Device5*`, not add-ref'd.
        device: usize,
        /// `ID3D12CommandQueue*` of the direct queue, not add-ref'd.
        queue: usize,
    },
}

/// One image: its resource and native format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeImage {
    /// A Vulkan image.
    Vulkan {
        /// `VkImage`.
        image: u64,
        /// `VkFormat`.
        format: i32,
        /// `VkImageUsageFlags`.
        usage: u32,
    },
    /// A Direct3D 12 resource.
    Dx12 {
        /// `ID3D12Resource*`, not add-ref'd.
        resource: usize,
        /// `DXGI_FORMAT` of the resource (typeless for sampled depth).
        format: u32,
    },
}

/// The command list a native recording writes into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeCommandList {
    /// `VkCommandBuffer`.
    Vulkan(u64),
    /// `ID3D12GraphicsCommandList4*`, not add-ref'd.
    Dx12(usize),
}

/// A native recording: receives the backend command list and the bytes
/// recorded with it. The device rebinds its own pipelines and descriptor heaps
/// for every later command, so the callback may change them freely; it must
/// leave every image it touches in the state the device tracks for it.
pub type NativeRecord = fn(NativeCommandList, &[u8]) -> Result<(), String>;
