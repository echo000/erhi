use std::fmt;
use std::sync::Arc;

use crate::device::BackendDevice;
#[cfg(all(feature = "dx12", windows))]
use crate::dx12::Dx12ComputePipeline;
use crate::pipeline_layout::PipelineLayoutBackend;
use crate::shader::ShaderModuleBackend;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanComputePipeline;
use crate::{Backend, Device, PipelineLayout, RhiError, ShaderModule};

/// A compute pipeline description: one entry point of one
/// module over one common pipeline layout.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ComputePipelineDescriptor<'a> {
    /// A name for the pipeline, shown by debuggers and in errors.
    pub label: &'a str,
    /// The module the compute shader is in.
    pub module: &'a ShaderModule,
    /// Entry-point name. Vulkan validates it against the SPIR-V exports;
    /// DXIL compute blobs carry exactly one entry, fixed at compilation.
    pub entry: &'a str,
    /// The descriptor sets and push constants the shader uses.
    pub layout: &'a PipelineLayout,
}

impl<'a> ComputePipelineDescriptor<'a> {
    /// A pipeline that runs `shader`.
    #[must_use]
    pub const fn new(
        label: &'a str,
        layout: &'a PipelineLayout,
        shader: crate::ShaderEntry<'a>,
    ) -> Self {
        Self {
            label,
            module: shader.module,
            entry: shader.entry,
            layout,
        }
    }
}

/// A compute pipeline.
#[derive(Clone)]
pub struct ComputePipeline {
    device_id: u64,
    // The native pipeline references the layout at bind time.
    layout: PipelineLayout,
    pub(crate) backend: ComputePipelineBackend,
    _device: BackendDevice,
}

#[derive(Clone)]
pub(crate) enum ComputePipelineBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Arc<VulkanComputePipeline>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Arc<Dx12ComputePipeline>),
}

impl fmt::Debug for ComputePipeline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ComputePipeline")
            .field("backend", &self.backend())
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Create a compute pipeline from one module entry point.
    pub fn create_compute_pipeline(
        &self,
        descriptor: &ComputePipelineDescriptor<'_>,
    ) -> Result<ComputePipeline, RhiError> {
        if descriptor.module.device_id() != self.id() || descriptor.layout.device_id() != self.id()
        {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        if descriptor.entry.is_empty() {
            return Err(RhiError::InvalidComputePipeline {
                reason: "the entry-point name is empty",
            });
        }
        if descriptor.module.has_entry_point(descriptor.entry) == Some(false) {
            return Err(RhiError::MissingShaderExport {
                entry: descriptor.entry.to_owned(),
            });
        }
        let backend = match (&self.backend, descriptor.module.native()) {
            #[cfg(feature = "vulkan")]
            (BackendDevice::Vulkan(device), ShaderModuleBackend::Vulkan(module)) => {
                match descriptor.layout.native() {
                    PipelineLayoutBackend::Vulkan(layout) => {
                        let entry = descriptor.module.resolve_entry(descriptor.entry);
                        ComputePipelineBackend::Vulkan(Arc::new(device.create_compute_pipeline(
                            descriptor.label,
                            module,
                            entry,
                            layout,
                        )?))
                    }
                    #[cfg(all(feature = "dx12", windows))]
                    PipelineLayoutBackend::Dx12(_) => {
                        return Err(RhiError::ResourceBackendMismatch);
                    }
                }
            }
            #[cfg(all(feature = "dx12", windows))]
            (BackendDevice::Dx12(device), ShaderModuleBackend::Dx12(module)) => {
                match descriptor.layout.native() {
                    PipelineLayoutBackend::Dx12(layout) => ComputePipelineBackend::Dx12(Arc::new(
                        device.create_compute_pipeline(descriptor.label, module, layout)?,
                    )),
                    #[cfg(feature = "vulkan")]
                    PipelineLayoutBackend::Vulkan(_) => {
                        return Err(RhiError::ResourceBackendMismatch);
                    }
                }
            }
            #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
            _ => return Err(RhiError::ResourceBackendMismatch),
        };
        Ok(ComputePipeline {
            device_id: self.id(),
            layout: descriptor.layout.clone(),
            backend,
            _device: self.backend.clone(),
        })
    }
}

impl ComputePipeline {
    /// The backend of the device that made this.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            ComputePipelineBackend::Vulkan(_) => Backend::Vulkan,
            #[cfg(all(feature = "dx12", windows))]
            ComputePipelineBackend::Dx12(_) => Backend::Dx12,
        }
    }

    /// The common layout this pipeline binds with.
    #[must_use]
    pub fn layout(&self) -> &PipelineLayout {
        &self.layout
    }

    pub(crate) fn device_id(&self) -> u64 {
        self.device_id
    }
}
