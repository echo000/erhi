use std::fmt;
use std::sync::{Arc, OnceLock};

#[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
use crate::acceleration_structure::AccelerationStructureBackend;
use crate::buffer::BufferBackend;
use crate::device::BackendDevice;
#[cfg(all(feature = "dx12", windows))]
use crate::dx12::Dx12DescriptorSet;
#[cfg(all(feature = "dx12", windows))]
use crate::image::ImageBackend;
use crate::image::ImageViewBackend;
use crate::pipeline_layout::PipelineLayoutBackend;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanDescriptorSet;
use crate::{
    AccelerationStructure, AccelerationStructureKind, Backend, Buffer, BufferUsages,
    DescriptorBinding, DescriptorType, Device, ImageUsages, ImageView, PipelineLayout, RhiError,
};

/// Uniform-buffer offsets bound through descriptors use the Direct3D 12
/// constant-buffer alignment on both backends so scene code stays identical.
pub const UNIFORM_BUFFER_OFFSET_ALIGNMENT: u64 = 256;

/// Storage-buffer descriptor offsets and ranges are raw 32-bit-word views on
/// DX12, so both backends require four-byte alignment.
pub const STORAGE_BUFFER_ALIGNMENT: u64 = 4;

/// The part of a buffer a binding holds.
///
/// ```text
/// set.write_uniform_buffer(0, BufferBinding::whole(&constants))?;
/// set.write_storage_buffer(1, BufferBinding::range(&records, 0, 4096).with_stride(16))?;
/// ```
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct BufferBinding<'a> {
    /// The buffer.
    pub buffer: &'a Buffer,
    /// Where the part begins, in bytes.
    pub offset: u64,
    /// How long the part is, in bytes.
    pub size: u64,
    /// For a storage buffer, the size of one of its records, in bytes: what
    /// the shader declares it a buffer of. A uniform buffer has none.
    pub stride: u32,
}

impl<'a> BufferBinding<'a> {
    /// All of `buffer`.
    #[must_use]
    pub fn whole(buffer: &'a Buffer) -> Self {
        Self::range(buffer, 0, buffer.descriptor().size)
    }

    /// The `size` bytes of `buffer` from `offset`.
    #[must_use]
    pub const fn range(buffer: &'a Buffer, offset: u64, size: u64) -> Self {
        Self {
            buffer,
            offset,
            size,
            stride: 0,
        }
    }

    /// The same part, as records of `stride` bytes each: what a storage
    /// buffer is bound as.
    #[must_use]
    pub const fn with_stride(mut self, stride: u32) -> Self {
        self.stride = stride;
        self
    }
}

/// One owned descriptor snapshot for one layout set index.
///
/// Writes validate the target binding, descriptor type, resource usage, and
/// creating device before any native descriptor update, and the set retains
/// every written resource so native handles outlive the bound frame. Native tables
/// are allocated and populated when first bound during native recording.
#[derive(Clone)]
pub struct DescriptorSet {
    device_id: u64,
    set: u32,
    bindings: Arc<Vec<DescriptorBinding>>,
    backend: Arc<OnceLock<DescriptorSetBackend>>,
    layout: PipelineLayout,
    /// What was written, by binding and by element of it.
    retained: Arc<std::collections::HashMap<(u32, u32), RetainedResource>>,
    _device: BackendDevice,
}

pub(crate) enum DescriptorSetBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<VulkanDescriptorSet>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Box<Dx12DescriptorSet>),
}

/// Captures both ownership and binding ranges so a queued table can be detached
/// without changing the resources observed by an earlier frame.
#[derive(Clone)]
enum RetainedResource {
    Buffer(Buffer, u64, u64, Option<u32>),
    ImageView(ImageView, DescriptorType),
    AccelerationStructure(AccelerationStructure),
    Sampler(crate::Sampler),
}

impl fmt::Debug for DescriptorSet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DescriptorSet")
            .field("backend", &self.backend())
            .field("set", &self.set)
            .field("binding_count", &self.bindings.len())
            .field("retained_count", &self.retained.len())
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Create a descriptor snapshot for `set` of `layout` without native allocation.
    pub fn create_descriptor_set(
        &self,
        layout: &PipelineLayout,
        set: u32,
    ) -> Result<DescriptorSet, RhiError> {
        if layout.device_id() != self.id() {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        let owned = layout
            .set_layout(set)
            .ok_or(RhiError::InvalidDescriptorSetOrder {
                expected: set,
                actual: u32::MAX,
            })?;
        Ok(DescriptorSet {
            device_id: self.id(),
            set,
            bindings: Arc::new(owned.bindings.clone()),
            backend: Arc::new(OnceLock::new()),
            layout: layout.clone(),
            retained: Arc::new(std::collections::HashMap::new()),
            _device: self.backend.clone(),
        })
    }
}

impl DescriptorSet {
    /// The backend of the device that made this.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.layout.backend()
    }

    /// Which of its layout's sets this is.
    #[must_use]
    pub const fn set(&self) -> u32 {
        self.set
    }

    pub(crate) fn device_id(&self) -> u64 {
        self.device_id
    }

    /// Every image bound in this set, with the type of its binding and the
    /// shader stages that see it.
    pub(crate) fn images(
        &self,
    ) -> impl Iterator<Item = (&crate::ImageView, DescriptorType, crate::ShaderStages)> + '_ {
        self.retained.iter().filter_map(|((binding, _), resource)| {
            let RetainedResource::ImageView(view, kind) = resource else {
                return None;
            };
            let declared = self
                .bindings
                .iter()
                .find(|declared| declared.binding == *binding)?;
            Some((view, *kind, declared.visibility))
        })
    }

    /// Every buffer bound in this set, with the type of its binding and the
    /// shader stages that see it.
    pub(crate) fn buffers(
        &self,
    ) -> impl Iterator<Item = (&Buffer, DescriptorType, crate::ShaderStages)> + '_ {
        self.retained.iter().filter_map(|((binding, _), resource)| {
            let RetainedResource::Buffer(buffer, ..) = resource else {
                return None;
            };
            let declared = self
                .bindings
                .iter()
                .find(|declared| declared.binding == *binding)?;
            Some((buffer, declared.descriptor_type, declared.visibility))
        })
    }

    /// What tells one writing of this set from another: a new one each time
    /// a binding is written.
    pub(crate) fn version(&self) -> usize {
        Arc::as_ptr(&self.backend) as usize
    }

    /// The first binding the layout declares of which some element has had
    /// nothing written to it.
    pub(crate) fn unwritten(&self) -> Option<u32> {
        self.bindings
            .iter()
            .flat_map(|declared| (0..declared.count).map(|element| (declared.binding, element)))
            .find(|at| !self.retained.contains_key(at))
            .map(|(binding, _)| binding)
    }

    /// Whether `layout` declares this set as the layout it was made for
    /// does: the same bindings, of the same types and counts.
    pub(crate) fn declared_as_in(&self, layout: &PipelineLayout) -> bool {
        layout
            .set_layout(self.set)
            .is_some_and(|declared| declared.bindings == *self.bindings)
    }

    /// Materialize during native recording when the captured table is first bound.
    pub(crate) fn native(&self) -> Result<&DescriptorSetBackend, RhiError> {
        if let Some(backend) = self.backend.get() {
            return Ok(backend);
        }
        let backend = match (&self._device, self.layout.native()) {
            #[cfg(feature = "vulkan")]
            (BackendDevice::Vulkan(device), PipelineLayoutBackend::Vulkan(layout)) => {
                DescriptorSetBackend::Vulkan(Box::new(device.create_descriptor_set(
                    layout,
                    self.set,
                    &self.bindings,
                )?))
            }
            #[cfg(all(feature = "dx12", windows))]
            (BackendDevice::Dx12(device), PipelineLayoutBackend::Dx12(_)) => {
                // What each slot of the set's sampler table holds, in the
                // order the root signature appends the `Sampler` bindings.
                // They are handed over whole: sets that hold the same
                // samplers share one run of the device's sampler heap.
                let mut samplers = Vec::new();
                for declared in self
                    .bindings
                    .iter()
                    .filter(|declared| declared.descriptor_type == DescriptorType::Sampler)
                {
                    for element in 0..declared.count {
                        samplers.push(match self.retained.get(&(declared.binding, element)) {
                            Some(RetainedResource::Sampler(sampler)) => {
                                match &sampler.inner.backend {
                                    crate::sampler::SamplerBackend::Dx12(native) => Some(native),
                                    #[cfg(feature = "vulkan")]
                                    crate::sampler::SamplerBackend::Vulkan(_) => {
                                        return Err(RhiError::ResourceBackendMismatch);
                                    }
                                }
                            }
                            _ => None,
                        });
                    }
                }
                DescriptorSetBackend::Dx12(Box::new(
                    device.create_descriptor_set(&self.bindings, &samplers)?,
                ))
            }
            #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
            _ => return Err(RhiError::ResourceBackendMismatch),
        };
        for (&(binding, element), resource) in &*self.retained {
            self.write_native(&backend, binding, element, resource)?;
        }
        // Independent readers may race; both construct the same immutable table.
        Ok(self.backend.get_or_init(|| backend))
    }

    /// Write a top-level acceleration structure into `binding`.
    pub fn write_acceleration_structure(
        &mut self,
        binding: u32,
        acceleration_structure: &AccelerationStructure,
    ) -> Result<(), RhiError> {
        self.write_acceleration_structure_at(binding, 0, acceleration_structure)
    }

    /// As [`Self::write_acceleration_structure`], into one element of a binding that declares
    /// several descriptors.
    pub fn write_acceleration_structure_at(
        &mut self,
        binding: u32,
        element: u32,
        acceleration_structure: &AccelerationStructure,
    ) -> Result<(), RhiError> {
        self.require_binding(binding, element, &[DescriptorType::AccelerationStructure])?;
        if acceleration_structure.device_id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        if acceleration_structure.kind() != AccelerationStructureKind::TopLevel {
            return Err(RhiError::InvalidDescriptorBinding {
                set: self.set,
                binding,
                reason: "acceleration-structure descriptors require a top-level structure",
            });
        }
        self.retain(
            binding,
            element,
            RetainedResource::AccelerationStructure(acceleration_structure.clone()),
        );
        Ok(())
    }

    /// Write a storage-image view into `binding`.
    pub fn write_storage_image(&mut self, binding: u32, view: &ImageView) -> Result<(), RhiError> {
        self.write_storage_image_at(binding, 0, view)
    }

    /// As [`Self::write_storage_image`], into one element of a binding that declares
    /// several descriptors.
    pub fn write_storage_image_at(
        &mut self,
        binding: u32,
        element: u32,
        view: &ImageView,
    ) -> Result<(), RhiError> {
        self.write_image(binding, element, view, DescriptorType::StorageImage)
    }

    /// Write a sampled-image view into `binding`.
    pub fn write_sampled_image(&mut self, binding: u32, view: &ImageView) -> Result<(), RhiError> {
        self.write_sampled_image_at(binding, 0, view)
    }

    /// As [`Self::write_sampled_image`], into one element of a binding that declares
    /// several descriptors.
    pub fn write_sampled_image_at(
        &mut self,
        binding: u32,
        element: u32,
        view: &ImageView,
    ) -> Result<(), RhiError> {
        self.write_image(binding, element, view, DescriptorType::SampledImage)
    }

    /// Write the image view of a combined-image-sampler `binding`; the
    /// sampler itself is immutable and lives in the pipeline layout.
    pub fn write_combined_image_sampler(
        &mut self,
        binding: u32,
        view: &ImageView,
    ) -> Result<(), RhiError> {
        let declared = self.find_binding(binding)?.descriptor_type;
        let DescriptorType::CombinedImageSampler(_) = declared else {
            return Err(RhiError::InvalidDescriptorBinding {
                set: self.set,
                binding,
                reason: "binding type does not accept this write",
            });
        };
        self.write_image(binding, 0, view, declared)
    }

    fn write_image(
        &mut self,
        binding: u32,
        element: u32,
        view: &ImageView,
        descriptor_type: DescriptorType,
    ) -> Result<(), RhiError> {
        self.require_binding(binding, element, &[descriptor_type])?;
        let image = view.image();
        if image.device_id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        let (usage, usage_name) = match descriptor_type {
            DescriptorType::StorageImage => (ImageUsages::STORAGE, "storage"),
            // Sampled and combined-image-sampler bindings both read through
            // the sampled usage.
            _ => (ImageUsages::SAMPLED, "sampled"),
        };
        image.require_usage(usage, usage_name)?;
        self.retain(
            binding,
            element,
            RetainedResource::ImageView(view.clone(), descriptor_type),
        );
        Ok(())
    }

    /// Write a sampler into a `Sampler` binding.
    pub fn write_sampler(
        &mut self,
        binding: u32,
        sampler: &crate::Sampler,
    ) -> Result<(), RhiError> {
        self.write_sampler_at(binding, 0, sampler)
    }

    /// As [`Self::write_sampler`], into one element of a binding that declares
    /// several descriptors.
    pub fn write_sampler_at(
        &mut self,
        binding: u32,
        element: u32,
        sampler: &crate::Sampler,
    ) -> Result<(), RhiError> {
        self.require_binding(binding, element, &[DescriptorType::Sampler])?;
        if sampler.device_id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        self.retain(binding, element, RetainedResource::Sampler(sampler.clone()));
        Ok(())
    }

    /// Write a storage-buffer range into `binding`.
    ///
    /// `element_stride` must match the shader's declared element size
    /// (`StructuredBuffer<T>` stride): DX12 creates a structured view from
    /// it, and the offset and size must be multiples of it. Vulkan's untyped
    /// storage buffers ignore the stride beyond that validation.
    pub fn write_storage_buffer(
        &mut self,
        binding: u32,
        bound: BufferBinding<'_>,
    ) -> Result<(), RhiError> {
        self.write_storage_buffer_at(binding, 0, bound)
    }

    /// As [`Self::write_storage_buffer`], into one element of a binding that declares
    /// several descriptors.
    pub fn write_storage_buffer_at(
        &mut self,
        binding: u32,
        element: u32,
        bound: BufferBinding<'_>,
    ) -> Result<(), RhiError> {
        let BufferBinding {
            buffer,
            offset,
            size,
            stride: element_stride,
        } = bound;
        self.require_binding(
            binding,
            element,
            &[
                DescriptorType::ReadOnlyStorageBuffer,
                DescriptorType::StorageBuffer,
            ],
        )?;
        if buffer.device_id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        buffer.require_usage(BufferUsages::STORAGE, "storage")?;
        buffer.validate_range(offset, size)?;
        if size == 0
            || element_stride == 0
            || !u64::from(element_stride).is_multiple_of(STORAGE_BUFFER_ALIGNMENT)
            || !offset.is_multiple_of(u64::from(element_stride))
            || !size.is_multiple_of(u64::from(element_stride))
        {
            return Err(RhiError::InvalidDescriptorBinding {
                set: self.set,
                binding,
                reason: "storage-buffer ranges must be non-empty multiples of a four-byte-aligned element stride",
            });
        }
        self.retain(
            binding,
            element,
            RetainedResource::Buffer(buffer.clone(), offset, size, Some(element_stride)),
        );
        Ok(())
    }

    /// Write a uniform-buffer range into `binding`.
    ///
    /// The offset must be 256-byte aligned to satisfy DX12 constant-buffer
    /// rules; the same rule applies on Vulkan for identical scene code.
    pub fn write_uniform_buffer(
        &mut self,
        binding: u32,
        bound: BufferBinding<'_>,
    ) -> Result<(), RhiError> {
        self.write_uniform_buffer_at(binding, 0, bound)
    }

    /// As [`Self::write_uniform_buffer`], into one element of a binding that declares
    /// several descriptors.
    pub fn write_uniform_buffer_at(
        &mut self,
        binding: u32,
        element: u32,
        bound: BufferBinding<'_>,
    ) -> Result<(), RhiError> {
        let BufferBinding {
            buffer,
            offset,
            size,
            ..
        } = bound;
        self.require_binding(binding, element, &[DescriptorType::UniformBuffer])?;
        if buffer.device_id() != self.device_id {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        buffer.require_usage(BufferUsages::UNIFORM, "uniform")?;
        buffer.validate_range(offset, size)?;
        if size == 0 || !offset.is_multiple_of(UNIFORM_BUFFER_OFFSET_ALIGNMENT) {
            return Err(RhiError::InvalidDescriptorBinding {
                set: self.set,
                binding,
                reason: "uniform-buffer ranges must be non-empty with a 256-byte-aligned offset",
            });
        }
        self.retain(
            binding,
            element,
            RetainedResource::Buffer(buffer.clone(), offset, size, None),
        );
        Ok(())
    }

    fn retain(&mut self, binding: u32, element: u32, resource: RetainedResource) {
        let unchanged = self
            .retained
            .get(&(binding, element))
            .is_some_and(|previous| match (previous, &resource) {
                (
                    RetainedResource::Buffer(a, ao, az, ast),
                    RetainedResource::Buffer(b, bo, bz, bst),
                ) => Arc::ptr_eq(&a.inner, &b.inner) && ao == bo && az == bz && ast == bst,
                (RetainedResource::ImageView(a, at), RetainedResource::ImageView(b, bt)) => {
                    Arc::ptr_eq(&a.inner, &b.inner) && at == bt
                }
                (
                    RetainedResource::AccelerationStructure(a),
                    RetainedResource::AccelerationStructure(b),
                ) => a.device_address() == b.device_address(),
                (RetainedResource::Sampler(a), RetainedResource::Sampler(b)) => {
                    Arc::ptr_eq(&a.inner, &b.inner)
                }
                _ => false,
            });
        if unchanged {
            return;
        }
        // Earlier packets keep their immutable native table and resources.
        self.backend = Arc::new(OnceLock::new());
        Arc::make_mut(&mut self.retained).insert((binding, element), resource);
    }

    fn write_native(
        &self,
        backend: &DescriptorSetBackend,
        binding: u32,
        element: u32,
        resource: &RetainedResource,
    ) -> Result<(), RhiError> {
        match resource {
            RetainedResource::Sampler(sampler) => match (backend, &sampler.inner.backend) {
                #[cfg(feature = "vulkan")]
                (
                    DescriptorSetBackend::Vulkan(set),
                    crate::sampler::SamplerBackend::Vulkan(native),
                ) => {
                    set.write_sampler(binding, element, native);
                }
                #[cfg(all(feature = "dx12", windows))]
                // (Written with the table, when the set was made.)
                (DescriptorSetBackend::Dx12(_), crate::sampler::SamplerBackend::Dx12(_)) => {}
                #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                _ => return Err(RhiError::ResourceBackendMismatch),
            },
            RetainedResource::AccelerationStructure(acceleration_structure) => {
                match (backend, acceleration_structure.native()) {
                    #[cfg(feature = "vulkan")]
                    (
                        DescriptorSetBackend::Vulkan(set),
                        AccelerationStructureBackend::Vulkan(native),
                    ) => {
                        set.write_acceleration_structure(binding, element, native);
                    }
                    #[cfg(all(feature = "dx12", windows))]
                    (
                        DescriptorSetBackend::Dx12(set),
                        AccelerationStructureBackend::Dx12(native),
                    ) => {
                        let _ = native;
                        set.write_acceleration_structure(
                            self.slot_of(binding)? + element,
                            acceleration_structure.device_address(),
                        );
                    }
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => return Err(RhiError::ResourceBackendMismatch),
                }
            }
            RetainedResource::ImageView(view, descriptor_type) => {
                let descriptor_type = *descriptor_type;
                match (backend, &view.inner.backend) {
                    #[cfg(feature = "vulkan")]
                    (DescriptorSetBackend::Vulkan(set), ImageViewBackend::Vulkan(native)) => {
                        set.write_image(binding, element, native, descriptor_type);
                    }
                    #[cfg(all(feature = "dx12", windows))]
                    (DescriptorSetBackend::Dx12(set), ImageViewBackend::Dx12(native)) => {
                        let image = view.image();
                        match &image.inner.backend {
                            ImageBackend::Dx12(native_image) => set.write_image(
                                self.slot_of(binding)? + element,
                                native_image,
                                native,
                                image.descriptor(),
                                descriptor_type,
                            ),
                            #[cfg(feature = "vulkan")]
                            ImageBackend::Vulkan(_) => {
                                return Err(RhiError::ResourceBackendMismatch);
                            }
                        }
                    }
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => return Err(RhiError::ResourceBackendMismatch),
                }
            }
            RetainedResource::Buffer(buffer, offset, size, Some(element_stride)) => {
                let (offset, size, element_stride) = (*offset, *size, *element_stride);
                match (backend, &buffer.inner.backend) {
                    #[cfg(feature = "vulkan")]
                    (DescriptorSetBackend::Vulkan(set), BufferBackend::Vulkan(native)) => {
                        let _ = element_stride;
                        set.write_storage_buffer(binding, element, native, offset, size);
                    }
                    #[cfg(all(feature = "dx12", windows))]
                    (DescriptorSetBackend::Dx12(set), BufferBackend::Dx12(native)) => {
                        let writable = self.binding_type(binding)? == DescriptorType::StorageBuffer;
                        set.write_storage_buffer(
                            self.slot_of(binding)? + element,
                            native,
                            offset,
                            size,
                            element_stride,
                            writable,
                        );
                    }
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => return Err(RhiError::ResourceBackendMismatch),
                }
            }
            RetainedResource::Buffer(buffer, offset, size, None) => {
                let (offset, size) = (*offset, *size);
                match (backend, &buffer.inner.backend) {
                    #[cfg(feature = "vulkan")]
                    (DescriptorSetBackend::Vulkan(set), BufferBackend::Vulkan(native)) => {
                        set.write_uniform_buffer(binding, element, native, offset, size);
                    }
                    #[cfg(all(feature = "dx12", windows))]
                    (DescriptorSetBackend::Dx12(set), BufferBackend::Dx12(native)) => {
                        set.write_uniform_buffer(
                            self.slot_of(binding)? + element,
                            native,
                            offset,
                            size,
                        )?;
                    }
                    #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
                    _ => return Err(RhiError::ResourceBackendMismatch),
                }
            }
        }
        Ok(())
    }

    fn find_binding(&self, binding: u32) -> Result<&DescriptorBinding, RhiError> {
        self.bindings
            .iter()
            .find(|declared| declared.binding == binding)
            .ok_or(RhiError::InvalidDescriptorBinding {
                set: self.set,
                binding,
                reason: "binding does not exist in this set",
            })
    }

    #[cfg(all(feature = "dx12", windows))]
    fn binding_type(&self, binding: u32) -> Result<DescriptorType, RhiError> {
        Ok(self.find_binding(binding)?.descriptor_type)
    }

    fn require_binding(
        &self,
        binding: u32,
        element: u32,
        accepted: &[DescriptorType],
    ) -> Result<(), RhiError> {
        let declared = self.find_binding(binding)?;
        if !accepted.contains(&declared.descriptor_type) {
            return Err(RhiError::InvalidDescriptorBinding {
                set: self.set,
                binding,
                reason: "binding type does not accept this write",
            });
        }
        if element >= declared.count {
            return Err(RhiError::InvalidDescriptorBinding {
                set: self.set,
                binding,
                reason: "the binding declares fewer descriptors than this element",
            });
        }
        Ok(())
    }

    /// Deterministic heap slot of the first element of `binding`: how many
    /// descriptors the set's non-sampler bindings before it declare. This
    /// matches the sequential `OffsetInDescriptorsFromTableStart` append
    /// order of the DX12 root signature, so scene code never sees
    /// backend-specific offsets.
    #[cfg(all(feature = "dx12", windows))]
    fn slot_of(&self, binding: u32) -> Result<u32, RhiError> {
        let mut slot = 0;
        for declared in self
            .bindings
            .iter()
            .filter(|declared| declared.descriptor_type != DescriptorType::Sampler)
        {
            if declared.binding == binding {
                return Ok(slot);
            }
            slot += declared.count;
        }
        Err(RhiError::InvalidDescriptorBinding {
            set: self.set,
            binding,
            reason: "binding does not exist in this set",
        })
    }
}

#[cfg(test)]
#[path = "tests/descriptor_set.rs"]
mod tests;
