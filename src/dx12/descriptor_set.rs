use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN;

use super::{Dx12Buffer, Dx12Device, Dx12Image, Dx12ImageView};
use crate::{DescriptorBinding, DescriptorType, ImageDescriptor, ImageViewDimension, RhiError};

/// How many descriptors the heap of tables begins with.
const FIRST_TABLE_HEAP: u32 = 65_536;

/// The most a heap that shaders read resources from holds on every device.
const LARGEST_TABLE_HEAP: u32 = 1_000_000;

/// The one heap every descriptor table of the device is a run of. A command
/// list reads resources from a single heap at a time, and a draw may bind
/// several tables at once, one for each set of its layout: so they all live
/// in the same heap, which is replaced by a larger one when it fills.
pub(super) struct TableHeap {
    /// What shaders read.
    visible: ID3D12DescriptorHeap,
    /// The same descriptors where they can be read back: what a larger heap
    /// is filled from.
    shadow: ID3D12DescriptorHeap,
    size: u32,
    /// Where the descriptors never yet leased begin.
    next: u32,
    /// Runs handed back, by their width.
    free: HashMap<u32, Vec<u32>>,
    /// The heaps outgrown: command lists recorded against them may not have
    /// run yet, and nothing else keeps them alive.
    outgrown: Vec<ID3D12DescriptorHeap>,
    increment: u32,
}

/// A heap of `size` descriptors that shaders read, and its twin that can be
/// read back.
fn create_heaps(
    device: &ID3D12Device5,
    size: u32,
) -> Result<(ID3D12DescriptorHeap, ID3D12DescriptorHeap), RhiError> {
    let create = |flags| {
        let description = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: size,
            Flags: flags,
            NodeMask: 0,
        };
        // SAFETY: the descriptor is fully initialized and the device is live.
        unsafe {
            device
                .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&raw const description)
                .map_err(|error| RhiError::dx12("ID3D12Device::CreateDescriptorHeap", error))
        }
    };
    Ok((
        create(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)?,
        create(D3D12_DESCRIPTOR_HEAP_FLAG_NONE)?,
    ))
}

impl TableHeap {
    fn new(device: &ID3D12Device5) -> Result<Self, RhiError> {
        let (visible, shadow) = create_heaps(device, FIRST_TABLE_HEAP)?;
        Ok(Self {
            visible,
            shadow,
            size: FIRST_TABLE_HEAP,
            next: 0,
            free: HashMap::new(),
            outgrown: Vec::new(),
            // SAFETY: the query only reads immutable device state.
            increment: unsafe {
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
            },
        })
    }

    /// A run of `width` descriptors: one handed back if there is one of
    /// that width, else the next never leased, in a larger heap if this
    /// one has no room.
    fn lease(&mut self, device: &ID3D12Device5, width: u32) -> Result<u32, RhiError> {
        if let Some(offset) = self.free.get_mut(&width).and_then(Vec::pop) {
            return Ok(offset);
        }
        let needed = self.next + width;
        if needed > self.size {
            let size = needed
                .max(self.size.saturating_mul(2))
                .min(LARGEST_TABLE_HEAP);
            if size < needed {
                return Err(RhiError::Dx12 {
                    operation: "descriptor tables",
                    code: 0,
                    message: "the descriptor sets alive hold more than a million descriptors \
                              between them"
                        .to_owned(),
                });
            }
            let (visible, shadow) = create_heaps(device, size)?;
            if self.next != 0 {
                // SAFETY: both new heaps hold at least `next` descriptors of
                // this type, and the source is the heap that is read back.
                unsafe {
                    let from = self.shadow.GetCPUDescriptorHandleForHeapStart();
                    for to in [&shadow, &visible] {
                        device.CopyDescriptorsSimple(
                            self.next,
                            to.GetCPUDescriptorHandleForHeapStart(),
                            from,
                            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        );
                    }
                }
            }
            self.outgrown
                .push(std::mem::replace(&mut self.visible, visible));
            self.shadow = shadow;
            self.size = size;
        }
        let offset = self.next;
        self.next = needed;
        Ok(offset)
    }
}

/// A leased run of the heap of tables. Its place is handed back only when
/// the owning set drops after its frame fence; a live table never moves
/// within the heap, though the heap may be replaced by a larger one.
pub(crate) struct Dx12DescriptorSet {
    device: ID3D12Device5,
    tables: Arc<Mutex<Option<TableHeap>>>,
    /// Where the run begins, in descriptors from the heap's start.
    pub(crate) offset: u32,
    width: u32,
    /// The set's `Sampler` bindings, in the device's sampler heap: none for
    /// a set without any. Shared with every set that holds the same ones.
    pub(crate) samplers: Option<Arc<super::sampler::SamplerTable>>,
}

/// Where one descriptor of a table is written. The heap of tables stays the
/// one it is while this is held; when it drops, what was written is copied
/// to where shaders read it.
struct Target<'a> {
    held: MutexGuard<'a, Option<TableHeap>>,
    device: &'a ID3D12Device5,
    index: u32,
    handle: D3D12_CPU_DESCRIPTOR_HANDLE,
}

impl Drop for Target<'_> {
    fn drop(&mut self) {
        let Some(heap) = self.held.as_ref() else {
            return;
        };
        // SAFETY: the index lies inside this set's run of both live heaps,
        // and the lock held keeps either from being replaced meanwhile.
        unsafe {
            let start = heap.visible.GetCPUDescriptorHandleForHeapStart();
            self.device.CopyDescriptorsSimple(
                1,
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: start.ptr + heap.increment as usize * self.index as usize,
                },
                self.handle,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            );
        }
    }
}

impl Dx12Device {
    /// `samplers` is what each slot of the set's `Sampler` bindings holds,
    /// in declaration order (`None` where nothing was written).
    pub(crate) fn create_descriptor_set(
        &self,
        bindings: &[DescriptorBinding],
        samplers: &[Option<&super::Dx12Sampler>],
    ) -> Result<Dx12DescriptorSet, RhiError> {
        let width = bindings
            .iter()
            .filter(|binding| binding.descriptor_type != DescriptorType::Sampler)
            .map(|binding| binding.count)
            .sum::<u32>()
            .max(1);
        let samplers = if samplers.is_empty() {
            None
        } else {
            Some(self.share_samplers(samplers)?)
        };
        let mut held = crate::sync::lock(&self.tables, "the descriptor tables")?;
        let heap = match &mut *held {
            Some(heap) => heap,
            None => held.insert(TableHeap::new(&self.native)?),
        };
        let offset = heap.lease(&self.native, width)?;
        Ok(Dx12DescriptorSet {
            device: self.native.clone(),
            tables: self.tables.clone(),
            offset,
            width,
            samplers,
        })
    }

    /// The heap tables are bound from as things stand, where shaders find
    /// its first descriptor, and how far apart its descriptors are. `None`
    /// before the first table was made.
    pub(crate) fn table_heap(
        &self,
    ) -> Option<(ID3D12DescriptorHeap, D3D12_GPU_DESCRIPTOR_HANDLE, u32)> {
        let held = self
            .tables
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let heap = held.as_ref()?;
        // SAFETY: reading the heap start has no side effects.
        let start = unsafe { heap.visible.GetGPUDescriptorHandleForHeapStart() };
        Some((heap.visible.clone(), start, heap.increment))
    }
}

impl Drop for Dx12DescriptorSet {
    fn drop(&mut self) {
        if let Some(heap) = self
            .tables
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_mut()
        {
            heap.free.entry(self.width).or_default().push(self.offset);
        }
    }
}

impl Dx12DescriptorSet {
    /// Where descriptor `slot` of this table is written.
    fn target(&self, slot: u32) -> Target<'_> {
        let held = self
            .tables
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let index = self.offset + slot;
        let handle = match held.as_ref() {
            Some(heap) => {
                // SAFETY: reading the heap start has no side effects.
                let start = unsafe { heap.shadow.GetCPUDescriptorHandleForHeapStart() };
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: start.ptr + heap.increment as usize * index as usize,
                }
            }
            None => D3D12_CPU_DESCRIPTOR_HANDLE::default(),
        };
        Target {
            held,
            device: &self.device,
            index,
            handle,
        }
    }

    pub(crate) fn write_acceleration_structure(&self, slot: u32, tlas_address: u64) {
        let target = self.target(slot);
        let description = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: Default::default(),
            ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV {
                    Location: tlas_address,
                },
            },
        };
        // SAFETY: acceleration-structure SRVs pass a null resource and carry
        // the TLAS GPU address inside the view description.
        unsafe {
            self.device
                .CreateShaderResourceView(None, Some(&raw const description), target.handle);
        }
    }

    pub(crate) fn write_image(
        &self,
        slot: u32,
        image: &Dx12Image,
        view: &Dx12ImageView,
        descriptor: ImageDescriptor,
        descriptor_type: DescriptorType,
    ) {
        let target = self.target(slot);
        let range = view.range;
        match descriptor_type {
            DescriptorType::StorageImage => {
                // A storage view is of one mip: the view's first.
                let (dimension, anonymous) = match range.dimension {
                    ImageViewDimension::D3 => (
                        D3D12_UAV_DIMENSION_TEXTURE3D,
                        D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                            Texture3D: D3D12_TEX3D_UAV {
                                MipSlice: range.base_mip,
                                FirstWSlice: 0,
                                WSize: descriptor.mip_extent(range.base_mip).2,
                            },
                        },
                    ),
                    ImageViewDimension::D2 if descriptor.layer_count() == 1 => (
                        D3D12_UAV_DIMENSION_TEXTURE2D,
                        D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_UAV {
                                MipSlice: range.base_mip,
                                PlaneSlice: view.plane,
                            },
                        },
                    ),
                    // Layers, and the faces of cubes, are written as an array.
                    _ => (
                        D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                        D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                            Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                                MipSlice: range.base_mip,
                                FirstArraySlice: range.base_layer,
                                ArraySize: range.layer_count,
                                PlaneSlice: view.plane,
                            },
                        },
                    ),
                };
                let description = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: view.format,
                    ViewDimension: dimension,
                    Anonymous: anonymous,
                };
                // SAFETY: the resource is live and the view matches its
                // format and dimension; validation ran in the common layer.
                unsafe {
                    self.device.CreateUnorderedAccessView(
                        image.resource(),
                        None,
                        Some(&raw const description),
                        target.handle,
                    );
                }
            }
            _ => {
                let (dimension, anonymous) = match range.dimension {
                    ImageViewDimension::D3 => (
                        D3D12_SRV_DIMENSION_TEXTURE3D,
                        D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture3D: D3D12_TEX3D_SRV {
                                MostDetailedMip: range.base_mip,
                                MipLevels: range.mip_count,
                                ResourceMinLODClamp: 0.0,
                            },
                        },
                    ),
                    // A multisampled image is read sample by sample
                    // (`Texture2DMS`).
                    ImageViewDimension::D2 if descriptor.sample_count > 1 => (
                        D3D12_SRV_DIMENSION_TEXTURE2DMS,
                        D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2DMS: Default::default(),
                        },
                    ),
                    ImageViewDimension::D2 if descriptor.layer_count() == 1 => (
                        D3D12_SRV_DIMENSION_TEXTURE2D,
                        D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV {
                                MostDetailedMip: range.base_mip,
                                MipLevels: range.mip_count,
                                PlaneSlice: view.plane,
                                ResourceMinLODClamp: 0.0,
                            },
                        },
                    ),
                    // One layer of an array, read as a 2D image, is an array
                    // view of one: the shader's `Texture2D` reads its first.
                    ImageViewDimension::D2 | ImageViewDimension::D2Array => (
                        D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                        D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                                MostDetailedMip: range.base_mip,
                                MipLevels: range.mip_count,
                                FirstArraySlice: range.base_layer,
                                ArraySize: range.layer_count,
                                PlaneSlice: view.plane,
                                ResourceMinLODClamp: 0.0,
                            },
                        },
                    ),
                    ImageViewDimension::Cube if range.base_layer == 0 => (
                        D3D12_SRV_DIMENSION_TEXTURECUBE,
                        D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            TextureCube: D3D12_TEXCUBE_SRV {
                                MostDetailedMip: range.base_mip,
                                MipLevels: range.mip_count,
                                ResourceMinLODClamp: 0.0,
                            },
                        },
                    ),
                    // (A cube that does not start at the image's first layer
                    // can only be named as an array of one cube.)
                    ImageViewDimension::Cube | ImageViewDimension::CubeArray => (
                        D3D12_SRV_DIMENSION_TEXTURECUBEARRAY,
                        D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            TextureCubeArray: D3D12_TEXCUBE_ARRAY_SRV {
                                MostDetailedMip: range.base_mip,
                                MipLevels: range.mip_count,
                                First2DArrayFace: range.base_layer,
                                NumCubes: range.layer_count / 6,
                                ResourceMinLODClamp: 0.0,
                            },
                        },
                    ),
                };
                let description = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: view.format,
                    ViewDimension: dimension,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: anonymous,
                };
                // SAFETY: the resource is live and the view matches its
                // format and dimension; validation ran in the common layer.
                unsafe {
                    self.device.CreateShaderResourceView(
                        image.resource(),
                        Some(&raw const description),
                        target.handle,
                    );
                }
            }
        }
    }

    pub(crate) fn write_storage_buffer(
        &self,
        slot: u32,
        buffer: &Dx12Buffer,
        offset: u64,
        size: u64,
        element_stride: u32,
        writable: bool,
    ) {
        let target = self.target(slot);
        // Structured views must match the shader's declared element stride
        // (Slang `StructuredBuffer<T>`); the common layer validated that the
        // offset and size are stride multiples.
        let first_element = offset / u64::from(element_stride);
        // (Validated: the range lies inside a buffer no larger than 4 GiB of
        // elements.)
        let element_count = (size / u64::from(element_stride)) as u32;
        if writable {
            let description = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: first_element,
                        NumElements: element_count,
                        StructureByteStride: element_stride,
                        CounterOffsetInBytes: 0,
                        Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                    },
                },
            };
            // SAFETY: the buffer resource is live and the structured view
            // range was validated by the common layer.
            unsafe {
                self.device.CreateUnorderedAccessView(
                    buffer.resource(),
                    None,
                    Some(&raw const description),
                    target.handle,
                );
            }
        } else {
            let description = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: first_element,
                        NumElements: element_count,
                        StructureByteStride: element_stride,
                        Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                    },
                },
            };
            // SAFETY: the buffer resource is live and the structured view
            // range was validated by the common layer.
            unsafe {
                self.device.CreateShaderResourceView(
                    buffer.resource(),
                    Some(&raw const description),
                    target.handle,
                );
            }
        }
    }

    pub(crate) fn write_uniform_buffer(
        &self,
        slot: u32,
        buffer: &Dx12Buffer,
        offset: u64,
        size: u64,
    ) -> Result<(), RhiError> {
        let target = self.target(slot);
        let address = buffer.device_address()?;
        let description = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: address + offset,
            // Constant-buffer views must be 256-byte multiples; committed
            // buffer resources are padded well past this rounding.
            SizeInBytes: (size.div_ceil(256) * 256) as u32,
        };
        // SAFETY: the buffer is live and the 256-byte offset alignment was
        // validated by the common layer.
        unsafe {
            self.device
                .CreateConstantBufferView(Some(&raw const description), target.handle);
        }
        Ok(())
    }
}
