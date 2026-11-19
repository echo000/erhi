use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use super::Dx12Device;
use super::render_pipeline::comparison;
use crate::sampler::{BorderColor, SamplerDescriptor};
use crate::{AddressMode, FilterMode, RhiError};

/// A sampler as Direct3D 12 has it: a description, written into a sampler
/// heap wherever a set that holds it is bound.
pub(crate) struct Dx12Sampler {
    pub(crate) description: D3D12_SAMPLER_DESC,
    /// The description as plain words: two samplers with one key read alike.
    key: SamplerKey,
}

type SamplerKey = [u32; 13];

fn sampler_key(description: &D3D12_SAMPLER_DESC) -> SamplerKey {
    let border = description.BorderColor.map(f32::to_bits);
    [
        description.Filter.0 as u32,
        description.AddressU.0 as u32,
        description.AddressV.0 as u32,
        description.AddressW.0 as u32,
        description.MipLODBias.to_bits(),
        description.MaxAnisotropy,
        description.ComparisonFunc.0 as u32,
        border[0],
        border[1],
        border[2],
        border[3],
        description.MinLOD.to_bits(),
        description.MaxLOD.to_bits(),
    ]
}

pub(crate) fn dx12_address_mode(mode: AddressMode) -> D3D12_TEXTURE_ADDRESS_MODE {
    match mode {
        AddressMode::ClampToEdge => D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressMode::Repeat => D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressMode::MirrorRepeat => D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
        AddressMode::ClampToBorder => D3D12_TEXTURE_ADDRESS_MODE_BORDER,
    }
}

impl Dx12Sampler {
    pub(crate) fn new(descriptor: &SamplerDescriptor) -> Self {
        // A filter is three choices of point or linear (bits 4, 2 and 0 for
        // minification, magnification and mips), or anisotropic, with the
        // comparison bit beside them.
        let linear = |filter: FilterMode, bit: i32| match filter {
            FilterMode::Nearest => 0,
            FilterMode::Linear => 1 << bit,
        };
        let mut filter = if descriptor.anisotropy > 1 {
            D3D12_FILTER_ANISOTROPIC.0
        } else {
            linear(descriptor.min_filter, 4)
                | linear(descriptor.mag_filter, 2)
                | linear(descriptor.mip_filter, 0)
        };
        if descriptor.compare.is_some() {
            filter |= D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT.0;
        }
        let description = D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER(filter),
            AddressU: dx12_address_mode(descriptor.address_u),
            AddressV: dx12_address_mode(descriptor.address_v),
            AddressW: dx12_address_mode(descriptor.address_w),
            MipLODBias: descriptor.lod_bias,
            MaxAnisotropy: u32::from(descriptor.anisotropy),
            ComparisonFunc: descriptor
                .compare
                .map_or(D3D12_COMPARISON_FUNC_NEVER, comparison),
            BorderColor: match descriptor.border {
                BorderColor::TransparentBlack => [0.0; 4],
                BorderColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
                BorderColor::OpaqueWhite => [1.0; 4],
            },
            MinLOD: descriptor.lod_min,
            MaxLOD: descriptor.lod_max,
        };
        Self {
            key: sampler_key(&description),
            description,
        }
    }
}

/// The most samplers one heap holds (a Direct3D 12 limit). A list binds one
/// sampler heap at a time, so this is the most sampler slots the sets alive
/// at one time use between them, where sets that hold the same samplers in
/// the same order use the same slots.
const SAMPLER_HEAP_SIZE: u32 = 2048;

/// The device's one shader-visible sampler heap. Every set's samplers live
/// in it, as a run of slots: one for each of its `Sampler` bindings, in order.
pub(crate) struct SamplerHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    increment: u32,
    state: Mutex<SamplerRuns>,
}

/// What a run holds: each slot's sampler, or nothing where none was written.
type RunKey = Vec<Option<SamplerKey>>;

#[derive(Default)]
struct SamplerRuns {
    /// The first slot never leased.
    next: u32,
    /// Runs given back: first slot and length.
    free: Vec<(u32, u32)>,
    /// The runs in use, by what they hold. A native set is never rewritten,
    /// so every set that holds the same samplers binds the same run.
    shared: HashMap<RunKey, Weak<SamplerTable>>,
}

/// A run of the sampler heap, shared by every set alive that holds its
/// samplers.
pub(crate) struct SamplerTable {
    owner: Arc<SamplerHeap>,
    first: u32,
    width: u32,
    key: RunKey,
}

impl Dx12Device {
    /// The run holding `samplers` in this order (`None` for a slot nothing
    /// was written to): the one a set alive already uses, else a new one.
    /// Creates the heap on first use.
    pub(crate) fn share_samplers(
        &self,
        samplers: &[Option<&Dx12Sampler>],
    ) -> Result<Arc<SamplerTable>, RhiError> {
        let mut held = crate::sync::lock(&self.sampler_heap, "the sampler heap")?;
        let owner = match &*held {
            Some(owner) => owner.clone(),
            None => {
                let description = D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                    NumDescriptors: SAMPLER_HEAP_SIZE,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                };
                // SAFETY: the descriptor is fully initialized and the device
                // is live; the increment query only reads device state.
                let (heap, increment) = unsafe {
                    let heap: ID3D12DescriptorHeap = self
                        .native
                        .CreateDescriptorHeap(&raw const description)
                        .map_err(|error| {
                            RhiError::dx12("ID3D12Device::CreateDescriptorHeap(sampler)", error)
                        })?;
                    (
                        heap,
                        self.native
                            .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER),
                    )
                };
                let owner = Arc::new(SamplerHeap {
                    heap,
                    increment,
                    state: Mutex::new(SamplerRuns::default()),
                });
                *held = Some(owner.clone());
                owner
            }
        };
        drop(held);
        let key: RunKey = samplers
            .iter()
            .map(|sampler| sampler.map(|sampler| sampler.key))
            .collect();
        let width = samplers.len() as u32;
        let mut runs = crate::sync::lock(&owner.state, "the shared samplers")?;
        if let Some(table) = runs.shared.get(&key).and_then(Weak::upgrade) {
            return Ok(table);
        }
        let first = match runs.free.iter().position(|(_, length)| *length == width) {
            Some(index) => runs.free.swap_remove(index).0,
            None if runs.next + width <= SAMPLER_HEAP_SIZE => {
                let first = runs.next;
                runs.next += width;
                first
            }
            None => {
                return Err(RhiError::InvalidSampler {
                    reason: "the descriptor sets alive hold more than 2048 samplers between them, counting each distinct run of samplers once",
                });
            }
        };
        // SAFETY: the slots lie inside a run of a live heap that no other
        // set holds, so nothing reads them until this returns.
        unsafe {
            let start = owner.heap.GetCPUDescriptorHandleForHeapStart();
            for (slot, sampler) in samplers.iter().enumerate() {
                if let Some(sampler) = sampler {
                    self.native.CreateSampler(
                        &raw const sampler.description,
                        D3D12_CPU_DESCRIPTOR_HANDLE {
                            ptr: start.ptr + owner.increment as usize * (first as usize + slot),
                        },
                    );
                }
            }
        }
        let table = Arc::new(SamplerTable {
            owner: owner.clone(),
            first,
            width,
            key: key.clone(),
        });
        runs.shared.insert(key, Arc::downgrade(&table));
        Ok(table)
    }
}

impl SamplerTable {
    pub(crate) fn heap(&self) -> &ID3D12DescriptorHeap {
        &self.owner.heap
    }

    /// Where the run starts, as the root descriptor table binds it.
    pub(crate) fn gpu_handle(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: reading the heap start has no side effects.
        let start = unsafe { self.owner.heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + u64::from(self.owner.increment) * u64::from(self.first),
        }
    }
}

impl Drop for SamplerTable {
    fn drop(&mut self) {
        let mut runs = self
            .owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        runs.free.push((self.first, self.width));
        // (Another thread may already have made the same run anew: then the
        // entry is that one's, and stays.)
        if runs
            .shared
            .get(&self.key)
            .is_some_and(|table| table.strong_count() == 0)
        {
            runs.shared.remove(&self.key);
        }
    }
}
