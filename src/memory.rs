//! Allocation accounting, independent of GPU timing and process working set.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::Device;
use crate::device::BackendDevice;

/// The allocating subsystem and optional asset/entity/viewport identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MemoryOwner {
    /// The part of the program that made the allocation, by name.
    pub subsystem: &'static str,
    /// Which thing of that part it was made for (an asset, a viewport), or
    /// zero.
    pub id: u64,
}

thread_local! {
    static OWNER: Cell<MemoryOwner> = const {
        Cell::new(MemoryOwner {
            subsystem: "unclassified",
            id: 0,
        })
    };
}

/// Restores nested attribution; cannot move across threads. No allocation per scope.
#[must_use]
#[derive(Debug)]
pub struct MemoryScope(MemoryOwner, PhantomData<Rc<()>>);

/// Say who owns what this thread allocates from now until the returned scope
/// is dropped, when the owner before it is restored. Scopes nest.
pub fn memory_scope(subsystem: &'static str, id: u64) -> MemoryScope {
    MemoryScope(OWNER.replace(MemoryOwner { subsystem, id }), PhantomData)
}

/// A number for `owner`, to tell one owner from another in a [`MemoryOwner`].
pub fn memory_owner_id(owner: &impl std::hash::Hash) -> u64 {
    use std::hash::Hasher;

    let mut hash = std::collections::hash_map::DefaultHasher::new();
    owner.hash(&mut hash);
    hash.finish()
}

/// [`memory_scope`] with the id made from `owner` by [`memory_owner_id`].
pub fn memory_scope_for(subsystem: &'static str, owner: &impl std::hash::Hash) -> MemoryScope {
    memory_scope(subsystem, memory_owner_id(owner))
}

impl Drop for MemoryScope {
    fn drop(&mut self) {
        OWNER.set(self.0);
    }
}

/// The kind of an allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum MemoryKind {
    /// A buffer in device memory.
    DeviceBuffer,
    /// An upload buffer.
    Upload,
    /// A readback buffer.
    Readback,
    /// An image.
    Image,
}

/// The allocations of one kind that one owner holds.
#[derive(Clone, Debug)]
pub struct MemoryRow {
    /// Who made them.
    pub owner: MemoryOwner,
    /// What they are.
    pub kind: MemoryKind,
    /// How many there are.
    pub allocations: u64,
    /// How many bytes they come to.
    pub bytes: u64,
    /// Deduplicated allocations held by explicit frame retirements. This is a
    /// subset of bytes, not extra memory; some may also have active owners.
    pub fence_retained_bytes: u64,
    /// Proven frame-only buffer allocations. Descriptor/view-only retirements
    /// are not inferred from reference counts, so this is a lower bound.
    pub retired_buffer_bytes: u64,
}

/// A snapshot of a device's allocations, grouped by owner and kind.
#[derive(Clone, Debug, Default)]
pub struct MemorySnapshot {
    /// The device's id.
    pub device: u64,
    /// One row for each owner and kind with anything allocated.
    pub rows: Vec<MemoryRow>,
    /// The bytes of every buffer and image alive.
    pub tracked_bytes: u64,
    /// The most `tracked_bytes` has been.
    pub peak_tracked_bytes: u64,
    /// Native allocator totals include allocations not made through Buffer/Image.
    pub allocator_used_bytes: u64,
    /// Bytes the allocator has reserved from the device, whether in use or not.
    pub allocator_reserved_bytes: u64,
}

#[derive(Default)]
pub(crate) struct MemoryTracker {
    next: u64,
    live: BTreeMap<u64, Allocation>,
    bytes: u64,
    peak: u64,
}

struct Allocation {
    owner: MemoryOwner,
    kind: MemoryKind,
    bytes: u64,
    retained: Arc<AtomicUsize>,
    buffer: Weak<crate::buffer::BufferInner>,
}

/// One ticket per native allocation, stored inside the shared resource inner.
pub(crate) struct MemoryTicket {
    tracker: Arc<Mutex<MemoryTracker>>,
    id: u64,
    retained: Arc<AtomicUsize>,
}

impl MemoryTicket {
    pub(crate) fn observe_buffer(&self, buffer: &Arc<crate::buffer::BufferInner>) {
        if let Some(entry) = crate::sync::lock_anyway(&self.tracker)
            .live
            .get_mut(&self.id)
        {
            entry.buffer = Arc::downgrade(buffer);
        }
    }

    pub(crate) fn new(tracker: &Arc<Mutex<MemoryTracker>>, kind: MemoryKind, bytes: u64) -> Self {
        let mut state = crate::sync::lock_anyway(tracker);
        state.next += 1;
        let id = state.next;
        let retained = Arc::new(AtomicUsize::new(0));
        state.live.insert(
            id,
            Allocation {
                owner: OWNER.get(),
                kind,
                bytes,
                retained: retained.clone(),
                buffer: Weak::new(),
            },
        );
        state.bytes += bytes;
        state.peak = state.peak.max(state.bytes);
        Self {
            tracker: tracker.clone(),
            id,
            retained,
        }
    }

    pub(crate) fn retain(&self) -> MemoryRetention {
        self.retained.fetch_add(1, Ordering::Relaxed);
        MemoryRetention(self.retained.clone())
    }
}

impl Drop for MemoryTicket {
    fn drop(&mut self) {
        let mut state = crate::sync::lock_anyway(&self.tracker);
        if let Some(allocation) = state.live.remove(&self.id) {
            state.bytes -= allocation.bytes;
        }
    }
}

pub(crate) struct MemoryRetention(Arc<AtomicUsize>);

impl Drop for MemoryRetention {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl std::fmt::Debug for MemoryTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryTracker")
            .field("bytes", &self.bytes)
            .field("peak", &self.peak)
            .finish()
    }
}

impl MemoryTracker {
    fn snapshot(&self) -> MemorySnapshot {
        let mut rows = BTreeMap::new();
        for allocation in self.live.values() {
            let row = rows
                .entry((allocation.owner, allocation.kind))
                .or_insert(MemoryRow {
                    owner: allocation.owner,
                    kind: allocation.kind,
                    allocations: 0,
                    bytes: 0,
                    fence_retained_bytes: 0,
                    retired_buffer_bytes: 0,
                });
            row.allocations += 1;
            row.bytes += allocation.bytes;
            let held = allocation.retained.load(Ordering::Relaxed);
            if held != 0 {
                row.fence_retained_bytes += allocation.bytes;
                if allocation.buffer.strong_count() == held {
                    row.retired_buffer_bytes += allocation.bytes;
                }
            }
        }
        MemorySnapshot {
            rows: rows.into_values().collect(),
            tracked_bytes: self.bytes,
            peak_tracked_bytes: self.peak,
            ..Default::default()
        }
    }
}

impl Device {
    /// Explicit, throttled diagnostic read. Never waits for a GPU fence.
    pub fn memory_snapshot(&self) -> MemorySnapshot {
        let mut snapshot = crate::sync::lock_anyway(&self.memory).snapshot();
        snapshot.device = self.id();
        let report = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => device.lock_allocator().generate_report(),
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => device.lock_allocator().generate_report(),
        };
        snapshot.allocator_used_bytes = report.total_allocated_bytes;
        snapshot.allocator_reserved_bytes = report.total_capacity_bytes;
        snapshot
    }
}

#[cfg(test)]
#[path = "tests/memory.rs"]
mod tests;
