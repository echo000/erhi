//! Unit tests of `memory.rs`.

use super::*;

#[test]
fn attribution_retention_and_release_are_deduplicated() {
    let tracker = Arc::new(Mutex::new(MemoryTracker::default()));
    let owner = memory_scope("mesh", 42);
    let first = MemoryTicket::new(&tracker, MemoryKind::Upload, 128);
    {
        let _nested = memory_scope("post", 7);
        let second = MemoryTicket::new(&tracker, MemoryKind::Image, 512);
        assert_eq!(tracker.lock().unwrap().snapshot().tracked_bytes, 640);
        drop(second);
    }
    let held = first.retain();
    let held_twice = first.retain();
    let snapshot = tracker.lock().unwrap().snapshot();
    assert_eq!(snapshot.rows[0].fence_retained_bytes, 128);
    assert_eq!(snapshot.rows[0].owner.id, 42);
    assert_eq!(snapshot.peak_tracked_bytes, 640);
    drop(held);
    drop(held_twice);
    assert_eq!(
        tracker.lock().unwrap().snapshot().rows[0].fence_retained_bytes,
        0
    );
    drop(first);
    drop(owner);
    assert!(tracker.lock().unwrap().snapshot().rows.is_empty());
    assert_eq!(OWNER.get().subsystem, "unclassified");
}
