//! Unit tests of `ray_tracing_pipeline.rs`.

use super::*;

#[test]
fn dx12_sbt_layout_respects_record_and_table_alignment() {
    // DX12: 32-byte identifiers, 32-byte records, 64-byte table starts.
    let layout = compute_sbt_layout(32, 32, 64, 2, 1);
    assert_eq!(layout.raygen_offset, 0);
    assert_eq!(layout.raygen_size, 32);
    assert_eq!(layout.miss_offset, 64);
    assert_eq!(layout.miss_stride, 32);
    assert_eq!(layout.miss_size, 64);
    assert_eq!(layout.hit_offset, 128);
    assert_eq!(layout.hit_size, 32);
    assert_eq!(layout.total_size, 160);
}

#[test]
fn vulkan_sbt_layout_uses_device_alignments() {
    // The RTX 3060 reports handle 32, handle alignment 32, base 64.
    let layout = compute_sbt_layout(32, 32, 64, 2, 1);
    assert!(layout.miss_offset.is_multiple_of(64));
    assert!(layout.hit_offset.is_multiple_of(64));
    assert!(layout.miss_stride.is_multiple_of(32));
    // An oddball device: handle 16, alignment 64, base 128.
    let layout = compute_sbt_layout(16, 64, 128, 1, 1);
    assert_eq!(layout.raygen_size, 64);
    assert_eq!(layout.miss_offset, 128);
    assert_eq!(layout.hit_offset, 256);
    assert_eq!(layout.total_size, 320);
}

#[test]
fn empty_hit_tables_still_reserve_one_record() {
    let layout = compute_sbt_layout(32, 32, 64, 0, 0);
    assert_eq!(layout.miss_size, 0);
    assert!(layout.total_size >= layout.hit_offset + 32);
}
