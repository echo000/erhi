//! Unit tests of `shader.rs`.

use super::*;

#[test]
fn spirv_validation_checks_magic_and_word_size() {
    assert!(validate_spirv(&[0x03, 0x02, 0x23, 0x07, 0, 0, 0, 0]).is_err());
    let mut valid = vec![0_u8; 24];
    valid[..4].copy_from_slice(&SPIRV_MAGIC);
    assert!(validate_spirv(&valid).is_ok());
    let mut misaligned = vec![0_u8; 23];
    misaligned[..4].copy_from_slice(&SPIRV_MAGIC);
    assert!(validate_spirv(&misaligned).is_err());
}

#[test]
fn checked_in_rt_module_exports_the_five_entries() {
    let entries = parse_spirv_entry_points(include_bytes!("../../tests/fixtures/procedural.spv"));
    for expected in [
        "ray_gen",
        "primary_miss",
        "shadow_miss",
        "intersection",
        "closest_hit",
    ] {
        assert!(
            entries.iter().any(|name| name == expected),
            "missing entry {expected}, found {entries:?}"
        );
    }
}

#[test]
fn checked_in_binaries_pass_blob_validation() {
    assert!(validate_spirv(include_bytes!("../../tests/fixtures/procedural.spv")).is_ok());
    assert!(validate_dxil(include_bytes!("../../tests/fixtures/procedural.dxil")).is_ok());
    assert!(validate_dxil(include_bytes!("../../tests/fixtures/compute.main.dxil")).is_ok());
}
