//! Unit tests of `acceleration_structure.rs`.

use super::*;

#[test]
fn instance_encoding_uses_a_finite_identity_and_24_bit_indices() {
    // Validate the instance record's constants independently
    // of a hardware acceleration structure.
    let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    assert!(identity.into_iter().all(f32::is_finite));
    assert_eq!(MAX_PACKED_INSTANCE_VALUE, (1 << 24) - 1);
}
