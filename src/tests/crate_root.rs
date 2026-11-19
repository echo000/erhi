//! Unit tests of `lib.rs`.

use super::*;

#[test]
fn aabb_retains_valid_extents() {
    let chunk = Aabb::new([0.0; 3], [64.0; 3]).unwrap();
    assert_eq!(chunk.min, [0.0; 3]);
    assert_eq!(chunk.max, [64.0; 3]);
}

#[test]
fn malformed_aabbs_are_rejected() {
    assert!(Aabb::new([0.0; 3], [0.0, 1.0, 1.0]).is_err());
    assert!(Aabb::new([f32::NAN, 0.0, 0.0], [1.0; 3]).is_err());
}
