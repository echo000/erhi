//! Unit tests of `buffer.rs`.

use super::*;

#[test]
fn row_uploads_validate_stride_and_complete_source_rows() {
    assert_eq!(row_write_size(&[1; 12], 4, 256).unwrap(), 768);
    assert!(row_write_size(&[1; 12], 0, 256).is_err());
    assert!(row_write_size(&[1; 12], 4, 2).is_err());
    assert!(row_write_size(&[1; 13], 4, 256).is_err());
    assert!(row_write_size(&[1; 12], 4, usize::MAX).is_err());
}
