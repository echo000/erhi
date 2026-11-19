//! Unit tests of `profiling.rs`.

use super::*;

#[test]
fn timestamp_ranges_reject_reverse_and_invalid_clocks_and_accept_wrap() {
    let clock = TimestampDomain {
        valid_bits: 8,
        period_nanoseconds: 1_000_000.,
    };
    assert_eq!(clock.elapsed_milliseconds(250, 10), Some(16.));
    assert_eq!(clock.elapsed_milliseconds(10, 9), None);
    assert_eq!(clock.elapsed_milliseconds(10, 10), Some(0.));
    assert_eq!(
        TimestampDomain {
            valid_bits: 0,
            ..clock
        }
        .elapsed_milliseconds(0, 1),
        None
    );
    assert_eq!(
        TimestampDomain {
            period_nanoseconds: f64::NAN,
            ..clock
        }
        .elapsed_milliseconds(0, 1),
        None
    );
    assert_eq!(
        TimestampDomain {
            valid_bits: 64,
            ..clock
        }
        .elapsed_milliseconds(u64::MAX, 0),
        Some(1.)
    );
}
