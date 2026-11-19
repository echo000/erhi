//! The graphics queue's clock: what a renderer needs to turn two timestamp
//! query results into a duration.

/// Graphics-queue clock. Its origin is unrelated to CPU Instant timestamps.
#[derive(Clone, Copy, Debug)]
pub struct TimestampDomain {
    /// How many of a timestamp's low bits mean anything; the rest are zero.
    pub valid_bits: u32,
    /// How many nanoseconds one tick of a timestamp is.
    pub period_nanoseconds: f64,
}

impl TimestampDomain {
    /// Accept a forward interval shorter than half the clock range. Longer spans
    /// are ambiguous after wrap and must be reported as unavailable, not zero.
    pub fn elapsed_milliseconds(self, start: u64, end: u64) -> Option<f64> {
        if !(1..=64).contains(&self.valid_bits)
            || !self.period_nanoseconds.is_finite()
            || self.period_nanoseconds <= 0.
        {
            return None;
        }
        let mask = u64::MAX >> (64 - self.valid_bits);
        let ticks = end.wrapping_sub(start) & mask;
        if ticks > (mask >> 1) {
            return None;
        }
        let ms = ticks as f64 * self.period_nanoseconds / 1_000_000.;
        ms.is_finite().then_some(ms)
    }
}

#[cfg(test)]
#[path = "tests/profiling.rs"]
mod tests;
