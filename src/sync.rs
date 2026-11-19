//! Locks that do not panic.

use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::RhiError;

/// Lock `mutex`, or say that a thread panicked while it held it: what it
/// guards may have been left half changed.
pub(crate) fn lock<'a, T>(
    mutex: &'a Mutex<T>,
    what: &'static str,
) -> Result<MutexGuard<'a, T>, RhiError> {
    mutex.lock().map_err(|_| RhiError::LockPoisoned { what })
}

/// Lock `mutex` whether or not a thread panicked while it held it. For a
/// lock that only keeps threads apart, or guards what no panic can leave
/// half changed, and where no error can be returned.
pub(crate) fn lock_anyway<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
