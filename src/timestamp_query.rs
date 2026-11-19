use std::fmt;
use std::sync::Arc;

use crate::command_encoder::CommandEncoder;
use crate::device::BackendDevice;
#[cfg(all(feature = "dx12", windows))]
use crate::dx12::Dx12TimestampQuerySet;
use crate::render_commands::Command;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanTimestampQuerySet;
use crate::{Backend, Buffer, BufferUsages, Device, RhiError};

/// Resolved timestamps are 64-bit ticks, so buffer offsets and the resolve
/// destination stride are eight bytes.
pub const TIMESTAMP_SIZE: u64 = 8;

/// One owned set of GPU timestamp query slots.
///
/// Slots are written with [`CommandEncoder::write_timestamp`] and copied into
/// a readback buffer with [`CommandEncoder::resolve_timestamps`] as raw
/// little-endian `u64` ticks; multiply by [`Device::timestamp_period_nanoseconds`]
/// to convert spans to time.
#[derive(Clone)]
pub struct TimestampQuerySet {
    device_id: u64,
    count: u32,
    pub(crate) backend: TimestampQuerySetBackend,
    _device: BackendDevice,
}

#[derive(Clone)]
pub(crate) enum TimestampQuerySetBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Arc<VulkanTimestampQuerySet>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Arc<Dx12TimestampQuerySet>),
}

impl fmt::Debug for TimestampQuerySet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TimestampQuerySet")
            .field("backend", &self.backend())
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Whether pass-level GPU timestamps are implemented on this device.
    #[must_use]
    pub fn supports_timestamps(&self) -> bool {
        self.timestamp_domain().is_ok()
    }

    /// Clock of the single graphics/direct queue used by the current device.
    pub fn timestamp_domain(&self) -> Result<crate::TimestampDomain, RhiError> {
        let valid_bits = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => device.timestamp_valid_bits()?,
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(_) => 64,
        };
        let period_nanoseconds = self.timestamp_period_nanoseconds()?;
        let domain = crate::TimestampDomain {
            valid_bits,
            period_nanoseconds,
        };
        if domain.elapsed_milliseconds(0, 1).is_none() {
            return Err(RhiError::TimestampQueriesUnavailable);
        }
        Ok(domain)
    }

    /// Create `count` timestamp query slots.
    pub fn create_timestamp_query_set(&self, count: u32) -> Result<TimestampQuerySet, RhiError> {
        if count == 0 {
            return Err(RhiError::InvalidTimestampQuery {
                reason: "a timestamp query set needs at least one slot",
            });
        }
        let backend: TimestampQuerySetBackend = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => TimestampQuerySetBackend::Vulkan(Arc::new(
                device.create_timestamp_query_set(count)?,
            )),
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => {
                TimestampQuerySetBackend::Dx12(Arc::new(device.create_timestamp_query_set(count)?))
            }
        };
        Ok(TimestampQuerySet {
            device_id: self.id(),
            count,
            backend,
            _device: self.backend.clone(),
        })
    }

    /// Nanoseconds represented by one timestamp tick on this device.
    pub fn timestamp_period_nanoseconds(&self) -> Result<f64, RhiError> {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => device.timestamp_period_nanoseconds(),
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => device.timestamp_period_nanoseconds(),
        }
    }
}

impl TimestampQuerySet {
    /// The backend of the device that made this.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match self.backend {
            #[cfg(feature = "vulkan")]
            TimestampQuerySetBackend::Vulkan(_) => Backend::Vulkan,
            #[cfg(all(feature = "dx12", windows))]
            TimestampQuerySetBackend::Dx12(_) => Backend::Dx12,
        }
    }

    /// Number of slots created with this set.
    #[must_use]
    pub const fn count(&self) -> u32 {
        self.count
    }

    pub(crate) fn device_id(&self) -> u64 {
        self.device_id
    }
}

impl CommandEncoder<'_> {
    /// Reset every slot so it can be written again in this recording.
    ///
    /// Vulkan requires a reset before each slot's first write in a
    /// submission; on DX12 this records nothing. Call it at the top of any
    /// recording that writes timestamps and both backends stay valid.
    pub fn reset_timestamps(&mut self, query_set: &TimestampQuerySet) -> Result<(), RhiError> {
        if query_set.device_id() != self.device_id() {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        self.commands.push(Command::ResetTimestamps {
            query_set: query_set.clone(),
        });
        Ok(())
    }

    /// Write the GPU clock into `index` once all previously recorded work
    /// has completed (bottom-of-pipe semantics on both backends).
    pub fn write_timestamp(
        &mut self,
        query_set: &TimestampQuerySet,
        index: u32,
    ) -> Result<(), RhiError> {
        if query_set.device_id() != self.device_id() {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        if index >= query_set.count() {
            return Err(RhiError::InvalidTimestampQuery {
                reason: "the slot index is out of range for this query set",
            });
        }
        self.commands.push(Command::WriteTimestamp {
            query_set: query_set.clone(),
            index,
        });
        Ok(())
    }

    /// Copy `count` slots starting at `first` into `destination` as raw
    /// little-endian `u64` ticks at `destination_offset`.
    ///
    /// Every copied slot must have been written earlier in this recording.
    pub fn resolve_timestamps(
        &mut self,
        query_set: &TimestampQuerySet,
        first: u32,
        count: u32,
        destination: &Buffer,
        destination_offset: u64,
    ) -> Result<(), RhiError> {
        self.require_no_render_pass("timestamps are resolved outside a render pass")?;
        if query_set.device_id() != self.device_id() || destination.device_id() != self.device_id()
        {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        if count == 0
            || first
                .checked_add(count)
                .is_none_or(|end| end > query_set.count())
        {
            return Err(RhiError::InvalidTimestampQuery {
                reason: "the resolve range exceeds this query set",
            });
        }
        if !destination_offset.is_multiple_of(TIMESTAMP_SIZE) {
            return Err(RhiError::InvalidTimestampQuery {
                reason: "the destination offset must be eight-byte aligned",
            });
        }
        destination.require_usage(BufferUsages::COPY_DESTINATION, "copy destination")?;
        destination.validate_range(destination_offset, u64::from(count) * TIMESTAMP_SIZE)?;
        self.use_buffer(
            destination,
            crate::command_encoder::BufferNeed::COPY_DESTINATION,
        )?;
        self.commands.push(Command::ResolveTimestamps {
            query_set: query_set.clone(),
            first,
            count,
            destination: destination.clone(),
            destination_offset,
        });
        Ok(())
    }
}
