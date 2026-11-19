//! Occlusion queries: how many samples the draws between a begin and an
//! end put through the depth and stencil tests.

use std::fmt;
use std::sync::Arc;

use crate::command_encoder::CommandEncoder;
use crate::device::BackendDevice;
#[cfg(all(feature = "dx12", windows))]
use crate::dx12::Dx12TimestampQuerySet as Dx12QuerySet;
use crate::render_commands::Command;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanTimestampQuerySet as VulkanQuerySet;
use crate::{Backend, Buffer, BufferUsages, Device, RhiError};

/// A resolved occlusion count is one little-endian `u64`.
pub const OCCLUSION_QUERY_SIZE: u64 = 8;

/// One owned set of occlusion query slots.
///
/// A slot counts between [`CommandEncoder::begin_occlusion_query`] and
/// [`CommandEncoder::end_occlusion_query`] inside a pass, and is copied
/// into a buffer with [`CommandEncoder::resolve_occlusion_queries`].
#[derive(Clone)]
pub struct OcclusionQuerySet {
    device_id: u64,
    count: u32,
    pub(crate) backend: OcclusionQuerySetBackend,
    _device: BackendDevice,
}

#[derive(Clone)]
pub(crate) enum OcclusionQuerySetBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Arc<VulkanQuerySet>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Arc<Dx12QuerySet>),
}

impl fmt::Debug for OcclusionQuerySet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OcclusionQuerySet")
            .field("backend", &self.backend())
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Create `count` occlusion query slots.
    pub fn create_occlusion_query_set(&self, count: u32) -> Result<OcclusionQuerySet, RhiError> {
        if count == 0 {
            return Err(RhiError::InvalidOcclusionQuery {
                reason: "an occlusion query set needs at least one slot",
            });
        }
        let backend = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => OcclusionQuerySetBackend::Vulkan(Arc::new(
                device.create_occlusion_query_set(count)?,
            )),
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => {
                OcclusionQuerySetBackend::Dx12(Arc::new(device.create_occlusion_query_set(count)?))
            }
        };
        Ok(OcclusionQuerySet {
            device_id: self.id(),
            count,
            backend,
            _device: self.backend.clone(),
        })
    }

    /// Whether a resolved occlusion query is the number of samples that
    /// passed. Where it is not (a Vulkan device without precise queries),
    /// it is only zero or not zero.
    #[must_use]
    pub fn occlusion_queries_are_precise(&self) -> bool {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => device.occlusion_query_precise,
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(_) => true,
        }
    }
}

impl OcclusionQuerySet {
    /// The backend of the device that made this.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match self.backend {
            #[cfg(feature = "vulkan")]
            OcclusionQuerySetBackend::Vulkan(_) => Backend::Vulkan,
            #[cfg(all(feature = "dx12", windows))]
            OcclusionQuerySetBackend::Dx12(_) => Backend::Dx12,
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
    /// Reset every slot so it can count again in this recording, outside
    /// any pass.
    ///
    /// Vulkan requires a reset before each slot's first use in a
    /// submission; on DX12 this records nothing. Call it at the top of any
    /// recording that counts and both backends stay valid.
    pub fn reset_occlusion_queries(
        &mut self,
        query_set: &OcclusionQuerySet,
    ) -> Result<(), RhiError> {
        self.require_no_render_pass("occlusion queries are reset outside a render pass")?;
        if query_set.device_id() != self.device_id() {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        self.commands.push(Command::ResetOcclusionQueries {
            query_set: query_set.clone(),
        });
        Ok(())
    }

    /// Start counting into slot `index`: the samples of the draws from here
    /// to [`Self::end_occlusion_query`] that pass the depth and stencil
    /// tests. Inside a pass, one query at a time, each slot at most once
    /// between resets.
    pub fn begin_occlusion_query(
        &mut self,
        query_set: &OcclusionQuerySet,
        index: u32,
    ) -> Result<(), RhiError> {
        let invalid = |reason| Err(RhiError::InvalidOcclusionQuery { reason });
        if query_set.device_id() != self.device_id() {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        if !self.in_render_pass() {
            return invalid("an occlusion query counts inside a render pass");
        }
        if self.occlusion_query_open {
            return invalid("an occlusion query is already open");
        }
        if index >= query_set.count() {
            return invalid("the slot index is out of range for this query set");
        }
        self.occlusion_query_open = true;
        self.commands.push(Command::BeginOcclusionQuery {
            query_set: query_set.clone(),
            index,
        });
        Ok(())
    }

    /// Stop the query [`Self::begin_occlusion_query`] opened, naming the
    /// same set and slot.
    pub fn end_occlusion_query(
        &mut self,
        query_set: &OcclusionQuerySet,
        index: u32,
    ) -> Result<(), RhiError> {
        if query_set.device_id() != self.device_id() {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        let opened = matches!(
            self.commands.iter().rev().find(|command| matches!(
                command,
                Command::BeginOcclusionQuery { .. } | Command::EndOcclusionQuery { .. }
            )),
            Some(Command::BeginOcclusionQuery { query_set: open, index: slot })
                if *slot == index && open.same_set(query_set)
        );
        if !self.occlusion_query_open || !opened {
            return Err(RhiError::InvalidOcclusionQuery {
                reason: "no occlusion query of this set and slot is open",
            });
        }
        self.occlusion_query_open = false;
        self.commands.push(Command::EndOcclusionQuery {
            query_set: query_set.clone(),
            index,
        });
        Ok(())
    }

    /// Copy `count` slots starting at `first` into `destination` as
    /// little-endian `u64` counts at `destination_offset`, outside any pass.
    ///
    /// Every copied slot must have counted earlier in this recording.
    pub fn resolve_occlusion_queries(
        &mut self,
        query_set: &OcclusionQuerySet,
        first: u32,
        count: u32,
        destination: &Buffer,
        destination_offset: u64,
    ) -> Result<(), RhiError> {
        let invalid = |reason| Err(RhiError::InvalidOcclusionQuery { reason });
        self.require_no_render_pass("occlusion queries are resolved outside a render pass")?;
        if query_set.device_id() != self.device_id() || destination.device_id() != self.device_id()
        {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        if count == 0
            || first
                .checked_add(count)
                .is_none_or(|end| end > query_set.count())
        {
            return invalid("the resolve range exceeds this query set");
        }
        if !destination_offset.is_multiple_of(OCCLUSION_QUERY_SIZE) {
            return invalid("the destination offset must be eight-byte aligned");
        }
        destination.require_usage(BufferUsages::COPY_DESTINATION, "copy destination")?;
        destination.validate_range(destination_offset, u64::from(count) * OCCLUSION_QUERY_SIZE)?;
        self.use_buffer(
            destination,
            crate::command_encoder::BufferNeed::COPY_DESTINATION,
        )?;
        self.commands.push(Command::ResolveOcclusionQueries {
            query_set: query_set.clone(),
            first,
            count,
            destination: destination.clone(),
            destination_offset,
        });
        Ok(())
    }
}

impl OcclusionQuerySet {
    /// Whether `other` is this set (a clone of it), not merely one like it.
    fn same_set(&self, other: &Self) -> bool {
        match (&self.backend, &other.backend) {
            #[cfg(feature = "vulkan")]
            (OcclusionQuerySetBackend::Vulkan(a), OcclusionQuerySetBackend::Vulkan(b)) => {
                Arc::ptr_eq(a, b)
            }
            #[cfg(all(feature = "dx12", windows))]
            (OcclusionQuerySetBackend::Dx12(a), OcclusionQuerySetBackend::Dx12(b)) => {
                Arc::ptr_eq(a, b)
            }
            #[cfg(all(feature = "vulkan", feature = "dx12", windows))]
            _ => false,
        }
    }
}
