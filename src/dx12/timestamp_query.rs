use windows::Win32::Graphics::Direct3D12::{
    D3D12_QUERY_HEAP_DESC, D3D12_QUERY_HEAP_TYPE, D3D12_QUERY_HEAP_TYPE_OCCLUSION,
    D3D12_QUERY_HEAP_TYPE_TIMESTAMP, ID3D12QueryHeap,
};

use super::Dx12Device;
use crate::RhiError;

/// One owned Direct3D 12 query heap: of timestamps, or of occlusion counts.
pub(crate) struct Dx12TimestampQuerySet {
    pub(crate) heap: ID3D12QueryHeap,
}

impl Dx12Device {
    pub(crate) fn create_timestamp_query_set(
        &self,
        count: u32,
    ) -> Result<Dx12TimestampQuerySet, RhiError> {
        self.create_query_heap(D3D12_QUERY_HEAP_TYPE_TIMESTAMP, count)
    }

    pub(crate) fn create_occlusion_query_set(
        &self,
        count: u32,
    ) -> Result<Dx12TimestampQuerySet, RhiError> {
        self.create_query_heap(D3D12_QUERY_HEAP_TYPE_OCCLUSION, count)
    }

    fn create_query_heap(
        &self,
        kind: D3D12_QUERY_HEAP_TYPE,
        count: u32,
    ) -> Result<Dx12TimestampQuerySet, RhiError> {
        let description = D3D12_QUERY_HEAP_DESC {
            Type: kind,
            Count: count,
            NodeMask: 0,
        };
        let mut created: Option<ID3D12QueryHeap> = None;
        // SAFETY: the descriptor is fully initialized and the device is live.
        unsafe {
            self.native
                .CreateQueryHeap(&raw const description, &mut created)
                .map_err(|error| RhiError::dx12("ID3D12Device::CreateQueryHeap", error))?;
        }
        let heap = created.ok_or_else(|| RhiError::Dx12 {
            operation: "ID3D12Device::CreateQueryHeap returned no heap",
            code: 0,
            message: "successful call returned a null interface".to_owned(),
        })?;
        Ok(Dx12TimestampQuerySet { heap })
    }

    pub(crate) fn timestamp_period_nanoseconds(&self) -> Result<f64, RhiError> {
        // SAFETY: the queue is live; the frequency query has no side effects.
        let frequency = unsafe { self.direct_queue.GetTimestampFrequency() }
            .map_err(|error| RhiError::dx12("ID3D12CommandQueue::GetTimestampFrequency", error))?;
        Ok(1.0e9 / frequency as f64)
    }
}
