use std::ffi::c_void;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_SHADER_BYTECODE, ID3D12PipelineState,
};

use super::pipeline_layout::Dx12PipelineLayout;
use super::{Dx12Device, Dx12ShaderModule};
use crate::RhiError;

/// One owned compute pipeline state object.
pub(crate) struct Dx12ComputePipeline {
    pub(crate) pipeline: ID3D12PipelineState,
}

impl Dx12Device {
    pub(crate) fn create_compute_pipeline(
        &self,
        label: &str,
        module: &Dx12ShaderModule,
        layout: &Dx12PipelineLayout,
    ) -> Result<Dx12ComputePipeline, RhiError> {
        let description = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(layout.root_signature().clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: module.bytes.as_ptr().cast::<c_void>(),
                BytecodeLength: module.bytes.len(),
            },
            ..Default::default()
        };
        let created: Result<ID3D12PipelineState, _> = self.cached_compute_pipeline(
            &super::pipeline_key(&[b"compute", &module.bytes]),
            &description,
        );
        // Release the COM reference cloned into the description.
        let _ = std::mem::ManuallyDrop::into_inner(description.pRootSignature);
        let pipeline = created
            .map_err(|error| RhiError::dx12("ID3D12Device::CreateComputePipelineState", error))?;
        super::name_object(&pipeline, label);
        Ok(Dx12ComputePipeline { pipeline })
    }
}
