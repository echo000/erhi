//! Direct3D 12 shader modules: validated DXIL bytes retained until a
//! pipeline (state object, compute, or graphics) consumes them. DXIL has no

//! device-side module object, so the bytes themselves are the module.

use super::Dx12Device;
use crate::RhiError;

/// One owned DXIL blob bound to the creating device's backend.
pub(crate) struct Dx12ShaderModule {
    pub(crate) bytes: Vec<u8>,
}

impl Dx12Device {
    pub(crate) fn create_shader_module(
        &self,
        bytes: Vec<u8>,
    ) -> Result<Dx12ShaderModule, RhiError> {
        Ok(Dx12ShaderModule { bytes })
    }
}
