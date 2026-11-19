use std::ffi::{CString, c_void};
use std::mem::ManuallyDrop;

use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_TOPOLOGY, D3D_PRIMITIVE_TOPOLOGY_LINELIST, D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
    D3D_PRIMITIVE_TOPOLOGY_POINTLIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::image::dxgi_format;
use super::pipeline_layout::Dx12PipelineLayout;
use super::{Dx12Device, Dx12ShaderModule};
use crate::RhiError;
use crate::render_pipeline::{
    BlendFactor, BlendOperation, CompareFunction, CullMode, FrontFace, PrimitiveTopology,
    RenderPipelineDescriptor, StencilFaceState, StencilOperation, VertexFormat, VertexSemantic,
    VertexStep,
};

/// One owned graphics pipeline state object, with what a draw binds beside
/// it: the primitive topology and the stencil reference.
pub(crate) struct Dx12RenderPipeline {
    pub(crate) pipeline: ID3D12PipelineState,
    pub(crate) topology: D3D_PRIMITIVE_TOPOLOGY,
    pub(crate) stencil_reference: Option<u32>,
}

fn vertex_dxgi_format(format: VertexFormat) -> DXGI_FORMAT {
    match format {
        VertexFormat::Float32 => DXGI_FORMAT_R32_FLOAT,
        VertexFormat::Float32x2 => DXGI_FORMAT_R32G32_FLOAT,
        VertexFormat::Float32x3 => DXGI_FORMAT_R32G32B32_FLOAT,
        VertexFormat::Float32x4 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        VertexFormat::Float16x2 => DXGI_FORMAT_R16G16_FLOAT,
        VertexFormat::Float16x4 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        VertexFormat::Uint32 => DXGI_FORMAT_R32_UINT,
        VertexFormat::Uint32x2 => DXGI_FORMAT_R32G32_UINT,
        VertexFormat::Uint32x3 => DXGI_FORMAT_R32G32B32_UINT,
        VertexFormat::Uint32x4 => DXGI_FORMAT_R32G32B32A32_UINT,
        VertexFormat::Sint32 => DXGI_FORMAT_R32_SINT,
        VertexFormat::Sint32x2 => DXGI_FORMAT_R32G32_SINT,
        VertexFormat::Sint32x3 => DXGI_FORMAT_R32G32B32_SINT,
        VertexFormat::Sint32x4 => DXGI_FORMAT_R32G32B32A32_SINT,
        VertexFormat::Uint16x2 => DXGI_FORMAT_R16G16_UINT,
        VertexFormat::Uint16x4 => DXGI_FORMAT_R16G16B16A16_UINT,
        VertexFormat::Sint16x2 => DXGI_FORMAT_R16G16_SINT,
        VertexFormat::Sint16x4 => DXGI_FORMAT_R16G16B16A16_SINT,
        VertexFormat::Unorm16x2 => DXGI_FORMAT_R16G16_UNORM,
        VertexFormat::Unorm16x4 => DXGI_FORMAT_R16G16B16A16_UNORM,
        VertexFormat::Snorm16x2 => DXGI_FORMAT_R16G16_SNORM,
        VertexFormat::Snorm16x4 => DXGI_FORMAT_R16G16B16A16_SNORM,
        VertexFormat::Uint8x2 => DXGI_FORMAT_R8G8_UINT,
        VertexFormat::Uint8x4 => DXGI_FORMAT_R8G8B8A8_UINT,
        VertexFormat::Sint8x2 => DXGI_FORMAT_R8G8_SINT,
        VertexFormat::Sint8x4 => DXGI_FORMAT_R8G8B8A8_SINT,
        VertexFormat::Unorm8x2 => DXGI_FORMAT_R8G8_UNORM,
        VertexFormat::Unorm8x4 => DXGI_FORMAT_R8G8B8A8_UNORM,
        VertexFormat::Snorm8x2 => DXGI_FORMAT_R8G8_SNORM,
        VertexFormat::Snorm8x4 => DXGI_FORMAT_R8G8B8A8_SNORM,
    }
}

/// A blend factor for the colour channels, or for alpha (where the colour
/// factors are named by their alpha counterparts).
fn blend_factor(factor: BlendFactor, alpha: bool) -> D3D12_BLEND {
    match (factor, alpha) {
        (BlendFactor::Zero, _) => D3D12_BLEND_ZERO,
        (BlendFactor::One, _) => D3D12_BLEND_ONE,
        (BlendFactor::Src, false) => D3D12_BLEND_SRC_COLOR,
        (BlendFactor::OneMinusSrc, false) => D3D12_BLEND_INV_SRC_COLOR,
        (BlendFactor::Src | BlendFactor::SrcAlpha, _) => D3D12_BLEND_SRC_ALPHA,
        (BlendFactor::OneMinusSrc | BlendFactor::OneMinusSrcAlpha, _) => D3D12_BLEND_INV_SRC_ALPHA,
        (BlendFactor::Dst, false) => D3D12_BLEND_DEST_COLOR,
        (BlendFactor::OneMinusDst, false) => D3D12_BLEND_INV_DEST_COLOR,
        (BlendFactor::Dst | BlendFactor::DstAlpha, _) => D3D12_BLEND_DEST_ALPHA,
        (BlendFactor::OneMinusDst | BlendFactor::OneMinusDstAlpha, _) => D3D12_BLEND_INV_DEST_ALPHA,
        // (For alpha the factor is one, as Vulkan defines it.)
        (BlendFactor::SrcAlphaSaturated, false) => D3D12_BLEND_SRC_ALPHA_SAT,
        (BlendFactor::SrcAlphaSaturated, true) => D3D12_BLEND_ONE,
    }
}

fn blend_operation(operation: BlendOperation) -> D3D12_BLEND_OP {
    match operation {
        BlendOperation::Add => D3D12_BLEND_OP_ADD,
        BlendOperation::Subtract => D3D12_BLEND_OP_SUBTRACT,
        BlendOperation::ReverseSubtract => D3D12_BLEND_OP_REV_SUBTRACT,
        BlendOperation::Min => D3D12_BLEND_OP_MIN,
        BlendOperation::Max => D3D12_BLEND_OP_MAX,
    }
}

pub(crate) fn comparison(function: CompareFunction) -> D3D12_COMPARISON_FUNC {
    match function {
        CompareFunction::Never => D3D12_COMPARISON_FUNC_NEVER,
        CompareFunction::Less => D3D12_COMPARISON_FUNC_LESS,
        CompareFunction::Equal => D3D12_COMPARISON_FUNC_EQUAL,
        CompareFunction::LessEqual => D3D12_COMPARISON_FUNC_LESS_EQUAL,
        CompareFunction::Greater => D3D12_COMPARISON_FUNC_GREATER,
        CompareFunction::NotEqual => D3D12_COMPARISON_FUNC_NOT_EQUAL,
        CompareFunction::GreaterEqual => D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        CompareFunction::Always => D3D12_COMPARISON_FUNC_ALWAYS,
    }
}

fn stencil_operation(operation: StencilOperation) -> D3D12_STENCIL_OP {
    match operation {
        StencilOperation::Keep => D3D12_STENCIL_OP_KEEP,
        StencilOperation::Zero => D3D12_STENCIL_OP_ZERO,
        StencilOperation::Replace => D3D12_STENCIL_OP_REPLACE,
        StencilOperation::IncrementClamp => D3D12_STENCIL_OP_INCR_SAT,
        StencilOperation::DecrementClamp => D3D12_STENCIL_OP_DECR_SAT,
        StencilOperation::Invert => D3D12_STENCIL_OP_INVERT,
        StencilOperation::IncrementWrap => D3D12_STENCIL_OP_INCR,
        StencilOperation::DecrementWrap => D3D12_STENCIL_OP_DECR,
    }
}

fn stencil_face(face: StencilFaceState) -> D3D12_DEPTH_STENCILOP_DESC {
    D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: stencil_operation(face.fail),
        StencilDepthFailOp: stencil_operation(face.depth_fail),
        StencilPassOp: stencil_operation(face.pass),
        StencilFunc: comparison(face.compare),
    }
}

impl Dx12Device {
    pub(crate) fn create_render_pipeline(
        &self,
        descriptor: &RenderPipelineDescriptor<'_>,
        vertex_module: &Dx12ShaderModule,
        fragment_module: &Dx12ShaderModule,
        layout: &Dx12PipelineLayout,
    ) -> Result<Dx12RenderPipeline, RhiError> {
        // Each input by the semantic its attribute has. A Slang shader's
        // `ATTRIB<n>` reaches the input signature as semantic `ATTRIB` with
        // index `n * 10`: its DXIL lowering appends each user semantic's
        // element index. The pipelines hardware test checks this.
        let attributes: Vec<_> = descriptor
            .vertex_buffers
            .iter()
            .enumerate()
            .flat_map(|(slot, layout)| {
                layout
                    .attributes
                    .iter()
                    .map(move |attribute| (slot as u32, layout.step, attribute))
            })
            .collect();
        // The names live here until the pipeline is made.
        let semantic_names: Vec<CString> = attributes
            .iter()
            .map(|(_, _, attribute)| match attribute.semantic {
                VertexSemantic::Location => Ok(c"ATTRIB".to_owned()),
                VertexSemantic::Named { name, .. } => {
                    CString::new(name).map_err(|_| RhiError::InvalidRenderPipeline {
                        reason: "a vertex attribute's semantic name must be non-empty text",
                    })
                }
            })
            .collect::<Result<_, _>>()?;
        let input_elements: Vec<D3D12_INPUT_ELEMENT_DESC> = attributes
            .iter()
            .zip(&semantic_names)
            .map(|(&(slot, step, attribute), name)| {
                let instanced = step == VertexStep::Instance;
                D3D12_INPUT_ELEMENT_DESC {
                    SemanticName: PCSTR(name.as_ptr().cast()),
                    SemanticIndex: match attribute.semantic {
                        VertexSemantic::Location => attribute.location * 10,
                        VertexSemantic::Named { index, .. } => index,
                    },
                    Format: vertex_dxgi_format(attribute.format),
                    InputSlot: slot,
                    AlignedByteOffset: attribute.offset,
                    InputSlotClass: if instanced {
                        D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                    } else {
                        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA
                    },
                    InstanceDataStepRate: u32::from(instanced),
                }
            })
            .collect();

        let mut blend = D3D12_BLEND_DESC {
            // Each target blends as its own state says.
            IndependentBlendEnable: true.into(),
            AlphaToCoverageEnable: descriptor.multisample.alpha_to_coverage.into(),
            ..Default::default()
        };
        for (index, target) in descriptor.color_targets.iter().enumerate() {
            let mut state = D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: false.into(),
                LogicOpEnable: false.into(),
                SrcBlend: D3D12_BLEND_ONE,
                DestBlend: D3D12_BLEND_ZERO,
                BlendOp: D3D12_BLEND_OP_ADD,
                SrcBlendAlpha: D3D12_BLEND_ONE,
                DestBlendAlpha: D3D12_BLEND_ZERO,
                BlendOpAlpha: D3D12_BLEND_OP_ADD,
                // (The write mask's bits are Direct3D's own: R 1, G 2, B 4,
                // A 8.)
                RenderTargetWriteMask: target.write_mask.0,
                ..Default::default()
            };
            if let Some(blending) = target.blend {
                state.BlendEnable = true.into();
                state.SrcBlend = blend_factor(blending.color.src_factor, false);
                state.DestBlend = blend_factor(blending.color.dst_factor, false);
                state.BlendOp = blend_operation(blending.color.operation);
                state.SrcBlendAlpha = blend_factor(blending.alpha.src_factor, true);
                state.DestBlendAlpha = blend_factor(blending.alpha.dst_factor, true);
                state.BlendOpAlpha = blend_operation(blending.alpha.operation);
            }
            blend.RenderTarget[index] = state;
        }

        let (topology, topology_type) = match descriptor.primitive.topology {
            PrimitiveTopology::TriangleList => (
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            ),
            PrimitiveTopology::TriangleStrip => (
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            ),
            PrimitiveTopology::LineList => (
                D3D_PRIMITIVE_TOPOLOGY_LINELIST,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
            ),
            PrimitiveTopology::LineStrip => (
                D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
            ),
            PrimitiveTopology::PointList => (
                D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
            ),
        };
        let mut description = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(layout.root_signature().clone())),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vertex_module.bytes.as_ptr().cast::<c_void>(),
                BytecodeLength: vertex_module.bytes.len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: fragment_module.bytes.as_ptr().cast::<c_void>(),
                BytecodeLength: fragment_module.bytes.len(),
            },
            BlendState: blend,
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: match descriptor.primitive.cull_mode {
                    CullMode::None => D3D12_CULL_MODE_NONE,
                    CullMode::Front => D3D12_CULL_MODE_FRONT,
                    CullMode::Back => D3D12_CULL_MODE_BACK,
                },
                FrontCounterClockwise: (descriptor.primitive.front_face
                    == FrontFace::CounterClockwise)
                    .into(),
                DepthClipEnable: true.into(),
                DepthBias: descriptor.depth_bias.map_or(0, |bias| bias.constant),
                SlopeScaledDepthBias: descriptor.depth_bias.map_or(0., |bias| bias.slope),
                ..Default::default()
            },
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: input_elements.as_ptr(),
                NumElements: input_elements.len() as u32,
            },
            PrimitiveTopologyType: topology_type,
            NumRenderTargets: descriptor.color_targets.len() as u32,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: descriptor.multisample.count,
                Quality: 0,
            },
            ..Default::default()
        };
        for (index, target) in descriptor.color_targets.iter().enumerate() {
            description.RTVFormats[index] = dxgi_format(target.format);
        }
        let mut stencil_reference = None;
        if let Some(depth) = descriptor.depth_stencil {
            description.DSVFormat = dxgi_format(depth.format);
            description.DepthStencilState = D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: if depth.depth_write {
                    D3D12_DEPTH_WRITE_MASK_ALL
                } else {
                    D3D12_DEPTH_WRITE_MASK_ZERO
                },
                DepthFunc: comparison(depth.depth_compare),
                ..Default::default()
            };
            if let Some(stencil) = depth.stencil {
                description.DepthStencilState.StencilEnable = true.into();
                description.DepthStencilState.StencilReadMask = stencil.read_mask;
                description.DepthStencilState.StencilWriteMask = stencil.write_mask;
                description.DepthStencilState.FrontFace = stencil_face(stencil.front);
                description.DepthStencilState.BackFace = stencil_face(stencil.back);
                stencil_reference = Some(u32::from(stencil.reference));
            }
        }

        // Everything that tells one pipeline's state from another's, for
        // its name in the pipeline library.
        let state = format!(
            "{:?}",
            (
                descriptor.vertex_buffers,
                descriptor.primitive,
                descriptor.multisample,
                descriptor.color_targets,
                descriptor.depth_stencil,
                descriptor.depth_bias,
            )
        );
        let created: Result<ID3D12PipelineState, _> = self.cached_graphics_pipeline(
            &super::pipeline_key(&[
                b"render",
                &vertex_module.bytes,
                &fragment_module.bytes,
                state.as_bytes(),
            ]),
            &description,
        );
        // Release the COM reference cloned into the description.
        // SAFETY: the field was constructed above and is not used again.
        unsafe {
            let _ = ManuallyDrop::take(&mut description.pRootSignature);
        }
        let pipeline = created
            .map_err(|error| RhiError::dx12("ID3D12Device::CreateGraphicsPipelineState", error))?;
        super::name_object(&pipeline, descriptor.label);
        Ok(Dx12RenderPipeline {
            pipeline,
            topology,
            stencil_reference,
        })
    }
}
