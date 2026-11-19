use std::ffi::CString;

use ash::vk;

use super::VulkanDevice;
use super::image::vk_format;
use super::pipeline_layout::VulkanPipelineLayout;
use super::shader::VulkanShaderModule;
use crate::RhiError;
use crate::render_pipeline::{
    BlendFactor, BlendOperation, CompareFunction, CullMode, FrontFace, PrimitiveTopology,
    RenderPipelineDescriptor, StencilOperation, VertexFormat, VertexStep,
};

/// One owned graphics `VkPipeline` for the dynamic-rendering raster path.
pub(crate) struct VulkanRenderPipeline {
    device: ash::Device,
    pub(crate) pipeline: vk::Pipeline,
}

fn vertex_format(format: VertexFormat) -> vk::Format {
    match format {
        VertexFormat::Float32 => vk::Format::R32_SFLOAT,
        VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
        VertexFormat::Float16x2 => vk::Format::R16G16_SFLOAT,
        VertexFormat::Float16x4 => vk::Format::R16G16B16A16_SFLOAT,
        VertexFormat::Uint32 => vk::Format::R32_UINT,
        VertexFormat::Uint32x2 => vk::Format::R32G32_UINT,
        VertexFormat::Uint32x3 => vk::Format::R32G32B32_UINT,
        VertexFormat::Uint32x4 => vk::Format::R32G32B32A32_UINT,
        VertexFormat::Sint32 => vk::Format::R32_SINT,
        VertexFormat::Sint32x2 => vk::Format::R32G32_SINT,
        VertexFormat::Sint32x3 => vk::Format::R32G32B32_SINT,
        VertexFormat::Sint32x4 => vk::Format::R32G32B32A32_SINT,
        VertexFormat::Uint16x2 => vk::Format::R16G16_UINT,
        VertexFormat::Uint16x4 => vk::Format::R16G16B16A16_UINT,
        VertexFormat::Sint16x2 => vk::Format::R16G16_SINT,
        VertexFormat::Sint16x4 => vk::Format::R16G16B16A16_SINT,
        VertexFormat::Unorm16x2 => vk::Format::R16G16_UNORM,
        VertexFormat::Unorm16x4 => vk::Format::R16G16B16A16_UNORM,
        VertexFormat::Snorm16x2 => vk::Format::R16G16_SNORM,
        VertexFormat::Snorm16x4 => vk::Format::R16G16B16A16_SNORM,
        VertexFormat::Uint8x2 => vk::Format::R8G8_UINT,
        VertexFormat::Uint8x4 => vk::Format::R8G8B8A8_UINT,
        VertexFormat::Sint8x2 => vk::Format::R8G8_SINT,
        VertexFormat::Sint8x4 => vk::Format::R8G8B8A8_SINT,
        VertexFormat::Unorm8x2 => vk::Format::R8G8_UNORM,
        VertexFormat::Unorm8x4 => vk::Format::R8G8B8A8_UNORM,
        VertexFormat::Snorm8x2 => vk::Format::R8G8_SNORM,
        VertexFormat::Snorm8x4 => vk::Format::R8G8B8A8_SNORM,
    }
}

fn blend_factor(factor: BlendFactor) -> vk::BlendFactor {
    match factor {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::Src => vk::BlendFactor::SRC_COLOR,
        BlendFactor::OneMinusSrc => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        BlendFactor::Dst => vk::BlendFactor::DST_COLOR,
        BlendFactor::OneMinusDst => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        BlendFactor::SrcAlphaSaturated => vk::BlendFactor::SRC_ALPHA_SATURATE,
    }
}

fn blend_operation(operation: BlendOperation) -> vk::BlendOp {
    match operation {
        BlendOperation::Add => vk::BlendOp::ADD,
        BlendOperation::Subtract => vk::BlendOp::SUBTRACT,
        BlendOperation::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        BlendOperation::Min => vk::BlendOp::MIN,
        BlendOperation::Max => vk::BlendOp::MAX,
    }
}

pub(crate) fn compare_op(function: CompareFunction) -> vk::CompareOp {
    match function {
        CompareFunction::Never => vk::CompareOp::NEVER,
        CompareFunction::Less => vk::CompareOp::LESS,
        CompareFunction::Equal => vk::CompareOp::EQUAL,
        CompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareFunction::Greater => vk::CompareOp::GREATER,
        CompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        CompareFunction::Always => vk::CompareOp::ALWAYS,
    }
}

fn stencil_op(operation: StencilOperation) -> vk::StencilOp {
    match operation {
        StencilOperation::Keep => vk::StencilOp::KEEP,
        StencilOperation::Zero => vk::StencilOp::ZERO,
        StencilOperation::Replace => vk::StencilOp::REPLACE,
        StencilOperation::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        StencilOperation::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        StencilOperation::Invert => vk::StencilOp::INVERT,
        StencilOperation::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        StencilOperation::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    }
}

impl VulkanDevice {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_render_pipeline(
        &self,
        descriptor: &RenderPipelineDescriptor<'_>,
        vertex_entry: &str,
        fragment_entry: &str,
        vertex_module: &VulkanShaderModule,
        fragment_module: &VulkanShaderModule,
        layout: &VulkanPipelineLayout,
    ) -> Result<VulkanRenderPipeline, RhiError> {
        let vertex_entry =
            CString::new(vertex_entry).map_err(|_| RhiError::InvalidRenderPipeline {
                reason: "the vertex entry-point name contains an interior NUL byte",
            })?;
        let fragment_entry =
            CString::new(fragment_entry).map_err(|_| RhiError::InvalidRenderPipeline {
                reason: "the fragment entry-point name contains an interior NUL byte",
            })?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_module.handle)
                .name(&vertex_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_module.handle)
                .name(&fragment_entry),
        ];

        let layouts = descriptor.vertex_buffers;
        let bindings: Vec<_> = layouts
            .iter()
            .enumerate()
            .map(|(slot, layout)| {
                vk::VertexInputBindingDescription::default()
                    .binding(slot as u32)
                    .stride(layout.stride)
                    .input_rate(match layout.step {
                        VertexStep::Vertex => vk::VertexInputRate::VERTEX,
                        VertexStep::Instance => vk::VertexInputRate::INSTANCE,
                    })
            })
            .collect();
        let attributes: Vec<_> = layouts
            .iter()
            .enumerate()
            .flat_map(|(slot, layout)| {
                layout.attributes.iter().map(move |attribute| {
                    vk::VertexInputAttributeDescription::default()
                        .location(attribute.location)
                        .binding(slot as u32)
                        .format(vertex_format(attribute.format))
                        .offset(attribute.offset)
                })
            })
            .collect();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(
            match descriptor.primitive.topology {
                PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
                PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
                PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
                PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
                PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
            },
        );
        // Viewport and scissor are dynamic: the render pass sets them from
        // the actual attachment extent.
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(match descriptor.primitive.cull_mode {
                CullMode::None => vk::CullModeFlags::NONE,
                CullMode::Front => vk::CullModeFlags::FRONT,
                CullMode::Back => vk::CullModeFlags::BACK,
            })
            // Winding is judged as the triangle lies on the screen, and the
            // flipped viewport (negative height) puts it there as Direct3D
            // does: the two backends agree with no correction. (Guarded by
            // the pipelines hardware test, which culls on both.)
            .front_face(match descriptor.primitive.front_face {
                FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
                FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
            })
            .line_width(1.0)
            .depth_bias_enable(descriptor.depth_bias.is_some())
            .depth_bias_constant_factor(
                descriptor
                    .depth_bias
                    .map_or(0., |bias| bias.constant as f32),
            )
            .depth_bias_slope_factor(descriptor.depth_bias.map_or(0., |bias| bias.slope));
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(super::image::vk_samples(descriptor.multisample.count))
            .alpha_to_coverage_enable(descriptor.multisample.alpha_to_coverage);
        let blend_attachments: Vec<_> = descriptor
            .color_targets
            .iter()
            .map(|target| {
                let state = vk::PipelineColorBlendAttachmentState::default().color_write_mask(
                    vk::ColorComponentFlags::from_raw(u32::from(target.write_mask.0)),
                );
                match target.blend {
                    None => state.blend_enable(false),
                    Some(blend) => state
                        .blend_enable(true)
                        .src_color_blend_factor(blend_factor(blend.color.src_factor))
                        .dst_color_blend_factor(blend_factor(blend.color.dst_factor))
                        .color_blend_op(blend_operation(blend.color.operation))
                        .src_alpha_blend_factor(blend_factor(blend.alpha.src_factor))
                        .dst_alpha_blend_factor(blend_factor(blend.alpha.dst_factor))
                        .alpha_blend_op(blend_operation(blend.alpha.operation)),
                }
            })
            .collect();
        let blend_state =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

        let color_formats: Vec<_> = descriptor
            .color_targets
            .iter()
            .map(|target| vk_format(target.format))
            .collect();
        let mut depth_state = vk::PipelineDepthStencilStateCreateInfo::default();
        let mut depth_format = vk::Format::UNDEFINED;
        let mut stencil_format = vk::Format::UNDEFINED;
        if let Some(depth) = descriptor.depth_stencil {
            depth_format = vk_format(depth.format);
            if depth.format.has_stencil() {
                stencil_format = depth_format;
            }
            depth_state = depth_state
                .depth_test_enable(true)
                .depth_write_enable(depth.depth_write)
                .depth_compare_op(compare_op(depth.depth_compare));
            if let Some(stencil) = depth.stencil {
                let face = |face: crate::StencilFaceState| vk::StencilOpState {
                    fail_op: stencil_op(face.fail),
                    pass_op: stencil_op(face.pass),
                    depth_fail_op: stencil_op(face.depth_fail),
                    compare_op: compare_op(face.compare),
                    compare_mask: u32::from(stencil.read_mask),
                    write_mask: u32::from(stencil.write_mask),
                    reference: u32::from(stencil.reference),
                };
                depth_state = depth_state
                    .stencil_test_enable(true)
                    .front(face(stencil.front))
                    .back(face(stencil.back));
            }
        }
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format)
            .stencil_attachment_format(stencil_format);
        let create = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&blend_state)
            .depth_stencil_state(&depth_state)
            .dynamic_state(&dynamic_state)
            .layout(layout.handle())
            .push_next(&mut rendering_info);
        let (_held, cache) = self.pipeline_cache();
        // SAFETY: every referenced module, layout, and state array is live
        // for this immediate call; dynamic rendering is an enabled feature.
        let pipeline = unsafe {
            self.logical
                .handle
                .create_graphics_pipelines(cache, &[create], None)
        }
        .map_err(|(_, result)| RhiError::vk("vkCreateGraphicsPipelines", result))?[0];
        self.name_object(pipeline, descriptor.label);
        Ok(VulkanRenderPipeline {
            device: self.logical.handle.clone(),
            pipeline,
        })
    }
}

impl Drop for VulkanRenderPipeline {
    fn drop(&mut self) {
        // SAFETY: draw completion is awaited before pipeline owners drop.
        unsafe { self.device.destroy_pipeline(self.pipeline, None) };
    }
}
