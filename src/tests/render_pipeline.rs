//! Unit tests of `render_pipeline.rs`.

use super::*;

#[test]
fn instance_attributes_must_fit_and_have_locations_distinct_from_vertices() {
    let vertex_attributes = [VertexAttribute::new(0, 0, VertexFormat::Float32x4)];
    let mut instance_attributes = [VertexAttribute::new(4, 0, VertexFormat::Float32x4)];
    let vertex = VertexBufferLayout {
        stride: 16,
        step: VertexStep::Vertex,
        attributes: &vertex_attributes,
    };
    let validate = |attributes: &[VertexAttribute], stride| {
        validate_vertex_layouts(&[
            vertex,
            VertexBufferLayout {
                stride,
                step: VertexStep::Instance,
                attributes,
            },
        ])
    };
    // No streams at all, and as many as there may be.
    assert!(validate_vertex_layouts(&[]).is_ok());
    assert!(validate_vertex_layouts(&[vertex; MAX_VERTEX_BUFFERS + 1]).is_err());
    assert!(validate(&instance_attributes, 16).is_ok());
    assert!(validate(&instance_attributes, 8).is_err());
    assert!(validate(&instance_attributes, 0).is_err());
    assert!(validate(&[], 16).is_err());
    instance_attributes[0].location = 0;
    assert!(validate(&instance_attributes, 16).is_err());
    instance_attributes[0].location = 4;
    instance_attributes[0].offset = u32::MAX;
    assert!(validate(&instance_attributes, 16).is_err());
}

#[test]
fn vertex_formats_have_the_expected_sizes() {
    assert_eq!(VertexFormat::Float32.size(), 4);
    assert_eq!(VertexFormat::Float32x2.size(), 8);
    assert_eq!(VertexFormat::Float32x4.size(), 16);
    assert_eq!(VertexFormat::Unorm8x4.size(), 4);
    assert_eq!(VertexFormat::Float32x3.size(), 12);
    assert_eq!(VertexFormat::Float16x4.size(), 8);
    assert_eq!(VertexFormat::Uint8x2.size(), 2);
}

#[test]
fn blend_modes_are_the_factors_they_always_were() {
    use BlendFactor::{One, OneMinusSrcAlpha, SrcAlpha, Zero};
    assert_eq!(BlendMode::Opaque.state(), None);
    let alpha = BlendMode::Alpha.state().unwrap();
    assert_eq!(
        (alpha.color.src_factor, alpha.color.dst_factor),
        (SrcAlpha, OneMinusSrcAlpha)
    );
    assert_eq!(
        (alpha.alpha.src_factor, alpha.alpha.dst_factor),
        (One, OneMinusSrcAlpha)
    );
    let additive = BlendMode::Additive.state().unwrap();
    assert_eq!(
        (additive.color.dst_factor, additive.alpha.src_factor),
        (One, Zero)
    );
    let target = ColorTargetState::new(ImageFormat::Rgba8Unorm, BlendMode::PremultipliedAlpha);
    assert_eq!(target.write_mask, ColorWrites::ALL);
    assert_eq!(target.blend.unwrap().color.src_factor, One);
}
