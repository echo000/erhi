//! Unit tests of `sampler.rs`.

use super::*;

#[test]
fn descriptions_no_device_takes_are_refused() {
    assert!(SamplerDescriptor::default().validate().is_ok());
    let wrong = |descriptor: SamplerDescriptor| {
        matches!(descriptor.validate(), Err(RhiError::InvalidSampler { .. }))
    };
    assert!(wrong(SamplerDescriptor {
        lod_min: 4.0,
        lod_max: 2.0,
        ..Default::default()
    }));
    assert!(wrong(SamplerDescriptor {
        lod_bias: f32::NAN,
        ..Default::default()
    }));
    assert!(wrong(SamplerDescriptor {
        anisotropy: 0,
        ..Default::default()
    }));
    assert!(wrong(SamplerDescriptor {
        anisotropy: 17,
        ..Default::default()
    }));
    // Anisotropy with a nearest filter anywhere.
    assert!(wrong(SamplerDescriptor {
        anisotropy: 8,
        mip_filter: FilterMode::Nearest,
        ..Default::default()
    }));
    assert!(
        SamplerDescriptor {
            anisotropy: 16,
            ..Default::default()
        }
        .validate()
        .is_ok()
    );
}
