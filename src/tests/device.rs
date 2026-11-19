//! Unit tests of `device.rs`.

use super::*;

#[test]
fn backend_names_parse_without_platform_types() {
    assert_eq!("auto".parse::<Backend>().expect("auto"), Backend::Auto);
    assert_eq!("vk".parse::<Backend>().expect("vulkan"), Backend::Vulkan);
    assert_eq!("d3d12".parse::<Backend>().expect("dx12"), Backend::Dx12);
    // A backend that is not written has no name.
    assert!("metal".parse::<Backend>().is_err());
}

#[test]
fn default_selection_is_runtime_auto() {
    assert_eq!(DeviceOptions::default().backend, Backend::Auto);
    assert!(Backend::Auto.is_compiled());
    assert_eq!(
        Backend::Dx12.is_compiled(),
        cfg!(all(feature = "dx12", windows))
    );
}

#[test]
fn automatic_backend_order_matches_platform_policy() {
    let backends = Device::auto_backend_order();
    assert!(backends.iter().all(|backend| backend.is_compiled()));
    #[cfg(all(windows, feature = "dx12"))]
    assert_eq!(backends.first(), Some(&Backend::Dx12));
    #[cfg(all(not(all(windows, feature = "dx12")), feature = "vulkan"))]
    assert_eq!(backends.first(), Some(&Backend::Vulkan));
    #[cfg(all(windows, feature = "dx12", feature = "vulkan"))]
    assert_eq!(backends, &[Backend::Dx12, Backend::Vulkan]);
}

#[cfg(not(all(feature = "dx12", windows)))]
#[test]
fn explicit_uncompiled_backend_has_a_concrete_error() {
    let error = Device::open(DeviceOptions {
        backend: Backend::Dx12,
        ..DeviceOptions::default()
    })
    .expect_err("DX12 is intentionally not implemented yet");
    assert!(matches!(error, RhiError::BackendNotCompiled("dx12")));
}
