//! Unit tests of `dx12/mod.rs`.

use super::*;

#[test]
fn adapter_names_stop_at_the_first_nul() {
    let mut description = [0_u16; 128];
    description[..4].copy_from_slice(&[b'T' as u16, b'e' as u16, b's' as u16, b't' as u16]);
    assert_eq!(adapter_name(&description), "Test");
}

#[test]
#[ignore = "requires a DX12 adapter and debug layer"]
fn debug_counter_reads_errors_and_excludes_warnings() {
    use windows::Win32::Graphics::Direct3D12::D3D12_MESSAGE_SEVERITY_WARNING;

    let _messages = crate::VALIDATION_MESSAGES
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let device = Dx12Device::open(AdapterPreference::HighPerformance, true, false).unwrap();
    let queue = device.native.cast::<ID3D12InfoQueue>().unwrap();
    assert_eq!(device.debug_error_count(), Some(0));
    // SAFETY: diagnostic messages enter this isolated test device's queue;
    // they do not issue invalid GPU work or change any application device.
    unsafe {
        queue
            .AddApplicationMessage(
                D3D12_MESSAGE_SEVERITY_WARNING,
                windows::core::s!("counter warning probe"),
            )
            .unwrap();
        queue
            .AddApplicationMessage(
                D3D12_MESSAGE_SEVERITY_ERROR,
                windows::core::s!("counter error probe"),
            )
            .unwrap();
    }
    assert_eq!(device.debug_error_count(), Some(1));
    assert_eq!(
        device.debug_error_count(),
        Some(1),
        "reading does not drain diagnostics"
    );
    // What each said is kept with how serious it was.
    let messages = device.debug_messages().unwrap();
    for (severity, text) in [
        (crate::DebugSeverity::Warning, "counter warning probe"),
        (crate::DebugSeverity::Error, "counter error probe"),
    ] {
        assert!(
            messages
                .iter()
                .any(|message| message.severity == severity && message.text == text),
            "{text} is not among {messages:?}"
        );
    }
    // The device may outlive this test in another's hands: leave its
    // queue as it was found.
    // SAFETY: the queue belongs to this live device.
    unsafe { queue.ClearStoredMessages() };
    assert_eq!(device.debug_error_count(), Some(0));
}
