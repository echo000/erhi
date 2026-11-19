//! Unit tests of `vulkan/mod.rs`.

use super::*;

#[test]
fn required_extensions_are_unique() {
    let extensions: Vec<&CStr> = base_device_extensions()
        .into_iter()
        .chain(ray_tracing_device_extensions())
        .collect();
    for (index, extension) in extensions.iter().enumerate() {
        assert!(!extensions[..index].contains(extension));
    }
}

#[test]
fn memory_type_filter_respects_flags_and_bits() {
    let mut properties = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: 2,
        ..Default::default()
    };
    properties.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    properties.memory_types[1].property_flags =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    assert_eq!(
        find_memory_type(
            &properties,
            0b11,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        ),
        Some(1)
    );
    assert_eq!(
        find_memory_type(&properties, 0b01, vk::MemoryPropertyFlags::HOST_VISIBLE),
        None
    );
}

#[test]
#[ignore = "requires a Vulkan device and the validation layer"]
fn debug_messages_keep_what_the_layer_said() {
    let _messages = crate::VALIDATION_MESSAGES
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let device =
        VulkanDevice::open(AdapterPreference::HighPerformance, true, false, (&[], &[])).unwrap();
    let Some(messenger) = &device.debug_messenger else {
        println!("no validation layer is installed");
        return;
    };
    let before = device.debug_error_count().unwrap();
    for (severity, text) in [
        (
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            c"erhi warning probe",
        ),
        (
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            c"erhi error probe",
        ),
    ] {
        let data = vk::DebugUtilsMessengerCallbackDataEXT::default().message(text);
        // SAFETY: a message of the application's own enters this test
        // device's messengers; no GPU work is issued.
        unsafe {
            messenger.loader.submit_debug_utils_message(
                severity,
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                &data,
            );
        }
    }
    assert_eq!(device.debug_error_count(), Some(before + 1));
    let messages = device.debug_messages();
    for (severity, text) in [
        (crate::DebugSeverity::Warning, "erhi warning probe"),
        (crate::DebugSeverity::Error, "erhi error probe"),
    ] {
        assert!(
            messages
                .iter()
                .any(|message| message.severity == severity && message.text == text),
            "{text} is not among {messages:?}"
        );
    }
}
