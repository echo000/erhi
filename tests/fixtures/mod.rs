//! The shaders the hardware tests run, compiled ahead and checked in beside
//! their source: `compile_shaders.ps1` at the crate's root rebuilds them.
//!
//! A ray-tracing library `name.slang` or `name.hlsl` is `name.spv` and
//! `name.dxil`; any other entry point of one is `name.<entry>.spv` and
//! `name.<entry>.dxil`. A GLSL stage `name.comp` is `name.comp.spv`, for
//! Vulkan alone.

#![allow(dead_code)]

use std::path::Path;

use erhi::{
    Device, RhiError, ShaderBinaryFormat, ShaderKind, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

/// The compiled bytes of `file` for the backend of `device`.
pub fn bytes(device: &Device, file: &str, kind: ShaderKind<'_>) -> Vec<u8> {
    let stem = match kind {
        ShaderKind::Compute { entry }
        | ShaderKind::Vertex { entry }
        | ShaderKind::Fragment { entry } => format!("{file}.{entry}"),
        _ => file.to_owned(),
    };
    let extension = match device.shader_binary_format() {
        ShaderBinaryFormat::SpirV => "spv",
        ShaderBinaryFormat::Dxil => "dxil",
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{stem}.{extension}"));
    std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error} (compile_shaders.ps1 builds it)",
            path.display()
        )
    })
}

/// A module of `file`'s entry point of `kind`, for the backend of `device`.
pub fn module(
    device: &Device,
    label: &str,
    file: &str,
    kind: ShaderKind<'_>,
) -> Result<ShaderModule, RhiError> {
    let bytes = bytes(device, file, kind);
    let source = match device.shader_binary_format() {
        ShaderBinaryFormat::SpirV => ShaderSource::SpirV(&bytes),
        ShaderBinaryFormat::Dxil => ShaderSource::Dxil(&bytes),
    };
    device.create_shader_module(&ShaderModuleDescriptor::new(label, kind, source))
}

/// A module of the GLSL stage `file` (`name.comp`, `name.rgen`), which only
/// a Vulkan device takes.
pub fn glsl(
    device: &Device,
    label: &str,
    file: &str,
    kind: ShaderKind<'_>,
) -> Result<ShaderModule, RhiError> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{file}.spv"));
    let bytes = std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error} (compile_shaders.ps1 builds it)",
            path.display()
        )
    });
    device.create_shader_module(&ShaderModuleDescriptor::new(
        label,
        kind,
        ShaderSource::SpirV(&bytes),
    ))
}
