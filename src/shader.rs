use std::fmt;

use crate::device::BackendDevice;
#[cfg(all(feature = "dx12", windows))]
use crate::dx12::Dx12ShaderModule;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanShaderModule;
use crate::{Backend, Device, RhiError};

/// The precompiled binary format a backend consumes, so shader libraries
/// can select the right blob without matching on [`crate::Backend`].
/// A future Metal backend adds a `MetalLib` arm here and every library
/// keyed on this enum keeps working.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderBinaryFormat {
    /// SPIR-V, for Vulkan.
    SpirV,
    /// DXIL, for Direct3D 12.
    Dxil,
}

impl Device {
    /// Which compiled form of a shader this device takes: what to pick the
    /// bytes of a [`ShaderSource`] by.
    #[must_use]
    pub fn shader_binary_format(&self) -> ShaderBinaryFormat {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(_) => ShaderBinaryFormat::SpirV,
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(_) => ShaderBinaryFormat::Dxil,
        }
    }
}

/// What a shader module is made from. Each backend takes its own compiled
/// bytes: SPIR-V for Vulkan, DXIL for Direct3D 12
/// ([`Device::shader_binary_format`] says which a device wants). The crate
/// compiles nothing and runs no tool: what language the bytes were compiled
/// from is the caller's own business.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum ShaderSource<'a> {
    /// Precompiled SPIR-V for the Vulkan backend.
    SpirV(&'a [u8]),
    /// Precompiled DXIL for the Direct3D 12 backend.
    Dxil(&'a [u8]),
}

impl fmt::Debug for ShaderSource<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SpirV(bytes) => write!(formatter, "SpirV({} bytes)", bytes.len()),
            Self::Dxil(bytes) => write!(formatter, "Dxil({} bytes)", bytes.len()),
        }
    }
}

/// What a shader module contains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShaderKind<'a> {
    /// A ray-tracing library holding every `[shader("...")]` entry point.
    RayTracingLibrary,
    /// One compute entry point with the given name.
    Compute {
        /// The entry point's name.
        entry: &'a str,
    },
    /// One vertex entry point with the given name.
    Vertex {
        /// The entry point's name.
        entry: &'a str,
    },
    /// One fragment (pixel) entry point with the given name.
    Fragment {
        /// The entry point's name.
        entry: &'a str,
    },
}

/// What a shader module is made from: compiled bytes, and what is in them.
/// [`Device::create_shader_module`] makes one.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ShaderModuleDescriptor<'a> {
    /// A name for the module, used in error messages.
    pub label: &'a str,
    /// What the bytes hold: one entry point of one stage, or a ray-tracing
    /// library.
    pub kind: ShaderKind<'a>,
    /// The compiled bytecode, in the format the device's backend accepts.
    pub source: ShaderSource<'a>,
}

impl<'a> ShaderModuleDescriptor<'a> {
    /// A module called `label`, holding what `kind` says, from `source`.
    #[must_use]
    pub const fn new(label: &'a str, kind: ShaderKind<'a>, source: ShaderSource<'a>) -> Self {
        Self {
            label,
            kind,
            source,
        }
    }
}

/// One owned shader module bound to the creating device's backend.
pub struct ShaderModule {
    device_id: u64,
    /// Entry-point names parsed from the binary where the format makes that
    /// cheap (SPIR-V `OpEntryPoint`); `None` when unknown (DXIL).
    entry_points: Option<Vec<String>>,
    /// The single entry name the descriptor's kind declared, when it has
    /// one. `slangc` renames a single-entry SPIR-V module's export to
    /// `main`, so the declared name is accepted as an alias for it.
    declared_entry: Option<String>,
    backend: ShaderModuleBackend,
}

pub(crate) enum ShaderModuleBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<VulkanShaderModule>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Box<Dx12ShaderModule>),
}

impl fmt::Debug for ShaderModule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ShaderModule")
            .field("backend", &self.backend())
            .finish_non_exhaustive()
    }
}

#[cfg(any(feature = "vulkan", test))]
const SPIRV_MAGIC: [u8; 4] = 0x0723_0203_u32.to_le_bytes();

impl Device {
    /// Create a shader module from compiled bytes.
    ///
    /// Bytes for the other backend, and bytes that are not what they are
    /// given as, are refused before any native call.
    pub fn create_shader_module(
        &self,
        descriptor: &ShaderModuleDescriptor<'_>,
    ) -> Result<ShaderModule, RhiError> {
        let (entry_points, backend) = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => {
                let bytes = match descriptor.source {
                    ShaderSource::SpirV(bytes) => validate_spirv(bytes)?.to_vec(),
                    ShaderSource::Dxil(_) => {
                        return Err(RhiError::ShaderSourceUnsupported {
                            backend: "vulkan",
                            source: "precompiled DXIL",
                        });
                    }
                };
                let entry_points = Some(parse_spirv_entry_points(&bytes));
                let native =
                    ShaderModuleBackend::Vulkan(Box::new(device.create_shader_module(&bytes)?));
                (entry_points, native)
            }
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => {
                let bytes = match descriptor.source {
                    ShaderSource::Dxil(bytes) => validate_dxil(bytes)?.to_vec(),
                    ShaderSource::SpirV(_) => {
                        return Err(RhiError::ShaderSourceUnsupported {
                            backend: "dx12",
                            source: "precompiled SPIR-V",
                        });
                    }
                };
                (
                    None,
                    ShaderModuleBackend::Dx12(Box::new(device.create_shader_module(bytes)?)),
                )
            }
        };
        let declared_entry = match descriptor.kind {
            ShaderKind::RayTracingLibrary => None,
            ShaderKind::Compute { entry }
            | ShaderKind::Vertex { entry }
            | ShaderKind::Fragment { entry } => Some(entry.to_owned()),
        };
        Ok(ShaderModule {
            device_id: self.id(),
            entry_points,
            declared_entry,
            backend,
        })
    }
}

impl ShaderModule {
    /// The backend of the device that made this.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            ShaderModuleBackend::Vulkan(_) => Backend::Vulkan,
            #[cfg(all(feature = "dx12", windows))]
            ShaderModuleBackend::Dx12(_) => Backend::Dx12,
        }
    }

    pub(crate) fn device_id(&self) -> u64 {
        self.device_id
    }

    pub(crate) fn native(&self) -> &ShaderModuleBackend {
        &self.backend
    }

    /// Whether `entry` exists in this module, when the binary format lets the
    /// common layer know without reflection (`None` means unknown). The
    /// declared single-entry name counts when the binary renamed it `main`.
    pub(crate) fn has_entry_point(&self, entry: &str) -> Option<bool> {
        self.entry_points
            .as_ref()
            .map(|entries| entries.iter().any(|name| name == entry) || self.is_alias_of_main(entry))
    }

    /// The name the native binary actually exports for `entry`.
    ///
    /// Only SPIR-V modules can rename entries, so DXIL-only builds never
    /// call this.
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    pub(crate) fn resolve_entry<'entry>(&self, entry: &'entry str) -> &'entry str {
        if self.is_alias_of_main(entry)
            && !self
                .entry_points
                .as_ref()
                .is_some_and(|entries| entries.iter().any(|name| name == entry))
        {
            "main"
        } else {
            entry
        }
    }

    fn is_alias_of_main(&self, entry: &str) -> bool {
        self.declared_entry.as_deref() == Some(entry)
            && self
                .entry_points
                .as_ref()
                .is_some_and(|entries| entries.iter().any(|name| name == "main"))
    }
}

/// Extract every `OpEntryPoint` name from a SPIR-V module. The header was
/// already validated; malformed instruction streams simply end the walk.
#[cfg(any(feature = "vulkan", test))]
fn parse_spirv_entry_points(bytes: &[u8]) -> Vec<String> {
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    let mut entries = Vec::new();
    let mut index = 5; // Header is five words.
    while index < words.len() {
        let word = words[index];
        let word_count = (word >> 16) as usize;
        let opcode = word & 0xffff;
        if word_count == 0 || index + word_count > words.len() {
            break;
        }
        const OP_ENTRY_POINT: u32 = 15;
        if opcode == OP_ENTRY_POINT && word_count > 3 {
            let name_words = &words[index + 3..index + word_count];
            let mut name_bytes: Vec<u8> = name_words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            if let Some(nul) = name_bytes.iter().position(|byte| *byte == 0) {
                name_bytes.truncate(nul);
            }
            if let Ok(name) = String::from_utf8(name_bytes) {
                entries.push(name);
            }
        }
        index += word_count;
    }
    entries
}

#[cfg(any(feature = "vulkan", test))]
fn validate_spirv(bytes: &[u8]) -> Result<&[u8], RhiError> {
    if bytes.len() < 20 || !bytes.len().is_multiple_of(4) || bytes[..4] != SPIRV_MAGIC {
        return Err(RhiError::InvalidShaderBlob { expected: "SPIR-V" });
    }
    Ok(bytes)
}

/// Signed and unsigned dxc output both use the DXBC container fourcc.
#[cfg(any(all(feature = "dx12", windows), test))]
fn validate_dxil(bytes: &[u8]) -> Result<&[u8], RhiError> {
    if bytes.len() < 8 || &bytes[..4] != b"DXBC" {
        return Err(RhiError::InvalidShaderBlob { expected: "DXIL" });
    }
    Ok(bytes)
}

#[cfg(test)]
#[path = "tests/shader.rs"]
mod tests;
