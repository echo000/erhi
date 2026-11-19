//! Unit tests of `descriptor_set.rs`.

use super::*;
use crate::RegisterOffsets;

#[test]
#[ignore = "DX12/Vulkan descriptor allocation and lifetime check"]
fn pooled_tables_remain_unique_across_growth_and_recycling() {
    let _messages = crate::VALIDATION_MESSAGES
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    use crate::{
        DescriptorSetLayoutDescriptor, DeviceOptions, PipelineLayoutDescriptor, ShaderStages,
    };

    fn identity(set: &DescriptorSet) -> (u64, u64) {
        match set.native().unwrap() {
            #[cfg(feature = "vulkan")]
            DescriptorSetBackend::Vulkan(set) => {
                use ash::vk::Handle;
                (0, set.handle.as_raw())
            }
            #[cfg(all(feature = "dx12", windows))]
            DescriptorSetBackend::Dx12(set) => (1, u64::from(set.offset)),
        }
    }

    for backend in [Backend::Dx12, Backend::Vulkan]
        .into_iter()
        .filter(|b| b.is_compiled())
    {
        let device = Device::open(DeviceOptions {
            backend,
            debug: true,
            ..Default::default()
        })
        .unwrap();
        let bindings: Vec<_> = (0..8)
            .map(|binding| DescriptorBinding {
                binding,
                descriptor_type: DescriptorType::UniformBuffer,
                count: 1,
                visibility: ShaderStages::COMPUTE,
            })
            .collect();
        let layout = device
            .create_pipeline_layout(PipelineLayoutDescriptor {
                sets: &[DescriptorSetLayoutDescriptor {
                    set: 0,
                    bindings: &bindings,
                    register_offsets: RegisterOffsets::NONE,
                }],
                push_constants: None,
            })
            .unwrap();
        // (More descriptors than the heap of tables begins with: it grows.)
        let mut sets: Vec<_> = (0..9000)
            .map(|_| device.create_descriptor_set(&layout, 0).unwrap())
            .collect();
        let identities: std::collections::HashSet<_> = sets.iter().map(identity).collect();
        assert_eq!(identities.len(), sets.len());
        let retained: Vec<_> = sets.iter().take(100).map(identity).collect();
        sets.truncate(100);
        sets.extend((0..8900).map(|_| device.create_descriptor_set(&layout, 0).unwrap()));
        assert_eq!(
            sets.iter().take(100).map(identity).collect::<Vec<_>>(),
            retained
        );
        assert_eq!(
            sets.iter()
                .map(identity)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            sets.len()
        );
        assert_eq!(device.debug_error_count().unwrap_or(0), 0);
        let buffer = device
            .create_buffer(crate::BufferDescriptor {
                size: 512,
                usages: BufferUsages::UNIFORM,
                memory: crate::BufferMemory::Upload,
            })
            .unwrap();
        sets[0]
            .write_uniform_buffer(0, BufferBinding::range(&buffer, 0, 256))
            .unwrap();
        sets[0]
            .write_uniform_buffer(1, BufferBinding::range(&buffer, 0, 256))
            .unwrap();
        assert!(sets[0].backend.get().is_none());
        let snapshot = sets[0].clone();
        let original = identity(&snapshot);
        sets[0]
            .write_uniform_buffer(0, BufferBinding::range(&buffer, 0, 256))
            .unwrap();
        assert!(Arc::ptr_eq(&snapshot.backend, &sets[0].backend));
        assert_eq!(identity(&sets[0]), original);
        sets[0]
            .write_uniform_buffer(0, BufferBinding::range(&buffer, 256, 256))
            .unwrap();
        assert!(sets[0].backend.get().is_none());
        assert_eq!(identity(&snapshot), original);
        assert_ne!(identity(&sets[0]), original);
        assert!(matches!(
            snapshot.retained.get(&(0, 0)),
            Some(RetainedResource::Buffer(_, 0, 256, None))
        ));
        assert!(matches!(
            sets[0].retained.get(&(0, 0)),
            Some(RetainedResource::Buffer(_, 256, 256, None))
        ));
        assert!(matches!(
            sets[0].retained.get(&(1, 0)),
            Some(RetainedResource::Buffer(_, 0, 256, None))
        ));
        drop(layout);
        drop(device);
        // Even unwritten sets keep their pool and native device alive.
        drop(sets);
    }
}

#[test]
fn shared_alignments_match_the_dx12_rules() {
    assert_eq!(UNIFORM_BUFFER_OFFSET_ALIGNMENT, 256);
    assert_eq!(STORAGE_BUFFER_ALIGNMENT, 4);
}
