//! Unit tests of `pipeline_layout.rs`.

use super::*;

#[test]
fn duplicate_bindings_are_rejected() {
    let bindings = [
        DescriptorBinding {
            binding: 2,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
            visibility: ShaderStages::COMPUTE,
        },
        DescriptorBinding {
            binding: 2,
            descriptor_type: DescriptorType::UniformBuffer,
            count: 1,
            visibility: ShaderStages::COMPUTE,
        },
    ];
    let sets = [DescriptorSetLayoutDescriptor {
        set: 0,
        bindings: &bindings,
        register_offsets: RegisterOffsets::NONE,
    }];
    let error = validate_pipeline_layout(PipelineLayoutDescriptor {
        sets: &sets,
        push_constants: None,
    })
    .expect_err("duplicate binding must fail");
    assert!(matches!(
        error,
        RhiError::InvalidDescriptorBinding {
            reason: "binding number is duplicated within the set",
            ..
        }
    ));
}

#[test]
fn a_layout_has_its_sets_in_order_and_no_more_than_eight() {
    let bindings = [DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
        visibility: ShaderStages::COMPUTE,
    }];
    let set = |set| DescriptorSetLayoutDescriptor {
        set,
        bindings: &bindings,
        register_offsets: RegisterOffsets::NONE,
    };
    assert!(
        validate_pipeline_layout(PipelineLayoutDescriptor {
            sets: &[set(0)],
            push_constants: None,
        })
        .is_ok()
    );
    let sets: Vec<_> = (0..MAX_DESCRIPTOR_SETS as u32 + 1).map(set).collect();
    assert!(
        validate_pipeline_layout(PipelineLayoutDescriptor {
            sets: &sets[..MAX_DESCRIPTOR_SETS],
            push_constants: None,
        })
        .is_ok()
    );
    assert!(matches!(
        validate_pipeline_layout(PipelineLayoutDescriptor {
            sets: &sets,
            push_constants: None,
        }),
        Err(RhiError::InvalidDescriptorSetOrder { .. })
    ));
    // Numbered from nought, with none left out.
    assert!(matches!(
        validate_pipeline_layout(PipelineLayoutDescriptor {
            sets: &[set(0), set(2)],
            push_constants: None,
        }),
        Err(RhiError::InvalidDescriptorSetOrder {
            expected: 1,
            actual: 2
        })
    ));
}
