//! Unit tests of `command_encoder.rs`.

use super::*;

#[test]
fn what_an_image_is_needed_for_is_served_by_one_state_or_none() {
    use ResourceState as S;

    // One need, one state.
    for (need, state) in [
        (ImageNeed::SAMPLED, S::ShaderResource),
        (ImageNeed::STORAGE, S::UnorderedAccess),
        (ImageNeed::COLOR_TARGET, S::RenderTarget),
        (ImageNeed::DEPTH_TARGET, S::DepthWrite),
        (ImageNeed::COPY_SOURCE, S::CopySource),
        (ImageNeed::COPY_DESTINATION, S::CopyDest),
    ] {
        let (accepted, preferred) = need.states().expect("served");
        assert_eq!(preferred, state, "{need:?}");
        assert!(accepted.contains(state), "{need:?}");
        assert!(!accepted.contains(S::Undefined), "{need:?}");
        assert_eq!(need.written(), need == ImageNeed::STORAGE, "{need:?}");
    }
    // Nothing serves an attachment that is also read, nor an image that is
    // sampled and written at once.
    for unserved in [
        ImageNeed(ImageNeed::COLOR_TARGET.0 | ImageNeed::SAMPLED.0),
        ImageNeed(ImageNeed::DEPTH_TARGET.0 | ImageNeed::SAMPLED.0),
        ImageNeed(ImageNeed::STORAGE.0 | ImageNeed::SAMPLED.0),
        ImageNeed(ImageNeed::COPY_SOURCE.0 | ImageNeed::COPY_DESTINATION.0),
        ImageNeed(0),
    ] {
        assert!(unserved.states().is_none(), "{unserved:?}");
    }
}

#[test]
fn what_a_buffer_is_needed_for_is_served_by_one_state_or_none() {
    use ResourceState as S;

    let states = |need: BufferNeed| need.states().expect("served");
    // Each read has its own state, and the state that is every read
    // serves it as well.
    for (need, preferred) in [
        (BufferNeed::VERTEX, S::VertexBuffer),
        (BufferNeed::INDEX, S::IndexBuffer),
        (BufferNeed::INDIRECT, S::IndirectArgument),
        (BufferNeed::UNIFORM, S::ConstantBuffer),
        (BufferNeed::READ, S::ShaderResource),
        (BufferNeed::COPY_SOURCE, S::CopySource),
    ] {
        let (accepted, moved_to) = states(need);
        assert_eq!(moved_to, preferred, "{need:?}");
        assert!(
            accepted.contains(preferred) && accepted.contains(S::GenericRead),
            "{need:?}"
        );
        assert!(!accepted.contains(S::UnorderedAccess), "{need:?}");
    }
    // Reads of more than one kind: only the state that is all of them.
    let (accepted, preferred) = states(BufferNeed(BufferNeed::VERTEX.0 | BufferNeed::INDEX.0));
    assert_eq!(preferred, S::GenericRead);
    assert!(!accepted.contains(S::VertexBuffer));
    // Written: by a shader, or by a copy, and by nothing else at once.
    assert_eq!(states(BufferNeed::STORAGE).1, S::UnorderedAccess);
    assert_eq!(states(BufferNeed::COPY_DESTINATION).1, S::CopyDest);
    for unserved in [
        BufferNeed(BufferNeed::STORAGE.0 | BufferNeed::VERTEX.0),
        BufferNeed(BufferNeed::STORAGE.0 | BufferNeed::READ.0),
        BufferNeed(BufferNeed::COPY_DESTINATION.0 | BufferNeed::COPY_SOURCE.0),
        BufferNeed(BufferNeed::COPY_DESTINATION.0 | BufferNeed::STORAGE.0),
        BufferNeed(0),
    ] {
        assert!(unserved.states().is_none(), "{unserved:?}");
    }
}

#[test]
fn every_state_survives_being_kept_as_a_byte() {
    use ResourceState as S;

    for state in [
        S::Undefined,
        S::VertexBuffer,
        S::IndexBuffer,
        S::ConstantBuffer,
        S::IndirectArgument,
        S::ShaderResource,
        S::UnorderedAccess,
        S::RenderTarget,
        S::DepthWrite,
        S::CopySource,
        S::CopyDest,
        S::GenericRead,
    ] {
        assert_eq!(S::from_bits(state.bits()), state);
    }
}
