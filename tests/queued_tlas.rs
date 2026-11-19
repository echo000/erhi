//! Queued TLAS ownership, validation and bounded retirement on native backends.

#![cfg(feature = "vulkan")]

use erhi::*;

#[test]
#[ignore = "requires Vulkan RT and DXR adapters"]
fn queued_builds_and_refits_retain_inputs_without_memory_growth() -> Result<(), RhiError> {
    for backend in [Backend::Vulkan, Backend::Dx12]
        .into_iter()
        .filter(|backend| backend.is_compiled())
    {
        let device = Device::open(DeviceOptions {
            backend,
            debug: true,
            frames_in_flight: 3,
            ..Default::default()
        })?;
        let blas = device.build_procedural_aabb_blas(&[Aabb::new([0.; 3], [1.; 3])?])?;
        let other = device.build_procedural_aabb_blas(&[Aabb::new([2.; 3], [3.; 3])?])?;
        let mut ring = device.create_frame_ring()?;
        let mut frame = ring.begin_frame(&device)?;
        assert!(matches!(
            frame.build_tlas(&[]),
            Err(RhiError::EmptyAccelerationStructureInstances)
        ));
        let mut records = vec![AccelerationStructureInstance::new(&blas); 16];
        let tlas = frame.build_updatable_tlas(&records)?;
        let static_tlas = frame.build_tlas(&records)?;
        assert!(matches!(
            frame.update_tlas(&static_tlas, &records),
            Err(RhiError::AccelerationStructureNotUpdatable)
        ));
        assert!(matches!(
            frame.update_tlas(&tlas, &records[..15]),
            Err(RhiError::AccelerationStructureShapeChanged(_))
        ));
        records[0].acceleration_structure = &other;
        assert!(matches!(
            frame.update_tlas(&tlas, &records),
            Err(RhiError::AccelerationStructureShapeChanged(_))
        ));
        records[0].acceleration_structure = &blas;
        records[0].transform[3] = f32::NAN;
        assert!(matches!(
            frame.update_tlas(&tlas, &records),
            Err(RhiError::InvalidAccelerationStructureInstance(_))
        ));
        records[0].transform[3] = 0.;
        // Build then refit in the same command stream. A private input snapshot
        // must preserve both operations even after these CPU records change.
        frame.update_tlas(&tlas, &records)?;
        drop(static_tlas);
        frame.finish()?;

        let address = tlas.device_address();
        let mut baseline = 0;
        for tick in 0..256 {
            for (i, record) in records.iter_mut().enumerate() {
                record.transform[3] = (tick + i) as f32;
                record.custom_index = i as u32;
                record.mask = if tick % 2 == 0 { 255 } else { 1 };
            }
            let mut frame = ring.begin_frame(&device)?;
            frame.update_tlas(&tlas, &records)?;
            frame.finish()?;
            assert_eq!(tlas.device_address(), address);
            if tick == 32 {
                baseline = device.allocated_bytes();
            }
        }
        ring.wait_idle(&device)?;
        assert!(
            device.allocated_bytes() <= baseline,
            "{backend:?}: per-frame input snapshots leaked"
        );

        // The synchronous compatibility path shares the same preparation and
        // barriers, and can follow queued updates without an external idle wait.
        device.update_tlas(&tlas, &records)?;
        let synchronous = device.build_tlas(&records)?;
        assert_ne!(synchronous.device_address(), 0);
        drop(records);
        drop(blas);
        let mut frame = ring.begin_frame(&device)?;
        let owned = device.build_procedural_aabb_blas(&[Aabb::new([-1.; 3], [1.; 3])?])?;
        let temporary = frame.build_tlas(&[AccelerationStructureInstance::new(&owned)])?;
        drop(temporary);
        drop(owned);
        frame.finish()?;
        // Drop the ring with its final build outstanding; the fence must protect
        // storage, BLAS dependencies, input and scratch even without user handles.
        drop(ring);
        eprintln!("{:#?}", device.debug_messages());
        assert_eq!(
            device.debug_error_count().unwrap_or(0),
            0,
            "{backend:?} validation"
        );
        println!(
            "{backend:?}: queued build/refit validation, 256 updates, stable allocation and pending-drop ownership passed"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires Vulkan RT and DXR adapters"]
fn batched_blas_builds_match_single_builds_in_a_tlas() -> Result<(), RhiError> {
    for backend in [Backend::Vulkan, Backend::Dx12]
        .into_iter()
        .filter(|backend| backend.is_compiled())
    {
        let device = Device::open(DeviceOptions {
            backend,
            debug: true,
            frames_in_flight: 2,
            ..Default::default()
        })?;
        assert!(device.build_procedural_aabb_blas_batch(&[])?.is_empty());
        let sets: Vec<Vec<Aabb>> = (0..5)
            .map(|i| {
                let base = i as f32 * 4.;
                vec![
                    Aabb::new([base; 3], [base + 1.; 3]).unwrap(),
                    Aabb::new([base + 2.; 3], [base + 3.; 3]).unwrap(),
                ]
            })
            .collect();
        let slices: Vec<&[Aabb]> = sets.iter().map(Vec::as_slice).collect();
        let batch = device.build_procedural_aabb_blas_batch(&slices)?;
        assert_eq!(batch.len(), 5, "{backend:?}: one BLAS per set, in order");
        let single = device.build_procedural_aabb_blas(&sets[0])?;
        assert_eq!(
            batch[0].report().acceleration_structure_size,
            single.report().acceleration_structure_size,
            "{backend:?}: a batched build sizes like a single one"
        );
        assert!(matches!(
            device.build_procedural_aabb_blas_batch(&[&sets[0], &[]]),
            Err(RhiError::InvalidAccelerationStructureInstance(_))
        ));
        let mut ring = device.create_frame_ring()?;
        let mut frame = ring.begin_frame(&device)?;
        let records: Vec<_> = batch
            .iter()
            .map(AccelerationStructureInstance::new)
            .collect();
        let tlas = frame.build_tlas(&records)?;
        assert_eq!(tlas.kind(), AccelerationStructureKind::TopLevel);
        frame.finish()?;
        device.wait_idle()?;
        if let Some(count) = device.debug_error_count() {
            assert_eq!(count, 0, "{backend:?}");
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires Vulkan RT and DXR adapters"]
fn frame_queued_blas_builds_feed_a_tlas_in_the_same_frame() -> Result<(), RhiError> {
    for backend in [Backend::Vulkan, Backend::Dx12]
        .into_iter()
        .filter(|backend| backend.is_compiled())
    {
        let device = Device::open(DeviceOptions {
            backend,
            debug: true,
            frames_in_flight: 2,
            ..Default::default()
        })?;
        let sets: Vec<Vec<Aabb>> = (0..3)
            .map(|i| vec![Aabb::new([i as f32 * 2.; 3], [i as f32 * 2. + 1.; 3]).unwrap()])
            .collect();
        let slices: Vec<&[Aabb]> = sets.iter().map(Vec::as_slice).collect();
        let mut ring = device.create_frame_ring()?;
        let mut frame = ring.begin_frame(&device)?;
        assert!(frame.build_procedural_aabb_blas_batch(&[])?.is_empty());
        let batch = frame.build_procedural_aabb_blas_batch(&slices)?;
        assert_eq!(batch.len(), 3, "{backend:?}");
        let records: Vec<_> = batch
            .iter()
            .map(AccelerationStructureInstance::new)
            .collect();
        // The TLAS build records after the BLAS builds in the same frame.
        let tlas = frame.build_updatable_tlas(&records)?;
        frame.finish()?;
        let mut frame = ring.begin_frame(&device)?;
        frame.update_tlas(&tlas, &records)?;
        frame.finish()?;
        device.wait_idle()?;
        if let Some(count) = device.debug_error_count() {
            assert_eq!(count, 0, "{backend:?}");
        }
    }
    Ok(())
}
