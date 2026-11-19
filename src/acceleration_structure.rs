use std::fmt;
use std::sync::Arc;

use crate::command_encoder::CommandEncoder;
use crate::device::BackendDevice;
#[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
#[cfg(all(feature = "dx12", windows))]
use crate::dx12::{Dx12AccelerationStructure, Dx12TlasBuild};
#[cfg(feature = "vulkan")]
use crate::vulkan::{BuiltAcceleration, VulkanTlasBuild};
use crate::{Aabb, AccelerationStructureBuildReport, Backend, Device, RhiError};

const MAX_PACKED_INSTANCE_VALUE: u32 = 0x00ff_ffff;

/// Which of the two levels an acceleration structure is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccelerationStructureKind {
    /// Geometry: triangles, or boxes for an intersection shader.
    BottomLevel,
    /// Instances of bottom-level structures: what rays are traced against.
    TopLevel,
}

/// Whether a top-level structure can later be refit rather than rebuilt.
///
/// A refit keeps the tree the build produced and only recomputes bounding
/// boxes from new instance transforms. It is far cheaper than a build, and it
/// is the difference between an entity that moves every frame costing a
/// whole-world rebuild and costing almost nothing.
///
/// `Updatable` is opt-in because it has a cost: `ALLOW_UPDATE` makes the
/// structure larger and slightly slower to trace, because the builder
/// cannot use layouts that assume the tree is final, and the refit scratch
/// buffer must stay alive for the structure's lifetime. Each queued build
/// or refit owns its input snapshot. A structure that is built once and
/// never moves should stay `Static`.
///
/// It combines with the fast-trace preference rather than replacing it; the
/// pair that is mutually exclusive is fast-trace against fast-build, which
/// this abstraction does not expose.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccelerationStructureUsage {
    /// Built once. Fastest to trace.
    #[default]
    Static,
    /// Built so its instances can be moved afterwards without a rebuild.
    Updatable,
}

/// The triangles of one bottom-level acceleration structure.
///
/// Rays hit them without an intersection shader: a hit group for them has
/// none ([`crate::HitGroup::intersection`] is `None`), and `PrimitiveIndex()`
/// is the triangle's place in this order.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TriangleGeometry<'a> {
    /// The corners of the triangles.
    pub positions: &'a [[f32; 3]],
    /// Three to a triangle, each the place of a position. `None` takes the
    /// positions three at a time.
    pub indices: Option<&'a [u32]>,
    /// Whether every hit counts without an any-hit shader being asked.
    /// Faster to trace; turn it off for cut-outs.
    pub opaque: bool,
}

impl<'a> TriangleGeometry<'a> {
    /// Opaque triangles of `positions` taken three at a time.
    #[must_use]
    pub const fn new(positions: &'a [[f32; 3]]) -> Self {
        Self {
            positions,
            indices: None,
            opaque: true,
        }
    }

    /// The same positions joined into triangles by `indices`, three to a
    /// triangle.
    #[must_use]
    pub const fn with_indices(mut self, indices: &'a [u32]) -> Self {
        self.indices = Some(indices);
        self
    }

    /// The same triangles, opaque or not: where they are not, an any-hit
    /// shader is asked of every hit.
    #[must_use]
    pub const fn with_opaque(mut self, opaque: bool) -> Self {
        self.opaque = opaque;
        self
    }
}

impl TriangleGeometry<'_> {
    /// How many triangles this is, once it is known to be whole ones.
    pub(crate) fn validate(&self) -> Result<u32, RhiError> {
        let invalid = |reason| Err(RhiError::InvalidAccelerationStructureInstance(reason));
        if self.positions.is_empty() {
            return invalid("a triangle BLAS needs at least one triangle");
        }
        if !self
            .positions
            .iter()
            .flatten()
            .all(|value| value.is_finite())
        {
            return invalid("triangle positions must be finite");
        }
        let corners = match self.indices {
            Some(indices) => {
                if indices
                    .iter()
                    .any(|index| *index as usize >= self.positions.len())
                {
                    return invalid("a triangle index names a position that is not there");
                }
                indices.len()
            }
            None => self.positions.len(),
        };
        if corners == 0 || !corners.is_multiple_of(3) {
            return invalid("triangles take their corners three at a time");
        }
        u32::try_from(corners / 3).or(invalid("a triangle BLAS holds at most 2^32 triangles"))
    }
}

/// One TLAS instance referencing an owned BLAS.
///
/// `transform` is a row-major 3x4 affine transform, matching both Vulkan KHR
/// and DXR instance records. Custom and shader-table indices are 24-bit values
/// in both native APIs and are validated before command recording.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct AccelerationStructureInstance<'a> {
    /// The bottom-level structure placed.
    pub acceleration_structure: &'a AccelerationStructure,
    /// Where it is placed: the three rows of a three-by-four matrix, the last
    /// column the translation.
    pub transform: [f32; 12],
    /// A number of your choosing that hit shaders read back (`InstanceID()`):
    /// twenty-four bits.
    pub custom_index: u32,
    /// The instance is hit only by rays whose own mask shares a bit with this.
    pub mask: u8,
    /// Which of the pipeline's hit groups the instance's geometry runs: added
    /// to the offset the trace call gives.
    pub shader_binding_table_record_offset: u32,
}

impl<'a> AccelerationStructureInstance<'a> {
    /// An instance of `acceleration_structure` where it was built, hit by
    /// every ray, running the first hit group.
    #[must_use]
    pub const fn new(acceleration_structure: &'a AccelerationStructure) -> Self {
        Self {
            acceleration_structure,
            transform: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            custom_index: 0,
            mask: u8::MAX,
            shader_binding_table_record_offset: 0,
        }
    }
}

/// A ray-tracing acceleration structure.
///
/// Clones share the native object. A TLAS retains clones of all referenced
/// BLAS objects, preventing their native storage from being destroyed while
/// instance addresses remain reachable by ray dispatches.
#[derive(Clone)]
pub struct AccelerationStructure {
    inner: Arc<AccelerationStructureInner>,
}

struct AccelerationStructureInner {
    device_id: u64,
    kind: AccelerationStructureKind,
    usage: AccelerationStructureUsage,
    report: AccelerationStructureBuildReport,
    backend: AccelerationStructureBackend,
    dependencies: Vec<AccelerationStructure>,
    /// Referenced BLAS device addresses in instance order, as built.
    ///
    /// A refit may move instances but must not repoint them: `dependencies`
    /// retains exactly the structures the build named, so an instance that
    /// swapped its geometry would leave the TLAS reaching a BLAS nothing here
    /// keeps alive. Comparing these per refit is a few thousand `u64`s
    /// against the millisecond a build would cost.
    instance_blas: Vec<u64>,
}

pub(crate) enum AccelerationStructureBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<BuiltAcceleration>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(Box<Dx12AccelerationStructure>),
}

#[derive(Clone)]
pub(crate) struct AccelerationBuild {
    pub acceleration: AccelerationStructure,
    pub(crate) backend: AccelerationBuildBackend,
}

#[derive(Clone)]
pub(crate) enum AccelerationBuildBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(std::sync::Arc<VulkanTlasBuild>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(std::sync::Arc<Dx12TlasBuild>),
}

/// A batch of procedural BLAS builds and the structures they produce; a
/// frame retains it until its fence so the inputs outlive the recording.
#[derive(Clone)]
pub(crate) struct BlasBatchBuild {
    pub(crate) accelerations: Vec<AccelerationStructure>,
    pub(crate) backend: BlasBatchBackend,
}

#[derive(Clone)]
pub(crate) enum BlasBatchBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(std::sync::Arc<Vec<crate::vulkan::VulkanBlasBuild>>),
    #[cfg(all(feature = "dx12", windows))]
    Dx12(std::sync::Arc<Vec<crate::dx12::Dx12BlasBuild>>),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NativeAccelerationStructureInstance {
    #[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
    pub transform: [f32; 12],
    #[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
    pub custom_index: u32,
    #[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
    pub mask: u8,
    #[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
    pub shader_binding_table_record_offset: u32,
    pub blas_address: u64,
}

impl fmt::Debug for AccelerationStructure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AccelerationStructure")
            .field("backend", &self.backend())
            .field("kind", &self.kind())
            .field("report", &self.report())
            .field("dependency_count", &self.inner.dependencies.len())
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Build one opaque procedural bottom-level acceleration structure over
    /// `aabbs`. The AABB order is the shader's `PrimitiveIndex()` order.
    pub fn build_procedural_aabb_blas(
        &self,
        aabbs: &[Aabb],
    ) -> Result<AccelerationStructure, RhiError> {
        self.build_procedural_aabb_blas_batch(&[aabbs])?
            .pop()
            .ok_or(RhiError::InvalidAccelerationStructureInstance(
                "a procedural BLAS needs at least one AABB",
            ))
    }

    /// Build one procedural BLAS per AABB set with a single submission and
    /// a single wait, returned in input order. An empty slice builds
    /// nothing. Streaming scenes should record the builds into a frame with
    /// [`crate::Frame::build_procedural_aabb_blas_batch`] instead, which
    /// waits for nothing.
    pub fn build_procedural_aabb_blas_batch(
        &self,
        sets: &[&[Aabb]],
    ) -> Result<Vec<AccelerationStructure>, RhiError> {
        let build = self.prepare_blas_batch(sets)?;
        if build.accelerations.is_empty() {
            return Ok(Vec::new());
        }
        let mut encoder = self.begin_commands()?;
        self.record_blas_batch(&mut encoder, &build)?;
        encoder.finish()?;
        Ok(build.accelerations)
    }

    /// Build one bottom-level acceleration structure of triangles.
    pub fn build_triangle_blas(
        &self,
        geometry: &TriangleGeometry<'_>,
    ) -> Result<AccelerationStructure, RhiError> {
        self.build_triangle_blas_batch(std::slice::from_ref(geometry))?
            .pop()
            .ok_or(RhiError::InvalidAccelerationStructureInstance(
                "a triangle BLAS needs at least one triangle",
            ))
    }

    /// Build one triangle BLAS per geometry with a single submission and a
    /// single wait, returned in input order. Streaming scenes should record
    /// the builds into a frame with
    /// [`crate::Frame::build_triangle_blas_batch`] instead, which waits for
    /// nothing.
    pub fn build_triangle_blas_batch(
        &self,
        geometries: &[TriangleGeometry<'_>],
    ) -> Result<Vec<AccelerationStructure>, RhiError> {
        let build = self.prepare_triangle_blas_batch(geometries)?;
        if build.accelerations.is_empty() {
            return Ok(Vec::new());
        }
        let mut encoder = self.begin_commands()?;
        self.record_blas_batch(&mut encoder, &build)?;
        encoder.finish()?;
        Ok(build.accelerations)
    }

    /// Validate and allocate every geometry's BLAS; the structures are
    /// usable once the returned build has been recorded and has executed.
    pub(crate) fn prepare_triangle_blas_batch(
        &self,
        geometries: &[TriangleGeometry<'_>],
    ) -> Result<BlasBatchBuild, RhiError> {
        if !self.capabilities().ray_tracing {
            return Err(RhiError::RayTracingUnsupported);
        }
        let mut triangles = Vec::with_capacity(geometries.len());
        for geometry in geometries {
            triangles.push(geometry.validate()?);
        }
        let wrap = |backend, report| {
            AccelerationStructure::new(
                self.id(),
                AccelerationStructureKind::BottomLevel,
                AccelerationStructureUsage::Static,
                report,
                backend,
                Vec::new(),
                Vec::new(),
            )
        };
        let mut accelerations = Vec::with_capacity(geometries.len());
        let backend = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => {
                let mut builds = Vec::with_capacity(geometries.len());
                for (geometry, triangles) in geometries.iter().zip(triangles) {
                    let (acceleration, report, build) =
                        device.prepare_triangle_blas(geometry, triangles)?;
                    accelerations.push(wrap(
                        AccelerationStructureBackend::Vulkan(Box::new(acceleration)),
                        report,
                    ));
                    builds.push(build);
                }
                BlasBatchBackend::Vulkan(std::sync::Arc::new(builds))
            }
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => {
                let mut builds = Vec::with_capacity(geometries.len());
                for (geometry, triangles) in geometries.iter().zip(triangles) {
                    let (acceleration, report, build) =
                        device.prepare_triangle_blas(geometry, triangles)?;
                    accelerations.push(wrap(
                        AccelerationStructureBackend::Dx12(Box::new(acceleration)),
                        report,
                    ));
                    builds.push(build);
                }
                BlasBatchBackend::Dx12(std::sync::Arc::new(builds))
            }
        };
        Ok(BlasBatchBuild {
            accelerations,
            backend,
        })
    }

    /// Validate and allocate every set's BLAS; the structures are usable
    /// once the returned build has been recorded and has executed.
    pub(crate) fn prepare_blas_batch(&self, sets: &[&[Aabb]]) -> Result<BlasBatchBuild, RhiError> {
        if !self.capabilities().ray_tracing {
            return Err(RhiError::RayTracingUnsupported);
        }
        let mut validated = Vec::with_capacity(sets.len());

        for aabbs in sets {
            if aabbs.is_empty() {
                return Err(RhiError::InvalidAccelerationStructureInstance(
                    "a procedural BLAS needs at least one AABB",
                ));
            }
            let mut set = Vec::with_capacity(aabbs.len());
            for bounds in *aabbs {
                set.push(Aabb::new(bounds.min, bounds.max)?);
            }
            validated.push(set);
        }
        let wrap = |backend, report| {
            AccelerationStructure::new(
                self.id(),
                AccelerationStructureKind::BottomLevel,
                AccelerationStructureUsage::Static,
                report,
                backend,
                Vec::new(),
                Vec::new(),
            )
        };
        let (accelerations, backend) = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => {
                let mut accelerations = Vec::with_capacity(validated.len());
                let mut builds = Vec::with_capacity(validated.len());
                for (acceleration, report, build) in
                    device.prepare_procedural_aabb_blas_batch(&validated)?
                {
                    accelerations.push(wrap(
                        AccelerationStructureBackend::Vulkan(Box::new(acceleration)),
                        report,
                    ));
                    builds.push(build);
                }
                (
                    accelerations,
                    BlasBatchBackend::Vulkan(std::sync::Arc::new(builds)),
                )
            }
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => {
                let mut accelerations = Vec::with_capacity(validated.len());
                let mut builds = Vec::with_capacity(validated.len());
                for (acceleration, report, build) in
                    device.prepare_procedural_aabb_blas_batch(&validated)?
                {
                    accelerations.push(wrap(
                        AccelerationStructureBackend::Dx12(Box::new(acceleration)),
                        report,
                    ));
                    builds.push(build);
                }
                (
                    accelerations,
                    BlasBatchBackend::Dx12(std::sync::Arc::new(builds)),
                )
            }
        };
        Ok(BlasBatchBuild {
            accelerations,
            backend,
        })
    }

    pub(crate) fn record_blas_batch(
        &self,
        encoder: &mut CommandEncoder<'_>,
        build: &BlasBatchBuild,
    ) -> Result<(), RhiError> {
        encoder.require_no_render_pass(
            "acceleration structure builds are invalid inside a render pass",
        )?;
        if encoder.device_id() != self.id()
            || build
                .accelerations
                .iter()
                .any(|acceleration| acceleration.inner.device_id != self.id())
        {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        encoder
            .commands
            .push(crate::render_commands::Command::BlasBatch {
                build: build.clone(),
            });
        Ok(())
    }

    /// Build a TLAS and retain every BLAS referenced by its instances.
    ///
    /// The result cannot be refit. Use
    /// [`Self::build_updatable_tlas`] when the
    /// instances will move.
    pub fn build_tlas(
        &self,
        instances: &[AccelerationStructureInstance<'_>],
    ) -> Result<AccelerationStructure, RhiError> {
        self.build_tlas_for(AccelerationStructureUsage::Static, instances)
    }

    /// Build a TLAS that [`Self::update_tlas`]
    /// can later refit in place.
    pub fn build_updatable_tlas(
        &self,
        instances: &[AccelerationStructureInstance<'_>],
    ) -> Result<AccelerationStructure, RhiError> {
        self.build_tlas_for(AccelerationStructureUsage::Updatable, instances)
    }

    /// Refit `tlas` in place against new instance transforms.
    ///
    /// The instance array must name the same geometry in the same order as
    /// the build did: a refit reuses that build's tree, so the transforms,
    /// masks and custom indices may change but the shape may not. Anything
    /// else is a rebuild, and is rejected here rather than silently
    /// producing a structure that traces against geometry nothing retains.
    ///
    /// Submits on the graphics queue and waits. Queue barriers order previous
    /// traces before the write; the input snapshot is private to this call. Use
    /// [`crate::Frame::update_tlas`] to queue a refit
    /// as part of a frame without a CPU wait.
    pub fn update_tlas(
        &self,
        tlas: &AccelerationStructure,
        instances: &[AccelerationStructureInstance<'_>],
    ) -> Result<(), RhiError> {
        let build = self.prepare_tlas_update(tlas, instances)?;
        let mut encoder = self.begin_commands()?;
        self.record_tlas(&mut encoder, &build)?;
        encoder.finish()
    }

    pub(crate) fn prepare_tlas_update(
        &self,
        tlas: &AccelerationStructure,
        instances: &[AccelerationStructureInstance<'_>],
    ) -> Result<AccelerationBuild, RhiError> {
        if !self.capabilities().ray_tracing {
            return Err(RhiError::RayTracingUnsupported);
        }
        if tlas.inner.device_id != self.id() {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        if tlas.kind() != AccelerationStructureKind::TopLevel {
            return Err(RhiError::InvalidAccelerationStructureInstance(
                "only a top-level acceleration structure can be refit",
            ));
        }
        if tlas.inner.usage != AccelerationStructureUsage::Updatable {
            return Err(RhiError::AccelerationStructureNotUpdatable);
        }
        if instances.len() != tlas.inner.instance_blas.len() {
            return Err(RhiError::AccelerationStructureShapeChanged(
                "the number of instances",
            ));
        }
        let native_instances = self.native_instances(instances)?;
        if native_instances
            .iter()
            .zip(&tlas.inner.instance_blas)
            .any(|(instance, built)| instance.blas_address != *built)
        {
            return Err(RhiError::AccelerationStructureShapeChanged(
                "which geometry an instance references",
            ));
        }

        let backend: AccelerationBuildBackend = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => match tlas.native() {
                AccelerationStructureBackend::Vulkan(acceleration) => {
                    AccelerationBuildBackend::Vulkan(std::sync::Arc::new(
                        device.prepare_tlas_update(acceleration, &native_instances)?,
                    ))
                }
                #[cfg(all(feature = "dx12", windows))]
                AccelerationStructureBackend::Dx12(_) => {
                    return Err(RhiError::ResourceBackendMismatch);
                }
            },
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => match tlas.native() {
                AccelerationStructureBackend::Dx12(acceleration) => {
                    AccelerationBuildBackend::Dx12(std::sync::Arc::new(
                        device.prepare_tlas_update(acceleration, &native_instances)?,
                    ))
                }
                #[cfg(feature = "vulkan")]
                AccelerationStructureBackend::Vulkan(_) => {
                    return Err(RhiError::ResourceBackendMismatch);
                }
            },
        };
        Ok(AccelerationBuild {
            acceleration: tlas.clone(),
            backend,
        })
    }

    fn build_tlas_for(
        &self,
        usage: AccelerationStructureUsage,
        instances: &[AccelerationStructureInstance<'_>],
    ) -> Result<AccelerationStructure, RhiError> {
        let build = self.prepare_tlas(usage, instances)?;
        let mut encoder = self.begin_commands()?;
        self.record_tlas(&mut encoder, &build)?;
        encoder.finish()?;
        Ok(build.acceleration)
    }

    pub(crate) fn prepare_tlas(
        &self,
        usage: AccelerationStructureUsage,
        instances: &[AccelerationStructureInstance<'_>],
    ) -> Result<AccelerationBuild, RhiError> {
        if !self.capabilities().ray_tracing {
            return Err(RhiError::RayTracingUnsupported);
        }
        if instances.is_empty() {
            return Err(RhiError::EmptyAccelerationStructureInstances);
        }
        if instances.len() as u64 > self.capabilities().max_instance_count {
            return Err(RhiError::TooManyInstances {
                count: instances.len(),
                maximum: self.capabilities().max_instance_count,
            });
        }
        let native_instances = self.native_instances(instances)?;
        let instance_blas: Vec<u64> = native_instances
            .iter()
            .map(|instance| instance.blas_address)
            .collect();
        let dependencies: Vec<AccelerationStructure> = instances
            .iter()
            .map(|instance| instance.acceleration_structure.clone())
            .collect();

        let (backend, report, build): (
            AccelerationStructureBackend,
            AccelerationStructureBuildReport,
            AccelerationBuildBackend,
        ) = match &self.backend {
            #[cfg(feature = "vulkan")]
            BackendDevice::Vulkan(device) => {
                let (acceleration, report, build) =
                    device.prepare_tlas(usage, &native_instances)?;
                (
                    AccelerationStructureBackend::Vulkan(Box::new(acceleration)),
                    report,
                    AccelerationBuildBackend::Vulkan(std::sync::Arc::new(build)),
                )
            }
            #[cfg(all(feature = "dx12", windows))]
            BackendDevice::Dx12(device) => {
                let (acceleration, report, build) =
                    device.prepare_tlas(usage, &native_instances)?;
                (
                    AccelerationStructureBackend::Dx12(Box::new(acceleration)),
                    report,
                    AccelerationBuildBackend::Dx12(std::sync::Arc::new(build)),
                )
            }
        };

        Ok(AccelerationBuild {
            acceleration: AccelerationStructure::new(
                self.id(),
                AccelerationStructureKind::TopLevel,
                usage,
                report,
                backend,
                dependencies,
                instance_blas,
            ),
            backend: build,
        })
    }

    pub(crate) fn record_tlas(
        &self,
        encoder: &mut CommandEncoder<'_>,
        build: &AccelerationBuild,
    ) -> Result<(), RhiError> {
        encoder.require_no_render_pass(
            "acceleration structure builds are invalid inside a render pass",
        )?;
        if encoder.device_id() != self.id() || build.acceleration.inner.device_id != self.id() {
            return Err(RhiError::ResourceDeviceMismatch);
        }
        encoder
            .commands
            .push(crate::render_commands::Command::Tlas {
                build: build.clone(),
            });
        Ok(())
    }

    /// Validate instances and convert them to the common instance record.
    fn native_instances(
        &self,
        instances: &[AccelerationStructureInstance<'_>],
    ) -> Result<Vec<NativeAccelerationStructureInstance>, RhiError> {
        let mut native_instances = Vec::with_capacity(instances.len());
        for instance in instances {
            let acceleration = instance.acceleration_structure;
            if acceleration.kind() != AccelerationStructureKind::BottomLevel {
                return Err(RhiError::InvalidAccelerationStructureInstance(
                    "instances must reference a bottom-level acceleration structure",
                ));
            }
            if acceleration.inner.device_id != self.id() {
                return Err(RhiError::ResourceDeviceMismatch);
            }
            if !instance.transform.iter().all(|value| value.is_finite()) {
                return Err(RhiError::InvalidAccelerationStructureInstance(
                    "transform contains a non-finite value",
                ));
            }
            if instance.custom_index > MAX_PACKED_INSTANCE_VALUE {
                return Err(RhiError::InvalidAccelerationStructureInstance(
                    "custom index exceeds 24 bits",
                ));
            }
            if instance.shader_binding_table_record_offset > MAX_PACKED_INSTANCE_VALUE {
                return Err(RhiError::InvalidAccelerationStructureInstance(
                    "shader binding table record offset exceeds 24 bits",
                ));
            }
            native_instances.push(NativeAccelerationStructureInstance {
                #[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
                transform: instance.transform,
                #[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
                custom_index: instance.custom_index,
                #[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
                mask: instance.mask,
                #[cfg(any(feature = "vulkan", all(feature = "dx12", windows)))]
                shader_binding_table_record_offset: instance.shader_binding_table_record_offset,
                blas_address: acceleration.device_address(),
            });
        }
        Ok(native_instances)
    }
}

impl AccelerationStructure {
    fn new(
        device_id: u64,
        kind: AccelerationStructureKind,
        usage: AccelerationStructureUsage,
        report: AccelerationStructureBuildReport,
        backend: AccelerationStructureBackend,
        dependencies: Vec<AccelerationStructure>,
        instance_blas: Vec<u64>,
    ) -> Self {
        Self {
            inner: Arc::new(AccelerationStructureInner {
                device_id,
                kind,
                usage,
                report,
                backend,
                dependencies,
                instance_blas,
            }),
        }
    }

    /// Whether this is a bottom-level or a top-level structure.
    #[must_use]
    pub fn kind(&self) -> AccelerationStructureKind {
        self.inner.kind
    }

    /// Whether this structure can be refit rather than rebuilt.
    #[must_use]
    pub fn usage(&self) -> AccelerationStructureUsage {
        self.inner.usage
    }

    /// The sizes and the address the build ended with.
    #[must_use]
    pub fn report(&self) -> AccelerationStructureBuildReport {
        self.inner.report
    }

    /// The structure's address on the device.
    #[must_use]
    pub fn device_address(&self) -> u64 {
        self.inner.report.device_address
    }

    pub(crate) fn device_id(&self) -> u64 {
        self.inner.device_id
    }

    pub(crate) fn native(&self) -> &AccelerationStructureBackend {
        &self.inner.backend
    }

    /// The backend of the device that made this.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match self.inner.backend {
            #[cfg(feature = "vulkan")]
            AccelerationStructureBackend::Vulkan(ref acceleration) => {
                let _ = acceleration;
                Backend::Vulkan
            }
            #[cfg(all(feature = "dx12", windows))]
            AccelerationStructureBackend::Dx12(ref acceleration) => {
                let _ = acceleration;
                Backend::Dx12
            }
        }
    }
}

#[cfg(test)]
#[path = "tests/acceleration_structure.rs"]
mod tests;
