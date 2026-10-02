//! R3 renderer-visible representation contracts and narrow semantic query protocols.
//!
//! A representation is not classified by a closed `Mesh | Sdf | Volume | ...` root enum. It is a
//! renderer-owned record with intrinsic coverage/refinement evidence and the small typed protocols
//! it supports. Request-relative applicability, current availability/residency, dispatch strategy,
//! and physical realization remain later concerns.

use super::field_input::RenderFieldSemanticInputRequirement;
use super::space_time::{
    CanonicalF64, RenderAffineTransform3, RenderSemanticValueError, RenderSpatialCoverage,
    RenderTemporalSupport, RenderTimePoint,
};
use super::surface_input::RenderSurfaceSemanticInputRequirement;
use std::error::Error;
use std::fmt;
use std::num::NonZeroU64;

pub const RENDER_SURFACE_QUERY_PROTOCOL_REVISION: u32 = 1;
pub const RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION: u32 = 1;
pub const RENDER_FIELD_DISTANCE_PROTOCOL_REVISION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderRepresentationId(NonZeroU64);

impl RenderRepresentationId {
    pub(crate) fn from_raw(raw: u64) -> Option<Self> {
        NonZeroU64::new(raw).map(Self)
    }

    #[cfg(test)]
    pub(crate) const fn raw(self) -> u64 {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderRepresentationProtocol {
    SurfaceQuery,
    OrientedSurfaceQuery,
    FieldDistance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderProtocolCompatibilityError {
    Unsupported {
        protocol: RenderRepresentationProtocol,
    },
    VersionMismatch {
        protocol: RenderRepresentationProtocol,
        requested_revision: u32,
        supported_revision: u32,
    },
}

impl fmt::Display for RenderProtocolCompatibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { protocol } => {
                write!(f, "representation does not support {protocol:?}")
            }
            Self::VersionMismatch {
                protocol,
                requested_revision,
                supported_revision,
            } => write!(
                f,
                "{protocol:?} protocol revision {requested_revision} is incompatible with supported revision {supported_revision}"
            ),
        }
    }
}

impl Error for RenderProtocolCompatibilityError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderRepresentationValidationError {
    SemanticValue(RenderSemanticValueError),
    NoProtocols,
    InvalidProtocolRevision,
    NegativeErrorBound,
    ZeroQueryDirection,
    NegativeSurfaceHitDistance,
    ZeroSurfaceNormal,
    ExactFieldSampleHasError,
    FieldSampleExceedsDeclaredError,
}

impl From<RenderSemanticValueError> for RenderRepresentationValidationError {
    fn from(value: RenderSemanticValueError) -> Self {
        Self::SemanticValue(value)
    }
}

impl fmt::Display for RenderRepresentationValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SemanticValue(error) => fmt::Display::fmt(error, f),
            Self::NoProtocols => {
                write!(f, "representation must support at least one typed protocol")
            }
            Self::InvalidProtocolRevision => write!(f, "protocol revision must be non-zero"),
            Self::NegativeErrorBound => write!(f, "semantic error bound must be non-negative"),
            Self::ZeroQueryDirection => write!(f, "surface-query direction must be non-zero"),
            Self::NegativeSurfaceHitDistance => {
                write!(f, "surface-hit distance must be non-negative")
            }
            Self::ZeroSurfaceNormal => {
                write!(f, "oriented surface geometric normal must be non-zero")
            }
            Self::ExactFieldSampleHasError => {
                write!(
                    f,
                    "exact field-distance protocol requires zero sample error"
                )
            }
            Self::FieldSampleExceedsDeclaredError => {
                write!(
                    f,
                    "field-distance sample error exceeds intrinsic declared bound"
                )
            }
        }
    }
}

impl Error for RenderRepresentationValidationError {}

/// Exact oriented-surface query evidence.
///
/// The oriented protocol refines exact surface-query semantics with one representation-defined
/// geometric front-side normal in canonical scene coordinates. Because every oriented result
/// contains a valid exact surface hit, oriented capability is advertised through the corresponding
/// surface evidence rather than as an unrelated representation-family field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderOrientedSurfaceProtocolEvidence {
    revision: u32,
}

impl RenderOrientedSurfaceProtocolEvidence {
    pub fn exact(revision: u32) -> Result<Self, RenderRepresentationValidationError> {
        validate_protocol_revision(revision)?;
        Ok(Self { revision })
    }

    pub const fn revision(self) -> u32 {
        self.revision
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderSurfaceProtocolEvidence {
    revision: u32,
    oriented_surface: Option<RenderOrientedSurfaceProtocolEvidence>,
    semantic_input_requirement: Option<RenderSurfaceSemanticInputRequirement>,
}

impl RenderSurfaceProtocolEvidence {
    pub fn exact(revision: u32) -> Result<Self, RenderRepresentationValidationError> {
        validate_protocol_revision(revision)?;
        Ok(Self {
            revision,
            oriented_surface: None,
            semantic_input_requirement: None,
        })
    }

    pub const fn with_oriented_surface(
        mut self,
        evidence: RenderOrientedSurfaceProtocolEvidence,
    ) -> Self {
        self.oriented_surface = Some(evidence);
        self
    }

    /// Declare that this concrete representation's surface protocol requires one current
    /// request-scoped semantic surface value before use.
    pub const fn with_semantic_input_requirement(
        mut self,
        requirement: RenderSurfaceSemanticInputRequirement,
    ) -> Self {
        self.semantic_input_requirement = Some(requirement);
        self
    }

    pub const fn revision(self) -> u32 {
        self.revision
    }

    pub const fn oriented_surface(self) -> Option<RenderOrientedSurfaceProtocolEvidence> {
        self.oriented_surface
    }

    pub const fn semantic_input_requirement(self) -> Option<RenderSurfaceSemanticInputRequirement> {
        self.semantic_input_requirement
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderFieldDistanceGuarantee {
    kind: RenderFieldDistanceGuaranteeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RenderFieldDistanceGuaranteeKind {
    Exact,
    Conservative {
        max_absolute_error_meters: CanonicalF64,
    },
}

impl RenderFieldDistanceGuarantee {
    pub const fn exact() -> Self {
        Self {
            kind: RenderFieldDistanceGuaranteeKind::Exact,
        }
    }

    pub fn conservative(
        max_absolute_error_meters: f64,
    ) -> Result<Self, RenderRepresentationValidationError> {
        let max_absolute_error_meters =
            CanonicalF64::new(max_absolute_error_meters, "field_max_absolute_error_meters")?;
        if max_absolute_error_meters.get() < 0.0 {
            return Err(RenderRepresentationValidationError::NegativeErrorBound);
        }
        Ok(Self {
            kind: RenderFieldDistanceGuaranteeKind::Conservative {
                max_absolute_error_meters,
            },
        })
    }

    pub const fn is_exact(self) -> bool {
        matches!(self.kind, RenderFieldDistanceGuaranteeKind::Exact)
    }

    pub fn max_absolute_error_meters(self) -> f64 {
        match self.kind {
            RenderFieldDistanceGuaranteeKind::Exact => 0.0,
            RenderFieldDistanceGuaranteeKind::Conservative {
                max_absolute_error_meters,
            } => max_absolute_error_meters.get(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderRefinementEvidence {
    kind: RenderRefinementEvidenceKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RenderRefinementEvidenceKind {
    None,
    Bounded {
        finest_absolute_error_meters: CanonicalF64,
    },
}

impl RenderRefinementEvidence {
    pub const fn none() -> Self {
        Self {
            kind: RenderRefinementEvidenceKind::None,
        }
    }

    pub fn bounded(
        finest_absolute_error_meters: f64,
    ) -> Result<Self, RenderRepresentationValidationError> {
        let finest_absolute_error_meters =
            CanonicalF64::new(finest_absolute_error_meters, "finest_absolute_error_meters")?;
        if finest_absolute_error_meters.get() < 0.0 {
            return Err(RenderRepresentationValidationError::NegativeErrorBound);
        }
        Ok(Self {
            kind: RenderRefinementEvidenceKind::Bounded {
                finest_absolute_error_meters,
            },
        })
    }

    pub fn finest_absolute_error_meters(self) -> Option<f64> {
        match self.kind {
            RenderRefinementEvidenceKind::None => None,
            RenderRefinementEvidenceKind::Bounded {
                finest_absolute_error_meters,
            } => Some(finest_absolute_error_meters.get()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderFieldDistanceProtocolEvidence {
    revision: u32,
    guarantee: RenderFieldDistanceGuarantee,
    semantic_input_requirement: Option<RenderFieldSemanticInputRequirement>,
}

impl RenderFieldDistanceProtocolEvidence {
    pub fn new(
        revision: u32,
        guarantee: RenderFieldDistanceGuarantee,
    ) -> Result<Self, RenderRepresentationValidationError> {
        validate_protocol_revision(revision)?;
        Ok(Self {
            revision,
            guarantee,
            semantic_input_requirement: None,
        })
    }

    pub const fn revision(self) -> u32 {
        self.revision
    }

    pub const fn guarantee(self) -> RenderFieldDistanceGuarantee {
        self.guarantee
    }

    pub const fn with_semantic_input_requirement(
        mut self,
        requirement: RenderFieldSemanticInputRequirement,
    ) -> Self {
        self.semantic_input_requirement = Some(requirement);
        self
    }

    pub const fn semantic_input_requirement(self) -> Option<RenderFieldSemanticInputRequirement> {
        self.semantic_input_requirement
    }

    pub fn validate_sample(
        self,
        sample: RenderFieldDistanceSample,
    ) -> Result<(), RenderRepresentationValidationError> {
        let sample_error = sample.max_absolute_error_meters();
        if self.guarantee.is_exact() && sample_error != 0.0 {
            return Err(RenderRepresentationValidationError::ExactFieldSampleHasError);
        }
        if sample_error > self.guarantee.max_absolute_error_meters() {
            return Err(RenderRepresentationValidationError::FieldSampleExceedsDeclaredError);
        }
        Ok(())
    }
}

fn validate_protocol_revision(revision: u32) -> Result<(), RenderRepresentationValidationError> {
    if revision == 0 {
        return Err(RenderRepresentationValidationError::InvalidProtocolRevision);
    }
    Ok(())
}

/// One renderer-visible representation record.
///
/// The record carries only intrinsic semantics plus protocol-local declarations of any current
/// request-scoped semantic value required to realize those semantics. It deliberately has no
/// availability, residency, physical resource, request-applicability, method, or execution fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderRepresentationRecord {
    id: RenderRepresentationId,
    spatial_coverage: RenderSpatialCoverage,
    temporal_support: RenderTemporalSupport,
    refinement: RenderRefinementEvidence,
    surface_query: Option<RenderSurfaceProtocolEvidence>,
    field_distance: Option<RenderFieldDistanceProtocolEvidence>,
}

impl RenderRepresentationRecord {
    pub fn new(
        id: RenderRepresentationId,
        spatial_coverage: RenderSpatialCoverage,
        temporal_support: RenderTemporalSupport,
        refinement: RenderRefinementEvidence,
        surface_query: Option<RenderSurfaceProtocolEvidence>,
        field_distance: Option<RenderFieldDistanceProtocolEvidence>,
    ) -> Result<Self, RenderRepresentationValidationError> {
        if surface_query.is_none() && field_distance.is_none() {
            return Err(RenderRepresentationValidationError::NoProtocols);
        }
        Ok(Self {
            id,
            spatial_coverage,
            temporal_support,
            refinement,
            surface_query,
            field_distance,
        })
    }

    pub const fn id(&self) -> RenderRepresentationId {
        self.id
    }

    pub const fn spatial_coverage(&self) -> &RenderSpatialCoverage {
        &self.spatial_coverage
    }

    pub const fn temporal_support(&self) -> RenderTemporalSupport {
        self.temporal_support
    }

    pub const fn refinement(&self) -> RenderRefinementEvidence {
        self.refinement
    }

    pub const fn supports_field_distance(&self) -> bool {
        self.field_distance.is_some()
    }

    /// Return the request-scoped surface semantic-input prerequisite declared by this concrete
    /// representation, if any. Surface and oriented-surface protocol uses share this declaration;
    /// field-distance use remains independent.
    pub const fn surface_semantic_input_requirement(
        &self,
    ) -> Option<RenderSurfaceSemanticInputRequirement> {
        match self.surface_query {
            Some(evidence) => evidence.semantic_input_requirement(),
            None => None,
        }
    }

    pub const fn field_semantic_input_requirement(
        &self,
    ) -> Option<RenderFieldSemanticInputRequirement> {
        match self.field_distance {
            Some(evidence) => evidence.semantic_input_requirement(),
            None => None,
        }
    }

    pub fn surface_query_protocol(
        &self,
        requested_revision: u32,
    ) -> Result<RenderSurfaceProtocolEvidence, RenderProtocolCompatibilityError> {
        let evidence = self
            .surface_query
            .ok_or(RenderProtocolCompatibilityError::Unsupported {
                protocol: RenderRepresentationProtocol::SurfaceQuery,
            })?;
        require_revision(
            RenderRepresentationProtocol::SurfaceQuery,
            requested_revision,
            evidence.revision(),
        )?;
        Ok(evidence)
    }

    pub fn oriented_surface_query_protocol(
        &self,
        requested_revision: u32,
    ) -> Result<RenderOrientedSurfaceProtocolEvidence, RenderProtocolCompatibilityError> {
        let evidence = self
            .surface_query
            .and_then(RenderSurfaceProtocolEvidence::oriented_surface)
            .ok_or(RenderProtocolCompatibilityError::Unsupported {
                protocol: RenderRepresentationProtocol::OrientedSurfaceQuery,
            })?;
        require_revision(
            RenderRepresentationProtocol::OrientedSurfaceQuery,
            requested_revision,
            evidence.revision(),
        )?;
        Ok(evidence)
    }

    pub fn field_distance_protocol(
        &self,
        requested_revision: u32,
    ) -> Result<RenderFieldDistanceProtocolEvidence, RenderProtocolCompatibilityError> {
        let evidence =
            self.field_distance
                .ok_or(RenderProtocolCompatibilityError::Unsupported {
                    protocol: RenderRepresentationProtocol::FieldDistance,
                })?;
        require_revision(
            RenderRepresentationProtocol::FieldDistance,
            requested_revision,
            evidence.revision(),
        )?;
        Ok(evidence)
    }
}

fn require_revision(
    protocol: RenderRepresentationProtocol,
    requested_revision: u32,
    supported_revision: u32,
) -> Result<(), RenderProtocolCompatibilityError> {
    if requested_revision != supported_revision {
        return Err(RenderProtocolCompatibilityError::VersionMismatch {
            protocol,
            requested_revision,
            supported_revision,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderSurfaceQuery {
    origin_scene_meters: [CanonicalF64; 3],
    direction_scene: [CanonicalF64; 3],
    time: RenderTimePoint,
}

impl RenderSurfaceQuery {
    pub fn new(
        origin_scene_meters: [f64; 3],
        direction_scene: [f64; 3],
        time: RenderTimePoint,
    ) -> Result<Self, RenderRepresentationValidationError> {
        Ok(Self {
            origin_scene_meters: canonical_point(origin_scene_meters, "surface_query_origin")?,
            direction_scene: canonical_unit_direction(direction_scene)?,
            time,
        })
    }

    pub fn origin_scene_meters(self) -> [f64; 3] {
        self.origin_scene_meters.map(CanonicalF64::get)
    }

    pub fn direction_scene(self) -> [f64; 3] {
        self.direction_scene.map(CanonicalF64::get)
    }

    pub const fn time(self) -> RenderTimePoint {
        self.time
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderFieldDistanceQuery {
    position_scene_meters: [CanonicalF64; 3],
    time: RenderTimePoint,
}

impl RenderFieldDistanceQuery {
    pub fn new(
        position_scene_meters: [f64; 3],
        time: RenderTimePoint,
    ) -> Result<Self, RenderRepresentationValidationError> {
        Ok(Self {
            position_scene_meters: canonical_point(
                position_scene_meters,
                "field_query_position_scene_meters",
            )?,
            time,
        })
    }

    pub fn position_scene_meters(self) -> [f64; 3] {
        self.position_scene_meters.map(CanonicalF64::get)
    }

    pub const fn time(self) -> RenderTimePoint {
        self.time
    }
}

/// A bounded signed-distance estimate.
///
/// The true signed distance is guaranteed to lie within `estimate ± max_absolute_error`. A safe
/// unsigned scene-space step lower bound is therefore `max(0, abs(estimate) - error)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderFieldDistanceSample {
    signed_distance_estimate_meters: CanonicalF64,
    max_absolute_error_meters: CanonicalF64,
}

impl RenderFieldDistanceSample {
    pub fn new(
        signed_distance_estimate_meters: f64,
        max_absolute_error_meters: f64,
    ) -> Result<Self, RenderRepresentationValidationError> {
        let signed_distance_estimate_meters = CanonicalF64::new(
            signed_distance_estimate_meters,
            "signed_distance_estimate_meters",
        )?;
        let max_absolute_error_meters = CanonicalF64::new(
            max_absolute_error_meters,
            "field_sample_max_absolute_error_meters",
        )?;
        if max_absolute_error_meters.get() < 0.0 {
            return Err(RenderRepresentationValidationError::NegativeErrorBound);
        }
        Ok(Self {
            signed_distance_estimate_meters,
            max_absolute_error_meters,
        })
    }

    pub fn signed_distance_estimate_meters(self) -> f64 {
        self.signed_distance_estimate_meters.get()
    }

    pub fn max_absolute_error_meters(self) -> f64 {
        self.max_absolute_error_meters.get()
    }

    pub fn safe_distance_lower_bound_meters(self) -> f64 {
        (self.signed_distance_estimate_meters.get().abs() - self.max_absolute_error_meters.get())
            .max(0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderFieldTransformClassification {
    kind: RenderFieldTransformKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RenderFieldTransformKind {
    Exact {
        distance_scale: CanonicalF64,
    },
    Conservative {
        distance_scale_lower_bound: CanonicalF64,
    },
    Invalid,
}

impl RenderFieldTransformClassification {
    fn exact(distance_scale: f64) -> Option<Self> {
        let distance_scale =
            CanonicalF64::new(distance_scale, "field_exact_distance_scale").ok()?;
        (distance_scale.get() > 0.0).then_some(Self {
            kind: RenderFieldTransformKind::Exact { distance_scale },
        })
    }

    fn conservative(distance_scale_lower_bound: f64) -> Option<Self> {
        let distance_scale_lower_bound = CanonicalF64::new(
            distance_scale_lower_bound,
            "field_conservative_distance_scale_lower_bound",
        )
        .ok()?;
        (distance_scale_lower_bound.get() > 0.0).then_some(Self {
            kind: RenderFieldTransformKind::Conservative {
                distance_scale_lower_bound,
            },
        })
    }

    pub const fn invalid() -> Self {
        Self {
            kind: RenderFieldTransformKind::Invalid,
        }
    }

    pub fn exact_distance_scale(self) -> Option<f64> {
        match self.kind {
            RenderFieldTransformKind::Exact { distance_scale } => Some(distance_scale.get()),
            _ => None,
        }
    }

    pub fn conservative_distance_scale_lower_bound(self) -> Option<f64> {
        match self.kind {
            RenderFieldTransformKind::Conservative {
                distance_scale_lower_bound,
            } => Some(distance_scale_lower_bound.get()),
            _ => None,
        }
    }

    pub const fn is_invalid(self) -> bool {
        matches!(self.kind, RenderFieldTransformKind::Invalid)
    }
}

/// Classify a renderer object transform for field/distance guarantees.
///
/// Similarity transforms preserve Euclidean distance up to one exact positive scale. Other
/// invertible affine transforms use the conservative lower bound `|det(A)| / ||A||_F^2`, expressed
/// in a scale-normalized form to avoid unnecessary overflow. This bound never exceeds the minimum
/// singular value and therefore cannot overestimate scene-space distance. Singular or numerically
/// unprovable transforms are invalid for the founding field-distance protocol.
pub fn classify_field_distance_transform(
    transform: RenderAffineTransform3,
) -> RenderFieldTransformClassification {
    let matrix = transform.row_major_3x4();
    let linear = [
        matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
        matrix[10],
    ];
    let scale = linear.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return RenderFieldTransformClassification::invalid();
    }

    let b = linear.map(|value| value / scale);
    let c0 = [b[0], b[3], b[6]];
    let c1 = [b[1], b[4], b[7]];
    let c2 = [b[2], b[5], b[8]];
    let n0 = dot(c0, c0);
    let n1 = dot(c1, c1);
    let n2 = dot(c2, c2);
    let d01 = dot(c0, c1);
    let d02 = dot(c0, c2);
    let d12 = dot(c1, c2);

    let determinant = b[0] * (b[4] * b[8] - b[5] * b[7]) - b[1] * (b[3] * b[8] - b[5] * b[6])
        + b[2] * (b[3] * b[7] - b[4] * b[6]);
    if determinant == 0.0 {
        return RenderFieldTransformClassification::invalid();
    }

    if n0 > 0.0 && n0 == n1 && n1 == n2 && d01 == 0.0 && d02 == 0.0 && d12 == 0.0 {
        let exact_scale = scale * n0.sqrt();
        return RenderFieldTransformClassification::exact(exact_scale)
            .unwrap_or_else(RenderFieldTransformClassification::invalid);
    }

    let frobenius_squared = b.iter().copied().map(|value| value * value).sum::<f64>();
    let lower_bound = scale * determinant.abs() / frobenius_squared;
    RenderFieldTransformClassification::conservative(lower_bound)
        .unwrap_or_else(RenderFieldTransformClassification::invalid)
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn canonical_point(
    point: [f64; 3],
    field: &'static str,
) -> Result<[CanonicalF64; 3], RenderRepresentationValidationError> {
    Ok([
        CanonicalF64::new(point[0], field)?,
        CanonicalF64::new(point[1], field)?,
        CanonicalF64::new(point[2], field)?,
    ])
}

fn canonical_unit_direction(
    direction: [f64; 3],
) -> Result<[CanonicalF64; 3], RenderRepresentationValidationError> {
    let values = [
        CanonicalF64::new(direction[0], "surface_query_direction")?.get(),
        CanonicalF64::new(direction[1], "surface_query_direction")?.get(),
        CanonicalF64::new(direction[2], "surface_query_direction")?.get(),
    ];
    let scale = values.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(RenderRepresentationValidationError::ZeroQueryDirection);
    }
    let scaled = values.map(|value| value / scale);
    let length = scaled
        .iter()
        .copied()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let normalized = scaled.map(|value| value / length);
    Ok([
        CanonicalF64::new(normalized[0], "surface_query_direction")?,
        CanonicalF64::new(normalized[1], "surface_query_direction")?,
        CanonicalF64::new(normalized[2], "surface_query_direction")?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space_time::{RenderTimeInterval, RenderTimePoint};
    use crate::surface_input::RenderSurfaceSemanticInputRequirement;

    fn instant() -> RenderTimePoint {
        RenderTimePoint::from_seconds(0.0).expect("finite time")
    }

    fn representation(
        id: u64,
        surface: Option<RenderSurfaceProtocolEvidence>,
        field: Option<RenderFieldDistanceProtocolEvidence>,
    ) -> RenderRepresentationRecord {
        RenderRepresentationRecord::new(
            RenderRepresentationId::from_raw(id).expect("non-zero representation id"),
            RenderSpatialCoverage::unbounded(),
            RenderTemporalSupport::unbounded(),
            RenderRefinementEvidence::none(),
            surface,
            field,
        )
        .expect("valid representation")
    }

    #[test]
    fn representation_requires_typed_protocol_without_family_enum() {
        let id = RenderRepresentationId::from_raw(1).expect("non-zero representation id");
        assert_eq!(
            RenderRepresentationRecord::new(
                id,
                RenderSpatialCoverage::unbounded(),
                RenderTemporalSupport::unbounded(),
                RenderRefinementEvidence::none(),
                None,
                None,
            ),
            Err(RenderRepresentationValidationError::NoProtocols)
        );
    }

    #[test]
    fn exact_analytic_surface_protocol_has_narrow_query_and_result() {
        use crate::surface_result::RenderSurfaceQueryResult;

        let record = representation(
            1,
            Some(
                RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
                    .expect("valid protocol"),
            ),
            None,
        );
        record
            .surface_query_protocol(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
            .expect("surface protocol should be supported");
        let query = RenderSurfaceQuery::new([0.0, 0.0, 3.0], [0.0, 0.0, -2.0], instant())
            .expect("valid query");

        // R3 protocol proof fixture: exact unit sphere at the scene origin. The fixture is deliberately
        // test-local; the public protocol is not named after or coupled to the founding primitive.
        let origin = query.origin_scene_meters();
        let direction = query.direction_scene();
        let b = origin[0] * direction[0] + origin[1] * direction[1] + origin[2] * direction[2];
        let c = origin[0] * origin[0] + origin[1] * origin[1] + origin[2] * origin[2] - 1.0;
        let distance = -b - (b * b - c).sqrt();
        let result = RenderSurfaceQueryResult::hit_at_distance(query, distance)
            .expect("valid exact hit result");
        let hit = result.hit().expect("surface should hit");
        assert_eq!(hit.distance_meters(), 2.0);
        assert_eq!(hit.position_scene_meters(), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn oriented_surface_protocol_refines_but_is_not_implied_by_plain_surface_query() {
        let plain = representation(
            1,
            Some(
                RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
                    .expect("plain surface protocol"),
            ),
            None,
        );
        assert_eq!(
            plain.oriented_surface_query_protocol(RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION),
            Err(RenderProtocolCompatibilityError::Unsupported {
                protocol: RenderRepresentationProtocol::OrientedSurfaceQuery,
            })
        );

        let oriented = RenderOrientedSurfaceProtocolEvidence::exact(
            RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
        )
        .expect("oriented surface protocol");
        let surface = RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
            .expect("surface protocol")
            .with_oriented_surface(oriented);
        let record = representation(2, Some(surface), None);
        assert_eq!(
            record.surface_query_protocol(RENDER_SURFACE_QUERY_PROTOCOL_REVISION),
            Ok(surface)
        );
        assert_eq!(
            record.oriented_surface_query_protocol(RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION),
            Ok(oriented)
        );
        assert_eq!(
            record.oriented_surface_query_protocol(
                RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION + 1,
            ),
            Err(RenderProtocolCompatibilityError::VersionMismatch {
                protocol: RenderRepresentationProtocol::OrientedSurfaceQuery,
                requested_revision: RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION + 1,
                supported_revision: RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
            })
        );
    }

    #[test]
    fn surface_input_prerequisite_is_representation_owned_and_optional() {
        let self_contained = representation(
            1,
            Some(
                RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
                    .expect("surface protocol"),
            ),
            None,
        );
        assert_eq!(self_contained.surface_semantic_input_requirement(), None);

        let requirement = RenderSurfaceSemanticInputRequirement::current();
        let surface = RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
            .expect("surface protocol")
            .with_oriented_surface(
                RenderOrientedSurfaceProtocolEvidence::exact(
                    RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
                )
                .expect("oriented protocol"),
            )
            .with_semantic_input_requirement(requirement);
        let externally_bound = representation(2, Some(surface), None);
        assert_eq!(
            externally_bound.surface_semantic_input_requirement(),
            Some(requirement)
        );

        let field = RenderFieldDistanceProtocolEvidence::new(
            RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
            RenderFieldDistanceGuarantee::exact(),
        )
        .expect("field protocol");
        assert_eq!(
            representation(3, None, Some(field)).surface_semantic_input_requirement(),
            None
        );
    }

    #[test]
    fn conservative_field_protocol_exposes_bounded_safe_distance_evidence() {
        let evidence = RenderFieldDistanceProtocolEvidence::new(
            RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
            RenderFieldDistanceGuarantee::conservative(0.25).expect("valid field guarantee"),
        )
        .expect("valid field protocol");
        let record = representation(1, None, Some(evidence));
        let supported = record
            .field_distance_protocol(RENDER_FIELD_DISTANCE_PROTOCOL_REVISION)
            .expect("field protocol should be supported");
        let query =
            RenderFieldDistanceQuery::new([0.0, 0.0, 2.0], instant()).expect("valid field query");
        assert_eq!(query.position_scene_meters(), [0.0, 0.0, 2.0]);

        let sample = RenderFieldDistanceSample::new(2.0, 0.25).expect("bounded sample");
        supported
            .validate_sample(sample)
            .expect("sample should satisfy intrinsic bound");
        assert_eq!(sample.safe_distance_lower_bound_meters(), 1.75);
    }

    #[test]
    fn protocol_version_mismatch_and_unsupported_are_structured() {
        let record = representation(
            1,
            Some(
                RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
                    .expect("valid protocol"),
            ),
            None,
        );
        assert_eq!(
            record.surface_query_protocol(RENDER_SURFACE_QUERY_PROTOCOL_REVISION + 1),
            Err(RenderProtocolCompatibilityError::VersionMismatch {
                protocol: RenderRepresentationProtocol::SurfaceQuery,
                requested_revision: RENDER_SURFACE_QUERY_PROTOCOL_REVISION + 1,
                supported_revision: RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            })
        );
        assert_eq!(
            record.field_distance_protocol(RENDER_FIELD_DISTANCE_PROTOCOL_REVISION),
            Err(RenderProtocolCompatibilityError::Unsupported {
                protocol: RenderRepresentationProtocol::FieldDistance,
            })
        );
    }

    #[test]
    fn refinement_and_intrinsic_coverage_are_representation_evidence() {
        let id = RenderRepresentationId::from_raw(1).expect("non-zero representation id");
        let interval = RenderTimeInterval::instant(instant());
        let record = RenderRepresentationRecord::new(
            id,
            RenderSpatialCoverage::axis_aligned_bounds([-1.0; 3], [1.0; 3])
                .expect("valid coverage"),
            RenderTemporalSupport::interval(interval),
            RenderRefinementEvidence::bounded(0.01).expect("valid refinement evidence"),
            Some(
                RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
                    .expect("valid protocol"),
            ),
            None,
        )
        .expect("valid representation");
        assert_eq!(
            record.refinement().finest_absolute_error_meters(),
            Some(0.01)
        );
        assert_eq!(record.temporal_support().bounded_interval(), Some(interval));
        assert_eq!(
            record.spatial_coverage().axis_aligned_bounds_value(),
            Some(([-1.0; 3], [1.0; 3]))
        );
    }

    #[test]
    fn field_transform_classifies_exact_conservative_and_invalid_semantics() {
        let exact = classify_field_distance_transform(
            RenderAffineTransform3::from_row_major_3x4([
                2.0, 0.0, 0.0, 5.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
            ])
            .expect("finite similarity transform"),
        );
        assert_eq!(exact.exact_distance_scale(), Some(2.0));

        let conservative = classify_field_distance_transform(
            RenderAffineTransform3::from_row_major_3x4([
                2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ])
            .expect("finite non-uniform transform"),
        );
        let lower_bound = conservative
            .conservative_distance_scale_lower_bound()
            .expect("non-uniform invertible transform should be conservative");
        assert!(lower_bound > 0.0);
        assert!(lower_bound <= 1.0);

        let invalid = classify_field_distance_transform(
            RenderAffineTransform3::from_row_major_3x4([
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ])
            .expect("finite singular transform"),
        );
        assert!(invalid.is_invalid());
    }
}
