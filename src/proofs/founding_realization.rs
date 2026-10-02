//! Proof-local concrete realization for the bounded founding renderer.
//!
//! This module deliberately exists only in the internal R6 proof. It does not create public
//! sphere/plane/field representation families or a provider/dispatch authority. Concrete values are
//! correlated by the existing R3 `RenderRepresentationId`, while semantic query/result meaning stays
//! owned by the permanent R2/R3 contracts.

use crate::derived_transform::{RenderCompiledObjectTransform, RenderCompiledObjectTransformError};
use crate::representation::{
    RenderFieldDistanceProtocolEvidence, RenderFieldDistanceQuery, RenderFieldDistanceSample,
    RenderRepresentationId, RenderRepresentationValidationError, RenderSurfaceQuery,
    classify_field_distance_transform,
};
use crate::scene::RenderObjectState;
use crate::space_time::{CanonicalF64, RenderSemanticValueError};
use crate::surface_result::RenderOrientedSurfaceQueryResult;
use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FoundingRepresentationRealization {
    representation_id: RenderRepresentationId,
    kind: FoundingRepresentationRealizationKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FoundingRepresentationRealizationKind {
    AnalyticSphere {
        center_local_units: [CanonicalF64; 3],
        radius_local_units: CanonicalF64,
    },
    AnalyticPlane {
        point_local_units: [CanonicalF64; 3],
        normal_local: [CanonicalF64; 3],
    },
    FieldSphere {
        center_local_units: [CanonicalF64; 3],
        radius_local_units: CanonicalF64,
    },
}

impl FoundingRepresentationRealization {
    pub(super) fn analytic_sphere(
        representation_id: RenderRepresentationId,
        center_local_units: [f64; 3],
        radius_local_units: f64,
    ) -> Result<Self, ProofRealizationError> {
        Ok(Self {
            representation_id,
            kind: FoundingRepresentationRealizationKind::AnalyticSphere {
                center_local_units: canonical_point(center_local_units, "r6_sphere_center")?,
                radius_local_units: positive_value(radius_local_units, "r6_sphere_radius")?,
            },
        })
    }

    pub(super) fn analytic_plane(
        representation_id: RenderRepresentationId,
        point_local_units: [f64; 3],
        normal_local: [f64; 3],
    ) -> Result<Self, ProofRealizationError> {
        Ok(Self {
            representation_id,
            kind: FoundingRepresentationRealizationKind::AnalyticPlane {
                point_local_units: canonical_point(point_local_units, "r6_plane_point")?,
                normal_local: canonical_unit_direction(normal_local, "r6_plane_normal")?,
            },
        })
    }

    pub(super) fn field_sphere(
        representation_id: RenderRepresentationId,
        center_local_units: [f64; 3],
        radius_local_units: f64,
    ) -> Result<Self, ProofRealizationError> {
        Ok(Self {
            representation_id,
            kind: FoundingRepresentationRealizationKind::FieldSphere {
                center_local_units: canonical_point(center_local_units, "r6_field_center")?,
                radius_local_units: positive_value(radius_local_units, "r6_field_radius")?,
            },
        })
    }

    pub(super) const fn representation_id(&self) -> RenderRepresentationId {
        self.representation_id
    }

    pub(super) fn oriented_surface_query(
        &self,
        object_state: &RenderObjectState,
        query: RenderSurfaceQuery,
    ) -> Result<RenderOrientedSurfaceQueryResult, ProofRealizationError> {
        let transform = RenderCompiledObjectTransform::compile(object_state.spatial())?;
        match &self.kind {
            FoundingRepresentationRealizationKind::AnalyticSphere {
                center_local_units,
                radius_local_units,
            }
            | FoundingRepresentationRealizationKind::FieldSphere {
                center_local_units,
                radius_local_units,
            } => sphere_surface_query(
                query,
                &transform,
                center_local_units.map(CanonicalF64::get),
                radius_local_units.get(),
            ),
            FoundingRepresentationRealizationKind::AnalyticPlane {
                point_local_units,
                normal_local,
            } => plane_surface_query(
                query,
                &transform,
                point_local_units.map(CanonicalF64::get),
                normal_local.map(CanonicalF64::get),
            ),
        }
    }

    fn field_distance_query(
        &self,
        object_state: &RenderObjectState,
        query: RenderFieldDistanceQuery,
    ) -> Result<RenderFieldDistanceSample, ProofRealizationError> {
        let FoundingRepresentationRealizationKind::FieldSphere {
            center_local_units,
            radius_local_units,
        } = &self.kind
        else {
            return Err(ProofRealizationError::ProtocolMismatch);
        };

        let transform = RenderCompiledObjectTransform::compile(object_state.spatial())?;
        let local_position = transform.local_point_from_scene(query.position_scene_meters());
        let center = center_local_units.map(CanonicalF64::get);
        let local_signed_distance = length(sub(local_position, center)) - radius_local_units.get();
        let classification =
            classify_field_distance_transform(object_state.spatial().local_to_scene());
        let Some(transform_distance_scale) = classification.exact_distance_scale() else {
            return Err(ProofRealizationError::FoundingFieldRequiresSimilarityTransform);
        };
        let scene_signed_distance = local_signed_distance
            * object_state.spatial().local_space().meters_per_unit()
            * transform_distance_scale;
        Ok(RenderFieldDistanceSample::new(scene_signed_distance, 0.0)?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ProofRealizationError {
    SemanticValue(RenderSemanticValueError),
    Representation(RenderRepresentationValidationError),
    NonPositiveExtent,
    ZeroDirection,
    NonInvertibleObjectTransform,
    FoundingFieldRequiresSimilarityTransform,
    ProtocolMismatch,
    DuplicateRealization(RenderRepresentationId),
    ForeignRealization(RenderRepresentationId),
    MissingRealization(RenderRepresentationId),
}

impl From<RenderSemanticValueError> for ProofRealizationError {
    fn from(value: RenderSemanticValueError) -> Self {
        Self::SemanticValue(value)
    }
}

impl From<RenderRepresentationValidationError> for ProofRealizationError {
    fn from(value: RenderRepresentationValidationError) -> Self {
        Self::Representation(value)
    }
}

impl From<RenderCompiledObjectTransformError> for ProofRealizationError {
    fn from(value: RenderCompiledObjectTransformError) -> Self {
        match value {
            RenderCompiledObjectTransformError::NonInvertibleObjectTransform => {
                Self::NonInvertibleObjectTransform
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FoundingRealizationSet {
    by_representation: BTreeMap<RenderRepresentationId, FoundingRepresentationRealization>,
}

impl FoundingRealizationSet {
    fn normalize(
        required: &BTreeSet<RenderRepresentationId>,
        inputs: Vec<FoundingRepresentationRealization>,
    ) -> Result<Self, ProofRealizationError> {
        let mut by_representation = BTreeMap::new();
        let mut duplicates = BTreeSet::new();
        for input in inputs {
            let representation_id = input.representation_id();
            match by_representation.entry(representation_id) {
                Entry::Vacant(entry) => {
                    entry.insert(input);
                }
                Entry::Occupied(_) => {
                    duplicates.insert(representation_id);
                }
            }
        }

        if let Some(representation_id) = duplicates.into_iter().next() {
            return Err(ProofRealizationError::DuplicateRealization(
                representation_id,
            ));
        }
        if let Some(representation_id) = by_representation
            .keys()
            .copied()
            .find(|representation_id| !required.contains(representation_id))
        {
            return Err(ProofRealizationError::ForeignRealization(representation_id));
        }
        if let Some(representation_id) = required
            .iter()
            .copied()
            .find(|representation_id| !by_representation.contains_key(representation_id))
        {
            return Err(ProofRealizationError::MissingRealization(representation_id));
        }

        Ok(Self { by_representation })
    }

    fn get(&self, representation_id: RenderRepresentationId) -> &FoundingRepresentationRealization {
        self.by_representation
            .get(&representation_id)
            .expect("normalized R6 proof realization must exist for each required identity")
    }
}

fn sphere_surface_query(
    query: RenderSurfaceQuery,
    transform: &RenderCompiledObjectTransform,
    center_local_units: [f64; 3],
    radius_local_units: f64,
) -> Result<RenderOrientedSurfaceQueryResult, ProofRealizationError> {
    let origin_local = transform.local_point_from_scene(query.origin_scene_meters());
    let direction_local = transform.local_direction_per_scene_meter(query.direction_scene());
    let relative_origin = sub(origin_local, center_local_units);
    let a = dot(direction_local, direction_local);
    let half_b = dot(relative_origin, direction_local);
    let c = dot(relative_origin, relative_origin) - radius_local_units * radius_local_units;
    let discriminant = half_b * half_b - a * c;
    if discriminant < 0.0 {
        return Ok(RenderOrientedSurfaceQueryResult::miss());
    }
    let root = discriminant.sqrt();
    let near = (-half_b - root) / a;
    let far = (-half_b + root) / a;
    let distance_meters = if near >= 0.0 {
        near
    } else if far >= 0.0 {
        far
    } else {
        return Ok(RenderOrientedSurfaceQueryResult::miss());
    };
    let local_hit = add(origin_local, scale(direction_local, distance_meters));
    let local_normal = normalize(sub(local_hit, center_local_units))
        .expect("positive sphere radius gives a non-zero hit normal");
    Ok(RenderOrientedSurfaceQueryResult::hit_at_distance(
        query,
        distance_meters,
        transform.scene_normal_from_local(local_normal),
    )?)
}

fn plane_surface_query(
    query: RenderSurfaceQuery,
    transform: &RenderCompiledObjectTransform,
    point_local_units: [f64; 3],
    normal_local: [f64; 3],
) -> Result<RenderOrientedSurfaceQueryResult, ProofRealizationError> {
    let point_scene = transform.scene_point_from_local(point_local_units);
    let normal_scene = transform.scene_normal_from_local(normal_local);
    let denominator = dot(normal_scene, query.direction_scene());
    if denominator.abs() <= f64::EPSILON {
        return Ok(RenderOrientedSurfaceQueryResult::miss());
    }
    let distance_meters =
        dot(sub(point_scene, query.origin_scene_meters()), normal_scene) / denominator;
    if distance_meters < 0.0 {
        return Ok(RenderOrientedSurfaceQueryResult::miss());
    }
    Ok(RenderOrientedSurfaceQueryResult::hit_at_distance(
        query,
        distance_meters,
        normal_scene,
    )?)
}

fn canonical_point(
    values: [f64; 3],
    field: &'static str,
) -> Result<[CanonicalF64; 3], ProofRealizationError> {
    Ok([
        CanonicalF64::new(values[0], field)?,
        CanonicalF64::new(values[1], field)?,
        CanonicalF64::new(values[2], field)?,
    ])
}

fn positive_value(value: f64, field: &'static str) -> Result<CanonicalF64, ProofRealizationError> {
    let value = CanonicalF64::new(value, field)?;
    if value.get() <= 0.0 {
        return Err(ProofRealizationError::NonPositiveExtent);
    }
    Ok(value)
}

fn canonical_unit_direction(
    direction: [f64; 3],
    field: &'static str,
) -> Result<[CanonicalF64; 3], ProofRealizationError> {
    let values = [
        CanonicalF64::new(direction[0], field)?.get(),
        CanonicalF64::new(direction[1], field)?.get(),
        CanonicalF64::new(direction[2], field)?.get(),
    ];
    let Some(normalized) = normalize(values) else {
        return Err(ProofRealizationError::ZeroDirection);
    };
    Ok([
        CanonicalF64::new(normalized[0], field)?,
        CanonicalF64::new(normalized[1], field)?,
        CanonicalF64::new(normalized[2], field)?,
    ])
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn length(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let magnitude = length(vector);
    (magnitude > 0.0 && magnitude.is_finite()).then(|| scale(vector, magnitude.recip()))
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::representation::{
        RENDER_FIELD_DISTANCE_PROTOCOL_REVISION, RenderFieldDistanceGuarantee,
    };
    use crate::space_time::{
        RenderAffineTransform3, RenderHandedness, RenderObjectSpatialState,
        RenderObjectTemporalState, RenderSpaceSpec, RenderSpatialCoverage, RenderTemporalSupport,
        RenderTimePoint,
    };

    fn representation_id(raw: u64) -> RenderRepresentationId {
        RenderRepresentationId::from_raw(raw).expect("non-zero proof representation id")
    }

    fn object_state(
        meters_per_unit: f64,
        local_to_scene: RenderAffineTransform3,
    ) -> RenderObjectState {
        RenderObjectState::new(
            RenderObjectSpatialState::new(
                RenderSpaceSpec::new(meters_per_unit, RenderHandedness::Right)
                    .expect("valid proof local space"),
                local_to_scene,
                RenderSpatialCoverage::unbounded(),
            ),
            RenderObjectTemporalState::new(RenderTemporalSupport::unbounded()),
        )
    }

    fn instant() -> RenderTimePoint {
        RenderTimePoint::from_seconds(0.0).expect("finite proof time")
    }

    #[test]
    fn realization_identity_normalization_is_deterministic_and_structural() {
        let first_id = representation_id(1);
        let second_id = representation_id(2);
        let foreign_id = representation_id(3);
        let first = FoundingRepresentationRealization::analytic_sphere(first_id, [0.0; 3], 1.0)
            .expect("sphere realization");
        let second =
            FoundingRepresentationRealization::analytic_plane(second_id, [0.0; 3], [0.0, 1.0, 0.0])
                .expect("plane realization");
        let foreign = FoundingRepresentationRealization::field_sphere(foreign_id, [0.0; 3], 1.0)
            .expect("field realization");
        let required = BTreeSet::from([first_id, second_id]);

        let ordered =
            FoundingRealizationSet::normalize(&required, vec![second.clone(), first.clone()])
                .expect("complete realization set");
        assert_eq!(ordered.get(first_id).representation_id(), first_id);
        assert_eq!(ordered.get(second_id).representation_id(), second_id);

        assert_eq!(
            FoundingRealizationSet::normalize(
                &required,
                vec![first.clone(), first, second.clone()]
            ),
            Err(ProofRealizationError::DuplicateRealization(first_id))
        );
        assert_eq!(
            FoundingRealizationSet::normalize(&required, vec![second.clone(), foreign]),
            Err(ProofRealizationError::ForeignRealization(foreign_id))
        );
        assert_eq!(
            FoundingRealizationSet::normalize(&required, vec![second]),
            Err(ProofRealizationError::MissingRealization(first_id))
        );
    }

    #[test]
    fn analytic_sphere_uses_r2_units_transform_and_r3_oriented_hit_semantics() {
        let realization =
            FoundingRepresentationRealization::analytic_sphere(representation_id(1), [0.0; 3], 1.0)
                .expect("sphere realization");
        let state = object_state(
            0.5,
            RenderAffineTransform3::from_row_major_3x4([
                2.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
            ])
            .expect("finite similarity transform"),
        );
        let query = RenderSurfaceQuery::new([1.0, 0.0, 3.0], [0.0, 0.0, -2.0], instant())
            .expect("valid surface query");
        let result = realization
            .oriented_surface_query(&state, query)
            .expect("exact sphere query");
        let hit = result.hit().expect("sphere hit");
        assert_eq!(hit.surface_hit().distance_meters(), 2.0);
        assert_eq!(hit.surface_hit().position_scene_meters(), [1.0, 0.0, 1.0]);
        assert_eq!(hit.geometric_normal_scene(), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn analytic_plane_transforms_representation_defined_orientation_without_view_flipping() {
        let realization = FoundingRepresentationRealization::analytic_plane(
            representation_id(2),
            [0.0; 3],
            [1.0, 0.0, 0.0],
        )
        .expect("plane realization");
        let state = object_state(
            1.0,
            RenderAffineTransform3::from_row_major_3x4([
                0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ])
            .expect("finite rotation"),
        );
        let query = RenderSurfaceQuery::new([0.0, 2.0, 0.0], [0.0, -1.0, 0.0], instant())
            .expect("valid surface query");
        let result = realization
            .oriented_surface_query(&state, query)
            .expect("exact plane query");
        let hit = result.hit().expect("plane hit");
        assert_eq!(hit.surface_hit().distance_meters(), 2.0);
        assert_eq!(hit.surface_hit().position_scene_meters(), [0.0, 0.0, 0.0]);
        assert_eq!(hit.geometric_normal_scene(), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn field_sphere_preserves_exact_r3_distance_and_explicit_surface_semantics() {
        let realization =
            FoundingRepresentationRealization::field_sphere(representation_id(3), [0.0; 3], 1.0)
                .expect("field realization");
        let state = object_state(
            0.5,
            RenderAffineTransform3::from_row_major_3x4([
                2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
            ])
            .expect("finite similarity transform"),
        );
        let field_query =
            RenderFieldDistanceQuery::new([0.0, 0.0, 3.0], instant()).expect("valid field query");
        let sample = realization
            .field_distance_query(&state, field_query)
            .expect("exact field sample");
        assert_eq!(sample.signed_distance_estimate_meters(), 2.0);
        assert_eq!(sample.max_absolute_error_meters(), 0.0);
        assert_eq!(sample.safe_distance_lower_bound_meters(), 2.0);

        let evidence = RenderFieldDistanceProtocolEvidence::new(
            RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
            RenderFieldDistanceGuarantee::exact(),
        )
        .expect("exact field evidence");
        evidence
            .validate_sample(sample)
            .expect("proof sample preserves R3 field guarantee");

        let surface_query = RenderSurfaceQuery::new([0.0, 0.0, 3.0], [0.0, 0.0, -1.0], instant())
            .expect("valid surface query");
        let surface = realization
            .oriented_surface_query(&state, surface_query)
            .expect("explicit exact surface capability");
        let hit = surface.hit().expect("field-backed sphere surface hit");
        assert_eq!(hit.surface_hit().distance_meters(), 2.0);
        assert_eq!(hit.surface_hit().position_scene_meters(), [0.0, 0.0, 1.0]);
        assert_eq!(hit.geometric_normal_scene(), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn founding_realizations_reject_invalid_numeric_and_protocol_use() {
        assert_eq!(
            FoundingRepresentationRealization::analytic_sphere(representation_id(1), [0.0; 3], 0.0,),
            Err(ProofRealizationError::NonPositiveExtent)
        );
        assert_eq!(
            FoundingRepresentationRealization::analytic_plane(
                representation_id(2),
                [0.0; 3],
                [0.0; 3],
            ),
            Err(ProofRealizationError::ZeroDirection)
        );
        let sphere =
            FoundingRepresentationRealization::analytic_sphere(representation_id(1), [0.0; 3], 1.0)
                .expect("sphere realization");
        let state = object_state(1.0, RenderAffineTransform3::identity());
        let field_query =
            RenderFieldDistanceQuery::new([0.0; 3], instant()).expect("valid field query");
        assert_eq!(
            sphere.field_distance_query(&state, field_query),
            Err(ProofRealizationError::ProtocolMismatch)
        );
    }
}
