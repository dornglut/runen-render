//! R7 object-local compiled transform and bounded retained-state proof.
//!
//! This module owns one renderer-derived CPU transform compiled from accepted R2 spatial semantics.
//! It is deliberately narrower than a compiled-representation/cache framework: the retained value
//! carries one exact #516 scene dependency plus enough scene-owned continuity provenance to decide
//! whether that transform may be reused across accepted commits.

#[cfg(test)]
use super::derived_state::{RenderDerivedSceneDependencies, RenderDerivedSceneDependency};
use super::representation::classify_field_distance_transform;
#[cfg(test)]
use super::scene::{
    RenderObjectId, RenderSceneCommit, RenderSceneContinuity, RenderSceneResync,
    RenderSceneRevision, RenderSceneSnapshot,
};
use super::space_time::RenderObjectSpatialState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RenderCompiledObjectTransformError {
    NonInvertibleObjectTransform,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RenderCompiledObjectTransform {
    #[cfg(test)]
    local_units_to_scene: [[f64; 3]; 3],
    scene_to_local_units: [[f64; 3]; 3],
    normal_local_to_scene: [[f64; 3]; 3],
    translation_scene: [f64; 3],
}

impl RenderCompiledObjectTransform {
    pub(super) fn compile(
        spatial: &RenderObjectSpatialState,
    ) -> Result<Self, RenderCompiledObjectTransformError> {
        let source = spatial.local_to_scene().row_major_3x4();
        let meters_per_unit = spatial.local_space().meters_per_unit();
        let local_units_to_scene = [
            [
                source[0] * meters_per_unit,
                source[1] * meters_per_unit,
                source[2] * meters_per_unit,
            ],
            [
                source[4] * meters_per_unit,
                source[5] * meters_per_unit,
                source[6] * meters_per_unit,
            ],
            [
                source[8] * meters_per_unit,
                source[9] * meters_per_unit,
                source[10] * meters_per_unit,
            ],
        ];
        let Some(scene_to_local_units) = inverse_3x3(local_units_to_scene) else {
            return Err(RenderCompiledObjectTransformError::NonInvertibleObjectTransform);
        };
        Ok(Self {
            #[cfg(test)]
            local_units_to_scene,
            scene_to_local_units,
            normal_local_to_scene: transpose_3x3(scene_to_local_units),
            translation_scene: [source[3], source[7], source[11]],
        })
    }

    #[cfg(test)]
    pub(super) fn scene_point_from_local(&self, point: [f64; 3]) -> [f64; 3] {
        add(
            mul_matrix_vector(self.local_units_to_scene, point),
            self.translation_scene,
        )
    }

    #[cfg(test)]
    pub(super) fn local_point_from_scene(&self, point: [f64; 3]) -> [f64; 3] {
        mul_matrix_vector(
            self.scene_to_local_units,
            sub(point, self.translation_scene),
        )
    }

    #[cfg(test)]
    pub(super) fn local_direction_per_scene_meter(&self, direction: [f64; 3]) -> [f64; 3] {
        mul_matrix_vector(self.scene_to_local_units, direction)
    }

    #[cfg(test)]
    pub(super) fn scene_normal_from_local(&self, normal: [f64; 3]) -> [f64; 3] {
        normalize(mul_matrix_vector(self.normal_local_to_scene, normal))
            .expect("invertible transform cannot map a non-zero normal to zero")
    }

    pub(super) fn scene_to_local_units_row_major(&self) -> [f64; 9] {
        flatten_3x3(self.scene_to_local_units)
    }

    pub(super) const fn translation_scene(&self) -> [f64; 3] {
        self.translation_scene
    }

    pub(super) fn normal_local_to_scene_row_major(&self) -> [f64; 9] {
        flatten_3x3(self.normal_local_to_scene)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RenderCompiledMetricSimilarityTransformError {
    NotPositiveSimilarity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RenderCompiledMetricSimilarityTransform {
    scene_to_local_meters: [[f64; 3]; 3],
    normal_local_to_scene: [[f64; 3]; 3],
    translation_scene: [f64; 3],
    scene_meters_per_local_meter: f64,
}

impl RenderCompiledMetricSimilarityTransform {
    pub(super) fn compile(
        spatial: &RenderObjectSpatialState,
    ) -> Result<Self, RenderCompiledMetricSimilarityTransformError> {
        // R3 representation semantics own field-transform classification. This derived layer only
        // materializes the accepted exact-similarity case for execution.
        let classification = classify_field_distance_transform(spatial.local_to_scene());
        let Some(scene_meters_per_local_meter) = classification.exact_distance_scale() else {
            return Err(RenderCompiledMetricSimilarityTransformError::NotPositiveSimilarity);
        };

        // RenderObjectSpatialState::local_to_scene consumes metre-normalized local coordinates.
        // Field semantic inputs are already expressed in those local metres, so unlike
        // RenderCompiledObjectTransform this path must not apply local_space.meters_per_unit.
        let source = spatial.local_to_scene().row_major_3x4();
        let local_meters_to_scene = [
            [source[0], source[1], source[2]],
            [source[4], source[5], source[6]],
            [source[8], source[9], source[10]],
        ];
        let Some(scene_to_local_meters) = inverse_3x3(local_meters_to_scene) else {
            return Err(RenderCompiledMetricSimilarityTransformError::NotPositiveSimilarity);
        };

        Ok(Self {
            scene_to_local_meters,
            normal_local_to_scene: transpose_3x3(scene_to_local_meters),
            translation_scene: [source[3], source[7], source[11]],
            scene_meters_per_local_meter,
        })
    }

    #[cfg(test)]
    pub(super) fn local_point_meters_from_scene(&self, point: [f64; 3]) -> [f64; 3] {
        mul_matrix_vector(
            self.scene_to_local_meters,
            sub(point, self.translation_scene),
        )
    }

    pub(super) fn scene_to_local_meters_row_major(&self) -> [f64; 9] {
        flatten_3x3(self.scene_to_local_meters)
    }

    pub(super) fn normal_local_to_scene_row_major(&self) -> [f64; 9] {
        flatten_3x3(self.normal_local_to_scene)
    }

    pub(super) const fn translation_scene(&self) -> [f64; 3] {
        self.translation_scene
    }

    pub(super) const fn scene_meters_per_local_meter(&self) -> f64 {
        self.scene_meters_per_local_meter
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RenderRetainedObjectTransformError {
    ObjectStateMissing,
    NonInvertibleObjectTransform,
    ForeignSceneLineage,
    MissingSceneContinuity {
        expected_previous: RenderSceneRevision,
        actual_previous: RenderSceneRevision,
    },
    SceneInvalidated,
}

#[cfg(test)]
impl From<RenderCompiledObjectTransformError> for RenderRetainedObjectTransformError {
    fn from(value: RenderCompiledObjectTransformError) -> Self {
        match value {
            RenderCompiledObjectTransformError::NonInvertibleObjectTransform => {
                Self::NonInvertibleObjectTransform
            }
        }
    }
}

/// One bounded retained CPU transform proving #516 dependency-driven reuse.
///
/// Advancement consumes `self`. Once continuity, lineage, resynchronization, or a relevant scene
/// delta rejects reuse there is no API that can resurrect the rejected value; callers must rebuild
/// explicitly from a current snapshot.
#[cfg(test)]
#[derive(Debug)]
pub(super) struct RenderRetainedObjectTransform {
    object_id: RenderObjectId,
    dependencies: RenderDerivedSceneDependencies,
    continuity: RenderSceneContinuity,
    compiled: RenderCompiledObjectTransform,
}

#[cfg(test)]
impl RenderRetainedObjectTransform {
    pub(super) fn from_snapshot(
        snapshot: &RenderSceneSnapshot,
        object_id: RenderObjectId,
    ) -> Result<Self, RenderRetainedObjectTransformError> {
        let state = snapshot
            .object_state(object_id)
            .ok_or(RenderRetainedObjectTransformError::ObjectStateMissing)?;
        let compiled = RenderCompiledObjectTransform::compile(state.spatial())?;
        Ok(Self {
            object_id,
            dependencies: RenderDerivedSceneDependencies::new([
                RenderDerivedSceneDependency::ObjectSpatialState(object_id),
            ]),
            continuity: snapshot.continuity(),
            compiled,
        })
    }

    pub(super) fn advance(
        mut self,
        commit: &RenderSceneCommit,
    ) -> Result<Self, RenderRetainedObjectTransformError> {
        if commit.previous_revision() != self.continuity.revision() {
            return Err(RenderRetainedObjectTransformError::MissingSceneContinuity {
                expected_previous: self.continuity.revision(),
                actual_previous: commit.previous_revision(),
            });
        }
        if !commit.directly_follows(&self.continuity) {
            return Err(RenderRetainedObjectTransformError::ForeignSceneLineage);
        }
        if self.dependencies.is_invalidated_by(commit.change_set()) {
            return Err(RenderRetainedObjectTransformError::SceneInvalidated);
        }
        self.continuity = commit.continuity();
        Ok(self)
    }

    pub(super) fn observe_resync(
        self,
        resync: &RenderSceneResync,
    ) -> Result<Self, RenderRetainedObjectTransformError> {
        if self.dependencies.is_invalidated_by(resync.change_set()) {
            return Err(RenderRetainedObjectTransformError::SceneInvalidated);
        }
        unreachable!("the retained transform has one non-empty scene dependency")
    }

    pub(super) const fn object_id(&self) -> RenderObjectId {
        self.object_id
    }

    pub(super) fn dependencies(&self) -> &RenderDerivedSceneDependencies {
        &self.dependencies
    }

    pub(super) const fn validated_revision(&self) -> RenderSceneRevision {
        self.continuity.revision()
    }

    pub(super) const fn compiled(&self) -> RenderCompiledObjectTransform {
        self.compiled
    }
}

fn inverse_3x3(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    if determinant == 0.0 || !determinant.is_finite() {
        return None;
    }
    let inverse_determinant = determinant.recip();
    Some([
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inverse_determinant,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inverse_determinant,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inverse_determinant,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inverse_determinant,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inverse_determinant,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inverse_determinant,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inverse_determinant,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inverse_determinant,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inverse_determinant,
        ],
    ])
}

fn transpose_3x3(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn flatten_3x3(matrix: [[f64; 3]; 3]) -> [f64; 9] {
    [
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2],
    ]
}

#[cfg(test)]
fn mul_matrix_vector(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        dot(matrix[0], vector),
        dot(matrix[1], vector),
        dot(matrix[2], vector),
    ]
}

#[cfg(test)]
fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

#[cfg(test)]
fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let magnitude = dot(vector, vector).sqrt();
    (magnitude > 0.0 && magnitude.is_finite()).then(|| scale(vector, magnitude.recip()))
}

#[cfg(test)]
fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

#[cfg(test)]
fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

#[cfg(test)]
fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{RenderObjectState, RenderSceneStore, RenderSceneUpdate};
    use crate::space_time::{
        RenderAffineTransform3, RenderHandedness, RenderObjectTemporalState, RenderSpaceSpec,
        RenderSpatialCoverage, RenderTemporalSupport, RenderTimeInterval, RenderTimePoint,
    };

    fn spatial(translation_x: f64, meters_per_unit: f64) -> RenderObjectSpatialState {
        RenderObjectSpatialState::new(
            RenderSpaceSpec::new(meters_per_unit, RenderHandedness::Right).expect("proof space"),
            RenderAffineTransform3::from_row_major_3x4([
                2.0,
                0.0,
                0.0,
                translation_x,
                0.0,
                3.0,
                0.0,
                2.0,
                0.0,
                0.0,
                4.0,
                3.0,
            ])
            .expect("proof transform"),
            RenderSpatialCoverage::unbounded(),
        )
    }

    fn state(translation_x: f64, temporal_end: f64) -> RenderObjectState {
        let interval = RenderTimeInterval::new(
            RenderTimePoint::from_seconds(0.0).expect("proof start"),
            RenderTimePoint::from_seconds(temporal_end).expect("proof end"),
        )
        .expect("proof interval");
        RenderObjectState::new(
            spatial(translation_x, 0.5),
            RenderObjectTemporalState::new(RenderTemporalSupport::interval(interval)),
        )
    }

    fn insert_pair(
        store: &mut RenderSceneStore,
    ) -> (RenderObjectId, RenderObjectId, RenderSceneCommit) {
        let target = store.allocate_object_id().expect("target id");
        let unrelated = store.allocate_object_id().expect("unrelated id");
        let mut update = RenderSceneUpdate::new();
        update
            .insert_with_state(target, state(1.0, 1.0))
            .insert_with_state(unrelated, state(10.0, 1.0));
        let commit = store.commit(update).expect("initial scene");
        (target, unrelated, commit)
    }

    #[test]
    fn compiled_transform_preserves_existing_r6_coordinate_semantics() {
        let compiled = RenderCompiledObjectTransform::compile(&spatial(1.0, 0.5))
            .expect("invertible proof transform");
        assert_eq!(
            compiled.scene_point_from_local([1.0, 1.0, 1.0]),
            [2.0, 3.5, 5.0]
        );
        assert_eq!(
            compiled.local_point_from_scene([2.0, 3.5, 5.0]),
            [1.0, 1.0, 1.0]
        );
        assert_eq!(
            compiled.scene_to_local_units_row_major(),
            [1.0, 0.0, 0.0, 0.0, 2.0 / 3.0, 0.0, 0.0, 0.0, 0.5]
        );
        assert_eq!(compiled.translation_scene(), [1.0, 2.0, 3.0]);
        assert_eq!(
            compiled.normal_local_to_scene_row_major(),
            [1.0, 0.0, 0.0, 0.0, 2.0 / 3.0, 0.0, 0.0, 0.0, 0.5]
        );
    }

    #[test]
    fn metric_field_similarity_uses_metre_normalized_transform_without_local_unit_scale() {
        let spatial = RenderObjectSpatialState::new(
            RenderSpaceSpec::new(0.01, RenderHandedness::Right).expect("proof local space"),
            RenderAffineTransform3::from_row_major_3x4([
                0.0, -2.0, 0.0, 3.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, -2.0, 5.0,
            ])
            .expect("exact rotation/reflection/uniform-scale transform"),
            RenderSpatialCoverage::unbounded(),
        );
        let compiled = RenderCompiledMetricSimilarityTransform::compile(&spatial)
            .expect("exact metric similarity");
        assert_eq!(compiled.scene_meters_per_local_meter(), 2.0);
        assert_eq!(
            compiled.local_point_meters_from_scene([1.0, 6.0, 3.0]),
            [1.0, 1.0, 1.0]
        );
        assert_eq!(compiled.translation_scene(), [3.0, 4.0, 5.0]);
    }

    #[test]
    fn metric_field_similarity_rejects_non_uniform_scale_and_shear() {
        for values in [
            [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ] {
            let spatial = RenderObjectSpatialState::new(
                RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("proof local space"),
                RenderAffineTransform3::from_row_major_3x4(values).expect("finite affine"),
                RenderSpatialCoverage::unbounded(),
            );
            assert_eq!(
                RenderCompiledMetricSimilarityTransform::compile(&spatial),
                Err(RenderCompiledMetricSimilarityTransformError::NotPositiveSimilarity)
            );
        }
    }

    #[test]
    fn non_invertible_semantic_spatial_state_fails_only_at_compilation() {
        let semantic = RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("proof space"),
            RenderAffineTransform3::from_row_major_3x4([
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ])
            .expect("finite singular semantic transform"),
            RenderSpatialCoverage::unbounded(),
        );
        assert_eq!(
            RenderCompiledObjectTransform::compile(&semantic),
            Err(RenderCompiledObjectTransformError::NonInvertibleObjectTransform)
        );
    }

    #[test]
    fn retained_transform_reuses_only_exact_spatial_dependency_across_direct_commits() {
        let mut store = RenderSceneStore::new();
        let (target, unrelated, initial) = insert_pair(&mut store);
        let retained = RenderRetainedObjectTransform::from_snapshot(initial.snapshot(), target)
            .expect("retained target transform");
        assert_eq!(retained.object_id(), target);
        assert_eq!(
            retained.dependencies().dependencies(),
            &[RenderDerivedSceneDependency::ObjectSpatialState(target)]
        );
        let original = retained.compiled();

        let no_op = store
            .commit(RenderSceneUpdate::new())
            .expect("accepted no-op");
        let retained = retained.advance(&no_op).expect("no-op continuity");
        assert_eq!(retained.compiled(), original);

        let mut unrelated_update = RenderSceneUpdate::new();
        unrelated_update.replace_state(unrelated, state(20.0, 1.0));
        let unrelated_commit = store
            .commit(unrelated_update)
            .expect("unrelated spatial change");
        let retained = retained
            .advance(&unrelated_commit)
            .expect("unrelated object does not invalidate");
        assert_eq!(retained.compiled(), original);
        assert_eq!(retained.validated_revision(), unrelated_commit.revision());
        assert_eq!(
            retained.compiled(),
            RenderCompiledObjectTransform::compile(
                unrelated_commit
                    .snapshot()
                    .object_state(target)
                    .expect("target state")
                    .spatial(),
            )
            .expect("clean recomputation"),
        );

        let current = unrelated_commit
            .snapshot()
            .object_state(target)
            .expect("target state");
        let later_interval = RenderTimeInterval::new(
            RenderTimePoint::from_seconds(0.0).expect("proof start"),
            RenderTimePoint::from_seconds(2.0).expect("proof end"),
        )
        .expect("proof interval");
        let temporal_only = RenderObjectState::new(
            current.spatial().clone(),
            RenderObjectTemporalState::new(RenderTemporalSupport::interval(later_interval)),
        );
        let mut temporal_update = RenderSceneUpdate::new();
        temporal_update.replace_state(target, temporal_only);
        let temporal_commit = store
            .commit(temporal_update)
            .expect("target temporal-only change");
        let retained = retained
            .advance(&temporal_commit)
            .expect("temporal-only change does not invalidate spatial transform");
        assert_eq!(retained.compiled(), original);
    }

    #[test]
    fn spatial_change_remove_and_full_resync_reject_reuse_and_clean_rebuild_recovers() {
        let mut store = RenderSceneStore::new();
        let (target, _, initial) = insert_pair(&mut store);
        let retained = RenderRetainedObjectTransform::from_snapshot(initial.snapshot(), target)
            .expect("retained target transform");
        let mut spatial_update = RenderSceneUpdate::new();
        spatial_update.replace_state(target, state(5.0, 1.0));
        let spatial_commit = store.commit(spatial_update).expect("target spatial change");
        assert!(matches!(
            retained.advance(&spatial_commit),
            Err(RenderRetainedObjectTransformError::SceneInvalidated)
        ));

        let rebuilt =
            RenderRetainedObjectTransform::from_snapshot(spatial_commit.snapshot(), target)
                .expect("clean rebuild after spatial change");
        assert_eq!(rebuilt.compiled().translation_scene(), [5.0, 2.0, 3.0]);
        assert_eq!(rebuilt.validated_revision(), spatial_commit.revision());

        let resync_retained =
            RenderRetainedObjectTransform::from_snapshot(spatial_commit.snapshot(), target)
                .expect("retained before resync");
        assert!(matches!(
            resync_retained.observe_resync(&store.full_resync()),
            Err(RenderRetainedObjectTransformError::SceneInvalidated)
        ));

        let remove_retained =
            RenderRetainedObjectTransform::from_snapshot(spatial_commit.snapshot(), target)
                .expect("retained before remove");
        let mut remove_update = RenderSceneUpdate::new();
        remove_update.remove(target);
        let remove_commit = store.commit(remove_update).expect("target removal");
        assert!(matches!(
            remove_retained.advance(&remove_commit),
            Err(RenderRetainedObjectTransformError::SceneInvalidated)
        ));
        assert!(matches!(
            RenderRetainedObjectTransform::from_snapshot(remove_commit.snapshot(), target),
            Err(RenderRetainedObjectTransformError::ObjectStateMissing)
        ));
    }

    #[test]
    fn skipped_commit_and_foreign_lineage_fail_closed_even_when_numeric_positions_match() {
        let mut store = RenderSceneStore::new();
        let (target, unrelated, initial) = insert_pair(&mut store);
        let retained = RenderRetainedObjectTransform::from_snapshot(initial.snapshot(), target)
            .expect("retained target transform");

        let mut first_update = RenderSceneUpdate::new();
        first_update.replace_state(unrelated, state(11.0, 1.0));
        let skipped = store.commit(first_update).expect("skipped commit");
        let mut later_update = RenderSceneUpdate::new();
        later_update.replace_state(unrelated, state(12.0, 1.0));
        let later = store.commit(later_update).expect("later commit");
        assert!(matches!(
            retained.advance(&later),
            Err(RenderRetainedObjectTransformError::MissingSceneContinuity {
                expected_previous,
                actual_previous,
            }) if expected_previous == initial.revision() && actual_previous == skipped.revision()
        ));

        let mut first_store = RenderSceneStore::new();
        let (first_target, _, first_initial) = insert_pair(&mut first_store);
        let first_retained =
            RenderRetainedObjectTransform::from_snapshot(first_initial.snapshot(), first_target)
                .expect("first lineage retained transform");

        let mut foreign_store = RenderSceneStore::new();
        let (foreign_target, foreign_unrelated, foreign_initial) = insert_pair(&mut foreign_store);
        assert_eq!(foreign_target, first_target);
        assert_eq!(foreign_initial.revision(), first_initial.revision());
        let mut foreign_update = RenderSceneUpdate::new();
        foreign_update.replace_state(foreign_unrelated, state(13.0, 1.0));
        let foreign = foreign_store
            .commit(foreign_update)
            .expect("foreign direct numeric successor");
        assert_eq!(foreign.previous_revision(), first_initial.revision());
        assert!(matches!(
            first_retained.advance(&foreign),
            Err(RenderRetainedObjectTransformError::ForeignSceneLineage)
        ));
    }
}
