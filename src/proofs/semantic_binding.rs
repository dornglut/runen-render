//! Focused proof for request-scoped semantic surface binding laws.
//!
//! This module proves protocol locality and one-binding temporal validity around the maintained R5
//! semantic-binding implementation. It does not create a second planning or binding authority.

use crate::field_input::{
    RenderFieldSemanticInput, RenderFieldSemanticInputBinding, RenderFieldSemanticInputRequirement,
};
use crate::method::{
    RenderAbstractExecutionRequirement, RenderFieldDistanceInputRequirement, RenderMethodContract,
    RenderMethodId, RenderMethodOutputContract, RenderMethodOutputGuarantee,
    RenderMethodOutputKind, RenderMethodRepresentationRequirement, RenderObservationKind,
    RenderRepresentationProtocolRequirement,
};
use crate::participation::RenderObjectParticipation;
use crate::representation::{
    RENDER_FIELD_DISTANCE_PROTOCOL_REVISION, RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
    RENDER_SURFACE_QUERY_PROTOCOL_REVISION, RenderFieldDistanceGuarantee,
    RenderFieldDistanceProtocolEvidence, RenderOrientedSurfaceProtocolEvidence,
    RenderRepresentationId, RenderRepresentationProtocol, RenderRepresentationRecord,
    RenderSurfaceProtocolEvidence,
};
use crate::request::{
    RenderDistanceConvention, RenderObservationSpec, RenderOutputSpec, RenderOutputValue,
    RenderProbeObservation, RenderRequest, RenderRequestedOutput, RenderResultTopology,
    RenderSamplingSupport, RenderSemanticTolerance,
};
use crate::scene::{RenderObjectId, RenderObjectState, RenderSceneStore, RenderSceneUpdate};
use crate::semantic_binding::{RenderNormalizedSemanticInputs, RenderSemanticBindingInputError};
use crate::semantic_plan::{RenderPlan, plan_render};
use crate::space_time::{
    RenderAffineTransform3, RenderHandedness, RenderObjectSpatialState, RenderObjectTemporalState,
    RenderSpaceSpec, RenderSpatialCoverage, RenderTemporalSupport, RenderTimeInterval,
    RenderTimePoint,
};
use crate::surface_input::{
    RenderSurfaceSemanticInput, RenderSurfaceSemanticInputBinding,
    RenderSurfaceSemanticInputRequirement,
};

fn point(seconds: f64) -> RenderTimePoint {
    RenderTimePoint::from_seconds(seconds).expect("finite R7 proof time")
}

fn interval(start: f64, end: f64) -> RenderTimeInterval {
    RenderTimeInterval::new(point(start), point(end)).expect("ordered R7 proof interval")
}

fn object_state() -> RenderObjectState {
    RenderObjectState::new(
        RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("R7 proof space"),
            RenderAffineTransform3::identity(),
            RenderSpatialCoverage::unbounded(),
        ),
        RenderObjectTemporalState::new(RenderTemporalSupport::unbounded()),
    )
}

fn request_at_times(times: &[f64]) -> RenderRequest {
    let start = *times.first().expect("at least one observation time");
    let end = *times.last().expect("at least one observation time");
    let observations = times
        .iter()
        .copied()
        .map(|time| {
            RenderObservationSpec::Probe(
                RenderProbeObservation::new(
                    RenderAffineTransform3::identity(),
                    RenderTimeInterval::instant(point(time)),
                    RenderSamplingSupport::ideal_ray(),
                )
                .expect("R7 proof probe"),
            )
        })
        .collect::<Vec<_>>();
    let outputs = times
        .iter()
        .enumerate()
        .map(|(observation_index, _)| {
            RenderRequestedOutput::new(
                observation_index,
                RenderOutputSpec::new(
                    RenderOutputValue::Distance {
                        convention: RenderDistanceConvention::RayDistance,
                    },
                    RenderResultTopology::scalar(),
                    RenderSemanticTolerance::exact(),
                )
                .expect("R7 proof distance output"),
            )
        })
        .collect::<Vec<_>>();
    RenderRequest::new(interval(start, end), observations, outputs).expect("R7 proof request")
}

fn method(requirements: Vec<RenderMethodRepresentationRequirement>) -> RenderMethodContract {
    let output = RenderMethodOutputContract::new(
        RenderObservationKind::Probe,
        RenderMethodOutputKind::Distance {
            convention: RenderDistanceConvention::RayDistance,
        },
        RenderMethodOutputGuarantee::Exact,
        requirements,
        false,
    )
    .expect("R7 proof output contract");
    RenderMethodContract::new(
        RenderMethodId::new(1).expect("R7 proof method id"),
        vec![output],
        vec![RenderAbstractExecutionRequirement::GeneralParallelWork],
    )
    .expect("R7 proof method")
}

fn plan_for(
    request: RenderRequest,
    surface: Option<RenderSurfaceProtocolEvidence>,
    field: Option<RenderFieldDistanceProtocolEvidence>,
    requirements: Vec<RenderMethodRepresentationRequirement>,
) -> (RenderPlan, RenderRepresentationId, RenderObjectId) {
    let mut store = RenderSceneStore::new();
    let object_id = store.allocate_object_id().expect("R7 proof object id");
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(object_id, object_state());
    store.commit(insert).expect("insert R7 proof object");

    let representation_id = store
        .allocate_representation_id(object_id)
        .expect("R7 proof representation id");
    let spatial_coverage = if field.is_some() {
        RenderSpatialCoverage::axis_aligned_bounds([-1.0; 3], [1.0; 3])
            .expect("R7 proof field coverage")
    } else {
        RenderSpatialCoverage::unbounded()
    };
    let representation = RenderRepresentationRecord::builder(
        representation_id,
        spatial_coverage,
        RenderTemporalSupport::unbounded(),
    )
    .surface_query(surface)
    .field_distance(field)
    .build()
    .expect("R7 proof representation");
    let participation = RenderObjectParticipation::from_representations(vec![representation])
        .expect("R7 proof participation");
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object_id, participation);
    store.commit(attach).expect("attach R7 proof participation");

    let method = method(requirements);
    let plan = plan_render(&store.snapshot(), &request, std::slice::from_ref(&method))
        .expect("R7 proof plan");
    (plan, representation_id, object_id)
}

fn surface_requirement() -> RenderMethodRepresentationRequirement {
    RenderMethodRepresentationRequirement::new(
        RenderRepresentationProtocolRequirement::SurfaceQuery {
            revision: RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
        },
        None,
    )
    .expect("surface requirement")
}

fn oriented_requirement() -> RenderMethodRepresentationRequirement {
    RenderMethodRepresentationRequirement::new(
        RenderRepresentationProtocolRequirement::OrientedSurfaceQuery {
            revision: RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
        },
        None,
    )
    .expect("oriented-surface requirement")
}

fn field_requirement() -> RenderMethodRepresentationRequirement {
    RenderMethodRepresentationRequirement::new(
        RenderRepresentationProtocolRequirement::FieldDistance {
            revision: RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
            input: RenderFieldDistanceInputRequirement::Exact,
        },
        None,
    )
    .expect("field requirement")
}

fn bound_surface_evidence() -> RenderSurfaceProtocolEvidence {
    RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
        .expect("surface protocol")
        .with_oriented_surface(
            RenderOrientedSurfaceProtocolEvidence::exact(
                RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
            )
            .expect("oriented-surface protocol"),
        )
        .with_semantic_input_requirement(RenderSurfaceSemanticInputRequirement::current())
}

fn exact_field_evidence() -> RenderFieldDistanceProtocolEvidence {
    RenderFieldDistanceProtocolEvidence::new(
        RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
        RenderFieldDistanceGuarantee::exact(),
    )
    .expect("field protocol")
}

fn exact_bound_field_evidence() -> RenderFieldDistanceProtocolEvidence {
    exact_field_evidence()
        .with_semantic_input_requirement(RenderFieldSemanticInputRequirement::current())
}

fn exact_bound_field_requirement() -> RenderMethodRepresentationRequirement {
    RenderMethodRepresentationRequirement::new(
        RenderRepresentationProtocolRequirement::FieldDistance {
            revision: RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
            input: RenderFieldDistanceInputRequirement::Exact,
        },
        None,
    )
    .expect("exact field requirement")
}

fn conservative_field_requirement() -> RenderMethodRepresentationRequirement {
    RenderMethodRepresentationRequirement::new(
        RenderRepresentationProtocolRequirement::FieldDistance {
            revision: RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
            input: RenderFieldDistanceInputRequirement::Conservative,
        },
        None,
    )
    .expect("conservative field requirement")
}

fn conservative_bound_field_evidence(
    max_absolute_error_meters: f64,
) -> RenderFieldDistanceProtocolEvidence {
    RenderFieldDistanceProtocolEvidence::new(
        RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
        RenderFieldDistanceGuarantee::conservative(max_absolute_error_meters)
            .expect("finite conservative field guarantee"),
    )
    .expect("field protocol")
    .with_semantic_input_requirement(RenderFieldSemanticInputRequirement::current())
}

fn sampled_field(validity: RenderTemporalSupport) -> RenderFieldSemanticInput {
    sampled_field_with_error(0.0, validity)
}

fn sampled_field_with_error(
    max_absolute_query_error_local_meters: f64,
    validity: RenderTemporalSupport,
) -> RenderFieldSemanticInput {
    RenderFieldSemanticInput::dense(
        [-1.0; 3],
        [1.0; 3],
        [3, 3, 3],
        vec![0.0; 27],
        max_absolute_query_error_local_meters,
        validity,
    )
    .expect("field semantic input")
}

fn sphere(validity: RenderTemporalSupport) -> RenderSurfaceSemanticInput {
    RenderSurfaceSemanticInput::sphere([0.0; 3], 1.0, validity).expect("surface semantic input")
}

#[test]
fn surface_prerequisite_does_not_apply_to_field_use_on_same_representation() {
    let (plan, representation_id, _) = plan_for(
        request_at_times(&[0.0]),
        Some(bound_surface_evidence()),
        Some(exact_field_evidence()),
        vec![field_requirement()],
    );
    let inputs = RenderNormalizedSemanticInputs::normalize(&plan, &[], &[])
        .expect("field-only use needs no surface binding");
    let candidate = inputs
        .specialize_candidate(&plan, &plan.candidates()[0])
        .expect("field use remains semantically admissible");
    let uses = candidate.outputs()[0].objects()[0].uses();
    assert_eq!(uses.len(), 1);
    assert_eq!(uses[0].representation_id(), representation_id);
    assert_eq!(
        uses[0].requirement().protocol().protocol(),
        RenderRepresentationProtocol::FieldDistance
    );
}

#[test]
fn surface_and_oriented_uses_share_one_canonical_binding() {
    let (plan, representation_id, object_id) = plan_for(
        request_at_times(&[0.0]),
        Some(bound_surface_evidence()),
        None,
        vec![surface_requirement(), oriented_requirement()],
    );
    let binding = RenderSurfaceSemanticInputBinding::new(
        representation_id,
        sphere(RenderTemporalSupport::unbounded()),
    );
    let inputs = RenderNormalizedSemanticInputs::normalize(&plan, &[binding], &[])
        .expect("one canonical surface binding");
    let candidate = inputs
        .specialize_candidate(&plan, &plan.candidates()[0])
        .expect("both surface protocol alternatives remain admissible");
    let uses = candidate.outputs()[0].objects()[0].uses();
    assert_eq!(uses.len(), 2);
    let first = inputs
        .surface_binding_for_selected_use(&plan, object_id, uses[0])
        .expect("first use binding");
    let second = inputs
        .surface_binding_for_selected_use(&plan, object_id, uses[1])
        .expect("second use binding");
    assert!(std::ptr::eq(first, second));
    assert_eq!(first.representation_id(), representation_id);
}

#[test]
fn field_prerequisite_requires_matching_typed_binding() {
    let (plan, representation_id, _) = plan_for(
        request_at_times(&[0.0]),
        None,
        Some(exact_bound_field_evidence()),
        vec![exact_bound_field_requirement()],
    );

    let missing = RenderNormalizedSemanticInputs::normalize(&plan, &[], &[])
        .expect("missing binding is absence, not malformed input");
    missing
        .specialize_candidate(&plan, &plan.candidates()[0])
        .expect_err("required field binding must fail closed when absent");

    let binding = RenderFieldSemanticInputBinding::new(
        representation_id,
        sampled_field(RenderTemporalSupport::unbounded()),
    );
    let supplied = RenderNormalizedSemanticInputs::normalize(&plan, &[], &[binding])
        .expect("matching typed field binding");
    supplied
        .specialize_candidate(&plan, &plan.candidates()[0])
        .expect("matching field binding should admit the field representation");
}

#[test]
fn exact_field_representation_rejects_nonzero_current_query_error() {
    let (plan, representation_id, _) = plan_for(
        request_at_times(&[0.0]),
        None,
        Some(exact_bound_field_evidence()),
        vec![exact_bound_field_requirement()],
    );
    let binding = RenderFieldSemanticInputBinding::new(
        representation_id,
        sampled_field_with_error(0.001, RenderTemporalSupport::unbounded()),
    );
    let inputs = RenderNormalizedSemanticInputs::normalize(&plan, &[], &[binding])
        .expect("field binding shape");

    inputs
        .specialize_candidate(&plan, &plan.candidates()[0])
        .expect_err("exact field guarantee must reject a non-zero current query error");
}

#[test]
fn conservative_field_representation_accepts_only_within_current_query_error_bound() {
    let (plan, representation_id, _) = plan_for(
        request_at_times(&[0.0]),
        None,
        Some(conservative_bound_field_evidence(0.25)),
        vec![conservative_field_requirement()],
    );

    let within = RenderNormalizedSemanticInputs::normalize(
        &plan,
        &[],
        &[RenderFieldSemanticInputBinding::new(
            representation_id,
            sampled_field_with_error(0.25, RenderTemporalSupport::unbounded()),
        )],
    )
    .expect("bounded field binding shape");
    within
        .specialize_candidate(&plan, &plan.candidates()[0])
        .expect("query error equal to the declared field guarantee should remain admissible");

    let exceeds = RenderNormalizedSemanticInputs::normalize(
        &plan,
        &[],
        &[RenderFieldSemanticInputBinding::new(
            representation_id,
            sampled_field_with_error(0.250_001, RenderTemporalSupport::unbounded()),
        )],
    )
    .expect("bounded field binding shape");
    exceeds
        .specialize_candidate(&plan, &plan.candidates()[0])
        .expect_err("current query error above the representation guarantee must fail closed");
}

#[test]
fn field_query_error_admission_does_not_require_refinement_evidence() {
    let (plan, representation_id, _) = plan_for(
        request_at_times(&[0.0]),
        None,
        Some(conservative_bound_field_evidence(0.1)),
        vec![conservative_field_requirement()],
    );
    let binding = RenderFieldSemanticInputBinding::new(
        representation_id,
        sampled_field_with_error(0.05, RenderTemporalSupport::unbounded()),
    );
    let inputs = RenderNormalizedSemanticInputs::normalize(&plan, &[], &[binding])
        .expect("field binding shape");

    inputs
        .specialize_candidate(&plan, &plan.candidates()[0])
        .expect("query-error admission is independent from absent refinement evidence");
}

#[test]
fn finite_field_binding_must_match_representation_coverage() {
    let (plan, representation_id, _) = plan_for(
        request_at_times(&[0.0]),
        None,
        Some(conservative_bound_field_evidence(0.1)),
        vec![conservative_field_requirement()],
    );
    let mismatched = RenderFieldSemanticInput::dense(
        [-2.0; 3],
        [1.5; 3],
        [3, 3, 3],
        vec![0.0; 27],
        0.05,
        RenderTemporalSupport::unbounded(),
    )
    .expect("finite mismatched field input");
    let inputs = RenderNormalizedSemanticInputs::normalize(
        &plan,
        &[],
        &[RenderFieldSemanticInputBinding::new(
            representation_id,
            mismatched,
        )],
    )
    .expect("field binding shape");

    inputs
        .specialize_candidate(&plan, &plan.candidates()[0])
        .expect_err("finite field input must not satisfy a different intrinsic coverage");
}

#[test]
fn field_binding_for_foreign_representation_fails_closed() {
    let (plan, _, _) = plan_for(
        request_at_times(&[0.0]),
        None,
        Some(exact_bound_field_evidence()),
        vec![exact_bound_field_requirement()],
    );
    let foreign = RenderRepresentationId::from_raw(999).expect("foreign representation id");
    let binding = RenderFieldSemanticInputBinding::new(
        foreign,
        sampled_field(RenderTemporalSupport::unbounded()),
    );
    let error = RenderNormalizedSemanticInputs::normalize(&plan, &[], &[binding])
        .expect_err("foreign field binding must fail closed");
    assert_eq!(
        error,
        RenderSemanticBindingInputError::ForeignFieldBinding {
            representation_id: foreign,
        }
    );
}

#[test]
fn one_binding_must_cover_every_selected_observation_shutter() {
    let (plan, representation_id, _) = plan_for(
        request_at_times(&[0.0, 1.0]),
        Some(bound_surface_evidence()),
        None,
        vec![surface_requirement()],
    );
    let only_first = RenderNormalizedSemanticInputs::normalize(
        &plan,
        &[RenderSurfaceSemanticInputBinding::new(
            representation_id,
            sphere(RenderTemporalSupport::interval(interval(0.0, 0.0))),
        )],
        &[],
    )
    .expect("binding shape");
    let rejection = only_first
        .specialize_candidate(&plan, &plan.candidates()[0])
        .expect_err("the one canonical value must cover every selected shutter");
    assert_eq!(rejection.output_index(), 1);

    let covers_both = RenderNormalizedSemanticInputs::normalize(
        &plan,
        &[RenderSurfaceSemanticInputBinding::new(
            representation_id,
            sphere(RenderTemporalSupport::interval(interval(0.0, 1.0))),
        )],
        &[],
    )
    .expect("binding shape");
    let candidate = covers_both
        .specialize_candidate(&plan, &plan.candidates()[0])
        .expect("one value valid for both shutters");
    assert_eq!(candidate.outputs().len(), 2);
}
