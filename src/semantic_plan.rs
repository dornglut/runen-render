//! R4 device-independent conditional semantic planning.
//!
//! This module consumes only accepted renderer-semantic scene/request/method/representation facts.
//! It does not inspect current availability/residency, output bindings, device capabilities, GPU
//! handles, pipelines, allocations, queues, submissions, ECS resources, or product fallback state.

use super::method::{
    RenderAbstractExecutionRequirement, RenderDistanceErrorBound,
    RenderFieldDistanceInputRequirement, RenderMethodContract, RenderMethodId,
    RenderMethodOutputContract, RenderMethodOutputGuarantee, RenderMethodRepresentationRequirement,
    RenderObservationKind, RenderRepresentationProtocolRequirement,
};
use super::representation::{
    RenderFieldDistanceProtocolEvidence, RenderProtocolCompatibilityError, RenderRepresentationId,
    RenderRepresentationProtocol, RenderRepresentationRecord,
};
use super::request::{
    RenderObservationSpec, RenderRequest, RenderRequestedOutput, RenderSemanticTolerance,
};
use super::scene::{RenderObjectId, RenderSceneRevision, RenderSceneSnapshot};
use std::error::Error;
use std::fmt;

/// Final output accuracy retained in a conditional R4 candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderOutputApproximation {
    Exact,
    BoundedAbsoluteDistance {
        max_error_meters: RenderDistanceErrorBound,
    },
}

impl RenderOutputApproximation {
    pub fn max_absolute_distance_error_meters(self) -> Option<f64> {
        match self {
            Self::Exact => None,
            Self::BoundedAbsoluteDistance { max_error_meters } => Some(max_error_meters.meters()),
        }
    }
}

/// One legal representation/protocol use for one object and one logical output obligation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderApplicableRepresentationUse {
    representation_id: RenderRepresentationId,
    requirement: RenderMethodRepresentationRequirement,
}

impl RenderApplicableRepresentationUse {
    pub const fn representation_id(self) -> RenderRepresentationId {
        self.representation_id
    }

    pub const fn requirement(self) -> RenderMethodRepresentationRequirement {
        self.requirement
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderObjectRepresentationOptions {
    object_id: RenderObjectId,
    uses: Vec<RenderApplicableRepresentationUse>,
}

impl RenderObjectRepresentationOptions {
    pub const fn object_id(&self) -> RenderObjectId {
        self.object_id
    }

    pub fn uses(&self) -> &[RenderApplicableRepresentationUse] {
        &self.uses
    }
}

/// One request output and the request-static semantic choices that can satisfy it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderPlannedOutput {
    output_index: usize,
    observation_index: usize,
    approximation: RenderOutputApproximation,
    object_representations: Vec<RenderObjectRepresentationOptions>,
}

impl RenderPlannedOutput {
    pub const fn output_index(&self) -> usize {
        self.output_index
    }

    pub const fn observation_index(&self) -> usize {
        self.observation_index
    }

    pub const fn approximation(&self) -> RenderOutputApproximation {
        self.approximation
    }

    pub fn object_representations(&self) -> &[RenderObjectRepresentationOptions] {
        &self.object_representations
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderPlanCandidate {
    method_id: RenderMethodId,
    outputs: Vec<RenderPlannedOutput>,
    abstract_execution_requirements: Vec<RenderAbstractExecutionRequirement>,
}

impl RenderPlanCandidate {
    pub const fn method_id(&self) -> RenderMethodId {
        self.method_id
    }

    pub fn outputs(&self) -> &[RenderPlannedOutput] {
        &self.outputs
    }

    pub fn abstract_execution_requirements(&self) -> &[RenderAbstractExecutionRequirement] {
        &self.abstract_execution_requirements
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderPlan {
    scene: RenderSceneSnapshot,
    request: RenderRequest,
    candidates: Vec<RenderPlanCandidate>,
    rejected_methods: Vec<RenderMethodRejection>,
}

impl RenderPlan {
    pub const fn scene_revision(&self) -> RenderSceneRevision {
        self.scene.revision()
    }

    /// The exact immutable R1-R3 semantic scene dependency anchor used to derive this plan.
    ///
    /// `RenderSceneSnapshot` is structurally shared, so retaining it preserves material/emitter,
    /// object-state, participation, and representation semantics without copying or mirroring scene
    /// authority and without later stages reaching back into mutable scene state.
    pub const fn scene(&self) -> &RenderSceneSnapshot {
        &self.scene
    }

    /// The complete accepted R2 semantic request whose obligations this plan preserves.
    pub const fn request(&self) -> &RenderRequest {
        &self.request
    }

    pub fn outputs(&self) -> &[RenderRequestedOutput] {
        self.request.outputs()
    }

    pub fn candidates(&self) -> &[RenderPlanCandidate] {
        &self.candidates
    }

    pub fn rejected_methods(&self) -> &[RenderMethodRejection] {
        &self.rejected_methods
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderMethodRejection {
    method_id: RenderMethodId,
    reason: RenderMethodRejectionReason,
}

impl RenderMethodRejection {
    pub const fn method_id(&self) -> RenderMethodId {
        self.method_id
    }

    pub const fn reason(&self) -> &RenderMethodRejectionReason {
        &self.reason
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderMethodRejectionReason {
    UnsupportedObservation {
        observation_index: usize,
        observation_kind: RenderObservationKind,
    },
    UnsupportedOutput {
        output_index: usize,
    },
    OutputDomainMismatch {
        output_index: usize,
    },
    ExactOutputRequired {
        output_index: usize,
    },
    RelativeOutputToleranceNotProvable {
        output_index: usize,
    },
    OutputApproximationExceedsTolerance {
        output_index: usize,
        max_error_meters: RenderDistanceErrorBound,
        allowed_error_meters: RenderDistanceErrorBound,
    },
    RequiredMaterialAssignmentMissing {
        output_index: usize,
        object_id: RenderObjectId,
    },
    NoApplicableRepresentation {
        output_index: usize,
        object_id: RenderObjectId,
        representation_rejections: Vec<RenderRepresentationRejection>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderRepresentationRejection {
    representation_id: RenderRepresentationId,
    requirement_rejections: Vec<RenderRequirementRejection>,
}

impl RenderRepresentationRejection {
    pub const fn representation_id(&self) -> RenderRepresentationId {
        self.representation_id
    }

    pub fn requirement_rejections(&self) -> &[RenderRequirementRejection] {
        &self.requirement_rejections
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderRequirementRejection {
    requirement: RenderMethodRepresentationRequirement,
    reason: RenderRepresentationRejectionReason,
}

impl RenderRequirementRejection {
    pub const fn requirement(&self) -> RenderMethodRepresentationRequirement {
        self.requirement
    }

    pub const fn reason(&self) -> &RenderRepresentationRejectionReason {
        &self.reason
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderRepresentationRejectionReason {
    ProtocolUnsupported {
        protocol: RenderRepresentationProtocol,
    },
    ProtocolVersionMismatch {
        protocol: RenderRepresentationProtocol,
        requested_revision: u32,
        supported_revision: u32,
    },
    FieldDistanceExactGuaranteeRequired {
        available_max_error_meters: RenderDistanceErrorBound,
    },
    FieldDistanceGuaranteeInsufficient {
        required_max_error_meters: RenderDistanceErrorBound,
        available_max_error_meters: RenderDistanceErrorBound,
    },
    TemporalCoverage {
        observation_index: usize,
    },
    ObjectStateTemporalCoverage {
        observation_index: usize,
    },
    RefinementEvidenceMissing,
    RefinementInsufficient {
        required_max_error_meters: RenderDistanceErrorBound,
        available_finest_error_meters: RenderDistanceErrorBound,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderPlanningFailure {
    NoMethods,
    DuplicateMethodId {
        method_id: RenderMethodId,
    },
    NoSemanticSolution {
        rejections: Vec<RenderMethodRejection>,
    },
}

impl fmt::Display for RenderPlanningFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoMethods => formatter.write_str("semantic planning requires a render method"),
            Self::DuplicateMethodId { method_id } => {
                write!(formatter, "duplicate render method identity {method_id:?}")
            }
            Self::NoSemanticSolution { .. } => {
                formatter.write_str("no render method can satisfy the requested semantics")
            }
        }
    }
}

impl Error for RenderPlanningFailure {}

/// Build one CPU-only, device-independent conditional semantic plan.
///
/// The founding implementation scans the explicitly supplied method set and represented objects.
/// The public plan retains the immutable semantic scene/request dependencies, request output
/// obligations, and eligible representation/protocol uses; it does not materialize a Cartesian
/// product of representation choices or expose execution topology.
pub fn plan_render(
    scene: &RenderSceneSnapshot,
    request: &RenderRequest,
    methods: &[RenderMethodContract],
) -> Result<RenderPlan, RenderPlanningFailure> {
    if methods.is_empty() {
        return Err(RenderPlanningFailure::NoMethods);
    }

    let mut methods = methods.iter().collect::<Vec<_>>();
    methods.sort_by_key(|method| method.id());
    for pair in methods.windows(2) {
        if pair[0].id() == pair[1].id() {
            return Err(RenderPlanningFailure::DuplicateMethodId {
                method_id: pair[0].id(),
            });
        }
    }

    let mut candidates = Vec::new();
    let mut rejected_methods = Vec::new();
    for method in methods {
        match plan_method(scene, request, method) {
            Ok(candidate) => candidates.push(candidate),
            Err(reason) => rejected_methods.push(RenderMethodRejection {
                method_id: method.id(),
                reason,
            }),
        }
    }

    if candidates.is_empty() {
        return Err(RenderPlanningFailure::NoSemanticSolution {
            rejections: rejected_methods,
        });
    }

    Ok(RenderPlan {
        scene: scene.clone(),
        request: request.clone(),
        candidates,
        rejected_methods,
    })
}

fn plan_method(
    scene: &RenderSceneSnapshot,
    request: &RenderRequest,
    method: &RenderMethodContract,
) -> Result<RenderPlanCandidate, RenderMethodRejectionReason> {
    let mut planned_outputs = Vec::with_capacity(request.outputs().len());

    for (output_index, requested_output) in request.outputs().iter().copied().enumerate() {
        let observation_index = requested_output.observation_index();
        let observation = request.observations()[observation_index];
        let observation_kind = RenderObservationKind::of(observation);
        let output_value = requested_output.spec().value();

        let Some(output_contract) = method.output_contract(observation_kind, output_value) else {
            if !method.supports_observation(observation_kind) {
                return Err(RenderMethodRejectionReason::UnsupportedObservation {
                    observation_index,
                    observation_kind,
                });
            }
            if method.has_output_family(observation_kind, output_value) {
                return Err(RenderMethodRejectionReason::OutputDomainMismatch { output_index });
            }
            return Err(RenderMethodRejectionReason::UnsupportedOutput { output_index });
        };

        let approximation = validate_output_guarantee(
            output_index,
            requested_output.spec().tolerance(),
            output_contract.guarantee(),
        )?;
        let object_representations = plan_output_representations(
            scene,
            output_index,
            observation_index,
            observation,
            output_contract,
        )?;

        planned_outputs.push(RenderPlannedOutput {
            output_index,
            observation_index,
            approximation,
            object_representations,
        });
    }

    Ok(RenderPlanCandidate {
        method_id: method.id(),
        outputs: planned_outputs,
        abstract_execution_requirements: method.abstract_execution_requirements().to_vec(),
    })
}

fn plan_output_representations(
    scene: &RenderSceneSnapshot,
    output_index: usize,
    observation_index: usize,
    observation: RenderObservationSpec,
    output_contract: &RenderMethodOutputContract,
) -> Result<Vec<RenderObjectRepresentationOptions>, RenderMethodRejectionReason> {
    let mut object_representations = Vec::new();

    for object_id in scene.object_ids() {
        let Some(participation) = scene.object_participation(object_id) else {
            continue;
        };
        if participation.representations().is_empty() {
            continue;
        }

        if output_contract.requires_material_assignment()
            && participation.material_assignment().is_none()
        {
            return Err(
                RenderMethodRejectionReason::RequiredMaterialAssignmentMissing {
                    output_index,
                    object_id,
                },
            );
        }

        if output_contract.representation_requirements().is_empty() {
            continue;
        }

        let mut applicable_uses = Vec::new();
        let mut representation_rejections = Vec::new();
        for representation in participation.representations() {
            let mut requirement_rejections = Vec::new();
            for requirement in output_contract.representation_requirements() {
                match evaluate_representation_use(
                    scene,
                    object_id,
                    observation_index,
                    observation,
                    representation,
                    *requirement,
                ) {
                    Ok(()) => applicable_uses.push(RenderApplicableRepresentationUse {
                        representation_id: representation.id(),
                        requirement: *requirement,
                    }),
                    Err(reason) => requirement_rejections.push(RenderRequirementRejection {
                        requirement: *requirement,
                        reason,
                    }),
                }
            }

            if requirement_rejections.len() == output_contract.representation_requirements().len() {
                representation_rejections.push(RenderRepresentationRejection {
                    representation_id: representation.id(),
                    requirement_rejections,
                });
            }
        }

        if applicable_uses.is_empty() {
            return Err(RenderMethodRejectionReason::NoApplicableRepresentation {
                output_index,
                object_id,
                representation_rejections,
            });
        }

        object_representations.push(RenderObjectRepresentationOptions {
            object_id,
            uses: applicable_uses,
        });
    }

    Ok(object_representations)
}

fn evaluate_representation_use(
    scene: &RenderSceneSnapshot,
    object_id: RenderObjectId,
    observation_index: usize,
    observation: RenderObservationSpec,
    representation: &RenderRepresentationRecord,
    requirement: RenderMethodRepresentationRequirement,
) -> Result<(), RenderRepresentationRejectionReason> {
    match requirement.protocol() {
        RenderRepresentationProtocolRequirement::SurfaceQuery { revision } => {
            representation
                .surface_query_protocol(revision)
                .map_err(protocol_rejection)?;
        }
        RenderRepresentationProtocolRequirement::OrientedSurfaceQuery { revision } => {
            representation
                .oriented_surface_query_protocol(revision)
                .map_err(protocol_rejection)?;
        }
        RenderRepresentationProtocolRequirement::FieldDistance { revision, input } => {
            let evidence = representation
                .field_distance_protocol(revision)
                .map_err(protocol_rejection)?;
            validate_field_input_guarantee(evidence, input)?;
            let state = scene
                .object_state(object_id)
                .expect("accepted R3 field participation always carries R2 object state");
            if !state
                .temporal()
                .validity()
                .contains_interval(observation.shutter())
            {
                return Err(
                    RenderRepresentationRejectionReason::ObjectStateTemporalCoverage {
                        observation_index,
                    },
                );
            }
        }
    }

    if !representation
        .temporal_support()
        .contains_interval(observation.shutter())
    {
        return Err(RenderRepresentationRejectionReason::TemporalCoverage { observation_index });
    }

    validate_refinement(requirement, representation)
}

fn validate_field_input_guarantee(
    evidence: RenderFieldDistanceProtocolEvidence,
    requirement: RenderFieldDistanceInputRequirement,
) -> Result<(), RenderRepresentationRejectionReason> {
    let guarantee = evidence.guarantee();
    match requirement {
        RenderFieldDistanceInputRequirement::Exact => {
            if guarantee.is_exact() {
                Ok(())
            } else {
                Err(
                    RenderRepresentationRejectionReason::FieldDistanceExactGuaranteeRequired {
                        available_max_error_meters: distance_bound(
                            guarantee.max_absolute_error_meters(),
                        ),
                    },
                )
            }
        }
        RenderFieldDistanceInputRequirement::Conservative => Ok(()),
        RenderFieldDistanceInputRequirement::Bounded {
            max_absolute_error_meters,
        } => {
            let available = guarantee.max_absolute_error_meters();
            if available <= max_absolute_error_meters.meters() {
                Ok(())
            } else {
                Err(
                    RenderRepresentationRejectionReason::FieldDistanceGuaranteeInsufficient {
                        required_max_error_meters: max_absolute_error_meters,
                        available_max_error_meters: distance_bound(available),
                    },
                )
            }
        }
    }
}

fn validate_refinement(
    requirement: RenderMethodRepresentationRequirement,
    representation: &RenderRepresentationRecord,
) -> Result<(), RenderRepresentationRejectionReason> {
    let Some(required) = requirement.maximum_refinement_error() else {
        return Ok(());
    };
    let Some(available) = representation.refinement().finest_absolute_error_meters() else {
        return Err(RenderRepresentationRejectionReason::RefinementEvidenceMissing);
    };
    if available > required.meters() {
        return Err(
            RenderRepresentationRejectionReason::RefinementInsufficient {
                required_max_error_meters: required,
                available_finest_error_meters: distance_bound(available),
            },
        );
    }
    Ok(())
}

fn protocol_rejection(
    error: RenderProtocolCompatibilityError,
) -> RenderRepresentationRejectionReason {
    match error {
        RenderProtocolCompatibilityError::Unsupported { protocol } => {
            RenderRepresentationRejectionReason::ProtocolUnsupported { protocol }
        }
        RenderProtocolCompatibilityError::VersionMismatch {
            protocol,
            requested_revision,
            supported_revision,
        } => RenderRepresentationRejectionReason::ProtocolVersionMismatch {
            protocol,
            requested_revision,
            supported_revision,
        },
    }
}

fn validate_output_guarantee(
    output_index: usize,
    tolerance: RenderSemanticTolerance,
    guarantee: RenderMethodOutputGuarantee,
) -> Result<RenderOutputApproximation, RenderMethodRejectionReason> {
    match guarantee {
        RenderMethodOutputGuarantee::Exact => Ok(RenderOutputApproximation::Exact),
        RenderMethodOutputGuarantee::BoundedAbsoluteDistance { max_error_meters } => {
            let max_error = max_error_meters.meters();
            if max_error == 0.0 {
                return Ok(RenderOutputApproximation::Exact);
            }
            if tolerance.is_exact() {
                return Err(RenderMethodRejectionReason::ExactOutputRequired { output_index });
            }
            if let Some(allowed) = tolerance.absolute_max_error() {
                if max_error <= allowed {
                    return Ok(RenderOutputApproximation::BoundedAbsoluteDistance {
                        max_error_meters,
                    });
                }
                return Err(
                    RenderMethodRejectionReason::OutputApproximationExceedsTolerance {
                        output_index,
                        max_error_meters,
                        allowed_error_meters: distance_bound(allowed),
                    },
                );
            }
            Err(RenderMethodRejectionReason::RelativeOutputToleranceNotProvable { output_index })
        }
    }
}

fn distance_bound(value: f64) -> RenderDistanceErrorBound {
    RenderDistanceErrorBound::new(value)
        .expect("accepted R2/R3 distance/refinement bounds are finite and non-negative")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::render::appearance::RenderDiffuseMaterial;
    use crate::plugins::render::method::{
        RenderAbstractExecutionRequirement, RenderFieldDistanceInputRequirement,
        RenderMethodOutputGuarantee, RenderMethodOutputKind, RenderMethodRepresentationRequirement,
        RenderRepresentationProtocolRequirement, RenderSpectralRadianceSupport,
    };
    use crate::plugins::render::participation::{
        RenderMaterialAssignment, RenderObjectParticipation,
    };
    use crate::plugins::render::representation::{
        RENDER_FIELD_DISTANCE_PROTOCOL_REVISION, RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
        RENDER_SURFACE_QUERY_PROTOCOL_REVISION, RenderFieldDistanceGuarantee,
        RenderFieldDistanceProtocolEvidence, RenderOrientedSurfaceProtocolEvidence,
        RenderRefinementEvidence, RenderRepresentationRecord, RenderSurfaceProtocolEvidence,
    };
    use crate::plugins::render::request::{
        RenderDistanceConvention, RenderObservationSpec, RenderOutputSpec, RenderOutputValue,
        RenderPerspectiveObservation, RenderProbeObservation, RenderRadiometricRepresentation,
        RenderResultTopology, RenderSamplingSupport,
    };
    use crate::plugins::render::scene::{RenderObjectState, RenderSceneStore, RenderSceneUpdate};
    use crate::plugins::render::space_time::{
        RenderAffineTransform3, RenderHandedness, RenderObjectSpatialState,
        RenderObjectTemporalState, RenderSpaceSpec, RenderSpatialCoverage, RenderTemporalSupport,
        RenderTimeInterval, RenderTimePoint,
    };

    fn interval(start: f64, end: f64) -> RenderTimeInterval {
        RenderTimeInterval::new(
            RenderTimePoint::from_seconds(start).expect("time"),
            RenderTimePoint::from_seconds(end).expect("time"),
        )
        .expect("ordered interval")
    }

    fn instant(time: f64) -> RenderTimeInterval {
        interval(time, time)
    }

    fn object_state(
        translation_x: f64,
        temporal_validity: RenderTemporalSupport,
    ) -> RenderObjectState {
        RenderObjectState::new(
            RenderObjectSpatialState::new(
                RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("space"),
                RenderAffineTransform3::from_row_major_3x4([
                    1.0,
                    0.0,
                    0.0,
                    translation_x,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ])
                .expect("transform"),
                RenderSpatialCoverage::unbounded(),
            ),
            RenderObjectTemporalState::new(temporal_validity),
        )
    }

    fn distance_request(
        tolerance: RenderSemanticTolerance,
        shutter: RenderTimeInterval,
    ) -> RenderRequest {
        let observation = RenderObservationSpec::Probe(
            RenderProbeObservation::new(
                RenderAffineTransform3::identity(),
                shutter,
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("probe"),
        );
        let output = RenderOutputSpec::new(
            RenderOutputValue::Distance {
                convention: RenderDistanceConvention::RayDistance,
            },
            RenderResultTopology::scalar(),
            tolerance,
        )
        .expect("distance output");
        RenderRequest::new(
            shutter,
            vec![observation],
            vec![RenderRequestedOutput::new(0, output)],
        )
        .expect("request")
    }

    fn two_probe_distance_request() -> RenderRequest {
        let first = RenderObservationSpec::Probe(
            RenderProbeObservation::new(
                RenderAffineTransform3::identity(),
                instant(0.0),
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("first probe"),
        );
        let second = RenderObservationSpec::Probe(
            RenderProbeObservation::new(
                RenderAffineTransform3::identity(),
                instant(1.0),
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("second probe"),
        );
        let output = || {
            RenderOutputSpec::new(
                RenderOutputValue::Distance {
                    convention: RenderDistanceConvention::RayDistance,
                },
                RenderResultTopology::scalar(),
                RenderSemanticTolerance::exact(),
            )
            .expect("distance output")
        };
        RenderRequest::new(
            interval(0.0, 1.0),
            vec![first, second],
            vec![
                RenderRequestedOutput::new(0, output()),
                RenderRequestedOutput::new(1, output()),
            ],
        )
        .expect("coordinated request")
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

    fn surface_requirement_revision(revision: u32) -> RenderMethodRepresentationRequirement {
        RenderMethodRepresentationRequirement::new(
            RenderRepresentationProtocolRequirement::SurfaceQuery { revision },
            None,
        )
        .expect("surface requirement")
    }

    fn oriented_surface_requirement() -> RenderMethodRepresentationRequirement {
        oriented_surface_requirement_revision(RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION)
    }

    fn oriented_surface_requirement_revision(
        revision: u32,
    ) -> RenderMethodRepresentationRequirement {
        RenderMethodRepresentationRequirement::new(
            RenderRepresentationProtocolRequirement::OrientedSurfaceQuery { revision },
            None,
        )
        .expect("oriented surface requirement")
    }

    fn field_requirement_exact(refinement: Option<f64>) -> RenderMethodRepresentationRequirement {
        RenderMethodRepresentationRequirement::new(
            RenderRepresentationProtocolRequirement::FieldDistance {
                revision: RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
                input: RenderFieldDistanceInputRequirement::Exact,
            },
            refinement.map(|value| RenderDistanceErrorBound::new(value).expect("refinement")),
        )
        .expect("exact field requirement")
    }

    fn field_requirement_bounded(
        max_query_error: f64,
        refinement: Option<f64>,
    ) -> RenderMethodRepresentationRequirement {
        RenderMethodRepresentationRequirement::new(
            RenderRepresentationProtocolRequirement::FieldDistance {
                revision: RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
                input: RenderFieldDistanceInputRequirement::Bounded {
                    max_absolute_error_meters: RenderDistanceErrorBound::new(max_query_error)
                        .expect("field query bound"),
                },
            },
            refinement.map(|value| RenderDistanceErrorBound::new(value).expect("refinement")),
        )
        .expect("bounded field requirement")
    }

    fn probe_distance_contract(
        guarantee: RenderMethodOutputGuarantee,
        requirements: Vec<RenderMethodRepresentationRequirement>,
    ) -> RenderMethodOutputContract {
        RenderMethodOutputContract::new(
            RenderObservationKind::Probe,
            RenderMethodOutputKind::Distance {
                convention: RenderDistanceConvention::RayDistance,
            },
            guarantee,
            requirements,
            false,
        )
        .expect("probe distance contract")
    }

    fn method(raw: u32, outputs: Vec<RenderMethodOutputContract>) -> RenderMethodContract {
        RenderMethodContract::new(
            RenderMethodId::new(raw).expect("method id"),
            outputs,
            vec![RenderAbstractExecutionRequirement::GeneralParallelWork],
        )
        .expect("method")
    }

    fn insert_object(
        store: &mut RenderSceneStore,
        temporal_validity: RenderTemporalSupport,
    ) -> RenderObjectId {
        let object_id = store.allocate_object_id().expect("object id");
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(object_id, object_state(0.0, temporal_validity));
        store.commit(insert).expect("insert");
        object_id
    }

    fn attach(
        store: &mut RenderSceneStore,
        object_id: RenderObjectId,
        representations: Vec<RenderRepresentationRecord>,
        with_material: bool,
    ) {
        let material = with_material.then(|| {
            RenderMaterialAssignment::new(
                RenderDiffuseMaterial::new(0.5).expect("diffuse material"),
            )
        });
        let participation =
            RenderObjectParticipation::new(representations, material, None).expect("participation");
        let mut update = RenderSceneUpdate::new();
        update.replace_participation(object_id, participation);
        store.commit(update).expect("attach representations");
    }

    fn surface_representation(
        store: &mut RenderSceneStore,
        object_id: RenderObjectId,
        revision: u32,
        temporal_support: RenderTemporalSupport,
        refinement: RenderRefinementEvidence,
    ) -> RenderRepresentationRecord {
        let id = store
            .allocate_representation_id(object_id)
            .expect("representation id");
        RenderRepresentationRecord::new(
            id,
            RenderSpatialCoverage::unbounded(),
            temporal_support,
            refinement,
            Some(RenderSurfaceProtocolEvidence::exact(revision).expect("surface protocol")),
            None,
        )
        .expect("surface representation")
    }

    fn oriented_surface_representation(
        store: &mut RenderSceneStore,
        object_id: RenderObjectId,
        surface_revision: u32,
        oriented_revision: u32,
        temporal_support: RenderTemporalSupport,
        refinement: RenderRefinementEvidence,
    ) -> RenderRepresentationRecord {
        let id = store
            .allocate_representation_id(object_id)
            .expect("representation id");
        let oriented = RenderOrientedSurfaceProtocolEvidence::exact(oriented_revision)
            .expect("oriented surface protocol");
        let surface = RenderSurfaceProtocolEvidence::exact(surface_revision)
            .expect("surface protocol")
            .with_oriented_surface(oriented);
        RenderRepresentationRecord::new(
            id,
            RenderSpatialCoverage::unbounded(),
            temporal_support,
            refinement,
            Some(surface),
            None,
        )
        .expect("oriented surface representation")
    }

    fn field_representation(
        store: &mut RenderSceneStore,
        object_id: RenderObjectId,
        revision: u32,
        refinement: RenderRefinementEvidence,
        guarantee: RenderFieldDistanceGuarantee,
        temporal_support: RenderTemporalSupport,
    ) -> RenderRepresentationRecord {
        let id = store
            .allocate_representation_id(object_id)
            .expect("representation id");
        RenderRepresentationRecord::new(
            id,
            RenderSpatialCoverage::unbounded(),
            temporal_support,
            refinement,
            None,
            Some(
                RenderFieldDistanceProtocolEvidence::new(revision, guarantee)
                    .expect("field protocol"),
            ),
        )
        .expect("field representation")
    }

    fn insert_surface_only(
        store: &mut RenderSceneStore,
        revision: u32,
        temporal_support: RenderTemporalSupport,
        with_material: bool,
    ) -> RenderObjectId {
        let object_id = insert_object(store, RenderTemporalSupport::unbounded());
        let representation = surface_representation(
            store,
            object_id,
            revision,
            temporal_support,
            RenderRefinementEvidence::none(),
        );
        attach(store, object_id, vec![representation], with_material);
        object_id
    }

    fn insert_oriented_surface(
        store: &mut RenderSceneStore,
        surface_revision: u32,
        oriented_revision: u32,
        temporal_support: RenderTemporalSupport,
        with_material: bool,
    ) -> RenderObjectId {
        let object_id = insert_object(store, RenderTemporalSupport::unbounded());
        let representation = oriented_surface_representation(
            store,
            object_id,
            surface_revision,
            oriented_revision,
            temporal_support,
            RenderRefinementEvidence::none(),
        );
        attach(store, object_id, vec![representation], with_material);
        object_id
    }

    fn insert_field_only(
        store: &mut RenderSceneStore,
        object_validity: RenderTemporalSupport,
        refinement: RenderRefinementEvidence,
        guarantee: RenderFieldDistanceGuarantee,
        temporal_support: RenderTemporalSupport,
    ) -> RenderObjectId {
        let object_id = insert_object(store, object_validity);
        let representation = field_representation(
            store,
            object_id,
            RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
            refinement,
            guarantee,
            temporal_support,
        );
        attach(store, object_id, vec![representation], false);
        object_id
    }

    fn only_method_reason(failure: RenderPlanningFailure) -> RenderMethodRejectionReason {
        let RenderPlanningFailure::NoSemanticSolution { rejections } = failure else {
            panic!("expected semantic no-solution failure");
        };
        rejections[0].reason().clone()
    }

    fn only_requirement_reason(
        failure: RenderPlanningFailure,
    ) -> RenderRepresentationRejectionReason {
        let RenderMethodRejectionReason::NoApplicableRepresentation {
            representation_rejections,
            ..
        } = only_method_reason(failure)
        else {
            panic!("expected representation rejection");
        };
        representation_rejections[0].requirement_rejections()[0]
            .reason()
            .clone()
    }

    #[test]
    fn deterministic_plan_preserves_request_and_multiple_solution_families() {
        let mut store = RenderSceneStore::new();
        let object_id = insert_object(&mut store, RenderTemporalSupport::unbounded());
        let surface = surface_representation(
            &mut store,
            object_id,
            RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            RenderTemporalSupport::unbounded(),
            RenderRefinementEvidence::none(),
        );
        let field = field_representation(
            &mut store,
            object_id,
            RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
            RenderRefinementEvidence::bounded(0.0).expect("refinement"),
            RenderFieldDistanceGuarantee::exact(),
            RenderTemporalSupport::unbounded(),
        );
        attach(&mut store, object_id, vec![field, surface], false);

        let snapshot = store.snapshot();
        let request = distance_request(RenderSemanticTolerance::exact(), instant(0.0));
        let methods = vec![
            method(
                2,
                vec![probe_distance_contract(
                    RenderMethodOutputGuarantee::Exact,
                    vec![field_requirement_exact(Some(0.0))],
                )],
            ),
            method(
                1,
                vec![probe_distance_contract(
                    RenderMethodOutputGuarantee::Exact,
                    vec![surface_requirement()],
                )],
            ),
        ];
        let first = plan_render(&snapshot, &request, &methods).expect("plan");
        let reversed = vec![methods[1].clone(), methods[0].clone()];
        let second = plan_render(&snapshot, &request, &reversed).expect("plan");

        assert_eq!(first, second);
        assert_eq!(first.scene(), &snapshot);
        assert_eq!(first.request(), &request);
        assert_eq!(first.candidates().len(), 2);
        assert_eq!(first.candidates()[0].method_id().raw(), 1);
        assert_eq!(first.candidates()[1].method_id().raw(), 2);
        assert_eq!(
            first.candidates()[0].outputs()[0].approximation(),
            RenderOutputApproximation::Exact
        );
        assert_eq!(
            first.candidates()[1].outputs()[0].approximation(),
            RenderOutputApproximation::Exact
        );
    }

    #[test]
    fn one_method_accepts_heterogeneous_surface_and_field_objects() {
        let mut store = RenderSceneStore::new();
        insert_surface_only(
            &mut store,
            RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            RenderTemporalSupport::unbounded(),
            false,
        );
        insert_field_only(
            &mut store,
            RenderTemporalSupport::unbounded(),
            RenderRefinementEvidence::bounded(0.01).expect("refinement"),
            RenderFieldDistanceGuarantee::conservative(0.01).expect("field guarantee"),
            RenderTemporalSupport::unbounded(),
        );
        let method = method(
            1,
            vec![probe_distance_contract(
                RenderMethodOutputGuarantee::BoundedAbsoluteDistance {
                    max_error_meters: RenderDistanceErrorBound::new(0.02).expect("output bound"),
                },
                vec![
                    surface_requirement(),
                    field_requirement_bounded(0.02, Some(0.05)),
                ],
            )],
        );
        let plan = plan_render(
            &store.snapshot(),
            &distance_request(
                RenderSemanticTolerance::absolute(0.02).expect("tolerance"),
                instant(0.0),
            ),
            &[method],
        )
        .expect("heterogeneous representations are one legal method family");

        let objects = plan.candidates()[0].outputs()[0].object_representations();
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].uses().len(), 1);
        assert_eq!(objects[1].uses().len(), 1);
        assert_eq!(
            objects[0].uses()[0].requirement().protocol().protocol(),
            RenderRepresentationProtocol::SurfaceQuery
        );
        assert_eq!(
            objects[1].uses()[0].requirement().protocol().protocol(),
            RenderRepresentationProtocol::FieldDistance
        );
    }

    #[test]
    fn oriented_surface_requirement_rejects_plain_surface_and_accepts_compatible_evidence() {
        let request = distance_request(RenderSemanticTolerance::exact(), instant(0.0));
        let oriented_method = method(
            1,
            vec![probe_distance_contract(
                RenderMethodOutputGuarantee::Exact,
                vec![oriented_surface_requirement()],
            )],
        );

        let mut plain_store = RenderSceneStore::new();
        insert_surface_only(
            &mut plain_store,
            RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            RenderTemporalSupport::unbounded(),
            false,
        );
        let unsupported = plan_render(
            &plain_store.snapshot(),
            &request,
            std::slice::from_ref(&oriented_method),
        )
        .expect_err("plain surface evidence must not imply oriented surface support");
        assert_eq!(
            only_requirement_reason(unsupported),
            RenderRepresentationRejectionReason::ProtocolUnsupported {
                protocol: RenderRepresentationProtocol::OrientedSurfaceQuery,
            }
        );

        let mut oriented_store = RenderSceneStore::new();
        insert_oriented_surface(
            &mut oriented_store,
            RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
            RenderTemporalSupport::unbounded(),
            false,
        );
        let plan = plan_render(
            &oriented_store.snapshot(),
            &request,
            std::slice::from_ref(&oriented_method),
        )
        .expect("compatible oriented evidence should use the ordinary R4 planning path");
        assert_eq!(
            plan.candidates()[0].outputs()[0].object_representations()[0].uses()[0]
                .requirement()
                .protocol()
                .protocol(),
            RenderRepresentationProtocol::OrientedSurfaceQuery
        );

        let mut mismatch_store = RenderSceneStore::new();
        insert_oriented_surface(
            &mut mismatch_store,
            RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION + 1,
            RenderTemporalSupport::unbounded(),
            false,
        );
        let mismatch = plan_render(
            &mismatch_store.snapshot(),
            &request,
            &[method(
                1,
                vec![probe_distance_contract(
                    RenderMethodOutputGuarantee::Exact,
                    vec![oriented_surface_requirement_revision(
                        RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
                    )],
                )],
            )],
        )
        .expect_err("oriented protocol revision must match exactly");
        assert_eq!(
            only_requirement_reason(mismatch),
            RenderRepresentationRejectionReason::ProtocolVersionMismatch {
                protocol: RenderRepresentationProtocol::OrientedSurfaceQuery,
                requested_revision: RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
                supported_revision: RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION + 1,
            }
        );
    }

    #[test]
    fn representation_applicability_is_scoped_to_each_output_observation() {
        let mut store = RenderSceneStore::new();
        let object_id = insert_object(&mut store, RenderTemporalSupport::unbounded());
        let at_zero = surface_representation(
            &mut store,
            object_id,
            RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            RenderTemporalSupport::interval(instant(0.0)),
            RenderRefinementEvidence::none(),
        );
        let at_zero_id = at_zero.id();
        let at_one = surface_representation(
            &mut store,
            object_id,
            RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            RenderTemporalSupport::interval(instant(1.0)),
            RenderRefinementEvidence::none(),
        );
        let at_one_id = at_one.id();
        attach(&mut store, object_id, vec![at_one, at_zero], false);
        let method = method(
            1,
            vec![probe_distance_contract(
                RenderMethodOutputGuarantee::Exact,
                vec![surface_requirement()],
            )],
        );
        let plan = plan_render(&store.snapshot(), &two_probe_distance_request(), &[method])
            .expect("different representations may satisfy different observations");

        let outputs = plan.candidates()[0].outputs();
        assert_eq!(outputs.len(), 2);
        assert_eq!(
            outputs[0].object_representations()[0].uses()[0].representation_id(),
            at_zero_id
        );
        assert_eq!(
            outputs[1].object_representations()[0].uses()[0].representation_id(),
            at_one_id
        );
    }

    #[test]
    fn field_applicability_respects_required_object_state_temporal_validity() {
        let mut store = RenderSceneStore::new();
        insert_field_only(
            &mut store,
            RenderTemporalSupport::interval(interval(0.0, 0.25)),
            RenderRefinementEvidence::bounded(0.0).expect("refinement"),
            RenderFieldDistanceGuarantee::exact(),
            RenderTemporalSupport::unbounded(),
        );
        let failure = plan_render(
            &store.snapshot(),
            &distance_request(RenderSemanticTolerance::exact(), instant(0.5)),
            &[method(
                1,
                vec![probe_distance_contract(
                    RenderMethodOutputGuarantee::Exact,
                    vec![field_requirement_exact(Some(0.0))],
                )],
            )],
        )
        .expect_err("field interpretation depends on the R2 state validity");
        assert_eq!(
            only_requirement_reason(failure),
            RenderRepresentationRejectionReason::ObjectStateTemporalCoverage {
                observation_index: 0,
            }
        );
    }

    #[test]
    fn field_query_error_is_an_input_precondition_not_the_rendered_output_bound() {
        let mut store = RenderSceneStore::new();
        insert_field_only(
            &mut store,
            RenderTemporalSupport::unbounded(),
            RenderRefinementEvidence::bounded(0.01).expect("refinement"),
            RenderFieldDistanceGuarantee::conservative(0.01).expect("field guarantee"),
            RenderTemporalSupport::unbounded(),
        );
        let output_bound = RenderDistanceErrorBound::new(0.05).expect("output bound");
        let plan = plan_render(
            &store.snapshot(),
            &distance_request(
                RenderSemanticTolerance::absolute(0.06).expect("tolerance"),
                instant(0.0),
            ),
            &[method(
                1,
                vec![probe_distance_contract(
                    RenderMethodOutputGuarantee::BoundedAbsoluteDistance {
                        max_error_meters: output_bound,
                    },
                    vec![field_requirement_bounded(0.02, Some(0.05))],
                )],
            )],
        )
        .expect("method owns the final output guarantee");
        assert_eq!(
            plan.candidates()[0].outputs()[0]
                .approximation()
                .max_absolute_distance_error_meters(),
            Some(0.05)
        );
    }

    #[test]
    fn bounded_output_guarantee_must_fit_the_request_envelope() {
        let mut store = RenderSceneStore::new();
        insert_surface_only(
            &mut store,
            RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            RenderTemporalSupport::unbounded(),
            false,
        );
        let bounded = method(
            1,
            vec![probe_distance_contract(
                RenderMethodOutputGuarantee::BoundedAbsoluteDistance {
                    max_error_meters: RenderDistanceErrorBound::new(0.01).expect("bound"),
                },
                vec![surface_requirement()],
            )],
        );

        let too_tight = plan_render(
            &store.snapshot(),
            &distance_request(
                RenderSemanticTolerance::absolute(0.005).expect("tolerance"),
                instant(0.0),
            ),
            std::slice::from_ref(&bounded),
        )
        .expect_err("output approximation exceeds request envelope");
        let RenderMethodRejectionReason::OutputApproximationExceedsTolerance {
            max_error_meters,
            allowed_error_meters,
            ..
        } = only_method_reason(too_tight)
        else {
            panic!("expected output approximation rejection");
        };
        assert_eq!(max_error_meters.meters(), 0.01);
        assert_eq!(allowed_error_meters.meters(), 0.005);

        let exact = plan_render(
            &store.snapshot(),
            &distance_request(RenderSemanticTolerance::exact(), instant(0.0)),
            std::slice::from_ref(&bounded),
        )
        .expect_err("bounded method cannot silently satisfy exact output");
        assert_eq!(
            only_method_reason(exact),
            RenderMethodRejectionReason::ExactOutputRequired { output_index: 0 }
        );

        let relative = plan_render(
            &store.snapshot(),
            &distance_request(
                RenderSemanticTolerance::relative(0.1).expect("relative tolerance"),
                instant(0.0),
            ),
            &[bounded],
        )
        .expect_err("non-zero absolute guarantee cannot prove arbitrary relative tolerance");
        assert_eq!(
            only_method_reason(relative),
            RenderMethodRejectionReason::RelativeOutputToleranceNotProvable { output_index: 0 }
        );
    }

    #[test]
    fn protocol_and_field_input_guarantee_failures_are_structured() {
        let request = distance_request(RenderSemanticTolerance::exact(), instant(0.0));

        let mut unsupported_store = RenderSceneStore::new();
        insert_surface_only(
            &mut unsupported_store,
            RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            RenderTemporalSupport::unbounded(),
            false,
        );
        let unsupported = plan_render(
            &unsupported_store.snapshot(),
            &request,
            &[method(
                1,
                vec![probe_distance_contract(
                    RenderMethodOutputGuarantee::Exact,
                    vec![field_requirement_exact(None)],
                )],
            )],
        )
        .expect_err("field protocol is unsupported");
        assert_eq!(
            only_requirement_reason(unsupported),
            RenderRepresentationRejectionReason::ProtocolUnsupported {
                protocol: RenderRepresentationProtocol::FieldDistance,
            }
        );

        let mut mismatch_store = RenderSceneStore::new();
        insert_surface_only(
            &mut mismatch_store,
            RENDER_SURFACE_QUERY_PROTOCOL_REVISION + 1,
            RenderTemporalSupport::unbounded(),
            false,
        );
        let mismatch = plan_render(
            &mismatch_store.snapshot(),
            &request,
            &[method(
                1,
                vec![probe_distance_contract(
                    RenderMethodOutputGuarantee::Exact,
                    vec![surface_requirement_revision(
                        RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
                    )],
                )],
            )],
        )
        .expect_err("protocol revision must match exactly");
        assert_eq!(
            only_requirement_reason(mismatch),
            RenderRepresentationRejectionReason::ProtocolVersionMismatch {
                protocol: RenderRepresentationProtocol::SurfaceQuery,
                requested_revision: RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
                supported_revision: RENDER_SURFACE_QUERY_PROTOCOL_REVISION + 1,
            }
        );

        let mut guarantee_store = RenderSceneStore::new();
        insert_field_only(
            &mut guarantee_store,
            RenderTemporalSupport::unbounded(),
            RenderRefinementEvidence::none(),
            RenderFieldDistanceGuarantee::conservative(0.1).expect("field guarantee"),
            RenderTemporalSupport::unbounded(),
        );
        let guarantee = plan_render(
            &guarantee_store.snapshot(),
            &distance_request(
                RenderSemanticTolerance::absolute(0.2).expect("tolerance"),
                instant(0.0),
            ),
            &[method(
                1,
                vec![probe_distance_contract(
                    RenderMethodOutputGuarantee::BoundedAbsoluteDistance {
                        max_error_meters: RenderDistanceErrorBound::new(0.2).expect("output bound"),
                    },
                    vec![field_requirement_bounded(0.05, None)],
                )],
            )],
        )
        .expect_err("field input guarantee is insufficient");
        let RenderRepresentationRejectionReason::FieldDistanceGuaranteeInsufficient {
            required_max_error_meters,
            available_max_error_meters,
        } = only_requirement_reason(guarantee)
        else {
            panic!("expected field guarantee rejection");
        };
        assert_eq!(required_max_error_meters.meters(), 0.05);
        assert_eq!(available_max_error_meters.meters(), 0.1);
    }

    #[test]
    fn temporal_coverage_and_refinement_failures_are_distinct() {
        let mut coverage_store = RenderSceneStore::new();
        insert_field_only(
            &mut coverage_store,
            RenderTemporalSupport::unbounded(),
            RenderRefinementEvidence::bounded(0.01).expect("refinement"),
            RenderFieldDistanceGuarantee::exact(),
            RenderTemporalSupport::interval(interval(0.0, 0.25)),
        );
        let coverage = plan_render(
            &coverage_store.snapshot(),
            &distance_request(RenderSemanticTolerance::exact(), instant(0.5)),
            &[method(
                1,
                vec![probe_distance_contract(
                    RenderMethodOutputGuarantee::Exact,
                    vec![field_requirement_exact(Some(0.05))],
                )],
            )],
        )
        .expect_err("representation temporal coverage must reject");
        assert_eq!(
            only_requirement_reason(coverage),
            RenderRepresentationRejectionReason::TemporalCoverage {
                observation_index: 0,
            }
        );

        let mut missing_store = RenderSceneStore::new();
        insert_field_only(
            &mut missing_store,
            RenderTemporalSupport::unbounded(),
            RenderRefinementEvidence::none(),
            RenderFieldDistanceGuarantee::exact(),
            RenderTemporalSupport::unbounded(),
        );
        let missing = plan_render(
            &missing_store.snapshot(),
            &distance_request(RenderSemanticTolerance::exact(), instant(0.0)),
            &[method(
                1,
                vec![probe_distance_contract(
                    RenderMethodOutputGuarantee::Exact,
                    vec![field_requirement_exact(Some(0.05))],
                )],
            )],
        )
        .expect_err("refinement evidence is required");
        assert_eq!(
            only_requirement_reason(missing),
            RenderRepresentationRejectionReason::RefinementEvidenceMissing
        );

        let mut insufficient_store = RenderSceneStore::new();
        insert_field_only(
            &mut insufficient_store,
            RenderTemporalSupport::unbounded(),
            RenderRefinementEvidence::bounded(0.1).expect("refinement"),
            RenderFieldDistanceGuarantee::exact(),
            RenderTemporalSupport::unbounded(),
        );
        let insufficient = plan_render(
            &insufficient_store.snapshot(),
            &distance_request(RenderSemanticTolerance::exact(), instant(0.0)),
            &[method(
                1,
                vec![probe_distance_contract(
                    RenderMethodOutputGuarantee::Exact,
                    vec![field_requirement_exact(Some(0.05))],
                )],
            )],
        )
        .expect_err("refinement evidence is too weak");
        assert!(matches!(
            only_requirement_reason(insufficient),
            RenderRepresentationRejectionReason::RefinementInsufficient { .. }
        ));
    }

    #[test]
    fn stronger_field_and_refinement_evidence_remains_admissible() {
        let mut store = RenderSceneStore::new();
        insert_field_only(
            &mut store,
            RenderTemporalSupport::unbounded(),
            RenderRefinementEvidence::bounded(0.0).expect("refinement"),
            RenderFieldDistanceGuarantee::exact(),
            RenderTemporalSupport::unbounded(),
        );
        plan_render(
            &store.snapshot(),
            &distance_request(
                RenderSemanticTolerance::absolute(0.1).expect("tolerance"),
                instant(0.0),
            ),
            &[method(
                1,
                vec![probe_distance_contract(
                    RenderMethodOutputGuarantee::BoundedAbsoluteDistance {
                        max_error_meters: RenderDistanceErrorBound::new(0.05)
                            .expect("output bound"),
                    },
                    vec![field_requirement_bounded(0.05, Some(0.05))],
                )],
            )],
        )
        .expect("equal/stronger input evidence remains admissible");
    }

    #[test]
    fn observation_output_domain_and_material_requirements_are_structured() {
        let mut store = RenderSceneStore::new();
        insert_surface_only(
            &mut store,
            RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            RenderTemporalSupport::unbounded(),
            false,
        );

        let perspective_only = RenderMethodOutputContract::new(
            RenderObservationKind::Perspective,
            RenderMethodOutputKind::Distance {
                convention: RenderDistanceConvention::RayDistance,
            },
            RenderMethodOutputGuarantee::Exact,
            vec![surface_requirement()],
            false,
        )
        .expect("perspective distance");
        let unsupported_observation = plan_render(
            &store.snapshot(),
            &distance_request(RenderSemanticTolerance::exact(), instant(0.0)),
            &[method(1, vec![perspective_only])],
        )
        .expect_err("probe observation is unsupported");
        assert_eq!(
            only_method_reason(unsupported_observation),
            RenderMethodRejectionReason::UnsupportedObservation {
                observation_index: 0,
                observation_kind: RenderObservationKind::Probe,
            }
        );

        let radiance_contract = RenderMethodOutputContract::new(
            RenderObservationKind::Probe,
            RenderMethodOutputKind::Radiance {
                spectral: RenderSpectralRadianceSupport::new(400e-9, 700e-9)
                    .expect("spectral support"),
            },
            RenderMethodOutputGuarantee::Exact,
            vec![surface_requirement()],
            true,
        )
        .expect("probe radiance");
        let unsupported_output = plan_render(
            &store.snapshot(),
            &distance_request(RenderSemanticTolerance::exact(), instant(0.0)),
            &[method(1, vec![radiance_contract.clone()])],
        )
        .expect_err("probe distance is not implied by probe radiance support");
        assert_eq!(
            only_method_reason(unsupported_output),
            RenderMethodRejectionReason::UnsupportedOutput { output_index: 0 }
        );

        let shutter = instant(0.0);
        let probe = RenderObservationSpec::Probe(
            RenderProbeObservation::new(
                RenderAffineTransform3::identity(),
                shutter,
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("probe"),
        );
        let radiance = RenderOutputSpec::new(
            RenderOutputValue::Radiance {
                representation: RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                    800e-9,
                )
                .expect("wavelength"),
            },
            RenderResultTopology::scalar(),
            RenderSemanticTolerance::exact(),
        )
        .expect("radiance output");
        let domain_request = RenderRequest::new(
            shutter,
            vec![probe],
            vec![RenderRequestedOutput::new(0, radiance)],
        )
        .expect("request");
        let domain = plan_render(
            &store.snapshot(),
            &domain_request,
            &[method(1, vec![radiance_contract.clone()])],
        )
        .expect_err("radiometric domain must match");
        assert_eq!(
            only_method_reason(domain),
            RenderMethodRejectionReason::OutputDomainMismatch { output_index: 0 }
        );

        let material_request = RenderRequest::new(
            shutter,
            vec![RenderObservationSpec::Probe(
                RenderProbeObservation::new(
                    RenderAffineTransform3::identity(),
                    shutter,
                    RenderSamplingSupport::ideal_ray(),
                )
                .expect("probe"),
            )],
            vec![RenderRequestedOutput::new(
                0,
                RenderOutputSpec::new(
                    RenderOutputValue::Radiance {
                        representation:
                            RenderRadiometricRepresentation::spectral_at_wavelength_meters(550e-9)
                                .expect("wavelength"),
                    },
                    RenderResultTopology::scalar(),
                    RenderSemanticTolerance::exact(),
                )
                .expect("radiance output"),
            )],
        )
        .expect("request");
        let material = plan_render(
            &store.snapshot(),
            &material_request,
            &[method(1, vec![radiance_contract])],
        )
        .expect_err("radiance contract requires material assignment");
        assert!(matches!(
            only_method_reason(material),
            RenderMethodRejectionReason::RequiredMaterialAssignmentMissing { .. }
        ));
    }

    #[test]
    fn asymmetric_r6_observation_output_shape_is_representable() {
        let mut store = RenderSceneStore::new();
        insert_oriented_surface(
            &mut store,
            RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
            RenderTemporalSupport::unbounded(),
            true,
        );
        let spectral =
            RenderSpectralRadianceSupport::new(400e-9, 700e-9).expect("spectral support");
        let outputs = vec![
            RenderMethodOutputContract::new(
                RenderObservationKind::Perspective,
                RenderMethodOutputKind::Radiance { spectral },
                RenderMethodOutputGuarantee::Exact,
                vec![oriented_surface_requirement()],
                true,
            )
            .expect("perspective radiance"),
            RenderMethodOutputContract::new(
                RenderObservationKind::Perspective,
                RenderMethodOutputKind::Distance {
                    convention: RenderDistanceConvention::ObservationForwardDepth,
                },
                RenderMethodOutputGuarantee::Exact,
                vec![surface_requirement()],
                false,
            )
            .expect("perspective depth"),
            RenderMethodOutputContract::new(
                RenderObservationKind::Perspective,
                RenderMethodOutputKind::ObjectIdentity,
                RenderMethodOutputGuarantee::Exact,
                vec![surface_requirement()],
                false,
            )
            .expect("perspective identity"),
            RenderMethodOutputContract::new(
                RenderObservationKind::Probe,
                RenderMethodOutputKind::Radiance { spectral },
                RenderMethodOutputGuarantee::Exact,
                vec![oriented_surface_requirement()],
                true,
            )
            .expect("probe radiance"),
        ];
        let shutter = instant(0.0);
        let perspective = RenderObservationSpec::Perspective(
            RenderPerspectiveObservation::new(
                RenderAffineTransform3::identity(),
                std::f64::consts::FRAC_PI_2,
                1.0,
                shutter,
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("perspective"),
        );
        let probe = RenderObservationSpec::Probe(
            RenderProbeObservation::new(
                RenderAffineTransform3::identity(),
                shutter,
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("probe"),
        );
        let radiance = |topology| {
            RenderOutputSpec::new(
                RenderOutputValue::Radiance {
                    representation: RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                        550e-9,
                    )
                    .expect("wavelength"),
                },
                topology,
                RenderSemanticTolerance::exact(),
            )
            .expect("radiance output")
        };
        let request = RenderRequest::new(
            shutter,
            vec![perspective, probe],
            vec![
                RenderRequestedOutput::new(
                    0,
                    radiance(RenderResultTopology::sample_lattice_2d(2, 2).expect("lattice")),
                ),
                RenderRequestedOutput::new(
                    0,
                    RenderOutputSpec::new(
                        RenderOutputValue::Distance {
                            convention: RenderDistanceConvention::ObservationForwardDepth,
                        },
                        RenderResultTopology::sample_lattice_2d(2, 2).expect("lattice"),
                        RenderSemanticTolerance::exact(),
                    )
                    .expect("depth output"),
                ),
                RenderRequestedOutput::new(
                    0,
                    RenderOutputSpec::new(
                        RenderOutputValue::ObjectIdentity,
                        RenderResultTopology::sample_lattice_2d(2, 2).expect("lattice"),
                        RenderSemanticTolerance::exact(),
                    )
                    .expect("identity output"),
                ),
                RenderRequestedOutput::new(1, radiance(RenderResultTopology::scalar())),
            ],
        )
        .expect("R6-shaped request");
        let plan = plan_render(&store.snapshot(), &request, &[method(1, outputs)])
            .expect("asymmetric observation/output relation is representable");
        let planned = plan.candidates()[0].outputs();
        assert_eq!(planned.len(), 4);
        assert_eq!(
            planned[0].object_representations()[0].uses()[0]
                .requirement()
                .protocol()
                .protocol(),
            RenderRepresentationProtocol::OrientedSurfaceQuery
        );
        assert_eq!(
            planned[1].object_representations()[0].uses()[0]
                .requirement()
                .protocol()
                .protocol(),
            RenderRepresentationProtocol::SurfaceQuery
        );
        assert_eq!(
            planned[2].object_representations()[0].uses()[0]
                .requirement()
                .protocol()
                .protocol(),
            RenderRepresentationProtocol::SurfaceQuery
        );
        assert_eq!(
            planned[3].object_representations()[0].uses()[0]
                .requirement()
                .protocol()
                .protocol(),
            RenderRepresentationProtocol::OrientedSurfaceQuery
        );
    }

    #[test]
    fn retained_snapshot_is_plannable_after_a_meaningful_newer_commit() {
        let mut store = RenderSceneStore::new();
        let object_id = insert_surface_only(
            &mut store,
            RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            RenderTemporalSupport::unbounded(),
            false,
        );
        let retained = store.snapshot();
        let retained_revision = retained.revision();
        let request = distance_request(RenderSemanticTolerance::exact(), instant(0.0));
        let method = method(
            1,
            vec![probe_distance_contract(
                RenderMethodOutputGuarantee::Exact,
                vec![surface_requirement()],
            )],
        );
        let first = plan_render(&retained, &request, std::slice::from_ref(&method))
            .expect("retained snapshot plan");

        let mut replace = RenderSceneUpdate::new();
        replace.replace_state(
            object_id,
            object_state(2.0, RenderTemporalSupport::unbounded()),
        );
        let newer = store.commit(replace).expect("meaningful state replacement");
        assert_ne!(newer.revision(), retained_revision);
        let current_revision = store.revision();

        let retained_again = plan_render(&retained, &request, &[method])
            .expect("old snapshot remains independently plannable");
        assert_eq!(first, retained_again);
        assert_eq!(retained_again.scene(), &retained);
        assert_eq!(retained_again.scene_revision(), retained_revision);
        assert_eq!(store.revision(), current_revision);
    }

    #[test]
    fn duplicate_method_identity_and_empty_method_set_reject_deterministically() {
        let store = RenderSceneStore::new();
        let request = distance_request(RenderSemanticTolerance::exact(), instant(0.0));
        let method = method(
            1,
            vec![probe_distance_contract(
                RenderMethodOutputGuarantee::Exact,
                vec![surface_requirement()],
            )],
        );
        assert_eq!(
            plan_render(&store.snapshot(), &request, &[method.clone(), method]),
            Err(RenderPlanningFailure::DuplicateMethodId {
                method_id: RenderMethodId::new(1).expect("method id")
            })
        );
        assert_eq!(
            plan_render(&store.snapshot(), &request, &[]),
            Err(RenderPlanningFailure::NoMethods)
        );
    }
}
