//! R5 request-scoped semantic binding admission for typed surface and field representations.
//!
//! This module remains narrower than a dynamic-input registry or second planning graph. It
//! normalizes one invocation's concrete typed bindings and specializes already-planned R4
//! representation alternatives without inspecting availability, GPU state, or physical outputs.

use super::field_input::RenderFieldSemanticInputBinding;
use super::representation::{
    RenderFieldDistanceSample, RenderRepresentationId, RenderRepresentationProtocol,
    classify_field_distance_transform,
};
use super::scene::RenderObjectId;
use super::semantic_plan::{
    RenderApplicableRepresentationUse, RenderOutputApproximation, RenderPlan, RenderPlanCandidate,
    RenderPlannedOutput,
};
use super::surface_input::RenderSurfaceSemanticInputBinding;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderSemanticBindingInputError {
    DuplicateSurfaceBinding {
        representation_id: RenderRepresentationId,
    },
    ForeignSurfaceBinding {
        representation_id: RenderRepresentationId,
    },
    UndeclaredSurfaceBinding {
        representation_id: RenderRepresentationId,
    },
    DuplicateFieldBinding {
        representation_id: RenderRepresentationId,
    },
    ForeignFieldBinding {
        representation_id: RenderRepresentationId,
    },
    UndeclaredFieldBinding {
        representation_id: RenderRepresentationId,
    },
}

impl fmt::Display for RenderSemanticBindingInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSurfaceBinding { representation_id } => write!(
                formatter,
                "duplicate request-scoped surface binding for {representation_id:?}"
            ),
            Self::ForeignSurfaceBinding { representation_id } => write!(
                formatter,
                "surface binding {representation_id:?} is not referenced by this render plan"
            ),
            Self::UndeclaredSurfaceBinding { representation_id } => write!(
                formatter,
                "surface binding {representation_id:?} was supplied for no planned surface-protocol use that declares the prerequisite"
            ),
            Self::DuplicateFieldBinding { representation_id } => write!(
                formatter,
                "duplicate request-scoped field binding for {representation_id:?}"
            ),
            Self::ForeignFieldBinding { representation_id } => write!(
                formatter,
                "field binding {representation_id:?} is not referenced by this render plan"
            ),
            Self::UndeclaredFieldBinding { representation_id } => write!(
                formatter,
                "field binding {representation_id:?} was supplied for no planned field-distance use that declares the prerequisite"
            ),
        }
    }
}

impl Error for RenderSemanticBindingInputError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RenderNormalizedSemanticInputs {
    surface_by_representation: BTreeMap<RenderRepresentationId, RenderSurfaceSemanticInputBinding>,
    field_by_representation: BTreeMap<RenderRepresentationId, RenderFieldSemanticInputBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RenderSemanticallyAdmittedCandidate {
    outputs: Vec<RenderSemanticallyAdmittedOutput>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RenderSemanticallyAdmittedOutput {
    output_index: usize,
    observation_index: usize,
    approximation: RenderOutputApproximation,
    objects: Vec<RenderSemanticallyAdmittedObject>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RenderSemanticallyAdmittedObject {
    object_id: RenderObjectId,
    uses: Vec<RenderApplicableRepresentationUse>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RenderSemanticCandidateRejection {
    output_index: usize,
    object_id: RenderObjectId,
    representation_ids: Vec<RenderRepresentationId>,
}

impl RenderSemanticallyAdmittedCandidate {
    pub(super) fn outputs(&self) -> &[RenderSemanticallyAdmittedOutput] {
        &self.outputs
    }

    #[cfg(test)]
    pub(super) fn vacuous(candidate: &RenderPlanCandidate) -> Self {
        Self {
            outputs: candidate
                .outputs()
                .iter()
                .map(|output| RenderSemanticallyAdmittedOutput {
                    output_index: output.output_index(),
                    observation_index: output.observation_index(),
                    approximation: output.approximation(),
                    objects: output
                        .object_representations()
                        .iter()
                        .map(|object| RenderSemanticallyAdmittedObject {
                            object_id: object.object_id(),
                            uses: object.uses().to_vec(),
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

impl RenderSemanticallyAdmittedOutput {
    pub(super) const fn output_index(&self) -> usize {
        self.output_index
    }

    pub(super) const fn observation_index(&self) -> usize {
        self.observation_index
    }

    pub(super) const fn approximation(&self) -> RenderOutputApproximation {
        self.approximation
    }

    pub(super) fn objects(&self) -> &[RenderSemanticallyAdmittedObject] {
        &self.objects
    }
}

impl RenderSemanticallyAdmittedObject {
    pub(super) const fn object_id(&self) -> RenderObjectId {
        self.object_id
    }

    pub(super) fn uses(&self) -> &[RenderApplicableRepresentationUse] {
        &self.uses
    }
}

impl RenderSemanticCandidateRejection {
    pub(super) const fn output_index(&self) -> usize {
        self.output_index
    }

    pub(super) const fn object_id(&self) -> RenderObjectId {
        self.object_id
    }

    pub(super) fn representation_ids(&self) -> &[RenderRepresentationId] {
        &self.representation_ids
    }
}

impl RenderNormalizedSemanticInputs {
    pub(super) fn normalize(
        plan: &RenderPlan,
        surface_bindings: &[RenderSurfaceSemanticInputBinding],
        field_bindings: &[RenderFieldSemanticInputBinding],
    ) -> Result<Self, RenderSemanticBindingInputError> {
        let mut planned_representations = BTreeSet::new();
        let mut declared_surface_prerequisites = BTreeSet::new();
        let mut declared_field_prerequisites = BTreeSet::new();
        for candidate in plan.candidates() {
            for output in candidate.outputs() {
                for object in output.object_representations() {
                    for representation in object.uses().iter().copied() {
                        let representation_id = representation.representation_id();
                        planned_representations.insert(representation_id);
                        if use_declares_surface_input(plan, object.object_id(), representation) {
                            declared_surface_prerequisites.insert(representation_id);
                        }
                        if use_declares_field_input(plan, object.object_id(), representation) {
                            declared_field_prerequisites.insert(representation_id);
                        }
                    }
                }
            }
        }

        let mut surface_by_representation = BTreeMap::new();
        for binding in surface_bindings.iter().cloned() {
            let representation_id = binding.representation_id();
            if surface_by_representation
                .insert(representation_id, binding)
                .is_some()
            {
                return Err(RenderSemanticBindingInputError::DuplicateSurfaceBinding {
                    representation_id,
                });
            }
            if !planned_representations.contains(&representation_id) {
                return Err(RenderSemanticBindingInputError::ForeignSurfaceBinding {
                    representation_id,
                });
            }
            if !declared_surface_prerequisites.contains(&representation_id) {
                return Err(RenderSemanticBindingInputError::UndeclaredSurfaceBinding {
                    representation_id,
                });
            }
        }

        let mut field_by_representation = BTreeMap::new();
        for binding in field_bindings.iter().cloned() {
            let representation_id = binding.representation_id();
            if field_by_representation
                .insert(representation_id, binding)
                .is_some()
            {
                return Err(RenderSemanticBindingInputError::DuplicateFieldBinding {
                    representation_id,
                });
            }
            if !planned_representations.contains(&representation_id) {
                return Err(RenderSemanticBindingInputError::ForeignFieldBinding {
                    representation_id,
                });
            }
            if !declared_field_prerequisites.contains(&representation_id) {
                return Err(RenderSemanticBindingInputError::UndeclaredFieldBinding {
                    representation_id,
                });
            }
        }

        Ok(Self {
            surface_by_representation,
            field_by_representation,
        })
    }

    pub(super) fn specialize_candidate(
        &self,
        plan: &RenderPlan,
        candidate: &RenderPlanCandidate,
    ) -> Result<RenderSemanticallyAdmittedCandidate, RenderSemanticCandidateRejection> {
        let mut outputs = Vec::with_capacity(candidate.outputs().len());
        for output in candidate.outputs() {
            let mut objects = Vec::with_capacity(output.object_representations().len());
            for object in output.object_representations() {
                let uses = object
                    .uses()
                    .iter()
                    .copied()
                    .filter(|representation| {
                        self.allows_use(plan, output, object.object_id(), *representation)
                    })
                    .collect::<Vec<_>>();
                if uses.is_empty() {
                    return Err(RenderSemanticCandidateRejection {
                        output_index: output.output_index(),
                        object_id: object.object_id(),
                        representation_ids: object
                            .uses()
                            .iter()
                            .map(|representation| representation.representation_id())
                            .collect(),
                    });
                }
                objects.push(RenderSemanticallyAdmittedObject {
                    object_id: object.object_id(),
                    uses,
                });
            }
            outputs.push(RenderSemanticallyAdmittedOutput {
                output_index: output.output_index(),
                observation_index: output.observation_index(),
                approximation: output.approximation(),
                objects,
            });
        }
        Ok(RenderSemanticallyAdmittedCandidate { outputs })
    }

    fn allows_use(
        &self,
        plan: &RenderPlan,
        output: &RenderPlannedOutput,
        object_id: RenderObjectId,
        representation: RenderApplicableRepresentationUse,
    ) -> bool {
        let Some(record) = plan
            .scene()
            .object_participation(object_id)
            .and_then(|participation| {
                participation.representation(representation.representation_id())
            })
        else {
            return false;
        };

        match representation.requirement().protocol().protocol() {
            RenderRepresentationProtocol::SurfaceQuery
            | RenderRepresentationProtocol::OrientedSurfaceQuery => {
                if record.surface_semantic_input_requirement().is_none() {
                    return true;
                }
                let Some(binding) = self
                    .surface_by_representation
                    .get(&representation.representation_id())
                else {
                    return false;
                };
                self.binding_valid_for_output(
                    plan,
                    output,
                    object_id,
                    record.temporal_support(),
                    binding.input().validity(),
                )
            }
            RenderRepresentationProtocol::FieldDistance => {
                if record.field_semantic_input_requirement().is_none() {
                    return true;
                }
                let Some(binding) = self
                    .field_by_representation
                    .get(&representation.representation_id())
                else {
                    return false;
                };
                if !self.binding_valid_for_output(
                    plan,
                    output,
                    object_id,
                    record.temporal_support(),
                    binding.input().validity(),
                ) {
                    return false;
                }
                self.field_binding_satisfies_protocol_guarantee(
                    plan,
                    object_id,
                    record,
                    representation,
                    binding,
                )
            }
        }
    }

    fn field_binding_satisfies_protocol_guarantee(
        &self,
        plan: &RenderPlan,
        object_id: RenderObjectId,
        record: &super::representation::RenderRepresentationRecord,
        representation: RenderApplicableRepresentationUse,
        binding: &RenderFieldSemanticInputBinding,
    ) -> bool {
        let Some((coverage_min, coverage_max)) =
            record.spatial_coverage().axis_aligned_bounds_value()
        else {
            return false;
        };
        if coverage_min != binding.input().origin_local_meters()
            || coverage_max != binding.input().max_local_meters()
        {
            return false;
        }

        let Some(state) = plan.scene().object_state(object_id) else {
            return false;
        };
        let Some(distance_scale) =
            classify_field_distance_transform(state.spatial().local_to_scene())
                .exact_distance_scale()
        else {
            return false;
        };
        let scene_error_meters =
            binding.input().max_absolute_query_error_local_meters() * distance_scale;
        if !scene_error_meters.is_finite() {
            return false;
        }
        let Ok(evidence) =
            record.field_distance_protocol(representation.requirement().protocol().revision())
        else {
            return false;
        };
        let Ok(sample) = RenderFieldDistanceSample::new(0.0, scene_error_meters) else {
            return false;
        };
        evidence.validate_sample(sample).is_ok()
    }

    fn binding_valid_for_output(
        &self,
        plan: &RenderPlan,
        output: &RenderPlannedOutput,
        object_id: RenderObjectId,
        representation_validity: super::space_time::RenderTemporalSupport,
        input_validity: super::space_time::RenderTemporalSupport,
    ) -> bool {
        let shutter = plan.request().observations()[output.observation_index()].shutter();
        if !representation_validity.contains_interval(shutter)
            || !input_validity.contains_interval(shutter)
        {
            return false;
        }
        plan.scene()
            .object_state(object_id)
            .is_some_and(|state| state.temporal().validity().contains_interval(shutter))
    }

    pub(super) fn surface_binding_for_selected_use(
        &self,
        plan: &RenderPlan,
        object_id: RenderObjectId,
        representation: RenderApplicableRepresentationUse,
    ) -> Option<&RenderSurfaceSemanticInputBinding> {
        use_declares_surface_input(plan, object_id, representation)
            .then(|| {
                self.surface_by_representation
                    .get(&representation.representation_id())
            })
            .flatten()
    }

    pub(super) fn field_binding_for_selected_use(
        &self,
        plan: &RenderPlan,
        object_id: RenderObjectId,
        representation: RenderApplicableRepresentationUse,
    ) -> Option<&RenderFieldSemanticInputBinding> {
        use_declares_field_input(plan, object_id, representation)
            .then(|| {
                self.field_by_representation
                    .get(&representation.representation_id())
            })
            .flatten()
    }
}

fn is_surface_protocol(representation: RenderApplicableRepresentationUse) -> bool {
    matches!(
        representation.requirement().protocol().protocol(),
        RenderRepresentationProtocol::SurfaceQuery
            | RenderRepresentationProtocol::OrientedSurfaceQuery
    )
}

fn use_declares_surface_input(
    plan: &RenderPlan,
    object_id: RenderObjectId,
    representation: RenderApplicableRepresentationUse,
) -> bool {
    if !is_surface_protocol(representation) {
        return false;
    }
    plan.scene()
        .object_participation(object_id)
        .and_then(|participation| participation.representation(representation.representation_id()))
        .and_then(|record| record.surface_semantic_input_requirement())
        .is_some()
}

fn use_declares_field_input(
    plan: &RenderPlan,
    object_id: RenderObjectId,
    representation: RenderApplicableRepresentationUse,
) -> bool {
    if !matches!(
        representation.requirement().protocol().protocol(),
        RenderRepresentationProtocol::FieldDistance
    ) {
        return false;
    }
    plan.scene()
        .object_participation(object_id)
        .and_then(|participation| participation.representation(representation.representation_id()))
        .and_then(|record| record.field_semantic_input_requirement())
        .is_some()
}
