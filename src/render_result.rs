use super::admission::AdmittedRenderPlan;
use super::field_input::RenderFieldSemanticInputBinding;
use super::method::RenderMethodId;
use super::request::{RenderOutputHandle, RenderRequest};
use super::scene::{RenderObjectId, RenderSceneRevision, RenderSceneSnapshot};
use super::semantic_plan::{RenderApplicableRepresentationUse, RenderOutputApproximation};
use super::surface_input::RenderSurfaceSemanticInputBinding;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Private renderer-owned evidence that one exact admitted execution completed every requested
/// output inside the result-formation contract.
///
/// Method-specific verification owns when this witness may be minted. The semantic result layer
/// consumes only this generic evidence and therefore does not depend on any maintained-method
/// verifier type. The admitted plan is retained by value so output-completion evidence cannot be
/// detached from the exact scene/request/binding authority that was verified.
#[derive(Debug)]
pub(super) struct RenderResultFormationEvidence {
    admitted: AdmittedRenderPlan,
    completed_outputs: BTreeSet<usize>,
}

impl RenderResultFormationEvidence {
    pub(super) fn complete(
        admitted: &AdmittedRenderPlan,
        output_indices: impl IntoIterator<Item = usize>,
    ) -> Result<Self, RenderResultFormationError> {
        let output_count = admitted.outputs().len();
        let mut completed_outputs = BTreeSet::new();
        for output_index in output_indices {
            if output_index >= output_count {
                return Err(RenderResultFormationError::OutputOutOfRange {
                    output_index,
                    output_count,
                });
            }
            if !completed_outputs.insert(output_index) {
                return Err(RenderResultFormationError::DuplicateOutput { output_index });
            }
        }
        for output in admitted.outputs() {
            let output_index = output.output_index();
            if !completed_outputs.contains(&output_index) {
                return Err(RenderResultFormationError::MissingOutput { output_index });
            }
        }
        Ok(Self {
            admitted: admitted.clone(),
            completed_outputs,
        })
    }
}

/// Immutable renderer-semantic representation provenance retained for one completed output.
///
/// Physical bindings, GPU execution evidence, readback identities, and output bytes deliberately do
/// not participate in this semantic result evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderResultObjectRepresentation {
    object_id: RenderObjectId,
    representation: RenderApplicableRepresentationUse,
}

impl RenderResultObjectRepresentation {
    pub const fn object_id(&self) -> RenderObjectId {
        self.object_id
    }

    pub const fn representation(&self) -> RenderApplicableRepresentationUse {
        self.representation
    }
}

/// Semantic evidence for one successfully formed requested output.
///
/// Output-to-observation correlation is intentionally not duplicated here. `RenderResult` retains
/// the exact immutable `RenderRequest`, so callers derive that relation through the requested output
/// at `output_index`.
#[derive(Debug, Clone)]
pub struct RenderResultOutputEvidence {
    output: RenderOutputHandle,
    approximation: RenderOutputApproximation,
    object_representations: Vec<RenderResultObjectRepresentation>,
}

impl PartialEq for RenderResultOutputEvidence {
    fn eq(&self, other: &Self) -> bool {
        self.output.position() == other.output.position()
            && self.approximation == other.approximation
            && self.object_representations == other.object_representations
    }
}

impl Eq for RenderResultOutputEvidence {}

impl RenderResultOutputEvidence {
    /// Exact request-local output correlation, distinct from deterministic output order.
    pub const fn output(&self) -> &RenderOutputHandle {
        &self.output
    }

    pub const fn approximation(&self) -> RenderOutputApproximation {
        self.approximation
    }

    pub fn object_representations(&self) -> &[RenderResultObjectRepresentation] {
        &self.object_representations
    }
}

/// Complete RunenRender semantic outcome evidence for one admitted execution.
///
/// This is intentionally not a value container. Output values may remain in physical bindings,
/// retained renderer products, readback results, or presentation destinations. The result retains
/// only immutable semantic provenance projected from the exact admitted plan that produced the work,
/// including the exact selected request-scoped semantic surface and field inputs once at result
/// level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderResult {
    scene: RenderSceneSnapshot,
    request: RenderRequest,
    method_id: RenderMethodId,
    surface_semantic_inputs: Vec<RenderSurfaceSemanticInputBinding>,
    field_semantic_inputs: Vec<RenderFieldSemanticInputBinding>,
    outputs: Vec<RenderResultOutputEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RenderResultFormationError {
    OutputOutOfRange {
        output_index: usize,
        output_count: usize,
    },
    DuplicateOutput {
        output_index: usize,
    },
    MissingOutput {
        output_index: usize,
    },
}

impl fmt::Display for RenderResultFormationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutputOutOfRange {
                output_index,
                output_count,
            } => write!(
                formatter,
                "formed render output {output_index} is outside {output_count} admitted outputs"
            ),
            Self::DuplicateOutput { output_index } => {
                write!(
                    formatter,
                    "render output {output_index} was formed more than once"
                )
            }
            Self::MissingOutput { output_index } => {
                write!(formatter, "render output {output_index} was not formed")
            }
        }
    }
}

impl Error for RenderResultFormationError {}

impl RenderResult {
    /// Form semantic result provenance from one private generic completion witness.
    ///
    /// Physical submission/readback identity and method-specific verification machinery have already
    /// been consumed before this boundary. The witness retains the exact admitted semantic authority
    /// by value and is not public or caller-mintable.
    pub(super) fn from_formation_evidence(evidence: RenderResultFormationEvidence) -> Self {
        let RenderResultFormationEvidence {
            admitted,
            completed_outputs,
        } = evidence;
        debug_assert_eq!(completed_outputs.len(), admitted.outputs().len());

        let outputs = admitted
            .outputs()
            .iter()
            .map(|output| RenderResultOutputEvidence {
                output: admitted
                    .plan()
                    .request()
                    .output_handle(output.output_index())
                    .expect("admitted request output"),
                approximation: output.approximation(),
                object_representations: output
                    .object_representations()
                    .iter()
                    .map(|object| RenderResultObjectRepresentation {
                        object_id: object.object_id(),
                        representation: object.representation(),
                    })
                    .collect(),
            })
            .collect();

        Self {
            scene: admitted.plan().scene().clone(),
            request: admitted.plan().request().clone(),
            method_id: admitted.selected_candidate().method_id(),
            surface_semantic_inputs: admitted.surface_semantic_inputs().to_vec(),
            field_semantic_inputs: admitted.field_semantic_inputs().to_vec(),
            outputs,
        }
    }

    pub const fn scene_revision(&self) -> RenderSceneRevision {
        self.scene.revision()
    }

    pub const fn scene(&self) -> &RenderSceneSnapshot {
        &self.scene
    }

    pub const fn request(&self) -> &RenderRequest {
        &self.request
    }

    pub const fn method_id(&self) -> RenderMethodId {
        self.method_id
    }

    pub fn surface_semantic_inputs(&self) -> &[RenderSurfaceSemanticInputBinding] {
        &self.surface_semantic_inputs
    }

    pub fn field_semantic_inputs(&self) -> &[RenderFieldSemanticInputBinding] {
        &self.field_semantic_inputs
    }

    pub fn outputs(&self) -> &[RenderResultOutputEvidence] {
        &self.outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{
        RenderObservationSpec, RenderOutputSpec, RenderOutputValue, RenderProbeObservation,
        RenderRequestBuilder, RenderResultTopology, RenderSamplingSupport,
        RenderSemanticTolerance,
    };
    use crate::space_time::{RenderAffineTransform3, RenderTimeInterval, RenderTimePoint};

    fn equivalent_request() -> RenderRequest {
        let shutter = RenderTimeInterval::instant(
            RenderTimePoint::from_seconds(0.0).expect("finite time"),
        );
        let observation = RenderObservationSpec::Probe(
            RenderProbeObservation::new(
                RenderAffineTransform3::identity(),
                shutter,
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("probe"),
        );
        let output = RenderOutputSpec::new(
            RenderOutputValue::ObjectIdentity,
            RenderResultTopology::scalar(),
            RenderSemanticTolerance::exact(),
        )
        .expect("output");
        let mut builder = RenderRequestBuilder::new(shutter);
        let observation = builder.add_observation(observation);
        builder.add_output(&observation, output).expect("own observation");
        builder.finish().expect("valid request")
    }

    #[test]
    fn semantic_result_evidence_equality_does_not_acquire_request_pointer_identity() {
        let first = equivalent_request();
        let second = equivalent_request();
        assert_eq!(first, second);
        let first_handle = first.output_handle(0).expect("first output");
        let second_handle = second.output_handle(0).expect("second output");
        assert_ne!(first_handle, second_handle);
        let evidence = |output| RenderResultOutputEvidence {
            output,
            approximation: RenderOutputApproximation::Exact,
            object_representations: Vec::new(),
        };
        let left = evidence(first_handle);
        let right = evidence(second_handle);
        assert_eq!(left, right);
        assert_ne!(left.output(), right.output());
    }
}
