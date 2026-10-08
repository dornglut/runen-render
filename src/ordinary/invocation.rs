use super::*;
use std::collections::BTreeSet;

/// All renderer-owned facts for one bounded ordinary invocation.
///
/// This source-neutral value does not own GPU context, frame policy, producer identity, or
/// presentation. Scene, request, semantic bindings, availability, and physical destinations
/// retain their own distinct semantic owners and are validated at their respective boundaries.
#[derive(Debug, Clone)]
pub struct RenderInvocation {
    scene: RenderSceneSnapshot,
    request: RenderRequest,
    surface_inputs: Vec<RenderSurfaceSemanticInputBinding>,
    field_inputs: Vec<RenderFieldSemanticInputBinding>,
    availability: Vec<RenderRepresentationAvailabilityFact>,
    output_bindings: Vec<RenderOutputBinding>,
}

/// Structural output-binding failure prior to GPU-dependent semantic admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderInvocationError {
    ForeignOutput {
        output: crate::request::RenderOutputHandle,
    },
    DuplicateOutput {
        position: usize,
    },
    MissingOutput {
        position: usize,
    },
}

impl fmt::Display for RenderInvocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignOutput { output } => write!(
                formatter,
                "output handle at position {} is not owned by this request",
                output.position()
            ),
            Self::DuplicateOutput { position } => {
                write!(
                    formatter,
                    "output position {position} has multiple physical bindings"
                )
            }
            Self::MissingOutput { position } => {
                write!(
                    formatter,
                    "requested output position {position} has no physical binding"
                )
            }
        }
    }
}

impl Error for RenderInvocationError {}

impl RenderInvocation {
    /// Construct the renderer invocation after physical output destinations are available.
    ///
    /// Semantic admission and GPU-dependent destination validation remain the renderer's
    /// canonical planning/admission responsibility. Construction proves the bounded mapping
    /// includes each request output exactly once, regardless of caller-provided binding order.
    pub fn new(
        scene: RenderSceneSnapshot,
        request: RenderRequest,
        surface_inputs: Vec<RenderSurfaceSemanticInputBinding>,
        field_inputs: Vec<RenderFieldSemanticInputBinding>,
        availability: Vec<RenderRepresentationAvailabilityFact>,
        output_bindings: Vec<RenderOutputBinding>,
    ) -> Result<Self, RenderInvocationError> {
        let output_count = request.outputs().len();
        let mut seen = BTreeSet::new();
        for binding in &output_bindings {
            if !request.contains_output(binding.output()) {
                return Err(RenderInvocationError::ForeignOutput {
                    output: binding.output().clone(),
                });
            }
            let position = binding.output_index();
            if !seen.insert(position) {
                return Err(RenderInvocationError::DuplicateOutput { position });
            }
        }
        for position in 0..output_count {
            if !seen.contains(&position) {
                return Err(RenderInvocationError::MissingOutput { position });
            }
        }
        Ok(Self {
            scene,
            request,
            surface_inputs,
            field_inputs,
            availability,
            output_bindings,
        })
    }

    pub const fn scene(&self) -> &RenderSceneSnapshot {
        &self.scene
    }

    pub const fn request(&self) -> &RenderRequest {
        &self.request
    }

    pub fn surface_inputs(&self) -> &[RenderSurfaceSemanticInputBinding] {
        &self.surface_inputs
    }

    pub fn field_inputs(&self) -> &[RenderFieldSemanticInputBinding] {
        &self.field_inputs
    }

    pub fn availability(&self) -> &[RenderRepresentationAvailabilityFact] {
        &self.availability
    }

    pub fn output_bindings(&self) -> &[RenderOutputBinding] {
        &self.output_bindings
    }
}
