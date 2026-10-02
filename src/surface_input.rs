//! Request-scoped semantic surface input for representations whose surface protocol depends on a
//! current source/adaptor value.
//!
//! The value is renderer-semantic input, not intrinsic representation evidence, source identity,
//! product scenario identity, or RunenGPU realization. Representation identity is supplied only by
//! [`RenderSurfaceSemanticInputBinding`], allowing the same immutable semantic value shape to be
//! bound independently to distinct representation records when their owning source contracts allow
//! it.

use super::representation::RenderRepresentationId;
use super::space_time::{CanonicalF64, RenderSemanticValueError, RenderTemporalSupport};
use std::error::Error;
use std::fmt;

/// Narrow typed declaration that one surface protocol requires a current request-scoped semantic
/// surface value before the representation use is semantically admissible.
///
/// This marker intentionally has no registry key, provider identity, generation, GPU handle, or
/// dynamic type token. A representation that can satisfy the same protocol intrinsically simply
/// omits this prerequisite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderSurfaceSemanticInputRequirement {
    _private: (),
}

impl RenderSurfaceSemanticInputRequirement {
    pub const fn current() -> Self {
        Self { _private: () }
    }
}

/// Immutable request-scoped surface semantic value.
///
/// The public construction surface is deliberately limited to the concrete pressure demonstrated by
/// the maintained deterministic renderer. This is not a closed representation-family ontology.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderSurfaceSemanticInput {
    kind: RenderSurfaceSemanticInputKind,
    validity: RenderTemporalSupport,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RenderSurfaceSemanticInputKind {
    Sphere {
        center_local_units: [CanonicalF64; 3],
        radius_local_units: CanonicalF64,
    },
    Plane {
        point_local_units: [CanonicalF64; 3],
        normal_local: [CanonicalF64; 3],
    },
}

/// Crate-private value view used only by the maintained evaluator's physical realization.
///
/// Keeping this view non-public prevents the current sphere/plane pressure from becoming a closed
/// representation-family ontology. The public semantic input remains the sole source of truth; this
/// is a lossless projection for renderer-owned execution code, not independent semantic state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum RenderSurfaceSemanticInputView {
    Sphere {
        center_local_units: [f64; 3],
        radius_local_units: f64,
    },
    Plane {
        point_local_units: [f64; 3],
        normal_local: [f64; 3],
    },
}

impl RenderSurfaceSemanticInput {
    pub fn sphere(
        center_local_units: [f64; 3],
        radius_local_units: f64,
        validity: RenderTemporalSupport,
    ) -> Result<Self, RenderSurfaceSemanticInputError> {
        Ok(Self {
            kind: RenderSurfaceSemanticInputKind::Sphere {
                center_local_units: canonical_point(
                    center_local_units,
                    "surface_input_sphere_center",
                )?,
                radius_local_units: positive_value(
                    radius_local_units,
                    "surface_input_sphere_radius",
                )?,
            },
            validity,
        })
    }

    pub fn plane(
        point_local_units: [f64; 3],
        normal_local: [f64; 3],
        validity: RenderTemporalSupport,
    ) -> Result<Self, RenderSurfaceSemanticInputError> {
        Ok(Self {
            kind: RenderSurfaceSemanticInputKind::Plane {
                point_local_units: canonical_point(point_local_units, "surface_input_plane_point")?,
                normal_local: canonical_unit_direction(normal_local, "surface_input_plane_normal")?,
            },
            validity,
        })
    }

    pub const fn validity(&self) -> RenderTemporalSupport {
        self.validity
    }

    pub(super) fn execution_view(&self) -> RenderSurfaceSemanticInputView {
        match &self.kind {
            RenderSurfaceSemanticInputKind::Sphere {
                center_local_units,
                radius_local_units,
            } => RenderSurfaceSemanticInputView::Sphere {
                center_local_units: center_local_units.map(CanonicalF64::get),
                radius_local_units: radius_local_units.get(),
            },
            RenderSurfaceSemanticInputKind::Plane {
                point_local_units,
                normal_local,
            } => RenderSurfaceSemanticInputView::Plane {
                point_local_units: point_local_units.map(CanonicalF64::get),
                normal_local: normal_local.map(CanonicalF64::get),
            },
        }
    }
}

/// Opaque source-owner generation correlated with one request-scoped semantic input binding.
///
/// The number is meaningful only within the source contract that publishes the binding for one
/// representation identity. RunenRender compares it for dependency validity but does not interpret
/// ordering, synthesize it from scene/frame/GPU state, or treat it as renderer-owned revision truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderSurfaceSemanticInputGeneration(u64);

impl RenderSurfaceSemanticInputGeneration {
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }
}

/// Invocation-local correlation between one exact representation identity and one immutable current
/// semantic surface value.
///
/// Existing consumers may omit source-generation evidence when they do not retain derived state
/// across invocations. Consumers whose correctness depends on retained semantic-input validity must
/// require an explicit generation and fail closed when it is absent.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderSurfaceSemanticInputBinding {
    representation_id: RenderRepresentationId,
    input: RenderSurfaceSemanticInput,
    generation: Option<RenderSurfaceSemanticInputGeneration>,
}

impl RenderSurfaceSemanticInputBinding {
    pub fn new(
        representation_id: RenderRepresentationId,
        input: RenderSurfaceSemanticInput,
    ) -> Self {
        Self {
            representation_id,
            input,
            generation: None,
        }
    }

    pub const fn with_generation(
        mut self,
        generation: RenderSurfaceSemanticInputGeneration,
    ) -> Self {
        self.generation = Some(generation);
        self
    }

    pub const fn representation_id(&self) -> RenderRepresentationId {
        self.representation_id
    }

    pub const fn input(&self) -> &RenderSurfaceSemanticInput {
        &self.input
    }

    pub const fn generation(&self) -> Option<RenderSurfaceSemanticInputGeneration> {
        self.generation
    }

    pub fn into_input(self) -> RenderSurfaceSemanticInput {
        self.input
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderSurfaceSemanticInputError {
    SemanticValue(RenderSemanticValueError),
    NonPositiveRadius,
    ZeroNormal,
}

impl From<RenderSemanticValueError> for RenderSurfaceSemanticInputError {
    fn from(value: RenderSemanticValueError) -> Self {
        Self::SemanticValue(value)
    }
}

impl fmt::Display for RenderSurfaceSemanticInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SemanticValue(error) => fmt::Display::fmt(error, formatter),
            Self::NonPositiveRadius => {
                write!(formatter, "surface-input sphere radius must be positive")
            }
            Self::ZeroNormal => write!(formatter, "surface-input plane normal must be non-zero"),
        }
    }
}

impl Error for RenderSurfaceSemanticInputError {}

fn canonical_point(
    values: [f64; 3],
    field: &'static str,
) -> Result<[CanonicalF64; 3], RenderSurfaceSemanticInputError> {
    Ok([
        CanonicalF64::new(values[0], field)?,
        CanonicalF64::new(values[1], field)?,
        CanonicalF64::new(values[2], field)?,
    ])
}

fn positive_value(
    value: f64,
    field: &'static str,
) -> Result<CanonicalF64, RenderSurfaceSemanticInputError> {
    let value = CanonicalF64::new(value, field)?;
    if value.get() <= 0.0 {
        return Err(RenderSurfaceSemanticInputError::NonPositiveRadius);
    }
    Ok(value)
}

fn canonical_unit_direction(
    direction: [f64; 3],
    field: &'static str,
) -> Result<[CanonicalF64; 3], RenderSurfaceSemanticInputError> {
    let values = [
        CanonicalF64::new(direction[0], field)?.get(),
        CanonicalF64::new(direction[1], field)?.get(),
        CanonicalF64::new(direction[2], field)?.get(),
    ];
    let normalized = normalize(values).ok_or(RenderSurfaceSemanticInputError::ZeroNormal)?;
    Ok([
        CanonicalF64::new(normalized[0], field)?,
        CanonicalF64::new(normalized[1], field)?,
        CanonicalF64::new(normalized[2], field)?,
    ])
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let magnitude_squared = dot(vector, vector);
    if magnitude_squared <= 0.0 || !magnitude_squared.is_finite() {
        return None;
    }
    let magnitude = magnitude_squared.sqrt();
    if !magnitude.is_finite() {
        return None;
    }
    Some(scale(vector, magnitude.recip()))
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}
