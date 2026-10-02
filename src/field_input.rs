//! Request-scoped semantic sampled field-distance input.
//!
//! This module is deliberately parallel to `surface_input`: it carries one concrete renderer-semantic
//! value family required by the maintained exact field-distance evaluator without creating a generic
//! dynamic-input registry, source-domain product ontology, or GPU-resource contract.

use super::representation::RenderRepresentationId;
use super::space_time::{CanonicalF64, RenderSemanticValueError, RenderTemporalSupport};
use std::error::Error;
use std::fmt;

/// Narrow typed declaration that one field-distance protocol requires a current request-scoped
/// sampled field value before that representation use is semantically admissible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderFieldSemanticInputRequirement {
    _private: (),
}

impl RenderFieldSemanticInputRequirement {
    pub const fn current() -> Self {
        Self { _private: () }
    }
}

/// Finite sampled signed-distance estimates in representation-local metric space.
///
/// The current binding carries one conservative pointwise query-error bound for deterministic
/// trilinear reconstruction. Samples use x-fastest dense ordering:
/// `index = z * (width * height) + y * width + x`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderFieldSemanticInput {
    origin_local_meters: [CanonicalF64; 3],
    sample_spacing_meters: [CanonicalF64; 3],
    dimensions: [u32; 3],
    signed_distance_samples_meters: Vec<CanonicalF64>,
    max_absolute_query_error_local_meters: CanonicalF64,
    validity: RenderTemporalSupport,
}

impl RenderFieldSemanticInput {
    pub fn dense(
        origin_local_meters: [f64; 3],
        sample_spacing_meters: [f64; 3],
        dimensions: [u32; 3],
        signed_distance_samples_meters: Vec<f64>,
        max_absolute_query_error_local_meters: f64,
        validity: RenderTemporalSupport,
    ) -> Result<Self, RenderFieldSemanticInputError> {
        let origin_local_meters =
            canonical_vector(origin_local_meters, "field_input_origin_local_meters")?;
        let sample_spacing_meters =
            positive_vector(sample_spacing_meters, "field_input_sample_spacing_meters")?;
        let expected_sample_count = checked_sample_count(dimensions)?;
        validate_finite_bounds(origin_local_meters, sample_spacing_meters, dimensions)?;
        if signed_distance_samples_meters.len() != expected_sample_count {
            return Err(RenderFieldSemanticInputError::SampleCountMismatch {
                expected: expected_sample_count,
                actual: signed_distance_samples_meters.len(),
            });
        }
        let signed_distance_samples_meters = signed_distance_samples_meters
            .into_iter()
            .map(|value| CanonicalF64::new(value, "field_input_signed_distance_sample_meters"))
            .collect::<Result<Vec<_>, _>>()?;
        let max_absolute_query_error_local_meters = CanonicalF64::new(
            max_absolute_query_error_local_meters,
            "field_input_max_absolute_query_error_local_meters",
        )?;
        if max_absolute_query_error_local_meters.get() < 0.0 {
            return Err(RenderFieldSemanticInputError::NegativeQueryErrorBound);
        }
        Ok(Self {
            origin_local_meters,
            sample_spacing_meters,
            dimensions,
            signed_distance_samples_meters,
            max_absolute_query_error_local_meters,
            validity,
        })
    }

    pub fn origin_local_meters(&self) -> [f64; 3] {
        self.origin_local_meters.map(CanonicalF64::get)
    }

    pub fn sample_spacing_meters(&self) -> [f64; 3] {
        self.sample_spacing_meters.map(CanonicalF64::get)
    }

    pub const fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }

    pub fn max_local_meters(&self) -> [f64; 3] {
        let origin = self.origin_local_meters();
        let spacing = self.sample_spacing_meters();
        [
            origin[0] + spacing[0] * f64::from(self.dimensions[0] - 1),
            origin[1] + spacing[1] * f64::from(self.dimensions[1] - 1),
            origin[2] + spacing[2] * f64::from(self.dimensions[2] - 1),
        ]
    }

    pub fn sample_count(&self) -> usize {
        self.signed_distance_samples_meters.len()
    }

    pub fn signed_distance_sample_meters(&self, index: usize) -> Option<f64> {
        self.signed_distance_samples_meters
            .get(index)
            .copied()
            .map(CanonicalF64::get)
    }

    pub fn max_absolute_query_error_local_meters(&self) -> f64 {
        self.max_absolute_query_error_local_meters.get()
    }

    pub const fn validity(&self) -> RenderTemporalSupport {
        self.validity
    }
}

/// Opaque source-owner generation correlated with one sampled field binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderFieldSemanticInputGeneration(u64);

impl RenderFieldSemanticInputGeneration {
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }
}

/// Invocation-local correlation between one exact representation and one immutable sampled field.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderFieldSemanticInputBinding {
    representation_id: RenderRepresentationId,
    input: RenderFieldSemanticInput,
    generation: Option<RenderFieldSemanticInputGeneration>,
}

impl RenderFieldSemanticInputBinding {
    pub fn new(representation_id: RenderRepresentationId, input: RenderFieldSemanticInput) -> Self {
        Self {
            representation_id,
            input,
            generation: None,
        }
    }

    pub const fn with_generation(mut self, generation: RenderFieldSemanticInputGeneration) -> Self {
        self.generation = Some(generation);
        self
    }

    pub const fn representation_id(&self) -> RenderRepresentationId {
        self.representation_id
    }

    pub const fn input(&self) -> &RenderFieldSemanticInput {
        &self.input
    }

    pub const fn generation(&self) -> Option<RenderFieldSemanticInputGeneration> {
        self.generation
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderFieldSemanticInputError {
    SemanticValue(RenderSemanticValueError),
    DegenerateDimensions,
    SampleCountOverflow,
    SampleCountMismatch { expected: usize, actual: usize },
    NonPositiveSampleSpacing,
    NegativeQueryErrorBound,
}

impl From<RenderSemanticValueError> for RenderFieldSemanticInputError {
    fn from(value: RenderSemanticValueError) -> Self {
        Self::SemanticValue(value)
    }
}

impl fmt::Display for RenderFieldSemanticInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SemanticValue(error) => fmt::Display::fmt(error, formatter),
            Self::DegenerateDimensions => formatter
                .write_str("field-input dimensions must contain at least two samples per axis"),
            Self::SampleCountOverflow => {
                formatter.write_str("field-input dimensions overflow the addressable sample count")
            }
            Self::SampleCountMismatch { expected, actual } => write!(
                formatter,
                "field-input sample count mismatch: expected {expected}, found {actual}"
            ),
            Self::NonPositiveSampleSpacing => {
                formatter.write_str("field-input sample spacing must be positive on every axis")
            }
            Self::NegativeQueryErrorBound => {
                formatter.write_str("field-input maximum absolute query error must be non-negative")
            }
        }
    }
}

impl Error for RenderFieldSemanticInputError {}

fn checked_sample_count(dimensions: [u32; 3]) -> Result<usize, RenderFieldSemanticInputError> {
    if dimensions.iter().any(|dimension| *dimension < 2) {
        return Err(RenderFieldSemanticInputError::DegenerateDimensions);
    }
    dimensions
        .into_iter()
        .try_fold(1_usize, |count, dimension| {
            count.checked_mul(dimension as usize)
        })
        .ok_or(RenderFieldSemanticInputError::SampleCountOverflow)
}

fn validate_finite_bounds(
    origin_local_meters: [CanonicalF64; 3],
    sample_spacing_meters: [CanonicalF64; 3],
    dimensions: [u32; 3],
) -> Result<(), RenderFieldSemanticInputError> {
    for axis in 0..3 {
        let max_local_meters = origin_local_meters[axis].get()
            + sample_spacing_meters[axis].get() * f64::from(dimensions[axis] - 1);
        CanonicalF64::new(max_local_meters, "field_input_max_local_meters")?;
    }
    Ok(())
}

fn canonical_vector(
    values: [f64; 3],
    field: &'static str,
) -> Result<[CanonicalF64; 3], RenderFieldSemanticInputError> {
    Ok([
        CanonicalF64::new(values[0], field)?,
        CanonicalF64::new(values[1], field)?,
        CanonicalF64::new(values[2], field)?,
    ])
}

fn positive_vector(
    values: [f64; 3],
    field: &'static str,
) -> Result<[CanonicalF64; 3], RenderFieldSemanticInputError> {
    let values = canonical_vector(values, field)?;
    if values.iter().any(|value| value.get() <= 0.0) {
        return Err(RenderFieldSemanticInputError::NonPositiveSampleSpacing);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::render::space_time::RenderTemporalSupport;

    #[test]
    fn dense_field_rejects_degenerate_shape_and_invalid_spacing() {
        let validity = RenderTemporalSupport::unbounded();
        assert!(matches!(
            RenderFieldSemanticInput::dense([0.0; 3], [1.0; 3], [1, 2, 2], vec![], 0.0, validity),
            Err(RenderFieldSemanticInputError::DegenerateDimensions)
        ));
        assert!(matches!(
            RenderFieldSemanticInput::dense(
                [0.0; 3],
                [1.0; 3],
                [2, 2, 2],
                vec![0.0; 7],
                0.0,
                validity,
            ),
            Err(RenderFieldSemanticInputError::SampleCountMismatch {
                expected: 8,
                actual: 7,
            })
        ));
        assert!(matches!(
            RenderFieldSemanticInput::dense(
                [0.0; 3],
                [1.0, 0.0, 1.0],
                [2, 2, 2],
                vec![0.0; 8],
                0.0,
                validity,
            ),
            Err(RenderFieldSemanticInputError::NonPositiveSampleSpacing)
        ));
    }

    #[test]
    fn dense_field_rejects_non_finite_derived_bounds() {
        let validity = RenderTemporalSupport::unbounded();
        assert!(matches!(
            RenderFieldSemanticInput::dense(
                [f64::MAX, 0.0, 0.0],
                [f64::MAX, 1.0, 1.0],
                [2, 2, 2],
                vec![0.0; 8],
                0.0,
                validity,
            ),
            Err(RenderFieldSemanticInputError::SemanticValue(_))
        ));
    }

    #[test]
    fn dense_field_rejects_negative_query_error_bound() {
        assert!(matches!(
            RenderFieldSemanticInput::dense(
                [0.0; 3],
                [1.0; 3],
                [2, 2, 2],
                vec![0.0; 8],
                -0.001,
                RenderTemporalSupport::unbounded(),
            ),
            Err(RenderFieldSemanticInputError::NegativeQueryErrorBound)
        ));
    }

    #[test]
    fn dense_field_preserves_metric_values() {
        let input = RenderFieldSemanticInput::dense(
            [-1.0, -2.0, -3.0],
            [0.5, 1.0, 2.0],
            [2, 2, 2],
            (0..8).map(|value| f64::from(value) * 0.25).collect(),
            0.125,
            RenderTemporalSupport::unbounded(),
        )
        .expect("valid sampled field");
        assert_eq!(input.dimensions(), [2, 2, 2]);
        assert_eq!(input.max_local_meters(), [-0.5, -1.0, -1.0]);
        assert_eq!(input.sample_count(), 8);
        assert_eq!(input.signed_distance_sample_meters(7), Some(1.75));
        assert_eq!(input.max_absolute_query_error_local_meters(), 0.125);
    }
}
