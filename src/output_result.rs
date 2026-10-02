//! R2 per-output-sample result-definedness semantics.
//!
//! This module owns only whether one logical requested-output sample has a renderer-semantic value
//! and how the current R2 output-value families interpret the absence of a represented surface hit.
//! It deliberately does not introduce a `RenderResult` container, numeric payload/storage format,
//! validity mask, reserved object identity, background/environment policy, readback/session state, or
//! another physical result-binding authority.

use super::request::RenderOutputValue;

/// Why one logical requested-output sample has no renderer-semantic value.
///
/// Undefined is not a numeric/object value. Any physical payload bits associated with an undefined
/// sample are non-semantic and must not be interpreted as a distance, radiance, object identity, or
/// other result value merely because they occupy the same storage destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderOutputSampleUndefinedReason {
    /// The requested geometric result had no represented surface hit for this logical sample.
    NoRepresentedSurfaceHit,
}

/// Renderer-semantic definedness of one logical sample for one requested output.
///
/// Definedness is orthogonal to result topology and to physical payload representation. A defined
/// zero is therefore distinct from an undefined sample even if a downstream physical encoding uses
/// zero bits as an otherwise convenient placeholder for the undefined payload. Semantic tolerance
/// applies only to a defined value; it does not turn an undefined sample into a numeric value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderOutputSampleDefinedness {
    Defined,
    Undefined(RenderOutputSampleUndefinedReason),
}

impl RenderOutputSampleDefinedness {
    pub const fn is_defined(self) -> bool {
        matches!(self, Self::Defined)
    }

    pub const fn undefined_reason(self) -> Option<RenderOutputSampleUndefinedReason> {
        match self {
            Self::Defined => None,
            Self::Undefined(reason) => Some(reason),
        }
    }
}

/// Effect of no represented surface hit on one requested output-value family.
///
/// This is semantic meaning, not an instruction to choose a physical sentinel/mask. In particular,
/// `DoesNotDetermineDefinedness` means the geometric event alone says nothing about whether the
/// requested value exists or what it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderOutputNoSurfaceHitSemantics {
    Undefined(RenderOutputSampleUndefinedReason),
    DoesNotDetermineDefinedness,
}

impl RenderOutputValue {
    /// Renderer-semantic effect of a logical sample having no represented surface hit.
    ///
    /// Distance/depth and renderer object identity require a represented geometric point/object, so
    /// they are undefined on a miss. Radiance is different: a miss does not inherently mean black
    /// or undefined because environment, emission, volume, or other transport semantics may still
    /// define radiance. The geometric miss therefore does not determine radiance definedness.
    pub const fn no_surface_hit_semantics(self) -> RenderOutputNoSurfaceHitSemantics {
        match self {
            Self::Radiance { .. } => RenderOutputNoSurfaceHitSemantics::DoesNotDetermineDefinedness,
            Self::Distance { .. } | Self::ObjectIdentity => {
                RenderOutputNoSurfaceHitSemantics::Undefined(
                    RenderOutputSampleUndefinedReason::NoRepresentedSurfaceHit,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::render::request::{
        RenderDistanceConvention, RenderRadiometricRepresentation,
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ProofPhysicalWord {
        payload: u32,
        defined: bool,
    }

    fn proof_encode(payload: u32, definedness: RenderOutputSampleDefinedness) -> ProofPhysicalWord {
        ProofPhysicalWord {
            payload,
            defined: definedness.is_defined(),
        }
    }

    #[test]
    fn valid_zero_and_undefined_sample_are_semantically_distinct() {
        let defined_zero = proof_encode(0, RenderOutputSampleDefinedness::Defined);
        let undefined_zero = proof_encode(
            0,
            RenderOutputSampleDefinedness::Undefined(
                RenderOutputSampleUndefinedReason::NoRepresentedSurfaceHit,
            ),
        );

        assert_eq!(defined_zero.payload, undefined_zero.payload);
        assert!(defined_zero.defined);
        assert!(!undefined_zero.defined);
    }

    #[test]
    fn distance_and_identity_miss_are_undefined_without_value_sentinels() {
        let expected = RenderOutputNoSurfaceHitSemantics::Undefined(
            RenderOutputSampleUndefinedReason::NoRepresentedSurfaceHit,
        );
        assert_eq!(
            RenderOutputValue::Distance {
                convention: RenderDistanceConvention::RayDistance,
            }
            .no_surface_hit_semantics(),
            expected
        );
        assert_eq!(
            RenderOutputValue::Distance {
                convention: RenderDistanceConvention::ObservationForwardDepth,
            }
            .no_surface_hit_semantics(),
            expected
        );
        assert_eq!(
            RenderOutputValue::ObjectIdentity.no_surface_hit_semantics(),
            expected
        );

        let undefined = RenderOutputSampleDefinedness::Undefined(
            RenderOutputSampleUndefinedReason::NoRepresentedSurfaceHit,
        );
        assert_eq!(
            undefined.undefined_reason(),
            Some(RenderOutputSampleUndefinedReason::NoRepresentedSurfaceHit)
        );
    }

    #[test]
    fn radiance_miss_does_not_imply_black_or_undefined() {
        let radiance = RenderOutputValue::Radiance {
            representation: RenderRadiometricRepresentation::spectral_at_wavelength_meters(550e-9)
                .expect("valid proof wavelength"),
        };
        assert_eq!(
            radiance.no_surface_hit_semantics(),
            RenderOutputNoSurfaceHitSemantics::DoesNotDetermineDefinedness
        );
    }
}
