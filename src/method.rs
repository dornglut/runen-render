//! R4 coherent renderer-method contracts.
//!
//! A method contract describes semantic compatibility and abstract future execution requirements.
//! It deliberately contains no current device capabilities, GPU handles, residency, surfaces,
//! pipelines, output bindings, or method-internal pass/work topology.

use super::representation::RenderRepresentationProtocol;
use super::request::{RenderDistanceConvention, RenderObservationSpec, RenderOutputValue};
use super::space_time::{CanonicalF64, RenderSemanticValueError};
use std::error::Error;
use std::fmt;
use std::num::NonZeroU32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderMethodId(NonZeroU32);

impl RenderMethodId {
    /// Runtime renderer-semantic method identity used to correlate conditional plan candidates.
    ///
    /// This is not persistence, wire, source, artifact, or RunenGPU identity.
    pub const fn new(raw: u32) -> Option<Self> {
        match NonZeroU32::new(raw) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    #[cfg(test)]
    pub(crate) const fn raw(self) -> u32 {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RenderObservationKind {
    Perspective,
    Probe,
}

impl RenderObservationKind {
    pub const fn of(observation: RenderObservationSpec) -> Self {
        match observation {
            RenderObservationSpec::Perspective(_) => Self::Perspective,
            RenderObservationSpec::Probe(_) => Self::Probe,
        }
    }
}

/// Non-negative renderer-semantic distance error bound in scene metres.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderDistanceErrorBound {
    meters: CanonicalF64,
}

impl RenderDistanceErrorBound {
    pub fn new(meters: f64) -> Result<Self, RenderMethodValidationError> {
        let meters = CanonicalF64::new(meters, "render_method_distance_error_meters")?;
        if meters.get() < 0.0 {
            return Err(RenderMethodValidationError::NegativeDistanceErrorBound);
        }
        Ok(Self { meters })
    }

    pub fn meters(self) -> f64 {
        self.meters.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderSpectralRadianceSupport {
    minimum_wavelength_meters: CanonicalF64,
    maximum_wavelength_meters: CanonicalF64,
}

impl RenderSpectralRadianceSupport {
    pub fn new(
        minimum_wavelength_meters: f64,
        maximum_wavelength_meters: f64,
    ) -> Result<Self, RenderMethodValidationError> {
        let minimum_wavelength_meters = CanonicalF64::new(
            minimum_wavelength_meters,
            "method_minimum_wavelength_meters",
        )?;
        let maximum_wavelength_meters = CanonicalF64::new(
            maximum_wavelength_meters,
            "method_maximum_wavelength_meters",
        )?;
        if minimum_wavelength_meters.get() <= 0.0
            || maximum_wavelength_meters.get() <= 0.0
            || minimum_wavelength_meters.get() > maximum_wavelength_meters.get()
        {
            return Err(RenderMethodValidationError::InvalidSpectralRange);
        }
        Ok(Self {
            minimum_wavelength_meters,
            maximum_wavelength_meters,
        })
    }

    pub fn contains_wavelength_meters(self, wavelength_meters: f64) -> bool {
        wavelength_meters >= self.minimum_wavelength_meters.get()
            && wavelength_meters <= self.maximum_wavelength_meters.get()
    }

    pub fn range_meters(self) -> (f64, f64) {
        (
            self.minimum_wavelength_meters.get(),
            self.maximum_wavelength_meters.get(),
        )
    }
}

/// One output family/domain supported by a method for one observation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderMethodOutputKind {
    Radiance {
        spectral: RenderSpectralRadianceSupport,
    },
    Distance {
        convention: RenderDistanceConvention,
    },
    ObjectIdentity,
}

impl RenderMethodOutputKind {
    pub fn supports_value(self, value: RenderOutputValue) -> bool {
        match (self, value) {
            (Self::Radiance { spectral }, RenderOutputValue::Radiance { representation }) => {
                spectral.contains_wavelength_meters(representation.wavelength_meters())
            }
            (
                Self::Distance {
                    convention: supported,
                },
                RenderOutputValue::Distance {
                    convention: requested,
                },
            ) => supported == requested,
            (Self::ObjectIdentity, RenderOutputValue::ObjectIdentity) => true,
            _ => false,
        }
    }

    pub const fn same_family(self, value: RenderOutputValue) -> bool {
        matches!(
            (self, value),
            (Self::Radiance { .. }, RenderOutputValue::Radiance { .. })
                | (Self::Distance { .. }, RenderOutputValue::Distance { .. })
                | (Self::ObjectIdentity, RenderOutputValue::ObjectIdentity)
        )
    }

    const fn key(self) -> (u8, u8) {
        match self {
            Self::Radiance { .. } => (0, 0),
            Self::Distance {
                convention: RenderDistanceConvention::RayDistance,
            } => (1, 0),
            Self::Distance {
                convention: RenderDistanceConvention::ObservationForwardDepth,
            } => (1, 1),
            Self::ObjectIdentity => (2, 0),
        }
    }
}

/// Renderer-semantic/model approximation promised by a method when its declared prerequisites hold.
///
/// This contract describes only the semantic/model relation between the requested output and the
/// method's modeled output. It does not establish finite-evaluation fidelity for one concrete
/// execution, satisfy `RenderSemanticTolerance` by itself, or promise any bit-exact/numeric GPU
/// realization. Those concerns require their own evidence later in the execution/result path.
///
/// R4 intentionally has only the bounded distance guarantee required by the founding proof. Other
/// output-specific approximation contracts must be added only when a concrete method requires them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderMethodOutputGuarantee {
    Exact,
    BoundedAbsoluteDistance {
        max_error_meters: RenderDistanceErrorBound,
    },
}

/// Input guarantee a method requires from a field-distance representation.
///
/// This is deliberately distinct from `RenderMethodOutputGuarantee`: a bound on a pointwise field
/// query is not itself a bound on a later rendered depth/distance result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderFieldDistanceInputRequirement {
    Exact,
    /// Accept any explicitly finite conservative FieldDistance guarantee.
    ///
    /// This is not unknown/unbounded error: the selected representation still carries one concrete
    /// finite maximum absolute error and current request-scoped bindings must satisfy it.
    Conservative,
    Bounded {
        max_absolute_error_meters: RenderDistanceErrorBound,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderRepresentationProtocolRequirement {
    SurfaceQuery {
        revision: u32,
    },
    OrientedSurfaceQuery {
        revision: u32,
    },
    FieldDistance {
        revision: u32,
        input: RenderFieldDistanceInputRequirement,
    },
}

impl RenderRepresentationProtocolRequirement {
    pub const fn revision(self) -> u32 {
        match self {
            Self::SurfaceQuery { revision }
            | Self::OrientedSurfaceQuery { revision }
            | Self::FieldDistance { revision, .. } => revision,
        }
    }

    pub const fn protocol(self) -> RenderRepresentationProtocol {
        match self {
            Self::SurfaceQuery { .. } => RenderRepresentationProtocol::SurfaceQuery,
            Self::OrientedSurfaceQuery { .. } => RenderRepresentationProtocol::OrientedSurfaceQuery,
            Self::FieldDistance { .. } => RenderRepresentationProtocol::FieldDistance,
        }
    }

    const fn sort_key(self) -> u8 {
        match self {
            Self::SurfaceQuery { .. } => 0,
            Self::OrientedSurfaceQuery { .. } => 1,
            Self::FieldDistance { .. } => 2,
        }
    }
}

/// One representation-side prerequisite under which an output contract is valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderMethodRepresentationRequirement {
    protocol: RenderRepresentationProtocolRequirement,
    maximum_refinement_error_meters: Option<RenderDistanceErrorBound>,
}

impl RenderMethodRepresentationRequirement {
    pub fn new(
        protocol: RenderRepresentationProtocolRequirement,
        maximum_refinement_error_meters: Option<RenderDistanceErrorBound>,
    ) -> Result<Self, RenderMethodValidationError> {
        if protocol.revision() == 0 {
            return Err(RenderMethodValidationError::InvalidProtocolRevision);
        }
        Ok(Self {
            protocol,
            maximum_refinement_error_meters,
        })
    }

    pub const fn protocol(self) -> RenderRepresentationProtocolRequirement {
        self.protocol
    }

    pub const fn maximum_refinement_error(self) -> Option<RenderDistanceErrorBound> {
        self.maximum_refinement_error_meters
    }

    const fn sort_key(self) -> u8 {
        self.protocol.sort_key()
    }
}

/// The semantic relation between one observation kind and one output family for a method.
///
/// Representation requirements are alternatives: each listed requirement is one semantic input
/// form under which this same output guarantee holds. Keeping them on the output contract avoids
/// incorrectly asserting a global observation x output x protocol Cartesian product.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderMethodOutputContract {
    observation_kind: RenderObservationKind,
    output_kind: RenderMethodOutputKind,
    guarantee: RenderMethodOutputGuarantee,
    representation_requirements: Vec<RenderMethodRepresentationRequirement>,
    requires_material_assignment: bool,
}

impl RenderMethodOutputContract {
    pub fn new(
        observation_kind: RenderObservationKind,
        output_kind: RenderMethodOutputKind,
        guarantee: RenderMethodOutputGuarantee,
        mut representation_requirements: Vec<RenderMethodRepresentationRequirement>,
        requires_material_assignment: bool,
    ) -> Result<Self, RenderMethodValidationError> {
        if matches!(
            guarantee,
            RenderMethodOutputGuarantee::BoundedAbsoluteDistance { .. }
        ) && !matches!(output_kind, RenderMethodOutputKind::Distance { .. })
        {
            return Err(
                RenderMethodValidationError::BoundedDistanceGuaranteeRequiresDistanceOutput,
            );
        }

        representation_requirements.sort_by_key(|requirement| requirement.sort_key());
        for pair in representation_requirements.windows(2) {
            if pair[0].protocol().protocol() == pair[1].protocol().protocol() {
                return Err(
                    RenderMethodValidationError::DuplicateRepresentationProtocol {
                        protocol: pair[0].protocol().protocol(),
                    },
                );
            }
        }

        Ok(Self {
            observation_kind,
            output_kind,
            guarantee,
            representation_requirements,
            requires_material_assignment,
        })
    }

    pub const fn observation_kind(&self) -> RenderObservationKind {
        self.observation_kind
    }

    pub const fn output_kind(&self) -> RenderMethodOutputKind {
        self.output_kind
    }

    pub const fn guarantee(&self) -> RenderMethodOutputGuarantee {
        self.guarantee
    }

    pub fn representation_requirements(&self) -> &[RenderMethodRepresentationRequirement] {
        &self.representation_requirements
    }

    pub const fn requires_material_assignment(&self) -> bool {
        self.requires_material_assignment
    }

    fn key(&self) -> (RenderObservationKind, u8, u8) {
        let (family, domain) = self.output_kind.key();
        (self.observation_kind, family, domain)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RenderAbstractExecutionRequirement {
    /// Candidate lowering requires general-purpose parallel programmable work, but does not name a
    /// current device, capability set, pipeline, resource, or submission.
    GeneralParallelWork,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderMethodContract {
    id: RenderMethodId,
    output_contracts: Vec<RenderMethodOutputContract>,
    abstract_execution_requirements: Vec<RenderAbstractExecutionRequirement>,
}

impl RenderMethodContract {
    pub fn new(
        id: RenderMethodId,
        mut output_contracts: Vec<RenderMethodOutputContract>,
        mut abstract_execution_requirements: Vec<RenderAbstractExecutionRequirement>,
    ) -> Result<Self, RenderMethodValidationError> {
        if output_contracts.is_empty() {
            return Err(RenderMethodValidationError::NoOutputs);
        }

        output_contracts.sort_by_key(RenderMethodOutputContract::key);
        for pair in output_contracts.windows(2) {
            if pair[0].key() == pair[1].key() {
                return Err(RenderMethodValidationError::DuplicateOutputContract {
                    observation_kind: pair[0].observation_kind(),
                    output_kind: pair[0].output_kind(),
                });
            }
        }

        abstract_execution_requirements.sort_unstable();
        abstract_execution_requirements.dedup();

        Ok(Self {
            id,
            output_contracts,
            abstract_execution_requirements,
        })
    }

    pub const fn id(&self) -> RenderMethodId {
        self.id
    }

    pub fn output_contracts(&self) -> &[RenderMethodOutputContract] {
        &self.output_contracts
    }

    pub fn supports_observation(&self, observation_kind: RenderObservationKind) -> bool {
        self.output_contracts
            .iter()
            .any(|contract| contract.observation_kind() == observation_kind)
    }

    pub fn output_contract(
        &self,
        observation_kind: RenderObservationKind,
        value: RenderOutputValue,
    ) -> Option<&RenderMethodOutputContract> {
        self.output_contracts.iter().find(|contract| {
            contract.observation_kind() == observation_kind
                && contract.output_kind().supports_value(value)
        })
    }

    pub fn has_output_family(
        &self,
        observation_kind: RenderObservationKind,
        value: RenderOutputValue,
    ) -> bool {
        self.output_contracts.iter().any(|contract| {
            contract.observation_kind() == observation_kind
                && contract.output_kind().same_family(value)
        })
    }

    pub fn abstract_execution_requirements(&self) -> &[RenderAbstractExecutionRequirement] {
        &self.abstract_execution_requirements
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMethodValidationError {
    SemanticValue(RenderSemanticValueError),
    NoOutputs,
    InvalidSpectralRange,
    InvalidProtocolRevision,
    NegativeDistanceErrorBound,
    BoundedDistanceGuaranteeRequiresDistanceOutput,
    DuplicateRepresentationProtocol {
        protocol: RenderRepresentationProtocol,
    },
    DuplicateOutputContract {
        observation_kind: RenderObservationKind,
        output_kind: RenderMethodOutputKind,
    },
}

impl From<RenderSemanticValueError> for RenderMethodValidationError {
    fn from(value: RenderSemanticValueError) -> Self {
        Self::SemanticValue(value)
    }
}

impl fmt::Display for RenderMethodValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SemanticValue(error) => fmt::Display::fmt(error, formatter),
            Self::NoOutputs => formatter.write_str("render method must support an output"),
            Self::InvalidSpectralRange => formatter
                .write_str("render method spectral range must be finite, positive, and ordered"),
            Self::InvalidProtocolRevision => {
                formatter.write_str("render method protocol revision must be non-zero")
            }
            Self::NegativeDistanceErrorBound => {
                formatter.write_str("render method distance error bound must be non-negative")
            }
            Self::BoundedDistanceGuaranteeRequiresDistanceOutput => formatter.write_str(
                "bounded distance output guarantee is valid only for distance output semantics",
            ),
            Self::DuplicateRepresentationProtocol { protocol } => write!(
                formatter,
                "render method output contract contains duplicate {protocol:?} representation requirement"
            ),
            Self::DuplicateOutputContract {
                observation_kind,
                output_kind,
            } => write!(
                formatter,
                "render method contains duplicate {observation_kind:?}/{output_kind:?} output contract"
            ),
        }
    }
}

impl Error for RenderMethodValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::representation::{
        RENDER_FIELD_DISTANCE_PROTOCOL_REVISION, RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
        RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
    };

    fn surface_requirement() -> RenderMethodRepresentationRequirement {
        RenderMethodRepresentationRequirement::new(
            RenderRepresentationProtocolRequirement::SurfaceQuery {
                revision: RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            },
            None,
        )
        .expect("surface requirement")
    }

    fn oriented_surface_requirement() -> RenderMethodRepresentationRequirement {
        RenderMethodRepresentationRequirement::new(
            RenderRepresentationProtocolRequirement::OrientedSurfaceQuery {
                revision: RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
            },
            None,
        )
        .expect("oriented surface requirement")
    }

    fn field_requirement() -> RenderMethodRepresentationRequirement {
        RenderMethodRepresentationRequirement::new(
            RenderRepresentationProtocolRequirement::FieldDistance {
                revision: RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
                input: RenderFieldDistanceInputRequirement::Bounded {
                    max_absolute_error_meters: RenderDistanceErrorBound::new(0.05)
                        .expect("field input bound"),
                },
            },
            Some(RenderDistanceErrorBound::new(0.05).expect("refinement bound")),
        )
        .expect("field requirement")
    }

    #[test]
    fn method_contract_canonicalizes_output_and_representation_relations() {
        let distance = RenderMethodOutputContract::new(
            RenderObservationKind::Perspective,
            RenderMethodOutputKind::Distance {
                convention: RenderDistanceConvention::RayDistance,
            },
            RenderMethodOutputGuarantee::Exact,
            vec![
                field_requirement(),
                oriented_surface_requirement(),
                surface_requirement(),
            ],
            false,
        )
        .expect("distance output contract");
        let radiance = RenderMethodOutputContract::new(
            RenderObservationKind::Probe,
            RenderMethodOutputKind::Radiance {
                spectral: RenderSpectralRadianceSupport::new(400e-9, 700e-9)
                    .expect("spectral support"),
            },
            RenderMethodOutputGuarantee::Exact,
            vec![surface_requirement()],
            true,
        )
        .expect("radiance output contract");

        let method = RenderMethodContract::new(
            RenderMethodId::new(7).expect("method id"),
            vec![radiance, distance],
            vec![
                RenderAbstractExecutionRequirement::GeneralParallelWork,
                RenderAbstractExecutionRequirement::GeneralParallelWork,
            ],
        )
        .expect("method");

        assert_eq!(method.id().raw(), 7);
        assert_eq!(method.output_contracts().len(), 2);
        assert_eq!(method.abstract_execution_requirements().len(), 1);
        let perspective = &method.output_contracts()[0];
        assert_eq!(
            perspective.observation_kind(),
            RenderObservationKind::Perspective
        );
        assert_eq!(perspective.representation_requirements().len(), 3);
        assert_eq!(
            perspective.representation_requirements()[0]
                .protocol()
                .protocol(),
            RenderRepresentationProtocol::SurfaceQuery
        );
        assert_eq!(
            perspective.representation_requirements()[1]
                .protocol()
                .protocol(),
            RenderRepresentationProtocol::OrientedSurfaceQuery
        );
        assert_eq!(
            perspective.representation_requirements()[2]
                .protocol()
                .protocol(),
            RenderRepresentationProtocol::FieldDistance
        );
    }

    #[test]
    fn observation_output_relation_does_not_imply_cartesian_product() {
        let perspective_distance = RenderMethodOutputContract::new(
            RenderObservationKind::Perspective,
            RenderMethodOutputKind::Distance {
                convention: RenderDistanceConvention::ObservationForwardDepth,
            },
            RenderMethodOutputGuarantee::Exact,
            vec![surface_requirement()],
            false,
        )
        .expect("perspective distance");
        let probe_radiance = RenderMethodOutputContract::new(
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
        let method = RenderMethodContract::new(
            RenderMethodId::new(1).expect("method id"),
            vec![probe_radiance, perspective_distance],
            Vec::new(),
        )
        .expect("method");

        let probe_distance = RenderOutputValue::Distance {
            convention: RenderDistanceConvention::ObservationForwardDepth,
        };
        assert!(method.supports_observation(RenderObservationKind::Probe));
        assert!(
            method
                .output_contract(RenderObservationKind::Probe, probe_distance)
                .is_none()
        );
        assert!(!method.has_output_family(RenderObservationKind::Probe, probe_distance));
    }

    #[test]
    fn bounded_output_guarantee_is_output_specific() {
        let radiance = RenderMethodOutputKind::Radiance {
            spectral: RenderSpectralRadianceSupport::new(400e-9, 700e-9).expect("spectral support"),
        };
        assert_eq!(
            RenderMethodOutputContract::new(
                RenderObservationKind::Probe,
                radiance,
                RenderMethodOutputGuarantee::BoundedAbsoluteDistance {
                    max_error_meters: RenderDistanceErrorBound::new(0.01).expect("bound"),
                },
                vec![surface_requirement()],
                false,
            ),
            Err(RenderMethodValidationError::BoundedDistanceGuaranteeRequiresDistanceOutput)
        );
    }

    #[test]
    fn spectral_support_is_semantic_and_bounded() {
        let support = RenderSpectralRadianceSupport::new(400e-9, 700e-9)
            .expect("valid visible-like interval");
        assert!(support.contains_wavelength_meters(550e-9));
        assert!(!support.contains_wavelength_meters(800e-9));
        assert_eq!(
            RenderSpectralRadianceSupport::new(700e-9, 400e-9),
            Err(RenderMethodValidationError::InvalidSpectralRange)
        );
    }
}
