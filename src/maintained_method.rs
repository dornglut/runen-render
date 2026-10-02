//! RunenRender-owned contract for the first maintained deterministic render method.
//!
//! The contract is semantic method authority only. It does not expose a method plugin/registry,
//! evaluator object, execution topology, GPU program, proof fixture, or product-facing method family.

use super::method::{
    RenderAbstractExecutionRequirement, RenderFieldDistanceInputRequirement, RenderMethodContract,
    RenderMethodId, RenderMethodOutputContract, RenderMethodOutputGuarantee,
    RenderMethodOutputKind, RenderMethodRepresentationRequirement, RenderObservationKind,
    RenderRepresentationProtocolRequirement, RenderSpectralRadianceSupport,
};
use super::representation::{
    RENDER_FIELD_DISTANCE_PROTOCOL_REVISION, RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
    RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
};
use super::request::RenderDistanceConvention;

pub(super) fn maintained_deterministic_method() -> RenderMethodContract {
    let spectral = RenderSpectralRadianceSupport::new(400e-9, 700e-9)
        .expect("maintained deterministic spectral range is valid");
    let oriented_surface = || {
        RenderMethodRepresentationRequirement::new(
            RenderRepresentationProtocolRequirement::OrientedSurfaceQuery {
                revision: RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
            },
            None,
        )
        .expect("maintained oriented-surface requirement is valid")
    };
    let surface = || {
        RenderMethodRepresentationRequirement::new(
            RenderRepresentationProtocolRequirement::SurfaceQuery {
                revision: RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
            },
            None,
        )
        .expect("maintained surface requirement is valid")
    };
    let field = || {
        RenderMethodRepresentationRequirement::new(
            RenderRepresentationProtocolRequirement::FieldDistance {
                revision: RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
                input: RenderFieldDistanceInputRequirement::Conservative,
            },
            None,
        )
        .expect("maintained conservative field-distance requirement is valid")
    };
    let outputs = vec![
        RenderMethodOutputContract::new(
            RenderObservationKind::Perspective,
            RenderMethodOutputKind::Radiance { spectral },
            RenderMethodOutputGuarantee::Exact,
            vec![oriented_surface(), field()],
            true,
        )
        .expect("maintained perspective-radiance contract is valid"),
        RenderMethodOutputContract::new(
            RenderObservationKind::Perspective,
            RenderMethodOutputKind::Distance {
                convention: RenderDistanceConvention::ObservationForwardDepth,
            },
            RenderMethodOutputGuarantee::Exact,
            vec![surface(), field()],
            false,
        )
        .expect("maintained perspective-depth contract is valid"),
        RenderMethodOutputContract::new(
            RenderObservationKind::Perspective,
            RenderMethodOutputKind::ObjectIdentity,
            RenderMethodOutputGuarantee::Exact,
            vec![surface(), field()],
            false,
        )
        .expect("maintained perspective-identity contract is valid"),
        RenderMethodOutputContract::new(
            RenderObservationKind::Probe,
            RenderMethodOutputKind::Radiance { spectral },
            RenderMethodOutputGuarantee::Exact,
            vec![oriented_surface()],
            true,
        )
        .expect("maintained probe-radiance contract is valid"),
    ];
    RenderMethodContract::new(
        RenderMethodId::new(1).expect("maintained method identity is non-zero"),
        outputs,
        vec![RenderAbstractExecutionRequirement::GeneralParallelWork],
    )
    .expect("maintained deterministic method contract is internally valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maintained_method_contract_is_stable_and_not_request_shaped() {
        let method = maintained_deterministic_method();
        assert_eq!(method.id(), RenderMethodId::new(1).unwrap());
        assert_eq!(method.output_contracts().len(), 4);
        assert_eq!(
            method.abstract_execution_requirements(),
            &[RenderAbstractExecutionRequirement::GeneralParallelWork]
        );
        for contract in method
            .output_contracts()
            .iter()
            .filter(|contract| contract.observation_kind() == RenderObservationKind::Perspective)
        {
            assert!(
                contract
                    .representation_requirements()
                    .iter()
                    .any(|requirement| {
                        matches!(
                            requirement.protocol(),
                            RenderRepresentationProtocolRequirement::FieldDistance {
                                input: RenderFieldDistanceInputRequirement::Conservative,
                                ..
                            }
                        )
                    })
            );
        }
        let probe = method
            .output_contracts()
            .iter()
            .find(|contract| contract.observation_kind() == RenderObservationKind::Probe)
            .expect("maintained probe contract");
        assert!(
            probe
                .representation_requirements()
                .iter()
                .all(|requirement| {
                    !matches!(
                        requirement.protocol(),
                        RenderRepresentationProtocolRequirement::FieldDistance { .. }
                    )
                })
        );
    }
}
