//! Focused Vulkan-capable proof for the maintained R7 deterministic execution seam.
//!
//! The fixture intentionally owns no method or evaluator authority. It reaches execution only
//! through `admit_deterministic_render_with_semantic_inputs`, then proves the maintained ordinary and verified paths use
//! their intended observation policy: ordinary execution authors no readback at all, while verified
//! execution retains exactly the renderer-private canonical-output, definedness, and evaluator-status
//! readbacks from the same exact `GpuSubmission`, can establish the private RR566-EVAL-001 witness,
//! and can form public FORM-001 result evidence through the maintained submitted-render surface.

use super::admission::{
    RenderOutputBinding, RenderOutputDestination, RenderRepresentationAvailabilityFact,
    RenderRepresentationAvailabilityState,
};
use super::appearance::RenderDiffuseMaterial;
use super::deterministic_admission::{
    AdmittedDeterministicRender, admit_deterministic_render_with_semantic_inputs,
};
use super::deterministic_execution::{
    RenderDeterministicRadianceCaptureRequestError, RenderDeterministicResultFormationError,
    SubmittedDeterministicRender, submit_deterministic_render,
    submit_deterministic_render_for_verified_result,
};
use super::deterministic_verification::{
    submit_deterministic_render_for_verified_formation, verify_completed_deterministic_render,
};
use super::participation::RenderMaterialAssignment;
use super::participation::RenderObjectParticipation;
use super::representation::{
    RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION, RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
    RenderOrientedSurfaceProtocolEvidence, RenderRefinementEvidence, RenderRepresentationRecord,
    RenderSurfaceProtocolEvidence,
};
use super::request::{
    RenderDistanceConvention, RenderObservationSpec, RenderOutputSpec, RenderOutputValue,
    RenderPerspectiveObservation, RenderRequest, RenderRequestedOutput, RenderResultTopology,
    RenderSamplingSupport, RenderSemanticTolerance,
};
use super::scene::{RenderObjectState, RenderSceneSnapshot, RenderSceneStore, RenderSceneUpdate};
use super::space_time::{
    RenderAffineTransform3, RenderHandedness, RenderObjectSpatialState, RenderObjectTemporalState,
    RenderSpaceSpec, RenderSpatialCoverage, RenderTemporalSupport, RenderTimeInterval,
    RenderTimePoint,
};
use super::surface_input::{
    RenderSurfaceSemanticInput, RenderSurfaceSemanticInputBinding,
    RenderSurfaceSemanticInputRequirement,
};
use runen_gpu::{
    GpuCapabilityProfile, GpuContext, GpuContextDescriptor, GpuContextRequestErrorCategory,
    GpuCopyExtent, GpuCopyOperation, GpuFormatRole, GpuLimitKind, GpuReadbackId,
    GpuReadbackOperation, GpuReadbackStatus, GpuReconstruction, GpuResourceLifetime, GpuSubmission,
    GpuSubmissionStatus, GpuTextureCopyRegion, GpuTextureDescriptor, GpuTextureFormat,
    GpuTextureInitialization, GpuTextureOrigin, GpuTextureUsage, GpuTransferRegion,
    GpuWorkFragment, GpuWorkResourceIdAllocator,
};
use std::time::{Duration, Instant};

pub(super) struct MaintainedExecutionFixture {
    pub(super) scene: RenderSceneSnapshot,
    pub(super) request: RenderRequest,
    pub(super) semantic_inputs: Vec<RenderSurfaceSemanticInputBinding>,
    pub(super) availability: Vec<RenderRepresentationAvailabilityFact>,
}

fn object_identity_lattice_request(width: u32, height: u32) -> RenderRequest {
    let shutter = instant();
    let observation = RenderObservationSpec::Perspective(
        RenderPerspectiveObservation::new(
            RenderAffineTransform3::identity(),
            std::f64::consts::FRAC_PI_3,
            1.0,
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("R7 maintained perspective observation"),
    );
    RenderRequest::new(
        shutter,
        vec![observation],
        vec![RenderRequestedOutput::new(
            0,
            RenderOutputSpec::new(
                RenderOutputValue::ObjectIdentity,
                RenderResultTopology::sample_lattice_2d(width, height)
                    .expect("R7 maintained object-identity lattice topology"),
                RenderSemanticTolerance::exact(),
            )
            .expect("R7 maintained object-identity output"),
        )],
    )
    .expect("R7 maintained proof request")
}

fn instant() -> RenderTimeInterval {
    RenderTimeInterval::instant(
        RenderTimePoint::from_seconds(0.0).expect("finite R7 maintained proof time"),
    )
}

fn translated_identity_state() -> RenderObjectState {
    RenderObjectState::new(
        RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right)
                .expect("R7 maintained proof local space"),
            RenderAffineTransform3::from_row_major_3x4([
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -3.0,
            ])
            .expect("finite translation-only R7 maintained proof transform"),
            RenderSpatialCoverage::unbounded(),
        ),
        RenderObjectTemporalState::new(RenderTemporalSupport::unbounded()),
    )
}

fn maintained_surface_evidence() -> RenderSurfaceProtocolEvidence {
    RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
        .expect("R7 maintained surface protocol")
        .with_oriented_surface(
            RenderOrientedSurfaceProtocolEvidence::exact(
                RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
            )
            .expect("R7 maintained oriented-surface protocol"),
        )
        .with_semantic_input_requirement(RenderSurfaceSemanticInputRequirement::current())
}

pub(super) fn maintained_fixture() -> MaintainedExecutionFixture {
    let mut store = RenderSceneStore::new();
    let object_id = store
        .allocate_object_id()
        .expect("R7 maintained proof object id");
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(object_id, translated_identity_state());
    store
        .commit(insert)
        .expect("insert R7 maintained proof object");

    let representation_id = store
        .allocate_representation_id(object_id)
        .expect("R7 maintained proof representation id");
    let representation = RenderRepresentationRecord::new(
        representation_id,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
        RenderRefinementEvidence::none(),
        Some(maintained_surface_evidence()),
        None,
    )
    .expect("R7 maintained proof representation");
    let material = RenderMaterialAssignment::new(
        RenderDiffuseMaterial::new(0.5).expect("R7 maintained proof diffuse material"),
    );
    let participation = RenderObjectParticipation::new(vec![representation], Some(material), None)
        .expect("R7 maintained proof participation");
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object_id, participation);
    store
        .commit(attach)
        .expect("attach R7 maintained proof participation");

    let request = object_identity_lattice_request(2, 2);
    let semantic_input = RenderSurfaceSemanticInput::sphere(
        [0.0, 0.0, 0.0],
        2.0,
        RenderTemporalSupport::unbounded(),
    )
    .expect("R7 maintained sphere semantic input");

    MaintainedExecutionFixture {
        scene: store.snapshot(),
        request,
        semantic_inputs: vec![RenderSurfaceSemanticInputBinding::new(
            representation_id,
            semantic_input,
        )],
        availability: vec![RenderRepresentationAvailabilityFact::new(
            representation_id,
            RenderRepresentationAvailabilityState::Available,
        )],
    }
}

fn reordered_8x6_request() -> RenderRequest {
    let shutter = instant();
    let observation = RenderObservationSpec::Perspective(
        RenderPerspectiveObservation::new(
            RenderAffineTransform3::identity(),
            std::f64::consts::FRAC_PI_2 * 1.25,
            8.0 / 6.0,
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("semantically legal wide R7 perspective"),
    );
    let lattice = || {
        RenderResultTopology::sample_lattice_2d(8, 6).expect("R7 maintained structural 8x6 lattice")
    };
    RenderRequest::new(
        shutter,
        vec![observation],
        vec![
            RenderRequestedOutput::new(
                0,
                RenderOutputSpec::new(
                    RenderOutputValue::ObjectIdentity,
                    lattice(),
                    RenderSemanticTolerance::exact(),
                )
                .expect("R7 reordered identity output"),
            ),
            RenderRequestedOutput::new(
                0,
                RenderOutputSpec::new(
                    RenderOutputValue::Distance {
                        convention: RenderDistanceConvention::ObservationForwardDepth,
                    },
                    lattice(),
                    RenderSemanticTolerance::exact(),
                )
                .expect("R7 reordered depth output"),
            ),
        ],
    )
    .expect("R7 reordered 8x6 request")
}

fn radiance_lattice_request() -> RenderRequest {
    let shutter = instant();
    let observation = RenderObservationSpec::Perspective(
        RenderPerspectiveObservation::new(
            RenderAffineTransform3::identity(),
            std::f64::consts::FRAC_PI_3,
            1.0,
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("R7 maintained radiance perspective observation"),
    );
    RenderRequest::new(
        shutter,
        vec![observation],
        vec![RenderRequestedOutput::new(
            0,
            RenderOutputSpec::new(
                RenderOutputValue::Radiance {
                    representation: super::request::RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                        550.0e-9,
                    )
                    .expect("R7 maintained radiance representation"),
                },
                RenderResultTopology::sample_lattice_2d(2, 2)
                    .expect("R7 maintained radiance lattice topology"),
                RenderSemanticTolerance::exact(),
            )
            .expect("R7 maintained radiance output"),
        )],
    )
    .expect("R7 maintained radiance lattice request")
}

fn request_execution_context() -> Option<GpuContext> {
    request_execution_context_with_dispatch_limit(None)
}

fn request_execution_context_with_dispatch_limit(
    max_workgroups_per_dimension: Option<u64>,
) -> Option<GpuContext> {
    let mut descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .require_format_role(GpuTextureFormat::R32Uint, GpuFormatRole::CopyDestination)
            .require_format_role(GpuTextureFormat::R32Uint, GpuFormatRole::CopySource)
            .with_label("RunenRender R7 maintained execution proof");
    if let Some(maximum) = max_workgroups_per_dimension {
        descriptor =
            descriptor.permit_limit(GpuLimitKind::MaxComputeWorkgroupsPerDimension, maximum);
    }
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => Some(context),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1"),
                "permanent R7 maintained execution CI requires a public RunenGPU adapter"
            );
            None
        }
        Err(error) => panic!("unexpected R7 maintained RunenGPU context failure: {error}"),
    }
}

fn admit_with_writable_only_destination(
    fixture: &MaintainedExecutionFixture,
    context: &GpuContext,
    label: &str,
) -> AdmittedDeterministicRender {
    admit_with_writable_only_lattice(fixture, context, label, 2, 2)
}

fn admit_with_writable_only_lattice(
    fixture: &MaintainedExecutionFixture,
    context: &GpuContext,
    label: &str,
    width: u32,
    height: u32,
) -> AdmittedDeterministicRender {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let destination = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::ordinary_owned_2d(
                label,
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                width,
                height,
                GpuTextureFormat::R32Uint,
                [GpuTextureUsage::CopyDestination],
                GpuTextureInitialization::Uninitialized,
            )
            .expect("R7 maintained writable-only lattice descriptor"),
        )
        .expect("R7 maintained writable-only lattice handle");
    let output_bindings = [RenderOutputBinding::new(
        0,
        RenderOutputDestination::SampleLatticeTexture(destination),
    )];
    admit_deterministic_render_with_semantic_inputs(
        &fixture.scene,
        &fixture.request,
        &fixture.semantic_inputs,
        &[],
        &fixture.availability,
        &output_bindings,
        context,
    )
    .expect("R7 maintained fixture must reach maintained deterministic admission")
}

fn admit_with_retained_destination(
    fixture: &MaintainedExecutionFixture,
    request: RenderRequest,
    context: &GpuContext,
    label: &str,
) -> AdmittedDeterministicRender {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let destination = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::ordinary_owned_2d(
                label,
                GpuResourceLifetime::Retained,
                GpuReconstruction::SourceBacked,
                2,
                2,
                GpuTextureFormat::R32Uint,
                [
                    GpuTextureUsage::CopyDestination,
                    GpuTextureUsage::CopySource,
                ],
                GpuTextureInitialization::Uninitialized,
            )
            .expect("R7 maintained retained radiance descriptor"),
        )
        .expect("R7 maintained retained radiance handle");
    let output_bindings = [RenderOutputBinding::new(
        0,
        RenderOutputDestination::SampleLatticeTexture(destination),
    )];
    admit_deterministic_render_with_semantic_inputs(
        &fixture.scene,
        &request,
        &fixture.semantic_inputs,
        &[],
        &fixture.availability,
        &output_bindings,
        context,
    )
    .expect("R7 maintained retained radiance fixture must admit")
}

pub(super) fn admit_with_retained_radiance_destination(
    fixture: &MaintainedExecutionFixture,
    context: &GpuContext,
) -> AdmittedDeterministicRender {
    admit_with_retained_destination(
        fixture,
        radiance_lattice_request(),
        context,
        "R7 maintained retained radiance destination",
    )
}

fn lattice_bindings(width: u32, height: u32, output_count: usize) -> Vec<RenderOutputBinding> {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    (0..output_count)
        .map(|output_index| {
            let destination = allocator
                .allocate_texture_handle(
                    GpuTextureDescriptor::ordinary_owned_2d(
                        format!("R7 structural output {output_index}"),
                        GpuResourceLifetime::Transient,
                        GpuReconstruction::SourceBacked,
                        width,
                        height,
                        GpuTextureFormat::R32Uint,
                        [GpuTextureUsage::CopyDestination],
                        GpuTextureInitialization::Uninitialized,
                    )
                    .expect("R7 structural lattice descriptor"),
                )
                .expect("R7 structural lattice handle");
            RenderOutputBinding::new(
                output_index,
                RenderOutputDestination::SampleLatticeTexture(destination),
            )
        })
        .collect()
}

fn wait_for_submission(context: &GpuContext, submission: &GpuSubmission) {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => return,
            GpuSubmissionStatus::Failed(failure) => {
                panic!("R7 maintained submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "R7 maintained submission did not complete before timeout"
        );
        std::thread::yield_now();
    }
}

fn wait_for_readbacks(
    context: &GpuContext,
    submission: &GpuSubmission,
    readback_ids: &[GpuReadbackId],
) {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        context.progress();
        if let GpuSubmissionStatus::Failed(failure) = submission.status() {
            panic!("R7 maintained verified submission failed: {failure:?}");
        }

        let mut all_ready = true;
        for readback_id in readback_ids {
            let readback = submission
                .readback(*readback_id)
                .expect("R7 maintained correlation must stay on the exact submission");
            match readback.status() {
                GpuReadbackStatus::Ready(_) => {}
                GpuReadbackStatus::Failed(failure) => {
                    panic!("R7 maintained private readback failed: {failure:?}")
                }
                GpuReadbackStatus::Pending => all_ready = false,
            }
        }

        if all_ready && matches!(submission.status(), GpuSubmissionStatus::Completed) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "R7 maintained private readbacks did not become ready before timeout"
        );
        std::thread::yield_now();
    }
}

fn form_verified_render(
    submitted: SubmittedDeterministicRender,
    context: &GpuContext,
) -> SubmittedDeterministicRender {
    let mut submitted = submitted;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        context.progress();
        match submitted.try_form_verified_result() {
            Ok(Some(_)) => return submitted,
            Ok(None) => {}
            Err(error) => panic!("R7 maintained verified result formation failed: {error}"),
        }
        assert!(
            Instant::now() < deadline,
            "R7 maintained verified result formation did not complete before timeout"
        );
        std::thread::yield_now();
    }
}

fn submit_product_readback(
    context: &GpuContext,
    source: GpuTransferRegion,
    readback_id: GpuReadbackId,
) -> GpuSubmission {
    let readback = GpuReadbackOperation::new(source, readback_id)
        .expect("product readback source must be publicly constructible");
    let fragment = GpuWorkFragment::build("R7 negative radiance readback", |work| {
        work.operation("read back radiance lattice", readback)?;
        Ok(())
    })
    .expect("product readback fragment must be authorable");
    let submission =
        pollster::block_on(context.submit_work("R7 negative radiance readback", [fragment]))
            .expect("product readback must submit");
    wait_for_readbacks(context, &submission, &[readback_id]);
    submission
}

fn partial_texture_source(source: &GpuTransferRegion) -> GpuTransferRegion {
    let GpuTransferRegion::Texture(region) = source else {
        panic!("radiance capture source must be a texture region");
    };
    GpuTextureCopyRegion::new(
        region.texture(),
        region.mip_level(),
        GpuTextureOrigin::new(0, 0, 0),
        region.aspect(),
        GpuCopyExtent::new(1, 2, 1).expect("partial radiance source extent"),
    )
    .expect("partial texture source must be publicly constructible")
    .into()
}

fn submit_intervening_texture_write(
    context: &GpuContext,
    destination: &GpuTransferRegion,
) -> GpuSubmission {
    let GpuTransferRegion::Texture(destination) = destination else {
        panic!("radiance capture destination must be a texture region");
    };
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let source_texture = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::ordinary_owned_2d(
                "R7 intervening radiance writer source",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                2,
                2,
                GpuTextureFormat::R32Uint,
                [GpuTextureUsage::CopySource],
                GpuTextureInitialization::Zeroed,
            )
            .expect("intervening writer source descriptor"),
        )
        .expect("intervening writer source handle");
    let operation = GpuCopyOperation::texture_to_texture(
        GpuTextureCopyRegion::whole_base_mip(&source_texture)
            .expect("intervening writer source region"),
        destination.clone(),
    )
    .expect("intervening texture writer must use public copy validation");
    let fragment = GpuWorkFragment::build("R7 intervening radiance writer", |work| {
        work.operation("overwrite retained radiance destination", operation)?;
        Ok(())
    })
    .expect("intervening writer fragment must be authorable");
    let submission =
        pollster::block_on(context.submit_work("R7 intervening radiance writer", [fragment]))
            .expect("intervening writer must submit");
    wait_for_submission(context, &submission);
    submission
}

#[test]
fn maintained_execution_keeps_ordinary_unobserved_and_verified_same_submission_observed() {
    let Some(context) = request_execution_context() else {
        return;
    };
    let fixture = maintained_fixture();

    let ordinary_admitted =
        admit_with_writable_only_destination(&fixture, &context, "R7 ordinary destination");
    let ordinary = pollster::block_on(submit_deterministic_render(ordinary_admitted, &context))
        .expect("ordinary maintained execution must submit");
    assert!(
        ordinary.submission().readbacks().is_empty(),
        "ordinary maintained execution must author no CPU readbacks"
    );
    wait_for_submission(&context, ordinary.submission());

    let verified_admitted =
        admit_with_writable_only_destination(&fixture, &context, "R7 verified destination");
    let verified = pollster::block_on(submit_deterministic_render_for_verified_formation(
        verified_admitted,
        &context,
    ))
    .expect("verified maintained execution must submit with same-submission correlation");
    assert_eq!(
        verified.readbacks().len(),
        1,
        "one admitted output must retain one private verification correlation"
    );
    let correlation = verified.readbacks()[0];
    assert_eq!(correlation.output_index(), 0);
    let readback_ids = [
        correlation.canonical_output(),
        correlation.definedness(),
        correlation.status(),
    ];
    let submission = verified.submitted().submission();
    assert_eq!(
        submission.readbacks().len(),
        readback_ids.len(),
        "verified execution must add exactly three renderer-private readbacks per output"
    );
    for readback_id in readback_ids {
        assert!(
            submission.readback(readback_id).is_some(),
            "every retained correlation must resolve through the exact returned submission"
        );
    }
    wait_for_readbacks(&context, submission, &readback_ids);

    let raw_canonical = match submission
        .readback(correlation.canonical_output())
        .expect("canonical correlation")
        .status()
    {
        GpuReadbackStatus::Ready(bytes) => bytes,
        status => panic!("canonical observation must be ready after wait: {status:?}"),
    };
    assert_eq!(
        raw_canonical.layout().byte_len(),
        u64::try_from(raw_canonical.as_bytes().len()).expect("readback byte length must fit u64")
    );
    assert!(
        raw_canonical.layout().byte_len() >= 4 * 4,
        "2x2 canonical observation must contain at least four one-word logical samples"
    );

    assert!(matches!(
        verified.submitted().submission().status(),
        GpuSubmissionStatus::Completed
    ));
    let _formation = verify_completed_deterministic_render(verified)
        .expect("same-submission object-identity samples must satisfy RR566-EVAL-001");
}

#[test]
fn public_verified_result_path_forms_once_from_exact_submission() {
    let Some(context) = request_execution_context() else {
        return;
    };
    let fixture = maintained_fixture();
    let admitted =
        admit_with_writable_only_destination(&fixture, &context, "R7 public verified destination");
    let mut submitted = pollster::block_on(submit_deterministic_render_for_verified_result(
        admitted, &context,
    ))
    .expect("public verified-result path must author one maintained submission");

    assert_eq!(
        submitted.submission().readbacks().len(),
        3,
        "one public verified output must retain exactly three renderer-private observations"
    );

    let deadline = Instant::now() + Duration::from_secs(15);
    let result = loop {
        context.progress();
        match submitted.try_form_verified_result() {
            Ok(Some(result)) => break result,
            Ok(None) => {}
            Err(error) => panic!("public verified-result formation failed: {error}"),
        }
        assert!(
            Instant::now() < deadline,
            "public verified-result formation did not complete before timeout"
        );
        std::thread::yield_now();
    };

    assert_eq!(result.request().outputs().len(), 1);
    assert_eq!(
        result.surface_semantic_inputs(),
        fixture.semantic_inputs.as_slice()
    );
    assert_eq!(result.outputs().len(), 1);
    assert_eq!(result.outputs()[0].output_index(), 0);
    assert_eq!(
        submitted.try_form_verified_result(),
        Err(RenderDeterministicResultFormationError::ResultAlreadyFormed),
        "one exact verified submission must not mint semantic result evidence twice"
    );
}

#[test]
fn public_radiance_capture_uses_a_separate_product_readback_submission() {
    let Some(context) = request_execution_context() else {
        return;
    };
    let fixture = maintained_fixture();
    let admitted = admit_with_retained_radiance_destination(&fixture, &context);
    let mut submitted = pollster::block_on(submit_deterministic_render_for_verified_result(
        admitted, &context,
    ))
    .expect("public maintained radiance execution must submit");
    assert!(matches!(
        submitted.request_deterministic_radiance_capture(0),
        Err(RenderDeterministicRadianceCaptureRequestError::VerificationNotFormed)
    ));

    let deadline = Instant::now() + Duration::from_secs(15);
    let result = loop {
        context.progress();
        match submitted.try_form_verified_result() {
            Ok(Some(result)) => break result,
            Ok(None) => {}
            Err(error) => panic!("public maintained radiance result formation failed: {error}"),
        }
        assert!(
            Instant::now() < deadline,
            "public maintained radiance result formation did not complete before timeout"
        );
        std::thread::yield_now();
    };

    let request = submitted
        .request_deterministic_radiance_capture(0)
        .expect("formed radiance result must mint a capture request");
    assert!(matches!(
        submitted.request_deterministic_radiance_capture(usize::MAX),
        Err(RenderDeterministicRadianceCaptureRequestError::OutputIndexOutOfRange)
    ));
    let readback = GpuReadbackOperation::new(request.source().clone(), request.readback_id())
        .expect("product must construct the ordinary public readback from request evidence");
    let fragment = GpuWorkFragment::build("R7 product radiance readback", |work| {
        work.operation("read back retained radiance lattice", readback)?;
        Ok(())
    })
    .expect("product readback fragment must be authorable through public RunenGPU");
    let product_submission =
        pollster::block_on(context.submit_work("R7 product radiance readback", [fragment]))
            .expect("product radiance readback must submit separately");
    wait_for_readbacks(&context, &product_submission, &[request.readback_id()]);

    let captured = submitted
        .capture_deterministic_radiance(request, &context, &product_submission)
        .expect("RunenRender must interpret the exact product-owned readback");
    assert_eq!(captured.output_index(), 0);
    assert_eq!(
        captured.topology().sample_lattice_dimensions(),
        Some((2, 2))
    );
    assert_eq!(captured.samples().len(), 4);
    assert!(captured.samples().iter().all(|sample| sample.is_finite()));
    assert!(captured.samples().iter().all(|sample| *sample == 0.0));
    assert_eq!(result.outputs().len(), 1);
}

#[test]
fn formed_non_radiance_output_rejects_radiance_capture_request() {
    let Some(context) = request_execution_context() else {
        return;
    };
    let fixture = maintained_fixture();
    let admitted = admit_with_retained_destination(
        &fixture,
        fixture.request.clone(),
        &context,
        "R7 retained non-radiance destination",
    );
    let submitted = form_verified_render(
        pollster::block_on(submit_deterministic_render_for_verified_result(
            admitted, &context,
        ))
        .expect("formed non-radiance submission must submit"),
        &context,
    );
    assert!(matches!(
        submitted.request_deterministic_radiance_capture(0),
        Err(RenderDeterministicRadianceCaptureRequestError::OutputNotRadiance)
    ));
}

#[test]
fn radiance_capture_rejects_correlation_affinity_and_replaced_writer_evidence() {
    let Some(context_a) = request_execution_context() else {
        return;
    };
    let Some(context_b) = request_execution_context() else {
        return;
    };
    let fixture = maintained_fixture();
    let submitted_a = form_verified_render(
        pollster::block_on(submit_deterministic_render_for_verified_result(
            admit_with_retained_radiance_destination(&fixture, &context_a),
            &context_a,
        ))
        .expect("first radiance submission must submit"),
        &context_a,
    );

    let missing_request = submitted_a
        .request_deterministic_radiance_capture(0)
        .expect("formed radiance result must mint a missing-ID proof request");
    let missing_submission = submit_product_readback(
        &context_a,
        missing_request.source().clone(),
        GpuReadbackId::allocate().expect("missing-ID proof allocation"),
    );
    assert_eq!(
        submitted_a.capture_deterministic_radiance(
            missing_request,
            &context_a,
            &missing_submission,
        ),
        Err(super::deterministic_execution::RenderDeterministicRadianceCaptureError::ReadbackCorrelationMissing)
    );

    let wrong_source_request = submitted_a
        .request_deterministic_radiance_capture(0)
        .expect("formed radiance result must mint a wrong-source proof request");
    let wrong_source_submission = submit_product_readback(
        &context_a,
        partial_texture_source(wrong_source_request.source()),
        wrong_source_request.readback_id(),
    );
    assert_eq!(
        submitted_a.capture_deterministic_radiance(
            wrong_source_request,
            &context_a,
            &wrong_source_submission,
        ),
        Err(super::deterministic_execution::RenderDeterministicRadianceCaptureError::ReadbackSourceMismatch)
    );

    let submitted_b = form_verified_render(
        pollster::block_on(submit_deterministic_render_for_verified_result(
            admit_with_retained_radiance_destination(&fixture, &context_b),
            &context_b,
        ))
        .expect("second radiance submission must submit"),
        &context_b,
    );
    assert_ne!(
        submitted_a.submission().affinity(),
        submitted_b.submission().affinity(),
        "fresh contexts must have distinct submission affinities"
    );
    assert_eq!(
        submitted_a.submission().id(),
        submitted_b.submission().id(),
        "fresh contexts should independently allocate colliding submission IDs"
    );

    let foreign_context_request = submitted_a
        .request_deterministic_radiance_capture(0)
        .expect("formed radiance result must mint a foreign-context proof request");
    assert_eq!(
        submitted_a.capture_deterministic_radiance(
            foreign_context_request,
            &context_b,
            submitted_b.submission(),
        ),
        Err(super::deterministic_execution::RenderDeterministicRadianceCaptureError::ContextAffinityMismatch)
    );

    let mismatched_render_request = submitted_a
        .request_deterministic_radiance_capture(0)
        .expect("formed radiance result must mint a render-correlation proof request");
    assert_eq!(
        submitted_b.capture_deterministic_radiance(
            mismatched_render_request,
            &context_b,
            submitted_b.submission(),
        ),
        Err(super::deterministic_execution::RenderDeterministicRadianceCaptureError::RequestCorrelationMismatch)
    );

    let foreign_product_request = submitted_a
        .request_deterministic_radiance_capture(0)
        .expect("formed radiance result must mint a product-affinity proof request");
    assert_eq!(
        submitted_a.capture_deterministic_radiance(
            foreign_product_request,
            &context_a,
            submitted_b.submission(),
        ),
        Err(super::deterministic_execution::RenderDeterministicRadianceCaptureError::ProductSubmissionAffinityMismatch)
    );

    let replaced_writer_request = submitted_a
        .request_deterministic_radiance_capture(0)
        .expect("formed radiance result must mint a replaced-writer proof request");
    submit_intervening_texture_write(&context_a, replaced_writer_request.source());
    let replaced_writer_submission = submit_product_readback(
        &context_a,
        replaced_writer_request.source().clone(),
        replaced_writer_request.readback_id(),
    );
    assert_eq!(
        submitted_a.capture_deterministic_radiance(
            replaced_writer_request,
            &context_a,
            &replaced_writer_submission,
        ),
        Err(super::deterministic_execution::RenderDeterministicRadianceCaptureError::RendererWriteNoLongerCurrent)
    );
}

#[test]
fn admitted_dispatch_budget_tiles_large_lattice_through_verified_execution() {
    let Some(context) = request_execution_context_with_dispatch_limit(Some(8)) else {
        return;
    };
    assert_eq!(
        context
            .device_facts()
            .workload_budget()
            .limits()
            .max_compute_workgroups_per_dimension(),
        8,
        "the proof must execute under the explicitly admitted dispatch budget"
    );

    let mut fixture = maintained_fixture();
    fixture.request = object_identity_lattice_request(27, 19);
    let admitted = admit_deterministic_render_with_semantic_inputs(
        &fixture.scene,
        &fixture.request,
        &fixture.semantic_inputs,
        &[],
        &fixture.availability,
        &lattice_bindings(27, 19, fixture.request.outputs().len()),
        &context,
    )
    .expect("27x19 maintained lattice must be admitted under the capped context");
    let mut submitted = pollster::block_on(submit_deterministic_render_for_verified_result(
        admitted, &context,
    ))
    .expect("the tiled dispatch must be accepted by the real RunenGPU submission boundary");
    assert!(matches!(
        submitted.submission().status(),
        GpuSubmissionStatus::Accepted | GpuSubmissionStatus::Completed
    ));

    let deadline = Instant::now() + Duration::from_secs(15);
    let result = loop {
        context.progress();
        match submitted.try_form_verified_result() {
            Ok(Some(result)) => break result,
            Ok(None) => {}
            Err(error) => panic!("capped tiled verified execution failed: {error}"),
        }
        assert!(
            Instant::now() < deadline,
            "capped tiled verified execution did not complete before timeout"
        );
        std::thread::yield_now();
    };

    assert_eq!(
        result.request().outputs()[0]
            .spec()
            .topology()
            .sample_lattice_dimensions(),
        Some((27, 19))
    );
    assert!(matches!(
        submitted.submission().status(),
        GpuSubmissionStatus::Completed
    ));
}

#[test]
fn maintained_ordinary_execution_supports_reordered_8x6_subset_outside_verifier_domain() {
    let Some(context) = request_execution_context() else {
        return;
    };
    let fixture = maintained_fixture();
    let request = reordered_8x6_request();
    let output_bindings = lattice_bindings(8, 6, request.outputs().len());
    let admitted = admit_deterministic_render_with_semantic_inputs(
        &fixture.scene,
        &request,
        &fixture.semantic_inputs,
        &[],
        &fixture.availability,
        &output_bindings,
        &context,
    )
    .expect("wide reordered 8x6 subset must remain legal maintained evaluator work");

    let requested = admitted.admitted().plan().request().outputs();
    assert_eq!(
        requested.len(),
        2,
        "structural proof intentionally uses an output subset"
    );
    assert!(matches!(
        requested[0].spec().value(),
        RenderOutputValue::ObjectIdentity
    ));
    assert!(matches!(
        requested[1].spec().value(),
        RenderOutputValue::Distance {
            convention: RenderDistanceConvention::ObservationForwardDepth
        }
    ));
    for output in requested {
        assert_eq!(
            output.spec().topology().sample_lattice_dimensions(),
            Some((8, 6)),
            "maintained topology must come from the exact request"
        );
    }

    let submitted = pollster::block_on(submit_deterministic_render(admitted, &context))
        .expect("reordered 8x6 subset must submit through ordinary maintained execution");
    assert!(
        submitted.submission().readbacks().is_empty(),
        "structural ordinary execution must not gain verification readbacks"
    );
    wait_for_submission(&context, submitted.submission());
}
