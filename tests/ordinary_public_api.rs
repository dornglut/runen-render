use runen_gpu::{
    GpuCapabilityProfile, GpuContext, GpuContextDescriptor, GpuContextRequestErrorCategory,
    GpuFormatRole, GpuReadbackId, GpuReadbackOperation, GpuReadbackStatus, GpuReconstruction,
    GpuResourceLifetime, GpuSubmission, GpuSubmissionFailureKind, GpuSubmissionStatus,
    GpuTextureCopyRegion, GpuTextureDescriptor, GpuTextureFormat, GpuTextureHandle,
    GpuTextureInitialization, GpuTextureUsage, GpuWorkFragment, GpuWorkResourceIdAllocator,
};
use runen_render::admission::{
    RenderOutputBinding, RenderOutputDestination, RenderRepresentationAvailabilityFact,
    RenderRepresentationAvailabilityState,
};
use runen_render::appearance::{RenderDiffuseMaterial, RenderDirectionalEmitter};
use runen_render::participation::{RenderMaterialAssignment, RenderObjectParticipation};
use runen_render::representation::{
    RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION, RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
    RenderOrientedSurfaceProtocolEvidence, RenderRefinementEvidence, RenderRepresentationRecord,
    RenderSurfaceProtocolEvidence,
};
use runen_render::request::{
    RenderObservationSpec, RenderOutputSpec, RenderOutputValue, RenderPerspectiveObservation,
    RenderRadiometricRepresentation, RenderRequestBuilder, RenderResultTopology,
    RenderSamplingSupport, RenderSemanticTolerance,
};
use runen_render::scene::{RenderObjectId, RenderObjectState, RenderSceneStore, RenderSceneUpdate};
use runen_render::space_time::{
    RenderAffineTransform3, RenderHandedness, RenderObjectSpatialState, RenderObjectTemporalState,
    RenderSpaceSpec, RenderSpatialCoverage, RenderTemporalSupport, RenderTimeInterval,
    RenderTimePoint,
};
use runen_render::surface_input::{
    RenderSurfaceSemanticInput, RenderSurfaceSemanticInputBinding,
    RenderSurfaceSemanticInputGeneration, RenderSurfaceSemanticInputRequirement,
};
use runen_render::{
    AdmittedRender, AssociatedRenderOccurrence, PreparedRadianceOutput, PreparedRender,
    PreparedRenderOccurrence, RenderAdmissionError, RenderCapturedRadiance,
    RenderEvaluationSelection, RenderExecutionError, RenderExecutionErrorKind,
    RenderExecutionSession, RenderExecutionSessionError, RenderInvocation,
    RenderObjectIdentityDecoderError, RenderObjectIdentityDecoderErrorKind,
    RenderRadianceCaptureError, RenderRadianceCaptureErrorKind, RenderRadianceCaptureRequest,
    RenderRadianceCaptureRequestError, RenderRadianceCaptureRequestErrorKind,
    RenderResultFormationError, RenderResultFormationErrorKind, RenderResultSubmissionError,
    RenderResultSubmissionErrorKind, RenderTemporalExecutionEvidence,
    RenderVerificationEligibilityErrorKind, SubmittedRender, SubmittedRenderForResult,
    admit_render, prepare_render, submit_render, submit_render_for_result,
};
use std::time::{Duration, Instant};

#[test]
fn ordinary_semantic_renderer_surface_is_public_to_downstream_consumers() {
    let _ = admit_render;
    let _ = prepare_render;
    let _ = submit_render;
    let _ = submit_render_for_result;
    let _ = AdmittedRender::admitted_plan;
    let _ = PreparedRender::admitted_plan;
    let _ = PreparedRender::work_set;
    let _ = PreparedRender::radiance_outputs;
    let _ = PreparedRender::radiance_output;
    let _ = PreparedRadianceOutput::temporal_execution_evidence;
    let _ = PreparedRadianceOutput::import;
    let _ = SubmittedRender::admitted_plan;
    let _ = SubmittedRender::submission_status;
    let _ = SubmittedRender::object_identity_decoder;
    let _ = SubmittedRenderForResult::admitted_plan;
    let _ = SubmittedRenderForResult::submission_status;
    let _ = SubmittedRenderForResult::object_identity_decoder;
    let _ = SubmittedRenderForResult::try_form_result;
    let _ = SubmittedRenderForResult::request_radiance_capture;
    let _ = SubmittedRenderForResult::capture_radiance;
    let _ = RenderRadianceCaptureRequest::source;
    let _ = RenderRadianceCaptureRequest::readback_id;
    let _ = GpuReadbackOperation::new;
    let _ = RenderAdmissionError::kind;
    let _ = RenderExecutionError::kind;
    let _ = RenderExecutionError::runen_shader_compilation_source;
    let _ = RenderExecutionError::runen_gpu_preparation_source;
    let _ = RenderExecutionError::submission_error;
    let _ = RenderResultSubmissionError::kind;
    let _ = RenderResultSubmissionError::verification_eligibility_kind;
    let _ = RenderResultSubmissionError::observation;
    let _ = RenderResultSubmissionError::object_id;
    let _ = RenderResultSubmissionError::readback_cardinality;
    let _ = RenderResultSubmissionError::output_correlation;
    let _ = RenderResultSubmissionError::correlation_output;
    let _ = RenderResultSubmissionError::correlation_channel;
    let _ = RenderResultFormationError::kind;
    let _ = RenderResultFormationError::verification_eligibility_kind;
    let _ = RenderResultFormationError::output;
    let _ = RenderResultFormationError::sample_index;
    let _ = RenderResultFormationError::channel;
    let _ = RenderResultFormationError::gpu_failure_kind;
    let _ = RenderRadianceCaptureRequestError::kind;
    let _ = RenderRadianceCaptureError::kind;
    let _ = RenderRadianceCaptureError::gpu_failure_kind;
    let _ = RenderExecutionSession::new;
    let _ = RenderExecutionSession::is_in_flight;
    let _ = RenderExecutionSession::reconcile;
    let _ = RenderExecutionSession::prepare;
    let _ = RenderExecutionSession::associate_submission;
    let _ = AssociatedRenderOccurrence::admitted_plan;
    let _ = AssociatedRenderOccurrence::submission_status;
    let _ = AssociatedRenderOccurrence::request_radiance_capture;
    let _ = AssociatedRenderOccurrence::capture_radiance;
    let _ = AssociatedRenderOccurrence::object_identity_decoder;
    let _ = PreparedRenderOccurrence::admitted_plan;
    let _ = PreparedRenderOccurrence::work_set;
    let _ = PreparedRenderOccurrence::radiance_outputs;
    let _ = PreparedRenderOccurrence::radiance_output;
    fn assert_selection(output: runen_render::request::RenderOutputHandle) {
        let selection =
            RenderEvaluationSelection::new(output.clone(), 64, 32).expect("non-zero extent");
        assert_eq!(selection.output(), &output);
        assert_eq!(selection.extent(), (64, 32));
    }
    let _ = assert_selection;

    fn assert_temporal_evidence(evidence: &RenderTemporalExecutionEvidence) {
        let _ = (
            evidence.requested_extent,
            evidence.evaluation_extent,
            evidence.sequence_revision,
            evidence.reconstruction_revision,
            evidence.phase,
            evidence.history_generation,
            evidence.history_age,
            evidence.history_reset,
            evidence.camera_reprojection_eligible,
        );
    }
    let _ = assert_temporal_evidence;

    fn assert_error_kinds(
        execution: RenderExecutionErrorKind,
        submission: RenderResultSubmissionErrorKind,
        formation: RenderResultFormationErrorKind,
        capture_request: RenderRadianceCaptureRequestErrorKind,
        capture: RenderRadianceCaptureErrorKind,
    ) {
        let _ = (execution, submission, formation, capture_request, capture);
    }
    let _ = assert_error_kinds;
    let _ = RenderVerificationEligibilityErrorKind::SamplingSupportUnsupported;

    fn assert_prepared_output_surface(output: &PreparedRadianceOutput<'_>) {
        let _ = output.output();
        let _ = output.resource();
        let _ = output.texture();
        let _ = output.export_relationship();
    }

    let _ = assert_prepared_output_surface;

    fn assert_capture_surface(
        submitted: &SubmittedRenderForResult,
        request: RenderRadianceCaptureRequest,
        context: &GpuContext,
        product_submission: &GpuSubmission,
    ) -> Result<RenderCapturedRadiance, RenderRadianceCaptureError> {
        submitted.capture_radiance(request, context, product_submission)
    }

    let _ = assert_capture_surface;

    fn assert_public_error<E: std::error::Error + 'static>() {}
    assert_public_error::<RenderAdmissionError>();
    assert_public_error::<RenderExecutionError>();
    assert_public_error::<RenderExecutionSessionError>();
    assert_public_error::<RenderObjectIdentityDecoderError>();
    assert_public_error::<RenderResultSubmissionError>();
    assert_public_error::<RenderResultFormationError>();
    assert_public_error::<RenderRadianceCaptureRequestError>();
    assert_public_error::<RenderRadianceCaptureError>();
}

#[test]
fn ordinary_surface_executes_headless_through_public_runengpu_only() {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .require_format_role(GpuTextureFormat::R32Uint, GpuFormatRole::CopyDestination)
            .with_label("RunenRender R8 ordinary public consumer");
    let context = match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => context,
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1"),
                "R8 ordinary public consumer CI requires a public RunenGPU adapter"
            );
            return;
        }
        Err(error) => panic!("unexpected R8 ordinary public RunenGPU context failure: {error}"),
    };

    let mut scene = RenderSceneStore::new();
    let object_id = scene.allocate_object_id().expect("R8 public object id");
    let object_state = RenderObjectState::new(
        RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("metric object space"),
            RenderAffineTransform3::from_row_major_3x4([
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -3.0,
            ])
            .expect("finite object transform"),
            RenderSpatialCoverage::unbounded(),
        ),
        RenderObjectTemporalState::new(RenderTemporalSupport::unbounded()),
    );
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(object_id, object_state);
    scene.commit(insert).expect("insert public consumer object");

    let representation_id = scene
        .allocate_representation_id(object_id)
        .expect("R8 public representation id");
    let surface = RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
        .expect("surface protocol")
        .with_semantic_input_requirement(RenderSurfaceSemanticInputRequirement::current());
    let representation = RenderRepresentationRecord::new(
        representation_id,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
        RenderRefinementEvidence::none(),
        Some(surface),
        None,
    )
    .expect("public surface representation");
    let participation = RenderObjectParticipation::new(vec![representation], None, None)
        .expect("public object participation");
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object_id, participation);
    scene
        .commit(attach)
        .expect("attach public object participation");

    let shutter = RenderTimeInterval::instant(
        RenderTimePoint::from_seconds(0.0).expect("finite public consumer time"),
    );
    let observation = RenderObservationSpec::Perspective(
        RenderPerspectiveObservation::new(
            RenderAffineTransform3::identity(),
            std::f64::consts::FRAC_PI_3,
            1.0,
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("public perspective observation"),
    );
    let mut builder = RenderRequestBuilder::new(shutter);
    let observation_handle = builder.add_observation(observation);
    builder
        .add_output(
            &observation_handle,
            RenderOutputSpec::new(
                RenderOutputValue::ObjectIdentity,
                RenderResultTopology::sample_lattice_2d(1, 1).expect("1x1 public lattice"),
                RenderSemanticTolerance::exact(),
            )
            .expect("public object-identity output"),
        )
        .expect("own observation");
    let request = builder.finish().expect("public render request");

    let semantic_inputs = [RenderSurfaceSemanticInputBinding::new(
        representation_id,
        RenderSurfaceSemanticInput::sphere(
            [0.0, 0.0, 0.0],
            1.0,
            RenderTemporalSupport::unbounded(),
        )
        .expect("public sphere semantic input"),
    )];
    let availability = [RenderRepresentationAvailabilityFact::new(
        representation_id,
        RenderRepresentationAvailabilityState::Available,
    )];

    let mut allocator = GpuWorkResourceIdAllocator::new();
    let destination = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::ordinary_owned_2d(
                "R8 ordinary public consumer output",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                1,
                1,
                GpuTextureFormat::R32Uint,
                [GpuTextureUsage::CopyDestination],
                GpuTextureInitialization::Uninitialized,
            )
            .expect("public output descriptor"),
        )
        .expect("public output handle");
    let output_bindings = [RenderOutputBinding::new(
        request.output_handle(0).expect("request output handle"),
        RenderOutputDestination::SampleLatticeTexture(destination),
    )];

    let admitted = admit_render(
        &RenderInvocation::new(
            scene.snapshot(),
            request.clone(),
            semantic_inputs.to_vec(),
            Vec::new(),
            availability.to_vec(),
            output_bindings.to_vec(),
        )
        .expect("valid correlated invocation"),
        &context,
    )
    .expect("public ordinary admission");
    let submitted =
        pollster::block_on(submit_render(admitted, &context)).expect("public ordinary submission");

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        match submitted.submission_status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(failure) => {
                panic!("public ordinary RunenGPU submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Accepted if Instant::now() < deadline => {
                std::thread::yield_now();
            }
            GpuSubmissionStatus::Accepted => {
                panic!("public ordinary RunenGPU submission did not complete before timeout")
            }
        }
    }

    assert_eq!(submitted.admitted_plan().scene_revision(), scene.revision());
    assert_eq!(submitted.admitted_plan().outputs().len(), 1);
}

fn wait_for_submission(context: &GpuContext, submission: &GpuSubmission) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => return,
            GpuSubmissionStatus::Failed(failure) => {
                panic!("retained-session RunenGPU submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Accepted if Instant::now() < deadline => {
                std::thread::yield_now();
            }
            GpuSubmissionStatus::Accepted => {
                panic!("retained-session RunenGPU submission did not complete before timeout")
            }
        }
    }
}

fn wait_for_readback(context: &GpuContext, submission: &GpuSubmission, id: GpuReadbackId) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        let readback = submission
            .readback(id)
            .expect("submitted readback correlation must remain observable");
        match (submission.status(), readback.status()) {
            (GpuSubmissionStatus::Completed, GpuReadbackStatus::Ready(_)) => return,
            (GpuSubmissionStatus::Failed(failure), _) => {
                panic!("retained-output readback submission failed: {failure:?}")
            }
            (_, GpuReadbackStatus::Failed(failure)) => {
                panic!("retained-output readback failed: {failure:?}")
            }
            _ if Instant::now() < deadline => std::thread::yield_now(),
            _ => panic!("retained-output readback did not complete before timeout"),
        }
    }
}

fn retained_context(format: GpuTextureFormat) -> Option<(GpuContext, GpuContextDescriptor)> {
    let mut descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .require_format_role(format, GpuFormatRole::CopyDestination)
            .with_label("RunenRender retained-session public consumer");
    if matches!(
        format,
        GpuTextureFormat::R32Float | GpuTextureFormat::R32Uint
    ) {
        descriptor = descriptor.require_format_role(format, GpuFormatRole::CopySource);
    }
    match pollster::block_on(GpuContext::request(descriptor.clone())) {
        Ok(context) => Some((context, descriptor)),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1"),
                "retained-session public consumer CI requires a public RunenGPU adapter"
            );
            None
        }
        Err(error) => panic!("unexpected retained-session RunenGPU context failure: {error}"),
    }
}

fn admitted_identity_render(
    context: &GpuContext,
    label: &str,
    output_count: usize,
) -> AdmittedRender {
    let mut scene = RenderSceneStore::new();
    let object_id = scene
        .allocate_object_id()
        .expect("retained public object id");
    let object_state = RenderObjectState::new(
        RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("metric object space"),
            RenderAffineTransform3::from_row_major_3x4([
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -3.0,
            ])
            .expect("finite object transform"),
            RenderSpatialCoverage::unbounded(),
        ),
        RenderObjectTemporalState::new(RenderTemporalSupport::unbounded()),
    );
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(object_id, object_state);
    scene.commit(insert).expect("insert retained public object");

    let representation_id = scene
        .allocate_representation_id(object_id)
        .expect("retained public representation id");
    let surface = RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
        .expect("surface protocol")
        .with_semantic_input_requirement(RenderSurfaceSemanticInputRequirement::current());
    let representation = RenderRepresentationRecord::new(
        representation_id,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
        RenderRefinementEvidence::none(),
        Some(surface),
        None,
    )
    .expect("retained surface representation");
    let participation = RenderObjectParticipation::new(vec![representation], None, None)
        .expect("retained object participation");
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object_id, participation);
    scene.commit(attach).expect("attach retained participation");

    let shutter = RenderTimeInterval::instant(
        RenderTimePoint::from_seconds(0.0).expect("finite retained time"),
    );
    let observation = RenderObservationSpec::Perspective(
        RenderPerspectiveObservation::new(
            RenderAffineTransform3::identity(),
            std::f64::consts::FRAC_PI_3,
            1.0,
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("retained perspective observation"),
    );
    let mut builder = RenderRequestBuilder::new(shutter);
    let observation_handle = builder.add_observation(observation);
    for _ in 0..output_count {
        builder
            .add_output(
                &observation_handle,
                RenderOutputSpec::new(
                    RenderOutputValue::ObjectIdentity,
                    RenderResultTopology::sample_lattice_2d(1, 1).expect("retained lattice"),
                    RenderSemanticTolerance::exact(),
                )
                .expect("retained identity output"),
            )
            .expect("own observation");
    }
    let request = builder.finish().expect("retained render request");

    let semantic_inputs = [RenderSurfaceSemanticInputBinding::new(
        representation_id,
        RenderSurfaceSemanticInput::sphere(
            [0.0, 0.0, 0.0],
            1.0,
            RenderTemporalSupport::unbounded(),
        )
        .expect("retained sphere semantic input"),
    )];
    let availability = [RenderRepresentationAvailabilityFact::new(
        representation_id,
        RenderRepresentationAvailabilityState::Available,
    )];

    let mut allocator = GpuWorkResourceIdAllocator::new();
    let mut output_bindings = Vec::new();
    for output_index in 0..output_count {
        let destination = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::ordinary_owned_2d(
                    format!("{label} output {output_index}"),
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    1,
                    1,
                    GpuTextureFormat::R32Uint,
                    [GpuTextureUsage::CopyDestination],
                    GpuTextureInitialization::Uninitialized,
                )
                .expect("retained output descriptor"),
            )
            .expect("retained output handle");
        output_bindings.push(RenderOutputBinding::new(
            request
                .output_handle(output_index)
                .expect("request output handle"),
            RenderOutputDestination::SampleLatticeTexture(destination),
        ));
    }

    admit_render(
        &RenderInvocation::new(
            scene.snapshot(),
            request.clone(),
            semantic_inputs.to_vec(),
            Vec::new(),
            availability.to_vec(),
            output_bindings.to_vec(),
        )
        .expect("valid correlated invocation"),
        context,
    )
    .expect("retained ordinary admission")
}

fn admitted_temporal_radiance_render(context: &GpuContext, label: &str) -> AdmittedRender {
    admitted_temporal_radiance_render_at_extent(context, label, 2)
}

fn admitted_temporal_radiance_render_at_extent(
    context: &GpuContext,
    label: &str,
    extent: u32,
) -> AdmittedRender {
    let mut scene = RenderSceneStore::new();
    let object_id = scene
        .allocate_object_id()
        .expect("temporal public object id");
    let object_state = RenderObjectState::new(
        RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("metric object space"),
            RenderAffineTransform3::from_row_major_3x4([
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -3.0,
            ])
            .expect("finite object transform"),
            RenderSpatialCoverage::unbounded(),
        ),
        RenderObjectTemporalState::new(RenderTemporalSupport::unbounded()),
    );
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(object_id, object_state);
    scene.commit(insert).expect("insert temporal public object");

    let representation_id = scene
        .allocate_representation_id(object_id)
        .expect("temporal public representation id");
    let surface = RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
        .expect("surface protocol")
        .with_oriented_surface(
            RenderOrientedSurfaceProtocolEvidence::exact(
                RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
            )
            .expect("oriented surface protocol"),
        )
        .with_semantic_input_requirement(RenderSurfaceSemanticInputRequirement::current());
    let representation = RenderRepresentationRecord::new(
        representation_id,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
        RenderRefinementEvidence::none(),
        Some(surface),
        None,
    )
    .expect("temporal oriented representation");
    let participation = RenderObjectParticipation::new(
        vec![representation],
        Some(RenderMaterialAssignment::new(
            RenderDiffuseMaterial::new(0.8).expect("temporal diffuse material"),
        )),
        Some(
            RenderDirectionalEmitter::new([0.0, 1.0, 0.0], 550e-9, 1.0)
                .expect("temporal directional illumination"),
        ),
    )
    .expect("temporal object participation");
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object_id, participation);
    scene.commit(attach).expect("attach temporal participation");

    let shutter = RenderTimeInterval::instant(
        RenderTimePoint::from_seconds(0.0).expect("finite temporal time"),
    );
    let observation = RenderObservationSpec::Perspective(
        RenderPerspectiveObservation::new(
            RenderAffineTransform3::identity(),
            std::f64::consts::FRAC_PI_3,
            1.0,
            shutter,
            RenderSamplingSupport::perspective_lattice_cell(),
        )
        .expect("temporal perspective observation"),
    );
    let mut builder = RenderRequestBuilder::new(shutter);
    let observation_handle = builder.add_observation(observation);
    builder
        .add_output(
            &observation_handle,
            RenderOutputSpec::new(
                RenderOutputValue::Radiance {
                    representation: RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                        550e-9,
                    )
                    .expect("visible spectral radiance"),
                },
                RenderResultTopology::sample_lattice_2d(extent, extent).expect("temporal lattice"),
                RenderSemanticTolerance::exact(),
            )
            .expect("temporal radiance output"),
        )
        .expect("own observation");
    let request = builder.finish().expect("temporal render request");

    let semantic_inputs = [RenderSurfaceSemanticInputBinding::new(
        representation_id,
        RenderSurfaceSemanticInput::sphere(
            [0.0, 0.0, 0.0],
            1.0,
            RenderTemporalSupport::unbounded(),
        )
        .expect("temporal sphere semantic input"),
    )
    .with_generation(RenderSurfaceSemanticInputGeneration::new(1))];
    let availability = [RenderRepresentationAvailabilityFact::new(
        representation_id,
        RenderRepresentationAvailabilityState::Available,
    )];

    let mut allocator = GpuWorkResourceIdAllocator::new();
    let destination = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::ordinary_owned_2d(
                label,
                GpuResourceLifetime::Retained,
                GpuReconstruction::SourceBacked,
                extent,
                extent,
                GpuTextureFormat::R32Float,
                [
                    GpuTextureUsage::CopyDestination,
                    GpuTextureUsage::CopySource,
                ],
                GpuTextureInitialization::Uninitialized,
            )
            .expect("temporal output descriptor"),
        )
        .expect("temporal output handle");
    let output_bindings = [RenderOutputBinding::new(
        request.output_handle(0).expect("request output handle"),
        RenderOutputDestination::SampleLatticeTexture(destination),
    )];

    admit_render(
        &RenderInvocation::new(
            scene.snapshot(),
            request.clone(),
            semantic_inputs.to_vec(),
            Vec::new(),
            availability.to_vec(),
            output_bindings.to_vec(),
        )
        .expect("valid correlated invocation"),
        context,
    )
    .expect("temporal ordinary admission")
}

fn admitted_retained_capture_render(
    context: &GpuContext,
    label: &str,
) -> (AdmittedRender, RenderObjectId, GpuTextureHandle) {
    const EXTENT: u32 = 4;

    let mut scene = RenderSceneStore::new();
    let object_id = scene
        .allocate_object_id()
        .expect("retained capture object id");
    let object_state = RenderObjectState::new(
        RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("metric object space"),
            RenderAffineTransform3::from_row_major_3x4([
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -3.0,
            ])
            .expect("finite retained capture transform"),
            RenderSpatialCoverage::unbounded(),
        ),
        RenderObjectTemporalState::new(RenderTemporalSupport::unbounded()),
    );
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(object_id, object_state);
    scene
        .commit(insert)
        .expect("insert retained capture object");

    let representation_id = scene
        .allocate_representation_id(object_id)
        .expect("retained capture representation id");
    let surface = RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
        .expect("surface protocol")
        .with_oriented_surface(
            RenderOrientedSurfaceProtocolEvidence::exact(
                RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
            )
            .expect("oriented surface protocol"),
        )
        .with_semantic_input_requirement(RenderSurfaceSemanticInputRequirement::current());
    let representation = RenderRepresentationRecord::new(
        representation_id,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
        RenderRefinementEvidence::none(),
        Some(surface),
        None,
    )
    .expect("retained capture representation");
    let participation = RenderObjectParticipation::new(
        vec![representation],
        Some(RenderMaterialAssignment::new(
            RenderDiffuseMaterial::new(0.8).expect("retained capture material"),
        )),
        Some(
            RenderDirectionalEmitter::new([0.0, 1.0, 0.0], 550e-9, 8.0)
                .expect("retained capture illumination"),
        ),
    )
    .expect("retained capture participation");
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object_id, participation);
    scene
        .commit(attach)
        .expect("attach retained capture participation");

    let shutter = RenderTimeInterval::instant(
        RenderTimePoint::from_seconds(0.0).expect("finite retained capture time"),
    );
    let observation = RenderObservationSpec::Perspective(
        RenderPerspectiveObservation::new(
            RenderAffineTransform3::identity(),
            std::f64::consts::FRAC_PI_3,
            1.0,
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("retained capture observation"),
    );
    let topology = || {
        RenderResultTopology::sample_lattice_2d(EXTENT, EXTENT).expect("retained capture lattice")
    };
    let mut builder = RenderRequestBuilder::new(shutter);
    let observation_handle = builder.add_observation(observation);
    builder
        .add_output(
            &observation_handle,
            RenderOutputSpec::new(
                RenderOutputValue::Radiance {
                    representation: RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                        550e-9,
                    )
                    .expect("retained capture spectral radiance"),
                },
                topology(),
                RenderSemanticTolerance::absolute(2.0e-4).expect("retained capture tolerance"),
            )
            .expect("retained capture radiance output"),
        )
        .expect("own observation");
    builder
        .add_output(
            &observation_handle,
            RenderOutputSpec::new(
                RenderOutputValue::ObjectIdentity,
                topology(),
                RenderSemanticTolerance::exact(),
            )
            .expect("retained capture identity output"),
        )
        .expect("own observation");
    let request = builder.finish().expect("retained capture request");

    let semantic_inputs = [RenderSurfaceSemanticInputBinding::new(
        representation_id,
        RenderSurfaceSemanticInput::sphere(
            [0.0, 0.0, 0.0],
            1.0,
            RenderTemporalSupport::unbounded(),
        )
        .expect("retained capture sphere"),
    )
    .with_generation(RenderSurfaceSemanticInputGeneration::new(1))];
    let availability = [RenderRepresentationAvailabilityFact::new(
        representation_id,
        RenderRepresentationAvailabilityState::Available,
    )];

    let mut allocator = GpuWorkResourceIdAllocator::new();
    let mut target = |name: &str| {
        allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::ordinary_owned_2d(
                    format!("{label} {name}"),
                    GpuResourceLifetime::Retained,
                    GpuReconstruction::SourceBacked,
                    EXTENT,
                    EXTENT,
                    GpuTextureFormat::R32Uint,
                    [
                        GpuTextureUsage::CopyDestination,
                        GpuTextureUsage::CopySource,
                    ],
                    GpuTextureInitialization::Uninitialized,
                )
                .expect("retained capture output descriptor"),
            )
            .expect("retained capture output handle")
    };
    let radiance_target = target("radiance");
    let identity_target = target("identity");
    let output_bindings = [
        RenderOutputBinding::new(
            request.output_handle(0).expect("request output handle"),
            RenderOutputDestination::SampleLatticeTexture(radiance_target),
        ),
        RenderOutputBinding::new(
            request.output_handle(1).expect("request output handle"),
            RenderOutputDestination::SampleLatticeTexture(identity_target.clone()),
        ),
    ];

    let admitted = admit_render(
        &RenderInvocation::new(
            scene.snapshot(),
            request.clone(),
            semantic_inputs.to_vec(),
            Vec::new(),
            availability.to_vec(),
            output_bindings.to_vec(),
        )
        .expect("valid correlated invocation"),
        context,
    )
    .expect("retained capture admission");
    (admitted, object_id, identity_target)
}

#[test]
fn associated_occurrence_exposes_exact_retained_capture_and_identity_decoder() {
    let Some((context, _)) = retained_context(GpuTextureFormat::R32Uint) else {
        return;
    };
    let (admitted, object_id, identity_target) =
        admitted_retained_capture_render(&context, "associated output proof");
    let mut session = RenderExecutionSession::new();
    let occurrence = session
        .prepare(admitted, &context, None)
        .expect("retained multi-output occurrence");
    let renderer_submission = pollster::block_on(context.submit_work(
        "associated multi-output renderer work",
        occurrence.work_set().fragments().iter().cloned(),
    ))
    .expect("associated renderer submission");
    assert_eq!(renderer_submission.status(), GpuSubmissionStatus::Accepted);

    let associated = session
        .associate_submission(occurrence, &renderer_submission)
        .expect("exact retained occurrence association");
    let Err(pending_capture_error) = associated.request_radiance_capture(
        &associated
            .admitted_plan()
            .plan()
            .request()
            .output_handle(0)
            .expect("capture output"),
    ) else {
        panic!("pending renderer submission cannot mint capture")
    };
    assert_eq!(
        pending_capture_error.kind(),
        RenderRadianceCaptureRequestErrorKind::RendererSubmissionPending
    );
    assert_eq!(
        associated
            .object_identity_decoder(
                &associated
                    .admitted_plan()
                    .plan()
                    .request()
                    .output_handle(1)
                    .expect("decoder output"),
            )
            .expect_err("pending renderer submission cannot expose decoder")
            .kind(),
        RenderObjectIdentityDecoderErrorKind::RendererSubmissionPending
    );

    wait_for_submission(&context, &renderer_submission);
    session.reconcile();
    assert!(!session.is_in_flight());
    drop(session);

    assert_eq!(
        associated.submission_status(),
        GpuSubmissionStatus::Completed
    );
    // A new request with the same semantic output positions must not acquire the already
    // associated occurrence's capture or decoder authority.
    let (foreign_admitted, _, _) =
        admitted_retained_capture_render(&context, "foreign request correlation");
    let foreign_request = foreign_admitted.admitted_plan().plan().request();
    assert_eq!(associated.admitted_plan().plan().request(), foreign_request);
    let foreign_radiance = foreign_request.output_handle(0).expect("foreign radiance");
    let foreign_identity = foreign_request.output_handle(1).expect("foreign identity");
    assert_ne!(
        &foreign_radiance,
        &associated
            .admitted_plan()
            .plan()
            .request()
            .output_handle(0)
            .expect("own radiance"),
    );
    let capture_error = associated
        .request_radiance_capture(&foreign_radiance)
        .err()
        .expect("foreign request must not capture this occurrence");
    assert_eq!(
        capture_error.kind(),
        RenderRadianceCaptureRequestErrorKind::OutputIndexOutOfRange
    );
    assert_eq!(capture_error.output(), &foreign_radiance);
    let decoder_error = associated
        .object_identity_decoder(&foreign_identity)
        .expect_err("foreign request must not select this occurrence decoder");
    assert_eq!(
        decoder_error.kind(),
        RenderObjectIdentityDecoderErrorKind::OutputIndexOutOfRange
    );
    assert_eq!(decoder_error.output(), &foreign_identity);

    assert_eq!(
        associated
            .object_identity_decoder(
                &associated
                    .admitted_plan()
                    .plan()
                    .request()
                    .output_handle(0)
                    .expect("decoder output"),
            )
            .expect_err("radiance output is not object identity")
            .kind(),
        RenderObjectIdentityDecoderErrorKind::OutputNotObjectIdentity
    );
    let decoder = associated
        .object_identity_decoder(
            &associated
                .admitted_plan()
                .plan()
                .request()
                .output_handle(1)
                .expect("decoder output"),
        )
        .expect("completed identity output decoder");

    let capture_request = associated
        .request_radiance_capture(
            &associated
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("capture output"),
        )
        .expect("completed retained radiance capture request");
    assert_eq!(capture_request.output().position(), 0);
    let capture_operation = GpuReadbackOperation::new(
        capture_request.source().clone(),
        capture_request.readback_id(),
    )
    .expect("retained radiance readback operation");
    let capture_fragment = GpuWorkFragment::build("retained radiance readback", |work| {
        work.operation("read retained radiance", capture_operation)?;
        Ok(())
    })
    .expect("retained radiance readback fragment");
    let capture_submission = pollster::block_on(
        context.submit_work("retained radiance readback submission", [capture_fragment]),
    )
    .expect("retained radiance readback submission");
    wait_for_readback(&context, &capture_submission, capture_request.readback_id());
    let captured = associated
        .capture_radiance(capture_request, &context, &capture_submission)
        .expect("interpret retained radiance");
    assert_eq!(captured.output().position(), 0);
    assert_eq!(
        captured.topology().sample_lattice_dimensions(),
        Some((4, 4))
    );
    assert_eq!(captured.samples().len(), 16);
    assert!(captured.samples().iter().all(|sample| sample.is_finite()));

    let identity_readback = GpuReadbackId::allocate().expect("identity readback id");
    let identity_operation = GpuReadbackOperation::new(
        GpuTextureCopyRegion::whole_base_mip(&identity_target)
            .expect("identity whole-base-mip source")
            .into(),
        identity_readback,
    )
    .expect("retained identity readback operation");
    let identity_fragment = GpuWorkFragment::build("retained identity readback", |work| {
        work.operation("read retained identity", identity_operation)?;
        Ok(())
    })
    .expect("retained identity readback fragment");
    let identity_submission = pollster::block_on(
        context.submit_work("retained identity readback submission", [identity_fragment]),
    )
    .expect("retained identity readback submission");
    wait_for_readback(&context, &identity_submission, identity_readback);
    let bytes = match identity_submission
        .readback(identity_readback)
        .expect("identity readback correlation")
        .status()
    {
        GpuReadbackStatus::Ready(bytes) => bytes.as_bytes().to_vec(),
        status => panic!("identity readback not ready after wait: {status:?}"),
    };
    assert!(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_ne_bytes(*word))
            .filter_map(|word| decoder.decode(word))
            .any(|observed| observed == object_id),
        "completed identity output must contain the fixture object in the execution-local codebook"
    );
}

#[test]
fn associated_occurrence_failure_is_machine_actionable() {
    let Some((context, _)) = retained_context(GpuTextureFormat::R32Uint) else {
        return;
    };
    let (admitted, _, _) = admitted_retained_capture_render(&context, "failed output proof");
    let mut session = RenderExecutionSession::new();
    let occurrence = session
        .prepare(admitted, &context, None)
        .expect("failed-output occurrence");
    let submission = pollster::block_on(context.submit_work(
        "failed associated renderer work",
        occurrence.work_set().fragments().iter().cloned(),
    ))
    .expect("failed-output submission");
    let associated = session
        .associate_submission(occurrence, &submission)
        .expect("associate before context failure");
    drop(context);
    assert!(matches!(
        submission.status(),
        GpuSubmissionStatus::Failed(failure)
            if failure.kind() == GpuSubmissionFailureKind::ContextDropped
    ));
    let Err(capture_error) = associated.request_radiance_capture(
        &associated
            .admitted_plan()
            .plan()
            .request()
            .output_handle(0)
            .expect("capture output"),
    ) else {
        panic!("failed renderer submission cannot mint capture")
    };
    assert_eq!(
        capture_error.kind(),
        RenderRadianceCaptureRequestErrorKind::RendererSubmissionFailed
    );
    assert_eq!(
        capture_error.gpu_failure_kind(),
        Some(GpuSubmissionFailureKind::ContextDropped)
    );
    let identity_error = associated
        .object_identity_decoder(
            &associated
                .admitted_plan()
                .plan()
                .request()
                .output_handle(1)
                .expect("decoder output"),
        )
        .expect_err("failed renderer submission cannot expose decoder");
    assert_eq!(
        identity_error.kind(),
        RenderObjectIdentityDecoderErrorKind::RendererSubmissionFailed
    );
    assert_eq!(
        identity_error.gpu_failure_kind(),
        Some(GpuSubmissionFailureKind::ContextDropped)
    );
}

#[test]
fn associated_capture_requests_do_not_cross_correlate_between_occurrences() {
    let Some((context, _)) = retained_context(GpuTextureFormat::R32Uint) else {
        return;
    };
    let (first_admitted, _, _) = admitted_retained_capture_render(&context, "first exact capture");
    let (second_admitted, _, _) =
        admitted_retained_capture_render(&context, "second exact capture");
    let mut first_session = RenderExecutionSession::new();
    let mut second_session = RenderExecutionSession::new();
    let first = first_session
        .prepare(first_admitted, &context, None)
        .expect("first exact occurrence");
    let second = second_session
        .prepare(second_admitted, &context, None)
        .expect("second exact occurrence");
    let fragments = first
        .work_set()
        .fragments()
        .iter()
        .cloned()
        .chain(second.work_set().fragments().iter().cloned())
        .collect::<Vec<_>>();
    let renderer_submission =
        pollster::block_on(context.submit_work("shared exact renderer submission", fragments))
            .expect("shared exact renderer submission");
    let first_associated = first_session
        .associate_submission(first, &renderer_submission)
        .expect("first exact association");
    let second_associated = second_session
        .associate_submission(second, &renderer_submission)
        .expect("second exact association");
    wait_for_submission(&context, &renderer_submission);
    first_session.reconcile();
    second_session.reconcile();

    let request = first_associated
        .request_radiance_capture(
            &first_associated
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("capture output"),
        )
        .expect("first exact capture request");
    let operation = GpuReadbackOperation::new(request.source().clone(), request.readback_id())
        .expect("first exact capture readback");
    let fragment = GpuWorkFragment::build("first exact capture readback", |work| {
        work.operation("read first exact capture", operation)?;
        Ok(())
    })
    .expect("first exact capture fragment");
    let product_submission = pollster::block_on(
        context.submit_work("first exact capture product submission", [fragment]),
    )
    .expect("first exact capture product submission");
    wait_for_readback(&context, &product_submission, request.readback_id());

    let error = second_associated
        .capture_radiance(request, &context, &product_submission)
        .expect_err("same submission and output index must not cross occurrence destinations");
    assert_eq!(
        error.kind(),
        RenderRadianceCaptureErrorKind::RequestCorrelationMismatch
    );
}

#[test]
fn associated_capture_rejects_a_superseded_retained_writer() {
    let Some((context, _)) = retained_context(GpuTextureFormat::R32Uint) else {
        return;
    };
    let (admitted, _, _) = admitted_retained_capture_render(&context, "stale retained capture");
    let mut session = RenderExecutionSession::new();

    let first = session
        .prepare(admitted.clone(), &context, None)
        .expect("first retained writer");
    let first_submission = pollster::block_on(context.submit_work(
        "first retained writer submission",
        first.work_set().fragments().iter().cloned(),
    ))
    .expect("first retained writer submission");
    let first_associated = session
        .associate_submission(first, &first_submission)
        .expect("first retained writer association");
    wait_for_submission(&context, &first_submission);
    session.reconcile();

    let second = session
        .prepare(admitted, &context, None)
        .expect("second retained writer");
    let second_submission = pollster::block_on(context.submit_work(
        "second retained writer submission",
        second.work_set().fragments().iter().cloned(),
    ))
    .expect("second retained writer submission");
    let _second_associated = session
        .associate_submission(second, &second_submission)
        .expect("second retained writer association");
    wait_for_submission(&context, &second_submission);
    session.reconcile();

    let request = first_associated
        .request_radiance_capture(
            &first_associated
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("capture output"),
        )
        .expect("old occurrence can still form an exact readback request");
    let operation = GpuReadbackOperation::new(request.source().clone(), request.readback_id())
        .expect("stale capture readback operation");
    let fragment = GpuWorkFragment::build("stale retained capture readback", |work| {
        work.operation("read stale retained destination", operation)?;
        Ok(())
    })
    .expect("stale retained capture readback fragment");
    let product_submission =
        pollster::block_on(context.submit_work("stale retained capture submission", [fragment]))
            .expect("stale retained capture submission");
    wait_for_readback(&context, &product_submission, request.readback_id());

    let error = first_associated
        .capture_radiance(request, &context, &product_submission)
        .expect_err("newer renderer write must invalidate old capture interpretation");
    assert_eq!(
        error.kind(),
        RenderRadianceCaptureErrorKind::RendererWriteNoLongerCurrent
    );
}

#[test]
fn retained_sessions_require_exact_occurrence_membership_and_compose_independently() {
    let Some((context, _)) = retained_context(GpuTextureFormat::R32Uint) else {
        return;
    };
    let mut first = RenderExecutionSession::new();
    let mut second = RenderExecutionSession::new();

    let first_occurrence = first
        .prepare(
            admitted_identity_render(&context, "first unrelated", 1),
            &context,
            None,
        )
        .expect("first retained preparation");
    let second_occurrence = second
        .prepare(
            admitted_identity_render(&context, "second unrelated", 1),
            &context,
            None,
        )
        .expect("second retained preparation");
    let second_submission = pollster::block_on(context.submit_work(
        "second retained occurrence",
        second_occurrence.work_set().fragments().iter().cloned(),
    ))
    .expect("second retained submission");
    assert!(matches!(
        first.associate_submission(first_occurrence, &second_submission),
        Err(RenderExecutionSessionError::SubmissionMissingRendererWork)
    ));
    second
        .associate_submission(second_occurrence, &second_submission)
        .expect("matching second occurrence");
    wait_for_submission(&context, &second_submission);
    second.reconcile();
    assert!(!second.is_in_flight());

    let first_occurrence = first
        .prepare(
            admitted_identity_render(&context, "first combined", 1),
            &context,
            None,
        )
        .expect("first combined preparation");
    let second_occurrence = second
        .prepare(
            admitted_identity_render(&context, "second combined", 1),
            &context,
            None,
        )
        .expect("second combined preparation");
    let fragments = first_occurrence
        .work_set()
        .fragments()
        .iter()
        .cloned()
        .chain(second_occurrence.work_set().fragments().iter().cloned())
        .collect::<Vec<_>>();
    let combined_submission =
        pollster::block_on(context.submit_work("combined retained occurrences", fragments))
            .expect("combined retained submission");
    first
        .associate_submission(first_occurrence, &combined_submission)
        .expect("combined first occurrence");
    second
        .associate_submission(second_occurrence, &combined_submission)
        .expect("combined second occurrence");
    wait_for_submission(&context, &combined_submission);
    first.reconcile();
    second.reconcile();
    assert!(!first.is_in_flight());
    assert!(!second.is_in_flight());

    let multi_output = first
        .prepare(
            admitted_identity_render(&context, "partial occurrence", 2),
            &context,
            None,
        )
        .expect("multi-output retained preparation");
    assert_eq!(multi_output.work_set().fragments().len(), 2);
    let partial_submission = pollster::block_on(context.submit_work(
        "partial retained occurrence",
        [multi_output.work_set().fragments()[0].clone()],
    ))
    .expect("partial retained submission");
    assert!(matches!(
        first.associate_submission(multi_output, &partial_submission),
        Err(RenderExecutionSessionError::SubmissionMissingRendererWork)
    ));
    wait_for_submission(&context, &partial_submission);
}

#[test]
fn associated_temporal_r32float_radiance_is_capturable_through_exact_occurrence() {
    let Some((context, _)) = retained_context(GpuTextureFormat::R32Float) else {
        return;
    };
    let mut session = RenderExecutionSession::new();
    let admitted = admitted_temporal_radiance_render(&context, "capturable temporal radiance");
    let evaluation = RenderEvaluationSelection::new(
        admitted
            .admitted_plan()
            .plan()
            .request()
            .output_handle(0)
            .expect("evaluation output"),
        2,
        2,
    )
    .expect("full temporal evaluation");

    let occurrence = session
        .prepare(admitted, &context, Some(evaluation.clone()))
        .expect("temporal retained occurrence");
    let temporal = occurrence
        .radiance_output(
            &occurrence
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("radiance output"),
        )
        .and_then(|output| output.temporal_execution_evidence())
        .expect("ordinary R32Float radiance must expose temporal evidence");
    assert_eq!(temporal.requested_extent, (2, 2));
    assert_eq!(temporal.evaluation_extent, (2, 2));
    assert!(temporal.history_reset);
    assert_eq!(temporal.history_age, 0);

    let renderer_submission = pollster::block_on(context.submit_work(
        "capturable temporal renderer work",
        occurrence.work_set().fragments().iter().cloned(),
    ))
    .expect("temporal renderer submission");
    let associated = session
        .associate_submission(occurrence, &renderer_submission)
        .expect("exact temporal occurrence association");
    wait_for_submission(&context, &renderer_submission);
    session.reconcile();
    assert!(!session.is_in_flight());

    let request = associated
        .request_radiance_capture(
            &associated
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("capture output"),
        )
        .expect("completed R32Float retained capture request");
    let readback_id = request.readback_id();
    let operation = GpuReadbackOperation::new(request.source().clone(), readback_id)
        .expect("R32Float retained capture readback");
    let fragment = GpuWorkFragment::build("R32Float retained capture readback", |work| {
        work.operation("read R32Float retained radiance", operation)?;
        Ok(())
    })
    .expect("R32Float retained capture fragment");
    let product_submission =
        pollster::block_on(context.submit_work("R32Float retained capture submission", [fragment]))
            .expect("R32Float retained capture submission");
    wait_for_readback(&context, &product_submission, readback_id);

    let captured = associated
        .capture_radiance(request, &context, &product_submission)
        .expect("interpret ordinary R32Float retained radiance");
    assert_eq!(captured.output().position(), 0);
    assert_eq!(
        captured.topology().sample_lattice_dimensions(),
        Some((2, 2))
    );
    assert_eq!(captured.samples().len(), 4);
    assert!(captured.samples().iter().all(|sample| sample.is_finite()));
}

#[test]
fn retained_radiance_sessions_use_independent_composable_graph_wiring() {
    let Some((context, _)) = retained_context(GpuTextureFormat::R32Float) else {
        return;
    };
    let mut first = RenderExecutionSession::new();
    let mut second = RenderExecutionSession::new();

    let first_occurrence = first
        .prepare(
            admitted_temporal_radiance_render(&context, "first radiance output"),
            &context,
            None,
        )
        .expect("first radiance preparation");
    let second_occurrence = second
        .prepare(
            admitted_temporal_radiance_render(&context, "second radiance output"),
            &context,
            None,
        )
        .expect("second radiance preparation");

    let first_key = first_occurrence
        .radiance_output(
            &first_occurrence
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("radiance output"),
        )
        .expect("first composable radiance output")
        .export_relationship()
        .export_key()
        .clone();
    let second_key = second_occurrence
        .radiance_output(
            &second_occurrence
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("radiance output"),
        )
        .expect("second composable radiance output")
        .export_relationship()
        .export_key()
        .clone();
    assert_ne!(first_key, second_key);

    let fragments = first_occurrence
        .work_set()
        .fragments()
        .iter()
        .cloned()
        .chain(second_occurrence.work_set().fragments().iter().cloned())
        .collect::<Vec<_>>();
    let submission =
        pollster::block_on(context.submit_work("independent radiance sessions", fragments))
            .expect("combined radiance submission");
    first
        .associate_submission(first_occurrence, &submission)
        .expect("first radiance association");
    second
        .associate_submission(second_occurrence, &submission)
        .expect("second radiance association");
    wait_for_submission(&context, &submission);
    first.reconcile();
    second.reconcile();
}

#[test]
fn retained_session_abandonment_and_device_generation_reset_preserve_temporal_truth() {
    let Some((mut context, descriptor)) = retained_context(GpuTextureFormat::R32Float) else {
        return;
    };
    let mut session = RenderExecutionSession::new();
    let admitted = admitted_temporal_radiance_render(&context, "temporal retained output");
    let evaluation = RenderEvaluationSelection::new(
        admitted
            .admitted_plan()
            .plan()
            .request()
            .output_handle(0)
            .expect("evaluation output"),
        2,
        2,
    )
    .expect("temporal evaluation");

    let first = session
        .prepare(admitted.clone(), &context, Some(evaluation.clone()))
        .expect("first temporal preparation");
    let first_evidence = first
        .radiance_output(
            &first
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("radiance output"),
        )
        .and_then(|output| output.temporal_execution_evidence())
        .expect("first temporal evidence");
    assert!(first_evidence.history_reset);
    assert_eq!(first_evidence.history_age, 0);
    let first_generation = first_evidence.history_generation;
    let first_submission = pollster::block_on(context.submit_work(
        "first temporal retained occurrence",
        first.work_set().fragments().iter().cloned(),
    ))
    .expect("first temporal submission");
    session
        .associate_submission(first, &first_submission)
        .expect("accept first temporal occurrence");
    wait_for_submission(&context, &first_submission);
    session.reconcile();

    let abandoned = session
        .prepare(admitted.clone(), &context, Some(evaluation.clone()))
        .expect("abandoned temporal preparation");
    let abandoned_evidence = abandoned
        .radiance_output(
            &abandoned
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("radiance output"),
        )
        .and_then(|output| output.temporal_execution_evidence())
        .expect("abandoned temporal evidence");
    assert!(!abandoned_evidence.history_reset);
    assert_eq!(abandoned_evidence.history_age, 1);
    let abandoned_export = abandoned
        .radiance_output(
            &abandoned
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("radiance output"),
        )
        .expect("abandoned radiance output")
        .export_relationship()
        .export_key()
        .clone();
    // The GPU may execute this work despite the caller never associating its occurrence.
    let unassociated_submission = pollster::block_on(context.submit_work(
        "unassociated retained occurrence",
        abandoned.work_set().fragments().iter().cloned(),
    ))
    .expect("unassociated submission");
    drop(abandoned);
    session.reconcile();

    let after_abandonment = session
        .prepare(admitted, &context, Some(evaluation.clone()))
        .expect("post-abandonment temporal preparation");
    let after_abandonment_evidence = after_abandonment
        .radiance_output(
            &after_abandonment
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("radiance output"),
        )
        .and_then(|output| output.temporal_execution_evidence())
        .expect("post-abandonment temporal evidence");
    assert!(after_abandonment_evidence.history_reset);
    assert_eq!(after_abandonment_evidence.history_age, 0);
    assert!(after_abandonment_evidence.history_generation > first_generation);
    assert_ne!(
        abandoned_export,
        after_abandonment
            .radiance_output(
                &after_abandonment
                    .admitted_plan()
                    .plan()
                    .request()
                    .output_handle(0)
                    .expect("radiance output")
            )
            .expect("fresh radiance output")
            .export_relationship()
            .export_key()
            .clone(),
        "an unassociated occurrence must not alias new graph wiring",
    );
    wait_for_submission(&context, &unassociated_submission);
    drop(after_abandonment);
    session.reconcile();

    pollster::block_on(context.replace_device_generation(descriptor))
        .expect("replace retained-session device generation");
    let new_generation_admitted =
        admitted_temporal_radiance_render(&context, "new-generation temporal output");
    let after_generation_change = session
        .prepare(
            new_generation_admitted.clone(),
            &context,
            Some(
                RenderEvaluationSelection::new(
                    new_generation_admitted
                        .admitted_plan()
                        .plan()
                        .request()
                        .output_handle(0)
                        .expect("new-generation output"),
                    2,
                    2,
                )
                .expect("fresh evaluation"),
            ),
        )
        .expect("new-generation temporal preparation");
    let reset_evidence = after_generation_change
        .radiance_output(
            &after_generation_change
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("radiance output"),
        )
        .and_then(|output| output.temporal_execution_evidence())
        .expect("new-generation temporal evidence");
    assert!(reset_evidence.history_reset);
    assert_eq!(reset_evidence.history_age, 0);
    assert!(reset_evidence.history_generation > first_generation);
}

#[test]
fn retained_subnative_unassociated_work_cannot_reuse_in_place_history() {
    let Some((context, _)) = retained_context(GpuTextureFormat::R32Float) else {
        return;
    };
    let mut session = RenderExecutionSession::new();
    let admitted =
        admitted_temporal_radiance_render_at_extent(&context, "sub-native in-place history", 4);
    let evaluation = RenderEvaluationSelection::new(
        admitted
            .admitted_plan()
            .plan()
            .request()
            .output_handle(0)
            .expect("evaluation output"),
        2,
        2,
    )
    .expect("sub-native evaluation");

    let first = session
        .prepare(admitted.clone(), &context, Some(evaluation.clone()))
        .expect("bootstrap sub-native history");
    let bootstrap = first
        .radiance_output(
            &first
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("radiance output"),
        )
        .and_then(|output| output.temporal_execution_evidence())
        .expect("bootstrap temporal evidence");
    assert!(bootstrap.history_reset);
    let first_submission = pollster::block_on(context.submit_work(
        "bootstrap sub-native retained occurrence",
        first.work_set().fragments().iter().cloned(),
    ))
    .expect("bootstrap submission");
    session
        .associate_submission(first, &first_submission)
        .expect("associate bootstrap occurrence");
    wait_for_submission(&context, &first_submission);
    session.reconcile();

    let abandoned = session
        .prepare(admitted.clone(), &context, Some(evaluation.clone()))
        .expect("prepare sub-native in-place update");
    let previous = abandoned
        .radiance_output(
            &abandoned
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("radiance output"),
        )
        .and_then(|output| output.temporal_execution_evidence())
        .expect("previous temporal evidence");
    assert!(!previous.history_reset);
    assert_eq!(previous.history_age, 1);
    let unassociated_submission = pollster::block_on(context.submit_work(
        "submitted but unassociated sub-native update",
        abandoned.work_set().fragments().iter().cloned(),
    ))
    .expect("unassociated in-place submission");
    drop(abandoned);
    session.reconcile();

    let renewed = session
        .prepare(admitted, &context, Some(evaluation.clone()))
        .expect("fresh history after unassociated GPU write");
    let next = renewed
        .radiance_output(
            &renewed
                .admitted_plan()
                .plan()
                .request()
                .output_handle(0)
                .expect("radiance output"),
        )
        .and_then(|output| output.temporal_execution_evidence())
        .expect("renewed temporal evidence");
    assert!(next.history_reset);
    assert_eq!(next.history_age, 0);
    assert!(next.history_generation > bootstrap.history_generation);
    wait_for_submission(&context, &unassociated_submission);
}
