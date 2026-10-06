use runen_gpu::{
    GpuCapabilityProfile, GpuContext, GpuContextDescriptor, GpuContextRequestErrorCategory,
    GpuFormatRole, GpuReadbackOperation, GpuReconstruction, GpuResourceLifetime, GpuSubmission,
    GpuSubmissionStatus, GpuTextureDescriptor, GpuTextureFormat, GpuTextureInitialization,
    GpuTextureUsage, GpuWorkResourceIdAllocator,
};
use runen_render::admission::{
    RenderOutputBinding, RenderOutputDestination, RenderRepresentationAvailabilityFact,
    RenderRepresentationAvailabilityState,
};
use runen_render::participation::RenderObjectParticipation;
use runen_render::representation::{
    RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION, RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
    RenderOrientedSurfaceProtocolEvidence, RenderRefinementEvidence, RenderRepresentationRecord,
    RenderSurfaceProtocolEvidence,
};
use runen_render::request::{
    RenderObservationSpec, RenderOutputSpec, RenderOutputValue, RenderPerspectiveObservation,
    RenderRadiometricRepresentation, RenderRequest, RenderRequestedOutput, RenderResultTopology,
    RenderSamplingSupport, RenderSemanticTolerance,
};
use runen_render::scene::{RenderObjectState, RenderSceneStore, RenderSceneUpdate};
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
    AdmittedRender, PreparedRadianceOutput, PreparedRender, RenderAdmissionError,
    PreparedRenderOccurrence, RenderCapturedRadiance, RenderEvaluationSelection,
    RenderExecutionError, RenderExecutionErrorKind, RenderExecutionSession,
    RenderExecutionSessionError,
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
    let _ = RenderResultSubmissionError::observation_index;
    let _ = RenderResultSubmissionError::object_id;
    let _ = RenderResultSubmissionError::readback_cardinality;
    let _ = RenderResultSubmissionError::output_correlation;
    let _ = RenderResultSubmissionError::correlation_output_index;
    let _ = RenderResultSubmissionError::correlation_channel;
    let _ = RenderResultFormationError::kind;
    let _ = RenderResultFormationError::verification_eligibility_kind;
    let _ = RenderResultFormationError::output_index;
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
    let _ = PreparedRenderOccurrence::admitted_plan;
    let _ = PreparedRenderOccurrence::work_set;
    let _ = PreparedRenderOccurrence::radiance_outputs;
    let _ = PreparedRenderOccurrence::radiance_output;
    let selection = RenderEvaluationSelection::new(0, 64, 32).expect("non-zero extent");
    assert_eq!(selection.output_index(), 0);
    assert_eq!(selection.extent(), (64, 32));

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
        let _ = output.output_index();
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
    let request = RenderRequest::new(
        shutter,
        vec![observation],
        vec![RenderRequestedOutput::new(
            0,
            RenderOutputSpec::new(
                RenderOutputValue::ObjectIdentity,
                RenderResultTopology::sample_lattice_2d(1, 1).expect("1x1 public lattice"),
                RenderSemanticTolerance::exact(),
            )
            .expect("public object-identity output"),
        )],
    )
    .expect("public render request");

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
        0,
        RenderOutputDestination::SampleLatticeTexture(destination),
    )];

    let admitted = admit_render(
        &scene.snapshot(),
        &request,
        &semantic_inputs,
        &[],
        &availability,
        &output_bindings,
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

fn retained_context(
    format: GpuTextureFormat,
) -> Option<(GpuContext, GpuContextDescriptor)> {
    let descriptor = GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
        .require_format_role(format, GpuFormatRole::CopyDestination)
        .with_label("RunenRender retained-session public consumer");
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
    let object_id = scene.allocate_object_id().expect("retained public object id");
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
    let outputs = (0..output_count)
        .map(|_| {
            RenderRequestedOutput::new(
                0,
                RenderOutputSpec::new(
                    RenderOutputValue::ObjectIdentity,
                    RenderResultTopology::sample_lattice_2d(1, 1).expect("retained lattice"),
                    RenderSemanticTolerance::exact(),
                )
                .expect("retained identity output"),
            )
        })
        .collect::<Vec<_>>();
    let request = RenderRequest::new(shutter, vec![observation], outputs)
        .expect("retained render request");

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
            output_index,
            RenderOutputDestination::SampleLatticeTexture(destination),
        ));
    }

    admit_render(
        &scene.snapshot(),
        &request,
        &semantic_inputs,
        &[],
        &availability,
        &output_bindings,
        context,
    )
    .expect("retained ordinary admission")
}

fn admitted_temporal_radiance_render(context: &GpuContext, label: &str) -> AdmittedRender {
    let mut scene = RenderSceneStore::new();
    let object_id = scene.allocate_object_id().expect("temporal public object id");
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
    let participation = RenderObjectParticipation::new(vec![representation], None, None)
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
    let request = RenderRequest::new(
        shutter,
        vec![observation],
        vec![RenderRequestedOutput::new(
            0,
            RenderOutputSpec::new(
                RenderOutputValue::Radiance {
                    representation: RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                        550e-9,
                    )
                    .expect("visible spectral radiance"),
                },
                RenderResultTopology::sample_lattice_2d(2, 2).expect("temporal lattice"),
                RenderSemanticTolerance::exact(),
            )
            .expect("temporal radiance output"),
        )],
    )
    .expect("temporal render request");

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
                2,
                2,
                GpuTextureFormat::R32Float,
                [GpuTextureUsage::CopyDestination, GpuTextureUsage::CopySource],
                GpuTextureInitialization::Uninitialized,
            )
            .expect("temporal output descriptor"),
        )
        .expect("temporal output handle");
    let output_bindings = [RenderOutputBinding::new(
        0,
        RenderOutputDestination::SampleLatticeTexture(destination),
    )];

    admit_render(
        &scene.snapshot(),
        &request,
        &semantic_inputs,
        &[],
        &availability,
        &output_bindings,
        context,
    )
    .expect("temporal ordinary admission")
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
fn retained_session_abandonment_and_device_generation_reset_preserve_temporal_truth() {
    let Some((mut context, descriptor)) = retained_context(GpuTextureFormat::R32Float) else {
        return;
    };
    let mut session = RenderExecutionSession::new();
    let admitted = admitted_temporal_radiance_render(&context, "temporal retained output");
    let evaluation = RenderEvaluationSelection::new(0, 2, 2).expect("temporal evaluation");

    let first = session
        .prepare(admitted.clone(), &context, Some(evaluation))
        .expect("first temporal preparation");
    let first_evidence = first
        .radiance_output(0)
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
        .prepare(admitted.clone(), &context, Some(evaluation))
        .expect("abandoned temporal preparation");
    let abandoned_evidence = abandoned
        .radiance_output(0)
        .and_then(|output| output.temporal_execution_evidence())
        .expect("abandoned temporal evidence");
    assert!(!abandoned_evidence.history_reset);
    assert_eq!(abandoned_evidence.history_age, 1);
    drop(abandoned);
    session.reconcile();

    let after_abandonment = session
        .prepare(admitted, &context, Some(evaluation))
        .expect("post-abandonment temporal preparation");
    let after_abandonment_evidence = after_abandonment
        .radiance_output(0)
        .and_then(|output| output.temporal_execution_evidence())
        .expect("post-abandonment temporal evidence");
    assert!(!after_abandonment_evidence.history_reset);
    assert_eq!(after_abandonment_evidence.history_age, 1);
    drop(after_abandonment);
    session.reconcile();

    pollster::block_on(context.replace_device_generation(descriptor))
        .expect("replace retained-session device generation");
    let new_generation_admitted =
        admitted_temporal_radiance_render(&context, "new-generation temporal output");
    let after_generation_change = session
        .prepare(new_generation_admitted, &context, Some(evaluation))
        .expect("new-generation temporal preparation");
    let reset_evidence = after_generation_change
        .radiance_output(0)
        .and_then(|output| output.temporal_execution_evidence())
        .expect("new-generation temporal evidence");
    assert!(reset_evidence.history_reset);
    assert_eq!(reset_evidence.history_age, 0);
    assert!(reset_evidence.history_generation > first_generation);
}
