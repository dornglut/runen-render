use super::*;
use crate::admission::RenderOutputDestination;
use crate::proofs::execution::{MaintainedExecutionFixture, maintained_fixture};
use crate::request::{
    RenderObservationSpec, RenderOutputSpec, RenderOutputValue, RenderPerspectiveObservation,
    RenderRequestedOutput, RenderSamplingSupport, RenderSemanticTolerance,
};
use runen_gpu::{
    GpuCapabilityProfile, GpuContextDescriptor, GpuContextRequestErrorCategory, GpuFormatRole,
    GpuReconstruction, GpuResourceLifetime, GpuTextureDescriptor, GpuTextureFormat,
    GpuTextureInitialization, GpuTextureUsage, GpuWorkResourceIdAllocator,
};
use std::time::{Duration, Instant};

fn context() -> Option<GpuContext> {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::CopyDestination)
            .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::CopySource)
            .with_label("P1 associated terminal-failure proof");
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => Some(context),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1"),
                "P1 terminal-failure conformance requires a public RunenGPU adapter",
            );
            None
        }
        Err(error) => panic!("P1 terminal-failure context: {error}"),
    }
}

fn temporal_fixture(extent: u32) -> MaintainedExecutionFixture {
    let mut fixture = maintained_fixture();
    let RenderObservationSpec::Perspective(previous) = fixture.request.observations()[0] else {
        unreachable!("maintained fixture has a perspective observation")
    };
    let observation = RenderPerspectiveObservation::new(
        previous.observation_to_scene(),
        previous.vertical_field_of_view_radians(),
        previous.aspect_ratio(),
        previous.shutter(),
        RenderSamplingSupport::perspective_lattice_cell(),
    )
    .expect("temporal perspective observation");
    fixture.request = RenderRequest::new(
        fixture.request.render_interval(),
        vec![RenderObservationSpec::Perspective(observation)],
        vec![RenderRequestedOutput::new(
            0,
            RenderOutputSpec::new(
                RenderOutputValue::Radiance {
                    representation: RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                        550e-9,
                    )
                    .expect("spectral radiance"),
                },
                RenderResultTopology::sample_lattice_2d(extent, extent).expect("temporal lattice"),
                RenderSemanticTolerance::exact(),
            )
            .expect("temporal output"),
        )],
    )
    .expect("temporal request");
    for binding in &mut fixture.semantic_inputs {
        *binding = binding
            .clone()
            .with_generation(RenderSurfaceSemanticInputGeneration::new(1));
    }
    fixture
}

fn admit(fixture: &MaintainedExecutionFixture, context: &GpuContext) -> AdmittedRender {
    let (width, height) = fixture.request.outputs()[0]
        .spec()
        .topology()
        .sample_lattice_dimensions()
        .expect("lattice dimensions");
    let destination = GpuWorkResourceIdAllocator::new()
        .allocate_texture_handle(
            GpuTextureDescriptor::ordinary_owned_2d(
                "associated failure radiance destination",
                GpuResourceLifetime::Retained,
                GpuReconstruction::SourceBacked,
                width,
                height,
                GpuTextureFormat::R32Float,
                [
                    GpuTextureUsage::CopyDestination,
                    GpuTextureUsage::CopySource,
                ],
                GpuTextureInitialization::Uninitialized,
            )
            .expect("radiance destination descriptor"),
        )
        .expect("radiance destination identity");
    admit_render(
        &fixture.scene,
        &fixture.request,
        &fixture.semantic_inputs,
        &[],
        &fixture.availability,
        &[RenderOutputBinding::new(
            0,
            RenderOutputDestination::SampleLatticeTexture(destination),
        )],
        context,
    )
    .expect("ordinary temporal admission")
}

fn wait_for_completion(context: &GpuContext, submission: &GpuSubmission) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => return,
            GpuSubmissionStatus::Accepted if Instant::now() < deadline => std::thread::yield_now(),
            status => panic!("bootstrap history did not complete: {status:?}"),
        }
    }
}

fn prove_associated_failure_invalidates_history(requested_extent: u32) {
    let Some(context) = context() else {
        return;
    };
    let affinity = context.affinity();
    let admitted = admit(&temporal_fixture(requested_extent), &context);
    let evaluation = RenderEvaluationSelection::new(0, 2, 2).expect("2x2 evaluation");
    let mut session = RenderExecutionSession::new();

    let bootstrap = session
        .prepare(admitted.clone(), &context, Some(evaluation))
        .expect("bootstrap occurrence");
    let initial = bootstrap
        .radiance_output(0)
        .and_then(|output| output.temporal_execution_evidence())
        .expect("bootstrap temporal evidence");
    assert!(initial.history_reset);
    assert_eq!(initial.history_age, 0);
    assert_eq!(
        initial.requested_extent,
        (requested_extent, requested_extent)
    );
    assert_eq!(initial.evaluation_extent, (2, 2));
    assert_eq!(
        initial.camera_reprojection_revision.is_some(),
        requested_extent == 2
    );
    let successful = pollster::block_on(context.submit_work(
        "successful retained history control",
        bootstrap.work_set().fragments().iter().cloned(),
    ))
    .expect("bootstrap submission");
    session
        .associate_submission(bootstrap, &successful)
        .expect("associate bootstrap occurrence");
    wait_for_completion(&context, &successful);
    session.reconcile();

    let occurrence = session
        .prepare(admitted, &context, Some(evaluation))
        .expect("next compatible occurrence");
    let retained = occurrence
        .radiance_output(0)
        .and_then(|output| output.temporal_execution_evidence())
        .expect("retained temporal evidence");
    assert!(!retained.history_reset);
    assert_eq!(retained.history_generation, initial.history_generation);
    assert_eq!(retained.history_age, 1);
    assert_eq!(retained.phase, 1);
    assert!(session.resources.has_temporal_history(0));

    let failing = pollster::block_on(context.submit_work(
        "associated occurrence whose context will be dropped",
        occurrence.work_set().fragments().iter().cloned(),
    ))
    .expect("accepted submission");
    assert_eq!(failing.status(), GpuSubmissionStatus::Accepted);
    session
        .associate_submission(occurrence, &failing)
        .expect("associate all required renderer work before terminal failure");
    assert!(session.is_in_flight());

    // No failure injection: the public RunenGPU owner-drop contract terminalizes the
    // accepted submission. Submission handles and the renderer session retain no context owner.
    drop(context);
    let GpuSubmissionStatus::Failed(failure) = failing.status() else {
        panic!("last context drop must fail the accepted submission")
    };
    assert_eq!(failure.kind(), GpuSubmissionFailureKind::ContextDropped);

    session.reconcile();
    assert!(!session.is_in_flight());
    assert!(session.submission.is_none());
    assert!(session.prepared_occurrence.is_none());
    assert_eq!(session.affinity, Some(affinity));
    // Inspect the owner before any new-context preparation can mask a false successful
    // advance through its independent affinity-reset law. Failed history must be absent,
    // not retained with either the old successful age or the failed occurrence's next age.
    assert!(!session.resources.has_temporal_history(0));
    session.reconcile();
    assert!(!session.resources.has_temporal_history(0));
    assert_eq!(successful.status(), GpuSubmissionStatus::Completed);
}

#[test]
fn associated_context_drop_failure_invalidates_camera_temporal_history() {
    prove_associated_failure_invalidates_history(2);
}

#[test]
fn associated_context_drop_failure_invalidates_subnative_temporal_history() {
    prove_associated_failure_invalidates_history(4);
}
