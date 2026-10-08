//! A standalone, headless public-API consumer of RunenRender and RunenGPU.
//!
//! The scene is renderer-owned, not an ECS. Radiance is monochrome 550 nm
//! spectral radiance; output PNGs are diagnostic mappings, not semantic RGB.

use runen_gpu::{
    GpuCapabilityProfile, GpuContext, GpuContextDescriptor, GpuContextRequestErrorCategory,
    GpuFormatRole, GpuReadbackId, GpuReadbackOperation, GpuReadbackStatus, GpuReconstruction,
    GpuResourceLifetime, GpuSubmission, GpuSubmissionStatus, GpuTextureCopyRegion,
    GpuTextureDescriptor, GpuTextureFormat, GpuTextureHandle, GpuTextureInitialization,
    GpuTextureUsage, GpuWorkFragment, GpuWorkResourceIdAllocator,
};
use runen_render::admission::{
    RenderOutputBinding, RenderOutputDestination, RenderRepresentationAvailabilityFact,
    RenderRepresentationAvailabilityState,
};
use runen_render::appearance::{RenderDiffuseMaterial, RenderDirectionalEmitter};
use runen_render::participation::{RenderMaterialAssignment, RenderObjectParticipation};
use runen_render::representation::{
    RENDER_FIELD_DISTANCE_PROTOCOL_REVISION, RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
    RENDER_SURFACE_QUERY_PROTOCOL_REVISION, RenderFieldDistanceGuarantee,
    RenderFieldDistanceProtocolEvidence, RenderOrientedSurfaceProtocolEvidence,
    RenderRefinementEvidence, RenderRepresentationId, RenderRepresentationRecord,
    RenderSurfaceProtocolEvidence,
};
use runen_render::request::{
    RenderObservationSpec, RenderOutputSpec, RenderOutputValue, RenderPerspectiveObservation,
    RenderRadiometricRepresentation, RenderRequest, RenderRequestBuilder, RenderResultTopology,
    RenderSamplingSupport, RenderSemanticTolerance,
};
use runen_render::scene::{
    RenderObjectId, RenderObjectState, RenderSceneSnapshot, RenderSceneStore, RenderSceneUpdate,
};
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
    AdmittedRender, RenderEvaluationSelection, RenderExecutionSession, RenderInvocation,
    RenderTemporalExecutionEvidence, admit_render,
};
use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

type ExampleResult<T> = Result<T, Box<dyn Error>>;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;
const WAVELENGTH_METERS: f64 = 550.0e-9;
const EXPOSURE: f32 = 0.25;
const SPHERE_FRAME_002_X: f64 = 0.75;
const DEADLINE: Duration = Duration::from_secs(30);
const RADIANCE: usize = 0;
const OBJECT_ID: usize = 1;
const COMPARISON_GAP: u32 = 8;
const COMPARISON_LABEL_BAND: u32 = 12;

#[derive(Clone, Copy)]
struct SceneIds {
    sphere: RenderObjectId,
    plane: RenderObjectId,
    sphere_rep: RenderRepresentationId,
    plane_rep: RenderRepresentationId,
}

struct Targets {
    radiance: GpuTextureHandle,
    identity: GpuTextureHandle,
}

#[derive(Debug)]
struct FrameObserved {
    scene_revision: String,
    sphere_translation_x: f64,
    radiance_min: u8,
    radiance_max: u8,
    radiance_distinct: usize,
    sphere_pixels: usize,
    plane_pixels: usize,
    undecoded_pixels: usize,
    grayscale: Vec<u8>,
    identity_rgb: Vec<u8>,
    temporal: RenderTemporalExecutionEvidence,
    adapter_backend: String,
}

#[derive(Debug)]
struct Observed {
    frame_001: FrameObserved,
    frame_002: FrameObserved,
    radiance_changed_pixels: usize,
    identity_changed_pixels: usize,
}

fn object_state_with_translation(x: f64, y: f64, z: f64) -> ExampleResult<RenderObjectState> {
    Ok(RenderObjectState::new(
        RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right)?,
            RenderAffineTransform3::from_row_major_3x4([
                1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z,
            ])?,
            RenderSpatialCoverage::unbounded(),
        ),
        RenderObjectTemporalState::new(RenderTemporalSupport::unbounded()),
    ))
}

fn build_request() -> ExampleResult<RenderRequest> {
    let time = RenderTimeInterval::instant(RenderTimePoint::from_seconds(0.0)?);
    let observation = RenderObservationSpec::Perspective(RenderPerspectiveObservation::new(
        RenderAffineTransform3::identity(),
        std::f64::consts::FRAC_PI_3,
        f64::from(WIDTH) / f64::from(HEIGHT),
        time,
        RenderSamplingSupport::perspective_lattice_cell(),
    )?);
    let lattice = || RenderResultTopology::sample_lattice_2d(WIDTH, HEIGHT);
    let mut builder = RenderRequestBuilder::new(time);
    let observation_handle = builder.add_observation(observation);
    builder.add_output(
        &observation_handle,
        RenderOutputSpec::new(
            RenderOutputValue::Radiance {
                representation: RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                    WAVELENGTH_METERS,
                )?,
            },
            lattice()?,
            RenderSemanticTolerance::absolute(2.0e-4)?,
        )?,
    )?;
    builder.add_output(
        &observation_handle,
        RenderOutputSpec::new(
            RenderOutputValue::ObjectIdentity,
            lattice()?,
            RenderSemanticTolerance::exact(),
        )?,
    )?;
    let request = builder.finish()?;

    if !matches!(
        request.outputs()[RADIANCE].spec().value(),
        RenderOutputValue::Radiance { .. }
    ) || !matches!(
        request.outputs()[OBJECT_ID].spec().value(),
        RenderOutputValue::ObjectIdentity
    ) {
        return Err(io::Error::other("request/output ordering changed").into());
    }
    Ok(request)
}

fn build_surface_inputs(ids: SceneIds) -> ExampleResult<[RenderSurfaceSemanticInputBinding; 2]> {
    Ok([
        RenderSurfaceSemanticInputBinding::new(
            ids.sphere_rep,
            RenderSurfaceSemanticInput::sphere(
                [0.0, 0.0, -3.0],
                1.0,
                RenderTemporalSupport::unbounded(),
            )?,
        )
        .with_generation(RenderSurfaceSemanticInputGeneration::new(1)),
        RenderSurfaceSemanticInputBinding::new(
            ids.plane_rep,
            RenderSurfaceSemanticInput::plane(
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                RenderTemporalSupport::unbounded(),
            )?,
        )
        .with_generation(RenderSurfaceSemanticInputGeneration::new(1)),
    ])
}

fn build_availability(ids: SceneIds) -> [RenderRepresentationAvailabilityFact; 2] {
    [ids.sphere_rep, ids.plane_rep].map(|representation_id| {
        RenderRepresentationAvailabilityFact::new(
            representation_id,
            RenderRepresentationAvailabilityState::Available,
        )
    })
}

fn create_targets() -> ExampleResult<Targets> {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let radiance = allocator.allocate_texture_handle(GpuTextureDescriptor::ordinary_owned_2d(
        "headless retained 550nm radiance",
        GpuResourceLifetime::Retained,
        GpuReconstruction::SourceBacked,
        WIDTH,
        HEIGHT,
        GpuTextureFormat::R32Float,
        [
            GpuTextureUsage::CopyDestination,
            GpuTextureUsage::CopySource,
        ],
        GpuTextureInitialization::Uninitialized,
    )?)?;
    let identity = allocator.allocate_texture_handle(GpuTextureDescriptor::ordinary_owned_2d(
        "headless retained object identity",
        GpuResourceLifetime::Retained,
        GpuReconstruction::SourceBacked,
        WIDTH,
        HEIGHT,
        GpuTextureFormat::R32Uint,
        [
            GpuTextureUsage::CopyDestination,
            GpuTextureUsage::CopySource,
        ],
        GpuTextureInitialization::Uninitialized,
    )?)?;
    Ok(Targets { radiance, identity })
}

fn output_bindings(request: &RenderRequest, targets: &Targets) -> [RenderOutputBinding; 2] {
    [
        RenderOutputBinding::new(
            request.output_handle(RADIANCE).expect("radiance output"),
            RenderOutputDestination::SampleLatticeTexture(targets.radiance.clone()),
        ),
        RenderOutputBinding::new(
            request.output_handle(OBJECT_ID).expect("identity output"),
            RenderOutputDestination::SampleLatticeTexture(targets.identity.clone()),
        ),
    ]
}

fn validate_output_correlation(
    admitted: &AdmittedRender,
    snapshot: &RenderSceneSnapshot,
    targets: &Targets,
) -> ExampleResult<()> {
    if admitted.admitted_plan().scene_revision() != snapshot.revision() {
        return Err(io::Error::other("admitted scene revision changed").into());
    }
    if admitted.admitted_plan().outputs().len() != 2 {
        return Err(io::Error::other("admitted output count changed").into());
    }
    for (index, target) in [
        (RADIANCE, &targets.radiance),
        (OBJECT_ID, &targets.identity),
    ] {
        let output = admitted
            .admitted_plan()
            .outputs()
            .iter()
            .find(|output| output.output().position() == index)
            .ok_or_else(|| io::Error::other("admitted output correlation disappeared"))?;
        if output.observation_index() != 0
            || output.binding().output()
                != &admitted
                    .admitted_plan()
                    .plan()
                    .request()
                    .output_handle(index)
                    .expect("request output")
            || output.binding().destination()
                != &RenderOutputDestination::SampleLatticeTexture(target.clone())
        {
            return Err(io::Error::other("admitted output/destination correlation changed").into());
        }
    }
    Ok(())
}

fn wait_for_submission(context: &GpuContext, submission: &GpuSubmission) -> ExampleResult<()> {
    let deadline = Instant::now() + DEADLINE;
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => return Ok(()),
            GpuSubmissionStatus::Failed(failure) => {
                return Err(
                    io::Error::other(format!("renderer submission failed: {failure:?}")).into(),
                );
            }
            GpuSubmissionStatus::Accepted if Instant::now() < deadline => {
                std::thread::yield_now();
            }
            GpuSubmissionStatus::Accepted => {
                return Err(io::Error::other("renderer submission did not complete").into());
            }
        }
    }
}

fn wait_for_readbacks(
    context: &GpuContext,
    submission: &GpuSubmission,
    ids: &[GpuReadbackId],
) -> ExampleResult<()> {
    let deadline = Instant::now() + DEADLINE;
    loop {
        context.progress();
        if let GpuSubmissionStatus::Failed(failure) = submission.status() {
            return Err(
                io::Error::other(format!("readback submission failed: {failure:?}")).into(),
            );
        }
        let mut ready = true;
        for &id in ids {
            let readback = submission
                .readback(id)
                .ok_or_else(|| io::Error::other("readback correlation disappeared"))?;
            match readback.status() {
                GpuReadbackStatus::Ready(_) => {}
                GpuReadbackStatus::Pending => ready = false,
                GpuReadbackStatus::Failed(failure) => {
                    return Err(io::Error::other(format!("readback failed: {failure:?}")).into());
                }
            }
        }
        if ready && matches!(submission.status(), GpuSubmissionStatus::Completed) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(io::Error::other("readbacks did not complete").into());
        }
        std::thread::yield_now();
    }
}

fn render_frame(
    context: &GpuContext,
    session: &mut RenderExecutionSession,
    snapshot: &RenderSceneSnapshot,
    ids: SceneIds,
    targets: &Targets,
    frame_label: &str,
) -> ExampleResult<FrameObserved> {
    let request = build_request()?;
    let surface_inputs = build_surface_inputs(ids)?;
    let availability = build_availability(ids);
    let bindings = output_bindings(&request, targets);
    let invocation = RenderInvocation::new(
        snapshot.clone(),
        request.clone(),
        surface_inputs.to_vec(),
        Vec::new(),
        availability.to_vec(),
        bindings.to_vec(),
    )?;
    let admitted = admit_render(&invocation, context)?;
    validate_output_correlation(&admitted, snapshot, targets)?;

    let evaluation = RenderEvaluationSelection::new(
        request.output_handle(RADIANCE).expect("radiance handle"),
        WIDTH,
        HEIGHT,
    )
    .ok_or_else(|| io::Error::other("full retained evaluation selection is invalid"))?;
    let occurrence = session.prepare(admitted, context, Some(evaluation))?;
    let temporal = occurrence
        .radiance_output(&request.output_handle(RADIANCE).expect("radiance output"))
        .and_then(|output| output.temporal_execution_evidence())
        .ok_or_else(|| io::Error::other("retained radiance temporal evidence is missing"))?;
    let expected_generation = RenderSurfaceSemanticInputGeneration::new(1);
    if temporal.requested_extent != (WIDTH, HEIGHT)
        || temporal.evaluation_extent != (WIDTH, HEIGHT)
        || temporal.semantic_input_generations.len() != 2
        || ![ids.sphere_rep, ids.plane_rep]
            .iter()
            .all(|representation_id| {
                temporal
                    .semantic_input_generations
                    .iter()
                    .any(|(actual_id, generation)| {
                        actual_id == representation_id && *generation == expected_generation
                    })
            })
        || !temporal.field_semantic_input_generations.is_empty()
    {
        return Err(io::Error::other(
            "retained temporal source-generation/evaluation evidence changed",
        )
        .into());
    }

    let renderer_submission = pollster::block_on(context.submit_work(
        format!("{frame_label} retained renderer work"),
        occurrence.work_set().fragments().iter().cloned(),
    ))?;
    let associated = session.associate_submission(occurrence, &renderer_submission)?;
    wait_for_submission(context, &renderer_submission)?;
    session.reconcile();
    if session.is_in_flight()
        || !matches!(
            associated.submission_status(),
            GpuSubmissionStatus::Completed
        )
    {
        return Err(io::Error::other("retained session did not reconcile completed work").into());
    }

    let capture = associated
        .request_radiance_capture(&request.output_handle(RADIANCE).expect("radiance output"))?;
    let decoder = associated
        .object_identity_decoder(&request.output_handle(OBJECT_ID).expect("identity output"))?;
    let identity_target = associated
        .admitted_plan()
        .outputs()
        .iter()
        .find(|output| output.output().position() == OBJECT_ID)
        .and_then(|output| match output.binding().destination() {
            RenderOutputDestination::SampleLatticeTexture(texture) => Some(texture.clone()),
            RenderOutputDestination::ScalarBuffer(_) => None,
        })
        .ok_or_else(|| io::Error::other("associated identity destination is not a texture"))?;
    let identity_readback = GpuReadbackId::allocate()?;
    let capture_operation =
        GpuReadbackOperation::new(capture.source().clone(), capture.readback_id())?;
    let identity_operation = GpuReadbackOperation::new(
        GpuTextureCopyRegion::whole_base_mip(&identity_target)?.into(),
        identity_readback,
    )?;
    let fragment = GpuWorkFragment::build(format!("{frame_label} diagnostic readback"), |work| {
        work.operation("capture retained radiance", capture_operation)?;
        work.operation("capture retained object IDs", identity_operation)?;
        Ok(())
    })?;
    let readback_submission = pollster::block_on(
        context.submit_work(format!("{frame_label} diagnostic readbacks"), [fragment]),
    )?;
    wait_for_readbacks(
        context,
        &readback_submission,
        &[capture.readback_id(), identity_readback],
    )?;

    let radiance = associated.capture_radiance(capture, context, &readback_submission)?;
    let expected_topology = RenderResultTopology::sample_lattice_2d(WIDTH, HEIGHT)?;
    if radiance.output() != &request.output_handle(RADIANCE).expect("radiance output")
        || radiance.topology() != expected_topology
    {
        return Err(io::Error::other("captured radiance output/topology changed").into());
    }
    let count = usize::try_from(u64::from(WIDTH) * u64::from(HEIGHT))?;
    if radiance.samples().len() != count {
        return Err(io::Error::other("radiance sample count mismatch").into());
    }

    let mut grayscale = Vec::with_capacity(count);
    for &value in radiance.samples() {
        if !value.is_finite() {
            return Err(io::Error::other("nonfinite spectral radiance").into());
        }
        let mapped = (value * EXPOSURE).clamp(0.0, 1.0);
        grayscale.push((mapped * 255.0).round() as u8);
    }

    let identity_bytes = match readback_submission
        .readback(identity_readback)
        .ok_or_else(|| io::Error::other("identity readback missing"))?
        .status()
    {
        GpuReadbackStatus::Ready(bytes) => bytes.as_bytes().to_vec(),
        _ => return Err(io::Error::other("identity readback not ready").into()),
    };
    if identity_bytes.len() != count * 4 {
        return Err(io::Error::other("unexpected normalized R32Uint readback layout").into());
    }

    let mut identity_rgb = Vec::with_capacity(count * 3);
    let mut sphere_pixels = 0;
    let mut plane_pixels = 0;
    let mut undecoded_pixels = 0;
    for word in identity_bytes.as_chunks::<4>().0 {
        let code = u32::from_le_bytes(*word);
        let rgb = match decoder.decode(code) {
            Some(id) if id == ids.sphere => {
                sphere_pixels += 1;
                [245, 105, 72]
            }
            Some(id) if id == ids.plane => {
                plane_pixels += 1;
                [95, 145, 215]
            }
            Some(_) => return Err(io::Error::other("unexpected object identity").into()),
            None => {
                // No decoded identity; the physical payload of an undefined
                // renderer sample carries no semantic object or miss meaning.
                undecoded_pixels += 1;
                [0, 0, 0]
            }
        };
        identity_rgb.extend_from_slice(&rgb);
    }

    let radiance_min = *grayscale
        .iter()
        .min()
        .ok_or_else(|| io::Error::other("no radiance pixels"))?;
    let radiance_max = *grayscale
        .iter()
        .max()
        .ok_or_else(|| io::Error::other("no radiance pixels"))?;
    let radiance_distinct = grayscale.iter().copied().collect::<BTreeSet<_>>().len();

    Ok(FrameObserved {
        scene_revision: format!("{:?}", snapshot.revision()),
        sphere_translation_x: snapshot
            .object_state(ids.sphere)
            .ok_or_else(|| io::Error::other("sphere state disappeared"))?
            .spatial()
            .local_to_scene()
            .row_major_3x4()[3],
        radiance_min,
        radiance_max,
        radiance_distinct,
        sphere_pixels,
        plane_pixels,
        undecoded_pixels,
        grayscale,
        identity_rgb,
        temporal,
        adapter_backend: format!("{:?}", context.adapter_facts().backend()),
    })
}

/// Render two exact occurrences through one retained session.
/// Returns None only when an adapter is unavailable outside GPU-required CI.
fn inspect(output_dir: Option<&Path>) -> ExampleResult<Option<Observed>> {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::CopyDestination)
            .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::CopySource)
            .require_format_role(GpuTextureFormat::R32Uint, GpuFormatRole::CopyDestination)
            .require_format_role(GpuTextureFormat::R32Uint, GpuFormatRole::CopySource)
            .with_label("RunenRender retained headless scene inspector");
    let context = match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => context,
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            if std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref() == Some("1") {
                return Err(io::Error::other("GPU-required test has no RunenGPU adapter").into());
            }
            return Ok(None);
        }
        Err(error) => return Err(error.into()),
    };

    let mut scene = RenderSceneStore::new();
    let sphere = scene.allocate_object_id()?;
    let plane = scene.allocate_object_id()?;
    let sphere_state_001 = object_state_with_translation(0.0, 0.0, 0.0)?;
    let plane_state = object_state_with_translation(0.0, 0.0, 0.0)?;
    let mut insert = RenderSceneUpdate::new();
    insert
        .insert_with_state(sphere, sphere_state_001.clone())
        .insert_with_state(plane, plane_state);
    scene.commit(insert)?;

    let sphere_rep = scene.allocate_representation_id(sphere)?;
    let plane_rep = scene.allocate_representation_id(plane)?;
    let oriented = RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)?
        .with_oriented_surface(RenderOrientedSurfaceProtocolEvidence::exact(
            RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
        )?)
        .with_semantic_input_requirement(RenderSurfaceSemanticInputRequirement::current());
    let sphere_record = RenderRepresentationRecord::new(
        sphere_rep,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
        RenderRefinementEvidence::none(),
        Some(oriented),
        Some(RenderFieldDistanceProtocolEvidence::new(
            RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
            RenderFieldDistanceGuarantee::exact(),
        )?),
    )?;
    let plane_record = RenderRepresentationRecord::new(
        plane_rep,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
        RenderRefinementEvidence::none(),
        Some(oriented),
        None,
    )?;
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(
        sphere,
        RenderObjectParticipation::new(
            vec![sphere_record],
            Some(RenderMaterialAssignment::new(RenderDiffuseMaterial::new(
                0.72,
            )?)),
            None,
        )?,
    );
    attach.replace_participation(
        plane,
        RenderObjectParticipation::new(
            vec![plane_record],
            Some(RenderMaterialAssignment::new(RenderDiffuseMaterial::new(
                0.48,
            )?)),
            Some(RenderDirectionalEmitter::new(
                [0.45, 0.80, 0.35],
                WAVELENGTH_METERS,
                12.0,
            )?),
        )?,
    );
    scene.commit(attach)?;

    let ids = SceneIds {
        sphere,
        plane,
        sphere_rep,
        plane_rep,
    };
    let targets = create_targets()?;
    let mut session = RenderExecutionSession::new();

    let snapshot_001 = scene.snapshot();
    let frame_001 = render_frame(
        &context,
        &mut session,
        &snapshot_001,
        ids,
        &targets,
        "frame 001",
    )?;

    let sphere_state_002 = object_state_with_translation(SPHERE_FRAME_002_X, 0.0, 0.0)?;
    let mut move_sphere = RenderSceneUpdate::new();
    move_sphere.replace_state(sphere, sphere_state_002.clone());
    let move_commit = scene.commit(move_sphere)?;
    if !move_commit
        .change_set()
        .spatial_changed()
        .is_some_and(|changed| changed.len() == 1 && changed[0] == sphere)
    {
        return Err(io::Error::other("sphere move did not publish precise spatial change").into());
    }

    let snapshot_002 = scene.snapshot();
    if snapshot_001.revision() == snapshot_002.revision()
        || snapshot_001.object_state(sphere) != Some(&sphere_state_001)
        || snapshot_002.object_state(sphere) != Some(&sphere_state_002)
    {
        return Err(io::Error::other("immutable scene snapshot retention failed").into());
    }

    let frame_002 = render_frame(
        &context,
        &mut session,
        &snapshot_002,
        ids,
        &targets,
        "frame 002",
    )?;

    if !frame_001.temporal.history_reset
        || frame_001.temporal.history_age != 0
        || !frame_002.temporal.history_reset
        || frame_002.temporal.history_age != 0
        || frame_002.temporal.history_generation <= frame_001.temporal.history_generation
    {
        return Err(io::Error::other(
            "scene revision change did not produce truthful retained-history reset evidence",
        )
        .into());
    }

    let radiance_changed_pixels = frame_001
        .grayscale
        .iter()
        .zip(&frame_002.grayscale)
        .filter(|(first, second)| first != second)
        .count();
    let identity_changed_pixels = frame_001
        .identity_rgb
        .as_chunks::<3>()
        .0
        .iter()
        .zip(frame_002.identity_rgb.as_chunks::<3>().0.iter())
        .filter(|(first, second)| first != second)
        .count();
    if radiance_changed_pixels == 0 || identity_changed_pixels == 0 {
        return Err(io::Error::other("sphere move did not change both diagnostic outputs").into());
    }

    let observed = Observed {
        frame_001,
        frame_002,
        radiance_changed_pixels,
        identity_changed_pixels,
    };
    if let Some(dir) = output_dir {
        write_frame_artifacts(dir, "frame_001", &observed.frame_001)?;
        write_frame_artifacts(dir, "frame_002", &observed.frame_002)?;
        write_comparison(
            &dir.join("comparison.png"),
            &observed.frame_001.grayscale,
            &observed.frame_002.grayscale,
        )?;
    }
    Ok(Some(observed))
}

fn write_frame_artifacts(dir: &Path, name: &str, frame: &FrameObserved) -> ExampleResult<()> {
    let frame_dir = dir.join(name);
    fs::create_dir_all(&frame_dir)?;
    write_png(
        &frame_dir.join("radiance.png"),
        WIDTH,
        HEIGHT,
        1,
        &frame.grayscale,
    )?;
    write_png(
        &frame_dir.join("object_ids.png"),
        WIDTH,
        HEIGHT,
        3,
        &frame.identity_rgb,
    )?;
    let evidence = format!(
        concat!(
            "{{\n",
            "  \"schema_revision\": 2,\n",
            "  \"frame\": \"{}\",\n",
            "  \"scene_revision\": \"{}\",\n",
            "  \"source_snapshot_retained_immutably\": true,\n",
            "  \"same_logical_render_execution_session\": true,\n",
            "  \"sphere_translation_x_meters\": {},\n",
            "  \"observation_count\": 1,\n",
            "  \"output_count\": 2,\n",
            "  \"radiance_output_index\": 0,\n",
            "  \"object_identity_output_index\": 1,\n",
            "  \"output_index_scope\": \"request-local; exact occurrence witness is correlation authority\",\n",
            "  \"topology\": [{}, {}],\n",
            "  \"adapter_backend\": \"{}\",\n",
            "  \"radiance_carrier_format\": \"R32Float\",\n",
            "  \"object_identity_carrier_format\": \"R32Uint\",\n",
            "  \"radiance_wavelength_meters\": {},\n",
            "  \"radiance_visualization\": \"clamp(spectral_radiance * 0.25, 0, 1) -> grayscale PNG\",\n",
            "  \"object_identity_visualization\": \"sphere=coral, plane=blue, undecoded=black; PNG colors are diagnostic, not semantic identity\",\n",
            "  \"object_identity_interpretation\": \"execution-local associated-occurrence decoder; decoded physical words do not prove per-pixel semantic definedness or miss reasons\",\n",
            "  \"renderer_execution\": \"associated_submission_completed\",\n",
            "  \"semantic_result_formed\": false,\n",
            "  \"retained_capture_interpretation\": \"maintained physical radiance observation; not RenderResult certification\",\n",
            "  \"readback_submission\": \"completed\",\n",
            "  \"surface_semantic_input_generations_validated\": true,\n",
            "  \"field_semantic_input_generations\": 0,\n",
            "  \"temporal_evaluation_extent\": [{}, {}],\n",
            "  \"history_generation\": {},\n",
            "  \"history_age\": {},\n",
            "  \"history_reset\": {},\n",
            "  \"camera_reprojection_eligible\": {},\n",
            "  \"previous_observation_available\": {},\n",
            "  \"camera_pose_changed\": {},\n",
            "  \"sphere_pixels\": {},\n",
            "  \"plane_pixels\": {},\n",
            "  \"undecoded_pixels\": {},\n",
            "  \"radiance_distinct_grayscale_values\": {}\n",
            "}}\n"
        ),
        name,
        frame.scene_revision,
        frame.sphere_translation_x,
        WIDTH,
        HEIGHT,
        frame.adapter_backend,
        WAVELENGTH_METERS,
        frame.temporal.evaluation_extent.0,
        frame.temporal.evaluation_extent.1,
        frame.temporal.history_generation,
        frame.temporal.history_age,
        frame.temporal.history_reset,
        frame.temporal.camera_reprojection_eligible,
        frame.temporal.previous_observation_available,
        frame.temporal.camera_pose_changed,
        frame.sphere_pixels,
        frame.plane_pixels,
        frame.undecoded_pixels,
        frame.radiance_distinct,
    );
    fs::write(frame_dir.join("evidence.json"), evidence)?;
    Ok(())
}

fn write_comparison(path: &Path, first: &[u8], second: &[u8]) -> ExampleResult<()> {
    let expected = usize::try_from(u64::from(WIDTH) * u64::from(HEIGHT))?;
    if first.len() != expected || second.len() != expected {
        return Err(io::Error::other("comparison inputs do not match frame dimensions").into());
    }
    let width = WIDTH
        .checked_mul(2)
        .and_then(|value| value.checked_add(COMPARISON_GAP))
        .ok_or_else(|| io::Error::other("comparison width overflow"))?;
    let height = HEIGHT
        .checked_add(COMPARISON_LABEL_BAND)
        .ok_or_else(|| io::Error::other("comparison height overflow"))?;
    let width_usize = usize::try_from(width)?;
    let height_usize = usize::try_from(height)?;
    let mut pixels = vec![0; width_usize * height_usize];

    blit_grayscale(
        &mut pixels,
        width_usize,
        0,
        usize::try_from(COMPARISON_LABEL_BAND)?,
        first,
    )?;
    blit_grayscale(
        &mut pixels,
        width_usize,
        usize::try_from(WIDTH + COMPARISON_GAP)?,
        usize::try_from(COMPARISON_LABEL_BAND)?,
        second,
    )?;
    draw_label(&mut pixels, width_usize, 2, 2, "FRAME 001")?;
    draw_label(
        &mut pixels,
        width_usize,
        usize::try_from(WIDTH + COMPARISON_GAP + 2)?,
        2,
        "FRAME 002",
    )?;
    write_png(path, width, height, 1, &pixels)
}

fn blit_grayscale(
    destination: &mut [u8],
    destination_width: usize,
    x: usize,
    y: usize,
    source: &[u8],
) -> ExampleResult<()> {
    let source_width = usize::try_from(WIDTH)?;
    let source_height = usize::try_from(HEIGHT)?;
    for row in 0..source_height {
        let source_start = row * source_width;
        let destination_start = (y + row) * destination_width + x;
        let destination_end = destination_start + source_width;
        let destination_row = destination
            .get_mut(destination_start..destination_end)
            .ok_or_else(|| io::Error::other("comparison blit exceeded destination"))?;
        destination_row.copy_from_slice(&source[source_start..source_start + source_width]);
    }
    Ok(())
}

fn draw_label(
    pixels: &mut [u8],
    width: usize,
    x: usize,
    y: usize,
    label: &str,
) -> ExampleResult<()> {
    let mut cursor = x;
    for character in label.chars() {
        let rows = glyph_rows(character)
            .ok_or_else(|| io::Error::other("comparison label contains unsupported glyph"))?;
        for (row, bits) in rows.into_iter().enumerate() {
            for column in 0..5 {
                if bits & (1 << (4 - column)) != 0 {
                    let index = (y + row) * width + cursor + column;
                    let pixel = pixels
                        .get_mut(index)
                        .ok_or_else(|| io::Error::other("comparison label exceeded destination"))?;
                    *pixel = u8::MAX;
                }
            }
        }
        cursor += 6;
    }
    Ok(())
}

fn glyph_rows(character: char) -> Option<[u8; 7]> {
    match character {
        ' ' => Some([0, 0, 0, 0, 0, 0, 0]),
        'F' => Some([
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ]),
        'R' => Some([
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ]),
        'A' => Some([
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ]),
        'M' => Some([
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ]),
        'E' => Some([
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ]),
        '0' => Some([
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ]),
        '1' => Some([
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ]),
        '2' => Some([
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ]),
        _ => None,
    }
}

// Small dependency-free PNG writer for this executable example only.
// Uncompressed DEFLATE keeps the implementation transparent and avoids
// extending RunenRender's production dependency graph or artifact authority.
fn write_png(
    path: &Path,
    width: u32,
    height: u32,
    channels: usize,
    pixels: &[u8],
) -> ExampleResult<()> {
    if channels != 1 && channels != 3 {
        return Err(io::Error::other("PNG must be grayscale or RGB").into());
    }
    let stride = usize::try_from(width)? * channels;
    if pixels.len() != stride * usize::try_from(height)? {
        return Err(io::Error::other("PNG dimensions do not match pixels").into());
    }
    let mut scanlines = Vec::with_capacity(pixels.len() + usize::try_from(height)?);
    for row in pixels.chunks_exact(stride) {
        scanlines.push(0);
        scanlines.extend_from_slice(row);
    }
    let mut zlib = vec![0x78, 0x01];
    for (index, block) in scanlines.chunks(65535).enumerate() {
        let last = (index + 1) * 65535 >= scanlines.len();
        zlib.push(u8::from(last));
        let len = u16::try_from(block.len())?;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &scanlines {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    zlib.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut png = vec![137, 80, 78, 71, 13, 10, 26, 10];
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, if channels == 1 { 0 } else { 2 }, 0, 0, 0]);
    push_chunk(&mut png, b"IHDR", &header)?;
    push_chunk(&mut png, b"IDAT", &zlib)?;
    push_chunk(&mut png, b"IEND", &[])?;
    fs::write(path, png)?;
    Ok(())
}

fn push_chunk(png: &mut Vec<u8>, kind: &[u8; 4], bytes: &[u8]) -> ExampleResult<()> {
    png.extend_from_slice(&u32::try_from(bytes.len())?.to_be_bytes());
    png.extend_from_slice(kind);
    png.extend_from_slice(bytes);
    let mut crc = !0u32;
    for &byte in kind.iter().chain(bytes) {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (if crc & 1 != 0 { 0xedb8_8320 } else { 0 });
        }
    }
    png.extend_from_slice(&(!crc).to_be_bytes());
    Ok(())
}

fn main() -> ExampleResult<()> {
    let mut args = std::env::args().skip(1);
    let output = match (args.next().as_deref(), args.next()) {
        (None, None) => None,
        (Some("--output"), Some(path)) if args.next().is_none() => Some(path),
        (Some("--help"), None) => {
            println!("Usage: cargo run --example headless_scene -- [--output DIRECTORY]");
            return Ok(());
        }
        _ => return Err(io::Error::other("usage: --output DIRECTORY").into()),
    };
    match inspect(output.as_deref().map(Path::new))? {
        Some(observed) => {
            println!(
                "Headless retained RunenRender scene inspector: frame 001 radiance {}..{} ({} distinct), frame 002 radiance {}..{} ({} distinct), {} radiance pixels changed, {} identity pixels changed",
                observed.frame_001.radiance_min,
                observed.frame_001.radiance_max,
                observed.frame_001.radiance_distinct,
                observed.frame_002.radiance_min,
                observed.frame_002.radiance_max,
                observed.frame_002.radiance_distinct,
                observed.radiance_changed_pixels,
                observed.identity_changed_pixels,
            );
            if let Some(output) = output {
                println!("Diagnostic PNG/JSON artifacts and comparison: {output}");
            }
            Ok(())
        }
        None => Err(io::Error::other("no supported RunenGPU adapter available").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_png(path: &Path, width: u32, height: u32, color_type: u8) -> ExampleResult<()> {
        let png = fs::read(path)?;
        assert_eq!(&png[..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[16..20], &width.to_be_bytes());
        assert_eq!(&png[20..24], &height.to_be_bytes());
        assert_eq!(png[25], color_type);
        assert!(png.windows(4).any(|value| value == b"IDAT"));
        assert!(png.windows(4).any(|value| value == b"IEND"));
        Ok(())
    }

    #[test]
    fn public_headless_two_frame_retained_scene_conformance() -> ExampleResult<()> {
        let directory_name = format!("runen-render-headless-retained-{}", std::process::id());
        let directory = std::env::temp_dir().join(directory_name);
        let Some(observed) = inspect(Some(&directory))? else {
            eprintln!("No adapter available outside GPU-required CI; skipping");
            return Ok(());
        };

        for frame in [&observed.frame_001, &observed.frame_002] {
            assert!(frame.sphere_pixels > 0, "sphere must have decoded words");
            assert!(frame.plane_pixels > 0, "plane must have decoded words");
            assert_eq!(
                frame.sphere_pixels + frame.plane_pixels + frame.undecoded_pixels,
                usize::try_from(u64::from(WIDTH) * u64::from(HEIGHT))?,
                "every physical word must be classified by this exact decoder",
            );
            assert!(
                frame.radiance_distinct >= 8,
                "radiance must not be nearly uniform"
            );
            assert!(frame.radiance_min < frame.radiance_max);
            assert!(frame.temporal.history_reset);
            assert_eq!(frame.temporal.history_age, 0);
        }
        assert_ne!(
            observed.frame_001.scene_revision,
            observed.frame_002.scene_revision
        );
        assert_eq!(observed.frame_001.sphere_translation_x, 0.0);
        assert_eq!(observed.frame_002.sphere_translation_x, SPHERE_FRAME_002_X);
        assert!(
            observed.frame_002.temporal.history_generation
                > observed.frame_001.temporal.history_generation
        );
        assert!(observed.radiance_changed_pixels > 0);
        assert!(observed.identity_changed_pixels > 0);

        for frame in ["frame_001", "frame_002"] {
            assert_png(
                &directory.join(frame).join("radiance.png"),
                WIDTH,
                HEIGHT,
                0,
            )?;
            assert_png(
                &directory.join(frame).join("object_ids.png"),
                WIDTH,
                HEIGHT,
                2,
            )?;
            let evidence = fs::read_to_string(directory.join(frame).join("evidence.json"))?;
            for required in [
                "\"radiance_output_index\": 0",
                "\"object_identity_output_index\": 1",
                "\"radiance_carrier_format\": \"R32Float\"",
                "\"object_identity_carrier_format\": \"R32Uint\"",
                "\"same_logical_render_execution_session\": true",
                "\"renderer_execution\": \"associated_submission_completed\"",
                "\"semantic_result_formed\": false",
                "\"surface_semantic_input_generations_validated\": true",
                "\"field_semantic_input_generations\": 0",
                "\"history_reset\": true",
                "exact occurrence witness is correlation authority",
                "not RenderResult certification",
                "decoded physical words do not prove per-pixel semantic definedness or miss reasons",
            ] {
                assert!(evidence.contains(required), "missing evidence: {required}");
            }
        }

        assert_png(
            &directory.join("comparison.png"),
            WIDTH * 2 + COMPARISON_GAP,
            HEIGHT + COMPARISON_LABEL_BAND,
            0,
        )?;
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn tiny_png_encoder_emits_valid_chunk_structure() -> ExampleResult<()> {
        let path = std::env::temp_dir().join(format!(
            "runen-render-headless-png-{}.png",
            std::process::id()
        ));
        write_png(&path, 1, 1, 3, &[1, 2, 3])?;
        assert_png(&path, 1, 1, 2)?;
        fs::remove_file(&path)?;
        Ok(())
    }
}
