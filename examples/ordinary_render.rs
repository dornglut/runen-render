//! One bounded ordinary invocation using only public, source-neutral contracts.
//!
//! Run with `cargo run --example ordinary_render --locked`. A GPU adapter is required;
//! adapter, admission, submission, timeout, and readback failures propagate from `main`.
//! The observed radiance and identity are physical output interpretation, not a certified
//! `RenderResult`. See `submit_render_for_result` for the separate one-shot verification path.

use runen_gpu::{
    GpuCapabilityProfile, GpuContext, GpuContextDescriptor, GpuFormatRole, GpuReadbackId,
    GpuReadbackOperation, GpuReadbackStatus, GpuReconstruction, GpuResourceLifetime, GpuSubmission,
    GpuSubmissionStatus, GpuTextureCopyRegion, GpuTextureDescriptor, GpuTextureFormat,
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
    RenderOrientedSurfaceProtocolEvidence, RenderRepresentationRecord,
    RenderSurfaceProtocolEvidence,
};
use runen_render::request::{
    RenderObservationSpec, RenderOutputSpec, RenderOutputValue, RenderPerspectiveObservation,
    RenderRadiometricRepresentation, RenderRequestBuilder, RenderResultTopology,
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
    RenderSurfaceSemanticInputRequirement,
};
use runen_render::{RenderExecutionSession, RenderInvocation, admit_render};
use std::error::Error;
use std::io;
use std::time::{Duration, Instant};

type ExampleResult<T> = Result<T, Box<dyn Error>>;
const WAVELENGTH_METERS: f64 = 550e-9;

fn main() -> ExampleResult<()> {
    // An engine adapter would extract immutable source facts into these same scene contracts.
    // Insertion and participation attachment are separate accepted transactions.
    let mut scene = RenderSceneStore::new();
    let object = scene.allocate_object_id()?;
    let state = RenderObjectState::new(
        RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right)?,
            RenderAffineTransform3::identity(),
            RenderSpatialCoverage::unbounded(),
        ),
        RenderObjectTemporalState::new(RenderTemporalSupport::unbounded()),
    );
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(object, state);
    scene.commit(insert)?;

    let representation = scene.allocate_representation_id(object)?;
    let surface = RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)?
        .with_oriented_surface(RenderOrientedSurfaceProtocolEvidence::exact(
            RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
        )?)
        .with_semantic_input_requirement(RenderSurfaceSemanticInputRequirement::current());
    let record = RenderRepresentationRecord::builder(
        representation,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
    )
    .surface_query(Some(surface))
    .build()?;
    let participation = RenderObjectParticipation::from_representations([record])?
        .with_material_assignment(Some(RenderMaterialAssignment::new(
            RenderDiffuseMaterial::new(0.5)?,
        )))
        .with_emitter(Some(RenderDirectionalEmitter::new(
            [0.0, 0.0, 1.0],
            WAVELENGTH_METERS,
            12.0,
        )?));
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object, participation);
    let snapshot = scene.commit(attach)?.snapshot().clone();

    // Keep the handles minted by this builder. Their positions are not output identity.
    let shutter = RenderTimeInterval::instant(RenderTimePoint::from_seconds(0.0)?);
    let mut request = RenderRequestBuilder::new(shutter);
    let observation = request.add_observation(RenderObservationSpec::Perspective(
        RenderPerspectiveObservation::new(
            RenderAffineTransform3::identity(),
            std::f64::consts::FRAC_PI_3,
            1.0,
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )?,
    ));
    let topology = RenderResultTopology::sample_lattice_2d(1, 1)?;
    let radiance = request.add_output(
        &observation,
        RenderOutputSpec::new(
            RenderOutputValue::Radiance {
                representation: RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                    WAVELENGTH_METERS,
                )?,
            },
            topology,
            RenderSemanticTolerance::exact(),
        )?,
    )?;
    let identity = request.add_output(
        &observation,
        RenderOutputSpec::new(
            RenderOutputValue::ObjectIdentity,
            topology,
            RenderSemanticTolerance::exact(),
        )?,
    )?;
    let request = request.finish()?;

    // Physical destinations become available later; the caller owns RunenGPU and its scheduling.
    let mut descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements());
    for format in [GpuTextureFormat::R32Float, GpuTextureFormat::R32Uint] {
        descriptor = descriptor
            .require_format_role(format, GpuFormatRole::CopyDestination)
            .require_format_role(format, GpuFormatRole::CopySource);
    }
    let context = pollster::block_on(GpuContext::request(descriptor))?;
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let mut target = |label: &str, format| -> ExampleResult<_> {
        Ok(
            allocator.allocate_texture_handle(GpuTextureDescriptor::ordinary_owned_2d(
                label,
                GpuResourceLifetime::Retained,
                GpuReconstruction::SourceBacked,
                1,
                1,
                format,
                [
                    GpuTextureUsage::CopyDestination,
                    GpuTextureUsage::CopySource,
                ],
                GpuTextureInitialization::Uninitialized,
            )?)?,
        )
    };
    let radiance_target = target("ordinary 550nm radiance", GpuTextureFormat::R32Float)?;
    let identity_target = target("ordinary object identity", GpuTextureFormat::R32Uint)?;
    let invocation = RenderInvocation::new(
        snapshot,
        request,
        vec![RenderSurfaceSemanticInputBinding::new(
            representation,
            RenderSurfaceSemanticInput::sphere(
                [0.0, 0.0, -3.0],
                1.0,
                RenderTemporalSupport::unbounded(),
            )?,
        )],
        Vec::new(),
        vec![RenderRepresentationAvailabilityFact::new(
            representation,
            RenderRepresentationAvailabilityState::Available,
        )],
        // Deliberately bind in the opposite order to the semantic request.
        vec![
            RenderOutputBinding::new(
                identity.clone(),
                RenderOutputDestination::SampleLatticeTexture(identity_target.clone()),
            ),
            RenderOutputBinding::new(
                radiance.clone(),
                RenderOutputDestination::SampleLatticeTexture(radiance_target.clone()),
            ),
        ],
    )?;
    let admitted = admit_render(&invocation, &context)?;
    let mut session = RenderExecutionSession::new();
    let occurrence = session.prepare(admitted, &context, None)?;
    let prepared = occurrence
        .radiance_output(&radiance)
        .ok_or_else(|| io::Error::other("radiance output missing from prepared occurrence"))?;
    if prepared.output() != &radiance || prepared.texture() != Some(&radiance_target) {
        return Err(io::Error::other("prepared radiance correlation changed").into());
    }
    let submission = pollster::block_on(context.submit_work(
        "ordinary renderer work",
        occurrence.work_set().fragments().iter().cloned(),
    ))?;
    let associated = session.associate_submission(occurrence, &submission)?;
    wait_for_submission(&context, &submission)?;
    session.reconcile();

    // Optional readback is another caller-owned submission, interpreted through the exact witness.
    let capture = associated.request_radiance_capture(&radiance)?;
    let decoder = associated.object_identity_decoder(&identity)?;
    let identity_readback = GpuReadbackId::allocate()?;
    let radiance_readback = capture.readback_id();
    let radiance_operation =
        GpuReadbackOperation::new(capture.source().clone(), radiance_readback)?;
    let identity_operation = GpuReadbackOperation::new(
        GpuTextureCopyRegion::whole_base_mip(&identity_target)?.into(),
        identity_readback,
    )?;
    let readback = GpuWorkFragment::build("ordinary output readback", |work| {
        work.operation("observe radiance", radiance_operation)?;
        work.operation("observe identity", identity_operation)?;
        Ok(())
    })?;
    let readback_submission =
        pollster::block_on(context.submit_work("ordinary output observation", [readback]))?;
    wait_for_submission(&context, &readback_submission)?;
    let captured = associated.capture_radiance(capture, &context, &readback_submission)?;
    if captured.output() != &radiance
        || captured.topology() != topology
        || captured.samples().len() != 1
        || !captured.samples()[0].is_finite()
        || captured.samples()[0] <= 0.0
    {
        return Err(io::Error::other("unexpected correlated radiance observation").into());
    }
    let bytes = match readback_submission
        .readback(identity_readback)
        .ok_or_else(|| io::Error::other("identity readback missing"))?
        .status()
    {
        GpuReadbackStatus::Ready(bytes) => bytes,
        status => {
            return Err(
                io::Error::other(format!("identity readback not ready: {status:?}")).into(),
            );
        }
    };
    let word: [u8; 4] = bytes.as_bytes().try_into()?;
    let decoded = decoder.decode(u32::from_le_bytes(word));
    if decoded != Some(object) {
        return Err(io::Error::other("identity word did not decode to the scene object").into());
    }
    println!(
        "Completed on {:?}: 550nm radiance {}, decoded object {decoded:?}",
        context.adapter_facts().backend(),
        captured.samples()[0]
    );
    // Physical words and capture do not establish semantic definedness or certify a RenderResult.
    Ok(())
}

fn wait_for_submission(context: &GpuContext, submission: &GpuSubmission) -> ExampleResult<()> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => return Ok(()),
            GpuSubmissionStatus::Accepted if Instant::now() < deadline => std::thread::yield_now(),
            status => {
                return Err(
                    io::Error::other(format!("submission did not complete: {status:?}")).into(),
                );
            }
        }
    }
}
