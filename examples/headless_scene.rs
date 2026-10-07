//! A standalone, headless public-API consumer of RunenRender and RunenGPU.
//!
//! The scene is renderer-owned, not an ECS. Radiance is monochrome 550 nm
//! spectral radiance; output PNGs are diagnostic mappings, not semantic RGB.

use runen_gpu::{
    GpuCapabilityProfile, GpuContext, GpuContextDescriptor, GpuContextRequestErrorCategory,
    GpuFormatRole, GpuReadbackId, GpuReadbackOperation, GpuReadbackStatus, GpuReconstruction,
    GpuResourceLifetime, GpuSubmission, GpuSubmissionStatus, GpuTextureCopyRegion,
    GpuTextureDescriptor, GpuTextureFormat, GpuTextureInitialization, GpuTextureUsage,
    GpuWorkFragment, GpuWorkResourceIdAllocator,
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
    RenderRefinementEvidence, RenderRepresentationRecord, RenderSurfaceProtocolEvidence,
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
use runen_render::{SubmittedRenderForResult, admit_render, submit_render_for_result};
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
const DEADLINE: Duration = Duration::from_secs(30);

#[derive(Debug)]
struct Observed {
    radiance_min: u8,
    radiance_max: u8,
    radiance_distinct: usize,
    sphere_pixels: usize,
    plane_pixels: usize,
    undecoded_pixels: usize,
}

fn object_state() -> ExampleResult<RenderObjectState> {
    Ok(RenderObjectState::new(
        RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right)?,
            RenderAffineTransform3::identity(),
            RenderSpatialCoverage::unbounded(),
        ),
        RenderObjectTemporalState::new(RenderTemporalSupport::unbounded()),
    ))
}

fn wait_for_result(
    context: &GpuContext,
    submitted: &mut SubmittedRenderForResult,
) -> ExampleResult<()> {
    let deadline = Instant::now() + DEADLINE;
    loop {
        context.progress();
        match submitted.try_form_result()? {
            Some(_) => return Ok(()),
            None if Instant::now() < deadline => std::thread::yield_now(),
            None => return Err(io::Error::other("verified render timed out").into()),
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

/// Use one scene snapshot/request to render radiance and object identity together.
/// Returns None only when an adapter is unavailable outside GPU-required CI.
fn inspect(output_dir: Option<&Path>) -> ExampleResult<Option<Observed>> {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .require_format_role(GpuTextureFormat::R32Uint, GpuFormatRole::CopyDestination)
            .require_format_role(GpuTextureFormat::R32Uint, GpuFormatRole::CopySource)
            .with_label("RunenRender headless scene inspector");
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

    // Renderer-local object/representation IDs are allocated explicitly by today's API.
    let mut scene = RenderSceneStore::new();
    let sphere = scene.allocate_object_id()?;
    let plane = scene.allocate_object_id()?;
    let mut insert = RenderSceneUpdate::new();
    insert
        .insert_with_state(sphere, object_state()?)
        .insert_with_state(plane, object_state()?);
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
    let snapshot = scene.snapshot();

    let time = RenderTimeInterval::instant(RenderTimePoint::from_seconds(0.0)?);
    let observation = RenderObservationSpec::Perspective(RenderPerspectiveObservation::new(
        RenderAffineTransform3::identity(),
        std::f64::consts::FRAC_PI_3,
        f64::from(WIDTH) / f64::from(HEIGHT),
        time,
        RenderSamplingSupport::ideal_ray(),
    )?);
    let lattice = || RenderResultTopology::sample_lattice_2d(WIDTH, HEIGHT);
    // Current public contracts use positional output indices. This example keeps the
    // correlation visible rather than obscuring it with a private convenience layer.
    const RADIANCE: usize = 0;
    const OBJECT_ID: usize = 1;
    let request = RenderRequest::new(
        time,
        vec![observation],
        vec![
            RenderRequestedOutput::new(
                0,
                RenderOutputSpec::new(
                    RenderOutputValue::Radiance {
                        representation:
                            RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                                WAVELENGTH_METERS,
                            )?,
                    },
                    lattice()?,
                    RenderSemanticTolerance::absolute(2.0e-4)?,
                )?,
            ),
            RenderRequestedOutput::new(
                0,
                RenderOutputSpec::new(
                    RenderOutputValue::ObjectIdentity,
                    lattice()?,
                    RenderSemanticTolerance::exact(),
                )?,
            ),
        ],
    )?;
    if !matches!(
        request.outputs()[RADIANCE].spec().value(),
        RenderOutputValue::Radiance { .. }
    ) || !matches!(
        request.outputs()[OBJECT_ID].spec().value(),
        RenderOutputValue::ObjectIdentity
    ) {
        return Err(io::Error::other("request/output ordering changed").into());
    }

    let surface_inputs = [
        RenderSurfaceSemanticInputBinding::new(
            sphere_rep,
            RenderSurfaceSemanticInput::sphere(
                [0.0, 0.0, -3.0],
                1.0,
                RenderTemporalSupport::unbounded(),
            )?,
        )
        .with_generation(RenderSurfaceSemanticInputGeneration::new(1)),
        RenderSurfaceSemanticInputBinding::new(
            plane_rep,
            RenderSurfaceSemanticInput::plane(
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                RenderTemporalSupport::unbounded(),
            )?,
        )
        .with_generation(RenderSurfaceSemanticInputGeneration::new(1)),
    ];
    let availability = [sphere_rep, plane_rep].map(|representation_id| {
        RenderRepresentationAvailabilityFact::new(
            representation_id,
            RenderRepresentationAvailabilityState::Available,
        )
    });

    let mut allocator = GpuWorkResourceIdAllocator::new();
    let mut target = |label| -> ExampleResult<_> {
        Ok(
            allocator.allocate_texture_handle(GpuTextureDescriptor::ordinary_owned_2d(
                label,
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
            )?)?,
        )
    };
    let radiance_target = target("headless 550nm radiance")?;
    let object_target = target("headless object identity")?;
    let bindings = [
        RenderOutputBinding::new(
            RADIANCE,
            RenderOutputDestination::SampleLatticeTexture(radiance_target),
        ),
        RenderOutputBinding::new(
            OBJECT_ID,
            RenderOutputDestination::SampleLatticeTexture(object_target.clone()),
        ),
    ];
    let admitted = admit_render(
        &snapshot,
        &request,
        &surface_inputs,
        &[],
        &availability,
        &bindings,
        &context,
    )?;
    let mut submitted = pollster::block_on(submit_render_for_result(admitted, &context))?;
    wait_for_result(&context, &mut submitted)?;
    let capture = submitted.request_radiance_capture(RADIANCE)?;
    let identity_readback = GpuReadbackId::allocate()?;
    let capture_operation =
        GpuReadbackOperation::new(capture.source().clone(), capture.readback_id())?;
    let identity_operation = GpuReadbackOperation::new(
        GpuTextureCopyRegion::whole_base_mip(&object_target)?.into(),
        identity_readback,
    )?;
    let fragment = GpuWorkFragment::build("headless scene readback", |work| {
        work.operation("capture radiance", capture_operation)?;
        work.operation("capture object IDs", identity_operation)?;
        Ok(())
    })?;
    let readback_submission =
        pollster::block_on(context.submit_work("headless scene diagnostic readbacks", [fragment]))?;
    wait_for_readbacks(
        &context,
        &readback_submission,
        &[capture.readback_id(), identity_readback],
    )?;

    let radiance = submitted.capture_radiance(capture, &context, &readback_submission)?;
    let radiance_samples = radiance.samples();
    let count = usize::try_from(u64::from(WIDTH) * u64::from(HEIGHT))?;
    if radiance_samples.len() != count {
        return Err(io::Error::other("radiance sample count mismatch").into());
    }
    let mut grayscale = Vec::with_capacity(count);
    for &value in radiance_samples {
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
    let decoder = submitted.object_identity_decoder();
    let mut identity_rgb = Vec::with_capacity(count * 3);
    let mut sphere_pixels = 0;
    let mut plane_pixels = 0;
    let mut undecoded_pixels = 0;
    for word in identity_bytes.as_chunks::<4>().0 {
        let code = u32::from_le_bytes(*word);
        let rgb = match decoder.decode(code) {
            Some(id) if id == sphere => {
                sphere_pixels += 1;
                [245, 105, 72]
            }
            Some(id) if id == plane => {
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
    let radiance_min = *grayscale.iter().min().ok_or("no pixels")?;
    let radiance_max = *grayscale.iter().max().ok_or("no pixels")?;
    let radiance_distinct = grayscale.iter().copied().collect::<BTreeSet<_>>().len();
    let observed = Observed {
        radiance_min,
        radiance_max,
        radiance_distinct,
        sphere_pixels,
        plane_pixels,
        undecoded_pixels,
    };

    if let Some(dir) = output_dir {
        let frame = dir.join("frame_001");
        fs::create_dir_all(&frame)?;
        write_png(&frame.join("radiance.png"), WIDTH, HEIGHT, 1, &grayscale)?;
        write_png(
            &frame.join("object_ids.png"),
            WIDTH,
            HEIGHT,
            3,
            &identity_rgb,
        )?;
        let evidence = format!(
            concat!(
                "{{\n",
                "  \"schema_revision\": 1,\n",
                "  \"scene_revision\": \"{:?}\",\n",
                "  \"observation_count\": 1,\n",
                "  \"output_count\": 2,\n",
                "  \"radiance_output_index\": 0,\n",
                "  \"object_identity_output_index\": 1,\n",
                "  \"topology\": [{}, {}],\n",
                "  \"radiance_wavelength_meters\": {},\n",
                "  \"radiance_visualization\": \"clamp(spectral_radiance * 0.25, 0, 1) -> grayscale PNG\",\n",
                "  \"object_identity_visualization\": \"sphere=coral, plane=blue, undecoded=black; PNG colors are diagnostic, not semantic identity\",\n",
                "  \"renderer_execution\": \"verified_result_formed\",\n",
                "  \"readback_submission\": \"completed\",\n",
                "  \"sphere_pixels\": {},\n",
                "  \"plane_pixels\": {},\n",
                "  \"undecoded_pixels\": {},\n",
                "  \"radiance_distinct_grayscale_values\": {}\n",
                "}}\n"
            ),
            snapshot.revision(),
            WIDTH,
            HEIGHT,
            WAVELENGTH_METERS,
            sphere_pixels,
            plane_pixels,
            undecoded_pixels,
            radiance_distinct
        );
        fs::write(frame.join("evidence.json"), evidence)?;
    }
    Ok(Some(observed))
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
        scanlines.push(0); // PNG filter None.
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
                "Headless RunenRender scene inspector: radiance {}..{} ({} distinct), sphere {} pixels, plane {} pixels, undecoded {} pixels",
                observed.radiance_min,
                observed.radiance_max,
                observed.radiance_distinct,
                observed.sphere_pixels,
                observed.plane_pixels,
                observed.undecoded_pixels,
            );
            if let Some(output) = output {
                println!("Diagnostic PNG/JSON artifacts: {}/frame_001", output);
            }
            Ok(())
        }
        None => Err(io::Error::other("no supported RunenGPU adapter available").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_headless_multi_output_scene_conformance() -> ExampleResult<()> {
        let Some(observed) = inspect(None)? else {
            eprintln!("No adapter available outside GPU-required CI; skipping");
            return Ok(());
        };
        assert!(observed.sphere_pixels > 0, "sphere must be visible");
        assert!(observed.plane_pixels > 0, "plane must be visible");
        assert!(
            observed.undecoded_pixels > 0,
            "unmapped pixels must be visible"
        );
        assert!(
            observed.radiance_distinct >= 8,
            "radiance must not be nearly uniform"
        );
        assert!(observed.radiance_min < observed.radiance_max);
        Ok(())
    }

    #[test]
    fn tiny_png_encoder_emits_valid_chunk_structure() -> ExampleResult<()> {
        let path = std::env::temp_dir().join(format!(
            "runen-render-headless-png-{}.png",
            std::process::id()
        ));
        write_png(&path, 1, 1, 3, &[1, 2, 3])?;
        let bytes = fs::read(&path)?;
        fs::remove_file(&path)?;
        assert_eq!(&bytes[..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
        assert!(bytes.windows(4).any(|w| w == b"IHDR"));
        assert!(bytes.windows(4).any(|w| w == b"IDAT"));
        assert!(bytes.windows(4).any(|w| w == b"IEND"));
        Ok(())
    }
}
