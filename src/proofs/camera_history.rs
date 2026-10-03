//! Private GPU behavior proof. Runs the maintained evaluator and camera shader together,
//! then reads their exact output; it does not substitute a CPU copy of shader decisions.

use super::*;
use crate::runtime::program::{
    abi::{camera, emitter, execution_mode, geometry, header, observation_kind, shape},
    build_maintained_program_sources,
};
use runen_gpu::{
    GpuCapabilityProfile, GpuContextDescriptor, GpuContextRequestErrorCategory, GpuDispatchSize,
};
use std::time::{Duration, Instant};

fn context() -> Option<GpuContext> {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .with_label("RunenRender camera correspondence behavior proof");
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => {
            let facts = context.adapter_facts();
            eprintln!(
                "camera proof adapter: backend={:?} name={:?} class={:?} software={:?} fallback={:?}",
                facts.backend(),
                facts.diagnostic_name(),
                facts.class(),
                facts.software(),
                facts.fallback(),
            );
            Some(context)
        }
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1")
            );
            None
        }
        Err(error) => panic!("camera proof context: {error}"),
    }
}

fn scene_input(sphere: bool) -> Vec<u32> {
    let count = if sphere { 2 } else { 1 };
    let emitter = header::WORDS + count * geometry::WORDS;
    let mut input = vec![0; emitter + emitter::WORDS];
    input[..8].copy_from_slice(&[
        1,
        1,
        1,
        1,
        count as u32,
        1,
        execution_mode::RADIANCE,
        observation_kind::PERSPECTIVE_FOOTPRINT,
    ]);
    pack_matrix3(
        &mut input,
        11,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    input[20] = 1.0_f32.to_bits();
    input[21] = 1.0_f32.to_bits();
    input[22] = 1;
    input[23] = 1;
    input[29] = emitter as u32;
    for index in 0..count {
        let base = header::WORDS + index * geometry::WORDS;
        input[base] = if index == 1 {
            shape::SPHERE
        } else {
            shape::PLANE
        };
        input[base + 1] = index as u32 + 1;
        input[base + 2] = 1.0_f32.to_bits();
        pack_matrix3(
            &mut input,
            base + 4,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        pack_matrix3(
            &mut input,
            base + 16,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        if index == 0 {
            pack_vec3(&mut input, base + 25, [0.0, 0.0, -4.0]).unwrap();
            pack_vec3(&mut input, base + 28, [0.0, 0.0, 1.0]).unwrap();
        } else {
            pack_vec3(&mut input, base + 25, [0.0, 0.0, -3.0]).unwrap();
            input[base + 28] = 2.0_f32.to_bits();
        }
    }
    let light = if sphere {
        [1.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    pack_vec3(&mut input, emitter, light).unwrap();
    input[emitter + 3] = std::f32::consts::PI.to_bits();
    input
}

fn camera_words(previous_available: bool, changed: bool, completed: u32) -> Vec<u32> {
    let mut words = vec![0; 35];
    words[0] = u32::from(previous_available);
    words[1] = u32::from(changed);
    words[23] = completed;
    pack_matrix3(&mut words, 9, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    pack_vec3(&mut words, 18, [0.0, 0.0, -1.0]).unwrap();
    words[21] = 1.0_f32.to_bits();
    words[22] = 1.0_f32.to_bits();
    pack_matrix3(
        &mut words,
        24,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    words
}

struct CameraProofPipelines {
    evaluation: GpuComputePipelineDescriptor,
    reconstruction: GpuComputePipelineDescriptor,
}

fn camera_proof_pipelines() -> CameraProofPipelines {
    let programs = build_maintained_program_sources().expect(
        "camera proof programs must compile through RunenShader and admit through RunenGPU",
    );
    CameraProofPipelines {
        evaluation: GpuComputePipelineDescriptor::ordinary(programs.evaluator().clone(), "main")
            .unwrap(),
        reconstruction: GpuComputePipelineDescriptor::ordinary(
            programs.camera_reprojection().clone(),
            "main",
        )
        .unwrap(),
    }
}

fn execute(
    context: &GpuContext,
    pipelines: &CameraProofPipelines,
    input: Vec<u32>,
    previous: [u32; 8],
    mut camera: Vec<u32>,
) -> [u32; 8] {
    let diagnose_motion = camera[1] != 0;
    camera[33] = u32::from(diagnose_motion);
    let payloads = [
        input,
        vec![0],
        vec![0],
        vec![0],
        vec![0; 4],
        previous.to_vec(),
        vec![0; 8 + camera::history::DIAGNOSTIC_WORDS as usize],
        camera,
        vec![0],
    ];
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let handles = payloads
        .iter()
        .enumerate()
        .map(|(index, words)| {
            allocator
                .allocate_buffer_handle(
                    GpuBufferDescriptor::ordinary_owned(
                        format!("camera proof buffer {index}"),
                        GpuResourceLifetime::Transient,
                        GpuReconstruction::SourceBacked,
                        words.len() as u64 * WORD_BYTES,
                        [
                            GpuBufferUsage::Storage,
                            GpuBufferUsage::CopySource,
                            GpuBufferUsage::CopyDestination,
                        ],
                        GpuBufferInitialization::Uninitialized,
                    )
                    .unwrap(),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    let bindings = pipelines
        .evaluation
        .runtime_bindings([
            GpuRuntimeBindingValue::whole_buffer(0, 0, &handles[0]),
            GpuRuntimeBindingValue::whole_buffer(0, 1, &handles[1]),
            GpuRuntimeBindingValue::whole_buffer(0, 2, &handles[2]),
            GpuRuntimeBindingValue::whole_buffer(0, 3, &handles[8]),
            GpuRuntimeBindingValue::whole_buffer(0, 4, &handles[3]),
            GpuRuntimeBindingValue::whole_buffer(0, 5, &handles[4]),
        ])
        .unwrap();
    let evaluation = GpuComputeOperation::new(
        pipelines.evaluation.clone(),
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap();
    let bindings =
        pipelines
            .reconstruction
            .runtime_bindings(handles[..8].iter().enumerate().map(|(index, handle)| {
                GpuRuntimeBindingValue::whole_buffer(0, index as u32, handle)
            }))
            .unwrap();
    let reconstruction = GpuComputeOperation::new(
        pipelines.reconstruction.clone(),
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap();
    let readback =
        GpuReadbackOperation::ordinary(GpuBufferRegion::whole(&handles[6]).unwrap().into())
            .unwrap();
    let readback_id = readback.id();
    let work = GpuWorkFragment::build("camera history behavior", |work| {
        for (index, (handle, words)) in handles.iter().zip(&payloads).enumerate() {
            let payload =
                PreparedGpuData::<TransferData>::ordinary_pod_transfer("camera proof input", words)
                    .unwrap();
            work.operation(
                format!("upload {index}"),
                GpuUploadOperation::whole_buffer(handle, payload).unwrap(),
            )?;
        }
        work.operation("current-only evaluation", evaluation)?;
        work.operation("history validation", reconstruction)?;
        work.operation("read exact history", readback)?;
        Ok(())
    })
    .unwrap();
    let submission =
        pollster::block_on(context.submit_work("camera history behavior", [work])).unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        context.progress();
        match submission.readback(readback_id).unwrap().status() {
            GpuReadbackStatus::Ready(bytes) => {
                if !matches!(submission.status(), GpuSubmissionStatus::Completed) {
                    assert!(
                        Instant::now() < deadline,
                        "camera proof completion timed out"
                    );
                    std::thread::yield_now();
                    continue;
                }
                let words = bytes
                    .as_bytes()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| u32::from_ne_bytes(*word))
                    .collect::<Vec<_>>();
                if diagnose_motion {
                    let diagnostic = &words[8..40];
                    eprintln!(
                        "camera proof motion: outcome={} retained_anchor={:?} visible_point={:?} retained_radiance={} revalidated_radiance={} result_count={}",
                        diagnostic[8],
                        diagnostic[19..22]
                            .iter()
                            .map(|w| f32::from_bits(*w))
                            .collect::<Vec<_>>(),
                        diagnostic[22..25]
                            .iter()
                            .map(|w| f32::from_bits(*w))
                            .collect::<Vec<_>>(),
                        f32::from_bits(diagnostic[18]),
                        f32::from_bits(diagnostic[25]),
                        words[2],
                    );
                }
                return words[..8].try_into().unwrap();
            }
            GpuReadbackStatus::Failed(failure) => panic!("camera proof readback: {failure:?}"),
            GpuReadbackStatus::Pending => {}
        }
        assert!(Instant::now() < deadline, "camera proof readback timed out");
        std::thread::yield_now();
    }
}

fn axis_aligned_plane_motion_camera() -> Vec<u32> {
    let mut camera = camera_words(true, true, 4);
    // The retained phase-3 point is (0, 0, -4) from origin (-2, 2, 0).
    // Current origin (0, 0, 1/8) makes its validation ray exactly axial. This
    // positive fixture does not depend on backend rounding of a decimal pan.
    pack_vec3(&mut camera, 6, [-2.0, 2.0, 0.0]).unwrap();
    camera
}

#[test]
fn stationary_aggregate_is_distinct_from_latest_phase_and_first_motion_uses_coherent_sample() {
    let Some(context) = context() else {
        return;
    };
    let pipelines = camera_proof_pipelines();
    let mut previous = [0; 8];
    let mut samples = Vec::new();
    for index in 0..8 {
        let mut input = scene_input(true);
        input[24] = index % 4;
        let current = execute(
            &context,
            &pipelines,
            input,
            previous,
            camera_words(index != 0, false, index.min(4)),
        );
        samples.push(f32::from_bits(current[4]));
        if index >= 3 {
            let expected = samples[..4].iter().sum::<f32>() / 4.0;
            assert!((f32::from_bits(current[0]) - expected).abs() < 0.000001);
            assert_eq!(current[2], 4);
        }
        previous = current;
    }
    assert_ne!(
        previous[0], previous[4],
        "smooth-sphere phases must distinguish the settled estimate and latest sample"
    );
    let mut input = scene_input(true);
    input[8] = 0.08_f32.to_bits();
    let current_only = execute(
        &context,
        &pipelines,
        input.clone(),
        previous,
        camera_words(false, false, 0),
    );
    let moving = execute(
        &context,
        &pipelines,
        input,
        previous,
        camera_words(true, true, 4),
    );
    assert_eq!(
        moving[4], current_only[4],
        "the retained coherent lane remains raw current radiance"
    );
    let expected = if moving[2] == 2 {
        (f32::from_bits(previous[4]) + f32::from_bits(current_only[4])) * 0.5
    } else {
        assert_eq!(moving[2], 1);
        f32::from_bits(current_only[4])
    };
    assert_eq!(
        moving[0],
        expected.to_bits(),
        "the aggregate must never feed moving reconstruction"
    );
}

#[test]
fn matched_plane_reuses_one_sample_and_rejects_wrong_radiance_or_outside_current_footprint() {
    let Some(context) = context() else {
        return;
    };
    let pipelines = camera_proof_pipelines();
    let mut input = scene_input(false);
    input[10] = 0.125_f32.to_bits();
    let previous = [
        0.25_f32.to_bits(),
        4.0_f32.to_bits(),
        4,
        1,
        1.0_f32.to_bits(),
        0,
        0,
        (-4.0_f32).to_bits(),
    ];
    let accepted = execute(
        &context,
        &pipelines,
        input.clone(),
        previous,
        axis_aligned_plane_motion_camera(),
    );
    assert_eq!(
        accepted[2], 2,
        "validation must allow useful moving-camera reuse"
    );
    assert_eq!(f32::from_bits(accepted[0]), 1.0);
    let mut wrong_radiance = previous;
    wrong_radiance[4] = 0;
    let rejected = execute(
        &context,
        &pipelines,
        input.clone(),
        wrong_radiance,
        axis_aligned_plane_motion_camera(),
    );
    assert_eq!(
        rejected[2], 1,
        "point/depth agreement must not validate mismatched shading"
    );
    assert_eq!(f32::from_bits(rejected[0]), 1.0);
    let mut outside = previous;
    outside[5] = (-4.25_f32).to_bits();
    let rejected = execute(
        &context,
        &pipelines,
        input,
        outside,
        axis_aligned_plane_motion_camera(),
    );
    assert_eq!(
        rejected[2], 1,
        "previous depth alone must not admit a sample outside current support"
    );
    assert_eq!(f32::from_bits(rejected[0]), 1.0);
}

#[test]
fn hard_shadow_revalidation_distinguishes_depth_and_geometry_from_radiance() {
    let Some(context) = context() else {
        return;
    };
    let pipelines = camera_proof_pipelines();
    let mut input = scene_input(true);
    let blocker = header::WORDS + geometry::WORDS;
    pack_vec3(&mut input, blocker + 25, [2.0, 0.0, -2.0]).unwrap();
    input[blocker + 28] = 0.25_f32.to_bits();
    let emitter = input[29] as usize;
    pack_vec3(&mut input, emitter, [1.0, 0.0, 1.0]).unwrap();
    input[10] = 0.125_f32.to_bits();
    let previous = [
        1.0_f32.to_bits(),
        4.0_f32.to_bits(),
        4,
        1,
        0,
        0,
        0,
        (-4.0_f32).to_bits(),
    ];
    let current_only = execute(
        &context,
        &pipelines,
        input.clone(),
        previous,
        camera_words(false, false, 0),
    );
    assert!(
        f32::from_bits(current_only[4]) > 0.0,
        "current phase is lit"
    );
    let accepted = execute(
        &context,
        &pipelines,
        input.clone(),
        previous,
        axis_aligned_plane_motion_camera(),
    );
    assert_eq!(
        accepted[2], 2,
        "visible retained plane anchor is shadowed but in this footprint"
    );
    assert_eq!(
        f32::from_bits(accepted[0]),
        f32::from_bits(current_only[4]) * 0.5,
        "one lit and one validated shadow sample"
    );
    assert_eq!(
        accepted[4], current_only[4],
        "the new coherent lane must remain raw"
    );
    let mut wrong_shadow = previous;
    wrong_shadow[4] = 1.0_f32.to_bits();
    let rejected = execute(
        &context,
        &pipelines,
        input.clone(),
        wrong_shadow,
        axis_aligned_plane_motion_camera(),
    );
    assert_eq!(
        rejected[2], 1,
        "matching hit, anchor and depth do not certify shadow radiance"
    );
    assert_eq!(rejected[0], current_only[0]);
    let mut control = axis_aligned_plane_motion_camera();
    control[34] = 1;
    let oracle = execute(&context, &pipelines, input, previous, control);
    assert_eq!(oracle[0], current_only[0]);
    assert_eq!(oracle[2], 1);
}

#[test]
fn settled_hit_miss_boundary_and_invalid_inputs_fail_closed_on_first_motion() {
    let Some(context) = context() else {
        return;
    };
    let pipelines = camera_proof_pipelines();
    let mut input = scene_input(true);
    input[4] = 1;
    input[header::WORDS] = shape::SPHERE;
    pack_vec3(&mut input, header::WORDS + 25, [1.5, -1.5, -3.0]).unwrap();
    input[header::WORDS + 28] = 0.5_f32.to_bits();
    let emitter = input[29] as usize;
    pack_vec3(&mut input, emitter, [0.0, 0.0, 1.0]).unwrap();
    let mut previous = [0; 8];
    for index in 0..8 {
        input[24] = index % 4;
        previous = execute(
            &context,
            &pipelines,
            input.clone(),
            previous,
            camera_words(index != 0, false, index.min(4)),
        );
        assert_eq!(
            previous[3] != 0,
            index % 4 == 3,
            "three miss phases and one hit phase"
        );
    }
    assert_eq!(previous[2], 4);
    assert!(f32::from_bits(previous[0]) > 0.0);
    assert_ne!(
        previous[0], previous[4],
        "stationary hit/miss footprint retains all four phases"
    );
    input[24] = 0;
    input[8] = 0.08_f32.to_bits();
    let moving = execute(
        &context,
        &pipelines,
        input.clone(),
        previous,
        camera_words(true, true, 4),
    );
    assert_eq!(moving[3], 0);
    assert_eq!(
        moving[0], 0,
        "a current miss cannot inherit a bright aggregate or anchor"
    );
    assert_eq!(moving[2], 1);
    input[8] = f32::NAN.to_bits();
    let invalid = execute(
        &context,
        &pipelines,
        input,
        previous,
        camera_words(true, true, 4),
    );
    assert_eq!(invalid[0], 0);
    assert_eq!(invalid[2], u32::MAX);
    assert_eq!(invalid[3], 0);
    assert!(f32::from_bits(invalid[4]).is_nan());
}

#[test]
fn invalid_retained_coherent_inputs_cannot_authorize_motion_reuse() {
    let Some(context) = context() else {
        return;
    };
    let pipelines = camera_proof_pipelines();
    let mut input = scene_input(false);
    input[10] = 0.125_f32.to_bits();
    let previous = [
        0.25_f32.to_bits(),
        4.0_f32.to_bits(),
        4,
        1,
        1.0_f32.to_bits(),
        0,
        0,
        (-4.0_f32).to_bits(),
    ];
    assert_eq!(
        execute(
            &context,
            &pipelines,
            input.clone(),
            previous,
            axis_aligned_plane_motion_camera()
        )[2],
        2,
        "the valid control must accept before testing corrupted retained inputs"
    );
    for (slot, value) in [
        (1, f32::NAN.to_bits()),
        (2, 0),
        (2, u32::MAX),
        (3, 0),
        (4, f32::NAN.to_bits()),
        (5, f32::INFINITY.to_bits()),
    ] {
        let mut invalid = previous;
        invalid[slot] = value;
        let result = execute(
            &context,
            &pipelines,
            input.clone(),
            invalid,
            axis_aligned_plane_motion_camera(),
        );
        assert_eq!(result[2], 1, "invalid history slot {slot}");
        assert_eq!(f32::from_bits(result[0]), 1.0);
    }
}
