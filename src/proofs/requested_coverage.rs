//! Actual current-coverage execution proof. Readbacks are authored
//! only here, after the ordinary producer fragment, and correlated to that same completed submission.

use super::prepare::build_object_identity_decoder;
use super::*;
use crate::admission::{RenderOutputBinding, RenderOutputDestination};
use crate::field_input::RenderFieldSemanticInput;
use crate::proofs::execution::{MaintainedExecutionFixture, maintained_fixture};
use crate::request::{
    RenderDistanceConvention, RenderRequest, RenderRequestedOutput, RenderResultTopology,
    RenderSemanticTolerance,
};
use crate::runtime::admission::admit_deterministic_render_with_semantic_inputs;
use crate::runtime::program::abi::{execution_mode, geometry, header};
use crate::space_time::{RenderAffineTransform3, RenderTemporalSupport};
use crate::surface_input::RenderSurfaceSemanticInput;
use runen_gpu::{
    GpuCapabilityProfile, GpuContextDescriptor, GpuContextRequestErrorCategory, GpuFormatRole,
    GpuTextureDescriptor, GpuTextureInitialization, GpuTextureUsage, GpuWorkNodeKind,
};
use std::time::{Duration, Instant};

fn context() -> Option<GpuContext> {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::CopyDestination)
            .with_label("requested current coverage proof");
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => {
            let facts = context.adapter_facts();
            eprintln!(
                "requested coverage adapter: backend={:?} name={:?} class={:?} software={:?} fallback={:?}",
                facts.backend(),
                facts.diagnostic_name(),
                facts.class(),
                facts.software(),
                facts.fallback()
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
        Err(error) => panic!("coverage context: {error}"),
    }
}

fn fixture(
    extent: (u32, u32),
    camera_x: f64,
    sphere_x: f64,
    generation: u64,
) -> MaintainedExecutionFixture {
    let mut fixture = maintained_fixture();
    let RenderObservationSpec::Perspective(old) = fixture.request.observations()[0] else {
        unreachable!()
    };
    let observation = RenderPerspectiveObservation::new(
        RenderAffineTransform3::from_row_major_3x4([
            1.0, 0.0, 0.0, camera_x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ])
        .unwrap(),
        std::f64::consts::FRAC_PI_2,
        f64::from(extent.0) / f64::from(extent.1),
        old.shutter(),
        RenderSamplingSupport::perspective_lattice_cell(),
    )
    .unwrap();
    fixture.request = RenderRequest::new(fixture.request.render_interval(),
        vec![RenderObservationSpec::Perspective(observation)],
        vec![RenderRequestedOutput::new(0, RenderOutputSpec::new(
            RenderOutputValue::Radiance { representation: crate::request::RenderRadiometricRepresentation::spectral_at_wavelength_meters(550e-9).unwrap() },
            RenderResultTopology::sample_lattice_2d(extent.0, extent.1).unwrap(), RenderSemanticTolerance::exact(),
        ).unwrap())],
    ).unwrap();
    let id = fixture.semantic_inputs[0].representation_id();
    fixture.semantic_inputs = vec![
        RenderSurfaceSemanticInputBinding::new(
            id,
            RenderSurfaceSemanticInput::sphere(
                [sphere_x, 0.0, 0.0],
                1.0,
                RenderTemporalSupport::unbounded(),
            )
            .unwrap(),
        )
        .with_generation(RenderSurfaceSemanticInputGeneration::new(generation)),
    ];
    fixture
}

fn admit(
    fixture: &MaintainedExecutionFixture,
    context: &GpuContext,
) -> AdmittedDeterministicRender {
    admit_inputs(fixture, &[], context)
}

fn admit_inputs(
    fixture: &MaintainedExecutionFixture,
    fields: &[RenderFieldSemanticInputBinding],
    context: &GpuContext,
) -> AdmittedDeterministicRender {
    let extent = fixture.request.outputs()[0]
        .spec()
        .topology()
        .sample_lattice_dimensions()
        .unwrap();
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let texture = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::ordinary_owned_2d(
                "coverage proof radiance destination",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                extent.0,
                extent.1,
                GpuTextureFormat::R32Float,
                [GpuTextureUsage::CopyDestination],
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    admit_deterministic_render_with_semantic_inputs(
        &fixture.scene,
        &fixture.request,
        &fixture.semantic_inputs,
        fields,
        &fixture.availability,
        &[RenderOutputBinding::new(
            fixture.request.output_handle(0).expect("output handle"),
            RenderOutputDestination::SampleLatticeTexture(texture),
        )],
        context,
    )
    .unwrap()
}

fn prepare_with_requested_coverage(
    admitted: AdmittedDeterministicRender,
    context: &GpuContext,
    resources: &mut DeterministicResourceCache,
    finite_evaluation: Option<(usize, (u32, u32))>,
) -> Result<PreparedDeterministicRender, RenderDeterministicExecutionError> {
    super::prepare_deterministic_render_with_cache_and_evaluation(
        admitted,
        context,
        resources,
        finite_evaluation,
        true,
    )
}

fn words(context: &GpuContext, submission: &GpuSubmission, id: GpuReadbackId) -> Vec<u32> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        match submission.readback(id).unwrap().status() {
            GpuReadbackStatus::Ready(bytes)
                if matches!(submission.status(), GpuSubmissionStatus::Completed) =>
            {
                return bytes
                    .as_bytes()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| u32::from_ne_bytes(*word))
                    .collect();
            }
            GpuReadbackStatus::Failed(error) => panic!("coverage readback: {error:?}"),
            _ => {}
        }
        assert!(
            Instant::now() < deadline,
            "coverage proof terminal submission timed out"
        );
        std::thread::yield_now();
    }
}

fn observe(
    context: &GpuContext,
    fragments: Vec<GpuWorkFragment>,
    handles: &[GpuBufferHandle],
) -> Vec<Vec<u32>> {
    let readbacks = handles
        .iter()
        .map(|handle| {
            GpuReadbackOperation::ordinary(GpuBufferRegion::whole(handle).unwrap().into()).unwrap()
        })
        .collect::<Vec<_>>();
    let ids = readbacks
        .iter()
        .map(GpuReadbackOperation::id)
        .collect::<Vec<_>>();
    let proof = GpuWorkFragment::build("bounded coverage observations", |work| {
        for fragment in fragments {
            assert!(
                fragment.inputs().is_empty()
                    && fragment.imports().is_empty()
                    && fragment.explicit_orders().is_empty(),
                "this proof requires a self-contained ordinary producer"
            );
            for node in fragment.nodes() {
                work.operation(node.label().as_str(), node.operation().clone())?;
            }
            for output in fragment.outputs() {
                // This GPU readback oracle appends its observations *after*
                // production, changing the observed buffer's final access
                // from storage ReadWrite to copy Read. Its real export was
                // already authored and separately validated by production
                // submissions; do not attach that now-stale final-access
                // declaration to this augmented, readback-only proof graph.
                let observed_buffer_export = match output.relationship().resource() {
                    GpuResourceRef::Buffer(buffer) => handles.contains(buffer),
                    _ => false,
                };
                if !observed_buffer_export {
                    work.add_output(output.clone())?;
                }
            }
        }
        for (index, readback) in readbacks.into_iter().enumerate() {
            work.operation(format!("observe coverage {index}"), readback)?;
        }
        Ok(())
    })
    .unwrap();
    let submission =
        pollster::block_on(context.submit_work("requested coverage proof", [proof])).unwrap();
    ids.into_iter()
        .map(|id| words(context, &submission, id))
        .collect()
}

fn packed(
    fixture: &MaintainedExecutionFixture,
    context: &GpuContext,
    phase: u32,
    kind: MaintainedExecutionKind,
) -> PackedOutput {
    packed_admitted(&admit(fixture, context), context, phase, kind)
}

fn packed_admitted(
    admitted: &AdmittedDeterministicRender,
    context: &GpuContext,
    phase: u32,
    kind: MaintainedExecutionKind,
) -> PackedOutput {
    let decoder = build_object_identity_decoder(admitted.admitted()).unwrap();
    let codes = decoder
        .objects_by_code
        .iter()
        .copied()
        .enumerate()
        .map(|(index, object)| (object, index as u32 + 1))
        .collect();
    let bytes_per_row_alignment = context
        .device_facts()
        .device_limits()
        .alignments()
        .bytes_per_row;
    let mut packed = pack_output(
        admitted.admitted(),
        &admitted.admitted().outputs()[0],
        kind,
        admitted.admitted().plan().request().observations()[0],
        &codes,
        OutputPackingInput {
            finite_evaluation_extent: None,
            bytes_per_row_alignment,
            temporal: None,
        },
    )
    .unwrap();
    // Explicit proof phase, shared by the coverage and P100 controls.
    packed.input_words[24] = phase;
    packed
}

/// Execute the actual maintained shader; sentinel scratch proves coverage never enters ordinary
/// invalidation/sidecar writes, even for undefined primary geometry. This is not a CPU query oracle.
fn evaluate(context: &GpuContext, packed: PackedOutput) -> Vec<Vec<u32>> {
    let coverage = packed.input_words[6] == execution_mode::REQUESTED_COVERAGE;
    let count = packed.sample_count as usize;
    let padded = packed.output_byte_len as usize / 4;
    let initial = if coverage { 0x1234abcd } else { 0 };
    let payloads = [
        packed.input_words,
        vec![0; padded],
        vec![0; count],
        vec![initial; count],
        vec![initial; padded],
        vec![initial; padded * 4],
    ];
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let handles = payloads
        .iter()
        .enumerate()
        .map(|(index, payload)| {
            allocator
                .allocate_buffer_handle(
                    GpuBufferDescriptor::ordinary_owned(
                        format!("coverage oracle {index}"),
                        GpuResourceLifetime::Transient,
                        GpuReconstruction::SourceBacked,
                        payload.len() as u64 * WORD_BYTES,
                        [
                            GpuBufferUsage::Storage,
                            GpuBufferUsage::CopyDestination,
                            GpuBufferUsage::CopySource,
                        ],
                        GpuBufferInitialization::Uninitialized,
                    )
                    .unwrap(),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    let pipeline = GpuComputePipelineDescriptor::ordinary(
        crate::runtime::program::retained_maintained_evaluator_source().unwrap(),
        "main",
    )
    .unwrap();
    let bindings =
        pipeline
            .runtime_bindings(handles.iter().enumerate().map(|(index, handle)| {
                GpuRuntimeBindingValue::whole_buffer(0, index as u32, handle)
            }))
            .unwrap();
    let compute = GpuComputeOperation::new(
        pipeline,
        bindings,
        GpuDispatchIntent::direct(
            super::passes::deterministic_dispatch_size(
                count as u32,
                context
                    .device_facts()
                    .workload_budget()
                    .limits()
                    .max_compute_workgroups_per_dimension(),
            )
            .unwrap(),
        ),
    )
    .unwrap();
    let work = GpuWorkFragment::build("coverage same-query oracle", |work| {
        for (index, (handle, payload)) in handles.iter().zip(&payloads).enumerate() {
            work.operation(
                format!("upload oracle {index}"),
                GpuUploadOperation::whole_buffer(
                    handle,
                    PreparedGpuData::<TransferData>::ordinary_pod_transfer(
                        "coverage oracle",
                        payload,
                    )
                    .unwrap(),
                )
                .unwrap(),
            )?;
        }
        work.compute("execute maintained primary query", compute)?;
        Ok(())
    })
    .unwrap();
    let observed = observe(context, vec![work], &handles[1..]);
    if coverage {
        assert!(
            observed[2..].iter().flatten().all(|word| *word == initial),
            "coverage must not access ordinary scratch bindings"
        );
    }
    observed
}

#[test]
fn ordinary_sub_native_preparation_omits_unconsumed_requested_coverage() {
    let Some(context) = context() else { return };
    let fixture = fixture((8, 8), 0.0, 0.0, 1);
    let mut cache = DeterministicResourceCache::default();
    let prepared = super::prepare_deterministic_render_with_cache_and_evaluation(
        admit(&fixture, &context),
        &context,
        &mut cache,
        Some((0, (4, 4))),
        false,
    )
    .unwrap();
    let evidence = prepared
        .radiance_output(0)
        .unwrap()
        .temporal_execution_evidence()
        .unwrap();
    assert_eq!(evidence.evaluation_extent, (4, 4));
    assert_eq!(evidence.requested_extent, (8, 8));
    assert_eq!(evidence.current_coverage, None);
    for kind in [
        DeterministicBufferKind::CoverageInput,
        DeterministicBufferKind::CoverageDepth,
        DeterministicBufferKind::CoverageState,
        DeterministicBufferKind::CoverageStatusScratch,
        DeterministicBufferKind::CoverageDepthScratch,
        DeterministicBufferKind::CoverageHitScratch,
    ] {
        assert!(
            !cache.buffers.contains_key(&(0, kind)),
            "ordinary runtime must not allocate unconsumed requested coverage {kind:?}"
        );
    }
}

#[test]
fn requested_coverage_executes_all_profiles_phases_and_odd_extents_without_aliasing() {
    let Some(context) = context() else { return };
    for requested in [(8_u32, 8_u32), (7, 5)] {
        for evaluation in [
            requested,
            (requested.0 * 3 / 4, requested.1 * 3 / 4),
            (requested.0 * 2 / 3, requested.1 * 2 / 3),
            (requested.0.div_ceil(2), requested.1.div_ceil(2)),
        ] {
            let fixture = fixture(requested, 0.0, 0.0, 7);
            let mut cache = DeterministicResourceCache::default();
            for phase in 0..4 {
                let prepared = prepare_with_requested_coverage(
                    admit(&fixture, &context),
                    &context,
                    &mut cache,
                    Some((0, evaluation)),
                )
                .unwrap();
                assert!(
                    prepared
                        .work_set()
                        .fragments()
                        .iter()
                        .flat_map(|fragment| fragment.nodes())
                        .all(|node| node.kind() != GpuWorkNodeKind::Readback)
                );
                let evidence = prepared
                    .radiance_output(0)
                    .unwrap()
                    .temporal_execution_evidence()
                    .unwrap();
                assert_eq!(evidence.requested_extent, requested);
                assert_eq!(evidence.evaluation_extent, evaluation);
                assert_eq!(evidence.phase, phase);
                assert_eq!(
                    evidence.semantic_input_generations[0].1,
                    RenderSurfaceSemanticInputGeneration::new(7)
                );
                if evaluation == requested {
                    assert!(evidence.current_coverage.is_none());
                    assert!(evidence.camera_reprojection_eligible);
                    break;
                }
                assert!(!evidence.camera_reprojection_eligible);
                assert_eq!(
                    evidence.current_coverage.as_ref().unwrap().extent,
                    requested
                );
                let carrier = |kind| cache.buffers.get(&(0, kind)).unwrap().clone();
                let handles = [
                    carrier(DeterministicBufferKind::CoverageDepth),
                    carrier(DeterministicBufferKind::CoverageState),
                ];
                let all = cache
                    .buffers
                    .iter()
                    .filter(|((output, _), _)| *output == 0)
                    .map(|(_, h)| h.diagnostic_identity())
                    .collect::<Vec<_>>();
                assert_eq!(
                    all.iter().collect::<BTreeSet<_>>().len(),
                    all.len(),
                    "private coverage must never alias phase-sparse carriers"
                );
                let observed =
                    observe(&context, prepared.work_set().fragments().to_vec(), &handles);
                cache.reconcile_temporal_outputs(true);
                let primary = packed(
                    &fixture,
                    &context,
                    phase,
                    MaintainedExecutionKind::Semantic(fixture.request.outputs()[0].spec().value()),
                );
                let stride = primary.input_words[3] as usize;
                let native = evaluate(&context, primary);
                assert!(
                    observed[0]
                        .iter()
                        .enumerate()
                        .all(|(physical, word)| physical % stride < requested.0 as usize
                            || *word == 0),
                    "padding is cleared, not coverage"
                );
                assert!(
                    observed[1].contains(&1) && observed[1].contains(&2),
                    "silhouette fixture must prove both hits and background"
                );
                for (index, state) in observed[1].iter().enumerate() {
                    assert!(*state <= 2);
                    let physical =
                        index / requested.0 as usize * stride + index % requested.0 as usize;
                    let hit = native[4][physical * 4] != 0;
                    assert_eq!(
                        *state,
                        if native[1][index] == 0 {
                            0
                        } else if hit {
                            2
                        } else {
                            1
                        }
                    );
                    assert_eq!(
                        observed[0][physical], native[3][physical],
                        "requested depth must match exact same-phase P100 primary ray"
                    );
                    if *state == 2 {
                        assert!(f32::from_bits(observed[0][physical]).is_finite());
                    }
                }
            }
        }
    }
}

#[test]
fn requested_coverage_overwrites_camera_and_source_changes_and_keeps_subnative_reset() {
    let Some(context) = context() else { return };
    let mut cache = DeterministicResourceCache::default();
    let mut previous_generation = None;
    let mut prior_depth = None;
    for (camera_x, sphere_x, generation) in
        [(0.0, 0.0, 1), (0.5, 0.0, 1), (0.5, 40.0, 2), (0.0, 0.0, 3)]
    {
        let fixture = fixture((8, 8), camera_x, sphere_x, generation);
        let prepared = prepare_with_requested_coverage(
            admit(&fixture, &context),
            &context,
            &mut cache,
            Some((0, (4, 4))),
        )
        .unwrap();
        let evidence = prepared
            .radiance_output(0)
            .unwrap()
            .temporal_execution_evidence()
            .unwrap();
        assert!(evidence.history_reset && !evidence.camera_reprojection_eligible);
        if let Some(previous) = previous_generation {
            assert_ne!(evidence.history_generation, previous);
        }
        previous_generation = Some(evidence.history_generation);
        let handles = [
            cache.buffers[&(0, DeterministicBufferKind::CoverageDepth)].clone(),
            cache.buffers[&(0, DeterministicBufferKind::CoverageState)].clone(),
        ];
        let observed = observe(&context, prepared.work_set().fragments().to_vec(), &handles);
        cache.reconcile_temporal_outputs(true);
        if sphere_x == 40.0 {
            assert!(observed[1].iter().all(|word| *word == 1));
            assert!(
                observed[0].iter().all(|word| *word == 0),
                "hit-to-miss must overwrite retained depth"
            );
        } else {
            assert!(observed[1].contains(&2));
        }
        if let Some(prior) = prior_depth {
            assert_ne!(observed[0], prior);
        }
        prior_depth = Some(observed[0].clone());
    }
    let mut fixture = fixture((8, 8), 0.0, 0.0, 7);
    fixture.semantic_inputs = fixture
        .semantic_inputs
        .into_iter()
        .map(|binding| {
            RenderSurfaceSemanticInputBinding::new(
                binding.representation_id(),
                binding.into_input(),
            )
        })
        .collect();
    assert!(matches!(
        prepare_with_requested_coverage(
            admit(&fixture, &context),
            &context,
            &mut cache,
            Some((0, (4, 4)))
        ),
        Err(RenderDeterministicExecutionError::Lowering(
            RenderDeterministicLoweringError::MissingTemporalSurfaceInputGeneration { .. }
        ))
    ));
}

#[test]
fn requested_coverage_invalid_primary_is_distinct_from_background_and_skips_lighting() {
    let Some(context) = context() else { return };
    let fixture = fixture((8, 8), 0.0, 0.0, 1);
    let mut invalid = packed(
        &fixture,
        &context,
        0,
        MaintainedExecutionKind::RequestedCoverage,
    );
    invalid.input_words[11..20].fill(0); // Invalid current direction, not a known miss.
    let invalid = evaluate(&context, invalid);
    assert!(invalid[1].iter().all(|state| *state == 0));
    let mut coverage = packed(
        &fixture,
        &context,
        0,
        MaintainedExecutionKind::RequestedCoverage,
    );
    assert_eq!(
        coverage.input_words[5], 0,
        "coverage does not pack emitters"
    );
    assert_eq!(
        coverage.input_words[header::WORDS + 2],
        0,
        "coverage does not require reflectance"
    );
    coverage.input_words[header::WORDS + 2] = f32::NAN.to_bits(); // Lighting-only invalidity is irrelevant to primary visibility.
    let coverage = evaluate(&context, coverage);
    assert!(coverage[1].contains(&1) && coverage[1].contains(&2));
}

fn field_fixture(
    input: RenderFieldSemanticInput,
    generation: Option<u64>,
) -> (
    MaintainedExecutionFixture,
    Vec<RenderFieldSemanticInputBinding>,
) {
    use crate::admission::{
        RenderRepresentationAvailabilityFact, RenderRepresentationAvailabilityState,
    };
    use crate::appearance::{RenderDiffuseMaterial, RenderDirectionalEmitter};
    use crate::field_input::RenderFieldSemanticInputRequirement;
    use crate::participation::{RenderMaterialAssignment, RenderObjectParticipation};
    use crate::representation::{
        RENDER_FIELD_DISTANCE_PROTOCOL_REVISION, RenderFieldDistanceGuarantee,
        RenderFieldDistanceProtocolEvidence, RenderRepresentationRecord,
    };
    use crate::scene::{RenderSceneStore, RenderSceneUpdate};
    use crate::space_time::RenderSpatialCoverage;
    let mut fixture = fixture((8, 8), 0.0, 0.0, 1);
    let state = fixture
        .scene
        .object_state(fixture.scene.object_ids()[0])
        .unwrap()
        .clone();
    let mut store = RenderSceneStore::new();
    let object = store.allocate_object_id().unwrap();
    let mut update = RenderSceneUpdate::new();
    update.insert_with_state(object, state.clone());
    store.commit(update).unwrap();
    let id = store.allocate_representation_id(object).unwrap();
    let representation = RenderRepresentationRecord::builder(
        id,
        RenderSpatialCoverage::axis_aligned_bounds(
            input.origin_local_meters(),
            input.max_local_meters(),
        )
        .unwrap(),
        RenderTemporalSupport::unbounded(),
    )
    .field_distance(Some(
        RenderFieldDistanceProtocolEvidence::new(
            RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
            RenderFieldDistanceGuarantee::conservative(0.2).unwrap(),
        )
        .unwrap()
        .with_semantic_input_requirement(RenderFieldSemanticInputRequirement::current()),
    ))
    .build()
    .unwrap();
    let mut update = RenderSceneUpdate::new();
    update.replace_participation(
        object,
        RenderObjectParticipation::from_representations(vec![representation])
            .unwrap()
            .with_material_assignment(Some(RenderMaterialAssignment::new(
                RenderDiffuseMaterial::new(0.5).unwrap(),
            ))),
    );
    store.commit(update).unwrap();
    let light = store.allocate_object_id().unwrap();
    let mut update = RenderSceneUpdate::new();
    update.insert_with_state(light, state);
    store.commit(update).unwrap();
    let mut update = RenderSceneUpdate::new();
    update.replace_participation(
        light,
        RenderObjectParticipation::from_representations([])
            .unwrap()
            .with_emitter(Some(
                RenderDirectionalEmitter::new([0.0, 0.0, 1.0], 550e-9, std::f64::consts::PI)
                    .unwrap(),
            )),
    );
    store.commit(update).unwrap();
    fixture.scene = store.snapshot();
    fixture.semantic_inputs.clear();
    fixture.availability = vec![RenderRepresentationAvailabilityFact::new(
        id,
        RenderRepresentationAvailabilityState::Available,
    )];
    let mut binding = RenderFieldSemanticInputBinding::new(id, input);
    if let Some(generation) = generation {
        binding = binding.with_generation(RenderFieldSemanticInputGeneration::new(generation));
    }
    (fixture, vec![binding])
}

fn sphere_field() -> RenderFieldSemanticInput {
    let mut values = Vec::new();
    for z in 0..17 {
        for y in 0..17 {
            for x in 0..17 {
                let p = [x, y, z].map(|v| -1.5 + f64::from(v) * 3.0 / 16.0);
                values.push((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - 0.75);
            }
        }
    }
    RenderFieldSemanticInput::dense(
        [-1.5; 3],
        [3.0 / 16.0; 3],
        [17; 3],
        values,
        0.2,
        RenderTemporalSupport::unbounded(),
    )
    .unwrap()
}

fn constant_field(value: f64) -> RenderFieldSemanticInput {
    RenderFieldSemanticInput::dense(
        [-1.5; 3],
        [3.0; 3],
        [2; 3],
        vec![value; 8],
        0.0,
        RenderTemporalSupport::unbounded(),
    )
    .unwrap()
}

#[test]
fn requested_coverage_preserves_field_payload_generations_and_invalid_vs_background() {
    let Some(context) = context() else { return };
    let mut cache = DeterministicResourceCache::default();
    let mut prior_generation = None;
    for (index, input) in [
        sphere_field(),
        constant_field(2.0),
        constant_field(0.0),
        constant_field(0.001),
        sphere_field(),
    ]
    .into_iter()
    .enumerate()
    {
        let source_generation = index as u64 + 11;
        let (fixture, fields) = field_fixture(input, Some(source_generation));
        let admitted = admit_inputs(&fixture, &fields, &context);
        let coverage = packed_admitted(
            &admitted,
            &context,
            0,
            MaintainedExecutionKind::RequestedCoverage,
        );
        let radiance = packed_admitted(
            &admitted,
            &context,
            0,
            MaintainedExecutionKind::Semantic(fixture.request.outputs()[0].spec().value()),
        );
        assert_eq!(coverage.input_words[5], 0);
        assert_eq!(
            radiance.input_words[5], 1,
            "lighting fixture must exercise shifted field payload offsets"
        );
        let field_base = header::WORDS;
        let coverage_samples = coverage.input_words[field_base + 34] as usize;
        let radiance_samples = radiance.input_words[field_base + 34] as usize;
        assert_ne!(coverage_samples, radiance_samples);
        assert_eq!(
            &coverage.input_words[coverage_samples..],
            &radiance.input_words[radiance_samples..],
            "same resolved field payload, despite omitted emitter block"
        );
        // All transform/query policy words are identical; reflectance and sample pointer differ.
        for word in 0..geometry::WORDS {
            if word != 2 && word != 34 {
                assert_eq!(
                    coverage.input_words[field_base + word],
                    radiance.input_words[field_base + word]
                );
            }
        }
        let oracle = evaluate(
            &context,
            packed_admitted(
                &admitted,
                &context,
                0,
                MaintainedExecutionKind::Semantic(RenderOutputValue::Distance {
                    convention: RenderDistanceConvention::ObservationForwardDepth,
                }),
            ),
        );
        let prepared =
            prepare_with_requested_coverage(admitted, &context, &mut cache, Some((0, (4, 4))))
                .unwrap();
        let evidence = prepared
            .radiance_output(0)
            .unwrap()
            .temporal_execution_evidence()
            .unwrap();
        assert_eq!(
            evidence.field_semantic_input_generations[0].1,
            RenderFieldSemanticInputGeneration::new(source_generation)
        );
        assert!(evidence.history_reset);
        if let Some(prior) = prior_generation {
            assert_ne!(prior, evidence.history_generation);
        }
        prior_generation = Some(evidence.history_generation);
        let handles = [
            cache.buffers[&(0, DeterministicBufferKind::CoverageDepth)].clone(),
            cache.buffers[&(0, DeterministicBufferKind::CoverageState)].clone(),
        ];
        let stride = handles[0].descriptor().size_bytes() as usize / 4 / 8;
        let observed = observe(&context, prepared.work_set().fragments().to_vec(), &handles);
        cache.reconcile_temporal_outputs(true);
        for (cell, state) in observed[1].iter().enumerate() {
            let physical = cell / 8 * stride + cell % 8;
            let hit = oracle[1][cell] != 0;
            assert_eq!(
                *state,
                if oracle[2][cell] != 0 {
                    0
                } else if hit {
                    2
                } else {
                    1
                }
            );
            assert_eq!(observed[0][physical], oracle[0][physical]);
        }
        match index {
            0 | 4 => assert!(observed[1].contains(&2), "sampled sphere must prove a hit"),
            1 => assert!(
                observed[1].iter().all(|state| *state == 1),
                "valid miss must be explicit background"
            ),
            2 | 3 => assert!(
                observed[1].contains(&0),
                "invalid field normal / exhausted query must remain Invalid"
            ),
            _ => unreachable!(),
        }
    }
    let (fixture, fields) = field_fixture(sphere_field(), None);
    assert!(matches!(
        prepare_with_requested_coverage(
            admit_inputs(&fixture, &fields, &context),
            &context,
            &mut cache,
            Some((0, (4, 4)))
        ),
        Err(RenderDeterministicExecutionError::Lowering(
            RenderDeterministicLoweringError::MissingTemporalFieldInputGeneration { .. }
        ))
    ));
}

#[test]
fn requested_coverage_reuses_compatible_resources_and_replaces_on_resize() {
    let Some(context) = context() else { return };
    let mut cache = DeterministicResourceCache::default();
    let mut identities = Vec::new();
    for extent in [(8, 8), (8, 8), (7, 5)] {
        let fixture = fixture(extent, 0.0, 0.0, 1);
        let prepared = prepare_with_requested_coverage(
            admit(&fixture, &context),
            &context,
            &mut cache,
            Some((0, (extent.0.div_ceil(2), extent.1.div_ceil(2)))),
        )
        .unwrap();
        let handles = [
            cache.buffers[&(0, DeterministicBufferKind::CoverageDepth)].clone(),
            cache.buffers[&(0, DeterministicBufferKind::CoverageState)].clone(),
        ];
        identities.push(handles[1].diagnostic_identity());
        let observed = observe(&context, prepared.work_set().fragments().to_vec(), &handles);
        assert_eq!(observed[1].len(), (extent.0 * extent.1) as usize);
        assert!(observed[1].contains(&2));
        cache.reconcile_temporal_outputs(true);
    }
    assert_eq!(identities[0], identities[1]);
    assert_ne!(
        identities[1], identities[2],
        "descriptor-changing resize must replace the coverage resource identity"
    );
}
#[test]
fn requested_coverage_plane_depth_matches_the_same_phase_primary_query() {
    let Some(context) = context() else { return };
    let mut fixture = fixture((7, 5), 0.0, 0.0, 1);
    let id = fixture.semantic_inputs[0].representation_id();
    fixture.semantic_inputs = vec![
        RenderSurfaceSemanticInputBinding::new(
            id,
            RenderSurfaceSemanticInput::plane(
                [0.0; 3],
                [0.0, 0.0, 1.0],
                RenderTemporalSupport::unbounded(),
            )
            .unwrap(),
        )
        .with_generation(RenderSurfaceSemanticInputGeneration::new(1)),
    ];
    for phase in 0..4 {
        let coverage = evaluate(
            &context,
            packed(
                &fixture,
                &context,
                phase,
                MaintainedExecutionKind::RequestedCoverage,
            ),
        );
        let depth = evaluate(
            &context,
            packed(
                &fixture,
                &context,
                phase,
                MaintainedExecutionKind::Semantic(RenderOutputValue::Distance {
                    convention: RenderDistanceConvention::ObservationForwardDepth,
                }),
            ),
        );
        assert!(coverage[1].iter().all(|state| *state == 2));
        assert_eq!(coverage[0], depth[0]);
    }
}

#[test]
fn requested_coverage_uses_current_scene_state_even_with_unchanged_surface_generation() {
    use crate::scene::{RenderObjectState, RenderSceneStore, RenderSceneUpdate};
    use crate::space_time::RenderObjectSpatialState;
    let Some(context) = context() else { return };
    let mut fixture = fixture((8, 8), 0.0, 0.0, 7);
    let mut cache = DeterministicResourceCache::default();
    let prepared = prepare_with_requested_coverage(
        admit(&fixture, &context),
        &context,
        &mut cache,
        Some((0, (4, 4))),
    )
    .unwrap();
    let first_generation = prepared
        .radiance_output(0)
        .unwrap()
        .temporal_execution_evidence()
        .unwrap()
        .history_generation;
    let handles = [
        cache.buffers[&(0, DeterministicBufferKind::CoverageDepth)].clone(),
        cache.buffers[&(0, DeterministicBufferKind::CoverageState)].clone(),
    ];
    let first = observe(&context, prepared.work_set().fragments().to_vec(), &handles);
    cache.reconcile_temporal_outputs(true);
    let object = fixture.scene.object_ids()[0];
    let state = fixture.scene.object_state(object).unwrap().clone();
    let participation = fixture.scene.object_participation(object).unwrap().clone();
    let old_revision = fixture.scene.revision();
    let mut store = RenderSceneStore::new();
    assert_eq!(store.allocate_object_id().unwrap(), object);
    let mut update = RenderSceneUpdate::new();
    update.insert_with_state(object, state.clone());
    store.commit(update).unwrap();
    assert_eq!(
        store.allocate_representation_id(object).unwrap(),
        fixture.semantic_inputs[0].representation_id()
    );
    let mut update = RenderSceneUpdate::new();
    update.replace_participation(object, participation);
    store.commit(update).unwrap();
    assert_eq!(store.snapshot().revision(), old_revision);
    let mut transform = state.spatial().local_to_scene().row_major_3x4();
    transform[11] = -4.0;
    let moved = RenderObjectState::new(
        RenderObjectSpatialState::new(
            state.spatial().local_space(),
            RenderAffineTransform3::from_row_major_3x4(transform).unwrap(),
            state.spatial().scene_coverage().clone(),
        ),
        *state.temporal(),
    );
    let mut update = RenderSceneUpdate::new();
    update.replace_state(object, moved);
    store.commit(update).unwrap();
    fixture.scene = store.snapshot();
    assert_ne!(fixture.scene.revision(), old_revision);
    let prepared = prepare_with_requested_coverage(
        admit(&fixture, &context),
        &context,
        &mut cache,
        Some((0, (4, 4))),
    )
    .unwrap();
    let evidence = prepared
        .radiance_output(0)
        .unwrap()
        .temporal_execution_evidence()
        .unwrap();
    assert!(evidence.history_reset);
    assert_ne!(first_generation, evidence.history_generation);
    assert_eq!(
        evidence.semantic_input_generations[0].1,
        RenderSurfaceSemanticInputGeneration::new(7)
    );
    let current = observe(&context, prepared.work_set().fragments().to_vec(), &handles);
    assert_ne!(
        first[0], current[0],
        "scene transform must form fresh requested depth without a surface-generation change"
    );
    assert!(current[1].contains(&2));
}

#[test]
fn phase_aligned_fallback_gpu_matches_same_phase_full_requested_evaluator() {
    use crate::runtime::program::abi::temporal_fallback;
    let Some(context) = context() else { return };
    for requested in [(8_u32, 8_u32), (7, 5), (9, 1), (1, 9)] {
        let profile = |numerator: u32, denominator: u32| {
            (
                ((requested.0 * numerator) / denominator).max(requested.0.div_ceil(2)),
                ((requested.1 * numerator) / denominator).max(requested.1.div_ceil(2)),
            )
        };
        for evaluation in [profile(3, 4), profile(2, 3), profile(1, 2)] {
            if evaluation == requested {
                continue;
            }
            let fixture = fixture(requested, 0.0, 0.0, 9);
            let mut cache = DeterministicResourceCache::default();
            let mut identities = None;
            for phase in 0..4 {
                let native = evaluate(
                    &context,
                    packed(
                        &fixture,
                        &context,
                        phase,
                        MaintainedExecutionKind::Semantic(
                            fixture.request.outputs()[0].spec().value(),
                        ),
                    ),
                );
                let prepared = super::prepare_deterministic_render_with_cache_and_evaluation(
                    admit(&fixture, &context),
                    &context,
                    &mut cache,
                    Some((0, evaluation)),
                    false,
                )
                .expect("admitted sub-native fallback");
                let radiance = prepared.radiance_output(0).unwrap();
                let evidence = radiance.temporal_execution_evidence().unwrap();
                assert_eq!(evidence.phase, phase);
                assert!(!evidence.camera_reprojection_eligible);
                assert!(radiance.availability_export_relationship().is_some());
                assert!(
                    radiance
                        .availability_import(GpuResourceProvenance::new(
                            radiance
                                .texture()
                                .unwrap()
                                .descriptor()
                                .common()
                                .label()
                                .clone(),
                            None,
                            None
                        ))
                        .is_some()
                );
                assert!(
                    prepared
                        .work_set()
                        .fragments()
                        .iter()
                        .flat_map(|fragment| fragment.nodes())
                        .all(|node| node.kind() != GpuWorkNodeKind::Readback),
                    "ordinary fallback may not introduce per-frame CPU readback"
                );
                let get = |kind| cache.buffers[&(0, kind)].clone();
                let resolved = get(DeterministicBufferKind::TemporalProvisional);
                let availability = get(DeterministicBufferKind::TemporalAvailability);
                let visited = get(DeterministicBufferKind::TemporalPhasePresence);
                let (history, counts) = match &cache.temporal_histories[&0].storage {
                    DeterministicTemporalStorage::Static {
                        handle,
                        sample_counts,
                        ..
                    } => (handle.clone(), sample_counts.clone()),
                    _ => panic!("sub-native uses finite static estimator"),
                };
                let current_identifiers = [
                    resolved.diagnostic_identity(),
                    availability.diagnostic_identity(),
                    visited.diagnostic_identity(),
                ];
                assert_eq!(
                    current_identifiers.iter().collect::<BTreeSet<_>>().len(),
                    3,
                    "distinct non-aliased output, presence, availability carriers"
                );
                if let Some(prior) = identities {
                    assert_eq!(prior, current_identifiers, "no per-frame buffer IDs");
                }
                identities = Some(current_identifiers);
                let observations = observe(
                    &context,
                    prepared.work_set().fragments().to_vec(),
                    &[resolved, availability, visited, counts, history],
                );
                let stride = observations[0].len() / requested.1 as usize;
                assert_eq!(observations[1].len(), (requested.0 * requested.1) as usize);
                assert_eq!(observations[2].len(), observations[1].len());
                assert_eq!(
                    observations[2].iter().filter(|state| **state != 0).count(),
                    (evaluation.0 * evaluation.1) as usize,
                    "every primary sample must map to a distinct requested cell"
                );
                for (cell, availability) in observations[1].iter().copied().enumerate() {
                    let x = cell % requested.0 as usize;
                    let y = cell / requested.0 as usize;
                    let physical = y * stride + x;
                    let bits = observations[0][physical];
                    let historical = observations[4][physical];
                    let count = observations[3][physical];
                    assert!(availability <= temporal_fallback::CURRENT_PHASE);
                    match availability {
                        temporal_fallback::UNRESOLVED => {
                            assert_eq!(bits, temporal_fallback::UNRESOLVED_RADIANCE_BITS);
                            assert_eq!(
                                native[1][cell], 0,
                                "same-phase unresolved must be genuinely undefined"
                            );
                        }
                        temporal_fallback::COMPATIBLE_HISTORY => {
                            assert_ne!(count, 0);
                            assert_ne!(count, u32::MAX);
                            assert_eq!(bits, historical);
                            assert!(f32::from_bits(bits).is_finite());
                        }
                        temporal_fallback::CURRENT_PHASE => {
                            assert_eq!(native[1][cell], 1);
                            assert_eq!(
                                bits, native[0][physical],
                                "fallback is the maintained *same phase* direct radiance ray"
                            );
                            assert!(f32::from_bits(bits).is_finite());
                        }
                        _ => unreachable!(),
                    }
                    if phase == 0 && observations[2][cell] == 0 {
                        assert_eq!(
                            count, 0,
                            "provisional current-only reconstruction never populates history"
                        );
                    }
                }
                cache.reconcile_temporal_outputs(true);
            }
        }
    }
}

#[test]
fn invalid_field_current_fallback_is_never_silent_background_black() {
    use crate::runtime::program::abi::temporal_fallback;
    let Some(context) = context() else { return };
    for input in [
        constant_field(2.0),
        constant_field(0.0),
        constant_field(0.001),
    ] {
        let (fixture, fields) = field_fixture(input, Some(55));
        let mut cache = DeterministicResourceCache::default();
        let native = evaluate(
            &context,
            packed_admitted(
                &admit_inputs(&fixture, &fields, &context),
                &context,
                0,
                MaintainedExecutionKind::Semantic(fixture.request.outputs()[0].spec().value()),
            ),
        );
        let prepared = super::prepare_deterministic_render_with_cache_and_evaluation(
            admit_inputs(&fixture, &fields, &context),
            &context,
            &mut cache,
            Some((0, (4, 4))),
            false,
        )
        .unwrap();
        let handles = [
            cache.buffers[&(0, DeterministicBufferKind::TemporalProvisional)].clone(),
            cache.buffers[&(0, DeterministicBufferKind::TemporalAvailability)].clone(),
        ];
        let observations = observe(&context, prepared.work_set().fragments().to_vec(), &handles);
        let stride = observations[0].len() / 8;
        for (cell, availability) in observations[1].iter().copied().enumerate() {
            let physical = cell / 8 * stride + cell % 8;
            if native[1][cell] == 0 {
                assert_eq!(availability, temporal_fallback::UNRESOLVED);
                assert_eq!(
                    observations[0][physical],
                    temporal_fallback::UNRESOLVED_RADIANCE_BITS,
                    "invalid shading/query is not a known-zero miss"
                );
            } else {
                assert_eq!(availability, temporal_fallback::CURRENT_PHASE);
                assert_eq!(observations[0][physical], native[0][physical]);
            }
        }
        cache.reconcile_temporal_outputs(true);
    }
}

#[test]
fn phase_fallback_prepared_but_abandoned_does_not_certify_history() {
    let Some(context) = context() else { return };
    let fixture = fixture((7, 5), 0.0, 0.0, 12);
    let mut cache = DeterministicResourceCache::default();
    let prepared = super::prepare_deterministic_render_with_cache_and_evaluation(
        admit(&fixture, &context),
        &context,
        &mut cache,
        Some((0, (4, 3))),
        false,
    )
    .unwrap();
    let first = prepared
        .radiance_output(0)
        .unwrap()
        .temporal_execution_evidence()
        .unwrap()
        .clone();
    assert_eq!(first.phase, 0);
    assert!(first.history_reset);
    drop(prepared);
    cache.reconcile_temporal_outputs(false);
    let next = super::prepare_deterministic_render_with_cache_and_evaluation(
        admit(&fixture, &context),
        &context,
        &mut cache,
        Some((0, (4, 3))),
        false,
    )
    .unwrap();
    let second = next
        .radiance_output(0)
        .unwrap()
        .temporal_execution_evidence()
        .unwrap();
    assert_eq!(second.phase, 0);
    assert_eq!(second.history_age, 0);
    assert!(second.history_reset);
    assert_ne!(second.history_generation, first.history_generation);
}

fn fixture_with_directional_illumination(
    extent: (u32, u32),
    light_direction: [f64; 3],
) -> MaintainedExecutionFixture {
    use crate::appearance::RenderDirectionalEmitter;
    use crate::participation::RenderObjectParticipation;
    use crate::scene::{RenderSceneStore, RenderSceneUpdate};
    let mut result = fixture(extent, 0.0, 0.0, 32);
    let original = result.scene.object_ids()[0];
    let state = result.scene.object_state(original).unwrap().clone();
    let participation = result.scene.object_participation(original).unwrap().clone();
    let mut store = RenderSceneStore::new();

    let object = store.allocate_object_id().unwrap();
    assert_eq!(object, original);
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(object, state.clone());
    store.commit(insert).unwrap();
    assert_eq!(
        store.allocate_representation_id(object).unwrap(),
        result.semantic_inputs[0].representation_id()
    );
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object, participation);
    store.commit(attach).unwrap();

    let light = store.allocate_object_id().unwrap();
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(light, state);
    store.commit(insert).unwrap();
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(
        light,
        RenderObjectParticipation::from_representations([])
            .unwrap()
            .with_emitter(Some(
                RenderDirectionalEmitter::new(light_direction, 550e-9, std::f64::consts::PI)
                    .unwrap(),
            )),
    );
    store.commit(attach).unwrap();
    result.scene = store.snapshot();
    result
}

#[test]
fn gpu_current_fallback_uses_actual_illumination_not_only_identical_depth() {
    use crate::runtime::program::abi::temporal_fallback;
    let Some(context) = context() else { return };
    let requested = (16_u32, 16_u32);
    let evaluation = (8_u32, 8_u32);
    let fixtures = [
        fixture_with_directional_illumination(requested, [0.0, 0.0, 1.0]),
        fixture_with_directional_illumination(requested, [0.0, 0.0, -1.0]),
    ];
    let mut expected = Vec::new();
    let mut produced = Vec::new();
    for fixture in &fixtures {
        let reference = evaluate(
            &context,
            packed(
                fixture,
                &context,
                0,
                MaintainedExecutionKind::Semantic(fixture.request.outputs()[0].spec().value()),
            ),
        );
        let mut cache = DeterministicResourceCache::default();
        let prepared = super::prepare_deterministic_render_with_cache_and_evaluation(
            admit(fixture, &context),
            &context,
            &mut cache,
            Some((0, evaluation)),
            false,
        )
        .unwrap();
        let handles = [
            cache.buffers[&(0, DeterministicBufferKind::TemporalProvisional)].clone(),
            cache.buffers[&(0, DeterministicBufferKind::TemporalAvailability)].clone(),
            cache.buffers[&(0, DeterministicBufferKind::TemporalPhasePresence)].clone(),
        ];
        let results = observe(&context, prepared.work_set().fragments().to_vec(), &handles);
        assert_eq!(results[1].len(), 256);
        assert_eq!(
            results[2].iter().filter(|state| **state != 0).count(),
            (evaluation.0 * evaluation.1) as usize
        );
        let stride = results[0].len() / requested.1 as usize;
        for cell in 0..256_usize {
            let physical = (cell / requested.0 as usize) * stride + cell % requested.0 as usize;
            if reference[1][cell] == 1 {
                assert_eq!(results[1][cell], temporal_fallback::CURRENT_PHASE);
                assert_eq!(
                    results[0][physical], reference[0][physical],
                    "real WGSL fallback shading must equal the same-phase full evaluator"
                );
            } else {
                assert_eq!(results[1][cell], temporal_fallback::UNRESOLVED);
                assert_eq!(
                    results[0][physical],
                    temporal_fallback::UNRESOLVED_RADIANCE_BITS
                );
            }
        }
        cache.reconcile_temporal_outputs(true);
        expected.push(reference);
        produced.push(results);
    }
    // Illumination is the only changed semantic variable; all primary
    // hit/depth evidence must remain identical, but current radiance differs.
    assert_eq!(expected[0][1], expected[1][1]);
    assert_eq!(expected[0][3], expected[1][3]);
    assert_eq!(expected[0][4], expected[1][4]);
    assert!(
        expected[0][0]
            .iter()
            .zip(&expected[1][0])
            .any(|(lit, dark)| { f32::from_bits(*lit) > 0.0 && f32::from_bits(*dark) == 0.0 }),
        "the GPU scene must prove genuinely illuminated versus unlit geometry"
    );
    assert!(
        produced[0][0]
            .iter()
            .zip(&produced[1][0])
            .any(|(lit, dark)| { f32::from_bits(*lit) > 0.0 && f32::from_bits(*dark) == 0.0 }),
        "actual current-only fallback must reflect illumination differences"
    );
}

#[test]
fn equal_numeric_scene_revisions_do_not_certify_different_illumination_histories() {
    let Some(context) = context() else { return };
    let first_fixture = fixture_with_directional_illumination((8, 8), [0.0, 0.0, 1.0]);
    let second_fixture = fixture_with_directional_illumination((8, 8), [0.0, 0.0, -1.0]);
    assert_eq!(
        first_fixture.scene.revision(),
        second_fixture.scene.revision()
    );
    assert_ne!(first_fixture.scene, second_fixture.scene);
    let mut cache = DeterministicResourceCache::default();
    let first = super::prepare_deterministic_render_with_cache_and_evaluation(
        admit(&first_fixture, &context),
        &context,
        &mut cache,
        Some((0, (4, 4))),
        false,
    )
    .unwrap();
    let previous = first
        .radiance_output(0)
        .unwrap()
        .temporal_execution_evidence()
        .unwrap()
        .clone();
    let availability = cache.buffers[&(0, DeterministicBufferKind::TemporalAvailability)].clone();
    let result = observe(
        &context,
        first.work_set().fragments().to_vec(),
        &[availability],
    );
    assert!(result[0].contains(&2));
    cache.reconcile_temporal_outputs(true);
    let second = super::prepare_deterministic_render_with_cache_and_evaluation(
        admit(&second_fixture, &context),
        &context,
        &mut cache,
        Some((0, (4, 4))),
        false,
    )
    .unwrap();
    let current = second
        .radiance_output(0)
        .unwrap()
        .temporal_execution_evidence()
        .unwrap();
    assert!(
        current.history_reset,
        "distinct emitter facts reject same-revision history"
    );
    assert_eq!(current.phase, 0);
    assert_eq!(current.history_age, 0);
    assert_ne!(current.history_generation, previous.history_generation);
}
