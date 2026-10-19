//! Test-only connection of admitted Gaussian queries to the maintained F3E
//! mask-fill and final resolve. Only final color is read back; preparation
//! never reads query values back or replaces original-region authority.
use super::*;
use std::time::{Duration, Instant};

/// Data lifetime inventory for this bounded, sequential field/compositor
/// sequence. Device/program realization and compiler/backend bookkeeping
/// retain their existing RunenGPU/RunenShader authorities.
struct FieldStorage {
    map: u64,
    upload: u64,
    scratch: u64,
    resident: u64,
    readback: u64,
    prepared_host: u64,
}

impl FieldStorage {
    fn new(extent: [u32; 2], queries: [u64; 2]) -> Result<Self, Render2dExecutionError> {
        let resource = || failure("Gaussian field data extent exceeds bounded storage");
        if extent.contains(&0) || extent.iter().any(|v| v % SAMPLES != 0) {
            return Err(resource());
        }
        let cells = u64::from(extent[0])
            .checked_mul(u64::from(extent[1]))
            .ok_or_else(resource)?;
        let map = cells.checked_mul(4).ok_or_else(resource)?;
        let vertices = 2 * 6 * VERTEX_STRIDE;
        let upload = queries[0]
            .checked_add(map)
            .and_then(|v| v.checked_add(16 + vertices))
            .ok_or_else(resource)?;
        let scratch = cells.checked_mul(12).ok_or_else(resource)?;
        let output_row = u64::from(extent[0] / SAMPLES) * 4;
        // The ordinary copy/readback carrier pads GPU rows to 256 bytes.
        let readback = output_row
            .checked_add(255)
            .map(|v| v & !255)
            .and_then(|v| v.checked_mul(u64::from(extent[1] / SAMPLES)))
            .ok_or_else(resource)?;
        let output = cells / u64::from(SAMPLES * SAMPLES) * 4;
        let resident = upload
            .checked_add(queries[1])
            .and_then(|v| v.checked_add(scratch))
            .and_then(|v| v.checked_add(output))
            .ok_or_else(resource)?;
        // Caller-owned encoded map/input/radii coexist with immutable POD
        // transfer Arcs and one temporary conversion Vec. Conservatively keep
        // the largest transfer Vec through all final-readback copies as well.
        let prepared_host = map
            .checked_add(queries[0])
            .and_then(|v| v.checked_add(queries[1])) // radii <= query output
            .and_then(|v| v.checked_add(upload))
            .and_then(|v| v.checked_add(map.max(queries[0]).max(vertices).max(16)))
            .and_then(|v| v.checked_add(3 * readback))
            .and_then(|v| v.checked_add(output))
            .ok_or_else(resource)?;
        Ok(Self {
            map,
            upload,
            scratch,
            resident,
            readback,
            prepared_host,
        })
    }

    fn admit(&self, policy: GpuExecutionPolicy) -> Result<(), Render2dExecutionError> {
        if self.prepared_host > shadow::MAX_SHADOW_COVERAGE_BYTES
            || self.upload > shadow::MAX_SHADOW_MASK_UPLOAD_BYTES
            || self.scratch > MAX_PRIVATE_SCRATCH_BYTES
            || self.upload > policy.max_upload_bytes_in_flight()
            || self.readback > policy.max_readback_bytes_in_flight()
            || policy.max_pending_readbacks() < 1
        {
            return Err(failure(
                "Gaussian field data exceeds host, scratch or transfer admission",
            ));
        }
        Ok(())
    }
}

pub(in crate::runtime::execution_2d) fn execute(
    samples: &[u32],
    extent: [u32; 2],
    query_input: &[u8],
    query_count: u32,
) -> Vec<u8> {
    assert_eq!(extent[0], extent[1], "this qualification fixture is square");
    assert_eq!(extent[0] % SAMPLES, 0);
    assert_eq!(samples.len(), extent[0] as usize * extent[1] as usize);
    let storage = FieldStorage::new(
        extent,
        [query_input.len() as u64, u64::from(query_count) * 16],
    )
    .unwrap();
    let policy = GpuExecutionPolicy::default();
    storage.admit(policy).unwrap();
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Compute,
        ))
        .unwrap();
    let mut descriptor = GpuContextDescriptor::new(requirements)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_label("F3F complete Gaussian to maintained F3E compositor proof");
    for (format, role) in [
        (GpuTextureFormat::R32Float, GpuFormatRole::ColorAttachment),
        (GpuTextureFormat::R32Float, GpuFormatRole::Sampled),
        (
            GpuTextureFormat::Rgba16Float,
            GpuFormatRole::ColorAttachment,
        ),
        (GpuTextureFormat::Rgba16Float, GpuFormatRole::Blendable),
        (GpuTextureFormat::Rgba16Float, GpuFormatRole::Sampled),
        (
            GpuTextureFormat::Rgba8UnormSrgb,
            GpuFormatRole::ColorAttachment,
        ),
        (GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Blendable),
        (GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::CopySource),
    ] {
        descriptor = descriptor.require_format_role(format, role);
    }
    let context = pollster::block_on(GpuContext::request_with_policies(
        descriptor,
        GpuRealizationPolicies::default(),
        policy,
    ))
    .expect("required real Vulkan Gaussian/compositor adapter");
    let facts = context.device_facts();
    let native = facts.device_limits().values();
    let limits = facts.workload_budget().limits();
    assert!(
        storage.map
            <= native
                .max_storage_buffer_binding_size()
                .min(limits.max_storage_buffer_binding_size())
    );
    assert!(storage.map <= native.max_buffer_size().min(limits.max_buffer_size()));
    assert!(extent.into_iter().all(|v| {
        v <= native
            .max_texture_dimension_2d()
            .min(limits.max_texture_dimension_2d())
    }));
    let mut resources = GpuResourceScope::new();
    let (mut operations, queries, field_pipeline) =
        super::gaussian::prepare_queries(query_input, query_count, &[0], &context, &mut resources)
            .expect("qualified Gaussian query lowering must succeed");
    operations.try_reserve_exact(4).unwrap();
    assert_eq!(operations.capacity(), 6);
    let mapping = resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                "F3F encoded field samples",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                std::mem::size_of_val(samples) as u64,
                [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                GpuBufferInitialization::Prepared(
                    PreparedGpuData::<TransferData>::ordinary_pod_transfer(
                        "F3F encoded field samples",
                        samples,
                    )
                    .unwrap(),
                ),
            )
            .unwrap(),
        )
        .unwrap();
    let dimensions = [extent[0], extent[1], 0, 0];
    let dimensions = resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                "F3F field dimensions",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                16,
                [GpuBufferUsage::Uniform, GpuBufferUsage::CopyDestination],
                GpuBufferInitialization::Prepared(
                    PreparedGpuData::<TransferData>::ordinary_pod_transfer(
                        "F3F field dimensions",
                        &dimensions,
                    )
                    .unwrap(),
                ),
            )
            .unwrap(),
        )
        .unwrap();
    let mask = scratch(
        &mut resources,
        "F3F Gaussian sample mask",
        extent[0],
        GpuTextureFormat::R32Float,
    )
    .unwrap();
    let field_bindings = field_pipeline
        .runtime_bindings([
            GpuRuntimeBindingValue::whole_buffer(0, 2, &mapping),
            GpuRuntimeBindingValue::whole_buffer(0, 3, &queries),
            GpuRuntimeBindingValue::whole_buffer(0, 4, &dimensions),
        ])
        .unwrap();
    let field_draw = GpuRenderDraw::new(
        field_pipeline,
        field_bindings,
        [],
        None,
        GpuDrawIntent::direct(
            GpuDrawRange::new(0, 3).unwrap(),
            GpuDrawRange::new(0, 1).unwrap(),
        ),
        GpuViewport::new(0.0, 0.0, extent[0] as f32, extent[1] as f32, 0.0, 1.0).unwrap(),
        GpuScissorRect::new(0, 0, extent[0], extent[1]).unwrap(),
        GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        0,
    )
    .unwrap();
    operations.push(operation(&mask, true, vec![field_draw]).unwrap().into());

    // These are the existing production F3E program, geometry draw builder,
    // scalar-mask fill and once-only correlated resolve, with no copied WGSL.
    let colors = scratch(
        &mut resources,
        "F3F maintained premultiplied sample colors",
        extent[0],
        GpuTextureFormat::Rgba16Float,
    )
    .unwrap();
    let fill = pipeline(
        GpuTextureFormat::Rgba16Float,
        "fs_sample_mask_fill",
        Some(0),
    )
    .unwrap();
    let mut vertices = Vec::new();
    vertices
        .try_reserve_exact(6 * VERTEX_STRIDE as usize / size_of::<f32>())
        .unwrap();
    rectangle(
        &mut vertices,
        [0.0, 0.0, f64::from(extent[0]), f64::from(extent[1])],
        extent,
        [1.0, 0.0, 0.0, 1.0],
        [0.0; 2],
    );
    let fill_draw = draw(
        &fill,
        Some((0, &mask)),
        None,
        None,
        &vertices,
        extent,
        &mut resources,
    )
    .unwrap();
    operations.push(operation(&colors, true, vec![fill_draw]).unwrap().into());
    let output_extent = [extent[0] / SAMPLES, extent[1] / SAMPLES];
    let output = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                "F3F final sRGB Gaussian proof",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                output_extent[0],
                output_extent[1],
                GpuTextureFormat::Rgba8UnormSrgb,
                [
                    GpuTextureUsage::ColorAttachment,
                    GpuTextureUsage::CopySource,
                ],
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let view = resources
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned("F3F final color", &output).unwrap(),
        )
        .unwrap();
    let resolve = pipeline(
        GpuTextureFormat::Rgba8UnormSrgb,
        "fs_sample_resolve",
        Some(6),
    )
    .unwrap();
    vertices.clear();
    rectangle(
        &mut vertices,
        [
            0.0,
            0.0,
            f64::from(output_extent[0]),
            f64::from(output_extent[1]),
        ],
        output_extent,
        [1.0; 4],
        [0.0; 2],
    );
    let resolve_draw = draw(
        &resolve,
        Some((6, &colors)),
        None,
        None,
        &vertices,
        output_extent,
        &mut resources,
    )
    .unwrap();
    operations.push(operation(&view, true, vec![resolve_draw]).unwrap().into());
    let readback = GpuReadbackOperation::ordinary(
        GpuTextureCopyRegion::whole_base_mip(&output)
            .unwrap()
            .into(),
    )
    .unwrap();
    let readback_id = readback.id();
    assert_eq!(operations.len(), 5);
    // Exercise the ordinary contribution authoring/evidence boundary with
    // the initialization, compute, field and compositor nodes together.
    // Diagnostic readback remains caller-owned work outside that token.
    let contribution = crate::execution_2d::Render2dPreparedContribution::new(operations, view);
    assert!(contribution.has_render_work());
    let mut token = None;
    let fragment = GpuWorkFragment::build("F3F query-field-compositor sequence", |work| {
        token = contribution.append_to(work)?;
        work.operation("F3F caller final color readback", readback)?;
        Ok(())
    })
    .unwrap();
    let token = token.expect("the complete field has authored work");
    assert_eq!(token.authored_nodes().len(), 5);
    assert_eq!(fragment.nodes().len(), 6);
    let graph = GpuPreparedWorkGraph::prepare(
        GpuResourceLabel::new("F3F exact Gaussian final color").unwrap(),
        [fragment],
    )
    .unwrap();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 1);
    // Source-backed conditional initial content is charged by RunenGPU at
    // ordered submission acceptance, not while merely preparing a graph.
    assert_eq!(stats.upload_bytes_in_flight(), 0);
    let submission = context.submit_prepared(prepared).unwrap();
    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 0);
    assert_eq!(stats.in_flight_submissions(), 1);
    assert_eq!(stats.upload_bytes_in_flight(), storage.upload);
    assert_eq!(stats.readback_bytes_in_flight(), storage.readback);
    let deadline = Instant::now() + Duration::from_secs(45);
    loop {
        context.progress();
        match submission.readback(readback_id).unwrap().status() {
            GpuReadbackStatus::Ready(bytes)
                if submission.status() == GpuSubmissionStatus::Completed =>
            {
                token.completed_by(&submission).unwrap();
                let stats = context.execution_stats();
                assert_eq!(stats.in_flight_submissions(), 0);
                assert_eq!(stats.upload_bytes_in_flight(), 0);
                assert_eq!(stats.readback_bytes_in_flight(), 0);
                println!(
                    "GAUSSIAN_FIELD_DATA resident={} upload={} readback={} conservative_host={} completed_transfer_reservations_released=true",
                    storage.resident, storage.upload, storage.readback, storage.prepared_host
                );
                return bytes.as_bytes().to_vec();
            }
            GpuReadbackStatus::Failed(error) => {
                panic!("Gaussian maintained compositor readback: {error:?}")
            }
            _ => {
                assert!(
                    Instant::now() < deadline,
                    "Gaussian/compositor Vulkan timeout"
                );
                std::thread::yield_now();
            }
        }
    }
}

#[test]
fn gaussian_field_data_admission_rejects_overflow_and_inflight_pressure() {
    assert!(FieldStorage::new([u32::MAX - 3; 2], [143_768, 2_976]).is_err());
    assert!(FieldStorage::new([0, 256], [143_768, 2_976]).is_err());
    let storage = FieldStorage::new([256; 2], [143_768, 2_976]).unwrap();
    let policy = GpuExecutionPolicy::default();
    storage.admit(policy).unwrap();
    assert_eq!(storage.resident, 1_212_104);
    assert_eq!(storage.upload, 406_312);
    assert_eq!(storage.readback, 16_384);
    // An otherwise-capable context with an exhausted transfer category must
    // reject before any resource or caller target is created or mutated.
    for (upload, readback, pending) in [
        (storage.upload - 1, policy.max_readback_bytes_in_flight(), 1),
        (policy.max_upload_bytes_in_flight(), storage.readback - 1, 1),
        (
            policy.max_upload_bytes_in_flight(),
            policy.max_readback_bytes_in_flight(),
            0,
        ),
    ] {
        assert!(
            storage
                .admit(GpuExecutionPolicy::new(
                    policy.max_prepared_submissions(),
                    policy.max_in_flight_submissions(),
                    upload,
                    readback,
                    pending,
                ))
                .is_err()
        );
    }
    assert!(
        FieldStorage::new([4096; 2], [143_768, 2_976])
            .unwrap()
            .admit(policy)
            .is_err()
    );
}
