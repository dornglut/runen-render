//! Test-only connection of admitted Gaussian queries to the maintained F3E
//! mask-fill and final resolve. Only final color is read back; preparation
//! never reads query values back or replaces original-region authority.
use super::gaussian::TILE_UNIFORM_BYTES;
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
    readbacks: u64,
}

impl FieldStorage {
    fn new(extent: [u32; 2], queries: [u64; 2]) -> Result<Self, Render2dExecutionError> {
        Self::for_tiles(extent, extent, queries, 1)
    }

    fn for_tiles(
        map_extent: [u32; 2],
        extent: [u32; 2],
        queries: [u64; 2],
        tiles: u64,
    ) -> Result<Self, Render2dExecutionError> {
        let resource = || failure("Gaussian field data extent exceeds bounded storage");
        if tiles == 0
            || tiles > MAX_OPERATIONS as u64 / 4
            || map_extent.contains(&0)
            || extent.contains(&0)
            || extent.iter().any(|v| v % SAMPLES != 0)
        {
            return Err(resource());
        }
        let cells = u64::from(extent[0])
            .checked_mul(u64::from(extent[1]))
            .ok_or_else(resource)?;
        let map = u64::from(map_extent[0])
            .checked_mul(u64::from(map_extent[1]))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(resource)?;
        let vertices = 2 * 6 * VERTEX_STRIDE;
        let upload = queries[0]
            .checked_add(map)
            .and_then(|v| v.checked_add(tiles.checked_mul(TILE_UNIFORM_BYTES)?))
            .and_then(|v| v.checked_add(vertices))
            .ok_or_else(resource)?;
        let scratch = cells.checked_mul(12).ok_or_else(resource)?;
        let output_row = u64::from(extent[0] / SAMPLES) * 4;
        // The ordinary copy footprint pads between rows to 256 bytes;
        // the final row retains only its logical bytes.
        let readback = output_row
            .checked_add(255)
            .map(|v| v & !255)
            .and_then(|v| v.checked_mul(u64::from(extent[1] / SAMPLES - 1)))
            .and_then(|v| v.checked_add(output_row))
            .and_then(|v| v.checked_mul(tiles))
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
            .and_then(|v| v.checked_add(map.max(queries[0]).max(vertices).max(TILE_UNIFORM_BYTES)))
            .and_then(|v| v.checked_add(readback.checked_mul(3)?))
            .and_then(|v| v.checked_add(output))
            .ok_or_else(resource)?;
        Ok(Self {
            map,
            upload,
            scratch,
            resident,
            readback,
            prepared_host,
            readbacks: tiles,
        })
    }

    fn admit(&self, policy: GpuExecutionPolicy) -> Result<(), Render2dExecutionError> {
        if self.prepared_host > shadow::MAX_SHADOW_COVERAGE_BYTES
            || self.upload > shadow::MAX_SHADOW_MASK_UPLOAD_BYTES
            || self.scratch > MAX_PRIVATE_SCRATCH_BYTES
            || self.upload > policy.max_upload_bytes_in_flight()
            || self.readback > policy.max_readback_bytes_in_flight()
            || (policy.max_pending_readbacks() as u64) < self.readbacks
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
    execute_tiles(
        samples,
        extent,
        query_input,
        query_count,
        extent,
        &[[0, 0]],
        extent,
    )
    .pop()
    .unwrap()
}

fn execute_tiles(
    samples: &[u32],
    map_extent: [u32; 2],
    query_input: &[u8],
    query_count: u32,
    extent: [u32; 2],
    offsets: &[[i32; 2]],
    visible_extent: [u32; 2],
) -> Vec<Vec<u8>> {
    assert_eq!(
        extent[0], extent[1],
        "this compositor proof uses square scratch"
    );
    assert_eq!(extent[0] % SAMPLES, 0);
    assert_eq!(
        samples.len(),
        map_extent[0] as usize * map_extent[1] as usize
    );
    let storage = FieldStorage::for_tiles(
        map_extent,
        extent,
        if query_count == 0 {
            [16, 0]
        } else {
            [query_input.len() as u64, u64::from(query_count) * 16]
        },
        offsets.len() as u64,
    )
    .unwrap();
    let policy = GpuExecutionPolicy::default();
    storage.admit(policy).unwrap();
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    if query_count > 0 {
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Compute,
            ))
            .unwrap();
    }
    let mut descriptor = GpuContextDescriptor::new(requirements)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_label("F3F Gaussian tiled maintained F3E compositor proof");
    if offsets.len() > 1 {
        // The buffer map is wider than this texture ceiling. Only the
        // reusable tile must fit the admitted (not merely native) texture.
        descriptor =
            descriptor.permit_limit(GpuLimitKind::MaxTextureDimension2d, u64::from(extent[0]));
    }
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
    assert_eq!(
        facts.is_enabled(GpuCapabilityFeature::Compute),
        query_count > 0
    );
    let mut resources = GpuResourceScope::new();
    let mut field = super::gaussian::prepare_field(
        super::gaussian::FieldData {
            samples,
            extent: map_extent,
            visible_extent,
            query_input,
            query_count,
        },
        &[0],
        &context,
        &mut resources,
    )
    .unwrap();
    let query_operations = if query_count == 0 { 0 } else { 2 };
    assert_eq!(field.operations.len(), query_operations);
    let tile = field
        .tile(extent, MAX_PRIVATE_SCRATCH_BYTES, &[0], &mut resources)
        .unwrap();
    assert_eq!(
        tile.scratch_bytes,
        u64::from(extent[0]) * u64::from(extent[1]) * 4
    );
    if offsets.len() > 1 {
        // Reject before creating a per-tile uniform, then reuse these SAME
        // admitted field/scratch/resource owners for the valid lookup sequence.
        assert!(
            matches!(field.tile(extent, tile.scratch_bytes - 1, &[2, 1], &mut resources),
            Err(Render2dExecutionError::SampleSpace { kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(owner), .. }) if owner == [2, 1])
        );
        assert!(
            matches!(field.tile([extent[0] + 1, extent[1]], MAX_PRIVATE_SCRATCH_BYTES,
            &[2, 1], &mut resources), Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit, path: Some(owner), ..
            }) if owner == [2, 1])
        );
        assert!(
            matches!(tile.lookup(&field, [i32::MAX, 0], &[2, 1], &mut resources),
            Err(Render2dExecutionError::SampleSpace { kind: Render2dSampleSpaceError::PrecisionLimit,
                path: Some(owner), .. }) if owner == [2, 1])
        );
    }
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
        Some((0, &tile.view)),
        None,
        None,
        &vertices,
        extent,
        &mut resources,
    )
    .unwrap();
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
    // Draw payloads and the map/query buffers upload ONCE. Only each small
    // origin uniform differs. One scratch/color/output allocation is reused by
    // ordered tile lookup -> F3E fill -> resolve -> caller readback operations.
    let mut sequences = Vec::new();
    sequences.try_reserve_exact(offsets.len()).unwrap();
    for &offset in offsets {
        let mut operations = std::mem::take(&mut field.operations);
        operations.try_reserve_exact(3).unwrap();
        operations.push(
            tile.lookup(&field, offset, &[0], &mut resources)
                .unwrap()
                .into(),
        );
        operations.push(
            operation(&colors, true, vec![fill_draw.clone()])
                .unwrap()
                .into(),
        );
        operations.push(
            operation(&view, true, vec![resolve_draw.clone()])
                .unwrap()
                .into(),
        );
        let readback = GpuReadbackOperation::ordinary(
            GpuTextureCopyRegion::whole_base_mip(&output)
                .unwrap()
                .into(),
        )
        .unwrap();
        sequences.push((
            crate::execution_2d::Render2dPreparedContribution::new(operations, view.clone()),
            readback,
        ));
    }
    let mut tokens = Vec::new();
    let mut readbacks = Vec::new();
    let fragment = GpuWorkFragment::build("F3F ordered Gaussian tile consumers", |work| {
        for (contribution, readback) in sequences {
            let token = contribution
                .append_to(work)?
                .expect("tile has three render consumers");
            readbacks.push(readback.id());
            work.operation("F3F caller tile color readback", readback)?;
            tokens.push(token);
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(tokens[0].authored_nodes().len(), query_operations + 3);
    for token in &tokens[1..] {
        assert_eq!(token.authored_nodes().len(), 3);
    }
    assert_eq!(fragment.nodes().len(), query_operations + 4 * offsets.len());
    let graph = GpuPreparedWorkGraph::prepare(
        GpuResourceLabel::new("F3F exact Gaussian tile sequence").unwrap(),
        [fragment],
    )
    .unwrap();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 1);
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
        if submission.status() == GpuSubmissionStatus::Completed {
            for token in tokens.drain(..) {
                token.completed_by(&submission).unwrap();
            }
            let pixels = readbacks
                .iter()
                .map(|&id| match submission.readback(id).unwrap().status() {
                    GpuReadbackStatus::Ready(bytes) => bytes.as_bytes().to_vec(),
                    other => panic!("completed Gaussian tile readback: {other:?}"),
                })
                .collect();
            let stats = context.execution_stats();
            assert_eq!(stats.in_flight_submissions(), 0);
            assert_eq!(stats.upload_bytes_in_flight(), 0);
            assert_eq!(stats.readback_bytes_in_flight(), 0);
            println!(
                "GAUSSIAN_FIELD_DATA resident={} upload={} readback={} conservative_host={} tiles={} completed_transfer_reservations_released=true",
                storage.resident,
                storage.upload,
                storage.readback,
                storage.prepared_host,
                offsets.len()
            );
            return pixels;
        }
        for &id in &readbacks {
            if let GpuReadbackStatus::Failed(error) = submission.readback(id).unwrap().status() {
                panic!("Gaussian tile final color readback failed: {error:?}");
            }
        }
        assert!(
            Instant::now() < deadline,
            "Gaussian tile compositor Vulkan timeout"
        );
        std::thread::yield_now();
    }
}

#[test]
#[ignore = "requires Vulkan: literal-only scalar fields without compute admission"]
fn gaussian_literal_fields_use_only_graphics_and_complete_all_nodes_on_vulkan() {
    // Each pixel resolves sixteen identical values. Independent sRGB reference:
    // round(255*(1.055*(1/2)^(1/2.4)-0.055))=188, alpha round(255/2)=128.
    // Include the exact cutoff endpoints and a mixed field in one submission.
    let mut samples = [0_u32; 8 * 8];
    for y in 0..8 {
        for x in 0..8 {
            samples[y * 8 + x] = match (x / 4, y / 4) {
                (0, 0) => 0,
                (1, 0) => 1 << 23,
                _ => 1 << 24,
            };
        }
    }
    let pixels = execute(&samples, [8; 2], &[], 0);
    assert_eq!(&pixels[..4], &[0, 0, 0, 0]);
    // Use the existing complete-plane two-code sRGB target tolerance. Q24
    // half coverage and its alpha stay exact; color conversion belongs to
    // the same maintained F3E compositor as every numerical field.
    assert!(pixels[4].abs_diff(188) <= 2);
    assert_eq!(&pixels[5..8], &[0, 0, 128]);
    assert_eq!(&pixels[8..], &[255, 0, 0, 255, 255, 0, 0, 255]);
}

#[test]
fn gaussian_field_data_admission_rejects_overflow_and_inflight_pressure() {
    assert_eq!(FieldStorage::new([8; 2], [16, 0]).unwrap().readback, 264);
    assert_eq!(FieldStorage::new([4; 2], [16, 0]).unwrap().readback, 4);
    assert!(FieldStorage::new([u32::MAX - 3; 2], [143_768, 2_976]).is_err());
    assert!(FieldStorage::new([0, 256], [143_768, 2_976]).is_err());
    let storage = FieldStorage::new([256; 2], [143_768, 2_976]).unwrap();
    let policy = GpuExecutionPolicy::default();
    storage.admit(policy).unwrap();
    assert_eq!(storage.resident, 1_212_120);
    assert_eq!(storage.upload, 406_328);
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

#[test]
fn gaussian_tiles_inventory_one_map_and_scratch_with_all_uniforms_and_readbacks() {
    let storage = FieldStorage::for_tiles([16, 8], [8; 2], [16, 0], 5).unwrap();
    assert_eq!(storage.map, 512);
    assert_eq!(storage.scratch, 768);
    assert_eq!(
        storage.upload,
        16 + 512 + 5 * TILE_UNIFORM_BYTES + 2 * 6 * VERTEX_STRIDE
    );
    assert_eq!(storage.readback, 5 * 264);
    let policy = GpuExecutionPolicy::default();
    storage.admit(policy).unwrap();
    assert!(
        storage
            .admit(GpuExecutionPolicy::new(
                policy.max_prepared_submissions(),
                policy.max_in_flight_submissions(),
                policy.max_upload_bytes_in_flight(),
                policy.max_readback_bytes_in_flight(),
                4
            ))
            .is_err()
    );
    assert!(FieldStorage::for_tiles([16, 8], [8; 2], [16, 0], 0).is_err());
}

#[test]
#[ignore = "requires Vulkan: signed tile crop/seams and shared field/query/scratch ownership"]
fn gaussian_tile_offsets_and_seams_share_one_map_through_f3e_on_vulkan() {
    // Each map pixel is sixteen equal samples. The five adjacent/cropped
    // tiles cover negative origins, the internal x seam and both upper edges.
    // Zero-node queries exercise the physical query-result path independently
    // of kernel/quadrature correctness; literal values exercise no-Compute.
    let offsets = [[-1, -1], [7, -1], [-1, 7], [7, 7], [15, 7]];
    for queried in [false, true] {
        let mut samples = [0_u32; 16 * 8];
        for y in 0..8 {
            for x in 0..16 {
                samples[y * 16 + x] = match (x / 4, queried) {
                    (0..=2, true) => (1 << 31) | (x / 4) as u32,
                    (0, false) => 1 << 24,
                    (1, false) => 1 << 23,
                    (2, false) => 1 << 22,
                    _ => 0,
                };
            }
        }
        let mut words = vec![0_u32; 836 + 3 * 5];
        words[0] = 3;
        for (index, value) in [1_u64 << 48, 1 << 47, 1 << 46].into_iter().enumerate() {
            words[836 + 5 * index + 2] = value as u32;
            words[836 + 5 * index + 3] = (value >> 32) as u32;
        }
        let bytes = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        let results = execute_tiles(
            &samples,
            [16, 8],
            if queried { &bytes } else { &[] },
            if queried { 3 } else { 0 },
            [8; 2],
            &offsets,
            [16, 8],
        );
        assert_eq!(results.len(), offsets.len());
        for (pixels, offset) in results.iter().zip(offsets) {
            for py in 0..2 {
                for px in 0..2 {
                    // Independent per-sample palette sum. Origins are NOT
                    // pixel aligned, so each resolved pixel can straddle the
                    // map cutoff or a phase boundary, including tile seams.
                    let mut total = 0.0;
                    for dy in 0..4 {
                        for dx in 0..4 {
                            let x = offset[0] + px * 4 + dx;
                            let y = offset[1] + py * 4 + dy;
                            if (0..16).contains(&x) && (0..8).contains(&y) {
                                total += [1.0_f64, 0.5, 0.25, 0.0][(x / 4) as usize];
                            }
                        }
                    }
                    let alpha = total / 16.0;
                    let red = if alpha <= 0.003_130_8 {
                        alpha * 12.92
                    } else {
                        1.055 * alpha.powf(1.0 / 2.4) - 0.055
                    };
                    let expected = [
                        (red * 255.0).round() as u8,
                        0,
                        0,
                        (alpha * 255.0).round() as u8,
                    ];
                    let actual = &pixels[((py * 2 + px) * 4) as usize..][..4];
                    assert_eq!(
                        &actual[1..],
                        &expected[1..],
                        "crop/alpha at {offset:?},{px},{py}"
                    );
                    assert!(
                        actual[0].abs_diff(expected[0]) <= 2,
                        "independent sRGB at {offset:?},{px},{py}: {actual:?}"
                    );
                    if expected[3] == 0 || expected[3] == 255 {
                        assert_eq!(actual, &expected, "exact cutoff endpoint");
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires Vulkan: fractional canvas cutoff with globally phased cropped field samples"]
fn gaussian_fractional_canvas_preserves_sample_phase_and_signed_crop_on_vulkan() {
    // This map begins at global sample (3,3). Canvas 1.2 admits global centers
    // 3.5 and 4.5, while the next center 5.5 is outside. The negative lookup
    // offset leaves two admitted samples on each axis in the first pixel:
    // alpha = 4/16, then ordinary linear-to-sRGB yields red 137 and alpha 64.
    let visible =
        super::gaussian::canvas_extent([1.2; 2], [3, 3, 11, 11], &mut 0_usize, &[0]).unwrap();
    assert_eq!(visible, [2; 2]);
    let mut words = vec![0_u32; 836 + 5];
    words[0] = 1;
    words[836 + 3] = 65536; // Q48 one, zero-node query.
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    for queried in [false, true] {
        let samples = [if queried { 1_u32 << 31 } else { 1_u32 << 24 }; 64];
        let pixels = execute_tiles(
            &samples,
            [8; 2],
            if queried { &bytes } else { &[] },
            u32::from(queried),
            [8; 2],
            &[[-1, -1]],
            visible,
        )
        .pop()
        .unwrap();
        assert!(pixels[0].abs_diff(137) <= 2);
        assert_eq!(&pixels[1..4], &[0, 0, 64]);
        assert_eq!(
            &pixels[4..],
            &[0; 12],
            "canvas clipping precedes F3E resolve"
        );
    }
}
