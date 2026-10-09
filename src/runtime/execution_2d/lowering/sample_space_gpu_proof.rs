//! GPU-required hardware admission of the exact F3E sample-plane resolve.
//! This is a physical prerequisite, not an assertion that group lowering
//! is already implemented. The reference is independently derived from
//! sample locations and premultiplied source-over.
use super::*;
use runen_gpu::{
    GpuBackendFamily, GpuCapabilityProfile, GpuContextDescriptor, GpuFormatRole,
    GpuReadbackOperation, GpuReadbackStatus, GpuSoftwareFallbackPolicy, GpuSubmissionStatus,
    GpuTextureCopyRegion, GpuWorkFragment,
};
use std::time::{Duration, Instant};

fn pipeline(
    target_format: GpuTextureFormat,
    entry: &str,
    blend: bool,
) -> GpuRenderPipelineDescriptor {
    let source =
        crate::runtime::program::retained_vector_source().expect("maintained vector shader source");
    let vertex = GpuEntryPointName::new("vs_main").unwrap();
    let fragment = GpuEntryPointName::new(entry).unwrap();
    let mut refinements = if matches!(
        entry,
        "fs_sample_resolve" | "fs_sample_merge" | "fs_sample_merge_clipped"
    ) {
        vec![
            GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 6).unwrap())
                .with_texture_sample_class(GpuTextureSampleClass::FloatUnfilterable),
        ]
    } else {
        Vec::new()
    };
    if entry == "fs_sample_merge_clipped" {
        refinements.push(
            GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 4).unwrap())
                .with_texture_sample_class(GpuTextureSampleClass::FloatUnfilterable),
        );
    }
    let program =
        GpuProgramDescriptor::new(source, [vertex.clone(), fragment.clone()], refinements).unwrap();
    let vertex_layout = GpuVertexBufferLayoutDescriptor::new(
        0,
        VERTEX_STRIDE,
        GpuVertexStepMode::Vertex,
        [
            GpuVertexAttribute::new(0, 0, GpuVertexFormat::Float32x2),
            GpuVertexAttribute::new(1, 8, GpuVertexFormat::Float32x2),
            GpuVertexAttribute::new(2, 16, GpuVertexFormat::Float32x4),
        ],
    )
    .unwrap();
    let blend = blend.then(|| {
        let component = GpuBlendComponent::new(
            GpuBlendFactor::One,
            GpuBlendFactor::OneMinusSrcAlpha,
            GpuBlendOperation::Add,
        )
        .unwrap();
        GpuBlendState::new(component, component)
    });
    let output =
        GpuColorTargetStateDescriptor::new(target_format, blend, GpuColorWriteMask::ALL).unwrap();
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([vertex_layout]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([output])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .unwrap();
    GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(vertex, Some(fragment)),
        state,
        GpuPipelineConfiguration::default(),
    )
    .unwrap()
}

fn rectangle(vertices: &mut Vec<f32>, bounds: [f32; 4], extent: [f32; 2], color: [f32; 4]) {
    rectangle_local(vertices, bounds, extent, color, [0.0, 0.0]);
}

fn rectangle_local(
    vertices: &mut Vec<f32>,
    bounds: [f32; 4],
    extent: [f32; 2],
    color: [f32; 4],
    tile_origin: [f32; 2],
) {
    let [left, top, right, bottom] = bounds;
    for [x, y] in [
        [left, top],
        [right, top],
        [left, bottom],
        [left, bottom],
        [right, top],
        [right, bottom],
    ] {
        vertices.extend([
            x / extent[0] * 2.0 - 1.0,
            1.0 - y / extent[1] * 2.0,
            x - tile_origin[0],
            y - tile_origin[1],
            color[0],
            color[1],
            color[2],
            color[3],
        ]);
    }
}

fn sample_layer(resources: &mut GpuResourceScope, label: &str) -> GpuTextureViewHandle {
    let texture = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                label,
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                256,
                256,
                GpuTextureFormat::Rgba16Float,
                [GpuTextureUsage::ColorAttachment, GpuTextureUsage::Sampled],
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    resources
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned(format!("{label} view"), &texture)
                .unwrap(),
        )
        .unwrap()
}

#[test]
#[ignore = "GPU-required F3E proof: explicitly executed by the Vulkan workflow"]
fn sample_space_isolates_child_colors_then_applies_group_opacity_once() {
    let mut request =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("F3E once-only isolated group opacity");
    for (format, role) in [
        (
            GpuTextureFormat::Rgba8UnormSrgb,
            GpuFormatRole::ColorAttachment,
        ),
        (GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Blendable),
        (GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::CopySource),
        (
            GpuTextureFormat::Rgba16Float,
            GpuFormatRole::ColorAttachment,
        ),
        (GpuTextureFormat::Rgba16Float, GpuFormatRole::Blendable),
        (GpuTextureFormat::Rgba16Float, GpuFormatRole::Sampled),
        (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled),
        (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopyDestination),
    ] {
        request = request.require_format_role(format, role);
    }
    let context = pollster::block_on(GpuContext::request(request))
        .expect("isolated F3E group proof requires the real admitted Vulkan format roles");

    let mut resources = GpuResourceScope::new();
    let child_view = sample_layer(&mut resources, "F3E isolated child");
    let parent_view = sample_layer(&mut resources, "F3E parent sample plane");
    let output = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                "F3E group opacity result",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                64,
                64,
                GpuTextureFormat::Rgba8UnormSrgb,
                [
                    GpuTextureUsage::ColorAttachment,
                    GpuTextureUsage::CopySource,
                ],
                GpuTextureInitialization::Zeroed,
            )
            .unwrap(),
        )
        .unwrap();
    let output_view = resources
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned("F3E group opacity output", &output)
                .unwrap(),
        )
        .unwrap();

    let mut child_vertices = Vec::new();
    for color in [[1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.5]] {
        rectangle(
            &mut child_vertices,
            [40.0, 80.0, 44.0, 84.0],
            [256.0, 256.0],
            color,
        );
    }
    let fill_pipeline = pipeline(GpuTextureFormat::Rgba16Float, "fs_sample_fill", true);
    let fill_draw = vector::vector_draw(
        fill_pipeline.clone(),
        fill_pipeline.runtime_bindings([]).unwrap(),
        &child_vertices,
        [256, 256],
        &mut resources,
    )
    .unwrap();
    let child_op = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            child_view.clone(),
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 0.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        [fill_draw],
        None,
    )
    .unwrap();

    let merge_pipeline = pipeline(GpuTextureFormat::Rgba16Float, "fs_sample_merge", true);
    let mut merge_vertices = Vec::new();
    rectangle(
        &mut merge_vertices,
        [0.0, 0.0, 256.0, 256.0],
        [256.0, 256.0],
        [1.0, 1.0, 1.0, 0.5],
    );
    let merge_draw = vector::vector_draw(
        merge_pipeline.clone(),
        merge_pipeline
            .runtime_bindings([texture_binding(6, &child_view).unwrap()])
            .unwrap(),
        &merge_vertices,
        [256, 256],
        &mut resources,
    )
    .unwrap();
    let parent_op = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            parent_view.clone(),
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 0.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        [merge_draw],
        None,
    )
    .unwrap();

    let resolve_pipeline = pipeline(GpuTextureFormat::Rgba8UnormSrgb, "fs_sample_resolve", true);
    let mut resolve_vertices = Vec::new();
    rectangle(
        &mut resolve_vertices,
        [0.0, 0.0, 64.0, 64.0],
        [64.0, 64.0],
        [1.0, 1.0, 1.0, 1.0],
    );
    let resolve_draw = vector::vector_draw(
        resolve_pipeline.clone(),
        resolve_pipeline
            .runtime_bindings([texture_binding(6, &parent_view).unwrap()])
            .unwrap(),
        &resolve_vertices,
        [64, 64],
        &mut resources,
    )
    .unwrap();
    let output_op = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            output_view.clone(),
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 0.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        [resolve_draw],
        None,
    )
    .unwrap();
    let readback = GpuReadbackOperation::ordinary(
        GpuTextureCopyRegion::whole_base_mip(&output)
            .unwrap()
            .into(),
    )
    .unwrap();
    let readback_id = readback.id();

    // F3D's actual packed 16-bit mask storage, not an averaged alpha:
    // 0x3333 = the left two samples of each of four horizontal sample rows.
    let public_target = Render2dTarget::new(output_view.clone(), 64.0, 64.0, 1.0).unwrap();
    let admitted = admit_target(&context, &public_target, false).unwrap();
    let clip_mask = crate::runtime::execution_2d::clip::ClipMask {
        root_index: 0,
        origin: [10, 20],
        extent: [1, 1],
        rgba: vec![0x33, 0x33, 0, 255],
    };
    let clipped = clip::upload(&admitted, &clip_mask, &mut resources).unwrap();
    let clip_pipeline = pipeline(
        GpuTextureFormat::Rgba16Float,
        "fs_sample_merge_clipped",
        true,
    );
    let mut clip_values = vec![texture_binding(6, &child_view).unwrap()];
    clip_values.extend(clip::bindings(&clipped, 4, 5).unwrap());
    let clip_draw = vector::vector_draw(
        clip_pipeline.clone(),
        clip_pipeline.runtime_bindings(clip_values).unwrap(),
        &merge_vertices,
        [256, 256],
        &mut resources,
    )
    .unwrap();
    let clipped_parent_op = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            parent_view.clone(),
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 0.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        [clip_draw],
        None,
    )
    .unwrap();
    let clipped_resolve_draw = vector::vector_draw(
        resolve_pipeline.clone(),
        resolve_pipeline
            .runtime_bindings([texture_binding(6, &parent_view).unwrap()])
            .unwrap(),
        &resolve_vertices,
        [64, 64],
        &mut resources,
    )
    .unwrap();
    let clipped_output_op = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            output_view,
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 0.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        [clipped_resolve_draw],
        None,
    )
    .unwrap();
    let clipped_readback = GpuReadbackOperation::ordinary(
        GpuTextureCopyRegion::whole_base_mip(&output)
            .unwrap()
            .into(),
    )
    .unwrap();
    let clipped_id = clipped_readback.id();
    let fragment = GpuWorkFragment::build("F3E isolated group sample passes", |work| {
        work.operation("compose children onto isolated sample scratch", child_op)?;
        work.operation("merge group with opacity once", parent_op)?;
        work.operation("resolve accumulated parent sample colors", output_op)?;
        work.operation("read group proof", readback)?;
        work.operation(
            "merge completed group with exact packed clip",
            clipped_parent_op,
        )?;
        work.operation("resolve once after binary-sample clip", clipped_output_op)?;
        work.operation("read correlated group clip proof", clipped_readback)?;
        Ok(())
    })
    .unwrap();
    let graph = GpuPreparedWorkGraph::prepare(
        GpuResourceLabel::new("F3E nested group sample sequence").unwrap(),
        [fragment],
    )
    .unwrap();
    let submission = context
        .submit_prepared(pollster::block_on(context.prepare_submission(graph)).unwrap())
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(45);
    let bytes = loop {
        context.progress();
        match submission.readback(readback_id).unwrap().status() {
            GpuReadbackStatus::Ready(bytes)
                if submission.status() == GpuSubmissionStatus::Completed =>
            {
                break bytes;
            }
            GpuReadbackStatus::Failed(error) => panic!("F3E group opacity readback: {error:?}"),
            _ => {
                assert!(
                    Instant::now() < deadline,
                    "F3E group opacity readback timed out"
                );
                std::thread::yield_now();
            }
        }
    };
    let offset = (20 * 64 + 10) * 4;
    let pixel = &bytes.as_bytes()[offset..offset + 4];
    // Child red α=.5 then blue α=.5 gives premultiplied
    // (red=.25, blue=.5, α=.75); group opacity .5 yields
    // (red=.125, blue=.25, α=.375) ONCE after composition.
    // Per-child opacity would instead give red=.1875, α=.4375.
    for (actual, expected) in pixel.iter().zip([99_u8, 0, 137, 96]) {
        assert!(
            actual.abs_diff(expected) <= 4,
            "group opacity channel {actual}, expected {expected}"
        );
    }
    let outside = (20 * 64 + 11) * 4;
    assert_eq!(&bytes.as_bytes()[outside..outside + 4], &[0, 0, 0, 0]);

    let clipped_bytes = match submission.readback(clipped_id).unwrap().status() {
        GpuReadbackStatus::Ready(bytes) => bytes,
        status => panic!("F3E clipped group output was not completed: {status:?}"),
    };
    let clipped_pixel = &clipped_bytes.as_bytes()[offset..offset + 4];
    // Exactly eight of sixteen samples survive the parent's 0x3333 mask:
    // premul (red=.0625, blue=.125, α=.1875).
    // Pixel-averaged per-child clipping is numerically different.
    for (actual, expected) in clipped_pixel.iter().zip([71_u8, 0, 99, 48]) {
        assert!(
            actual.abs_diff(expected) <= 4,
            "correlated group clip channel {actual}, expected {expected}"
        );
    }
    assert_eq!(
        &clipped_bytes.as_bytes()[outside..outside + 4],
        &[0, 0, 0, 0]
    );
}

#[test]
#[ignore = "GPU-required F3E proof: explicitly executed by the Vulkan workflow"]
fn retained_gpu_plane_resolves_correlated_siblings_once_over_prior_target() {
    let mut request =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("F3E float sample-plane and final resolve proof");
    for (format, role) in [
        (
            GpuTextureFormat::Rgba8UnormSrgb,
            GpuFormatRole::ColorAttachment,
        ),
        (GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Blendable),
        (GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::CopySource),
        (
            GpuTextureFormat::Rgba16Float,
            GpuFormatRole::ColorAttachment,
        ),
        (GpuTextureFormat::Rgba16Float, GpuFormatRole::Blendable),
        (GpuTextureFormat::Rgba16Float, GpuFormatRole::Sampled),
    ] {
        request = request.require_format_role(format, role);
    }
    let context = pollster::block_on(GpuContext::request(request))
        .expect("F3E GPU-only proof requires admitted Lavapipe Vulkan and Rgba16Float roles");

    let mut resources = GpuResourceScope::new();
    let sample = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                "F3E sample layer",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                8,
                8,
                GpuTextureFormat::Rgba16Float,
                [GpuTextureUsage::ColorAttachment, GpuTextureUsage::Sampled],
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let sample_view = resources
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned("F3E sample view", &sample).unwrap(),
        )
        .unwrap();
    let output = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                "F3E one-time resolved output",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                64,
                64,
                GpuTextureFormat::Rgba8UnormSrgb,
                [
                    GpuTextureUsage::ColorAttachment,
                    GpuTextureUsage::CopySource,
                ],
                GpuTextureInitialization::Zeroed,
            )
            .unwrap(),
        )
        .unwrap();
    let output_view = resources
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned("F3E output view", &output).unwrap(),
        )
        .unwrap();

    let mut sample_vertices = Vec::new();
    // The cropped 8x8 sample texture represents global target pixel (10,20).
    // Red covers 8 of its 16 samples, blue covers the other 8. Per-item pixel averaging would give alpha .75,
    // whereas one coherent sample space has alpha 1.0 before the resolve.
    rectangle(
        &mut sample_vertices,
        [0.0, 0.0, 2.0, 4.0],
        [8.0, 8.0],
        [1.0, 0.0, 0.0, 1.0],
    );
    rectangle(
        &mut sample_vertices,
        [2.0, 0.0, 4.0, 4.0],
        [8.0, 8.0],
        [0.0, 0.0, 1.0, 1.0],
    );
    let sample_pipeline = pipeline(GpuTextureFormat::Rgba16Float, "fs_sample_fill", true);
    let sample_draw = vector::vector_draw(
        sample_pipeline.clone(),
        sample_pipeline.runtime_bindings([]).unwrap(),
        &sample_vertices,
        [8, 8],
        &mut resources,
    )
    .unwrap();
    let samples = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            sample_view.clone(),
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 0.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        [sample_draw],
        None,
    )
    .unwrap();

    let resolve_pipeline = pipeline(GpuTextureFormat::Rgba8UnormSrgb, "fs_sample_resolve", true);
    let mut output_vertices = Vec::new();
    rectangle_local(
        &mut output_vertices,
        [10.0, 20.0, 11.0, 21.0],
        [64.0, 64.0],
        [1.0, 1.0, 1.0, 1.0],
        [10.0, 20.0],
    );
    let resolve_draw = vector::vector_draw(
        resolve_pipeline.clone(),
        resolve_pipeline
            .runtime_bindings([texture_binding(6, &sample_view).unwrap()])
            .unwrap(),
        &output_vertices,
        [64, 64],
        &mut resources,
    )
    .unwrap();
    let clear = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            output_view.clone(),
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 1.0, 1.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        [],
        None,
    )
    .unwrap();
    let resolve = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            output_view,
            GpuColorAttachmentLoad::Load,
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        [resolve_draw],
        None,
    )
    .unwrap();
    let readback = GpuReadbackOperation::ordinary(
        GpuTextureCopyRegion::whole_base_mip(&output)
            .unwrap()
            .into(),
    )
    .unwrap();
    let readback_id = readback.id();
    let fragment = GpuWorkFragment::build("F3E float sample source and resolve", |builder| {
        builder.operation("caller prior blue", clear)?;
        builder.operation("16 sample premultiplied colors", samples)?;
        builder.operation("single physical pixel resolve", resolve)?;
        builder.operation("terminal output readback", readback)?;
        Ok(())
    })
    .unwrap();
    let graph = GpuPreparedWorkGraph::prepare(
        GpuResourceLabel::new("F3E correlated GPU sample proof").unwrap(),
        [fragment],
    )
    .unwrap();
    let submission = context
        .submit_prepared(pollster::block_on(context.prepare_submission(graph)).unwrap())
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(45);
    let bytes = loop {
        context.progress();
        match submission.readback(readback_id).unwrap().status() {
            GpuReadbackStatus::Ready(bytes)
                if submission.status() == GpuSubmissionStatus::Completed =>
            {
                break bytes;
            }
            GpuReadbackStatus::Failed(problem) => panic!("F3E readback: {problem:?}"),
            _ => {
                assert!(Instant::now() < deadline, "F3E GPU submission timed out");
                std::thread::yield_now();
            }
        }
    };
    let pixel = |x: usize, y: usize| {
        let offset = (y * 64 + x) * 4;
        &bytes.as_bytes()[offset..offset + 4]
    };
    // Linear 0.5 is sRGB 188. The last 0.5 blue comes from the
    // *composed sample plane*, not from partially obscured caller blue.
    for (actual, expected) in pixel(10, 20).iter().zip([188_u8, 0, 188, 255]) {
        assert!(
            actual.abs_diff(expected) <= 4,
            "correlated color {actual}, expected {expected}"
        );
    }
    assert_eq!(pixel(11, 20), &[0, 0, 255, 255]);
}
