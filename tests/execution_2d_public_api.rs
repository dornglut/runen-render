use runen_gpu::{
    GpuAttachmentStore, GpuColorAttachmentLoad, GpuColorClearValue, GpuRenderColorAttachment,
    GpuRenderOperation, GpuSubmission,
};
use runen_gpu::{
    GpuBackendFamily, GpuCapabilityProfile, GpuContext, GpuContextDescriptor,
    GpuContextRequestErrorCategory, GpuExportKey, GpuFormatRole, GpuPreparedWorkGraph,
    GpuReadbackBytes, GpuReadbackOperation, GpuReadbackStatus, GpuReconstruction,
    GpuResourceAccessIntent, GpuResourceLabel, GpuResourceLifetime, GpuResourceProvenance,
    GpuResourceRef, GpuResourceScope, GpuSoftwareFallbackPolicy, GpuSubmissionStatus,
    GpuTextureCopyRegion, GpuTextureDescriptor, GpuTextureFormat, GpuTextureHandle,
    GpuTextureInitialization, GpuTextureUsage, GpuTextureViewDescriptor, GpuWorkFragment,
    GpuWorkImport,
};
use runen_render::composition_2d::{
    Render2dAffineTransform, Render2dColorRgba8, Render2dComposition, Render2dEntry,
    Render2dFontBinding, Render2dGlyph, Render2dGroup, Render2dItem, Render2dOpacity,
    Render2dPoint, Render2dPrimitive, Render2dResourceBinding, Render2dResourceBindings,
    Render2dResourceId, Render2dResourceValue, Render2dShapedTextPrimitive,
    Render2dShapedTextResource,
};
use runen_render::execution_2d::{
    Render2dContributionEvidence, Render2dContributionEvidenceError, Render2dContributionToken,
    Render2dExecutionError, Render2dExecutor, Render2dPreparedContribution,
    Render2dShapedTextError, Render2dTarget, Render2dTargetError, Render2dUnsupportedContent,
    Render2dUnsupportedGlyphKind, Render2dWorkBinding,
};
use std::time::{Duration, Instant};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const OUTLINE_FONT: &[u8] = include_bytes!("fixtures/f2_outline.ttf");
const COLR_V0_FONT: &[u8] = include_bytes!("fixtures/f2_colrv0.ttf");
const COLR_V1_FONT: &[u8] = include_bytes!("fixtures/f2_colrv1.ttf");
const SVG_FONT: &[u8] = include_bytes!("fixtures/f2_svg.ttf");
const BITMAP_FONT: &[u8] = include_bytes!("fixtures/f2_bitmap.ttf");
const SPACE_GLYPH: u32 = 1;
const BOX_GLYPH: u32 = 2;

#[path = "execution_2d/clip.rs"]
mod clip;
#[path = "execution_2d/control_order.rs"]
mod control_order;
#[path = "execution_2d/gradient.rs"]
mod gradient;
#[path = "execution_2d/image.rs"]
mod image;
#[path = "execution_2d/vector.rs"]
mod vector;

#[test]
fn f2_public_execution_surface_is_available_to_downstream_consumers() {
    let _ = Render2dTarget::new;
    let _ = Render2dTarget::view;
    let _ = Render2dTarget::logical_width;
    let _ = Render2dTarget::logical_height;
    let _ = Render2dTarget::raster_scale;
    let _ = Render2dExecutor::new;
    let _ = Render2dExecutor::discard_cache;
    let _ = Render2dExecutor::prepare;
    let _ = Render2dPreparedContribution::has_render_work;
    let _ = Render2dPreparedContribution::into_fragment;
    let _ = Render2dPreparedContribution::append_to;
    let _ = Render2dWorkBinding::new;
    let _ = Render2dWorkBinding::after;
    let _ = Render2dContributionToken::authored_nodes;
    let _ = Render2dContributionToken::completed_by;
    let _ = Render2dContributionEvidence::submission_id;

    fn assert_public_error<E: std::error::Error + 'static>() {}
    assert_public_error::<Render2dTargetError>();
    assert_public_error::<Render2dExecutionError>();
    assert_public_error::<Render2dContributionEvidenceError>();
}

#[test]
fn f2_shaped_text_executes_and_preserves_retained_semantics_on_public_runengpu() {
    let Some(context) = f2_context() else {
        return;
    };

    let id = Render2dResourceId::new(1).expect("nonzero resource id");
    let resource = shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0);
    let composition = shaped_composition(id, [8.0, 32.0], Render2dColorRgba8::new(255, 0, 0, 255));
    let resource_bindings = bindings(id, resource.clone());
    let mut executor = Render2dExecutor::new();

    let first = execute(
        &context,
        &mut executor,
        &composition,
        &resource_bindings,
        "first",
    );
    assert_rendered_box(&first);

    let reused = execute(
        &context,
        &mut executor,
        &composition,
        &resource_bindings,
        "same-tier reuse",
    );
    assert_eq!(first.as_bytes(), reused.as_bytes());

    executor.discard_cache();
    let rebound = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 25.0));
    let (_, rebound_target) = target("rebind target");
    assert!(matches!(
        executor.prepare(&context, &composition, &rebound, &rebound_target),
        Err(Render2dExecutionError::ResourceIdentityRebound { resource_id }) if resource_id == id
    ));

    let second = execute(
        &context,
        &mut executor,
        &composition,
        &resource_bindings,
        "after cache discard",
    );
    assert_eq!(first.as_bytes(), second.as_bytes());

    let mut reconstructed = Render2dExecutor::new();
    let fresh_context = f2_context().expect("fresh admitted software Vulkan device");
    let third = execute(
        &fresh_context,
        &mut reconstructed,
        &composition,
        &resource_bindings,
        "fresh executor",
    );
    assert_eq!(first.as_bytes(), third.as_bytes());

    let second_id = Render2dResourceId::new(2).expect("second id");
    let second_composition = Render2dComposition::new(vec![
        shaped_entry(id, [8.0, 32.0], Render2dColorRgba8::WHITE),
        shaped_entry(second_id, [36.0, 32.0], Render2dColorRgba8::WHITE),
    ])
    .expect("two-resource composition");
    let distinct_equal = Render2dResourceBindings::new(vec![
        Render2dResourceBinding::new(id, Render2dResourceValue::ShapedText(resource.clone())),
        Render2dResourceBinding::new(second_id, Render2dResourceValue::ShapedText(resource)),
    ])
    .expect("distinct equal resource values");
    let (_, distinct_target) = target("distinct equal identities target");
    assert!(
        reconstructed
            .prepare(
                &context,
                &second_composition,
                &distinct_equal,
                &distinct_target,
            )
            .expect("distinct semantic identities stay independently admissible")
            .has_render_work()
    );
}

#[test]
fn f2_execution_evidence_rejects_other_work_and_inflight_submissions() {
    let Some(context) = f2_context() else {
        return;
    };
    let id = Render2dResourceId::new(50).unwrap();
    let composition = shaped_composition(id, [8.0, 32.0], Render2dColorRgba8::WHITE);
    let bindings = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0));
    let mut executor = Render2dExecutor::new();
    let (_, target) = target("evidence target");
    let (other_fragment, other_token) = executor
        .prepare(&context, &composition, &bindings, &target)
        .unwrap()
        .into_fragment(&work_binding("contribution"))
        .unwrap();
    let (fragment, token) = executor
        .prepare(&context, &composition, &bindings, &target)
        .unwrap()
        .into_fragment(&work_binding("contribution"))
        .unwrap();
    let graph =
        GpuPreparedWorkGraph::prepare(GpuResourceLabel::new("inflight proof").unwrap(), [fragment])
            .unwrap();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    assert_eq!(submission.status(), GpuSubmissionStatus::Accepted);
    assert_eq!(
        token.unwrap().completed_by(&submission),
        Err(Render2dContributionEvidenceError::SubmissionNotCompleted)
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    while submission.status() == GpuSubmissionStatus::Accepted {
        assert!(Instant::now() < deadline);
        context.progress();
    }
    assert_eq!(submission.status(), GpuSubmissionStatus::Completed);
    assert_eq!(
        other_token.unwrap().completed_by(&submission),
        Err(Render2dContributionEvidenceError::MissingWorkNode)
    );
    drop(other_fragment);
}

#[test]
fn f2_failed_preparation_does_not_observe_resource_identities() {
    let Some(context) = f2_context() else {
        return;
    };
    let first = Render2dResourceId::new(60).unwrap();
    let invalid = Render2dResourceId::new(61).unwrap();
    let composition = Render2dComposition::new(vec![
        shaped_entry(first, [8.0, 32.0], Render2dColorRgba8::WHITE),
        shaped_entry(invalid, [36.0, 32.0], Render2dColorRgba8::WHITE),
    ])
    .unwrap();
    let bindings = Render2dResourceBindings::new(vec![
        Render2dResourceBinding::new(
            first,
            Render2dResourceValue::ShapedText(shaped_resource(
                OUTLINE_FONT,
                false,
                BOX_GLYPH,
                24.0,
            )),
        ),
        Render2dResourceBinding::new(
            invalid,
            Render2dResourceValue::ShapedText(shaped_resource(OUTLINE_FONT, false, u32::MAX, 24.0)),
        ),
    ])
    .unwrap();
    let (_, target) = target("transaction target");
    let mut executor = Render2dExecutor::new();
    assert!(
        matches!(executor.prepare(&context, &composition, &bindings, &target), Err(Render2dExecutionError::ShapedText(Render2dShapedTextError::InvalidOutline { resource_id, .. })) if resource_id == invalid)
    );
    let changed = self::bindings(first, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 25.0));
    let first_only = shaped_composition(first, [8.0, 32.0], Render2dColorRgba8::WHITE);
    assert!(
        executor
            .prepare(&context, &first_only, &changed, &target)
            .unwrap()
            .has_render_work()
    );
}

#[test]
fn f2_preserves_painter_order_and_linear_light_source_over() {
    let Some(context) = f2_context() else {
        return;
    };

    let id = Render2dResourceId::new(10).expect("blend resource id");
    let resource = shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0);
    let composition = Render2dComposition::new(vec![
        shaped_entry(id, [8.0, 32.0], Render2dColorRgba8::new(255, 0, 0, 128)),
        shaped_entry(id, [8.0, 32.0], Render2dColorRgba8::new(0, 255, 0, 128)),
    ])
    .expect("ordered overlapping shaped text");
    let bindings = bindings(id, resource);
    let mut executor = Render2dExecutor::new();
    let bytes = execute(
        &context,
        &mut executor,
        &composition,
        &bindings,
        "linear source-over",
    );

    // The fixture box covers this sample well inside its outline, so coverage is 1.0.
    // Red is painted first, then green. With alpha = 128/255 and premultiplied
    // linear-light source-over, the encoded RGBA8 result is approximately
    // [137, 188, 0, 192]. Reversing painter order swaps red/green, while sRGB-space
    // blending or replace blending produces materially different values.
    let observed = pixel(&bytes, 20, 20);
    let expected = [137_u8, 188, 0, 192];
    for (channel, (actual, expected)) in observed.into_iter().zip(expected).enumerate() {
        assert!(
            actual.abs_diff(expected) <= 3,
            "source-over channel {channel} expected {expected}±3, observed {actual}"
        );
    }
}

#[test]
fn f2_nonpainting_and_unsupported_content_fail_or_elide_structurally() {
    let Some(context) = f2_context() else {
        return;
    };

    let mut executor = Render2dExecutor::new();
    let space_id = Render2dResourceId::new(20).expect("space id");
    let space = shaped_resource(OUTLINE_FONT, false, SPACE_GLYPH, 24.0);
    let space_composition = shaped_composition(space_id, [8.0, 32.0], Render2dColorRgba8::WHITE);
    let space_bindings = bindings(space_id, space);
    let (_, space_target) = target("space target");
    let prepared = executor
        .prepare(&context, &space_composition, &space_bindings, &space_target)
        .expect("outline-free space glyph is valid non-painting content");
    assert!(!prepared.has_render_work());
    let (_, token) = prepared
        .into_fragment(&work_binding("contribution"))
        .unwrap();
    assert!(token.is_none());

    let group = Render2dGroup::new(
        Vec::new(),
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        Vec::new(),
    );
    let grouped = Render2dComposition::new(vec![Render2dEntry::group(group)])
        .expect("structurally valid group composition");
    let (_, grouped_target) = target("group target");
    assert!(matches!(
        executor.prepare(
            &context,
            &grouped,
            &Render2dResourceBindings::default(),
            &grouped_target,
        ),
        Err(Render2dExecutionError::UnsupportedContent(
            Render2dUnsupportedContent::Group { root_index: 0 }
        ))
    ));

    for (index, (font, expected)) in [
        (COLR_V0_FONT, Render2dUnsupportedGlyphKind::ColrV0),
        (COLR_V1_FONT, Render2dUnsupportedGlyphKind::ColrV1),
        (SVG_FONT, Render2dUnsupportedGlyphKind::Svg),
        (BITMAP_FONT, Render2dUnsupportedGlyphKind::Bitmap),
    ]
    .into_iter()
    .enumerate()
    {
        let id = Render2dResourceId::new(30 + u64::try_from(index).expect("small fixture index"))
            .expect("fixture id");
        assert_unsupported_glyph(&context, id, font, false, Some(BOX_GLYPH), expected);
    }

    let faux_bold_id = Render2dResourceId::new(40).expect("faux bold id");
    assert_unsupported_glyph(
        &context,
        faux_bold_id,
        OUTLINE_FONT,
        true,
        None,
        Render2dUnsupportedGlyphKind::FauxBold,
    );
}

fn assert_unsupported_glyph(
    context: &GpuContext,
    id: Render2dResourceId,
    font: &[u8],
    faux_bold: bool,
    expected_glyph_id: Option<u32>,
    expected_kind: Render2dUnsupportedGlyphKind,
) {
    let composition = shaped_composition(id, [8.0, 32.0], Render2dColorRgba8::WHITE);
    let bindings = bindings(id, shaped_resource(font, faux_bold, BOX_GLYPH, 24.0));
    let (_, target) = target("unsupported glyph target");
    let mut executor = Render2dExecutor::new();
    assert!(matches!(
        executor.prepare(context, &composition, &bindings, &target),
        Err(Render2dExecutionError::ShapedText(
            Render2dShapedTextError::UnsupportedGlyph {
                resource_id,
                glyph_id,
                kind,
            }
        )) if resource_id == id && glyph_id == expected_glyph_id && kind == expected_kind
    ));
}

fn f2_context() -> Option<GpuContext> {
    f2_context_with_blendable_role(true)
}

fn f2_context_with_blendable_role(blendable: bool) -> Option<GpuContext> {
    let mut descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .require_format_role(
                GpuTextureFormat::Rgba8UnormSrgb,
                GpuFormatRole::ColorAttachment,
            )
            .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::CopySource)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Filterable)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopyDestination)
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("RunenRender F2 public shaped-text proof");

    if blendable {
        descriptor = descriptor
            .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Blendable);
    }

    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => Some(context),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1"),
                "F2 public consumer CI requires the retained Vulkan software adapter"
            );
            None
        }
        Err(error) => panic!("unexpected F2 public RunenGPU context failure: {error}"),
    }
}

#[test]
fn f2_source_over_requires_explicit_blendable_admission() {
    let Some(context) = f2_context_with_blendable_role(false) else {
        return;
    };
    let id = Render2dResourceId::new(90).unwrap();
    let composition = shaped_composition(id, [8.0, 32.0], Render2dColorRgba8::WHITE);
    let bindings = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0));
    let (_, target) = target("non-blendable admission");
    assert!(matches!(
        Render2dExecutor::new().prepare(&context, &composition, &bindings, &target),
        Err(Render2dExecutionError::Target(
            runen_render::execution_2d::Render2dTargetAdmissionError::TargetFormatNotBlendable
        ))
    ));
}

fn shaped_resource(
    font: &[u8],
    faux_bold: bool,
    glyph_id: u32,
    font_size: f64,
) -> Render2dShapedTextResource {
    let font = Render2dFontBinding::new(font.to_vec(), 0, Vec::new(), faux_bold, None)
        .expect("fixture font binding");
    let glyph = Render2dGlyph::new(glyph_id, 0.0, 0.0, font_size).expect("finite shaped glyph");
    Render2dShapedTextResource::new(font, font_size, vec![glyph]).expect("finite shaped resource")
}

fn bindings(
    id: Render2dResourceId,
    resource: Render2dShapedTextResource,
) -> Render2dResourceBindings {
    Render2dResourceBindings::new(vec![Render2dResourceBinding::new(
        id,
        Render2dResourceValue::ShapedText(resource),
    )])
    .expect("one semantic binding")
}

fn shaped_composition(
    id: Render2dResourceId,
    origin: [f64; 2],
    color: Render2dColorRgba8,
) -> Render2dComposition {
    Render2dComposition::new(vec![shaped_entry(id, origin, color)])
        .expect("one shaped-text composition")
}

fn shaped_entry(
    id: Render2dResourceId,
    origin: [f64; 2],
    color: Render2dColorRgba8,
) -> Render2dEntry {
    let text = Render2dShapedTextPrimitive::new(
        id,
        Render2dPoint::new(origin[0], origin[1]).expect("finite shaped origin"),
        color,
    );
    Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::ShapedText(text),
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
    ))
}

fn target(name: &str) -> (GpuTextureHandle, Render2dTarget) {
    let mut resources = GpuResourceScope::new();
    let texture = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                name,
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                WIDTH,
                HEIGHT,
                GpuTextureFormat::Rgba8UnormSrgb,
                [
                    GpuTextureUsage::ColorAttachment,
                    GpuTextureUsage::CopySource,
                ],
                GpuTextureInitialization::Zeroed,
            )
            .expect("F2 target descriptor"),
        )
        .expect("F2 target identity");
    let view = resources
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned(format!("{name} view"), &texture)
                .expect("F2 target view descriptor"),
        )
        .expect("F2 target view identity");
    let target = Render2dTarget::new(view, f64::from(WIDTH), f64::from(HEIGHT), 1.0)
        .expect("exact F2 target mapping");
    (texture, target)
}

fn execute(
    context: &GpuContext,
    executor: &mut Render2dExecutor,
    composition: &Render2dComposition,
    bindings: &Render2dResourceBindings,
    label: &str,
) -> GpuReadbackBytes {
    let (texture, target) = target(label);
    let contribution = executor
        .prepare(context, composition, bindings, &target)
        .expect("F2 contribution preparation");
    assert!(contribution.has_render_work());
    let (fragment, token) = contribution
        .into_fragment(&work_binding("contribution"))
        .unwrap();
    let token = token.expect("painting contribution carries exact work-node token");
    let output_key = fragment.outputs()[0].relationship().export_key().clone();

    let readback = GpuReadbackOperation::ordinary(
        GpuTextureCopyRegion::whole_base_mip(&texture)
            .expect("full target readback region")
            .into(),
    )
    .expect("readback operation");
    let readback_id = readback.id();
    let readback_fragment = GpuWorkFragment::build(format!("{label} readback"), |work| {
        work.operation("read F2 target", readback)?;
        work.add_import(GpuWorkImport::new(
            GpuResourceRef::Texture(texture.clone()),
            output_key,
            GpuResourceAccessIntent::Read,
            GpuResourceProvenance::new(
                GpuResourceLabel::new("F2 readback import").unwrap(),
                None,
                None,
            ),
        ))?;
        Ok(())
    })
    .expect("readback fragment");
    let graph = GpuPreparedWorkGraph::prepare(
        GpuResourceLabel::new(format!("{label} graph")).expect("graph label"),
        [fragment, readback_fragment],
    )
    .expect("downstream-composed F2 graph");
    let prepared =
        pollster::block_on(context.prepare_submission(graph)).expect("F2 submission preparation");
    let submission = context
        .submit_prepared(prepared)
        .expect("F2 submission acceptance");

    let bytes = await_readback(context, &submission, readback_id);

    let evidence = token
        .completed_by(&submission)
        .expect("exact F2 work-node membership plus terminal completion");
    assert_eq!(evidence.submission_id(), submission.id());
    assert_eq!(
        bytes.texture_format(),
        Some(GpuTextureFormat::Rgba8UnormSrgb)
    );
    bytes
}

fn await_readback(
    context: &GpuContext,
    submission: &GpuSubmission,
    readback_id: runen_gpu::GpuReadbackId,
) -> GpuReadbackBytes {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Failed(failure) => {
                panic!("F2 public submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Accepted | GpuSubmissionStatus::Completed => {}
        }
        let readback = submission
            .readback(readback_id)
            .expect("submission retains requested F2 readback");
        match readback.status() {
            GpuReadbackStatus::Ready(bytes)
                if submission.status() == GpuSubmissionStatus::Completed =>
            {
                return bytes;
            }
            GpuReadbackStatus::Failed(failure) => {
                panic!("F2 public readback failed: {failure:?}")
            }
            GpuReadbackStatus::Pending | GpuReadbackStatus::Ready(_) => {
                assert!(
                    Instant::now() < deadline,
                    "F2 public submission/readback did not complete before timeout"
                );
                std::thread::yield_now();
            }
        }
    }
}

fn execute_inline(
    context: &GpuContext,
    executor: &mut Render2dExecutor,
    composition: &Render2dComposition,
    bindings: &Render2dResourceBindings,
    texture: &GpuTextureHandle,
    target: &Render2dTarget,
    clear: Option<[f64; 4]>,
) -> GpuReadbackBytes {
    let contribution = executor
        .prepare(context, composition, bindings, target)
        .unwrap();
    let readback = GpuReadbackOperation::ordinary(
        GpuTextureCopyRegion::whole_base_mip(texture)
            .unwrap()
            .into(),
    )
    .unwrap();
    let readback_id = readback.id();
    let mut token = None;
    let fragment = GpuWorkFragment::build("caller-owned lexical composition", |work| {
        if let Some([r, g, b, a]) = clear {
            work.operation(
                "caller-owned clear",
                GpuRenderOperation::new(
                    [GpuRenderColorAttachment::new(
                        target.view().clone(),
                        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(r, g, b, a).unwrap()),
                        GpuAttachmentStore::Store,
                        None,
                    )
                    .unwrap()],
                    None,
                    [],
                    None,
                )
                .unwrap(),
            )?;
        }
        token = contribution.append_to(work)?;
        work.operation("caller-owned readback", readback)?;
        Ok(())
    })
    .unwrap();
    let graph =
        GpuPreparedWorkGraph::prepare(GpuResourceLabel::new("lexical graph").unwrap(), [fragment])
            .unwrap();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = await_readback(context, &submission, readback_id);
    assert_eq!(
        token
            .unwrap()
            .completed_by(&submission)
            .unwrap()
            .submission_id(),
        submission.id()
    );
    bytes
}

#[test]
fn f2_appends_between_caller_clear_and_readback_and_loads_prior_contents() {
    let Some(context) = f2_context() else {
        return;
    };
    let id = Render2dResourceId::new(70).unwrap();
    let composition = shaped_composition(id, [8.0, 32.0], Render2dColorRgba8::new(255, 0, 0, 128));
    let bindings = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0));
    let (texture, target) = target("caller blue target");
    let bytes = execute_inline(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &bindings,
        &texture,
        &target,
        Some([0.0, 0.0, 1.0, 1.0]),
    );
    assert_eq!(pixel(&bytes, 0, 0), [0, 0, 255, 255]);
    for (actual, expected) in pixel(&bytes, 20, 20).into_iter().zip([188_u8, 0, 187, 255]) {
        assert!(actual.abs_diff(expected) <= 3);
    }
}

#[test]
fn f2_separate_fragment_imports_prior_target_contents_instead_of_array_order() {
    let Some(context) = f2_context() else {
        return;
    };
    let id = Render2dResourceId::new(75).unwrap();
    let composition = shaped_composition(id, [8.0, 32.0], Render2dColorRgba8::WHITE);
    let bindings = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0));
    let (texture, target) = target("typed prior target");
    let prior_key = GpuExportKey::new("caller.prior").unwrap();
    let provenance =
        GpuResourceProvenance::new(GpuResourceLabel::new("caller prior").unwrap(), None, None);
    let clear = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            target.view().clone(),
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
    let prior = GpuWorkFragment::build("caller prior fragment", |work| {
        work.operation("caller clear", clear)?;
        work.declare_resource(GpuResourceRef::Texture(texture.clone()))?;
        work.add_output(runen_gpu::GpuWorkOutput::new(
            runen_gpu::GpuExportRelationship::new(
                GpuResourceRef::Texture(texture.clone()),
                prior_key.clone(),
                GpuResourceAccessIntent::Write,
                provenance.clone(),
            ),
            runen_gpu::GpuInitialCoverage::texture_subresources(
                &runen_gpu::GpuTextureAccessResource::Texture(texture.clone()),
                [target.view().descriptor().subresources()],
            )?,
        )?)?;
        Ok(())
    })
    .unwrap();
    let contribution = Render2dExecutor::new()
        .prepare(&context, &composition, &bindings, &target)
        .unwrap();
    let (fragment, token) = contribution
        .into_fragment(&work_binding("after prior").after(prior_key))
        .unwrap();
    let graph = GpuPreparedWorkGraph::prepare(
        GpuResourceLabel::new("reversed fragment inventory").unwrap(),
        [fragment, prior],
    )
    .unwrap();
    assert_eq!(graph.topological_order()[0].fragment_ordinal(), 1);
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    while submission.status() == GpuSubmissionStatus::Accepted {
        assert!(Instant::now() < deadline);
        context.progress();
    }
    assert_eq!(
        token
            .unwrap()
            .completed_by(&submission)
            .unwrap()
            .submission_id(),
        submission.id()
    );
}

#[test]
fn f2_exact_raster_scale_translation_and_continuous_canvas_clipping_are_preserved() {
    let Some(context) = f2_context() else {
        return;
    };
    let id = Render2dResourceId::new(80).unwrap();
    let bindings = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0));
    let composition = shaped_composition(id, [44.0, 32.0], Render2dColorRgba8::new(255, 0, 0, 255));
    let translated = Render2dComposition::new(vec![Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::ShapedText(Render2dShapedTextPrimitive::new(
            id,
            Render2dPoint::new(40.0, 30.0).unwrap(),
            Render2dColorRgba8::new(255, 0, 0, 255),
        )),
        Render2dAffineTransform::translation(4.0, 2.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
    ))])
    .unwrap();
    let mut executor = Render2dExecutor::new();
    let (texture, unit_target) = target("fractional canvas");
    let fractional_target =
        Render2dTarget::new(unit_target.view().clone(), 50.6, 51.2, 1.25).unwrap();
    let first = execute_inline(
        &context,
        &mut executor,
        &composition,
        &bindings,
        &texture,
        &fractional_target,
        None,
    );
    let (texture2, unit_target2) = target("translated fractional canvas");
    let fractional_target2 =
        Render2dTarget::new(unit_target2.view().clone(), 50.6, 51.2, 1.25).unwrap();
    let second = execute_inline(
        &context,
        &mut executor,
        &translated,
        &bindings,
        &texture2,
        &fractional_target2,
        None,
    );
    assert_eq!(first.as_bytes(), second.as_bytes());
    assert!(pixel(&first, 62, 25)[0] > 200);
    assert_eq!(pixel(&first, 63, 25), [0, 0, 0, 0]);
    let mismatched = Render2dTarget::new(unit_target.view().clone(), 50.0, 51.2, 1.25).unwrap();
    assert!(matches!(
        executor.prepare(&context, &composition, &bindings, &mismatched),
        Err(Render2dExecutionError::Target(
            runen_render::execution_2d::Render2dTargetAdmissionError::PhysicalExtentMismatch { .. }
        ))
    ));
}

fn work_binding(label: &str) -> Render2dWorkBinding {
    Render2dWorkBinding::new(GpuExportKey::new(format!("f2.{label}")).unwrap())
}

fn assert_rendered_box(bytes: &GpuReadbackBytes) {
    assert_eq!(
        bytes.as_bytes().len(),
        usize::try_from(WIDTH * HEIGHT * 4).expect("small proof extent")
    );
    assert_eq!(pixel(bytes, 0, 0), [0, 0, 0, 0]);
    assert!(
        bytes
            .as_bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .any(|pixel| pixel[0] > 200 && pixel[3] > 200),
        "outline fixture must produce visibly covered red pixels"
    );
}

fn pixel(bytes: &GpuReadbackBytes, x: u32, y: u32) -> [u8; 4] {
    let offset = usize::try_from((y * WIDTH + x) * 4).expect("small proof offset");
    bytes.as_bytes()[offset..offset + 4]
        .try_into()
        .expect("RGBA8 pixel")
}
