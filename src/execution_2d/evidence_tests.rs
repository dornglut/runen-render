use super::*;
use runen_gpu::*;
use std::time::{Duration, Instant};

#[test]
fn terminal_subset_cannot_certify_a_complete_multi_node_contribution() {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require);
    let context = match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => context,
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1")
            );
            return;
        }
        Err(error) => panic!("evidence context: {error}"),
    };
    let mut resources = GpuResourceScope::new();
    let texture = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                "evidence target",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                [GpuTextureUsage::ColorAttachment],
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let view = resources
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned("evidence view", &texture).unwrap(),
        )
        .unwrap();
    let clear = || {
        GpuRenderOperation::new(
            [GpuRenderColorAttachment::new(
                view.clone(),
                GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 0.0).unwrap()),
                GpuAttachmentStore::Store,
                None,
            )
            .unwrap()],
            None,
            [],
            None,
        )
        .unwrap()
    };
    let first = GpuWorkFragment::build("present node", |work| {
        work.operation("first", clear())?;
        Ok(())
    })
    .unwrap();
    let absent = GpuWorkFragment::build("omitted node", |work| {
        work.operation("second", clear())?;
        Ok(())
    })
    .unwrap();
    let present = first.nodes()[0].id().clone();
    let missing = absent.nodes()[0].id().clone();
    let graph =
        GpuPreparedWorkGraph::prepare(GpuResourceLabel::new("subset graph").unwrap(), [first])
            .unwrap();
    let submission = context
        .submit_prepared(pollster::block_on(context.prepare_submission(graph)).unwrap())
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    while submission.status() == GpuSubmissionStatus::Accepted {
        assert!(Instant::now() < deadline);
        context.progress();
    }
    assert_eq!(submission.status(), GpuSubmissionStatus::Completed);
    assert!(submission.contains_work_node(&present));
    assert!(!submission.contains_work_node(&missing));
    // Both prefix-only and last-node-only implementations would accept one of these.
    for nodes in [
        vec![present.clone(), missing.clone()],
        vec![missing, present.clone()],
    ] {
        assert_eq!(
            Render2dContributionToken { nodes }.completed_by(&submission),
            Err(Render2dContributionEvidenceError::MissingWorkNode)
        );
    }
    assert!(
        Render2dContributionToken {
            nodes: vec![present]
        }
        .completed_by(&submission)
        .is_ok()
    );
}
