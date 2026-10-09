//! Public exact-fragment-order and completion proof for composable F2.
use super::*;
use runen_gpu::{GpuExplicitOrder, GpuWorkNodeId};
use runen_render::composition_2d::{Render2dBrush, Render2dRect, Render2dShape};

fn clear(target: &Render2dTarget, red: f64) -> GpuRenderOperation {
    GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            target.view().clone(),
            GpuColorAttachmentLoad::Clear(
                GpuColorClearValue::new(red, 0.0, 0.0, 1.0).expect("finite color"),
            ),
            GpuAttachmentStore::Store,
            None,
        ).expect("valid attachment")],
        None,
        [],
        None,
    ).expect("independent GPU clear")
}

fn position(graph: &GpuPreparedWorkGraph, node_id: &GpuWorkNodeId) -> usize {
    let prepared = graph.nodes().iter()
        .find(|node| node.node().id() == node_id)
        .expect("actual authored GPU node").id();
    graph.topological_order().iter().position(|node| *node == prepared)
        .expect("ordered GPU node")
}

fn vector_item(x: f64) -> Render2dEntry {
    Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::rect(
                Render2dRect::new(x, x, 8.0, 8.0).expect("rectangle")
            ),
            brush: Render2dBrush::solid(Render2dColorRgba8::WHITE),
        },
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
    ))
}

#[test]
fn independent_predecessor_and_successor_bracket_every_authored_f2_node() {
    let Some(context) = f2_context() else { return };
    let id = Render2dResourceId::new(923).expect("test id");
    let composition = Render2dComposition::new(vec![
        vector_item(0.0),
        shaped_entry(id, [8.0, 32.0], Render2dColorRgba8::WHITE),
        vector_item(48.0),
    ]).expect("vector/text/vector paint");
    let source = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0));
    let (main_texture, target) = target("controlled F2 surface");
    let (_, before_target) = target("independent predecessor surface");
    let (_, after_target) = target("independent successor surface");
    let prepared_f2 = Render2dExecutor::new()
        .prepare(&context, &composition, &source, &target)
        .expect("real mixed contribution");
    assert!(prepared_f2.has_render_work());
    let readback = GpuReadbackOperation::ordinary(
        GpuTextureCopyRegion::whole_base_mip(&main_texture)
            .expect("full target").into(),
    ).expect("terminal readback");
    let readback_id = readback.id();
    let mut token = None;
    let mut control = None;
    let fragment = GpuWorkFragment::build("mixed F2 inside one fragment", |work| {
        work.operation("initialize target", clear(&target, 0.0))?;
        let predecessor = work.operation("unrelated before", clear(&before_target, 1.0))?;
        let f2 = prepared_f2.append_to(work)?.expect("painting token");
        assert!(f2.authored_nodes().len() >= 2, "F2 multi-node token must be complete");
        work.add_explicit_order(GpuExplicitOrder::new(
            &predecessor,
            f2.authored_nodes().first().expect("first"),
            "independent before first F2",
        )?)?;
        let successor = work.operation("unrelated after", clear(&after_target, 0.5))?;
        work.add_explicit_order(GpuExplicitOrder::new(
            f2.authored_nodes().last().expect("last"),
            &successor,
            "last F2 before independent successor",
        )?)?;
        control = Some((predecessor, f2.authored_nodes().to_vec(), successor));
        token = Some(f2);
        work.operation("terminal readback", readback)?;
        Ok(())
    }).expect("fragment-local authoring");
    let graph = GpuPreparedWorkGraph::prepare(
        GpuResourceLabel::new("single graph with independent controls").unwrap(),
        [fragment],
    ).expect("RunenGPU prepares explicit control constraints");
    let (before, nodes, after) = control.expect("captured authored identities");
    for node in nodes {
        assert!(
            position(&graph, &before) < position(&graph, &node)
                && position(&graph, &node) < position(&graph, &after),
            "every F2 node must be bracketed by independent controls"
        );
    }
    let submission = context.submit_prepared(
        pollster::block_on(context.prepare_submission(graph))
            .expect("GPU submission prepared"),
    ).expect("GPU work accepted");
    let bytes = await_readback(&context, &submission, readback_id);
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8UnormSrgb));
    assert_eq!(
        token.expect("exact token").completed_by(&submission)
            .expect("all F2 nodes completed").submission_id(),
        submission.id()
    );
}

#[test]
fn nonpainting_f2_authors_no_node_and_no_evidence_token() {
    let Some(context) = f2_context() else { return };
    let id = Render2dResourceId::new(924).expect("test id");
    let (_, target) = target("nonpainting target");
    let prepared_f2 = Render2dExecutor::new().prepare(
        &context,
        &shaped_composition(id, [8.0, 32.0], Render2dColorRgba8::WHITE),
        &bindings(id, shaped_resource(OUTLINE_FONT, false, SPACE_GLYPH, 24.0)),
        &target,
    ).expect("valid invisible content");
    let mut token = None;
    let fragment = GpuWorkFragment::build("nonpainting F2", |work| {
        token = prepared_f2.append_to(work)?;
        Ok(())
    }).expect("legal no-work fragment");
    assert!(token.is_none());
    assert!(fragment.nodes().is_empty());
}
