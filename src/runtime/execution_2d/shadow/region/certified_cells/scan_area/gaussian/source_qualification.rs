//! Qualification through the current real borrowed-F1 source preparation.
use super::*;
use crate::composition_2d::Render2dResourceBindings;
use crate::composition_2d::{
    Render2dColorRgba8, Render2dComposition, Render2dEntry, Render2dGroup, Render2dOpacity,
};
use crate::runtime::execution_2d::scene;
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
enum Variant {
    Sheared,
    Reflected,
    Clipped,
}

fn prepare_fixture(variant: Variant) -> super::plane::PreparedPlane {
    let original = super::tests::original_ring();
    let item = if matches!(variant, Variant::Clipped) {
        Render2dItem::new(
            original.primitive().clone(),
            original.local_to_parent(),
            vec![Render2dClip::new(
                Render2dShape::rect(Render2dRect::new(0.0, 15.0, 64.0, 49.0).unwrap()),
                Render2dAffineTransform::IDENTITY,
            )],
            original.opacity(),
        )
    } else {
        original
    };
    let child = Render2dGroup::new(
        [Render2dEntry::item(item)],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![
            Render2dDropShadow::new(8.0, 0.0, 0.03125, -1.0, Render2dColorRgba8::TRANSPARENT)
                .unwrap(),
        ],
    );
    // The shear maps the COMPLETED child support, after its disk/finite
    // square operations in the child's immediate-parent frame.
    let ancestor = if matches!(variant, Variant::Reflected) {
        Render2dAffineTransform::new(-1.0, 0.0, -0.25, 1.0, 64.0, 0.0).unwrap()
    } else {
        Render2dAffineTransform::new(1.0, 0.0, 0.25, 1.0, 0.0, 0.0).unwrap()
    };
    let wrapper = Render2dGroup::new(
        [Render2dEntry::group(child)],
        ancestor,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        Vec::new(),
    );
    let parent = Render2dGroup::new(
        [Render2dEntry::group(wrapper)],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![
            Render2dDropShadow::new(0.0, 0.0, 0.1, 0.0, Render2dColorRgba8::new(255, 0, 0, 255))
                .unwrap(),
        ],
    );
    let composition = Render2dComposition::new([Render2dEntry::group(parent)]).unwrap();
    // This qualification owns exactly three groups and one item. Prepay the
    // borrowed-plan traversal, path copies, affine composition, activity and
    // frame passes BEFORE analysis. Its temporary vectors fit the separate
    // 64 KiB construction reserve below; no general scene-cost claim follows.
    let mut fuel = 16_777_216;
    debit(&mut fuel, 4096).unwrap();
    let plan = scene::analyze(&composition).unwrap();
    assert_eq!(plan.events.len(), 7);
    let bindings = Render2dResourceBindings::default();
    let fields = BTreeMap::new();
    let sources = shadow::build_group_sources(
        &plan,
        &bindings,
        &fields,
        shadow::NeutralPreparationScale {
            field_raster_scale: 1.0,
            geometry_sample_scale: 4.0,
        },
        64 * 1024 * 1024,
    )
    .unwrap();
    println!(
        "CURRENT_SOURCE_PREPARATION graph_work={} storage={:?} original_F1_no_polygon_demand",
        sources.graph_work(),
        sources.storage(),
    );
    fuel = fuel.checked_sub(sources.graph_work()).unwrap();
    // Lookup, Arc retention and scalar/capacity admission are actual caller
    // preparation, not an excluded diagnostic observer.
    debit(&mut fuel, 1024).unwrap();
    let root = sources
        .source(0)
        .expect("actual consuming parent original F1 support");
    let storage = sources.storage();
    let ambient = storage
        .plan_bytes
        .checked_add(size_of::<Render2dResourceBindings>() + size_of_val(&fields))
        .unwrap();
    let construction = storage
        .construction_peak_bytes
        .checked_add(ambient)
        .unwrap()
        // Plan-construction temporaries are gone before DAG construction.
        .max(65_536);
    // Every numeric construction/packing phase checks the existing fixed
    // reserve. Width-dependent map/row data is separate; DAG + input plan
    // remain live during those phases and are released on function return.
    let query_peak = storage
        .retained_bytes
        .checked_add(ambient)
        .and_then(|v| v.checked_add(MAX_AREA_QUERY_FIXED_BYTES as usize))
        .and_then(|v| v.checked_add(256 * 256 * size_of::<u32>() + 256 * size_of::<f64>()))
        .unwrap();
    assert!(construction <= MAX_BYTES && query_peak <= MAX_BYTES);
    println!(
        "CURRENT_SOURCE_HOST construction_bound={construction} combined_numeric_phase_bound={query_peak}"
    );
    let prepared = PreparedBoundaries::prepare(&root, &mut fuel).unwrap();
    let plan = ScanPlan::prepare(&root, None, &prepared, &mut fuel).unwrap();
    let strips =
        Strips::prepare(&plan, &root, &prepared, [-1.0, -1.0, 65.0, 65.0], &mut fuel).unwrap();
    let field = super::plane::prepare(
        &strips,
        &plan,
        &prepared,
        &root,
        Lattice {
            root: [0, 0, 256, 256],
            scale: 4.0,
            sigma: 0.1,
        },
        &mut fuel,
    )
    .unwrap();
    // Returning an owned plane enforces release of ALL original-source and
    // numeric preparation, including the borrowed ScenePlan and composition.
    // Its type cannot retain a hidden F1 source lifetime through GPU execution.
    field
}

fn run(variant: Variant, execute: bool) {
    let field = prepare_fixture(variant);
    if execute {
        field.execute_proof(
            matches!(variant, Variant::Reflected),
            matches!(variant, Variant::Clipped),
        );
    }
}

#[test]
fn complete_plane_pays_current_original_source_preparation() {
    run(Variant::Sheared, false);
}

#[test]
#[ignore = "requires Vulkan: actual current source preparation and complete Gaussian sample plane"]
fn current_source_complete_plane_matches_independent_references_on_vulkan() {
    run(Variant::Sheared, true);
}

#[test]
fn reflected_complete_plane_pays_current_original_source_preparation() {
    run(Variant::Reflected, false);
}

#[test]
#[ignore = "requires Vulkan: reflected completed support through the original F1 source path"]
fn reflected_source_complete_plane_matches_independent_references_on_vulkan() {
    run(Variant::Reflected, true);
}

#[test]
fn clipped_complete_plane_pays_current_original_source_preparation() {
    run(Variant::Clipped, false);
}

#[test]
#[ignore = "requires Vulkan: original source clip and independent finite-Gaussian cutoff pixels"]
fn clipped_source_complete_plane_matches_independent_references_on_vulkan() {
    run(Variant::Clipped, true);
}
