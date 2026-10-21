//! Original image destinations qualify the same geometric ring independently
//! of texel alpha and source crop. No fill path is constructed as query truth.
use super::*;
use crate::composition_2d::{
    Render2dColorRgba8, Render2dComposition, Render2dEntry, Render2dGroup, Render2dImagePatch,
    Render2dImagePrimitive, Render2dImageResource, Render2dImageSourceRect, Render2dOpacity,
    Render2dPixelExtent, Render2dResourceBinding, Render2dResourceBindings, Render2dResourceId,
    Render2dResourceValue,
};
use std::collections::BTreeMap;

fn ring_image() -> Render2dImagePrimitive {
    // Four rectangles have disjoint interiors and their closed union is
    // [10,30]^2 minus the OPEN hole (18,22)^2. Its measure and exposed
    // boundary equal the independent ring references; internal seams vanish.
    let patches = [
        [10.0, 10.0, 20.0, 8.0],
        [10.0, 22.0, 20.0, 8.0],
        [10.0, 18.0, 8.0, 4.0],
        [22.0, 18.0, 8.0, 4.0],
    ]
    .into_iter()
    .enumerate()
    .map(|(index, [x, y, w, h])| {
        Render2dImagePatch::new(
            // F3F-NI-01: positive destination with zero-area source crop.
            Render2dImageSourceRect::new(0.0, 0.0, if index == 0 { 0.0 } else { 1.0 }, 1.0)
                .unwrap(),
            Render2dRect::new(x, y, w, h).unwrap(),
        )
    })
    .collect::<Vec<_>>();
    Render2dImagePrimitive::new(
        Render2dResourceId::new(1).unwrap(),
        Render2dPixelExtent::new(1, 1).unwrap(),
        patches,
    )
    .unwrap()
}

fn image_item(local_to_parent: Render2dAffineTransform, clips: Vec<Render2dClip>) -> Render2dItem {
    Render2dItem::new(
        Render2dPrimitive::Image(ring_image()),
        local_to_parent,
        clips,
        Render2dOpacity::TRANSPARENT,
    )
}

fn prepare_fixture() -> super::plane::PreparedPlane {
    let item = image_item(Render2dAffineTransform::IDENTITY, Vec::new());
    let child = Render2dGroup::new(
        [Render2dEntry::item(item)],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        vec![
            Render2dDropShadow::new(8.0, 0.0, 0.03125, -1.0, Render2dColorRgba8::TRANSPARENT)
                .unwrap(),
        ],
    );
    let wrapper = Render2dGroup::new(
        [Render2dEntry::group(child)],
        Render2dAffineTransform::new(1.0, 0.0, 0.25, 1.0, 0.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
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
    let bindings = Render2dResourceBindings::new(vec![Render2dResourceBinding::new(
        Render2dResourceId::new(1).unwrap(),
        Render2dResourceValue::ImageRgba8Srgb(
            Render2dImageResource::new(Render2dPixelExtent::new(1, 1).unwrap(), vec![0; 4])
                .unwrap(),
        ),
    )])
    .unwrap();
    // Exactly three groups, one item and one binding: prepay immutable binding
    // validation. This source/numerical proof does not construct an affine Plan.
    let mut fuel = 16_777_216;
    debit(&mut fuel, 512).unwrap();
    composition.validate_bindings(&bindings).unwrap();
    let fields = BTreeMap::new();
    let mut preparation = shadow::SourcePreparation::new(
        shadow::NeutralPreparationScale {
            field_raster_scale: 1.0,
            geometry_sample_scale: 4.0,
        },
        MAX_BYTES as u64,
    );
    assert_eq!(preparation.admit_effects(&composition).unwrap(), 2);
    let sources =
        shadow::build_group_sources(&composition, 0, &bindings, &fields, preparation).unwrap();
    fuel = fuel.checked_sub(sources.graph_work()).unwrap();
    debit(&mut fuel, 1024).unwrap();
    let source = sources.source(0).unwrap();
    let storage = sources.storage();
    // Bound the single immutable resource/binding payload and map control
    // separately from the original renderer DAG retention.
    let ambient = storage
        .input_bytes
        .checked_add(size_of::<Render2dResourceBindings>() + size_of_val(&fields) + 1024)
        .unwrap();
    let construction = storage
        .construction_peak_bytes
        .checked_add(ambient)
        .unwrap()
        .max(65_536);
    let query_peak = storage
        .retained_bytes
        .checked_add(ambient)
        .and_then(|v| v.checked_add(MAX_AREA_QUERY_FIXED_BYTES as usize))
        .and_then(|v| v.checked_add(256 * 256 * size_of::<u32>() + 256 * size_of::<f64>()))
        .unwrap();
    assert!(construction <= MAX_BYTES && query_peak <= MAX_BYTES);
    let field = source
        .prepare_gaussian_plane(
            GaussianPlaneRequest {
                root: [0, 0, 256, 256],
                scale: 4.0,
                sigma: 0.1,
                offset: [0.0; 2],
                parent_to_root: Render2dAffineTransform::IDENTITY,
            },
            &mut fuel,
            MAX_BYTES
                .checked_sub(storage.retained_bytes + ambient)
                .unwrap(),
        )
        .unwrap();
    assert_eq!(field.samples().len(), 65_536);
    assert!(fuel > 0);
    println!(
        "IMAGE_SOURCE_COMPLETE graph_work={} remaining={fuel} construction_bound={construction} numeric_phase_bound={query_peak} retained_field_bytes={} GPU_node_visits={}; transparent_bytes_and_empty_source_crop_original_destination_ring",
        sources.graph_work(),
        field.retained_bytes().unwrap(),
        field.query_node_visits(),
    );
    field
}

#[test]
fn image_destinations_preserve_hole_crop_transform_clips_and_query_fuel() {
    let identity = image_item(Render2dAffineTransform::IDENTITY, Vec::new());
    let mut fuel = 16_777_216;
    let query = |item: &Render2dItem, [x, y]: [f64; 2], fuel: &mut usize| {
        certify_authored_item(item, Cell::new([x, y, x, y]).unwrap(), fuel)
    };
    // Independent direct rectangle membership, including the empty-crop arm.
    for point in [[12.0, 12.0], [12.0, 20.0], [28.0, 20.0], [20.0, 28.0]] {
        assert_eq!(query(&identity, point, &mut fuel), CellCertificate::Inside);
    }
    for point in [[20.0, 20.0], [8.0, 20.0], [32.0, 20.0]] {
        assert_eq!(query(&identity, point, &mut fuel), CellCertificate::Outside);
    }
    assert_eq!(
        query(&identity, [10.0, 12.0], &mut fuel),
        CellCertificate::Unknown
    );
    let moved = image_item(
        Render2dAffineTransform::translation(4.0, -2.0).unwrap(),
        vec![Render2dClip::new(
            Render2dShape::rect(Render2dRect::new(15.0, 9.0, 2.0, 2.0).unwrap()),
            Render2dAffineTransform::IDENTITY,
        )],
    );
    assert_eq!(
        query(&moved, [16.0, 10.0], &mut fuel),
        CellCertificate::Inside
    );
    assert_eq!(
        query(&moved, [18.0, 10.0], &mut fuel),
        CellCertificate::Outside
    );
    // One identity preimage plus one patch classification is required even
    // though that patch's source crop has zero area. No refund or guessed alpha.
    let mut short = 16 + 96 - 1;
    assert_eq!(
        query(&identity, [12.0, 12.0], &mut short),
        CellCertificate::Unknown
    );
    assert_eq!(short, 0);
    let mut exact = 16 + 96;
    assert_eq!(
        query(&identity, [12.0, 12.0], &mut exact),
        CellCertificate::Inside
    );
    assert_eq!(exact, 0);
    let empty = Render2dItem::new(
        Render2dPrimitive::Image(
            Render2dImagePrimitive::new(
                Render2dResourceId::new(1).unwrap(),
                Render2dPixelExtent::new(1, 1).unwrap(),
                vec![Render2dImagePatch::new(
                    Render2dImageSourceRect::new(0.0, 0.0, 1.0, 1.0).unwrap(),
                    Render2dRect::new(12.0, 12.0, 0.0, 2.0).unwrap(),
                )],
            )
            .unwrap(),
        ),
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
    );
    let mut short = 16 + 96 - 1;
    assert_eq!(
        query(&empty, [12.0, 12.0], &mut short),
        CellCertificate::Unknown
    );
    assert_eq!(short, 0);
    let mut exact = 16 + 96;
    assert_eq!(
        query(&empty, [12.0, 12.0], &mut exact),
        CellCertificate::Outside
    );
    assert_eq!(exact, 0);
}

#[test]
fn image_source_complete_plane_pays_current_original_preparation() {
    let _ = prepare_fixture();
}

#[test]
#[ignore = "requires Vulkan: transparent image destinations and empty source crop through original F1"]
fn image_source_complete_plane_matches_independent_references_on_vulkan() {
    // The authored image union has the SAME continuous geometry as the
    // independently integrated ring; reuse its 32 sample and four pixel refs,
    // rather than deriving any expected value from a tessellated fill path.
    prepare_fixture().execute_proof(false, false);
}
