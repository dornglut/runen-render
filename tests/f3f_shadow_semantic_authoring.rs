//! F3F public authoring contract: immutable semantic facts, not renderer conformance.
//! GPU proof of neutral coverage and shadow painting belongs to F3F delivery.

use runen_render::composition_2d::{
    Render2dAffineTransform, Render2dBrush, Render2dColorRgba8, Render2dComposition,
    Render2dDropShadow, Render2dEntry, Render2dGroup, Render2dItem, Render2dOpacity,
    Render2dPaintError, Render2dPrimitive, Render2dRect, Render2dShape,
};

#[test]
fn transparent_child_and_group_keep_ordered_signed_shadow_facts() {
    let red = Render2dDropShadow::new(
        12.0,
        -2.0,
        0.0,
        -0.5,
        Render2dColorRgba8::new(255, 0, 0, 128),
    )
    .expect("signed spread");
    let blue = Render2dDropShadow::new(0.0, 1.0, 2.0, 3.0, Render2dColorRgba8::new(0, 0, 255, 128))
        .expect("finite blur");

    let child = Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::rect(
                Render2dRect::new(-10.0, 0.0, 9.0, 2.0).expect("offscreen rectangle"),
            ),
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        },
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
    ));
    let transform = Render2dAffineTransform::new(2.0, 0.0, 0.0, 1.0, 0.0, 0.0).expect("scale");
    let group = Render2dGroup::new(
        vec![child],
        transform,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        vec![red, blue],
    );
    let composition =
        Render2dComposition::new(vec![Render2dEntry::group(group)]).expect("composition");
    let Render2dEntry::Group(retained) = &composition.root_entries()[0] else {
        panic!("group semantic structure must remain intact");
    };

    assert_eq!(retained.shadows(), &[red, blue]);
    assert_eq!(retained.local_to_parent(), transform);
    assert_eq!(retained.opacity(), Render2dOpacity::TRANSPARENT);
    assert_eq!(retained.entries().len(), 1);
    assert!(composition.resource_requirements().is_empty());
    assert_eq!(retained.shadows()[0].spread(), -0.5);
    assert_eq!(retained.shadows()[1].sigma(), 2.0);
}

#[test]
fn negative_sigma_is_not_a_valid_shadow_blur() {
    assert_eq!(
        Render2dDropShadow::new(0.0, 0.0, -0.5, 0.0, Render2dColorRgba8::BLACK),
        Err(Render2dPaintError::NegativeShadowSigma),
    );
}
