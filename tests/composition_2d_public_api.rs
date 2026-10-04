use runen_render::composition_2d::{
    Render2dAffineTransform, Render2dColorRgba8, Render2dComposition, Render2dEntry,
    Render2dFontBinding, Render2dGlyph, Render2dImagePatch, Render2dImagePrimitive,
    Render2dImageResource, Render2dImageSourceRect, Render2dItem, Render2dOpacity,
    Render2dPixelExtent, Render2dPoint, Render2dPrimitive, Render2dRect, Render2dResourceBinding,
    Render2dResourceBindings, Render2dResourceId, Render2dResourceValue,
    Render2dShapedTextPrimitive, Render2dShapedTextResource,
};

#[test]
fn source_neutral_2d_semantics_are_publicly_composable() {
    let image_id = Render2dResourceId::new(1).expect("image id");
    let text_id = Render2dResourceId::new(2).expect("text id");

    let image_extent = Render2dPixelExtent::new(2, 2).expect("image extent");
    let image_patch = Render2dImagePatch::new(
        Render2dImageSourceRect::new(0.25, 0.5, 1.5, 1.0).expect("source"),
        Render2dRect::new(0.0, 0.0, 32.0, 16.0).expect("destination"),
    );
    let image = Render2dImagePrimitive::new(image_id, image_extent, vec![image_patch])
        .expect("semantic image");

    let text = Render2dShapedTextPrimitive::new(
        text_id,
        Render2dPoint::new(4.0, 8.0).expect("text origin"),
        Render2dColorRgba8::WHITE,
    );

    let composition = Render2dComposition::new(vec![
        Render2dEntry::item(Render2dItem::new(
            Render2dPrimitive::Image(image),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
        )),
        Render2dEntry::item(Render2dItem::new(
            Render2dPrimitive::ShapedText(text),
            Render2dAffineTransform::translation(1.0, 2.0).expect("translation"),
            Vec::new(),
            Render2dOpacity::OPAQUE,
        )),
    ])
    .expect("composition");

    let image_resource =
        Render2dImageResource::new(image_extent, vec![255; 16]).expect("image resource");
    let font = Render2dFontBinding::new(vec![0, 1, 2, 3], 0, Vec::<i16>::new(), false, None)
        .expect("font facts");
    let glyph = Render2dGlyph::new(42, 0.0, 0.0, 9.0).expect("glyph");
    let shaped = Render2dShapedTextResource::new(font, 16.0, vec![glyph]).expect("shaped resource");

    let bindings = Render2dResourceBindings::new(vec![
        Render2dResourceBinding::new(
            image_id,
            Render2dResourceValue::ImageRgba8Srgb(image_resource),
        ),
        Render2dResourceBinding::new(text_id, Render2dResourceValue::ShapedText(shaped)),
    ])
    .expect("bindings");

    composition
        .validate_bindings(&bindings)
        .expect("exact resources satisfy composition");

    assert_eq!(composition.root_entries().len(), 2);
    assert_eq!(composition.resource_requirements().len(), 2);
}
