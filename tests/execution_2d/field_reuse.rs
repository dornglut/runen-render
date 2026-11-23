use super::*;

#[test]
fn changed_text_reuse_preserves_output_identity_and_failed_transaction() {
    let Some(context) = f2_context() else { return };
    let mut retained = Render2dExecutor::new();
    let seed = Render2dResourceId::new(980).unwrap();
    let composition = shaped_composition(seed, [8., 32.], Render2dColorRgba8::WHITE);
    execute(
        &context,
        &mut retained,
        &composition,
        &bindings(seed, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.)),
        "seed prior fields",
    );

    let next = Render2dResourceId::new(981).unwrap();
    let invalid = Render2dResourceId::new(982).unwrap();
    let failed = Render2dComposition::new(vec![
        shaped_entry(next, [8., 32.], Render2dColorRgba8::WHITE),
        shaped_entry(invalid, [36., 32.], Render2dColorRgba8::WHITE),
    ])
    .unwrap();
    let attempted = Render2dResourceBindings::new(vec![
        Render2dResourceBinding::new(
            next,
            Render2dResourceValue::ShapedText(shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.)),
        ),
        Render2dResourceBinding::new(
            invalid,
            Render2dResourceValue::ShapedText(shaped_resource(OUTLINE_FONT, false, u32::MAX, 24.)),
        ),
    ])
    .unwrap();
    let (_, target) = target("failed reuse transaction");
    assert!(
        matches!(retained.prepare(&context, &failed, &attempted, &target),
        Err(Render2dExecutionError::ShapedText(Render2dShapedTextError::InvalidOutline {resource_id,..})) if resource_id==invalid)
    );

    // The failed transaction must not observe next's attempted size. The retry
    // changes size within the same field tier and uses a new placement/color.
    let composition =
        shaped_composition(next, [12., 34.], Render2dColorRgba8::new(25, 180, 220, 190));
    let current = bindings(next, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 20.));
    let reused = execute(
        &context,
        &mut retained,
        &composition,
        &current,
        "changed identity reuse",
    );
    let mut fresh = Render2dExecutor::new();
    let expected = execute(
        &context,
        &mut fresh,
        &composition,
        &current,
        "independent fresh output",
    );
    assert_eq!(reused.as_bytes(), expected.as_bytes());
    let rebound = bindings(next, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 21.));
    assert!(
        matches!(retained.prepare(&context, &composition, &rebound, &target),
        Err(Render2dExecutionError::ResourceIdentityRebound{resource_id}) if resource_id==next)
    );

    let fresh_context = f2_context().expect("second required conformance adapter");
    let reconstructed = execute(
        &fresh_context,
        &mut retained,
        &composition,
        &current,
        "CPU fields across devices",
    );
    assert_eq!(reconstructed.as_bytes(), expected.as_bytes());
    retained.discard_cache();
    let discarded = execute(
        &context,
        &mut retained,
        &composition,
        &current,
        "reconstruct discarded fields",
    );
    assert_eq!(discarded.as_bytes(), expected.as_bytes());
    assert!(
        matches!(retained.prepare(&context, &composition, &rebound, &target),
        Err(Render2dExecutionError::ResourceIdentityRebound{resource_id}) if resource_id==next)
    );
}
