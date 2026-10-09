//! Physical F1 painter-event inspection, derived resources and deterministic preflight bounds.
use super::*;

/// One derived, preflight-only solid-vector snapshot. Its fields are not
/// authored semantic state; the immutable F1 plan remains the only authority.
pub(super) struct PreparedGlyph {
    pub(super) occurrence: super::super::GlyphOccurrence,
    pub(super) placement: super::super::GlyphPlacement,
    pub(super) bounds: [u32; 4],
}

pub(super) enum PreparedItem {
    Vector(geometry::VectorMesh),
    Image(Vec<image_semantics::ImagePatchWork>),
    Text(Vec<PreparedGlyph>),
}

impl PreparedItem {
    /// Physical pixel extent, derived from the same validated F1 item.
    pub(super) fn bounds(&self) -> [u32; 4] {
        match self {
            Self::Vector(mesh) => {
                let [left, top, width, height] = mesh.bounds;
                [
                    left,
                    top,
                    left.saturating_add(width),
                    top.saturating_add(height),
                ]
            }
            Self::Image(patches) => {
                let mut bounds = [u32::MAX, u32::MAX, 0, 0];
                for patch in patches {
                    bounds[0] = bounds[0].min(patch.bounds[0]);
                    bounds[1] = bounds[1].min(patch.bounds[1]);
                    bounds[2] = bounds[2].max(patch.bounds[2]);
                    bounds[3] = bounds[3].max(patch.bounds[3]);
                }
                bounds
            }
            Self::Text(glyphs) => {
                let mut bounds = [u32::MAX, u32::MAX, 0, 0];
                for glyph in glyphs {
                    let b = glyph.bounds;
                    bounds[0] = bounds[0].min(b[0]);
                    bounds[1] = bounds[1].min(b[1]);
                    bounds[2] = bounds[2].max(b[2]);
                    bounds[3] = bounds[3].max(b[3]);
                }
                bounds
            }
        }
    }
}

/// Original F2 pixels cover an MSDF glyph when their centers lie inside
/// its physical quad; tile-local 4x4 replicas must preserve that membership.
fn text_pixel_bounds(placement: super::super::GlyphPlacement) -> [u32; 4] {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "finite glyph placement is clipped to admitted u32 target dimensions"
    )]
    let first = |edge: f64| (edge - 0.5).ceil().max(0.0) as u32;
    [
        first(placement.left),
        first(placement.top),
        first(placement.right),
        first(placement.bottom),
    ]
}

pub(super) fn text_quad(
    glyph: &PreparedGlyph,
    origin: [u32; 2],
    end: [u32; 2],
    dimension: u32,
) -> Vec<f32> {
    let b = glyph.bounds;
    let [left, top, right, bottom] = [
        b[0].max(origin[0]),
        b[1].max(origin[1]),
        b[2].min(end[0]),
        b[3].min(end[1]),
    ];
    if left >= right || top >= bottom {
        return Vec::new();
    }
    let p = glyph.placement;
    let color = linear_color(glyph.occurrence.color);
    let mut result = Vec::with_capacity(6 * FLOATS_PER_VERTEX);
    for [x, y] in [
        [left, top],
        [right, top],
        [left, bottom],
        [left, bottom],
        [right, top],
        [right, bottom],
    ] {
        let sx = f64::from(x - origin[0]) * f64::from(SAMPLES);
        let sy = f64::from(y - origin[1]) * f64::from(SAMPLES);
        result.extend([
            physical_x_to_ndc(sx, dimension),
            physical_y_to_ndc(sy, dimension),
            f32_from_f64((f64::from(x) - p.x0) / p.width),
            f32_from_f64((f64::from(y) - p.y0) / p.height),
            color[0],
            color[1],
            color[2],
            color[3],
        ]);
    }
    result
}

/// An indexed physical realization only, aligned with the ONE F1 painter plan.
pub(super) struct Inspected {
    pub(super) items: Vec<Option<PreparedItem>>,
    pub(super) group_bounds: Vec<Option<[u32; 4]>>,
    pub(super) bounds: [u32; 4],
    pub(super) peak_group_depth: usize,
}

/// Pure, conservative physical bounds; only semantic clips decide exact
/// correlated-sample coverage after this non-authoritative culling step.
pub(super) fn intersection(a: [u32; 4], b: [u32; 4]) -> Option<[u32; 4]> {
    let bounds = [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ];
    (bounds[0] < bounds[2] && bounds[1] < bounds[3]).then_some(bounds)
}

fn union(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

pub(super) fn charge_vector_geometry(
    retained: &mut usize,
    vertices: usize,
) -> Result<(), Render2dExecutionError> {
    let next = retained
        .checked_add(vertices)
        .ok_or_else(|| failure("aggregate vector geometry size overflow"))?;
    if next > MAX_RETAINED_VECTOR_VERTICES {
        return Err(failure(
            "aggregate retained vector geometry exceeds bounded admission",
        ));
    }
    *retained = next;
    Ok(())
}

pub(super) fn admit_tile_work(
    plan_events: usize,
    items: &[Option<PreparedItem>],
    tiles: u64,
) -> Result<(), Render2dExecutionError> {
    let mut units = u64::try_from(plan_events).map_err(|_| failure("tile event count overflow"))?;
    for item in items.iter().flatten() {
        let count = match item {
            PreparedItem::Vector(mesh) => mesh.triangles.len(),
            PreparedItem::Image(patches) => patches.len(),
            PreparedItem::Text(glyphs) => glyphs.len(),
        };
        units = units
            .checked_add(u64::try_from(count).map_err(|_| failure("tile work count overflow"))?)
            .ok_or_else(|| failure("tile work count overflow"))?;
    }
    let total = units
        .checked_mul(tiles)
        .ok_or_else(|| failure("aggregate tile work count overflow"))?;
    if total > MAX_TILE_WORK_UNITS {
        return Err(failure(
            "aggregate tile preparation work exceeds bounded admission",
        ));
    }
    Ok(())
}

pub(super) fn inspect(
    plan: &scene::Plan<'_>,
    target: &AdmittedTarget,
    bindings: &Render2dResourceBindings,
    glyphs_by_event: &BTreeMap<usize, Vec<super::super::GlyphOccurrence>>,
) -> Result<Inspected, Render2dExecutionError> {
    let mut items = Vec::with_capacity(plan.events.len());
    let mut group_bounds = Vec::with_capacity(plan.events.len());
    let mut active_groups = Vec::new();
    let mut bounds = [u32::MAX, u32::MAX, 0, 0];
    let mut depth = 0usize;
    let mut peak = 0usize;
    let mut retained_vector_vertices = 0usize;
    for (event_index, event) in plan.events.iter().enumerate() {
        match event {
            scene::Event::BeginGroup { group, path, .. } => {
                if !group.shadows().is_empty() {
                    let kind = Render2dUnsupportedContent::Shadows {
                        root_index: path[0],
                    };
                    return Err(super::super::super::super::unsupported_at(path, kind));
                }
                depth += 1;
                peak = peak.max(depth);
                active_groups.push(items.len());
                items.push(None);
                group_bounds.push(None);
            }
            scene::Event::EndGroup => {
                depth -= 1;
                active_groups.pop().expect("balanced accepted group plan");
                items.push(None);
                group_bounds.push(None);
            }
            scene::Event::Item {
                item,
                path,
                to_root,
                ..
            } => {
                let [a, b, c, d, tx, ty] = to_root.coefficients();
                let transform = Render2dAffineTransform::new(a, b, c, d, tx, ty)
                    .map_err(|_| precision_failure(path, "unrepresentable cumulative transform"))?;
                let derived = Render2dItem::new(
                    item.primitive().clone(),
                    transform,
                    Vec::new(),
                    item.opacity(),
                );
                let realized = match item.primitive() {
                    Render2dPrimitive::Fill { .. } | Render2dPrimitive::Stroke { .. } => {
                        geometry::realize(
                            &derived,
                            path[0],
                            target.raster_scale(),
                            target.canvas(),
                            target.max_buffer_bytes(),
                        )?
                        .map(PreparedItem::Vector)
                    }
                    Render2dPrimitive::Image(image) => {
                        if !target.image_format {
                            return Err(image_semantics::failure(
                                path[0],
                                crate::execution_2d::Render2dImageError::FormatUnsupported,
                            ));
                        }
                        let value = bindings.get(image.resource_id()).expect(
                            "the composition already validated immutable image resource bindings",
                        );
                        let Render2dResourceValue::ImageRgba8Srgb(source) = value else {
                            unreachable!("validated source-neutral F1 image binding kind");
                        };
                        let patches = image_semantics::realize(
                            &derived,
                            image,
                            source,
                            path[0],
                            target.raster_scale(),
                            target.canvas(),
                            target.max_texture_dimension_2d(),
                        )?;
                        (!patches.is_empty()).then_some(PreparedItem::Image(patches))
                    }
                    Render2dPrimitive::ShapedText(_) => {
                        let mut glyphs = Vec::new();
                        for glyph in glyphs_by_event
                            .get(&event_index)
                            .into_iter()
                            .flat_map(|values| values.iter())
                        {
                            let Some(placement) = super::super::glyph_placement(target, glyph)?
                            else {
                                continue;
                            };
                            let bounds = text_pixel_bounds(placement);
                            if bounds[0] < bounds[2] && bounds[1] < bounds[3] {
                                glyphs.push(PreparedGlyph {
                                    occurrence: glyph.clone(),
                                    placement,
                                    bounds,
                                });
                            }
                        }
                        (!glyphs.is_empty()).then_some(PreparedItem::Text(glyphs))
                    }
                };
                if let Some(PreparedItem::Vector(mesh)) = realized.as_ref() {
                    charge_vector_geometry(&mut retained_vector_vertices, mesh.triangles.len())?;
                }
                if let Some(ref content) = realized {
                    let b = content.bounds();
                    bounds = union(bounds, b);
                    for group_index in &active_groups {
                        let entry = &mut group_bounds[*group_index];
                        *entry = Some(entry.map_or(b, |previous| union(previous, b)));
                    }
                }
                items.push(realized);
                group_bounds.push(None);
            }
        }
    }
    debug_assert_eq!(depth, 0);
    debug_assert!(active_groups.is_empty());
    Ok(Inspected {
        items,
        group_bounds,
        bounds,
        peak_group_depth: peak,
    })
}
