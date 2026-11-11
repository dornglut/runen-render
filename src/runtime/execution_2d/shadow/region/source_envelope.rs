//! Original SOURCE reach and immutable glyph-fact capture from F1/F2 facts.
//! Never infer original Bezier/stroke support from a disposable Lyon mesh.
//! This is an enclosing bound, NOT a membership or Gaussian-alpha oracle.

use super::interval::Interval;
use crate::composition_2d::{
    Render2dAffineTransform, Render2dItem, Render2dPathCommand, Render2dPrimitive, Render2dShape,
    Render2dStrokeStyle,
};
use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};

/// One outward-rounded control-hull projection. Source-specific placement
/// precedes these two frames; neither composed rounded affines nor physical
/// tessellation can replace the original controls.
struct ControlEnvelope {
    item: [Interval; 6],
    after: [Interval; 6],
    bounds: Option<[f64; 4]>,
}

impl ControlEnvelope {
    fn new(
        item: &Render2dItem,
        after: Option<Render2dAffineTransform>,
        path: &[usize],
    ) -> Result<Self, Render2dExecutionError> {
        fn coefficients(
            transform: Render2dAffineTransform,
            path: &[usize],
        ) -> Result<[Interval; 6], Render2dExecutionError> {
            let [Some(a), Some(b), Some(c), Some(d), Some(tx), Some(ty)] =
                transform.components().map(Interval::scalar)
            else {
                return Err(precision(path));
            };
            Ok([a, b, c, d, tx, ty])
        }
        Ok(Self {
            item: coefficients(item.local_to_parent(), path)?,
            after: coefficients(after.unwrap_or(Render2dAffineTransform::IDENTITY), path)?,
            bounds: None,
        })
    }

    fn visit(
        &mut self,
        x: Interval,
        y: Interval,
        path: &[usize],
    ) -> Result<(), Render2dExecutionError> {
        fn project(
            [a, b, c, d, tx, ty]: [Interval; 6],
            x: Interval,
            y: Interval,
        ) -> Option<[Interval; 2]> {
            Some([
                a.mul(x)?.add(c.mul(y)?)?.add(tx)?,
                b.mul(x)?.add(d.mul(y)?)?.add(ty)?,
            ])
        }
        let [px, py] = project(self.item, x, y).ok_or_else(|| precision(path))?;
        let [xx, yy] = project(self.after, px, py).ok_or_else(|| precision(path))?;
        self.bounds = Some(match self.bounds {
            Some([x0, y0, x1, y1]) => [x0.min(xx.lo), y0.min(yy.lo), x1.max(xx.hi), y1.max(yy.hi)],
            None => [xx.lo, yy.lo, xx.hi, yy.hi],
        });
        Ok(())
    }
}

fn precision(path: &[usize]) -> Render2dExecutionError {
    Render2dExecutionError::SampleSpace {
        kind: Render2dSampleSpaceError::PrecisionLimit,
        path: Some(path.to_vec()),
        detail: "original F1 geometric reach cannot be certified at finite precision".to_owned(),
    }
}

/// Budget the examination of original immutable curve control points. This
/// count is an upper bound even when contours contain only move/close events.
pub(super) fn control_visits(item: &Render2dItem) -> usize {
    match item.primitive() {
        Render2dPrimitive::Fill { shape, .. } | Render2dPrimitive::Stroke { shape, .. } => shape
            .as_path()
            .map_or(4, |p| p.commands().len().saturating_mul(3)),
        Render2dPrimitive::Image(image) => image.patches().len().saturating_mul(4),
        Render2dPrimitive::ShapedText(_) => 0,
    }
}

fn abs_upper(value: Interval) -> f64 {
    value.lo.abs().max(value.hi.abs())
}

/// Certify only reach from the SAME retained F2 outlines used for glyph
/// painting. Only original outline handles and raw placement survive; no
/// font, field bitmap, resource or control-body copy is retained.
/// Every glyph lookup and command/control visit is paid before examination,
/// including repeated and valid empty glyphs. The hull alone is not a
/// membership proof; later queries consume the shared original controls.
pub(super) fn shaped_source(
    item: &Render2dItem,
    resource: &crate::composition_2d::Render2dShapedTextResource,
    fields: &crate::runtime::execution_2d::field::ResourceFields,
    path: &[usize],
    graph: &mut super::NeutralRegionBuilder,
) -> Result<Option<super::GlyphSources>, Render2dExecutionError> {
    graph.charge_graph_work(512, path)?;
    let Render2dPrimitive::ShapedText(primitive) = item.primitive() else {
        return Err(precision(path));
    };
    let fail = || precision(path);
    let size = Interval::scalar(resource.font_size()).ok_or_else(fail)?;
    let origin_x = Interval::scalar(primitive.origin().x()).ok_or_else(fail)?;
    let origin_y = Interval::scalar(primitive.origin().y()).ok_or_else(fail)?;
    let mut envelope = ControlEnvelope::new(item, None, path)?;
    let mut instances = Vec::new();
    let mut outline_bytes = 0_usize;
    let lookup_work = fields.outline_lookup_work();
    for glyph in resource.glyphs() {
        graph.charge_graph_work(lookup_work, path)?;
        let Some(field) = fields
            .admitted_outline(glyph.id())
            .map_err(|()| precision(path))?
        else {
            continue;
        };
        let commands = field.outline_commands();
        let work = commands
            .len()
            .checked_mul(3 * (super::SOURCE_REACH_CONTROL_WORK + 512))
            .ok_or_else(|| super::resource(path, "original glyph reach work overflow"))?;
        graph.charge_graph_work(work, path)?;
        // Enclose the original sum, rather than treating rounded placement
        // or a rounded fontsize/affine product as exact source geometry.
        let dx = origin_x
            .add(Interval::scalar(glyph.x()).ok_or_else(fail)?)
            .ok_or_else(fail)?;
        let dy = origin_y
            .add(Interval::scalar(glyph.y()).ok_or_else(fail)?)
            .ok_or_else(fail)?;
        let mut visit = |point: crate::composition_2d::Render2dPoint| {
            let x = Interval::scalar(point.x())
                .and_then(|v| v.mul(size))
                .and_then(|v| v.add(dx))
                .ok_or_else(fail)?;
            let y = Interval::scalar(point.y())
                .and_then(|v| v.mul(size))
                .and_then(|v| v.add(dy))
                .ok_or_else(fail)?;
            envelope.visit(x, y, path)
        };
        for command in commands {
            match command.map_err(|()| precision(path))? {
                Render2dPathCommand::MoveTo(to) | Render2dPathCommand::LineTo(to) => visit(to)?,
                Render2dPathCommand::QuadraticTo { control, to } => {
                    visit(control)?;
                    visit(to)?;
                }
                Render2dPathCommand::CubicTo {
                    control1,
                    control2,
                    to,
                } => {
                    visit(control1)?;
                    visit(control2)?;
                    visit(to)?;
                }
                Render2dPathCommand::Close => {}
            }
        }
        let grows = instances.len() == instances.capacity();
        let copies = if grows { instances.len() } else { 0 };
        // Pay handle reads/writes, geometric relocation, Arc clone and host
        // bookkeeping BEFORE allocation or retaining the shared body.
        let work = copies
            .checked_add(1)
            .and_then(|copies| copies.checked_mul(2 * size_of::<super::GlyphSource>()))
            .and_then(|work| work.checked_add(192))
            .ok_or_else(|| super::resource(path, "original glyph metadata work overflow"))?;
        graph.charge_graph_work(work, path)?;
        let body = field
            .outline_storage_bytes()
            .ok_or_else(|| super::resource(path, "original glyph outline storage overflow"))?;
        let next_bytes = outline_bytes
            .checked_add(body)
            .ok_or_else(|| super::resource(path, "original glyph retained storage overflow"))?;
        graph.host.reserve(body, path)?;
        if grows {
            let capacity = instances
                .capacity()
                .checked_mul(2)
                .map(|capacity| capacity.max(4).min(resource.glyphs().len()))
                .ok_or_else(|| {
                    super::resource(path, "original glyph instance capacity overflow")
                })?;
            graph.host.reserve_vector(&mut instances, capacity, path)?;
        }
        instances.push(super::GlyphSource {
            outline: field.original_outline(),
            position: [glyph.x(), glyph.y()],
        });
        outline_bytes = next_bytes;
    }
    Ok(envelope.bounds.map(|reach| super::GlyphSources {
        instances,
        font_size: resource.font_size(),
        reach,
        outline_bytes,
    }))
}

/// The Bézier convex-hull property encloses ALL points of each segment by
/// its authored controls, independent of tessellation quality. A rounded
/// rectangle or ellipse fits inside its rectangle. Caps/joins of a centered
/// stroke extend at most radius * max(2,miter_limit) from the centerline,
/// conservatively including square caps and miter joins.
pub(super) fn bounds(
    item: &Render2dItem,
    after: Option<Render2dAffineTransform>,
    path: &[usize],
) -> Result<Option<[f64; 4]>, Render2dExecutionError> {
    let (shape, stroke): (Option<&Render2dShape>, Option<Render2dStrokeStyle>) =
        match item.primitive() {
            Render2dPrimitive::Fill { shape, .. } => (Some(shape), None),
            Render2dPrimitive::Stroke { shape, style, .. } => (Some(shape), Some(*style)),
            Render2dPrimitive::Image(_) => (None, None),
            Render2dPrimitive::ShapedText(_) => return Ok(None),
        };
    let mut envelope = ControlEnvelope::new(item, after, path)?;
    let [a, b, c, d, _, _] = envelope.item;
    let [aa, bb, cc, dd, _, _] = envelope.after;
    let fail = || precision(path);
    let mut visit = |x: Interval, y: Interval| -> Result<(), Render2dExecutionError> {
        envelope.visit(x, y, path)
    };
    if let Some(rect) = shape.and_then(|shape| {
        shape
            .as_rect()
            .or_else(|| shape.as_rounded_rect().map(|(rect, _)| rect))
            .or_else(|| shape.as_ellipse())
    }) {
        if rect.is_empty() {
            return Ok(None);
        }
        let x0 = Interval::scalar(rect.x()).ok_or_else(fail)?;
        let y0 = Interval::scalar(rect.y()).ok_or_else(fail)?;
        let x1 = x0
            .add(Interval::scalar(rect.width()).ok_or_else(fail)?)
            .ok_or_else(fail)?;
        let y1 = y0
            .add(Interval::scalar(rect.height()).ok_or_else(fail)?)
            .ok_or_else(fail)?;
        for (x, y) in [(x0, y0), (x0, y1), (x1, y0), (x1, y1)] {
            visit(x, y)?;
        }
    } else if let Some(authored) = shape.and_then(Render2dShape::as_path) {
        let mut visit_point = |point: crate::composition_2d::Render2dPoint| {
            visit(
                Interval::scalar(point.x()).ok_or_else(fail)?,
                Interval::scalar(point.y()).ok_or_else(fail)?,
            )
        };
        for command in authored.commands() {
            match *command {
                Render2dPathCommand::MoveTo(to) | Render2dPathCommand::LineTo(to) => {
                    visit_point(to)?
                }
                Render2dPathCommand::QuadraticTo { control, to } => {
                    visit_point(control)?;
                    visit_point(to)?;
                }
                Render2dPathCommand::CubicTo {
                    control1,
                    control2,
                    to,
                } => {
                    visit_point(control1)?;
                    visit_point(control2)?;
                    visit_point(to)?;
                }
                Render2dPathCommand::Close => {}
            }
        }
    } else if let Render2dPrimitive::Image(image) = item.primitive() {
        // F1 image support is each nonempty DESTINATION patch, regardless
        // of source-crop area, pixel alpha, brush alpha or item opacity.
        // The original affine must act before any ancestor affine.
        for patch in image.patches() {
            let rect = patch.destination();
            if rect.is_empty() {
                continue;
            }
            let x0 = Interval::scalar(rect.x()).ok_or_else(fail)?;
            let y0 = Interval::scalar(rect.y()).ok_or_else(fail)?;
            let x1 = x0
                .add(Interval::scalar(rect.width()).ok_or_else(fail)?)
                .ok_or_else(fail)?;
            let y1 = y0
                .add(Interval::scalar(rect.height()).ok_or_else(fail)?)
                .ok_or_else(fail)?;
            for (x, y) in [(x0, y0), (x0, y1), (x1, y0), (x1, y1)] {
                visit(x, y)?;
            }
        }
    }
    let Some([x0, y0, x1, y1]) = envelope.bounds else {
        return Ok(None);
    };
    let (mut x0, mut y0, mut x1, mut y1) = (x0, y0, x1, y1);
    if let Some(style) = stroke {
        let half_width = Interval::scalar(style.width())
            .and_then(|v| v.mul(Interval::scalar(0.5)?))
            .ok_or_else(fail)?;
        let radius = half_width
            .mul(Interval::scalar(style.miter_limit().max(2.0)).ok_or_else(fail)?)
            .ok_or_else(fail)?;
        let ex = aa.mul(a).and_then(|v| v.add(cc.mul(b)?)).ok_or_else(fail)?;
        let ey = aa.mul(c).and_then(|v| v.add(cc.mul(d)?)).ok_or_else(fail)?;
        let fx = bb.mul(a).and_then(|v| v.add(dd.mul(b)?)).ok_or_else(fail)?;
        let fy = bb.mul(c).and_then(|v| v.add(dd.mul(d)?)).ok_or_else(fail)?;
        let margin_x = Interval::scalar(abs_upper(ex))
            .and_then(|v| v.add(Interval::scalar(abs_upper(ey))?))
            .and_then(|v| v.mul(radius))
            .ok_or_else(fail)?
            .hi;
        let margin_y = Interval::scalar(abs_upper(fx))
            .and_then(|v| v.add(Interval::scalar(abs_upper(fy))?))
            .and_then(|v| v.mul(radius))
            .ok_or_else(fail)?
            .hi;
        x0 = Interval::scalar(x0)
            .and_then(|v| v.sub(Interval::scalar(margin_x)?))
            .ok_or_else(fail)?
            .lo;
        x1 = Interval::scalar(x1)
            .and_then(|v| v.add(Interval::scalar(margin_x)?))
            .ok_or_else(fail)?
            .hi;
        y0 = Interval::scalar(y0)
            .and_then(|v| v.sub(Interval::scalar(margin_y)?))
            .ok_or_else(fail)?
            .lo;
        y1 = Interval::scalar(y1)
            .and_then(|v| v.add(Interval::scalar(margin_y)?))
            .ok_or_else(fail)?
            .hi;
    }
    Ok(Some([x0, y0, x1, y1]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition_2d::{
        Render2dImagePatch, Render2dImagePrimitive, Render2dImageSourceRect, Render2dOpacity,
        Render2dPixelExtent, Render2dRect, Render2dResourceId,
    };

    #[test]
    fn original_image_destination_patches_enclose_transformed_transparent_support() {
        let image = Render2dImagePrimitive::new(
            Render2dResourceId::new(7).unwrap(),
            Render2dPixelExtent::new(4, 4).unwrap(),
            vec![
                Render2dImagePatch::new(
                    Render2dImageSourceRect::new(0.0, 0.0, 0.0, 4.0).unwrap(),
                    Render2dRect::new(-8.0, -3.0, 4.0, 2.0).unwrap(),
                ),
                Render2dImagePatch::new(
                    Render2dImageSourceRect::new(1.0, 1.0, 2.0, 2.0).unwrap(),
                    Render2dRect::new(2.0, 4.0, 3.0, 1.0).unwrap(),
                ),
            ],
        )
        .unwrap();
        let item_transform = Render2dAffineTransform::new(1.0, 0.5, 0.25, 1.0, 3.0, -2.0).unwrap();
        let parent_transform =
            Render2dAffineTransform::new(0.5, 0.0, -1.0, 1.0, 7.0, -3.0).unwrap();
        let item = Render2dItem::new(
            Render2dPrimitive::Image(image),
            item_transform,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        assert_eq!(control_visits(&item), 8);
        let extent = bounds(&item, Some(parent_transform), &[0])
            .unwrap()
            .unwrap();
        for [x, y] in [
            [-8.0, -3.0],
            [-4.0, -3.0],
            [-8.0, -1.0],
            [-4.0, -1.0],
            [2.0, 4.0],
            [5.0, 4.0],
            [2.0, 5.0],
            [5.0, 5.0],
        ] {
            let [a, b, c, d, tx, ty] = item_transform.components();
            let px = a.mul_add(x, c.mul_add(y, tx));
            let py = b.mul_add(x, d.mul_add(y, ty));
            let [a, b, c, d, tx, ty] = parent_transform.components();
            let xx = a.mul_add(px, c.mul_add(py, tx));
            let yy = b.mul_add(px, d.mul_add(py, ty));
            assert!(extent[0] <= xx && xx <= extent[2]);
            assert!(extent[1] <= yy && yy <= extent[3]);
        }
    }

    #[test]
    fn destination_empty_image_has_no_authored_support() {
        let image = Render2dImagePrimitive::new(
            Render2dResourceId::new(8).unwrap(),
            Render2dPixelExtent::new(1, 1).unwrap(),
            vec![Render2dImagePatch::new(
                Render2dImageSourceRect::new(0.0, 0.0, 1.0, 1.0).unwrap(),
                Render2dRect::new(2.0, 3.0, 0.0, 1.0).unwrap(),
            )],
        )
        .unwrap();
        let item = Render2dItem::new(
            Render2dPrimitive::Image(image),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        assert_eq!(control_visits(&item), 4);
        assert_eq!(bounds(&item, None, &[0]).unwrap(), None);
    }
}
