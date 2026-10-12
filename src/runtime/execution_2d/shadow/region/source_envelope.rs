//! Conservative SOURCE reach from original, borrowed F1 authoring facts.
//! Never infer original Bezier/stroke support from a disposable Lyon mesh.
//! This is an enclosing bound, NOT a membership or Gaussian-alpha oracle.

use super::interval::Interval;
use crate::composition_2d::{
    Render2dAffineTransform, Render2dItem, Render2dPathCommand, Render2dPrimitive, Render2dShape,
    Render2dStrokeStyle,
};
use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};

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
    let shape = match item.primitive() {
        Render2dPrimitive::Fill { shape, .. }
        | Render2dPrimitive::Stroke { shape, .. } => shape,
        _ => return 0,
    };
    shape
        .as_path()
        .map_or(4, |p| p.commands().len().saturating_mul(3))
}

fn abs_upper(value: Interval) -> f64 {
    value.lo.abs().max(value.hi.abs())
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
    let (shape, stroke): (&Render2dShape, Option<Render2dStrokeStyle>) = match item.primitive() {
        Render2dPrimitive::Fill { shape, .. } => (shape, None),
        Render2dPrimitive::Stroke { shape, style, .. } => (shape, Some(*style)),
        _ => return Ok(None),
    };
    let [a, b, c, d, tx, ty] = item.local_to_parent().components().map(Interval::scalar);
    let (a, b, c, d, tx, ty) = (
        a.ok_or_else(|| precision(path))?,
        b.ok_or_else(|| precision(path))?,
        c.ok_or_else(|| precision(path))?,
        d.ok_or_else(|| precision(path))?,
        tx.ok_or_else(|| precision(path))?,
        ty.ok_or_else(|| precision(path))?,
    );
    let (aa, bb, cc, dd, ttx, tty) = match after {
        Some(next) => {
            let [aa, bb, cc, dd, ttx, tty] = next.components().map(Interval::scalar);
            (
                aa.ok_or_else(|| precision(path))?,
                bb.ok_or_else(|| precision(path))?,
                cc.ok_or_else(|| precision(path))?,
                dd.ok_or_else(|| precision(path))?,
                ttx.ok_or_else(|| precision(path))?,
                tty.ok_or_else(|| precision(path))?,
            )
        }
        None => (
            Interval::scalar(1.0).expect("finite identity"),
            Interval::scalar(0.0).expect("finite identity"),
            Interval::scalar(0.0).expect("finite identity"),
            Interval::scalar(1.0).expect("finite identity"),
            Interval::scalar(0.0).expect("finite identity"),
            Interval::scalar(0.0).expect("finite identity"),
        ),
    };
    let fail = || precision(path);
    let mut coverage: Option<[f64; 4]> = None;
    let mut visit = |x: Interval, y: Interval| -> Result<(), Render2dExecutionError> {
        let px = a.mul(x).and_then(|v| v.add(c.mul(y)?)).and_then(|v| v.add(tx)).ok_or_else(fail)?;
        let py = b.mul(x).and_then(|v| v.add(d.mul(y)?)).and_then(|v| v.add(ty)).ok_or_else(fail)?;
        let xx = aa.mul(px).and_then(|v| v.add(cc.mul(py)?)).and_then(|v| v.add(ttx)).ok_or_else(fail)?;
        let yy = bb.mul(px).and_then(|v| v.add(dd.mul(py)?)).and_then(|v| v.add(tty)).ok_or_else(fail)?;
        coverage = Some(match coverage {
            Some([x0, y0, x1, y1]) => [x0.min(xx.lo), y0.min(yy.lo), x1.max(xx.hi), y1.max(yy.hi)],
            None => [xx.lo, yy.lo, xx.hi, yy.hi],
        });
        Ok(())
    };
    if let Some(rect) = shape
        .as_rect()
        .or_else(|| shape.as_rounded_rect().map(|(rect, _)| rect))
        .or_else(|| shape.as_ellipse())
    {
        if rect.is_empty() {
            return Ok(None);
        }
        let x0 = Interval::scalar(rect.x()).ok_or_else(fail)?;
        let y0 = Interval::scalar(rect.y()).ok_or_else(fail)?;
        let x1 = x0.add(Interval::scalar(rect.width()).ok_or_else(fail)?).ok_or_else(fail)?;
        let y1 = y0.add(Interval::scalar(rect.height()).ok_or_else(fail)?).ok_or_else(fail)?;
        for (x, y) in [(x0, y0), (x0, y1), (x1, y0), (x1, y1)] {
            visit(x, y)?;
        }
    } else if let Some(authored) = shape.as_path() {
        let mut visit_point = |point: crate::composition_2d::Render2dPoint| {
            visit(
                Interval::scalar(point.x()).ok_or_else(fail)?,
                Interval::scalar(point.y()).ok_or_else(fail)?,
            )
        };
        for command in authored.commands() {
            match *command {
                Render2dPathCommand::MoveTo(to) | Render2dPathCommand::LineTo(to) => visit_point(to)?,
                Render2dPathCommand::QuadraticTo { control, to } => {
                    visit_point(control)?;
                    visit_point(to)?;
                }
                Render2dPathCommand::CubicTo { control1, control2, to } => {
                    visit_point(control1)?;
                    visit_point(control2)?;
                    visit_point(to)?;
                }
                Render2dPathCommand::Close => {}
            }
        }
    }
    let Some([x0, y0, x1, y1]) = coverage else { return Ok(None) };
    let (mut x0, mut y0, mut x1, mut y1) = (x0, y0, x1, y1);
    if let Some(style) = stroke {
        let half_width = Interval::scalar(style.width()).and_then(|v| v.mul(Interval::scalar(0.5)?)).ok_or_else(fail)?;
        let radius = half_width.mul(Interval::scalar(style.miter_limit().max(2.0)).ok_or_else(fail)?).ok_or_else(fail)?;
        let ex = aa.mul(a).and_then(|v| v.add(cc.mul(b)?)).ok_or_else(fail)?;
        let ey = aa.mul(c).and_then(|v| v.add(cc.mul(d)?)).ok_or_else(fail)?;
        let fx = bb.mul(a).and_then(|v| v.add(dd.mul(b)?)).ok_or_else(fail)?;
        let fy = bb.mul(c).and_then(|v| v.add(dd.mul(d)?)).ok_or_else(fail)?;
        let margin_x = Interval::scalar(abs_upper(ex)).and_then(|v| v.add(Interval::scalar(abs_upper(ey))?))
            .and_then(|v| v.mul(radius)).ok_or_else(fail)?.hi;
        let margin_y = Interval::scalar(abs_upper(fx)).and_then(|v| v.add(Interval::scalar(abs_upper(fy))?))
            .and_then(|v| v.mul(radius)).ok_or_else(fail)?.hi;
        x0 = Interval::scalar(x0).and_then(|v| v.sub(Interval::scalar(margin_x)?)).ok_or_else(fail)?.lo;
        x1 = Interval::scalar(x1).and_then(|v| v.add(Interval::scalar(margin_x)?)).ok_or_else(fail)?.hi;
        y0 = Interval::scalar(y0).and_then(|v| v.sub(Interval::scalar(margin_y)?)).ok_or_else(fail)?.lo;
        y1 = Interval::scalar(y1).and_then(|v| v.add(Interval::scalar(margin_y)?)).ok_or_else(fail)?.hi;
    }
    Ok(Some([x0, y0, x1, y1]))
}
