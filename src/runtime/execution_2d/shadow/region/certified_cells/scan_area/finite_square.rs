//! Integration of the complete source-derived finite-square cover.
//! The original region remains membership authority; these are candidate events.
use super::*;

impl ScanPlan {
    /// Compositional finite-support qualification. Candidate curves
    /// cover geometry; the original prepared source remains membership owner.
    pub(super) fn finite_square_effect(
        &mut self,
        region: &NeutralRegion<'_>,
        shadow: Render2dDropShadow,
        affine: Affine,
        prepared: &PreparedBoundaries,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        let boundary = prepared.get(region.id, fuel).ok_or({
            if *fuel == 0 {
                AreaQueryError::Resource
            } else {
                AreaQueryError::Unsupported
            }
        })?;
        charge(fuel, 128).ok_or(AreaQueryError::Resource)?;
        let halo = exact_mul(3.0, shadow.sigma())
            .and_then(Interval::scalar)
            .or_else(|| Interval::scalar(3.0)?.mul(Interval::scalar(shadow.sigma())?))
            .ok_or(AreaQueryError::Precision)?;
        let inventory =
            finite_square_geometry::Inventory::prepare(boundary, -shadow.spread(), halo, fuel)?;
        let shift = Render2dAffineTransform::translation(shadow.offset_x(), shadow.offset_y())
            .map_err(|_| AreaQueryError::Precision)?;
        let frame = self.frame().ok_or(AreaQueryError::Resource)?;
        let shifted = affine.after(shift, frame, fuel)?;
        for line in inventory.lines {
            self.interval_line(line, shifted, fuel)?;
        }
        for circle in inventory.circles {
            if circle.center.iter().any(|v| v.lo != v.hi) {
                return Err(AreaQueryError::Precision);
            }
            self.finite_circle(circle, shifted, fuel)?;
        }
        Ok(())
    }

    fn finite_circle(
        &mut self,
        circle: finite_square_geometry::Circle,
        affine: Affine,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        if let (Some([a, 0.0, c, d, _, _]), Some([sx, sy])) = (affine.2, circle.quadrant)
            && a.abs() == 1.0
            && d.abs() == 1.0
        {
            charge(fuel, 4096).ok_or(AreaQueryError::Resource)?;
            let build = || -> Option<Curve> {
                let [h, k] = affine.point(circle.center.map(|v| v.lo))?;
                if h.lo != h.hi || k.lo != k.hi {
                    return None;
                }
                let tip = exact_add(k.lo, d * sy * circle.radius)?;
                let raw = [h.lo, c / d, a * sx, k.lo, circle.radius, 1.0];
                let [h, a, b, k, r, c] = raw.map(|v| Interval::scalar(v).unwrap());
                Some(Curve {
                    function: Conic { h, a, b, k, r, c },
                    start: Interval::scalar(k.lo.min(tip))?,
                    end: Interval::scalar(k.hi.max(tip))?,
                    key: CurveKey::ExactConic(raw),
                    reach: Interval::scalar(0.0)?,
                })
            };
            return self.push(build().ok_or(AreaQueryError::Precision)?, fuel);
        }
        // Foreign affines can add root-space extrema. Retaining the complete
        // circle is a conservative cover until that domain split is qualified.
        self.circle(circle.center.map(|v| v.lo), circle.radius, affine, fuel)
    }

    fn interval_line(
        &mut self,
        line: finite_square_geometry::Line,
        affine: Affine,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        charge(fuel, 3072).ok_or(AreaQueryError::Resource)?;
        let calculate = || -> Option<(Conic, Interval, Interval, bool)> {
            let [a, b, c, d, tx, ty] = affine.0;
            let [[px, py], [qx, qy]] = line.ends;
            let point = |x: Interval, y: Interval| -> Option<[Interval; 2]> {
                Some([
                    a.mul(x)?.add(c.mul(y)?)?.add(tx)?,
                    b.mul(x)?.add(d.mul(y)?)?.add(ty)?,
                ])
            };
            let [_, y0] = point(px, py)?;
            let [_, y1] = point(qx, qy)?;
            let (slope, intercept, horizontal) = if line.axis == 0 {
                if d.lo == 0.0 && d.hi == 0.0 {
                    (Interval::scalar(0.0)?, Interval::scalar(0.0)?, true)
                } else {
                    let slope = c.div(d)?;
                    (
                        slope,
                        a.sub(slope.mul(b)?)?
                            .mul(px)?
                            .add(tx)?
                            .sub(slope.mul(ty)?)?,
                        false,
                    )
                }
            } else if b.lo == 0.0 && b.hi == 0.0 {
                (Interval::scalar(0.0)?, Interval::scalar(0.0)?, true)
            } else {
                let slope = a.div(b)?;
                (
                    slope,
                    c.sub(slope.mul(d)?)?
                        .mul(py)?
                        .add(tx)?
                        .sub(slope.mul(ty)?)?,
                    false,
                )
            };
            let (start, end) = if y0.hi < y1.lo {
                (y0, y1)
            } else if y1.hi < y0.lo {
                (y1, y0)
            } else if horizontal {
                (Interval::new(y0.lo.min(y1.lo), y0.hi.max(y1.hi))?, y1)
            } else {
                return None;
            };
            Some((
                Conic {
                    h: intercept,
                    a: slope,
                    b: Interval::scalar(0.0)?,
                    k: Interval::scalar(0.0)?,
                    r: Interval::scalar(1.0)?,
                    c: Interval::scalar(1.0)?,
                },
                start,
                end,
                horizontal,
            ))
        };
        let (function, start, end, horizontal) = calculate().ok_or(AreaQueryError::Precision)?;
        if horizontal {
            if self.extra_y_events.len() == MAX_BANDS {
                return Err(AreaQueryError::Resource);
            }
            self.extra_y_events.push(start);
            return Ok(());
        }
        if line.ends.iter().flatten().all(|v| v.lo == v.hi) {
            return self.line(
                line.ends[0].map(|v| v.lo),
                line.ends[1].map(|v| v.lo),
                affine,
                fuel,
            );
        }
        let frame = self.frame().ok_or(AreaQueryError::Resource)?;
        self.push(
            Curve {
                function,
                start,
                end,
                key: CurveKey::Vertical(frame, 0.0),
                reach: Interval::scalar(0.0).unwrap(),
            },
            fuel,
        )
    }
}
