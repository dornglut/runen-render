//! Bounded private tessellation. Stroke geometry stays local until after expansion;
//! affine transforms therefore preserve centered widths under non-uniform scale.

use crate::composition_2d::*;
use crate::execution_2d::{Render2dExecutionError, Render2dVectorError};
use lyon_tessellation::{
    FillOptions, FillRule, FillTessellator, FillVertex, LineCap, LineJoin, StrokeOptions,
    StrokeTessellator, StrokeVertex, VertexId,
    geometry_builder::{
        FillGeometryBuilder, GeometryBuilder, GeometryBuilderError, StrokeGeometryBuilder,
    },
    math::{Point, point},
    path::{Path, iterator::PathIterator},
};

const PHYSICAL_TOLERANCE: f64 = 1.0 / 64.0;
const MAX_ELEMENTS: usize = 1_048_576;

#[derive(Debug)]
pub(super) struct VectorMesh {
    pub triangles: Vec<[f64; 2]>,
    pub bounds: [u32; 4], // left, top, width, height on the target pixel lattice
    pub brush: Render2dBrush,
    pub transform: Render2dAffineTransform,
    pub raster_scale: f64,
    pub opacity: f64,
    pub root_index: usize,
}

pub(super) fn error(root_index: usize, kind: Render2dVectorError) -> Render2dExecutionError {
    Render2dExecutionError::Vector { root_index, kind }
}

pub(super) fn realize(
    item: &Render2dItem,
    root_index: usize,
    scale: f64,
    canvas: [f64; 2],
    max_buffer_bytes: u64,
) -> Result<Option<VectorMesh>, Render2dExecutionError> {
    let (shape, brush, stroke) = match item.primitive() {
        Render2dPrimitive::Fill { shape, brush } => (shape, brush, None),
        Render2dPrimitive::Stroke {
            shape,
            brush,
            style,
        } => (shape, brush, Some(*style)),
        _ => unreachable!("vector admission"),
    };
    let fail = |kind| error(root_index, kind);
    let [a, b, c, d, tx, ty] = item.local_to_parent().components();
    // Frobenius norm is a conservative upper bound on all directional scale.
    let stretch = a.hypot(b).hypot(c.hypot(d)) * scale;
    if !stretch.is_finite() {
        return Err(fail(Render2dVectorError::PrecisionLimit));
    }
    if stretch == 0.0
        || item.opacity().get() == 0.0
        || match brush {
            Render2dBrush::Solid(color) => color.channels()[3] == 0,
            Render2dBrush::Linear(gradient) => gradient
                .stops()
                .as_slice()
                .iter()
                .all(|stop| stop.color().channels()[3] == 0),
            Render2dBrush::Radial(gradient) => gradient
                .stops()
                .as_slice()
                .iter()
                .all(|stop| stop.color().channels()[3] == 0),
        }
        || stroke.is_some_and(|style| style.width() == 0.0)
    {
        return Ok(None);
    }
    let tolerance = PHYSICAL_TOLERANCE / stretch;
    if !tolerance.is_finite() || tolerance < f64::from(f32::MIN_POSITIVE) {
        return Err(fail(Render2dVectorError::PrecisionLimit));
    }
    let path = build_path(shape, tolerance).map_err(fail)?;
    // Bound flattened input before the sweep allocates internal edge storage.
    for (index, _) in path.iter().flattened(tolerance as f32).enumerate() {
        if index >= MAX_ELEMENTS {
            return Err(fail(Render2dVectorError::ResourceLimit));
        }
    }
    // Keep authored curves for stroke endpoint tangents; the probe above bounds
    // preprocessing without replacing the path by a second flattened authority.
    let mut geometry = BoundedGeometry::default();
    let result = if let Some(style) = stroke {
        let width = narrow(style.width(), tolerance).map_err(fail)?;
        let miter = narrow(style.miter_limit(), tolerance).map_err(fail)?;
        let cap = match style.cap() {
            Render2dStrokeCap::Butt => LineCap::Butt,
            Render2dStrokeCap::Round => LineCap::Round,
            Render2dStrokeCap::Square => LineCap::Square,
        };
        let join = match style.join() {
            Render2dStrokeJoin::Miter => LineJoin::Miter,
            Render2dStrokeJoin::Bevel => LineJoin::Bevel,
            Render2dStrokeJoin::Round => LineJoin::Round,
        };
        StrokeTessellator::new().tessellate_path(
            &path,
            &StrokeOptions::default()
                .with_tolerance(tolerance as f32)
                .with_line_width(width)
                .with_line_cap(cap)
                .with_line_join(join)
                .with_miter_limit(miter),
            &mut geometry,
        )
    } else {
        let rule = match shape
            .as_path()
            .map(Render2dPath::fill_rule)
            .unwrap_or_default()
        {
            Render2dFillRule::NonZero => FillRule::NonZero,
            Render2dFillRule::EvenOdd => FillRule::EvenOdd,
        };
        FillTessellator::new().tessellate_path(
            &path,
            &FillOptions::default()
                .with_fill_rule(rule)
                .with_tolerance(tolerance as f32),
            &mut geometry,
        )
    };
    if geometry.exceeded {
        return Err(fail(Render2dVectorError::ResourceLimit));
    }
    result.map_err(|_| fail(Render2dVectorError::TessellationFailed))?;
    let mut triangles = Vec::new();
    for indices in geometry.indices.as_chunks::<3>().0 {
        let mut polygon = Vec::with_capacity(7);
        for index in indices {
            let p = geometry.vertices[*index as usize];
            let x = (a.mul_add(f64::from(p.x), c.mul_add(f64::from(p.y), tx))) * scale;
            let y = (b.mul_add(f64::from(p.x), d.mul_add(f64::from(p.y), ty))) * scale;
            if !x.is_finite()
                || !y.is_finite()
                || x.abs().max(y.abs()) * f64::EPSILON > PHYSICAL_TOLERANCE / 8.0
            {
                return Err(fail(Render2dVectorError::PrecisionLimit));
            }
            polygon.push([x, y]);
        }
        let u = [polygon[1][0] - polygon[0][0], polygon[1][1] - polygon[0][1]];
        let v = [polygon[2][0] - polygon[0][0], polygon[2][1] - polygon[0][1]];
        if u[0] * v[1] - u[1] * v[0] == 0.0 {
            continue;
        }
        // Clip in f64 before the GPU ABI narrowing. Continuous canvas edges, including
        // a fractional final pixel, participate in coverage rather than a rounded scissor.
        for (axis, edge, greater) in [
            (0, 0.0, true),
            (1, 0.0, true),
            (0, canvas[0], false),
            (1, canvas[1], false),
        ] {
            polygon = clip(&polygon, axis, edge, greater);
        }
        for index in 1..polygon.len().saturating_sub(1) {
            if triangles.len() + 3 > MAX_ELEMENTS
                || (triangles.len() as u64 + 3)
                    * crate::runtime::program::abi::COMPOSITION_VERTEX_STRIDE
                    > max_buffer_bytes
            {
                return Err(fail(Render2dVectorError::ResourceLimit));
            }
            triangles.extend_from_slice(&[polygon[0], polygon[index], polygon[index + 1]]);
        }
    }
    if triangles.is_empty() {
        return Ok(None);
    }
    let (mut left, mut top, mut right, mut bottom) = (canvas[0], canvas[1], 0.0_f64, 0.0_f64);
    for [x, y] in &triangles {
        left = left.min(*x);
        top = top.min(*y);
        right = right.max(*x);
        bottom = bottom.max(*y);
    }
    if right <= left || bottom <= top {
        return Ok(None);
    }
    let left = left.floor() as u32;
    let top = top.floor() as u32;
    Ok(Some(VectorMesh {
        triangles,
        bounds: [
            left,
            top,
            right.ceil() as u32 - left,
            bottom.ceil() as u32 - top,
        ],
        brush: brush.clone(),
        transform: item.local_to_parent(),
        raster_scale: scale,
        opacity: item.opacity().get(),
        root_index,
    }))
}

fn narrow(value: f64, tolerance: f64) -> Result<f32, Render2dVectorError> {
    let narrowed = value as f32;
    // Reserve most of the error budget for flattening and arithmetic, not conversion.
    if !narrowed.is_finite()
        || (value - f64::from(narrowed)).abs() > tolerance / 8.0
        || value.abs() * f64::from(f32::EPSILON) > tolerance / 2.0
    {
        return Err(Render2dVectorError::PrecisionLimit);
    }
    Ok(narrowed)
}

fn build_path(shape: &Render2dShape, tolerance: f64) -> Result<Path, Render2dVectorError> {
    let mut builder = Path::builder();
    let p = |x, y| Ok(point(narrow(x, tolerance)?, narrow(y, tolerance)?));
    if let Some(path) = shape.as_path() {
        if path.commands().len() > MAX_ELEMENTS {
            return Err(Render2dVectorError::ResourceLimit);
        }
        let mut open = false;
        for command in path.commands() {
            match *command {
                Render2dPathCommand::MoveTo(to) => {
                    if open {
                        builder.end(false);
                    }
                    builder.begin(p(to.x(), to.y())?);
                    open = true;
                }
                Render2dPathCommand::LineTo(to) => {
                    builder.line_to(p(to.x(), to.y())?);
                }
                Render2dPathCommand::QuadraticTo { control, to } => {
                    builder.quadratic_bezier_to(p(control.x(), control.y())?, p(to.x(), to.y())?);
                }
                Render2dPathCommand::CubicTo {
                    control1,
                    control2,
                    to,
                } => {
                    builder.cubic_bezier_to(
                        p(control1.x(), control1.y())?,
                        p(control2.x(), control2.y())?,
                        p(to.x(), to.y())?,
                    );
                }
                Render2dPathCommand::Close => {
                    builder.end(true);
                    open = false;
                }
            }
        }
        if open {
            builder.end(false);
        }
    } else {
        let mut contour = Vec::new();
        if let Some(rect) = shape.as_rect() {
            if rect.is_empty() {
                return Ok(builder.build());
            }
            contour.extend_from_slice(&[
                [rect.x(), rect.y()],
                [rect.x() + rect.width(), rect.y()],
                [rect.x() + rect.width(), rect.y() + rect.height()],
                [rect.x(), rect.y() + rect.height()],
            ]);
        } else if let Some(rect) = shape.as_ellipse() {
            if rect.is_empty() {
                return Ok(builder.build());
            }
            arc(
                &mut contour,
                [
                    rect.x() + rect.width() / 2.0,
                    rect.y() + rect.height() / 2.0,
                ],
                [rect.width() / 2.0, rect.height() / 2.0],
                0.0,
                std::f64::consts::TAU,
                tolerance,
            )?;
        } else if let Some((rect, radii)) = shape.as_rounded_rect() {
            if rect.is_empty() {
                return Ok(builder.build());
            }
            let [x, y, w, h] = [rect.x(), rect.y(), rect.width(), rect.height()];
            for (center, radius, start) in [
                (
                    [x + w - radii.top_right(), y + radii.top_right()],
                    radii.top_right(),
                    -std::f64::consts::FRAC_PI_2,
                ),
                (
                    [x + w - radii.bottom_right(), y + h - radii.bottom_right()],
                    radii.bottom_right(),
                    0.0,
                ),
                (
                    [x + radii.bottom_left(), y + h - radii.bottom_left()],
                    radii.bottom_left(),
                    std::f64::consts::FRAC_PI_2,
                ),
                (
                    [x + radii.top_left(), y + radii.top_left()],
                    radii.top_left(),
                    std::f64::consts::PI,
                ),
            ] {
                arc(
                    &mut contour,
                    center,
                    [radius, radius],
                    start,
                    std::f64::consts::FRAC_PI_2,
                    tolerance,
                )?;
            }
        }
        if let Some(first) = contour.first() {
            builder.begin(p(first[0], first[1])?);
            for point in &contour[1..] {
                builder.line_to(p(point[0], point[1])?);
            }
            builder.end(true);
        }
    }
    Ok(builder.build())
}

fn arc(
    points: &mut Vec<[f64; 2]>,
    center: [f64; 2],
    radii: [f64; 2],
    start: f64,
    sweep: f64,
    tolerance: f64,
) -> Result<(), Render2dVectorError> {
    let radius = radii[0].max(radii[1]);
    if radius == 0.0 {
        points.push(center);
        return Ok(());
    }
    // Sagitta <= tolerance/2; the larger ellipse radius bounds both axes.
    let angle = 2.0 * (1.0 - (tolerance / (2.0 * radius)).min(1.0)).acos();
    let count = (sweep / angle).ceil().max(1.0);
    if !count.is_finite()
        || count > MAX_ELEMENTS as f64
        || points.len() + count as usize + 1 > MAX_ELEMENTS
    {
        return Err(Render2dVectorError::ResourceLimit);
    }
    let count = count as usize;
    for index in 0..=count {
        let angle = start + sweep * index as f64 / count as f64;
        points.push([
            center[0] + radii[0] * angle.cos(),
            center[1] + radii[1] * angle.sin(),
        ]);
    }
    Ok(())
}

fn clip(points: &[[f64; 2]], axis: usize, edge: f64, greater: bool) -> Vec<[f64; 2]> {
    let mut output = Vec::with_capacity(7);
    let Some(mut previous) = points.last().copied() else {
        return output;
    };
    let inside = |p: [f64; 2]| {
        if greater {
            p[axis] >= edge
        } else {
            p[axis] <= edge
        }
    };
    for current in points.iter().copied() {
        if inside(previous) != inside(current) {
            let fraction = (edge - previous[axis]) / (current[axis] - previous[axis]);
            let other = 1 - axis;
            let mut intersection = [0.0; 2];
            intersection[axis] = edge;
            intersection[other] = previous[other] + fraction * (current[other] - previous[other]);
            output.push(intersection);
        }
        if inside(current) {
            output.push(current);
        }
        previous = current;
    }
    output
}

#[derive(Default)]
struct BoundedGeometry {
    vertices: Vec<Point>,
    indices: Vec<u32>,
    exceeded: bool,
}
impl BoundedGeometry {
    fn vertex(&mut self, point: Point) -> Result<VertexId, GeometryBuilderError> {
        if self.vertices.len() >= MAX_ELEMENTS
            || self.exceeded
            || !point.x.is_finite()
            || !point.y.is_finite()
        {
            self.exceeded = true;
            return Err(GeometryBuilderError::TooManyVertices);
        }
        let id = VertexId(self.vertices.len() as u32);
        self.vertices.push(point);
        Ok(id)
    }
}
impl GeometryBuilder for BoundedGeometry {
    fn add_triangle(&mut self, a: VertexId, b: VertexId, c: VertexId) {
        if self.indices.len() + 3 > MAX_ELEMENTS {
            self.exceeded = true;
            return;
        }
        self.indices.extend_from_slice(&[a.0, b.0, c.0]);
    }
}
impl FillGeometryBuilder for BoundedGeometry {
    fn add_fill_vertex(
        &mut self,
        vertex: FillVertex<'_>,
    ) -> Result<VertexId, GeometryBuilderError> {
        self.vertex(vertex.position())
    }
}
impl StrokeGeometryBuilder for BoundedGeometry {
    fn add_stroke_vertex(
        &mut self,
        vertex: StrokeVertex<'_, '_>,
    ) -> Result<VertexId, GeometryBuilderError> {
        self.vertex(vertex.position())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_buffer_limit_and_collapsed_affine_are_explicit() {
        let primitive = Render2dPrimitive::Fill {
            shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 8.0, 8.0).unwrap()),
            brush: Render2dBrush::solid(Render2dColorRgba8::WHITE),
        };
        let item = Render2dItem::new(
            primitive.clone(),
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::OPAQUE,
        );
        assert!(matches!(
            realize(&item, 7, 1.0, [64.0, 64.0], 32),
            Err(Render2dExecutionError::Vector {
                root_index: 7,
                kind: Render2dVectorError::ResourceLimit
            })
        ));
        let collapsed = Render2dItem::new(
            primitive,
            Render2dAffineTransform::new(1.0, 1.0, 1.0, 1.0, 0.0, 0.0).unwrap(),
            vec![],
            Render2dOpacity::OPAQUE,
        );
        assert!(
            realize(&collapsed, 0, 1.0, [64.0, 64.0], 1024)
                .unwrap()
                .is_none()
        );
    }
}
