//! Bounded private tessellation. Stroke geometry stays local until after expansion;
//! affine transforms therefore preserve centered widths under non-uniform scale.

use super::support::NeutralMesh;
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

/// Disposable backend-neutral tessellation shared by visible paint and neutral support.
fn tessellate_shape(
    shape: &Render2dShape,
    stroke: Option<Render2dStrokeStyle>,
    root_index: usize,
    stretch: f64,
    mut admitted: Option<(&mut super::shadow::NeutralPreparationWork<'_>, &[usize])>,
) -> Result<BoundedGeometry, Render2dExecutionError> {
    let fail = |kind| error(root_index, kind);
    let tolerance = PHYSICAL_TOLERANCE / stretch;
    if !tolerance.is_finite() || tolerance < f64::from(f32::MIN_POSITIVE) {
        return Err(fail(Render2dVectorError::PrecisionLimit));
    }
    let path = build_path(shape, tolerance).map_err(fail)?;
    // Bound flattened input before the sweep allocates internal edge storage.
    // Each observed flattened event is real CPU work owned by the same
    // invocation as the later index scan. Charge it before requesting the
    // next event or allocating tessellation output; failed work is not refunded.
    for (index, _) in path.iter().flattened(tolerance as f32).enumerate() {
        if index >= MAX_ELEMENTS {
            return Err(fail(Render2dVectorError::ResourceLimit));
        }
        if let Some((work, path)) = admitted.as_mut() {
            work.charge(1, path)?;
        }
    }
    // Keep authored curves for stroke endpoint tangents; the probe above bounds
    // preprocessing without replacing the path by a second flattened authority.
    let mut sink = TessellationSink::new(admitted);
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
            &mut sink,
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
            &mut sink,
        )
    };
    if sink.geometry.exceeded {
        return Err(fail(Render2dVectorError::ResourceLimit));
    }
    result.map_err(|_| fail(Render2dVectorError::TessellationFailed))?;
    Ok(sink.into_geometry())
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
    let geometry = tessellate_shape(shape, stroke, root_index, stretch, None)?;
    let mut triangles = Vec::new();
    for indices in geometry.indices.as_chunks::<3>().0 {
        let Some(triangle) =
            transformed_triangle(&geometry, indices, [a, b, c, d, tx, ty], scale, root_index)?
        else {
            continue;
        };
        // Each clipped triangle is a convex polygon with at most seven
        // vertices after four half-planes. Keep both intermediate buffers
        // on the stack rather than allocating five Vecs per triangle.
        let mut polygon = [[0.0; 2]; 8];
        polygon[..3].copy_from_slice(&triangle);
        let mut count = 3;
        let mut scratch = [[0.0; 2]; 8];
        // Clip in f64 before the GPU ABI narrowing. Continuous canvas edges,
        // including a fractional final pixel, participate in coverage.
        for (axis, edge, greater) in [
            (0, 0.0, true),
            (1, 0.0, true),
            (0, canvas[0], false),
            (1, canvas[1], false),
        ] {
            count =
                clip_into(&polygon[..count], &mut scratch, axis, edge, greater).map_err(fail)?;
            std::mem::swap(&mut polygon, &mut scratch);
        }
        for index in 1..count.saturating_sub(1) {
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

/// Retained shaped-glyph geometry in the owning item's immediate-parent frame.
///
/// The same unhinted, variation-resolved font outline used for F2 MSDF
/// generation is re-tessellated by the F3 vector authority. MSDF pixels,
/// foreground alpha, field overscan and final-target clipping never define
/// neutral glyph support. An ancestor affine is deliberately absent: effects
/// using this support operate in their group's parent frame first.
pub(super) struct NeutralGlyphInstance<'a> {
    pub(super) field: &'a super::field::GlyphField,
    pub(super) resource_id: Render2dResourceId,
    pub(super) origin: [f64; 2],
    pub(super) font_size: f64,
    pub(super) to_parent: Render2dAffineTransform,
}

pub(super) fn neutral_shaped_glyph(
    glyph: NeutralGlyphInstance<'_>,
    path: &[usize],
    samples_per_parent_logical_unit: f64,
    max_buffer_bytes: u64,
    work: &mut super::shadow::NeutralPreparationWork<'_>,
) -> Result<Option<NeutralMesh>, Render2dExecutionError> {
    work.charge(512, path)?;
    let NeutralGlyphInstance {
        field,
        resource_id,
        origin: glyph_origin,
        font_size,
        to_parent: glyph_to_parent,
    } = glyph;
    let precision = || Render2dExecutionError::SampleSpace {
        kind: crate::execution_2d::Render2dSampleSpaceError::PrecisionLimit,
        path: Some(path.to_vec()),
        detail: "unrepresentable shaped-glyph parent-frame support".to_owned(),
    };
    if !font_size.is_finite()
        || font_size <= 0.0
        || !samples_per_parent_logical_unit.is_finite()
        || samples_per_parent_logical_unit <= 0.0
        || !glyph_origin.iter().all(|component| component.is_finite())
    {
        return Err(precision());
    }
    let [a, b, c, d, tx, ty] = glyph_to_parent.components();
    let transform = Render2dAffineTransform::new(
        a * font_size,
        b * font_size,
        c * font_size,
        d * font_size,
        a.mul_add(glyph_origin[0], c.mul_add(glyph_origin[1], tx)),
        b.mul_add(glyph_origin[0], d.mul_add(glyph_origin[1], ty)),
    )
    .map_err(|_| precision())?;
    let shape = field.neutral_shape(resource_id, path, work)?;
    let command_bytes = size_of_val(shape.as_path().expect("neutral glyph path").commands())
        + 2 * size_of::<usize>();
    let item = Render2dItem::new(
        Render2dPrimitive::Fill {
            shape,
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        },
        transform,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
    );
    let result = neutral_support_admitted(
        &item,
        path[0],
        samples_per_parent_logical_unit,
        max_buffer_bytes,
        work,
        path,
    );
    drop(item);
    work.release_bytes(command_bytes);
    result
}

/// A source-neutral vector's tessellated physical support before paint/canvas culling.
///
/// Triangles are disposable physical approximations of the accepted semantic geometry;
/// neither their boundaries nor these bounds define a second semantic support authority.
fn transformed_triangle(
    geometry: &BoundedGeometry,
    indices: &[u32; 3],
    [a, b, c, d, tx, ty]: [f64; 6],
    scale: f64,
    root_index: usize,
) -> Result<Option<[[f64; 2]; 3]>, Render2dExecutionError> {
    let mut polygon = [[0.0; 2]; 3];
    for (position, index) in indices.iter().enumerate() {
        let point = geometry.vertices[*index as usize];
        let x = (a.mul_add(f64::from(point.x), c.mul_add(f64::from(point.y), tx))) * scale;
        let y = (b.mul_add(f64::from(point.x), d.mul_add(f64::from(point.y), ty))) * scale;
        if !x.is_finite()
            || !y.is_finite()
            || x.abs().max(y.abs()) * f64::EPSILON > PHYSICAL_TOLERANCE / 8.0
        {
            return Err(error(root_index, Render2dVectorError::PrecisionLimit));
        }
        polygon[position] = [x, y];
    }
    let u = [polygon[1][0] - polygon[0][0], polygon[1][1] - polygon[0][1]];
    let v = [polygon[2][0] - polygon[0][0], polygon[2][1] - polygon[0][1]];
    Ok((u[0] * v[1] - u[1] * v[0] != 0.0).then_some(polygon))
}

/// Reconstructs the geometry that may cast shadows, before visible-paint alpha
/// or final-canvas culling. Signed coordinates are retained: off-canvas casters
/// may still reach the target after later group offsets, spreads or blurs.
///
/// This private geometry retains the same bounded tessellation and precision
/// admission as visible vectors. Shadow-specific morphology, clips, halo, and
/// group support are derived by the F3F composition compiler, not by this leaf.
#[cfg(test)]
pub(super) fn neutral_support(
    item: &Render2dItem,
    root_index: usize,
    scale: f64,
    max_buffer_bytes: u64,
) -> Result<Option<NeutralMesh>, Render2dExecutionError> {
    neutral_support_impl(item, root_index, scale, max_buffer_bytes, None)
}

/// The clip compiler owns the same result buffer throughout construction and
/// subsequent intersection. Admit its exact emitted-vertex capacity before
/// allocation; the caller releases that lease when clip source storage dies.
/// Lyon's private sweep/path allocations are separate, still-open obligations.
pub(super) fn neutral_support_admitted(
    item: &Render2dItem,
    root_index: usize,
    scale: f64,
    max_buffer_bytes: u64,
    work: &mut super::shadow::NeutralPreparationWork<'_>,
    path: &[usize],
) -> Result<Option<NeutralMesh>, Render2dExecutionError> {
    neutral_support_impl(
        item,
        root_index,
        scale,
        max_buffer_bytes,
        Some((work, path)),
    )
}

fn neutral_support_impl(
    item: &Render2dItem,
    root_index: usize,
    scale: f64,
    max_buffer_bytes: u64,
    mut admitted: Option<(&mut super::shadow::NeutralPreparationWork<'_>, &[usize])>,
) -> Result<Option<NeutralMesh>, Render2dExecutionError> {
    let (shape, stroke) = match item.primitive() {
        Render2dPrimitive::Fill { shape, .. } => (shape, None),
        Render2dPrimitive::Stroke { shape, style, .. } => (shape, Some(*style)),
        _ => unreachable!("vector neutral geometry admission"),
    };
    let [a, b, c, d, tx, ty] = item.local_to_parent().components();
    let stretch = a.hypot(b).hypot(c.hypot(d)) * scale;
    if !stretch.is_finite() {
        return Err(error(root_index, Render2dVectorError::PrecisionLimit));
    }
    if stretch == 0.0 || stroke.is_some_and(|style| style.width() == 0.0) {
        return Ok(None);
    }
    // Move the existing scan setup admission ahead of any original Lyon path
    // construction. An exhausted shared invocation must not allocate a path
    // or enter tessellation before paying the same required setup work.
    if let Some((work, path)) = admitted.as_mut() {
        work.charge(256, path)?;
    }
    let geometry = tessellate_shape(
        shape,
        stroke,
        root_index,
        stretch,
        admitted.as_mut().map(|(work, path)| (&mut **work, *path)),
    )?;
    // Keep geometry output and final source storage under the same host owner
    // until source conversion completes (including every error exit).
    let prepared = (|| -> Result<Option<NeutralMesh>, Render2dExecutionError> {
        let mut triangles = Vec::new();
        if let Some((work, path)) = admitted.as_mut() {
            // Only the per-index scan work remains: the 256 setup was paid before
            // Lyon allocation. Do not double-charge successful source work.
            // Lyon preprocessing/sweep remain separate unproved owners.
            let scan_cost = geometry
                .indices
                .len()
                .checked_mul(64)
                .ok_or_else(|| error(root_index, Render2dVectorError::ResourceLimit))?;
            work.charge(scan_cost, path)?;
            // Preflight all *actual* emitted triangles before committing a host
            // lease, rather than reserving tessellator triangles later rejected
            // by a singular transform. This source is immutable and the second
            // pass uses the identical deterministic conversion.
            let mut required = 0_usize;
            for indices in geometry.indices.as_chunks::<3>().0 {
                if transformed_triangle(
                    &geometry,
                    indices,
                    [a, b, c, d, tx, ty],
                    scale,
                    root_index,
                )?
                .is_some()
                {
                    required = required
                        .checked_add(3)
                        .ok_or_else(|| error(root_index, Render2dVectorError::ResourceLimit))?;
                    let bytes = u64::try_from(required)
                        .ok()
                        .and_then(|n| {
                            n.checked_mul(crate::runtime::program::abi::COMPOSITION_VERTEX_STRIDE)
                        })
                        .ok_or_else(|| error(root_index, Render2dVectorError::ResourceLimit))?;
                    if required > MAX_ELEMENTS || bytes > max_buffer_bytes {
                        return Err(error(root_index, Render2dVectorError::ResourceLimit));
                    }
                }
            }
            work.reserve_vector(&mut triangles, required, path)?;
        }
        let mut bounds = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        let emitted = (|| -> Result<(), Render2dExecutionError> {
            for indices in geometry.indices.as_chunks::<3>().0 {
                let Some(triangle) = transformed_triangle(
                    &geometry,
                    indices,
                    [a, b, c, d, tx, ty],
                    scale,
                    root_index,
                )?
                else {
                    continue;
                };
                let count = triangles
                    .len()
                    .checked_add(3)
                    .ok_or_else(|| error(root_index, Render2dVectorError::ResourceLimit))?;
                let bytes = u64::try_from(count)
                    .ok()
                    .and_then(|n| {
                        n.checked_mul(crate::runtime::program::abi::COMPOSITION_VERTEX_STRIDE)
                    })
                    .ok_or_else(|| error(root_index, Render2dVectorError::ResourceLimit))?;
                if count > MAX_ELEMENTS || bytes > max_buffer_bytes {
                    return Err(error(root_index, Render2dVectorError::ResourceLimit));
                }
                for [x, y] in triangle {
                    bounds[0] = bounds[0].min(x);
                    bounds[1] = bounds[1].min(y);
                    bounds[2] = bounds[2].max(x);
                    bounds[3] = bounds[3].max(y);
                    triangles.push([x, y]);
                }
            }
            Ok(())
        })();
        if let Err(err) = emitted {
            if let Some((work, _)) = admitted.as_mut() {
                work.release_bytes(triangles.capacity() * std::mem::size_of::<[f64; 2]>());
            }
            return Err(err);
        }
        Ok((!triangles.is_empty()).then_some(NeutralMesh {
            units_per_parent_logical_unit: scale,
            triangles,
            bounds,
        }))
    })();
    // The physical scratch is destroyed before its host lease is retired.
    let scratch_bytes = geometry.leased_bytes();
    drop(geometry);
    if let Some((work, _)) = admitted {
        work.release_bytes(scratch_bytes);
    }
    prepared
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
        // Generate Lyon path events directly. A separately allocated contour
        // used to coexist with the path and later tessellator scratch.
        let mut open = false;
        {
            let mut emit = |[x, y]: [f64; 2]| -> Result<(), Render2dVectorError> {
                let to = p(x, y)?;
                if open {
                    builder.line_to(to);
                } else {
                    builder.begin(to);
                    open = true;
                }
                Ok(())
            };
            let mut emitted = 0_usize;
            if let Some(rect) = shape.as_rect() {
                if !rect.is_empty() {
                    for point in [
                        [rect.x(), rect.y()],
                        [rect.x() + rect.width(), rect.y()],
                        [rect.x() + rect.width(), rect.y() + rect.height()],
                        [rect.x(), rect.y() + rect.height()],
                    ] {
                        emit(point)?;
                    }
                }
            } else if let Some(rect) = shape.as_ellipse() {
                if !rect.is_empty() {
                    arc(
                        &mut emit,
                        &mut emitted,
                        [
                            rect.x() + rect.width() / 2.0,
                            rect.y() + rect.height() / 2.0,
                        ],
                        [rect.width() / 2.0, rect.height() / 2.0],
                        0.0,
                        std::f64::consts::TAU,
                        tolerance,
                    )?;
                }
            } else if let Some((rect, radii)) = shape.as_rounded_rect()
                && !rect.is_empty()
            {
                let [x, y, w, h] = [rect.x(), rect.y(), rect.width(), rect.height()];
                let arcs = [
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
                ];
                // All four authored corner arcs previously finished counting
                // before any point was narrowed into the Lyon path. Preserve
                // that resource-before-precision failure priority without a Vec.
                let mut total = 0_usize;
                for (_, radius, _) in arcs {
                    let steps =
                        arc_count([radius, radius], std::f64::consts::FRAC_PI_2, tolerance)?;
                    total = total
                        .checked_add(steps)
                        .filter(|&count| count <= MAX_ELEMENTS)
                        .ok_or(Render2dVectorError::ResourceLimit)?;
                }
                for (center, radius, start) in arcs {
                    arc(
                        &mut emit,
                        &mut emitted,
                        center,
                        [radius, radius],
                        start,
                        std::f64::consts::FRAC_PI_2,
                        tolerance,
                    )?;
                }
            }
        }
        if open {
            builder.end(true);
        }
    }
    Ok(builder.build())
}

fn arc_count(radii: [f64; 2], sweep: f64, tolerance: f64) -> Result<usize, Render2dVectorError> {
    let radius = radii[0].max(radii[1]);
    if radius == 0.0 {
        return Ok(1);
    }
    // Sagitta <= tolerance/2; preserve every endpoint of each authored arc.
    let angle = 2.0 * (1.0 - (tolerance / (2.0 * radius)).min(1.0)).acos();
    let count = (sweep / angle).ceil().max(1.0);
    if !count.is_finite() || count > MAX_ELEMENTS as f64 {
        return Err(Render2dVectorError::ResourceLimit);
    }
    Ok(count as usize + 1)
}

fn arc(
    emit: &mut impl FnMut([f64; 2]) -> Result<(), Render2dVectorError>,
    emitted: &mut usize,
    center: [f64; 2],
    radii: [f64; 2],
    start: f64,
    sweep: f64,
    tolerance: f64,
) -> Result<(), Render2dVectorError> {
    let steps = arc_count(radii, sweep, tolerance)?;
    *emitted = emitted
        .checked_add(steps)
        .filter(|&total| total <= MAX_ELEMENTS)
        .ok_or(Render2dVectorError::ResourceLimit)?;
    if radii[0].max(radii[1]) == 0.0 {
        return emit(center);
    }
    let count = steps - 1;
    for index in 0..=count {
        let angle = start + sweep * index as f64 / count as f64;
        emit([
            center[0] + radii[0] * angle.cos(),
            center[1] + radii[1] * angle.sin(),
        ])?;
    }
    Ok(())
}

/// The four half-plane passes on a convex triangle produce at most seven
/// vertices. A capacity violation fails closed rather than allocating scratch.
fn clip_into(
    points: &[[f64; 2]],
    output: &mut [[f64; 2]; 8],
    axis: usize,
    edge: f64,
    greater: bool,
) -> Result<usize, Render2dVectorError> {
    let Some(mut previous) = points.last().copied() else {
        return Ok(0);
    };
    let inside = |p: [f64; 2]| {
        if greater {
            p[axis] >= edge
        } else {
            p[axis] <= edge
        }
    };
    let mut count = 0;
    for current in points.iter().copied() {
        if inside(previous) != inside(current) {
            let fraction = (edge - previous[axis]) / (current[axis] - previous[axis]);
            let other = 1 - axis;
            let mut intersection = [0.0; 2];
            intersection[axis] = edge;
            intersection[other] = previous[other] + fraction * (current[other] - previous[other]);
            if count == output.len() {
                return Err(Render2dVectorError::ResourceLimit);
            }
            output[count] = intersection;
            count += 1;
        }
        if inside(current) {
            if count == output.len() {
                return Err(Render2dVectorError::ResourceLimit);
            }
            output[count] = current;
            count += 1;
        }
        previous = current;
    }
    Ok(count)
}

#[derive(Default)]
struct BoundedGeometry {
    vertices: Vec<Point>,
    indices: Vec<u32>,
    exceeded: bool,
    // Transferred output leases; root visible paint has no shadow lease.
    vertex_lease: usize,
    index_lease: usize,
}

impl BoundedGeometry {
    fn leased_bytes(&self) -> usize {
        self.vertex_lease + self.index_lease
    }
}

/// Lyon owns geometry formation; the owning invocation admits the callback
/// output Vecs before growth. Failure drops the sink and retires its leases.
/// Success transfers both buffers and their leases through neutral conversion.
struct TessellationSink<'w, 'a> {
    geometry: BoundedGeometry,
    admitted: Option<(
        &'w mut super::shadow::NeutralPreparationWork<'a>,
        &'w [usize],
    )>,
}

impl<'w, 'a> TessellationSink<'w, 'a> {
    fn new(
        admitted: Option<(
            &'w mut super::shadow::NeutralPreparationWork<'a>,
            &'w [usize],
        )>,
    ) -> Self {
        Self {
            geometry: BoundedGeometry::default(),
            admitted,
        }
    }

    fn next_capacity(current: usize, needed: usize) -> usize {
        current
            .max(4)
            .saturating_mul(2)
            .max(needed)
            .min(MAX_ELEMENTS)
    }

    fn into_geometry(mut self) -> BoundedGeometry {
        std::mem::take(&mut self.geometry)
    }

    fn vertex(&mut self, point: Point) -> Result<VertexId, GeometryBuilderError> {
        if self.geometry.vertices.len() >= MAX_ELEMENTS
            || self.geometry.exceeded
            || !point.x.is_finite()
            || !point.y.is_finite()
        {
            self.geometry.exceeded = true;
            return Err(GeometryBuilderError::TooManyVertices);
        }
        let id = VertexId(self.geometry.vertices.len() as u32);
        if self.geometry.vertices.len() == self.geometry.vertices.capacity()
            && let Some((work, path)) = self.admitted.as_mut()
        {
            let capacity =
                Self::next_capacity(self.geometry.vertices.capacity(), id.0 as usize + 1);
            if work
                .reserve_vector(&mut self.geometry.vertices, capacity, path)
                .is_err()
            {
                self.geometry.exceeded = true;
                return Err(GeometryBuilderError::TooManyVertices);
            }
            self.geometry.vertex_lease =
                self.geometry.vertices.capacity() * std::mem::size_of::<Point>();
        }
        self.geometry.vertices.push(point);
        Ok(id)
    }
}

impl Drop for TessellationSink<'_, '_> {
    fn drop(&mut self) {
        if let Some((work, _)) = self.admitted.as_mut() {
            work.release_bytes(self.geometry.leased_bytes());
        }
    }
}

impl GeometryBuilder for TessellationSink<'_, '_> {
    fn add_triangle(&mut self, a: VertexId, b: VertexId, c: VertexId) {
        let Some(needed) = self.geometry.indices.len().checked_add(3) else {
            self.geometry.exceeded = true;
            return;
        };
        if self.geometry.exceeded || needed > MAX_ELEMENTS {
            self.geometry.exceeded = true;
            return;
        }
        if needed > self.geometry.indices.capacity()
            && let Some((work, path)) = self.admitted.as_mut()
        {
            let capacity = Self::next_capacity(self.geometry.indices.capacity(), needed);
            if work
                .reserve_vector(&mut self.geometry.indices, capacity, path)
                .is_err()
            {
                self.geometry.exceeded = true;
                return;
            }
            self.geometry.index_lease =
                self.geometry.indices.capacity() * std::mem::size_of::<u32>();
        }
        self.geometry.indices.extend_from_slice(&[a.0, b.0, c.0]);
    }
}

impl FillGeometryBuilder for TessellationSink<'_, '_> {
    fn add_fill_vertex(
        &mut self,
        vertex: FillVertex<'_>,
    ) -> Result<VertexId, GeometryBuilderError> {
        self.vertex(vertex.position())
    }
}

impl StrokeGeometryBuilder for TessellationSink<'_, '_> {
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
    fn stack_canvas_clipping_preserves_prior_polygon_events() {
        // Independent former Vec clipping sequence, retained only as an
        // oracle. Compare ordered vertices, duplicates and fractional edges.
        fn reference(points: &[[f64; 2]], axis: usize, edge: f64, greater: bool) -> Vec<[f64; 2]> {
            let mut out = Vec::with_capacity(7);
            let Some(mut prev) = points.last().copied() else {
                return out;
            };
            let inside = |p: [f64; 2]| {
                if greater {
                    p[axis] >= edge
                } else {
                    p[axis] <= edge
                }
            };
            for curr in points.iter().copied() {
                if inside(prev) != inside(curr) {
                    let fraction = (edge - prev[axis]) / (curr[axis] - prev[axis]);
                    let other = 1 - axis;
                    let mut intersection = [0.0; 2];
                    intersection[axis] = edge;
                    intersection[other] = prev[other] + fraction * (curr[other] - prev[other]);
                    out.push(intersection);
                }
                if inside(curr) {
                    out.push(curr);
                }
                prev = curr;
            }
            out
        }
        for triangle in [
            [[-2.0, -1.0], [3.0, 0.75], [0.25, 4.0]],
            [[0.0, 0.0], [2.5, 0.0], [0.0, 2.5]],
            [[4.0, 4.0], [5.0, 4.0], [4.0, 5.0]],
            [[-1.0, 0.0], [-1.0, 1.0], [0.0, 0.0]],
            [[0.5, 0.5], [0.5, 0.5], [1.5, 1.5]],
        ] {
            let mut actual = [[0.0; 2]; 8];
            actual[..3].copy_from_slice(&triangle);
            let mut len = 3;
            let mut scratch = [[0.0; 2]; 8];
            let mut expected = triangle.to_vec();
            for (axis, edge, greater) in [
                (0, 0.0, true),
                (1, 0.0, true),
                (0, 2.25, false),
                (1, 1.75, false),
            ] {
                len = clip_into(&actual[..len], &mut scratch, axis, edge, greater).unwrap();
                std::mem::swap(&mut actual, &mut scratch);
                expected = reference(&expected, axis, edge, greater);
                assert_eq!(&actual[..len], expected.as_slice());
            }
        }
        let mut output = [[0.0; 2]; 8];
        assert_eq!(clip_into(&[], &mut output, 0, 1.0, true).unwrap(), 0);
        // This artificial non-convex event stream violates the capacity
        // contract; it must reject without heap fallback.
        let excessive = [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [0.0, 3.0],
            [2.0, 3.0],
        ];
        assert!(matches!(
            clip_into(&excessive, &mut output, 0, 1.0, true),
            Err(Render2dVectorError::ResourceLimit)
        ));
    }

    #[test]
    fn streamed_ellipse_events_match_reference_contour_and_reject_excess() {
        let shape = Render2dShape::ellipse(Render2dRect::new(1.0, 2.0, 4.0, 6.0).unwrap());
        let tolerance = 0.125;
        let actual = build_path(&shape, tolerance).unwrap();
        // Independent reference retains the former temporary contour only
        // in this test. Position arithmetic, event order and closure must
        // remain byte-for-byte equivalent under the production refactor.
        let center = [3.0, 5.0];
        let radii = [2.0, 3.0];
        let angle = 2.0 * (1.0 - tolerance / (2.0 * radii[1])).acos();
        let segments = (std::f64::consts::TAU / angle).ceil().max(1.0) as usize;
        let mut points = Vec::new();
        for index in 0..=segments {
            let theta = std::f64::consts::TAU * index as f64 / segments as f64;
            points.push(point(
                narrow(center[0] + radii[0] * theta.cos(), tolerance).unwrap(),
                narrow(center[1] + radii[1] * theta.sin(), tolerance).unwrap(),
            ));
        }
        let mut reference = Path::builder();
        reference.begin(points[0]);
        for &p in &points[1..] {
            reference.line_to(p);
        }
        reference.end(true);
        assert_eq!(
            actual.iter().collect::<Vec<_>>(),
            reference.build().iter().collect::<Vec<_>>(),
        );
        // An event count above the existing hard cap rejects before any
        // contour-length allocation, not as a successfully drawn ellipse.
        assert!(matches!(
            build_path(&shape, 1.0e-14),
            Err(Render2dVectorError::ResourceLimit)
        ));
    }

    #[test]
    fn streamed_rounded_contours_match_independent_prior_event_order() {
        // The old factory first collected all four arcs, including shared
        // endpoints and zero-radius corners, then emitted a single closed path.
        // Preserve exactly those events without retaining its heap contour.
        let tolerance = 0.125;
        for radii in [
            Render2dCornerRadii::new(1.0, 2.0, 1.5, 0.5).unwrap(),
            Render2dCornerRadii::new(0.0, 2.0, 1.5, 0.0).unwrap(),
        ] {
            let shape =
                Render2dShape::rounded_rect(Render2dRect::new(-2.0, 1.0, 9.0, 8.0).unwrap(), radii);
            let actual = build_path(&shape, tolerance).unwrap();
            let (rect, radii) = shape.as_rounded_rect().unwrap();
            let [x, y, w, h] = [rect.x(), rect.y(), rect.width(), rect.height()];
            let arcs = [
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
            ];
            let mut points = Vec::new();
            for (center, radius, start) in arcs {
                if radius == 0.0 {
                    points.push(point(
                        narrow(center[0], tolerance).unwrap(),
                        narrow(center[1], tolerance).unwrap(),
                    ));
                    continue;
                }
                let angle = 2.0 * (1.0 - (tolerance / (2.0 * radius)).min(1.0)).acos();
                let segments = (std::f64::consts::FRAC_PI_2 / angle).ceil().max(1.0) as usize;
                for index in 0..=segments {
                    let theta =
                        start + std::f64::consts::FRAC_PI_2 * index as f64 / segments as f64;
                    points.push(point(
                        narrow(center[0] + radius * theta.cos(), tolerance).unwrap(),
                        narrow(center[1] + radius * theta.sin(), tolerance).unwrap(),
                    ));
                }
            }
            let mut reference = Path::builder();
            reference.begin(points[0]);
            for &point in &points[1..] {
                reference.line_to(point);
            }
            reference.end(true);
            assert_eq!(
                actual.iter().collect::<Vec<_>>(),
                reference.build().iter().collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn rounded_contour_resource_preflight_precedes_point_precision() {
        // Each quarter arc is individually within the event cap, but all
        // four combined exceed it. The large X would fail f32 precision
        // during the first arc if we emitted before counting the full path.
        let shape = Render2dShape::rounded_rect(
            Render2dRect::new(1.0e12, 0.0, 20.0, 20.0).unwrap(),
            Render2dCornerRadii::new(3.0, 3.0, 3.0, 3.0).unwrap(),
        );
        assert!(matches!(
            build_path(&shape, 1.0e-11),
            Err(Render2dVectorError::ResourceLimit)
        ));
    }

    #[test]
    fn neutral_support_preserves_transparent_offscreen_fill_and_signed_bounds() {
        let source = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(-10.0, -2.0, 9.0, 4.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        assert!(
            realize(&source, 3, 1.0, [16.0, 16.0], 4096)
                .unwrap()
                .is_none()
        );
        let support = neutral_support(&source, 3, 1.0, 4096)
            .unwrap()
            .expect("transparent offscreen source has semantic geometry");
        assert_eq!(support.bounds, [-10.0, -2.0, -1.0, 2.0]);
        assert_eq!(support.triangles.len(), 6);
    }

    #[test]
    fn flattened_input_scan_spends_shared_work_before_lyon_sweep() {
        let source = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let path = [4, 7];
        let mut graph = super::super::support::MAX_COMPOSITIONAL_SHADOW_WORK - 256 - 2;
        let mut clip_pairs = 0;
        let mut host = super::super::support::PreparationHostBudget::new();
        // Two flatten events are admitted after setup. The third event
        // fails before any Lyon callback output is allocated.
        let result = neutral_support_admitted(
            &source,
            4,
            1.0,
            4096,
            &mut super::super::shadow::NeutralPreparationWork::new(
                &mut graph,
                &mut clip_pairs,
                &mut host,
            ),
            &path,
        );
        assert!(matches!(
            result,
            Err(Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(owner),
                ..
            }) if owner == path
        ));
        assert_eq!(graph, super::super::support::MAX_COMPOSITIONAL_SHADOW_WORK);
        assert_eq!((host.live(), host.peak()), (0, 0));
        // A separate invocation owns its own complete receipt.
        let mut graph = 0;
        let mesh = neutral_support_admitted(
            &source,
            4,
            1.0,
            4096,
            &mut super::super::shadow::NeutralPreparationWork::new(
                &mut graph,
                &mut clip_pairs,
                &mut host,
            ),
            &path,
        )
        .unwrap()
        .unwrap();
        assert_eq!(mesh.triangles.len(), 6);
        assert!(graph > 256 + 6 * 64);
        assert_eq!(
            host.live(),
            mesh.triangles.capacity() * std::mem::size_of::<[f64; 2]>()
        );
    }

    #[test]
    fn original_vector_setup_is_charged_before_lyon_path_construction() {
        let owner = [7, 3];
        let badly_narrowed = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::path(
                    Render2dPath::new(
                        Render2dFillRule::NonZero,
                        vec![
                            Render2dPathCommand::MoveTo(Render2dPoint::new(0.0, 0.0).unwrap()),
                            Render2dPathCommand::LineTo(Render2dPoint::new(1.0e20, 0.0).unwrap()),
                            Render2dPathCommand::LineTo(Render2dPoint::new(0.0, 1.0).unwrap()),
                            Render2dPathCommand::Close,
                        ],
                    )
                    .unwrap(),
                ),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        // A one-short work receipt beats physical f32 source precision, and
        // neither consumes fuel nor allocates admitted source output.
        let mut work_used = super::super::support::MAX_COMPOSITIONAL_SHADOW_WORK - 255;
        let mut clips = 0;
        let mut host = super::super::support::PreparationHostBudget::new();
        let rejected = neutral_support_admitted(
            &badly_narrowed,
            7,
            1.0,
            4096,
            &mut super::super::shadow::NeutralPreparationWork::new(
                &mut work_used,
                &mut clips,
                &mut host,
            ),
            &owner,
        );
        assert!(matches!(
            rejected,
            Err(Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == owner
        ));
        assert_eq!(
            work_used,
            super::super::support::MAX_COMPOSITIONAL_SHADOW_WORK - 255
        );
        assert_eq!((host.live(), host.peak()), (0, 0));
        // At the exact setup threshold the same F1 input now reaches
        // physical construction and is correctly rejected for precision.
        work_used -= 1;
        assert!(matches!(
            neutral_support_admitted(
                &badly_narrowed,
                7,
                1.0,
                4096,
                &mut super::super::shadow::NeutralPreparationWork::new(
                    &mut work_used,
                    &mut clips,
                    &mut host,
                ),
                &owner,
            ),
            Err(Render2dExecutionError::Vector {
                root_index: 7,
                kind: Render2dVectorError::PrecisionLimit
            })
        ));
        assert_eq!(
            work_used,
            super::super::support::MAX_COMPOSITIONAL_SHADOW_WORK
        );
        assert_eq!((host.live(), host.peak()), (0, 0));
    }

    #[test]
    fn tessellation_callback_buffers_share_host_and_refund_late_failure() {
        let path = [7, 2];
        let first_vertices = 8 * std::mem::size_of::<Point>();
        let first_indices = 8 * std::mem::size_of::<u32>();
        let scratch = first_vertices + first_indices;
        let mut host = super::super::support::PreparationHostBudget::new();
        let prior = super::super::support::MAX_SHADOW_HOST_BYTES - scratch;
        host.reserve(prior, &path).unwrap();
        let mut graph_work = 0;
        let mut clip_work = 0;
        {
            let mut work = super::super::shadow::NeutralPreparationWork::new(
                &mut graph_work,
                &mut clip_work,
                &mut host,
            );
            let mut sink = TessellationSink::new(Some((&mut work, &path)));
            for index in 0..8 {
                sink.vertex(point(index as f32, 0.0)).unwrap();
            }
            sink.add_triangle(VertexId(0), VertexId(1), VertexId(2));
            assert_eq!(sink.geometry.vertices.capacity(), 8);
            assert_eq!(sink.geometry.indices.capacity(), 8);
            assert_eq!(sink.geometry.leased_bytes(), scratch);
            // Growing the ninth vertex would relocate the old eight-vertex
            // buffer. Admit the new allocation while the old one is live,
            // reject at the exact host cap, and do not mutate the output.
            assert!(matches!(
                sink.vertex(point(8.0, 0.0)),
                Err(GeometryBuilderError::TooManyVertices)
            ));
            assert!(sink.geometry.exceeded);
            assert_eq!(sink.geometry.vertices.len(), 8);
            assert_eq!(sink.geometry.indices.len(), 3);
        }
        assert_eq!(host.live(), prior);
        assert_eq!(host.peak(), super::super::support::MAX_SHADOW_HOST_BYTES);
        assert_eq!(graph_work, 0);
    }

    #[test]
    fn tessellation_callback_index_growth_refunds_previous_vertex_output() {
        let path = [8, 1];
        let vertex_bytes = 8 * std::mem::size_of::<Point>();
        let index_bytes = 8 * std::mem::size_of::<u32>();
        let prior = super::super::support::MAX_SHADOW_HOST_BYTES - vertex_bytes - index_bytes;
        let mut host = super::super::support::PreparationHostBudget::new();
        host.reserve(prior, &path).unwrap();
        let mut graph_work = 0;
        let mut clip_work = 0;
        {
            let mut work = super::super::shadow::NeutralPreparationWork::new(
                &mut graph_work,
                &mut clip_work,
                &mut host,
            );
            let mut sink = TessellationSink::new(Some((&mut work, &path)));
            sink.vertex(point(0.0, 0.0)).unwrap();
            sink.vertex(point(1.0, 0.0)).unwrap();
            sink.vertex(point(0.0, 1.0)).unwrap();
            sink.add_triangle(VertexId(0), VertexId(1), VertexId(2));
            sink.add_triangle(VertexId(0), VertexId(1), VertexId(2));
            assert_eq!(sink.geometry.indices.len(), 6);
            // The third triangle crosses the admitted capacity of eight.
            sink.add_triangle(VertexId(0), VertexId(1), VertexId(2));
            assert!(sink.geometry.exceeded);
            assert_eq!(sink.geometry.indices.len(), 6);
        }
        assert_eq!(host.live(), prior);
        assert_eq!(host.peak(), super::super::support::MAX_SHADOW_HOST_BYTES);
    }

    #[test]
    fn admitted_clip_source_output_owns_exact_host_receipt() {
        let source = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(-10.0, -2.0, 9.0, 4.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let bytes = 6 * std::mem::size_of::<[f64; 2]>();
        // Four rectangle vertices and six indices fit in one eight-element
        // callback reservation per buffer, overlapping the final output.
        let scratch = 8 * std::mem::size_of::<Point>() + 8 * std::mem::size_of::<u32>();
        let mut owner = super::super::support::PreparationHostBudget::new();
        owner
            .reserve(
                super::super::support::MAX_SHADOW_HOST_BYTES - bytes - scratch,
                &[7, 2],
            )
            .unwrap();
        let mut graph_work = 0;
        let mut clip_pairs = 0;
        let mut work = super::super::shadow::NeutralPreparationWork::new(
            &mut graph_work,
            &mut clip_pairs,
            &mut owner,
        );
        let mesh = neutral_support_admitted(&source, 7, 1.0, 4096, &mut work, &[7, 2])
            .unwrap()
            .unwrap();
        assert_eq!(mesh.triangles.len(), 6);
        assert_eq!(mesh.triangles.capacity(), 6);
        assert_eq!(mesh.bounds, [-10.0, -2.0, -1.0, 2.0]);
        assert_eq!(
            owner.live(),
            super::super::support::MAX_SHADOW_HOST_BYTES - scratch
        );
        assert_eq!(owner.peak(), super::super::support::MAX_SHADOW_HOST_BYTES);
        owner.release(mesh.triangles.capacity() * std::mem::size_of::<[f64; 2]>());
        assert_eq!(
            owner.live(),
            super::super::support::MAX_SHADOW_HOST_BYTES - bytes - scratch
        );
    }

    #[test]
    fn admitted_clip_source_rejects_one_short_before_output_allocation() {
        let source = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let bytes = 6 * std::mem::size_of::<[f64; 2]>();
        // Four rectangle vertices and six indices fit in one eight-element
        // callback reservation per buffer, overlapping the final output.
        let scratch = 8 * std::mem::size_of::<Point>() + 8 * std::mem::size_of::<u32>();
        let mut owner = super::super::support::PreparationHostBudget::new();
        let prior = super::super::support::MAX_SHADOW_HOST_BYTES - bytes - scratch + 1;
        owner.reserve(prior, &[9, 3]).unwrap();
        let mut graph_work = 0;
        let mut clip_pairs = 0;
        let mut work = super::super::shadow::NeutralPreparationWork::new(
            &mut graph_work,
            &mut clip_pairs,
            &mut owner,
        );
        assert!(matches!(
            neutral_support_admitted(&source, 9, 1.0, 4096, &mut work, &[9, 3]),
            Err(Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [9, 3]
        ));
        // A source with no physical support does not lease an output vector.
        let collapsed = Render2dItem::new(
            source.primitive().clone(),
            Render2dAffineTransform::new(1.0, 1.0, 1.0, 1.0, 0.0, 0.0).unwrap(),
            vec![],
            Render2dOpacity::OPAQUE,
        );
        assert!(
            neutral_support_admitted(&collapsed, 9, 1.0, 4096, &mut work, &[9, 3])
                .unwrap()
                .is_none()
        );
        assert_eq!((owner.live(), owner.peak()), (prior, prior + scratch));
    }

    #[test]
    fn neutral_support_applies_local_to_parent_before_parent_frame_effects() {
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 1.0, 1.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::new(2.0, 0.0, 0.0, 1.0, 4.0, 0.0).unwrap(),
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let support = neutral_support(&item, 0, 1.0, 4096).unwrap().unwrap();
        assert_eq!(support.bounds, [4.0, 0.0, 6.0, 1.0]);
        assert_eq!(support.triangles.len(), 6);
        assert!(matches!(
            neutral_support(&item, 0, 1.0, 32),
            Err(Render2dExecutionError::Vector {
                root_index: 0,
                kind: Render2dVectorError::ResourceLimit
            })
        ));
    }

    #[test]
    fn neutral_support_preserves_transparent_stroke_and_singular_empty_geometry() {
        let stroke = Render2dItem::new(
            Render2dPrimitive::Stroke {
                shape: Render2dShape::rect(Render2dRect::new(-2.0, -2.0, 4.0, 4.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
                style: Render2dStrokeStyle::new(
                    2.0,
                    Render2dStrokeCap::Round,
                    Render2dStrokeJoin::Round,
                    4.0,
                )
                .unwrap(),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let support = neutral_support(&stroke, 0, 1.0, 1_048_576)
            .unwrap()
            .unwrap();
        assert!(support.bounds[0] < -2.0 && support.bounds[2] > 2.0);
        let collapsed = Render2dItem::new(
            stroke.primitive().clone(),
            Render2dAffineTransform::new(1.0, 1.0, 1.0, 1.0, 0.0, 0.0).unwrap(),
            vec![],
            Render2dOpacity::OPAQUE,
        );
        assert!(
            neutral_support(&collapsed, 0, 1.0, 1_048_576)
                .unwrap()
                .is_none()
        );
    }

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
