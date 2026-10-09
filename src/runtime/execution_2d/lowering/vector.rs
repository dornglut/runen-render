//! Reusable source-backed GPU vertex submission for one correlated sample pass.
use super::{create_vertex_buffer, f32_from_f64, f32_from_u32, gpu, linear_color, vertex_count};
use crate::composition_2d::Render2dBrush;
use crate::execution_2d::{Render2dExecutionError, Render2dVectorError};
use runen_gpu::*;

pub(super) fn vector_draw(
    pipeline: GpuRenderPipelineDescriptor,
    bindings: GpuRuntimeBindingSet,
    vertices: &[f32],
    extent: [u32; 2],
    resources: &mut GpuResourceScope,
) -> Result<GpuRenderDraw, Render2dExecutionError> {
    let buffer = create_vertex_buffer(resources, vertices)?;
    let binding = GpuVertexBufferBinding::new(
        0,
        &buffer,
        GpuBufferRange::whole(&buffer).map_err(|e| gpu("vector range", e))?,
    )
    .map_err(|e| gpu("vector binding", e))?;
    GpuRenderDraw::new(
        pipeline,
        bindings,
        [binding],
        None,
        GpuDrawIntent::direct(
            GpuDrawRange::new(0, vertex_count(vertices)?)
                .map_err(|e| gpu("vector draw range", e))?,
            GpuDrawRange::new(0, 1).map_err(|e| gpu("vector instance", e))?,
        ),
        GpuViewport::new(
            0.0,
            0.0,
            f32_from_u32(extent[0]),
            f32_from_u32(extent[1]),
            0.0,
            1.0,
        )
        .map_err(|e| gpu("vector viewport", e))?,
        GpuScissorRect::new(0, 0, extent[0], extent[1]).map_err(|e| gpu("vector scissor", e))?,
        GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).map_err(|e| gpu("vector blend constant", e))?,
        0,
    )
    .map_err(|e| gpu("vector draw", e))
}

// Four 16-byte header rows followed by stable authored stop pairs. Every stop is
// (offset, unused, unused, unused) and premultiplied linear RGBA, with no sorting
// or deduplication. This is private GPU representation, not an authoring format.
pub(super) fn gradient_payload(
    mesh: &crate::runtime::execution_2d::vector::VectorMesh,
) -> Result<Option<Vec<f32>>, Render2dExecutionError> {
    let (kind, geometry, stops) = match &mesh.brush {
        Render2dBrush::Solid(_) => return Ok(None),
        Render2dBrush::Linear(linear) => (
            1.0_f32,
            [
                linear.start().x(),
                linear.start().y(),
                linear.end().x() - linear.start().x(),
                linear.end().y() - linear.start().y(),
            ],
            linear.stops(),
        ),
        Render2dBrush::Radial(radial) => (
            2.0_f32,
            [
                radial.center().x(),
                radial.center().y(),
                radial.radius(),
                0.0,
            ],
            radial.stops(),
        ),
    };
    let fail = |kind| crate::runtime::execution_2d::vector::error(mesh.root_index, kind);
    const MAX_STOPS: usize = 256;
    if stops.as_slice().len() > MAX_STOPS {
        return Err(fail(Render2dVectorError::ResourceLimit));
    }
    let [a, b, c, d, tx, ty] = mesh.transform.components();
    let determinant = a.mul_add(d, -(b * c));
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(fail(Render2dVectorError::PrecisionLimit));
    }
    let inv = [
        d / determinant / mesh.raster_scale,
        -c / determinant / mesh.raster_scale,
        (c.mul_add(ty, -(d * tx))) / determinant,
        -b / determinant / mesh.raster_scale,
        a / determinant / mesh.raster_scale,
        (b.mul_add(tx, -(a * ty))) / determinant,
    ];
    let mut words = Vec::with_capacity(16 + stops.as_slice().len() * 8);
    let pack = |words: &mut Vec<f32>, value: f64| -> Result<(), Render2dExecutionError> {
        let narrowed = value as f32;
        if !narrowed.is_finite()
            || (f64::from(narrowed) - value).abs() > (1.0e-7_f64).max(value.abs() * 1.0e-7)
        {
            return Err(fail(Render2dVectorError::PrecisionLimit));
        }
        words.push(narrowed);
        Ok(())
    };
    for number in [
        f64::from(kind),
        stops.as_slice().len() as f64,
        f64::from(mesh.bounds[0]),
        f64::from(mesh.bounds[1]),
        inv[0],
        inv[1],
        inv[2],
        mesh.opacity,
        inv[3],
        inv[4],
        inv[5],
        0.0,
        geometry[0],
        geometry[1],
        geometry[2],
        geometry[3],
    ] {
        pack(&mut words, number)?;
    }
    if kind == 1.0 {
        let len_sq =
            f64::from(words[14]).mul_add(f64::from(words[14]), f64::from(words[15]).powi(2));
        if !len_sq.is_finite() || len_sq < 1.0e-16 {
            return Err(fail(Render2dVectorError::PrecisionLimit));
        }
    } else if words[14] <= 1.0e-8 {
        return Err(fail(Render2dVectorError::PrecisionLimit));
    }
    let mut previous = None;
    for stop in stops.as_slice() {
        let offset = stop.offset();
        let narrowed = offset as f32;
        // Distinct authored hard-stop coordinates MUST NOT collapse into one GPU coordinate.
        if previous
            .is_some_and(|(prior, narrowed_prior)| prior < offset && narrowed_prior == narrowed)
        {
            return Err(fail(Render2dVectorError::PrecisionLimit));
        }
        pack(&mut words, offset)?;
        for _ in 0..3 {
            pack(&mut words, 0.0)?;
        }
        let mut rgba = linear_color(stop.color());
        let alpha = rgba[3];
        for channel in &mut rgba[..3] {
            *channel *= alpha;
        }
        words.extend_from_slice(&rgba);
        previous = Some((offset, narrowed));
    }
    Ok(Some(words))
}
