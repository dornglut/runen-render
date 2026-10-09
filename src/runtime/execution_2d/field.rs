//! Private deterministic shaped-outline and MSDF field realization.

use crate::composition_2d::{
    Render2dFillRule, Render2dPath, Render2dPathCommand, Render2dPoint, Render2dResourceId,
    Render2dShape, Render2dShapedTextResource,
};
use crate::execution_2d::{
    Render2dExecutionError, Render2dShapedTextError, Render2dUnsupportedGlyphKind,
};
use bymsdfgen_core::{
    Bitmap, Contour, DistanceMapping, EdgeSegment, MsdfGeneratorConfig, Projection, Range,
    SdfTransformation, Shape, Vector2, coloring::edge_coloring_simple, generate_msdf,
};
use skrifa::raw::TableProvider;
use skrifa::{
    FontRef, MetadataProvider,
    instance::{LocationRef, NormalizedCoord, Size},
    outline::{DrawSettings, OutlinePen},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

const FIELD_RANGE: f64 = 4.0;
const FIELD_BORDER: f64 = 4.0;
// The 4-byte retained RGBA field is preceded by a 3-channel f32 MSDF
// generation bitmap. Bound aggregate field area before either allocation.
pub(super) const MAX_TEXT_FIELD_BYTES: u64 = 32 * 1024 * 1024;
// The same immutable glyph outlines must be retained for F3F neutral geometry.
// They count against the existing aggregate shaped-text preparation budget.
const MAX_OUTLINE_VERBS: usize = 1_048_576;

#[derive(Debug, Default)]
pub(super) struct FieldBudget {
    bytes: u64,
}

impl FieldBudget {
    fn charge(
        &mut self,
        resource_id: Render2dResourceId,
        glyph_id: u32,
        width: u32,
        height: u32,
        outline_verbs: usize,
    ) -> Result<(), Render2dShapedTextError> {
        let fail = || Render2dShapedTextError::FieldBudgetExceeded {
            resource_id,
            glyph_id,
            maximum_bytes: MAX_TEXT_FIELD_BYTES,
        };
        let bytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(4))
            .and_then(|pixels| {
                u64::try_from(outline_verbs)
                    .ok()
                    .and_then(|verbs| verbs.checked_mul(std::mem::size_of::<OutlineVerb>() as u64))
                    .and_then(|outline_bytes| pixels.checked_add(outline_bytes))
            })
            .ok_or_else(fail)?;
        let next = self.bytes.checked_add(bytes).ok_or_else(fail)?;
        if next > MAX_TEXT_FIELD_BYTES {
            return Err(fail());
        }
        self.bytes = next;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) enum QualityTier {
    P16,
    P24,
    P32,
    P48,
}

impl QualityTier {
    pub(super) fn select(font_size: f64, raster_scale: f64) -> Option<Self> {
        let effective = font_size * raster_scale;
        if !effective.is_finite() || effective <= 0.0 || effective > f64::from(f32::MAX) {
            return None;
        }
        Some(if effective <= 16.0 {
            Self::P16
        } else if effective <= 24.0 {
            Self::P24
        } else if effective <= 32.0 {
            Self::P32
        } else {
            Self::P48
        })
    }

    pub(super) const fn pixels_per_em(self) -> f64 {
        match self {
            Self::P16 => 16.0,
            Self::P24 => 24.0,
            Self::P32 => 32.0,
            Self::P48 => 48.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct FieldSetKey {
    resource_id: Render2dResourceId,
    quality: QualityTier,
}

impl FieldSetKey {
    pub(super) const fn new(resource_id: Render2dResourceId, quality: QualityTier) -> Self {
        Self {
            resource_id,
            quality,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct GlyphField {
    glyph_id: u32,
    origin_x: f64,
    origin_y: f64,
    width: u32,
    height: u32,
    rgba8: Arc<[u8]>,
    /// Unhinted, variation/synthesis-resolved immutable outline, never an
    /// approximate MSDF or a text foreground-alpha footprint.
    outline: Arc<[OutlineVerb]>,
}

impl GlyphField {
    pub(super) const fn glyph_id(&self) -> u32 {
        self.glyph_id
    }

    pub(super) const fn origin_x(&self) -> f64 {
        self.origin_x
    }

    pub(super) const fn origin_y(&self) -> f64 {
        self.origin_y
    }

    pub(super) const fn width(&self) -> u32 {
        self.width
    }

    pub(super) const fn height(&self) -> u32 {
        self.height
    }

    pub(super) fn rgba8(&self) -> &[u8] {
        &self.rgba8
    }

    /// Reuses the exact F2 retained outline authority for F3F neutral path
    /// tessellation. Coordinates are normalized em units; glyph placement and
    /// font-size/parent transforms belong to the receiving F1 item.
    #[allow(
        dead_code,
        reason = "F3F neutral shaped-text support awaiting group lowering"
    )]
    pub(super) fn neutral_shape(
        &self,
        resource_id: Render2dResourceId,
    ) -> Result<Render2dShape, Render2dExecutionError> {
        let invalid = || Render2dShapedTextError::InvalidOutline {
            resource_id,
            glyph_id: self.glyph_id,
        };
        let mut commands = Vec::new();
        commands
            .try_reserve_exact(self.outline.len())
            .map_err(|_| invalid())?;
        let point = |p: OutlinePoint| Render2dPoint::new(p.x, p.y).map_err(|_| invalid());
        for verb in self.outline.iter().copied() {
            commands.push(match verb {
                OutlineVerb::MoveTo(to) => Render2dPathCommand::MoveTo(point(to)?),
                OutlineVerb::LineTo(to) => Render2dPathCommand::LineTo(point(to)?),
                OutlineVerb::QuadraticTo { control, to } => Render2dPathCommand::QuadraticTo {
                    control: point(control)?,
                    to: point(to)?,
                },
                OutlineVerb::CubicTo {
                    control1,
                    control2,
                    to,
                } => Render2dPathCommand::CubicTo {
                    control1: point(control1)?,
                    control2: point(control2)?,
                    to: point(to)?,
                },
                OutlineVerb::Close => Render2dPathCommand::Close,
            });
        }
        let path = Render2dPath::new(Render2dFillRule::NonZero, commands).map_err(|_| invalid())?;
        Ok(Render2dShape::path(path))
    }
}

#[derive(Clone, Debug)]
pub(super) struct ResourceFields {
    by_glyph: BTreeMap<u32, Option<Arc<GlyphField>>>,
}

impl ResourceFields {
    pub(super) fn glyph(&self, glyph_id: u32) -> Option<&Arc<GlyphField>> {
        self.by_glyph.get(&glyph_id).and_then(Option::as_ref)
    }

    pub(super) fn charge_cached(
        &self,
        resource_id: Render2dResourceId,
        budget: &mut FieldBudget,
    ) -> Result<(), Render2dShapedTextError> {
        for field in self.by_glyph.values().flatten() {
            budget.charge(
                resource_id,
                field.glyph_id(),
                field.width(),
                field.height(),
                field.outline.len(),
            )?;
        }
        Ok(())
    }

    pub(super) fn validate_texture_limit(
        &self,
        resource_id: Render2dResourceId,
        maximum: u32,
    ) -> Result<(), Render2dShapedTextError> {
        for field in self.by_glyph.values().flatten() {
            if field.width() > maximum || field.height() > maximum {
                return Err(Render2dShapedTextError::GlyphExtentExceedsLimit {
                    resource_id,
                    glyph_id: field.glyph_id(),
                    width: field.width(),
                    height: field.height(),
                    maximum,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OutlinePoint {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum OutlineVerb {
    MoveTo(OutlinePoint),
    LineTo(OutlinePoint),
    QuadraticTo {
        control: OutlinePoint,
        to: OutlinePoint,
    },
    CubicTo {
        control1: OutlinePoint,
        control2: OutlinePoint,
        to: OutlinePoint,
    },
    Close,
}

#[derive(Clone, Debug, PartialEq)]
struct GlyphOutline {
    verbs: Arc<[OutlineVerb]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ContourState {
    Closed,
    OpenMoveOnly,
    OpenSegmentBearing,
}

impl ContourState {
    const fn is_open(self) -> bool {
        !matches!(self, Self::Closed)
    }
}

struct OutlinePathPen {
    verbs: Vec<OutlineVerb>,
    contour: ContourState,
    has_any_segment: bool,
    invalid: bool,
    skew: f64,
}

impl OutlinePathPen {
    fn new(faux_skew: Option<f64>) -> Self {
        let skew = faux_skew.map_or(0.0, f64::tan);
        Self {
            verbs: Vec::new(),
            contour: ContourState::Closed,
            has_any_segment: false,
            invalid: !skew.is_finite(),
            skew,
        }
    }

    fn point(&mut self, x: f32, y: f32) -> Option<OutlinePoint> {
        let y = -f64::from(y);
        let x = self.skew.mul_add(y, f64::from(x));
        if !x.is_finite() || !y.is_finite() {
            self.invalid = true;
            return None;
        }
        Some(OutlinePoint { x, y })
    }

    fn push_verb(&mut self, verb: OutlineVerb) {
        if self.verbs.len() >= MAX_OUTLINE_VERBS {
            self.invalid = true;
        } else {
            self.verbs.push(verb);
        }
    }

    fn finish_contour(&mut self) {
        if matches!(self.contour, ContourState::OpenSegmentBearing) {
            self.push_verb(OutlineVerb::Close);
        }
        self.contour = ContourState::Closed;
    }

    fn finish(mut self) -> Result<Option<GlyphOutline>, ()> {
        self.finish_contour();
        if self.invalid {
            return Err(());
        }
        Ok(self.has_any_segment.then(|| GlyphOutline {
            verbs: self.verbs.into(),
        }))
    }

    const fn mark_segment(&mut self) {
        self.contour = ContourState::OpenSegmentBearing;
        self.has_any_segment = true;
    }
}

impl OutlinePen for OutlinePathPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.finish_contour();
        if let Some(point) = self.point(x, y) {
            self.push_verb(OutlineVerb::MoveTo(point));
            self.contour = ContourState::OpenMoveOnly;
        }
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let Some(to) = self.point(x, y) else {
            return;
        };
        if !self.contour.is_open() {
            self.invalid = true;
            return;
        }
        self.push_verb(OutlineVerb::LineTo(to));
        self.mark_segment();
    }

    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let Some(control) = self.point(cx, cy) else {
            return;
        };
        let Some(to) = self.point(x, y) else {
            return;
        };
        if !self.contour.is_open() {
            self.invalid = true;
            return;
        }
        self.push_verb(OutlineVerb::QuadraticTo { control, to });
        self.mark_segment();
    }

    fn curve_to(&mut self, c0x: f32, c0y: f32, c1x: f32, c1y: f32, x: f32, y: f32) {
        let Some(control1) = self.point(c0x, c0y) else {
            return;
        };
        let Some(control2) = self.point(c1x, c1y) else {
            return;
        };
        let Some(to) = self.point(x, y) else {
            return;
        };
        if !self.contour.is_open() {
            self.invalid = true;
            return;
        }
        self.push_verb(OutlineVerb::CubicTo {
            control1,
            control2,
            to,
        });
        self.mark_segment();
    }

    fn close(&mut self) {
        self.finish_contour();
    }
}

pub(super) fn realize(
    resource_id: Render2dResourceId,
    resource: &Render2dShapedTextResource,
    quality: QualityTier,
    max_texture_dimension_2d: u32,
    budget: &mut FieldBudget,
) -> Result<ResourceFields, Render2dExecutionError> {
    if resource.font().faux_bold() {
        return Err(Render2dShapedTextError::UnsupportedGlyph {
            resource_id,
            glyph_id: None,
            kind: Render2dUnsupportedGlyphKind::FauxBold,
        }
        .into());
    }

    let font = FontRef::from_index(resource.font().bytes(), resource.font().face_index())
        .map_err(|_| Render2dShapedTextError::InvalidFont { resource_id })?;
    let glyph_count = font
        .maxp()
        .map_err(|_| Render2dShapedTextError::InvalidFont { resource_id })?
        .num_glyphs();
    let upem = font
        .head()
        .map_err(|_| Render2dShapedTextError::InvalidFont { resource_id })?
        .units_per_em();
    if !(16..=16384).contains(&upem) {
        return Err(Render2dShapedTextError::InvalidFont { resource_id }.into());
    }
    if resource.font().normalized_coords().len() > font.axes().len()
        || resource
            .font()
            .normalized_coords()
            .iter()
            .any(|coordinate| !(-16384..=16384).contains(coordinate))
    {
        return Err(Render2dShapedTextError::InvalidFont { resource_id }.into());
    }
    let normalized = resource
        .font()
        .normalized_coords()
        .iter()
        .copied()
        .map(NormalizedCoord::from_bits)
        .collect::<Vec<_>>();
    let location = LocationRef::new(&normalized);
    let outlines = font.outline_glyphs();
    let intrinsic = super::intrinsic::IntrinsicGlyphs::new(&font, resource_id)?;
    finite_f32(resource.font_size())
        .ok_or(crate::execution_2d::Render2dUnsupportedContent::FontSize { resource_id })?;
    let mut seen = BTreeSet::new();
    let mut by_glyph = BTreeMap::new();

    for glyph in resource.glyphs() {
        if !seen.insert(glyph.id()) {
            continue;
        }
        let glyph_id = skrifa::GlyphId::new(glyph.id());
        if glyph.id() >= u32::from(glyph_count) {
            return Err(Render2dShapedTextError::InvalidOutline {
                resource_id,
                glyph_id: glyph.id(),
            }
            .into());
        }

        intrinsic.admit(glyph_id)?;

        let outline = if let Some(outline) = outlines.get(glyph_id) {
            let mut pen = OutlinePathPen::new(resource.font().faux_skew());
            outline
                .draw(DrawSettings::unhinted(Size::new(1.0), location), &mut pen)
                .map_err(|_| Render2dShapedTextError::InvalidOutline {
                    resource_id,
                    glyph_id: glyph.id(),
                })?;
            pen.finish()
                .map_err(|()| Render2dShapedTextError::InvalidOutline {
                    resource_id,
                    glyph_id: glyph.id(),
                })?
        } else {
            return Err(Render2dShapedTextError::InvalidOutline {
                resource_id,
                glyph_id: glyph.id(),
            }
            .into());
        };

        let field = outline
            .map(|outline| {
                generate_field(
                    resource_id,
                    glyph.id(),
                    &outline,
                    quality,
                    max_texture_dimension_2d,
                    budget,
                )
                .map(Arc::new)
            })
            .transpose()?;
        by_glyph.insert(glyph.id(), field);
    }

    Ok(ResourceFields { by_glyph })
}

fn finite_f32(value: f64) -> Option<f32> {
    if !value.is_finite() || value <= 0.0 || value > f64::from(f32::MAX) {
        return None;
    }
    let narrowed = narrow_f32(value);
    (narrowed > 0.0).then_some(narrowed)
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "validated finite f64 semantic scalars narrow only at the private font/GPU realization boundary"
)]
fn narrow_f32(value: f64) -> f32 {
    value as f32
}

fn msdf_shape(outline: &GlyphOutline) -> Result<Shape, ()> {
    let mut shape = Shape::new();
    let mut contour = None::<Contour>;
    let mut current = None::<Vector2>;
    let mut start = None::<Vector2>;

    for verb in outline.verbs.iter().copied() {
        match verb {
            OutlineVerb::MoveTo(point) => {
                finish_contour(&mut shape, &mut contour, &mut current, &mut start);
                let point = Vector2::new(point.x, point.y);
                contour = Some(Contour::new());
                current = Some(point);
                start = Some(point);
            }
            OutlineVerb::LineTo(point) => {
                let Some(from) = current else {
                    return Err(());
                };
                let Some(active) = contour.as_mut() else {
                    return Err(());
                };
                let to = Vector2::new(point.x, point.y);
                active.add_edge(EdgeSegment::line(from, to));
                current = Some(to);
            }
            OutlineVerb::QuadraticTo { control, to } => {
                let Some(from) = current else {
                    return Err(());
                };
                let Some(active) = contour.as_mut() else {
                    return Err(());
                };
                let control = Vector2::new(control.x, control.y);
                let to = Vector2::new(to.x, to.y);
                active.add_edge(EdgeSegment::quadratic(from, control, to));
                current = Some(to);
            }
            OutlineVerb::CubicTo {
                control1,
                control2,
                to,
            } => {
                let Some(from) = current else {
                    return Err(());
                };
                let Some(active) = contour.as_mut() else {
                    return Err(());
                };
                let control1 = Vector2::new(control1.x, control1.y);
                let control2 = Vector2::new(control2.x, control2.y);
                let to = Vector2::new(to.x, to.y);
                active.add_edge(EdgeSegment::cubic(from, control1, control2, to));
                current = Some(to);
            }
            OutlineVerb::Close => {
                finish_contour(&mut shape, &mut contour, &mut current, &mut start);
            }
        }
    }
    finish_contour(&mut shape, &mut contour, &mut current, &mut start);
    Ok(shape)
}

fn finish_contour(
    shape: &mut Shape,
    contour: &mut Option<Contour>,
    current: &mut Option<Vector2>,
    start: &mut Option<Vector2>,
) {
    let Some(mut finished) = contour.take() else {
        *current = None;
        *start = None;
        return;
    };
    if let (Some(current), Some(start)) = (current.take(), start.take())
        && current != start
    {
        finished.add_edge(EdgeSegment::line(current, start));
    }
    if !finished.is_empty() {
        shape.add_contour(finished);
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "MSDF extents and quantized channels narrow only after explicit finite/non-negative checks"
)]
fn generate_field(
    resource_id: Render2dResourceId,
    glyph_id: u32,
    outline: &GlyphOutline,
    quality: QualityTier,
    max_texture_dimension_2d: u32,
    budget: &mut FieldBudget,
) -> Result<GlyphField, Render2dExecutionError> {
    let mut shape = msdf_shape(outline).map_err(|()| Render2dShapedTextError::InvalidOutline {
        resource_id,
        glyph_id,
    })?;
    if shape.contours.is_empty() {
        return Err(Render2dShapedTextError::InvalidOutline {
            resource_id,
            glyph_id,
        }
        .into());
    }
    if !shape.validate() {
        return Err(Render2dShapedTextError::InvalidOutline {
            resource_id,
            glyph_id,
        }
        .into());
    }
    shape.normalize();
    shape.orient_contours();
    edge_coloring_simple(&mut shape, 3.0, 0);

    let bounds = shape.get_bounds(0.0);
    let scale = quality.pixels_per_em();
    let origin_x = bounds.l - FIELD_BORDER / scale;
    let origin_y = bounds.b - FIELD_BORDER / scale;
    let width_f = (((bounds.r - bounds.l) + 2.0 * FIELD_BORDER / scale) * scale)
        .ceil()
        .max(1.0);
    let height_f = (((bounds.t - bounds.b) + 2.0 * FIELD_BORDER / scale) * scale)
        .ceil()
        .max(1.0);
    if !width_f.is_finite()
        || !height_f.is_finite()
        || width_f > f64::from(u32::MAX)
        || height_f > f64::from(u32::MAX)
    {
        return Err(Render2dShapedTextError::GlyphExtentExceedsLimit {
            resource_id,
            glyph_id,
            width: u32::MAX,
            height: u32::MAX,
            maximum: max_texture_dimension_2d,
        }
        .into());
    }
    let width = width_f as u32;
    let height = height_f as u32;
    if width > max_texture_dimension_2d || height > max_texture_dimension_2d {
        return Err(Render2dShapedTextError::GlyphExtentExceedsLimit {
            resource_id,
            glyph_id,
            width,
            height,
            maximum: max_texture_dimension_2d,
        }
        .into());
    }

    let width_usize =
        usize::try_from(width).map_err(|_| Render2dShapedTextError::GlyphExtentExceedsLimit {
            resource_id,
            glyph_id,
            width,
            height,
            maximum: max_texture_dimension_2d,
        })?;
    let height_usize =
        usize::try_from(height).map_err(|_| Render2dShapedTextError::GlyphExtentExceedsLimit {
            resource_id,
            glyph_id,
            width,
            height,
            maximum: max_texture_dimension_2d,
        })?;

    // Reject aggregate work before MSDF's temporary 3xf32 bitmap is created.
    // This is the same per-invocation accounting used for preexisting cache entries.
    budget.charge(resource_id, glyph_id, width, height, outline.verbs.len())?;

    let projection = Projection::new(Vector2::splat(scale), Vector2::new(-origin_x, -origin_y));
    let mapping = DistanceMapping::from_range(Range::symmetric(FIELD_RANGE / scale));
    let transformation = SdfTransformation::new(projection, mapping);
    let mut bitmap: Bitmap<f32, 3> = Bitmap::new(width_usize, height_usize);
    generate_msdf(
        &mut bitmap,
        &shape,
        &transformation,
        &MsdfGeneratorConfig::default(),
    );

    let capacity = width_usize
        .checked_mul(height_usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(Render2dShapedTextError::GlyphExtentExceedsLimit {
            resource_id,
            glyph_id,
            width,
            height,
            maximum: max_texture_dimension_2d,
        })?;
    let mut rgba8 = Vec::with_capacity(capacity);
    for y in 0..height_usize {
        for x in 0..width_usize {
            for channel in bitmap.pixel(x, y) {
                if !channel.is_finite() {
                    return Err(Render2dShapedTextError::InvalidOutline {
                        resource_id,
                        glyph_id,
                    }
                    .into());
                }
                rgba8.push((channel.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
            rgba8.push(u8::MAX);
        }
    }

    Ok(GlyphField {
        glyph_id,
        origin_x,
        origin_y,
        width,
        height,
        rgba8: rgba8.into(),
        outline: Arc::clone(&outline.verbs),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition_2d::{Render2dFontBinding, Render2dGlyph};

    const OUTLINE: &[u8] = include_bytes!("../../../tests/fixtures/f2_outline.ttf");

    fn resource(bytes: &[u8], glyph_id: u32) -> Render2dShapedTextResource {
        Render2dShapedTextResource::new(
            Render2dFontBinding::new(bytes.to_vec(), 0, Vec::new(), false, None).unwrap(),
            24.0,
            vec![Render2dGlyph::new(glyph_id, 0.0, 0.0, 24.0).unwrap()],
        )
        .unwrap()
    }

    fn realize_fixture(
        bytes: &[u8],
        glyph_id: u32,
    ) -> Result<ResourceFields, Render2dExecutionError> {
        realize(
            Render2dResourceId::new(1).unwrap(),
            &resource(bytes, glyph_id),
            QualityTier::P24,
            4096,
            &mut FieldBudget::default(),
        )
    }

    #[test]
    fn cumulative_msdf_budget_accounts_cached_and_new_fields_transactionally() {
        let id = Render2dResourceId::new(920).unwrap();
        let mut budget = FieldBudget::default();
        budget
            .charge(id, 1, 2048, 2048, 0)
            .expect("first sixteen MiB field");
        budget
            .charge(id, 2, 2048, 2048, 0)
            .expect("second sixteen MiB field");
        assert_eq!(budget.bytes, MAX_TEXT_FIELD_BYTES);
        assert!(matches!(
            budget.charge(id, 3, 1, 1, 0),
            Err(Render2dShapedTextError::FieldBudgetExceeded {
                resource_id,
                glyph_id: 3,
                maximum_bytes: MAX_TEXT_FIELD_BYTES,
            }) if resource_id == id
        ));
        assert_eq!(budget.bytes, MAX_TEXT_FIELD_BYTES);
        assert!(budget.charge(id, 4, u32::MAX, u32::MAX, 0).is_err());
        assert_eq!(budget.bytes, MAX_TEXT_FIELD_BYTES);
        let mut cached = BTreeMap::new();
        cached.insert(
            5,
            Some(Arc::new(GlyphField {
                glyph_id: 5,
                origin_x: 0.0,
                origin_y: 0.0,
                width: 2,
                height: 2,
                rgba8: Arc::from([0_u8; 16]),
                outline: Arc::from([]),
            })),
        );
        let fields = ResourceFields { by_glyph: cached };
        assert!(matches!(
            fields.charge_cached(id, &mut budget),
            Err(Render2dShapedTextError::FieldBudgetExceeded { glyph_id: 5, .. })
        ));
        assert_eq!(budget.bytes, MAX_TEXT_FIELD_BYTES);
    }

    #[test]
    fn neutral_text_shape_comes_from_immutable_outline_independently_of_msdf() {
        let source = realize_fixture(OUTLINE, 2).unwrap();
        let reproduced = realize_fixture(OUTLINE, 2).unwrap();
        let field = source.glyph(2).expect("fixture glyph has an outline");
        let again = reproduced.glyph(2).expect("recreated outline exists");
        assert!(!field.outline.is_empty());
        let id = Render2dResourceId::new(1).unwrap();
        let a = field.neutral_shape(id).unwrap();
        let b = again.neutral_shape(id).unwrap();
        assert_eq!(a, b);
        let path = a.as_path().unwrap();
        assert_eq!(path.fill_rule(), Render2dFillRule::NonZero);
        assert!(
            path.commands()
                .iter()
                .any(|v| matches!(v, Render2dPathCommand::Close))
        );
    }

    #[test]
    fn cached_field_budget_includes_retained_outline_work() {
        let id = Render2dResourceId::new(1).unwrap();
        let fields = realize_fixture(OUTLINE, 2).unwrap();
        let field = fields.glyph(2).unwrap();
        let mut budget = FieldBudget::default();
        fields.charge_cached(id, &mut budget).unwrap();
        assert_eq!(
            budget.bytes,
            u64::from(field.width()) * u64::from(field.height()) * 4
                + (field.outline.len() * std::mem::size_of::<OutlineVerb>()) as u64
        );
    }

    #[test]
    fn deterministic_fields_and_valid_empty_outlines_are_distinct_from_invalid_glyphs() {
        let first = realize_fixture(OUTLINE, 2).unwrap();
        let second = realize_fixture(OUTLINE, 2).unwrap();
        assert_eq!(
            first.glyph(2).unwrap().rgba8(),
            second.glyph(2).unwrap().rgba8()
        );
        assert!(realize_fixture(OUTLINE, 1).unwrap().glyph(1).is_none());
        assert!(matches!(
            realize_fixture(OUTLINE, u32::MAX),
            Err(Render2dExecutionError::ShapedText(
                Render2dShapedTextError::InvalidOutline { .. }
            ))
        ));
    }

    #[test]
    fn variation_coordinates_must_belong_to_the_bound_face() {
        let font = Render2dFontBinding::new(OUTLINE.to_vec(), 0, vec![8192], false, None).unwrap();
        let resource = Render2dShapedTextResource::new(
            font,
            24.0,
            vec![Render2dGlyph::new(2, 0.0, 0.0, 24.0).unwrap()],
        )
        .unwrap();
        assert!(matches!(
            realize(
                Render2dResourceId::new(1).unwrap(),
                &resource,
                QualityTier::P24,
                4096,
                &mut FieldBudget::default()
            ),
            Err(Render2dExecutionError::ShapedText(
                Render2dShapedTextError::InvalidFont { .. }
            ))
        ));
    }

    #[test]
    fn malformed_intrinsic_tables_cannot_disappear_into_outline_fallback() {
        for (font, tag) in [
            (
                include_bytes!("../../../tests/fixtures/f2_colrv0.ttf").as_slice(),
                b"COLR",
            ),
            (
                include_bytes!("../../../tests/fixtures/f2_svg.ttf").as_slice(),
                b"SVG ",
            ),
            (
                include_bytes!("../../../tests/fixtures/f2_bitmap.ttf").as_slice(),
                b"sbix",
            ),
        ] {
            let index = FontRef::new(font)
                .unwrap()
                .table_directory()
                .table_records()
                .iter()
                .position(|record| record.tag() == skrifa::raw::types::Tag::new(tag))
                .unwrap();
            let mut corrupt = font.to_vec();
            // A present but truncated table is malformed, never an absent intrinsic class.
            let length_offset = 12 + index * 16 + 12;
            corrupt[length_offset..length_offset + 4].copy_from_slice(&1_u32.to_be_bytes());
            assert!(matches!(
                realize_fixture(&corrupt, 2),
                Err(Render2dExecutionError::ShapedText(
                    Render2dShapedTextError::InvalidFont { .. }
                ))
            ));
        }
    }

    #[test]
    fn every_sbix_graphic_type_is_intrinsic_bitmap_content() {
        let mut font = include_bytes!("../../../tests/fixtures/f2_bitmap.ttf").to_vec();
        let png_tag = font.windows(4).position(|bytes| bytes == b"png ").unwrap();
        font[png_tag..png_tag + 4].copy_from_slice(b"jpg ");
        assert!(matches!(
            realize_fixture(&font, 2),
            Err(Render2dExecutionError::ShapedText(
                Render2dShapedTextError::UnsupportedGlyph {
                    kind: Render2dUnsupportedGlyphKind::Bitmap,
                    ..
                }
            ))
        ));
    }

    #[test]
    fn overflowing_bitmap_strike_count_is_not_an_empty_intrinsic_table() {
        let mut font = include_bytes!("../../../tests/fixtures/f2_bitmap.ttf").to_vec();
        let offset = usize::try_from(
            FontRef::new(&font)
                .unwrap()
                .table_directory()
                .table_records()
                .iter()
                .find(|record| record.tag() == skrifa::raw::types::Tag::new(b"sbix"))
                .unwrap()
                .offset(),
        )
        .unwrap();
        font[offset + 4..offset + 8].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(
            realize_fixture(&font, 2),
            Err(Render2dExecutionError::ShapedText(
                Render2dShapedTextError::InvalidFont { .. }
            ))
        ));
    }

    #[test]
    fn tier_depends_only_on_exact_size_and_scale_and_cached_fields_obey_later_limits() {
        assert_eq!(
            QualityTier::select(12.0, 2.0),
            QualityTier::select(24.0, 1.0)
        );
        assert_ne!(
            QualityTier::select(24.0, 1.0),
            QualityTier::select(24.0, 2.0)
        );
        let fields = realize_fixture(OUTLINE, 2).unwrap();
        assert!(matches!(
            fields.validate_texture_limit(Render2dResourceId::new(1).unwrap(), 1),
            Err(Render2dShapedTextError::GlyphExtentExceedsLimit { .. })
        ));
    }
}
