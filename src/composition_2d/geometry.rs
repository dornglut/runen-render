//! Source-neutral two-dimensional geometry semantics.

use core::{error::Error, fmt};
use std::sync::Arc;

/// One finite point in source-neutral two-dimensional logical coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dPoint {
    x: f64,
    y: f64,
}

impl Render2dPoint {
    /// Creates one finite logical point.
    ///
    /// # Errors
    ///
    /// Returns [`Render2dGeometryError::NonFiniteScalar`] when either component is non-finite.
    pub fn new(x: f64, y: f64) -> Result<Self, Render2dGeometryError> {
        if x.is_finite() && y.is_finite() {
            Ok(Self { x, y })
        } else {
            Err(Render2dGeometryError::NonFiniteScalar)
        }
    }

    /// Returns the x coordinate.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.x
    }

    /// Returns the y coordinate.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.y
    }
}

/// One finite non-negative logical rectangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dRect {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

impl Render2dRect {
    /// Creates one finite rectangle with non-negative extents.
    ///
    /// # Errors
    ///
    /// Returns a geometry error for non-finite components or negative extents.
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Result<Self, Render2dGeometryError> {
        if ![x, y, width, height].into_iter().all(f64::is_finite) {
            return Err(Render2dGeometryError::NonFiniteScalar);
        }
        if width < 0.0 || height < 0.0 {
            return Err(Render2dGeometryError::NegativeExtent);
        }
        if !(x + width).is_finite() || !(y + height).is_finite() {
            return Err(Render2dGeometryError::ExtentOverflow);
        }
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }

    /// Returns the left coordinate.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.x
    }

    /// Returns the top coordinate.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.y
    }

    /// Returns the non-negative width.
    #[must_use]
    pub const fn width(self) -> f64 {
        self.width
    }

    /// Returns the non-negative height.
    #[must_use]
    pub const fn height(self) -> f64 {
        self.height
    }

    /// Returns whether the rectangle has no positive-area interior.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }
}

/// Finite non-negative circular corner radii.
///
/// A standalone radius tuple is not yet rectangle-relative semantic geometry.
/// [`Render2dShape::rounded_rect`] applies the single-factor corner-overlap
/// normalization and stores only the canonical renderer-semantic radii.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dCornerRadii {
    top_left: f64,
    top_right: f64,
    bottom_right: f64,
    bottom_left: f64,
}

impl Render2dCornerRadii {
    /// Creates finite non-negative corner radii.
    ///
    /// # Errors
    ///
    /// Returns a geometry error for non-finite or negative radii.
    pub fn new(
        top_left: f64,
        top_right: f64,
        bottom_right: f64,
        bottom_left: f64,
    ) -> Result<Self, Render2dGeometryError> {
        let values = [top_left, top_right, bottom_right, bottom_left];
        if !values.into_iter().all(f64::is_finite) {
            return Err(Render2dGeometryError::NonFiniteScalar);
        }
        if values.into_iter().any(|value| value < 0.0) {
            return Err(Render2dGeometryError::NegativeRadius);
        }
        Ok(Self {
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        })
    }

    /// Returns the top-left radius.
    #[must_use]
    pub const fn top_left(self) -> f64 {
        self.top_left
    }

    /// Returns the top-right radius.
    #[must_use]
    pub const fn top_right(self) -> f64 {
        self.top_right
    }

    /// Returns the bottom-right radius.
    #[must_use]
    pub const fn bottom_right(self) -> f64 {
        self.bottom_right
    }

    /// Returns the bottom-left radius.
    #[must_use]
    pub const fn bottom_left(self) -> f64 {
        self.bottom_left
    }
}

/// Finite affine transform mapping local 2D coordinates into one parent coordinate space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dAffineTransform {
    m11: f64,
    m12: f64,
    m21: f64,
    m22: f64,
    tx: f64,
    ty: f64,
}

impl Render2dAffineTransform {
    /// Identity transform.
    pub const IDENTITY: Self = Self {
        m11: 1.0,
        m12: 0.0,
        m21: 0.0,
        m22: 1.0,
        tx: 0.0,
        ty: 0.0,
    };

    /// Creates one finite affine transform.
    ///
    /// # Errors
    ///
    /// Returns [`Render2dGeometryError::NonFiniteScalar`] when a component is non-finite.
    pub fn new(
        m11: f64,
        m12: f64,
        m21: f64,
        m22: f64,
        tx: f64,
        ty: f64,
    ) -> Result<Self, Render2dGeometryError> {
        if [m11, m12, m21, m22, tx, ty].into_iter().all(f64::is_finite) {
            Ok(Self {
                m11,
                m12,
                m21,
                m22,
                tx,
                ty,
            })
        } else {
            Err(Render2dGeometryError::NonFiniteScalar)
        }
    }

    /// Creates one finite translation.
    ///
    /// # Errors
    ///
    /// Returns [`Render2dGeometryError::NonFiniteScalar`] for a non-finite offset.
    pub fn translation(x: f64, y: f64) -> Result<Self, Render2dGeometryError> {
        Self::new(1.0, 0.0, 0.0, 1.0, x, y)
    }

    /// Returns `(m11, m12, m21, m22, tx, ty)`.
    #[must_use]
    pub const fn components(self) -> [f64; 6] {
        [self.m11, self.m12, self.m21, self.m22, self.tx, self.ty]
    }
}

/// Fill rule applied to structural path contours.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum Render2dFillRule {
    /// Non-zero winding fill.
    #[default]
    NonZero,
    /// Even-odd parity fill.
    EvenOdd,
}

/// One exact structural path command.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Render2dPathCommand {
    /// Starts a new contour.
    MoveTo(Render2dPoint),
    /// Adds one straight segment.
    LineTo(Render2dPoint),
    /// Adds one quadratic Bézier segment.
    QuadraticTo {
        /// Control point.
        control: Render2dPoint,
        /// Segment endpoint.
        to: Render2dPoint,
    },
    /// Adds one cubic Bézier segment.
    CubicTo {
        /// First control point.
        control1: Render2dPoint,
        /// Second control point.
        control2: Render2dPoint,
        /// Segment endpoint.
        to: Render2dPoint,
    },
    /// Explicitly closes the current segment-bearing contour.
    Close,
}

/// Immutable validated structural path content.
///
/// Empty and move-only paths are valid non-painting semantic values. Point-degenerate authored
/// segments remain structural content. Open segment-bearing contours remain structurally open;
/// fill evaluation derives the implicit closing edge required by the selected fill rule without
/// rewriting this structural value.
#[derive(Clone, Debug, PartialEq)]
pub struct Render2dPath {
    fill_rule: Render2dFillRule,
    commands: Arc<[Render2dPathCommand]>,
}

impl Render2dPath {
    /// Validates and freezes exact authored path content.
    ///
    /// # Errors
    ///
    /// Returns a geometry error for malformed contour ordering.
    pub fn new(
        fill_rule: Render2dFillRule,
        commands: impl Into<Vec<Render2dPathCommand>>,
    ) -> Result<Self, Render2dGeometryError> {
        let commands = commands.into();
        validate_path_commands(&commands)?;
        Ok(Self {
            fill_rule,
            commands: commands.into(),
        })
    }

    /// Returns the exact authored fill rule.
    #[must_use]
    pub const fn fill_rule(&self) -> Render2dFillRule {
        self.fill_rule
    }

    /// Returns exact authored commands in stable order.
    #[must_use]
    pub fn commands(&self) -> &[Render2dPathCommand] {
        &self.commands
    }
}

fn validate_path_commands(commands: &[Render2dPathCommand]) -> Result<(), Render2dGeometryError> {
    let mut has_contour = false;
    let mut has_segment = false;
    let mut closed = false;

    for (index, command) in commands.iter().enumerate() {
        match command {
            Render2dPathCommand::MoveTo(_) => {
                has_contour = true;
                has_segment = false;
                closed = false;
            }
            Render2dPathCommand::LineTo(_)
            | Render2dPathCommand::QuadraticTo { .. }
            | Render2dPathCommand::CubicTo { .. } => {
                if !has_contour {
                    return Err(Render2dGeometryError::SegmentWithoutContour { index });
                }
                if closed {
                    return Err(Render2dGeometryError::SegmentAfterClose { index });
                }
                has_segment = true;
            }
            Render2dPathCommand::Close => {
                if !has_contour || !has_segment {
                    return Err(Render2dGeometryError::CloseWithoutSegment { index });
                }
                if closed {
                    return Err(Render2dGeometryError::AlreadyClosed { index });
                }
                closed = true;
            }
        }
    }
    Ok(())
}

/// Public source-neutral shape category.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Render2dShapeKind {
    /// Rectangle.
    Rect,
    /// Canonical rounded rectangle.
    RoundedRect,
    /// Analytic ellipse.
    Ellipse,
    /// Structural path.
    Path,
}

#[derive(Clone, Debug, PartialEq)]
enum Render2dShapeData {
    Rect(Render2dRect),
    RoundedRect {
        rect: Render2dRect,
        radii: Render2dCornerRadii,
    },
    Ellipse(Render2dRect),
    Path(Render2dPath),
}

/// Canonical source-neutral structural geometry.
///
/// Rounded rectangles are normalized at construction with one uniform factor so
/// semantically equivalent oversized corner-radius inputs cannot create distinct
/// renderer-semantic values. That normalization is semantic, not a later physical
/// tessellation or coverage policy.
#[derive(Clone, Debug, PartialEq)]
pub struct Render2dShape {
    data: Render2dShapeData,
}

impl Render2dShape {
    /// Creates rectangular geometry.
    #[must_use]
    pub const fn rect(rect: Render2dRect) -> Self {
        Self {
            data: Render2dShapeData::Rect(rect),
        }
    }

    /// Creates canonical rounded-rectangle geometry.
    #[must_use]
    pub fn rounded_rect(rect: Render2dRect, radii: Render2dCornerRadii) -> Self {
        Self {
            data: Render2dShapeData::RoundedRect {
                rect,
                radii: normalize_corner_radii(rect, radii),
            },
        }
    }

    /// Creates analytic ellipse geometry inscribed in the supplied rectangle.
    #[must_use]
    pub const fn ellipse(rect: Render2dRect) -> Self {
        Self {
            data: Render2dShapeData::Ellipse(rect),
        }
    }

    /// Creates structural arbitrary-path geometry.
    #[must_use]
    pub const fn path(path: Render2dPath) -> Self {
        Self {
            data: Render2dShapeData::Path(path),
        }
    }

    /// Returns the public geometry category.
    #[must_use]
    pub const fn kind(&self) -> Render2dShapeKind {
        match &self.data {
            Render2dShapeData::Rect(_) => Render2dShapeKind::Rect,
            Render2dShapeData::RoundedRect { .. } => Render2dShapeKind::RoundedRect,
            Render2dShapeData::Ellipse(_) => Render2dShapeKind::Ellipse,
            Render2dShapeData::Path(_) => Render2dShapeKind::Path,
        }
    }

    /// Returns rectangular geometry when this shape is a rectangle.
    #[must_use]
    pub const fn as_rect(&self) -> Option<Render2dRect> {
        match &self.data {
            Render2dShapeData::Rect(rect) => Some(*rect),
            _ => None,
        }
    }

    /// Returns canonical rounded-rectangle geometry when present.
    #[must_use]
    pub const fn as_rounded_rect(&self) -> Option<(Render2dRect, Render2dCornerRadii)> {
        match &self.data {
            Render2dShapeData::RoundedRect { rect, radii } => Some((*rect, *radii)),
            _ => None,
        }
    }

    /// Returns the ellipse bounding rectangle when present.
    #[must_use]
    pub const fn as_ellipse(&self) -> Option<Render2dRect> {
        match &self.data {
            Render2dShapeData::Ellipse(rect) => Some(*rect),
            _ => None,
        }
    }

    /// Returns structural path geometry when present.
    #[must_use]
    pub const fn as_path(&self) -> Option<&Render2dPath> {
        match &self.data {
            Render2dShapeData::Path(path) => Some(path),
            _ => None,
        }
    }
}

fn normalize_corner_radii(rect: Render2dRect, radii: Render2dCornerRadii) -> Render2dCornerRadii {
    let values = [
        radii.top_left,
        radii.top_right,
        radii.bottom_right,
        radii.bottom_left,
    ];
    let mut factor = 1.0_f64;
    for (extent, first, second) in [
        (rect.width, values[0], values[1]),
        (rect.width, values[3], values[2]),
        (rect.height, values[0], values[3]),
        (rect.height, values[1], values[2]),
    ] {
        let scale = first.max(second);
        if scale > 0.0 {
            let normalized_sum = first / scale + second / scale;
            factor = factor.min((extent / scale) / normalized_sum);
        }
    }

    Render2dCornerRadii {
        top_left: values[0] * factor,
        top_right: values[1] * factor,
        bottom_right: values[2] * factor,
        bottom_left: values[3] * factor,
    }
}

/// Centered stroke endpoint cap.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum Render2dStrokeCap {
    /// Stop at the endpoint.
    #[default]
    Butt,
    /// Semicircular cap.
    Round,
    /// Square extension by half the stroke width.
    Square,
}

/// Centered stroke join.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum Render2dStrokeJoin {
    /// Miter with deterministic bevel fallback above the miter limit.
    #[default]
    Miter,
    /// Bevel join.
    Bevel,
    /// Round join.
    Round,
}

/// Validated centered stroke semantics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dStrokeStyle {
    width: f64,
    cap: Render2dStrokeCap,
    join: Render2dStrokeJoin,
    miter_limit: f64,
}

impl Render2dStrokeStyle {
    /// Creates one finite centered stroke style.
    ///
    /// # Errors
    ///
    /// Returns a geometry error for non-finite values, negative width, or miter limit below one.
    pub fn new(
        width: f64,
        cap: Render2dStrokeCap,
        join: Render2dStrokeJoin,
        miter_limit: f64,
    ) -> Result<Self, Render2dGeometryError> {
        if !width.is_finite() || !miter_limit.is_finite() {
            return Err(Render2dGeometryError::NonFiniteScalar);
        }
        if width < 0.0 {
            return Err(Render2dGeometryError::NegativeStrokeWidth);
        }
        if miter_limit < 1.0 {
            return Err(Render2dGeometryError::MiterLimitBelowOne);
        }
        Ok(Self {
            width,
            cap,
            join,
            miter_limit,
        })
    }

    /// Returns centered stroke width.
    #[must_use]
    pub const fn width(self) -> f64 {
        self.width
    }

    /// Returns endpoint cap semantics.
    #[must_use]
    pub const fn cap(self) -> Render2dStrokeCap {
        self.cap
    }

    /// Returns join semantics.
    #[must_use]
    pub const fn join(self) -> Render2dStrokeJoin {
        self.join
    }

    /// Returns miter-limit ratio.
    #[must_use]
    pub const fn miter_limit(self) -> f64 {
        self.miter_limit
    }
}

/// One self-contained conjunctive clip in an owner's parent coordinate space.
///
/// `clip_to_parent` maps clip-local geometry directly into the same parent space reached by the
/// attached item's or group's `local_to_parent`. The owner transform is not applied to the clip
/// again; an adapter that wants a clip to follow owner-local motion must resolve that relationship
/// into `clip_to_parent` before constructing the semantic value.
#[derive(Clone, Debug, PartialEq)]
pub struct Render2dClip {
    shape: Render2dShape,
    clip_to_parent: Render2dAffineTransform,
}

impl Render2dClip {
    /// Creates one already-validated self-contained clip.
    #[must_use]
    pub const fn new(shape: Render2dShape, clip_to_parent: Render2dAffineTransform) -> Self {
        Self {
            shape,
            clip_to_parent,
        }
    }

    /// Returns structural clip geometry.
    #[must_use]
    pub const fn shape(&self) -> &Render2dShape {
        &self.shape
    }

    /// Returns the clip-local to owner-parent transform.
    #[must_use]
    pub const fn clip_to_parent(&self) -> Render2dAffineTransform {
        self.clip_to_parent
    }
}

/// Validation failure for source-neutral 2D geometry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Render2dGeometryError {
    /// One numeric component was NaN or infinite.
    NonFiniteScalar,
    /// Rectangle extent was negative.
    NegativeExtent,
    /// Finite rectangle origin plus extent overflowed to a non-finite bound.
    ExtentOverflow,
    /// Corner radius was negative.
    NegativeRadius,
    /// Stroke width was negative.
    NegativeStrokeWidth,
    /// Stroke miter limit was below one.
    MiterLimitBelowOne,
    /// A segment appeared before any contour move.
    SegmentWithoutContour {
        /// Command index.
        index: usize,
    },
    /// `close` appeared before a segment-bearing contour existed.
    CloseWithoutSegment {
        /// Command index.
        index: usize,
    },
    /// The current contour was already explicitly closed.
    AlreadyClosed {
        /// Command index.
        index: usize,
    },
    /// A segment followed an explicit close without a new move.
    SegmentAfterClose {
        /// Command index.
        index: usize,
    },
}

impl fmt::Display for Render2dGeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteScalar => formatter.write_str("2D geometry scalar must be finite"),
            Self::NegativeExtent => formatter.write_str("2D rectangle extent must be non-negative"),
            Self::ExtentOverflow => {
                formatter.write_str("2D rectangle derived bounds must remain finite")
            }
            Self::NegativeRadius => formatter.write_str("2D corner radius must be non-negative"),
            Self::NegativeStrokeWidth => {
                formatter.write_str("2D stroke width must be non-negative")
            }
            Self::MiterLimitBelowOne => {
                formatter.write_str("2D stroke miter limit must be at least 1")
            }
            Self::SegmentWithoutContour { index } => {
                write!(
                    formatter,
                    "2D path segment at {index} requires a preceding move"
                )
            }
            Self::CloseWithoutSegment { index } => write!(
                formatter,
                "2D path Close at {index} requires a segment-bearing contour"
            ),
            Self::AlreadyClosed { index } => {
                write!(formatter, "2D path contour is already closed at {index}")
            }
            Self::SegmentAfterClose { index } => write!(
                formatter,
                "2D path segment at {index} after Close requires a new move"
            ),
        }
    }
}

impl Error for Render2dGeometryError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f64, y: f64) -> Render2dPoint {
        Render2dPoint::new(x, y).expect("finite point")
    }

    #[test]
    fn rounded_rectangles_store_only_canonical_semantic_radii() {
        let rect = Render2dRect::new(0.0, 0.0, 10.0, 4.0).expect("rect");
        let radii = Render2dCornerRadii::new(8.0, 8.0, 8.0, 8.0).expect("radii");
        let shape = Render2dShape::rounded_rect(rect, radii);
        let (_, normalized) = shape.as_rounded_rect().expect("rounded rect");
        assert_eq!(normalized.top_left(), 2.0);
        assert_eq!(normalized.top_right(), 2.0);
        assert_eq!(normalized.bottom_right(), 2.0);
        assert_eq!(normalized.bottom_left(), 2.0);
    }

    #[test]
    fn rectangle_derived_bounds_must_remain_finite() {
        assert_eq!(
            Render2dRect::new(f64::MAX, 0.0, f64::MAX, 1.0),
            Err(Render2dGeometryError::ExtentOverflow)
        );
    }

    #[test]
    fn empty_and_move_only_paths_are_valid_non_painting_structure() {
        Render2dPath::new(Render2dFillRule::NonZero, Vec::new()).expect("empty path is valid");
        Render2dPath::new(
            Render2dFillRule::NonZero,
            vec![Render2dPathCommand::MoveTo(point(0.0, 0.0))],
        )
        .expect("move-only path is valid");
    }

    #[test]
    fn point_degenerate_segment_remains_valid_structure() {
        let at = point(4.0, 5.0);
        Render2dPath::new(
            Render2dFillRule::EvenOdd,
            vec![
                Render2dPathCommand::MoveTo(at),
                Render2dPathCommand::LineTo(at),
                Render2dPathCommand::Close,
            ],
        )
        .expect("point-degenerate segment is structural content");
    }

    #[test]
    fn segment_after_close_requires_new_move() {
        let commands = vec![
            Render2dPathCommand::MoveTo(point(0.0, 0.0)),
            Render2dPathCommand::LineTo(point(1.0, 0.0)),
            Render2dPathCommand::Close,
            Render2dPathCommand::LineTo(point(2.0, 0.0)),
        ];
        assert_eq!(
            Render2dPath::new(Render2dFillRule::NonZero, commands),
            Err(Render2dGeometryError::SegmentAfterClose { index: 3 })
        );
    }
}
