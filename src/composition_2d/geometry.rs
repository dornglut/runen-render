//! Source-neutral two-dimensional geometry semantics.

use core::{error::Error, fmt};
use std::sync::Arc;

/// One finite point in source-neutral two-dimensional logical coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dPoint {
    x: f32,
    y: f32,
}

impl Render2dPoint {
    /// Creates one finite logical point.
    ///
    /// # Errors
    ///
    /// Returns [`Render2dGeometryError::NonFiniteScalar`] when either component is non-finite.
    pub fn new(x: f32, y: f32) -> Result<Self, Render2dGeometryError> {
        if x.is_finite() && y.is_finite() {
            Ok(Self { x, y })
        } else {
            Err(Render2dGeometryError::NonFiniteScalar)
        }
    }

    /// Returns the x coordinate.
    #[must_use]
    pub const fn x(self) -> f32 {
        self.x
    }

    /// Returns the y coordinate.
    #[must_use]
    pub const fn y(self) -> f32 {
        self.y
    }
}

/// One finite non-negative logical rectangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dRect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl Render2dRect {
    /// Creates one finite rectangle with non-negative extents.
    ///
    /// # Errors
    ///
    /// Returns a geometry error for non-finite components or negative extents.
    pub fn new(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) -> Result<Self, Render2dGeometryError> {
        if ![x, y, width, height].into_iter().all(f32::is_finite) {
            return Err(Render2dGeometryError::NonFiniteScalar);
        }
        if width < 0.0 || height < 0.0 {
            return Err(Render2dGeometryError::NegativeExtent);
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
    pub const fn x(self) -> f32 {
        self.x
    }

    /// Returns the top coordinate.
    #[must_use]
    pub const fn y(self) -> f32 {
        self.y
    }

    /// Returns the non-negative width.
    #[must_use]
    pub const fn width(self) -> f32 {
        self.width
    }

    /// Returns the non-negative height.
    #[must_use]
    pub const fn height(self) -> f32 {
        self.height
    }

    /// Returns whether the rectangle has no positive-area interior.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }
}

/// Authored circular corner radii for one rounded rectangle.
///
/// These values remain structural semantic content. Any normalization against a rectangle's
/// extents is derived realization policy and does not rewrite this value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dCornerRadii {
    top_left: f32,
    top_right: f32,
    bottom_right: f32,
    bottom_left: f32,
}

impl Render2dCornerRadii {
    /// Creates finite non-negative authored corner radii.
    ///
    /// # Errors
    ///
    /// Returns a geometry error for non-finite or negative radii.
    pub fn new(
        top_left: f32,
        top_right: f32,
        bottom_right: f32,
        bottom_left: f32,
    ) -> Result<Self, Render2dGeometryError> {
        let values = [top_left, top_right, bottom_right, bottom_left];
        if !values.into_iter().all(f32::is_finite) {
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
    pub const fn top_left(self) -> f32 {
        self.top_left
    }

    /// Returns the top-right radius.
    #[must_use]
    pub const fn top_right(self) -> f32 {
        self.top_right
    }

    /// Returns the bottom-right radius.
    #[must_use]
    pub const fn bottom_right(self) -> f32 {
        self.bottom_right
    }

    /// Returns the bottom-left radius.
    #[must_use]
    pub const fn bottom_left(self) -> f32 {
        self.bottom_left
    }
}

/// Finite affine transform mapping local 2D coordinates into one parent coordinate space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dAffineTransform {
    m11: f32,
    m12: f32,
    m21: f32,
    m22: f32,
    tx: f32,
    ty: f32,
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
        m11: f32,
        m12: f32,
        m21: f32,
        m22: f32,
        tx: f32,
        ty: f32,
    ) -> Result<Self, Render2dGeometryError> {
        if [m11, m12, m21, m22, tx, ty]
            .into_iter()
            .all(f32::is_finite)
        {
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
    pub fn translation(x: f32, y: f32) -> Result<Self, Render2dGeometryError> {
        Self::new(1.0, 0.0, 0.0, 1.0, x, y)
    }

    /// Returns `(m11, m12, m21, m22, tx, ty)`.
    #[must_use]
    pub const fn components(self) -> [f32; 6] {
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
/// physical fill realization may derive an implicit closing edge without rewriting this value.
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

/// Source-neutral structural geometry.
#[derive(Clone, Debug, PartialEq)]
pub enum Render2dShape {
    /// Finite rectangle.
    Rect(Render2dRect),
    /// Finite rectangle with authored circular corner radii.
    RoundedRect {
        /// Base rectangle.
        rect: Render2dRect,
        /// Authored radii.
        radii: Render2dCornerRadii,
    },
    /// Analytic ellipse inscribed in the supplied rectangle.
    Ellipse(Render2dRect),
    /// Structural arbitrary path.
    Path(Render2dPath),
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
    width: f32,
    cap: Render2dStrokeCap,
    join: Render2dStrokeJoin,
    miter_limit: f32,
}

impl Render2dStrokeStyle {
    /// Creates one finite centered stroke style.
    ///
    /// # Errors
    ///
    /// Returns a geometry error for non-finite values, negative width, or miter limit below one.
    pub fn new(
        width: f32,
        cap: Render2dStrokeCap,
        join: Render2dStrokeJoin,
        miter_limit: f32,
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
    pub const fn width(self) -> f32 {
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
    pub const fn miter_limit(self) -> f32 {
        self.miter_limit
    }
}

/// One conjunctive clip: structural local geometry plus its local-to-parent mapping.
#[derive(Clone, Debug, PartialEq)]
pub struct Render2dClip {
    shape: Render2dShape,
    local_to_parent: Render2dAffineTransform,
}

impl Render2dClip {
    /// Creates one already-validated clip.
    #[must_use]
    pub const fn new(
        shape: Render2dShape,
        local_to_parent: Render2dAffineTransform,
    ) -> Self {
        Self {
            shape,
            local_to_parent,
        }
    }

    /// Returns structural clip geometry.
    #[must_use]
    pub const fn shape(&self) -> &Render2dShape {
        &self.shape
    }

    /// Returns the clip-local to parent transform.
    #[must_use]
    pub const fn local_to_parent(&self) -> Render2dAffineTransform {
        self.local_to_parent
    }
}

/// Validation failure for source-neutral 2D geometry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Render2dGeometryError {
    /// One numeric component was NaN or infinite.
    NonFiniteScalar,
    /// Rectangle extent was negative.
    NegativeExtent,
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
            Self::NegativeExtent => {
                formatter.write_str("2D rectangle extent must be non-negative")
            }
            Self::NegativeRadius => formatter.write_str("2D corner radius must be non-negative"),
            Self::NegativeStrokeWidth => {
                formatter.write_str("2D stroke width must be non-negative")
            }
            Self::MiterLimitBelowOne => {
                formatter.write_str("2D stroke miter limit must be at least 1")
            }
            Self::SegmentWithoutContour { index } => {
                write!(formatter, "2D path segment at {index} requires a preceding move")
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

    fn point(x: f32, y: f32) -> Render2dPoint {
        Render2dPoint::new(x, y).expect("finite point")
    }

    #[test]
    fn empty_and_move_only_paths_are_valid_non_painting_structure() {
        Render2dPath::new(Render2dFillRule::NonZero, Vec::new())
            .expect("empty path is valid");
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
