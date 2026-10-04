//! Source-neutral two-dimensional paint and ordinary effect semantics.

use core::{error::Error, fmt};
use std::sync::Arc;

use super::geometry::Render2dPoint;

/// Straight-alpha sRGB8 literal color.
///
/// RGB bytes carry sRGB-encoded literal channels and alpha carries linear coverage. Ordered
/// composition is ordinary source-over in linear-light space; private realizations may
/// premultiply after decoding but must preserve this public meaning.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Render2dColorRgba8 {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl Render2dColorRgba8 {
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self::new(0, 0, 0, 0);
    /// Opaque black.
    pub const BLACK: Self = Self::new(0, 0, 0, u8::MAX);
    /// Opaque white.
    pub const WHITE: Self = Self::new(u8::MAX, u8::MAX, u8::MAX, u8::MAX);

    /// Creates one exact straight-alpha sRGB8 color.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Returns `[r, g, b, a]`.
    #[must_use]
    pub const fn channels(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

/// Validated item/group opacity applied once at its semantic owner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dOpacity(f64);

impl Render2dOpacity {
    /// Fully opaque.
    pub const OPAQUE: Self = Self(1.0);
    /// Fully transparent.
    pub const TRANSPARENT: Self = Self(0.0);

    /// Creates one finite opacity in the inclusive range `0..=1`.
    ///
    /// # Errors
    ///
    /// Returns [`Render2dPaintError`] for non-finite or out-of-range values.
    pub fn new(value: f64) -> Result<Self, Render2dPaintError> {
        if !value.is_finite() {
            return Err(Render2dPaintError::NonFiniteScalar);
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(Render2dPaintError::OpacityOutOfRange);
        }
        Ok(Self(value))
    }

    /// Returns the validated opacity scalar.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// One exact authored gradient stop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dGradientStop {
    offset: f64,
    color: Render2dColorRgba8,
}

impl Render2dGradientStop {
    /// Creates one normalized gradient stop.
    ///
    /// # Errors
    ///
    /// Returns [`Render2dPaintError`] for non-finite or out-of-range offset.
    pub fn new(
        offset: f64,
        color: Render2dColorRgba8,
    ) -> Result<Self, Render2dPaintError> {
        if !offset.is_finite() {
            return Err(Render2dPaintError::NonFiniteScalar);
        }
        if !(0.0..=1.0).contains(&offset) {
            return Err(Render2dPaintError::GradientOffsetOutOfRange);
        }
        Ok(Self { offset, color })
    }

    /// Returns the normalized stop offset.
    #[must_use]
    pub const fn offset(self) -> f64 {
        self.offset
    }

    /// Returns the literal stop color.
    #[must_use]
    pub const fn color(self) -> Render2dColorRgba8 {
        self.color
    }
}

/// Immutable validated authored gradient stops.
///
/// Stops retain stable nondecreasing authored order. Equal offsets are preserved exactly and
/// therefore retain deterministic hard-stop semantics.
#[derive(Clone, Debug, PartialEq)]
pub struct Render2dGradientStops(Arc<[Render2dGradientStop]>);

impl Render2dGradientStops {
    /// Validates and freezes authored stops.
    ///
    /// # Errors
    ///
    /// Returns [`Render2dPaintError`] for fewer than two stops or decreasing offsets.
    pub fn new(
        stops: impl Into<Vec<Render2dGradientStop>>,
    ) -> Result<Self, Render2dPaintError> {
        let stops = stops.into();
        if stops.len() < 2 {
            return Err(Render2dPaintError::TooFewGradientStops);
        }
        if stops
            .windows(2)
            .any(|pair| pair[0].offset > pair[1].offset)
        {
            return Err(Render2dPaintError::DecreasingGradientOffsets);
        }
        Ok(Self(stops.into()))
    }

    /// Returns exact authored stops in stable order.
    #[must_use]
    pub fn as_slice(&self) -> &[Render2dGradientStop] {
        &self.0
    }
}

/// Primitive-local nondegenerate linear gradient.
///
/// Color interpolation is semantically premultiplied linear-sRGB between the stable stops.
#[derive(Clone, Debug, PartialEq)]
pub struct Render2dLinearGradient {
    start: Render2dPoint,
    end: Render2dPoint,
    stops: Render2dGradientStops,
}

impl Render2dLinearGradient {
    /// Creates one nondegenerate primitive-local linear gradient.
    ///
    /// # Errors
    ///
    /// Returns [`Render2dPaintError::EqualLinearGradientEndpoints`] when endpoints match.
    pub fn new(
        start: Render2dPoint,
        end: Render2dPoint,
        stops: Render2dGradientStops,
    ) -> Result<Self, Render2dPaintError> {
        if start == end {
            return Err(Render2dPaintError::EqualLinearGradientEndpoints);
        }
        Ok(Self { start, end, stops })
    }

    /// Returns the primitive-local start point.
    #[must_use]
    pub const fn start(&self) -> Render2dPoint {
        self.start
    }

    /// Returns the primitive-local end point.
    #[must_use]
    pub const fn end(&self) -> Render2dPoint {
        self.end
    }

    /// Returns stable authored stops.
    #[must_use]
    pub const fn stops(&self) -> &Render2dGradientStops {
        &self.stops
    }
}

/// Primitive-local concentric radial gradient.
///
/// Color interpolation is semantically premultiplied linear-sRGB between the stable stops.
#[derive(Clone, Debug, PartialEq)]
pub struct Render2dRadialGradient {
    center: Render2dPoint,
    radius: f64,
    stops: Render2dGradientStops,
}

impl Render2dRadialGradient {
    /// Creates one positive-radius primitive-local radial gradient.
    ///
    /// # Errors
    ///
    /// Returns a paint error for non-finite or non-positive radius.
    pub fn new(
        center: Render2dPoint,
        radius: f64,
        stops: Render2dGradientStops,
    ) -> Result<Self, Render2dPaintError> {
        if !radius.is_finite() {
            return Err(Render2dPaintError::NonFiniteScalar);
        }
        if radius <= 0.0 {
            return Err(Render2dPaintError::NonPositiveRadialRadius);
        }
        Ok(Self {
            center,
            radius,
            stops,
        })
    }

    /// Returns the primitive-local center.
    #[must_use]
    pub const fn center(&self) -> Render2dPoint {
        self.center
    }

    /// Returns positive radius.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// Returns stable authored stops.
    #[must_use]
    pub const fn stops(&self) -> &Render2dGradientStops {
        &self.stops
    }
}

/// Source-neutral initial brush vocabulary.
#[derive(Clone, Debug, PartialEq)]
pub enum Render2dBrush {
    /// Straight-alpha sRGB8 literal solid color.
    Solid(Render2dColorRgba8),
    /// Primitive-local linear gradient.
    Linear(Render2dLinearGradient),
    /// Primitive-local radial gradient.
    Radial(Render2dRadialGradient),
}

impl Render2dBrush {
    /// Creates a solid brush.
    #[must_use]
    pub const fn solid(color: Render2dColorRgba8) -> Self {
        Self::Solid(color)
    }
}

/// Ordinary source-neutral drop shadow.
///
/// Shadow support derives from neutral semantic child geometry, not cached sampled alpha.
/// Offset and signed spread are logical semantic values; sigma is finite and non-negative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dDropShadow {
    offset_x: f64,
    offset_y: f64,
    sigma: f64,
    spread: f64,
    color: Render2dColorRgba8,
}

impl Render2dDropShadow {
    /// Creates one finite ordinary shadow.
    ///
    /// # Errors
    ///
    /// Returns a paint error for non-finite values or negative sigma.
    pub fn new(
        offset_x: f64,
        offset_y: f64,
        sigma: f64,
        spread: f64,
        color: Render2dColorRgba8,
    ) -> Result<Self, Render2dPaintError> {
        if ![offset_x, offset_y, sigma, spread]
            .into_iter()
            .all(f64::is_finite)
        {
            return Err(Render2dPaintError::NonFiniteScalar);
        }
        if sigma < 0.0 {
            return Err(Render2dPaintError::NegativeShadowSigma);
        }
        Ok(Self {
            offset_x,
            offset_y,
            sigma,
            spread,
            color,
        })
    }

    /// Returns horizontal offset.
    #[must_use]
    pub const fn offset_x(self) -> f64 {
        self.offset_x
    }

    /// Returns vertical offset.
    #[must_use]
    pub const fn offset_y(self) -> f64 {
        self.offset_y
    }

    /// Returns non-negative Gaussian sigma.
    #[must_use]
    pub const fn sigma(self) -> f64 {
        self.sigma
    }

    /// Returns signed Euclidean spread radius.
    #[must_use]
    pub const fn spread(self) -> f64 {
        self.spread
    }

    /// Returns literal shadow color.
    #[must_use]
    pub const fn color(self) -> Render2dColorRgba8 {
        self.color
    }
}

/// Validation failure for 2D paint/effect semantic values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Render2dPaintError {
    /// One numeric component was NaN or infinite.
    NonFiniteScalar,
    /// Opacity was outside `0..=1`.
    OpacityOutOfRange,
    /// Gradient offset was outside `0..=1`.
    GradientOffsetOutOfRange,
    /// Fewer than two gradient stops were supplied.
    TooFewGradientStops,
    /// Gradient offsets decreased.
    DecreasingGradientOffsets,
    /// Linear gradient endpoints were equal.
    EqualLinearGradientEndpoints,
    /// Radial gradient radius was zero or negative.
    NonPositiveRadialRadius,
    /// Shadow sigma was negative.
    NegativeShadowSigma,
}

impl fmt::Display for Render2dPaintError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NonFiniteScalar => "2D paint scalar must be finite",
            Self::OpacityOutOfRange => "2D opacity must be in the inclusive range 0..=1",
            Self::GradientOffsetOutOfRange => {
                "2D gradient offset must be in the inclusive range 0..=1"
            }
            Self::TooFewGradientStops => "2D gradient requires at least two stops",
            Self::DecreasingGradientOffsets => {
                "2D gradient stop offsets must be nondecreasing"
            }
            Self::EqualLinearGradientEndpoints => {
                "2D linear gradient endpoints must differ"
            }
            Self::NonPositiveRadialRadius => "2D radial gradient radius must be positive",
            Self::NegativeShadowSigma => "2D shadow sigma must be non-negative",
        })
    }
}

impl Error for Render2dPaintError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_gradient_offsets_are_preserved_for_hard_stops() {
        let first =
            Render2dGradientStop::new(0.5, Render2dColorRgba8::BLACK).expect("stop");
        let second =
            Render2dGradientStop::new(0.5, Render2dColorRgba8::WHITE).expect("stop");
        let stops =
            Render2dGradientStops::new(vec![first, second]).expect("stable hard stop");
        assert_eq!(stops.as_slice(), &[first, second]);
    }

    #[test]
    fn decreasing_gradient_offsets_reject() {
        let first =
            Render2dGradientStop::new(0.8, Render2dColorRgba8::BLACK).expect("stop");
        let second =
            Render2dGradientStop::new(0.2, Render2dColorRgba8::WHITE).expect("stop");
        assert_eq!(
            Render2dGradientStops::new(vec![first, second]),
            Err(Render2dPaintError::DecreasingGradientOffsets)
        );
    }
}
