//! Source-neutral immutable 2D semantic resource values and invocation bindings.

use core::{error::Error, fmt};
use std::{
    collections::BTreeMap,
    num::{NonZeroU32, NonZeroU64},
    sync::Arc,
};

/// Opaque semantic identity of one immutable 2D resource value.
///
/// The same identity denotes the same immutable semantic value across compatible
/// executions. Cache entries, atlas locations, device generations, backend handles,
/// and other prepared/resident forms are never part of this identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Render2dResourceId(NonZeroU64);

impl Render2dResourceId {
    /// Creates a non-zero source-neutral semantic resource identity.
    #[must_use]
    pub fn new(raw: u64) -> Option<Self> {
        NonZeroU64::new(raw).map(Self)
    }

    /// Returns the opaque numeric payload.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Initial immutable semantic resource classes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Render2dResourceKind {
    /// Tightly packed unpremultiplied RGBA8 sRGB image content.
    ImageRgba8Srgb,
    /// Already-shaped exact glyph/font facts.
    ShapedText,
}

/// Exact non-zero intrinsic image pixel extent.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Render2dPixelExtent {
    width: NonZeroU32,
    height: NonZeroU32,
}

impl Render2dPixelExtent {
    /// Creates one exact non-zero intrinsic pixel extent.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Option<Self> {
        Some(Self {
            width: NonZeroU32::new(width)?,
            height: NonZeroU32::new(height)?,
        })
    }

    /// Returns intrinsic pixel width.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width.get()
    }

    /// Returns intrinsic pixel height.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height.get()
    }
}

/// Immutable exact initial image semantic payload.
#[derive(Clone, Debug, PartialEq)]
pub struct Render2dImageResource {
    extent: Render2dPixelExtent,
    rgba8_srgb: Arc<[u8]>,
}

impl Render2dImageResource {
    /// Validates and freezes tightly packed unpremultiplied RGBA8 sRGB bytes.
    ///
    /// # Errors
    ///
    /// Returns ImageByteLengthOverflow when the exact byte length is not addressable
    /// and ImageByteLength when the supplied byte count does not equal width * height * 4.
    pub fn new(
        extent: Render2dPixelExtent,
        rgba8_srgb: impl Into<Vec<u8>>,
    ) -> Result<Self, Render2dResourceError> {
        let bytes = rgba8_srgb.into();
        let expected_u64 = u64::from(extent.width())
            .checked_mul(u64::from(extent.height()))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(Render2dResourceError::ImageByteLengthOverflow)?;
        let expected = usize::try_from(expected_u64)
            .map_err(|_| Render2dResourceError::ImageByteLengthOverflow)?;
        if bytes.len() != expected {
            return Err(Render2dResourceError::ImageByteLength {
                expected,
                actual: bytes.len(),
            });
        }
        Ok(Self {
            extent,
            rgba8_srgb: bytes.into(),
        })
    }

    /// Returns exact intrinsic pixel extent.
    #[must_use]
    pub const fn extent(&self) -> Render2dPixelExtent {
        self.extent
    }

    /// Returns immutable tightly packed unpremultiplied RGBA8 sRGB bytes.
    #[must_use]
    pub fn rgba8_srgb(&self) -> &[u8] {
        &self.rgba8_srgb
    }
}

/// Exact immutable font-face facts selected before RunenRender.
#[derive(Clone, PartialEq)]
pub struct Render2dFontBinding {
    bytes: Arc<[u8]>,
    face_index: u32,
    normalized_coords: Arc<[i16]>,
    faux_bold: bool,
    faux_skew: Option<f64>,
}

impl Render2dFontBinding {
    /// Validates and freezes one exact shaped-text font binding.
    ///
    /// Font bytes, face index, normalized variation coordinates and synthesis
    /// requirements are caller-owned semantic facts. RunenRender must not rediscover
    /// or silently substitute another font.
    ///
    /// # Errors
    ///
    /// Returns EmptyFontBytes for an empty font payload or
    /// NonFiniteShapedTextScalar for non-finite faux skew.
    pub fn new(
        bytes: impl Into<Vec<u8>>,
        face_index: u32,
        normalized_coords: impl Into<Vec<i16>>,
        faux_bold: bool,
        faux_skew: Option<f64>,
    ) -> Result<Self, Render2dResourceError> {
        let bytes = bytes.into();
        if bytes.is_empty() {
            return Err(Render2dResourceError::EmptyFontBytes);
        }
        if faux_skew.is_some_and(|value| !value.is_finite()) {
            return Err(Render2dResourceError::NonFiniteShapedTextScalar);
        }
        let normalized_coords: Vec<i16> = normalized_coords.into();
        Ok(Self {
            bytes: bytes.into(),
            face_index,
            normalized_coords: normalized_coords.into(),
            faux_bold,
            faux_skew,
        })
    }

    /// Returns immutable exact font-file bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns exact face index inside the font bytes.
    #[must_use]
    pub const fn face_index(&self) -> u32 {
        self.face_index
    }

    /// Returns normalized variation coordinates selected during shaping.
    #[must_use]
    pub fn normalized_coords(&self) -> &[i16] {
        &self.normalized_coords
    }

    /// Returns whether faux emboldening is required by exact upstream shaping facts.
    #[must_use]
    pub const fn faux_bold(&self) -> bool {
        self.faux_bold
    }

    /// Returns exact faux skew when required.
    #[must_use]
    pub const fn faux_skew(&self) -> Option<f64> {
        self.faux_skew
    }
}

impl fmt::Debug for Render2dFontBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Render2dFontBinding")
            .field("byte_len", &self.bytes.len())
            .field("face_index", &self.face_index)
            .field("normalized_coords", &self.normalized_coords)
            .field("faux_bold", &self.faux_bold)
            .field("faux_skew", &self.faux_skew)
            .finish()
    }
}

/// One already-shaped glyph occurrence in resource-local logical coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dGlyph {
    id: u32,
    x: f64,
    y: f64,
    advance: f64,
}

impl Render2dGlyph {
    /// Creates one exact finite shaped glyph occurrence.
    ///
    /// # Errors
    ///
    /// Returns NonFiniteShapedTextScalar for non-finite position or advance.
    pub fn new(id: u32, x: f64, y: f64, advance: f64) -> Result<Self, Render2dResourceError> {
        if [x, y, advance].into_iter().all(f64::is_finite) {
            Ok(Self { id, x, y, advance })
        } else {
            Err(Render2dResourceError::NonFiniteShapedTextScalar)
        }
    }

    /// Returns upstream-shaped glyph id.
    #[must_use]
    pub const fn id(self) -> u32 {
        self.id
    }

    /// Returns resource-local logical x position.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.x
    }

    /// Returns resource-local logical y position.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.y
    }

    /// Returns caller-owned logical inline advance.
    #[must_use]
    pub const fn advance(self) -> f64 {
        self.advance
    }
}

/// Immutable exact already-shaped text semantic payload.
///
/// Intrinsic outline, COLR, SVG and bitmap capability is derived later from these
/// exact immutable font/glyph facts. No caller-authored duplicate intrinsic-class
/// field is introduced here.
#[derive(Clone, Debug, PartialEq)]
pub struct Render2dShapedTextResource {
    font: Render2dFontBinding,
    font_size: f64,
    glyphs: Arc<[Render2dGlyph]>,
}

impl Render2dShapedTextResource {
    /// Validates and freezes one already-shaped exact resource.
    ///
    /// Empty glyph sequences remain valid non-painting shaped resources.
    ///
    /// # Errors
    ///
    /// Returns NonFiniteShapedTextScalar for non-finite logical font size.
    pub fn new(
        font: Render2dFontBinding,
        font_size: f64,
        glyphs: impl Into<Vec<Render2dGlyph>>,
    ) -> Result<Self, Render2dResourceError> {
        if !font_size.is_finite() {
            return Err(Render2dResourceError::NonFiniteShapedTextScalar);
        }
        let glyphs: Vec<Render2dGlyph> = glyphs.into();
        Ok(Self {
            font,
            font_size,
            glyphs: glyphs.into(),
        })
    }

    /// Returns exact immutable font binding.
    #[must_use]
    pub const fn font(&self) -> &Render2dFontBinding {
        &self.font
    }

    /// Returns logical font size used during upstream shaping.
    #[must_use]
    pub const fn font_size(&self) -> f64 {
        self.font_size
    }

    /// Returns shaped glyph occurrences in exact upstream order.
    #[must_use]
    pub fn glyphs(&self) -> &[Render2dGlyph] {
        &self.glyphs
    }
}

/// One immutable semantic resource value.
#[derive(Clone, Debug, PartialEq)]
pub enum Render2dResourceValue {
    /// Initial exact image payload class.
    ImageRgba8Srgb(Render2dImageResource),
    /// Already-shaped exact text resource.
    ShapedText(Render2dShapedTextResource),
}

impl Render2dResourceValue {
    /// Returns semantic resource class.
    #[must_use]
    pub const fn kind(&self) -> Render2dResourceKind {
        match self {
            Self::ImageRgba8Srgb(_) => Render2dResourceKind::ImageRgba8Srgb,
            Self::ShapedText(_) => Render2dResourceKind::ShapedText,
        }
    }
}

/// One semantic identity-to-value binding for an invocation-compatible resource set.
#[derive(Clone, Debug, PartialEq)]
pub struct Render2dResourceBinding {
    id: Render2dResourceId,
    value: Render2dResourceValue,
}

impl Render2dResourceBinding {
    /// Creates one exact semantic binding.
    #[must_use]
    pub const fn new(id: Render2dResourceId, value: Render2dResourceValue) -> Self {
        Self { id, value }
    }

    /// Returns semantic resource identity.
    #[must_use]
    pub const fn id(&self) -> Render2dResourceId {
        self.id
    }

    /// Returns immutable semantic value.
    #[must_use]
    pub const fn value(&self) -> &Render2dResourceValue {
        &self.value
    }
}

/// Immutable normalized semantic resource set.
///
/// Unreferenced bindings are allowed so one immutable invocation set can satisfy
/// multiple compatible compositions. Every resource referenced by a composition is
/// still required to exist with its exact required semantic kind and compatibility.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Render2dResourceBindings {
    by_id: BTreeMap<Render2dResourceId, Render2dResourceValue>,
}

impl Render2dResourceBindings {
    /// Normalizes one semantic resource set.
    ///
    /// # Errors
    ///
    /// Returns DuplicateResource when an identity appears more than once.
    pub fn new(
        bindings: impl Into<Vec<Render2dResourceBinding>>,
    ) -> Result<Self, Render2dResourceBindingError> {
        let mut by_id = BTreeMap::new();
        for binding in bindings.into() {
            if by_id.insert(binding.id, binding.value).is_some() {
                return Err(Render2dResourceBindingError::DuplicateResource { id: binding.id });
            }
        }
        Ok(Self { by_id })
    }

    /// Looks up one semantic resource value by identity.
    #[must_use]
    pub fn get(&self, id: Render2dResourceId) -> Option<&Render2dResourceValue> {
        self.by_id.get(&id)
    }

    /// Returns number of bindings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Returns whether no semantic resources are bound.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

/// Malformed immutable 2D resource payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Render2dResourceError {
    /// Exact image byte count overflowed addressable memory.
    ImageByteLengthOverflow,
    /// Supplied image bytes do not match exact intrinsic extent.
    ImageByteLength {
        /// Required byte count.
        expected: usize,
        /// Supplied byte count.
        actual: usize,
    },
    /// Exact font binding contained no bytes.
    EmptyFontBytes,
    /// A shaped-text scalar was NaN or infinite.
    NonFiniteShapedTextScalar,
}

impl fmt::Display for Render2dResourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImageByteLengthOverflow => {
                formatter.write_str("2D image byte length overflows addressable memory")
            }
            Self::ImageByteLength { expected, actual } => write!(
                formatter,
                "2D image requires exactly {expected} RGBA8 bytes, got {actual}"
            ),
            Self::EmptyFontBytes => {
                formatter.write_str("2D shaped-text font bytes must not be empty")
            }
            Self::NonFiniteShapedTextScalar => {
                formatter.write_str("2D shaped-text scalar must be finite")
            }
        }
    }
}

impl Error for Render2dResourceError {}

/// Invocation resource-set compatibility failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Render2dResourceBindingError {
    /// One semantic identity was supplied more than once.
    DuplicateResource {
        /// Duplicated semantic identity.
        id: Render2dResourceId,
    },
    /// A required semantic resource is absent.
    MissingResource {
        /// Missing identity.
        id: Render2dResourceId,
        /// Required resource class.
        expected: Render2dResourceKind,
    },
    /// A resource identity resolves to the wrong semantic class.
    ResourceKindMismatch {
        /// Affected identity.
        id: Render2dResourceId,
        /// Required resource class.
        expected: Render2dResourceKind,
        /// Bound resource class.
        actual: Render2dResourceKind,
    },
    /// Bound image intrinsic extent differs from the exact composition requirement.
    ImageIntrinsicExtentMismatch {
        /// Affected image identity.
        id: Render2dResourceId,
        /// Exact extent carried by the composition.
        expected: Render2dPixelExtent,
        /// Exact extent carried by the immutable image binding.
        actual: Render2dPixelExtent,
    },
}

impl fmt::Display for Render2dResourceBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateResource { id } => {
                write!(formatter, "duplicate 2D semantic resource {}", id.get())
            }
            Self::MissingResource { id, expected } => write!(
                formatter,
                "missing 2D semantic resource {} of kind {expected:?}",
                id.get()
            ),
            Self::ResourceKindMismatch {
                id,
                expected,
                actual,
            } => write!(
                formatter,
                "2D semantic resource {} has kind {actual:?}, expected {expected:?}",
                id.get()
            ),
            Self::ImageIntrinsicExtentMismatch {
                id,
                expected,
                actual,
            } => write!(
                formatter,
                "2D image resource {} has intrinsic extent {actual:?}, expected {expected:?}",
                id.get()
            ),
        }
    }
}

impl Error for Render2dResourceBindingError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_resource_identity_rejects() {
        let id = Render2dResourceId::new(7).expect("nonzero id");
        let extent = Render2dPixelExtent::new(1, 1).expect("extent");
        let image = Render2dImageResource::new(extent, vec![0; 4]).expect("image");
        let binding =
            Render2dResourceBinding::new(id, Render2dResourceValue::ImageRgba8Srgb(image));
        assert_eq!(
            Render2dResourceBindings::new(vec![binding.clone(), binding]),
            Err(Render2dResourceBindingError::DuplicateResource { id })
        );
    }

    #[test]
    fn empty_glyph_sequence_is_valid_shaped_semantic_content() {
        let font =
            Render2dFontBinding::new(vec![1], 0, Vec::<i16>::new(), false, None).expect("font");
        let shaped = Render2dShapedTextResource::new(font, 16.0, Vec::<Render2dGlyph>::new())
            .expect("shaped resource");
        assert!(shaped.glyphs().is_empty());
    }
}
