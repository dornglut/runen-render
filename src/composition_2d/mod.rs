//! Source-neutral immutable two-dimensional composition semantics.
//!
//! This module owns renderer-domain meaning only. It deliberately excludes source-framework
//! publication identity, GPU/display-list commands, caches, atlas residency, device handles,
//! native presentation policy, and concrete physical realization-family names.
//!
//! Entries compose in exact authored order using ordinary linear-light source-over semantics.
//! Literal colors are straight-alpha sRGB8. Gradient interpolation is defined in premultiplied
//! linear-sRGB. Item opacity applies once to its item; group opacity applies once to the composed
//! atomic group result. Clips are conjunctive. Ordered group shadows derive support from neutral
//! semantic child geometry rather than cached sampled alpha, so transparent child color does not
//! erase semantic shadow support.

mod geometry;
mod paint;
mod resource;

pub use geometry::{
    Render2dAffineTransform, Render2dClip, Render2dCornerRadii, Render2dFillRule,
    Render2dGeometryError, Render2dPath, Render2dPathCommand, Render2dPoint, Render2dRect,
    Render2dShape, Render2dStrokeCap, Render2dStrokeJoin, Render2dStrokeStyle,
};
pub use paint::{
    Render2dBrush, Render2dColorRgba8, Render2dDropShadow, Render2dGradientStop,
    Render2dGradientStops, Render2dLinearGradient, Render2dOpacity, Render2dPaintError,
    Render2dRadialGradient,
};
pub use resource::{
    Render2dFontBinding, Render2dGlyph, Render2dImageResource, Render2dPixelExtent,
    Render2dResourceBinding, Render2dResourceBindingError, Render2dResourceBindings,
    Render2dResourceError, Render2dResourceId, Render2dResourceKind, Render2dResourceValue,
    Render2dShapedTextResource,
};

use core::{error::Error, fmt};
use std::{collections::BTreeMap, sync::Arc};

/// Exact finite source-pixel rectangle used by one already-resolved image mapping.
///
/// Coordinates are continuous intrinsic-pixel coordinates. Fractional and zero-area
/// rectangles are retained exactly because source adapters have already resolved fit,
/// crop, alignment and slicing policy before RunenRender observes the composition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dImageSourceRect {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

impl Render2dImageSourceRect {
    /// Creates one finite non-negative source rectangle.
    ///
    /// # Errors
    ///
    /// Returns NonFiniteScalar for NaN/infinite components or NegativeExtent for a
    /// negative coordinate or extent.
    pub fn new(
        x: f64,
        y: f64,
        width: f64,
        height: f64,
    ) -> Result<Self, Render2dImageSourceRectError> {
        if ![x, y, width, height].into_iter().all(f64::is_finite) {
            return Err(Render2dImageSourceRectError::NonFiniteScalar);
        }
        if x < 0.0 || y < 0.0 || width < 0.0 || height < 0.0 {
            return Err(Render2dImageSourceRectError::NegativeExtent);
        }
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }

    /// Returns source x coordinate in intrinsic-pixel space.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.x
    }

    /// Returns source y coordinate in intrinsic-pixel space.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.y
    }

    /// Returns source width in intrinsic-pixel space.
    #[must_use]
    pub const fn width(self) -> f64 {
        self.width
    }

    /// Returns source height in intrinsic-pixel space.
    #[must_use]
    pub const fn height(self) -> f64 {
        self.height
    }

    fn fits_within(self, extent: Render2dPixelExtent) -> bool {
        let max_x = f64::from(extent.width());
        let max_y = f64::from(extent.height());
        self.width <= max_x
            && self.height <= max_y
            && self.x <= max_x - self.width
            && self.y <= max_y - self.height
    }
}

/// Validation failure for one already-resolved image source rectangle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Render2dImageSourceRectError {
    /// One source-pixel component was NaN or infinite.
    NonFiniteScalar,
    /// A coordinate or extent was negative.
    NegativeExtent,
}

impl fmt::Display for Render2dImageSourceRectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NonFiniteScalar => "2D image source rectangle components must be finite",
            Self::NegativeExtent => {
                "2D image source rectangle coordinates and extents must be non-negative"
            }
        })
    }
}

impl Error for Render2dImageSourceRectError {}

/// One exact already-resolved image source-to-destination mapping patch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dImagePatch {
    source: Render2dImageSourceRect,
    destination: Render2dRect,
}

impl Render2dImagePatch {
    /// Creates one exact source/destination mapping patch.
    ///
    /// Zero-area source or destination patches remain structural semantic content.
    #[must_use]
    pub const fn new(source: Render2dImageSourceRect, destination: Render2dRect) -> Self {
        Self {
            source,
            destination,
        }
    }

    /// Returns exact source-pixel rectangle.
    #[must_use]
    pub const fn source(self) -> Render2dImageSourceRect {
        self.source
    }

    /// Returns exact destination logical rectangle.
    #[must_use]
    pub const fn destination(self) -> Render2dRect {
        self.destination
    }
}

/// Exact source-neutral image primitive.
///
/// Intrinsic extent is carried by composition semantics so source mappings are validated
/// before resource binding and so an invocation binding must prove compatible immutable
/// image metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Render2dImagePrimitive {
    resource_id: Render2dResourceId,
    intrinsic_extent: Render2dPixelExtent,
    patches: Arc<[Render2dImagePatch]>,
}

impl Render2dImagePrimitive {
    /// Validates and freezes one already-resolved image primitive.
    ///
    /// Empty patch sequences are valid non-painting image content.
    ///
    /// # Errors
    ///
    /// Returns ImageSourceOutsideIntrinsicExtent when any resolved source rectangle
    /// exceeds the exact intrinsic extent carried by the composition.
    pub fn new(
        resource_id: Render2dResourceId,
        intrinsic_extent: Render2dPixelExtent,
        patches: impl Into<Vec<Render2dImagePatch>>,
    ) -> Result<Self, Render2dCompositionError> {
        let patches = patches.into();
        for (patch_index, patch) in patches.iter().enumerate() {
            if !patch.source.fits_within(intrinsic_extent) {
                return Err(Render2dCompositionError::ImageSourceOutsideIntrinsicExtent {
                    id: resource_id,
                    patch_index,
                });
            }
        }
        Ok(Self {
            resource_id,
            intrinsic_extent,
            patches: patches.into(),
        })
    }

    /// Returns semantic image resource identity.
    #[must_use]
    pub const fn resource_id(&self) -> Render2dResourceId {
        self.resource_id
    }

    /// Returns exact intrinsic extent required by this mapping.
    #[must_use]
    pub const fn intrinsic_extent(&self) -> Render2dPixelExtent {
        self.intrinsic_extent
    }

    /// Returns exact resolved patches in authored mapping order.
    #[must_use]
    pub fn patches(&self) -> &[Render2dImagePatch] {
        &self.patches
    }
}

/// One already-shaped text occurrence placed in composition-local coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Render2dShapedTextPrimitive {
    resource_id: Render2dResourceId,
    origin: Render2dPoint,
    color: Render2dColorRgba8,
}

impl Render2dShapedTextPrimitive {
    /// Creates one shaped-text occurrence without changing resource-owned glyph metrics.
    #[must_use]
    pub const fn new(
        resource_id: Render2dResourceId,
        origin: Render2dPoint,
        color: Render2dColorRgba8,
    ) -> Self {
        Self {
            resource_id,
            origin,
            color,
        }
    }

    /// Returns shaped semantic resource identity.
    #[must_use]
    pub const fn resource_id(self) -> Render2dResourceId {
        self.resource_id
    }

    /// Returns placement of resource-local logical origin.
    #[must_use]
    pub const fn origin(self) -> Render2dPoint {
        self.origin
    }

    /// Returns ordinary straight-alpha sRGB8 foreground color.
    #[must_use]
    pub const fn color(self) -> Render2dColorRgba8 {
        self.color
    }
}

/// One renderer-semantic 2D primitive.
#[derive(Clone, Debug, PartialEq)]
pub enum Render2dPrimitive {
    /// Fill structural geometry with one brush.
    Fill {
        /// Structural geometry.
        shape: Render2dShape,
        /// Primitive-local brush.
        brush: Render2dBrush,
    },
    /// Centered stroke of structural geometry.
    Stroke {
        /// Structural geometry.
        shape: Render2dShape,
        /// Primitive-local brush.
        brush: Render2dBrush,
        /// Exact centered stroke semantics.
        style: Render2dStrokeStyle,
    },
    /// Exact resolved image mapping.
    Image(Render2dImagePrimitive),
    /// Already-shaped text occurrence.
    ShapedText(Render2dShapedTextPrimitive),
}

/// One ordinary composition item.
#[derive(Clone, Debug, PartialEq)]
pub struct Render2dItem {
    primitive: Render2dPrimitive,
    local_to_parent: Render2dAffineTransform,
    clips: Arc<[Render2dClip]>,
    opacity: Render2dOpacity,
}

impl Render2dItem {
    /// Freezes one item and its conjunctive clips.
    #[must_use]
    pub fn new(
        primitive: Render2dPrimitive,
        local_to_parent: Render2dAffineTransform,
        clips: impl Into<Vec<Render2dClip>>,
        opacity: Render2dOpacity,
    ) -> Self {
        let clips: Vec<Render2dClip> = clips.into();
        Self {
            primitive,
            local_to_parent,
            clips: clips.into(),
            opacity,
        }
    }

    /// Returns primitive semantic content.
    #[must_use]
    pub const fn primitive(&self) -> &Render2dPrimitive {
        &self.primitive
    }

    /// Returns item-local to parent transform.
    #[must_use]
    pub const fn local_to_parent(&self) -> Render2dAffineTransform {
        self.local_to_parent
    }

    /// Returns conjunctive clips in stable authored order.
    #[must_use]
    pub fn clips(&self) -> &[Render2dClip] {
        &self.clips
    }

    /// Returns opacity applied once to this item.
    #[must_use]
    pub const fn opacity(&self) -> Render2dOpacity {
        self.opacity
    }
}

/// One atomic nested composition group.
///
/// The owned recursive tree contains no public group identity or back-reference, so
/// cycles and ambiguous membership cannot be constructed through the safe API.
#[derive(Clone, Debug, PartialEq)]
pub struct Render2dGroup {
    entries: Arc<[Render2dEntry]>,
    local_to_parent: Render2dAffineTransform,
    clips: Arc<[Render2dClip]>,
    opacity: Render2dOpacity,
    shadows: Arc<[Render2dDropShadow]>,
}

impl Render2dGroup {
    /// Freezes one atomic group and exact child/effect order.
    #[must_use]
    pub fn new(
        entries: impl Into<Vec<Render2dEntry>>,
        local_to_parent: Render2dAffineTransform,
        clips: impl Into<Vec<Render2dClip>>,
        opacity: Render2dOpacity,
        shadows: impl Into<Vec<Render2dDropShadow>>,
    ) -> Self {
        let entries: Vec<Render2dEntry> = entries.into();
        let clips: Vec<Render2dClip> = clips.into();
        let shadows: Vec<Render2dDropShadow> = shadows.into();
        Self {
            entries: entries.into(),
            local_to_parent,
            clips: clips.into(),
            opacity,
            shadows: shadows.into(),
        }
    }

    /// Returns child entries in exact painter order.
    #[must_use]
    pub fn entries(&self) -> &[Render2dEntry] {
        &self.entries
    }

    /// Returns group-local to parent transform.
    #[must_use]
    pub const fn local_to_parent(&self) -> Render2dAffineTransform {
        self.local_to_parent
    }

    /// Returns conjunctive group clips.
    #[must_use]
    pub fn clips(&self) -> &[Render2dClip] {
        &self.clips
    }

    /// Returns opacity applied exactly once to the composed group result.
    #[must_use]
    pub const fn opacity(&self) -> Render2dOpacity {
        self.opacity
    }

    /// Returns ordinary shadows in exact authored order.
    #[must_use]
    pub fn shadows(&self) -> &[Render2dDropShadow] {
        &self.shadows
    }
}

/// One ordered root/group composition entry.
#[derive(Clone, Debug, PartialEq)]
pub enum Render2dEntry {
    /// Ordinary item.
    Item(Render2dItem),
    /// Atomic nested group.
    Group(Render2dGroup),
}

impl Render2dEntry {
    /// Creates an item entry.
    #[must_use]
    pub const fn item(item: Render2dItem) -> Self {
        Self::Item(item)
    }

    /// Creates a group entry.
    #[must_use]
    pub const fn group(group: Render2dGroup) -> Self {
        Self::Group(group)
    }
}

/// Exact semantic resource requirement derived from immutable composition content.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Render2dResourceRequirement {
    id: Render2dResourceId,
    kind: Render2dResourceKind,
    image_intrinsic_extent: Option<Render2dPixelExtent>,
}

impl Render2dResourceRequirement {
    /// Returns semantic identity.
    #[must_use]
    pub const fn id(self) -> Render2dResourceId {
        self.id
    }

    /// Returns required semantic resource class.
    #[must_use]
    pub const fn kind(self) -> Render2dResourceKind {
        self.kind
    }

    /// Returns exact intrinsic image extent when this requirement is image-kind.
    #[must_use]
    pub const fn image_intrinsic_extent(self) -> Option<Render2dPixelExtent> {
        self.image_intrinsic_extent
    }
}

/// Lifetime-neutral immutable source-neutral 2D composition semantic root.
#[derive(Clone, Debug, PartialEq)]
pub struct Render2dComposition {
    root_entries: Arc<[Render2dEntry]>,
    resource_requirements: Arc<[Render2dResourceRequirement]>,
}

impl Render2dComposition {
    /// Validates and freezes exact root painter/group order.
    ///
    /// Empty compositions are valid non-painting semantic values.
    ///
    /// # Errors
    ///
    /// Returns a composition error when one semantic resource identity is used
    /// incompatibly within the immutable composition.
    pub fn new(
        root_entries: impl Into<Vec<Render2dEntry>>,
    ) -> Result<Self, Render2dCompositionError> {
        let root_entries = root_entries.into();
        let mut requirements = BTreeMap::new();
        let mut pending = root_entries.iter().rev().collect::<Vec<_>>();

        while let Some(entry) = pending.pop() {
            match entry {
                Render2dEntry::Item(item) => match item.primitive() {
                    Render2dPrimitive::Image(image) => {
                        collect_requirement(
                            &mut requirements,
                            Render2dResourceRequirement {
                                id: image.resource_id(),
                                kind: Render2dResourceKind::ImageRgba8Srgb,
                                image_intrinsic_extent: Some(image.intrinsic_extent()),
                            },
                        )?;
                    }
                    Render2dPrimitive::ShapedText(text) => {
                        collect_requirement(
                            &mut requirements,
                            Render2dResourceRequirement {
                                id: text.resource_id(),
                                kind: Render2dResourceKind::ShapedText,
                                image_intrinsic_extent: None,
                            },
                        )?;
                    }
                    Render2dPrimitive::Fill { .. } | Render2dPrimitive::Stroke { .. } => {}
                },
                Render2dEntry::Group(group) => {
                    pending.extend(group.entries().iter().rev());
                }
            }
        }

        let resource_requirements = requirements.into_values().collect::<Vec<_>>();
        Ok(Self {
            root_entries: root_entries.into(),
            resource_requirements: resource_requirements.into(),
        })
    }

    /// Returns root entries in exact painter order.
    #[must_use]
    pub fn root_entries(&self) -> &[Render2dEntry] {
        &self.root_entries
    }

    /// Returns deterministic unique semantic resource requirements ordered by identity.
    #[must_use]
    pub fn resource_requirements(&self) -> &[Render2dResourceRequirement] {
        &self.resource_requirements
    }

    /// Validates one immutable invocation resource set against this composition.
    ///
    /// # Errors
    ///
    /// Returns a binding error for a missing resource, semantic-kind mismatch or
    /// incompatible exact image intrinsic extent.
    pub fn validate_bindings(
        &self,
        bindings: &Render2dResourceBindings,
    ) -> Result<(), Render2dResourceBindingError> {
        for requirement in self.resource_requirements.iter().copied() {
            let Some(value) = bindings.get(requirement.id) else {
                return Err(Render2dResourceBindingError::MissingResource {
                    id: requirement.id,
                    expected: requirement.kind,
                });
            };
            let actual = value.kind();
            if actual != requirement.kind {
                return Err(Render2dResourceBindingError::ResourceKindMismatch {
                    id: requirement.id,
                    expected: requirement.kind,
                    actual,
                });
            }

            if let (
                Some(expected),
                Render2dResourceValue::ImageRgba8Srgb(image),
            ) = (requirement.image_intrinsic_extent, value)
            {
                let actual = image.extent();
                if actual != expected {
                    return Err(Render2dResourceBindingError::ImageIntrinsicExtentMismatch {
                        id: requirement.id,
                        expected,
                        actual,
                    });
                }
            }
        }
        Ok(())
    }
}

fn collect_requirement(
    requirements: &mut BTreeMap<Render2dResourceId, Render2dResourceRequirement>,
    requirement: Render2dResourceRequirement,
) -> Result<(), Render2dCompositionError> {
    if let Some(existing) = requirements.get(&requirement.id).copied() {
        if existing.kind != requirement.kind {
            return Err(Render2dCompositionError::ConflictingResourceKinds {
                id: requirement.id,
                first: existing.kind,
                second: requirement.kind,
            });
        }
        if existing.image_intrinsic_extent != requirement.image_intrinsic_extent {
            let (Some(first), Some(second)) = (
                existing.image_intrinsic_extent,
                requirement.image_intrinsic_extent,
            ) else {
                unreachable!("equal image resource kinds carry image extents");
            };
            return Err(Render2dCompositionError::ConflictingImageIntrinsicExtent {
                id: requirement.id,
                first,
                second,
            });
        }
        return Ok(());
    }

    requirements.insert(requirement.id, requirement);
    Ok(())
}

/// Malformed or internally incompatible immutable composition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Render2dCompositionError {
    /// One identity is referenced as two different semantic resource classes.
    ConflictingResourceKinds {
        /// Affected identity.
        id: Render2dResourceId,
        /// First observed class.
        first: Render2dResourceKind,
        /// Conflicting class.
        second: Render2dResourceKind,
    },
    /// One image identity carries conflicting exact intrinsic extents.
    ConflictingImageIntrinsicExtent {
        /// Affected image identity.
        id: Render2dResourceId,
        /// First observed exact image extent.
        first: Render2dPixelExtent,
        /// Conflicting exact image extent.
        second: Render2dPixelExtent,
    },
    /// One already-resolved source patch exceeds its composition-declared intrinsic extent.
    ImageSourceOutsideIntrinsicExtent {
        /// Affected image identity.
        id: Render2dResourceId,
        /// Zero-based patch index.
        patch_index: usize,
    },
}

impl fmt::Display for Render2dCompositionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConflictingResourceKinds { id, first, second } => write!(
                formatter,
                "2D semantic resource {} is referenced as both {first:?} and {second:?}",
                id.get()
            ),
            Self::ConflictingImageIntrinsicExtent { id, first, second } => write!(
                formatter,
                "2D image resource {} has conflicting intrinsic extents {first:?} and {second:?}",
                id.get()
            ),
            Self::ImageSourceOutsideIntrinsicExtent { id, patch_index } => write!(
                formatter,
                "2D image resource {} patch {patch_index} exceeds its exact intrinsic extent",
                id.get()
            ),
        }
    }
}

impl Error for Render2dCompositionError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f32, y: f32) -> Render2dPoint {
        Render2dPoint::new(x, y).expect("finite point")
    }

    fn rect() -> Render2dRect {
        Render2dRect::new(0.0, 0.0, 10.0, 10.0).expect("rect")
    }

    fn solid_item(color: Render2dColorRgba8) -> Render2dEntry {
        Render2dEntry::item(Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::Rect(rect()),
                brush: Render2dBrush::solid(color),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
        ))
    }

    #[test]
    fn nested_entries_preserve_exact_authored_order() {
        let first = solid_item(Render2dColorRgba8::BLACK);
        let second = solid_item(Render2dColorRgba8::WHITE);
        let group = Render2dEntry::group(Render2dGroup::new(
            vec![first.clone(), second.clone()],
            Render2dAffineTransform::translation(2.0, 3.0).expect("transform"),
            Vec::new(),
            Render2dOpacity::OPAQUE,
            Vec::new(),
        ));
        let composition = Render2dComposition::new(vec![group]).expect("composition");
        let Render2dEntry::Group(group) = &composition.root_entries()[0] else {
            unreachable!("group retained");
        };
        assert_eq!(group.entries(), &[first, second]);
    }

    #[test]
    fn one_resource_identity_cannot_mean_image_and_text() {
        let id = Render2dResourceId::new(1).expect("id");
        let extent = Render2dPixelExtent::new(2, 2).expect("extent");
        let image = Render2dEntry::item(Render2dItem::new(
            Render2dPrimitive::Image(
                Render2dImagePrimitive::new(id, extent, Vec::new()).expect("image"),
            ),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
        ));
        let text = Render2dEntry::item(Render2dItem::new(
            Render2dPrimitive::ShapedText(Render2dShapedTextPrimitive::new(
                id,
                point(0.0, 0.0),
                Render2dColorRgba8::WHITE,
            )),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
        ));
        assert!(matches!(
            Render2dComposition::new(vec![image, text]),
            Err(Render2dCompositionError::ConflictingResourceKinds {
                id: conflicting,
                ..
            }) if conflicting == id
        ));
    }

    #[test]
    fn fractional_and_zero_area_image_patches_are_retained_exactly() {
        let id = Render2dResourceId::new(3).expect("id");
        let extent = Render2dPixelExtent::new(8, 8).expect("extent");
        let source = Render2dImageSourceRect::new(0.5, 1.25, 0.0, 2.5).expect("source");
        let patch = Render2dImagePatch::new(source, Render2dRect::new(1.0, 2.0, 0.0, 4.0).expect("destination"));
        let image = Render2dImagePrimitive::new(id, extent, vec![patch]).expect("image");
        assert_eq!(image.patches(), &[patch]);
    }

    #[test]
    fn image_source_must_fit_composition_intrinsic_extent() {
        let id = Render2dResourceId::new(4).expect("id");
        let extent = Render2dPixelExtent::new(2, 2).expect("extent");
        let patch = Render2dImagePatch::new(
            Render2dImageSourceRect::new(1.5, 0.0, 1.0, 1.0).expect("source"),
            rect(),
        );
        assert_eq!(
            Render2dImagePrimitive::new(id, extent, vec![patch]),
            Err(Render2dCompositionError::ImageSourceOutsideIntrinsicExtent {
                id,
                patch_index: 0,
            })
        );
    }

    #[test]
    fn image_binding_must_match_exact_intrinsic_extent() {
        let id = Render2dResourceId::new(5).expect("id");
        let expected = Render2dPixelExtent::new(2, 2).expect("extent");
        let entry = Render2dEntry::item(Render2dItem::new(
            Render2dPrimitive::Image(
                Render2dImagePrimitive::new(id, expected, Vec::new()).expect("image"),
            ),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
        ));
        let composition = Render2dComposition::new(vec![entry]).expect("composition");

        let actual = Render2dPixelExtent::new(1, 4).expect("extent");
        let image = Render2dImageResource::new(actual, vec![0; 16]).expect("resource");
        let bindings = Render2dResourceBindings::new(vec![Render2dResourceBinding::new(
            id,
            Render2dResourceValue::ImageRgba8Srgb(image),
        )])
        .expect("bindings");

        assert_eq!(
            composition.validate_bindings(&bindings),
            Err(Render2dResourceBindingError::ImageIntrinsicExtentMismatch {
                id,
                expected,
                actual,
            })
        );
    }
}
