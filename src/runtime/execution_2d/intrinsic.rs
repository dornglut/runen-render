//! Fail-closed intrinsic glyph admission without convenience APIs that erase parse errors.

use crate::composition_2d::Render2dResourceId;
use crate::execution_2d::{Render2dShapedTextError, Render2dUnsupportedGlyphKind};
use skrifa::raw::{
    FontData, FontRef, ReadError, TableProvider,
    tables::{
        bitmap::BitmapSize, cblc::Cblc, colr::Colr, eblc::Eblc, sbix::Sbix, svg::SVGDocumentList,
    },
    types::{GlyphId, Tag},
};

pub(super) struct IntrinsicGlyphs<'a> {
    resource_id: Render2dResourceId,
    colr: Option<Colr<'a>>,
    svg: Option<SVGDocumentList<'a>>,
    sbix: Option<Sbix<'a>>,
    cblc: Option<Cblc<'a>>,
    eblc: Option<Eblc<'a>>,
}

impl<'a> IntrinsicGlyphs<'a> {
    pub(super) fn new(
        font: &FontRef<'a>,
        resource_id: Render2dResourceId,
    ) -> Result<Self, Render2dShapedTextError> {
        let colr = optional(font, b"COLR", font.colr(), resource_id)?;
        if colr.as_ref().is_some_and(|table| table.version() > 1) {
            return Err(Render2dShapedTextError::InvalidFont { resource_id });
        }
        let svg_table = optional(font, b"SVG ", font.svg(), resource_id)?;
        if svg_table.as_ref().is_some_and(|table| table.version() != 0) {
            return Err(Render2dShapedTextError::InvalidFont { resource_id });
        }
        let svg = svg_table
            .map(|table| table.svg_document_list())
            .transpose()
            .map_err(|_| Render2dShapedTextError::InvalidFont { resource_id })?;
        if svg
            .as_ref()
            .is_some_and(|list| usize::from(list.num_entries()) != list.document_records().len())
        {
            return Err(Render2dShapedTextError::InvalidFont { resource_id });
        }
        let sbix = optional(font, b"sbix", font.sbix(), resource_id)?;
        let cblc = optional(font, b"CBLC", font.cblc(), resource_id)?;
        let cbdt = optional(font, b"CBDT", font.cbdt(), resource_id)?;
        let eblc = optional(font, b"EBLC", font.eblc(), resource_id)?;
        let ebdt = optional(font, b"EBDT", font.ebdt(), resource_id)?;
        if sbix.as_ref().is_some_and(|table| {
            table.version() != 1
                || usize::try_from(table.num_strikes()) != Ok(table.strike_offsets().len())
        }) || cblc.as_ref().is_some_and(|table| {
            usize::try_from(table.num_sizes()) != Ok(table.bitmap_sizes().len())
        }) || eblc.as_ref().is_some_and(|table| {
            usize::try_from(table.num_sizes()) != Ok(table.bitmap_sizes().len())
        }) {
            return Err(Render2dShapedTextError::InvalidFont { resource_id });
        }
        if cblc.is_some() != cbdt.is_some() || eblc.is_some() != ebdt.is_some() {
            return Err(Render2dShapedTextError::InvalidFont { resource_id });
        }
        Ok(Self {
            resource_id,
            colr,
            svg,
            sbix,
            cblc,
            eblc,
        })
    }

    pub(super) fn admit(&self, glyph_id: GlyphId) -> Result<(), Render2dShapedTextError> {
        let invalid = || Render2dShapedTextError::InvalidFont {
            resource_id: self.resource_id,
        };
        let unsupported = |kind| Render2dShapedTextError::UnsupportedGlyph {
            resource_id: self.resource_id,
            glyph_id: Some(glyph_id.to_u32()),
            kind,
        };
        if let Some(svg) = &self.svg {
            // Classify from declared glyph ranges, even when the SVG payload is corrupt.
            if svg
                .document_records()
                .iter()
                .any(|record| (record.start_glyph_id()..=record.end_glyph_id()).contains(&glyph_id))
            {
                return Err(unsupported(Render2dUnsupportedGlyphKind::Svg));
            }
        }
        if let Some(colr) = &self.colr {
            if let Some(list) = colr.base_glyph_list().transpose().map_err(|_| invalid())? {
                if usize::try_from(list.num_base_glyph_paint_records())
                    != Ok(list.base_glyph_paint_records().len())
                {
                    return Err(invalid());
                }
                if list
                    .base_glyph_paint_records()
                    .iter()
                    .any(|record| u32::from(record.glyph_id().to_u16()) == glyph_id.to_u32())
                {
                    return Err(unsupported(Render2dUnsupportedGlyphKind::ColrV1));
                }
            }
            if let Some(records) = colr
                .base_glyph_records()
                .transpose()
                .map_err(|_| invalid())?
            {
                if records.len() != usize::from(colr.num_base_glyph_records()) {
                    return Err(invalid());
                }
                if records
                    .iter()
                    .any(|record| u32::from(record.glyph_id().to_u16()) == glyph_id.to_u32())
                {
                    return Err(unsupported(Render2dUnsupportedGlyphKind::ColrV0));
                }
            } else if colr.num_base_glyph_records() != 0 {
                return Err(invalid());
            }
        }
        if let Some(sbix) = &self.sbix {
            // Inspect every strike and every sbix graphic type, including duplicate/JPEG/TIFF.
            for strike in sbix.strikes().iter() {
                let strike = strike.map_err(|_| invalid())?;
                if strike
                    .glyph_data(glyph_id)
                    .map_err(|_| invalid())?
                    .is_some()
                {
                    return Err(unsupported(Render2dUnsupportedGlyphKind::Bitmap));
                }
            }
        }
        for (sizes, data) in self
            .cblc
            .iter()
            .map(|table| (table.bitmap_sizes(), table.offset_data()))
            .chain(
                self.eblc
                    .iter()
                    .map(|table| (table.bitmap_sizes(), table.offset_data())),
            )
        {
            if has_bitmap(sizes, data, glyph_id).map_err(|_| invalid())? {
                return Err(unsupported(Render2dUnsupportedGlyphKind::Bitmap));
            }
        }
        Ok(())
    }
}

fn optional<T>(
    font: &FontRef<'_>,
    tag: &[u8; 4],
    parsed: Result<T, ReadError>,
    resource_id: Render2dResourceId,
) -> Result<Option<T>, Render2dShapedTextError> {
    // Directory presence matters: out-of-bounds/zero-offset tables must not look absent.
    if font
        .table_directory()
        .table_records()
        .iter()
        .any(|record| record.tag() == Tag::new(tag))
    {
        parsed
            .map(Some)
            .map_err(|_| Render2dShapedTextError::InvalidFont { resource_id })
    } else {
        Ok(None)
    }
}

fn has_bitmap(
    sizes: &[BitmapSize],
    data: FontData<'_>,
    glyph_id: GlyphId,
) -> Result<bool, ReadError> {
    for size in sizes {
        if !(size.start_glyph_index()..=size.end_glyph_index()).contains(&glyph_id) {
            continue;
        }
        let list = size.index_subtable_list(data)?;
        if usize::try_from(size.number_of_index_subtables())
            != Ok(list.index_subtable_records().len())
        {
            return Err(ReadError::OutOfBounds);
        }
        if !list.index_subtable_records().iter().any(|record| {
            (record.first_glyph_index()..=record.last_glyph_index()).contains(&glyph_id)
        }) {
            continue;
        }
        match size.location(data, glyph_id) {
            Ok(location) if !location.is_empty() => return Ok(true),
            Ok(_) | Err(ReadError::InvalidCollectionIndex(_)) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(false)
}
