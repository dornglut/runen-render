#!/usr/bin/env python3
"""Regenerate the tiny deterministic RunenRender F2 font fixtures.

Requires fonttools==4.63.0. The generated files are owned test material and intentionally contain
only the tables needed to exercise scalable-outline and intrinsic-glyph classification paths.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from io import BytesIO
from pathlib import Path

import fontTools
from fontTools.colorLib.builder import buildCOLR, buildCPAL
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.S_V_G_ import SVGDocument
from fontTools.ttLib.tables.sbixGlyph import Glyph as SbixGlyph
from fontTools.ttLib.tables.sbixStrike import Strike

EXPECTED_FONTTOOLS = "4.63.0"
EXPECTED_SHA256 = {
    "f2_outline.ttf": "957683eb49945364e8324d713b72fcc5bc464412a20cc74f2cef30874248112c",
    "f2_colrv0.ttf": "993a004991d7deb87027eb94d7d2984f30c497595166e48d418a4b13d0ecf1da",
    "f2_colrv1.ttf": "c61233da173b9bb4a17a2dc21137b4f6cc2a90d0457ab2f7d83c65ce5e2596a3",
    "f2_svg.ttf": "a9bf4da3e5ebb40e0ea879fcccde1fb36be624283ec0e7c307246ae0429a83b6",
    "f2_bitmap.ttf": "6fa436c6862ebb8b7d1651906871725f2e685f320ac9b722539e4dcd19e03bb9",
}


def empty_glyph():
    return TTGlyphPen(None).glyph()


def box_glyph():
    pen = TTGlyphPen(None)
    pen.moveTo((100, 0))
    pen.lineTo((900, 0))
    pen.lineTo((900, 800))
    pen.lineTo((100, 800))
    pen.closePath()
    return pen.glyph()


def base_font(*, layer: bool = False):
    glyph_order = [".notdef", "space", "box"] + (["layer"] if layer else [])
    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder(glyph_order)
    builder.setupCharacterMap({0x20: "space", 0x41: "box"})
    glyphs = {".notdef": empty_glyph(), "space": empty_glyph(), "box": box_glyph()}
    if layer:
        glyphs["layer"] = box_glyph()
    builder.setupGlyf(glyphs)
    metrics = {name: (1000, 0) for name in glyph_order}
    metrics["space"] = (500, 0)
    builder.setupHorizontalMetrics(metrics)
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupOS2(
        sTypoAscender=800,
        sTypoDescender=-200,
        sTypoLineGap=0,
        usWinAscent=800,
        usWinDescent=200,
    )
    builder.setupNameTable(
        {
            "familyName": "RunenRender F2 Fixture",
            "styleName": "Regular",
            "uniqueFontIdentifier": "RunenRenderF2Fixture-Regular",
            "fullName": "RunenRender F2 Fixture Regular",
            "psName": "RunenRenderF2Fixture-Regular",
            "version": "Version 1.000",
        }
    )
    builder.setupPost()
    builder.setupMaxp()
    font = builder.font
    font.recalcTimestamp = False
    font["head"].created = 0
    font["head"].modified = 0
    return font


def encode(font) -> bytes:
    output = BytesIO()
    font.save(output, reorderTables=True)
    return output.getvalue()


def outline_font() -> bytes:
    return encode(base_font())


def colr_v0_font() -> bytes:
    font = base_font(layer=True)
    font["COLR"] = buildCOLR(
        {"box": [("layer", 0)]},
        version=0,
        glyphMap=font.getReverseGlyphMap(),
    )
    font["CPAL"] = buildCPAL([[(1.0, 0.0, 0.0, 1.0)]])
    return encode(font)


def colr_v1_font() -> bytes:
    font = base_font(layer=True)
    paint = {
        "Format": int(ot.PaintFormat.PaintGlyph),
        "Glyph": "layer",
        "Paint": {
            "Format": int(ot.PaintFormat.PaintSolid),
            "PaletteIndex": 0,
            "Alpha": 1.0,
        },
    }
    font["COLR"] = buildCOLR(
        {"box": paint},
        version=1,
        glyphMap=font.getReverseGlyphMap(),
    )
    font["CPAL"] = buildCPAL([[(1.0, 0.0, 0.0, 1.0)]])
    return encode(font)


def svg_font() -> bytes:
    font = base_font()
    svg = newTable("SVG ")
    svg.docList = [
        SVGDocument(
            '<svg xmlns="http://www.w3.org/2000/svg"><rect x="0" y="0" width="1000" height="1000"/></svg>',
            2,
            2,
            False,
        )
    ]
    font["SVG "] = svg
    return encode(font)


def rgba_png() -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFF_FFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00\xff", 9)
    return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def bitmap_font() -> bytes:
    font = base_font()
    sbix = newTable("sbix")
    strike = Strike(ppem=24, resolution=72)
    strike.glyphs["box"] = SbixGlyph(
        glyphName="box",
        originOffsetX=0,
        originOffsetY=0,
        graphicType="png ",
        imageData=rgba_png(),
        gid=2,
    )
    sbix.strikes[16] = strike
    font["sbix"] = sbix
    return encode(font)


def main() -> None:
    if fontTools.__version__ != EXPECTED_FONTTOOLS:
        raise SystemExit(
            f"expected fonttools {EXPECTED_FONTTOOLS}, observed {fontTools.__version__}"
        )
    outputs = {
        "f2_outline.ttf": outline_font(),
        "f2_colrv0.ttf": colr_v0_font(),
        "f2_colrv1.ttf": colr_v1_font(),
        "f2_svg.ttf": svg_font(),
        "f2_bitmap.ttf": bitmap_font(),
    }
    root = Path(__file__).resolve().parent
    for name, data in outputs.items():
        digest = hashlib.sha256(data).hexdigest()
        if digest != EXPECTED_SHA256[name]:
            raise SystemExit(f"{name}: expected {EXPECTED_SHA256[name]}, observed {digest}")
        (root / name).write_bytes(data)
        print(f"{name}: {len(data)} bytes sha256={digest}")


if __name__ == "__main__":
    main()
