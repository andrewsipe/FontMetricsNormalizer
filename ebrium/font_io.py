"""Font I/O helper functions for reading and analyzing font files."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from fontTools.ttLib import TTFont


def _read_ttfont(path: str):
    from fontTools.ttLib import TTFont

    return TTFont(path)


def _get_upm(font: TTFont) -> int:
    return int(font["head"].unitsPerEm)


def _get_best_cmap(font: TTFont) -> Dict[int, str]:
    try:
        cmap = font.getBestCmap()
        if cmap:
            return dict(cmap)
    except Exception:
        pass
    # fallback: merge all subtables
    mapping: Dict[int, str] = {}
    try:
        for st in font["cmap"].tables:
            if getattr(st, "cmap", None):
                mapping.update(st.cmap)
    except Exception:
        pass
    return mapping


def _glyph_bounds(
    font: TTFont, glyph_name: str
) -> Optional[Tuple[float, float, float, float]]:
    try:
        glyph_set = font.getGlyphSet()
        if glyph_name not in glyph_set:
            return None
        from fontTools.pens.boundsPen import BoundsPen

        pen = BoundsPen(glyph_set)
        glyph_set[glyph_name].draw(pen)
        if pen.bounds is None:
            return None
        xMin, yMin, xMax, yMax = pen.bounds
        return float(xMin), float(yMin), float(xMax), float(yMax)
    except Exception:
        return None


def _codepoint_bounds(
    font: TTFont, codepoint: int
) -> Optional[Tuple[float, float, float, float]]:
    cmap = _get_best_cmap(font)
    name = cmap.get(codepoint)
    if not name:
        return None
    return _glyph_bounds(font, name)


def _font_cmap_glyph_names(font: TTFont) -> List[str]:
    try:
        cmap = _get_best_cmap(font)
        names = list({name for name in cmap.values() if isinstance(name, str)})
        if names:
            return names
    except Exception:
        pass
    # fallback: enumerate glyphOrder
    try:
        return list(font.getGlyphOrder())
    except Exception:
        return []


def _font_overall_bounds(font: TTFont) -> Optional[Tuple[float, float]]:
    names = _font_cmap_glyph_names(font)
    if not names:
        return None
    min_y = None
    max_y = None
    for gn in names:
        b = _glyph_bounds(font, gn)
        if not b:
            continue
        _, yMin, _, yMax = b
        if min_y is None or yMin < min_y:
            min_y = yMin
        if max_y is None or yMax > max_y:
            max_y = yMax
    if min_y is None or max_y is None:
        return None
    return float(min_y), float(max_y)
