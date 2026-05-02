"""Read-only MVAR/HVAR inspection for variable fonts.

Uses the same VarStoreInstancer + normalized location path as fontTools.varLib.mutator.
Does not modify fonts. Intended to decide whether MVAR/HVAR encode meaningful deltas
before any normalization logic consumes them."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import FontCore.core_console_styles as cs
from FontCore.core_console_styles import get_console
from fontTools.misc.fixedTools import floatToFixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools.ttLib import TTFont
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.models import normalizeLocation, piecewiseLinearMap
from fontTools.varLib.varStore import NO_VARIATION_INDEX, VarStoreInstancer

# Tags most relevant to ebrium-style vertical typo / line box (+ Win fallbacks).
MVAR_VERTICAL_LINE_TAGS = frozenset(
    {"hasc", "hdsc", "hlgp", "hcla", "hcld", "xhgt", "cpht"}
)


def normalized_variation_location(
    varfont: TTFont, location_user: Mapping[str, float]
) -> Dict[str, float]:
    """Normalize user-space coordinates (−1..1-ish) applying avar then F2Dot14 quantization."""
    fvar = varfont["fvar"]
    axes = {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in fvar.axes}
    loc = normalizeLocation(dict(location_user), axes)
    if "avar" in varfont:
        maps = varfont["avar"].segments
        loc = {k: piecewiseLinearMap(v, maps[k]) for k, v in loc.items()}
    return {k: floatToFixedToFloat(v, 14) for k, v in loc.items()}


def _ascii_tag(tag: object) -> str:
    if isinstance(tag, bytes):
        return tag.decode("latin-1")
    return str(tag)


def axis_pole_user_locations(varfont: TTFont) -> List[Tuple[str, Dict[str, float]]]:
    """Default + each axis pinned to min or max while others stay at fvar defaults."""
    fvar = varfont["fvar"]
    defaults = {a.axisTag: float(a.defaultValue) for a in fvar.axes}
    out: List[Tuple[str, Dict[str, float]]] = [
        ("default (axis default values)", dict(defaults))
    ]
    for a in fvar.axes:
        t = a.axisTag
        lo = dict(defaults)
        lo[t] = float(a.minValue)
        out.append((f"{t}=min · others default", lo))
        hi = dict(defaults)
        hi[t] = float(a.maxValue)
        out.append((f"{t}=max · others default", hi))
    return out


def mvar_delta_map(varfont: TTFont, loc_norm: Dict[str, float]) -> Dict[str, int]:
    """Rounded MVAR deltas at a normalized location (0 at defaults)."""
    mvar_tbl = varfont["MVAR"].table
    inst = VarStoreInstancer(mvar_tbl.VarStore, varfont["fvar"].axes, loc_norm)
    out: Dict[str, int] = {}
    for rec in mvar_tbl.ValueRecord:
        tag = _ascii_tag(rec.ValueTag)
        vidx = rec.VarIdx
        if vidx == NO_VARIATION_INDEX:
            d = 0
        else:
            d = otRound(inst[vidx])
        out[tag] = int(d)
    return out


def mvar_aggregate_ranges(
    varfont: TTFont, samples: Sequence[Tuple[str, Dict[str, float]]]
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, str]]:
    """Per tag: (min_delta, max_delta) across samples; unknown tags tracked separately."""
    per_tag_vals: Dict[str, List[int]] = {}
    unknown_tags: Dict[str, str] = {}

    for _label, user_loc in samples:
        ln = normalized_variation_location(varfont, user_loc)
        snap = mvar_delta_map(varfont, ln)
        for tag, d in snap.items():
            per_tag_vals.setdefault(tag, []).append(int(d))
            if tag not in MVAR_ENTRIES:
                unknown_tags.setdefault(tag, "no OpenType registry mapping in FontTools")

    ranges: Dict[str, Tuple[int, int]] = {}
    for tag, vals in per_tag_vals.items():
        ranges[tag] = (min(vals), max(vals))
    return ranges, unknown_tags


def _sample_glyph_names(varfont: TTFont, limit: int = 200) -> List[str]:
    order = list(varfont.getGlyphOrder())
    cmap = varfont.getBestCmap() or {}
    preferred: List[str] = []
    for cp in list(range(0x0041, 0x005B)) + list(range(0x0061, 0x007B)) + list(
        range(0x0030, 0x003A)
    ):
        gn = cmap.get(cp)
        if gn and gn not in preferred:
            preferred.append(str(gn))
    for gn in order:
        if gn and gn != ".notdef" and gn not in preferred:
            preferred.append(str(gn))
        if len(preferred) >= limit:
            break
    return preferred[:limit]


def hvar_advance_delta_stats(
    varfont: TTFont,
    loc_norm: Dict[str, float],
    glyph_names: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, object]]:
    """Summarize advance-width deltas via HVAR at one location."""
    if "HVAR" not in varfont:
        return None
    hvar = varfont["HVAR"].table
    vstore = getattr(hvar, "VarStore", None)
    if vstore is None:
        return None

    names = list(glyph_names) if glyph_names else _sample_glyph_names(varfont)
    inst = VarStoreInstancer(vstore, varfont["fvar"].axes, loc_norm)
    order = varfont.getGlyphOrder()
    index_of = {str(g): i for i, g in enumerate(order)}

    deltas: List[int] = []
    skipped = 0
    map_obj = getattr(hvar, "AdvWidthMap", None)
    mapping = getattr(map_obj, "mapping", None) if map_obj is not None else None

    for g in names:
        if g not in index_of:
            skipped += 1
            continue
        gid = index_of[g]
        if mapping is not None:
            if g not in mapping:
                continue
            widx = mapping[g]
        else:
            widx = gid
        if widx == NO_VARIATION_INDEX:
            deltas.append(0)
        else:
            deltas.append(int(otRound(inst[widx])))

    nonz = [d for d in deltas if d != 0]
    return {
        "glyphs_checked": len(deltas),
        "skipped_unmapped": skipped,
        "nonzero_advances": len(nonz),
        "max_abs_delta": max(abs(d) for d in deltas) if deltas else 0,
        "mean_abs_nonzero": (sum(abs(x) for x in nonz) / len(nonz)) if nonz else 0.0,
    }


def probe_font_file(path: str, verbose: bool = False) -> None:
    """Print MVAR/HVAR summary for one font path."""
    console = get_console()
    fp = Path(path)
    try:
        font = TTFont(str(fp))
    except Exception as e:
        cs.StatusIndicator("error").add_file(path, filename_only=False).with_explanation(
            str(e)
        ).emit(console)
        return

    try:
        if "fvar" not in font:
            cs.StatusIndicator("unchanged").add_file(path, filename_only=False).add_message(
                "No fvar table — treat as static for variation metrics.",
            ).emit(console)
            return

        fvar = font["fvar"]
        axis_line = ", ".join(
            f"{a.axisTag} [{a.minValue:g} … {a.defaultValue:g} … {a.maxValue:g}]"
            for a in fvar.axes
        )

        ind = (
            cs.StatusIndicator("info")
            .add_file(str(fp), filename_only=False)
            .add_message(f"[dim]Axes:[/dim] {axis_line}")
        )

        samples = axis_pole_user_locations(font)
        default_norm = normalized_variation_location(
            font, samples[0][1]
        )

        if "MVAR" in font:
            ranges, unknown = mvar_aggregate_ranges(font, samples)
            nrec = len(font["MVAR"].table.ValueRecord)
            ind.add_item(f"MVAR: {nrec} value record(s)", indent_level=1)

            # Planning-relevant summary
            vert_lines: List[str] = []
            for tag in sorted(MVAR_VERTICAL_LINE_TAGS & set(ranges.keys())):
                lo, hi = ranges[tag]
                if lo == 0 and hi == 0:
                    continue
                entry = MVAR_ENTRIES.get(tag, ("?", "?"))
                vert_lines.append(
                    f"{tag} → {entry[0]}.{entry[1]}: delta range [{lo:+d} … {hi:+d}]"
                )
            if vert_lines:
                ind.add_item(
                    "[bold]Vertical / line-box tags (non-zero range over axis poles):[/bold]",
                    indent_level=1,
                )
                for line in vert_lines[:20]:
                    ind.add_item(line, indent_level=2)
                if len(vert_lines) > 20:
                    ind.add_item(f"… and {len(vert_lines) - 20} more", indent_level=2)
            else:
                ind.add_item(
                    "MVAR vertical/typo tags all zero across pole samples — "
                    "authored metrics likely stable vs axes (for these tags).",
                    indent_level=1,
                )

            if unknown:
                ind.add_item(
                    f"[warning]{len(unknown)} unknown ValueTag(s) (not in FontTools "
                    "MVAR_ENTRIES registry)[/warning]",
                    indent_level=1,
                )
                if verbose:
                    for ut in sorted(unknown.keys())[:12]:
                        ind.add_item(f"tag {ut!r}", indent_level=2)

            all_zero = all(lo == 0 and hi == 0 for lo, hi in ranges.values())
            if verbose or not all_zero:
                other = sorted(
                    t
                    for t, (lo, hi) in ranges.items()
                    if t not in MVAR_VERTICAL_LINE_TAGS and (lo != 0 or hi != 0)
                )
                if other and verbose:
                    ind.add_item("Other tags with variation:", indent_level=1)
                    for t in other[:25]:
                        lo, hi = ranges[t]
                        ind.add_item(f"{t}: [{lo:+d} … {hi:+d}]", indent_level=2)

            if verbose:
                ind.add_item("[dim]Per-pole deltas (non-zero only):[/dim]", indent_level=1)
                for label, uloc in samples:
                    ln = normalized_variation_location(font, uloc)
                    zm = mvar_delta_map(font, ln)
                    nz = [f"{k}={v:+d}" for k, v in sorted(zm.items()) if v != 0]
                    if nz:
                        ind.add_item(f"{label}: {', '.join(nz)}", indent_level=2)
        else:
            ind.add_item("MVAR: absent", indent_level=1)

        if "HVAR" in font:
            st_def = hvar_advance_delta_stats(font, default_norm)
            # Worst-case among poles for max-abs advance delta
            worst_max = 0
            worst_label = ""
            for label, uloc in samples:
                ln = normalized_variation_location(font, uloc)
                st = hvar_advance_delta_stats(font, ln)
                if st and isinstance(st["max_abs_delta"], int):
                    if st["max_abs_delta"] > worst_max:
                        worst_max = int(st["max_abs_delta"])
                        worst_label = label
            sample_n = len(_sample_glyph_names(font))
            line = (
                f"HVAR: sampled up to {sample_n} glyph(s) — "
                f"default max |Δadvance|={st_def['max_abs_delta'] if st_def else 0}"
            )
            if worst_max > (st_def["max_abs_delta"] if st_def else 0):
                # Avoid literal [...] here — Rich markup eats bracket groups and can blank the label.
                pole = worst_label or "?"
                line += f"; worst sample: {pole} → max |Δadvance|={worst_max}"
            ind.add_item(line, indent_level=1)
            if verbose and st_def:
                ind.add_item(
                    f"defaults: glyphs checked={st_def['glyphs_checked']}, "
                    f"non-zero advances={st_def['nonzero_advances']}",
                    indent_level=2,
                )
        else:
            ind.add_item("HVAR: absent", indent_level=1)

        ind.emit(console)
    finally:
        font.close()


def run_probe(paths: Iterable[str], verbose: bool = False) -> None:
    """Probe every path in the iterable."""
    console = get_console()
    lst = list(paths)
    if not lst:
        cs.StatusIndicator("error").add_message("No font files to probe").emit(console)
        return
    for p in lst:
        probe_font_file(p, verbose=verbose)
    cs.emit("", console=console)
