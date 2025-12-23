"""Application functions for applying metrics to font files."""

import sys
from typing import Dict, Tuple
from pathlib import Path

# Add project root to path for FontCore imports
# ruff: noqa: E402
_project_root = Path(__file__).parent.parent.parent
while (
    not (_project_root / "FontCore").exists() and _project_root.parent != _project_root
):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import FontCore.core_console_styles as cs
from FontCore.core_console_styles import get_console

from . import models

console = get_console()
FontMeasures = models.FontMeasures


def apply_metrics(fp: str, fm: FontMeasures, dry_run: bool) -> Tuple[bool, str]:
    """Apply computed metrics to font file."""
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(fp)
        orig_flavor = getattr(font, "flavor", None)

        old_vals: Dict[str, int] = {}
        new_vals: Dict[str, int] = {}

        os2 = font["OS/2"] if "OS/2" in font else None
        hhea = font["hhea"] if "hhea" in font else None

        # Gather old values
        if os2:
            old_vals["usWinAscent"] = int(getattr(os2, "usWinAscent", 0) or 0)
            old_vals["usWinDescent"] = int(getattr(os2, "usWinDescent", 0) or 0)
            old_vals["sTypoAscender"] = int(getattr(os2, "sTypoAscender", 0) or 0)
            old_vals["sTypoDescender"] = int(getattr(os2, "sTypoDescender", 0) or 0)
            old_vals["sTypoLineGap"] = int(getattr(os2, "sTypoLineGap", 0) or 0)
            old_vals["fsSelection"] = int(getattr(os2, "fsSelection", 0) or 0)
        if hhea:
            old_vals["hhea.ascent"] = int(getattr(hhea, "ascent", 0) or 0)
            old_vals["hhea.descent"] = int(getattr(hhea, "descent", 0) or 0)
            old_vals["hhea.lineGap"] = int(getattr(hhea, "lineGap", 0) or 0)

        # Compute new values (Win >= Typo already enforced in planning phase)
        win_asc = fm.target_win_asc or old_vals.get("usWinAscent", 0)
        win_desc = fm.target_win_desc or old_vals.get("usWinDescent", 0)
        typ_asc = fm.target_typo_asc or old_vals.get("sTypoAscender", 0)
        typ_desc = fm.target_typo_desc or old_vals.get("sTypoDescender", 0)

        new_vals = {
            "usWinAscent": win_asc,
            "usWinDescent": win_desc,
            "sTypoAscender": typ_asc,
            "sTypoDescender": typ_desc,
            "sTypoLineGap": 0,
            "hhea.ascent": typ_asc,
            "hhea.descent": typ_desc,
            "hhea.lineGap": 0,
        }

        # Determine changes
        diff_keys = [k for k, v in new_vals.items() if old_vals.get(k) != v]
        if not diff_keys:
            indicator = (
                cs.StatusIndicator("unchanged")
                .add_file(fp, filename_only=False)
                .add_message("(no metric changes)")
            )
            return False, indicator.build()

        # Track fsSelection USE_TYPO_METRICS bit (bit 7)
        if os2 and getattr(os2, "version", 0) >= 4:
            new_fs_selection = int(getattr(os2, "fsSelection", 0) or 0) | (1 << 7)
            if new_fs_selection != old_vals.get("fsSelection", 0):
                if "fsSelection" not in diff_keys:
                    diff_keys.append("fsSelection")
                new_vals["fsSelection"] = new_fs_selection
        else:
            new_vals["fsSelection"] = old_vals.get("fsSelection", 0)

        # Compute summary
        old_span = old_vals.get("sTypoAscender", 0) + abs(
            old_vals.get("sTypoDescender", 0)
        )
        new_span = new_vals["sTypoAscender"] + abs(new_vals["sTypoDescender"])
        span_diff = ((new_span - old_span) / float(fm.upm)) * 100 if fm.upm > 0 else 0

        # Format UPM indicator if different from family majority
        upm_note = ""
        if fm.family_upm_majority is not None and fm.upm != fm.family_upm_majority:
            upm_note = f" [dim](UPM: {fm.upm}, family majority: {fm.family_upm_majority})[/dim]"

        if dry_run:
            indicator = (
                cs.StatusIndicator("updated", dry_run=True)
                .add_file(fp, filename_only=False)
            )
            if upm_note:
                indicator.add_message(upm_note.strip())

            # Group OS/2 metrics
            os2_keys = [
                k
                for k in diff_keys
                if k
                in [
                    "usWinAscent",
                    "usWinDescent",
                    "sTypoAscender",
                    "sTypoDescender",
                    "sTypoLineGap",
                    "fsSelection",
                ]
            ]
            if os2_keys:
                indicator.add_item("[bold]OS/2 table:[/bold]", indent_level=0)
                for k in os2_keys:
                    old_v = old_vals.get(k, "—")
                    new_v = new_vals.get(k, "—")
                    if k == "fsSelection":
                        # Show USE_TYPO_METRICS bit state
                        old_bit7 = "set" if (int(old_v) & (1 << 7)) else "clear"
                        new_bit7 = "set" if (int(new_v) & (1 << 7)) else "clear"
                        indicator.add_item(
                            f"fsSelection (USE_TYPO_METRICS): {old_bit7} → {new_bit7}",
                            indent_level=1,
                        )
                    else:
                        indicator.add_item(
                            f"{k}: {cs.fmt_change(str(old_v), str(new_v))}",
                            indent_level=1,
                        )

            # Group hhea metrics
            hhea_keys = [k for k in diff_keys if k.startswith("hhea.")]
            if hhea_keys:
                indicator.add_item("[bold]hhea table:[/bold]", indent_level=0)
                for k in hhea_keys:
                    short_name = k.replace("hhea.", "")
                    indicator.add_item(
                        f"{short_name}: {cs.fmt_change(str(old_vals.get(k, '—')), str(new_vals.get(k, '—')))}",
                        indent_level=1,
                    )
            cs.emit("")
            indicator.add_item(
                f"[dim]vertical span:[/dim] {old_span} → {new_span} ({span_diff:+.1f}% UPM)",
                indent_level=1,
            )
            return False, indicator.build()

        # Apply changes
        if os2:
            os2.usWinAscent = int(win_asc)
            os2.usWinDescent = int(win_desc)
            os2.sTypoAscender = int(typ_asc)
            os2.sTypoDescender = int(typ_desc)
            os2.sTypoLineGap = 0
            try:
                # Always update sxHeight and sCapHeight with measured values
                # This ensures OS/2 metadata matches actual glyph measurements
                if fm.x_height and fm.x_height > 0:
                    os2.sxHeight = int(fm.x_height)
                if fm.cap_height and fm.cap_height > 0:
                    os2.sCapHeight = int(fm.cap_height)
            except Exception:
                pass
            try:
                if getattr(os2, "version", 0) >= 4:
                    os2.fsSelection = int(getattr(os2, "fsSelection", 0) or 0) | (
                        1 << 7
                    )
            except Exception:
                pass
        if hhea:
            hhea.ascent = int(typ_asc)
            hhea.descent = int(typ_desc)
            hhea.lineGap = 0

        font.flavor = orig_flavor
        font.save(fp)
        font.close()

        # Format UPM indicator if different from family majority
        upm_note = ""
        if fm.family_upm_majority is not None and fm.upm != fm.family_upm_majority:
            upm_note = f" [dim](UPM: {fm.upm}, family majority: {fm.family_upm_majority})[/dim]"

        # Compose report with grouped metrics using StatusIndicator
        indicator = cs.StatusIndicator("updated").add_file(fp, filename_only=False)
        if upm_note:
            indicator.add_message(upm_note.strip())

        # Group OS/2 metrics
        os2_keys = [
            k
            for k in diff_keys
            if k
            in [
                "usWinAscent",
                "usWinDescent",
                "sTypoAscender",
                "sTypoDescender",
                "sTypoLineGap",
                "fsSelection",
            ]
        ]
        if os2_keys:
            indicator.add_item("[bold]OS/2 table:[/bold]", indent_level=0)
            for k in os2_keys:
                old_v = old_vals.get(k, "—")
                new_v = new_vals.get(k, "—")
                if k == "fsSelection":
                    # Show USE_TYPO_METRICS bit state
                    old_bit7 = "set" if (int(old_v) & (1 << 7)) else "clear"
                    new_bit7 = "set" if (int(new_v) & (1 << 7)) else "clear"
                    indicator.add_item(
                        f"fsSelection (USE_TYPO_METRICS): {old_bit7} → {new_bit7}",
                        indent_level=1,
                    )
                else:
                    indicator.add_item(
                        f"{k}: {cs.fmt_change(str(old_v), str(new_v))}", indent_level=1
                    )

        # Group hhea metrics
        hhea_keys = [k for k in diff_keys if k.startswith("hhea.")]
        if hhea_keys:
            indicator.add_item("[bold]hhea table:[/bold]", indent_level=0)
            for k in hhea_keys:
                short_name = k.replace("hhea.", "")
                indicator.add_item(
                    f"{short_name}: {cs.fmt_change(str(old_vals.get(k, '—')), str(new_vals.get(k, '—')))}",
                    indent_level=1,
                )
        cs.emit("")
        indicator.add_item(
            f"[dim]vertical span:[/dim] {old_span} → {new_span} ({span_diff:+.1f}% UPM)",
            indent_level=1,
        )

        msg = indicator.build()
        return True, msg

    except Exception as e:
        indicator = (
            cs.StatusIndicator("error")
            .add_file(fp, filename_only=False)
            .with_explanation(str(e))
        )
        return False, indicator.build()


def process_all(measures, dry_run=False):
    updated = unchanged = errors = 0
    for fm in measures:
        ok, msg = apply_metrics(fm.path, fm, dry_run=dry_run)
        if dry_run:
            if "UNCHANGED" in msg or "unchanged" in msg:
                unchanged += 1
            else:
                updated += 1
        else:
            if ok:
                updated += 1
            elif "ERROR" in msg or "error" in msg:
                errors += 1
            else:
                unchanged += 1
        cs.emit(msg, console=console)

    cs.emit("")
    cs.StatusIndicator("success", dry_run=dry_run).add_message(
        "Processing Completed!"
    ).with_summary_block(
        updated=updated, unchanged=unchanged, errors=errors
    ).emit(console)

    return updated, unchanged, errors
