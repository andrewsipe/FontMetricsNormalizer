"""Metrics planning functions for computing normalization targets."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from FontCore.core_logging_config import Verbosity

from . import clustering
from . import config
from . import font_io
from . import models

console = get_console()
MetricsConfig = config.MetricsConfig
FontMeasures = models.FontMeasures

# Import clustering functions
detect_optical_clusters = clustering.detect_optical_clusters

# Import font I/O for analyze_family_impact
_read_ttfont = font_io._read_ttfont


def compute_family_normalized_extremes(
    measures: List[FontMeasures],
) -> Tuple[float, float]:
    """Compute family-wide extremes in normalized units (handles mixed UPMs)."""
    norm_mins: List[float] = []
    norm_maxs: List[float] = []

    for fm in measures:
        # Skip fonts excluded from calculations
        if getattr(fm, "is_excluded_from_calculations", False):
            continue
        if fm.min_y is None or fm.max_y is None or fm.upm <= 0:
            continue
        norm_mins.append(fm.min_y / fm.upm)
        norm_maxs.append(fm.max_y / fm.upm)

    if not norm_mins or not norm_maxs:
        return (-1.0, 1.0)  # Conservative fallback

    return (min(norm_mins), max(norm_maxs))


def compute_descender_for_centering(
    typo_asc: int, cap_h: int, actual_desc: Optional[int]
) -> int:
    """Center cap height within typo bounds.

    Returns deeper of:
    - Calculated descender (centers caps)
    - Actual descender (prevents clipping)

    Args:
        typo_asc: Typographic ascender value
        cap_h: Cap height value
        actual_desc: Actual measured descender (negative value, or None)

    Returns:
        Descender value (negative integer)
    """
    # Calculate descender that centers cap height within typo bounds
    desired_desc = -(typo_asc - cap_h)

    # Clamp to actual descender to prevent clipping
    # (min because descenders are negative - more negative = deeper)
    if actual_desc is not None:
        return min(desired_desc, actual_desc)

    return desired_desc


def compute_cluster_baseline_descender(
    cluster: List[FontMeasures],
    exclude_decorative: bool = True,
) -> Optional[float]:
    """Compute normalized descender for baseline alignment across cluster.

    Uses median of actual descenders in normalized units to maintain
    consistent baseline across fonts in the cluster. Should be called with
    main_cluster only (excluding decorative outliers).

    Args:
        cluster: Cluster of fonts to compute baseline descender for
            (should be main_cluster, excluding decorative outliers)
        exclude_decorative: If True, exclude decorative outliers from calculation

    Returns:
        Normalized descender value (negative float), or None if insufficient data
    """
    # Collect normalized descenders from cluster (excluding decorative if requested)
    norm_descenders: List[float] = []
    for fm in cluster:
        if exclude_decorative and getattr(fm, "is_decorative_outlier", False):
            continue
        if fm.descender_min and fm.upm > 0:
            norm_descenders.append(abs(float(fm.descender_min) / float(fm.upm)))

    if not norm_descenders:
        return None

    # Use median normalized descender for baseline consistency
    norm_descenders.sort()
    n = len(norm_descenders)
    norm_median_desc = (
        norm_descenders[n // 2]
        if n % 2 == 1
        else (norm_descenders[n // 2 - 1] + norm_descenders[n // 2]) / 2.0
    )

    return -norm_median_desc


def compute_cluster_target_percent(
    cluster: List[FontMeasures],
    config: MetricsConfig,
) -> float:
    """Compute recommended target for entire cluster (preserves alignment).

    If auto_adjust_target is enabled, recommends target based on median
    x-height to cap-height ratio across the cluster.

    Args:
        cluster: Cluster of fonts to compute target for
        config: Configuration with auto_adjust_target flag

    Returns:
        Recommended target span (never less than config.target_span)
    """
    if not config.auto_adjust_target:
        return config.target_span

    # Calculate x-height to cap-height ratios for all fonts in cluster
    x_ratios = [
        fm.x_height / fm.cap_height
        for fm in cluster
        if fm.x_height and fm.cap_height and fm.cap_height > 0
    ]

    if not x_ratios:
        return config.target_span

    # Use MEDIAN x-ratio for cluster (preserves baseline alignment)
    x_ratios.sort()
    n = len(x_ratios)
    median_x_ratio = (
        x_ratios[n // 2]
        if n % 2 == 1
        else (x_ratios[n // 2 - 1] + x_ratios[n // 2]) / 2.0
    )

    # Recommend based on median
    if median_x_ratio < 0.55:
        recommended = 1.20  # Compact spacing
    elif median_x_ratio < 0.65:
        recommended = 1.25  # Standard spacing
    elif median_x_ratio < 0.75:
        recommended = 1.30  # Large x-height
    else:
        recommended = 1.35  # Very large x-height

    # Use maximum (never reduce below user config)
    return max(config.target_span, recommended)


def compute_family_normalized_ascender(
    measures: List[FontMeasures], config: MetricsConfig
) -> float:
    """Compute family-wide ascender target based on cap height + fixed headroom.

    Strategy:
    1. Start with max cap height + configured headroom (e.g., 25% UPM)
    2. If any font's actual ascenders exceed this baseline significantly,
       use the tallest ascender instead (preserves actual geometry)
    3. Adjust headroom for unicase fonts (less headroom needed)
    """
    cap_ratios: List[float] = []
    ascender_ratios: List[float] = []

    # Filter out excluded fonts for calculations
    included_measures = [
        fm for fm in measures if not getattr(fm, "is_excluded_from_calculations", False)
    ]

    # Check if this is a unicase-only cluster (using included fonts only)
    is_unicase_cluster = (
        all(fm.is_unicase for fm in included_measures) and len(included_measures) > 0
    )

    for fm in included_measures:
        if fm.upm <= 0:
            continue
        cap = fm.cap_optical or fm.cap_height
        if cap:
            cap_ratios.append(cap / fm.upm)

        # Collect ascender measurements (if available)
        if fm.ascender_max and fm.ascender_max > 0:
            ascender_ratios.append(fm.ascender_max / fm.upm)

    if not cap_ratios:
        return 0.85  # Default fallback

    max_cap_ratio = max(cap_ratios)

    # Adjust top margin for unicase fonts (less margin needed)
    top_margin = config.top_margin * 0.65 if is_unicase_cluster else config.top_margin
    baseline_ascender = max_cap_ratio + top_margin

    # Check if actual ascenders exceed baseline
    if ascender_ratios:
        max_ascender_ratio = max(ascender_ratios)

        # Define "significant excess" using configurable threshold
        # Default: 50% of top_margin, or minimum 2% UPM (whichever is larger)
        significant_threshold = max(
            config.top_margin * config.ascender_override_threshold, 0.02
        )

        # If ascenders significantly exceed baseline, use actual ascender height
        # This preserves the font's actual geometry
        if max_ascender_ratio > (baseline_ascender + significant_threshold):
            cs.StatusIndicator("info").add_message(
                f"Ascenders exceed baseline by {((max_ascender_ratio - baseline_ascender) * 100):.1f}% UPM - using actual ascender height"
            ).emit(console)
            return max_ascender_ratio

        # If ascenders are close to baseline, use the larger of the two
        # (prevents descenders from being unnecessarily deep)
        if max_ascender_ratio > baseline_ascender:
            return max_ascender_ratio

    return baseline_ascender


def plan_identical_metrics(
    cluster: List[FontMeasures],
    family_norm_min: float,
    family_norm_max: float,
    family_norm_asc: float,
    config: MetricsConfig,
    verbosity: Verbosity = Verbosity.BRIEF,
) -> None:
    """Apply identical normalization to optically identical fonts.

    Works in normalized units, scales to each font's UPM.

    Note: This function intentionally does not apply per-font x-height adjustments
    (unlike plan_adaptive_metrics()) because identical clusters receive uniform
    treatment by design. All fonts in the cluster get the same normalized metrics
    regardless of minor x-height variations.
    """
    if not cluster:
        return

    # Compute shared cap_height ratio (median in normalized units)
    cap_height_ratios: List[float] = []
    for fm in cluster:
        cap = fm.cap_optical or fm.cap_height
        if cap and fm.upm > 0:
            cap_height_ratios.append(float(cap) / float(fm.upm))

    if not cap_height_ratios:
        norm_cap_h = 0.7
    else:
        cap_height_ratios.sort()
        n = len(cap_height_ratios)
        norm_cap_h = (
            cap_height_ratios[n // 2]
            if n % 2 == 1
            else (cap_height_ratios[n // 2 - 1] + cap_height_ratios[n // 2]) / 2.0
        )

    # Find deepest normalized descender
    norm_descenders: List[float] = []
    for fm in cluster:
        if fm.descender_min and fm.upm > 0:
            norm_descenders.append(abs(float(fm.descender_min) / float(fm.upm)))

    # Compute normalized typo metrics
    norm_typo_asc = family_norm_asc
    norm_desired_desc = -(norm_typo_asc - norm_cap_h)

    # Clamp to deepest actual descender (prevents clipping)
    if norm_descenders:
        norm_descender_limit = max(norm_descenders)
        if abs(norm_desired_desc) < norm_descender_limit:
            norm_desired_desc = -norm_descender_limit

    # Compute cluster-wide target (category-aware if enabled)
    cluster_target = compute_cluster_target_percent(cluster, config)

    # Apply target percent adjustment in normalized space (minimum only)
    norm_span = norm_typo_asc + abs(norm_desired_desc)

    # Only expand if below minimum target
    # No upper limit - let actual geometry determine final span
    if norm_span < cluster_target:
        extra_amt = cluster_target - norm_span
        # Expand both (60/40 split) to maintain centering
        norm_typo_asc += extra_amt * 0.6
        norm_desired_desc -= extra_amt * 0.4

    # Apply to each font, scaling to its UPM
    for fm in cluster:
        upm = fm.upm
        # Win metrics (family-wide extremes, scaled per UPM)
        fm.target_win_asc = int(
            round(max(family_norm_max * upm * (1.0 + config.win_buffer), 0))
        )
        fm.target_win_desc = int(
            round(abs(family_norm_min * upm) * (1.0 + config.win_buffer))
        )
        # Typo metrics (normalized and scaled to this font's UPM)
        fm.target_typo_asc = int(round(norm_typo_asc * upm))
        fm.target_typo_desc = int(round(norm_desired_desc * upm))

        # Validate: ensure typo descender doesn't exceed actual descender (prevent clipping)
        if fm.descender_min and fm.target_typo_desc:
            if fm.target_typo_desc > fm.descender_min:
                # Clipping would occur - adjust
                fm.target_typo_desc = fm.descender_min


def finalize_metrics(fm: FontMeasures) -> None:
    """Ensure Win >= Typo after all planning."""
    if fm.target_win_asc is not None and fm.target_typo_asc is not None:
        fm.target_win_asc = max(fm.target_win_asc, fm.target_typo_asc)
    if fm.target_win_desc is not None and fm.target_typo_desc is not None:
        fm.target_win_desc = max(fm.target_win_desc, abs(fm.target_typo_desc))


def get_cluster_normalized_typo(
    cluster: List[FontMeasures],
) -> Tuple[float, float]:
    """Get representative normalized typo metrics from cluster (after all adjustments)."""
    asc_ratios = [
        fm.target_typo_asc / fm.upm
        for fm in cluster
        if fm.target_typo_asc is not None and fm.upm > 0
    ]
    desc_ratios = [
        fm.target_typo_desc / fm.upm
        for fm in cluster
        if fm.target_typo_desc is not None and fm.upm > 0
    ]
    norm_asc = sum(asc_ratios) / len(asc_ratios) if asc_ratios else 0.85
    norm_desc = sum(desc_ratios) / len(desc_ratios) if desc_ratios else -0.25
    return (norm_asc, norm_desc)


def plan_safe_metrics(group: List[FontMeasures], config: MetricsConfig) -> None:
    """Conservative bbox approach - all fonts get same normalized bounds.

    Win and Typo both use family extremes. This guarantees no clipping but
    wastes vertical space. Good for extremely decorative fonts or fonts with
    unexpected outliers (mathematical symbols, dingbats).
    """
    fam_min, fam_max = compute_family_normalized_extremes(group)

    for fm in group:
        upm = fm.upm
        # Win and Typo both use family extremes
        win_asc = int(round(fam_max * upm * (1.0 + config.win_buffer)))
        win_desc = int(round(abs(fam_min * upm) * (1.0 + config.win_buffer)))

        fm.target_win_asc = win_asc
        fm.target_win_desc = win_desc
        fm.target_typo_asc = win_asc
        fm.target_typo_desc = -win_desc
        # Finalize ensures consistency (should be no-op here, but safe)
        finalize_metrics(fm)


def validate_cluster_consistency(cluster: List[FontMeasures]) -> None:
    """Validate that cluster fonts have identical normalized typo ratios."""
    if len(cluster) <= 1:
        return
    asc_ratios = [
        fm.target_typo_asc / fm.upm
        for fm in cluster
        if fm.target_typo_asc is not None and fm.upm > 0
    ]
    desc_ratios = [
        fm.target_typo_desc / fm.upm
        for fm in cluster
        if fm.target_typo_desc is not None and fm.upm > 0
    ]
    if asc_ratios:
        asc_range = max(asc_ratios) - min(asc_ratios)
        if asc_range > 0.001:
            cs.StatusIndicator("warning").add_message(
                f"Cluster normalization inconsistency detected: "
                f"ascender ratio range {asc_range:.6f} (should be < 0.001)"
            ).emit(console)
    if desc_ratios:
        desc_range = max(desc_ratios) - min(desc_ratios)
        if desc_range > 0.001:
            cs.StatusIndicator("warning").add_message(
                f"Cluster normalization inconsistency detected: "
                f"descender ratio range {desc_range:.6f} (should be < 0.001)"
            ).emit(console)


def plan_adaptive_metrics(
    cluster: List[FontMeasures],
    family_norm_min: float,
    family_norm_max: float,
    family_norm_asc: float,
    config: MetricsConfig,
    verbosity: Verbosity = Verbosity.BRIEF,
) -> None:
    """Apply per-font adaptive normalization for varied fonts."""
    for fm in cluster:
        upm = fm.upm

        # Win metrics (family-wide, in normalized units)
        fm.target_win_asc = int(
            round(max(family_norm_max * upm * (1.0 + config.win_buffer), 0))
        )
        fm.target_win_desc = int(
            round(abs(family_norm_min * upm) * (1.0 + config.win_buffer))
        )

        # Typo ascender from family baseline
        typo_asc = int(round(family_norm_asc * upm))

        # Apply centering logic
        cap_ref = fm.cap_optical or fm.cap_height
        cap_h = cap_ref if cap_ref else int(round(0.7 * upm))
        desired_desc = compute_descender_for_centering(
            typo_asc, cap_h, fm.descender_min
        )

        # Compute cluster-wide target (category-aware if enabled)
        # For adaptive metrics, compute target for the single-font cluster
        cluster_target = compute_cluster_target_percent([fm], config)

        # Adjust target percent using x-height if configured (additional adjustment)
        per_font_target_percent = cluster_target
        if config.adapt_for_xheight and fm.x_height and fm.x_height > 0 and cap_h > 0:
            x_ratio = fm.x_height / float(cap_h)
            extra = config.xheight_softener * max(0.0, x_ratio - 0.66)
            per_font_target_percent = min(1.6, cluster_target + extra)

        # Adjust span to meet minimum target (no upper limit)
        span = typo_asc + abs(desired_desc)
        per_font_target_span = int(round(per_font_target_percent * upm))

        # Only expand if below minimum target (1.2x default)
        # No upper limit - let actual geometry determine final span
        if span < per_font_target_span:
            extra_amt = per_font_target_span - span
            # Expand both (60/40 split) to maintain centering
            typo_asc += int(round(extra_amt * 0.6))
            desired_desc -= int(round(extra_amt * 0.4))

        fm.target_typo_asc = typo_asc
        fm.target_typo_desc = desired_desc

        # Validate: ensure typo descender doesn't exceed actual descender (prevent clipping)
        if fm.descender_min and fm.target_typo_desc:
            if fm.target_typo_desc > fm.descender_min:
                # Clipping would occur - adjust
                fm.target_typo_desc = fm.descender_min


def analyze_family_impact(
    measures: List[FontMeasures],
) -> Tuple[float, float, int, bool]:
    """Analyze the impact of planned changes.

    Returns: (avg_typo_change_pct, avg_span_change_pct, num_fonts, has_any_changes)
    """
    typo_changes: List[float] = []
    span_changes: List[float] = []
    has_any_changes = False

    for fm in measures:
        try:
            font = _read_ttfont(fm.path)
            os2 = font.get("OS/2")
            hhea = font.get("hhea")
            if not os2:
                font.close()
                continue

            old_typo_asc = int(getattr(os2, "sTypoAscender", 0) or 0)
            old_typo_desc = int(getattr(os2, "sTypoDescender", 0) or 0)
            old_typo_gap = int(getattr(os2, "sTypoLineGap", 0) or 0)
            old_win_asc = int(getattr(os2, "usWinAscent", 0) or 0)
            old_win_desc = int(getattr(os2, "usWinDescent", 0) or 0)
            old_span = old_typo_asc + abs(old_typo_desc)

            old_hhea_asc = int(getattr(hhea, "ascent", 0) or 0) if hhea else 0
            old_hhea_desc = int(getattr(hhea, "descent", 0) or 0) if hhea else 0
            old_hhea_gap = int(getattr(hhea, "lineGap", 0) or 0) if hhea else 0

            new_typo_asc = fm.target_typo_asc or old_typo_asc
            new_typo_desc = fm.target_typo_desc or old_typo_desc
            new_win_asc = fm.target_win_asc or old_win_asc
            new_win_desc = fm.target_win_desc or old_win_desc
            new_span = new_typo_asc + abs(new_typo_desc)

            if (
                old_typo_asc != new_typo_asc
                or old_typo_desc != new_typo_desc
                or old_win_asc != new_win_asc
                or old_win_desc != new_win_desc
                or old_typo_gap != 0
                or old_hhea_gap != 0
                or old_hhea_asc != new_typo_asc
                or old_hhea_desc != new_typo_desc
            ):
                has_any_changes = True

            if old_span > 0:
                span_change_pct = ((new_span - old_span) / float(old_span)) * 100.0
                span_changes.append(span_change_pct)

            typo_change = abs(new_typo_asc - old_typo_asc) + abs(
                new_typo_desc - old_typo_desc
            )
            if fm.upm > 0:
                typo_change_pct = (typo_change / float(fm.upm)) * 100.0
                typo_changes.append(typo_change_pct)

            font.close()
        except Exception:
            continue

    avg_typo = sum(typo_changes) / len(typo_changes) if typo_changes else 0.0
    avg_span = sum(span_changes) / len(span_changes) if span_changes else 0.0

    return (avg_typo, avg_span, len(measures), has_any_changes)


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
                cs.StatusIndicator("updated")
                .add_file(fp, filename_only=False)
                .as_preview()
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


def build_plans(
    families: Dict[str, List[FontMeasures]],
    config: MetricsConfig,
    verbosity: Verbosity = Verbosity.BRIEF,
    cached_clusters: Optional[Dict[str, Dict[str, List[str]]]] = None,
    grouping_mode: str = "family",
    force_hhea: bool = False,
) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, Dict[str, List[str]]]]:
    """Build normalization plans with optical clustering.

    Returns:
        Tuple of (family_plans, clusters_cache) where:
        - family_plans: Dict mapping family name to (fam_min, fam_max, fam_asc)
        - clusters_cache: Dict mapping family name to cluster assignments
    """
    family_plans = {}
    clusters_cache: Dict[str, Dict[str, List[str]]] = {}

    for fam, group in families.items():
        # Compute UPM majority for status reporting
        upm_counts: Dict[int, int] = {}
        for fm in group:
            upm_counts[fm.upm] = upm_counts.get(fm.upm, 0) + 1
        family_upm_majority = (
            max(upm_counts.items(), key=lambda x: x[1])[0] if upm_counts else None
        )
        # Store in each font measure for status output
        for fm in group:
            fm.family_upm_majority = family_upm_majority

        # Force-hhea mode: apply averaged hhea/typo values to all fonts
        if force_hhea:
            # Filter out excluded fonts for calculations
            included_fonts = [
                fm
                for fm in group
                if not getattr(fm, "is_excluded_from_calculations", False)
            ]

            if not included_fonts:
                # All fonts excluded - fall back to individual calculations
                cs.StatusIndicator("warning").add_message(
                    f"[field]Family:[/field] '{fam}' — "
                    "All fonts excluded from calculations, using individual mode"
                ).emit(console)
                for fm in group:
                    fam_min, fam_max = compute_family_normalized_extremes([fm])
                    core_asc = compute_family_normalized_ascender([fm], config)
                    plan_adaptive_metrics(
                        [fm], fam_min, fam_max, core_asc, config, verbosity
                    )
                    finalize_metrics(fm)
                family_plans[fam] = (
                    compute_family_normalized_extremes(group)[0],
                    compute_family_normalized_extremes(group)[1],
                    compute_family_normalized_ascender(group, config),
                )
                continue

            # Compute family extremes for Win values (excluding excluded fonts)
            fam_min, fam_max = compute_family_normalized_extremes(included_fonts)

            # Read existing hhea/typo values from included fonts and average them
            # This preserves the original vertical span rather than computing new values
            norm_typo_asc_list: List[float] = []
            norm_typo_desc_list: List[float] = []

            for fm in included_fonts:
                try:
                    font = _read_ttfont(fm.path)
                    os2 = font.get("OS/2")
                    hhea = font.get("hhea")

                    # Prefer hhea values, fallback to OS/2 typo values
                    if hhea:
                        typo_asc = int(getattr(hhea, "ascent", 0) or 0)
                        typo_desc = int(getattr(hhea, "descent", 0) or 0)
                    elif os2:
                        typo_asc = int(getattr(os2, "sTypoAscender", 0) or 0)
                        typo_desc = int(getattr(os2, "sTypoDescender", 0) or 0)
                    else:
                        font.close()
                        continue

                    if typo_asc > 0 and fm.upm > 0:
                        norm_typo_asc_list.append(typo_asc / fm.upm)
                    if typo_desc != 0 and fm.upm > 0:
                        norm_typo_desc_list.append(typo_desc / fm.upm)

                    font.close()
                except Exception:
                    continue

            # Compute averages of existing normalized typo values
            if norm_typo_asc_list and norm_typo_desc_list:
                avg_norm_typo_asc = sum(norm_typo_asc_list) / len(norm_typo_asc_list)
                avg_norm_typo_desc = sum(norm_typo_desc_list) / len(norm_typo_desc_list)
                # Use averaged typo ascender as core_asc for family plan reporting
                core_asc = avg_norm_typo_asc
            else:
                # Fallback: compute from measurements if existing values unavailable
                core_asc = compute_family_normalized_ascender(included_fonts, config)
                avg_norm_typo_asc = core_asc
                # Compute average cap height ratio for descender calculation
                cap_ratios = [
                    (fm.cap_optical or fm.cap_height) / fm.upm
                    for fm in included_fonts
                    if (fm.cap_optical or fm.cap_height) and fm.upm > 0
                ]
                if cap_ratios:
                    avg_cap_ratio = sum(cap_ratios) / len(cap_ratios)
                    avg_norm_typo_desc = -(avg_norm_typo_asc - avg_cap_ratio)
                else:
                    avg_norm_typo_desc = -0.25  # Default fallback

            # Apply averaged values to ALL fonts in family (including excluded ones)
            for fm in group:
                upm = fm.upm
                # Win metrics use family extremes (excluding excluded fonts)
                fm.target_win_asc = int(
                    round(max(fam_max * upm * (1.0 + config.win_buffer), 0))
                )
                fm.target_win_desc = int(
                    round(abs(fam_min * upm) * (1.0 + config.win_buffer))
                )
                # Typo metrics use averaged normalized values (scaled to this font's UPM)
                fm.target_typo_asc = int(round(avg_norm_typo_asc * upm))
                fm.target_typo_desc = int(round(avg_norm_typo_desc * upm))

                # Ensure Win >= Typo
                finalize_metrics(fm)

            # Report safe-hhea application
            if verbosity >= Verbosity.VERBOSE:
                cs.StatusIndicator("info").add_message(
                    f"[field]Family:[/field] '{fam}' — "
                    f"[bold]Safe-hhea mode:[/bold] Applied averaged typo values to all {len(group)} font(s)"
                ).add_item(
                    f"Average normalized typo: asc={avg_norm_typo_asc:.4f}, desc={avg_norm_typo_desc:.4f}",
                    indent_level=1,
                ).add_item(
                    f"Based on {len(included_fonts)} included font(s) (excluded: {len(group) - len(included_fonts)})",
                    indent_level=1,
                ).emit(console)

            family_plans[fam] = (fam_min, fam_max, core_asc)
            continue

        # Handle grouping modes that bypass clustering
        if grouping_mode == "individual":
            # Individual mode: skip all clustering and family normalization
            for fm in group:
                fam_min, fam_max = compute_family_normalized_extremes([fm])
                core_asc = compute_family_normalized_ascender([fm], config)
                plan_adaptive_metrics(
                    [fm], fam_min, fam_max, core_asc, config, verbosity
                )
                finalize_metrics(fm)
            family_plans[fam] = (
                compute_family_normalized_extremes(group)[0],
                compute_family_normalized_extremes(group)[1],
                compute_family_normalized_ascender(group, config),
            )
            continue

        if grouping_mode == "conservative":
            # Safe-max mode: use bbox extremes for all fonts (prevents clipping)
            plan_safe_metrics(group, config)
            family_plans[fam] = (
                compute_family_normalized_extremes(group)[0],
                compute_family_normalized_extremes(group)[1],
                compute_family_normalized_ascender(group, config),
            )
            continue

        # Report unicase detection
        unicase_count = sum(1 for fm in group if fm.is_unicase)
        non_unicase_count = len(group) - unicase_count
        if unicase_count > 0:
            unicase_names = [Path(fm.path).name for fm in group if fm.is_unicase]
            if non_unicase_count > 0:
                # Mixed family: unicase will inherit baseline from traditional
                indicator = cs.StatusIndicator("info").add_message(
                    f"[field]Family:[/field] '{fam}' — "
                    f"[bold]Unicase detected:[/bold] {cs.fmt_count(unicase_count)} unicase font(s) "
                    f"mixed with {cs.fmt_count(non_unicase_count)} traditional font(s)"
                )
                if verbosity >= Verbosity.VERBOSE:
                    indicator.add_item(
                        f"Unicase fonts: {', '.join(unicase_names[:5])}{'...' if len(unicase_names) > 5 else ''}",
                        indent_level=1,
                    )
                indicator.add_item(
                    "Unicase fonts will inherit baseline from traditional fonts for alignment",
                    indent_level=1,
                ).emit(console)
            else:
                # Pure unicase family: will cluster normally
                cs.StatusIndicator("info").add_message(
                    f"[field]Family:[/field] '{fam}' — "
                    f"[bold]Pure unicase family:[/bold] {cs.fmt_count(unicase_count)} font(s) "
                    f"(x-height ≈ cap-height, clustering normally)"
                ).emit(console)

        # Level 0: Handle grouping modes that bypass clustering
        # Note: grouping_mode is passed via args, not config
        # This check will be handled in build_plans() before calling this section

        # Level 1: Compute family-wide extremes (in normalized units) - for Win metrics
        fam_min, fam_max = compute_family_normalized_extremes(group)

        # Level 2: Detect optical clusters (always enabled)
        if len(group) > 1:
            if verbosity >= Verbosity.DEBUG:
                cs.StatusIndicator("info").add_message(
                    f"[field]Family:[/field] '{fam}' — "
                    f"[dim]DEBUG:[/dim] Detecting optical clusters from {len(group)} font(s)"
                ).emit(console)
            # Check for cached clusters first
            family_clusters = None
            if cached_clusters and fam in cached_clusters:
                family_clusters = cached_clusters[fam]
                # Validate cached clusters match current font set
                cached_paths = set()
                if family_clusters.get("main_cluster"):
                    cached_paths.update(family_clusters["main_cluster"])
                if family_clusters.get("decorative"):
                    cached_paths.update(family_clusters["decorative"])
                if family_clusters.get("script"):
                    cached_paths.update(family_clusters.get("script", []))
                if family_clusters.get("unicase"):
                    cached_paths.update(family_clusters["unicase"])
                current_paths = {fm.path for fm in group}

                if cached_paths == current_paths:
                    # Reconstruct clusters from cached data
                    path_to_fm = {fm.path: fm for fm in group}
                    clusters = []
                    decorative_outliers = []
                    script_outliers = []
                    main_cluster = []

                    # Reconstruct main cluster
                    if family_clusters.get("main_cluster"):
                        main_cluster = [
                            path_to_fm[path]
                            for path in family_clusters["main_cluster"]
                            if path in path_to_fm
                        ]
                        if main_cluster:
                            clusters.append(main_cluster)

                    # Reconstruct decorative outliers
                    if family_clusters.get("decorative"):
                        decorative_outliers = [
                            path_to_fm[path]
                            for path in family_clusters["decorative"]
                            if path in path_to_fm
                        ]
                        for fm in decorative_outliers:
                            fm.is_decorative_outlier = True

                    # Reconstruct script outliers
                    if family_clusters.get("script"):
                        script_outliers = [
                            path_to_fm[path]
                            for path in family_clusters["script"]
                            if path in path_to_fm
                        ]
                        for fm in script_outliers:
                            # is_script is already set during measurement, but confirm/refine here
                            fm.is_script = True

                    # Reconstruct unicase (already marked in decorative)
                    if family_clusters.get("unicase"):
                        for path in family_clusters["unicase"]:
                            if path in path_to_fm:
                                path_to_fm[path].is_unicase = True
                else:
                    # Paths don't match - invalidate cache and recompute
                    if verbosity >= Verbosity.DEBUG:
                        cs.StatusIndicator("info").add_message(
                            f"Cluster cache invalid for '{fam}' (file set changed) - reclustering"
                        ).emit(console)
                    family_clusters = None
                    main_cluster = []
                    script_outliers = []

            if not family_clusters:
                # No valid cache - compute clusters normally
                clusters, decorative_outliers, script_outliers = (
                    detect_optical_clusters(group, config.optical_threshold, config)
                )
                main_cluster = max(clusters, key=len) if clusters else []
            else:
                # Using cached clusters - main_cluster already set during reconstruction
                pass

            # Report clustering results (main_cluster is now defined in both paths)

            # Report font type detection (script, decorative, unicase)
            # These messages indicate which detector identified which fonts

            # Script detection: 2x+ span AND descender-dominant
            if script_outliers:
                # Calculate span ratios for reporting
                span_ratios = []
                if main_cluster:
                    cluster_spans = [
                        (cfm.max_y - cfm.min_y) / cfm.upm
                        for cfm in main_cluster
                        if cfm.max_y is not None
                        and cfm.min_y is not None
                        and cfm.upm > 0
                    ]
                    avg_cluster_span = (
                        sum(cluster_spans) / len(cluster_spans) if cluster_spans else 0
                    )

                    for fm in script_outliers:
                        if fm.max_y and fm.min_y and fm.upm > 0:
                            fm_span = (fm.max_y - fm.min_y) / fm.upm
                            ratio = (
                                fm_span / avg_cluster_span
                                if avg_cluster_span > 0
                                else 0
                            )
                            span_ratios.append(ratio)

                # Build message with span ratio info
                if span_ratios and verbosity >= Verbosity.VERBOSE:
                    avg_ratio = sum(span_ratios) / len(span_ratios)
                    ratio_text = f"span ratio: {avg_ratio:.1f}x vs cluster avg"
                else:
                    ratio_text = "2x+ span, descender-dominant"

                indicator = cs.StatusIndicator("info").add_message(
                    f"[field]Family:[/field] '{fam}' — "
                    f"[bold]Script detector:[/bold] {cs.fmt_count(len(script_outliers))} font(s) "
                    f"({ratio_text})"
                )
                if verbosity >= Verbosity.VERBOSE:
                    script_names = [Path(fm.path).name for fm in script_outliers]
                    indicator.add_item(
                        f"Detected as script: {', '.join(script_names[:5])}{'...' if len(script_names) > 5 else ''}",
                        indent_level=1,
                    )
                indicator.emit(console)

            # Decorative detection: expanded bounds, core metrics match (but not script/unicase)
            other_decorative = [
                fm
                for fm in decorative_outliers
                if not fm.is_unicase and not fm.is_script
            ]
            if other_decorative:
                decorative_names = [Path(fm.path).name for fm in other_decorative]
                indicator = cs.StatusIndicator("info").add_message(
                    f"[field]Family:[/field] '{fam}' — "
                    f"[bold]Decorative detector:[/bold] {cs.fmt_count(len(other_decorative))} font(s) "
                    f"(expanded bounds, core metrics match)"
                )
                indicator.add_item(
                    f"Detected as decorative: {', '.join(decorative_names)}",
                    indent_level=1,
                )
                if verbosity >= Verbosity.VERBOSE:
                    # Show span ratios for decorative fonts
                    if main_cluster:
                        cluster_spans = [
                            (cfm.max_y - cfm.min_y) / cfm.upm
                            for cfm in main_cluster
                            if cfm.max_y is not None
                            and cfm.min_y is not None
                            and cfm.upm > 0
                        ]
                        avg_cluster_span = (
                            sum(cluster_spans) / len(cluster_spans)
                            if cluster_spans
                            else 0
                        )
                        for fm in other_decorative:
                            if fm.max_y and fm.min_y and fm.upm > 0:
                                fm_span = (fm.max_y - fm.min_y) / fm.upm
                                ratio = (
                                    fm_span / avg_cluster_span
                                    if avg_cluster_span > 0
                                    else 0
                                )
                                indicator.add_item(
                                    f"{Path(fm.path).name}: span ratio {ratio:.2f}x vs cluster avg",
                                    indent_level=2,
                                )
                indicator.emit(console)

            # Unicase detection: x-height ≈ cap-height
            unicase_in_clusters = sum(
                1 for cluster in clusters for fm in cluster if fm.is_unicase
            )
            unicase_in_decorative = sum(
                1 for fm in decorative_outliers if fm.is_unicase
            )
            if unicase_in_clusters > 0 or unicase_in_decorative > 0:
                total_unicase = unicase_in_clusters + unicase_in_decorative
                if unicase_in_decorative > 0:
                    indicator = cs.StatusIndicator("info").add_message(
                        f"[field]Family:[/field] '{fam}' — "
                        f"[bold]Unicase detector:[/bold] {cs.fmt_count(total_unicase)} font(s) "
                        f"(x-height ≈ cap-height) - separated for baseline alignment"
                    )
                    if verbosity >= Verbosity.VERBOSE:
                        unicase_names = [
                            Path(fm.path).name
                            for fm in decorative_outliers
                            if fm.is_unicase
                        ]
                        indicator.add_item(
                            f"Detected as unicase: {', '.join(unicase_names[:5])}{'...' if len(unicase_names) > 5 else ''}",
                            indent_level=1,
                        )
                    indicator.emit(console)

            # Level 3: Compute typo baseline from CORE CLUSTER only (not decorative outliers)
            if main_cluster:
                core_asc = compute_family_normalized_ascender(main_cluster, config)
            else:
                core_asc = compute_family_normalized_ascender(group, config)

            if len(main_cluster) > 1:
                # Get UPM info
                upms = {fm.upm for fm in main_cluster}
                if len(upms) == 1:
                    upm_info = f"UPM: {list(upms)[0]}"
                else:
                    # Mixed UPMs - show normalized impact
                    upm_counts: Dict[int, int] = {}
                    for fm in main_cluster:
                        upm_counts[fm.upm] = upm_counts.get(fm.upm, 0) + 1

                    upm_list = ", ".join(
                        f"{upm} ({count})" for upm, count in sorted(upm_counts.items())
                    )

                    # Calculate normalization variance
                    cap_ratios = [
                        (fm.cap_optical or fm.cap_height) / fm.upm
                        for fm in main_cluster
                        if (fm.cap_optical or fm.cap_height) and fm.upm > 0
                    ]

                    if cap_ratios:
                        mean_cap_ratio = sum(cap_ratios) / len(cap_ratios)
                        variance = sum(
                            (r - mean_cap_ratio) ** 2 for r in cap_ratios
                        ) / len(cap_ratios)
                        std_dev = variance**0.5
                        cv = (
                            (std_dev / mean_cap_ratio * 100)
                            if mean_cap_ratio != 0
                            else 0.0
                        )

                        if cv < 1.0:
                            quality = "excellent normalization"
                        elif cv < 2.5:
                            quality = "good normalization"
                        else:
                            quality = "significant variance"

                        upm_info = f"[warning]Mixed UPM: {upm_list}[/warning] ([dim]{quality}, CV={cv:.1f}%[/dim])"
                    else:
                        upm_info = f"[warning]Mixed UPM: {upm_list}[/warning]"

                cluster_msg = (
                    f"[field]Family:[/field] '{fam}' — "
                    f"Core cluster: {cs.fmt_count(len(main_cluster))} fonts"
                )
                if verbosity >= Verbosity.VERBOSE:
                    cluster_msg += (
                        f" (cap height ≈ {main_cluster[0].cap_height}/{main_cluster[0].upm} = "
                        f"{(main_cluster[0].cap_height or 0) / main_cluster[0].upm:.3f})"
                    )
                cluster_msg += f" | {upm_info}"
                cs.StatusIndicator("info").add_message(cluster_msg).emit(console)

            # Check if max_adjustment limit should override family normalization
            fonts_exceeding_limit: List[Tuple[FontMeasures, float]] = []
            if config.max_adjustment is not None and len(main_cluster) > 1:
                for fm in main_cluster:
                    solo_asc = compute_family_normalized_ascender([fm], config)
                    pull_fraction = (
                        abs((core_asc - solo_asc) / solo_asc) if solo_asc > 0 else 0
                    )

                    if pull_fraction > config.max_adjustment:
                        fonts_exceeding_limit.append(
                            (fm, pull_fraction * 100)
                        )  # Store as percent for display

                if fonts_exceeding_limit:
                    # Report and switch to per-font for affected fonts
                    for fm, pull_percent in fonts_exceeding_limit:
                        cs.StatusIndicator("info").add_message(
                            f"{Path(fm.path).name}: Adjustment {pull_percent:.1f}% exceeds --max-adjustment {config.max_adjustment * 100:.1f}% - using individual calculation"
                        ).emit(console)

                    # Split main_cluster into two groups
                    within_limit = [
                        fm
                        for fm in main_cluster
                        if fm not in [f[0] for f in fonts_exceeding_limit]
                    ]
                    exceeded_fonts = [f[0] for f in fonts_exceeding_limit]

                    # Update clusters list: replace main_cluster with split clusters
                    clusters_new = []
                    for cluster in clusters:
                        if cluster == main_cluster:
                            # Replace main_cluster with split clusters
                            if len(within_limit) > 1:
                                clusters_new.append(within_limit)
                            elif within_limit:
                                # Single font becomes individual cluster
                                clusters_new.append(within_limit)

                            # Add exceeded fonts as individual clusters
                            for fm in exceeded_fonts:
                                clusters_new.append([fm])
                        else:
                            clusters_new.append(cluster)

                    clusters = clusters_new
                    # Update main_cluster reference for decorative outlier handling
                    # If within_limit is empty (all fonts exceeded), set main_cluster to empty
                    # so decorative outliers get adaptive metrics instead
                    if within_limit:
                        main_cluster = within_limit
                    else:
                        # All fonts exceeded - no cluster for decorative outliers to inherit from
                        main_cluster = []
                    # Now fall through to normal cluster processing

            if decorative_outliers:
                # Separate unicase from other decorative variants for reporting
                unicase_outliers = [fm for fm in decorative_outliers if fm.is_unicase]
                other_decorative = [
                    fm for fm in decorative_outliers if not fm.is_unicase
                ]

                if unicase_outliers:
                    indicator = cs.StatusIndicator("info").add_message(
                        f"[field]Family:[/field] '{fam}' — "
                        f"[bold]Unicase baseline preservation:[/bold] {cs.fmt_count(len(unicase_outliers))} font(s)"
                    )
                    if verbosity >= Verbosity.VERBOSE:
                        outlier_names = [Path(fm.path).name for fm in unicase_outliers]
                        indicator.add_item(
                            f"Unicase fonts: {', '.join(outlier_names[:5])}{'...' if len(outlier_names) > 5 else ''}",
                            indent_level=1,
                        )
                    indicator.add_item(
                        "Aligned by x-height (not cap-height) to maintain baseline alignment with traditional fonts",
                        indent_level=1,
                    ).emit(console)

                if other_decorative:
                    indicator = cs.StatusIndicator("info").add_message(
                        f"[field]Family:[/field] '{fam}' — "
                        f"[bold]Decorative variants:[/bold] {cs.fmt_count(len(other_decorative))} font(s)"
                    )
                    if verbosity >= Verbosity.VERBOSE:
                        outlier_names = [Path(fm.path).name for fm in other_decorative]
                        indicator.add_item(
                            f"Decorative fonts: {', '.join(outlier_names[:5])}{'...' if len(outlier_names) > 5 else ''}",
                            indent_level=1,
                        )
                    indicator.add_item(
                        "Inherit core typo, expand win bounds",
                        indent_level=1,
                    ).emit(console)

            # Report script font handling (already reported detection above, this is for processing)
            if script_outliers and verbosity >= Verbosity.VERBOSE:
                cs.StatusIndicator("info").add_message(
                    f"[field]Family:[/field] '{fam}' — "
                    f"Script fonts will inherit core typo metrics and expand win bounds with {config.script_win_buffer_multiplier}x buffer"
                ).emit(console)

            # Level 4: Apply normalization per cluster
            # Note: Decorative outliers are NOT in clusters, so they won't be processed here
            if verbosity >= Verbosity.DEBUG:
                cs.StatusIndicator("info").add_message(
                    f"[field]Family:[/field] '{fam}' — "
                    f"[dim]DEBUG:[/dim] Level 4: Processing {len(clusters)} cluster(s)"
                ).emit(console)
            for cluster in clusters:
                # Skip decorative outliers if they somehow ended up in clusters
                cluster_fonts = [
                    fm
                    for fm in cluster
                    if not getattr(fm, "is_decorative_outlier", False)
                ]
                decorative_in_cluster = [
                    fm for fm in cluster if getattr(fm, "is_decorative_outlier", False)
                ]
                if decorative_in_cluster and verbosity >= Verbosity.DEBUG:
                    decorative_names = [
                        Path(fm.path).name for fm in decorative_in_cluster
                    ]
                    cs.StatusIndicator("warning").add_message(
                        f"[field]Family:[/field] '{fam}' — "
                        f"WARNING: Decorative fonts found in clusters (should not happen): {', '.join(decorative_names)}"
                    ).emit(console)
                if not cluster_fonts:
                    continue  # Skip clusters that only contain decorative outliers

                if len(cluster_fonts) > 1:
                    # Core cluster: identical metrics (normalized across UPMs)
                    plan_identical_metrics(
                        cluster_fonts,
                        fam_min,
                        fam_max,
                        core_asc,
                        config,
                        verbosity,
                    )
                else:
                    # True outlier: compute own ascender (don't use main cluster's)
                    outlier_asc = compute_family_normalized_ascender(
                        cluster_fonts, config
                    )
                    plan_adaptive_metrics(
                        cluster_fonts,
                        fam_min,
                        fam_max,
                        outlier_asc,
                        config,
                        verbosity,
                    )

            # Level 5: FINALIZE: Ensure Win >= Typo for ALL fonts (before decorative inheritance)
            if verbosity >= Verbosity.DEBUG:
                cs.StatusIndicator("info").add_message(
                    f"[field]Family:[/field] '{fam}' — "
                    f"[dim]DEBUG:[/dim] Level 5: Finalizing metrics for {len(group)} font(s)"
                ).emit(console)
            for fm in group:
                finalize_metrics(fm)

            # Level 7: Now get normalized typo from main cluster (after finalization)
            if decorative_outliers:
                if verbosity >= Verbosity.DEBUG:
                    cs.StatusIndicator("info").add_message(
                        f"[field]Family:[/field] '{fam}' — "
                        f"[dim]DEBUG:[/dim] Level 7: Processing {len(decorative_outliers)} decorative outlier(s)"
                    ).emit(console)
                # Use main_cluster as typo source (fallback handled in max-pull path)
                typo_source = main_cluster if main_cluster else None
                if typo_source:
                    norm_typo_asc, norm_typo_desc = get_cluster_normalized_typo(
                        typo_source
                    )
                    # Always show decorative inheritance info (not just verbose)
                    decorative_names = [
                        Path(fm.path).name for fm in decorative_outliers
                    ]
                    main_cluster_names = [Path(fm.path).name for fm in typo_source]
                    cs.StatusIndicator("info").add_message(
                        f"[field]Family:[/field] '{fam}' — "
                        f"Decorative fonts inheriting typo metrics from main cluster"
                    ).add_item(
                        f"Main cluster fonts: {', '.join(main_cluster_names)}",
                        indent_level=1,
                    ).add_item(
                        f"Decorative fonts: {', '.join(decorative_names)}",
                        indent_level=1,
                    ).add_item(
                        f"Inherited typo ascender: {int(round(norm_typo_asc * (typo_source[0].upm if typo_source else 1000)))}, "
                        f"descender: {int(round(norm_typo_desc * (typo_source[0].upm if typo_source else 1000)))}",
                        indent_level=1,
                    ).emit(console)
                    # Individual font inheritance messages shown below at VERBOSE level
                else:
                    # Fallback: compute adaptive metrics for decorative outliers
                    for fm in decorative_outliers:
                        decorative_asc = compute_family_normalized_ascender(
                            [fm], config
                        )
                        plan_adaptive_metrics(
                            [fm],
                            fam_min,
                            fam_max,
                            decorative_asc,
                            config,
                            verbosity,
                        )
                    continue  # Skip the inherited typo logic below

                for fm in decorative_outliers:
                    # SPECIAL HANDLING FOR UNICASE: Inherit traditional metrics directly
                    if fm.is_unicase and typo_source:
                        # Unicase: Just inherit traditional typo metrics directly
                        # The unicase cap (which equals its x-height) will naturally
                        # align with the traditional x-height since both use same baseline
                        old_asc = fm.target_typo_asc
                        fm.target_typo_asc = int(round(norm_typo_asc * fm.upm))
                        fm.target_typo_desc = int(round(norm_typo_desc * fm.upm))
                        if (
                            verbosity >= Verbosity.VERBOSE
                            and old_asc != fm.target_typo_asc
                        ):
                            cs.StatusIndicator("info").add_message(
                                f"{Path(fm.path).name}: Inherited ascender {old_asc} → {fm.target_typo_asc}, descender → {fm.target_typo_desc}"
                            ).emit(console)

                        # Report the alignment for clarity
                        traditional_x = [
                            cf.x_height / cf.upm
                            for cf in typo_source
                            if cf.x_height and cf.upm > 0
                        ]
                        if traditional_x and fm.cap_height:
                            avg_trad_x = sum(traditional_x) / len(traditional_x)
                            unicase_cap = fm.cap_height / fm.upm
                            alignment_diff = (
                                abs(unicase_cap - avg_trad_x) * 100
                            )  # as % of UPM

                            if (
                                alignment_diff < 3.0 and verbosity >= Verbosity.VERBOSE
                            ):  # Within 3% UPM
                                cs.StatusIndicator("success").add_message(
                                    f"{Path(fm.path).name}: Unicase cap ({fm.cap_height}) aligns with "
                                    f"traditional x-height ({int(avg_trad_x * fm.upm)}) - "
                                    f"baseline preserved"
                                ).emit(console)
                    else:
                        # Regular decorative outlier: inherit typo as-is
                        old_asc = fm.target_typo_asc
                        old_desc = fm.target_typo_desc
                        new_asc = int(round(norm_typo_asc * fm.upm))
                        fm.target_typo_asc = new_asc
                        new_desc = int(round(norm_typo_desc * fm.upm))
                        fm.target_typo_desc = new_desc
                        # Show inheritance for decorative fonts at VERBOSE level
                        if verbosity >= Verbosity.VERBOSE:
                            if old_asc != new_asc or old_desc != new_desc:
                                cs.StatusIndicator("info").add_message(
                                    f"{Path(fm.path).name}: Inherited typo metrics from main cluster"
                                ).add_item(
                                    f"Ascender: {old_asc} → {new_asc} (norm: {norm_typo_asc:.4f})",
                                    indent_level=1,
                                ).add_item(
                                    f"Descender: {old_desc} → {new_desc}",
                                    indent_level=1,
                                ).emit(console)
                            else:
                                # Show even if no change (for transparency)
                                cs.StatusIndicator("info").add_message(
                                    f"{Path(fm.path).name}: Already matches main cluster metrics (asc: {new_asc}, desc: {new_desc})"
                                ).emit(console)

                    # Expand win for actual bounds (same for all decorative)
                    if fm.max_y and fm.min_y:
                        fm.target_win_asc = int(
                            round(fm.max_y * (1.0 + config.win_buffer))
                        )
                        fm.target_win_desc = int(
                            round(abs(fm.min_y) * (1.0 + config.win_buffer))
                        )
                    else:
                        # Fallback to family extremes
                        fm.target_win_asc = int(
                            round(fam_max * fm.upm * (1.0 + config.win_buffer))
                        )
                        fm.target_win_desc = int(
                            round(abs(fam_min * fm.upm) * (1.0 + config.win_buffer))
                        )
                    # Ensure Win >= Typo after decorative planning
                    finalize_metrics(fm)

            # Level 7b: Handle script outliers (similar to decorative but with larger buffer)
            if script_outliers:
                # Use main_cluster as typo source (same as decorative)
                typo_source = main_cluster if main_cluster else None
                if typo_source:
                    norm_typo_asc, norm_typo_desc = get_cluster_normalized_typo(
                        typo_source
                    )
                else:
                    # Fallback: compute adaptive metrics for script outliers
                    for fm in script_outliers:
                        script_asc = compute_family_normalized_ascender([fm], config)
                        plan_adaptive_metrics(
                            [fm],
                            fam_min,
                            fam_max,
                            script_asc,
                            config,
                            verbosity,
                        )
                    # Skip inherited typo logic below
                    script_outliers = []

                for fm in script_outliers:
                    # Script fonts: inherit typo from main cluster
                    fm.target_typo_asc = int(round(norm_typo_asc * fm.upm))
                    fm.target_typo_desc = int(round(norm_typo_desc * fm.upm))

                    # Expand win with script buffer (larger than decorative)
                    script_buffer = (
                        config.win_buffer * config.script_win_buffer_multiplier
                    )
                    if fm.max_y and fm.min_y:
                        fm.target_win_asc = int(round(fm.max_y * (1.0 + script_buffer)))
                        fm.target_win_desc = int(
                            round(abs(fm.min_y) * (1.0 + script_buffer))
                        )
                    else:
                        # Fallback to family extremes
                        fm.target_win_asc = int(
                            round(fam_max * fm.upm * (1.0 + script_buffer))
                        )
                        fm.target_win_desc = int(
                            round(abs(fam_min * fm.upm) * (1.0 + script_buffer))
                        )
                    # Ensure Win >= Typo after script planning
                    finalize_metrics(fm)

            # Level 8: VALIDATE: Check cluster consistency
            if main_cluster and len(main_cluster) > 1:
                validate_cluster_consistency(main_cluster)
        else:
            # Single font family: compute ascender and use adaptive
            core_asc = compute_family_normalized_ascender(group, config)
            plan_adaptive_metrics(group, fam_min, fam_max, core_asc, config, verbosity)
            # Finalize single font
            for fm in group:
                finalize_metrics(fm)

        family_plans[fam] = (
            fam_min,
            fam_max,
            core_asc
            if len(group) > 1
            else compute_family_normalized_ascender(group, config),
        )

        # Store cluster info for checkpoint (if clustering was performed)
        if len(group) > 1 and grouping_mode not in ("individual", "conservative"):
            main_cluster = max(clusters, key=len) if clusters else []
            clusters_cache[fam] = {
                "main_cluster": [fm.path for fm in main_cluster]
                if main_cluster
                else [],
                "decorative": [fm.path for fm in decorative_outliers],
                "script": [fm.path for fm in script_outliers],
                "unicase": [fm.path for fm in group if fm.is_unicase],
            }

    return family_plans, clusters_cache
