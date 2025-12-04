#!/usr/bin/env python3
"""
FamilyMetricsNormalizer: Normalize vertical metrics across a font family without
changing unitsPerEm or glyph outlines.

Strategy:
- Cap height is the stable anchor for identifying optically identical fonts
- Family-wide 'Win' metrics prevent clipping using normalized extremes
- Core cluster gets identical typo/hhea metrics (normalized across UPMs)
- Decorative outliers inherit typo metrics but expand win bounds
- Line gaps set to zero across typo and hhea
- USE_TYPO_METRICS flag enabled when OS/2 version >= 4

CLI:
  python3 FamilyMetricsNormalizer.py [paths...] [-r] [-n] [-y]
                                     [--target-percent 1.2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple
import unicodedata

if TYPE_CHECKING:
    from fontTools.ttLib import TTFont

import core.core_console_styles as cs
from core.core_file_collector import collect_font_files
from core.core_font_sorter import FontSorter, FontInfo

# Get the themed console singleton
console = cs.get_console()


# --- Unicode codepoints for referenced glyphs ---
LOWER_ASCENDER_CODEPOINTS: Tuple[int, ...] = (
    0x0062,
    0x0064,
    0x0068,
    0x006B,
    0x006C,
)  # b d h k l
LOWER_DESCENDER_CODEPOINTS: Tuple[int, ...] = (
    0x0067,
    0x006A,
    0x0070,
    0x0071,
    0x0079,
)  # g j p q y
CAP_HEIGHT_GLYPHS: Tuple[int, ...] = (0x0048, 0x0049)  # 'H', 'I'
U_LOWER_X: int = 0x0078  # 'x'
UPPER_A: int = 0x0041
UPPER_Z: int = 0x005A


@dataclass
class MetricsConfig:
    """Configuration constants for metrics normalization."""

    target_percent: float = 1.3
    win_buffer: float = 0.02
    xheight_softener: float = 0.6
    adapt_for_xheight: bool = True
    optical_threshold: float = (
        0.025  # 2.5% UPM for identical detection (validated optimal)
    )
    safe_mode: bool = False  # Use conservative bbox approach (no clustering)
    headroom_ratio: float = 0.25  # Fixed headroom as percentage of UPM (25% default)
    max_pull_percent: Optional[float] = (
        None  # Maximum % a font can be pulled by family extremes
    )
    decorative_span_threshold: float = (
        1.3  # Span ratio threshold for decorative variant detection (30% larger)
    )


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


def _cap_height(font: TTFont) -> Optional[int]:
    # Prefer OS/2.sCapHeight
    try:
        os2 = font["OS/2"]
        cap = int(getattr(os2, "sCapHeight", 0) or 0)
        if cap and cap > 0:
            return cap
    except Exception:
        pass
    # Fallback measure from 'H' or 'I'
    upm = _get_upm(font)
    for cp in CAP_HEIGHT_GLYPHS:
        b = _codepoint_bounds(font, cp)
        if b:
            _, _, _, yMax = b
            return int(round(yMax))
    # Last resort heuristic
    return int(round(upm * 0.7))


def _cap_height_optical(font: TTFont) -> Optional[int]:
    """Estimate optical cap height using median yMax across A–Z."""
    ymax_values: List[int] = []
    for cp in range(UPPER_A, UPPER_Z + 1):
        b = _codepoint_bounds(font, cp)
        if not b:
            continue
        ymax_values.append(int(round(b[3])))
    if len(ymax_values) < 3:
        return _cap_height(font)
    ymax_values.sort()
    n = len(ymax_values)
    if n % 2 == 1:
        return ymax_values[n // 2]
    return int(round((ymax_values[n // 2 - 1] + ymax_values[n // 2]) / 2))


def _ascender_max(font: TTFont) -> Optional[int]:
    """Measure maximum ascender height from lowercase letters (b, d, h, k, l).

    Returns None if fewer than 2 ascenders found (unreliable measurement).
    """
    max_y = None
    count = 0
    for cp in LOWER_ASCENDER_CODEPOINTS:
        b = _codepoint_bounds(font, cp)
        if not b:
            continue
        _, _, _, yMax = b
        count += 1
        if max_y is None or yMax > max_y:
            max_y = yMax
    # Require at least 2 ascenders for reliable measurement
    if max_y is None or count < 2:
        return None
    return int(round(max_y))


def _descender_min(font: TTFont) -> Optional[int]:
    min_y = None
    for cp in LOWER_DESCENDER_CODEPOINTS:
        b = _codepoint_bounds(font, cp)
        if not b:
            continue
        _, yMin, _, _ = b
        if min_y is None or yMin < min_y:
            min_y = yMin
    if min_y is None:
        return None
    return int(round(min_y))


def _x_height(font: TTFont) -> Optional[int]:
    # Prefer OS/2.sxHeight
    try:
        os2 = font["OS/2"]
        xh = int(getattr(os2, "sxHeight", 0) or 0)
        if xh and xh > 0:
            return xh
    except Exception:
        pass
    # Fallback: measure 'x'
    b = _codepoint_bounds(font, U_LOWER_X)
    if b:
        return int(round(b[3]))
    return None


class FontMeasures:
    def __init__(self, path: str, upm: int):
        self.path = path
        self.upm = upm
        self.family_name: str = "Unknown"

        # Measured geometry
        self.min_y: Optional[int] = None
        self.max_y: Optional[int] = None
        self.cap_height: Optional[int] = None
        self.cap_optical: Optional[int] = None
        self.x_height: Optional[int] = None
        self.ascender_max: Optional[int] = None
        self.descender_min: Optional[int] = None

        # Clustering results
        self.cluster_id: Optional[int] = None
        self.is_decorative_outlier: bool = False

        # Computed targets
        self.target_typo_asc: Optional[int] = None
        self.target_typo_desc: Optional[int] = None
        self.target_win_asc: Optional[int] = None
        self.target_win_desc: Optional[int] = None

        # Family context (for status reporting)
        self.family_upm_majority: Optional[int] = None


def scan_fonts(paths: Iterable[str], recursive: bool, include_ttx: bool) -> List[str]:
    files = collect_font_files(paths, recursive)
    if include_ttx:
        return files
    return [f for f in files if Path(f).suffix.lower() != ".ttx"]


def save_measurements_checkpoint(
    measures: List[FontMeasures], checkpoint_path: Path
) -> None:
    """Save font measurements to checkpoint file (JSON format)."""
    try:
        checkpoint_data = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "measures": [],
        }

        for fm in measures:
            # Only save measurement data, not computed targets or clustering results
            measure_data = {
                "path": fm.path,
                "upm": fm.upm,
                "family_name": fm.family_name,
                "min_y": fm.min_y,
                "max_y": fm.max_y,
                "cap_height": fm.cap_height,
                "cap_optical": fm.cap_optical,
                "x_height": fm.x_height,
                "ascender_max": fm.ascender_max,
                "descender_min": fm.descender_min,
            }
            checkpoint_data["measures"].append(measure_data)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        cs.StatusIndicator("warning").add_message(
            f"Could not save checkpoint to {checkpoint_path}: {e}"
        ).emit(console)


def load_measurements_checkpoint(
    checkpoint_path: Path, expected_files: Optional[List[str]] = None
) -> Tuple[List[FontMeasures], List[str]]:
    """Load font measurements from checkpoint file.

    Returns:
        Tuple of (loaded_measures, missing_files) where missing_files are
        expected files not found in checkpoint or files that no longer exist.
    """
    if not checkpoint_path.exists():
        return ([], expected_files or [])

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)

        if checkpoint_data.get("version") != "1.0":
            cs.StatusIndicator("warning").add_message(
                f"Checkpoint version mismatch: {checkpoint_data.get('version')}"
            ).emit(console)
            return ([], expected_files or [])

        loaded_measures: List[FontMeasures] = []
        checkpoint_paths = {m["path"] for m in checkpoint_data.get("measures", [])}

        # Reconstruct FontMeasures objects
        for measure_data in checkpoint_data.get("measures", []):
            path = measure_data.get("path")
            if not path or not Path(path).exists():
                continue  # Skip files that no longer exist

            upm = measure_data.get("upm", 1000)
            fm = FontMeasures(path, upm)
            fm.family_name = measure_data.get("family_name", "Unknown")
            fm.min_y = measure_data.get("min_y")
            fm.max_y = measure_data.get("max_y")
            fm.cap_height = measure_data.get("cap_height")
            fm.cap_optical = measure_data.get("cap_optical")
            fm.x_height = measure_data.get("x_height")
            fm.ascender_max = measure_data.get("ascender_max")
            fm.descender_min = measure_data.get("descender_min")
            loaded_measures.append(fm)

        # Determine missing files
        if expected_files:
            missing_files = [
                f
                for f in expected_files
                if f not in checkpoint_paths or not Path(f).exists()
            ]
        else:
            missing_files = []

        return (loaded_measures, missing_files)

    except json.JSONDecodeError as e:
        cs.StatusIndicator("warning").add_message(
            f"Checkpoint file corrupted: {e}"
        ).emit(console)
        return ([], expected_files or [])
    except Exception as e:
        cs.StatusIndicator("warning").add_message(
            f"Error loading checkpoint: {e}"
        ).emit(console)
        return ([], expected_files or [])


def measure_fonts(
    filepaths: List[str], existing_measures: Optional[List[FontMeasures]] = None
) -> List[FontMeasures]:
    """Measure fonts and extract family names.

    Args:
        filepaths: List of font file paths to measure
        existing_measures: Optional list of already-measured fonts (from checkpoint)
            These will be skipped and merged with new measurements.

    Returns:
        Combined list of existing + newly measured fonts.
    """
    existing_by_path = {
        fm.path: fm for fm in (existing_measures or []) if Path(fm.path).exists()
    }

    # Filter out files already measured
    files_to_measure = [fp for fp in filepaths if fp not in existing_by_path]
    measures: List[FontMeasures] = list(existing_by_path.values())

    if not files_to_measure:
        return measures  # All files already in checkpoint

    with cs.create_progress_bar(console) as progress:
        task = progress.add_task("Measuring fonts...", total=len(files_to_measure))

        for fp in files_to_measure:
            progress.console.print(f"[dim]→ {Path(fp).name}[/dim]", end="\r")
            try:
                font = _read_ttfont(fp)
            except Exception as e:
                cs.StatusIndicator("error").add_file(
                    fp, filename_only=False
                ).with_explanation(f"failed to open: {e}").emit(console)
                progress.advance(task)
                continue

            try:
                upm = _get_upm(font)
                fm = FontMeasures(fp, upm)

                # Extract family name (prefer name ID 16, fallback to ID 1)
                fam = None
                try:
                    if "name" in font:
                        name_tbl = font["name"]
                        rec16 = name_tbl.getName(16, 3, 1, 0x409) or name_tbl.getName(
                            16, 1, 0, 0
                        )
                        rec1 = name_tbl.getName(1, 3, 1, 0x409) or name_tbl.getName(
                            1, 1, 0, 0
                        )
                        if rec16:
                            fam = str(rec16.toUnicode())
                        elif rec1:
                            fam = str(rec1.toUnicode())
                except Exception:
                    pass
                fm.family_name = (
                    unicodedata.normalize("NFC", fam)
                    if isinstance(fam, str)
                    else "Unknown"
                )

                # Measure bounds and metrics
                overall = _font_overall_bounds(font)
                if overall:
                    fm.min_y = int(round(overall[0]))
                    fm.max_y = int(round(overall[1]))

                fm.cap_height = _cap_height(font)
                fm.cap_optical = _cap_height_optical(font)
                fm.ascender_max = _ascender_max(font)
                fm.descender_min = _descender_min(font)
                fm.x_height = _x_height(font)

                measures.append(fm)
            except Exception as e:
                cs.StatusIndicator("error").add_file(
                    fp, filename_only=False
                ).with_explanation(f"measure error: {e}").emit(console)
            finally:
                try:
                    font.close()
                except Exception:
                    pass
                progress.advance(task)

    return measures


def compute_optical_similarity(
    fm1: FontMeasures, fm2: FontMeasures, threshold: float, config: MetricsConfig
) -> bool:
    """Compare cap height and baseline alignment - the stable anchors.

    Cap height is the primary metric. X-height can vary (optical sizes).
    Descender is checked with more leniency.

    Always uses optical cap height measurement for clustering consistency.
    """

    def normalize(value: Optional[int], upm: int) -> Optional[float]:
        if value is None or upm <= 0:
            return None
        return float(value) / float(upm)

    # Pre-check: Overall bounds span (detect structurally different fonts)
    # Script fonts or highly decorative fonts may have same cap height but 2-3x span
    if (
        fm1.max_y is not None
        and fm1.min_y is not None
        and fm2.max_y is not None
        and fm2.min_y is not None
    ):
        span1 = normalize(fm1.max_y - fm1.min_y, fm1.upm)
        span2 = normalize(fm2.max_y - fm2.min_y, fm2.upm)

        if span1 and span2 and span1 > 0 and span2 > 0:
            span_ratio = max(span1, span2) / min(span1, span2)
            # Use configurable threshold
            if span_ratio > config.decorative_span_threshold:
                return False

    # Primary: Cap height ratio (the stable anchor) - always use optical
    if fm1.cap_optical is None or fm2.cap_optical is None:
        return False  # Require optical measurement for clustering
    cap1 = normalize(fm1.cap_optical, fm1.upm)
    cap2 = normalize(fm2.cap_optical, fm2.upm)

    if cap1 is None or cap2 is None:
        return False

    cap_diff = abs(cap1 - cap2)
    if cap_diff > threshold:
        return False

    # Secondary: x-height (more lenient - allows optical size variants)
    xh1 = normalize(fm1.x_height, fm1.upm)
    xh2 = normalize(fm2.x_height, fm2.upm)

    if xh1 is not None and xh2 is not None:
        xh_diff = abs(xh1 - xh2)
        if xh_diff > threshold * 2.0:  # 2x lenient
            return False

    # Tertiary: Descender ratio (somewhat lenient)
    desc1 = normalize(fm1.descender_min, fm1.upm)
    desc2 = normalize(fm2.descender_min, fm2.upm)

    if desc1 is not None and desc2 is not None:
        desc_diff = abs(abs(desc1) - abs(desc2))
        if desc_diff > threshold * 1.5:  # 1.5x lenient
            return False

    return True


def compute_optical_variance(measures: List[FontMeasures]) -> Dict[str, float]:
    """Calculate coefficient of variation for key metrics across a group.

    Returns dict with CV (coefficient of variation) for:
    - cap_height_ratio: cap_height/UPM
    - x_height_ratio: x_height/UPM
    - descender_ratio: descender_min/UPM

    Low CV (< 0.05 or 5%) indicates fonts are optically identical.
    High CV (> 0.15 or 15%) indicates significant variation.
    """
    if len(measures) < 2:
        return {"cap_height_cv": 0.0, "x_height_cv": 0.0, "descender_cv": 0.0}

    cap_ratios: List[float] = []
    x_height_ratios: List[float] = []
    descender_ratios: List[float] = []

    for fm in measures:
        if fm.cap_height is not None and fm.upm > 0:
            cap_ratios.append(float(fm.cap_height) / float(fm.upm))
        if fm.x_height is not None and fm.upm > 0:
            x_height_ratios.append(float(fm.x_height) / float(fm.upm))
        if fm.descender_min is not None and fm.upm > 0:
            descender_ratios.append(abs(float(fm.descender_min)) / float(fm.upm))

    def compute_cv(values: List[float]) -> float:
        if not values or len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std_dev = variance**0.5
        return (std_dev / mean) if mean != 0 else 0.0

    return {
        "cap_height_cv": compute_cv(cap_ratios) if cap_ratios else 0.0,
        "x_height_cv": compute_cv(x_height_ratios) if x_height_ratios else 0.0,
        "descender_cv": compute_cv(descender_ratios) if descender_ratios else 0.0,
    }


def detect_decorative_outlier(
    fm: FontMeasures,
    core_cluster: List[FontMeasures],
    threshold: float,
    config: MetricsConfig,
) -> bool:
    """Detect if font is decorative variant (shadow/rough/outline).

    Core metrics match cluster, but bounds are slightly larger.
    Script fonts or structurally different fonts are NOT decorative variants.
    """
    if not core_cluster:
        return False

    # Pre-check: Reject structurally different fonts (script, extreme display)
    # Compare span against cluster average
    if fm.max_y is not None and fm.min_y is not None and fm.upm > 0:
        fm_span = (fm.max_y - fm.min_y) / fm.upm
        cluster_spans = [
            (cfm.max_y - cfm.min_y) / cfm.upm
            for cfm in core_cluster
            if cfm.max_y is not None and cfm.min_y is not None and cfm.upm > 0
        ]

        if cluster_spans:
            avg_cluster_span = sum(cluster_spans) / len(cluster_spans)
            if avg_cluster_span > 0:
                span_ratio = max(fm_span, avg_cluster_span) / min(
                    fm_span, avg_cluster_span
                )
                # Use configurable threshold
                if span_ratio > config.decorative_span_threshold:
                    return False

    # Check if core metrics match any font in cluster
    matches_core = any(
        compute_optical_similarity(fm, core_fm, threshold, config)
        for core_fm in core_cluster
    )

    if not matches_core:
        return False  # Different structure, not a decorative variant

    # Compare bounds to cluster average (in normalized units)
    cluster_max_avg = sum(
        cfm.max_y / cfm.upm for cfm in core_cluster if cfm.max_y
    ) / len(core_cluster)
    cluster_min_avg = sum(
        cfm.min_y / cfm.upm for cfm in core_cluster if cfm.min_y
    ) / len(core_cluster)

    fm_max_norm = fm.max_y / fm.upm if fm.max_y and fm.upm > 0 else 0
    fm_min_norm = fm.min_y / fm.upm if fm.min_y and fm.upm > 0 else 0

    # If bounds are >15% larger than cluster average, it's decorative
    max_inflation = (
        (fm_max_norm - cluster_max_avg) / cluster_max_avg if cluster_max_avg else 0
    )
    min_inflation = (
        abs((fm_min_norm - cluster_min_avg) / cluster_min_avg) if cluster_min_avg else 0
    )

    return max_inflation > 0.15 or min_inflation > 0.15


def detect_optical_clusters(
    measures: List[FontMeasures], threshold: float, config: MetricsConfig
) -> Tuple[List[List[FontMeasures]], List[FontMeasures]]:
    """Detect core clusters and decorative outliers based on cap height.

    Returns: (core_clusters, decorative_outliers)
    """
    if len(measures) <= 1:
        return ([measures] if measures else [], [])

    # Build similarity graph based on cap height + x-height + descender
    n = len(measures)
    similar_pairs: List[Tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if compute_optical_similarity(measures[i], measures[j], threshold, config):
                similar_pairs.append((i, j))

    # Union-Find clustering
    parent = list(range(n))

    def find(x: int) -> int:
        # Iterative find with path compression to avoid recursion depth issues
        root = x
        # Find the root by traversing up the parent chain
        while parent[root] != root:
            root = parent[root]
        # Path compression: update all nodes along the path to point directly to root
        while parent[x] != root:
            next_node = parent[x]
            parent[x] = root
            x = next_node
        return root

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j in similar_pairs:
        union(i, j)

    # Group by cluster
    clusters_dict: Dict[int, List[FontMeasures]] = {}
    for idx, fm in enumerate(measures):
        root = find(idx)
        clusters_dict.setdefault(root, []).append(fm)
        fm.cluster_id = root

    core_clusters = [c for c in clusters_dict.values() if len(c) > 1]
    singletons = [c[0] for c in clusters_dict.values() if len(c) == 1]

    # Find main cluster (largest)
    if not core_clusters:
        return ([], singletons)

    main_cluster = max(core_clusters, key=len)

    # Check singletons: decorative variants or true outliers?
    decorative_outliers = []
    true_outliers = []

    for fm in singletons:
        if detect_decorative_outlier(fm, main_cluster, threshold, config):
            decorative_outliers.append(fm)
            fm.is_decorative_outlier = True
        else:
            true_outliers.append(fm)

    # True outliers become single-font clusters
    all_clusters = core_clusters + [[fm] for fm in true_outliers]

    return (all_clusters, decorative_outliers)


def compute_family_normalized_extremes(
    measures: List[FontMeasures],
) -> Tuple[float, float]:
    """Compute family-wide extremes in normalized units (handles mixed UPMs)."""
    norm_mins: List[float] = []
    norm_maxs: List[float] = []

    for fm in measures:
        if fm.min_y is None or fm.max_y is None or fm.upm <= 0:
            continue
        norm_mins.append(fm.min_y / fm.upm)
        norm_maxs.append(fm.max_y / fm.upm)

    if not norm_mins or not norm_maxs:
        return (-1.0, 1.0)  # Conservative fallback

    return (min(norm_mins), max(norm_maxs))


def compute_family_normalized_ascender(
    measures: List[FontMeasures], config: MetricsConfig
) -> float:
    """Compute family-wide ascender target based on cap height + fixed headroom.

    Strategy:
    1. Start with max cap height + configured headroom (e.g., 25% UPM)
    2. If any font's actual ascenders exceed this baseline significantly,
       use the tallest ascender instead (preserves actual geometry)
    """
    cap_ratios: List[float] = []
    ascender_ratios: List[float] = []

    for fm in measures:
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

    # Baseline: cap + configured headroom
    baseline_ascender = max_cap_ratio + config.headroom_ratio

    # Check if actual ascenders exceed baseline
    if ascender_ratios:
        max_ascender_ratio = max(ascender_ratios)

        # Define "significant excess" as more than half the planned headroom
        # or minimum 2% UPM (whichever is larger)
        significant_threshold = max(config.headroom_ratio / 2.0, 0.02)

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

    # Clamp to deepest actual descender
    if norm_descenders:
        norm_descender_limit = max(norm_descenders)
        if abs(norm_desired_desc) < norm_descender_limit:
            norm_desired_desc = -norm_descender_limit

    # Apply target percent adjustment in normalized space (minimum only)
    norm_span = norm_typo_asc + abs(norm_desired_desc)
    target_span_ratio = config.target_percent

    # Only expand if below minimum target (1.2x default)
    # No upper limit - let actual geometry determine final span
    if norm_span < target_span_ratio:
        extra_amt = target_span_ratio - norm_span
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

        # Descender: center capitals visually
        cap_ref = fm.cap_optical or fm.cap_height
        cap_h = cap_ref if cap_ref else int(round(0.7 * upm))
        desired_desc = -(typo_asc - cap_h)

        # Clamp to actual descender (preserve actual geometry)
        actual_descender = fm.descender_min if fm.descender_min else desired_desc
        if actual_descender < desired_desc:  # Actual descender is deeper
            desired_desc = actual_descender

        # Adjust target percent using x-height if configured
        per_font_target_percent = config.target_percent
        if config.adapt_for_xheight and fm.x_height and fm.x_height > 0 and cap_h > 0:
            x_ratio = fm.x_height / float(cap_h)
            extra = config.xheight_softener * max(0.0, x_ratio - 0.66)
            per_font_target_percent = min(1.6, config.target_percent + extra)

        # Adjust span to meet minimum target (no upper limit)
        span = typo_asc + abs(desired_desc)
        per_font_target_span = int(round(per_font_target_percent * upm))

        # Only expand if below minimum target (1.2x default)
        # No upper limit - let actual geometry determine final span
        if span < per_font_target_span:
            extra_amt = per_font_target_span - span
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
                if fm.x_height and fm.x_height > 0:
                    if getattr(os2, "sxHeight", 0) in (0, None):
                        os2.sxHeight = int(fm.x_height)
                if fm.cap_height and fm.cap_height > 0:
                    if getattr(os2, "sCapHeight", 0) in (0, None):
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


# ===============================
# CLI orchestration
# ===============================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize vertical metrics using cap height as stable anchor",
        epilog="Supported formats: TTF, OTF, WOFF, WOFF2",
    )
    parser.add_argument("paths", nargs="*", help="Font files or directories")
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Recurse into directories"
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="Preview without writing"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--target-percent",
        type=float,
        default=1.3,
        help="Target span as fraction of UPM (default: 1.3)",
    )
    parser.add_argument("--use-ttx", action="store_true", help="Include .ttx files")
    parser.add_argument(
        "--superfamily", action="store_true", help="Auto-group by common prefix"
    )
    parser.add_argument(
        "--ignore-term",
        "-it",
        action="append",
        help="Ignore token when grouping superfamilies (comma-separated or repeat flag)",
    )
    parser.add_argument(
        "--exclude-family",
        "-ef",
        action="append",
        help="Exclude families from superfamily grouping (comma-separated or repeat flag)",
    )
    parser.add_argument(
        "--group",
        "-g",
        action="append",
        help='Force merge families (comma-separated): --group "Font A,Font B"',
    )
    parser.add_argument(
        "-sm",
        "--safe-mode",
        action="store_true",
        help="Use conservative bbox approach (no clustering)",
    )
    parser.add_argument(
        "--headroom-ratio",
        type=float,
        default=0.25,
        help="Headroom as percentage of UPM above cap (default: 0.25 = 25%%)",
    )
    parser.add_argument(
        "--per-font",
        action="store_true",
        help="Process each font individually, ignoring family grouping",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show detailed analysis of family vs per-font calculations (implies --dry-run)",
    )
    parser.add_argument(
        "--max-pull",
        type=float,
        default=None,
        metavar="PERCENT",
        help="Maximum %% a font can be pulled UP or DOWN by family extremes (e.g., 8.0 for 8%%). "
        "Fonts exceeding this get per-font calculation. Default: unlimited",
    )
    parser.add_argument(
        "--decorative-threshold",
        type=float,
        default=1.3,
        metavar="RATIO",
        help="Span ratio threshold for decorative variant detection (default: 1.3 = 30%% larger). "
        "Fonts with span exceeding this ratio vs core cluster are treated as decorative variants.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_percent <= 0.5 or args.target_percent > 2.0:
        cs.StatusIndicator("warning").add_message(
            f"target-percent {args.target_percent} unusual; proceeding anyway"
        ).emit(console)

    # Validate headroom-ratio
    if args.headroom_ratio < 0.10:
        cs.StatusIndicator("warning").add_message(
            f"headroom-ratio {args.headroom_ratio} very low (<10% UPM) - may clip tall glyphs"
        ).emit(console)
    elif args.headroom_ratio > 0.40:
        cs.StatusIndicator("warning").add_message(
            f"headroom-ratio {args.headroom_ratio} very high (>40% UPM) - may waste vertical space"
        ).emit(console)

    # Warn about combined expansion
    if args.target_percent > 1.5 and args.headroom_ratio > 0.30:
        estimated_span = args.target_percent + args.headroom_ratio
        cs.StatusIndicator("warning").add_message(
            f"Combined target-percent ({args.target_percent}) + headroom-ratio ({args.headroom_ratio}) "
            f"≈ {estimated_span:.1f}x UPM - extremely loose spacing"
        ).emit(console)

    if (args.ignore_term or args.exclude_family) and not args.superfamily:
        cs.StatusIndicator("warning").add_message(
            "--ignore-term and --exclude-family only work with --superfamily"
        ).emit(console)

    if args.per_font:
        if args.superfamily:
            cs.StatusIndicator("warning").add_message(
                "--per-font ignores --superfamily (processing fonts individually)"
            ).emit(console)
        if args.group:
            cs.StatusIndicator("warning").add_message(
                "--per-font ignores --group (processing fonts individually)"
            ).emit(console)
        if args.ignore_term:
            cs.StatusIndicator("warning").add_message(
                "--per-font ignores --ignore-term (no family grouping)"
            ).emit(console)
        if args.exclude_family:
            cs.StatusIndicator("warning").add_message(
                "--per-font ignores --exclude-family (no family grouping)"
            ).emit(console)


def collect_groups(args) -> List[List[str]]:
    forced_groups = []
    if args.group:
        for group_str in args.group:
            families = [name.strip() for name in group_str.split(",")]
            if len(families) < 2:
                cs.StatusIndicator("warning").add_message(
                    f"--group requires at least 2 families, skipping: {group_str}"
                ).emit(console)
                continue
            forced_groups.append(families)
    return forced_groups


def expand_comma_separated_args(arg_list: Optional[List[str]]) -> List[str]:
    """Expand comma-separated values in argument list.

    Supports both formats:
      --flag "a,b,c"  ->  ['a', 'b', 'c']
      --flag a --flag b --flag c  ->  ['a', 'b', 'c']
      --flag "a,b" --flag c  ->  ['a', 'b', 'c']
    """
    if not arg_list:
        return []

    expanded = []
    for item in arg_list:
        # Split by comma and strip whitespace
        expanded.extend(term.strip() for term in item.split(",") if term.strip())
    return expanded


def group_families(args, measures, forced_groups):
    # Per-font mode: process each font individually
    if args.per_font:
        cs.StatusIndicator("info").add_message(
            f"Processing {cs.fmt_count(len(measures))} font(s) individually (no family grouping)"
        ).emit(console)
        # Create individual groups using filename (without extension) as group name
        return {Path(fm.path).stem: [fm] for fm in measures}

    font_infos = [FontInfo(path=fm.path, family_name=fm.family_name) for fm in measures]
    sorter = FontSorter(font_infos)

    if args.superfamily:
        groups = sorter.group_by_superfamily(
            ignore_terms=set(expand_comma_separated_args(args.ignore_term)),
            exclude_families=expand_comma_separated_args(args.exclude_family),
            forced_groups=forced_groups,
        )
        cs.StatusIndicator("info").add_message(
            f"Found {cs.fmt_count(len(groups))} superfamily group(s)"
        ).emit(console)
        sorter.get_superfamily_summary(groups)
        if forced_groups:
            cs.emit("", console=console)
            forced_info = sorter.get_forced_groups_info(forced_groups, "superfamily")
            for info in forced_info:
                cs.StatusIndicator("info").add_message(
                    f"Forced superfamily merge: [field]{info['group_name']}[/field] "
                    f"← {', '.join(info['merged_families'])}"
                ).emit(console)
    else:
        groups = sorter.group_by_family(forced_groups=forced_groups)
        cs.StatusIndicator("info").add_message(
            f"Found {cs.fmt_count(len(groups))} family group(s)"
        ).emit(console)
        if forced_groups:
            cs.emit("", console=console)
            forced_info = sorter.get_forced_groups_info(forced_groups, "family")
            for info in forced_info:
                cs.StatusIndicator("info").add_message(
                    f"Forced family merge: [field]{info['group_name']}[/field] "
                    f"← {', '.join(info['merged_families'])}"
                ).emit(console)

    path_to_measure = {fm.path: fm for fm in measures}
    return {
        group_name: [path_to_measure[fi.path] for fi in infos]
        for group_name, infos in groups.items()
    }


def build_plans(
    families: Dict[str, List[FontMeasures]], config: MetricsConfig
) -> Dict[str, Tuple[float, float, float]]:
    """Build normalization plans with optical clustering."""
    family_plans = {}

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

        # Level 0: Safe mode - skip all clustering logic
        if config.safe_mode:
            plan_safe_metrics(group, config)
            family_plans[fam] = (
                compute_family_normalized_extremes(group)[0],
                compute_family_normalized_extremes(group)[1],
                compute_family_normalized_ascender(group, config),
            )
            continue

        # Level 1: Compute family-wide extremes (in normalized units) - for Win metrics
        fam_min, fam_max = compute_family_normalized_extremes(group)

        # Level 2: Detect optical clusters (always enabled)
        if len(group) > 1:
            clusters, decorative_outliers = detect_optical_clusters(
                group, config.optical_threshold, config
            )

            # Report clustering results
            main_cluster = max(clusters, key=len) if clusters else []

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

                cs.StatusIndicator("info").add_message(
                    f"[field]Family:[/field] '{fam}' — "
                    f"Core cluster: {cs.fmt_count(len(main_cluster))} fonts "
                    f"(cap height ≈ {main_cluster[0].cap_height}/{main_cluster[0].upm} = "
                    f"{(main_cluster[0].cap_height or 0) / main_cluster[0].upm:.3f}) | {upm_info}"
                ).emit(console)

            # Check if max_pull limit should override family normalization
            fonts_exceeding_limit: List[Tuple[FontMeasures, float]] = []
            if config.max_pull_percent is not None and len(main_cluster) > 1:
                for fm in main_cluster:
                    solo_asc = compute_family_normalized_ascender([fm], config)
                    pull_percent = (
                        abs((core_asc - solo_asc) / solo_asc * 100)
                        if solo_asc > 0
                        else 0
                    )

                    if pull_percent > config.max_pull_percent:
                        fonts_exceeding_limit.append((fm, pull_percent))

                if fonts_exceeding_limit:
                    # Report and switch to per-font for affected fonts
                    for fm, pull_percent in fonts_exceeding_limit:
                        cs.StatusIndicator("info").add_message(
                            f"{Path(fm.path).name}: Pull {pull_percent:.1f}% exceeds --max-pull {config.max_pull_percent:.1f}% - using per-font calculation"
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
                outlier_names = [Path(fm.path).name for fm in decorative_outliers]
                cs.StatusIndicator("info").add_message(
                    f"[field]Family:[/field] '{fam}' — "
                    f"Decorative variants: {', '.join(outlier_names)} "
                    f"(inherit core typo, expand win bounds)"
                ).emit(console)

            # Level 4: Apply normalization per cluster
            for cluster in clusters:
                if len(cluster) > 1:
                    # Core cluster: identical metrics (normalized across UPMs)
                    plan_identical_metrics(cluster, fam_min, fam_max, core_asc, config)
                else:
                    # True outlier: compute own ascender (don't use main cluster's)
                    outlier_asc = compute_family_normalized_ascender(cluster, config)
                    plan_adaptive_metrics(
                        cluster, fam_min, fam_max, outlier_asc, config
                    )

            # Level 5: FINALIZE: Ensure Win >= Typo for ALL fonts (before decorative inheritance)
            for fm in group:
                finalize_metrics(fm)

            # Level 7: Now get normalized typo from main cluster (after finalization)
            if decorative_outliers:
                # Use main_cluster as typo source (fallback handled in max-pull path)
                typo_source = main_cluster if main_cluster else None
                if typo_source:
                    norm_typo_asc, norm_typo_desc = get_cluster_normalized_typo(
                        typo_source
                    )
                else:
                    # Fallback: compute adaptive metrics for decorative outliers
                    for fm in decorative_outliers:
                        decorative_asc = compute_family_normalized_ascender(
                            [fm], config
                        )
                        plan_adaptive_metrics(
                            [fm], fam_min, fam_max, decorative_asc, config
                        )
                    continue  # Skip the inherited typo logic below

                for fm in decorative_outliers:
                    # Inherit normalized typo, scale to this font's UPM
                    fm.target_typo_asc = int(round(norm_typo_asc * fm.upm))
                    fm.target_typo_desc = int(round(norm_typo_desc * fm.upm))
                    # Expand win for actual bounds
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

            # Level 8: VALIDATE: Check cluster consistency
            if main_cluster and len(main_cluster) > 1:
                validate_cluster_consistency(main_cluster)
        else:
            # Single font family: compute ascender and use adaptive
            core_asc = compute_family_normalized_ascender(group, config)
            plan_adaptive_metrics(group, fam_min, fam_max, core_asc, config)
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

    return family_plans


def report_changes(families, plans, args, forced_groups) -> bool:
    """Report planned changes per family."""
    any_changes_needed = False

    def get_impact_category(fam):
        group = families[fam]
        avg_typo, avg_span, num_fonts, has_changes = analyze_family_impact(group)
        if not has_changes or avg_typo < 0.1:
            return (0, fam)
        elif avg_typo < 2.0:
            return (1, fam)
        elif avg_typo < 8.0:
            return (2, fam)
        else:
            return (3, fam)

    sorted_families = sorted(plans.items(), key=lambda x: get_impact_category(x[0]))

    for fam, (fam_min, fam_max, fam_asc) in sorted_families:
        group = families[fam]
        avg_typo, avg_span, num_fonts, has_changes = analyze_family_impact(group)

        family_label = f"[bold]{fam}[/bold]"
        unique_names = set(fm.family_name for fm in group)
        if len(unique_names) > 1:
            if args.superfamily:
                family_label += " [darktext.dim](superfamily)[/darktext.dim]"
            elif forced_groups and any(fam in fg for fg in forced_groups):
                family_label += " [darktext.dim](forced group)[/darktext.dim]"

        # Adjust label for per-font mode
        field_label = "Font" if args.per_font else "Family"
        font_count_text = (
            ""
            if args.per_font
            else f"[darktext.dim]({cs.fmt_count(num_fonts)} fonts)[/darktext.dim]"
        )

        if not has_changes or avg_typo < 0.1:
            cs.StatusIndicator("info").add_message(
                f"[field]{field_label}:[/field] {family_label} "
                f"{font_count_text} — No changes needed"
            ).emit(console)
            cs.StatusIndicator("unchanged").add_message("Already normalized").emit(
                console
            )
            continue

        any_changes_needed = True

        if avg_typo < 2.0:
            impact_type, impact_desc = "minimal", "Minimal Normalization"
        elif avg_typo < 8.0:
            impact_type, impact_desc = "moderate", "Moderate Normalization"
        else:
            impact_type, impact_desc = "major", "Major Normalization"

        if abs(avg_span) < 1.0:
            span_info = "preserving vertical height"
        elif avg_span > 0:
            span_info = f"increasing vertical height by [count]{avg_span:.1f}[/count]%"
        else:
            span_info = f"reducing vertical height by [count]{avg_span:.1f}[/count]%"

        cs.StatusIndicator("info").add_message(
            f"[field]{field_label}:[/field] {family_label} "
            f"{font_count_text} — {impact_desc}"
        ).emit(console)

        cs.StatusIndicator(impact_type).add_message(
            f"UPM scaled by ~[count]{avg_typo:.1f}[/count]%, {span_info} [info]|[/info] linegap set to {cs.fmt_count(0)}"
        ).emit(console)

    return any_changes_needed


def generate_family_report(
    families: Dict[str, List[FontMeasures]],
    config: MetricsConfig,
    args: argparse.Namespace,
) -> None:
    """Generate detailed report comparing family vs per-font calculations.

    Example output:
        Font Sans (3 fonts)

          Family ascender: 0.8200 (normalized)
          Driven by: Sans-Black.ttf (asc: 820)

          ↑ Sans-Regular.ttf    + 70 units (+8.5%) - pulled up by family
          ↑ Sans-Bold.ttf       + 50 units (+6.5%) - pulled up by family
          ✓ Sans-Black.ttf      ±  0 units (+0.0%) - minimal impact

          ⚠️  High normalization impact detected (8.5% max pull)
              Consider using --per-font for heavily affected fonts
              Or use --max-pull 6.8 to limit pulling
    """

    cs.emit("")
    cs.StatusIndicator("info").add_message("Family Normalization Impact Report").emit(
        console
    )
    cs.emit("")

    for fam_name, group in families.items():
        if len(group) == 1:
            continue  # Skip single-font families

        cs.emit(
            f"[bold]{fam_name}[/bold] ({cs.fmt_count(len(group))} fonts)",
            console=console,
        )
        cs.emit("")

        # Calculate family-wide metrics
        fam_min, fam_max = compute_family_normalized_extremes(group)
        family_asc = compute_family_normalized_ascender(group, config)

        # Track pulling effects
        pulls: List[Tuple[FontMeasures, float, float, int]] = []

        for fm in group:
            # Calculate what this font would get individually
            solo_asc = compute_family_normalized_ascender([fm], config)

            # Calculate normalized difference
            family_value = int(family_asc * fm.upm)
            solo_value = int(solo_asc * fm.upm)
            diff_units = family_value - solo_value
            diff_percent = (
                ((family_asc - solo_asc) / solo_asc * 100) if solo_asc > 0 else 0
            )

            pulls.append((fm, diff_percent, solo_asc, diff_units))

        # Sort by impact (most pulled first)
        pulls.sort(key=lambda x: abs(x[1]), reverse=True)

        # Find the "driver" (font with tallest ascenders)
        driver = max(
            group,
            key=lambda fm: (fm.ascender_max or 0) / fm.upm if fm.upm > 0 else 0,
        )

        cs.emit(
            f"  [dim]Family ascender:[/dim] {family_asc:.4f} (normalized)",
            console=console,
        )
        cs.emit(
            f"  [dim]Driven by:[/dim] {Path(driver.path).name} (asc: {driver.ascender_max})",
            console=console,
        )
        cs.emit("")

        # Report per-font impact
        for fm, diff_pct, solo_asc, diff_units in pulls:
            filename = Path(fm.path).name

            if abs(diff_pct) < 1.0:
                # Minimal change
                indicator = "✓"
                style = "dim"
                msg = f"±{abs(diff_units):>4} units ({diff_pct:+.1f}%) - minimal impact"
            elif diff_units > 0:
                # Pulled up
                indicator = "↑"
                style = "warning"
                msg = f"+{diff_units:>4} units ({diff_pct:+.1f}%) - pulled up by family"
            else:
                # Pulled down (rare)
                indicator = "↓"
                style = "info"
                msg = f"{diff_units:>4} units ({diff_pct:.1f}%) - pulled down by family"

            cs.emit(
                f"  [{style}]{indicator}[/{style}] {filename:40s} {msg}",
                console=console,
            )

        cs.emit("")

        # Recommendation
        max_pull = max(abs(p[1]) for p in pulls)
        if max_pull > 8.0:
            cs.StatusIndicator("warning").add_message(
                f"⚠️  High normalization impact detected ({max_pull:.1f}% max pull)"
            ).add_item(
                "Consider using --per-font for heavily affected fonts", indent_level=1
            ).add_item(
                f"Or use --max-pull {max_pull * 0.8:.1f} to limit pulling",
                indent_level=1,
            ).emit(console)

        cs.emit("")

    # Summary: report single-font families that were skipped
    single_font_families = [fam for fam, group in families.items() if len(group) == 1]
    if single_font_families:
        cs.StatusIndicator("info").add_message(
            f"{cs.fmt_count(len(single_font_families))} single-font families skipped (no pulling analysis needed)"
        ).emit(console)


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

    label = "Preview" if dry_run else "Processing Completed!"
    cs.emit("")
    cs.StatusIndicator("success").add_message(label).with_summary_block(
        updated=updated, unchanged=unchanged, errors=errors
    ).emit(console)

    return updated, unchanged, errors


def confirm_or_exit(count: int) -> None:
    """Prompt for confirmation, looping until clear yes/no response."""
    while True:
        try:
            cs.emit("", console=console)
            response = (
                cs.prompt_input(
                    f"About to modify {cs.fmt_count(count)} file(s). Proceed? [y/N]: "
                )
                .strip()
                .lower()
            )

            if response in ["y", "yes"]:
                cs.emit("", console=console)
                return  # Proceed
            elif response in ["n", "no"]:
                cs.StatusIndicator("error").add_message("Aborted by user").emit(console)
                sys.exit(3)
            elif response == "":
                # Empty string - treat as mistake, re-prompt
                cs.StatusIndicator("warning").add_message(
                    "Please enter 'y' for yes or 'n' for no"
                ).emit(console)
                continue
            else:
                # Invalid input - re-prompt
                cs.StatusIndicator("warning").add_message(
                    f"Invalid input '{response}'. Please enter 'y' for yes or 'n' for no"
                ).emit(console)
                continue
        except (EOFError, KeyboardInterrupt):
            cs.StatusIndicator("error").add_message("Aborted by user").emit(console)
            sys.exit(3)


def main() -> None:
    start_time = time.time()
    args = parse_args()
    validate_args(args)

    # --report implies --dry-run
    if args.report:
        args.dry_run = True

    config = MetricsConfig(
        target_percent=args.target_percent if args.target_percent > 0 else 1.3,
        win_buffer=0.02,
        xheight_softener=0.6,
        adapt_for_xheight=True,
        optical_threshold=0.025,  # Validated optimal threshold (2.5% UPM)
        safe_mode=args.safe_mode,
        headroom_ratio=args.headroom_ratio,
        max_pull_percent=args.max_pull,
        decorative_span_threshold=args.decorative_threshold,
    )

    files = scan_fonts(args.paths or ["."], args.recursive, args.use_ttx)
    if not files:
        cs.StatusIndicator("error").add_message("No font files found").emit(console)
        sys.exit(1)

    # Determine checkpoint path
    checkpoint_path = Path(".metrics_checkpoint.json")

    # Try to load checkpoint
    loaded_measures: List[FontMeasures] = []
    checkpoint_files: List[str] = []
    if checkpoint_path.exists():
        try:
            loaded_measures, missing_files = load_measurements_checkpoint(
                checkpoint_path, expected_files=files
            )
            checkpoint_files = [fm.path for fm in loaded_measures]

            if loaded_measures:
                # Check if checkpoint matches current file set
                checkpoint_set = set(checkpoint_files)
                files_set = set(files)
                overlap = len(checkpoint_set & files_set)
                new_files = len(files_set - checkpoint_set)
                removed_files = len(checkpoint_set - files_set)

                # If no overlap, checkpoint is useless - skip it
                if overlap == 0:
                    cs.StatusIndicator("info").add_message(
                        "Checkpoint found but contains no matching fonts (skipping)"
                    ).emit(console)
                    measures = []
                else:
                    # Display checkpoint status with StatusIndicator
                    cs.emit("", console=console)
                    if checkpoint_set == files_set:
                        # Perfect match
                        cs.StatusIndicator("info").add_message(
                            f"Checkpoint found with measurements for all {cs.fmt_count(len(loaded_measures))} fonts"
                        ).add_item(
                            "Using checkpoint would skip all measurement",
                            indent_level=1,
                        ).emit(console)
                        prompt_msg = "Use checkpoint?"
                    elif new_files > 0 and removed_files == 0:
                        # Only new files added
                        cs.StatusIndicator("info").add_message(
                            f"Checkpoint found with measurements for {cs.fmt_count(overlap)} fonts"
                        ).add_item(
                            f"{cs.fmt_count(new_files)} new file(s) would still need measurement",
                            indent_level=1,
                        ).emit(console)
                        prompt_msg = "Use checkpoint for existing measurements?"
                    elif removed_files > 0 and new_files == 0:
                        # Some files removed
                        cs.StatusIndicator("info").add_message(
                            f"Checkpoint found with measurements for {cs.fmt_count(overlap)} fonts"
                        ).add_item(
                            f"({cs.fmt_count(removed_files)} files from checkpoint no longer in current set)",
                            indent_level=1,
                        ).emit(console)
                        prompt_msg = "Use checkpoint? (skips all measurement)"
                    else:
                        # Mixed changes
                        cs.StatusIndicator("info").add_message(
                            f"Checkpoint found: {cs.fmt_count(overlap)} matching fonts"
                        ).add_item(
                            f"{cs.fmt_count(new_files)} new file(s) would need measurement",
                            indent_level=1,
                        ).add_item(
                            f"{cs.fmt_count(removed_files)} file(s) removed from current set",
                            indent_level=1,
                        ).emit(console)
                        prompt_msg = "Use checkpoint for matching fonts?"

                    # Prompt user for checkpoint usage
                    cs.emit("", console=console)
                    resp = cs.prompt_confirm(prompt_msg, default=True)

                    if resp:
                        # Use checkpoint for matching files
                        measures = [
                            fm for fm in loaded_measures if fm.path in files_set
                        ]
                        # Update files list to only measure missing ones
                        files = [f for f in files if f not in checkpoint_files]

                        if new_files > 0:
                            cs.StatusIndicator("info").add_message(
                                f"Reusing {cs.fmt_count(overlap)} measurements, will measure {cs.fmt_count(new_files)} new file(s)"
                            ).emit(console)
                        else:
                            cs.StatusIndicator("info").add_message(
                                f"Reusing all {cs.fmt_count(len(measures))} measurements from checkpoint"
                            ).emit(console)
                    else:
                        # User wants fresh measurement
                        measures = []
                        cs.StatusIndicator("info").add_message(
                            f"Remeasuring all {cs.fmt_count(len(files))} fonts"
                        ).emit(console)
            else:
                measures = []
        except Exception as e:
            cs.StatusIndicator("warning").add_message(
                f"Error processing checkpoint: {e}. Proceeding with fresh measurement."
            ).emit(console)
            measures = []
    else:
        measures = []

    # Measure fonts (only those not in checkpoint)
    if files:
        cs.StatusIndicator("info").add_message(
            f"Measuring {cs.fmt_count(len(files))} file(s) for bounds and metrics"
        ).emit(console)

        try:
            measures = measure_fonts(files, existing_measures=measures)
        except KeyboardInterrupt:
            cs.emit("", console=console)
            cs.StatusIndicator("warning").add_message(
                "Measurement interrupted. Saving partial checkpoint..."
            ).emit(console)
            # Save partial checkpoint before exiting
            if measures:
                save_measurements_checkpoint(measures, checkpoint_path)
            cs.StatusIndicator("info").add_message(
                f"Partial checkpoint saved to {checkpoint_path}"
            ).emit(console)
            sys.exit(0)

    if not measures:
        cs.StatusIndicator("error").add_message("No measurable fonts found").emit(
            console
        )
        sys.exit(2)

    # Save checkpoint after successful measurement
    save_measurements_checkpoint(measures, checkpoint_path)

    cs.emit("", console=console)
    forced_groups = collect_groups(args)
    families = group_families(args, measures, forced_groups)
    cs.emit("", console=console)

    family_plans = build_plans(families, config)

    if args.report:
        # Generate detailed impact report
        generate_family_report(families, config, args)
        cs.emit("")
        cs.StatusIndicator("info").add_message(
            "Report complete. Use --dry-run to preview changes or remove --report to apply."
        ).emit(console)
        sys.exit(0)

    any_changes_needed = report_changes(families, family_plans, args, forced_groups)

    if not any_changes_needed:
        elapsed = time.time() - start_time
        cs.emit("")
        cs.StatusIndicator("success").add_message(
            "All families already normalized!"
        ).emit(console)
        cs.emit(
            f"{cs.INDENT}[darktext.dim]Total time: [bold]{elapsed:.1f}[/bold]s[/darktext.dim]",
            console=console,
        )
        sys.exit(0)

    if args.dry_run:
        process_all(measures, dry_run=True)
        cs.emit(
            f"{cs.INDENT}[darktext.dim]Total time: [bold]{time.time() - start_time:.1f}[/bold]s[/darktext.dim]",
            console=console,
        )
        return

    if not args.yes:
        confirm_or_exit(len(measures))

    process_all(measures, dry_run=False)
    cs.emit(
        f"{cs.INDENT}[darktext.dim]Total time: [bold]{time.time() - start_time:.1f}[/bold]s[/darktext.dim]",
        console=console,
    )


if __name__ == "__main__":
    main()
