"""Font measurement functions for extracting metrics from font files."""

import fnmatch
import sys
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from fontTools.ttLib import TTFont

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

from . import config
from . import font_io
from . import models

console = get_console()

# Import constants from config
CAP_HEIGHT_GLYPHS = config.CAP_HEIGHT_GLYPHS
U_LOWER_X = config.U_LOWER_X
UPPER_A = config.UPPER_A
UPPER_Z = config.UPPER_Z
LOWERCASE_XHEIGHT_SAMPLES = config.LOWERCASE_XHEIGHT_SAMPLES
UPPERCASE_CAPHEIGHT_SAMPLES = config.UPPERCASE_CAPHEIGHT_SAMPLES
LOWER_ASCENDER_CODEPOINTS = config.LOWER_ASCENDER_CODEPOINTS
LOWER_DESCENDER_CODEPOINTS = config.LOWER_DESCENDER_CODEPOINTS

# Import font I/O functions
_read_ttfont = font_io._read_ttfont
_get_upm = font_io._get_upm
_codepoint_bounds = font_io._codepoint_bounds
_font_overall_bounds = font_io._font_overall_bounds

FontMeasures = models.FontMeasures


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


def _get_cap_height_from_glyphs(font: TTFont) -> Optional[int]:
    """Get cap-height by measuring glyphs directly (ignores OS/2 metadata).

    Legacy function for backward compatibility. Use _get_cap_height_from_glyphs_robust() for better accuracy.
    """
    for cp in CAP_HEIGHT_GLYPHS:
        b = _codepoint_bounds(font, cp)
        if b:
            _, _, _, yMax = b
            return int(round(yMax))
    return None


def _get_x_height_from_glyphs(font: TTFont) -> Optional[int]:
    """Get x-height by measuring 'x' glyph directly (ignores OS/2 metadata).

    Legacy function for backward compatibility. Use _get_x_height_from_glyphs_robust() for better accuracy.
    """
    b = _codepoint_bounds(font, U_LOWER_X)
    if b:
        return int(round(b[3]))
    return None


def _get_x_height_from_glyphs_robust(font: TTFont) -> Optional[Tuple[int, int, int]]:
    """Get x-height by measuring multiple lowercase glyphs (robust sampling).

    Samples multiple lowercase letters to get a more representative x-height,
    using median to ignore outliers (e.g., 'o' with overshoot).

    Returns:
        Tuple of (median, min, max) x-heights, or None if insufficient data
    """
    heights: List[int] = []

    for cp in LOWERCASE_XHEIGHT_SAMPLES:
        b = _codepoint_bounds(font, cp)
        if b:
            heights.append(int(round(b[3])))  # yMax

    if len(heights) < 3:
        # Fallback to single 'x' measurement
        b = _codepoint_bounds(font, U_LOWER_X)
        if b:
            height = int(round(b[3]))
            return (height, height, height)
        return None

    heights.sort()
    n = len(heights)

    # Calculate median
    median = (
        heights[n // 2] if n % 2 == 1 else (heights[n // 2 - 1] + heights[n // 2]) // 2
    )

    return (median, heights[0], heights[-1])


def _get_cap_height_from_glyphs_robust(font: TTFont) -> Optional[Tuple[int, int, int]]:
    """Get cap-height by measuring multiple uppercase glyphs (robust sampling).

    Samples multiple uppercase letters to get a more representative cap-height,
    using median to ignore outliers.

    Returns:
        Tuple of (median, min, max) cap-heights, or None if insufficient data
    """
    heights: List[int] = []

    for cp in UPPERCASE_CAPHEIGHT_SAMPLES:
        b = _codepoint_bounds(font, cp)
        if b:
            heights.append(int(round(b[3])))  # yMax

    if len(heights) < 3:
        # Fallback to 'H' or 'I'
        for cp in CAP_HEIGHT_GLYPHS:
            b = _codepoint_bounds(font, cp)
            if b:
                height = int(round(b[3]))
                return (height, height, height)
        return None

    heights.sort()
    n = len(heights)

    # Calculate median
    median = (
        heights[n // 2] if n % 2 == 1 else (heights[n // 2 - 1] + heights[n // 2]) // 2
    )

    return (median, heights[0], heights[-1])


def detect_decorative_standalone(
    fm: FontMeasures,
    config,
) -> bool:
    """Detect decorative fonts using standalone heuristics.

    Decorative characteristics:
    - Span significantly larger than typical (1.4x+ UPM)
    - Not script (script is 2.0x+ and descender-dominant)
    - Not unicase (unicase has x-height ≈ cap-height)

    This is a conservative standalone check. Clustering will refine
    using cluster comparison.

    Args:
        fm: FontMeasures with min_y, max_y, and upm set
        config: MetricsConfig with decorative_span_threshold

    Returns:
        True if font is detected as decorative candidate
    """
    if fm.max_y is None or fm.min_y is None or fm.upm <= 0:
        return False

    # Already classified as script or unicase - not decorative
    if fm.is_script or fm.is_unicase:
        return False

    # Calculate span ratio
    fm_span = (fm.max_y - fm.min_y) / fm.upm

    # Decorative fonts have moderately large span (1.4x-2.0x)
    # Below 1.4x: normal font
    # Above 2.0x: likely script (should be caught by script detection)
    if config.decorative_span_threshold <= fm_span < config.script_span_threshold:
        return True

    return False


def is_script_font(
    fm: FontMeasures,
    script_span_threshold: float = 2.0,
    script_asymmetry_ratio: float = 1.2,
    min_span_ratio: float = 1.5,
) -> bool:
    """Detect script fonts based on span and descender-dominance characteristics.

    Script fonts typically have:
    - Large vertical span (swashes extend far above/below)
    - Descender-dominant (swashes go down more than up)

    This is a standalone detection that doesn't require cluster comparison.
    Clustering may refine this detection by comparing to family average.

    Args:
        fm: FontMeasures with min_y, max_y, and upm set
        script_span_threshold: Minimum span ratio vs typical font (default: 2.0x)
        script_asymmetry_ratio: Descender must exceed ascender by this ratio (default: 1.2x)
        min_span_ratio: Minimum span as ratio of UPM to consider (default: 1.5x)

    Returns:
        True if font is detected as script
    """
    if fm.max_y is None or fm.min_y is None or fm.upm <= 0:
        return False

    # Calculate span as ratio of UPM
    fm_span = (fm.max_y - fm.min_y) / fm.upm

    # Script fonts have unusually large span (at least min_span_ratio of UPM)
    if fm_span < min_span_ratio:
        return False

    # Check descender-dominance (script swashes go down more than up)
    fm_desc_ratio = abs(fm.min_y / fm.upm) if fm.min_y else 0
    fm_asc_ratio = (fm.max_y / fm.upm) if fm.max_y else 0

    # Script: descender deeper than ascender is tall (by configurable ratio)
    if fm_desc_ratio > fm_asc_ratio * script_asymmetry_ratio:
        return True

    return False


def is_unicase(
    fm: FontMeasures,
    threshold: float = 0.05,
    overshoot_tolerance: float = 0.03,
) -> bool:
    """Detect unicase fonts where x-height ≈ cap-height.

    Unicase fonts have uppercase-style letterforms at lowercase proportions.
    Key characteristic: x-height matches or is very close to cap-height.

    For proper detection:
    - Traditional: x-height (~500) < cap-height (~700)
    - Unicase: x-height (~500) ≈ cap-height (~500)

    Note: The unicase cap-height should match the traditional x-height,
    not the traditional cap-height.

    Uses glyph measurements from FontMeasures, which may come from OS/2
    metadata or glyph measurements depending on what's available.

    Args:
        fm: FontMeasures with x_height and cap_height set
        threshold: Maximum difference ratio to consider unicase (default: 5% UPM)
        overshoot_tolerance: Additional tolerance for overshoot (default: 3% UPM)

    Returns:
        True if font is detected as unicase
    """
    if fm.x_height is None or fm.cap_height is None or fm.upm <= 0:
        return False

    x_ratio = fm.x_height / fm.upm
    cap_ratio = fm.cap_height / fm.upm

    # Check difference between x-height and cap-height
    diff = abs(x_ratio - cap_ratio)

    # Safety check 1: Reject if cap is too tall (traditional font)
    # Traditional fonts have cap significantly taller than x-height
    # If cap is more than 10% UPM taller than x-height, it's traditional
    if (cap_ratio - x_ratio) > 0.10:
        return False  # Cap too tall, traditional font

    # Safety check 2: Avoid false positives from small caps
    # True unicase: both metrics reasonably tall (0.45-0.65 of UPM)
    # Small caps: both metrics very short (~0.40 of UPM or less)
    min_height = 0.45
    max_height = 0.65

    if cap_ratio < min_height or cap_ratio > max_height:
        return False  # Outside unicase range

    # Primary check: x-height and cap-height are very close
    # Allow for slight overshoot (letters like 'o', 'c' can be taller)
    effective_threshold = threshold + overshoot_tolerance

    return diff < effective_threshold


def measure_fonts(
    filepaths: List[str],
    existing_measures: Optional[List[FontMeasures]] = None,
    unicase_threshold: float = 0.05,
    script_span_threshold: float = 2.0,
    script_asymmetry_ratio: float = 1.2,
    decorative_span_threshold: float = 1.4,
    assume_script: Optional[List[str]] = None,
    assume_decorative: Optional[List[str]] = None,
    assume_unicase: Optional[List[str]] = None,
    exclude_measuring: Optional[List[str]] = None,
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

                # Check if font name suggests unicase (fallback for fonts that don't match geometric pattern)
                filename_hint = (
                    "Unicase" in Path(fp).name or "unicase" in Path(fp).name.lower()
                )
                family_name_hint = (
                    "unicase" in fm.family_name.lower() if fm.family_name else False
                )

                # For unicase detection, prefer robust glyph measurements over OS/2 metadata
                # OS/2 metadata can be incorrect, especially for unicase fonts
                x_measurement = _get_x_height_from_glyphs_robust(font)
                cap_measurement = _get_cap_height_from_glyphs_robust(font)

                # Create a temporary FontMeasures with glyph-based values for detection
                if x_measurement is not None and cap_measurement is not None:
                    x_median, x_min, x_max = x_measurement
                    cap_median, cap_min, cap_max = cap_measurement

                    # Use median values for detection (most representative)
                    detection_fm = FontMeasures(fp, fm.upm)
                    detection_fm.x_height = x_median
                    detection_fm.cap_height = cap_median
                    detection_fm.upm = fm.upm

                    # Calculate overshoot (natural variation in glyph heights)
                    x_variation = (x_max - x_min) / fm.upm if fm.upm > 0 else 0
                    cap_variation = (cap_max - cap_min) / fm.upm if fm.upm > 0 else 0
                    avg_variation = (x_variation + cap_variation) / 2.0

                    # Adjust overshoot tolerance based on actual font variation
                    # Fonts with more variation (e.g., decorative) need more tolerance
                    overshoot_tolerance = max(0.03, avg_variation * 1.5)

                    fm.is_unicase = is_unicase(
                        detection_fm,
                        threshold=unicase_threshold,
                        overshoot_tolerance=overshoot_tolerance,
                    )
                else:
                    # Fall back to simple glyph measurements if robust unavailable
                    x_from_glyph = _get_x_height_from_glyphs(font)
                    cap_from_glyph = _get_cap_height_from_glyphs(font)
                    if x_from_glyph is not None and cap_from_glyph is not None:
                        detection_fm = FontMeasures(fp, fm.upm)
                        detection_fm.x_height = x_from_glyph
                        detection_fm.cap_height = cap_from_glyph
                        detection_fm.upm = fm.upm
                        fm.is_unicase = is_unicase(
                            detection_fm, threshold=unicase_threshold
                        )
                    else:
                        # Fall back to OS/2-based values if glyph measurements unavailable
                        fm.is_unicase = is_unicase(fm, threshold=unicase_threshold)

                # Fallback: if geometric detection failed but name suggests unicase
                if not fm.is_unicase and (filename_hint or family_name_hint):
                    fm.is_unicase = True

                # Check pattern overrides FIRST (before automatic detection)
                filename = Path(fp).name
                forced_unicase = False
                forced_script = False
                forced_decorative = False

                # Check if font should be excluded from family calculations
                if exclude_measuring:
                    for pattern in exclude_measuring:
                        if fnmatch.fnmatch(filename, pattern):
                            fm.is_excluded_from_calculations = True
                            break

                if assume_unicase:
                    for pattern in assume_unicase:
                        if fnmatch.fnmatch(filename, pattern):
                            forced_unicase = True
                            break

                if assume_script:
                    for pattern in assume_script:
                        if fnmatch.fnmatch(filename, pattern):
                            forced_script = True
                            break

                if assume_decorative:
                    for pattern in assume_decorative:
                        if fnmatch.fnmatch(filename, pattern):
                            forced_decorative = True
                            break

                # Unified detection phase (with pattern overrides)
                # Priority: forced > automatic detection
                # Order matters: unicase → script → decorative

                # 1. Unicase detection
                if forced_unicase:
                    fm.is_unicase = True
                # (automatic unicase detection already done above)

                # 2. Script detection
                if forced_script:
                    fm.is_script = True
                else:
                    fm.is_script = is_script_font(
                        fm,
                        script_span_threshold=script_span_threshold,
                        script_asymmetry_ratio=script_asymmetry_ratio,
                    )

                # 3. Decorative detection (requires config object)
                from . import config

                temp_config = config.MetricsConfig(
                    decorative_span_threshold=decorative_span_threshold
                )
                if forced_decorative:
                    fm.is_decorative_candidate = True
                else:
                    fm.is_decorative_candidate = detect_decorative_standalone(
                        fm, temp_config
                    )

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
