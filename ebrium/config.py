"""Configuration constants and dataclasses for metrics normalization."""

from dataclasses import dataclass
from typing import Optional, Tuple


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

# Lowercase letters for x-height sampling (exclude ascenders/descenders)
LOWERCASE_XHEIGHT_SAMPLES: Tuple[int, ...] = (
    0x0061,  # a
    0x0063,  # c
    0x0065,  # e
    0x006D,  # m
    0x006E,  # n
    0x006F,  # o
    0x0072,  # r
    0x0073,  # s
    0x0075,  # u
    0x0076,  # v
    0x0077,  # w
    0x0078,  # x
    0x007A,  # z
)

# Uppercase letters for cap height sampling (flat tops, no curves that might exceed)
UPPERCASE_CAPHEIGHT_SAMPLES: Tuple[int, ...] = (
    0x0042,  # B
    0x0044,  # D
    0x0045,  # E
    0x0046,  # F
    0x0048,  # H
    0x0049,  # I
    0x004B,  # K
    0x004C,  # L
    0x004D,  # M
    0x004E,  # N
    0x0050,  # P
    0x0052,  # R
    0x0054,  # T
)


@dataclass
class MetricsConfig:
    """Configuration constants for metrics normalization."""

    # Internal representation: all stored as fractions (convert from percentage input)
    target_span: float = 1.3  # Internal: as multiplier (1.3x = 130% of UPM)
    win_buffer: float = 0.02  # Internal: as fraction (0.02 = 2%)
    xheight_softener: float = 0.6
    adapt_for_xheight: bool = True
    optical_threshold: float = (
        0.025  # 2.5% UPM for identical detection (validated optimal)
    )
    top_margin: float = 0.25  # Internal: as fraction (0.25 = 25% of UPM)
    ascender_override_threshold: float = (
        0.5  # Fraction of top_margin to trigger ascender override (default: 0.5 = 50%)
    )
    max_adjustment: Optional[float] = (
        None  # Internal: as fraction (0.08 = 8% max adjustment)
    )
    # Clustering thresholds
    max_span_ratio: float = (
        1.5  # Pre-check rejection threshold (separate from decorative detection)
    )
    # Type detection thresholds (in order of specificity)
    unicase_threshold: float = 0.05  # 5% UPM x-height ≈ cap-height
    script_span_threshold: float = 2.0  # 2.0x span minimum for script
    script_asymmetry_ratio: float = 1.2  # 1.2x descender-dominant
    decorative_span_threshold: float = (
        1.4  # 1.4x span minimum (changed from 1.3x to create gap with script)
    )
    script_win_buffer_multiplier: float = (
        1.5  # Buffer multiplier for script fonts (1.5x default)
    )
    auto_adjust_target: bool = (
        True  # Enable automatic target adjustment based on x-height
    )
