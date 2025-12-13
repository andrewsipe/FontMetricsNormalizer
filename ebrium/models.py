"""Font measurement data models."""

from typing import Optional


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

        # Detection flags (set during measurement)
        self.is_unicase: bool = False
        self.is_script: bool = False
        self.is_decorative_candidate: bool = False  # Standalone detection
        self.is_excluded_from_calculations: bool = (
            False  # Excluded from family calculations
        )

        # Clustering results (set during clustering refinement)
        self.cluster_id: Optional[int] = None
        self.is_decorative_outlier: bool = False  # Confirmed by clustering

        # Computed targets
        self.target_typo_asc: Optional[int] = None
        self.target_typo_desc: Optional[int] = None
        self.target_win_asc: Optional[int] = None
        self.target_win_desc: Optional[int] = None

        # Family context (for status reporting)
        self.family_upm_majority: Optional[int] = None
