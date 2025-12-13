"""Checkpoint save/load functions for font measurements."""

import hashlib
import json
import sys
from datetime import datetime
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

from . import config
from . import models

console = get_console()
MetricsConfig = config.MetricsConfig
FontMeasures = models.FontMeasures


def compute_config_hash(config: MetricsConfig) -> str:
    """Compute a hash of config values that affect clustering.

    Used to invalidate cached clusters when config changes.
    """
    config_str = (
        f"{config.optical_threshold}:"
        f"{config.decorative_span_threshold}:"
        f"{config.unicase_threshold}:"
        f"{config.script_span_threshold}:"
        f"{config.script_asymmetry_ratio}:"
        f"{config.max_span_ratio}"
    )
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def save_measurements_checkpoint(
    measures: List[FontMeasures],
    checkpoint_path: Path,
    config: Optional[MetricsConfig] = None,
    clusters: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> None:
    """Save font measurements to checkpoint file (JSON format).

    Args:
        measures: Font measurements to save
        checkpoint_path: Path to checkpoint file
        config: Optional config to store hash for cluster validation
        clusters: Optional cluster assignments per family to cache
    """
    try:
        checkpoint_data = {
            "version": "1.1",  # Bump version for cluster storage
            "timestamp": datetime.now().isoformat(),
            "measures": [],
        }

        # Store config hash if provided (for cluster validation)
        if config:
            checkpoint_data["config_hash"] = compute_config_hash(config)

        # Store cluster assignments if provided
        if clusters:
            checkpoint_data["clusters"] = clusters

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
                "is_unicase": fm.is_unicase,
                "is_script": fm.is_script,
                "is_decorative_candidate": fm.is_decorative_candidate,
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
    checkpoint_path: Path,
    expected_files: Optional[List[str]] = None,
    config: Optional[MetricsConfig] = None,
) -> Tuple[List[FontMeasures], List[str], Optional[Dict[str, Dict[str, List[str]]]]]:
    """Load font measurements from checkpoint file.

    Returns:
        Tuple of (loaded_measures, missing_files, cached_clusters) where:
        - loaded_measures: Font measurements loaded from checkpoint
        - missing_files: Files not found in checkpoint or that no longer exist
        - cached_clusters: Cluster assignments if valid, None otherwise
    """
    if not checkpoint_path.exists():
        return ([], expected_files or [], None)

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)

        version = checkpoint_data.get("version", "1.0")
        if version not in ("1.0", "1.1"):
            cs.StatusIndicator("warning").add_message(
                f"Checkpoint version mismatch: {version}"
            ).emit(console)
            return ([], expected_files or [], None)

        # Validate config hash if both config and hash are present
        cached_clusters = None
        if version == "1.1" and config:
            stored_hash = checkpoint_data.get("config_hash")
            current_hash = compute_config_hash(config)
            if stored_hash == current_hash:
                # Config matches - clusters are valid
                cached_clusters = checkpoint_data.get("clusters")
            # If hash doesn't match, clusters are invalid (config changed)

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
            fm.is_unicase = measure_data.get("is_unicase", False)
            fm.is_script = measure_data.get("is_script", False)
            fm.is_decorative_candidate = measure_data.get(
                "is_decorative_candidate", False
            )
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

        return (loaded_measures, missing_files, cached_clusters)

    except json.JSONDecodeError as e:
        cs.StatusIndicator("warning").add_message(
            f"Checkpoint file corrupted: {e}"
        ).emit(console)
        return ([], expected_files or [], None)
    except Exception as e:
        cs.StatusIndicator("warning").add_message(
            f"Error loading checkpoint: {e}"
        ).emit(console)
        return ([], expected_files or [], None)
