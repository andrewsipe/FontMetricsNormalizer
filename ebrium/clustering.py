"""Optical clustering logic for grouping similar fonts."""

from typing import Dict, List, Optional, Tuple

from . import config
from . import models

MetricsConfig = config.MetricsConfig
FontMeasures = models.FontMeasures


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
            # Use max_span_ratio for pre-check (separate from decorative detection)
            if span_ratio > config.max_span_ratio:
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
    """Refine decorative detection using cluster context.

    Confirms/rejects decorative candidate flag from measurement phase.

    Requirements:
    - Standalone detection flagged as candidate OR
    - Core metrics match cluster but span is 1.4x+ larger
    """
    if not core_cluster:
        # No cluster context - trust standalone detection
        return fm.is_decorative_candidate

    # Check if marked as candidate during measurement
    is_candidate = fm.is_decorative_candidate

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
                # Use max_span_ratio for structural rejection
                if span_ratio > config.max_span_ratio:
                    return False

    # Check if core metrics match any font in cluster
    matches_core = any(
        compute_optical_similarity(fm, core_fm, threshold, config)
        for core_fm in core_cluster
    )

    if not matches_core:
        # Different structure - not a decorative variant
        return False

    # If candidate OR matches cluster with inflated bounds
    if is_candidate:
        return True

    # Check bounds inflation vs cluster
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


def detect_script_font(
    fm: FontMeasures,
    core_cluster: List[FontMeasures],
    config: MetricsConfig,
) -> bool:
    """Refine script detection using cluster context.

    Confirms/rejects script flag from measurement phase.

    Requirements:
    - Standalone detection flagged as script OR
    - Core metrics match cluster but span is 2.0x+ larger
    """
    if not core_cluster:
        # No cluster context - trust standalone detection
        return fm.is_script

    # If already detected during measurement, confirm
    if fm.is_script:
        return True

    # Additional check using cluster context
    # Check if core metrics match any font in cluster (same letter body)
    matches_core = any(
        compute_optical_similarity(fm, core_fm, config.optical_threshold, config)
        for core_fm in core_cluster
    )

    if not matches_core:
        return False  # Different structure, not a script variant

    # Check span ratio (script fonts have much larger span)
    if fm.max_y is None or fm.min_y is None or fm.upm <= 0:
        return False

    fm_span = (fm.max_y - fm.min_y) / fm.upm
    cluster_spans = [
        (cfm.max_y - cfm.min_y) / cfm.upm
        for cfm in core_cluster
        if cfm.max_y is not None and cfm.min_y is not None and cfm.upm > 0
    ]

    if not cluster_spans:
        return False

    avg_cluster_span = sum(cluster_spans) / len(cluster_spans)
    if avg_cluster_span <= 0:
        return False

    span_ratio = fm_span / avg_cluster_span

    # Script threshold: 2.0x span (vs decorative 1.3x)
    if span_ratio < config.script_span_threshold:
        return False

    # AND descender-dominant (script swashes go down more than up)
    fm_desc_ratio = abs(fm.min_y / fm.upm) if fm.min_y else 0
    fm_asc_ratio = (fm.max_y / fm.upm) if fm.max_y else 0

    # Script: descender deeper than ascender is tall (by configurable ratio)
    if fm_desc_ratio > fm_asc_ratio * config.script_asymmetry_ratio:
        return True

    return False


def cluster_group_helper(
    group: List[FontMeasures],
    threshold: float,
    config: MetricsConfig,
) -> Tuple[List[List[FontMeasures]], List[FontMeasures], List[FontMeasures]]:
    """Cluster a single group of fonts (original clustering logic).

    Returns: (core_clusters, decorative_outliers, script_outliers)
    """
    if len(group) <= 1:
        return ([group] if group else [], [], [])

    # Build similarity graph based on cap height + x-height + descender
    n = len(group)
    similar_pairs: List[Tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if compute_optical_similarity(group[i], group[j], threshold, config):
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
    for idx, fm in enumerate(group):
        root = find(idx)
        clusters_dict.setdefault(root, []).append(fm)
        fm.cluster_id = root

    core_clusters = [c for c in clusters_dict.values() if len(c) > 1]
    singletons = [c[0] for c in clusters_dict.values() if len(c) == 1]

    # Find main cluster (largest)
    if not core_clusters:
        return ([], singletons, [])

    main_cluster = max(core_clusters, key=len)

    # Check singletons: script, decorative variants, or true outliers?
    # Check scripts FIRST (more specific pattern than decorative)
    script_outliers = []
    decorative_outliers = []
    true_outliers = []

    for fm in singletons:
        # Refine script detection (measurement flag + cluster check)
        is_script_refined = detect_script_font(fm, main_cluster, config)
        if is_script_refined or fm.is_script:
            script_outliers.append(fm)
            fm.is_script = True  # Confirm
            continue

        # Refine decorative detection (measurement flag + cluster check)
        is_decorative_refined = detect_decorative_outlier(
            fm, main_cluster, threshold, config
        )
        if is_decorative_refined:
            decorative_outliers.append(fm)
            fm.is_decorative_outlier = True  # Confirm
            fm.is_decorative_candidate = False  # No longer candidate
            continue

        # True outlier (neither script nor decorative)
        true_outliers.append(fm)
        fm.is_decorative_candidate = False  # Reject candidate status

    # True outliers become single-font clusters
    all_clusters = core_clusters + [[fm] for fm in true_outliers]

    return (all_clusters, decorative_outliers, script_outliers)


def detect_optical_clusters(
    measures: List[FontMeasures], threshold: float, config: MetricsConfig
) -> Tuple[List[List[FontMeasures]], List[FontMeasures], List[FontMeasures]]:
    """Detect core clusters and decorative outliers based on cap height.

    Unicase fonts are treated as special decorative variants that inherit
    baseline alignment from traditional fonts in mixed families.
    Script fonts are detected separately with larger span and descender-dominant characteristics.

    Returns: (core_clusters, decorative_outliers, script_outliers)
    """
    if len(measures) <= 1:
        return ([measures] if measures else [], [], [])

    # Separate unicase and non-unicase fonts
    unicase_fonts = [fm for fm in measures if fm.is_unicase]
    non_unicase_fonts = [fm for fm in measures if not fm.is_unicase]

    # If we have BOTH unicase and non-unicase fonts in this family,
    # treat unicase as decorative outliers (inherit baseline from traditional)
    if unicase_fonts and non_unicase_fonts:
        # Cluster only the traditional fonts normally
        clusters, decorative, scripts = cluster_group_helper(
            non_unicase_fonts, threshold, config
        )

        # Mark unicase fonts as decorative outliers
        for fm in unicase_fonts:
            fm.is_decorative_outlier = True

        # Add unicase to decorative outliers list
        all_decorative = decorative + unicase_fonts

        return (clusters, all_decorative, scripts)

    # If ALL fonts are unicase, cluster them normally (pure unicase family)
    elif unicase_fonts and not non_unicase_fonts:
        clusters, decorative, scripts = cluster_group_helper(
            unicase_fonts, threshold, config
        )
        return (clusters, decorative, scripts)

    # If no unicase fonts, cluster normally
    else:
        clusters, decorative, scripts = cluster_group_helper(
            non_unicase_fonts, threshold, config
        )
        return (clusters, decorative, scripts)
