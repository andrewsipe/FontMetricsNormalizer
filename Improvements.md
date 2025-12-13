# Font Metrics Normalization - Improvement Plan

## Executive Summary

This plan addresses fundamental architectural issues in the clustering and grouping system, focusing on clarity, predictability, and user experience. The current system has overlapping modes, inconsistent detection timing, and confusing thresholds that make debugging difficult and user intent unclear.

---

## Problem 1: Confusing Mode Overlap

### Current State
```
--per-font        → No grouping + No clustering
--safe-mode       → Family grouping + No clustering
(default)         → Family grouping + Clustering
--superfamily     → Superfamily grouping + Clustering
```

**Issues:**
- `--safe-mode` groups families but claims to be "safe" (conservative)
- `--per-font` does TWO things (no grouping AND no clustering)
- Users can't get "family grouping without clustering" intuitively
- Mixing `--per-font` + `--superfamily` creates undefined behavior

### Root Cause
Two orthogonal concerns (grouping strategy + clustering strategy) are conflated into single flags.

### Solution: Separate Concerns with Clear Modes

**Implementation:**

```python
# File: cli.py, parse_args()

# Replace current --per-font, --superfamily, --safe-mode with:
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--individual", 
    dest="grouping_mode", 
    action="store_const", 
    const="individual",
    help="Normalize each font independently (no grouping, no clustering)")

mode.add_argument("--family", 
    dest="grouping_mode", 
    action="store_const", 
    const="family",
    help="Group by family name, cluster within families (default)")

mode.add_argument("--superfamily", 
    dest="grouping_mode", 
    action="store_const", 
    const="superfamily",
    help="Merge families with shared prefix, cluster across superfamily")

mode.add_argument("--conservative", 
    dest="grouping_mode", 
    action="store_const", 
    const="conservative",
    help="Group by family, use bbox only (no clustering, safest)")

parser.set_defaults(grouping_mode="family")
```

**Changes Required:**

1. **File: `cli.py`**
   ```python
   # OLD:
   if args.per_font:
       cs.StatusIndicator("info").add_message(
           f"Processing {cs.fmt_count(len(measures))} font(s) individually..."
       ).emit(console)
       return {Path(fm.path).stem: [fm] for fm in measures}
   
   # NEW:
   if args.grouping_mode == "individual":
       cs.StatusIndicator("info").add_message(
           f"Processing {cs.fmt_count(len(measures))} font(s) individually (no grouping, no clustering)"
       ).emit(console)
       return {Path(fm.path).stem: [fm] for fm in measures}
   
   # OLD:
   if args.superfamily:
       groups = sorter.group_by_superfamily(...)
   
   # NEW:
   if args.grouping_mode == "superfamily":
       groups = sorter.group_by_superfamily(...)
   elif args.grouping_mode in ("family", "conservative"):
       groups = sorter.group_by_family(...)
   ```

2. **File: `planning.py`, function `build_plans()`**
   ```python
   # OLD:
   if config.safe_mode:
       plan_safe_metrics(group, config)
       continue
   
   # NEW:
   if args.grouping_mode == "individual":
       # Skip all clustering and family normalization
       for fm in group:
           plan_adaptive_metrics([fm], fam_min, fam_max, 
                                compute_family_normalized_ascender([fm], config), 
                                config, verbosity)
       continue
   
   if args.grouping_mode == "conservative":
       # Use bbox approach but still group by family
       plan_safe_metrics(group, config)
       continue
   
   # Otherwise proceed with clustering (family or superfamily mode)
   ```

3. **File: `config.py`**
   ```python
   # OLD:
   safe_mode: bool = False
   
   # NEW: Remove safe_mode from config
   # (Mode is now passed via args.grouping_mode, not config)
   ```

4. **File: `validation.py`, function `validate_args()`**
   ```python
   # Remove old validation:
   # if args.per_font and args.superfamily: ...
   
   # Add new validation:
   if args.grouping_mode == "individual":
       if args.ignore_prefix or args.exclude or args.combine:
           cs.StatusIndicator("warning").add_message(
               "--individual ignores grouping modifiers (--ignore-prefix, --exclude, --combine)"
           ).emit(console)
   
   if args.grouping_mode == "conservative":
       cs.StatusIndicator("info").add_message(
           "Conservative mode: using bbox approach (no clustering)"
       ).emit(console)
   ```

**Behavior Matrix:**

| Mode | Grouping | Clustering | Use Case |
|------|----------|------------|----------|
| `--individual` | None | None | Test individual fonts, debug issues |
| `--family` (default) | By family | Yes | Standard normalization |
| `--superfamily` | Superfamily | Yes | Large collections with variants |
| `--conservative` | By family | None | Problematic families, maximum safety |

**Reasoning:**
- Each mode has ONE clear purpose
- No overlapping behavior
- Natural hierarchy from most isolated (individual) to most grouped (superfamily)
- Conservative mode explicitly states "no clustering" in name

---

## Problem 2: Detection Phase Confusion

### Current State

**Unicase Detection:**
- Phase 1: Measurement (`measurements.py::is_unicase()`) → sets `fm.is_unicase`
- Phase 2: Clustering (`clustering.py::detect_optical_clusters()`) → decides treatment based on family composition

**Script Detection:**
- Phase 1: Measurement (`measurements.py::is_script_font()`) → sets `fm.is_script` (standalone)
- Phase 2: Clustering (`clustering.py::detect_script_font()`) → refines using cluster comparison
- Uses OR logic: `if is_script_by_cluster or fm.is_script`

**Decorative Detection:**
- Phase 1: None
- Phase 2: Clustering only (`clustering.py::detect_decorative_outlier()`)

**Issues:**
- Inconsistent timing makes debugging hard ("when was this font flagged?")
- Script detection can change between phases
- Decorative detection can't be debugged until clustering runs
- Unicase treatment depends on family composition (decided late)

### Root Cause
Detection logic is split across measurement and clustering phases with no clear ownership.

### Solution: Unified Detection in Measurement, Refinement in Clustering

**Principle:** All detection happens during measurement. Clustering refines/confirms using cluster context.

**Implementation:**

1. **File: `measurements.py` - Add standalone decorative detection**

```python
def detect_decorative_standalone(fm: FontMeasures, config: MetricsConfig) -> bool:
    """Detect decorative fonts using standalone heuristics.
    
    Decorative characteristics:
    - Span significantly larger than typical (1.4x+ UPM)
    - Not script (script is 2.0x+ and descender-dominant)
    - Not unicase (unicase has x-height ≈ cap-height)
    
    This is a conservative standalone check. Clustering will refine
    using cluster comparison.
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
    if 1.4 <= fm_span < 2.0:
        return True
    
    return False


def measure_fonts(
    filepaths: List[str],
    existing_measures: Optional[List[FontMeasures]] = None,
    unicase_threshold: float = 0.05,
    script_span_threshold: float = 2.0,
    script_asymmetry_ratio: float = 1.2,
    decorative_span_threshold: float = 1.4,  # NEW parameter
) -> List[FontMeasures]:
    """Measure fonts and extract family names.
    
    All font type detection happens here:
    - Unicase: x-height ≈ cap-height
    - Script: 2.0x+ span, descender-dominant
    - Decorative: 1.4x-2.0x span, not script/unicase
    
    Clustering will refine these detections using cluster context.
    """
    # ... existing measurement code ...
    
    for fp in filepaths:
        # ... existing bounds/metrics measurement ...
        
        # Unified detection phase (all at measurement time)
        # Order matters: unicase → script → decorative
        
        # 1. Unicase detection (first, most specific)
        # ... existing unicase detection code ...
        
        # 2. Script detection (second, checks span + asymmetry)
        fm.is_script = is_script_font(
            fm,
            script_span_threshold=script_span_threshold,
            script_asymmetry_ratio=script_asymmetry_ratio,
        )
        
        # 3. Decorative detection (third, catches remaining outliers)
        fm.is_decorative_candidate = detect_decorative_standalone(
            fm, 
            MetricsConfig(decorative_span_threshold=decorative_span_threshold)
        )
        
        measures.append(fm)
    
    return measures
```

2. **File: `models.py` - Add new flag**

```python
class FontMeasures:
    def __init__(self, path: str, upm: int):
        # ... existing fields ...
        
        # Detection flags (set during measurement)
        self.is_unicase: bool = False
        self.is_script: bool = False
        self.is_decorative_candidate: bool = False  # NEW: standalone detection
        
        # Clustering results (set during clustering refinement)
        self.cluster_id: Optional[int] = None
        self.is_decorative_outlier: bool = False  # Confirmed by clustering
```

3. **File: `clustering.py` - Refine detections, don't re-detect**

```python
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
    - Core metrics match cluster but span is 1.3x+ larger
    """
    if not core_cluster:
        # No cluster context - trust standalone detection
        return fm.is_decorative_candidate
    
    # Check if marked as candidate during measurement
    is_candidate = fm.is_decorative_candidate
    
    # Pre-check: Reject structurally different fonts
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
                span_ratio = max(fm_span, avg_cluster_span) / min(fm_span, avg_cluster_span)
                if span_ratio > config.decorative_span_threshold:
                    # Structurally different - not decorative variant
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
    # ... existing bounds comparison code ...
    
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
    # ... existing cluster-based detection ...


def cluster_group_helper(
    group: List[FontMeasures],
    threshold: float,
    config: MetricsConfig,
) -> Tuple[List[List[FontMeasures]], List[FontMeasures], List[FontMeasures]]:
    """Cluster a single group of fonts.
    
    Refines detection flags set during measurement using cluster context.
    """
    # ... existing clustering code ...
    
    # Refine singletons using cluster context
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
    
    # ... rest of clustering ...
```

4. **File: `cli.py` - Update measure_fonts call**

```python
# OLD:
measures = measure_fonts(
    files,
    existing_measures=measures,
    unicase_threshold=config.unicase_threshold,
    script_span_threshold=config.script_span_threshold,
    script_asymmetry_ratio=config.script_asymmetry_ratio,
)

# NEW:
measures = measure_fonts(
    files,
    existing_measures=measures,
    unicase_threshold=config.unicase_threshold,
    script_span_threshold=config.script_span_threshold,
    script_asymmetry_ratio=config.script_asymmetry_ratio,
    decorative_span_threshold=config.decorative_span_threshold,  # NEW
)
```

**Behavior Flow:**

```
Measurement Phase (measurements.py):
  For each font:
    1. Measure bounds (min_y, max_y, cap_height, etc.)
    2. Detect unicase    → fm.is_unicase = True/False
    3. Detect script     → fm.is_script = True/False
    4. Detect decorative → fm.is_decorative_candidate = True/False
  
  Result: All fonts have detection flags set

Clustering Phase (clustering.py):
  For each family:
    1. Build similarity graph
    2. Form clusters
    3. For each singleton:
       - Refine script:     cluster context → confirm/reject fm.is_script
       - Refine decorative: cluster context → confirm (set is_decorative_outlier)
                                            → or reject (clear is_decorative_candidate)
  
  Result: Detection flags refined with cluster context
```

**Reasoning:**
- Single source of truth: measurement phase sets initial flags
- Clustering refines using additional context (cluster comparison)
- Clear ownership: measurement = standalone detection, clustering = refinement
- Easier debugging: check measurement output to see initial detection
- Consistent timing: all fonts get initial flags before clustering

---

## Problem 3: Threshold Cascade & Overlap

### Current State

```python
optical_threshold = 0.025              # 2.5% UPM - similarity check
decorative_span_threshold = 1.3        # 1.3x - pre-check rejection + detection
script_span_threshold = 2.0            # 2.0x - script detection

# Overlap zone: 1.3x - 2.0x
# Fonts with 1.5x span could be:
#   - Rejected by pre-check (1.5x > 1.3x)
#   - Missed by script detection (1.5x < 2.0x)
#   - Never reach decorative detection
```

**Issues:**
- Pre-check rejection happens BEFORE decorative detection
- Fonts in overlap zone (1.3x-2.0x) might be rejected from clustering entirely
- No clear priority: script vs decorative for fonts around 1.5x-1.8x span

### Root Cause
`decorative_span_threshold` used for TWO purposes:
1. Pre-check rejection in similarity comparison
2. Decorative variant detection

### Solution: Separate Pre-Check from Detection Thresholds

**Implementation:**

1. **File: `config.py` - Split threshold**

```python
@dataclass
class MetricsConfig:
    """Configuration constants for metrics normalization."""
    
    # ... existing fields ...
    
    # Clustering thresholds
    optical_threshold: float = 0.025  # 2.5% UPM for optical similarity
    max_span_ratio: float = 1.5       # NEW: Pre-check rejection (was decorative_span_threshold)
    
    # Type detection thresholds (in order of specificity)
    unicase_threshold: float = 0.05         # 5% UPM x-height ≈ cap-height
    script_span_threshold: float = 2.0      # 2.0x span minimum for script
    script_asymmetry_ratio: float = 1.2     # 1.2x descender-dominant
    decorative_span_threshold: float = 1.4  # CHANGED: 1.4x span minimum (was 1.3x)
    
    # Note: Threshold order creates clear zones:
    #   < 1.4x  = Normal font
    #   1.4-2.0 = Decorative zone
    #   >= 2.0x = Script zone (if descender-dominant)
```

2. **File: `clustering.py` - Use separate thresholds**

```python
def compute_optical_similarity(
    fm1: FontMeasures, fm2: FontMeasures, threshold: float, config: MetricsConfig
) -> bool:
    """Compare cap height and baseline alignment.
    
    Pre-check: Overall bounds span (detect structurally different fonts).
    Uses max_span_ratio (1.5x) for pre-check, NOT decorative_span_threshold.
    """
    # ... existing code ...
    
    # Pre-check: Overall bounds span
    if (fm1.max_y is not None and fm1.min_y is not None and 
        fm2.max_y is not None and fm2.min_y is not None):
        span1 = normalize(fm1.max_y - fm1.min_y, fm1.upm)
        span2 = normalize(fm2.max_y - fm2.min_y, fm2.upm)
        
        if span1 and span2 and span1 > 0 and span2 > 0:
            span_ratio = max(span1, span2) / min(span1, span2)
            # Use max_span_ratio (1.5x) for pre-check
            if span_ratio > config.max_span_ratio:  # CHANGED: was decorative_span_threshold
                return False
    
    # ... rest of similarity checks ...
```

3. **File: `clustering.py` - Update decorative detection**

```python
def detect_decorative_outlier(
    fm: FontMeasures,
    core_cluster: List[FontMeasures],
    threshold: float,
    config: MetricsConfig,
) -> bool:
    """Refine decorative detection using cluster context.
    
    Uses decorative_span_threshold (1.4x) for detection, separate from
    pre-check threshold (max_span_ratio = 1.5x).
    """
    # ... existing code ...
    
    # Pre-check: Reject structurally different fonts
    if fm.max_y is not None and fm.min_y is not None and fm.upm > 0:
        fm_span = (fm.max_y - fm.min_y) / fm.upm
        cluster_spans = [...]
        
        if cluster_spans:
            avg_cluster_span = sum(cluster_spans) / len(cluster_spans)
            if avg_cluster_span > 0:
                span_ratio = max(fm_span, avg_cluster_span) / min(fm_span, avg_cluster_span)
                # Use max_span_ratio for structural rejection
                if span_ratio > config.max_span_ratio:  # CHANGED: was decorative_span_threshold
                    return False
    
    # ... rest of detection ...
```

4. **File: `cli.py` - Update argument**

```python
# OLD:
parser.add_argument(
    "--decorative-threshold",
    type=float,
    default=1.3,  # Used for both pre-check AND detection
    metavar="RATIO",
    help="Span ratio threshold for decorative variant detection..."
)

# NEW:
parser.add_argument(
    "--decorative-threshold",
    type=float,
    default=1.4,  # CHANGED: Now only for detection, 1.4x instead of 1.3x
    metavar="RATIO",
    help="Span ratio threshold for decorative variant detection (default: 1.4 = 40%% larger). "
         "Fonts with span exceeding this ratio vs core cluster are treated as decorative variants."
)

# Add new hidden argument for pre-check threshold
parser.add_argument(
    "--max-span-ratio",
    type=float,
    default=1.5,
    help=argparse.SUPPRESS  # Hide from --help (internal use)
)
```

**Threshold Zones:**

```
Span Ratio vs Normal Font:
< 1.4x        Normal font (will cluster normally)
1.4x - 1.5x   Decorative zone (detected, still clusters)
1.5x - 2.0x   Large decorative (detected, rejected from clustering)
>= 2.0x       Script zone (if descender-dominant)

Pre-Check (max_span_ratio = 1.5x):
  - Rejects fonts >1.5x from clustering
  - Prevents unrelated fonts from clustering together
  - More lenient than decorative detection (1.4x)

Decorative Detection (decorative_span_threshold = 1.4x):
  - Flags fonts 1.4x+ as decorative candidates
  - More conservative than old 1.3x (fewer false positives)
  - Creates gap with script detection (2.0x)

Script Detection (script_span_threshold = 2.0x):
  - Requires 2.0x+ span AND descender-dominant
  - No overlap with decorative zone (1.4x-2.0x)
  - Most specific detection
```

**Reasoning:**
- Separates pre-check (structural rejection) from type detection
- Creates clear, non-overlapping zones: normal < decorative < script
- Changing decorative from 1.3x to 1.4x creates buffer from pre-check (1.5x)
- More conservative thresholds = fewer false positives
- Pre-check remains lenient (1.5x) to allow some decorative variants to cluster

---

## Problem 4: Confusing Argument Names

### Current State

```python
--target-percent 1.3    # Span as multiple of UPM (1.3 = 130%)
--headroom-ratio 0.25   # Space above caps as percentage (0.25 = 25%)
--max-pull 8.0          # Max percentage adjustment
```

**Issues:**
- "target-percent" is actually a multiplier (1.3x), not a percentage (130%)
- "headroom-ratio" is actually a percentage (25%), not a ratio (0.25)
- Mixed units confuse users
- "max-pull" unclear direction (pull up? pull down?)

### Solution: Unified Percentage Units with Clear Names

**Implementation:**

1. **File: `cli.py` - Rename arguments**

```python
# OLD:
parser.add_argument(
    "--target-percent",
    type=float,
    default=1.3,
    help="Target span as fraction of UPM (default: 1.3)",
)
parser.add_argument(
    "--headroom-ratio",
    type=float,
    default=0.25,
    help="Headroom as percentage of UPM above cap (default: 0.25 = 25%%)",
)
parser.add_argument(
    "--max-pull",
    type=float,
    default=None,
    metavar="PERCENT",
    help="Maximum %% a font can be pulled UP or DOWN by family extremes...",
)

# NEW:
spacing = parser.add_argument_group("vertical spacing (all values as % of UPM)")
spacing.add_argument(
    "--letter-height",
    type=float,
    default=130,  # Store as percentage (130 = 130% = 1.3x UPM)
    metavar="PERCENT",
    help="Target vertical space for letters (default: 130%% of UPM)",
)
spacing.add_argument(
    "--top-margin",
    type=float,
    default=25,  # Store as percentage (25 = 25% = 0.25x UPM)
    metavar="PERCENT",
    help="Extra space above capitals (default: 25%% of UPM)",
)
spacing.add_argument(
    "--max-adjustment",
    type=float,
    default=None,
    metavar="PERCENT",
    help="Maximum %% adjustment by family extremes (default: unlimited). "
         "Fonts exceeding this get individual calculation.",
)
```

2. **File: `config.py` - Update internal representation**

```python
@dataclass
class MetricsConfig:
    """Configuration constants for metrics normalization."""
    
    # OLD:
    # target_percent: float = 1.3      # As multiplier
    # win_buffer: float = 0.02         # As fraction
    # headroom_ratio: float = 0.25     # As fraction
    # max_pull_percent: Optional[float] = None  # As percentage
    
    # NEW: Store internally as fractions (convert from percentage input)
    target_span: float = 1.3           # Internal: as multiplier (1.3x)
    win_buffer: float = 0.02           # Internal: as fraction (0.02)
    top_margin: float = 0.25           # Internal: as fraction (0.25)
    max_adjustment: Optional[float] = None  # Internal: as fraction (0.08 = 8%)
```

3. **File: `cli.py` - Convert percentages to fractions in main()**

```python
def main() -> None:
    # ... parse args ...
    
    # Convert percentage inputs to internal fraction representation
    config = MetricsConfig(
        target_span=args.letter_height / 100.0,  # 130 → 1.3
        top_margin=args.top_margin / 100.0,      # 25 → 0.25
        max_adjustment=args.max_adjustment / 100.0 if args.max_adjustment else None,  # 8 → 0.08
        win_buffer=0.02,
        xheight_softener=0.6,
        adapt_for_xheight=True,
        optical_threshold=0.025,
        safe_mode=False,  # Remove after implementing grouping_mode
        decorative_span_threshold=args.decorative_threshold,
        unicase_threshold=args.unicase_threshold,
        auto_adjust_target=not args.no_auto_adjust,
    )
```

4. **File: `planning.py` - Update all references**

```python
# OLD:
norm_span = norm_typo_asc + abs(norm_desired_desc)
if norm_span < cluster_target:  # cluster_target comes from config.target_percent
    extra_amt = cluster_target - norm_span
    # ...

# NEW:
norm_span = norm_typo_asc + abs(norm_desired_desc)
if norm_span < config.target_span:  # Use config.target_span (internal fraction)
    extra_amt = config.target_span - norm_span
    # ...

# OLD:
baseline_ascender = max_cap_ratio + config.headroom_ratio

# NEW:
baseline_ascender = max_cap_ratio + config.top_margin

# OLD:
if config.max_pull_percent is not None:
    pull_percent = abs((core_asc - solo_asc) / solo_asc * 100)
    if pull_percent > config.max_pull_percent:
        # ...

# NEW:
if config.max_adjustment is not None:
    pull_fraction = abs((core_asc - solo_asc) / solo_asc)
    if pull_fraction > config.max_adjustment:  # Compare fractions
        # ...
```

5. **File: `validation.py` - Update validation and reporting**

```python
# OLD:
if args.target_percent <= 0.5 or args.target_percent > 2.0:
    cs.StatusIndicator("warning").add_message(
        f"target-percent {args.target_percent} unusual..."
    ).emit(console)

# NEW:
if args.letter_height <= 50 or args.letter_height > 200:
    cs.StatusIndicator("warning").add_message(
        f"letter-height {args.letter_height}% unusual (typically 110-150%)"
    ).emit(console)

if args.top_margin < 10:
    cs.StatusIndicator("warning").add_message(
        f"top-margin {args.top_margin}% very low (<10% UPM) - may clip tall glyphs"
    ).emit(console)
elif args.top_margin > 40:
    cs.StatusIndicator("warning").add_message(
        f"top-margin {args.top_margin}% very high (>40% UPM) - may waste vertical space"
    ).emit(console)

# Combined expansion warning
if args.letter_height > 150 and args.top_margin > 30:
    estimated_span = args.letter_height + args.top_margin
    cs.StatusIndicator("warning").add_message(
        f"Combined letter-height ({args.letter_height}%) + top-margin ({args.top_margin}%) "
        f"≈ {estimated_span}% of UPM - extremely loose spacing"
    ).emit(console)
```

**Usage Examples:**

```bash
# OLD (confusing):
python script.py fonts/ --target-percent 1.3 --headroom-ratio 0.25 --max-pull 8.0

# NEW (clear):
python script.py fonts/ --letter-height 130 --top-margin 25 --max-adjustment 8.0

# All percentages, clear meaning:
# --letter-height 130    = 130% of UPM for letter space
# --top-margin 25        = 25% of UPM above capitals
# --max-adjustment 8.0   = Maximum 8% adjustment by family
```

**Reasoning:**
- All user-facing values are percentages (130%, 25%, 8%)
- Internally stored as fractions for calculation (1.3, 0.25, 0.08)
- Clear argument names (`--letter-height`, `--top-margin`, `--max-adjustment`)
- Consistent units eliminate confusion
- Better validation messages with percentage context

---

## Problem 5: Pattern-Based Overrides

### Current State

```python
--unicase-threshold 0.05        # Numeric threshold (5% UPM)
--decorative-threshold 1.3      # Numeric threshold (1.3x span)
--script-* (multiple thresholds)
```

**Issues:**
- Users don't understand what "0.05" or "1.3" means without reading code
- Can't easily say "this specific font is script"
- Tweaking thresholds affects ALL fonts globally
- No way to override failed detection

### Solution: Add Pattern-Based Overrides

**Implementation:**

1. **File: `cli.py` - Add pattern arguments**

```python
detection = parser.add_argument_group("detection overrides")
detection.add_argument(
    "--assume-script",
    action="append",
    metavar="PATTERN",
    help="Treat matching fonts as script style (e.g., '*Script*', '*Swash*', 'FontName-*'). "
         "Supports glob patterns. Can be used multiple times."
)
detection.add_argument(
    "--assume-decorative",
    action="append",
    metavar="PATTERN",
    help="Treat matching fonts as decorative variants (e.g., '*Rough*', '*Shadow*', '*Inline*'). "
         "Supports glob patterns. Can be used multiple times."
)
detection.add_argument(
    "--assume-unicase",
    action="append",
    metavar="PATTERN",
    help="Treat matching fonts as unicase (e.g., '*Unicase*', 'FontName-SC'). "
         "Supports glob patterns. Can be used multiple times."
)

# Keep numeric thresholds but hide from --help (experts only)
detection.add_argument(
    "--unicase-threshold",
    type=float,
    default=0.05,
    help=argparse.SUPPRESS
)
detection.add_argument(
    "--decorative-threshold",
    type=float,
    default=1.4,
    help=argparse.SUPPRESS
)
# (script thresholds already in config, not exposed as args)
```

2. **File: `measurements.py` - Apply pattern overrides**

```python
def measure_fonts(
    filepaths: List[str],
    existing_measures: Optional[List[FontMeasures]] = None,
    unicase_threshold: float = 0.05,
    script_span_threshold: float = 2.0,
    script_asymmetry_ratio: float = 1.2,
    decorative_span_threshold: float = 1.4,
    assume_script: Optional[List[str]] = None,      # NEW
    assume_decorative: Optional[List[str]] = None,  # NEW
    assume_unicase: Optional[List[str]] = None,     # NEW
) -> List[FontMeasures]:
    """Measure fonts and extract family names.
    
    Pattern overrides (assume_*) take precedence over automatic detection.
    """
    import fnmatch
    
    # ... existing measurement code ...
    
    for fp in filepaths:
        # ... existing bounds/metrics measurement ...
        
        filename = Path(fp).name
        
        # Check pattern overrides FIRST (before automatic detection)
        forced_unicase = False
        forced_script = False
        forced_decorative = False
        
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
        
        # 1. Unicase detection
        if forced_unicase:
            fm.is_unicase = True
        else:
            # ... existing automatic unicase detection ...
        
        # 2. Script detection
        if forced_script:
            fm.is_script = True
        else:
            fm.is_script = is_script_font(
                fm,
                script_span_threshold=script_span_threshold,
                script_asymmetry_ratio=script_asymmetry_ratio,
            )
        
        # 3. Decorative detection
        if forced_decorative:
            fm.is_decorative_candidate = True
        else:
            fm.is_decorative_candidate = detect_decorative_standalone(
                fm,
                MetricsConfig(decorative_span_threshold=decorative_span_threshold)
            )
        
        measures.append(fm)
    
    return measures
```

3. **File: `cli.py` - Pass pattern overrides to measure_fonts**

```python
def main() -> None:
    # ... parse args ...
    
    # Measure fonts (with pattern overrides)
    if files:
        cs.StatusIndicator("info").add_message(
            f"Measuring {cs.fmt_count(len(files))} file(s) for bounds and metrics"
        ).emit(console)
        
        try:
            measures = measure_fonts(
                files,
                existing_measures=measures,
                unicase_threshold=config.unicase_threshold,
                script_span_threshold=config.script_span_threshold,
                script_asymmetry_ratio=config.script_asymmetry_ratio,
                decorative_span_threshold=config.decorative_span_threshold,
                assume_script=args.assume_script,      # NEW
                assume_decorative=args.assume_decorative,  # NEW
                assume_unicase=args.assume_unicase,    # NEW
            )
        except KeyboardInterrupt:
            # ... existing interrupt handling ...
```

4. **Add validation and reporting**

```python
# File: validation.py
def validate_args(args: argparse.Namespace) -> None:
    # ... existing validation ...
    
    # Report pattern overrides
    if args.assume_script or args.assume_decorative or args.assume_unicase:
        cs.emit("", console=console)
        cs.StatusIndicator("info").add_message("Detection overrides active:").emit(console)
        
        if args.assume_script:
            for pattern in args.assume_script:
                cs.emit(f"  • Script: {pattern}", console=console)
        
        if args.assume_decorative:
            for pattern in args.assume_decorative:
                cs.emit(f"  • Decorative: {pattern}", console=console)
        
        if args.assume_unicase:
            for pattern in args.assume_unicase:
                cs.emit(f"  • Unicase: {pattern}", console=console)
```

**Usage Examples:**

```bash
# Force specific fonts to be treated as script
python script.py fonts/ --assume-script "*Script*" --assume-script "*Swash*"

# Force rough/shadow variants as decorative
python script.py fonts/ --assume-decorative "*Rough*" --assume-decorative "*Shadow*"

# Force unicase detection for specific family
python script.py fonts/ --assume-unicase "Kalliope*Unicase*"

# Combine multiple overrides
python script.py fonts/ \
    --assume-script "*Script*" \
    --assume-decorative "*Rough*" "*Shadow*" \
    --assume-unicase "*SC"

# Expert mode: tweak thresholds (hidden from --help)
python script.py fonts/ --unicase-threshold 0.08 --decorative-threshold 1.5
```

**Reasoning:**
- Users can say "this font IS script" instead of tweaking global thresholds
- Glob patterns are familiar (*, ?, etc.)
- Overrides apply before automatic detection (clear precedence)
- Numeric thresholds still available for experts (but hidden)
- Better debugging: forced fonts clearly marked in output

---

## Problem 6: Grouping Modifier Scope

### Current State

```python
--ignore-term    # Only applies to --superfamily
--exclude-family # Only applies to --superfamily
--group          # Works with both family and superfamily
```

**Issues:**
- `--ignore-term` only applies to superfamily, not regular family grouping
- No way to ignore terms in family grouping (e.g., ignore "LT" suffix)
- Inconsistent behavior: `--group` works everywhere, `--ignore-term` doesn't

### Solution: Apply Modifiers Based on Active Mode

**Implementation:**

1. **File: `cli.py` - Rename and clarify arguments**

```python
# OLD:
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

# NEW:
grouping_modifiers = parser.add_argument_group(
    "grouping modifiers",
    description="These options modify grouping behavior for --family, --superfamily, and --custom modes"
)
grouping_modifiers.add_argument(
    "--combine",
    action="append",
    metavar="FAMILIES",
    help='Force-merge families: --combine "Font A,Font B,Font C". '
         'Works with all grouping modes. Can be used multiple times.'
)
grouping_modifiers.add_argument(
    "--ignore-prefix",
    action="append",
    metavar="TOKEN",
    help='Ignore token when finding common prefixes: --ignore-prefix Adobe --ignore-prefix LT. '
         'Applies to --superfamily mode only. Can be used multiple times.'
)
grouping_modifiers.add_argument(
    "--exclude",
    action="append",
    metavar="NAME",
    help='Keep family separate from superfamily grouping: --exclude Script --exclude Display. '
         'Applies to --superfamily mode only. Can be used multiple times.'
)
```

2. **File: `grouping.py` - Update group_families logic**

```python
def group_families(args, measures, forced_groups):
    """Group fonts by family or superfamily based on args.grouping_mode.
    
    Args:
        args: Parsed command-line arguments
        measures: List of FontMeasures
        forced_groups: List of forced group merges from --combine
    
    Returns:
        Dict mapping group name to list of FontMeasures
    """
    # Individual mode: each font is its own group
    if args.grouping_mode == "individual":
        cs.StatusIndicator("info").add_message(
            f"Processing {cs.fmt_count(len(measures))} font(s) individually (no grouping)"
        ).emit(console)
        return {Path(fm.path).stem: [fm] for fm in measures}
    
    # Conservative mode: family grouping only (no modifiers apply)
    if args.grouping_mode == "conservative":
        font_infos = [FontInfo(path=fm.path, family_name=fm.family_name) for fm in measures]
        sorter = FontSorter(font_infos)
        groups = sorter.group_by_family(forced_groups=forced_groups)
        
        cs.StatusIndicator("info").add_message(
            f"Found {cs.fmt_count(len(groups))} family group(s) (conservative mode, no clustering)"
        ).emit(console)
        
        path_to_measure = {fm.path: fm for fm in measures}
        return {
            group_name: [path_to_measure[fi.path] for fi in infos]
            for group_name, infos in groups.items()
        }
    
    # Family or Superfamily mode
    font_infos = [FontInfo(path=fm.path, family_name=fm.family_name) for fm in measures]
    sorter = FontSorter(font_infos)
    
    if args.grouping_mode == "superfamily":
        # Superfamily mode: apply ignore-prefix and exclude modifiers
        groups = sorter.group_by_superfamily(
            ignore_terms=set(expand_comma_separated_args(args.ignore_prefix)),
            exclude_families=expand_comma_separated_args(args.exclude),
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
    
    else:  # args.grouping_mode == "family" (default)
        # Family mode: only apply forced groups (combine)
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
```

3. **File: `validation.py` - Add validation**

```python
def validate_args(args: argparse.Namespace) -> None:
    # ... existing validation ...
    
    # Warn if modifiers used with wrong mode
    if args.grouping_mode != "superfamily":
        if args.ignore_prefix:
            cs.StatusIndicator("warning").add_message(
                "--ignore-prefix only applies to --superfamily mode (ignored)"
            ).emit(console)
        if args.exclude:
            cs.StatusIndicator("warning").add_message(
                "--exclude only applies to --superfamily mode (ignored)"
            ).emit(console)
    
    if args.grouping_mode == "individual":
        if args.combine or args.ignore_prefix or args.exclude:
            cs.StatusIndicator("warning").add_message(
                "--individual ignores all grouping modifiers (--combine, --ignore-prefix, --exclude)"
            ).emit(console)
```

**Modifier Application Matrix:**

| Modifier | --individual | --family | --superfamily | --conservative |
|----------|-------------|----------|---------------|----------------|
| `--combine` | ❌ Ignored | ✅ Applied | ✅ Applied | ✅ Applied |
| `--ignore-prefix` | ❌ Ignored | ❌ Ignored | ✅ Applied | ❌ Ignored |
| `--exclude` | ❌ Ignored | ❌ Ignored | ✅ Applied | ❌ Ignored |

**Reasoning:**
- Clear scope: which modifiers apply to which modes
- `--combine` works everywhere (universal override)
- `--ignore-prefix` and `--exclude` only for superfamily (where they make sense)
- Validation warns if modifiers used with wrong mode
- Conservative mode supports `--combine` but not superfamily modifiers

---

## Summary of Changes

### Files Modified

1. **`cli.py`**
   - Replace `--per-font`, `--superfamily`, `--safe-mode` with `--individual`, `--family`, `--superfamily`, `--conservative`
   - Rename `--target-percent` → `--letter-height` (percentage input)
   - Rename `--headroom-ratio` → `--top-margin` (percentage input)
   - Rename `--max-pull` → `--max-adjustment` (percentage input)
   - Rename `--group` → `--combine`
   - Rename `--ignore-term` → `--ignore-prefix`
   - Rename `--exclude-family` → `--exclude`
   - Add `--assume-script`, `--assume-decorative`, `--assume-unicase` pattern overrides
   - Hide numeric thresholds from `--help` (experts only)
   - Convert percentage inputs to fractions for config
   - Pass pattern overrides to `measure_fonts()`

2. **`config.py`**
   - Remove `safe_mode` field (replaced by `grouping_mode` in args)
   - Rename `target_percent` → `target_span` (internal fraction)
   - Rename `headroom_ratio` → `top_margin` (internal fraction)
   - Rename `max_pull_percent` → `max_adjustment` (internal fraction)
   - Add `max_span_ratio = 1.5` (pre-check threshold, separate from decorative)
   - Change `decorative_span_threshold` from 1.3 to 1.4 (create gap with script)

3. **`measurements.py`**
   - Add `detect_decorative_standalone()` function
   - Add `assume_script`, `assume_decorative`, `assume_unicase` parameters to `measure_fonts()`
   - Apply pattern overrides before automatic detection
   - Set `is_decorative_candidate` flag during measurement
   - Add `decorative_span_threshold` parameter

4. **`models.py`**
   - Add `is_decorative_candidate: bool = False` field
   - Keep `is_decorative_outlier` (confirmed by clustering)

5. **`clustering.py`**
   - Update `compute_optical_similarity()` to use `max_span_ratio` instead of `decorative_span_threshold`
   - Update `detect_decorative_outlier()` to refine `is_decorative_candidate` using cluster context
   - Update `detect_script_font()` to refine measurement flag using cluster context
   - Update `cluster_group_helper()` to confirm/reject detection flags

6. **`planning.py`**
   - Update all references from `target_percent` → `target_span`
   - Update all references from `headroom_ratio` → `top_margin`
   - Update all references from `max_pull_percent` → `max_adjustment`
   - Update comparison logic (percentages → fractions)
   - Remove `safe_mode` checks (replaced by `grouping_mode`)

7. **`grouping.py`**
   - Update `group_families()` to handle `grouping_mode` (individual/family/superfamily/conservative)
   - Apply modifiers based on active mode
   - Update status messages to reflect mode

8. **`validation.py`**
   - Update argument validation for new names
   - Update percentage ranges (130% vs 1.3x)
   - Add modifier scope validation
   - Add pattern override reporting

### Backward Compatibility

**Breaking Changes:**
- `--per-font` → `--individual` (semantic change: old flag name invalid)
- `--safe-mode` → `--conservative` (semantic change: old flag name invalid)
- `--target-percent` → `--letter-height` (value changes: 1.3 → 130)
- `--headroom-ratio` → `--top-margin` (value changes: 0.25 → 25)
- `--max-pull` → `--max-adjustment` (name change only, values unchanged)
- `--group` → `--combine` (name change only)
- `--ignore-term` → `--ignore-prefix` (name change only)
- `--exclude-family` → `--exclude` (name change only)

**Migration Guide:**
```bash
# OLD:
python script.py fonts/ --per-font --target-percent 1.3 --headroom-ratio 0.25 --max-pull 8.0

# NEW:
python script.py fonts/ --individual --letter-height 130 --top-margin 25 --max-adjustment 8.0

# OLD:
python script.py fonts/ --superfamily --ignore-term Adobe --exclude-family Script --group "A,B"

# NEW:
python script.py fonts/ --superfamily --ignore-prefix Adobe --exclude Script --combine "A,B"

# OLD:
python script.py fonts/ --safe-mode

# NEW:
python script.py fonts/ --conservative
```

### Testing Strategy

1. **Unit Tests (if applicable)**
   - Test detection functions with pattern overrides
   - Test percentage → fraction conversion
   - Test modifier scope validation

2. **Integration Tests**
   - Test each grouping mode (individual/family/superfamily/conservative)
   - Test pattern overrides with real fonts
   - Test threshold separation (max_span_ratio vs decorative_span_threshold)
   - Test modifier application per mode

3. **Regression Tests**
   - Test with existing font collections
   - Compare output before/after changes
   - Verify clustering results unchanged (except decorative threshold 1.3→1.4)

---

## Implementation Priority

### Phase 1: Critical (Mode Confusion)
1. Implement `--individual`/`--family`/`--superfamily`/`--conservative` modes
2. Remove `--safe-mode` and `--per-font` flags
3. Update validation and help text

### Phase 2: Clarity (Naming)
1. Rename percentage arguments (`--letter-height`, `--top-margin`, `--max-adjustment`)
2. Convert percentage inputs to fractions
3. Update all internal references
4. Update validation ranges

### Phase 3: Detection (Timing & Thresholds)
1. Add `detect_decorative_standalone()` in measurement
2. Add pattern override arguments (`--assume-*`)
3. Separate `max_span_ratio` from `decorative_span_threshold`
4. Change decorative threshold from 1.3 to 1.4
5. Update clustering to refine detection flags

### Phase 4: Modifiers (Scope)
1. Rename grouping modifiers (`--combine`, `--ignore-prefix`, `--exclude`)
2. Update `group_families()` to apply modifiers based on mode
3. Add modifier scope validation

### Phase 5: Documentation
1. Update README with new arguments
2. Update help text with examples
3. Add migration guide
4. Document threshold zones (normal/decorative/script)

---

## Expected Benefits

1. **Clearer User Intent**
   - Four explicit modes replace overlapping flags
   - Users know exactly what each mode does
   - No confusion about "safe" vs "per-font"

2. **Predictable Detection**
   - All detection happens during measurement (single phase)
   - Clustering refines, doesn't re-detect
   - Easier debugging (check measurement output)

3. **Better Threshold Control**
   - Separate pre-check (1.5x) from decorative detection (1.4x)
   - Clear zones: normal < decorative (1.4-2.0x) < script (2.0x+)
   - No overlap confusion

4. **Consistent Units**
   - All user-facing values are percentages
   - No mixing multipliers and fractions
   - Better validation messages

5. **Flexible Overrides**
   - Pattern-based overrides for specific fonts
   - Numeric thresholds still available for experts
   - Clearer precedence (override > automatic)

6. **Scoped Modifiers**
   - Clear which modifiers apply to which modes
   - Validation warns about wrong usage
   - Consistent behavior