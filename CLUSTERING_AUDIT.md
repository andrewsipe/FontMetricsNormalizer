# Clustering & Grouping Strategies Audit

## Overview

The script uses a **multi-level grouping strategy** that operates in distinct phases:

1. **Family/Superfamily Grouping** (Level 0) - Groups fonts by name/metadata
2. **Optical Clustering** (Level 1) - Groups similar fonts within families
3. **Special Type Detection** (Level 2) - Identifies script, decorative, unicase fonts
4. **Normalization Planning** (Level 3) - Applies metrics based on groupings

Each level can be controlled, bypassed, or modified through CLI arguments and config options.

---

## Level 0: Family/Superfamily Grouping

**Purpose:** Group fonts into families or superfamilies before any clustering occurs.

**Location:** `grouping.py` → `group_families()`

### Strategies:

#### 1. **Per-Font Mode** (`--per-font`)
- **Type:** Direct control (bypasses all grouping)
- **Behavior:** Each font processed individually, no family grouping
- **Result:** Each font gets its own "family" (using filename stem)
- **Impact:** No family normalization, each font normalized independently
- **When to use:** When fonts shouldn't be grouped together

#### 2. **Family Grouping** (default)
- **Type:** Automatic (based on font metadata)
- **Behavior:** Groups fonts by family name (from name table)
- **Method:** Uses `FontSorter.group_by_family()` from FontCore
- **Result:** Fonts with same family name are grouped together
- **Impact:** Normalization applied per family

#### 3. **Superfamily Grouping** (`--superfamily`)
- **Type:** Automatic with manual control
- **Behavior:** Groups fonts by common prefix (e.g., "Roboto" + "Roboto Condensed")
- **Method:** Uses `FontSorter.group_by_superfamily()` from FontCore
- **Result:** Multiple related families treated as one group
- **Impact:** Normalization applied across superfamily
- **Control options:**
  - `--ignore-term` / `-it`: Ignore specific terms when grouping
  - `--exclude-family` / `-ef`: Exclude families from superfamily grouping

#### 4. **Forced Grouping** (`--group` / `-g`)
- **Type:** Direct control (manual override)
- **Behavior:** Manually merge specific families together
- **Format:** `--group "Family A,Family B"` or `--group "Family A,Family B,Family C"`
- **Result:** Specified families treated as one group
- **Impact:** Overrides automatic grouping for specified families
- **Works with:** Both family and superfamily modes

### Flow:
```
All Fonts
  ↓
[--per-font?] → Yes → Individual groups (by filename)
  ↓ No
[--superfamily?] → Yes → Superfamily grouping
  ↓ No
Family grouping (default)
  ↓
[--group applied] → Forced merges
  ↓
Grouped Families
```

### Arguments:
- `--per-font` - Bypasses all grouping
- `--superfamily` - Enables superfamily grouping
- `--ignore-term` / `-it` - Terms to ignore in superfamily grouping
- `--exclude-family` / `-ef` - Families to exclude from superfamily grouping
- `--group` / `-g` - Force merge specific families

---

## Level 1: Optical Clustering

**Purpose:** Within each family, group fonts with similar optical characteristics.

**Location:** `clustering.py` → `detect_optical_clusters()`

### Strategy:

#### **Optical Similarity Clustering** (default, unless `--safe-mode`)
- **Type:** Automatic (always enabled if not in safe mode)
- **Method:** Union-Find algorithm based on optical similarity
- **Similarity Criteria** (`compute_optical_similarity()`):
  1. **Primary:** Cap height ratio (normalized) - must match within `optical_threshold` (default: 2.5% UPM)
  2. **Secondary:** X-height ratio - more lenient (2x threshold)
  3. **Tertiary:** Descender ratio - somewhat lenient (1.5x threshold)
  4. **Pre-check:** Span ratio - rejects if > `decorative_span_threshold` (default: 1.3x)
- **Result:** Creates "core clusters" of optically identical fonts
- **Impact:** Fonts in same cluster get identical normalized metrics

#### **Safe Mode** (`--safe-mode` / `-sm`)
- **Type:** Direct control (bypasses clustering)
- **Behavior:** Skips all optical clustering, uses conservative bbox approach
- **Method:** `plan_safe_metrics()` - per-font calculation using bbox bounds
- **Result:** Each font normalized independently (no clustering)
- **Impact:** No family normalization, no outlier detection
- **When to use:** When clustering causes issues or for maximum safety

### Clustering Process:
```
Family Group
  ↓
[--safe-mode?] → Yes → Skip clustering, use plan_safe_metrics()
  ↓ No
Build similarity graph (cap height + x-height + descender)
  ↓
Union-Find clustering
  ↓
Core clusters (size > 1) + Singletons
  ↓
Main cluster (largest core cluster)
```

### Arguments:
- `--safe-mode` / `-sm` - Bypasses clustering
- `optical_threshold` (config, hardcoded: 0.025 = 2.5% UPM) - Similarity threshold

---

## Level 2: Special Type Detection

**Purpose:** Identify special font types (script, decorative, unicase) that need special handling.

**Location:** `clustering.py` → Detection functions, `measurements.py` → `is_unicase()`, `is_script_font()`

### Detection Strategies:

#### 1. **Unicase Detection** (during measurement)
- **Type:** Automatic (during measurement phase)
- **Location:** `measurements.py` → `is_unicase()`
- **Criteria:**
  - X-height ≈ cap-height (within `unicase_threshold`, default: 5% UPM)
  - Cap not too tall (rejects if cap > x-height + 10% UPM)
  - Both metrics in reasonable range (0.45-0.65 of UPM)
- **Control:** `--unicase-threshold` (default: 0.05)
- **Impact:** 
  - Mixed families: Unicase fonts treated as decorative outliers (inherit baseline)
  - Pure unicase families: Clustered normally
- **When set:** During `measure_fonts()` → saved to checkpoint

#### 2. **Script Detection** (during measurement + clustering refinement)
- **Type:** Automatic (two-phase: measurement + clustering)
- **Location:** 
  - `measurements.py` → `is_script_font()` (standalone detection)
  - `clustering.py` → `detect_script_font()` (cluster-based refinement)
- **Criteria:**
  - **Standalone (measurement):**
    - Large span (≥ 1.5x UPM)
    - Descender-dominant (descender ratio > ascender ratio × `script_asymmetry_ratio`)
  - **Cluster-based (clustering):**
    - Core metrics match cluster (same letter body)
    - Span ≥ 2x cluster average (`script_span_threshold`, default: 2.0)
    - Descender-dominant (same as standalone)
- **Control:**
  - `script_span_threshold` (config, default: 2.0) - Span ratio threshold
  - `script_asymmetry_ratio` (config, default: 1.2) - Descender dominance
- **Impact:** Script fonts inherit typo metrics from main cluster, expand win bounds with larger buffer
- **When set:** During `measure_fonts()` (initial), refined during clustering

#### 3. **Decorative Outlier Detection** (during clustering)
- **Type:** Automatic (during clustering phase)
- **Location:** `clustering.py` → `detect_decorative_outlier()`
- **Criteria:**
  - Core metrics match cluster (same letter body)
  - Span ratio > `decorative_span_threshold` (default: 1.3x) vs cluster average
  - Bounds inflation > 15% (max or min) vs cluster average
- **Control:** `--decorative-threshold` (default: 1.3)
- **Impact:** Decorative fonts inherit typo metrics from main cluster, expand win bounds
- **When set:** During clustering (singletons checked against main cluster)

### Detection Flow:
```
Fonts in Family
  ↓
[Measurement Phase]
  ├─→ is_unicase() → Sets fm.is_unicase
  └─→ is_script_font() → Sets fm.is_script (standalone)
  ↓
[Clustering Phase]
  ├─→ Main cluster identified
  ├─→ Singletons checked:
  │   ├─→ detect_script_font() → Refines is_script (cluster-based)
  │   ├─→ detect_decorative_outlier() → Sets is_decorative_outlier
  │   └─→ True outliers → Single-font clusters
  └─→ Unicase handling:
      ├─→ Mixed (unicase + traditional) → Unicase as decorative
      └─→ Pure unicase → Clustered normally
```

### Arguments:
- `--unicase-threshold` - Unicase detection threshold (default: 0.05 = 5% UPM)
- `--decorative-threshold` - Decorative span ratio threshold (default: 1.3 = 30% larger)
- `script_span_threshold` (config, default: 2.0) - Script span ratio threshold
- `script_asymmetry_ratio` (config, default: 1.2) - Script descender dominance

---

## Level 3: Normalization Planning

**Purpose:** Apply normalization metrics based on groupings and detections.

**Location:** `planning.py` → `build_plans()`

### Planning Strategies:

#### 1. **Safe Mode Planning** (`plan_safe_metrics()`)
- **Trigger:** `--safe-mode`
- **Behavior:** Per-font calculation using bbox bounds
- **No clustering, no family normalization**

#### 2. **Identical Metrics Planning** (`plan_identical_metrics()`)
- **Trigger:** Fonts in core cluster (size > 1)
- **Behavior:** All fonts in cluster get identical normalized metrics
- **Uses:** Family-normalized ascender, centered descender

#### 3. **Adaptive Metrics Planning** (`plan_adaptive_metrics()`)
- **Trigger:** Single-font clusters (true outliers)
- **Behavior:** Per-font calculation with family context
- **Uses:** Family-normalized ascender, per-font descender (centered)

#### 4. **Decorative Inheritance**
- **Trigger:** Fonts marked as `is_decorative_outlier`
- **Behavior:** Inherit typo metrics from main cluster
- **Uses:** Main cluster's normalized typo ascender/descender

#### 5. **Script Inheritance**
- **Trigger:** Fonts marked as `is_script`
- **Behavior:** Inherit typo metrics from main cluster, expand win bounds
- **Uses:** Main cluster's normalized typo metrics + larger win buffer

#### 6. **Unicase Inheritance** (mixed families)
- **Trigger:** Unicase fonts in mixed families
- **Behavior:** Inherit typo metrics from traditional fonts
- **Uses:** Traditional fonts' normalized typo metrics

### Planning Flow:
```
Family Group
  ↓
[--safe-mode?] → Yes → plan_safe_metrics() (per font)
  ↓ No
[Clustering Results]
  ├─→ Core clusters → plan_identical_metrics()
  ├─→ True outliers → plan_adaptive_metrics()
  ├─→ Decorative outliers → Inherit from main cluster
  ├─→ Script outliers → Inherit from main cluster + larger buffer
  └─→ Unicase (mixed) → Inherit from traditional fonts
```

---

## Interaction Matrix

### How Strategies Interact:

| Strategy | Affects Family Grouping | Affects Clustering | Affects Detection | Affects Planning |
|----------|------------------------|-------------------|-------------------|-----------------|
| `--per-font` | ✅ Bypasses | ✅ Bypasses | ⚠️ Partial (unicase/script still detected) | ✅ Per-font planning |
| `--superfamily` | ✅ Changes grouping | ❌ No effect | ❌ No effect | ❌ No effect |
| `--group` | ✅ Forces merges | ❌ No effect | ❌ No effect | ❌ No effect |
| `--safe-mode` | ❌ No effect | ✅ Bypasses | ✅ Bypasses | ✅ Changes planning |
| `--unicase-threshold` | ❌ No effect | ⚠️ Affects unicase handling | ✅ Controls detection | ⚠️ Affects inheritance |
| `--decorative-threshold` | ❌ No effect | ✅ Controls detection | ✅ Controls detection | ⚠️ Affects inheritance |
| `optical_threshold` | ❌ No effect | ✅ Controls similarity | ❌ No effect | ⚠️ Affects cluster formation |
| `script_span_threshold` | ❌ No effect | ⚠️ Used in detection | ✅ Controls detection | ⚠️ Affects inheritance |

### Dependency Graph:

```
Family Grouping (Level 0)
  ├─→ --per-font → Bypasses everything
  ├─→ --superfamily → Changes input to clustering
  └─→ --group → Forces merges
       ↓
Optical Clustering (Level 1)
  ├─→ --safe-mode → Bypasses clustering
  ├─→ optical_threshold → Controls similarity
  └─→ decorative_span_threshold → Used in similarity check
       ↓
Special Type Detection (Level 2)
  ├─→ --unicase-threshold → Controls unicase detection
  ├─→ --decorative-threshold → Controls decorative detection
  ├─→ script_span_threshold → Controls script detection
  └─→ script_asymmetry_ratio → Controls script detection
       ↓
Normalization Planning (Level 3)
  └─→ All above affect what planning strategy is used
```

---

## Direct vs Indirect vs Automatic

### Direct Control (User explicitly controls):
- `--per-font` - Direct bypass of all grouping
- `--safe-mode` - Direct bypass of clustering
- `--group` - Direct control of family merges
- `--superfamily` - Direct control of grouping mode

### Indirect Control (User sets parameters, script decides):
- `--unicase-threshold` - Sets threshold, script detects automatically
- `--decorative-threshold` - Sets threshold, script detects automatically
- `optical_threshold` - Sets threshold, script clusters automatically
- `script_span_threshold` - Sets threshold, script detects automatically

### Automatic (No user control, always happens):
- Family grouping (unless `--per-font`)
- Optical clustering (unless `--safe-mode` or `--per-font`)
- Script detection (standalone phase during measurement)
- Unicase detection (during measurement)
- Decorative detection (during clustering)

---

## Potential Issues & Observations

### 1. **Multiple Detection Phases**
- Script detection happens twice: measurement (standalone) + clustering (refined)
- Could be confusing - which one takes precedence?
- **Current:** Clustering refines measurement detection (OR logic)

### 2. **Unicase Special Handling**
- Unicase detection is automatic during measurement
- But handling differs based on family composition (mixed vs pure)
- This happens in clustering phase, not measurement phase
- **Impact:** Unicase flag set early, but treatment decided later

### 3. **Decorative vs Script Overlap**
- Both check span ratio vs cluster
- Script threshold (2.0x) > Decorative threshold (1.3x)
- Script checked first (more specific)
- **Potential issue:** Fonts between 1.3x-2.0x might be misclassified

### 4. **Safe Mode Bypass**
- `--safe-mode` bypasses clustering but still does family grouping
- `--per-font` bypasses both family grouping AND clustering
- **Inconsistency:** Safe mode still groups by family, per-font doesn't

### 5. **Forced Groups vs Clustering**
- `--group` forces families together, but clustering still happens within
- Could cluster fonts from different families together if forced
- **Impact:** Forced groups get clustered as one family

### 6. **Threshold Interactions**
- `optical_threshold` (2.5% UPM) used for similarity
- `decorative_span_threshold` (1.3x) used in similarity pre-check
- **Potential conflict:** Fonts with 1.3x span might be rejected from clustering entirely

---

## Recommendations for Better Approach

### Potential Improvements:

1. **Unify Detection Phases**
   - Move all detection to measurement phase (like unicase/script)
   - Clustering phase only refines, doesn't re-detect
   - Makes behavior more predictable

2. **Clarify Safe Mode**
   - Should `--safe-mode` also bypass family grouping?
   - Or rename to `--no-clustering` to be more explicit?

3. **Threshold Consistency**
   - Document how thresholds interact
   - Consider unified threshold system
   - Make decorative/script thresholds more distinct

4. **Grouping Strategy Options**
   - Add explicit `--family-only` mode (no superfamily)
   - Add `--no-grouping` mode (per-font but with clustering?)
   - Clarify relationship between grouping modes

5. **Detection Order**
   - Document explicit detection order
   - Make script vs decorative decision clearer
   - Consider priority system for overlapping detections

---

## Summary

The script uses a **hierarchical grouping system** with three main levels:

1. **Family/Superfamily Grouping** - Groups fonts by name/metadata
2. **Optical Clustering** - Groups similar fonts within families  
3. **Special Type Detection** - Identifies fonts needing special handling

Each level can be controlled, bypassed, or modified through CLI arguments. The system is flexible but complex, with multiple detection phases and threshold interactions that could benefit from simplification and clearer documentation.
