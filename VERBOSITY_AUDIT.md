# Console Output Verbosity Audit

## Current Verbosity Levels

- **BRIEF** (default, no `-v` flag): Essential information only
- **VERBOSE** (`-v` flag): Detailed diagnostic information
- **DEBUG** (`-vv` flag): Step-by-step processing flow and internal details

---

## ALWAYS SHOWN (Regardless of Verbosity Level)

### CLI Module (`cli.py`)

1. **Checkpoint Status** (lines 233-305)
   - "Checkpoint found with measurements for all X fonts"
   - "Using checkpoint would skip all measurement"
   - "X new file(s) would still need measurement"
   - "Reusing X measurements, will measure Y new file(s)"
   - "Remeasuring all X fonts"

2. **File Collection** (line 318)
   - "Measuring X file(s) for bounds and metrics"

3. **Grouping Results** (lines 77-78, 91-92)
   - "Found X family group(s)" or "Found X superfamily group(s)"
   - Forced group merges (lines 85-88, 98-101)

4. **Family Normalization Impact** (`validation.py` lines 183-190)
   - "Family: X — Minimal/Moderate/Major Normalization"
   - "UPM scaled by ~X%, preserving/increasing/reducing vertical height | linegap set to 0"

5. **Font Update Reports** (`planning.py` lines 728, 822-873)
   - "UPDATED" or "UNCHANGED" status for each font
   - OS/2 and hhea table changes with before/after values
   - Vertical span changes

6. **Success Summary** (`cli.py` line 384)
   - "Processing Completed! updated: X | unchanged: Y | errors: Z"

7. **Error Messages** (various)
   - All error and warning messages are always shown

### Planning Module (`planning.py`)

1. **Ascender Exceedance Warning** (line 290)
   - "Ascenders exceed baseline by X% UPM - using actual ascender height"

2. **Unicase Detection** (lines 922-942)
   - "Unicase detected: X unicase font(s) mixed with Y traditional font(s)"
   - "Pure unicase family: X font(s)"

3. **Core Cluster Info** (line 1245-1255)
   - "Core cluster: X fonts | UPM: Y" (cap height details only at VERBOSE)

4. **Script Detection** (lines 1081-1092)
   - "Script detector: X font(s) (span ratio: Yx vs cluster avg)" (ratio details at VERBOSE)

5. **Decorative Detection** (lines 1102-1138)
   - "Decorative detector: X font(s) (expanded bounds, core metrics match)"
   - Font names list (always shown)
   - Span ratios (only at VERBOSE)

6. **Unicase Detector** (lines 1150-1165)
   - "Unicase detector: X font(s) (x-height ≈ cap-height) - separated for baseline alignment"
   - Font names list (only at VERBOSE)

7. **Decorative Inheritance Summary** (lines 1441-1454)
   - "Decorative fonts inheriting typo metrics from main cluster"
   - Main cluster fonts list
   - Decorative fonts list
   - Inherited typo ascender/descender values

8. **Cluster Normalization Warnings** (lines 483-493)
   - Inconsistency warnings for ascender/descender ratio ranges

---

## VERBOSE LEVEL (`-v` flag)

### Planning Module (`planning.py`)

1. **Unicase Detection Details** (line 927)
   - List of unicase font names (truncated to 5)

2. **Script Detection Details** (lines 1075-1091)
   - Average span ratio calculation
   - List of script font names (truncated to 5)

3. **Decorative Detection Details** (lines 1111-1137)
   - Individual span ratios for each decorative font vs cluster average

4. **Unicase Detector Details** (line 1155)
   - List of unicase font names (truncated to 5)

5. **Baseline Priority Calculation** (lines 1180-1198)
   - "Baseline priority: Target baseline descender calculated"
   - Main cluster font names
   - Target baseline descender (normalized and scaled to UPM)

6. **Core Cluster Cap Height** (line 1249)
   - Cap height ratio added to cluster message: "(cap height ≈ X/Y = Z)"

7. **Script Font Processing** (line 1354)
   - "Script fonts will inherit core typo metrics and expand win bounds with Xx buffer"

8. **Unicase Inheritance** (lines 1492-1497)
   - Individual font inheritance: "Inherited ascender X → Y, descender → Z"

9. **Regular Decorative Inheritance** (lines 1537-1552)
   - Individual font inheritance with before/after values
   - "Already matches main cluster metrics" if no change

10. **Unicase Alignment Report** (line 1513)
    - "Unicase cap (X) aligns with traditional x-height (Y) - baseline preserved" (when alignment_diff < 3.0%)

---

## DEBUG LEVEL (`-vv` flag)

### Planning Module (`planning.py`)

1. **Ascender Calculation Details** (line 359)
   - "Cluster: Applying baseline alignment shift"
   - Centered descender → Target baseline (shift amount)

2. **Cluster Detection Start** (line 536)
   - "DEBUG: Detecting optical clusters from X font(s)"

3. **Cluster Detection Start (Level 2)** (line 959)
   - "DEBUG: Detecting optical clusters from X font(s)"

4. **Max Pull Limit Details** (line 1025)
   - Individual font pull percentages when exceeding max_pull_percent

5. **Level 4: Cluster Processing** (line 1362)
   - "DEBUG: Level 4: Processing X cluster(s)"

6. **Decorative in Cluster Warning** (line 1377)
   - "WARNING: Decorative fonts found in clusters (should not happen): [names]"

7. **Level 5: Finalization** (line 1415)
   - "DEBUG: Level 5: Finalizing metrics for X font(s)"

8. **Level 7: Decorative Processing** (line 1425)
   - "DEBUG: Level 7: Processing X decorative outlier(s)"

9. **Adaptive Metrics Baseline Shift** (line 536)
   - Individual font baseline alignment shifts: "X → Y (Z units)"

---

## Issues & Inconsistencies Found

### 1. **Mixed Verbosity Gating**
   - Some messages show partial info at BRIEF, full details at VERBOSE (e.g., core cluster cap height)
   - Some messages always show full details (e.g., decorative font names list)
   - Some messages only show at VERBOSE but should maybe be at BRIEF (e.g., baseline priority calculation)

### 2. **Redundant Output**
   - Decorative inheritance shows summary ALWAYS, but individual font details only at VERBOSE
   - This creates confusion: you see "inheriting from main cluster" but not which fonts or what values

### 3. **Missing Context**
   - Baseline priority calculation only shows at VERBOSE, but it's critical for understanding `--priority-baseline` behavior
   - Individual font baseline shifts only show at DEBUG, making it hard to see what's happening

### 4. **Inconsistent Detail Levels**
   - Script detection: shows count and ratio type at BRIEF, but ratio value only at VERBOSE
   - Decorative detection: shows font names always, but span ratios only at VERBOSE
   - Unicase: shows count always, but names only at VERBOSE

### 5. **DEBUG vs VERBOSE Confusion**
   - Some DEBUG messages are actually useful for understanding behavior (baseline shifts)
   - Some VERBOSE messages are more like DEBUG (step-by-step processing)

### 6. **Always-Shown Messages That Could Be Gated**
   - Decorative inheritance summary (lines 1441-1454) - very verbose, could be VERBOSE
   - Individual decorative font names in detection (line 1108) - could be VERBOSE
   - Script font names in detection (line 1088) - already gated at VERBOSE ✓

---

## Recommendations for Reorganization

### BRIEF (Default)
- Essential status: what's being processed, what changed
- High-level summaries: cluster counts, detection results (counts only)
- Critical warnings: normalization inconsistencies, errors
- **Remove**: Detailed decorative inheritance summary, individual font names in detection

### VERBOSE (`-v`)
- Diagnostic details: which fonts in which clusters, detection details
- Baseline priority calculations and shifts
- Individual font inheritance details
- Span ratios and metric calculations
- **Add**: Baseline shift details (currently DEBUG)

### DEBUG (`-vv`)
- Step-by-step processing flow (Level 4, 5, 7)
- Internal calculations and intermediate values
- Edge case warnings (decorative in clusters)
- **Keep**: Processing flow, internal state

---

## Specific Changes Needed

1. **Move baseline priority calculation to BRIEF** (or at least show summary)
   - Currently: VERBOSE only
   - Should: Show at BRIEF when `--priority-baseline` is used

2. **Move baseline shift details to VERBOSE**
   - Currently: DEBUG only
   - Should: Show at VERBOSE so users can see what's happening

3. **Gate decorative inheritance summary at VERBOSE**
   - Currently: Always shown
   - Should: Show summary at BRIEF, details at VERBOSE

4. **Gate decorative font names at VERBOSE**
   - Currently: Always shown in detection
   - Should: Show count at BRIEF, names at VERBOSE

5. **Clarify cluster info verbosity**
   - Currently: Cap height ratio only at VERBOSE
   - Should: Keep this, but maybe show UPM info more clearly at BRIEF
