# Checkpoint Measurement Audit

## Current State

### Measurements Collected (from `measurements.py`)

**Raw Font Data:**
- ✅ `upm` - Units per em (from font head table)
- ✅ `family_name` - Font family name (from name table, ID 16 or 1)
- ✅ `min_y` - Minimum Y coordinate (overall font bounds)
- ✅ `max_y` - Maximum Y coordinate (overall font bounds)
- ✅ `cap_height` - Cap height (from OS/2.sCapHeight or measured from 'H'/'I')
- ✅ `cap_optical` - Optical cap height (median of A-Z yMax)
- ✅ `x_height` - X-height (from OS/2.sxHeight or measured from 'x')
- ✅ `ascender_max` - Maximum ascender (from lowercase b, d, h, k, l)
- ✅ `descender_min` - Minimum descender (from lowercase g, j, p, q, y)

**Detection Results:**
- ✅ `is_unicase` - Detected during measurement (x-height ≈ cap-height)
- ✅ `is_script` - Detected during measurement (large span + descender-dominant)

### Measurements Saved to Checkpoint (from `checkpoints.py` lines 78-91)

All measured fields are saved:
- ✅ `path`
- ✅ `upm`
- ✅ `family_name`
- ✅ `min_y`
- ✅ `max_y`
- ✅ `cap_height`
- ✅ `cap_optical`
- ✅ `x_height`
- ✅ `ascender_max`
- ✅ `descender_min`
- ✅ `is_unicase`
- ✅ `is_script` (detected during measurement, refined during clustering)

### Fields NOT Saved (by design)

**Computed Targets** (recalculated each run):
- ❌ `target_typo_asc`
- ❌ `target_typo_desc`
- ❌ `target_win_asc`
- ❌ `target_win_desc`

**Clustering Results** (recalculated each run):
- ❌ `cluster_id`
- ❌ `is_decorative_outlier` (set during clustering)

**Family Context** (recalculated each run):
- ❌ `family_upm_majority`

---

## Issues Found

### 1. `is_script` Inconsistency ✅ **FIXED**
- **Problem**: `is_script` was saved to checkpoint but was NOT set during measurement phase
- **Previous behavior**: Set during clustering/planning phase (`clustering.py` line 325, `planning.py` line 910)
- **Solution**: Added standalone script detection to measurement phase (`measurements.py`)
  - New function `is_script_font()` detects script fonts based on span and descender-dominance
  - Called during `measure_fonts()` to set `is_script` flag before checkpoint save
  - Clustering still refines detection by comparing to cluster average
- **Status**: ✅ **Resolved** - `is_script` is now set during measurement and saved to checkpoint

### 2. `is_decorative_outlier` Not Saved
- **Current**: Not saved to checkpoint (set during clustering)
- **Impact**: Decorative detection is recalculated each run (may change if config changes)
- **Status**: ✅ **Correct** - This is a clustering result, not a raw measurement

---

## Potential Additional Measurements to Consider

### OS/2 Table Metrics (Currently Not Captured)
These are font metadata that don't change through normalization:

1. **`sTypoAscender` / `sTypoDescender`** (OS/2)
   - Current typographic metrics
   - **Use case**: Could help detect fonts that are already normalized
   - **Value**: Medium - useful for validation/comparison

2. **`usWinAscent` / `usWinDescent`** (OS/2)
   - Current Windows metrics
   - **Use case**: Could help detect fonts that are already normalized
   - **Value**: Medium - useful for validation/comparison

3. **`sxHeight`** (OS/2)
   - **Status**: Already captured as `x_height` (prefers OS/2, falls back to glyph measurement)
   - ✅ **Already handled**

4. **`sCapHeight`** (OS/2)
   - **Status**: Already captured as `cap_height` (prefers OS/2, falls back to glyph measurement)
   - ✅ **Already handled**

5. **`lineGap`** (hhea)
   - Line spacing metric
   - **Use case**: Could be useful for normalization validation
   - **Value**: Low - not currently used in normalization logic

6. **`ascender` / `descender`** (hhea)
   - **Status**: Similar to OS/2 metrics, but hhea is less authoritative
   - **Value**: Low - OS/2 is preferred

### Glyph-Based Measurements (Currently Not Captured)

1. **X-height min/max range** (from robust measurement)
   - Currently: Only median is used for detection
   - **Use case**: Could help with overshoot tolerance calculation
   - **Value**: Low - already calculated during measurement but not stored

2. **Cap-height min/max range** (from robust measurement)
   - Currently: Only median is used for detection
   - **Use case**: Could help with overshoot tolerance calculation
   - **Value**: Low - already calculated during measurement but not stored

3. **Glyph count** or **character coverage**
   - **Use case**: Could help identify incomplete fonts
   - **Value**: Very Low - not relevant to normalization

### Font Metadata (Currently Not Captured)

1. **Font version** (name table)
   - **Use case**: Could help detect if font was modified
   - **Value**: Low - not relevant to normalization

2. **Font style** (name table, ID 2)
   - **Status**: Already captured as part of `family_name` grouping
   - ✅ **Already handled via grouping**

3. **Font format** (TrueType vs OpenType)
   - **Use case**: Could help with format-specific handling
   - **Value**: Very Low - already handled transparently

---

## Recommendations

### High Priority

1. **Fix `is_script` inconsistency** ✅ **COMPLETED**
   - **Solution**: Added standalone script detection to measurement phase
   - **Implementation**: 
     - Created `is_script_font()` function in `measurements.py`
     - Calls script detection during `measure_fonts()` 
     - Clustering still refines detection by comparing to cluster average
   - **Result**: `is_script` is now consistently set during measurement and saved to checkpoint

### Medium Priority

2. **Consider saving original OS/2 metrics** (optional) ⚠️ **NOT RECOMMENDED**
   - **User preference**: Prefer glyph-based measurements over metadata
   - **Current approach**: Code already prioritizes glyph measurements:
     - `cap_height`: Prefers OS/2.sCapHeight, but falls back to glyph measurement
     - `x_height`: Prefers OS/2.sxHeight, but falls back to glyph measurement  
     - `ascender_max`, `descender_min`: Always from glyph measurements
     - `min_y`, `max_y`: Always from glyph measurements (overall bounds)
   - **Recommendation**: **Skip** saving original OS/2 metrics - not needed for calculations, and user prefers glyph-based data
   - **Rationale**: Glyph measurements are more accurate and reflect actual font geometry

### Low Priority

3. **Document measurement vs. detection distinction**
   - Clarify in code comments what's a raw measurement vs. a detection result
   - Helps future maintainers understand checkpoint design

---

## Summary

**Current Status**: ✅ **Mostly Complete**

- All raw measurements are captured and saved
- All detection results that are set during measurement are saved
- Computed targets and clustering results are correctly excluded

**Main Issue**: `is_script` is saved but not set during measurement, creating inconsistency with `is_unicase` which is set during measurement.

**Recommendation**: Move script detection to measurement phase for consistency, or remove it from checkpoint if it's truly a clustering result.
