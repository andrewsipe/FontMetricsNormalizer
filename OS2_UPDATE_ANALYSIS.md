# OS/2 Metadata Update Analysis

## Currently Updated OS/2 Fields

### Always Updated (from normalized targets):
- ✅ `sTypoAscender` - from `target_typo_asc` (normalized across family)
- ✅ `sTypoDescender` - from `target_typo_desc` (normalized across family)
- ✅ `usWinAscent` - from `target_win_asc` (normalized across family)
- ✅ `usWinDescent` - from `target_win_desc` (normalized across family)
- ✅ `sTypoLineGap` - set to 0 (standardized)

### Always Updated (with measured values):
- ✅ `sxHeight` - always updated with measured `x_height` (per font)
- ✅ `sCapHeight` - always updated with measured `cap_height` (per font)

### Currently Measured But Not Always Updated:
- `x_height` - measured from glyphs (or OS/2.sxHeight), but only writes to OS/2 if missing
- `cap_height` - measured from glyphs (or OS/2.sCapHeight), but only writes to OS/2 if missing

---

## What Would Be Needed to Always Update sxHeight and sCapHeight?

### Option 1: Update with Measured Values (No Normalization)
**Additional data needed:** ✅ **NONE** - We already have this!

**What we'd do:**
- Always write `x_height` to `OS/2.sxHeight` (not just when missing)
- Always write `cap_height` to `OS/2.sCapHeight` (not just when missing)

**Pros:**
- Ensures OS/2 metadata matches actual glyph measurements
- No additional data collection needed
- Simple change (remove the `if getattr(os2, "sxHeight", 0) in (0, None):` check)

**Cons:**
- Doesn't normalize x-height/cap-height across family (each font keeps its own)
- May create inconsistencies if fonts have slightly different x-heights

**Code change (implemented):**
```python
# Previous (conditional update):
if fm.x_height and fm.x_height > 0:
    if getattr(os2, "sxHeight", 0) in (0, None):
        os2.sxHeight = int(fm.x_height)
if fm.cap_height and fm.cap_height > 0:
    if getattr(os2, "sCapHeight", 0) in (0, None):
        os2.sCapHeight = int(fm.cap_height)

# Current (always update):
# Always update sxHeight and sCapHeight with measured values
# This ensures OS/2 metadata matches actual glyph measurements
if fm.x_height and fm.x_height > 0:
    os2.sxHeight = int(fm.x_height)
if fm.cap_height and fm.cap_height > 0:
    os2.sCapHeight = int(fm.cap_height)
```

---

### Option 2: Normalize x-height and cap-height Across Family
**Additional data needed:** ✅ **NONE** - We already have this!

**What we'd need to add:**
- Normalized x-height target (similar to how we compute `target_typo_asc`)
- Normalized cap-height target (similar to how we compute `target_typo_asc`)
- New fields in `FontMeasures`: `target_x_height`, `target_cap_height`

**How it would work:**
1. Compute family-normalized x-height (median or average of x-heights in main cluster)
2. Compute family-normalized cap-height (median or average of cap-heights in main cluster)
3. Apply to all fonts in family (similar to typo ascender/descender normalization)
4. Write normalized values to OS/2.sxHeight and OS/2.sCapHeight

**Pros:**
- Consistent x-height and cap-height across family (like typo metrics)
- Better typographic consistency

**Cons:**
- More complex (requires normalization logic)
- May not be desired (fonts might intentionally have different x-heights)
- Could affect visual appearance if fonts are meant to have different proportions

**Code changes needed:**
- Add `target_x_height` and `target_cap_height` to `FontMeasures` model
- Add normalization logic in `planning.py` (similar to typo ascender calculation)
- Update `application.py` to write normalized values

---

## Implementation Status

### ✅ Option 1 Implemented - Always Update with Measured Values
**Status:** ✅ **COMPLETED**
- ✅ No additional data needed
- ✅ Code updated in `application.py` and `planning.py`
- ✅ Ensures OS/2 metadata matches actual glyph measurements
- ✅ Preserves font-specific characteristics (per-font, not family-wide)

### For Normalizing Across Family:
**Use Option 2** - Only if you want consistent x-height/cap-height across family
- ⚠️ Requires normalization logic (similar to typo metrics)
- ⚠️ May not be desired for all font families
- ⚠️ Could be a config option (`--normalize-xheight`, `--normalize-capheight`)

---

## Summary

**To always update sxHeight and sCapHeight with measured values:**
- **Additional data needed:** None ✅
- **Code change:** Remove the conditional check (2 lines)
- **Impact:** OS/2 metadata will always match glyph measurements

**To normalize x-height and cap-height across family:**
- **Additional data needed:** None ✅ (we already measure them)
- **Code change:** Add normalization logic similar to typo metrics
- **Impact:** All fonts in family will have consistent x-height/cap-height

The data is already there - it's just a question of whether you want to:
1. Always write measured values (simple fix)
2. Normalize across family (more complex, may not be desired)
