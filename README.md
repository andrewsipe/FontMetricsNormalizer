# Metrics

Font metrics normalization tool for family-wide metric consistency.

## Overview

Normalizes vertical metrics across a font family without changing unitsPerEm or glyph outlines. Ensures consistent metrics for proper text rendering across different font styles.

## Scripts

### `FamilyMetricsNormalizer.py`
Normalize vertical metrics across a font family.

**Strategy:**
- Cap height is the stable anchor for identifying optically identical fonts
- Family-wide 'Win' metrics prevent clipping using normalized extremes
- Core cluster gets identical typo/hhea metrics (normalized across UPMs)
- Decorative outliers inherit typo metrics but expand win bounds
- Line gaps set to zero across typo and hhea
- USE_TYPO_METRICS flag enabled when OS/2 version >= 4

**Usage:**
```bash
# Normalize metrics for a font family
python FamilyMetricsNormalizer.py /path/to/fonts -R

# Preview changes
python FamilyMetricsNormalizer.py /path/to/fonts -R --dry-run

# Set target percent (default 1.2)
python FamilyMetricsNormalizer.py /path/to/fonts -R --target-percent 1.3

# Non-interactive mode (auto-confirm)
python FamilyMetricsNormalizer.py /path/to/fonts -R --confirm
```

**Options:**
- `-R, --recursive` - Process directories recursively
- `--dry-run` - Preview changes without modifying files
- `--target-percent` - Target percentage for metric normalization (default: 1.2)
- `--confirm` - Auto-confirm changes without prompting

## How It Works

1. **Analysis**: Analyzes all fonts in the family to identify core cluster (optically identical fonts)
2. **Normalization**: Normalizes typo and hhea metrics across the core cluster
3. **Win Metrics**: Sets family-wide Win metrics to prevent clipping
4. **Outliers**: Handles decorative outliers separately, expanding bounds as needed

## Use Cases

- **Family consistency**: Ensure all styles in a family have compatible metrics
- **Cross-platform rendering**: Prevent clipping on Windows with proper Win metrics
- **Line spacing**: Normalize line gaps for consistent text flow

## Dependencies

See `requirements.txt`:
- Core dependencies (fonttools, rich) provided by included `core/` library
- No additional dependencies required

## Installation

1. Clone this repository:
```bash
git clone https://github.com/andrewsipe/FontMetricsNormalizer.git
cd FontMetricsNormalizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Related Tools

- [FontNameID](https://github.com/andrewsipe/FontNameID) - Update font metadata
- [FontFileTools](https://github.com/andrewsipe/FontFileTools) - Various font fixing utilities

