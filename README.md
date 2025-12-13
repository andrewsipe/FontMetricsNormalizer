# FontMetricsNormalizer (ebrium)

Font metrics normalization tool for family-wide metric consistency.

## Overview

Normalizes vertical metrics across a font family without changing unitsPerEm or glyph outlines. Ensures consistent metrics for proper text rendering across different font styles.

**Strategy:**
- Cap height is the stable anchor for identifying optically identical fonts
- Family-wide 'Win' metrics prevent clipping using normalized extremes
- Core cluster gets identical typo/hhea metrics (normalized across UPMs)
- Decorative outliers inherit typo metrics but expand win bounds
- Line gaps set to zero across typo and hhea
- USE_TYPO_METRICS flag enabled when OS/2 version >= 4

## Installation

### Option 1: Install as a package (recommended - run from anywhere)

Install ebrium in editable mode so you can run it from any directory:

```bash
cd FontMetricsNormalizer
pip install -e .
```

After installation, you can run ebrium from anywhere using either:

```bash
# Using python -m (recommended)
python -m ebrium /path/to/fonts -r

# Or using the ebrium command directly
ebrium /path/to/fonts -r
```

### Option 2: Run from FontMetricsNormalizer directory

If you prefer not to install, you can run from the FontMetricsNormalizer directory:

```bash
cd FontMetricsNormalizer
python -m ebrium
```

**Note:** Option 1 is recommended as it allows you to use ebrium from any directory.

## Usage

After installation, run ebrium from any directory:

```bash
# Basic usage - normalize metrics for fonts in current directory
python -m ebrium

# Or use the ebrium command directly
ebrium

# Process specific paths
python -m ebrium /path/to/fonts

# Recursive directory processing
python -m ebrium /path/to/fonts -r

# Preview changes without writing
python -m ebrium /path/to/fonts -r -n

# Auto-confirm (skip prompts)
python -m ebrium /path/to/fonts -r -y

# Show detailed impact report
python -m ebrium /path/to/fonts -r --report
```

## Command-Line Options

### Basic Options
- `paths` - Font files or directories to process (default: current directory)
- `-r, --recursive` - Recurse into directories
- `-n, --dry-run` - Preview changes without modifying files
- `-y, --yes` - Skip confirmation prompt
- `-v, --verbose` - Show detailed file lists and measurement data

### Normalization Parameters
- `--target-percent FLOAT` - Target span as fraction of UPM (default: 1.3)
- `--headroom-ratio FLOAT` - Headroom as percentage of UPM above cap (default: 0.25 = 25%)
- `--no-auto-adjust` - Disable automatic target adjustment based on x-height

### Family Grouping
- `--superfamily` - Auto-group by common prefix
- `--ignore-term TERM, -it TERM` - Ignore token when grouping superfamilies (can be repeated)
- `--exclude-family FAMILY, -ef FAMILY` - Exclude families from superfamily grouping (can be repeated)
- `--group "Font A,Font B", -g "Font A,Font B"` - Force merge families (can be repeated)
- `--per-font` - Process each font individually, ignoring family grouping

### Advanced Options
- `-sm, --safe-mode` - Use conservative bbox approach (no clustering)
- `--max-pull PERCENT` - Maximum % a font can be pulled by family extremes (e.g., 8.0 for 8%)
- `--decorative-threshold RATIO` - Span ratio threshold for decorative variant detection (default: 1.3)
- `--unicase-threshold RATIO` - Threshold for unicase detection (default: 0.05 = 5% UPM)
- `--use-ttx` - Include .ttx files
- `--report` - Show detailed analysis of family vs per-font calculations (implies --dry-run)

## Examples

```bash
# Normalize a font family directory
python -m ebrium ~/fonts/MyFontFamily -r

# Preview changes with detailed report
python -m ebrium ~/fonts/MyFontFamily -r --report

# Process with custom target percent
python -m ebrium ~/fonts/MyFontFamily -r --target-percent 1.4

# Group multiple families together
python -m ebrium ~/fonts -r --group "Font Sans,Font Sans Pro"

# Process each font individually (no family grouping)
python -m ebrium ~/fonts -r --per-font

# Safe mode (no clustering, conservative approach)
python -m ebrium ~/fonts -r -sm
```

## How It Works

1. **Measurement**: Measures all fonts to extract cap height, x-height, ascenders, descenders, and bounds
2. **Clustering**: Identifies optically identical fonts using cap height as the stable anchor
3. **Planning**: Calculates normalized metrics for each font based on family/cluster analysis
4. **Application**: Applies the calculated metrics to font files

### Checkpoint System

ebrium uses a checkpoint system (`.metrics_checkpoint.json`) to cache measurements and cluster assignments. This allows:
- Fast reruns when fonts haven't changed
- Resuming interrupted measurements
- Preserving cluster assignments across runs

Checkpoints are automatically validated against the current configuration and font set.

## Use Cases

- **Family consistency**: Ensure all styles in a family have compatible metrics
- **Cross-platform rendering**: Prevent clipping on Windows with proper Win metrics
- **Line spacing**: Normalize line gaps for consistent text flow
- **Mixed UPM families**: Handle families with different unitsPerEm values
- **Decorative variants**: Properly handle decorative/script fonts in families

## Supported Formats

- TTF (TrueType)
- OTF (OpenType/CFF)
- WOFF (Web Open Font Format)
- WOFF2 (Web Open Font Format 2.0)

## Dependencies

See `requirements.txt`:
- `fonttools>=4.40.0` - Font manipulation library
- `rich>=13.0.0` - Terminal formatting and progress bars

## Project Structure

The tool is organized as a modular Python package:

```
ebrium/
├── __init__.py          # Package initialization
├── __main__.py          # Entry point (python -m ebrium)
├── cli.py               # Command-line interface and orchestration
├── config.py            # Configuration dataclass and constants
├── models.py            # FontMeasures dataclass
├── font_io.py           # Font reading/writing helpers
├── measurements.py      # Font measurement functions
├── clustering.py         # Optical clustering logic
├── planning.py          # Metrics planning and calculation
├── application.py       # Applying metrics to font files
├── checkpoints.py       # Checkpoint save/load functionality
├── grouping.py          # Family/superfamily grouping
└── validation.py        # Validation and reporting
```

See `Project structure.md` for detailed architecture documentation.

## Related Tools

- [FontNameID](https://github.com/andrewsipe/FontNameID) - Update font metadata
- [FontFileTools](https://github.com/andrewsipe/FontFileTools) - Various font fixing utilities
