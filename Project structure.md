"""
ebrium - Conservative Refactoring Plan
=======================================

Goal: Extract the monolithic script into modules while preserving ALL original logic.

Directory Structure:
-------------------
ebrium/
├── __init__.py
├── __main__.py              # Entry point (calls main())
├── config.py                # MetricsConfig dataclass + constants
├── models.py                # FontMeasures dataclass
├── font_io.py               # Font reading/writing helpers
├── measurements.py          # All measurement functions
├── clustering.py            # Optical clustering logic
├── planning.py              # Metrics planning functions
├── application.py           # apply_metrics function
├── checkpoints.py           # Checkpoint save/load
├── grouping.py              # Family/superfamily grouping
├── validation.py            # Validation and confirmation
└── cli.py                   # CLI parsing and orchestration

Extraction Strategy:
-------------------

1. config.py - Extract unchanged:
   - MetricsConfig
   - All CODEPOINT constants
   - SUPPORTED_EXTENSIONS if we add it

2. models.py - Extract unchanged:
   - FontMeasures class (copy as-is)

3. font_io.py - Extract existing helpers:
   - _read_ttfont()
   - _get_upm()
   - _get_best_cmap()
   - _glyph_bounds()
   - _codepoint_bounds()
   - _font_cmap_glyph_names()
   - _font_overall_bounds()

4. measurements.py - Extract measurement functions:
   - _cap_height()
   - _cap_height_optical()
   - _ascender_max()
   - _descender_min()
   - _x_height()
   - _get_cap_height_from_glyphs()
   - _get_x_height_from_glyphs()
   - _get_x_height_from_glyphs_robust()
   - _get_cap_height_from_glyphs_robust()
   - is_unicase()
   - measure_fonts()

5. clustering.py - Extract clustering logic:
   - compute_optical_similarity()
   - compute_optical_variance()
   - detect_decorative_outlier()
   - detect_script_font()
   - cluster_group_helper()
   - detect_optical_clusters()

6. planning.py - Extract planning functions:
   - compute_family_normalized_extremes()
   - compute_descender_for_centering()
   - compute_cluster_target_percent()
   - compute_family_normalized_ascender()
   - plan_identical_metrics()
   - finalize_metrics()
   - get_cluster_normalized_typo()
   - plan_safe_metrics()
   - validate_cluster_consistency()
   - plan_adaptive_metrics()
   - analyze_family_impact()
   - build_plans()

7. application.py - Extract application:
   - apply_metrics()
   - process_all()

8. checkpoints.py - Extract checkpoint functions:
   - compute_config_hash()
   - save_measurements_checkpoint()
   - load_measurements_checkpoint()

9. grouping.py - Extract grouping:
   - collect_groups()
   - expand_comma_separated_args()
   - group_families()

10. validation.py - Extract validation/UI:
    - validate_args()
    - confirm_or_exit()
    - report_changes()
    - generate_family_report()

11. cli.py - Orchestration:
    - parse_args()
    - scan_fonts()
    - main()

Key Principles:
--------------
1. Copy functions as-is initially
2. Keep all imports in each module
3. Functions keep their signatures
4. No logic changes in first pass
5. Test after each module extraction

Implementation Order:
--------------------
1. Create all module files
2. Extract config + models (no dependencies)
3. Extract font_io (minimal dependencies)
4. Extract measurements (depends on font_io)
5. Extract clustering (depends on measurements)
6. Extract planning (depends on clustering)
7. Extract application (depends on planning)
8. Extract remaining modules
9. Build main() orchestrator
10. Test against original with identical inputs

Testing Strategy:
----------------
- Run both scripts on same input
- Compare checkpoint files (should be identical)
- Compare final font files (should be byte-identical)
- Compare console output (should match)
"""