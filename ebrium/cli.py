"""CLI parsing and main orchestration."""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
from FontCore.core_file_collector import collect_font_files
from FontCore.core_logging_config import Verbosity

from . import application
from . import checkpoints
from . import config
from . import grouping
from . import measurements
from . import models
from . import planning
from . import validation

console = get_console()
MetricsConfig = config.MetricsConfig
FontMeasures = models.FontMeasures

# Import functions from modules
measure_fonts = measurements.measure_fonts
save_measurements_checkpoint = checkpoints.save_measurements_checkpoint
load_measurements_checkpoint = checkpoints.load_measurements_checkpoint
group_families = grouping.group_families
build_plans = planning.build_plans
report_changes = validation.report_changes
generate_family_report = validation.generate_family_report
process_all = application.process_all
confirm_or_exit = validation.confirm_or_exit
validate_args = validation.validate_args


def scan_fonts(paths: Iterable[str], recursive: bool, include_ttx: bool) -> List[str]:
    files = collect_font_files(paths, recursive)
    if include_ttx:
        return files
    return [f for f in files if Path(f).suffix.lower() != ".ttx"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize vertical metrics using cap height as stable anchor",
        epilog="Supported formats: TTF, OTF, WOFF, WOFF2",
    )
    parser.add_argument("paths", nargs="*", help="Font files or directories")
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Recurse into directories"
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="Preview without writing"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt"
    )
    # Vertical spacing (all values as % of UPM)
    spacing = parser.add_argument_group("vertical spacing (all values as % of UPM)")
    spacing.add_argument(
        "--letter-height",
        type=float,
        default=130,
        metavar="PERCENT",
        help="Target vertical space for letters (default: 130%% of UPM)",
    )
    parser.add_argument("--use-ttx", action="store_true", help="Include .ttx files")

    # Grouping mode (mutually exclusive with --safe-max)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--individual",
        dest="grouping_mode",
        action="store_const",
        const="individual",
        help="Normalize each font independently (no grouping, no clustering)",
    )
    mode.add_argument(
        "--family",
        dest="grouping_mode",
        action="store_const",
        const="family",
        help="Group by family name, cluster within families (default)",
    )
    mode.add_argument(
        "--superfamily",
        dest="grouping_mode",
        action="store_const",
        const="superfamily",
        help="Merge families with shared prefix, cluster across superfamily",
    )
    # Note: --safe-max is defined in the "safe normalization modes" group below
    # but also needs to be in this mutually exclusive group. We'll add it to both.
    parser.set_defaults(grouping_mode="family")

    # Grouping modifiers
    grouping_modifiers = parser.add_argument_group(
        "grouping modifiers",
        description="These options modify grouping behavior for --family, --superfamily, and --safe-max modes",
    )
    grouping_modifiers.add_argument(
        "--combine",
        action="append",
        metavar="FAMILIES",
        help='Force-merge families: --combine "Font A,Font B,Font C". '
        "Works with all grouping modes. Can be used multiple times.",
    )
    grouping_modifiers.add_argument(
        "--ignore-prefix",
        action="append",
        metavar="TOKEN",
        help="Ignore token when normalizing family names: --ignore-prefix Adobe --ignore-prefix LT. "
        "Applies to both --family and --superfamily modes. "
        "In family mode: normalizes prefixes before exact matching. "
        "In superfamily mode: normalizes prefixes before finding common prefixes. "
        "Can be used multiple times.",
    )
    grouping_modifiers.add_argument(
        "--exclude",
        action="append",
        metavar="NAME",
        help="Keep family separate from superfamily grouping: --exclude Script --exclude Display. "
        "Applies to --superfamily mode only. Can be used multiple times.",
    )
    parser.add_argument(
        "--top-margin",
        type=float,
        default=25,
        metavar="PERCENT",
        help="Extra space above capitals (default: 25%% of UPM)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show detailed analysis of family vs per-font calculations (implies --dry-run)",
    )
    spacing.add_argument(
        "--max-adjustment",
        type=float,
        default=None,
        metavar="PERCENT",
        help="Maximum %% adjustment by family extremes (default: unlimited). "
        "Fonts exceeding this get individual calculation.",
    )
    # Detection overrides
    detection = parser.add_argument_group("detection overrides")
    detection.add_argument(
        "--assume-script",
        action="append",
        metavar="PATTERN",
        help="Treat matching fonts as script style (e.g., '*Script*', '*Swash*', 'FontName-*'). "
        "Supports glob patterns. Can be used multiple times.",
    )
    detection.add_argument(
        "--assume-decorative",
        action="append",
        metavar="PATTERN",
        help="Treat matching fonts as decorative variants (e.g., '*Rough*', '*Shadow*', '*Inline*'). "
        "Supports glob patterns. Can be used multiple times.",
    )
    detection.add_argument(
        "--assume-unicase",
        action="append",
        metavar="PATTERN",
        help="Treat matching fonts as unicase (e.g., '*Unicase*', 'FontName-SC'). "
        "Supports glob patterns. Can be used multiple times.",
    )
    detection.add_argument(
        "--exclude-measuring",
        action="append",
        metavar="PATTERN",
        help="Exclude matching fonts from family calculations (e.g., '*Circle*', 'FontName-Decorative.otf'). "
        "Fonts are still measured but their measurements don't affect family averages. "
        "Supports glob patterns. Can be used multiple times.",
    )

    # Safe normalization modes
    safe_modes = parser.add_argument_group(
        "safe normalization modes",
        description="Safe approaches that preserve original vertical span (--safe-hhea) or use maximum bounds (--safe-max). "
        "Note: --safe-max is mutually exclusive with other grouping modes (--family, --superfamily, --individual).",
    )
    # Add --safe-max to safe_modes group (also added to mode group for mutual exclusivity)
    safe_max_action = safe_modes.add_argument(
        "--safe-max",
        dest="grouping_mode",
        action="store_const",
        const="conservative",
        help="Group by family, use bbox extremes for all fonts (no clustering, safest - prevents clipping). "
        "Mutually exclusive with --family, --superfamily, --individual.",
    )
    # Also add to mode group for mutual exclusivity
    mode._group_actions.append(safe_max_action)
    safe_modes.add_argument(
        "--safe-hhea",
        action="store_true",
        help="Apply averaged existing hhea/typo values across entire family (excluding --exclude-measuring fonts). "
        "Reads existing values from included fonts, averages them, and applies uniformly to all fonts. "
        "Preserves original vertical span while ensuring consistent hhea metrics. Win values use max ranges.",
    )

    # Hidden expert thresholds (not shown in --help)
    parser.add_argument(
        "--decorative-threshold",
        type=float,
        default=1.4,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--unicase-threshold",
        type=float,
        default=0.05,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-span-ratio",
        type=float,
        default=1.5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-auto-adjust",
        action="store_true",
        help="Disable automatic target adjustment based on x-height (use exact --letter-height)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level. Use -v for VERBOSE, -vv for DEBUG level output",
    )
    return parser.parse_args()


def main() -> None:
    start_time = time.time()
    args = parse_args()
    validate_args(args)

    # --report implies --dry-run
    if args.report:
        args.dry_run = True

    # Convert percentage inputs to internal fraction representation
    config = MetricsConfig(
        target_span=(args.letter_height / 100.0) if args.letter_height > 0 else 1.3,
        win_buffer=0.02,
        xheight_softener=0.6,
        adapt_for_xheight=True,
        optical_threshold=0.025,  # Validated optimal threshold (2.5% UPM)
        top_margin=(args.top_margin / 100.0) if args.top_margin >= 0 else 0.25,
        max_adjustment=(args.max_adjustment / 100.0) if args.max_adjustment else None,
        max_span_ratio=args.max_span_ratio,
        decorative_span_threshold=args.decorative_threshold,
        unicase_threshold=args.unicase_threshold,
        auto_adjust_target=not args.no_auto_adjust,
    )

    files = scan_fonts(args.paths or ["."], args.recursive, args.use_ttx)
    if not files:
        cs.StatusIndicator("error").add_message("No font files found").emit(console)
        sys.exit(1)

    # Determine checkpoint path
    checkpoint_path = Path(".metrics_checkpoint.json")

    # Try to load checkpoint
    loaded_measures: List[FontMeasures] = []
    checkpoint_files: List[str] = []
    cached_clusters: Optional[Dict[str, Dict[str, List[str]]]] = None
    if checkpoint_path.exists():
        try:
            loaded_measures, missing_files, cached_clusters = (
                load_measurements_checkpoint(
                    checkpoint_path, expected_files=files, config=config
                )
            )
            checkpoint_files = [fm.path for fm in loaded_measures]

            if loaded_measures:
                # Check if checkpoint matches current file set
                checkpoint_set = set(checkpoint_files)
                files_set = set(files)
                overlap = len(checkpoint_set & files_set)
                new_files = len(files_set - checkpoint_set)
                removed_files = len(checkpoint_set - files_set)

                # If no overlap, checkpoint is useless - skip it
                if overlap == 0:
                    cs.StatusIndicator("info").add_message(
                        "Checkpoint found but contains no matching fonts (skipping)"
                    ).emit(console)
                    measures = []
                else:
                    # Display checkpoint status with StatusIndicator
                    cs.emit("", console=console)
                    if checkpoint_set == files_set:
                        # Perfect match
                        cs.StatusIndicator("info").add_message(
                            f"Checkpoint found with measurements for all {cs.fmt_count(len(loaded_measures))} fonts"
                        ).add_item(
                            "Using checkpoint would skip all measurement",
                            indent_level=1,
                        ).emit(console)
                        prompt_msg = "Use checkpoint?"
                    elif new_files > 0 and removed_files == 0:
                        # Only new files added
                        cs.StatusIndicator("info").add_message(
                            f"Checkpoint found with measurements for {cs.fmt_count(overlap)} fonts"
                        ).add_item(
                            f"{cs.fmt_count(new_files)} new file(s) would still need measurement",
                            indent_level=1,
                        ).emit(console)
                        prompt_msg = "Use checkpoint for existing measurements?"
                    elif removed_files > 0 and new_files == 0:
                        # Some files removed
                        cs.StatusIndicator("info").add_message(
                            f"Checkpoint found with measurements for {cs.fmt_count(overlap)} fonts"
                        ).add_item(
                            f"({cs.fmt_count(removed_files)} files from checkpoint no longer in current set)",
                            indent_level=1,
                        ).emit(console)
                        prompt_msg = "Use checkpoint? (skips all measurement)"
                    else:
                        # Mixed changes
                        cs.StatusIndicator("info").add_message(
                            f"Checkpoint found: {cs.fmt_count(overlap)} matching fonts"
                        ).add_item(
                            f"{cs.fmt_count(new_files)} new file(s) would need measurement",
                            indent_level=1,
                        ).add_item(
                            f"{cs.fmt_count(removed_files)} file(s) removed from current set",
                            indent_level=1,
                        ).emit(console)
                        prompt_msg = "Use checkpoint for matching fonts?"

                    # Prompt user for checkpoint usage
                    cs.emit("", console=console)
                    resp = cs.prompt_confirm(prompt_msg, default=True)

                    if resp:
                        # Use checkpoint for matching files
                        measures = [
                            fm for fm in loaded_measures if fm.path in files_set
                        ]
                        # Apply exclusion patterns to checkpoint-loaded fonts
                        if args.exclude_measuring:
                            import fnmatch

                            for fm in measures:
                                filename = Path(fm.path).name
                                for pattern in args.exclude_measuring:
                                    if fnmatch.fnmatch(filename, pattern):
                                        fm.is_excluded_from_calculations = True
                                        break
                        # Update files list to only measure missing ones
                        files = [f for f in files if f not in checkpoint_files]

                        if new_files > 0:
                            cs.StatusIndicator("info").add_message(
                                f"Reusing {cs.fmt_count(overlap)} measurements, will measure {cs.fmt_count(new_files)} new file(s)"
                            ).emit(console)
                        else:
                            cs.StatusIndicator("info").add_message(
                                f"Reusing all {cs.fmt_count(len(measures))} measurements from checkpoint"
                            ).emit(console)
                    else:
                        # User wants fresh measurement
                        measures = []
                        cs.StatusIndicator("info").add_message(
                            f"Remeasuring all {cs.fmt_count(len(files))} fonts"
                        ).emit(console)
            else:
                measures = []
        except Exception as e:
            cs.StatusIndicator("warning").add_message(
                f"Error processing checkpoint: {e}. Proceeding with fresh measurement."
            ).emit(console)
            measures = []
    else:
        measures = []

    # Measure fonts (only those not in checkpoint)
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
                assume_script=args.assume_script,
                assume_decorative=args.assume_decorative,
                assume_unicase=args.assume_unicase,
                exclude_measuring=args.exclude_measuring,
            )
        except KeyboardInterrupt:
            cs.emit("", console=console)
            cs.StatusIndicator("warning").add_message(
                "Measurement interrupted. Saving partial checkpoint..."
            ).emit(console)
            # Save partial checkpoint before exiting
            if measures:
                save_measurements_checkpoint(measures, checkpoint_path, config=config)
            cs.StatusIndicator("info").add_message(
                f"Partial checkpoint saved to {checkpoint_path}"
            ).emit(console)
            sys.exit(0)

    if not measures:
        cs.StatusIndicator("error").add_message("No measurable fonts found").emit(
            console
        )
        sys.exit(2)

    # Save checkpoint after successful measurement (clusters will be saved after build_plans)
    save_measurements_checkpoint(measures, checkpoint_path, config=config)

    cs.emit("", console=console)
    # Collect forced groups from --combine argument
    forced_groups = []
    if args.combine:
        for group_str in args.combine:
            families = [name.strip() for name in group_str.split(",")]
            if len(families) < 2:
                cs.StatusIndicator("warning").add_message(
                    f"--combine requires at least 2 families, skipping: {group_str}"
                ).emit(console)
                continue
            forced_groups.append(families)
    families = group_families(args, measures, forced_groups)
    cs.emit("", console=console)

    # Map verbose count to Verbosity enum: 0=BRIEF, 1=VERBOSE, 2+=DEBUG
    verbosity = (
        Verbosity.DEBUG
        if args.verbose >= 2
        else (Verbosity.VERBOSE if args.verbose >= 1 else Verbosity.BRIEF)
    )
    family_plans, clusters_cache = build_plans(
        families,
        config,
        verbosity=verbosity,
        cached_clusters=cached_clusters,
        grouping_mode=args.grouping_mode,
        force_hhea=args.safe_hhea,
    )

    # Save checkpoint with cluster information
    save_measurements_checkpoint(
        measures, checkpoint_path, config=config, clusters=clusters_cache
    )

    if args.report:
        # Generate detailed impact report
        generate_family_report(families, config, args)
        cs.emit("")
        cs.StatusIndicator("info").add_message(
            "Report complete. Use --dry-run to preview changes or remove --report to apply."
        ).emit(console)
        sys.exit(0)

    any_changes_needed = report_changes(families, family_plans, args, forced_groups)

    if not any_changes_needed:
        elapsed = time.time() - start_time
        cs.emit("")
        cs.StatusIndicator("success").add_message(
            "All families already normalized!"
        ).emit(console)
        cs.emit(
            f"{cs.INDENT}[darktext.dim]Total time: [bold]{elapsed:.1f}[/bold]s[/darktext.dim]",
            console=console,
        )
        sys.exit(0)

    if args.dry_run:
        process_all(measures, dry_run=True)
        cs.emit(
            f"{cs.INDENT}[darktext.dim]Total time: [bold]{time.time() - start_time:.1f}[/bold]s[/darktext.dim]",
            console=console,
        )
        return

    if not args.yes:
        confirm_or_exit(len(measures))

    process_all(measures, dry_run=False)
    cs.emit(
        f"{cs.INDENT}[darktext.dim]Total time: [bold]{time.time() - start_time:.1f}[/bold]s[/darktext.dim]",
        console=console,
    )
