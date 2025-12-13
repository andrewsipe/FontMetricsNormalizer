"""Family grouping functions for organizing fonts into families/superfamilies."""

import sys
from pathlib import Path
from typing import List, Optional

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
from FontCore.core_font_sorter import FontSorter, FontInfo

from . import models

console = get_console()
FontMeasures = models.FontMeasures


# collect_groups removed - logic moved to cli.py


def expand_comma_separated_args(arg_list: Optional[List[str]]) -> List[str]:
    """Expand comma-separated values in argument list.

    Supports both formats:
      --flag "a,b,c"  ->  ['a', 'b', 'c']
      --flag a --flag b --flag c  ->  ['a', 'b', 'c']
      --flag "a,b" --flag c  ->  ['a', 'b', 'c']
    """
    if not arg_list:
        return []

    expanded = []
    for item in arg_list:
        # Split by comma and strip whitespace
        expanded.extend(term.strip() for term in item.split(",") if term.strip())
    return expanded


def group_families(args, measures, forced_groups):
    """Group fonts by family or superfamily based on args.grouping_mode.

    Args:
        args: Parsed command-line arguments (must have grouping_mode attribute)
        measures: List of FontMeasures
        forced_groups: List of forced group merges from --combine

    Returns:
        Dict mapping group name to list of FontMeasures
    """
    # Individual mode: each font is its own group
    if args.grouping_mode == "individual":
        cs.StatusIndicator("info").add_message(
            f"Processing {cs.fmt_count(len(measures))} font(s) individually (no grouping, no clustering)"
        ).emit(console)
        return {Path(fm.path).stem: [fm] for fm in measures}

    # Collect ignore_terms for both family and superfamily modes
    ignore_terms_set = set(
        expand_comma_separated_args(getattr(args, "ignore_prefix", None) or [])
    )

    # Conservative mode: family grouping only (no modifiers apply)
    if args.grouping_mode == "conservative":
        font_infos = [
            FontInfo(path=fm.path, family_name=fm.family_name) for fm in measures
        ]
        sorter = FontSorter(font_infos, ignore_terms=ignore_terms_set)
        groups = sorter.group_by_family(forced_groups=forced_groups)

        cs.StatusIndicator("info").add_message(
            f"Found {cs.fmt_count(len(groups))} family group(s) (safe-max mode, no clustering)"
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

    # Family or Superfamily mode
    font_infos = [FontInfo(path=fm.path, family_name=fm.family_name) for fm in measures]
    # Pass ignore_terms to constructor (works for both family and superfamily)
    sorter = FontSorter(font_infos, ignore_terms=ignore_terms_set)

    # Show normalization mapping if ignore_terms is active and verbose
    verbosity = getattr(args, "verbose", 0)
    if ignore_terms_set and verbosity >= 1:
        normalization_summary = sorter.get_normalization_summary()
        if normalization_summary:
            cs.emit("", console=console)
            cs.StatusIndicator("info").add_message(
                "Family name normalization (ignored prefixes removed):"
            ).emit(console)
            for original, normalized in sorted(normalization_summary.items()):
                cs.emit(
                    f"  {original} → {normalized}",
                    console=console,
                )

    if args.grouping_mode == "superfamily":
        # Superfamily mode: apply exclude modifiers (ignore-prefix handled in constructor)
        groups = sorter.group_by_superfamily(
            exclude_families=expand_comma_separated_args(
                getattr(args, "exclude", None) or []
            ),
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
        # ignore_terms is automatically applied via constructor
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
