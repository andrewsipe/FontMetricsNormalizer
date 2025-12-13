"""Validation and reporting functions."""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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

from . import config
from . import models
from . import planning

console = get_console()
MetricsConfig = config.MetricsConfig
FontMeasures = models.FontMeasures

# Import planning functions
analyze_family_impact = planning.analyze_family_impact
compute_family_normalized_extremes = planning.compute_family_normalized_extremes
compute_family_normalized_ascender = planning.compute_family_normalized_ascender


def validate_args(args: argparse.Namespace) -> None:
    # Validate letter-height (percentage input)
    if args.letter_height <= 50 or args.letter_height > 200:
        cs.StatusIndicator("warning").add_message(
            f"letter-height {args.letter_height}% unusual (typically 110-150%)"
        ).emit(console)

    # Validate top-margin (percentage input)
    if args.top_margin < 10:
        cs.StatusIndicator("warning").add_message(
            f"top-margin {args.top_margin}% very low (<10% UPM) - may clip tall glyphs"
        ).emit(console)
    elif args.top_margin > 40:
        cs.StatusIndicator("warning").add_message(
            f"top-margin {args.top_margin}% very high (>40% UPM) - may waste vertical space"
        ).emit(console)

    # Warn about combined expansion
    if args.letter_height > 150 and args.top_margin > 30:
        estimated_span = args.letter_height + args.top_margin
        cs.StatusIndicator("warning").add_message(
            f"Combined letter-height ({args.letter_height}%) + top-margin ({args.top_margin}%) "
            f"≈ {estimated_span}% of UPM - extremely loose spacing"
        ).emit(console)

    # Warn if modifiers used with wrong mode
    if args.grouping_mode != "superfamily":
        if getattr(args, "exclude", None):
            cs.StatusIndicator("warning").add_message(
                "--exclude only applies to --superfamily mode (ignored)"
            ).emit(console)

    # Report prefix normalization when active
    if getattr(args, "ignore_prefix", None):
        cs.StatusIndicator("info").add_message(
            f"Prefix normalization active: {', '.join(args.ignore_prefix)}"
        ).add_item(
            "Family names will be normalized before grouping",
            indent_level=1,
        ).emit(console)

    if args.grouping_mode == "individual":
        if (
            getattr(args, "combine", None)
            or getattr(args, "ignore_prefix", None)
            or getattr(args, "exclude", None)
        ):
            cs.StatusIndicator("warning").add_message(
                "--individual ignores all grouping modifiers (--combine, --ignore-prefix, --exclude)"
            ).emit(console)

    if args.grouping_mode == "conservative":
        cs.StatusIndicator("info").add_message(
            "Safe-max mode: using bbox extremes for all fonts (no clustering, prevents clipping)"
        ).emit(console)

    # Report pattern overrides
    if (
        getattr(args, "assume_script", None)
        or getattr(args, "assume_decorative", None)
        or getattr(args, "assume_unicase", None)
    ):
        cs.emit("", console=console)
        cs.StatusIndicator("info").add_message("Detection overrides active:").emit(
            console
        )

        if getattr(args, "assume_script", None):
            for pattern in args.assume_script:
                cs.emit(f"  • Script: {pattern}", console=console)

        if getattr(args, "assume_decorative", None):
            for pattern in args.assume_decorative:
                cs.emit(f"  • Decorative: {pattern}", console=console)

        if getattr(args, "assume_unicase", None):
            for pattern in args.assume_unicase:
                cs.emit(f"  • Unicase: {pattern}", console=console)


def confirm_or_exit(count: int) -> None:
    """Prompt for confirmation, looping until clear yes/no response."""
    while True:
        try:
            cs.emit("", console=console)
            response = (
                cs.prompt_input(
                    f"About to modify {cs.fmt_count(count)} file(s). Proceed? [y/N]: "
                )
                .strip()
                .lower()
            )

            if response in ["y", "yes"]:
                cs.emit("", console=console)
                return  # Proceed
            elif response in ["n", "no"]:
                cs.StatusIndicator("error").add_message("Aborted by user").emit(console)
                sys.exit(3)
            elif response == "":
                # Empty string - treat as mistake, re-prompt
                cs.StatusIndicator("warning").add_message(
                    "Please enter 'y' for yes or 'n' for no"
                ).emit(console)
                continue
            else:
                # Invalid input - re-prompt
                cs.StatusIndicator("warning").add_message(
                    f"Invalid input '{response}'. Please enter 'y' for yes or 'n' for no"
                ).emit(console)
                continue
        except (EOFError, KeyboardInterrupt):
            cs.StatusIndicator("error").add_message("Aborted by user").emit(console)
            sys.exit(3)


def report_changes(families, plans, args, forced_groups) -> bool:
    """Report planned changes per family."""
    any_changes_needed = False

    def get_impact_category(fam):
        group = families[fam]
        avg_typo, avg_span, num_fonts, has_changes = analyze_family_impact(group)
        if not has_changes or avg_typo < 0.1:
            return (0, fam)
        elif avg_typo < 2.0:
            return (1, fam)
        elif avg_typo < 8.0:
            return (2, fam)
        else:
            return (3, fam)

    sorted_families = sorted(plans.items(), key=lambda x: get_impact_category(x[0]))

    for fam, (fam_min, fam_max, fam_asc) in sorted_families:
        group = families[fam]
        avg_typo, avg_span, num_fonts, has_changes = analyze_family_impact(group)

        family_label = f"[bold]{fam}[/bold]"
        unique_names = set(fm.family_name for fm in group)
        if len(unique_names) > 1:
            if args.grouping_mode == "superfamily":
                family_label += " [darktext.dim](superfamily)[/darktext.dim]"
            elif forced_groups and any(fam in fg for fg in forced_groups):
                family_label += " [darktext.dim](forced group)[/darktext.dim]"

        # Adjust label for individual mode
        field_label = "Font" if args.grouping_mode == "individual" else "Family"
        font_count_text = (
            ""
            if args.grouping_mode == "individual"
            else f"[darktext.dim]({cs.fmt_count(num_fonts)} fonts)[/darktext.dim]"
        )

        if not has_changes or avg_typo < 0.1:
            cs.StatusIndicator("info").add_message(
                f"[field]{field_label}:[/field] {family_label} "
                f"{font_count_text} — No changes needed"
            ).emit(console)
            cs.StatusIndicator("unchanged").add_message("Already normalized").emit(
                console
            )
            continue

        any_changes_needed = True

        if avg_typo < 2.0:
            impact_type, impact_desc = "minimal", "Minimal Normalization"
        elif avg_typo < 8.0:
            impact_type, impact_desc = "moderate", "Moderate Normalization"
        else:
            impact_type, impact_desc = "major", "Major Normalization"

        if abs(avg_span) < 1.0:
            span_info = "preserving vertical height"
        elif avg_span > 0:
            span_info = f"increasing vertical height by [count]{avg_span:.1f}[/count]%"
        else:
            span_info = f"reducing vertical height by [count]{avg_span:.1f}[/count]%"

        cs.StatusIndicator("info").add_message(
            f"[field]{field_label}:[/field] {family_label} "
            f"{font_count_text} — {impact_desc}"
        ).emit(console)

        cs.StatusIndicator(impact_type).add_message(
            f"UPM scaled by ~[count]{avg_typo:.1f}[/count]%, {span_info} [info]|[/info] linegap set to {cs.fmt_count(0)}"
        ).emit(console)

    return any_changes_needed


def generate_family_report(
    families: Dict[str, List[FontMeasures]],
    config: MetricsConfig,
    args: argparse.Namespace,
) -> None:
    """Generate detailed report comparing family vs per-font calculations.

    Example output:
        Font Sans (3 fonts)

          Family ascender: 0.8200 (normalized)
          Driven by: Sans-Black.ttf (asc: 820)

          ↑ Sans-Regular.ttf    + 70 units (+8.5%) - pulled up by family
          ↑ Sans-Bold.ttf       + 50 units (+6.5%) - pulled up by family
          ✓ Sans-Black.ttf      ±  0 units (+0.0%) - minimal impact

          ⚠️  High normalization impact detected (8.5% max pull)
              Consider using --per-font for heavily affected fonts
              Or use --max-pull 6.8 to limit pulling
    """

    cs.emit("")
    cs.StatusIndicator("info").add_message("Family Normalization Impact Report").emit(
        console
    )
    cs.emit("")

    for fam_name, group in families.items():
        if len(group) == 1:
            continue  # Skip single-font families

        cs.emit(
            f"[bold]{fam_name}[/bold] ({cs.fmt_count(len(group))} fonts)",
            console=console,
        )
        cs.emit("")

        # Calculate family-wide metrics
        fam_min, fam_max = compute_family_normalized_extremes(group)
        family_asc = compute_family_normalized_ascender(group, config)

        # Track pulling effects
        pulls: List[Tuple[FontMeasures, float, float, int]] = []

        for fm in group:
            # Calculate what this font would get individually
            solo_asc = compute_family_normalized_ascender([fm], config)

            # Calculate normalized difference
            family_value = int(family_asc * fm.upm)
            solo_value = int(solo_asc * fm.upm)
            diff_units = family_value - solo_value
            diff_percent = (
                ((family_asc - solo_asc) / solo_asc * 100) if solo_asc > 0 else 0
            )

            pulls.append((fm, diff_percent, solo_asc, diff_units))

        # Sort by impact (most pulled first)
        pulls.sort(key=lambda x: abs(x[1]), reverse=True)

        # Find the "driver" (font with tallest ascenders)
        driver = max(
            group,
            key=lambda fm: (fm.ascender_max or 0) / fm.upm if fm.upm > 0 else 0,
        )

        cs.emit(
            f"  [dim]Family ascender:[/dim] {family_asc:.4f} (normalized)",
            console=console,
        )
        cs.emit(
            f"  [dim]Driven by:[/dim] {Path(driver.path).name} (asc: {driver.ascender_max})",
            console=console,
        )
        cs.emit("")

        # Report per-font impact
        for fm, diff_pct, solo_asc, diff_units in pulls:
            filename = Path(fm.path).name

            if abs(diff_pct) < 1.0:
                # Minimal change
                indicator = "✓"
                style = "dim"
                msg = f"±{abs(diff_units):>4} units ({diff_pct:+.1f}%) - minimal impact"
            elif diff_units > 0:
                # Pulled up
                indicator = "↑"
                style = "warning"
                msg = f"+{diff_units:>4} units ({diff_pct:+.1f}%) - pulled up by family"
            else:
                # Pulled down (rare)
                indicator = "↓"
                style = "info"
                msg = f"{diff_units:>4} units ({diff_pct:.1f}%) - pulled down by family"

            cs.emit(
                f"  [{style}]{indicator}[/{style}] {filename:40s} {msg}",
                console=console,
            )

        cs.emit("")

        # Recommendation
        max_pull = max(abs(p[1]) for p in pulls)
        if max_pull > 8.0:
            cs.StatusIndicator("warning").add_message(
                f"⚠️  High normalization impact detected ({max_pull:.1f}% max adjustment)"
            ).add_item(
                "Consider using --individual for heavily affected fonts", indent_level=1
            ).add_item(
                f"Or use --max-adjustment {max_pull * 0.8:.1f} to limit adjustment",
                indent_level=1,
            ).emit(console)

        cs.emit("")

    # Summary: report single-font families that were skipped
    single_font_families = [fam for fam, group in families.items() if len(group) == 1]
    if single_font_families:
        cs.StatusIndicator("info").add_message(
            f"{cs.fmt_count(len(single_font_families))} single-font families skipped (no pulling analysis needed)"
        ).emit(console)
