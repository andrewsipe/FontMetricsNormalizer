# Product refinement notes — FontMetricsNormalizer (ebrium)

Captured during the 2026-08-21 declutter pass. Use for a later product/release pass. **Not** user-facing docs.

## Declutter verdict

**No whole scripts to archive.** This is already a single packaged product (`ebrium`) with a clear CLI (`ebrium` / `python -m ebrium`). All modules are reachable from `cli.py`.

### Done in declutter

| Change | Why |
|--------|-----|
| Removed duplicate `apply_metrics` from `ebrium/planning.py` | Byte-identical copy of `application.apply_metrics`; never called (CLI uses `application.process_all` → `application.apply_metrics` only) |
| Deleted local `ebrium.egg-info/` | Gitignored build residue |

## Active package layout

| Module | Role |
|--------|------|
| `ebrium/cli.py` | Argparse + pipeline orchestration |
| `ebrium/config.py` | MetricsConfig |
| `ebrium/models.py` | FontMeasures etc. |
| `ebrium/measurements.py` | Measure fonts |
| `ebrium/grouping.py` | Family / superfamily grouping |
| `ebrium/clustering.py` | Optical / decorative clustering |
| `ebrium/planning.py` | Plan metrics (large: ~1.7k lines after declutter) |
| `ebrium/application.py` | Apply planned metrics to files |
| `ebrium/validation.py` | Validation / reports |
| `ebrium/variation_probe.py` | VF probe mode (CLI hook) |
| `ebrium/checkpoints.py` | Measurement/cluster checkpoint cache |
| `ebrium/font_io.py` | Font I/O helpers |

## Product-pass refinements (deferred)

1. **Naming** — Folder `FontMetricsNormalizer` vs package/CLI `ebrium`. Pick one public brand for release (README already leads with both).
2. **`planning.py` size** — Still the bulk of the logic; split plan/report helpers if refactoring for maintainability (do not confuse with the removed dead `apply_metrics`).
3. **Tests** — No pytest suite in-tree; add fixtures for family / superfamily / safe-mode / dry-run.
4. **Checkpoints** — Document on-disk checkpoint format and when caches invalidate (`compute_config_hash`).
5. **Placeholder author email** in `pyproject.toml` (`andrewsipe@example.com`) — fix before public release.
6. **Overlap** — FontFixer touches OS/2/style flags; ebrium owns vertical metrics + USE_TYPO_METRICS. Document “run order” if both are used in a pipeline.
7. **`raw_github_urls.txt`** — PushCore noise; exclude from release artifacts.

## Do not lose

- Cap-height anchoring, win vs typo/hhea strategy, line-gap → 0, USE_TYPO_METRICS when OS/2 ≥ 4 (README strategy section).
- Family vs `--superfamily` vs `--per-font` / `--safe-mode` semantics.
- Checkpoint invalidation when clustering-related config changes.
