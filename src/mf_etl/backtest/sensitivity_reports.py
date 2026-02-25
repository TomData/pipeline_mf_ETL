"""Markdown rendering for sensitivity grid artifacts."""

from __future__ import annotations

from typing import Any

import polars as pl


def _markdown_table(df: pl.DataFrame, *, max_rows: int = 12) -> str:
    if df.height == 0:
        return "(no rows)"
    head = df.head(max_rows)
    cols = head.columns
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in head.to_dicts():
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def render_grid_report(
    *,
    summary: dict[str, Any],
    source_summary: pl.DataFrame,
    best_payload: dict[str, Any],
    dim_sensitivity: pl.DataFrame,
    cost_fragility: pl.DataFrame,
    holdbars_profile: pl.DataFrame,
) -> str:
    """Render markdown report for one sensitivity grid run."""

    lines: list[str] = []
    lines.append("# Backtest Sensitivity Grid Report")
    lines.append("")
    lines.append("## Run")
    lines.append(f"- grid_run_id: `{summary.get('grid_run_id')}`")
    lines.append(f"- scope: `{summary.get('scope')}`")
    lines.append(f"- comparability: `{summary.get('comparability')}`")
    lines.append(f"- total_combos: `{summary.get('total_combos')}`")
    lines.append(f"- successful_combos: `{summary.get('successful_combos')}`")
    lines.append(f"- failed_combos: `{summary.get('failed_combos')}`")
    lines.append(f"- zero_trade_combos: `{summary.get('zero_trade_combos')}`")
    lines.append(f"- zero_trade_combo_share: `{summary.get('zero_trade_combo_share')}`")
    lines.append(
        f"- realism_profile_broken_for_universe: `{summary.get('realism_profile_broken_for_universe')}`"
    )
    lines.append(f"- non_finite_cells: `{summary.get('non_finite_cells')}`")
    lines.append(f"- null_metric_cells: `{summary.get('null_metric_cells')}`")
    lines.append("")

    lines.append("## Source Summary")
    lines.append(_markdown_table(source_summary, max_rows=20))
    lines.append("")

    lines.append("## Best Configs")
    for source, metric_payload in best_payload.items():
        lines.append(f"### {source}")
        for metric_name, rows in metric_payload.items():
            lines.append(f"- top by `{metric_name}`:")
            if not rows:
                lines.append("  - (no rows)")
            else:
                for row in rows[:5]:
                    lines.append(
                        "  - "
                        f"combo={row.get('combo_id')} hb={row.get('hold_bars')} sig={row.get('signal_mode')} "
                        f"exit={row.get('exit_mode')} fee={row.get('fee_bps_per_side')} slip={row.get('slippage_bps_per_side')} "
                        f"exp={row.get('expectancy')} pf={row.get('profit_factor')} ret_cv={row.get('ret_cv')} "
                        f"downside={row.get('downside_std')} dd={row.get('max_drawdown')} "
                        f"rob_v2={row.get('robustness_score_v2')}"
                    )
        lines.append("")

    lines.append("## Hold Bars Profile")
    lines.append(_markdown_table(holdbars_profile, max_rows=30))
    lines.append("")

    lines.append("## Dimension Sensitivity")
    lines.append(_markdown_table(dim_sensitivity, max_rows=30))
    lines.append("")

    lines.append("## Cost Fragility")
    lines.append(_markdown_table(cost_fragility, max_rows=30))
    lines.append("")

    lines.append("## Notes")
    lines.append("- D1-only, EOD signal, next-bar execution baseline.")
    lines.append("- No intraday fill model, optimizer, or corporate-actions model in this pack.")
    lines.append("- Robustness v2 is heuristic; use it for ranking, not as a statistical test.")
    lines.append("- `row_usage_rate` and `turnover_proxy` are MVP proxies, not full microstructure measures.")
    lines.append("")

    return "\n".join(lines) + "\n"


def render_grid_compare_report(
    *,
    summary: dict[str, Any],
    compare_table: pl.DataFrame,
) -> str:
    """Render markdown report for grid-vs-grid comparison."""

    lines: list[str] = []
    lines.append("# Backtest Grid Compare Report")
    lines.append("")
    lines.append(f"- compare_id: `{summary.get('compare_id')}`")
    lines.append(f"- grid_runs: `{len(summary.get('grid_run_dirs', []))}`")
    lines.append(f"- primary_metric: `{summary.get('primary_metric')}`")
    lines.append("")
    lines.append("## Comparison Table")
    lines.append(_markdown_table(compare_table, max_rows=40))
    lines.append("")
    lines.append("## Notes")
    lines.append("- Top-row metrics are taken from each run's best config by primary metric.")
    lines.append("- Overlap columns compare top-N parameter signatures between runs.")
    lines.append("- Overlay delta columns compare each run against the top-ranked run in this table.")
    lines.append("- `overlay_verdict` is heuristic: HELPFUL / NEUTRAL / HARMFUL.")
    lines.append(
        "- `realism_verdict` is heuristic: EDGE_RETAINED / EDGE_COMPROMISED_BUT_CLEANER / TOO_FRAGILE_UNDER_REALISM / NOT_TRADABLE."
    )
    lines.append("- Compare ret_cv / downside metrics and zero-trade rates before selecting candidates.")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_wf_grid_report(
    *,
    summary: dict[str, Any],
    by_split: pl.DataFrame,
    source_summary: pl.DataFrame,
    config_aggregate: pl.DataFrame,
    winner_stability: pl.DataFrame | None = None,
    cost_fragility_summary: pl.DataFrame | None = None,
    tail_risk_summary: pl.DataFrame | None = None,
    overlay_split_summary: pl.DataFrame | None = None,
    overlay_source_summary: pl.DataFrame | None = None,
    overlay_effectiveness_summary: pl.DataFrame | None = None,
) -> str:
    """Render markdown report for walk-forward sensitivity grid outputs."""

    lines: list[str] = []
    lines.append("# Backtest Walk-Forward Sensitivity Report")
    lines.append("")
    lines.append(f"- wf_grid_id: `{summary.get('wf_grid_id')}`")
    lines.append(f"- splits_total: `{summary.get('splits_total')}`")
    lines.append(f"- splits_successful: `{summary.get('splits_successful')}`")
    lines.append(f"- splits_failed: `{summary.get('splits_failed')}`")
    lines.append(f"- policy_filter_mode: `{summary.get('policy_filter_mode')}`")
    lines.append(f"- selected_train_ends: `{summary.get('selected_train_ends')}`")
    lines.append("")

    lines.append("## By Split")
    lines.append(_markdown_table(by_split, max_rows=60))
    lines.append("")

    lines.append("## Source Summary")
    lines.append(_markdown_table(source_summary, max_rows=30))
    lines.append("")

    lines.append("## Config Aggregate")
    lines.append(_markdown_table(config_aggregate, max_rows=40))
    lines.append("")

    if winner_stability is not None:
        lines.append("## Winner Stability")
        lines.append(_markdown_table(winner_stability, max_rows=60))
        lines.append("")

    if cost_fragility_summary is not None:
        lines.append("## Cost Fragility Summary")
        lines.append(_markdown_table(cost_fragility_summary, max_rows=30))
        lines.append("")

    if tail_risk_summary is not None:
        lines.append("## Tail Risk Summary")
        lines.append(_markdown_table(tail_risk_summary, max_rows=30))
        lines.append("")

    if overlay_split_summary is not None:
        lines.append("## Overlay Split Summary")
        lines.append(_markdown_table(overlay_split_summary, max_rows=60))
        lines.append("")
    if overlay_source_summary is not None:
        lines.append("## Overlay Source Summary")
        lines.append(_markdown_table(overlay_source_summary, max_rows=30))
        lines.append("")
    if overlay_effectiveness_summary is not None:
        lines.append("## Overlay Effectiveness Summary")
        lines.append(_markdown_table(overlay_effectiveness_summary, max_rows=30))
        lines.append("")

    lines.append("## Notes")
    lines.append("- Split-level grids are run independently; failures are captured and do not stop aggregation.")
    lines.append("- Cross-source comparability depends on row-universe alignment per split.")
    lines.append("")

    return "\n".join(lines) + "\n"
