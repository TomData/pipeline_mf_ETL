"""Markdown report builders for backtest harness outputs."""

from __future__ import annotations

from typing import Any

import polars as pl


def render_backtest_report(
    *,
    run_summary: dict[str, Any],
    summary_by_state: pl.DataFrame,
    summary_by_symbol: pl.DataFrame,
    policy_summary: dict[str, Any] | None,
    overlay_summary: dict[str, Any] | None,
    execution_summary: dict[str, Any] | None,
) -> str:
    """Render a single-run backtest markdown report."""

    lines: list[str] = []
    lines.append(f"# Backtest Report ({run_summary.get('run_id')})")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append(f"- input_type: {run_summary.get('input_type')}")
    lines.append(f"- input_file: `{run_summary.get('input_file')}`")
    lines.append(f"- signal_mode: {run_summary.get('signal_mode')}")
    lines.append(f"- exit_mode: {run_summary.get('exit_mode')}")
    lines.append(f"- hold_bars: {run_summary.get('hold_bars')}")
    lines.append(f"- allow_overlap: {run_summary.get('allow_overlap')}")
    lines.append(f"- allow_unconfirmed: {run_summary.get('allow_unconfirmed')}")
    lines.append(f"- fee_bps_per_side: {run_summary.get('fee_bps_per_side')}")
    lines.append(f"- slippage_bps_per_side: {run_summary.get('slippage_bps_per_side')}")
    lines.append("")

    head = run_summary.get("headline", {})
    lines.append("## Headline Metrics")
    for key in [
        "trade_count",
        "win_rate",
        "avg_return",
        "median_return",
        "profit_factor",
        "expectancy",
        "return_std",
        "avg_hold_bars",
        "skipped_signal_count",
        "invalid_trade_count",
    ]:
        lines.append(f"- {key}: {head.get(key)}")
    lines.append("")

    if policy_summary is not None:
        lines.append("## Cluster Policy Snapshot")
        lines.append(f"- allow_count: {policy_summary.get('allow_count')}")
        lines.append(f"- watch_count: {policy_summary.get('watch_count')}")
        lines.append(f"- block_count: {policy_summary.get('block_count')}")
        lines.append("")

    if overlay_summary is not None and bool(overlay_summary.get("overlay_enabled")):
        lines.append("## Hybrid Overlay")
        lines.append(f"- overlay_mode: {overlay_summary.get('overlay_mode')}")
        lines.append(f"- overlay_match_rate: {overlay_summary.get('overlay_match_rate')}")
        lines.append(f"- overlay_allow_rate: {overlay_summary.get('overlay_allow_rate')}")
        lines.append(f"- overlay_watch_rate: {overlay_summary.get('overlay_watch_rate')}")
        lines.append(f"- overlay_block_rate: {overlay_summary.get('overlay_block_rate')}")
        lines.append(f"- overlay_unknown_rate: {overlay_summary.get('overlay_unknown_rate')}")
        lines.append(f"- overlay_vetoed_signal_count: {overlay_summary.get('overlay_vetoed_signal_count')}")
        lines.append(f"- overlay_vetoed_signal_share: {overlay_summary.get('overlay_vetoed_signal_share')}")
        lines.append(f"- overlay_passed_signal_count: {overlay_summary.get('overlay_passed_signal_count')}")
        lines.append(
            f"- overlay_direction_conflict_share: {overlay_summary.get('overlay_direction_conflict_share')}"
        )
        lines.append("")

    if execution_summary is not None and bool(
        execution_summary.get("filters_enabled_or_profile_non_none")
    ):
        lines.append("## Execution Realism")
        lines.append(f"- execution_profile: {execution_summary.get('execution_profile')}")
        lines.append(f"- execution_filters_enabled: {execution_summary.get('execution_filters_enabled')}")
        lines.append(f"- vol_metric_source: {execution_summary.get('vol_metric_source')}")
        lines.append(f"- vol_unit_detected: {execution_summary.get('vol_unit_detected')}")
        lines.append(f"- vol_threshold_input: {execution_summary.get('vol_threshold_input')}")
        lines.append(
            f"- vol_threshold_effective_decimal: {execution_summary.get('vol_threshold_effective_decimal')}"
        )
        lines.append(
            f"- vol_threshold_effective_pct: {execution_summary.get('vol_threshold_effective_pct')}"
        )
        lines.append(f"- realism_profile_status: {execution_summary.get('realism_profile_status')}")
        lines.append(f"- exec_eligibility_rate: {execution_summary.get('exec_eligibility_rate')}")
        lines.append(
            f"- exec_suppressed_signal_share: {execution_summary.get('exec_suppressed_signal_share')}"
        )
        lines.append(
            f"- exec_suppressed_by_price_count: {execution_summary.get('exec_suppressed_by_price_count')}"
        )
        lines.append(
            f"- exec_suppressed_by_liquidity_count: {execution_summary.get('exec_suppressed_by_liquidity_count')}"
        )
        lines.append(
            f"- exec_suppressed_by_vol_count: {execution_summary.get('exec_suppressed_by_vol_count')}"
        )
        lines.append(
            f"- exec_suppressed_by_warmup_count: {execution_summary.get('exec_suppressed_by_warmup_count')}"
        )
        lines.append(
            f"- exec_suppressed_by_price_share: {execution_summary.get('exec_suppressed_by_price_share')}"
        )
        lines.append(
            f"- exec_suppressed_by_liquidity_share: {execution_summary.get('exec_suppressed_by_liquidity_share')}"
        )
        lines.append(f"- exec_suppressed_by_vol_share: {execution_summary.get('exec_suppressed_by_vol_share')}")
        lines.append(
            f"- exec_suppressed_by_warmup_share: {execution_summary.get('exec_suppressed_by_warmup_share')}"
        )
        lines.append(
            f"- exec_trade_avg_dollar_vol_20: {execution_summary.get('exec_trade_avg_dollar_vol_20')}"
        )
        lines.append(
            f"- exec_trade_p10_dollar_vol_20: {execution_summary.get('exec_trade_p10_dollar_vol_20')}"
        )
        lines.append(f"- exec_trade_avg_vol_pct: {execution_summary.get('exec_trade_avg_vol_pct')}")
        lines.append("")

    lines.append("## By-State (Top 20 by trade_count)")
    lines.append("| state_id | class | direction | trade_count | win_rate | avg_return | expectancy |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for row in summary_by_state.head(20).to_dicts():
        lines.append(
            f"| {row.get('entry_state_id')} | {row.get('entry_state_class')} | {row.get('entry_state_direction_hint')} | "
            f"{row.get('trade_count')} | {row.get('win_rate')} | {row.get('avg_return')} | {row.get('expectancy')} |"
        )
    lines.append("")

    lines.append("## By-Symbol (Top 20 by net contribution)")
    lines.append("| ticker | trade_count | win_rate | avg_return | contribution |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in summary_by_symbol.head(20).to_dicts():
        lines.append(
            f"| {row.get('ticker')} | {row.get('trade_count')} | {row.get('win_rate')} | {row.get('avg_return')} | {row.get('contribution')} |"
        )
    lines.append("")

    lines.append("## MVP Assumptions")
    lines.append("- D1 only, EOD signal with next-bar execution.")
    lines.append("- No intraday fill model, no optimizer, no position sizing optimization.")
    lines.append("- Slippage/fees default to zero unless configured.")
    lines.append("- Research baseline only; not a live execution engine.")
    lines.append("")
    return "\n".join(lines)


def render_backtest_compare_report(*, summary: dict[str, Any], compare_table: pl.DataFrame) -> str:
    """Render compare report markdown."""

    lines: list[str] = []
    lines.append(f"# Backtest Compare Report ({summary.get('compare_id')})")
    lines.append("")
    lines.append("## Runs")
    for path in summary.get("run_dirs", []):
        lines.append(f"- `{path}`")
    lines.append("")

    lines.append("## Table")
    lines.append("| run_id | input_type | trade_count | win_rate | avg_return | median_return | profit_factor | expectancy | max_drawdown | overlay_mode | overlay_match_rate | overlay_vetoed_signal_share | execution_profile | exec_eligibility_rate | exec_suppressed_signal_share |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|")
    for row in compare_table.to_dicts():
        lines.append(
            f"| {row.get('run_id')} | {row.get('input_type')} | {row.get('trade_count')} | {row.get('win_rate')} | "
            f"{row.get('avg_return')} | {row.get('median_return')} | {row.get('profit_factor')} | {row.get('expectancy')} | {row.get('max_drawdown')} | "
            f"{row.get('overlay_mode')} | {row.get('overlay_match_rate')} | {row.get('overlay_vetoed_signal_share')} | "
            f"{row.get('execution_profile')} | {row.get('exec_eligibility_rate')} | {row.get('exec_suppressed_signal_share')} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- Cluster runs may be ALLOW-only depending on cluster hardening policy.")
    lines.append("- Compare settings should use same hold/signal/cost assumptions.")
    lines.append("")
    return "\n".join(lines)


def render_wf_backtest_report(
    *,
    aggregate_summary: dict[str, Any],
    by_split: pl.DataFrame,
    model_summary: pl.DataFrame,
) -> str:
    """Render walk-forward backtest markdown report."""

    lines: list[str] = []
    lines.append(f"# Walk-Forward Backtest Report ({aggregate_summary.get('wf_bt_id')})")
    lines.append("")
    lines.append("## Topline")
    lines.append(f"- splits_total: {aggregate_summary.get('splits_total')}")
    lines.append(f"- splits_successful: {aggregate_summary.get('splits_successful')}")
    lines.append(f"- splits_failed: {aggregate_summary.get('splits_failed')}")
    lines.append("")

    lines.append("## Per-Split Metrics")
    lines.append("| train_end | model | trade_count | win_rate | avg_return | profit_factor | expectancy |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in by_split.to_dicts():
        lines.append(
            f"| {row.get('train_end')} | {row.get('model')} | {row.get('trade_count')} | {row.get('win_rate')} | "
            f"{row.get('avg_return')} | {row.get('profit_factor')} | {row.get('expectancy')} |"
        )
    lines.append("")

    lines.append("## Model Aggregate")
    lines.append("| model | splits | trade_count_sum | win_rate_mean | avg_return_mean | expectancy_mean | profit_factor_mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in model_summary.to_dicts():
        lines.append(
            f"| {row.get('model')} | {row.get('splits')} | {row.get('trade_count_sum')} | {row.get('win_rate_mean')} | "
            f"{row.get('avg_return_mean')} | {row.get('expectancy_mean')} | {row.get('profit_factor_mean')} |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("- Flow backtests use aligned split HMM rows when flow_state_code is present.")
    lines.append("- Cluster backtests use split cluster hardening policy outputs.")
    lines.append("- Next step: sensitivity runs for costs and hold-bars.")
    lines.append("")
    return "\n".join(lines)
