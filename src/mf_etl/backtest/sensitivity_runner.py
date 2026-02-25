"""Orchestration for backtest sensitivity grids, comparisons, and walk-forward runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import traceback
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from mf_etl.backtest.models import InputType
from mf_etl.backtest.pipeline import run_backtest_run
from mf_etl.backtest.sensitivity_aggregate import (
    best_configs_by_metric,
    build_cost_fragility,
    build_dimension_sensitivity,
    build_holdbars_profile,
    classify_hold_shape,
    build_source_summary,
    classify_universe_comparability,
    compute_metrics_table,
)
from mf_etl.backtest.sensitivity_grid import build_grid_combinations, combo_id, default_dimensions_from_settings
from mf_etl.backtest.sensitivity_models import (
    GridCompareResult,
    GridDimensionValues,
    GridRunResult,
    GridWalkForwardResult,
    PolicyFilterMode,
    SourceInputSpec,
)
from mf_etl.backtest.sensitivity_reports import (
    render_grid_compare_report,
    render_grid_report,
    render_wf_grid_report,
)
from mf_etl.backtest.writer import (
    write_csv_atomically,
    write_json_atomically,
    write_markdown_atomically,
    write_parquet_atomically,
)
from mf_etl.config import AppSettings
from mf_etl.validation.cluster_hardening import run_cluster_hardening_single

LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _finite_json(payload: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        return value

    return convert(payload)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_quantile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return _safe_float(float(np.quantile(values, q)))


def _trade_edge_metrics(
    *,
    trades_path: Path,
    rows_input: int,
    rows_eligible: int,
    eps: float = 1e-12,
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    """Extract finite-safe trade distribution metrics from one backtest run."""

    if not trades_path.exists():
        return (
            {
                "ret_cv": None,
                "ret_p10": None,
                "ret_p90": None,
                "downside_std": None,
                "worst_trade_return": None,
                "best_trade_return": None,
                "trades_per_1000_rows": None,
                "row_usage_rate": None,
                "turnover_proxy": None,
                "is_zero_trade_combo": True,
                "null_metric_count": 10,
            },
            [],
            None,
        )

    trades = pl.read_parquet(trades_path)
    if trades.height == 0:
        return (
            {
                "ret_cv": None,
                "ret_p10": None,
                "ret_p90": None,
                "downside_std": None,
                "worst_trade_return": None,
                "best_trade_return": None,
                "trades_per_1000_rows": _safe_float(0.0 if rows_input > 0 else None),
                "row_usage_rate": _safe_float(0.0 if rows_eligible > 0 else None),
                "turnover_proxy": None,
                "is_zero_trade_combo": True,
                "null_metric_count": 7,
            },
            [],
            None,
        )

    valid = trades.filter(pl.col("is_valid_trade") == True)
    if valid.height == 0:
        return (
            {
                "ret_cv": None,
                "ret_p10": None,
                "ret_p90": None,
                "downside_std": None,
                "worst_trade_return": None,
                "best_trade_return": None,
                "trades_per_1000_rows": _safe_float(0.0 if rows_input > 0 else None),
                "row_usage_rate": _safe_float(0.0 if rows_eligible > 0 else None),
                "turnover_proxy": None,
                "is_zero_trade_combo": True,
                "null_metric_count": 7,
            },
            [],
            None,
        )

    returns = valid.get_column("net_return").cast(pl.Float64, strict=False).to_numpy()
    returns = returns[np.isfinite(returns)]
    trade_count = int(returns.size)
    mean_ret = _safe_float(float(np.mean(returns)) if trade_count > 0 else None)
    std_ret = _safe_float(float(np.std(returns, ddof=0)) if trade_count > 0 else None)
    if mean_ret is None or std_ret is None or abs(mean_ret) < eps:
        ret_cv = None
    else:
        ret_cv = _safe_float(std_ret / abs(mean_ret))

    neg = returns[returns < 0]
    downside_std = _safe_float(float(np.std(neg, ddof=0)) if neg.size > 0 else None)
    avg_hold = _safe_float(valid.select(pl.col("hold_bars_realized").mean()).item())
    turnover_proxy = _safe_float((trade_count / avg_hold) if (avg_hold is not None and avg_hold > 0) else None)

    trade_mix = None
    if "entry_state_class" in valid.columns:
        counts = (
            valid.group_by("entry_state_class")
            .agg(pl.len().alias("trade_count"))
            .sort("entry_state_class")
            .to_dicts()
        )
        if counts:
            trade_mix = ";".join(f"{row.get('entry_state_class')}={row.get('trade_count')}" for row in counts)

    state_rows: list[dict[str, Any]] = []
    if "entry_state_id" in valid.columns:
        for row in valid.group_by(
            ["entry_state_id", "entry_state_label", "entry_state_class", "entry_state_direction_hint"]
        ).agg(pl.col("net_return").cast(pl.Float64, strict=False).alias("rets")).to_dicts():
            rets = np.array([float(v) for v in row.get("rets", []) if v is not None and np.isfinite(v)], dtype=np.float64)
            if rets.size == 0:
                continue
            mean_v = float(np.mean(rets))
            std_v = float(np.std(rets, ddof=0))
            cv_v = None if abs(mean_v) < eps else float(std_v / abs(mean_v))
            state_rows.append(
                {
                    "entry_state_id": int(row.get("entry_state_id")),
                    "entry_state_label": row.get("entry_state_label"),
                    "entry_state_class": row.get("entry_state_class"),
                    "entry_state_direction_hint": row.get("entry_state_direction_hint"),
                    "trade_count": int(rets.size),
                    "avg_return": _safe_float(mean_v),
                    "median_return": _safe_float(float(np.median(rets))),
                    "ret_cv": _safe_float(cv_v),
                    "ret_p10": _safe_float(float(np.quantile(rets, 0.10))),
                    "ret_p90": _safe_float(float(np.quantile(rets, 0.90))),
                    "downside_std": _safe_float(float(np.std(rets[rets < 0], ddof=0)) if np.any(rets < 0) else None),
                    "worst_trade_return": _safe_float(float(np.min(rets))),
                    "best_trade_return": _safe_float(float(np.max(rets))),
                }
            )

    out = {
        "ret_cv": ret_cv,
        "ret_p10": _safe_quantile(returns, 0.10),
        "ret_p90": _safe_quantile(returns, 0.90),
        "downside_std": downside_std,
        "worst_trade_return": _safe_float(float(np.min(returns)) if trade_count > 0 else None),
        "best_trade_return": _safe_float(float(np.max(returns)) if trade_count > 0 else None),
        "trades_per_1000_rows": _safe_float((trade_count / rows_input) * 1000.0 if rows_input > 0 else None),
        # MVP simplification: usage proxy approximates used opportunities via realized trade count.
        "row_usage_rate": _safe_float((trade_count / rows_eligible) if rows_eligible > 0 else None),
        # MVP simplification: turnover proxy from count and average hold.
        "turnover_proxy": turnover_proxy,
        "is_zero_trade_combo": trade_count == 0,
    }
    out["null_metric_count"] = int(sum(1 for key in out if key != "is_zero_trade_combo" and out[key] is None))
    return out, state_rows, trade_mix


def _parse_date_like(raw: str) -> tuple[int, int, int]:
    parts = raw.strip().split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid YYYY-MM-DD date: {raw}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _generate_train_ends(train_start: str, train_end_final: str, step_years: int) -> list[str]:
    sy, sm, sd = _parse_date_like(train_start)
    ey, em, ed = _parse_date_like(train_end_final)
    if step_years <= 0:
        raise ValueError("step_years must be >= 1")
    out: list[str] = []
    year = sy
    while True:
        current = f"{year:04d}-{sm:02d}-{sd:02d}"
        out.append(current)
        if year >= ey:
            break
        year += step_years
        if year > ey:
            final = f"{ey:04d}-{em:02d}-{ed:02d}"
            if out[-1] != final:
                out.append(final)
            break
    return out


def _scope_name(source_specs: list[SourceInputSpec]) -> str:
    if len(source_specs) == 1:
        return f"single-{source_specs[0].source_type}"
    ordered = "-".join(sorted(spec.source_type for spec in source_specs))
    return f"multi-{ordered}"


def _resolve_grid_output_dir(
    settings: AppSettings,
    *,
    scope: str,
    tag: str,
    force: bool,
    output_dir_override: Path | None,
    grid_run_id_override: str | None,
) -> tuple[str, Path]:
    if output_dir_override is not None and grid_run_id_override is not None:
        return grid_run_id_override, output_dir_override

    root = settings.paths.artifacts_root / "backtest_sensitivity"
    root.mkdir(parents=True, exist_ok=True)
    grid_run_id = f"grid-{uuid4().hex[:12]}"
    output_dir = root / f"{grid_run_id}_{scope}_{tag}"

    if force:
        existing = sorted(root.glob(f"*_{scope}_{tag}"), key=lambda p: p.stat().st_mtime)
        if existing:
            output_dir = existing[-1]
            grid_run_id = output_dir.name.split("_", 1)[0]
    return grid_run_id, output_dir


def _to_manifest_row(
    *,
    grid_run_id: str,
    combo_idx: int,
    source_spec: SourceInputSpec,
    combo: Any,
    status: str,
    error_message: str | None,
    backtest_run_dir: Path | None,
    summary_payload: dict[str, Any] | None,
    edge_metrics: dict[str, Any] | None,
    policy_trade_class_mix: str | None,
) -> dict[str, Any]:
    headline = (summary_payload or {}).get("headline", {})
    equity = (summary_payload or {}).get("equity_metrics") or {}
    adapter = (summary_payload or {}).get("adapter_summary", {})
    mapping = (summary_payload or {}).get("mapping_summary", {})
    overlay = (summary_payload or {}).get("overlay", {}) if isinstance((summary_payload or {}).get("overlay"), dict) else {}
    execution = (
        (summary_payload or {}).get("execution", {})
        if isinstance((summary_payload or {}).get("execution"), dict)
        else {}
    )

    def metric(name: str) -> float | None:
        return _safe_float(headline.get(name))

    non_finite_detected = False
    for value in [
        metric("avg_return"),
        metric("median_return"),
        metric("profit_factor"),
        metric("expectancy"),
        metric("return_std"),
        _safe_float(equity.get("max_drawdown")),
        _safe_float(equity.get("sharpe_proxy")),
    ]:
        if value is None:
            continue
        if not np.isfinite(value):
            non_finite_detected = True
            break

    rows_in = int(adapter.get("rows_in", 0) or 0)
    rows_eligible = int(mapping.get("eligible_rows", 0) or 0)
    rows_skipped = int(adapter.get("rows_dropped_missing_essential", 0) or 0) + int(
        adapter.get("rows_dropped_bad_price", 0) or 0
    ) + int(adapter.get("rows_deduped_ticker_date", 0) or 0)

    return {
        "grid_run_id": grid_run_id,
        "combo_id": combo_id(source_spec.source_type, combo),
        "combo_index": combo_idx,
        "source_type": source_spec.source_type,
        "status": status,
        "error_message": error_message,
        "backtest_run_dir": str(backtest_run_dir) if backtest_run_dir is not None else None,
        "hold_bars": combo.hold_bars,
        "signal_mode": combo.signal_mode,
        "exit_mode": combo.exit_mode,
        "fee_bps_per_side": combo.fee_bps_per_side,
        "slippage_bps_per_side": combo.slippage_bps_per_side,
        "allow_overlap": combo.allow_overlap,
        "equity_mode": combo.equity_mode,
        "include_watch": (combo.include_watch if source_spec.source_type == "cluster" else False),
        "policy_filter_mode": (source_spec.policy_filter_mode if source_spec.source_type == "cluster" else None),
        "state_subset_key": combo.state_subset_key,
        "input_file": str(source_spec.input_file),
        "validation_run_dir": str(source_spec.validation_run_dir) if source_spec.validation_run_dir else None,
        "cluster_hardening_dir": (
            str(source_spec.cluster_hardening_dir) if source_spec.cluster_hardening_dir else None
        ),
        "trade_count": int(headline.get("trade_count", 0) or 0),
        "win_rate": metric("win_rate"),
        "avg_return": metric("avg_return"),
        "median_return": metric("median_return"),
        "profit_factor": metric("profit_factor"),
        "expectancy": metric("expectancy"),
        "return_std": metric("return_std"),
        "avg_hold_bars": metric("avg_hold_bars"),
        "max_drawdown": _safe_float(equity.get("max_drawdown")),
        "sharpe_proxy": _safe_float(equity.get("sharpe_proxy")),
        "nan_warning_total": 1 if non_finite_detected else 0,
        "rows_input": rows_in,
        "rows_eligible": rows_eligible,
        "rows_skipped": rows_skipped,
        "ret_cv": (edge_metrics or {}).get("ret_cv"),
        "ret_p10": (edge_metrics or {}).get("ret_p10"),
        "ret_p90": (edge_metrics or {}).get("ret_p90"),
        "downside_std": (edge_metrics or {}).get("downside_std"),
        "worst_trade_return": (edge_metrics or {}).get("worst_trade_return"),
        "best_trade_return": (edge_metrics or {}).get("best_trade_return"),
        "trades_per_1000_rows": (edge_metrics or {}).get("trades_per_1000_rows"),
        "row_usage_rate": (edge_metrics or {}).get("row_usage_rate"),
        "turnover_proxy": (edge_metrics or {}).get("turnover_proxy"),
        "is_zero_trade_combo": bool((edge_metrics or {}).get("is_zero_trade_combo", True)),
        "null_metric_count": int((edge_metrics or {}).get("null_metric_count", 0)),
        "policy_trade_class_mix": policy_trade_class_mix,
        "overlay_enabled": bool(
            overlay.get("overlay_enabled", False)
            or (source_spec.overlay_cluster_file is not None and source_spec.overlay_cluster_hardening_dir is not None)
        ),
        "overlay_mode": str(overlay.get("overlay_mode") or source_spec.overlay_mode),
        "overlay_match_rate": _safe_float(overlay.get("overlay_match_rate")),
        "overlay_unknown_rate": _safe_float(overlay.get("overlay_unknown_rate")),
        "overlay_allow_rate": _safe_float(overlay.get("overlay_allow_rate")),
        "overlay_watch_rate": _safe_float(overlay.get("overlay_watch_rate")),
        "overlay_block_rate": _safe_float(overlay.get("overlay_block_rate")),
        "overlay_vetoed_signal_count": int(overlay.get("overlay_vetoed_signal_count", 0) or 0),
        "overlay_vetoed_signal_share": _safe_float(overlay.get("overlay_vetoed_signal_share")),
        "overlay_passed_signal_count": int(overlay.get("overlay_passed_signal_count", 0) or 0),
        "overlay_direction_conflict_share": _safe_float(
            overlay.get("overlay_direction_conflict_share")
        ),
        "execution_profile": execution.get("execution_profile"),
        "execution_filters_enabled": bool(execution.get("execution_filters_enabled", False)),
        "exec_vol_metric_source": execution.get("vol_metric_source"),
        "exec_vol_unit_detected": execution.get("vol_unit_detected"),
        "exec_vol_threshold_input": _safe_float(execution.get("vol_threshold_input")),
        "exec_vol_threshold_effective_decimal": _safe_float(
            execution.get("vol_threshold_effective_decimal")
        ),
        "exec_vol_threshold_effective_pct": _safe_float(
            execution.get("vol_threshold_effective_pct")
        ),
        "realism_profile_status": execution.get("realism_profile_status"),
        "exec_candidate_signals_before_filters": int(
            execution.get("candidate_signals_before_filters", 0) or 0
        ),
        "exec_candidate_signals_after_filters": int(
            execution.get("candidate_signals_after_filters", 0) or 0
        ),
        "exec_suppressed_signal_count": int(execution.get("exec_suppressed_signal_count", 0) or 0),
        "exec_eligibility_rate": _safe_float(execution.get("exec_eligibility_rate")),
        "exec_suppressed_signal_share": _safe_float(execution.get("exec_suppressed_signal_share")),
        "exec_suppressed_by_price_count": int(execution.get("exec_suppressed_by_price_count", 0) or 0),
        "exec_suppressed_by_liquidity_count": int(
            execution.get("exec_suppressed_by_liquidity_count", 0) or 0
        ),
        "exec_suppressed_by_vol_count": int(execution.get("exec_suppressed_by_vol_count", 0) or 0),
        "exec_suppressed_by_warmup_count": int(execution.get("exec_suppressed_by_warmup_count", 0) or 0),
        "exec_suppressed_by_liquidity_share": _safe_float(
            execution.get("exec_suppressed_by_liquidity_share")
        ),
        "exec_suppressed_by_price_share": _safe_float(execution.get("exec_suppressed_by_price_share")),
        "exec_suppressed_by_vol_share": _safe_float(execution.get("exec_suppressed_by_vol_share")),
        "exec_suppressed_by_warmup_share": _safe_float(execution.get("exec_suppressed_by_warmup_share")),
        "exec_trade_avg_dollar_vol_20": _safe_float(execution.get("exec_trade_avg_dollar_vol_20")),
        "exec_trade_p10_dollar_vol_20": _safe_float(execution.get("exec_trade_p10_dollar_vol_20")),
        "exec_trade_avg_vol_pct": _safe_float(execution.get("exec_trade_avg_vol_pct")),
        "built_ts": (summary_payload or {}).get("finished_ts"),
    }


def _ensure_cluster_hardening_dir(
    *,
    settings: AppSettings,
    source_spec: SourceInputSpec,
    inferred_validation_dir: Path | None,
    logger: logging.Logger,
) -> SourceInputSpec:
    if source_spec.source_type != "cluster":
        return source_spec
    if source_spec.cluster_hardening_dir is not None and source_spec.cluster_hardening_dir.exists():
        return source_spec
    if inferred_validation_dir is None:
        raise FileNotFoundError(
            "Cluster source requires --cluster-hardening-dir or inferable validation dir for auto hardening build"
        )
    output_dir = inferred_validation_dir / "cluster_hardening"
    run_cluster_hardening_single(
        settings,
        validation_run_dir=inferred_validation_dir,
        clustered_rows_file=source_spec.input_file if source_spec.input_file.exists() else None,
        export_filtered=False,
        output_dir=output_dir,
        force=False,
        logger=logger,
    )
    return SourceInputSpec(
        source_type=source_spec.source_type,
        input_file=source_spec.input_file,
        validation_run_dir=source_spec.validation_run_dir,
        cluster_hardening_dir=output_dir,
        state_map_file=source_spec.state_map_file,
        policy_filter_mode=source_spec.policy_filter_mode,
        overlay_cluster_file=source_spec.overlay_cluster_file,
        overlay_cluster_hardening_dir=source_spec.overlay_cluster_hardening_dir,
        overlay_mode=source_spec.overlay_mode,
        overlay_join_keys=source_spec.overlay_join_keys,
    )


def run_backtest_grid(
    settings: AppSettings,
    *,
    source_specs: list[SourceInputSpec],
    dimensions: GridDimensionValues | None,
    tag: str | None,
    max_combos: int | None,
    shuffle_grid: bool,
    seed: int,
    progress_every: int | None,
    stop_on_error: bool,
    force: bool,
    write_run_manifest: bool,
    include_ret_cv: bool = True,
    include_tail_metrics: bool = True,
    report_top_n: int = 10,
    execution_profile: str = "none",
    exec_min_price: float | None = None,
    exec_min_dollar_vol20: float | None = None,
    exec_max_vol_pct: float | None = None,
    exec_min_history_bars: int | None = None,
    report_min_trades: int | None = None,
    report_max_zero_trade_share: float | None = None,
    report_max_ret_cv: float | None = None,
    logger: logging.Logger | None = None,
    output_dir_override: Path | None = None,
    grid_run_id_override: str | None = None,
) -> GridRunResult:
    """Execute one sensitivity grid across one or many sources."""

    effective_logger = logger or LOGGER
    if not source_specs:
        raise ValueError("No source specs provided to run_backtest_grid")

    resolved_dimensions = dimensions or default_dimensions_from_settings(settings)
    combos = build_grid_combinations(
        resolved_dimensions,
        max_combos=max_combos or settings.backtest_sensitivity.max_combos,
        shuffle_grid=shuffle_grid,
        seed=seed,
    )
    if not combos:
        raise ValueError("Grid has zero combinations after applying limits")

    scope = _scope_name(source_specs)
    safe_tag = (tag or "default").replace(" ", "-")
    grid_run_id, output_dir = _resolve_grid_output_dir(
        settings,
        scope=scope,
        tag=safe_tag,
        force=force,
        output_dir_override=output_dir_override,
        grid_run_id_override=grid_run_id_override,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cluster_mix_rows: list[dict[str, Any]] = []
    cluster_state_profile_rows: list[dict[str, Any]] = []
    resolved_source_specs: dict[tuple[str, str], SourceInputSpec] = {}

    total = len(combos) * len(source_specs)
    processed = 0
    progress_n = progress_every or settings.backtest_sensitivity.progress_every

    for spec in source_specs:
        actual_spec = spec
        if spec.source_type == "cluster":
            inferred_val = spec.validation_run_dir
            actual_spec = _ensure_cluster_hardening_dir(
                settings=settings,
                source_spec=spec,
                inferred_validation_dir=inferred_val,
                logger=effective_logger,
            )
        resolved_source_specs[(actual_spec.source_type, str(actual_spec.input_file))] = actual_spec

        for idx, combo in enumerate(combos, start=1):
            processed += 1
            backtest_run_dir: Path | None = None
            summary_payload: dict[str, Any] | None = None
            edge_metrics: dict[str, Any] | None = None
            policy_trade_class_mix: str | None = None
            status = "SUCCESS"
            error_message: str | None = None

            try:
                run_tag = f"{safe_tag}-{grid_run_id}-{actual_spec.source_type}-c{idx:04d}"
                result = run_backtest_run(
                    settings,
                    input_type=actual_spec.source_type,
                    input_file=actual_spec.input_file,
                    validation_run_dir=actual_spec.validation_run_dir,
                    cluster_hardening_dir=actual_spec.cluster_hardening_dir,
                    state_map_file=actual_spec.state_map_file,
                    signal_mode=combo.signal_mode,
                    exit_mode=combo.exit_mode,
                    hold_bars=combo.hold_bars,
                    allow_overlap=combo.allow_overlap,
                    allow_unconfirmed=settings.backtest.allow_unconfirmed,
                    include_watch=(combo.include_watch if actual_spec.source_type == "cluster" else False),
                    policy_filter_mode=actual_spec.policy_filter_mode if actual_spec.source_type == "cluster" else "allow_only",
                    include_state_ids=combo.include_state_ids,
                    overlay_cluster_file=actual_spec.overlay_cluster_file,
                    overlay_cluster_hardening_dir=actual_spec.overlay_cluster_hardening_dir,
                    overlay_mode=actual_spec.overlay_mode,
                    overlay_join_keys=actual_spec.overlay_join_keys,
                    execution_profile=execution_profile,  # type: ignore[arg-type]
                    exec_min_price=exec_min_price,
                    exec_min_dollar_vol20=exec_min_dollar_vol20,
                    exec_max_vol_pct=exec_max_vol_pct,
                    exec_min_history_bars=exec_min_history_bars,
                    report_min_trades=report_min_trades,
                    report_max_zero_trade_share=report_max_zero_trade_share,
                    report_max_ret_cv=report_max_ret_cv,
                    fee_bps_per_side=combo.fee_bps_per_side,
                    slippage_bps_per_side=combo.slippage_bps_per_side,
                    equity_mode=combo.equity_mode,
                    export_joined_rows=False,
                    tag=run_tag,
                    force=False,
                    logger=effective_logger,
                )
                backtest_run_dir = result.output_dir
                summary_payload = _load_json(result.summary_path)
                payload_adapter = summary_payload.get("adapter_summary", {}) if summary_payload else {}
                payload_mapping = summary_payload.get("mapping_summary", {}) if summary_payload else {}
                edge_metrics, state_rows, policy_trade_class_mix = _trade_edge_metrics(
                    trades_path=result.trades_path,
                    rows_input=int(payload_adapter.get("rows_in", 0) or 0),
                    rows_eligible=int(payload_mapping.get("eligible_rows", 0) or 0),
                )
                if actual_spec.source_type == "cluster":
                    cluster_mix_rows.append(
                        {
                            "combo_id": combo_id(actual_spec.source_type, combo),
                            "combo_index": idx,
                            "policy_filter_mode": actual_spec.policy_filter_mode,
                            "policy_trade_class_mix": policy_trade_class_mix,
                            "trade_count": int(summary_payload.get("headline", {}).get("trade_count", 0) or 0),
                        }
                    )
                    for row in state_rows:
                        row["combo_id"] = combo_id(actual_spec.source_type, combo)
                        row["combo_index"] = idx
                        row["policy_filter_mode"] = actual_spec.policy_filter_mode
                        cluster_state_profile_rows.append(row)
            except Exception as exc:
                status = "FAILED"
                error_message = str(exc)
                tb = traceback.format_exc(limit=8)
                failures.append(
                    {
                        "grid_run_id": grid_run_id,
                        "source_type": actual_spec.source_type,
                        "combo_index": idx,
                        "combo_id": combo_id(actual_spec.source_type, combo),
                        "error_message": error_message,
                        "traceback": tb,
                    }
                )
                if stop_on_error:
                    raise

            manifest_rows.append(
                _to_manifest_row(
                    grid_run_id=grid_run_id,
                    combo_idx=idx,
                    source_spec=actual_spec,
                    combo=combo,
                    status=status,
                    error_message=error_message,
                    backtest_run_dir=backtest_run_dir,
                    summary_payload=summary_payload,
                    edge_metrics=edge_metrics if include_ret_cv or include_tail_metrics else None,
                    policy_trade_class_mix=policy_trade_class_mix,
                )
            )

            if processed % max(1, progress_n) == 0:
                succ = sum(1 for row in manifest_rows if row["status"] == "SUCCESS")
                fail = sum(1 for row in manifest_rows if row["status"] == "FAILED")
                effective_logger.info(
                    "backtest.grid.progress grid_run_id=%s processed=%s/%s success=%s failed=%s",
                    grid_run_id,
                    processed,
                    total,
                    succ,
                    fail,
                )

    manifest = pl.DataFrame(manifest_rows) if manifest_rows else pl.DataFrame()
    if manifest.height > 0:
        manifest = manifest.sort(["source_type", "combo_index"])

    metrics_table = compute_metrics_table(
        manifest,
        weights=settings.backtest_sensitivity.robustness_score_weights,
    )
    if not include_ret_cv:
        metrics_table = metrics_table.with_columns(
            pl.lit(None).cast(pl.Float64).alias("ret_cv"),
        )
    if not include_tail_metrics:
        metrics_table = metrics_table.with_columns(
            pl.lit(None).cast(pl.Float64).alias("ret_p10"),
            pl.lit(None).cast(pl.Float64).alias("ret_p90"),
            pl.lit(None).cast(pl.Float64).alias("downside_std"),
            pl.lit(None).cast(pl.Float64).alias("worst_trade_return"),
            pl.lit(None).cast(pl.Float64).alias("best_trade_return"),
        )

    dim_sensitivity = build_dimension_sensitivity(metrics_table, manifest=manifest)
    cost_fragility = build_cost_fragility(metrics_table)
    holdbars_profile = build_holdbars_profile(metrics_table)
    hold_shape = classify_hold_shape(holdbars_profile, metric_col="expectancy_mean")
    source_summary = build_source_summary(metrics_table, manifest)
    if hold_shape.height > 0 and source_summary.height > 0:
        source_summary = source_summary.join(hold_shape, on="source_type", how="left")
    best_payload = best_configs_by_metric(metrics_table, top_n=max(1, int(report_top_n)))
    comparability = classify_universe_comparability(metrics_table)

    total_combos = int(manifest.height)
    successful = int(manifest.filter(pl.col("status") == "SUCCESS").height) if manifest.height > 0 else 0
    failed = int(manifest.filter(pl.col("status") == "FAILED").height) if manifest.height > 0 else 0
    skipped = int(manifest.filter(pl.col("status") == "SKIPPED").height) if manifest.height > 0 else 0
    zero_trade_combos = (
        int(metrics_table.filter(pl.col("is_zero_trade_combo") == True).height) if metrics_table.height > 0 else 0
    )
    realism_profile_broken_for_universe = False
    if execution_profile != "none" and metrics_table.height > 0:
        success_df = metrics_table.filter(pl.col("status") == "SUCCESS")
        if success_df.height > 0:
            with_candidate = success_df.filter(
                pl.col("exec_candidate_signals_before_filters").cast(pl.Int64, strict=False) > 0
            )
            if with_candidate.height > 0:
                zero_after = int(
                    with_candidate.filter(
                        pl.col("exec_candidate_signals_after_filters").cast(pl.Int64, strict=False) <= 0
                    ).height
                )
                realism_profile_broken_for_universe = (zero_after == with_candidate.height)
    non_finite_cells = 0
    null_metric_cells = 0
    if metrics_table.height > 0:
        for col, dtype in metrics_table.schema.items():
            if dtype.is_numeric():
                non_finite_cells += int(
                    metrics_table.select(
                        (
                            (~pl.col(col).cast(pl.Float64, strict=False).is_finite())
                            & pl.col(col).cast(pl.Float64, strict=False).is_not_null()
                        ).sum()
                    ).item()
                )
                null_metric_cells += int(metrics_table.select(pl.col(col).is_null().sum()).item())

    summary_payload = {
        "grid_run_id": grid_run_id,
        "scope": scope,
        "tag": safe_tag,
        "comparability": comparability,
        "total_combos": total_combos,
        "successful_combos": successful,
        "failed_combos": failed,
        "skipped_combos": skipped,
        "zero_trade_combos": zero_trade_combos,
        "zero_trade_combo_share": _safe_float((zero_trade_combos / successful) if successful > 0 else None),
        "realism_profile_broken_for_universe": realism_profile_broken_for_universe,
        "non_finite_cells": non_finite_cells,
        "null_metric_cells": null_metric_cells,
        "sources": [
            {
                "source_type": spec.source_type,
                "input_file": str(spec.input_file),
                "validation_run_dir": str(spec.validation_run_dir) if spec.validation_run_dir else None,
                "cluster_hardening_dir": str(spec.cluster_hardening_dir) if spec.cluster_hardening_dir else None,
                "policy_filter_mode": spec.policy_filter_mode if spec.source_type == "cluster" else None,
                "overlay_cluster_file": str(spec.overlay_cluster_file) if spec.overlay_cluster_file else None,
                "overlay_cluster_hardening_dir": (
                    str(spec.overlay_cluster_hardening_dir) if spec.overlay_cluster_hardening_dir else None
                ),
                "overlay_mode": spec.overlay_mode,
                "overlay_join_keys": spec.overlay_join_keys,
            }
            for spec in resolved_source_specs.values()
        ],
        "dimensions": {
            "hold_bars": resolved_dimensions.hold_bars,
            "signal_mode": list(resolved_dimensions.signal_mode),
            "exit_mode": list(resolved_dimensions.exit_mode),
            "fee_bps_per_side": resolved_dimensions.fee_bps_per_side,
            "slippage_bps_per_side": resolved_dimensions.slippage_bps_per_side,
            "allow_overlap": resolved_dimensions.allow_overlap,
            "equity_mode": list(resolved_dimensions.equity_mode),
            "include_watch": resolved_dimensions.include_watch,
            "include_state_sets": resolved_dimensions.include_state_sets,
        },
        "include_ret_cv": include_ret_cv,
        "include_tail_metrics": include_tail_metrics,
        "report_top_n": report_top_n,
        "execution_profile": execution_profile,
        "exec_overrides": {
            "exec_min_price": exec_min_price,
            "exec_min_dollar_vol20": exec_min_dollar_vol20,
            "exec_max_vol_pct": exec_max_vol_pct,
            "exec_min_history_bars": exec_min_history_bars,
        },
        "report_threshold_overrides": {
            "report_min_trades": report_min_trades,
            "report_max_zero_trade_share": report_max_zero_trade_share,
            "report_max_ret_cv": report_max_ret_cv,
        },
    }

    config_path = output_dir / "grid_run_config.json"
    manifest_parquet = output_dir / "grid_manifest.parquet"
    manifest_csv = output_dir / "grid_manifest.csv"
    failures_path = output_dir / "grid_failures.json"
    summary_path = output_dir / "grid_summary.json"
    metrics_parquet = output_dir / "grid_metrics_table.parquet"
    metrics_csv = output_dir / "grid_metrics_table.csv"
    best_path = output_dir / "grid_best_configs.json"
    dim_parquet = output_dir / "grid_dimension_sensitivity.parquet"
    dim_csv = output_dir / "grid_dimension_sensitivity.csv"
    fragility_csv = output_dir / "grid_cost_fragility.csv"
    holdbars_csv = output_dir / "grid_holdbars_profile.csv"
    holdshape_csv = output_dir / "grid_holdbars_shape.csv"
    robustness_parquet = output_dir / "grid_robustness_v2_table.parquet"
    robustness_csv = output_dir / "grid_robustness_v2_table.csv"
    source_summary_csv = output_dir / "grid_source_summary.csv"
    cluster_mix_csv = output_dir / "cluster_policy_trade_mix.csv"
    cluster_state_csv = output_dir / "cluster_state_edge_profile.csv"
    report_path = output_dir / "grid_report.md"

    write_json_atomically(_finite_json(summary_payload), config_path)
    if write_run_manifest:
        write_parquet_atomically(manifest, manifest_parquet)
        write_csv_atomically(manifest, manifest_csv)
    write_json_atomically(_finite_json({"grid_run_id": grid_run_id, "failures": failures}), failures_path)
    write_parquet_atomically(metrics_table, metrics_parquet)
    write_csv_atomically(metrics_table, metrics_csv)
    write_json_atomically(_finite_json(summary_payload), summary_path)
    write_json_atomically(_finite_json(best_payload), best_path)
    write_parquet_atomically(dim_sensitivity, dim_parquet)
    write_csv_atomically(dim_sensitivity, dim_csv)
    write_csv_atomically(cost_fragility, fragility_csv)
    write_csv_atomically(holdbars_profile, holdbars_csv)
    write_csv_atomically(hold_shape, holdshape_csv)
    write_parquet_atomically(metrics_table, robustness_parquet)
    write_csv_atomically(metrics_table, robustness_csv)
    write_csv_atomically(source_summary, source_summary_csv)
    if cluster_mix_rows:
        write_csv_atomically(pl.DataFrame(cluster_mix_rows), cluster_mix_csv)
    if cluster_state_profile_rows:
        state_profile_df = pl.DataFrame(cluster_state_profile_rows)
        state_profile_summary = (
            state_profile_df.group_by(
                ["entry_state_id", "entry_state_label", "entry_state_class", "entry_state_direction_hint", "policy_filter_mode"]
            )
            .agg(
                pl.col("trade_count").sum().alias("trade_count"),
                pl.col("avg_return").cast(pl.Float64, strict=False).mean().alias("avg_return"),
                pl.col("median_return").cast(pl.Float64, strict=False).median().alias("median_return"),
                pl.col("ret_cv").cast(pl.Float64, strict=False).median().alias("ret_cv"),
                pl.col("ret_p10").cast(pl.Float64, strict=False).mean().alias("ret_p10"),
                pl.col("ret_p90").cast(pl.Float64, strict=False).mean().alias("ret_p90"),
                pl.col("downside_std").cast(pl.Float64, strict=False).mean().alias("downside_std"),
                pl.col("worst_trade_return").cast(pl.Float64, strict=False).min().alias("worst_trade_return"),
                pl.col("best_trade_return").cast(pl.Float64, strict=False).max().alias("best_trade_return"),
                pl.len().alias("combo_count"),
            )
            .sort(["policy_filter_mode", "trade_count"], descending=[False, True])
        )
        write_csv_atomically(state_profile_summary, cluster_state_csv)
    write_markdown_atomically(
        render_grid_report(
            summary=summary_payload,
            source_summary=source_summary,
            best_payload=best_payload,
            dim_sensitivity=dim_sensitivity,
            cost_fragility=cost_fragility,
            holdbars_profile=holdbars_profile,
        ),
        report_path,
    )

    effective_logger.info(
        "backtest.grid.complete grid_run_id=%s total=%s success=%s failed=%s output=%s",
        grid_run_id,
        total_combos,
        successful,
        failed,
        output_dir,
    )
    return GridRunResult(
        grid_run_id=grid_run_id,
        output_dir=output_dir,
        config_path=config_path,
        manifest_path=manifest_parquet,
        metrics_table_path=metrics_parquet,
        summary_path=summary_path,
        report_path=report_path,
    )


def run_backtest_grid_compare(
    settings: AppSettings,
    *,
    grid_run_dirs: list[Path],
    metrics: list[str] | None,
    logger: logging.Logger | None = None,
) -> GridCompareResult:
    """Compare two or more sensitivity grid runs."""

    effective_logger = logger or LOGGER
    if len(grid_run_dirs) < 2:
        raise ValueError("backtest-grid-compare requires at least two --grid-run-dir values")

    metric_order = metrics or [settings.backtest_sensitivity.ranking_metric_default]
    primary_metric = metric_order[0]

    rows: list[dict[str, Any]] = []
    top_sets: dict[str, set[str]] = {}
    for run_dir in grid_run_dirs:
        summary = _load_json(run_dir / "grid_summary.json")
        metrics_df = pl.read_parquet(run_dir / "grid_metrics_table.parquet")
        source_summary = pl.read_csv(run_dir / "grid_source_summary.csv") if (run_dir / "grid_source_summary.csv").exists() else pl.DataFrame()
        fragility_df = pl.read_csv(run_dir / "grid_cost_fragility.csv") if (run_dir / "grid_cost_fragility.csv").exists() else pl.DataFrame()

        if metrics_df.height == 0:
            best_row = {}
        else:
            descending = primary_metric not in {"max_drawdown", "return_std", "ret_cv", "downside_std", "total_cost_bps"}
            best_row = metrics_df.sort(primary_metric, descending=descending).head(1).to_dicts()[0]
        overlay_enabled_share = (
            _safe_float(
                metrics_df.select(pl.col("overlay_enabled").cast(pl.Float64, strict=False).mean()).item()
            )
            if metrics_df.height > 0 and "overlay_enabled" in metrics_df.columns
            else None
        )
        overlay_match_rate_mean = (
            _safe_float(
                metrics_df.select(pl.col("overlay_match_rate").cast(pl.Float64, strict=False).mean()).item()
            )
            if metrics_df.height > 0 and "overlay_match_rate" in metrics_df.columns
            else None
        )
        overlay_veto_share_mean = (
            _safe_float(
                metrics_df.select(
                    pl.col("overlay_vetoed_signal_share").cast(pl.Float64, strict=False).mean()
                ).item()
            )
            if metrics_df.height > 0 and "overlay_vetoed_signal_share" in metrics_df.columns
            else None
        )
        overlay_conflict_share_mean = (
            _safe_float(
                metrics_df.select(
                    pl.col("overlay_direction_conflict_share").cast(pl.Float64, strict=False).mean()
                ).item()
            )
            if metrics_df.height > 0 and "overlay_direction_conflict_share" in metrics_df.columns
            else None
        )
        exec_profile_set = (
            sorted(set(str(v) for v in metrics_df.get_column("execution_profile").drop_nulls().to_list()))
            if metrics_df.height > 0 and "execution_profile" in metrics_df.columns
            else []
        )
        execution_filters_enabled_share = (
            _safe_float(
                metrics_df.select(
                    pl.col("execution_filters_enabled").cast(pl.Float64, strict=False).mean()
                ).item()
            )
            if metrics_df.height > 0 and "execution_filters_enabled" in metrics_df.columns
            else None
        )
        exec_eligibility_rate_mean = (
            _safe_float(
                metrics_df.select(pl.col("exec_eligibility_rate").cast(pl.Float64, strict=False).mean()).item()
            )
            if metrics_df.height > 0 and "exec_eligibility_rate" in metrics_df.columns
            else None
        )
        exec_suppressed_signal_share_mean = (
            _safe_float(
                metrics_df.select(
                    pl.col("exec_suppressed_signal_share").cast(pl.Float64, strict=False).mean()
                ).item()
            )
            if metrics_df.height > 0 and "exec_suppressed_signal_share" in metrics_df.columns
            else None
        )
        exec_trade_avg_dollar_vol_20_mean = (
            _safe_float(
                metrics_df.select(
                    pl.col("exec_trade_avg_dollar_vol_20").cast(pl.Float64, strict=False).mean()
                ).item()
            )
            if metrics_df.height > 0 and "exec_trade_avg_dollar_vol_20" in metrics_df.columns
            else None
        )
        overlay_modes = (
            sorted(set(str(v) for v in metrics_df.get_column("overlay_mode").drop_nulls().to_list()))
            if metrics_df.height > 0 and "overlay_mode" in metrics_df.columns
            else []
        )
        expectancy_mean = (
            _safe_float(metrics_df.select(pl.col("expectancy").cast(pl.Float64, strict=False).mean()).item())
            if metrics_df.height > 0 and "expectancy" in metrics_df.columns
            else None
        )
        pf_mean = (
            _safe_float(metrics_df.select(pl.col("profit_factor").cast(pl.Float64, strict=False).mean()).item())
            if metrics_df.height > 0 and "profit_factor" in metrics_df.columns
            else None
        )
        robustness_v2_mean = (
            _safe_float(
                metrics_df.select(pl.col("robustness_score_v2").cast(pl.Float64, strict=False).mean()).item()
            )
            if metrics_df.height > 0 and "robustness_score_v2" in metrics_df.columns
            else None
        )

        signatures = set(
            metrics_df.sort(
                primary_metric,
                descending=(primary_metric not in {"max_drawdown", "return_std", "ret_cv", "downside_std", "total_cost_bps"}),
            )
            .head(10)
            .select(
                pl.concat_str(
                    [
                        pl.col("source_type").cast(pl.String),
                        pl.col("hold_bars").cast(pl.String),
                        pl.col("signal_mode").cast(pl.String),
                        pl.col("exit_mode").cast(pl.String),
                        pl.col("fee_bps_per_side").cast(pl.String),
                        pl.col("slippage_bps_per_side").cast(pl.String),
                        pl.col("allow_overlap").cast(pl.String),
                        pl.col("equity_mode").cast(pl.String),
                        pl.col("include_watch").cast(pl.String),
                        pl.col("state_subset_key").cast(pl.String),
                    ],
                    separator="|",
                ).alias("sig")
            )
            .get_column("sig")
            .to_list()
        )
        top_sets[str(run_dir)] = signatures

        rows.append(
            {
                "grid_run_dir": str(run_dir),
                "grid_run_id": summary.get("grid_run_id"),
                "scope": summary.get("scope"),
                "comparability": summary.get("comparability"),
                "successful_combos": summary.get("successful_combos"),
                "failed_combos": summary.get("failed_combos"),
                "best_combo_id": best_row.get("combo_id"),
                "best_source_type": best_row.get("source_type"),
                "best_expectancy": best_row.get("expectancy"),
                "best_profit_factor": best_row.get("profit_factor"),
                "best_ret_cv": best_row.get("ret_cv"),
                "best_downside_std": best_row.get("downside_std"),
                "best_max_drawdown": best_row.get("max_drawdown"),
                "best_robustness_score_v1": best_row.get("robustness_score_v1"),
                "best_robustness_score_v2": best_row.get("robustness_score_v2"),
                "best_zero_trade": best_row.get("is_zero_trade_combo"),
                "best_trade_count": best_row.get("trade_count"),
                "expectancy_mean": expectancy_mean,
                "profit_factor_mean": pf_mean,
                "robustness_v2_mean": robustness_v2_mean,
                "ret_cv_median": _safe_float(metrics_df.select(pl.col("ret_cv").cast(pl.Float64, strict=False).median()).item()) if metrics_df.height > 0 and "ret_cv" in metrics_df.columns else None,
                "downside_std_mean": _safe_float(metrics_df.select(pl.col("downside_std").cast(pl.Float64, strict=False).mean()).item()) if metrics_df.height > 0 and "downside_std" in metrics_df.columns else None,
                "zero_trade_combo_share": _safe_float(metrics_df.select(pl.col("is_zero_trade_combo").cast(pl.Float64, strict=False).mean()).item()) if metrics_df.height > 0 and "is_zero_trade_combo" in metrics_df.columns else None,
                "overlay_enabled_share": overlay_enabled_share,
                "overlay_mode_set": ",".join(overlay_modes),
                "overlay_match_rate_mean": overlay_match_rate_mean,
                "overlay_vetoed_signal_share_mean": overlay_veto_share_mean,
                "overlay_direction_conflict_share_mean": overlay_conflict_share_mean,
                "execution_profile_set": ",".join(exec_profile_set),
                "execution_filters_enabled_share": execution_filters_enabled_share,
                "exec_eligibility_rate_mean": exec_eligibility_rate_mean,
                "exec_suppressed_signal_share_mean": exec_suppressed_signal_share_mean,
                "exec_trade_avg_dollar_vol_20_mean": exec_trade_avg_dollar_vol_20_mean,
                "cost_fragility_expect_slope_mean": _safe_float(fragility_df.select(pl.col("expectancy_slope_per_10bps").cast(pl.Float64, strict=False).mean()).item()) if fragility_df.height > 0 and "expectancy_slope_per_10bps" in fragility_df.columns else None,
                "cost_fragility_pf_slope_mean": _safe_float(fragility_df.select(pl.col("profit_factor_slope_per_10bps").cast(pl.Float64, strict=False).mean()).item()) if fragility_df.height > 0 and "profit_factor_slope_per_10bps" in fragility_df.columns else None,
                "source_summary_rows": int(source_summary.height),
            }
        )

    overlap_rows: list[dict[str, Any]] = []
    run_keys = list(top_sets.keys())
    for idx, run_a in enumerate(run_keys):
        for run_b in run_keys[idx + 1 :]:
            sa = top_sets[run_a]
            sb = top_sets[run_b]
            overlap_rows.append(
                {
                    "run_a": run_a,
                    "run_b": run_b,
                    "top10_overlap_count": int(len(sa & sb)),
                }
            )

    table = pl.DataFrame(rows).sort("best_robustness_score_v2", descending=True)
    if table.height > 1:
        baseline = table.head(1).to_dicts()[0]
        base_exp = _safe_float(baseline.get("expectancy_mean"))
        base_pf = _safe_float(baseline.get("profit_factor_mean"))
        base_ret_cv = _safe_float(baseline.get("ret_cv_median"))
        base_rob = _safe_float(baseline.get("robustness_v2_mean"))
        base_exec_elig = _safe_float(baseline.get("exec_eligibility_rate_mean"))
        base_exec_supp = _safe_float(baseline.get("exec_suppressed_signal_share_mean"))
        table = table.with_columns(
            (
                pl.col("expectancy_mean").cast(pl.Float64, strict=False)
                - pl.lit(base_exp).cast(pl.Float64, strict=False)
            ).alias("delta_expectancy_vs_top"),
            (
                pl.col("profit_factor_mean").cast(pl.Float64, strict=False)
                - pl.lit(base_pf).cast(pl.Float64, strict=False)
            ).alias("delta_pf_vs_top"),
            (
                pl.col("ret_cv_median").cast(pl.Float64, strict=False)
                - pl.lit(base_ret_cv).cast(pl.Float64, strict=False)
            ).alias("delta_ret_cv_vs_top"),
            (
                pl.col("robustness_v2_mean").cast(pl.Float64, strict=False)
                - pl.lit(base_rob).cast(pl.Float64, strict=False)
            ).alias("delta_robustness_v2_vs_top"),
            (
                pl.col("exec_eligibility_rate_mean").cast(pl.Float64, strict=False)
                - pl.lit(base_exec_elig).cast(pl.Float64, strict=False)
            ).alias("delta_exec_eligibility_rate_vs_top"),
            (
                pl.col("exec_suppressed_signal_share_mean").cast(pl.Float64, strict=False)
                - pl.lit(base_exec_supp).cast(pl.Float64, strict=False)
            ).alias("delta_exec_suppressed_signal_share_vs_top"),
        ).with_columns(
            pl.when(
                (pl.col("delta_expectancy_vs_top") > 0)
                & (pl.col("delta_pf_vs_top") > 0)
                & (pl.col("delta_robustness_v2_vs_top") >= 0)
                & (pl.col("delta_ret_cv_vs_top") <= 0)
            )
            .then(pl.lit("HELPFUL"))
            .when(
                (pl.col("delta_expectancy_vs_top") >= 0)
                & (pl.col("delta_pf_vs_top") >= 0)
            )
            .then(pl.lit("NEUTRAL"))
            .otherwise(pl.lit("HARMFUL"))
            .alias("overlay_verdict")
        ).with_columns(
            pl.when(
                (pl.col("best_zero_trade").cast(pl.Boolean).fill_null(False))
                | (pl.col("best_trade_count").cast(pl.Int64, strict=False).fill_null(0) <= 0)
            )
            .then(pl.lit("NOT_TRADABLE"))
            .when(
                (pl.col("delta_expectancy_vs_top") >= -0.001)
                & (pl.col("delta_pf_vs_top") >= -0.05)
                & (pl.col("delta_robustness_v2_vs_top") >= -2.0)
                & (pl.col("delta_exec_suppressed_signal_share_vs_top") <= 0.10)
            )
            .then(pl.lit("EDGE_RETAINED"))
            .when(
                (pl.col("delta_expectancy_vs_top") < 0)
                & (pl.col("delta_ret_cv_vs_top") < 0)
                & (pl.col("delta_exec_suppressed_signal_share_vs_top") <= 0.35)
            )
            .then(pl.lit("EDGE_COMPROMISED_BUT_CLEANER"))
            .otherwise(pl.lit("TOO_FRAGILE_UNDER_REALISM"))
            .alias("realism_verdict")
        )
    overlap_df = pl.DataFrame(overlap_rows) if overlap_rows else pl.DataFrame()

    compare_id = f"backtest-grid-compare-{uuid4().hex[:12]}"
    output_dir = settings.paths.artifacts_root / "backtest_sensitivity" / compare_id
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "grid_compare_table.csv"
    summary_path = output_dir / "grid_compare_summary.json"
    overlap_path = output_dir / "grid_compare_overlap.csv"
    report_path = output_dir / "grid_compare_report.md"

    payload = {
        "compare_id": compare_id,
        "grid_run_dirs": [str(path) for path in grid_run_dirs],
        "primary_metric": primary_metric,
        "rows": rows,
        "overlap": overlap_rows,
    }

    write_csv_atomically(table, table_path)
    write_csv_atomically(overlap_df, overlap_path)
    write_json_atomically(_finite_json(payload), summary_path)
    write_markdown_atomically(
        render_grid_compare_report(summary=payload, compare_table=table),
        report_path,
    )

    effective_logger.info("backtest.grid.compare.complete compare_id=%s output=%s", compare_id, output_dir)
    return GridCompareResult(
        compare_id=compare_id,
        output_dir=output_dir,
        summary_path=summary_path,
        table_path=table_path,
        report_path=report_path,
    )


def _flow_file_for_split(
    *,
    hmm_file: Path,
    flow_dataset_file: Path | None,
) -> Path:
    if hmm_file.exists():
        return hmm_file
    if flow_dataset_file is None:
        raise FileNotFoundError("Flow input cannot be resolved for split (hmm decoded rows missing and no --flow-dataset-file)")
    return flow_dataset_file


def run_backtest_grid_walkforward(
    settings: AppSettings,
    *,
    wf_run_dir: Path,
    flow_dataset_file: Path | None,
    overlay_cluster_file: Path | None,
    overlay_cluster_hardening_dir: Path | None,
    overlay_mode: OverlayMode,
    overlay_join_keys: list[str] | None,
    train_ends: list[str] | None,
    train_start: str | None,
    train_end_final: str | None,
    step_years: int | None,
    sources: list[str] | None,
    policy_filter_mode: PolicyFilterMode,
    execution_profile: str,
    exec_min_price: float | None,
    exec_min_dollar_vol20: float | None,
    exec_max_vol_pct: float | None,
    exec_min_history_bars: int | None,
    report_min_trades: int | None,
    report_max_zero_trade_share: float | None,
    report_max_ret_cv: float | None,
    dimensions: GridDimensionValues | None,
    max_combos: int | None,
    shuffle_grid: bool,
    seed: int,
    progress_every: int | None,
    stop_on_error: bool,
    force: bool,
    tag: str | None,
    min_successful_splits: int,
    report_top_n: int,
    logger: logging.Logger | None = None,
) -> GridWalkForwardResult:
    """Run sensitivity grid across validation walk-forward splits and aggregate outputs."""

    effective_logger = logger or LOGGER
    manifest_path = wf_run_dir / "wf_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing walk-forward manifest: {manifest_path}")
    wf_manifest = _load_json(manifest_path)

    selected_train_ends: list[str] | None = None
    if train_ends:
        selected_train_ends = sorted({str(v) for v in train_ends})
    elif train_start and train_end_final and step_years is not None:
        selected_train_ends = _generate_train_ends(train_start, train_end_final, step_years)
    elif any(v is not None for v in [train_start, train_end_final, step_years]):
        raise ValueError("train_start, train_end_final, and step_years must all be provided for generated schedule mode")

    source_set = sorted({s.strip().lower() for s in (sources or ["hmm", "flow", "cluster"]) if s.strip()})
    if not source_set:
        raise ValueError("sources selection cannot be empty")
    invalid_sources = sorted(set(source_set) - {"hmm", "flow", "cluster"})
    if invalid_sources:
        raise ValueError(f"Unsupported sources: {','.join(invalid_sources)}")

    wf_grid_id = f"wfgrid-{uuid4().hex[:12]}"
    output_dir = settings.paths.artifacts_root / "backtest_sensitivity_walkforward" / wf_grid_id
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows: list[dict[str, Any]] = []
    split_metrics_frames: list[pl.DataFrame] = []
    failures: list[dict[str, Any]] = []

    all_splits = wf_manifest.get("splits", [])
    if selected_train_ends is not None:
        all_splits = [s for s in all_splits if str(s.get("train_end")) in selected_train_ends]

    for split in all_splits:
        train_end = str(split.get("train_end"))
        status = str(split.get("status"))
        if status != "SUCCESS":
            split_rows.append(
                {
                    "train_end": train_end,
                    "status": "FAILED_UPSTREAM",
                    "grid_run_dir": None,
                    "successful_combos": 0,
                    "failed_combos": 0,
                    "comparability_label": None,
                    "zero_trade_combo_share": None,
                    "robustness_score_v2_mean": None,
                    "exec_eligibility_rate_mean": None,
                    "exec_suppressed_signal_share_mean": None,
                    "error": split.get("error"),
                }
            )
            continue

        split_dir = output_dir / "splits" / train_end
        split_dir.mkdir(parents=True, exist_ok=True)

        hmm_run_dir = Path(str(split.get("hmm_run_dir")))
        cluster_run_dir = Path(str(split.get("cluster_run_dir")))
        val_hmm_dir = Path(str(split.get("val_hmm_dir")))
        val_cluster_dir = Path(str(split.get("val_cluster_dir")))

        hmm_file = hmm_run_dir / "decoded_rows.parquet"
        cluster_file = cluster_run_dir / "clustered_dataset_full.parquet"
        cluster_hardening_dir = wf_run_dir / "cluster_hardening" / "splits" / train_end

        if not cluster_hardening_dir.exists():
            run_cluster_hardening_single(
                settings,
                validation_run_dir=val_cluster_dir,
                clustered_rows_file=cluster_file if cluster_file.exists() else None,
                export_filtered=False,
                output_dir=cluster_hardening_dir,
                force=force,
                logger=effective_logger,
            )

        source_specs: list[SourceInputSpec] = []
        if "hmm" in source_set:
            source_specs.append(
                SourceInputSpec(
                    source_type="hmm",
                    input_file=hmm_file,
                    validation_run_dir=val_hmm_dir,
                    cluster_hardening_dir=None,
                    overlay_cluster_file=overlay_cluster_file,
                    overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,
                    overlay_mode=overlay_mode,
                    overlay_join_keys=overlay_join_keys,
                )
            )
        if "flow" in source_set:
            source_specs.append(
                SourceInputSpec(
                    source_type="flow",
                    input_file=_flow_file_for_split(hmm_file=hmm_file, flow_dataset_file=flow_dataset_file),
                    validation_run_dir=None,
                    cluster_hardening_dir=None,
                    overlay_cluster_file=overlay_cluster_file,
                    overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,
                    overlay_mode=overlay_mode,
                    overlay_join_keys=overlay_join_keys,
                )
            )
        if "cluster" in source_set:
            source_specs.append(
                SourceInputSpec(
                    source_type="cluster",
                    input_file=cluster_file,
                    validation_run_dir=val_cluster_dir,
                    cluster_hardening_dir=cluster_hardening_dir,
                    policy_filter_mode=policy_filter_mode,
                    overlay_cluster_file=overlay_cluster_file,
                    overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,
                    overlay_mode=overlay_mode,
                    overlay_join_keys=overlay_join_keys,
                )
            )

        try:
            split_result = run_backtest_grid(
                settings,
                source_specs=source_specs,
                dimensions=dimensions,
                tag=(tag or f"split-{train_end}"),
                max_combos=max_combos,
                shuffle_grid=shuffle_grid,
                seed=seed,
                progress_every=progress_every,
                stop_on_error=stop_on_error,
                force=force,
                write_run_manifest=True,
                include_ret_cv=True,
                include_tail_metrics=True,
                report_top_n=report_top_n,
                execution_profile=execution_profile,
                exec_min_price=exec_min_price,
                exec_min_dollar_vol20=exec_min_dollar_vol20,
                exec_max_vol_pct=exec_max_vol_pct,
                exec_min_history_bars=exec_min_history_bars,
                report_min_trades=report_min_trades,
                report_max_zero_trade_share=report_max_zero_trade_share,
                report_max_ret_cv=report_max_ret_cv,
                logger=effective_logger,
                output_dir_override=split_dir,
                grid_run_id_override=f"{wf_grid_id}-{train_end}",
            )
            split_summary = _load_json(split_result.summary_path)
            split_rows.append(
                {
                    "train_end": train_end,
                    "status": "SUCCESS",
                    "grid_run_dir": str(split_result.output_dir),
                    "successful_combos": split_summary.get("successful_combos"),
                    "failed_combos": split_summary.get("failed_combos"),
                    "comparability_label": split_summary.get("comparability"),
                    "zero_trade_combo_share": split_summary.get("zero_trade_combo_share"),
                    "robustness_score_v2_mean": None,
                    "exec_eligibility_rate_mean": None,
                    "exec_suppressed_signal_share_mean": None,
                    "error": None,
                }
            )
            metrics_df = pl.read_parquet(split_result.metrics_table_path)
            if metrics_df.height > 0:
                split_metrics_frames.append(metrics_df.with_columns(pl.lit(train_end).alias("train_end")))
                split_rows[-1]["robustness_score_v2_mean"] = _safe_float(
                    metrics_df.select(pl.col("robustness_score_v2").cast(pl.Float64, strict=False).mean()).item()
                )
                split_rows[-1]["exec_eligibility_rate_mean"] = _safe_float(
                    metrics_df.select(pl.col("exec_eligibility_rate").cast(pl.Float64, strict=False).mean()).item()
                ) if "exec_eligibility_rate" in metrics_df.columns else None
                split_rows[-1]["exec_suppressed_signal_share_mean"] = _safe_float(
                    metrics_df.select(
                        pl.col("exec_suppressed_signal_share").cast(pl.Float64, strict=False).mean()
                    ).item()
                ) if "exec_suppressed_signal_share" in metrics_df.columns else None
        except Exception as exc:
            failures.append(
                {
                    "train_end": train_end,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )
            split_rows.append(
                {
                    "train_end": train_end,
                    "status": "FAILED",
                    "grid_run_dir": str(split_dir),
                    "successful_combos": 0,
                    "failed_combos": 0,
                    "comparability_label": None,
                    "zero_trade_combo_share": None,
                    "robustness_score_v2_mean": None,
                    "exec_eligibility_rate_mean": None,
                    "exec_suppressed_signal_share_mean": None,
                    "error": str(exc),
                }
            )
            if stop_on_error:
                raise

    by_split = pl.DataFrame(split_rows) if split_rows else pl.DataFrame()
    all_metrics = pl.concat(split_metrics_frames, how="vertical") if split_metrics_frames else pl.DataFrame()

    if all_metrics.height > 0:
        config_aggregate = (
            all_metrics.group_by(
                [
                    "source_type",
                    "hold_bars",
                    "signal_mode",
                    "exit_mode",
                    "fee_bps_per_side",
                    "slippage_bps_per_side",
                    "allow_overlap",
                    "equity_mode",
                    "include_watch",
                    "policy_filter_mode",
                    "state_subset_key",
                ]
            )
            .agg(
                pl.len().alias("obs_count"),
                pl.col("expectancy").cast(pl.Float64, strict=False).mean().alias("expectancy_mean"),
                pl.col("expectancy").cast(pl.Float64, strict=False).median().alias("expectancy_median"),
                pl.col("profit_factor").cast(pl.Float64, strict=False).mean().alias("profit_factor_mean"),
                pl.col("ret_cv").cast(pl.Float64, strict=False).median().alias("ret_cv_median"),
                pl.col("downside_std").cast(pl.Float64, strict=False).mean().alias("downside_std_mean"),
                pl.col("robustness_score_v2").cast(pl.Float64, strict=False).mean().alias("robustness_v2_mean"),
                pl.col("win_rate").cast(pl.Float64, strict=False).mean().alias("win_rate_mean"),
                pl.col("trade_count").cast(pl.Float64, strict=False).mean().alias("trade_count_mean"),
                pl.col("is_zero_trade_combo").cast(pl.Float64, strict=False).mean().alias("zero_trade_combo_share"),
                pl.col("exec_eligibility_rate")
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("exec_eligibility_rate_mean"),
                pl.col("exec_suppressed_signal_share")
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("exec_suppressed_signal_share_mean"),
                pl.col("train_end").n_unique().alias("split_count"),
            )
            .sort(["source_type", "robustness_v2_mean"], descending=[False, True])
        )

        source_summary = (
            all_metrics.group_by("source_type")
            .agg(
                pl.len().alias("obs_count"),
                pl.col("train_end").n_unique().alias("splits_covered"),
                pl.col("expectancy").cast(pl.Float64, strict=False).mean().alias("expectancy_mean"),
                pl.col("expectancy").cast(pl.Float64, strict=False).median().alias("expectancy_median"),
                pl.col("profit_factor").cast(pl.Float64, strict=False).mean().alias("profit_factor_mean"),
                pl.col("ret_cv").cast(pl.Float64, strict=False).median().alias("ret_cv_median"),
                pl.col("downside_std").cast(pl.Float64, strict=False).mean().alias("downside_std_mean"),
                pl.col("robustness_score_v2").cast(pl.Float64, strict=False).mean().alias("robustness_v2_mean"),
                pl.col("is_zero_trade_combo").cast(pl.Float64, strict=False).mean().alias("zero_trade_combo_share"),
                pl.col("nan_warning_total").cast(pl.Float64, strict=False).sum().alias("nan_warning_total_sum"),
                pl.col("exec_eligibility_rate")
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("exec_eligibility_rate_mean"),
                pl.col("exec_suppressed_signal_share")
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("exec_suppressed_signal_share_mean"),
            )
            .sort("source_type")
        )

        source_dim_summary = (
            all_metrics.group_by(["source_type", "hold_bars"])
            .agg(
                pl.len().alias("obs_count"),
                pl.col("expectancy").cast(pl.Float64, strict=False).mean().alias("expectancy_mean"),
                pl.col("profit_factor").cast(pl.Float64, strict=False).mean().alias("profit_factor_mean"),
                pl.col("ret_cv").cast(pl.Float64, strict=False).median().alias("ret_cv_median"),
                pl.col("robustness_score_v2").cast(pl.Float64, strict=False).mean().alias("robustness_v2_mean"),
                pl.col("is_zero_trade_combo").cast(pl.Float64, strict=False).mean().alias("zero_trade_combo_share"),
                pl.col("exec_eligibility_rate")
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("exec_eligibility_rate_mean"),
                pl.col("exec_suppressed_signal_share")
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("exec_suppressed_signal_share_mean"),
            )
            .with_columns(pl.lit("hold_bars").alias("dimension"))
            .sort(["source_type", "hold_bars"])
        )

        # Rank consistency by config across splits.
        rank_ready = all_metrics.with_columns(
            pl.col("robustness_score_v2")
            .rank(method="dense", descending=True)
            .over(["train_end", "source_type"])
            .alias("rank_in_split")
        )
        split_sizes = (
            rank_ready.group_by(["train_end", "source_type"])
            .agg(pl.len().alias("split_combo_count"))
        )
        rank_ready = rank_ready.join(split_sizes, on=["train_end", "source_type"], how="left").with_columns(
            (
                pl.col("rank_in_split")
                <= (pl.col("split_combo_count").cast(pl.Float64, strict=False) * 0.25).ceil().cast(pl.Int64)
            )
            .cast(pl.Int8)
            .alias("is_top_quartile")
        )
        config_consistency = (
            rank_ready.group_by(
                [
                    "source_type",
                    "hold_bars",
                    "signal_mode",
                    "exit_mode",
                    "fee_bps_per_side",
                    "slippage_bps_per_side",
                    "allow_overlap",
                    "equity_mode",
                    "include_watch",
                    "policy_filter_mode",
                    "state_subset_key",
                ]
            )
            .agg(
                pl.col("train_end").n_unique().alias("split_count"),
                pl.col("is_top_quartile").cast(pl.Float64, strict=False).sum().alias("top_quartile_hits"),
                pl.col("rank_in_split").cast(pl.Float64, strict=False).median().alias("median_rank"),
                pl.col("rank_in_split").cast(pl.Float64, strict=False).std().alias("rank_std"),
                pl.col("is_zero_trade_combo").cast(pl.Int8).sum().alias("zero_trade_splits"),
                pl.col("expectancy").cast(pl.Float64, strict=False).mean().alias("expectancy_mean"),
                pl.col("robustness_score_v2").cast(pl.Float64, strict=False).mean().alias("robustness_v2_mean"),
            )
            .sort(["source_type", "robustness_v2_mean"], descending=[False, True])
        )

        # Split winners by metric.
        winner_rows: list[dict[str, Any]] = []
        per_split_source = (
            all_metrics.group_by(["train_end", "source_type"])
            .agg(
                pl.col("expectancy").cast(pl.Float64, strict=False).max().alias("best_expectancy"),
                pl.col("profit_factor").cast(pl.Float64, strict=False).max().alias("best_profit_factor"),
                pl.col("ret_cv").cast(pl.Float64, strict=False).min().alias("best_ret_cv"),
                pl.col("max_drawdown").cast(pl.Float64, strict=False).min().alias("best_max_drawdown"),
                pl.col("robustness_score_v2").cast(pl.Float64, strict=False).max().alias("best_robustness_v2"),
            )
        )
        for train_end in sorted(set(per_split_source.get_column("train_end").to_list())):
            sub = per_split_source.filter(pl.col("train_end") == train_end)
            if sub.height == 0:
                continue
            winners = {
                "expectancy": sub.sort("best_expectancy", descending=True).head(1),
                "profit_factor": sub.sort("best_profit_factor", descending=True).head(1),
                "ret_cv": sub.sort("best_ret_cv", descending=False).head(1),
                "max_drawdown": sub.sort("best_max_drawdown", descending=False).head(1),
                "robustness_v2": sub.sort("best_robustness_v2", descending=True).head(1),
            }
            for metric_name, row_df in winners.items():
                row = row_df.to_dicts()[0]
                winner_rows.append(
                    {
                        "train_end": train_end,
                        "metric": metric_name,
                        "winner_source_type": row.get("source_type"),
                        "winner_value": row.get(f"best_{metric_name}" if metric_name != "robustness_v2" else "best_robustness_v2"),
                    }
                )
        winner_stability = pl.DataFrame(winner_rows) if winner_rows else pl.DataFrame()
        if winner_stability.height > 0 and source_summary.height > 0:
            winner_counts = (
                winner_stability.group_by(["winner_source_type", "metric"])
                .agg(pl.len().alias("winner_count"))
                .pivot(index="winner_source_type", on="metric", values="winner_count")
                .fill_null(0)
                .rename({"winner_source_type": "source_type"})
            )
            winner_metric_cols = [col for col in winner_counts.columns if col != "source_type"]
            if winner_metric_cols:
                winner_counts = winner_counts.rename(
                    {col: f"winner_count_by_metric_{col}" for col in winner_metric_cols}
                )
            source_summary = source_summary.join(winner_counts, on="source_type", how="left").fill_null(0)

        cost_fragility_summary = (
            build_cost_fragility(all_metrics)
            .group_by("source_type")
            .agg(
                pl.col("expectancy_slope_per_10bps").cast(pl.Float64, strict=False).mean().alias("expectancy_slope_per_10bps_mean"),
                pl.col("profit_factor_slope_per_10bps").cast(pl.Float64, strict=False).mean().alias("profit_factor_slope_per_10bps_mean"),
                pl.col("ret_cv_slope_per_10bps").cast(pl.Float64, strict=False).mean().alias("ret_cv_slope_per_10bps_mean"),
                pl.col("trade_count_slope_per_10bps").cast(pl.Float64, strict=False).mean().alias("trade_count_slope_per_10bps_mean"),
                pl.col("zero_trade_rate_delta_0_to_max_cost")
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("zero_trade_rate_delta_0_to_max_cost_mean"),
            )
            .sort("source_type")
        )

        tail_risk_summary = (
            all_metrics.group_by("source_type")
            .agg(
                pl.col("ret_p10").cast(pl.Float64, strict=False).mean().alias("ret_p10_mean"),
                pl.col("ret_p90").cast(pl.Float64, strict=False).mean().alias("ret_p90_mean"),
                pl.col("downside_std").cast(pl.Float64, strict=False).mean().alias("downside_std_mean"),
                pl.col("worst_trade_return").cast(pl.Float64, strict=False).mean().alias("worst_trade_return_mean"),
            )
            .sort("source_type")
        )
        if "overlay_enabled" in all_metrics.columns:
            overlay_split_summary = (
                all_metrics.group_by(["train_end", "source_type"])
                .agg(
                    pl.col("overlay_enabled").cast(pl.Float64, strict=False).mean().alias("overlay_enabled_share"),
                    pl.col("overlay_mode").drop_nulls().first().alias("overlay_mode"),
                    pl.col("overlay_match_rate").cast(pl.Float64, strict=False).mean().alias("overlay_match_rate_mean"),
                    pl.col("overlay_unknown_rate").cast(pl.Float64, strict=False).mean().alias("overlay_unknown_rate_mean"),
                    pl.col("overlay_vetoed_signal_share")
                    .cast(pl.Float64, strict=False)
                    .mean()
                    .alias("overlay_vetoed_signal_share_mean"),
                    pl.col("overlay_direction_conflict_share")
                    .cast(pl.Float64, strict=False)
                    .mean()
                    .alias("overlay_direction_conflict_share_mean"),
                )
                .sort(["train_end", "source_type"])
            )
            overlay_source_summary = (
                overlay_split_summary.group_by("source_type")
                .agg(
                    pl.len().alias("split_rows"),
                    pl.col("overlay_enabled_share").mean().alias("overlay_enabled_share_mean"),
                    pl.col("overlay_match_rate_mean").mean().alias("overlay_match_rate_mean"),
                    pl.col("overlay_unknown_rate_mean").mean().alias("overlay_unknown_rate_mean"),
                    pl.col("overlay_vetoed_signal_share_mean").mean().alias("overlay_vetoed_signal_share_mean"),
                    pl.col("overlay_direction_conflict_share_mean")
                    .mean()
                    .alias("overlay_direction_conflict_share_mean"),
                )
                .sort("source_type")
            )
            overlay_effectiveness_summary = (
                all_metrics.group_by("source_type")
                .agg(
                    pl.col("expectancy").cast(pl.Float64, strict=False).mean().alias("expectancy_mean"),
                    pl.col("profit_factor").cast(pl.Float64, strict=False).mean().alias("profit_factor_mean"),
                    pl.col("ret_cv").cast(pl.Float64, strict=False).median().alias("ret_cv_median"),
                    pl.col("robustness_score_v2")
                    .cast(pl.Float64, strict=False)
                    .mean()
                    .alias("robustness_v2_mean"),
                    pl.col("exec_eligibility_rate")
                    .cast(pl.Float64, strict=False)
                    .mean()
                    .alias("exec_eligibility_rate_mean"),
                    pl.col("exec_suppressed_signal_share")
                    .cast(pl.Float64, strict=False)
                    .mean()
                    .alias("exec_suppressed_signal_share_mean"),
                    pl.col("overlay_vetoed_signal_share")
                    .cast(pl.Float64, strict=False)
                    .mean()
                    .alias("overlay_vetoed_signal_share_mean"),
                )
                .sort("source_type")
            )
        else:
            overlay_split_summary = pl.DataFrame()
            overlay_source_summary = pl.DataFrame()
            overlay_effectiveness_summary = pl.DataFrame()
    else:
        config_aggregate = pl.DataFrame()
        source_summary = pl.DataFrame()
        source_dim_summary = pl.DataFrame()
        config_consistency = pl.DataFrame()
        winner_stability = pl.DataFrame()
        cost_fragility_summary = pl.DataFrame()
        tail_risk_summary = pl.DataFrame()
        overlay_split_summary = pl.DataFrame()
        overlay_source_summary = pl.DataFrame()
        overlay_effectiveness_summary = pl.DataFrame()

    splits_total = int(len(all_splits))
    splits_successful = int(by_split.filter(pl.col("status") == "SUCCESS").height) if by_split.height > 0 else 0
    splits_failed = int(by_split.filter(pl.col("status") != "SUCCESS").height) if by_split.height > 0 else 0

    if source_summary.height > 0 and splits_total > 0:
        source_summary = source_summary.with_columns(
            (pl.col("splits_covered").cast(pl.Float64, strict=False) / float(splits_total)).alias("coverage_ratio")
        )

    summary_payload = {
        "wf_grid_id": wf_grid_id,
        "wf_run_dir": str(wf_run_dir),
        "sources": source_set,
        "policy_filter_mode": policy_filter_mode,
        "overlay_enabled": bool(overlay_cluster_file is not None and overlay_cluster_hardening_dir is not None),
        "overlay_mode": overlay_mode,
        "execution_profile": execution_profile,
        "exec_overrides": {
            "exec_min_price": exec_min_price,
            "exec_min_dollar_vol20": exec_min_dollar_vol20,
            "exec_max_vol_pct": exec_max_vol_pct,
            "exec_min_history_bars": exec_min_history_bars,
        },
        "selected_train_ends": selected_train_ends,
        "splits_total": splits_total,
        "splits_successful": splits_successful,
        "splits_failed": splits_failed,
        "tag": tag,
        "min_successful_splits": min_successful_splits,
    }

    manifest_out = {
        "wf_grid_id": wf_grid_id,
        "source_wf_run_dir": str(wf_run_dir),
        "splits": split_rows,
    }

    manifest_out_path = output_dir / "wf_grid_manifest.json"
    by_split_parquet = output_dir / "wf_grid_by_split.parquet"
    by_split_csv = output_dir / "wf_grid_by_split.csv"
    config_aggr_parquet = output_dir / "wf_grid_config_aggregate.parquet"
    config_aggr_csv = output_dir / "wf_grid_config_aggregate.csv"
    source_summary_parquet = output_dir / "wf_grid_source_summary.parquet"
    source_summary_csv = output_dir / "wf_grid_source_summary.csv"
    source_dim_csv = output_dir / "wf_grid_source_dimension_summary.csv"
    config_consistency_csv = output_dir / "wf_grid_config_consistency.csv"
    winner_stability_csv = output_dir / "wf_grid_winner_stability.csv"
    cost_fragility_csv = output_dir / "wf_grid_cost_fragility_summary.csv"
    tail_risk_csv = output_dir / "wf_grid_tail_risk_summary.csv"
    overlay_split_csv = output_dir / "wf_overlay_split_summary.csv"
    overlay_source_csv = output_dir / "wf_overlay_source_summary.csv"
    overlay_effect_csv = output_dir / "wf_overlay_effectiveness_summary.csv"
    summary_path = output_dir / "wf_grid_summary.json"
    failures_path = output_dir / "wf_grid_failures.json"
    report_path = output_dir / "wf_grid_report.md"

    write_json_atomically(_finite_json(manifest_out), manifest_out_path)
    write_parquet_atomically(by_split, by_split_parquet)
    write_csv_atomically(by_split, by_split_csv)
    write_parquet_atomically(config_aggregate, config_aggr_parquet)
    write_csv_atomically(config_aggregate, config_aggr_csv)
    write_parquet_atomically(source_summary, source_summary_parquet)
    write_csv_atomically(source_summary, source_summary_csv)
    write_csv_atomically(source_dim_summary, source_dim_csv)
    write_csv_atomically(config_consistency, config_consistency_csv)
    write_csv_atomically(winner_stability, winner_stability_csv)
    write_csv_atomically(cost_fragility_summary, cost_fragility_csv)
    write_csv_atomically(tail_risk_summary, tail_risk_csv)
    write_csv_atomically(overlay_split_summary, overlay_split_csv)
    write_csv_atomically(overlay_source_summary, overlay_source_csv)
    write_csv_atomically(overlay_effectiveness_summary, overlay_effect_csv)
    write_json_atomically(_finite_json(summary_payload), summary_path)
    write_json_atomically(_finite_json({"wf_grid_id": wf_grid_id, "failures": failures}), failures_path)
    write_markdown_atomically(
        render_wf_grid_report(
            summary=summary_payload,
            by_split=by_split,
            source_summary=source_summary,
            config_aggregate=config_aggregate,
            winner_stability=winner_stability,
            cost_fragility_summary=cost_fragility_summary,
            tail_risk_summary=tail_risk_summary,
            overlay_split_summary=overlay_split_summary,
            overlay_source_summary=overlay_source_summary,
            overlay_effectiveness_summary=overlay_effectiveness_summary,
        ),
        report_path,
    )

    if splits_successful < min_successful_splits:
        raise RuntimeError(
            f"WF grid completed with {splits_successful} successful splits, below required minimum {min_successful_splits}."
        )

    effective_logger.info(
        "backtest.grid.wf.complete wf_grid_id=%s splits_successful=%s splits_failed=%s output=%s",
        wf_grid_id,
        splits_successful,
        splits_failed,
        output_dir,
    )
    return GridWalkForwardResult(
        wf_grid_id=wf_grid_id,
        output_dir=output_dir,
        manifest_path=manifest_out_path,
        by_split_path=by_split_parquet,
        config_aggregate_path=config_aggr_parquet,
        source_summary_path=source_summary_parquet,
        report_path=report_path,
    )
