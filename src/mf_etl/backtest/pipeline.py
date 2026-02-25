"""Backtest harness orchestration: run, compare, and walk-forward execution."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from mf_etl.backtest.adapters import normalize_backtest_input
from mf_etl.backtest.engine import simulate_trades
from mf_etl.backtest.metrics import (
    build_exit_reason_summary,
    build_summary_by_state,
    build_summary_by_symbol,
    compute_trade_summary,
)
from mf_etl.backtest.models import (
    BacktestCompareResult,
    BacktestRunConfig,
    BacktestRunResult,
    BacktestWalkForwardResult,
    EquityMode,
    ExitMode,
    InputType,
    OverlayMode,
    PolicyFilterMode,
    SignalMode,
)
from mf_etl.backtest.portfolio import build_daily_equity_curve
from mf_etl.backtest.policy_overlay import (
    apply_policy_overlay,
    build_overlay_performance_breakdown,
)
from mf_etl.backtest.reports import (
    render_backtest_compare_report,
    render_backtest_report,
    render_wf_backtest_report,
)
from mf_etl.backtest.signals import generate_signals
from mf_etl.backtest.state_mapping import (
    apply_cluster_policy_mapping,
    apply_flow_state_mapping,
    apply_hmm_state_mapping,
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


def _resolve_output_dir(
    *,
    root: Path,
    input_type: str,
    tag: str,
    force: bool,
) -> tuple[str, Path]:
    suffix = f"{input_type}_{tag}"
    existing = sorted(root.glob(f"*_{suffix}"), key=lambda p: p.stat().st_mtime)
    if force and existing:
        output_dir = existing[-1]
        run_id = output_dir.name.split("_", 1)[0]
        return run_id, output_dir
    run_id = f"bt-{uuid4().hex[:12]}"
    output_dir = root / f"{run_id}_{suffix}"
    return run_id, output_dir


def _finite_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure non-finite floats become null in JSON payloads."""

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


def _apply_state_mapping(
    *,
    input_type: InputType,
    frame: pl.DataFrame,
    settings: AppSettings,
    validation_run_dir: Path | None,
    cluster_hardening_dir: Path | None,
    state_map_file: Path | None,
    include_watch: bool,
    policy_filter_mode: PolicyFilterMode,
    include_state_ids: list[int],
    allow_unconfirmed: bool,
) -> tuple[pl.DataFrame, dict[str, Any], dict[str, Any] | None]:
    if input_type == "flow":
        return apply_flow_state_mapping(
            frame,
            settings=settings,
            include_state_ids=include_state_ids,
            allow_unconfirmed=allow_unconfirmed,
        )
    if input_type == "hmm":
        return apply_hmm_state_mapping(
            frame,
            settings=settings,
            validation_run_dir=validation_run_dir,
            state_map_file=state_map_file,
            include_state_ids=include_state_ids,
            allow_unconfirmed=allow_unconfirmed,
        )
    if cluster_hardening_dir is None:
        raise ValueError("Cluster backtest requires --cluster-hardening-dir")
    return apply_cluster_policy_mapping(
        frame,
        settings=settings,
        cluster_hardening_dir=cluster_hardening_dir,
        include_watch=include_watch,
        policy_filter_mode=policy_filter_mode,
        include_state_ids=include_state_ids,
        allow_unconfirmed=allow_unconfirmed,
    )


def _default_tag(input_file: Path) -> str:
    return input_file.parent.name.replace(" ", "-")


def run_backtest_run(
    settings: AppSettings,
    *,
    input_type: InputType,
    input_file: Path,
    validation_run_dir: Path | None,
    cluster_hardening_dir: Path | None,
    state_map_file: Path | None,
    signal_mode: SignalMode,
    exit_mode: ExitMode,
    hold_bars: int,
    allow_overlap: bool,
    allow_unconfirmed: bool,
    include_watch: bool,
    policy_filter_mode: PolicyFilterMode = "allow_only",
    include_state_ids: list[int],
    overlay_cluster_file: Path | None,
    overlay_cluster_hardening_dir: Path | None,
    overlay_mode: OverlayMode,
    overlay_join_keys: list[str] | None,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
    equity_mode: EquityMode,
    export_joined_rows: bool,
    tag: str | None,
    force: bool,
    logger: logging.Logger | None = None,
    output_dir_override: Path | None = None,
    run_id_override: str | None = None,
) -> BacktestRunResult:
    """Run one backtest on FLOW/HMM/CLUSTER normalized state rows."""

    effective_logger = logger or LOGGER
    started = datetime.now(timezone.utc)

    config = BacktestRunConfig(
        input_type=input_type,
        input_file=input_file,
        signal_mode=signal_mode,
        exit_mode=exit_mode,
        hold_bars=hold_bars,
        allow_overlap=allow_overlap,
        allow_unconfirmed=allow_unconfirmed,
        include_watch=include_watch,
        policy_filter_mode=policy_filter_mode,
        include_state_ids=include_state_ids,
        overlay_cluster_file=overlay_cluster_file,
        overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,
        overlay_mode=overlay_mode,
        overlay_join_keys=(overlay_join_keys or list(settings.backtest_policy_overlay.join_keys)),
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        equity_mode=equity_mode,
        validation_run_dir=validation_run_dir,
        cluster_hardening_dir=cluster_hardening_dir,
        state_map_file=state_map_file,
        export_joined_rows=export_joined_rows,
        tag=tag,
    )

    adapter = normalize_backtest_input(input_file, input_type=input_type, logger=effective_logger)
    mapped, mapping_summary, policy_snapshot = _apply_state_mapping(
        input_type=input_type,
        frame=adapter.frame,
        settings=settings,
        validation_run_dir=validation_run_dir,
        cluster_hardening_dir=cluster_hardening_dir,
        state_map_file=state_map_file,
        include_watch=include_watch,
        policy_filter_mode=policy_filter_mode,
        include_state_ids=include_state_ids,
        allow_unconfirmed=allow_unconfirmed,
    )

    overlay_enabled = bool(overlay_cluster_file is not None or overlay_cluster_hardening_dir is not None)
    if overlay_enabled and (overlay_cluster_file is None or overlay_cluster_hardening_dir is None):
        raise ValueError(
            "Overlay requires both --overlay-cluster-file and --overlay-cluster-hardening-dir."
        )
    if (not overlay_enabled) and overlay_mode != "none":
        raise ValueError(
            "overlay_mode is not none, but overlay inputs are missing. Provide both overlay inputs or use --overlay-mode none."
        )

    overlay_join_summary: dict[str, Any] = {
        "overlay_enabled": False,
        "overlay_mode": "none",
        "join_keys": list(settings.backtest_policy_overlay.join_keys),
        "primary_rows_total": int(mapped.height),
        "primary_rows_matched": 0,
        "primary_rows_unmatched": int(mapped.height),
        "match_rate": None,
        "overlay_rows_total": 0,
        "overlay_rows_used": 0,
        "duplicate_key_count_primary": 0,
        "duplicate_key_count_overlay": 0,
        "overlay_allow_rate": None,
        "overlay_watch_rate": None,
        "overlay_block_rate": None,
        "overlay_unknown_rate": None,
    }
    overlay_coverage_by_year = pl.DataFrame(schema={"year": pl.Int32})
    overlay_duplicate_keys = pl.DataFrame(schema={"dataset": pl.String})
    overlay_policy_mix = pl.DataFrame(schema={"overlay_policy_class": pl.String})
    if overlay_enabled:
        overlay_result = apply_policy_overlay(
            mapped,
            overlay_cluster_file=overlay_cluster_file,  # type: ignore[arg-type]
            overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,  # type: ignore[arg-type]
            overlay_mode=overlay_mode,
            join_keys=(overlay_join_keys or list(settings.backtest_policy_overlay.join_keys)),
            allow_unknown_for_block_veto=settings.backtest_policy_overlay.allow_unknown_for_block_veto,
            min_overlay_match_rate_warn=settings.backtest_policy_overlay.min_overlay_match_rate_warn,
            logger=effective_logger,
        )
        mapped = overlay_result.frame
        overlay_join_summary = overlay_result.join_summary
        overlay_coverage_by_year = overlay_result.coverage_by_year
        overlay_duplicate_keys = overlay_result.duplicate_keys
        overlay_policy_mix = overlay_result.policy_mix_on_primary

    signal_result = generate_signals(
        mapped,
        signal_mode=signal_mode,
        confirm_bars=2,
    )

    engine_result = simulate_trades(
        signal_result.frame,
        exit_mode=exit_mode,
        hold_bars=hold_bars,
        allow_overlap=allow_overlap,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
    )

    summary_by_state = build_summary_by_state(engine_result.trades)
    summary_by_symbol = build_summary_by_symbol(engine_result.trades)
    headline = compute_trade_summary(
        engine_result.trades,
        signal_diagnostics=engine_result.signal_diagnostics,
    )
    exit_reason = build_exit_reason_summary(engine_result.trades)

    equity_curve = pl.DataFrame(schema={"trade_date": pl.Date})
    equity_metrics: dict[str, Any] | None = None
    if equity_mode == "daily_equity_curve":
        equity_curve, equity_metrics = build_daily_equity_curve(
            engine_result.trades,
            capital_base=settings.backtest.capital_base,
        )

    run_root = settings.paths.artifacts_root / "backtest_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    run_tag = (tag or _default_tag(input_file)).replace(" ", "-")
    if output_dir_override is not None and run_id_override is not None:
        run_id = run_id_override
        output_dir = output_dir_override
    else:
        run_id, output_dir = _resolve_output_dir(
            root=run_root,
            input_type=input_type,
            tag=run_tag,
            force=force,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "backtest_run_config.json"
    summary_path = output_dir / "backtest_summary.json"
    trades_parquet = output_dir / "trades.parquet"
    trades_csv = output_dir / "trades.csv"
    state_parquet = output_dir / "summary_by_state.parquet"
    state_csv = output_dir / "summary_by_state.csv"
    symbol_parquet = output_dir / "summary_by_symbol.parquet"
    symbol_csv = output_dir / "summary_by_symbol.csv"
    signal_diag_path = output_dir / "signal_diagnostics.json"
    overlay_join_summary_path = output_dir / "overlay_join_summary.json"
    overlay_coverage_by_year_path = output_dir / "overlay_join_coverage_by_year.csv"
    overlay_duplicate_keys_path = output_dir / "overlay_join_duplicate_keys.csv"
    overlay_policy_mix_path = output_dir / "overlay_policy_mix_on_primary.csv"
    overlay_signal_effect_path = output_dir / "overlay_signal_effect_summary.json"
    overlay_perf_path = output_dir / "overlay_performance_breakdown.csv"
    report_path = output_dir / "backtest_report.md"

    write_json_atomically(_finite_json(asdict(config)), config_path)
    write_parquet_atomically(engine_result.trades, trades_parquet)
    write_csv_atomically(engine_result.trades, trades_csv)
    write_parquet_atomically(summary_by_state, state_parquet)
    write_csv_atomically(summary_by_state, state_csv)
    write_parquet_atomically(summary_by_symbol, symbol_parquet)
    write_csv_atomically(summary_by_symbol, symbol_csv)
    write_json_atomically(_finite_json(engine_result.signal_diagnostics), signal_diag_path)
    overlay_signal_effect = {
        "overlay_enabled": bool(overlay_enabled),
        "overlay_mode": str(overlay_mode),
        "candidate_signals_before_overlay": int(
            signal_result.diagnostics.get("candidate_signals_before_overlay", 0) or 0
        ),
        "candidate_signals_after_overlay": int(
            signal_result.diagnostics.get("candidate_signals_after_overlay", 0) or 0
        ),
        "vetoed_count": int(signal_result.diagnostics.get("overlay_vetoed_signal_count", 0) or 0),
        "vetoed_share": _safe_float(signal_result.diagnostics.get("overlay_vetoed_signal_share")),
        "overlay_passed_signal_count": int(signal_result.diagnostics.get("overlay_passed_signal_count", 0) or 0),
        "overlay_vetoed_by_policy_class": signal_result.diagnostics.get("overlay_vetoed_by_policy_class", {}),
        "overlay_direction_conflict_count": int(
            signal_result.diagnostics.get("overlay_direction_conflict_count", 0) or 0
        ),
        "overlay_direction_conflict_share": _safe_float(
            signal_result.diagnostics.get("overlay_direction_conflict_share")
        ),
        "by_year": (
            signal_result.frame.with_columns(pl.col("trade_date").dt.year().alias("year"))
            .group_by("year")
            .agg(
                pl.col("overlay_candidate_before").cast(pl.Int64).sum().alias("candidate_before"),
                pl.col("overlay_candidate_after").cast(pl.Int64).sum().alias("candidate_after"),
                pl.col("overlay_vetoed_signal").cast(pl.Int64).sum().alias("vetoed"),
            )
            .with_columns(
                pl.when(pl.col("candidate_before") > 0)
                .then(pl.col("vetoed").cast(pl.Float64) / pl.col("candidate_before").cast(pl.Float64))
                .otherwise(None)
                .alias("vetoed_share")
            )
            .sort("year")
            .to_dicts()
            if signal_result.frame.height > 0
            else []
        ),
    }

    if export_joined_rows:
        write_parquet_atomically(signal_result.frame, output_dir / "normalized_rows_with_policy.parquet")

    if equity_mode == "daily_equity_curve":
        write_parquet_atomically(equity_curve, output_dir / "equity_curve.parquet")
        write_csv_atomically(equity_curve, output_dir / "equity_curve.csv")
        write_json_atomically(_finite_json(equity_metrics or {}), output_dir / "equity_metrics.json")

    if policy_snapshot is not None:
        write_json_atomically(_finite_json(policy_snapshot), output_dir / "policy_snapshot.json")
    write_json_atomically(_finite_json(mapping_summary), output_dir / "state_policy_join_summary.json")
    if overlay_enabled:
        write_json_atomically(_finite_json(overlay_join_summary), overlay_join_summary_path)
        write_csv_atomically(overlay_coverage_by_year, overlay_coverage_by_year_path)
        if overlay_duplicate_keys.height > 0:
            write_csv_atomically(overlay_duplicate_keys, overlay_duplicate_keys_path)
        write_csv_atomically(overlay_policy_mix, overlay_policy_mix_path)
        write_json_atomically(_finite_json(overlay_signal_effect), overlay_signal_effect_path)
        write_csv_atomically(build_overlay_performance_breakdown(engine_result.trades), overlay_perf_path)

    finished = datetime.now(timezone.utc)
    summary_payload = {
        "run_id": run_id,
        "input_type": input_type,
        "input_file": str(input_file),
        "signal_mode": signal_mode,
        "exit_mode": exit_mode,
        "hold_bars": hold_bars,
        "allow_overlap": allow_overlap,
        "allow_unconfirmed": allow_unconfirmed,
        "include_watch": include_watch,
        "policy_filter_mode": policy_filter_mode,
        "include_state_ids": include_state_ids,
        "fee_bps_per_side": fee_bps_per_side,
        "slippage_bps_per_side": slippage_bps_per_side,
        "equity_mode": equity_mode,
        "started_ts": started.isoformat(),
        "finished_ts": finished.isoformat(),
        "duration_sec": round((finished - started).total_seconds(), 3),
        "adapter_summary": adapter.summary,
        "mapping_summary": mapping_summary,
        "signal_diagnostics": signal_result.diagnostics,
        "engine_diagnostics": engine_result.signal_diagnostics,
        "headline": headline,
        "exit_reason_summary": exit_reason,
        "equity_metrics": equity_metrics,
        "overlay": {
            "overlay_enabled": bool(overlay_enabled),
            "overlay_mode": str(overlay_mode),
            "overlay_match_rate": _safe_float(overlay_join_summary.get("match_rate")),
            "overlay_unknown_rate": _safe_float(overlay_join_summary.get("overlay_unknown_rate")),
            "overlay_allow_rate": _safe_float(overlay_join_summary.get("overlay_allow_rate")),
            "overlay_watch_rate": _safe_float(overlay_join_summary.get("overlay_watch_rate")),
            "overlay_block_rate": _safe_float(overlay_join_summary.get("overlay_block_rate")),
            "overlay_vetoed_signal_count": int(
                signal_result.diagnostics.get("overlay_vetoed_signal_count", 0) or 0
            ),
            "overlay_vetoed_signal_share": _safe_float(
                signal_result.diagnostics.get("overlay_vetoed_signal_share")
            ),
            "overlay_passed_signal_count": int(
                signal_result.diagnostics.get("overlay_passed_signal_count", 0) or 0
            ),
            "overlay_direction_conflict_count": int(
                signal_result.diagnostics.get("overlay_direction_conflict_count", 0) or 0
            ),
            "overlay_direction_conflict_share": _safe_float(
                signal_result.diagnostics.get("overlay_direction_conflict_share")
            ),
            "overlay_join_summary": overlay_join_summary,
        },
        "outputs": {
            "summary": str(summary_path),
            "trades_parquet": str(trades_parquet),
            "summary_by_state_csv": str(state_csv),
            "summary_by_symbol_csv": str(symbol_csv),
            "signal_diagnostics": str(signal_diag_path),
            "report": str(report_path),
            "overlay_join_summary": str(overlay_join_summary_path) if overlay_enabled else None,
            "overlay_policy_mix_on_primary": str(overlay_policy_mix_path) if overlay_enabled else None,
            "overlay_signal_effect_summary": str(overlay_signal_effect_path) if overlay_enabled else None,
            "overlay_performance_breakdown": str(overlay_perf_path) if overlay_enabled else None,
        },
    }
    write_json_atomically(_finite_json(summary_payload), summary_path)

    report = render_backtest_report(
        run_summary=summary_payload,
        summary_by_state=summary_by_state,
        summary_by_symbol=summary_by_symbol,
        policy_summary=(policy_snapshot.get("summary") if isinstance(policy_snapshot, dict) else None),
        overlay_summary=(summary_payload.get("overlay") if isinstance(summary_payload.get("overlay"), dict) else None),
    )
    write_markdown_atomically(report, report_path)

    effective_logger.info(
        "backtest.run.complete run_id=%s input_type=%s trades=%s output=%s",
        run_id,
        input_type,
        engine_result.trades.height,
        output_dir,
    )
    return BacktestRunResult(
        run_id=run_id,
        output_dir=output_dir,
        summary_path=summary_path,
        trades_path=trades_parquet,
        report_path=report_path,
    )


def run_backtest_compare(
    settings: AppSettings,
    *,
    run_dirs: list[Path],
    logger: logging.Logger | None = None,
) -> BacktestCompareResult:
    """Compare multiple backtest runs and write summary artifacts."""

    effective_logger = logger or LOGGER
    if len(run_dirs) < 2:
        raise ValueError("backtest-compare requires at least two --run-dir entries")

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        payload = _load_json(run_dir / "backtest_summary.json")
        headline = payload.get("headline", {})
        equity = payload.get("equity_metrics") or {}
        overlay = payload.get("overlay") or {}
        rows.append(
            {
                "run_id": payload.get("run_id"),
                "run_dir": str(run_dir),
                "input_type": payload.get("input_type"),
                "trade_count": headline.get("trade_count"),
                "win_rate": headline.get("win_rate"),
                "avg_return": headline.get("avg_return"),
                "median_return": headline.get("median_return"),
                "profit_factor": headline.get("profit_factor"),
                "expectancy": headline.get("expectancy"),
                "max_drawdown": equity.get("max_drawdown"),
                "cumulative_return": equity.get("cumulative_return"),
                "overlay_enabled": overlay.get("overlay_enabled"),
                "overlay_mode": overlay.get("overlay_mode"),
                "overlay_match_rate": overlay.get("overlay_match_rate"),
                "overlay_vetoed_signal_share": overlay.get("overlay_vetoed_signal_share"),
                "overlay_direction_conflict_share": overlay.get("overlay_direction_conflict_share"),
            }
        )

    table = pl.DataFrame(rows)
    compare_id = f"backtest-compare-{uuid4().hex[:12]}"
    output_dir = settings.paths.artifacts_root / "backtest_runs" / compare_id
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "backtest_compare_table.csv"
    summary_path = output_dir / "backtest_compare_summary.json"
    report_path = output_dir / "backtest_compare_report.md"

    summary_payload = {
        "compare_id": compare_id,
        "run_dirs": [str(p) for p in run_dirs],
        "rows": rows,
        "generated_ts": datetime.now(timezone.utc).isoformat(),
    }

    write_csv_atomically(table, table_path)
    write_json_atomically(_finite_json(summary_payload), summary_path)
    write_markdown_atomically(
        render_backtest_compare_report(summary=summary_payload, compare_table=table),
        report_path,
    )

    effective_logger.info("backtest.compare.complete compare_id=%s output=%s", compare_id, output_dir)
    return BacktestCompareResult(
        compare_id=compare_id,
        output_dir=output_dir,
        summary_path=summary_path,
        table_path=table_path,
        report_path=report_path,
    )


def _safe_model_aggregate(df: pl.DataFrame, model: str) -> dict[str, Any]:
    sub = df.filter(pl.col("model") == model)
    if sub.height == 0:
        return {
            "model": model,
            "splits": 0,
            "trade_count_sum": 0,
            "win_rate_mean": None,
            "avg_return_mean": None,
            "expectancy_mean": None,
            "profit_factor_mean": None,
        }
    return {
        "model": model,
        "splits": int(sub.height),
        "trade_count_sum": int(sub.select(pl.col("trade_count").sum()).item() or 0),
        "win_rate_mean": _safe_float(sub.select(pl.col("win_rate").cast(pl.Float64, strict=False).mean()).item()),
        "avg_return_mean": _safe_float(sub.select(pl.col("avg_return").cast(pl.Float64, strict=False).mean()).item()),
        "expectancy_mean": _safe_float(sub.select(pl.col("expectancy").cast(pl.Float64, strict=False).mean()).item()),
        "profit_factor_mean": _safe_float(sub.select(pl.col("profit_factor").cast(pl.Float64, strict=False).mean()).item()),
    }


def run_backtest_walkforward(
    settings: AppSettings,
    *,
    wf_run_dir: Path,
    flow_dataset_file: Path | None,
    overlay_cluster_file: Path | None,
    overlay_cluster_hardening_dir: Path | None,
    overlay_mode: OverlayMode,
    overlay_join_keys: list[str] | None,
    signal_mode: SignalMode,
    exit_mode: ExitMode,
    hold_bars: int,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
    allow_overlap: bool,
    allow_unconfirmed: bool,
    include_watch: bool,
    equity_mode: EquityMode,
    force: bool,
    logger: logging.Logger | None = None,
) -> BacktestWalkForwardResult:
    """Run backtest for HMM/FLOW/CLUSTER across a validation walk-forward pack."""

    effective_logger = logger or LOGGER
    manifest_path = wf_run_dir / "wf_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing walk-forward manifest at {manifest_path}")
    wf_manifest = _load_json(manifest_path)

    wf_bt_id = f"wfbt-{uuid4().hex[:12]}"
    output_dir = settings.paths.artifacts_root / "backtest_walkforward" / wf_bt_id
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows: list[dict[str, Any]] = []
    split_manifest_rows: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []

    for split in wf_manifest.get("splits", []):
        train_end = str(split.get("train_end"))
        if split.get("status") != "SUCCESS":
            split_manifest_rows.append(
                {
                    "train_end": train_end,
                    "status": "FAILED_UPSTREAM",
                    "error": split.get("error"),
                }
            )
            continue

        split_dir = output_dir / "splits" / train_end
        split_dir.mkdir(parents=True, exist_ok=True)

        hmm_file = Path(str(split.get("hmm_run_dir"))) / "decoded_rows.parquet"
        hmm_validation = Path(str(split.get("val_hmm_dir")))
        cluster_file = Path(str(split.get("cluster_run_dir"))) / "clustered_dataset_full.parquet"
        cluster_validation = Path(str(split.get("val_cluster_dir")))
        cluster_hardening_dir = wf_run_dir / "cluster_hardening" / "splits" / train_end

        if not cluster_hardening_dir.exists():
            run_cluster_hardening_single(
                settings,
                validation_run_dir=cluster_validation,
                clustered_rows_file=cluster_file if cluster_file.exists() else None,
                export_filtered=False,
                output_dir=cluster_hardening_dir,
                force=force,
                logger=effective_logger,
            )

        flow_file = hmm_file
        if flow_dataset_file is not None and not flow_file.exists():
            flow_file = flow_dataset_file

        model_specs = [
            ("HMM", "hmm", hmm_file, hmm_validation, None),
            ("FLOW", "flow", flow_file, None, None),
            ("CLUSTER", "cluster", cluster_file, None, cluster_hardening_dir),
        ]

        for model_name, input_type, input_file, val_dir, hardening_dir in model_specs:
            try:
                run_id = f"{wf_bt_id}-{train_end}-{model_name.lower()}"
                run_output_dir = split_dir / model_name.lower()
                result = run_backtest_run(
                    settings,
                    input_type=input_type,  # type: ignore[arg-type]
                    input_file=input_file,
                    validation_run_dir=val_dir,
                    cluster_hardening_dir=hardening_dir,
                    state_map_file=None,
                    signal_mode=signal_mode,
                    exit_mode=exit_mode,
                    hold_bars=hold_bars,
                    allow_overlap=allow_overlap,
                    allow_unconfirmed=allow_unconfirmed,
                    include_watch=include_watch,
                    include_state_ids=[],
                    overlay_cluster_file=overlay_cluster_file,
                    overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,
                    overlay_mode=overlay_mode,
                    overlay_join_keys=overlay_join_keys,
                    fee_bps_per_side=fee_bps_per_side,
                    slippage_bps_per_side=slippage_bps_per_side,
                    equity_mode=equity_mode,
                    export_joined_rows=False,
                    tag=f"{train_end}-{model_name.lower()}",
                    force=True,
                    logger=effective_logger,
                    output_dir_override=run_output_dir,
                    run_id_override=run_id,
                )
                payload = _load_json(result.summary_path)
                head = payload.get("headline", {})
                overlay = payload.get("overlay", {}) if isinstance(payload.get("overlay"), dict) else {}
                split_rows.append(
                    {
                        "train_end": train_end,
                        "model": model_name,
                        "run_dir": str(result.output_dir),
                        "trade_count": head.get("trade_count"),
                        "win_rate": head.get("win_rate"),
                        "avg_return": head.get("avg_return"),
                        "profit_factor": head.get("profit_factor"),
                        "expectancy": head.get("expectancy"),
                        "status": "SUCCESS",
                        "error": None,
                    }
                )
                overlay_rows.append(
                    {
                        "train_end": train_end,
                        "model": model_name,
                        "overlay_enabled": bool(overlay.get("overlay_enabled")),
                        "overlay_mode": overlay.get("overlay_mode"),
                        "overlay_match_rate": _safe_float(overlay.get("overlay_match_rate")),
                        "overlay_unknown_rate": _safe_float(overlay.get("overlay_unknown_rate")),
                        "overlay_vetoed_signal_share": _safe_float(overlay.get("overlay_vetoed_signal_share")),
                        "overlay_direction_conflict_share": _safe_float(
                            overlay.get("overlay_direction_conflict_share")
                        ),
                    }
                )
            except Exception as exc:
                split_rows.append(
                    {
                        "train_end": train_end,
                        "model": model_name,
                        "run_dir": str(split_dir / model_name.lower()),
                        "trade_count": None,
                        "win_rate": None,
                        "avg_return": None,
                        "profit_factor": None,
                        "expectancy": None,
                        "status": "FAILED",
                        "error": str(exc),
                    }
                )
                if force:
                    # force here means do not stop; explicit continue behavior for wf pack.
                    pass

        split_manifest_rows.append({"train_end": train_end, "status": "DONE", "error": None})

    by_split = pl.DataFrame(split_rows) if split_rows else pl.DataFrame(schema={"train_end": pl.String})
    overlay_split = pl.DataFrame(overlay_rows) if overlay_rows else pl.DataFrame(schema={"train_end": pl.String})
    success_df = by_split.filter(pl.col("status") == "SUCCESS") if by_split.height > 0 else by_split

    model_summary_rows = [_safe_model_aggregate(success_df, model) for model in ["HMM", "FLOW", "CLUSTER"]]
    model_summary = pl.DataFrame(model_summary_rows)

    aggregate_summary = {
        "wf_bt_id": wf_bt_id,
        "wf_run_dir": str(wf_run_dir),
        "splits_total": int(len(wf_manifest.get("splits", []))),
        "splits_successful": int(len([x for x in split_manifest_rows if x.get("status") == "DONE"])),
        "splits_failed": int(len([x for x in split_manifest_rows if x.get("status") != "DONE"])),
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "overlay_enabled": bool(overlay_cluster_file is not None and overlay_cluster_hardening_dir is not None),
        "overlay_mode": overlay_mode,
    }

    overlay_source_summary = (
        overlay_split.group_by("model")
        .agg(
            pl.len().alias("rows"),
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
        .sort("model")
        if overlay_split.height > 0
        else pl.DataFrame(schema={"model": pl.String})
    )

    manifest_out = {
        "wf_bt_id": wf_bt_id,
        "source_wf_run_dir": str(wf_run_dir),
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "splits": split_manifest_rows,
    }

    manifest_out_path = output_dir / "wf_backtest_manifest.json"
    by_split_path = output_dir / "wf_backtest_by_split.csv"
    model_summary_path = output_dir / "wf_backtest_model_summary.csv"
    overlay_split_path = output_dir / "wf_overlay_split_summary.csv"
    overlay_source_path = output_dir / "wf_overlay_source_summary.csv"
    overlay_effect_path = output_dir / "wf_overlay_effectiveness_summary.csv"
    aggregate_path = output_dir / "wf_backtest_aggregate_summary.json"
    report_path = output_dir / "wf_backtest_report.md"

    write_json_atomically(_finite_json(manifest_out), manifest_out_path)
    write_csv_atomically(by_split, by_split_path)
    write_csv_atomically(model_summary, model_summary_path)
    write_csv_atomically(overlay_split, overlay_split_path)
    write_csv_atomically(overlay_source_summary, overlay_source_path)
    write_csv_atomically(overlay_source_summary, overlay_effect_path)
    write_json_atomically(_finite_json(aggregate_summary), aggregate_path)
    write_markdown_atomically(
        render_wf_backtest_report(
            aggregate_summary=aggregate_summary,
            by_split=by_split,
            model_summary=model_summary,
        ),
        report_path,
    )

    effective_logger.info("backtest.wf.complete wf_bt_id=%s output=%s", wf_bt_id, output_dir)
    return BacktestWalkForwardResult(
        wf_bt_id=wf_bt_id,
        output_dir=output_dir,
        manifest_path=manifest_out_path,
        aggregate_summary_path=aggregate_path,
        model_summary_path=model_summary_path,
        report_path=report_path,
    )
