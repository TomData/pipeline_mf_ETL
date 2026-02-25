"""Candidate Re-run Pack v1 orchestration and drift checks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from mf_etl.backtest.candidate_rerun_models import CandidateRerunPackResult
from mf_etl.backtest.candidate_rerun_reports import write_candidate_rerun_reports
from mf_etl.backtest.pipeline import run_backtest_run
from mf_etl.backtest.sensitivity_models import GridDimensionValues, SourceInputSpec
from mf_etl.backtest.sensitivity_runner import run_backtest_grid, run_backtest_grid_walkforward
from mf_etl.backtest.writer import (
    write_csv_atomically,
    write_json_atomically,
    write_markdown_atomically,
)
from mf_etl.config import AppSettings

LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_json(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct_delta(expected: float | None, observed: float | None, eps: float) -> float | None:
    if expected is None or observed is None:
        return None
    denom = max(abs(expected), eps)
    out = (observed - expected) / denom
    return out if np.isfinite(out) else None


def _compute_observed_metrics(
    *,
    summary_payload: dict[str, Any],
    trades_path: Path,
    eps: float,
) -> dict[str, Any]:
    headline = summary_payload.get("headline", {}) if isinstance(summary_payload, dict) else {}
    execution = summary_payload.get("execution", {}) if isinstance(summary_payload, dict) else {}
    overlay = summary_payload.get("overlay", {}) if isinstance(summary_payload, dict) else {}

    expectancy = _safe_float(headline.get("expectancy"))
    profit_factor = _safe_float(headline.get("profit_factor"))
    trade_count = _safe_int(headline.get("trade_count")) or 0
    return_std = _safe_float(headline.get("return_std"))
    ret_cv = None
    if return_std is not None and expectancy is not None and abs(expectancy) > eps:
        ret_cv = return_std / abs(expectancy)

    ret_p10: float | None = None
    downside_std: float | None = None
    if trades_path.exists():
        trades = pl.read_parquet(trades_path)
        if "net_return" in trades.columns and trades.height > 0:
            series = (
                trades.select(pl.col("net_return").cast(pl.Float64, strict=False).alias("net_return"))
                .filter(pl.col("net_return").is_not_null() & pl.col("net_return").is_finite())
                .get_column("net_return")
            )
            if series.len() > 0:
                ret_p10 = _safe_float(series.quantile(0.10))
                negative = series.filter(series < 0.0)
                if negative.len() > 0:
                    downside_std = _safe_float(negative.std())

    return {
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "robustness_v2": None,  # single-run backtest summary does not carry robustness score
        "ret_cv": _safe_float(ret_cv),
        "trade_count": trade_count,
        "zero_trade_share": 1.0 if trade_count <= 0 else 0.0,
        "ret_p10": ret_p10,
        "downside_std": downside_std,
        "exec_eligibility_rate": _safe_float(execution.get("exec_eligibility_rate")),
        "exec_suppressed_signal_share": _safe_float(execution.get("exec_suppressed_signal_share")),
        "exec_suppressed_by_price_share": _safe_float(execution.get("exec_suppressed_by_price_share")),
        "exec_suppressed_by_liquidity_share": _safe_float(execution.get("exec_suppressed_by_liquidity_share")),
        "exec_suppressed_by_vol_share": _safe_float(execution.get("exec_suppressed_by_vol_share")),
        "exec_suppressed_by_warmup_share": _safe_float(execution.get("exec_suppressed_by_warmup_share")),
        "overlay_match_rate": _safe_float(overlay.get("overlay_match_rate")),
        "overlay_vetoed_signal_share": _safe_float(overlay.get("overlay_vetoed_signal_share")),
    }


def _compute_drift(
    *,
    expected: dict[str, Any],
    observed: dict[str, Any],
    settings: AppSettings,
) -> dict[str, Any]:
    eps = settings.candidate_rerun.eps
    thr = settings.candidate_rerun.drift

    exp_expectancy = _safe_float(expected.get("best_expectancy"))
    exp_pf = _safe_float(expected.get("PF"))
    exp_rob = _safe_float(expected.get("robustness_v2"))
    exp_ret_cv = _safe_float(expected.get("ret_cv"))
    exp_trades = _safe_int(expected.get("trade_count"))
    exp_zero = _safe_float(expected.get("zero_trade_share"))
    exp_p10 = _safe_float(expected.get("ret_p10"))
    exp_downside = _safe_float(expected.get("downside_std"))

    obs_expectancy = _safe_float(observed.get("expectancy"))
    obs_pf = _safe_float(observed.get("profit_factor"))
    obs_rob = _safe_float(observed.get("robustness_v2"))
    obs_ret_cv = _safe_float(observed.get("ret_cv"))
    obs_trades = _safe_int(observed.get("trade_count"))
    obs_zero = _safe_float(observed.get("zero_trade_share"))
    obs_p10 = _safe_float(observed.get("ret_p10"))
    obs_downside = _safe_float(observed.get("downside_std"))

    exp_exec_elig = _safe_float(expected.get("exec_eligibility_rate"))
    exp_overlay_match = _safe_float(expected.get("overlay_match_rate"))
    obs_exec_elig = _safe_float(observed.get("exec_eligibility_rate"))
    obs_overlay_match = _safe_float(observed.get("overlay_match_rate"))

    deltas = {
        "delta_expectancy_abs": None if exp_expectancy is None or obs_expectancy is None else obs_expectancy - exp_expectancy,
        "delta_expectancy_pct": _pct_delta(exp_expectancy, obs_expectancy, eps),
        "delta_profit_factor_abs": None if exp_pf is None or obs_pf is None else obs_pf - exp_pf,
        "delta_profit_factor_pct": _pct_delta(exp_pf, obs_pf, eps),
        "delta_robustness_v2_abs": None if exp_rob is None or obs_rob is None else obs_rob - exp_rob,
        "delta_robustness_v2_pct": _pct_delta(exp_rob, obs_rob, eps),
        "delta_ret_cv_abs": None if exp_ret_cv is None or obs_ret_cv is None else obs_ret_cv - exp_ret_cv,
        "delta_ret_cv_pct": _pct_delta(exp_ret_cv, obs_ret_cv, eps),
        "delta_trade_count_abs": None if exp_trades is None or obs_trades is None else obs_trades - exp_trades,
        "delta_trade_count_pct": _pct_delta(float(exp_trades) if exp_trades is not None else None, float(obs_trades) if obs_trades is not None else None, eps),
        "delta_zero_trade_share_abs": None if exp_zero is None or obs_zero is None else obs_zero - exp_zero,
        "delta_ret_p10_abs": None if exp_p10 is None or obs_p10 is None else obs_p10 - exp_p10,
        "delta_downside_std_abs": None if exp_downside is None or obs_downside is None else obs_downside - exp_downside,
        "delta_exec_eligibility_rate_abs": None if exp_exec_elig is None or obs_exec_elig is None else obs_exec_elig - exp_exec_elig,
        "delta_exec_eligibility_rate_pct": _pct_delta(exp_exec_elig, obs_exec_elig, eps),
        "delta_overlay_match_rate_abs": None if exp_overlay_match is None or obs_overlay_match is None else obs_overlay_match - exp_overlay_match,
    }

    flags: list[str] = []
    if obs_trades is not None and obs_trades <= 0:
        flags.append("zero_trade_observed")

    if deltas["delta_expectancy_pct"] is not None and deltas["delta_expectancy_pct"] < -thr.expectancy_drop_pct_flag:
        flags.append("expectancy_drop")
    if deltas["delta_profit_factor_pct"] is not None and deltas["delta_profit_factor_pct"] < -thr.pf_drop_pct_flag:
        flags.append("profit_factor_drop")
    if deltas["delta_robustness_v2_abs"] is not None and deltas["delta_robustness_v2_abs"] < -thr.robustness_drop_points_flag:
        flags.append("robustness_v2_drop")
    if deltas["delta_ret_cv_pct"] is not None and deltas["delta_ret_cv_pct"] > thr.ret_cv_increase_pct_flag:
        flags.append("ret_cv_increase")
    if deltas["delta_trade_count_pct"] is not None and deltas["delta_trade_count_pct"] < -thr.trade_count_drop_pct_flag:
        flags.append("trade_count_drop")
    if (
        deltas["delta_exec_eligibility_rate_pct"] is not None
        and deltas["delta_exec_eligibility_rate_pct"] < -thr.exec_eligibility_drop_pct_flag
    ):
        flags.append("exec_eligibility_drop")
    if obs_overlay_match is not None and obs_overlay_match < thr.overlay_match_rate_min:
        flags.append("overlay_match_rate_low")

    if "zero_trade_observed" in flags or len(flags) >= 2:
        drift_status = "DRIFT_FAIL"
    elif len(flags) == 1:
        drift_status = "DRIFT_WARN"
    else:
        drift_status = "OK"

    return {
        **deltas,
        "flags": flags,
        "drift_status": drift_status,
    }


def _build_source_spec_from_candidate(
    *,
    candidate: dict[str, Any],
    input_file: Path,
    validation_run_dir: Path | None,
) -> SourceInputSpec:
    overlay = candidate.get("overlay", {})
    return SourceInputSpec(
        source_type=str(candidate.get("input_type", "hmm")),  # type: ignore[arg-type]
        input_file=input_file,
        validation_run_dir=validation_run_dir,
        cluster_hardening_dir=None,
        state_map_file=None,
        policy_filter_mode="allow_only",
        overlay_cluster_file=Path(str(overlay.get("overlay_cluster_file"))) if overlay.get("overlay_cluster_file") else None,
        overlay_cluster_hardening_dir=Path(str(overlay.get("overlay_cluster_hardening_dir"))) if overlay.get("overlay_cluster_hardening_dir") else None,
        overlay_mode=str(overlay.get("mode", "none")),  # type: ignore[arg-type]
        overlay_join_keys=["ticker", "trade_date"],
    )


def _single_combo_dimensions(candidate: dict[str, Any]) -> GridDimensionValues:
    strategy = candidate.get("strategy_params", {})
    return GridDimensionValues(
        hold_bars=[int(strategy.get("hold_bars", 10))],
        signal_mode=[str(strategy.get("signal_mode", "state_transition_entry"))],  # type: ignore[list-item]
        exit_mode=[str(strategy.get("exit_mode", "horizon_or_state"))],  # type: ignore[list-item]
        fee_bps_per_side=[float(strategy.get("fee_bps_per_side", 0.0))],
        slippage_bps_per_side=[float(strategy.get("slippage_bps_per_side", 0.0))],
        allow_overlap=[bool(strategy.get("allow_overlap", False))],
        equity_mode=["event_returns_only"],
        include_watch=[False],
        include_state_sets=[[]],
    )


def _micro_dimensions(candidate: dict[str, Any], settings: AppSettings) -> GridDimensionValues:
    strategy = candidate.get("strategy_params", {})
    hold = int(strategy.get("hold_bars", 10))
    fee = float(strategy.get("fee_bps_per_side", 0.0))
    slippage = float(strategy.get("slippage_bps_per_side", 0.0))
    hold_values = sorted(
        {
            max(1, hold + int(offset))
            for offset in settings.candidate_rerun.micro_grid.hold_bars_offsets
        }
    )
    fee_values = sorted(
        {
            max(0.0, fee + float(offset))
            for offset in settings.candidate_rerun.micro_grid.fee_bps_offsets
        }
        | {0.0, fee}
    )
    return GridDimensionValues(
        hold_bars=hold_values,
        signal_mode=[str(strategy.get("signal_mode", "state_transition_entry"))],  # type: ignore[list-item]
        exit_mode=[str(strategy.get("exit_mode", "horizon_or_state"))],  # type: ignore[list-item]
        fee_bps_per_side=fee_values,
        slippage_bps_per_side=[slippage],
        allow_overlap=[bool(strategy.get("allow_overlap", False))],
        equity_mode=["event_returns_only"],
        include_watch=[False],
        include_state_sets=[[]],
    )


def run_candidate_rerun_pack(
    settings: AppSettings,
    *,
    pcp_pack_dir: Path,
    as_of_tag: str | None,
    wf_run_dir: Path | None,
    override_input_file: Path | None,
    run_micro_grid: bool | None,
    logger: logging.Logger | None = None,
) -> CandidateRerunPackResult:
    """Rerun PCP candidates and generate drift/consistency artifacts."""

    effective_logger = logger or LOGGER
    packet_path = pcp_pack_dir / "production_policy_packet_v1.json"
    if not packet_path.exists():
        raise FileNotFoundError(f"Missing PCP policy packet: {packet_path}")
    policy_packet = _read_json(packet_path)
    candidates = policy_packet.get("candidates", {})
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("PCP policy packet has no candidates.")

    started = datetime.now(timezone.utc)
    run_id = f"crp-{uuid4().hex[:12]}"
    output_dir = settings.paths.artifacts_root / "candidate_reruns" / f"{run_id}_candidate_rerun_pack_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_root = output_dir / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)

    micro_enabled = (
        settings.candidate_rerun.micro_grid.enabled_default
        if run_micro_grid is None
        else run_micro_grid
    )

    candidate_rows: list[dict[str, Any]] = []
    manifest_candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    wf_rows: list[dict[str, Any]] = []

    for candidate_label in sorted(candidates.keys()):
        candidate = candidates[candidate_label]
        candidate_dir = candidates_root / candidate_label
        candidate_dir.mkdir(parents=True, exist_ok=True)

        input_file = (
            override_input_file
            if override_input_file is not None
            else Path(str(candidate.get("input_file")))
        )
        validation_run_dir = (
            Path(str(candidate.get("validation_run_dir")))
            if candidate.get("validation_run_dir")
            else None
        )
        if not input_file.exists():
            raise FileNotFoundError(f"{candidate_label}: input file does not exist: {input_file}")

        strategy = candidate.get("strategy_params", {})
        overlay = candidate.get("overlay", {})
        execution = candidate.get("execution_realism", {})
        expected_snapshot = candidate.get("expected_behavior_snapshot", {})
        expected_enriched = {
            **expected_snapshot,
            "exec_eligibility_rate": _safe_float(execution.get("eligibility_rate")),
            "overlay_match_rate": _safe_float(overlay.get("match_rate")),
        }

        bt_result = run_backtest_run(
            settings,
            input_type=str(candidate.get("input_type", "hmm")),  # type: ignore[arg-type]
            input_file=input_file,
            validation_run_dir=validation_run_dir,
            cluster_hardening_dir=None,
            state_map_file=None,
            signal_mode=str(strategy.get("signal_mode", "state_transition_entry")),  # type: ignore[arg-type]
            exit_mode=str(strategy.get("exit_mode", "horizon_or_state")),  # type: ignore[arg-type]
            hold_bars=int(strategy.get("hold_bars", 10)),
            allow_overlap=bool(strategy.get("allow_overlap", False)),
            allow_unconfirmed=settings.backtest.allow_unconfirmed,
            include_watch=False,
            policy_filter_mode="allow_only",
            include_state_ids=[],
            overlay_cluster_file=Path(str(overlay.get("overlay_cluster_file"))) if overlay.get("overlay_cluster_file") else None,
            overlay_cluster_hardening_dir=Path(str(overlay.get("overlay_cluster_hardening_dir")))
            if overlay.get("overlay_cluster_hardening_dir")
            else None,
            overlay_mode=str(overlay.get("mode", "none")),  # type: ignore[arg-type]
            overlay_join_keys=["ticker", "trade_date"],
            execution_profile=str(execution.get("profile", settings.backtest_execution_realism.default_profile)),  # type: ignore[arg-type]
            exec_min_price=_safe_float((execution.get("thresholds_used") or {}).get("min_price")),
            exec_min_dollar_vol20=_safe_float((execution.get("thresholds_used") or {}).get("min_dollar_vol20")),
            exec_max_vol_pct=_safe_float((execution.get("thresholds_used") or {}).get("max_vol_pct")),
            exec_min_history_bars=_safe_int((execution.get("thresholds_used") or {}).get("min_history_bars")),
            report_min_trades=settings.backtest_execution_realism.report_min_trades_default,
            report_max_zero_trade_share=settings.backtest_execution_realism.report_max_zero_trade_share_default,
            report_max_ret_cv=settings.backtest_execution_realism.report_max_ret_cv_default,
            fee_bps_per_side=float(strategy.get("fee_bps_per_side", 0.0)),
            slippage_bps_per_side=float(strategy.get("slippage_bps_per_side", 0.0)),
            equity_mode="event_returns_only",
            export_joined_rows=False,
            tag=f"{as_of_tag or 'rerun'}-{candidate_label.lower()}",
            force=False,
            logger=effective_logger,
        )

        backtest_summary = _read_json(bt_result.summary_path)
        observed = _compute_observed_metrics(
            summary_payload=backtest_summary,
            trades_path=bt_result.trades_path,
            eps=settings.candidate_rerun.eps,
        )
        drift = _compute_drift(
            expected=expected_enriched,
            observed=observed,
            settings=settings,
        )

        write_markdown_atomically(str(bt_result.output_dir) + "\n", candidate_dir / "backtest_run_dir.txt")
        write_markdown_atomically(str(bt_result.summary_path) + "\n", candidate_dir / "backtest_summary_path.txt")
        write_json_atomically(_finite_json(backtest_summary), candidate_dir / "backtest_summary.json")
        write_json_atomically(_finite_json(drift), candidate_dir / "drift_metrics.json")

        micro_grid_dir: Path | None = None
        if micro_enabled:
            micro_source = _build_source_spec_from_candidate(
                candidate=candidate,
                input_file=input_file,
                validation_run_dir=validation_run_dir,
            )
            micro_dims = _micro_dimensions(candidate, settings)
            micro_result = run_backtest_grid(
                settings,
                source_specs=[micro_source],
                dimensions=micro_dims,
                tag=f"{as_of_tag or 'rerun'}-{candidate_label.lower()}-micro",
                max_combos=settings.candidate_rerun.micro_grid.max_combos,
                shuffle_grid=False,
                seed=42,
                progress_every=settings.backtest_sensitivity.progress_every,
                stop_on_error=False,
                force=False,
                write_run_manifest=True,
                include_ret_cv=True,
                include_tail_metrics=True,
                report_top_n=5,
                execution_profile=str(execution.get("profile", settings.backtest_execution_realism.default_profile)),
                exec_min_price=_safe_float((execution.get("thresholds_used") or {}).get("min_price")),
                exec_min_dollar_vol20=_safe_float((execution.get("thresholds_used") or {}).get("min_dollar_vol20")),
                exec_max_vol_pct=_safe_float((execution.get("thresholds_used") or {}).get("max_vol_pct")),
                exec_min_history_bars=_safe_int((execution.get("thresholds_used") or {}).get("min_history_bars")),
                report_min_trades=settings.backtest_execution_realism.report_min_trades_default,
                report_max_zero_trade_share=settings.backtest_execution_realism.report_max_zero_trade_share_default,
                report_max_ret_cv=settings.backtest_execution_realism.report_max_ret_cv_default,
                logger=effective_logger,
            )
            micro_grid_dir = micro_result.output_dir
            write_markdown_atomically(str(micro_grid_dir) + "\n", candidate_dir / "micro_grid_dir.txt")

        wf_grid_dir: Path | None = None
        if wf_run_dir is not None:
            wf_source = str(candidate.get("input_type", "hmm")).lower()
            if wf_source != "hmm":
                warnings.append(f"{candidate_label}: wf rerun currently only implemented for hmm candidates; skipped.")
            else:
                wf_result = run_backtest_grid_walkforward(
                    settings,
                    wf_run_dir=wf_run_dir,
                    flow_dataset_file=None,
                    overlay_cluster_file=Path(str(overlay.get("overlay_cluster_file"))) if overlay.get("overlay_cluster_file") else None,
                    overlay_cluster_hardening_dir=Path(str(overlay.get("overlay_cluster_hardening_dir")))
                    if overlay.get("overlay_cluster_hardening_dir")
                    else None,
                    overlay_mode=str(overlay.get("mode", "none")),  # type: ignore[arg-type]
                    overlay_join_keys=["ticker", "trade_date"],
                    train_ends=None,
                    train_start=None,
                    train_end_final=None,
                    step_years=None,
                    sources=["hmm"],
                    policy_filter_mode="allow_only",
                    execution_profile=str(execution.get("profile", settings.backtest_execution_realism.default_profile)),
                    exec_min_price=_safe_float((execution.get("thresholds_used") or {}).get("min_price")),
                    exec_min_dollar_vol20=_safe_float((execution.get("thresholds_used") or {}).get("min_dollar_vol20")),
                    exec_max_vol_pct=_safe_float((execution.get("thresholds_used") or {}).get("max_vol_pct")),
                    exec_min_history_bars=_safe_int((execution.get("thresholds_used") or {}).get("min_history_bars")),
                    report_min_trades=settings.backtest_execution_realism.report_min_trades_default,
                    report_max_zero_trade_share=settings.backtest_execution_realism.report_max_zero_trade_share_default,
                    report_max_ret_cv=settings.backtest_execution_realism.report_max_ret_cv_default,
                    dimensions=_single_combo_dimensions(candidate),
                    max_combos=1,
                    shuffle_grid=False,
                    seed=42,
                    progress_every=settings.backtest_sensitivity.progress_every,
                    stop_on_error=False,
                    force=False,
                    tag=f"{as_of_tag or 'rerun'}-{candidate_label.lower()}-wf",
                    min_successful_splits=1,
                    report_top_n=3,
                    logger=effective_logger,
                )
                wf_grid_dir = wf_result.output_dir
                write_markdown_atomically(str(wf_grid_dir) + "\n", candidate_dir / "wf_grid_dir.txt")
                wf_summary_df = pl.read_parquet(wf_result.source_summary_path)
                if wf_summary_df.height > 0:
                    row = wf_summary_df.head(1).to_dicts()[0]
                    wf_rows.append(
                        {
                            "candidate_label": candidate_label,
                            "wf_grid_dir": str(wf_grid_dir),
                            "splits_covered": _safe_int(row.get("splits_covered")),
                            "expectancy_mean": _safe_float(row.get("expectancy_mean")),
                            "profit_factor_mean": _safe_float(row.get("profit_factor_mean")),
                            "ret_cv_median": _safe_float(row.get("ret_cv_median")),
                            "robustness_v2_mean": _safe_float(row.get("robustness_v2_mean")),
                            "exec_eligibility_rate_mean": _safe_float(row.get("exec_eligibility_rate_mean")),
                            "exec_suppressed_signal_share_mean": _safe_float(row.get("exec_suppressed_signal_share_mean")),
                        }
                    )

        candidate_row = {
            "candidate_label": candidate_label,
            "combo_id": candidate.get("combo_id"),
            "backtest_run_dir": str(bt_result.output_dir),
            "micro_grid_dir": str(micro_grid_dir) if micro_grid_dir is not None else None,
            "wf_grid_dir": str(wf_grid_dir) if wf_grid_dir is not None else None,
            "expected_expectancy": _safe_float(expected_snapshot.get("best_expectancy")),
            "observed_expectancy": _safe_float(observed.get("expectancy")),
            "delta_expectancy_abs": _safe_float(drift.get("delta_expectancy_abs")),
            "delta_expectancy_pct": _safe_float(drift.get("delta_expectancy_pct")),
            "expected_profit_factor": _safe_float(expected_snapshot.get("PF")),
            "observed_profit_factor": _safe_float(observed.get("profit_factor")),
            "delta_profit_factor_abs": _safe_float(drift.get("delta_profit_factor_abs")),
            "delta_profit_factor_pct": _safe_float(drift.get("delta_profit_factor_pct")),
            "expected_robustness_v2": _safe_float(expected_snapshot.get("robustness_v2")),
            "observed_robustness_v2": _safe_float(observed.get("robustness_v2")),
            "delta_robustness_v2_abs": _safe_float(drift.get("delta_robustness_v2_abs")),
            "expected_ret_cv": _safe_float(expected_snapshot.get("ret_cv")),
            "observed_ret_cv": _safe_float(observed.get("ret_cv")),
            "delta_ret_cv_abs": _safe_float(drift.get("delta_ret_cv_abs")),
            "delta_ret_cv_pct": _safe_float(drift.get("delta_ret_cv_pct")),
            "expected_trade_count": _safe_int(expected_snapshot.get("trade_count")),
            "observed_trade_count": _safe_int(observed.get("trade_count")),
            "delta_trade_count_abs": _safe_int(drift.get("delta_trade_count_abs")),
            "delta_trade_count_pct": _safe_float(drift.get("delta_trade_count_pct")),
            "expected_exec_eligibility_rate": _safe_float(expected_enriched.get("exec_eligibility_rate")),
            "observed_exec_eligibility_rate": _safe_float(observed.get("exec_eligibility_rate")),
            "delta_exec_eligibility_rate_pct": _safe_float(drift.get("delta_exec_eligibility_rate_pct")),
            "expected_overlay_match_rate": _safe_float(expected_enriched.get("overlay_match_rate")),
            "observed_overlay_match_rate": _safe_float(observed.get("overlay_match_rate")),
            "delta_overlay_match_rate_abs": _safe_float(drift.get("delta_overlay_match_rate_abs")),
            "drift_status": drift.get("drift_status"),
            "drift_reasons": ",".join(str(v) for v in (drift.get("flags") or [])),
        }
        candidate_rows.append(candidate_row)

        manifest_candidates.append(
            {
                "candidate_label": candidate_label,
                "combo_id": candidate.get("combo_id"),
                "backtest_run_dir": str(bt_result.output_dir),
                "backtest_summary_path": str(bt_result.summary_path),
                "trades_path": str(bt_result.trades_path),
                "candidate_dir": str(candidate_dir),
                "micro_grid_dir": str(micro_grid_dir) if micro_grid_dir is not None else None,
                "wf_grid_dir": str(wf_grid_dir) if wf_grid_dir is not None else None,
                "drift_status": drift.get("drift_status"),
                "drift_flags": drift.get("flags", []),
            }
        )

    finished = datetime.now(timezone.utc)
    candidate_table = pl.DataFrame(candidate_rows).sort("candidate_label")

    wf_summary_csv: Path | None = None
    wf_summary_md: Path | None = None
    if wf_rows:
        wf_df = pl.DataFrame(wf_rows).sort("candidate_label")
        wf_summary_csv = output_dir / "wf_single_combo_summary.csv"
        write_csv_atomically(wf_df, wf_summary_csv)
        lines = ["# WF Single Combo Summary", ""]
        for row in wf_df.to_dicts():
            lines.append(
                f"- {row.get('candidate_label')}: splits={row.get('splits_covered')} "
                f"exp_mean={row.get('expectancy_mean')} pf_mean={row.get('profit_factor_mean')} "
                f"ret_cv_median={row.get('ret_cv_median')} rob_v2_mean={row.get('robustness_v2_mean')}"
            )
        wf_summary_md = output_dir / "wf_single_combo_report.md"
        write_markdown_atomically("\n".join(lines) + "\n", wf_summary_md)

    rerun_manifest = {
        "rerun_id": run_id,
        "crp_version": "candidate_rerun_pack_v1",
        "pcp_pack_dir": str(pcp_pack_dir),
        "policy_packet_path": str(packet_path),
        "as_of_tag": as_of_tag or "none",
        "started_ts": started.isoformat(),
        "finished_ts": finished.isoformat(),
        "duration_sec": round((finished - started).total_seconds(), 3),
        "wf_run_dir": str(wf_run_dir) if wf_run_dir is not None else None,
        "override_input_file": str(override_input_file) if override_input_file is not None else None,
        "micro_grid_enabled": bool(micro_enabled),
        "candidates": manifest_candidates,
    }

    status_counts = (
        candidate_table.group_by("drift_status").len().rename({"len": "count"}).to_dicts()
        if candidate_table.height > 0 and "drift_status" in candidate_table.columns
        else []
    )
    summary = {
        "rerun_id": run_id,
        "candidate_count": int(candidate_table.height),
        "status_counts": status_counts,
        "warnings": warnings,
        "wf": {
            "enabled": wf_run_dir is not None,
            "candidates_with_wf": len(wf_rows),
            "wf_single_combo_summary_csv": str(wf_summary_csv) if wf_summary_csv is not None else None,
            "wf_single_combo_report_md": str(wf_summary_md) if wf_summary_md is not None else None,
        },
    }

    manifest_path, candidates_table_path, summary_path, report_path = write_candidate_rerun_reports(
        output_dir=output_dir,
        rerun_manifest=_finite_json(rerun_manifest),
        candidate_table=candidate_table,
        summary=_finite_json(summary),
    )

    effective_logger.info(
        "backtest.candidate_rerun.complete rerun_id=%s output=%s",
        run_id,
        output_dir,
    )
    return CandidateRerunPackResult(
        run_id=run_id,
        output_dir=output_dir,
        manifest_path=manifest_path,
        candidates_table_path=candidates_table_path,
        summary_path=summary_path,
        report_path=report_path,
    )


def summarize_candidate_rerun_pack(rerun_dir: Path) -> dict[str, Any]:
    """Validate CRP v1 artifacts and return concise sanity diagnostics."""

    manifest_path = rerun_dir / "rerun_manifest.json"
    table_path = rerun_dir / "rerun_candidates_table.csv"
    summary_path = rerun_dir / "rerun_summary.json"
    report_path = rerun_dir / "rerun_report.md"

    errors: list[str] = []
    warnings: list[str] = []
    for path in [manifest_path, table_path, summary_path, report_path]:
        if not path.exists():
            errors.append(f"missing_required_artifact: {path}")
    if errors:
        return {
            "rerun_dir": str(rerun_dir),
            "errors": errors,
            "warnings": warnings,
        }

    manifest = _read_json(manifest_path)
    summary = _read_json(summary_path)
    table = pl.read_csv(table_path)

    if table.height == 0:
        errors.append("empty_rerun_candidates_table")

    if "candidate_label" in table.columns:
        duplicated = (
            table.group_by("candidate_label")
            .len()
            .filter(pl.col("len") > 1)
        )
        if duplicated.height > 0:
            errors.append("duplicate_candidate_rows_in_table")

    numeric_cols = [col for col, dtype in table.schema.items() if dtype.is_numeric()]
    non_finite_cells = 0
    for col in numeric_cols:
        col_expr = pl.col(col).cast(pl.Float64, strict=False)
        finite = int(table.select(col_expr.is_finite().sum()).item() or 0)
        nulls = int(table.select(col_expr.is_null().sum()).item() or 0)
        non_finite_cells += max(table.height - finite - nulls, 0)
    if non_finite_cells > 0:
        errors.append(f"non_finite_cells_detected={non_finite_cells}")

    candidate_entries = manifest.get("candidates", [])
    if len(candidate_entries) != table.height:
        warnings.append(
            f"candidate_count_mismatch manifest={len(candidate_entries)} table={table.height}"
        )
    for item in candidate_entries:
        summary_path_item = Path(str(item.get("backtest_summary_path", "")))
        run_dir_item = Path(str(item.get("backtest_run_dir", "")))
        if not summary_path_item.exists():
            errors.append(f"missing_backtest_summary: {summary_path_item}")
        if not run_dir_item.exists():
            errors.append(f"missing_backtest_run_dir: {run_dir_item}")

    return {
        "rerun_dir": str(rerun_dir),
        "manifest_path": str(manifest_path),
        "table_path": str(table_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "candidate_count": int(table.height),
        "status_counts": summary.get("status_counts", []),
        "non_finite_cells": non_finite_cells,
        "errors": errors,
        "warnings": warnings,
    }
