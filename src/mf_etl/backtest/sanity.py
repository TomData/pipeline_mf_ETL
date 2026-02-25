"""Sanity checks and compact summaries for backtest artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _nan_count(df: pl.DataFrame) -> int:
    if df.height == 0:
        return 0
    numeric = [name for name, dtype in df.schema.items() if dtype.is_numeric()]
    total = 0
    for col in numeric:
        total += int(df.select(pl.col(col).cast(pl.Float64, strict=False).is_nan().fill_null(False).sum()).item())
    return total


def summarize_backtest_run(run_dir: Path) -> dict[str, Any]:
    """Load and validate one backtest run directory."""

    summary = json.loads(_require(run_dir / "backtest_summary.json").read_text(encoding="utf-8"))
    trades = pl.read_parquet(_require(run_dir / "trades.parquet"))
    by_state = pl.read_csv(_require(run_dir / "summary_by_state.csv"))
    by_symbol = pl.read_csv(_require(run_dir / "summary_by_symbol.csv"))

    errors: list[str] = []
    if trades.height > 0:
        if int(trades.filter(pl.col("hold_bars_realized") < 0).height) > 0:
            errors.append("negative_hold_bars")
        if int(trades.filter(pl.col("entry_date") > pl.col("exit_date")).height) > 0:
            errors.append("entry_date_after_exit_date")
        if int(trades.select(pl.col("position_id").n_unique()).item()) != trades.height:
            errors.append("duplicate_position_id")
        bad_price = trades.filter(
            (pl.col("is_valid_trade") == True)
            & (
                (~pl.col("entry_price").cast(pl.Float64, strict=False).is_finite())
                | (~pl.col("exit_price").cast(pl.Float64, strict=False).is_finite())
                | (pl.col("entry_price") <= 0)
                | (pl.col("exit_price") <= 0)
            )
        )
        if bad_price.height > 0:
            errors.append("non_finite_or_nonpositive_prices_in_valid_trades")

    headline = summary.get("headline", {})
    if int(headline.get("trade_count", 0)) != int(trades.height):
        errors.append("trade_count_mismatch")

    nan_warnings = {
        "trades_nan_count": _nan_count(trades),
        "by_state_nan_count": _nan_count(by_state),
        "by_symbol_nan_count": _nan_count(by_symbol),
    }

    policy_info: dict[str, Any] | None = None
    policy_snapshot = run_dir / "policy_snapshot.json"
    if policy_snapshot.exists():
        policy_payload = json.loads(policy_snapshot.read_text(encoding="utf-8"))
        policy_info = {
            "allow_count": policy_payload.get("summary", {}).get("allow_count"),
            "watch_count": policy_payload.get("summary", {}).get("watch_count"),
            "block_count": policy_payload.get("summary", {}).get("block_count"),
        }

    overlay_info: dict[str, Any] | None = None
    overlay = summary.get("overlay") if isinstance(summary.get("overlay"), dict) else {}
    if overlay and bool(overlay.get("overlay_enabled")):
        required_overlay = [
            run_dir / "overlay_join_summary.json",
            run_dir / "overlay_coverage_verdict.json",
            run_dir / "overlay_policy_mix_on_primary.csv",
            run_dir / "overlay_signal_effect_summary.json",
        ]
        for path in required_overlay:
            if not path.exists():
                errors.append(f"missing_overlay_artifact:{path.name}")
        overlay_join_summary_path = run_dir / "overlay_join_summary.json"
        overlay_verdict_path = run_dir / "overlay_coverage_verdict.json"
        overlay_join_payload = (
            json.loads(overlay_join_summary_path.read_text(encoding="utf-8"))
            if overlay_join_summary_path.exists()
            else {}
        )
        overlay_verdict_payload = (
            json.loads(overlay_verdict_path.read_text(encoding="utf-8"))
            if overlay_verdict_path.exists()
            else {}
        )
        match_rate = overlay_join_payload.get("match_rate")
        if match_rate is None:
            errors.append("overlay_match_rate_missing")
        else:
            match_rate_f = float(match_rate)
            if not (0.0 <= match_rate_f <= 1.0):
                errors.append("overlay_match_rate_out_of_range")
        unknown_rate = overlay_join_payload.get("unknown_rate")
        if unknown_rate is None:
            errors.append("overlay_unknown_rate_missing")
        else:
            unknown_rate_f = float(unknown_rate)
            if not (0.0 <= unknown_rate_f <= 1.0):
                errors.append("overlay_unknown_rate_out_of_range")
        coverage_status = str(overlay_verdict_payload.get("status", ""))
        coverage_mode = str(overlay.get("overlay_coverage_mode", "warn_only"))
        coverage_bypass = bool(overlay.get("overlay_coverage_bypass", False))
        if coverage_mode == "strict_fail" and (not coverage_bypass) and coverage_status.startswith("FAIL"):
            errors.append("overlay_strict_fail_verdict_present")
        veto_share = overlay.get("overlay_vetoed_signal_share")
        veto_value = float(veto_share) if veto_share is not None else None
        if veto_value is not None and not (0.0 <= veto_value <= 1.0):
            errors.append("overlay_vetoed_signal_share_out_of_range")
        overlay_info = {
            "overlay_mode": overlay.get("overlay_mode"),
            "overlay_coverage_mode": overlay.get("overlay_coverage_mode"),
            "overlay_coverage_status": overlay.get("overlay_coverage_status"),
            "overlay_match_rate": overlay.get("overlay_match_rate"),
            "overlay_unknown_rate": overlay.get("overlay_unknown_rate"),
            "overlay_vetoed_signal_share": overlay.get("overlay_vetoed_signal_share"),
            "overlay_direction_conflict_share": overlay.get("overlay_direction_conflict_share"),
        }

    execution_info: dict[str, Any] | None = None
    execution = summary.get("execution") if isinstance(summary.get("execution"), dict) else {}
    if execution and bool(execution.get("filters_enabled_or_profile_non_none")):
        required_execution = [
            run_dir / "execution_filter_summary.json",
            run_dir / "execution_filter_by_reason.csv",
            run_dir / "execution_filter_by_year.csv",
            run_dir / "execution_trade_context_summary.json",
        ]
        for path in required_execution:
            if not path.exists():
                errors.append(f"missing_execution_artifact:{path.name}")
        exec_elig = execution.get("exec_eligibility_rate")
        exec_elig_val = float(exec_elig) if exec_elig is not None else None
        if exec_elig_val is not None and not (0.0 <= exec_elig_val <= 1.0):
            errors.append("exec_eligibility_rate_out_of_range")
        exec_supp = execution.get("exec_suppressed_signal_share")
        exec_supp_val = float(exec_supp) if exec_supp is not None else None
        if exec_supp_val is not None and not (0.0 <= exec_supp_val <= 1.0):
            errors.append("exec_suppressed_signal_share_out_of_range")
        for field in [
            "exec_suppressed_by_price_share",
            "exec_suppressed_by_liquidity_share",
            "exec_suppressed_by_vol_share",
            "exec_suppressed_by_warmup_share",
        ]:
            value = execution.get(field)
            value_f = float(value) if value is not None else None
            if value_f is not None and not (0.0 <= value_f <= 1.0):
                errors.append(f"{field}_out_of_range")
        for field in [
            "exec_suppressed_by_price_count",
            "exec_suppressed_by_liquidity_count",
            "exec_suppressed_by_vol_count",
            "exec_suppressed_by_warmup_count",
        ]:
            value = execution.get(field)
            if value is None:
                continue
            value_f = float(value)
            if value_f < 0 or abs(value_f - round(value_f)) > 1e-9:
                errors.append(f"{field}_invalid")
        before_cnt = int(execution.get("candidate_signals_before_filters", 0) or 0)
        after_cnt = int(execution.get("candidate_signals_after_filters", 0) or 0)
        suppressed = int(execution.get("exec_suppressed_signal_count", 0) or 0)
        if after_cnt > before_cnt:
            errors.append("execution_candidate_count_invalid_order")
        if suppressed != (before_cnt - after_cnt):
            errors.append("execution_suppressed_count_mismatch")
        by_reason_path = run_dir / "execution_filter_by_reason.csv"
        if by_reason_path.exists():
            by_reason = pl.read_csv(by_reason_path)
            if "suppressed_signal_count" in by_reason.columns:
                reason_sum = int(
                    by_reason.select(pl.col("suppressed_signal_count").cast(pl.Int64, strict=False).sum()).item()
                    or 0
                )
                if reason_sum > suppressed:
                    errors.append("execution_reason_count_sum_above_suppressed")
            for field in ["suppressed_signal_share", "row_share_total"]:
                if field in by_reason.columns:
                    bad_share = int(
                        by_reason.filter(
                            (~pl.col(field).cast(pl.Float64, strict=False).is_finite())
                            | (pl.col(field).cast(pl.Float64, strict=False) < 0)
                            | (pl.col(field).cast(pl.Float64, strict=False) > 1)
                        ).height
                    )
                    if bad_share > 0:
                        errors.append(f"execution_by_reason_{field}_invalid")
        execution_info = {
            "execution_profile": execution.get("execution_profile"),
            "realism_profile_status": execution.get("realism_profile_status"),
            "exec_eligibility_rate": execution.get("exec_eligibility_rate"),
            "exec_suppressed_signal_share": execution.get("exec_suppressed_signal_share"),
            "exec_trade_avg_dollar_vol_20": execution.get("exec_trade_avg_dollar_vol_20"),
        }

    verdict_label = summary.get("verdict_label")
    if verdict_label in {"PRIMARY_CANDIDATE", "SECONDARY_CANDIDATE"}:
        trade_count = int(headline.get("trade_count", 0) or 0)
        if trade_count <= 0:
            errors.append("candidate_verdict_with_zero_trades")

    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "top_states": by_state.head(15).to_dicts(),
        "top_symbols": by_symbol.head(15).to_dicts(),
        "nan_warnings": nan_warnings,
        "errors": errors,
        "policy_info": policy_info,
        "overlay_info": overlay_info,
        "execution_info": execution_info,
    }
