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
            run_dir / "overlay_policy_mix_on_primary.csv",
            run_dir / "overlay_signal_effect_summary.json",
        ]
        for path in required_overlay:
            if not path.exists():
                errors.append(f"missing_overlay_artifact:{path.name}")
        veto_share = overlay.get("overlay_vetoed_signal_share")
        veto_value = float(veto_share) if veto_share is not None else None
        if veto_value is not None and not (0.0 <= veto_value <= 1.0):
            errors.append("overlay_vetoed_signal_share_out_of_range")
        overlay_info = {
            "overlay_mode": overlay.get("overlay_mode"),
            "overlay_match_rate": overlay.get("overlay_match_rate"),
            "overlay_unknown_rate": overlay.get("overlay_unknown_rate"),
            "overlay_vetoed_signal_share": overlay.get("overlay_vetoed_signal_share"),
            "overlay_direction_conflict_share": overlay.get("overlay_direction_conflict_share"),
        }

    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "top_states": by_state.head(15).to_dicts(),
        "top_symbols": by_symbol.head(15).to_dicts(),
        "nan_warnings": nan_warnings,
        "errors": errors,
        "policy_info": policy_info,
        "overlay_info": overlay_info,
    }
