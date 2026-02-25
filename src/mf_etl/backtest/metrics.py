"""Performance metrics and summaries for backtest harness outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def _finite_array(series: pl.Series) -> np.ndarray:
    arr = series.cast(pl.Float64, strict=False).to_numpy()
    return arr[np.isfinite(arr)]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _profit_factor(returns: np.ndarray) -> float | None:
    if returns.size == 0:
        return None
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_win = float(np.sum(wins)) if wins.size > 0 else 0.0
    gross_loss = float(np.sum(losses)) if losses.size > 0 else 0.0
    if gross_loss == 0:
        return None if gross_win == 0 else float("inf")
    return abs(gross_win / gross_loss)


def _expectancy(returns: np.ndarray) -> float | None:
    if returns.size == 0:
        return None
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = float(wins.size / returns.size)
    avg_win = float(np.mean(wins)) if wins.size > 0 else 0.0
    avg_loss = float(np.mean(losses)) if losses.size > 0 else 0.0
    return (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)


def compute_trade_summary(
    trades: pl.DataFrame,
    *,
    signal_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Compute headline trade metrics."""

    if trades.height == 0:
        return {
            "trade_count": 0,
            "long_trade_count": 0,
            "short_trade_count": 0,
            "win_rate": None,
            "avg_return": None,
            "median_return": None,
            "avg_win": None,
            "avg_loss": None,
            "profit_factor": None,
            "expectancy": None,
            "return_std": None,
            "avg_hold_bars": None,
            "median_hold_bars": None,
            "skipped_signal_count": int(signal_diagnostics.get("skipped_no_next_bar_entry", 0)) + int(signal_diagnostics.get("skipped_due_overlap", 0)),
            "invalid_trade_count": int(signal_diagnostics.get("invalid_trade_count", 0)),
        }

    valid = trades.filter(pl.col("is_valid_trade") == True)  # noqa: E712
    returns = _finite_array(valid["net_return"]) if valid.height > 0 else np.array([], dtype=float)

    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = float(wins.size / returns.size) if returns.size > 0 else None

    summary = {
        "trade_count": int(trades.height),
        "long_trade_count": int(trades.filter(pl.col("side") == "LONG").height),
        "short_trade_count": int(trades.filter(pl.col("side") == "SHORT").height),
        "win_rate": _safe_float(win_rate),
        "avg_return": _safe_float(float(np.mean(returns)) if returns.size > 0 else None),
        "median_return": _safe_float(float(np.median(returns)) if returns.size > 0 else None),
        "avg_win": _safe_float(float(np.mean(wins)) if wins.size > 0 else None),
        "avg_loss": _safe_float(float(np.mean(losses)) if losses.size > 0 else None),
        "profit_factor": _safe_float(_profit_factor(returns)),
        "expectancy": _safe_float(_expectancy(returns)),
        "return_std": _safe_float(float(np.std(returns, ddof=0)) if returns.size > 0 else None),
        "avg_hold_bars": _safe_float(valid.select(pl.col("hold_bars_realized").mean()).item() if valid.height > 0 else None),
        "median_hold_bars": _safe_float(valid.select(pl.col("hold_bars_realized").median()).item() if valid.height > 0 else None),
        "skipped_signal_count": int(signal_diagnostics.get("skipped_no_next_bar_entry", 0)) + int(signal_diagnostics.get("skipped_due_overlap", 0)),
        "invalid_trade_count": int(signal_diagnostics.get("invalid_trade_count", 0)),
    }
    return summary


def build_summary_by_state(trades: pl.DataFrame) -> pl.DataFrame:
    """Build by-entry-state summary table."""

    if trades.height == 0:
        return pl.DataFrame(schema={"entry_state_id": pl.Int32})

    valid = trades.filter(pl.col("is_valid_trade") == True).with_columns(
        pl.col("net_return").cast(pl.Float64, strict=False)
    )
    if valid.height == 0:
        return pl.DataFrame(schema={"entry_state_id": pl.Int32})

    grouped = valid.group_by(
        [
            "entry_state_id",
            "entry_state_label",
            "entry_state_class",
            "entry_state_direction_hint",
        ]
    ).agg(
        [
            pl.len().alias("trade_count"),
            pl.col("net_return").mean().alias("avg_return"),
            pl.col("net_return").median().alias("median_return"),
            pl.col("hold_bars_realized").mean().alias("avg_hold_bars"),
            pl.col("net_return").std(ddof=0).alias("return_std"),
            pl.col("net_return").filter(pl.col("net_return") > 0).mean().alias("avg_win"),
            pl.col("net_return").filter(pl.col("net_return") <= 0).mean().alias("avg_loss"),
            pl.col("net_return").filter(pl.col("net_return") > 0).len().alias("win_count"),
            pl.col("net_return").sum().alias("net_pnl"),
            pl.col("entry_state_score").mean().alias("state_score"),
        ]
    )

    total_trades = max(1, int(valid.height))
    total_abs_pnl = abs(float(valid.select(pl.col("net_return").sum()).item() or 0.0))
    if total_abs_pnl <= 0:
        total_abs_pnl = 1.0

    out = grouped.with_columns(
        [
            (pl.col("win_count") / pl.col("trade_count")).alias("win_rate"),
            (pl.col("net_pnl") / pl.lit(total_abs_pnl)).alias("contribution_share"),
            (pl.col("trade_count") / pl.lit(total_trades)).alias("trade_share"),
            (pl.col("avg_win") / pl.col("avg_loss").abs()).alias("profit_factor_proxy"),
            (
                (pl.col("win_count") / pl.col("trade_count")) * pl.col("avg_win")
                + (1 - (pl.col("win_count") / pl.col("trade_count"))) * pl.col("avg_loss")
            ).alias("expectancy"),
        ]
    ).sort("trade_count", descending=True)
    return out


def build_summary_by_symbol(trades: pl.DataFrame) -> pl.DataFrame:
    """Build by-symbol summary table."""

    if trades.height == 0:
        return pl.DataFrame(schema={"ticker": pl.String})
    valid = trades.filter(pl.col("is_valid_trade") == True).with_columns(
        pl.col("net_return").cast(pl.Float64, strict=False)
    )
    if valid.height == 0:
        return pl.DataFrame(schema={"ticker": pl.String})

    total_pnl = float(valid.select(pl.col("net_return").sum()).item() or 0.0)
    denom = total_pnl if abs(total_pnl) > 1e-12 else 1.0

    out = valid.group_by("ticker").agg(
        [
            pl.len().alias("trade_count"),
            pl.col("net_return").mean().alias("avg_return"),
            pl.col("net_return").median().alias("median_return"),
            pl.col("net_return").std(ddof=0).alias("return_std"),
            pl.col("net_return").filter(pl.col("net_return") > 0).len().alias("win_count"),
            pl.col("hold_bars_realized").mean().alias("avg_hold_bars"),
            pl.col("net_return").sum().alias("net_pnl"),
        ]
    ).with_columns(
        [
            (pl.col("win_count") / pl.col("trade_count")).alias("win_rate"),
            (pl.col("net_pnl") / pl.lit(denom)).alias("contribution"),
        ]
    ).sort("net_pnl", descending=True)
    return out


def build_exit_reason_summary(trades: pl.DataFrame) -> list[dict[str, Any]]:
    """Return exit-reason distribution diagnostics."""

    if trades.height == 0 or "exit_reason" not in trades.columns:
        return []
    out = (
        trades.group_by("exit_reason")
        .agg(
            [
                pl.len().alias("trade_count"),
                pl.col("net_return").cast(pl.Float64, strict=False).mean().alias("avg_return"),
                pl.col("net_return").cast(pl.Float64, strict=False).median().alias("median_return"),
            ]
        )
        .sort("trade_count", descending=True)
    )
    return out.to_dicts()
