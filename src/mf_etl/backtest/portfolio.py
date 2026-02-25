"""Portfolio-level aggregation utilities for backtest harness."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _max_drawdown_days(drawdowns: list[float]) -> int:
    longest = 0
    current = 0
    for dd in drawdowns:
        if dd < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def build_daily_equity_curve(
    trades: pl.DataFrame,
    *,
    capital_base: float,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Build a simple realized-PnL daily equity curve from trade exits."""

    if trades.height == 0 or "exit_date" not in trades.columns or "net_return" not in trades.columns:
        empty = pl.DataFrame(schema={"trade_date": pl.Date, "daily_return": pl.Float64, "equity": pl.Float64, "drawdown": pl.Float64})
        return empty, {
            "capital_base": capital_base,
            "cumulative_return": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_days": 0,
            "daily_vol": None,
            "sharpe_proxy": None,
            "cagr_proxy": None,
        }

    exits = (
        trades.filter(pl.col("is_valid_trade") == True)  # noqa: E712
        .select([pl.col("exit_date").cast(pl.Date).alias("trade_date"), pl.col("net_return").cast(pl.Float64, strict=False)])
        .drop_nulls(["trade_date", "net_return"])
    )
    if exits.height == 0:
        empty = pl.DataFrame(schema={"trade_date": pl.Date, "daily_return": pl.Float64, "equity": pl.Float64, "drawdown": pl.Float64})
        return empty, {
            "capital_base": capital_base,
            "cumulative_return": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_days": 0,
            "daily_vol": None,
            "sharpe_proxy": None,
            "cagr_proxy": None,
        }

    daily = exits.group_by("trade_date").agg(pl.col("net_return").mean().alias("daily_return")).sort("trade_date")
    returns = [float(v) for v in daily["daily_return"].to_list()]

    equity_vals: list[float] = []
    eq = float(capital_base)
    for ret in returns:
        if not np.isfinite(ret):
            ret = 0.0
        eq *= (1.0 + ret)
        equity_vals.append(eq)

    curve = daily.with_columns(pl.Series("equity", equity_vals))
    peak = curve.select(pl.col("equity").cum_max()).to_series().to_list()
    drawdowns = [((e / p) - 1.0) if p > 0 else 0.0 for e, p in zip(equity_vals, peak)]
    curve = curve.with_columns(pl.Series("drawdown", drawdowns))

    cumulative_return = (equity_vals[-1] / capital_base) - 1.0 if equity_vals else 0.0
    daily_ret_arr = np.array(returns, dtype=float)
    finite_ret = daily_ret_arr[np.isfinite(daily_ret_arr)]
    daily_vol = float(np.std(finite_ret, ddof=0)) if finite_ret.size > 0 else None
    sharpe = None
    if daily_vol is not None and daily_vol > 0 and finite_ret.size > 1:
        sharpe = float((np.mean(finite_ret) / daily_vol) * np.sqrt(252.0))

    cagr = None
    if curve.height > 1:
        start = curve["trade_date"][0]
        end = curve["trade_date"][-1]
        if start is not None and end is not None:
            days = max(1, int((end - start).days))
            years = days / 365.25
            if years > 0 and equity_vals[-1] > 0 and capital_base > 0:
                cagr = float((equity_vals[-1] / capital_base) ** (1.0 / years) - 1.0)

    metrics = {
        "capital_base": float(capital_base),
        "cumulative_return": _safe_float(cumulative_return),
        "max_drawdown": _safe_float(float(min(drawdowns)) if drawdowns else 0.0),
        "max_drawdown_days": int(_max_drawdown_days(drawdowns)),
        "daily_vol": _safe_float(daily_vol),
        "sharpe_proxy": _safe_float(sharpe),
        "cagr_proxy": _safe_float(cagr),
    }
    return curve, metrics
