"""Forward outcome label builders for cluster validation."""

from __future__ import annotations

import numpy as np
import polars as pl


def _forward_vol_proxy_from_daily_returns(values: np.ndarray, horizon: int) -> np.ndarray:
    """Compute forward-looking volatility proxy from daily returns.

    For row `t`, this computes std of daily returns for bars `(t+1 .. t+horizon)`.
    """

    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        window = values[idx + 1 : idx + 1 + horizon]
        if window.shape[0] == horizon and np.all(np.isfinite(window)):
            out[idx] = float(np.std(window, ddof=0))
    return out


def _normalize_non_finite_to_null(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Convert non-finite values (NaN/inf/-inf) to null for selected columns."""

    existing = [column for column in columns if column in df.columns]
    if not existing:
        return df
    expressions: list[pl.Expr] = []
    for column in existing:
        value = pl.col(column).cast(pl.Float64, strict=False)
        expressions.append(
            pl.when(value.is_finite().fill_null(False))
            .then(value)
            .otherwise(None)
            .alias(column)
        )
    return df.with_columns(expressions)


def add_forward_outcomes(
    df: pl.DataFrame,
    *,
    windows: list[int] | tuple[int, ...] = (5, 10, 20),
    vol_horizon: int = 10,
) -> pl.DataFrame:
    """Add forward return and forward volatility proxy columns by ticker."""

    if "ticker" not in df.columns or "trade_date" not in df.columns or "close" not in df.columns:
        raise ValueError("Dataframe must include ticker, trade_date, and close columns.")

    normalized_windows = sorted({int(window) for window in windows if int(window) > 0})
    if not normalized_windows:
        raise ValueError("forward windows must include at least one positive integer.")

    sorted_df = df.sort(["ticker", "trade_date"])
    expressions: list[pl.Expr] = []
    for window in normalized_windows:
        expressions.append(
            (pl.col("close").shift(-window).over("ticker") / pl.col("close") - 1.0).alias(f"fwd_ret_{window}")
        )

    out = sorted_df.with_columns(expressions)
    if 10 in normalized_windows:
        out = out.with_columns(pl.col("fwd_ret_10").abs().alias("fwd_abs_ret_10"))

    out = out.with_columns((pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("_daily_ret"))

    def _apply_group(group: pl.DataFrame) -> pl.DataFrame:
        values = group["_daily_ret"].to_numpy()
        fwd_vol = _forward_vol_proxy_from_daily_returns(values, vol_horizon)
        return group.with_columns(pl.Series(name="fwd_vol_proxy_10", values=fwd_vol))

    out = out.group_by("ticker", maintain_order=True).map_groups(_apply_group)
    forward_columns = [f"fwd_ret_{window}" for window in normalized_windows]
    if 10 in normalized_windows:
        forward_columns.append("fwd_abs_ret_10")
    forward_columns.append("fwd_vol_proxy_10")
    out = _normalize_non_finite_to_null(out, forward_columns)
    return out.drop("_daily_ret")
