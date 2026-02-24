"""Rolling-window stability diagnostics for state-labeled datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class StabilityDiagnosticsResult:
    """Stability artifact tables for one validation run."""

    rolling_state_metrics: pl.DataFrame
    state_stability_summary: pl.DataFrame
    transition_matrix_stability: pl.DataFrame


def _add_months(base: date, months: int) -> date:
    year = base.year + (base.month - 1 + months) // 12
    month = (base.month - 1 + months) % 12 + 1
    day = min(base.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return date(year, month, day)


def _window_ranges(min_date: date, max_date: date, *, window_months: int, step_months: int) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cursor = date(min_date.year, min_date.month, 1)
    while cursor <= max_date:
        end_exclusive = _add_months(cursor, window_months)
        end_inclusive = end_exclusive - timedelta(days=1)
        if end_inclusive >= min_date:
            windows.append((cursor, end_inclusive))
        cursor = _add_months(cursor, step_months)
    return windows


def _safe_cv(mean_value: float | None, std_value: float | None, eps: float) -> float | None:
    if mean_value is None or std_value is None:
        return None
    if not np.isfinite(mean_value) or not np.isfinite(std_value):
        return None
    return float(std_value / (abs(mean_value) + eps))


def _state_trend_slope(rolling_df: pl.DataFrame, value_col: str) -> pl.DataFrame:
    if rolling_df.height == 0 or value_col not in rolling_df.columns:
        return pl.DataFrame(schema={"state_id": pl.Int32, f"{value_col}_trend_slope": pl.Float64})

    def _map(group: pl.DataFrame) -> pl.DataFrame:
        ordered = group.sort("window_seq")
        x = ordered["window_seq"].to_numpy().astype(np.float64, copy=False)
        y = ordered[value_col].cast(pl.Float64, strict=False).to_numpy()
        finite = np.isfinite(y)
        if int(np.sum(finite)) < 2:
            slope = None
        else:
            slope = float(np.polyfit(x[finite], y[finite], deg=1)[0])
        return pl.DataFrame(
            {
                "state_id": [int(ordered["state_id"][0])],
                f"{value_col}_trend_slope": [slope],
            }
        )

    return rolling_df.group_by("state_id", maintain_order=True).map_groups(_map)


def _transition_matrix_from_state_rows(df: pl.DataFrame) -> dict[tuple[int, int], float]:
    if df.height == 0:
        return {}

    transitions = (
        df.sort(["ticker", "trade_date"])
        .with_columns(pl.col("state_id").shift(1).over("ticker").alias("state_prev"))
        .filter(pl.col("state_prev").is_not_null())
        .group_by(["state_prev", "state_id"])
        .len(name="transition_count")
        .with_columns(
            [
                pl.col("state_prev").cast(pl.Int32, strict=False),
                pl.col("state_id").cast(pl.Int32, strict=False),
            ]
        )
    )
    if transitions.height == 0:
        return {}

    normalized = (
        transitions.join(
            transitions.group_by("state_prev")
            .agg(pl.col("transition_count").sum().alias("from_total")),
            on="state_prev",
            how="left",
        )
        .with_columns((pl.col("transition_count") / pl.col("from_total")).alias("prob"))
        .select(["state_prev", "state_id", "prob"])
    )
    matrix: dict[tuple[int, int], float] = {}
    for row in normalized.to_dicts():
        key = (int(row["state_prev"]), int(row["state_id"]))
        matrix[key] = float(row["prob"])
    return matrix


def _transition_matrix_frobenius(global_matrix: dict[tuple[int, int], float], local_matrix: dict[tuple[int, int], float]) -> float:
    keys = set(global_matrix.keys()) | set(local_matrix.keys())
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        g = global_matrix.get(key, 0.0)
        l = local_matrix.get(key, 0.0)
        total += (g - l) ** 2
    return float(np.sqrt(total))


def build_rolling_stability_diagnostics(
    df: pl.DataFrame,
    *,
    window_months: int = 12,
    step_months: int = 3,
    eps: float = 1e-12,
    compute_transition_stability: bool = False,
) -> StabilityDiagnosticsResult:
    """Compute rolling state behavior and drift diagnostics."""

    if "trade_date" not in df.columns or "state_id" not in df.columns:
        raise ValueError("Dataframe must include trade_date and state_id for stability diagnostics.")
    if window_months < 1 or step_months < 1:
        raise ValueError("window_months and step_months must be >= 1")

    base = df.sort(["ticker", "trade_date"]).with_columns(
        [
            pl.col("trade_date").cast(pl.Date, strict=False).alias("trade_date"),
            pl.col("state_id").cast(pl.Int32, strict=False).alias("state_id"),
        ]
    )
    base = base.filter(pl.col("trade_date").is_not_null() & pl.col("state_id").is_not_null())
    if base.height == 0:
        empty_rolling = pl.DataFrame(
            schema={
                "window_seq": pl.Int32,
                "window_start": pl.Date,
                "window_end": pl.Date,
                "state_id": pl.Int32,
                "row_count": pl.Int64,
                "state_share": pl.Float64,
            }
        )
        empty_state = pl.DataFrame(schema={"state_id": pl.Int32})
        empty_transition = pl.DataFrame(
            schema={
                "window_seq": pl.Int32,
                "window_start": pl.Date,
                "window_end": pl.Date,
                "transition_count": pl.Int64,
                "frobenius_distance": pl.Float64,
            }
        )
        return StabilityDiagnosticsResult(
            rolling_state_metrics=empty_rolling,
            state_stability_summary=empty_state,
            transition_matrix_stability=empty_transition,
        )

    bounds = base.select(
        [
            pl.col("trade_date").min().alias("min_trade_date"),
            pl.col("trade_date").max().alias("max_trade_date"),
        ]
    ).to_dicts()[0]
    min_date = bounds["min_trade_date"]
    max_date = bounds["max_trade_date"]
    if min_date is None or max_date is None:
        raise ValueError("Failed to resolve date bounds for rolling stability diagnostics.")

    windows = _window_ranges(min_date, max_date, window_months=window_months, step_months=step_months)
    rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    global_matrix = _transition_matrix_from_state_rows(base)

    for window_seq, (window_start, window_end) in enumerate(windows):
        subset = base.filter(
            (pl.col("trade_date") >= pl.lit(window_start))
            & (pl.col("trade_date") <= pl.lit(window_end))
        )
        if subset.height == 0:
            continue

        total_rows = subset.height
        agg_exprs: list[pl.Expr] = [
            pl.len().alias("row_count"),
            pl.col("fwd_ret_10").mean().alias("fwd_ret_10_mean"),
            pl.col("fwd_ret_10").gt(0).mean().alias("fwd_ret_10_hit_rate"),
            pl.col("fwd_abs_ret_10").mean().alias("fwd_abs_ret_10_mean"),
            pl.col("fwd_vol_proxy_10").mean().alias("fwd_vol_proxy_10_mean"),
        ]
        for column in ("tmf_21", "flow_bias_20", "oscillation_index_20"):
            if column in subset.columns:
                agg_exprs.append(pl.col(column).mean().alias(f"{column}_mean"))

        grouped = subset.group_by("state_id").agg(agg_exprs).sort("state_id")
        grouped = grouped.with_columns(
            [
                pl.lit(window_seq).cast(pl.Int32).alias("window_seq"),
                pl.lit(window_start).cast(pl.Date).alias("window_start"),
                pl.lit(window_end).cast(pl.Date).alias("window_end"),
                (pl.col("row_count") / float(total_rows)).alias("state_share"),
            ]
        )
        rows.extend(grouped.to_dicts())

        if compute_transition_stability:
            local_matrix = _transition_matrix_from_state_rows(subset)
            frob = _transition_matrix_frobenius(global_matrix, local_matrix)
            transition_rows.append(
                {
                    "window_seq": window_seq,
                    "window_start": window_start,
                    "window_end": window_end,
                    "transition_count": int(
                        subset.with_columns(pl.col("state_id").shift(1).over("ticker").alias("state_prev"))
                        .filter(pl.col("state_prev").is_not_null())
                        .height
                    ),
                    "frobenius_distance": frob,
                }
            )

    rolling = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "window_seq": pl.Int32,
            "window_start": pl.Date,
            "window_end": pl.Date,
            "state_id": pl.Int32,
            "row_count": pl.Int64,
            "state_share": pl.Float64,
        }
    )

    if rolling.height == 0:
        summary = pl.DataFrame(schema={"state_id": pl.Int32})
    else:
        summary = (
            rolling.group_by("state_id")
            .agg(
                [
                    pl.len().alias("window_count"),
                    pl.col("state_share").mean().alias("state_share_mean"),
                    pl.col("state_share").std(ddof=0).alias("state_share_std"),
                    pl.col("fwd_ret_10_mean").mean().alias("fwd_ret_10_mean_mean"),
                    pl.col("fwd_ret_10_mean").std(ddof=0).alias("fwd_ret_10_mean_std"),
                    pl.col("fwd_ret_10_hit_rate").mean().alias("fwd_ret_10_hit_rate_mean"),
                    pl.col("fwd_ret_10_hit_rate").std(ddof=0).alias("fwd_ret_10_hit_rate_std"),
                    pl.col("fwd_abs_ret_10_mean").mean().alias("fwd_abs_ret_10_mean_mean"),
                    pl.col("fwd_abs_ret_10_mean").std(ddof=0).alias("fwd_abs_ret_10_mean_std"),
                    pl.col("fwd_vol_proxy_10_mean").mean().alias("fwd_vol_proxy_10_mean_mean"),
                    pl.col("fwd_vol_proxy_10_mean").std(ddof=0).alias("fwd_vol_proxy_10_mean_std"),
                    pl.col("fwd_ret_10_mean").gt(0).mean().alias("fwd_ret_10_sign_stability"),
                ]
            )
            .sort("state_id")
            .with_columns(
                [
                    pl.struct(["fwd_ret_10_mean_mean", "fwd_ret_10_mean_std"])
                    .map_elements(lambda x: _safe_cv(x["fwd_ret_10_mean_mean"], x["fwd_ret_10_mean_std"], eps), return_dtype=pl.Float64)
                    .alias("ret_mean_cv"),
                    pl.struct(["state_share_mean", "state_share_std"])
                    .map_elements(lambda x: _safe_cv(x["state_share_mean"], x["state_share_std"], eps), return_dtype=pl.Float64)
                    .alias("share_cv"),
                ]
            )
        )
        if "tmf_21_mean" in rolling.columns:
            summary = summary.join(
                rolling.group_by("state_id").agg(pl.col("tmf_21_mean").mean().alias("tmf_21_mean_mean")),
                on="state_id",
                how="left",
            )
        if "flow_bias_20_mean" in rolling.columns:
            summary = summary.join(
                rolling.group_by("state_id").agg(pl.col("flow_bias_20_mean").mean().alias("flow_bias_20_mean_mean")),
                on="state_id",
                how="left",
            )
        if "oscillation_index_20_mean" in rolling.columns:
            summary = summary.join(
                rolling.group_by("state_id").agg(
                    pl.col("oscillation_index_20_mean").mean().alias("oscillation_index_20_mean_mean")
                ),
                on="state_id",
                how="left",
            )

        trend = _state_trend_slope(rolling, "fwd_ret_10_mean")
        summary = summary.join(trend, on="state_id", how="left")

    transition_matrix_stability = (
        pl.DataFrame(transition_rows).sort("window_seq")
        if transition_rows
        else pl.DataFrame(
            schema={
                "window_seq": pl.Int32,
                "window_start": pl.Date,
                "window_end": pl.Date,
                "transition_count": pl.Int64,
                "frobenius_distance": pl.Float64,
            }
        )
    )

    return StabilityDiagnosticsResult(
        rolling_state_metrics=rolling,
        state_stability_summary=summary,
        transition_matrix_stability=transition_matrix_stability,
    )
