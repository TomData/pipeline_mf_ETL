"""Transition event-study helpers for state-labeled datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl

EVENT_CONTEXT_COLUMNS: tuple[str, ...] = (
    "tmf_21",
    "flow_bias_20",
    "oscillation_index_20",
    "flow_activity_20",
)
FORWARD_COLUMNS: tuple[str, ...] = (
    "fwd_ret_5",
    "fwd_ret_10",
    "fwd_ret_20",
)


@dataclass(frozen=True, slots=True)
class TransitionEventStudyResult:
    """Event-study tables for state transition analyses."""

    transition_events: pl.DataFrame
    transition_event_path_summary: pl.DataFrame
    transition_event_summary: pl.DataFrame
    transition_top_codes: dict[str, Any]


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _compute_daily_returns(close: np.ndarray) -> np.ndarray:
    daily = np.full(close.shape[0], np.nan, dtype=np.float64)
    for idx in range(1, close.shape[0]):
        prev = close[idx - 1]
        curr = close[idx]
        if np.isfinite(prev) and np.isfinite(curr) and prev != 0.0:
            daily[idx] = float(curr / prev - 1.0)
    return daily


def run_transition_event_study(
    df: pl.DataFrame,
    *,
    window_pre: int = 10,
    window_post: int = 20,
    min_events_per_transition: int = 50,
) -> TransitionEventStudyResult:
    """Build transition event and event-window summaries around state changes."""

    required = ["ticker", "trade_date", "state_id"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for event studies: {', '.join(missing)}")
    if "close" not in df.columns:
        raise ValueError("Event study requires close column for return path calculations.")

    sorted_df = df.sort(["ticker", "trade_date"]).with_columns(
        [
            pl.col("close").cast(pl.Float64, strict=False).alias("close"),
            pl.col("state_id").cast(pl.Int32, strict=False).alias("state_id"),
        ]
    )

    event_rows: list[dict[str, Any]] = []
    path_acc: dict[tuple[int, int], dict[str, float]] = {}

    context_cols = [column for column in EVENT_CONTEXT_COLUMNS if column in sorted_df.columns]
    forward_cols = [column for column in FORWARD_COLUMNS if column in sorted_df.columns]

    grouped = sorted_df.partition_by("ticker", maintain_order=True)
    for group in grouped:
        ticker = str(group["ticker"][0])
        n_rows = group.height
        if n_rows < 2:
            continue

        trade_dates = group["trade_date"].to_numpy()
        states = group["state_id"].to_numpy().astype(np.int32, copy=False)
        close = group["close"].to_numpy().astype(np.float64, copy=False)
        daily_ret = _compute_daily_returns(close)

        context_arrays = {
            column: group[column].cast(pl.Float64, strict=False).to_numpy() for column in context_cols
        }
        forward_arrays = {
            column: group[column].cast(pl.Float64, strict=False).to_numpy() for column in forward_cols
        }

        event_idx_in_ticker = 0
        for idx in range(1, n_rows):
            prev_state = int(states[idx - 1])
            curr_state = int(states[idx])
            if prev_state == curr_state:
                continue

            transition_code = int(prev_state * 1000 + curr_state)
            base_close = close[idx]
            event_row: dict[str, Any] = {
                "ticker": ticker,
                "event_trade_date": trade_dates[idx],
                "prev_state_id": prev_state,
                "next_state_id": curr_state,
                "state_id": curr_state,
                "transition_code": transition_code,
                "event_index_in_ticker": event_idx_in_ticker,
            }
            for column in context_cols:
                value = context_arrays[column][idx]
                event_row[column] = _safe_float(float(value)) if np.isfinite(value) else None
            for column in forward_cols:
                value = forward_arrays[column][idx]
                event_row[column] = _safe_float(float(value)) if np.isfinite(value) else None
            event_rows.append(event_row)
            event_idx_in_ticker += 1

            for rel_bar in range(-window_pre, window_post + 1):
                target = idx + rel_bar
                if target < 0 or target >= n_rows:
                    continue

                day_ret_val = daily_ret[target]
                if np.isfinite(base_close) and np.isfinite(close[target]) and base_close != 0.0:
                    cumulative = float(close[target] / base_close - 1.0)
                else:
                    cumulative = np.nan
                abs_daily = abs(day_ret_val) if np.isfinite(day_ret_val) else np.nan

                key = (transition_code, rel_bar)
                if key not in path_acc:
                    path_acc[key] = {
                        "count_points": 0.0,
                        "count_daily": 0.0,
                        "count_abs_daily": 0.0,
                        "count_cum": 0.0,
                        "sum_daily": 0.0,
                        "sum_abs_daily": 0.0,
                        "sum_cum": 0.0,
                    }
                bucket = path_acc[key]
                bucket["count_points"] += 1.0
                if np.isfinite(day_ret_val):
                    bucket["sum_daily"] += float(day_ret_val)
                    bucket["count_daily"] += 1.0
                if np.isfinite(abs_daily):
                    bucket["sum_abs_daily"] += float(abs_daily)
                    bucket["count_abs_daily"] += 1.0
                if np.isfinite(cumulative):
                    bucket["sum_cum"] += float(cumulative)
                    bucket["count_cum"] += 1.0

    transition_events = pl.DataFrame(event_rows) if event_rows else pl.DataFrame(
        schema={
            "ticker": pl.String,
            "event_trade_date": pl.Date,
            "prev_state_id": pl.Int32,
            "next_state_id": pl.Int32,
            "state_id": pl.Int32,
            "transition_code": pl.Int32,
            "event_index_in_ticker": pl.Int32,
        }
    )

    path_rows: list[dict[str, Any]] = []
    for (transition_code, rel_bar), bucket in path_acc.items():
        count = int(bucket["count_points"])
        if count == 0:
            continue
        path_rows.append(
            {
                "transition_code": transition_code,
                "rel_bar": int(rel_bar),
                "count_points": count,
                "daily_ret_1_mean": (
                    _safe_float(bucket["sum_daily"] / bucket["count_daily"])
                    if bucket["count_daily"] > 0
                    else None
                ),
                "abs_daily_ret_mean": (
                    _safe_float(bucket["sum_abs_daily"] / bucket["count_abs_daily"])
                    if bucket["count_abs_daily"] > 0
                    else None
                ),
                "cumulative_ret_from_event_mean": (
                    _safe_float(bucket["sum_cum"] / bucket["count_cum"])
                    if bucket["count_cum"] > 0
                    else None
                ),
            }
        )

    path_summary = (
        pl.DataFrame(path_rows).sort(["transition_code", "rel_bar"])
        if path_rows
        else pl.DataFrame(
            schema={
                "transition_code": pl.Int32,
                "rel_bar": pl.Int32,
                "count_points": pl.Int64,
                "daily_ret_1_mean": pl.Float64,
                "abs_daily_ret_mean": pl.Float64,
                "cumulative_ret_from_event_mean": pl.Float64,
            }
        )
    )

    if transition_events.height == 0:
        transition_summary = pl.DataFrame(
            schema={
                "transition_code": pl.Int32,
                "prev_state_id": pl.Int32,
                "next_state_id": pl.Int32,
                "count_events": pl.Int64,
            }
        )
    else:
        agg_exprs: list[pl.Expr] = [
            pl.len().alias("count_events"),
        ]
        for column in forward_cols:
            agg_exprs.extend(
                [
                    pl.col(column).mean().alias(f"{column}_mean"),
                    pl.col(column).median().alias(f"{column}_median"),
                ]
            )
            if column in {"fwd_ret_5", "fwd_ret_10", "fwd_ret_20"}:
                agg_exprs.append(pl.col(column).gt(0).mean().alias(f"{column}_hit_rate"))
        for column in context_cols:
            agg_exprs.append(pl.col(column).mean().alias(f"{column}_mean"))

        transition_summary = (
            transition_events.group_by(["transition_code", "prev_state_id", "next_state_id"])
            .agg(agg_exprs)
            .sort("count_events", descending=True)
            .filter(pl.col("count_events") >= min_events_per_transition)
        )

    top_codes = {
        "generated_ts": date.today().isoformat(),
        "min_events_per_transition": int(min_events_per_transition),
        "top_transitions": transition_summary.head(20).to_dicts(),
    }

    return TransitionEventStudyResult(
        transition_events=transition_events,
        transition_event_path_summary=path_summary,
        transition_event_summary=transition_summary,
        transition_top_codes=top_codes,
    )
