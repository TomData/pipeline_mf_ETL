"""Gold Event Grammar v1 derived from TMF and TTI-proxy indicators."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Sequence

import polars as pl

from mf_etl.transform.dtypes import GOLD_SCHEMA_VERSION
from mf_etl.utils.time_utils import now_utc

EVENT_SCHEMA_VERSION: Final[str] = GOLD_SCHEMA_VERSION
EVENT_CALC_VERSION: Final[str] = "event_grammar_v1"

FLOW_STATE_CODE_TO_LABEL: dict[int, str] = {
    0: "S0_QUIET",
    1: "S1_EARLY_DEMAND",
    2: "S2_PERSISTENT_DEMAND",
    3: "S3_EARLY_SUPPLY",
    4: "S4_PERSISTENT_SUPPLY",
}


def _normalize_windows(windows: Sequence[int]) -> list[int]:
    cleaned = sorted({int(window) for window in windows if int(window) > 0})
    if not cleaned:
        raise ValueError("activity_windows must contain at least one positive integer.")
    return cleaned


def _bars_since_expr(event_col: str, *, row_index_col: str = "_row_index") -> pl.Expr:
    """Compute bars-since counter per ticker for a boolean event column."""

    last_event_index = (
        pl.when(pl.col(event_col))
        .then(pl.col(row_index_col))
        .otherwise(None)
        .forward_fill()
        .over("ticker")
    )
    return (
        pl.when(last_event_index.is_null())
        .then(None)
        .otherwise((pl.col(row_index_col) - last_event_index).cast(pl.Int32))
    )


def _ensure_required_inputs(indicator_df: pl.DataFrame, *, fallback_run_id: str | None) -> pl.DataFrame:
    """Ensure event grammar has required columns and derive safe defaults."""

    required = {
        "ticker",
        "exchange",
        "trade_date",
        "trade_dt",
        "close",
        "volume",
        "tmf_21",
        "tmf_ready_21",
        "tmf_zero_cross_up",
        "tmf_zero_cross_down",
        "tti_proxy_v1_21",
        "tti_proxy_ready_21",
        "tti_proxy_zero_cross_up",
        "tti_proxy_zero_cross_down",
    }
    missing = sorted(required.difference(indicator_df.columns))
    if missing:
        raise ValueError(f"Indicator frame missing required columns: {', '.join(missing)}")

    out = indicator_df
    if "tmf_sign" not in out.columns:
        out = out.with_columns(
            pl.when(pl.col("tmf_21") > 0)
            .then(1)
            .when(pl.col("tmf_21") < 0)
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("tmf_sign")
        )
    if "tmf_slope_1" not in out.columns:
        out = out.with_columns((pl.col("tmf_21") - pl.col("tmf_21").shift(1).over("ticker")).alias("tmf_slope_1"))
    if "tmf_abs" not in out.columns:
        out = out.with_columns(pl.col("tmf_21").abs().alias("tmf_abs"))
    if "tti_proxy_sign" not in out.columns:
        out = out.with_columns(
            pl.when(pl.col("tti_proxy_v1_21") > 0)
            .then(1)
            .when(pl.col("tti_proxy_v1_21") < 0)
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("tti_proxy_sign")
        )

    if "tr" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("tr"))
    if "atr_14" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("atr_14"))
    if "quality_warn_count" not in out.columns:
        out = out.with_columns(pl.lit(0, dtype=pl.Int64).alias("quality_warn_count"))

    if "run_id" not in out.columns:
        out = out.with_columns(pl.lit(fallback_run_id, dtype=pl.String).alias("run_id"))
    elif fallback_run_id is not None:
        out = out.with_columns(pl.col("run_id").cast(pl.String, strict=False).fill_null(fallback_run_id).alias("run_id"))
    return out


def build_event_grammar_v1(
    indicator_df: pl.DataFrame,
    *,
    pivot_mode: str = "3bar",
    respect_fail_lookahead_bars: int = 10,
    hold_consecutive_bars: int = 5,
    tmf_burst_abs_threshold: float = 0.15,
    tmf_burst_slope_threshold: float = 0.05,
    activity_windows: Sequence[int] = (5, 20),
    eps: float = 1e-12,
    float_dtype: pl.DataType = pl.Float32,
    fallback_run_id: str | None = None,
    built_ts: datetime | None = None,
) -> pl.DataFrame:
    """Build deterministic Event Grammar v1 flags and state coding.

    Pivot events are emitted on the confirmation row `t` using a 3-bar local
    structure where the pivot itself is at `t-1`.
    """

    if pivot_mode != "3bar":
        raise ValueError(f"Unsupported pivot_mode: {pivot_mode}. Only '3bar' is supported in v1.")
    if respect_fail_lookahead_bars < 1:
        raise ValueError("respect_fail_lookahead_bars must be >= 1.")
    if hold_consecutive_bars < 1:
        raise ValueError("hold_consecutive_bars must be >= 1.")
    if tmf_burst_abs_threshold < 0 or tmf_burst_slope_threshold < 0:
        raise ValueError("burst thresholds must be >= 0.")
    if eps <= 0:
        raise ValueError("eps must be > 0.")

    windows = _normalize_windows(activity_windows)
    data = _ensure_required_inputs(indicator_df, fallback_run_id=fallback_run_id)
    data = data.sort([column for column in ("ticker", "trade_date") if column in data.columns])
    built_at = built_ts or now_utc()

    tmf_prev2 = pl.col("tmf_21").shift(2).over("ticker")
    tmf_prev1 = pl.col("tmf_21").shift(1).over("ticker")

    out = data.with_columns(
        [
            (pl.col("ticker").cum_count().over("ticker") - 1).cast(pl.Int64).alias("_row_index"),
            pl.col("tmf_zero_cross_up").cast(pl.Boolean, strict=False).fill_null(False).alias("ev_tmf_zero_up"),
            pl.col("tmf_zero_cross_down").cast(pl.Boolean, strict=False).fill_null(False).alias("ev_tmf_zero_down"),
            pl.col("tti_proxy_zero_cross_up").cast(pl.Boolean, strict=False).fill_null(False).alias("ev_tti_zero_up"),
            pl.col("tti_proxy_zero_cross_down").cast(pl.Boolean, strict=False).fill_null(False).alias("ev_tti_zero_down"),
        ]
    )

    out = out.with_columns(
        [
            ((tmf_prev2 > tmf_prev1) & (tmf_prev1 < pl.col("tmf_21"))).fill_null(False).alias("ev_tmf_pivot_low"),
            ((tmf_prev2 < tmf_prev1) & (tmf_prev1 > pl.col("tmf_21"))).fill_null(False).alias("ev_tmf_pivot_high"),
        ]
    )

    out = out.with_columns(
        [
            (
                pl.col("ev_tmf_pivot_low")
                & (tmf_prev1 > 0)
                & (pl.col("tmf_21") > 0)
                & (pl.col("tmf_slope_1") > 0)
            )
            .fill_null(False)
            .alias("ev_tmf_respect_zero_up"),
            (
                pl.col("ev_tmf_pivot_high")
                & (tmf_prev1 < 0)
                & (pl.col("tmf_21") < 0)
                & (pl.col("tmf_slope_1") < 0)
            )
            .fill_null(False)
            .alias("ev_tmf_respect_zero_down"),
        ]
    )

    prior_respect_up = (
        pl.col("ev_tmf_respect_zero_up")
        .cast(pl.Int8)
        .shift(1)
        .rolling_max(window_size=respect_fail_lookahead_bars, min_samples=1)
        .over("ticker")
        .fill_null(0)
    )
    prior_respect_down = (
        pl.col("ev_tmf_respect_zero_down")
        .cast(pl.Int8)
        .shift(1)
        .rolling_max(window_size=respect_fail_lookahead_bars, min_samples=1)
        .over("ticker")
        .fill_null(0)
    )
    out = out.with_columns(
        [
            (pl.col("ev_tmf_zero_down") & (prior_respect_up > 0)).fill_null(False).alias("ev_tmf_respect_fail_up"),
            (pl.col("ev_tmf_zero_up") & (prior_respect_down > 0)).fill_null(False).alias("ev_tmf_respect_fail_down"),
        ]
    )

    abs_threshold = max(tmf_burst_abs_threshold, eps)
    out = out.with_columns(
        [
            (
                pl.col("tmf_ready_21")
                & (pl.col("tmf_21") > 0)
                & (pl.col("tmf_slope_1") > tmf_burst_slope_threshold)
                & (pl.col("tmf_abs") > abs_threshold)
            )
            .fill_null(False)
            .alias("ev_tmf_burst_up"),
            (
                pl.col("tmf_ready_21")
                & (pl.col("tmf_21") < 0)
                & (pl.col("tmf_slope_1") < -tmf_burst_slope_threshold)
                & (pl.col("tmf_abs") > abs_threshold)
            )
            .fill_null(False)
            .alias("ev_tmf_burst_down"),
        ]
    )

    tmf_pos_ready = (pl.col("tmf_ready_21") & (pl.col("tmf_21") > 0)).cast(pl.Int16)
    tmf_neg_ready = (pl.col("tmf_ready_21") & (pl.col("tmf_21") < 0)).cast(pl.Int16)
    out = out.with_columns(
        [
            (tmf_pos_ready.rolling_sum(window_size=hold_consecutive_bars, min_samples=hold_consecutive_bars).over("ticker") == hold_consecutive_bars)
            .fill_null(False)
            .alias("ev_tmf_hold_pos"),
            (tmf_neg_ready.rolling_sum(window_size=hold_consecutive_bars, min_samples=hold_consecutive_bars).over("ticker") == hold_consecutive_bars)
            .fill_null(False)
            .alias("ev_tmf_hold_neg"),
        ]
    )

    tti_slope_1 = (pl.col("tti_proxy_v1_21") - pl.col("tti_proxy_v1_21").shift(1).over("ticker"))
    tti_abs = pl.col("tti_proxy_v1_21").abs()
    out = out.with_columns(
        [
            (
                pl.col("tti_proxy_ready_21")
                & (pl.col("tti_proxy_v1_21") > 0)
                & (tti_slope_1 > tmf_burst_slope_threshold)
                & (tti_abs > abs_threshold)
            )
            .fill_null(False)
            .alias("ev_tti_burst_up"),
            (
                pl.col("tti_proxy_ready_21")
                & (pl.col("tti_proxy_v1_21") < 0)
                & (tti_slope_1 < -tmf_burst_slope_threshold)
                & (tti_abs > abs_threshold)
            )
            .fill_null(False)
            .alias("ev_tti_burst_down"),
        ]
    )

    tti_pos_ready = (pl.col("tti_proxy_ready_21") & (pl.col("tti_proxy_v1_21") > 0)).cast(pl.Int16)
    tti_neg_ready = (pl.col("tti_proxy_ready_21") & (pl.col("tti_proxy_v1_21") < 0)).cast(pl.Int16)
    out = out.with_columns(
        [
            (tti_pos_ready.rolling_sum(window_size=hold_consecutive_bars, min_samples=hold_consecutive_bars).over("ticker") == hold_consecutive_bars)
            .fill_null(False)
            .alias("ev_tti_hold_pos"),
            (tti_neg_ready.rolling_sum(window_size=hold_consecutive_bars, min_samples=hold_consecutive_bars).over("ticker") == hold_consecutive_bars)
            .fill_null(False)
            .alias("ev_tti_hold_neg"),
        ]
    )

    bars_since_columns = {
        "bs_tmf_zero_up": "ev_tmf_zero_up",
        "bs_tmf_zero_down": "ev_tmf_zero_down",
        "bs_tmf_respect_zero_up": "ev_tmf_respect_zero_up",
        "bs_tmf_respect_zero_down": "ev_tmf_respect_zero_down",
        "bs_tmf_burst_up": "ev_tmf_burst_up",
        "bs_tmf_burst_down": "ev_tmf_burst_down",
        "bs_tti_zero_up": "ev_tti_zero_up",
        "bs_tti_zero_down": "ev_tti_zero_down",
    }
    out = out.with_columns(
        [_bars_since_expr(event_col).alias(target_col) for target_col, event_col in bars_since_columns.items()]
    )

    out = out.with_columns(
        [
            pl.sum_horizontal(
                [
                    pl.col("ev_tmf_zero_up").cast(pl.Int16),
                    pl.col("ev_tmf_respect_zero_up").cast(pl.Int16),
                    pl.col("ev_tmf_burst_up").cast(pl.Int16),
                ]
            ).alias("_tmf_long_events_base"),
            pl.sum_horizontal(
                [
                    pl.col("ev_tmf_zero_down").cast(pl.Int16),
                    pl.col("ev_tmf_respect_zero_down").cast(pl.Int16),
                    pl.col("ev_tmf_burst_down").cast(pl.Int16),
                ]
            ).alias("_tmf_short_events_base"),
            pl.sum_horizontal(
                [
                    pl.col("ev_tti_zero_up").cast(pl.Int16),
                    pl.col("ev_tti_zero_down").cast(pl.Int16),
                    pl.col("ev_tti_burst_up").cast(pl.Int16),
                    pl.col("ev_tti_burst_down").cast(pl.Int16),
                ]
            ).alias("_tti_events_base"),
        ]
    )

    activity_exprs: list[pl.Expr] = []
    for window in windows:
        activity_exprs.extend(
            [
                pl.col("_tmf_long_events_base")
                .rolling_sum(window_size=window, min_samples=1)
                .over("ticker")
                .cast(pl.Int32)
                .alias(f"tmf_long_events_{window}"),
                pl.col("_tmf_short_events_base")
                .rolling_sum(window_size=window, min_samples=1)
                .over("ticker")
                .cast(pl.Int32)
                .alias(f"tmf_short_events_{window}"),
            ]
        )
        if window == 20:
            activity_exprs.append(
                pl.col("_tti_events_base")
                .rolling_sum(window_size=20, min_samples=1)
                .over("ticker")
                .cast(pl.Int32)
                .alias("tti_events_20")
            )
    out = out.with_columns(activity_exprs)

    if "tmf_long_events_5" in out.columns and "tmf_short_events_5" in out.columns:
        out = out.with_columns((pl.col("tmf_long_events_5") - pl.col("tmf_short_events_5")).cast(pl.Int32).alias("tmf_event_asym_5"))
    if "tmf_long_events_20" in out.columns and "tmf_short_events_20" in out.columns:
        out = out.with_columns(
            [
                (pl.col("tmf_long_events_20") - pl.col("tmf_short_events_20")).cast(pl.Int32).alias("tmf_event_asym_20"),
                (pl.col("tmf_long_events_20") + pl.col("tmf_short_events_20")).cast(pl.Int32).alias("tmf_event_activity_20"),
            ]
        )

    recent_up = (pl.col("bs_tmf_zero_up").is_not_null() & (pl.col("bs_tmf_zero_up") <= 3)).fill_null(False)
    recent_down = (pl.col("bs_tmf_zero_down").is_not_null() & (pl.col("bs_tmf_zero_down") <= 3)).fill_null(False)
    long_adv_20 = (pl.col("tmf_long_events_20") > (pl.col("tmf_short_events_20") + 1)).fill_null(False)
    short_adv_20 = (pl.col("tmf_short_events_20") > (pl.col("tmf_long_events_20") + 1)).fill_null(False)

    persistent_demand = (pl.col("tmf_ready_21") & (pl.col("tmf_sign") > 0) & (pl.col("ev_tmf_hold_pos") | long_adv_20)).fill_null(False)
    persistent_supply = (pl.col("tmf_ready_21") & (pl.col("tmf_sign") < 0) & (pl.col("ev_tmf_hold_neg") | short_adv_20)).fill_null(False)
    early_demand = (
        pl.col("tmf_ready_21")
        & (pl.col("tmf_sign") > 0)
        & (~persistent_demand)
        & (pl.col("ev_tmf_zero_up") | pl.col("ev_tmf_burst_up") | pl.col("ev_tmf_respect_zero_up") | recent_up)
    ).fill_null(False)
    early_supply = (
        pl.col("tmf_ready_21")
        & (pl.col("tmf_sign") < 0)
        & (~persistent_supply)
        & (pl.col("ev_tmf_zero_down") | pl.col("ev_tmf_burst_down") | pl.col("ev_tmf_respect_zero_down") | recent_down)
    ).fill_null(False)

    out = out.with_columns(
        pl.when(persistent_demand)
        .then(2)
        .when(persistent_supply)
        .then(4)
        .when(early_demand)
        .then(1)
        .when(early_supply)
        .then(3)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("flow_state_code")
    )
    out = out.with_columns(
        pl.when(pl.col("flow_state_code") == 0)
        .then(pl.lit("S0_QUIET"))
        .when(pl.col("flow_state_code") == 1)
        .then(pl.lit("S1_EARLY_DEMAND"))
        .when(pl.col("flow_state_code") == 2)
        .then(pl.lit("S2_PERSISTENT_DEMAND"))
        .when(pl.col("flow_state_code") == 3)
        .then(pl.lit("S3_EARLY_SUPPLY"))
        .otherwise(pl.lit("S4_PERSISTENT_SUPPLY"))
        .alias("flow_state_label")
    )

    out = out.with_columns(
        [
            pl.col("close").cast(float_dtype, strict=False).alias("close"),
            pl.col("volume").cast(float_dtype, strict=False).alias("volume"),
            pl.col("tmf_21").cast(float_dtype, strict=False).alias("tmf_21"),
            pl.col("tti_proxy_v1_21").cast(float_dtype, strict=False).alias("tti_proxy_v1_21"),
            pl.col("tr").cast(float_dtype, strict=False).alias("tr"),
            pl.col("atr_14").cast(float_dtype, strict=False).alias("atr_14"),
            pl.lit(EVENT_SCHEMA_VERSION).alias("event_schema_version"),
            pl.lit(EVENT_CALC_VERSION).alias("event_calc_version"),
            pl.lit(built_at).cast(pl.Datetime(time_unit="us", time_zone="UTC")).alias("built_ts"),
        ]
    )

    ordered_columns = [
        "ticker",
        "exchange",
        "trade_date",
        "trade_dt",
        "close",
        "volume",
        "tmf_21",
        "tti_proxy_v1_21",
        "tr",
        "atr_14",
        "quality_warn_count",
        "ev_tmf_zero_up",
        "ev_tmf_zero_down",
        "ev_tmf_pivot_low",
        "ev_tmf_pivot_high",
        "ev_tmf_respect_zero_up",
        "ev_tmf_respect_zero_down",
        "ev_tmf_respect_fail_up",
        "ev_tmf_respect_fail_down",
        "ev_tmf_burst_up",
        "ev_tmf_burst_down",
        "ev_tmf_hold_pos",
        "ev_tmf_hold_neg",
        "ev_tti_zero_up",
        "ev_tti_zero_down",
        "ev_tti_burst_up",
        "ev_tti_burst_down",
        "ev_tti_hold_pos",
        "ev_tti_hold_neg",
        "bs_tmf_zero_up",
        "bs_tmf_zero_down",
        "bs_tmf_respect_zero_up",
        "bs_tmf_respect_zero_down",
        "bs_tmf_burst_up",
        "bs_tmf_burst_down",
        "bs_tti_zero_up",
        "bs_tti_zero_down",
        "tmf_long_events_5",
        "tmf_short_events_5",
        "tmf_event_asym_5",
        "tmf_long_events_20",
        "tmf_short_events_20",
        "tmf_event_asym_20",
        "tmf_event_activity_20",
        "tti_events_20",
        "flow_state_code",
        "flow_state_label",
        "event_schema_version",
        "event_calc_version",
        "run_id",
        "built_ts",
    ]
    existing = [column for column in ordered_columns if column in out.columns]
    remaining = [column for column in out.columns if column not in existing and not column.startswith("_")]
    return out.select(existing + remaining)
