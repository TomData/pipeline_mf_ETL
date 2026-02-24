"""Gold Features v1: ML-ready feature vectors from Gold event grammar outputs."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Sequence

import polars as pl

from mf_etl.transform.dtypes import GOLD_SCHEMA_VERSION
from mf_etl.utils.time_utils import now_utc

FEATURE_SCHEMA_VERSION: Final[str] = GOLD_SCHEMA_VERSION
FEATURE_CALC_VERSION: Final[str] = "gold_features_v1"


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


def _safe_div(numerator: pl.Expr, denominator: pl.Expr, *, eps: float) -> pl.Expr:
    return pl.when(denominator.abs() > eps).then(numerator / denominator).otherwise(None)


def _ensure_inputs(event_df: pl.DataFrame, *, fallback_run_id: str | None) -> pl.DataFrame:
    required = {
        "ticker",
        "exchange",
        "trade_date",
        "trade_dt",
        "flow_state_code",
        "flow_state_label",
        "tmf_21",
        "tti_proxy_v1_21",
    }
    missing = sorted(required.difference(event_df.columns))
    if missing:
        raise ValueError(f"Event dataframe missing required columns: {', '.join(missing)}")

    out = event_df
    bool_defaults = {
        "ev_tmf_zero_up": False,
        "ev_tmf_zero_down": False,
        "ev_tmf_respect_zero_up": False,
        "ev_tmf_respect_zero_down": False,
        "ev_tmf_respect_fail_up": False,
        "ev_tmf_respect_fail_down": False,
        "ev_tmf_burst_up": False,
        "ev_tmf_burst_down": False,
        "ev_tmf_hold_pos": False,
        "ev_tmf_hold_neg": False,
        "ev_tti_zero_up": False,
        "ev_tti_zero_down": False,
        "ev_tti_burst_up": False,
        "ev_tti_burst_down": False,
        "ev_tti_hold_pos": False,
        "ev_tti_hold_neg": False,
    }
    for column, default_value in bool_defaults.items():
        if column not in out.columns:
            out = out.with_columns(pl.lit(default_value).alias(column))
        else:
            out = out.with_columns(pl.col(column).cast(pl.Boolean, strict=False).fill_null(default_value).alias(column))

    bars_since_defaults = [
        "bs_tmf_zero_up",
        "bs_tmf_zero_down",
        "bs_tmf_respect_zero_up",
        "bs_tmf_respect_zero_down",
        "bs_tmf_burst_up",
        "bs_tmf_burst_down",
        "bs_tti_zero_up",
        "bs_tti_zero_down",
    ]
    for column in bars_since_defaults:
        if column not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=pl.Int32).alias(column))
        else:
            out = out.with_columns(pl.col(column).cast(pl.Int32, strict=False).alias(column))

    if "quality_warn_count" not in out.columns:
        out = out.with_columns(pl.lit(0, dtype=pl.Int64).alias("quality_warn_count"))
    if "tmf_abs" not in out.columns:
        out = out.with_columns(pl.col("tmf_21").abs().alias("tmf_abs"))
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
    if "tmf_slope_5" not in out.columns:
        out = out.with_columns((pl.col("tmf_21") - pl.col("tmf_21").shift(5).over("ticker")).alias("tmf_slope_5"))

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

    optional_float_columns = ["close", "volume", "atr_14", "atr_pct_14", "dollar_volume"]
    for column in optional_float_columns:
        if column not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))

    if "run_id" not in out.columns:
        out = out.with_columns(pl.lit(fallback_run_id, dtype=pl.String).alias("run_id"))
    elif fallback_run_id is not None:
        out = out.with_columns(pl.col("run_id").cast(pl.String, strict=False).fill_null(fallback_run_id).alias("run_id"))
    return out


def build_gold_features_v1(
    event_df: pl.DataFrame,
    *,
    activity_windows: Sequence[int] = (5, 20),
    zero_weight: float = 1.0,
    respect_weight: float = 2.0,
    burst_weight: float = 2.0,
    hold_weight: float = 1.5,
    recency_clip_bars: int = 20,
    eps: float = 1e-12,
    float_dtype: pl.DataType = pl.Float32,
    fallback_run_id: str | None = None,
    built_ts: datetime | None = None,
) -> pl.DataFrame:
    """Build Gold Features v1 from one symbol's event grammar dataframe."""

    if recency_clip_bars < 1:
        raise ValueError("recency_clip_bars must be >= 1.")
    if eps <= 0:
        raise ValueError("eps must be > 0.")

    windows = sorted(set([5, 20, *_normalize_windows(activity_windows)]))
    data = _ensure_inputs(event_df, fallback_run_id=fallback_run_id)
    data = data.sort([column for column in ("ticker", "trade_date") if column in data.columns])
    built_at = built_ts or now_utc()

    out = data.with_columns(
        [
            (pl.col("ticker").cum_count().over("ticker") - 1).cast(pl.Int64).alias("_row_index"),
            (pl.col("tmf_21") - pl.col("tmf_21").shift(10).over("ticker")).alias("tmf_slope_10"),
            (pl.col("tmf_slope_1") - pl.col("tmf_slope_1").shift(1).over("ticker")).alias("tmf_curvature_1"),
            (pl.col("tmf_sign") > 0).fill_null(False).alias("tmf_above_zero"),
            (pl.col("tmf_sign") < 0).fill_null(False).alias("tmf_below_zero"),
            (pl.col("tti_proxy_v1_21") - pl.col("tti_proxy_v1_21").shift(1).over("ticker")).alias("tti_proxy_slope_1"),
            (pl.col("tti_proxy_v1_21") - pl.col("tti_proxy_v1_21").shift(5).over("ticker")).alias("tti_proxy_slope_5"),
            (
                (pl.col("tmf_sign") == pl.col("tti_proxy_sign"))
                & (pl.col("tmf_sign") != 0)
                & (pl.col("tti_proxy_sign") != 0)
            )
            .fill_null(False)
            .alias("tmf_tti_sign_agree"),
            ((pl.col("tmf_sign").cast(pl.Int16) * pl.col("tti_proxy_sign").cast(pl.Int16)) < 0)
            .fill_null(False)
            .alias("tmf_tti_divergence"),
        ]
    )

    long_base = (
        (pl.col("ev_tmf_zero_up").cast(pl.Float64) * zero_weight)
        + (pl.col("ev_tmf_respect_zero_up").cast(pl.Float64) * respect_weight)
        + (pl.col("ev_tmf_burst_up").cast(pl.Float64) * burst_weight)
        + (pl.col("ev_tmf_hold_pos").cast(pl.Float64) * hold_weight)
    ).alias("_long_flow_base")
    short_base = (
        (pl.col("ev_tmf_zero_down").cast(pl.Float64) * zero_weight)
        + (pl.col("ev_tmf_respect_zero_down").cast(pl.Float64) * respect_weight)
        + (pl.col("ev_tmf_burst_down").cast(pl.Float64) * burst_weight)
        + (pl.col("ev_tmf_hold_neg").cast(pl.Float64) * hold_weight)
    ).alias("_short_flow_base")
    out = out.with_columns([long_base, short_base])

    rolling_exprs: list[pl.Expr] = []
    for window in windows:
        rolling_exprs.extend(
            [
                pl.col("_long_flow_base")
                .rolling_sum(window_size=window, min_samples=1)
                .over("ticker")
                .alias(f"long_flow_score_{window}"),
                pl.col("_short_flow_base")
                .rolling_sum(window_size=window, min_samples=1)
                .over("ticker")
                .alias(f"short_flow_score_{window}"),
            ]
        )
    out = out.with_columns(rolling_exprs)

    out = out.with_columns(
        [
            (pl.col("long_flow_score_5") - pl.col("short_flow_score_5")).alias("delta_flow_5"),
            (pl.col("long_flow_score_20") - pl.col("short_flow_score_20")).alias("delta_flow_20"),
            (pl.col("long_flow_score_20") + pl.col("short_flow_score_20")).alias("flow_activity_20"),
        ]
    ).with_columns(
        _safe_div(pl.col("delta_flow_20"), pl.col("flow_activity_20"), eps=eps).alias("flow_bias_20")
    )

    clip = float(recency_clip_bars)
    out = out.with_columns(
        [
            pl.when(pl.col("bs_tmf_zero_up").is_null())
            .then(None)
            .otherwise(pl.min_horizontal([pl.col("bs_tmf_zero_up"), pl.lit(recency_clip_bars)]).cast(pl.Float64) / clip)
            .alias("rec_tmf_zero_up_20"),
            pl.when(pl.col("bs_tmf_zero_down").is_null())
            .then(None)
            .otherwise(pl.min_horizontal([pl.col("bs_tmf_zero_down"), pl.lit(recency_clip_bars)]).cast(pl.Float64) / clip)
            .alias("rec_tmf_zero_down_20"),
            pl.when(pl.col("bs_tmf_burst_up").is_null())
            .then(None)
            .otherwise(pl.min_horizontal([pl.col("bs_tmf_burst_up"), pl.lit(recency_clip_bars)]).cast(pl.Float64) / clip)
            .alias("rec_tmf_burst_up_20"),
            pl.when(pl.col("bs_tmf_burst_down").is_null())
            .then(None)
            .otherwise(pl.min_horizontal([pl.col("bs_tmf_burst_down"), pl.lit(recency_clip_bars)]).cast(pl.Float64) / clip)
            .alias("rec_tmf_burst_down_20"),
            pl.when(pl.col("bs_tmf_respect_zero_up").is_null())
            .then(None)
            .otherwise(pl.min_horizontal([pl.col("bs_tmf_respect_zero_up"), pl.lit(recency_clip_bars)]).cast(pl.Float64) / clip)
            .alias("rec_tmf_respect_up_20"),
            pl.when(pl.col("bs_tmf_respect_zero_down").is_null())
            .then(None)
            .otherwise(pl.min_horizontal([pl.col("bs_tmf_respect_zero_down"), pl.lit(recency_clip_bars)]).cast(pl.Float64) / clip)
            .alias("rec_tmf_respect_down_20"),
        ]
    )

    out = out.with_columns(
        [
            pl.col("ev_tmf_burst_up").cast(pl.Int16).rolling_sum(window_size=5, min_samples=1).over("ticker").cast(pl.Int32).alias("long_burst_5"),
            pl.col("ev_tmf_burst_down").cast(pl.Int16).rolling_sum(window_size=5, min_samples=1).over("ticker").cast(pl.Int32).alias("short_burst_5"),
            pl.col("ev_tmf_burst_up").cast(pl.Int16).rolling_sum(window_size=20, min_samples=1).over("ticker").cast(pl.Int32).alias("long_burst_20"),
            pl.col("ev_tmf_burst_down").cast(pl.Int16).rolling_sum(window_size=20, min_samples=1).over("ticker").cast(pl.Int32).alias("short_burst_20"),
            pl.col("ev_tmf_hold_pos").cast(pl.Float64).rolling_mean(window_size=20, min_samples=1).over("ticker").alias("persistence_pos_20"),
            pl.col("ev_tmf_hold_neg").cast(pl.Float64).rolling_mean(window_size=20, min_samples=1).over("ticker").alias("persistence_neg_20"),
            (
                (pl.col("ev_tmf_zero_up").cast(pl.Int16) + pl.col("ev_tmf_zero_down").cast(pl.Int16))
                .rolling_sum(window_size=20, min_samples=1)
                .over("ticker")
                .cast(pl.Int32)
            ).alias("oscillation_index_20"),
            (
                pl.col("ev_tmf_respect_fail_down")
                .cast(pl.Int16)
                .rolling_sum(window_size=20, min_samples=1)
                .over("ticker")
                - pl.col("ev_tmf_respect_fail_up")
                .cast(pl.Int16)
                .rolling_sum(window_size=20, min_samples=1)
                .over("ticker")
            )
            .cast(pl.Int32)
            .alias("respect_fail_balance_20"),
        ]
    )

    state_prev = pl.col("flow_state_code").shift(1).over("ticker")
    state_changed = (pl.col("flow_state_code") != state_prev).fill_null(False)
    state_group = (
        pl.when(state_prev.is_null() | (pl.col("flow_state_code") != state_prev))
        .then(1)
        .otherwise(0)
        .cum_sum()
        .over("ticker")
        .alias("_state_group")
    )
    out = out.with_columns(state_group)
    out = out.with_columns(
        [
            state_prev.cast(pl.Int8).alias("state_prev"),
            state_changed.alias("state_changed"),
            pl.col("flow_state_code").cum_count().over(["ticker", "_state_group"]).cast(pl.Int32).alias("state_run_length"),
            pl.when(state_prev.is_null())
            .then(None)
            .otherwise(state_prev.cast(pl.Int16) * 10 + pl.col("flow_state_code").cast(pl.Int16))
            .cast(pl.Int16)
            .alias("state_transition_code"),
        ]
    )
    out = out.with_columns(_bars_since_expr("state_changed").alias("bs_state_change"))

    out = out.with_columns(
        [
            pl.col("tmf_21").cast(float_dtype, strict=False).alias("tmf_21"),
            pl.col("tmf_abs").cast(float_dtype, strict=False).alias("tmf_abs"),
            pl.col("tmf_slope_1").cast(float_dtype, strict=False).alias("tmf_slope_1"),
            pl.col("tmf_slope_5").cast(float_dtype, strict=False).alias("tmf_slope_5"),
            pl.col("tmf_slope_10").cast(float_dtype, strict=False).alias("tmf_slope_10"),
            pl.col("tmf_curvature_1").cast(float_dtype, strict=False).alias("tmf_curvature_1"),
            pl.col("tti_proxy_v1_21").cast(float_dtype, strict=False).alias("tti_proxy_v1_21"),
            pl.col("tti_proxy_slope_1").cast(float_dtype, strict=False).alias("tti_proxy_slope_1"),
            pl.col("tti_proxy_slope_5").cast(float_dtype, strict=False).alias("tti_proxy_slope_5"),
            pl.col("long_flow_score_5").cast(float_dtype, strict=False).alias("long_flow_score_5"),
            pl.col("short_flow_score_5").cast(float_dtype, strict=False).alias("short_flow_score_5"),
            pl.col("long_flow_score_20").cast(float_dtype, strict=False).alias("long_flow_score_20"),
            pl.col("short_flow_score_20").cast(float_dtype, strict=False).alias("short_flow_score_20"),
            pl.col("delta_flow_5").cast(float_dtype, strict=False).alias("delta_flow_5"),
            pl.col("delta_flow_20").cast(float_dtype, strict=False).alias("delta_flow_20"),
            pl.col("flow_activity_20").cast(float_dtype, strict=False).alias("flow_activity_20"),
            pl.col("flow_bias_20").cast(float_dtype, strict=False).alias("flow_bias_20"),
            pl.col("rec_tmf_zero_up_20").cast(float_dtype, strict=False).alias("rec_tmf_zero_up_20"),
            pl.col("rec_tmf_zero_down_20").cast(float_dtype, strict=False).alias("rec_tmf_zero_down_20"),
            pl.col("rec_tmf_burst_up_20").cast(float_dtype, strict=False).alias("rec_tmf_burst_up_20"),
            pl.col("rec_tmf_burst_down_20").cast(float_dtype, strict=False).alias("rec_tmf_burst_down_20"),
            pl.col("rec_tmf_respect_up_20").cast(float_dtype, strict=False).alias("rec_tmf_respect_up_20"),
            pl.col("rec_tmf_respect_down_20").cast(float_dtype, strict=False).alias("rec_tmf_respect_down_20"),
            pl.col("persistence_pos_20").cast(float_dtype, strict=False).alias("persistence_pos_20"),
            pl.col("persistence_neg_20").cast(float_dtype, strict=False).alias("persistence_neg_20"),
            pl.col("oscillation_index_20").cast(float_dtype, strict=False).alias("oscillation_index_20"),
            pl.col("respect_fail_balance_20").cast(float_dtype, strict=False).alias("respect_fail_balance_20"),
            pl.col("close").cast(float_dtype, strict=False).alias("close"),
            pl.col("volume").cast(float_dtype, strict=False).alias("volume"),
            pl.col("atr_14").cast(float_dtype, strict=False).alias("atr_14"),
            pl.col("atr_pct_14").cast(float_dtype, strict=False).alias("atr_pct_14"),
            pl.col("dollar_volume").cast(float_dtype, strict=False).alias("dollar_volume"),
            pl.lit(FEATURE_SCHEMA_VERSION).alias("feature_schema_version"),
            pl.lit(FEATURE_CALC_VERSION).alias("feature_calc_version"),
            pl.lit(built_at).cast(pl.Datetime(time_unit="us", time_zone="UTC")).alias("built_ts"),
        ]
    )

    ordered_columns = [
        "ticker",
        "exchange",
        "trade_date",
        "trade_dt",
        "flow_state_code",
        "flow_state_label",
        "quality_warn_count",
        "tmf_21",
        "tmf_abs",
        "tmf_slope_1",
        "tmf_slope_5",
        "tmf_slope_10",
        "tmf_curvature_1",
        "tmf_sign",
        "tmf_above_zero",
        "tmf_below_zero",
        "tti_proxy_v1_21",
        "tti_proxy_sign",
        "tti_proxy_slope_1",
        "tti_proxy_slope_5",
        "tmf_tti_sign_agree",
        "tmf_tti_divergence",
        "long_flow_score_5",
        "short_flow_score_5",
        "long_flow_score_20",
        "short_flow_score_20",
        "delta_flow_5",
        "delta_flow_20",
        "flow_activity_20",
        "flow_bias_20",
        "bs_tmf_zero_up",
        "bs_tmf_zero_down",
        "bs_tmf_respect_zero_up",
        "bs_tmf_respect_zero_down",
        "bs_tmf_burst_up",
        "bs_tmf_burst_down",
        "bs_tti_zero_up",
        "bs_tti_zero_down",
        "rec_tmf_zero_up_20",
        "rec_tmf_zero_down_20",
        "rec_tmf_burst_up_20",
        "rec_tmf_burst_down_20",
        "rec_tmf_respect_up_20",
        "rec_tmf_respect_down_20",
        "long_burst_5",
        "short_burst_5",
        "long_burst_20",
        "short_burst_20",
        "persistence_pos_20",
        "persistence_neg_20",
        "oscillation_index_20",
        "respect_fail_balance_20",
        "state_prev",
        "state_changed",
        "state_run_length",
        "state_transition_code",
        "bs_state_change",
        "close",
        "volume",
        "atr_14",
        "atr_pct_14",
        "dollar_volume",
        "feature_schema_version",
        "feature_calc_version",
        "run_id",
        "built_ts",
    ]
    existing = [column for column in ordered_columns if column in out.columns]
    remaining = [column for column in out.columns if column not in existing and not column.startswith("_")]
    return out.select(existing + remaining)

