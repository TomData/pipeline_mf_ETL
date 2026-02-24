"""Twiggs-style indicator calculations on Silver base series."""

from __future__ import annotations

from datetime import datetime
from typing import Final

import polars as pl

from mf_etl.transform.dtypes import SILVER_SCHEMA_VERSION
from mf_etl.utils.time_utils import now_utc

INDICATOR_SCHEMA_VERSION: Final[str] = SILVER_SCHEMA_VERSION
INDICATOR_CALC_VERSION: Final[str] = "twiggs_indicators_v1"
TTI_FORMULA_STATUS: Final[str] = "PROXY_UNDISCLOSED_ORIGINAL"
TTI_PROXY_VERSION: Final[str] = "v1"
DEFAULT_TMF_PERIOD: Final[int] = 21
DEFAULT_PROXY_PERIOD: Final[int] = 21
DEFAULT_EPS: Final[float] = 1e-12

FLOAT_COLUMNS: tuple[str, ...] = (
    "close",
    "volume",
    "tr",
    "atr_14",
    "tmf_21",
    "tmf_slope_1",
    "tmf_slope_5",
    "tmf_ema_5",
    "tmf_abs",
    "tti_proxy_v1_21",
)


def _safe_div(numerator: pl.Expr, denominator: pl.Expr, *, eps: float) -> pl.Expr:
    """Null-safe division with epsilon guard."""

    return pl.when(denominator.abs() > eps).then(numerator / denominator).otherwise(None)


def _wilder_smooth(expr: pl.Expr, period: int, *, by: str = "ticker") -> pl.Expr:
    """Return Wilder-style smoothed average using EMA(alpha=1/period)."""

    return expr.ewm_mean(alpha=1.0 / float(period), adjust=False, min_samples=period).over(by)


def _ensure_indicator_inputs(
    silver_df: pl.DataFrame,
    *,
    fallback_run_id: str | None,
) -> pl.DataFrame:
    """Ensure required Silver columns are present before indicator computation."""

    required = {"ticker", "exchange", "trade_date", "trade_dt", "high", "low", "close", "volume"}
    missing = sorted(required.difference(silver_df.columns))
    if missing:
        raise ValueError(f"Silver dataframe missing required columns: {', '.join(missing)}")

    out = silver_df
    if "prev_close" not in out.columns:
        out = out.with_columns(pl.col("close").shift(1).over("ticker").alias("prev_close"))

    if "tr" not in out.columns:
        prev_close = pl.col("prev_close")
        tr_hl = pl.col("high") - pl.col("low")
        tr_expr = pl.coalesce(
            [
                pl.max_horizontal(
                    [
                        tr_hl,
                        (pl.col("high") - prev_close).abs(),
                        (pl.col("low") - prev_close).abs(),
                    ]
                ),
                tr_hl,
            ]
        )
        out = out.with_columns(tr_expr.alias("tr"))

    if "atr_14" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("atr_14"))

    if "quality_warn_count" not in out.columns:
        out = out.with_columns(pl.lit(0, dtype=pl.Int64).alias("quality_warn_count"))

    if "run_id" not in out.columns:
        out = out.with_columns(pl.lit(fallback_run_id, dtype=pl.String).alias("run_id"))
    elif fallback_run_id is not None:
        out = out.with_columns(pl.col("run_id").cast(pl.String, strict=False).fill_null(fallback_run_id).alias("run_id"))
    return out


def build_twiggs_indicator_frame(
    silver_df: pl.DataFrame,
    *,
    tmf_period: int = DEFAULT_TMF_PERIOD,
    proxy_period: int = DEFAULT_PROXY_PERIOD,
    eps: float = DEFAULT_EPS,
    indicator_float_dtype: pl.DataType = pl.Float32,
    fallback_run_id: str | None = None,
    built_ts: datetime | None = None,
) -> pl.DataFrame:
    """Build TMF v1 and TTI proxy indicator columns for one-symbol Silver data."""

    if tmf_period < 1:
        raise ValueError("tmf_period must be >= 1.")
    if proxy_period < 1:
        raise ValueError("proxy_period must be >= 1.")
    if eps <= 0:
        raise ValueError("eps must be > 0.")

    normalized = _ensure_indicator_inputs(silver_df, fallback_run_id=fallback_run_id)
    sorted_df = normalized.sort([col for col in ("ticker", "trade_date") if col in normalized.columns])
    built_at = built_ts or now_utc()

    trh = (
        pl.when(pl.col("prev_close").is_null())
        .then(pl.col("high"))
        .otherwise(pl.max_horizontal([pl.col("high"), pl.col("prev_close")]))
    ).alias("trh")
    trl = (
        pl.when(pl.col("prev_close").is_null())
        .then(pl.col("low"))
        .otherwise(pl.min_horizontal([pl.col("low"), pl.col("prev_close")]))
    ).alias("trl")

    out = sorted_df.with_columns([trh, trl])
    tr_range = pl.max_horizontal([pl.col("trh") - pl.col("trl"), pl.lit(eps)])
    signed_tr_pos = ((2.0 * pl.col("close")) - pl.col("trh") - pl.col("trl")) / tr_range

    out = out.with_columns(
        [
            (signed_tr_pos * pl.col("volume")).alias("ad_raw"),
            (signed_tr_pos * pl.col("tr").abs()).alias("proxy_raw"),
            pl.col("tr").abs().alias("proxy_den"),
        ]
    )

    out = out.with_columns(
        [
            _wilder_smooth(pl.col("ad_raw"), tmf_period).alias("ad_wilder"),
            _wilder_smooth(pl.col("volume"), tmf_period).alias("volume_wilder"),
            _wilder_smooth(pl.col("proxy_raw"), proxy_period).alias("proxy_wilder"),
            _wilder_smooth(pl.col("proxy_den"), proxy_period).alias("proxy_den_wilder"),
        ]
    )

    out = out.with_columns(
        [
            _safe_div(pl.col("ad_wilder"), pl.col("volume_wilder"), eps=eps).alias("tmf_21"),
            _safe_div(pl.col("proxy_wilder"), pl.col("proxy_den_wilder"), eps=eps).alias("tti_proxy_v1_21"),
        ]
    )

    tmf_prev = pl.col("tmf_21").shift(1).over("ticker")
    tmf_prev5 = pl.col("tmf_21").shift(5).over("ticker")
    tti_prev = pl.col("tti_proxy_v1_21").shift(1).over("ticker")

    out = out.with_columns(
        [
            pl.col("tmf_21").is_not_null().alias("tmf_ready_21"),
            ((pl.col("tmf_21") > 0) & (tmf_prev <= 0)).fill_null(False).alias("tmf_zero_cross_up"),
            ((pl.col("tmf_21") < 0) & (tmf_prev >= 0)).fill_null(False).alias("tmf_zero_cross_down"),
            pl.when(pl.col("tmf_21") > 0)
            .then(1)
            .when(pl.col("tmf_21") < 0)
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("tmf_sign"),
            pl.col("tmf_21").abs().alias("tmf_abs"),
            (pl.col("tmf_21") - tmf_prev).alias("tmf_slope_1"),
            (pl.col("tmf_21") - tmf_prev5).alias("tmf_slope_5"),
            pl.col("tmf_21").ewm_mean(span=5, adjust=False, min_samples=5).over("ticker").alias("tmf_ema_5"),
            (pl.col("tmf_21") > 0).fill_null(False).alias("tmf_above_zero"),
            (pl.col("tmf_21") < 0).fill_null(False).alias("tmf_below_zero"),
            pl.col("tti_proxy_v1_21").is_not_null().alias("tti_proxy_ready_21"),
            ((pl.col("tti_proxy_v1_21") > 0) & (tti_prev <= 0)).fill_null(False).alias("tti_proxy_zero_cross_up"),
            ((pl.col("tti_proxy_v1_21") < 0) & (tti_prev >= 0)).fill_null(False).alias("tti_proxy_zero_cross_down"),
            pl.when(pl.col("tti_proxy_v1_21") > 0)
            .then(1)
            .when(pl.col("tti_proxy_v1_21") < 0)
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("tti_proxy_sign"),
            pl.lit(INDICATOR_SCHEMA_VERSION).alias("indicator_schema_version"),
            pl.lit(INDICATOR_CALC_VERSION).alias("indicator_calc_version"),
            pl.lit(tmf_period, dtype=pl.Int32).alias("tmf_period"),
            pl.lit(TTI_FORMULA_STATUS).alias("tti_formula_status"),
            pl.lit(TTI_PROXY_VERSION).alias("tti_proxy_version"),
            pl.lit(built_at).cast(pl.Datetime(time_unit="us", time_zone="UTC")).alias("built_ts"),
        ]
    )

    tmf_prev1 = pl.col("tmf_21").shift(1).over("ticker")
    tmf_prev2 = pl.col("tmf_21").shift(2).over("ticker")
    out = out.with_columns(
        [
            pl.col("tmf_above_zero").alias("tmf_pos"),
            pl.col("tmf_below_zero").alias("tmf_neg"),
            (
                pl.col("tmf_above_zero")
                & (pl.col("tmf_slope_1") > 0)
                & (tmf_prev1 < tmf_prev2)
            )
            .fill_null(False)
            .alias("tmf_respect_zero_up_candidate"),
            (
                pl.col("tmf_below_zero")
                & (pl.col("tmf_slope_1") < 0)
                & (tmf_prev1 > tmf_prev2)
            )
            .fill_null(False)
            .alias("tmf_respect_zero_down_candidate"),
        ]
    )

    cast_exprs: list[pl.Expr] = []
    for column in FLOAT_COLUMNS:
        if column in out.columns:
            cast_exprs.append(pl.col(column).cast(indicator_float_dtype, strict=False).alias(column))
    if cast_exprs:
        out = out.with_columns(cast_exprs)

    ordered_columns = [
        "ticker",
        "exchange",
        "trade_date",
        "trade_dt",
        "close",
        "volume",
        "tr",
        "atr_14",
        "quality_warn_count",
        "tmf_21",
        "tmf_ready_21",
        "tmf_zero_cross_up",
        "tmf_zero_cross_down",
        "tmf_sign",
        "tmf_slope_1",
        "tmf_slope_5",
        "tmf_ema_5",
        "tmf_abs",
        "tmf_above_zero",
        "tmf_below_zero",
        "tmf_pos",
        "tmf_neg",
        "tmf_respect_zero_up_candidate",
        "tmf_respect_zero_down_candidate",
        "tti_proxy_v1_21",
        "tti_proxy_ready_21",
        "tti_proxy_zero_cross_up",
        "tti_proxy_zero_cross_down",
        "tti_proxy_sign",
        "tti_formula_status",
        "tti_proxy_version",
        "indicator_schema_version",
        "indicator_calc_version",
        "tmf_period",
        "run_id",
        "built_ts",
    ]
    existing = [column for column in ordered_columns if column in out.columns]
    remaining = [column for column in out.columns if column not in existing]
    return out.select(existing + remaining)

