"""Silver base feature engineering on top of Bronze valid rows."""

from __future__ import annotations

from typing import Final

import polars as pl

from mf_etl.transform.dtypes import SILVER_SCHEMA_VERSION

SILVER_CALC_VERSION: Final[str] = "base_v1"
EPSILON: Final[float] = 1e-12

BASE_FLOAT_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "hl_range",
    "body",
    "body_abs",
    "range_safe",
    "body_to_range",
    "close_pos_in_range",
    "open_pos_in_range",
    "hlc3",
    "ohlc4",
    "prev_close",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "gap_from_prev_close",
    "tr",
    "atr_14",
    "atr_pct_14",
    "range_pct_close",
    "dollar_volume",
    "vol_sma_20",
    "vol_ratio_20",
    "dollar_vol_sma_20",
    "volume_z_20",
    "high_20",
    "low_20",
    "close_sma_20",
    "close_sma_50",
    "close_sma_200",
    "close_vs_sma20",
    "close_vs_sma50",
    "close_vs_sma200",
)


def _safe_div(numerator: pl.Expr, denominator: pl.Expr, eps: float = EPSILON) -> pl.Expr:
    """Null-safe division returning null when denominator is too small."""

    return pl.when(denominator.abs() > eps).then(numerator / denominator).otherwise(None)


def _ensure_required_columns(bronze_df: pl.DataFrame) -> pl.DataFrame:
    """Ensure required Bronze columns exist; raise on structural issues."""

    required = {
        "ticker",
        "exchange",
        "trade_date",
        "trade_dt",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source_file",
    }
    missing = sorted(required.difference(bronze_df.columns))
    if missing:
        raise ValueError(f"Bronze frame missing required columns: {', '.join(missing)}")

    out = bronze_df
    if "run_id" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.String).alias("run_id"))
    if "quality_warn_count" not in out.columns:
        out = out.with_columns(pl.lit(0, dtype=pl.Int64).alias("quality_warn_count"))
    if "is_valid_row" not in out.columns:
        out = out.with_columns(pl.lit(True).alias("is_valid_row"))
    if "source_line_no" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Int64).alias("source_line_no"))
    if "raw_ticker" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.String).alias("raw_ticker"))
    if "raw_per" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.String).alias("raw_per"))
    if "raw_date" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.String).alias("raw_date"))
    if "raw_time" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.String).alias("raw_time"))
    if "openint" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Int64).alias("openint"))
    return out


def build_silver_base_features(
    bronze_df: pl.DataFrame,
    *,
    silver_float_dtype: pl.DataType = pl.Float32,
    fallback_run_id: str | None = None,
    epsilon: float = EPSILON,
) -> pl.DataFrame:
    """Build Silver base helper features from one ticker's Bronze valid rows.

    `silver_ready_base` is set when the 50-day context is available.
    """

    base = _ensure_required_columns(bronze_df)

    sort_cols = [col for col in ("ticker", "trade_date", "source_line_no") if col in base.columns]
    sorted_df = base.sort(sort_cols) if sort_cols else base

    out = sorted_df.with_columns(
        [
            pl.col("ticker").cast(pl.String).str.strip_chars().str.to_uppercase().alias("ticker"),
            pl.col("exchange").cast(pl.String).str.strip_chars().str.to_uppercase().alias("exchange"),
            pl.col("run_id")
            .cast(pl.String, strict=False)
            .fill_null(fallback_run_id if fallback_run_id is not None else None)
            .alias("run_id"),
            pl.lit(SILVER_SCHEMA_VERSION).alias("schema_version"),
            pl.lit(SILVER_CALC_VERSION).alias("silver_calc_version"),
            pl.col("openint").cast(pl.Int64, strict=False).alias("openint"),
            pl.col("quality_warn_count").cast(pl.Int64, strict=False).fill_null(0).alias("quality_warn_count"),
            pl.col("is_valid_row").cast(pl.Boolean, strict=False).fill_null(True).alias("is_valid_row"),
        ]
    )

    prev_close = pl.col("close").shift(1).over("ticker")
    close_5 = pl.col("close").shift(5).over("ticker")
    close_20 = pl.col("close").shift(20).over("ticker")
    hl_range_raw = pl.col("high") - pl.col("low")
    range_safe = pl.max_horizontal([hl_range_raw, pl.lit(epsilon)])
    tr_expr = pl.coalesce(
        [
            pl.max_horizontal(
                [
                    hl_range_raw,
                    (pl.col("high") - prev_close).abs(),
                    (pl.col("low") - prev_close).abs(),
                ]
            ),
            hl_range_raw,
        ]
    )

    out = out.with_columns(
        [
            hl_range_raw.alias("hl_range"),
            (pl.col("close") - pl.col("open")).alias("body"),
            (pl.col("close") - pl.col("open")).abs().alias("body_abs"),
            range_safe.alias("range_safe"),
            _safe_div(pl.col("close") - pl.col("open"), range_safe, eps=epsilon).alias("body_to_range"),
            _safe_div(pl.col("close") - pl.col("low"), range_safe, eps=epsilon).alias("close_pos_in_range"),
            _safe_div(pl.col("open") - pl.col("low"), range_safe, eps=epsilon).alias("open_pos_in_range"),
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("hlc3"),
            ((pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4.0).alias("ohlc4"),
            prev_close.alias("prev_close"),
            _safe_div(pl.col("close"), prev_close, eps=epsilon).sub(1.0).alias("ret_1d"),
            _safe_div(pl.col("close"), close_5, eps=epsilon).sub(1.0).alias("ret_5d"),
            _safe_div(pl.col("close"), close_20, eps=epsilon).sub(1.0).alias("ret_20d"),
            _safe_div(pl.col("open"), prev_close, eps=epsilon).sub(1.0).alias("gap_from_prev_close"),
            tr_expr.alias("tr"),
        ]
    )

    out = out.with_columns(
        [
            pl.col("tr").rolling_mean(window_size=14, min_samples=14).over("ticker").alias("atr_14"),
            _safe_div(pl.col("tr").rolling_mean(window_size=14, min_samples=14).over("ticker"), pl.col("close"), eps=epsilon).alias("atr_pct_14"),
            _safe_div(hl_range_raw, pl.col("close"), eps=epsilon).alias("range_pct_close"),
            (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
            pl.col("volume").rolling_mean(window_size=20, min_samples=20).over("ticker").alias("vol_sma_20"),
            _safe_div(
                pl.col("volume"),
                pl.col("volume").rolling_mean(window_size=20, min_samples=20).over("ticker"),
                eps=epsilon,
            ).alias("vol_ratio_20"),
            (pl.col("close") * pl.col("volume"))
            .rolling_mean(window_size=20, min_samples=20)
            .over("ticker")
            .alias("dollar_vol_sma_20"),
            _safe_div(
                pl.col("volume") - pl.col("volume").rolling_mean(window_size=20, min_samples=20).over("ticker"),
                pl.col("volume").rolling_std(window_size=20, min_samples=20, ddof=0).over("ticker"),
                eps=epsilon,
            ).alias("volume_z_20"),
            pl.col("high").rolling_max(window_size=20, min_samples=20).over("ticker").alias("high_20"),
            pl.col("low").rolling_min(window_size=20, min_samples=20).over("ticker").alias("low_20"),
            pl.col("close").rolling_mean(window_size=20, min_samples=20).over("ticker").alias("close_sma_20"),
            pl.col("close").rolling_mean(window_size=50, min_samples=50).over("ticker").alias("close_sma_50"),
            pl.col("close").rolling_mean(window_size=200, min_samples=200).over("ticker").alias("close_sma_200"),
        ]
    )

    out = out.with_columns(
        [
            _safe_div(pl.col("close"), pl.col("close_sma_20"), eps=epsilon).sub(1.0).alias("close_vs_sma20"),
            _safe_div(pl.col("close"), pl.col("close_sma_50"), eps=epsilon).sub(1.0).alias("close_vs_sma50"),
            _safe_div(pl.col("close"), pl.col("close_sma_200"), eps=epsilon).sub(1.0).alias("close_vs_sma200"),
            pl.lit(1).cum_sum().over("ticker").ge(14).alias("warmup_14_complete"),
            pl.lit(1).cum_sum().over("ticker").ge(20).alias("warmup_20_complete"),
            pl.lit(1).cum_sum().over("ticker").ge(50).alias("warmup_50_complete"),
            pl.lit(1).cum_sum().over("ticker").ge(200).alias("warmup_200_complete"),
        ]
    ).with_columns(pl.col("warmup_50_complete").alias("silver_ready_base"))

    cast_exprs: list[pl.Expr] = []
    for column in BASE_FLOAT_COLUMNS:
        if column in out.columns:
            cast_exprs.append(pl.col(column).cast(silver_float_dtype, strict=False).alias(column))
    if cast_exprs:
        out = out.with_columns(cast_exprs)

    ordered_columns = [
        "ticker",
        "exchange",
        "trade_date",
        "trade_dt",
        "source_file",
        "run_id",
        "schema_version",
        "silver_calc_version",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "openint",
        "hl_range",
        "body",
        "body_abs",
        "range_safe",
        "body_to_range",
        "close_pos_in_range",
        "open_pos_in_range",
        "hlc3",
        "ohlc4",
        "prev_close",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "gap_from_prev_close",
        "tr",
        "atr_14",
        "atr_pct_14",
        "range_pct_close",
        "dollar_volume",
        "vol_sma_20",
        "vol_ratio_20",
        "dollar_vol_sma_20",
        "volume_z_20",
        "high_20",
        "low_20",
        "close_sma_20",
        "close_sma_50",
        "close_sma_200",
        "close_vs_sma20",
        "close_vs_sma50",
        "close_vs_sma200",
        "warmup_14_complete",
        "warmup_20_complete",
        "warmup_50_complete",
        "warmup_200_complete",
        "silver_ready_base",
        "quality_warn_count",
        "is_valid_row",
    ]
    existing_ordered = [column for column in ordered_columns if column in out.columns]
    remaining = [column for column in out.columns if column not in existing_ordered]
    return out.select(existing_ordered + remaining)
