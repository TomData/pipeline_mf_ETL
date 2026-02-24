"""Row-level validation rules for normalized Bronze rows."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

HARD_ERROR_FLAGS: tuple[str, ...] = (
    "q_missing_field",
    "q_parse_error",
    "q_bad_timeframe",
    "q_nonpositive_price",
    "q_high_lt_low",
    "q_ohlc_outside_hl",
    "q_negative_volume",
    "q_duplicate_ticker_date",
)

WARNING_FLAGS: tuple[str, ...] = (
    "q_suspicious_bar",
    "q_gap_in_calendar",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "quality_error_count",
    "quality_warn_count",
    "is_valid_row",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "ticker",
    "raw_per",
    "raw_date",
    "raw_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_date",
)


@dataclass(frozen=True, slots=True)
class ValidationThresholds:
    """Threshold configuration for warning-level validation rules."""

    suspicious_range_pct_threshold: float = 0.5
    suspicious_return_pct_threshold: float = 0.3
    gap_days_warn_threshold: int = 7


def _assert_required_schema(df: pl.DataFrame) -> None:
    """Raise on structural schema failures required by validation."""

    required = {
        "ticker",
        "exchange",
        "timeframe",
        "trade_date",
        "trade_dt",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "raw_per",
        "raw_date",
        "raw_time",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for validation: {', '.join(missing)}")


def _null_or_blank(column: str) -> pl.Expr:
    """Return expression that detects null/blank strings safely."""

    text = pl.col(column).cast(pl.String, strict=False).str.strip_chars()
    return pl.col(column).is_null() | text.is_null() | (text == "")


def apply_quality_flags(
    df: pl.DataFrame,
    thresholds: ValidationThresholds | None = None,
) -> pl.DataFrame:
    """Apply row-level quality flags and summary columns to normalized Bronze rows."""

    _assert_required_schema(df)
    rule_thresholds = thresholds or ValidationThresholds()

    with_index = df.with_row_index("__row_idx")

    range_ratio = pl.when(pl.col("close").abs() > 0).then((pl.col("high") - pl.col("low")) / pl.col("close").abs())
    return_ratio = pl.when(pl.col("open").abs() > 0).then((pl.col("close") / pl.col("open") - 1.0).abs())

    flagged = with_index.with_columns(
        [
            (
                _null_or_blank("ticker")
                | _null_or_blank("raw_per")
                | _null_or_blank("raw_date")
                | _null_or_blank("raw_time")
                | pl.col("open").is_null()
                | pl.col("high").is_null()
                | pl.col("low").is_null()
                | pl.col("close").is_null()
                | pl.col("volume").is_null()
                | pl.col("trade_date").is_null()
            )
            .fill_null(True)
            .alias("q_missing_field"),
            (
                pl.col("trade_date").is_null()
                | pl.col("trade_dt").is_null()
                | pl.col("open").is_null()
                | pl.col("high").is_null()
                | pl.col("low").is_null()
                | pl.col("close").is_null()
                | pl.col("volume").is_null()
            )
            .fill_null(True)
            .alias("q_parse_error"),
            (pl.col("timeframe") != "D1").fill_null(True).alias("q_bad_timeframe"),
            (
                (pl.col("open") <= 0)
                | (pl.col("high") <= 0)
                | (pl.col("low") <= 0)
                | (pl.col("close") <= 0)
            )
            .fill_null(False)
            .alias("q_nonpositive_price"),
            (pl.col("high") < pl.col("low")).fill_null(False).alias("q_high_lt_low"),
            (
                (pl.col("open") < pl.col("low"))
                | (pl.col("open") > pl.col("high"))
                | (pl.col("close") < pl.col("low"))
                | (pl.col("close") > pl.col("high"))
            )
            .fill_null(False)
            .alias("q_ohlc_outside_hl"),
            (pl.col("volume") < 0).fill_null(False).alias("q_negative_volume"),
            (
                pl.when(pl.col("ticker").is_not_null() & pl.col("trade_date").is_not_null())
                .then(pl.struct(["ticker", "trade_date"]).is_duplicated())
                .otherwise(False)
            )
            .fill_null(False)
            .alias("q_duplicate_ticker_date"),
            (
                (range_ratio > rule_thresholds.suspicious_range_pct_threshold).fill_null(False)
                | (return_ratio > rule_thresholds.suspicious_return_pct_threshold).fill_null(False)
            )
            .fill_null(False)
            .alias("q_suspicious_bar"),
        ]
    )

    gap_flags = (
        flagged.sort(["ticker", "trade_date", "__row_idx"])
        .with_columns(
            [
                (
                    pl.when(pl.col("ticker").is_not_null() & pl.col("trade_date").is_not_null())
                    .then(
                        (
                            (pl.col("trade_date") - pl.col("trade_date").shift(1).over("ticker"))
                            .dt.total_days()
                            > rule_thresholds.gap_days_warn_threshold
                        )
                    )
                    .otherwise(False)
                )
                .fill_null(False)
                .alias("q_gap_in_calendar")
            ]
        )
        .select(["__row_idx", "q_gap_in_calendar"])
    )

    flagged = flagged.join(gap_flags, on="__row_idx", how="left").with_columns(
        pl.col("q_gap_in_calendar").fill_null(False)
    )

    hard_error_exprs = [pl.col(flag).cast(pl.Int64) for flag in HARD_ERROR_FLAGS]
    warning_exprs = [pl.col(flag).cast(pl.Int64) for flag in WARNING_FLAGS]

    validated = flagged.with_columns(
        [
            pl.sum_horizontal(*hard_error_exprs).cast(pl.Int64).alias("quality_error_count"),
            pl.sum_horizontal(*warning_exprs).cast(pl.Int64).alias("quality_warn_count"),
        ]
    ).with_columns((pl.col("quality_error_count") == 0).alias("is_valid_row"))

    return validated.drop("__row_idx")


def validate_ohlcv(df: pl.DataFrame) -> list[str]:
    """Compatibility helper retained for legacy callers."""

    errors: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
    return errors
