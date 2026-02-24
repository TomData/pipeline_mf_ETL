"""Validation orchestration and quality reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import polars as pl

from mf_etl.validate.rules import HARD_ERROR_FLAGS, WARNING_FLAGS, ValidationThresholds, apply_quality_flags


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Container for validated frame, splits, and ticker-level quality report."""

    validated_df: pl.DataFrame
    valid_rows: pl.DataFrame
    reject_rows: pl.DataFrame
    quality_report: dict[str, Any]


def _extract_single_value(df: pl.DataFrame, column: str) -> Any:
    """Extract a scalar value from the first row, if available."""

    if df.height == 0 or column not in df.columns:
        return None
    value = df.select(pl.col(column).first().alias(column)).to_dicts()[0][column]
    return value


def build_ticker_quality_report(
    validated_df: pl.DataFrame,
    *,
    header_skipped: bool | None = None,
    malformed_raw_rows_count: int | None = None,
    generated_ts: datetime | None = None,
) -> dict[str, Any]:
    """Build ticker-level quality metrics from a validated Bronze dataframe."""

    ts = generated_ts or datetime.now(timezone.utc)

    rows_total = validated_df.height
    rows_valid = int(validated_df.filter(pl.col("is_valid_row")).height) if rows_total > 0 else 0
    rows_invalid = rows_total - rows_valid
    warnings_total = (
        int(validated_df.select(pl.col("quality_warn_count").sum()).item()) if rows_total > 0 else 0
    )
    duplicates_count = (
        int(validated_df.select(pl.col("q_duplicate_ticker_date").cast(pl.Int64).sum()).item())
        if rows_total > 0
        else 0
    )
    suspicious_bars_count = (
        int(validated_df.select(pl.col("q_suspicious_bar").cast(pl.Int64).sum()).item()) if rows_total > 0 else 0
    )
    gap_rows_count = (
        int(validated_df.select(pl.col("q_gap_in_calendar").cast(pl.Int64).sum()).item()) if rows_total > 0 else 0
    )

    min_trade_date = (
        _extract_single_value(validated_df.select(pl.col("trade_date").min().alias("trade_date")), "trade_date")
        if rows_total > 0
        else None
    )
    max_trade_date = (
        _extract_single_value(validated_df.select(pl.col("trade_date").max().alias("trade_date")), "trade_date")
        if rows_total > 0
        else None
    )

    report: dict[str, Any] = {
        "ticker": _extract_single_value(validated_df, "ticker"),
        "exchange": _extract_single_value(validated_df, "exchange"),
        "run_id": _extract_single_value(validated_df, "run_id"),
        "rows_total": rows_total,
        "rows_valid": rows_valid,
        "rows_invalid": rows_invalid,
        "warnings_total": warnings_total,
        "duplicates_count": duplicates_count,
        "suspicious_bars_count": suspicious_bars_count,
        "gap_rows_count": gap_rows_count,
        "min_trade_date": min_trade_date.isoformat() if min_trade_date is not None else None,
        "max_trade_date": max_trade_date.isoformat() if max_trade_date is not None else None,
        "header_skipped": bool(header_skipped) if header_skipped is not None else None,
        "malformed_raw_rows_count": int(malformed_raw_rows_count or 0),
        "generated_ts": ts.isoformat(),
    }
    return report


def validate_bronze_dataframe(
    normalized_df: pl.DataFrame,
    *,
    thresholds: ValidationThresholds | None = None,
    header_skipped: bool | None = None,
    malformed_raw_rows_count: int | None = None,
    generated_ts: datetime | None = None,
) -> ValidationResult:
    """Apply quality flags, split valid/reject rows, and generate ticker-level report."""

    validated = apply_quality_flags(normalized_df, thresholds=thresholds)
    valid_rows = validated.filter(pl.col("is_valid_row"))
    reject_rows = validated.filter(~pl.col("is_valid_row"))
    quality_report = build_ticker_quality_report(
        validated,
        header_skipped=header_skipped,
        malformed_raw_rows_count=malformed_raw_rows_count,
        generated_ts=generated_ts,
    )
    return ValidationResult(
        validated_df=validated,
        valid_rows=valid_rows,
        reject_rows=reject_rows,
        quality_report=quality_report,
    )


def format_validation_report(symbol: str, errors: list[str]) -> str:
    """Compatibility report formatter for legacy callers."""

    if not errors:
        return f"symbol={symbol} status=ok errors=0"
    joined = "; ".join(errors)
    return f"symbol={symbol} status=failed errors={len(errors)} details={joined}"


def quality_flag_counts(validated_df: pl.DataFrame) -> dict[str, int]:
    """Return per-flag counts for all hard and warning quality flags."""

    if validated_df.height == 0:
        return {flag: 0 for flag in (*HARD_ERROR_FLAGS, *WARNING_FLAGS)}

    counts: dict[str, int] = {}
    for flag in (*HARD_ERROR_FLAGS, *WARNING_FLAGS):
        counts[flag] = int(validated_df.select(pl.col(flag).cast(pl.Int64).sum()).item())
    return counts
