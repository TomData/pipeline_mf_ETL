"""Bronze artifact writers with atomic file replacement."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from mf_etl.transform.dtypes import BRONZE_SCHEMA_VERSION
from mf_etl.validate.reports import ValidationResult

DEFAULT_SORT_COLUMNS: tuple[str, ...] = ("ticker", "trade_date", "source_line_no")
REQUIRED_BRONZE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "ingest_ts",
    "source_file",
    "source_file_name",
    "source_line_no",
    "raw_ticker",
    "raw_per",
    "raw_date",
    "raw_time",
    "quality_error_count",
    "quality_warn_count",
    "is_valid_row",
    "schema_version",
)


@dataclass(frozen=True, slots=True)
class BronzeWriteResult:
    """Outcome of writing one ticker's Bronze artifacts."""

    ticker: str
    exchange: str
    bronze_path: Path
    rejects_path: Path | None
    quality_report_path: Path
    malformed_rows_path: Path | None
    rows_valid: int
    rows_invalid: int
    rows_total: int
    wrote_bronze: bool
    wrote_rejects: bool
    wrote_quality_report: bool
    wrote_malformed_rows: bool


def _atomic_temp_path(target_path: Path) -> Path:
    """Create a unique temp path next to the target for atomic replacement."""

    return target_path.parent / f".{target_path.name}.{uuid4().hex}.tmp"


def _write_parquet_atomically(
    df: pl.DataFrame,
    output_path: Path,
    *,
    compression: str,
    compression_level: int | None,
    statistics: bool,
) -> Path:
    """Write parquet atomically via temporary file then os.replace."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(output_path)
    try:
        df.write_parquet(
            temp_path,
            compression=compression,
            compression_level=compression_level,
            statistics=statistics,
        )
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return output_path


def _write_json_atomically(payload: dict[str, Any], output_path: Path) -> Path:
    """Write JSON atomically via temporary file then os.replace."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(output_path)
    try:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return output_path


def _extract_first_text(df: pl.DataFrame, column: str) -> str | None:
    """Extract first non-null, non-empty string value from a column."""

    if df.height == 0 or column not in df.columns:
        return None
    values = (
        df.select(pl.col(column).cast(pl.String, strict=False).str.strip_chars().alias(column))
        .filter(pl.col(column).is_not_null() & (pl.col(column) != ""))
        .head(1)
        .to_dicts()
    )
    if not values:
        return None
    return values[0][column]


def _normalize_identity(
    validation_result: ValidationResult,
    quality_report: dict[str, Any],
    fallback_ticker: str,
    fallback_exchange: str,
    fallback_run_id: str,
) -> tuple[str, str, str]:
    """Resolve ticker/exchange/run_id from validated data + report + fallback values."""

    ticker = _extract_first_text(validation_result.validated_df, "ticker")
    exchange = _extract_first_text(validation_result.validated_df, "exchange")
    run_id = _extract_first_text(validation_result.validated_df, "run_id")

    ticker = (
        ticker
        or str(quality_report.get("ticker") or "").strip()
        or fallback_ticker.strip().upper()
        or "UNKNOWN"
    )
    exchange = (
        exchange
        or str(quality_report.get("exchange") or "").strip()
        or fallback_exchange.strip().upper()
        or "UNKNOWN"
    )
    run_id = run_id or str(quality_report.get("run_id") or "").strip() or fallback_run_id.strip() or "unknown-run"
    return ticker.upper(), exchange.upper(), run_id


def _ensure_bronze_output_columns(valid_rows: pl.DataFrame, run_id: str) -> pl.DataFrame:
    """Guarantee required metadata and quality columns exist on Bronze output."""

    out = valid_rows
    existing = set(out.columns)
    for column in REQUIRED_BRONZE_COLUMNS:
        if column in existing:
            continue
        if column == "schema_version":
            out = out.with_columns(pl.lit(BRONZE_SCHEMA_VERSION).alias("schema_version"))
        elif column == "is_valid_row":
            out = out.with_columns(pl.lit(True).alias("is_valid_row"))
        elif column in {"quality_error_count", "quality_warn_count", "source_line_no"}:
            out = out.with_columns(pl.lit(0, dtype=pl.Int64).alias(column))
        elif column == "run_id":
            out = out.with_columns(pl.lit(run_id).cast(pl.String).alias("run_id"))
        else:
            out = out.with_columns(pl.lit(None, dtype=pl.String).alias(column))

    # Ensure schema_version value is set consistently for all rows.
    out = out.with_columns(pl.lit(BRONZE_SCHEMA_VERSION).cast(pl.String).alias("schema_version"))
    return out


def _sorted(df: pl.DataFrame, preferred_columns: tuple[str, ...] = DEFAULT_SORT_COLUMNS) -> pl.DataFrame:
    """Sort dataframe by available preferred columns."""

    columns = [column for column in preferred_columns if column in df.columns]
    return df.sort(columns) if columns else df


def write_bronze_parquet(
    df: pl.DataFrame,
    output_path: Path,
    compression: str = "zstd",
    compression_level: int | None = 3,
    statistics: bool = True,
) -> Path:
    """Write a Polars DataFrame to parquet atomically."""

    return _write_parquet_atomically(
        df,
        output_path,
        compression=compression,
        compression_level=compression_level,
        statistics=statistics,
    )


def write_bronze_artifacts(
    *,
    bronze_root: Path,
    validation_result: ValidationResult,
    quality_report: dict[str, Any],
    fallback_ticker: str,
    fallback_exchange: str,
    fallback_run_id: str,
    compression: str = "zstd",
    compression_level: int | None = 3,
    statistics: bool = True,
    malformed_rows: pl.DataFrame | None = None,
) -> BronzeWriteResult:
    """Write Bronze, rejects, and quality artifacts for one ticker run."""

    ticker, exchange, run_id = _normalize_identity(
        validation_result,
        quality_report,
        fallback_ticker=fallback_ticker,
        fallback_exchange=fallback_exchange,
        fallback_run_id=fallback_run_id,
    )

    prefix = ticker[0] if ticker else "_"
    if not prefix.strip():
        prefix = "_"

    bronze_path = (
        bronze_root
        / "ohlcv_by_symbol"
        / f"exchange={exchange}"
        / f"prefix={prefix}"
        / f"ticker={ticker}"
        / "part-000.parquet"
    )
    quality_report_path = (
        bronze_root / "quality" / "ticker_reports" / f"run_id={run_id}" / f"ticker={ticker}.json"
    )

    valid_rows = _ensure_bronze_output_columns(validation_result.valid_rows, run_id=run_id)
    valid_rows = _sorted(valid_rows)
    write_bronze_parquet(
        valid_rows,
        bronze_path,
        compression=compression,
        compression_level=compression_level,
        statistics=statistics,
    )

    rejects_path: Path | None = None
    wrote_rejects = False
    if validation_result.reject_rows.height > 0:
        rejects_path = bronze_root / "rejects" / f"run_id={run_id}" / f"ticker={ticker}" / "rejects.parquet"
        reject_rows = _sorted(validation_result.reject_rows)
        _write_parquet_atomically(
            reject_rows,
            rejects_path,
            compression=compression,
            compression_level=compression_level,
            statistics=statistics,
        )
        wrote_rejects = True

    malformed_rows_path: Path | None = None
    wrote_malformed_rows = False
    if malformed_rows is not None and malformed_rows.height > 0:
        malformed_rows_path = (
            bronze_root
            / "rejects_raw"
            / f"run_id={run_id}"
            / f"ticker={ticker}"
            / "malformed_rows.parquet"
        )
        _write_parquet_atomically(
            malformed_rows,
            malformed_rows_path,
            compression=compression,
            compression_level=compression_level,
            statistics=statistics,
        )
        wrote_malformed_rows = True

    report_payload = dict(quality_report)
    report_payload.update(
        {
            "ticker": ticker,
            "exchange": exchange,
            "run_id": run_id,
            "rows_total": validation_result.validated_df.height,
            "rows_valid": validation_result.valid_rows.height,
            "rows_invalid": validation_result.reject_rows.height,
            "bronze_path": str(bronze_path),
            "rejects_path": str(rejects_path) if rejects_path else None,
            "malformed_rows_path": str(malformed_rows_path) if malformed_rows_path else None,
            "schema_version": BRONZE_SCHEMA_VERSION,
        }
    )
    _write_json_atomically(report_payload, quality_report_path)

    return BronzeWriteResult(
        ticker=ticker,
        exchange=exchange,
        bronze_path=bronze_path,
        rejects_path=rejects_path,
        quality_report_path=quality_report_path,
        malformed_rows_path=malformed_rows_path,
        rows_valid=validation_result.valid_rows.height,
        rows_invalid=validation_result.reject_rows.height,
        rows_total=validation_result.validated_df.height,
        wrote_bronze=True,
        wrote_rejects=wrote_rejects,
        wrote_quality_report=True,
        wrote_malformed_rows=wrote_malformed_rows,
    )
