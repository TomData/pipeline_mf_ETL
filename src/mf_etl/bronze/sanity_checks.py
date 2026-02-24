"""Global Bronze sanity checks and QA artifact generation."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from mf_etl.bronze.symbol_master import (
    discover_bronze_parquet_files,
    load_latest_quality_reports,
    quality_reports_root,
    symbol_master_paths,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BronzeSanityResult:
    """Result bundle for Bronze global sanity checks."""

    summary: dict[str, Any]
    summary_path: Path
    by_exchange_path: Path
    by_exchange_csv_path: Path
    rows_by_year_path: Path
    rows_by_year_csv_path: Path
    by_exchange_df: pl.DataFrame
    rows_by_year_df: pl.DataFrame


def _atomic_temp_path(target_path: Path) -> Path:
    """Create temp path near target for atomic replacement."""

    return target_path.parent / f".{target_path.name}.{uuid4().hex}.tmp"


def _write_json_atomically(payload: dict[str, Any], path: Path) -> Path:
    """Write JSON file atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_temp_path(path)
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def _write_df_pair_atomically(df: pl.DataFrame, parquet_path: Path, csv_path: Path) -> tuple[Path, Path]:
    """Write dataframe as parquet and csv atomically."""

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_tmp = _atomic_temp_path(parquet_path)
    csv_tmp = _atomic_temp_path(csv_path)
    try:
        df.write_parquet(parquet_tmp)
        df.write_csv(csv_tmp)
        os.replace(parquet_tmp, parquet_path)
        os.replace(csv_tmp, csv_path)
    finally:
        if parquet_tmp.exists():
            parquet_tmp.unlink()
        if csv_tmp.exists():
            csv_tmp.unlink()
    return parquet_path, csv_path


def _empty_exchange_rollup() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "exchange": pl.String,
            "ticker_count": pl.Int64,
            "total_rows": pl.Int64,
            "min_date": pl.Date,
            "max_date": pl.Date,
            "total_warn_rows": pl.Int64,
        }
    )


def _empty_year_rollup() -> pl.DataFrame:
    return pl.DataFrame(schema={"year": pl.Int64, "row_count": pl.Int64})


def _rows_by_year_from_bronze_files(
    bronze_files: list[Path],
    logger: logging.Logger | None = None,
) -> tuple[pl.DataFrame, int]:
    """Build year-level row distribution by scanning trade_date from Bronze files."""

    effective_logger = logger or LOGGER
    year_counts: dict[int, int] = {}
    read_errors = 0

    for parquet_path in bronze_files:
        try:
            frame = pl.read_parquet(parquet_path, columns=["trade_date"])
        except Exception as exc:
            read_errors += 1
            effective_logger.warning("sanity_checks.rows_by_year_read_failed path=%s error=%s", parquet_path, exc)
            continue
        if frame.height == 0 or "trade_date" not in frame.columns:
            continue

        year_rows = (
            frame.drop_nulls(subset=["trade_date"])
            .with_columns(pl.col("trade_date").dt.year().cast(pl.Int64).alias("year"))
            .group_by("year")
            .len(name="row_count")
            .to_dicts()
        )
        for row in year_rows:
            year = int(row["year"])
            year_counts[year] = year_counts.get(year, 0) + int(row["row_count"])

    if not year_counts:
        return _empty_year_rollup(), read_errors

    df = pl.DataFrame(
        [{"year": year, "row_count": count} for year, count in sorted(year_counts.items())],
        schema_overrides={"year": pl.Int64, "row_count": pl.Int64},
    )
    return df, read_errors


def _top_rows(df: pl.DataFrame, sort_cols: list[tuple[str, bool]], limit: int = 20) -> list[dict[str, Any]]:
    """Return top rows as dictionaries with optional descending sort."""

    if df.height == 0:
        return []
    sort_by = [col for col, _ in sort_cols]
    descending = [desc for _, desc in sort_cols]
    return df.sort(sort_by, descending=descending).head(limit).to_dicts()


def _date_span_within_symbol_master(symbol_master_df: pl.DataFrame) -> pl.DataFrame:
    """Add date-span days for each ticker row."""

    if symbol_master_df.height == 0:
        return symbol_master_df
    return symbol_master_df.with_columns(
        (pl.col("last_date") - pl.col("first_date")).dt.total_days().cast(pl.Int64).alias("date_span_days")
    )


def run_bronze_sanity_checks(
    bronze_root: Path,
    artifacts_root: Path,
    *,
    symbol_master_df: pl.DataFrame | None = None,
    logger: logging.Logger | None = None,
) -> BronzeSanityResult:
    """Run global Bronze sanity checks and write QA artifacts."""

    effective_logger = logger or LOGGER
    generated_ts = datetime.now(timezone.utc)

    if symbol_master_df is None:
        symbol_master_parquet, _ = symbol_master_paths(bronze_root)
        if not symbol_master_parquet.exists():
            raise FileNotFoundError(
                f"Symbol master not found at {symbol_master_parquet}. Run build-symbol-master first."
            )
        symbol_master_df = pl.read_parquet(symbol_master_parquet)

    bronze_files = discover_bronze_parquet_files(bronze_root)
    bronze_file_count = len(bronze_files)
    empty_bronze_files_count = (
        int(symbol_master_df.filter(pl.col("row_count") <= 0).height) if symbol_master_df.height > 0 else 0
    )
    duplicate_ticker_entries_in_symbol_master = (
        int(symbol_master_df.select(pl.col("ticker").is_duplicated().cast(pl.Int64).sum()).item())
        if symbol_master_df.height > 0
        else 0
    )

    ticker_count = int(symbol_master_df.height)
    exchange_count_map = (
        {
            str(row["exchange"]): int(row["count"])
            for row in symbol_master_df.group_by("exchange").len(name="count").to_dicts()
        }
        if symbol_master_df.height > 0
        else {}
    )
    total_rows = int(symbol_master_df.select(pl.col("row_count").sum()).item()) if symbol_master_df.height > 0 else 0
    total_warn_rows = (
        int(symbol_master_df.select(pl.col("warn_row_count").sum()).item()) if symbol_master_df.height > 0 else 0
    )
    total_invalid_rows = (
        int(symbol_master_df.select(pl.col("invalid_row_count").sum()).item()) if symbol_master_df.height > 0 else 0
    )

    global_min_trade_date = None
    global_max_trade_date = None
    if symbol_master_df.height > 0:
        bounds = symbol_master_df.select(
            [pl.col("first_date").min().alias("min_date"), pl.col("last_date").max().alias("max_date")]
        ).to_dicts()[0]
        global_min_trade_date = bounds["min_date"]
        global_max_trade_date = bounds["max_date"]

    top_warn_tickers = _top_rows(
        symbol_master_df.select(
            ["ticker", "exchange", "warn_row_count", "row_count", "first_date", "last_date"]
        ),
        sort_cols=[("warn_row_count", True), ("row_count", True)],
        limit=20,
    )
    top_row_count_tickers = _top_rows(
        symbol_master_df.select(["ticker", "exchange", "row_count", "first_date", "last_date"]),
        sort_cols=[("row_count", True)],
        limit=20,
    )
    date_span_df = _date_span_within_symbol_master(
        symbol_master_df.select(["ticker", "exchange", "first_date", "last_date", "row_count"])
    )
    top_date_span_tickers = _top_rows(
        date_span_df.select(["ticker", "exchange", "date_span_days", "first_date", "last_date", "row_count"]),
        sort_cols=[("date_span_days", True), ("row_count", True)],
        limit=20,
    )

    by_exchange_df = (
        symbol_master_df.group_by("exchange")
        .agg(
            pl.len().alias("ticker_count"),
            pl.col("row_count").sum().alias("total_rows"),
            pl.col("first_date").min().alias("min_date"),
            pl.col("last_date").max().alias("max_date"),
            pl.col("warn_row_count").sum().alias("total_warn_rows"),
        )
        .sort("exchange")
        if symbol_master_df.height > 0
        else _empty_exchange_rollup()
    )

    rows_by_year_df, year_rollup_read_errors = _rows_by_year_from_bronze_files(bronze_files, logger=effective_logger)

    latest_quality_map, quality_reports_total_files = load_latest_quality_reports(
        quality_reports_root(bronze_root),
        logger=effective_logger,
    )
    symbol_tickers = set(symbol_master_df.select("ticker").to_series().to_list()) if symbol_master_df.height > 0 else set()
    quality_tickers = set(latest_quality_map.keys())
    quality_reports_missing_for_symbols = len(symbol_tickers - quality_tickers)
    quality_reports_extra_without_symbol = len(quality_tickers - symbol_tickers)

    artifacts_dir = artifacts_root / "bronze_qa"
    summary_path = artifacts_dir / "bronze_sanity_summary.json"
    by_exchange_path = artifacts_dir / "bronze_sanity_by_exchange.parquet"
    by_exchange_csv_path = artifacts_dir / "bronze_sanity_by_exchange.csv"
    rows_by_year_path = artifacts_dir / "bronze_rows_by_year.parquet"
    rows_by_year_csv_path = artifacts_dir / "bronze_rows_by_year.csv"

    summary: dict[str, Any] = {
        "generated_ts": generated_ts.isoformat(),
        "ticker_count": ticker_count,
        "exchange_counts": dict(sorted(exchange_count_map.items())),
        "total_rows": total_rows,
        "global_min_trade_date": global_min_trade_date.isoformat() if global_min_trade_date is not None else None,
        "global_max_trade_date": global_max_trade_date.isoformat() if global_max_trade_date is not None else None,
        "bronze_file_count": bronze_file_count,
        "empty_bronze_files_count": empty_bronze_files_count,
        "duplicate_ticker_entries_in_symbol_master": duplicate_ticker_entries_in_symbol_master,
        "total_warn_rows": total_warn_rows,
        "total_invalid_rows": total_invalid_rows,
        "top_tickers_by_warn_rows": top_warn_tickers,
        "top_tickers_by_row_count": top_row_count_tickers,
        "top_tickers_by_date_span_days": top_date_span_tickers,
        "quality_reports_count": quality_reports_total_files,
        "quality_reports_missing_for_symbols": quality_reports_missing_for_symbols,
        "quality_reports_extra_without_symbol": quality_reports_extra_without_symbol,
        "rows_by_year_read_errors": year_rollup_read_errors,
    }

    _write_json_atomically(summary, summary_path)
    _write_df_pair_atomically(by_exchange_df, by_exchange_path, by_exchange_csv_path)
    _write_df_pair_atomically(rows_by_year_df, rows_by_year_path, rows_by_year_csv_path)

    return BronzeSanityResult(
        summary=summary,
        summary_path=summary_path,
        by_exchange_path=by_exchange_path,
        by_exchange_csv_path=by_exchange_csv_path,
        rows_by_year_path=rows_by_year_path,
        rows_by_year_csv_path=rows_by_year_csv_path,
        by_exchange_df=by_exchange_df,
        rows_by_year_df=rows_by_year_df,
    )
