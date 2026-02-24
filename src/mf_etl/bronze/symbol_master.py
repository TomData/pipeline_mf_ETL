"""Build and persist symbol-level Bronze metadata tables."""

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

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SymbolMasterBuildResult:
    """Result of symbol-master build and artifact writes."""

    symbol_master_df: pl.DataFrame
    parquet_path: Path
    csv_path: Path
    bronze_file_count: int
    bronze_files_read_errors: int
    quality_reports_total_files: int
    quality_reports_latest_tickers: int


def symbol_master_paths(bronze_root: Path) -> tuple[Path, Path]:
    """Return standard symbol-master parquet/csv output paths."""

    base = bronze_root / "symbol_master"
    return base / "symbol_master.parquet", base / "symbol_master.csv"


def quality_reports_root(bronze_root: Path) -> Path:
    """Return root path containing per-run ticker quality JSON reports."""

    return bronze_root / "quality" / "ticker_reports"


def _atomic_temp_path(target_path: Path) -> Path:
    """Create temp path in target directory for atomic replacement."""

    return target_path.parent / f".{target_path.name}.{uuid4().hex}.tmp"


def _write_df_atomically(df: pl.DataFrame, parquet_path: Path, csv_path: Path) -> tuple[Path, Path]:
    """Write dataframe to parquet/csv atomically."""

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


def _empty_symbol_master() -> pl.DataFrame:
    """Return empty symbol-master frame with stable schema."""

    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "exchange": pl.String,
            "prefix": pl.String,
            "bronze_path": pl.String,
            "first_date": pl.Date,
            "last_date": pl.Date,
            "row_count": pl.Int64,
            "invalid_row_count": pl.Int64,
            "warn_row_count": pl.Int64,
            "min_price": pl.Float64,
            "max_price": pl.Float64,
            "avg_volume": pl.Float64,
            "source_file_last": pl.String,
            "schema_version": pl.String,
            "duplicates_count": pl.Int64,
            "suspicious_bars_count": pl.Int64,
            "gap_rows_count": pl.Int64,
            "rows_invalid_reported": pl.Int64,
            "rows_total_reported": pl.Int64,
            "built_ts": pl.Datetime(time_zone="UTC"),
        }
    )


def discover_bronze_parquet_files(bronze_root: Path) -> list[Path]:
    """Discover per-symbol Bronze parquet files recursively."""

    base = bronze_root / "ohlcv_by_symbol"
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*.parquet") if path.is_file())


def _partition_value(path: Path, key: str) -> str | None:
    prefix = f"{key}="
    for part in path.parts:
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _parse_quality_ts(report: dict[str, Any], fallback_mtime: float) -> datetime:
    """Extract generated timestamp from report, fallback to file mtime."""

    value = report.get("generated_ts")
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback_mtime, tz=timezone.utc)


def load_latest_quality_reports(
    reports_root: Path,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Load latest quality report per ticker and return (map, total_files_count)."""

    effective_logger = logger or LOGGER
    if not reports_root.exists():
        return {}, 0

    latest_by_ticker: dict[str, tuple[datetime, dict[str, Any]]] = {}
    total_files = 0
    for report_path in sorted(reports_root.rglob("*.json")):
        if not report_path.is_file():
            continue
        total_files += 1
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            effective_logger.warning("symbol_master.quality_report_read_failed path=%s error=%s", report_path, exc)
            continue

        ticker = str(payload.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        ts = _parse_quality_ts(payload, report_path.stat().st_mtime)
        previous = latest_by_ticker.get(ticker)
        if previous is None or ts >= previous[0]:
            latest_by_ticker[ticker] = (ts, payload)

    out: dict[str, dict[str, Any]] = {}
    for ticker, (_, payload) in latest_by_ticker.items():
        out[ticker] = {
            "duplicates_count": int(payload.get("duplicates_count") or 0),
            "suspicious_bars_count": int(payload.get("suspicious_bars_count") or 0),
            "gap_rows_count": int(payload.get("gap_rows_count") or 0),
            "rows_invalid_reported": int(payload.get("rows_invalid") or 0),
            "rows_total_reported": int(payload.get("rows_total") or 0),
        }
    return out, total_files


def _safe_symbol_row(
    parquet_path: Path,
    built_ts: datetime,
    logger: logging.Logger | None = None,
) -> dict[str, object] | None:
    """Read one Bronze parquet and return symbol-master row metadata."""

    effective_logger = logger or LOGGER
    exchange = (_partition_value(parquet_path, "exchange") or "UNKNOWN").upper()
    prefix = _partition_value(parquet_path, "prefix") or "_"
    ticker = (_partition_value(parquet_path, "ticker") or parquet_path.stem).upper()

    try:
        df = pl.read_parquet(parquet_path)
    except Exception as exc:
        effective_logger.warning("symbol_master.read_parquet_failed path=%s error=%s", parquet_path, exc)
        return None

    row_count = int(df.height)
    if row_count == 0:
        return {
            "ticker": ticker,
            "exchange": exchange,
            "prefix": prefix,
            "bronze_path": str(parquet_path),
            "first_date": None,
            "last_date": None,
            "row_count": 0,
            "invalid_row_count": 0,
            "warn_row_count": 0,
            "min_price": None,
            "max_price": None,
            "avg_volume": None,
            "source_file_last": None,
            "schema_version": None,
            "built_ts": built_ts,
        }

    invalid_row_count = 0
    if "is_valid_row" in df.columns:
        invalid_row_count = int(df.select((~pl.col("is_valid_row")).cast(pl.Int64).sum()).item())

    warn_row_count = 0
    if "quality_warn_count" in df.columns:
        warn_row_count = int(df.select((pl.col("quality_warn_count") > 0).cast(pl.Int64).sum()).item())

    first_date = None
    last_date = None
    if "trade_date" in df.columns:
        date_bounds = df.select(
            [pl.col("trade_date").min().alias("first_date"), pl.col("trade_date").max().alias("last_date")]
        ).to_dicts()[0]
        first_date = date_bounds["first_date"]
        last_date = date_bounds["last_date"]

    min_price = None
    max_price = None
    price_cols = [col for col in ("open", "high", "low", "close") if col in df.columns]
    if price_cols:
        price_stats = df.select(
            [
                pl.min_horizontal(price_cols).min().alias("min_price"),
                pl.max_horizontal(price_cols).max().alias("max_price"),
            ]
        ).to_dicts()[0]
        min_price = price_stats["min_price"]
        max_price = price_stats["max_price"]

    avg_volume = None
    if "volume" in df.columns:
        avg_volume = df.select(pl.col("volume").mean().alias("avg_volume")).to_dicts()[0]["avg_volume"]

    source_file_last = None
    if "source_file" in df.columns:
        if "trade_date" in df.columns and "source_line_no" in df.columns:
            ordered = df.sort(["trade_date", "source_line_no"]).tail(1).to_dicts()
            source_file_last = ordered[0].get("source_file") if ordered else None
        else:
            source_file_last = df.select(pl.col("source_file").max().alias("source_file_last")).to_dicts()[0][
                "source_file_last"
            ]

    schema_version = None
    if "schema_version" in df.columns:
        values = (
            df.select(pl.col("schema_version").cast(pl.String).drop_nulls().head(1).alias("schema_version"))
            .to_dicts()
        )
        schema_version = values[0]["schema_version"] if values else None

    return {
        "ticker": ticker,
        "exchange": exchange,
        "prefix": prefix,
        "bronze_path": str(parquet_path),
        "first_date": first_date,
        "last_date": last_date,
        "row_count": row_count,
        "invalid_row_count": invalid_row_count,
        "warn_row_count": warn_row_count,
        "min_price": min_price,
        "max_price": max_price,
        "avg_volume": avg_volume,
        "source_file_last": source_file_last,
        "schema_version": schema_version,
        "built_ts": built_ts,
    }


def build_symbol_master_dataframe(
    bronze_root: Path,
    *,
    logger: logging.Logger | None = None,
    built_ts: datetime | None = None,
) -> tuple[pl.DataFrame, int, int, int, int]:
    """Build symbol master DataFrame from Bronze parquet outputs."""

    effective_logger = logger or LOGGER
    stamp = built_ts or datetime.now(timezone.utc)
    bronze_files = discover_bronze_parquet_files(bronze_root)

    rows: list[dict[str, object]] = []
    read_errors = 0
    for parquet_path in bronze_files:
        row = _safe_symbol_row(parquet_path, stamp, logger=effective_logger)
        if row is None:
            read_errors += 1
            continue
        rows.append(row)

    if not rows:
        quality_map, quality_file_count = load_latest_quality_reports(quality_reports_root(bronze_root), logger=logger)
        return _empty_symbol_master(), len(bronze_files), read_errors, quality_file_count, len(quality_map)

    symbol_master = pl.DataFrame(
        rows,
        schema_overrides={
            "ticker": pl.String,
            "exchange": pl.String,
            "prefix": pl.String,
            "bronze_path": pl.String,
            "first_date": pl.Date,
            "last_date": pl.Date,
            "row_count": pl.Int64,
            "invalid_row_count": pl.Int64,
            "warn_row_count": pl.Int64,
            "min_price": pl.Float64,
            "max_price": pl.Float64,
            "avg_volume": pl.Float64,
            "source_file_last": pl.String,
            "schema_version": pl.String,
            "built_ts": pl.Datetime(time_zone="UTC"),
        },
    )

    quality_map, quality_file_count = load_latest_quality_reports(quality_reports_root(bronze_root), logger=logger)
    if quality_map:
        enrich_df = pl.DataFrame(
            [
                {"ticker": ticker, **values}
                for ticker, values in quality_map.items()
            ],
            schema_overrides={
                "ticker": pl.String,
                "duplicates_count": pl.Int64,
                "suspicious_bars_count": pl.Int64,
                "gap_rows_count": pl.Int64,
                "rows_invalid_reported": pl.Int64,
                "rows_total_reported": pl.Int64,
            },
        )
        symbol_master = symbol_master.join(enrich_df, on="ticker", how="left")
    else:
        symbol_master = symbol_master.with_columns(
            [
                pl.lit(0, dtype=pl.Int64).alias("duplicates_count"),
                pl.lit(0, dtype=pl.Int64).alias("suspicious_bars_count"),
                pl.lit(0, dtype=pl.Int64).alias("gap_rows_count"),
                pl.lit(0, dtype=pl.Int64).alias("rows_invalid_reported"),
                pl.lit(0, dtype=pl.Int64).alias("rows_total_reported"),
            ]
        )

    symbol_master = symbol_master.with_columns(
        [
            pl.col("duplicates_count").fill_null(0).cast(pl.Int64),
            pl.col("suspicious_bars_count").fill_null(0).cast(pl.Int64),
            pl.col("gap_rows_count").fill_null(0).cast(pl.Int64),
            pl.col("rows_invalid_reported").fill_null(0).cast(pl.Int64),
            pl.col("rows_total_reported").fill_null(0).cast(pl.Int64),
        ]
    ).sort(["exchange", "ticker"])

    return symbol_master, len(bronze_files), read_errors, quality_file_count, len(quality_map)


def write_symbol_master_artifacts(symbol_master: pl.DataFrame, bronze_root: Path) -> tuple[Path, Path]:
    """Write symbol-master parquet + csv outputs atomically."""

    parquet_path, csv_path = symbol_master_paths(bronze_root)
    return _write_df_atomically(symbol_master, parquet_path, csv_path)


def build_and_write_symbol_master(
    bronze_root: Path,
    *,
    logger: logging.Logger | None = None,
) -> SymbolMasterBuildResult:
    """Build symbol master from Bronze outputs and persist artifacts."""

    symbol_master, bronze_file_count, read_errors, quality_file_count, quality_latest_count = build_symbol_master_dataframe(
        bronze_root,
        logger=logger,
    )
    parquet_path, csv_path = write_symbol_master_artifacts(symbol_master, bronze_root)
    return SymbolMasterBuildResult(
        symbol_master_df=symbol_master,
        parquet_path=parquet_path,
        csv_path=csv_path,
        bronze_file_count=bronze_file_count,
        bronze_files_read_errors=read_errors,
        quality_reports_total_files=quality_file_count,
        quality_reports_latest_tickers=quality_latest_count,
    )
