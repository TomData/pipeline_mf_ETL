"""Silver pipeline orchestration for base helper series."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from mf_etl.config import AppSettings
from mf_etl.silver.features_base import SILVER_CALC_VERSION, build_silver_base_features
from mf_etl.silver.writer import SilverWriteResult, write_silver_parquet
from mf_etl.transform.dtypes import SILVER_SCHEMA_VERSION, dtype_for_precision_name

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SilverOneResult:
    """Result of one Bronze->Silver conversion."""

    ticker: str
    exchange: str
    bronze_file: Path
    silver_path: Path
    rows_in: int
    rows_out: int
    min_trade_date: date | None
    max_trade_date: date | None


@dataclass(frozen=True, slots=True)
class SilverRunOptions:
    """Runtime options for Silver batch runs."""

    limit: int | None = None
    progress_every: int = 100
    full: bool = False


@dataclass(frozen=True, slots=True)
class SilverRunResult:
    """Summary artifact metadata for a Silver batch run."""

    run_id: str
    summary: dict[str, Any]
    summary_path: Path
    ticker_results_path: Path


@dataclass(frozen=True, slots=True)
class SilverSanityResult:
    """Result from Silver sanity scan."""

    summary: dict[str, Any]
    silver_file_count: int
    read_errors: int


def _atomic_temp_path(target_path: Path) -> Path:
    return target_path.parent / f".{target_path.name}.{uuid4().hex}.tmp"


def _write_json_atomically(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(output_path)
    try:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return output_path


def _write_parquet_atomically(df: pl.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(output_path)
    try:
        df.write_parquet(temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return output_path


def _partition_value(path: Path, key: str) -> str | None:
    prefix = f"{key}="
    for part in path.parts:
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def discover_bronze_symbol_files(bronze_root: Path) -> list[Path]:
    """Discover Bronze valid per-symbol parquet files."""

    base = bronze_root / "ohlcv_by_symbol"
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*.parquet") if path.is_file())


def _bronze_inputs_from_symbol_master(bronze_root: Path) -> pl.DataFrame | None:
    symbol_master_path = bronze_root / "symbol_master" / "symbol_master.parquet"
    if not symbol_master_path.exists():
        return None
    frame = pl.read_parquet(symbol_master_path)
    required = {"ticker", "exchange", "bronze_path"}
    if not required.issubset(set(frame.columns)):
        return None
    return frame.select(["ticker", "exchange", "bronze_path"]).unique(subset=["ticker", "exchange"], keep="last")


def discover_bronze_inputs(bronze_root: Path) -> pl.DataFrame:
    """Discover Bronze input file list for Silver runs."""

    from_symbol_master = _bronze_inputs_from_symbol_master(bronze_root)
    if from_symbol_master is not None:
        return from_symbol_master.sort("ticker")

    rows: list[dict[str, object]] = []
    for path in discover_bronze_symbol_files(bronze_root):
        rows.append(
            {
                "ticker": (_partition_value(path, "ticker") or path.stem).upper(),
                "exchange": (_partition_value(path, "exchange") or "UNKNOWN").upper(),
                "bronze_path": str(path),
            }
        )
    if not rows:
        return pl.DataFrame(schema={"ticker": pl.String, "exchange": pl.String, "bronze_path": pl.String})
    return pl.DataFrame(rows, schema_overrides={"ticker": pl.String, "exchange": pl.String, "bronze_path": pl.String})


def resolve_bronze_file_for_ticker(bronze_root: Path, ticker: str) -> Path:
    """Resolve Bronze file path for ticker via symbol master or direct scan."""

    normalized = ticker.strip().upper()
    if normalized == "":
        raise ValueError("Ticker must be non-empty.")

    inputs = discover_bronze_inputs(bronze_root)
    matched = inputs.filter(pl.col("ticker") == normalized)
    if matched.height == 0:
        raise FileNotFoundError(f"No Bronze parquet found for ticker {normalized}")
    if matched.height > 1:
        exchanges = sorted(
            {
                str(value)
                for value in matched.select("exchange").to_series().to_list()
                if value is not None and str(value).strip() != ""
            }
        )
        raise ValueError(
            f"Ticker {normalized} maps to multiple exchanges ({', '.join(exchanges)}); "
            "use --bronze-file to disambiguate."
        )
    bronze_path = Path(str(matched.select("bronze_path").to_dicts()[0]["bronze_path"]))
    if not bronze_path.exists():
        raise FileNotFoundError(f"Bronze path for ticker {normalized} does not exist: {bronze_path}")
    return bronze_path


def run_silver_one_from_bronze_file(
    bronze_file: Path,
    settings: AppSettings,
    *,
    run_id: str,
    logger: logging.Logger | None = None,
) -> SilverOneResult:
    """Read one Bronze per-symbol parquet, compute Silver base features, and write output."""

    effective_logger = logger or LOGGER
    bronze_df = pl.read_parquet(bronze_file)
    rows_in = bronze_df.height
    if rows_in == 0:
        raise ValueError(f"Bronze file is empty: {bronze_file}")

    silver_float_dtype = dtype_for_precision_name(settings.precision.silver_float)
    silver_df = build_silver_base_features(
        bronze_df,
        silver_float_dtype=silver_float_dtype,
        fallback_run_id=run_id,
    )

    write_result: SilverWriteResult = write_silver_parquet(
        silver_df,
        silver_root=settings.paths.silver_root,
        compression=settings.parquet.compression,
        compression_level=settings.parquet.compression_level,
        statistics=settings.parquet.statistics,
    )
    bounds = silver_df.select(
        [
            pl.col("trade_date").min().alias("min_trade_date"),
            pl.col("trade_date").max().alias("max_trade_date"),
        ]
    ).to_dicts()[0]
    effective_logger.debug(
        "silver_one.complete ticker=%s exchange=%s rows_in=%s rows_out=%s output=%s",
        write_result.ticker,
        write_result.exchange,
        rows_in,
        write_result.rows_out,
        write_result.output_path,
    )
    return SilverOneResult(
        ticker=write_result.ticker,
        exchange=write_result.exchange,
        bronze_file=bronze_file,
        silver_path=write_result.output_path,
        rows_in=rows_in,
        rows_out=write_result.rows_out,
        min_trade_date=bounds["min_trade_date"],
        max_trade_date=bounds["max_trade_date"],
    )


def run_silver_pipeline(
    settings: AppSettings,
    *,
    options: SilverRunOptions | None = None,
    logger: logging.Logger | None = None,
) -> SilverRunResult:
    """Run Silver base-feature generation for selected Bronze symbols."""

    effective_logger = logger or LOGGER
    run_options = options or SilverRunOptions()
    progress_every = max(1, run_options.progress_every)
    run_id = f"silver-run-{uuid4().hex[:12]}"
    started_ts = time.time()

    inputs = discover_bronze_inputs(settings.paths.bronze_root).sort("ticker")
    if run_options.limit is not None:
        inputs = inputs.head(max(0, run_options.limit))

    selected_total = inputs.height
    effective_logger.info(
        "silver_run.start run_id=%s selected_symbols=%s full=%s limit=%s",
        run_id,
        selected_total,
        run_options.full,
        run_options.limit,
    )

    success_count = 0
    failed_count = 0
    total_rows_in = 0
    total_rows_out = 0
    global_min_trade_date: date | None = None
    global_max_trade_date: date | None = None
    failed_files: list[dict[str, str]] = []
    ticker_results_rows: list[dict[str, object]] = []

    for idx, row in enumerate(inputs.iter_rows(named=True), start=1):
        ticker = str(row["ticker"])
        exchange = str(row["exchange"])
        bronze_file = Path(str(row["bronze_path"]))
        try:
            result = run_silver_one_from_bronze_file(
                bronze_file,
                settings,
                run_id=run_id,
                logger=effective_logger,
            )
            success_count += 1
            total_rows_in += result.rows_in
            total_rows_out += result.rows_out
            if result.min_trade_date is not None:
                global_min_trade_date = (
                    result.min_trade_date
                    if global_min_trade_date is None
                    else min(global_min_trade_date, result.min_trade_date)
                )
            if result.max_trade_date is not None:
                global_max_trade_date = (
                    result.max_trade_date
                    if global_max_trade_date is None
                    else max(global_max_trade_date, result.max_trade_date)
                )

            ticker_results_rows.append(
                {
                    "ticker": result.ticker,
                    "exchange": result.exchange,
                    "source_file": str(result.bronze_file),
                    "rows_in": result.rows_in,
                    "rows_out": result.rows_out,
                    "silver_path": str(result.silver_path),
                    "success": True,
                    "error_message": None,
                }
            )
        except Exception as exc:
            failed_count += 1
            message = str(exc)
            failed_files.append({"source_file": str(bronze_file), "error": message})
            ticker_results_rows.append(
                {
                    "ticker": ticker,
                    "exchange": exchange,
                    "source_file": str(bronze_file),
                    "rows_in": 0,
                    "rows_out": 0,
                    "silver_path": None,
                    "success": False,
                    "error_message": message,
                }
            )
            effective_logger.exception("silver_run.symbol_failed ticker=%s source_file=%s", ticker, bronze_file)

        if idx % progress_every == 0 or idx == selected_total:
            elapsed = time.time() - started_ts
            effective_logger.info(
                "silver_run.progress processed=%s/%s success=%s failure=%s rows_in=%s rows_out=%s elapsed_sec=%.2f",
                idx,
                selected_total,
                success_count,
                failed_count,
                total_rows_in,
                total_rows_out,
                elapsed,
            )

    finished_ts = time.time()
    duration_sec = finished_ts - started_ts
    artifacts_dir = settings.paths.artifacts_root / "silver_run_summaries"
    summary_path = artifacts_dir / f"{run_id}_silver_run_summary.json"
    ticker_results_path = artifacts_dir / f"{run_id}_silver_ticker_results.parquet"

    summary: dict[str, Any] = {
        "run_id": run_id,
        "started_ts": datetime_from_epoch(started_ts),
        "finished_ts": datetime_from_epoch(finished_ts),
        "duration_sec": round(duration_sec, 3),
        "silver_schema_version": SILVER_SCHEMA_VERSION,
        "silver_calc_version": SILVER_CALC_VERSION,
        "symbols_selected_total": selected_total,
        "symbols_processed_success": success_count,
        "symbols_processed_failed": failed_count,
        "rows_in_total": total_rows_in,
        "rows_out_total": total_rows_out,
        "global_min_trade_date": global_min_trade_date.isoformat() if global_min_trade_date is not None else None,
        "global_max_trade_date": global_max_trade_date.isoformat() if global_max_trade_date is not None else None,
        "failed_files": failed_files[:200],
        "outputs": {
            "silver_root": str(settings.paths.silver_root),
            "summary_path": str(summary_path),
            "ticker_results_path": str(ticker_results_path),
        },
    }
    _write_json_atomically(summary, summary_path)

    results_df = (
        pl.DataFrame(
            ticker_results_rows,
            schema_overrides={
                "ticker": pl.String,
                "exchange": pl.String,
                "source_file": pl.String,
                "rows_in": pl.Int64,
                "rows_out": pl.Int64,
                "silver_path": pl.String,
                "success": pl.Boolean,
                "error_message": pl.String,
            },
        )
        if ticker_results_rows
        else pl.DataFrame(
            schema={
                "ticker": pl.String,
                "exchange": pl.String,
                "source_file": pl.String,
                "rows_in": pl.Int64,
                "rows_out": pl.Int64,
                "silver_path": pl.String,
                "success": pl.Boolean,
                "error_message": pl.String,
            }
        )
    )
    _write_parquet_atomically(results_df, ticker_results_path)

    return SilverRunResult(
        run_id=run_id,
        summary=summary,
        summary_path=summary_path,
        ticker_results_path=ticker_results_path,
    )


def datetime_from_epoch(value: float) -> str:
    """Return ISO timestamp for POSIX epoch seconds in UTC."""

    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def run_silver_sanity(
    silver_root: Path,
    *,
    logger: logging.Logger | None = None,
) -> SilverSanityResult:
    """Scan Silver outputs and produce compact sanity metrics."""

    effective_logger = logger or LOGGER
    base = silver_root / "base_series_by_symbol"
    files = sorted(path for path in base.rglob("*.parquet") if path.is_file()) if base.exists() else []
    if not files:
        return SilverSanityResult(
            summary={
                "ticker_count": 0,
                "total_rows": 0,
                "global_min_trade_date": None,
                "global_max_trade_date": None,
                "feature_columns_present": [],
                "key_feature_null_rates": {},
            },
            silver_file_count=0,
            read_errors=0,
        )

    key_features = ("ret_1d", "atr_14", "vol_ratio_20", "close_sma_50")
    total_rows = 0
    ticker_count = 0
    min_date: date | None = None
    max_date: date | None = None
    read_errors = 0
    null_counts = {feature: 0 for feature in key_features}
    feature_columns_present: set[str] = set()

    for idx, path in enumerate(files, start=1):
        try:
            schema = pl.read_parquet_schema(path)
        except Exception as exc:
            read_errors += 1
            effective_logger.warning("silver_sanity.schema_read_failed path=%s error=%s", path, exc)
            continue
        feature_columns_present.update(schema.keys())

        requested_cols = ["trade_date", "ticker", *[col for col in key_features if col in schema]]
        try:
            frame = pl.read_parquet(path, columns=requested_cols)
        except Exception as exc:
            read_errors += 1
            effective_logger.warning("silver_sanity.file_read_failed path=%s error=%s", path, exc)
            continue

        rows = frame.height
        total_rows += rows
        ticker_count += 1
        if rows > 0 and "trade_date" in frame.columns:
            bounds = frame.select(
                [
                    pl.col("trade_date").min().alias("min_date"),
                    pl.col("trade_date").max().alias("max_date"),
                ]
            ).to_dicts()[0]
            local_min = bounds["min_date"]
            local_max = bounds["max_date"]
            if local_min is not None:
                min_date = local_min if min_date is None else min(min_date, local_min)
            if local_max is not None:
                max_date = local_max if max_date is None else max(max_date, local_max)

        for feature in key_features:
            if feature in frame.columns:
                null_counts[feature] += int(frame.select(pl.col(feature).is_null().sum()).item())
            else:
                null_counts[feature] += rows

        if idx % 1000 == 0:
            effective_logger.info("silver_sanity.progress scanned=%s/%s", idx, len(files))

    null_rates = {
        feature: (null_counts[feature] / total_rows if total_rows > 0 else None)
        for feature in key_features
    }
    summary = {
        "ticker_count": ticker_count,
        "total_rows": total_rows,
        "global_min_trade_date": min_date.isoformat() if min_date is not None else None,
        "global_max_trade_date": max_date.isoformat() if max_date is not None else None,
        "feature_columns_present": sorted(feature_columns_present),
        "key_feature_null_rates": null_rates,
    }
    return SilverSanityResult(summary=summary, silver_file_count=len(files), read_errors=read_errors)
