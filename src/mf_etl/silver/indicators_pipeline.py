"""Batch/single orchestration for Silver indicator artifacts."""

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
from mf_etl.silver.indicators_twiggs import (
    INDICATOR_CALC_VERSION,
    INDICATOR_SCHEMA_VERSION,
    build_twiggs_indicator_frame,
)
from mf_etl.silver.indicators_writer import IndicatorWriteResult, write_indicator_parquet
from mf_etl.transform.dtypes import dtype_for_precision_name

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndicatorOneResult:
    """Result metadata for one symbol indicator build."""

    ticker: str
    exchange: str
    silver_file: Path
    indicator_path: Path
    rows_in: int
    rows_out: int
    min_trade_date: date | None
    max_trade_date: date | None
    tmf_null_count: int
    tti_proxy_null_count: int
    tmf_zero_cross_count: int
    tti_proxy_zero_cross_count: int


@dataclass(frozen=True, slots=True)
class IndicatorRunOptions:
    """Runtime options for indicator batch runs."""

    limit: int | None = None
    progress_every: int = 100
    full: bool = False


@dataclass(frozen=True, slots=True)
class IndicatorRunResult:
    """Run summary output paths for indicator batch processing."""

    run_id: str
    summary: dict[str, Any]
    summary_path: Path
    ticker_results_path: Path


@dataclass(frozen=True, slots=True)
class IndicatorSanityResult:
    """Result payload for indicator sanity scan."""

    summary: dict[str, Any]
    indicator_file_count: int
    read_errors: int
    summary_path: Path


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


def discover_silver_base_files(silver_root: Path) -> list[Path]:
    """Discover Silver base-series parquet files."""

    base = silver_root / "base_series_by_symbol"
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*.parquet") if path.is_file())


def discover_silver_base_inputs(silver_root: Path) -> pl.DataFrame:
    """Build symbol input frame from Silver base output paths."""

    rows: list[dict[str, object]] = []
    for path in discover_silver_base_files(silver_root):
        rows.append(
            {
                "ticker": (_partition_value(path, "ticker") or path.stem).strip().upper(),
                "exchange": (_partition_value(path, "exchange") or "UNKNOWN").strip().upper(),
                "silver_path": str(path),
            }
        )

    if not rows:
        return pl.DataFrame(schema={"ticker": pl.String, "exchange": pl.String, "silver_path": pl.String})
    return pl.DataFrame(rows, schema_overrides={"ticker": pl.String, "exchange": pl.String, "silver_path": pl.String})


def resolve_silver_base_file_for_ticker(silver_root: Path, ticker: str) -> Path:
    """Resolve one Silver base file for a ticker, with ambiguity guard."""

    normalized = ticker.strip().upper()
    if normalized == "":
        raise ValueError("Ticker must be non-empty.")

    inputs = discover_silver_base_inputs(silver_root)
    matched = inputs.filter(pl.col("ticker") == normalized)
    if matched.height == 0:
        raise FileNotFoundError(f"No Silver base parquet found for ticker {normalized}")
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
            "use --silver-file to disambiguate."
        )

    output = Path(str(matched.select("silver_path").to_dicts()[0]["silver_path"]))
    if not output.exists():
        raise FileNotFoundError(f"Silver file does not exist for ticker {normalized}: {output}")
    return output


def _indicator_float_dtype(settings: AppSettings) -> pl.DataType:
    precision_name = settings.indicators.float_dtype_override or settings.precision.silver_float
    return dtype_for_precision_name(precision_name)


def _frame_date_bounds(df: pl.DataFrame) -> tuple[date | None, date | None]:
    if df.height == 0 or "trade_date" not in df.columns:
        return None, None
    bounds = df.select(
        [
            pl.col("trade_date").min().alias("min_trade_date"),
            pl.col("trade_date").max().alias("max_trade_date"),
        ]
    ).to_dicts()[0]
    return bounds["min_trade_date"], bounds["max_trade_date"]


def _merge_bounds(
    current_min: date | None,
    current_max: date | None,
    next_min: date | None,
    next_max: date | None,
) -> tuple[date | None, date | None]:
    merged_min = current_min
    merged_max = current_max
    if next_min is not None:
        merged_min = next_min if merged_min is None else min(merged_min, next_min)
    if next_max is not None:
        merged_max = next_max if merged_max is None else max(merged_max, next_max)
    return merged_min, merged_max


def _iso_utc_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def run_indicators_one_from_silver_file(
    silver_file: Path,
    settings: AppSettings,
    *,
    run_id: str,
    logger: logging.Logger | None = None,
) -> IndicatorOneResult:
    """Read one Silver base file, compute indicators, and write output."""

    effective_logger = logger or LOGGER
    silver_df = pl.read_parquet(silver_file)
    rows_in = silver_df.height
    if rows_in == 0:
        raise ValueError(f"Silver base file is empty: {silver_file}")

    indicator_df = build_twiggs_indicator_frame(
        silver_df,
        tmf_period=settings.indicators.tmf_period,
        proxy_period=settings.indicators.proxy_period,
        eps=settings.indicators.eps,
        indicator_float_dtype=_indicator_float_dtype(settings),
        fallback_run_id=run_id,
    )

    write_result: IndicatorWriteResult = write_indicator_parquet(
        indicator_df,
        silver_root=settings.paths.silver_root,
        compression=settings.parquet.compression,
        compression_level=settings.parquet.compression_level,
        statistics=settings.parquet.statistics,
    )

    min_trade_date, max_trade_date = _frame_date_bounds(indicator_df)
    tmf_null_count = int(indicator_df.select(pl.col("tmf_21").is_null().sum()).item())
    tti_proxy_null_count = int(indicator_df.select(pl.col("tti_proxy_v1_21").is_null().sum()).item())
    tmf_zero_cross_count = int(
        indicator_df.select((pl.col("tmf_zero_cross_up").sum() + pl.col("tmf_zero_cross_down").sum()).alias("count")).item()
    )
    tti_proxy_zero_cross_count = int(
        indicator_df.select(
            (pl.col("tti_proxy_zero_cross_up").sum() + pl.col("tti_proxy_zero_cross_down").sum()).alias("count")
        ).item()
    )

    effective_logger.debug(
        "indicators_one.complete ticker=%s exchange=%s rows_in=%s rows_out=%s output=%s",
        write_result.ticker,
        write_result.exchange,
        rows_in,
        write_result.rows_out,
        write_result.output_path,
    )
    return IndicatorOneResult(
        ticker=write_result.ticker,
        exchange=write_result.exchange,
        silver_file=silver_file,
        indicator_path=write_result.output_path,
        rows_in=rows_in,
        rows_out=write_result.rows_out,
        min_trade_date=min_trade_date,
        max_trade_date=max_trade_date,
        tmf_null_count=tmf_null_count,
        tti_proxy_null_count=tti_proxy_null_count,
        tmf_zero_cross_count=tmf_zero_cross_count,
        tti_proxy_zero_cross_count=tti_proxy_zero_cross_count,
    )


def run_indicators_pipeline(
    settings: AppSettings,
    *,
    options: IndicatorRunOptions | None = None,
    logger: logging.Logger | None = None,
) -> IndicatorRunResult:
    """Run indicator calculation for all selected Silver base symbols."""

    effective_logger = logger or LOGGER
    run_options = options or IndicatorRunOptions()
    progress_every = max(1, run_options.progress_every)
    run_id = f"indicators-run-{uuid4().hex[:12]}"
    started_epoch = time.time()

    inputs = discover_silver_base_inputs(settings.paths.silver_root).sort("ticker")
    if run_options.limit is not None:
        inputs = inputs.head(max(0, run_options.limit))

    selected_total = inputs.height
    effective_logger.info(
        "indicators_run.start run_id=%s selected_symbols=%s full=%s limit=%s",
        run_id,
        selected_total,
        run_options.full,
        run_options.limit,
    )

    success_count = 0
    failed_count = 0
    rows_total = 0
    tmf_null_total = 0
    tti_proxy_null_total = 0
    global_min_trade_date: date | None = None
    global_max_trade_date: date | None = None
    failed_files: list[dict[str, str]] = []
    ticker_results_rows: list[dict[str, object]] = []

    for idx, row in enumerate(inputs.iter_rows(named=True), start=1):
        ticker = str(row["ticker"])
        exchange = str(row["exchange"])
        silver_file = Path(str(row["silver_path"]))
        try:
            result = run_indicators_one_from_silver_file(
                silver_file,
                settings,
                run_id=run_id,
                logger=effective_logger,
            )
            success_count += 1
            rows_total += result.rows_out
            tmf_null_total += result.tmf_null_count
            tti_proxy_null_total += result.tti_proxy_null_count
            global_min_trade_date, global_max_trade_date = _merge_bounds(
                global_min_trade_date,
                global_max_trade_date,
                result.min_trade_date,
                result.max_trade_date,
            )

            ticker_results_rows.append(
                {
                    "ticker": result.ticker,
                    "exchange": result.exchange,
                    "source_file": str(result.silver_file),
                    "rows_in": result.rows_in,
                    "rows_out": result.rows_out,
                    "indicator_path": str(result.indicator_path),
                    "tmf_null_count": result.tmf_null_count,
                    "tti_proxy_null_count": result.tti_proxy_null_count,
                    "success": True,
                    "error_message": None,
                }
            )
        except Exception as exc:
            failed_count += 1
            message = str(exc)
            failed_files.append({"source_file": str(silver_file), "error": message})
            ticker_results_rows.append(
                {
                    "ticker": ticker,
                    "exchange": exchange,
                    "source_file": str(silver_file),
                    "rows_in": 0,
                    "rows_out": 0,
                    "indicator_path": None,
                    "tmf_null_count": 0,
                    "tti_proxy_null_count": 0,
                    "success": False,
                    "error_message": message,
                }
            )
            effective_logger.exception("indicators_run.symbol_failed ticker=%s source_file=%s", ticker, silver_file)

        if idx % progress_every == 0 or idx == selected_total:
            elapsed = time.time() - started_epoch
            effective_logger.info(
                "indicators_run.progress processed=%s/%s success=%s failure=%s rows_total=%s elapsed_sec=%.2f",
                idx,
                selected_total,
                success_count,
                failed_count,
                rows_total,
                elapsed,
            )

    finished_epoch = time.time()
    duration_sec = finished_epoch - started_epoch
    tmf_null_rate = (tmf_null_total / rows_total) if rows_total > 0 else None
    tti_proxy_null_rate = (tti_proxy_null_total / rows_total) if rows_total > 0 else None

    artifacts_root = settings.paths.artifacts_root / "indicator_run_summaries"
    summary_path = artifacts_root / f"{run_id}_indicators_run_summary.json"
    ticker_results_path = artifacts_root / f"{run_id}_indicators_ticker_results.parquet"

    summary: dict[str, Any] = {
        "run_id": run_id,
        "started_ts": _iso_utc_from_epoch(started_epoch),
        "finished_ts": _iso_utc_from_epoch(finished_epoch),
        "duration_sec": round(duration_sec, 3),
        "indicator_schema_version": INDICATOR_SCHEMA_VERSION,
        "indicator_calc_version": INDICATOR_CALC_VERSION,
        "symbols_total_selected": selected_total,
        "symbols_success": success_count,
        "symbols_failed": failed_count,
        "rows_total": rows_total,
        "global_min_trade_date": global_min_trade_date.isoformat() if global_min_trade_date is not None else None,
        "global_max_trade_date": global_max_trade_date.isoformat() if global_max_trade_date is not None else None,
        "tmf_21_null_rate": tmf_null_rate,
        "tti_proxy_v1_21_null_rate": tti_proxy_null_rate,
        "failed_files": failed_files[:200],
        "output_root": str(settings.paths.silver_root / "indicators_by_symbol"),
        "outputs": {
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
                "indicator_path": pl.String,
                "tmf_null_count": pl.Int64,
                "tti_proxy_null_count": pl.Int64,
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
                "indicator_path": pl.String,
                "tmf_null_count": pl.Int64,
                "tti_proxy_null_count": pl.Int64,
                "success": pl.Boolean,
                "error_message": pl.String,
            }
        )
    )
    _write_parquet_atomically(results_df, ticker_results_path)

    return IndicatorRunResult(
        run_id=run_id,
        summary=summary,
        summary_path=summary_path,
        ticker_results_path=ticker_results_path,
    )


def run_indicators_sanity(
    silver_root: Path,
    artifacts_root: Path,
    *,
    logger: logging.Logger | None = None,
) -> IndicatorSanityResult:
    """Scan indicator parquet outputs and return compact QA summary."""

    effective_logger = logger or LOGGER
    indicator_root = silver_root / "indicators_by_symbol"
    files = sorted(path for path in indicator_root.rglob("*.parquet") if path.is_file()) if indicator_root.exists() else []

    tmf_null_count = 0
    tti_proxy_null_count = 0
    total_rows = 0
    symbol_count = 0
    read_errors = 0
    min_trade_date: date | None = None
    max_trade_date: date | None = None
    tmf_zero_cross_up_count = 0
    tmf_zero_cross_down_count = 0
    tti_proxy_zero_cross_up_count = 0
    tti_proxy_zero_cross_down_count = 0
    abs_tmf_by_symbol: list[dict[str, object]] = []

    for idx, path in enumerate(files, start=1):
        requested = [
            "ticker",
            "trade_date",
            "tmf_21",
            "tti_proxy_v1_21",
            "tmf_zero_cross_up",
            "tmf_zero_cross_down",
            "tti_proxy_zero_cross_up",
            "tti_proxy_zero_cross_down",
        ]
        try:
            frame = pl.read_parquet(path, columns=requested)
        except Exception as exc:
            read_errors += 1
            effective_logger.warning("indicators_sanity.read_failed path=%s error=%s", path, exc)
            continue

        rows = frame.height
        symbol_count += 1
        total_rows += rows

        bounds = frame.select(
            [
                pl.col("trade_date").min().alias("min_trade_date"),
                pl.col("trade_date").max().alias("max_trade_date"),
            ]
        ).to_dicts()[0]
        min_trade_date, max_trade_date = _merge_bounds(
            min_trade_date,
            max_trade_date,
            bounds["min_trade_date"],
            bounds["max_trade_date"],
        )

        tmf_null_count += int(frame.select(pl.col("tmf_21").is_null().sum()).item())
        tti_proxy_null_count += int(frame.select(pl.col("tti_proxy_v1_21").is_null().sum()).item())
        tmf_zero_cross_up_count += int(frame.select(pl.col("tmf_zero_cross_up").sum()).item())
        tmf_zero_cross_down_count += int(frame.select(pl.col("tmf_zero_cross_down").sum()).item())
        tti_proxy_zero_cross_up_count += int(frame.select(pl.col("tti_proxy_zero_cross_up").sum()).item())
        tti_proxy_zero_cross_down_count += int(frame.select(pl.col("tti_proxy_zero_cross_down").sum()).item())

        if rows > 0:
            ticker_value = str(frame.select(pl.col("ticker").first()).item())
            max_abs_tmf = frame.select(pl.col("tmf_21").abs().max()).item()
            abs_tmf_by_symbol.append(
                {
                    "ticker": ticker_value,
                    "max_abs_tmf_21": float(max_abs_tmf) if max_abs_tmf is not None else None,
                }
            )

        if idx % 1000 == 0:
            effective_logger.info("indicators_sanity.progress scanned=%s/%s", idx, len(files))

    top_abs = sorted(
        (row for row in abs_tmf_by_symbol if row["max_abs_tmf_21"] is not None),
        key=lambda row: float(row["max_abs_tmf_21"]),
        reverse=True,
    )[:20]
    summary: dict[str, Any] = {
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "symbol_count": symbol_count,
        "total_rows": total_rows,
        "global_min_trade_date": min_trade_date.isoformat() if min_trade_date is not None else None,
        "global_max_trade_date": max_trade_date.isoformat() if max_trade_date is not None else None,
        "tmf_21_null_rate": (tmf_null_count / total_rows) if total_rows > 0 else None,
        "tti_proxy_v1_21_null_rate": (tti_proxy_null_count / total_rows) if total_rows > 0 else None,
        "tmf_zero_cross_up_count": tmf_zero_cross_up_count,
        "tmf_zero_cross_down_count": tmf_zero_cross_down_count,
        "tti_proxy_zero_cross_up_count": tti_proxy_zero_cross_up_count,
        "tti_proxy_zero_cross_down_count": tti_proxy_zero_cross_down_count,
        "top_20_symbols_by_max_abs_tmf_21": top_abs,
    }

    summary_path = artifacts_root / "indicator_qa" / "indicator_sanity_summary.json"
    _write_json_atomically(summary, summary_path)
    return IndicatorSanityResult(
        summary=summary,
        indicator_file_count=len(files),
        read_errors=read_errors,
        summary_path=summary_path,
    )

