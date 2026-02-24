"""Batch Bronze pipeline orchestration for incremental processing."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from mf_etl.bronze.writer import write_bronze_artifacts
from mf_etl.config import AppSettings
from mf_etl.ingest.discover import discover_txt_files, extract_ticker_hint
from mf_etl.ingest.manifest import (
    ManifestStatus,
    build_manifest,
    classify_manifest,
    exchange_counts,
    get_manifest_paths,
    load_manifest_parquet,
    manifest_status_counts,
    persist_classified_current_manifest,
    promote_current_manifest_to_stable,
)
from mf_etl.ingest.read_txt import read_stock_txt_with_rejects
from mf_etl.transform.normalize import BronzeNormalizeMetadata, normalize_bronze_rows
from mf_etl.utils.time_utils import now_utc
from mf_etl.validate.reports import validate_bronze_dataframe
from mf_etl.validate.rules import ValidationThresholds

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BronzeRunOptions:
    """Runtime options for Bronze batch orchestration."""

    full: bool = False
    limit: int | None = None
    dry_run: bool = False
    progress_every: int = 100
    only_status: ManifestStatus | None = None


@dataclass(frozen=True, slots=True)
class BronzeRunResult:
    """Return object for Bronze run outcomes."""

    run_id: str
    summary: dict[str, Any]
    summary_path: Path
    manifest_current_path: Path
    manifest_stable_path: Path
    failed_files_path: Path | None
    ticker_results_path: Path | None


def _atomic_temp_path(target_path: Path) -> Path:
    """Create a temp path in the target directory for atomic replacement."""

    return target_path.parent / f".{target_path.name}.{uuid4().hex}.tmp"


def _write_json_atomically(payload: dict[str, Any], output_path: Path) -> Path:
    """Write JSON payload atomically to output path."""

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
    """Write parquet atomically to output path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(output_path)
    try:
        df.write_parquet(temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return output_path


def _select_files_for_processing(
    classified_manifest: pl.DataFrame,
    options: BronzeRunOptions,
) -> pl.DataFrame:
    """Select files according to status/full/limit options."""

    selected = classified_manifest.sort("source_file")
    if options.only_status:
        selected = selected.filter(pl.col("manifest_status") == options.only_status)
    elif not options.full:
        selected = selected.filter(pl.col("manifest_status").is_in(["NEW", "CHANGED"]))

    if options.limit is not None:
        selected = selected.head(max(0, options.limit))
    return selected


def _safe_trade_date_bounds(df: pl.DataFrame) -> tuple[date | None, date | None]:
    """Return min/max trade_date values from frame."""

    if df.height == 0 or "trade_date" not in df.columns:
        return None, None
    bounds = df.select(
        [
            pl.col("trade_date").min().alias("min_trade_date"),
            pl.col("trade_date").max().alias("max_trade_date"),
        ]
    ).to_dicts()[0]
    return bounds["min_trade_date"], bounds["max_trade_date"]


def _merge_date_bounds(
    current_min: date | None,
    current_max: date | None,
    next_min: date | None,
    next_max: date | None,
) -> tuple[date | None, date | None]:
    """Merge pairwise date bounds with null-safe handling."""

    merged_min = current_min
    merged_max = current_max
    if next_min is not None:
        merged_min = next_min if merged_min is None else min(merged_min, next_min)
    if next_max is not None:
        merged_max = next_max if merged_max is None else max(merged_max, next_max)
    return merged_min, merged_max


def _should_promote_manifest(options: BronzeRunOptions) -> bool:
    """Decide whether the stable manifest should be promoted for this run mode."""

    # We intentionally do not promote on partial test runs (`limit`) or filtered runs
    # (`only_status`) to avoid marking unprocessed files as up-to-date.
    return not options.dry_run and options.limit is None and options.only_status is None


def _empty_ticker_results_df() -> pl.DataFrame:
    """Create empty ticker-results frame with stable schema."""

    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "exchange": pl.String,
            "source_file": pl.String,
            "rows_total": pl.Int64,
            "rows_valid": pl.Int64,
            "rows_invalid": pl.Int64,
            "bronze_path": pl.String,
            "quality_report_path": pl.String,
            "success": pl.Boolean,
            "error_message": pl.String,
        }
    )


def run_bronze_pipeline(
    settings: AppSettings,
    *,
    options: BronzeRunOptions | None = None,
    logger: logging.Logger | None = None,
) -> BronzeRunResult:
    """Run incremental Bronze pipeline for all discovered files."""

    effective_logger = logger or LOGGER
    run_options = options or BronzeRunOptions()
    progress_every = max(1, run_options.progress_every)

    run_id = f"bronze-run-{uuid4().hex[:12]}"
    started_ts = now_utc()
    started_mono = time.monotonic()

    discovered = discover_txt_files(settings.paths.raw_root, logger=effective_logger)
    current_manifest = build_manifest(discovered, discovered_ts=started_ts, logger=effective_logger)

    manifest_paths = get_manifest_paths(settings.paths.bronze_root)
    previous_manifest = load_manifest_parquet(manifest_paths.stable_path)
    classified_manifest = classify_manifest(current_manifest, previous_manifest, logger=effective_logger)

    manifest_current_path = persist_classified_current_manifest(
        classified_manifest=classified_manifest,
        bronze_root=settings.paths.bronze_root,
        compression=settings.parquet.compression,
        compression_level=settings.parquet.compression_level,
        statistics=settings.parquet.statistics,
    )

    status_counts = manifest_status_counts(classified_manifest)
    exchange_count_map = exchange_counts(classified_manifest)
    selected_manifest = _select_files_for_processing(classified_manifest, run_options)

    files_discovered_total = classified_manifest.height
    files_selected_total = selected_manifest.height
    unchanged_selected = int(selected_manifest.filter(pl.col("manifest_status") == "UNCHANGED").height)
    files_skipped_unchanged = max(0, status_counts["UNCHANGED"] - unchanged_selected)

    effective_logger.info(
        "bronze_run.start run_id=%s discovered=%s selected=%s dry_run=%s full=%s limit=%s only_status=%s",
        run_id,
        files_discovered_total,
        files_selected_total,
        run_options.dry_run,
        run_options.full,
        run_options.limit,
        run_options.only_status,
    )
    effective_logger.info("bronze_run.status_counts %s", status_counts)
    effective_logger.info("bronze_run.exchange_counts %s", exchange_count_map)

    thresholds = ValidationThresholds(
        suspicious_range_pct_threshold=settings.validation.suspicious_range_pct_threshold,
        suspicious_return_pct_threshold=settings.validation.suspicious_return_pct_threshold,
        gap_days_warn_threshold=settings.validation.gap_days_warn_threshold,
    )

    success_count = 0
    failed_count = 0
    rows_total = 0
    rows_valid = 0
    rows_invalid = 0
    tickers_processed: set[str] = set()
    tickers_with_problems: set[str] = set()
    failed_files: list[dict[str, str]] = []
    ticker_results_rows: list[dict[str, object]] = []
    global_min_trade_date: date | None = None
    global_max_trade_date: date | None = None

    if not run_options.dry_run:
        for processed_idx, row in enumerate(selected_manifest.iter_rows(named=True), start=1):
            source_file = Path(str(row["source_file"]))
            source_exchange = str(row.get("exchange") or "UNKNOWN").strip().upper() or "UNKNOWN"
            fallback_ticker = str(row.get("ticker_hint") or extract_ticker_hint(source_file)).strip().upper()
            try:
                raw_result = read_stock_txt_with_rejects(source_file, logger=effective_logger)
                metadata = BronzeNormalizeMetadata.build(
                    source_file=source_file,
                    exchange=source_exchange,
                    run_id=run_id,
                )
                normalized = normalize_bronze_rows(raw_result.data, metadata=metadata)
                validation_result = validate_bronze_dataframe(
                    normalized,
                    thresholds=thresholds,
                    header_skipped=raw_result.skipped_header,
                    malformed_raw_rows_count=raw_result.rejects.height,
                )

                write_result = write_bronze_artifacts(
                    bronze_root=settings.paths.bronze_root,
                    validation_result=validation_result,
                    quality_report=validation_result.quality_report,
                    fallback_ticker=fallback_ticker,
                    fallback_exchange=source_exchange,
                    fallback_run_id=run_id,
                    compression=settings.parquet.compression,
                    compression_level=settings.parquet.compression_level,
                    statistics=settings.parquet.statistics,
                    malformed_rows=raw_result.rejects if raw_result.rejects.height > 0 else None,
                )

                success_count += 1
                rows_total += write_result.rows_total
                rows_valid += write_result.rows_valid
                rows_invalid += write_result.rows_invalid
                tickers_processed.add(write_result.ticker)
                if write_result.rows_invalid > 0 or raw_result.rejects.height > 0:
                    tickers_with_problems.add(write_result.ticker)

                min_trade_date, max_trade_date = _safe_trade_date_bounds(validation_result.validated_df)
                global_min_trade_date, global_max_trade_date = _merge_date_bounds(
                    global_min_trade_date,
                    global_max_trade_date,
                    min_trade_date,
                    max_trade_date,
                )

                ticker_results_rows.append(
                    {
                        "ticker": write_result.ticker,
                        "exchange": write_result.exchange,
                        "source_file": str(source_file),
                        "rows_total": write_result.rows_total,
                        "rows_valid": write_result.rows_valid,
                        "rows_invalid": write_result.rows_invalid,
                        "bronze_path": str(write_result.bronze_path),
                        "quality_report_path": str(write_result.quality_report_path),
                        "success": True,
                        "error_message": None,
                    }
                )
            except Exception as exc:
                failed_count += 1
                error_message = str(exc)
                failed_files.append({"source_file": str(source_file), "error": error_message})
                tickers_with_problems.add(fallback_ticker or "UNKNOWN")
                ticker_results_rows.append(
                    {
                        "ticker": fallback_ticker or "UNKNOWN",
                        "exchange": source_exchange,
                        "source_file": str(source_file),
                        "rows_total": 0,
                        "rows_valid": 0,
                        "rows_invalid": 0,
                        "bronze_path": None,
                        "quality_report_path": None,
                        "success": False,
                        "error_message": error_message,
                    }
                )
                effective_logger.exception("bronze_run.file_failed source_file=%s", source_file)

            if processed_idx % progress_every == 0 or processed_idx == files_selected_total:
                elapsed = time.monotonic() - started_mono
                effective_logger.info(
                    "bronze_run.progress processed=%s/%s success=%s failure=%s rows_total=%s rows_valid=%s rows_invalid=%s elapsed_sec=%.2f",
                    processed_idx,
                    files_selected_total,
                    success_count,
                    failed_count,
                    rows_total,
                    rows_valid,
                    rows_invalid,
                    elapsed,
                )

    manifest_stable_path = manifest_paths.stable_path
    promoted_manifest = False
    if _should_promote_manifest(run_options):
        # We intentionally promote even if some files failed: unchanged fingerprints
        # remain skipped next run, while failures are tracked in summary artifacts.
        promote_current_manifest_to_stable(
            classified_manifest=classified_manifest,
            bronze_root=settings.paths.bronze_root,
            compression=settings.parquet.compression,
            compression_level=settings.parquet.compression_level,
            statistics=settings.parquet.statistics,
        )
        promoted_manifest = True
    else:
        effective_logger.info(
            "bronze_run.manifest_promotion_skipped reason=partial_or_dry_run dry_run=%s limit=%s only_status=%s",
            run_options.dry_run,
            run_options.limit,
            run_options.only_status,
        )

    finished_ts = now_utc()
    duration_sec = time.monotonic() - started_mono

    artifacts_dir = settings.paths.artifacts_root / "run_summaries"
    summary_path = artifacts_dir / f"{run_id}_bronze_run_summary.json"
    failed_files_path = artifacts_dir / f"{run_id}_failed_files.json"
    ticker_results_path = artifacts_dir / f"{run_id}_ticker_results.parquet"

    failed_files_for_summary = failed_files[:200]
    tickers_with_problems_for_summary = sorted(tickers_with_problems)[:200]

    summary: dict[str, Any] = {
        "run_id": run_id,
        "started_ts": started_ts.isoformat(),
        "finished_ts": finished_ts.isoformat(),
        "duration_sec": round(duration_sec, 3),
        "raw_root": str(settings.paths.raw_root),
        "files_discovered_total": files_discovered_total,
        "files_selected_total": files_selected_total,
        "files_processed_success": success_count,
        "files_processed_failed": failed_count,
        "files_skipped_unchanged": files_skipped_unchanged,
        "status_counts": status_counts,
        "exchange_counts": exchange_count_map,
        "tickers_processed_count": len(tickers_processed),
        "rows_total": rows_total,
        "rows_valid": rows_valid,
        "rows_invalid": rows_invalid,
        "tickers_with_problems": tickers_with_problems_for_summary,
        "failed_files": failed_files_for_summary,
        "global_min_trade_date": global_min_trade_date.isoformat() if global_min_trade_date else None,
        "global_max_trade_date": global_max_trade_date.isoformat() if global_max_trade_date else None,
        "outputs": {
            "manifest_current_path": str(manifest_current_path),
            "manifest_stable_path": str(manifest_stable_path),
            "manifest_stable_promoted": promoted_manifest,
            "bronze_root": str(settings.paths.bronze_root),
            "quality_reports_root": str(settings.paths.bronze_root / "quality" / "ticker_reports"),
        },
    }
    _write_json_atomically(summary, summary_path)

    _write_json_atomically({"run_id": run_id, "failed_files": failed_files}, failed_files_path)

    ticker_results_df = (
        pl.DataFrame(
            ticker_results_rows,
            schema_overrides={
                "ticker": pl.String,
                "exchange": pl.String,
                "source_file": pl.String,
                "rows_total": pl.Int64,
                "rows_valid": pl.Int64,
                "rows_invalid": pl.Int64,
                "bronze_path": pl.String,
                "quality_report_path": pl.String,
                "success": pl.Boolean,
                "error_message": pl.String,
            },
        )
        if ticker_results_rows
        else _empty_ticker_results_df()
    )
    _write_parquet_atomically(ticker_results_df, ticker_results_path)

    effective_logger.info(
        "bronze_run.complete run_id=%s success=%s failed=%s selected=%s discovered=%s summary_path=%s",
        run_id,
        success_count,
        failed_count,
        files_selected_total,
        files_discovered_total,
        summary_path,
    )

    return BronzeRunResult(
        run_id=run_id,
        summary=summary,
        summary_path=summary_path,
        manifest_current_path=manifest_current_path,
        manifest_stable_path=manifest_stable_path,
        failed_files_path=failed_files_path,
        ticker_results_path=ticker_results_path,
    )
