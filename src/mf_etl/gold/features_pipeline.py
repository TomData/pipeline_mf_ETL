"""Gold Features v1 pipeline, sanity checks, and dataset export helpers."""

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
from mf_etl.gold.features_v1 import FEATURE_CALC_VERSION, FEATURE_SCHEMA_VERSION, build_gold_features_v1
from mf_etl.gold.features_writer import GoldFeatureWriteResult, write_gold_feature_parquet
from mf_etl.transform.dtypes import dtype_for_precision_name

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GoldFeatureOneResult:
    """Result metadata for one events->features conversion."""

    ticker: str
    exchange: str
    events_file: Path
    output_path: Path
    rows_in: int
    rows_out: int
    min_trade_date: date | None
    max_trade_date: date | None


@dataclass(frozen=True, slots=True)
class GoldFeatureRunOptions:
    """Runtime options for Gold feature batch runs."""

    limit: int | None = None
    progress_every: int = 100
    full: bool = False


@dataclass(frozen=True, slots=True)
class GoldFeatureRunResult:
    """Batch run summary outputs for Gold Features v1."""

    run_id: str
    summary: dict[str, Any]
    summary_path: Path
    ticker_results_path: Path


@dataclass(frozen=True, slots=True)
class GoldFeatureSanityResult:
    """Sanity scan result payload for Gold Features outputs."""

    summary: dict[str, Any]
    feature_file_count: int
    read_errors: int
    summary_path: Path


@dataclass(frozen=True, slots=True)
class GoldMLDatasetExportResult:
    """Output metadata for stacked ML dataset export."""

    run_id: str
    dataset_path: Path
    metadata_path: Path
    row_count: int
    symbol_count: int


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


def _write_parquet_atomically(
    df: pl.DataFrame,
    output_path: Path,
    *,
    compression: str,
    compression_level: int | None,
    statistics: bool,
) -> Path:
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


def _partition_value(path: Path, key: str) -> str | None:
    prefix = f"{key}="
    for part in path.parts:
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _iso_utc_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


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


def discover_event_files(gold_root: Path) -> list[Path]:
    """Discover Gold event parquet files."""

    base = gold_root / "events_by_symbol"
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*.parquet") if path.is_file())


def discover_event_inputs(gold_root: Path) -> pl.DataFrame:
    """Build ticker/exchange/path frame from Gold event files."""

    rows: list[dict[str, object]] = []
    for path in discover_event_files(gold_root):
        rows.append(
            {
                "ticker": (_partition_value(path, "ticker") or path.stem).strip().upper(),
                "exchange": (_partition_value(path, "exchange") or "UNKNOWN").strip().upper(),
                "events_path": str(path),
            }
        )

    if not rows:
        return pl.DataFrame(schema={"ticker": pl.String, "exchange": pl.String, "events_path": pl.String})
    return pl.DataFrame(rows, schema_overrides={"ticker": pl.String, "exchange": pl.String, "events_path": pl.String})


def resolve_events_file_for_ticker(gold_root: Path, ticker: str) -> Path:
    """Resolve one Gold event file for a ticker with ambiguity guard."""

    normalized = ticker.strip().upper()
    if normalized == "":
        raise ValueError("Ticker must be non-empty.")

    inputs = discover_event_inputs(gold_root)
    matched = inputs.filter(pl.col("ticker") == normalized)
    if matched.height == 0:
        raise FileNotFoundError(f"No Gold event parquet found for ticker {normalized}")
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
            "use --events-file to disambiguate."
        )

    output = Path(str(matched.select("events_path").to_dicts()[0]["events_path"]))
    if not output.exists():
        raise FileNotFoundError(f"Events file does not exist for ticker {normalized}: {output}")
    return output


def discover_feature_files(gold_root: Path) -> list[Path]:
    """Discover Gold feature parquet files."""

    base = gold_root / "features_by_symbol"
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*.parquet") if path.is_file())


def _feature_float_dtype(settings: AppSettings) -> pl.DataType:
    precision_name = settings.gold_features.float_dtype_override or settings.precision.gold_float
    return dtype_for_precision_name(precision_name)


def run_features_one_from_events_file(
    events_file: Path,
    settings: AppSettings,
    *,
    run_id: str,
    logger: logging.Logger | None = None,
) -> GoldFeatureOneResult:
    """Read one Gold event file, compute features, and write output."""

    effective_logger = logger or LOGGER
    events_df = pl.read_parquet(events_file)
    rows_in = events_df.height
    if rows_in == 0:
        raise ValueError(f"Events file is empty: {events_file}")

    features_df = build_gold_features_v1(
        events_df,
        activity_windows=settings.gold_features.activity_windows,
        zero_weight=settings.gold_features.score_weights.zero,
        respect_weight=settings.gold_features.score_weights.respect,
        burst_weight=settings.gold_features.score_weights.burst,
        hold_weight=settings.gold_features.score_weights.hold,
        recency_clip_bars=settings.gold_features.recency_clip_bars,
        eps=settings.gold_features.eps,
        float_dtype=_feature_float_dtype(settings),
        fallback_run_id=run_id,
    )
    write_result: GoldFeatureWriteResult = write_gold_feature_parquet(
        features_df,
        gold_root=settings.paths.gold_root,
        compression=settings.parquet.compression,
        compression_level=settings.parquet.compression_level,
        statistics=settings.parquet.statistics,
    )

    min_trade_date, max_trade_date = _frame_date_bounds(features_df)
    effective_logger.debug(
        "features_one.complete ticker=%s exchange=%s rows_in=%s rows_out=%s output=%s",
        write_result.ticker,
        write_result.exchange,
        rows_in,
        write_result.rows_out,
        write_result.output_path,
    )
    return GoldFeatureOneResult(
        ticker=write_result.ticker,
        exchange=write_result.exchange,
        events_file=events_file,
        output_path=write_result.output_path,
        rows_in=rows_in,
        rows_out=write_result.rows_out,
        min_trade_date=min_trade_date,
        max_trade_date=max_trade_date,
    )


def run_features_pipeline(
    settings: AppSettings,
    *,
    options: GoldFeatureRunOptions | None = None,
    logger: logging.Logger | None = None,
) -> GoldFeatureRunResult:
    """Run Gold Features v1 over all selected Gold event files."""

    effective_logger = logger or LOGGER
    run_options = options or GoldFeatureRunOptions()
    progress_every = max(1, run_options.progress_every)
    run_id = f"features-run-{uuid4().hex[:12]}"
    started_epoch = time.time()

    inputs = discover_event_inputs(settings.paths.gold_root).sort("ticker")
    if run_options.limit is not None:
        inputs = inputs.head(max(0, run_options.limit))

    selected_total = inputs.height
    effective_logger.info(
        "features_run.start run_id=%s selected_symbols=%s full=%s limit=%s",
        run_id,
        selected_total,
        run_options.full,
        run_options.limit,
    )

    success_count = 0
    failed_count = 0
    rows_total = 0
    global_min_trade_date: date | None = None
    global_max_trade_date: date | None = None
    failed_files: list[dict[str, str]] = []
    ticker_results_rows: list[dict[str, object]] = []

    for idx, row in enumerate(inputs.iter_rows(named=True), start=1):
        ticker = str(row["ticker"])
        exchange = str(row["exchange"])
        events_file = Path(str(row["events_path"]))
        try:
            result = run_features_one_from_events_file(
                events_file,
                settings,
                run_id=run_id,
                logger=effective_logger,
            )
            success_count += 1
            rows_total += result.rows_out
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
                    "source_file": str(result.events_file),
                    "rows_in": result.rows_in,
                    "rows_out": result.rows_out,
                    "output_path": str(result.output_path),
                    "success": True,
                    "error_message": None,
                }
            )
        except Exception as exc:
            failed_count += 1
            message = str(exc)
            failed_files.append({"source_file": str(events_file), "error": message})
            ticker_results_rows.append(
                {
                    "ticker": ticker,
                    "exchange": exchange,
                    "source_file": str(events_file),
                    "rows_in": 0,
                    "rows_out": 0,
                    "output_path": None,
                    "success": False,
                    "error_message": message,
                }
            )
            effective_logger.exception("features_run.symbol_failed ticker=%s source_file=%s", ticker, events_file)

        if idx % progress_every == 0 or idx == selected_total:
            elapsed = time.time() - started_epoch
            effective_logger.info(
                "features_run.progress processed=%s/%s success=%s failure=%s rows_total=%s elapsed_sec=%.2f",
                idx,
                selected_total,
                success_count,
                failed_count,
                rows_total,
                elapsed,
            )

    finished_epoch = time.time()
    duration_sec = finished_epoch - started_epoch
    artifacts_root = settings.paths.artifacts_root / "gold_feature_run_summaries"
    summary_path = artifacts_root / f"{run_id}_features_run_summary.json"
    ticker_results_path = artifacts_root / f"{run_id}_features_ticker_results.parquet"

    summary: dict[str, Any] = {
        "run_id": run_id,
        "started_ts": _iso_utc_from_epoch(started_epoch),
        "finished_ts": _iso_utc_from_epoch(finished_epoch),
        "duration_sec": round(duration_sec, 3),
        "symbols_total_selected": selected_total,
        "symbols_success": success_count,
        "symbols_failed": failed_count,
        "rows_total": rows_total,
        "global_min_trade_date": global_min_trade_date.isoformat() if global_min_trade_date is not None else None,
        "global_max_trade_date": global_max_trade_date.isoformat() if global_max_trade_date is not None else None,
        "feature_calc_version": FEATURE_CALC_VERSION,
        "output_root": str(settings.paths.gold_root / "features_by_symbol"),
        "failed_files": failed_files[:200],
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
                "output_path": pl.String,
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
                "output_path": pl.String,
                "success": pl.Boolean,
                "error_message": pl.String,
            }
        )
    )
    _write_parquet_atomically(
        results_df,
        ticker_results_path,
        compression=settings.parquet.compression,
        compression_level=settings.parquet.compression_level,
        statistics=settings.parquet.statistics,
    )

    return GoldFeatureRunResult(
        run_id=run_id,
        summary=summary,
        summary_path=summary_path,
        ticker_results_path=ticker_results_path,
    )


def run_features_sanity(
    gold_root: Path,
    artifacts_root: Path,
    *,
    logger: logging.Logger | None = None,
) -> GoldFeatureSanityResult:
    """Scan Gold feature outputs and compute compact QA metrics."""

    effective_logger = logger or LOGGER
    files = discover_feature_files(gold_root)
    key_features = [
        "tmf_21",
        "long_flow_score_20",
        "delta_flow_20",
        "flow_bias_20",
        "oscillation_index_20",
        "state_run_length",
    ]

    total_rows = 0
    symbol_count = 0
    read_errors = 0
    global_min_trade_date: date | None = None
    global_max_trade_date: date | None = None
    null_counts = {column: 0 for column in key_features}
    top_avg_flow_activity: list[dict[str, object]] = []
    top_max_abs_delta: list[dict[str, object]] = []
    top_avg_oscillation: list[dict[str, object]] = []

    for idx, path in enumerate(files, start=1):
        requested = list(dict.fromkeys(["ticker", "trade_date", "flow_activity_20", "delta_flow_20", "oscillation_index_20", *key_features]))
        try:
            schema = pl.read_parquet_schema(path)
            columns = [column for column in requested if column in schema]
            frame = pl.read_parquet(path, columns=columns)
        except Exception as exc:
            read_errors += 1
            effective_logger.warning("features_sanity.read_failed path=%s error=%s", path, exc)
            continue

        rows = frame.height
        symbol_count += 1
        total_rows += rows
        if rows > 0 and "trade_date" in frame.columns:
            bounds = frame.select(
                [
                    pl.col("trade_date").min().alias("min_trade_date"),
                    pl.col("trade_date").max().alias("max_trade_date"),
                ]
            ).to_dicts()[0]
            global_min_trade_date, global_max_trade_date = _merge_bounds(
                global_min_trade_date,
                global_max_trade_date,
                bounds["min_trade_date"],
                bounds["max_trade_date"],
            )

        for column in key_features:
            if column in frame.columns:
                null_counts[column] += int(frame.select(pl.col(column).is_null().sum()).item())
            else:
                null_counts[column] += rows

        ticker = str(frame.select(pl.col("ticker").first()).item()) if rows > 0 and "ticker" in frame.columns else "UNKNOWN"
        top_avg_flow_activity.append(
            {
                "ticker": ticker,
                "avg_flow_activity_20": float(frame.select(pl.col("flow_activity_20").mean()).item()) if "flow_activity_20" in frame.columns else None,
            }
        )
        top_max_abs_delta.append(
            {
                "ticker": ticker,
                "max_abs_delta_flow_20": float(frame.select(pl.col("delta_flow_20").abs().max()).item()) if "delta_flow_20" in frame.columns else None,
            }
        )
        top_avg_oscillation.append(
            {
                "ticker": ticker,
                "avg_oscillation_index_20": float(frame.select(pl.col("oscillation_index_20").mean()).item()) if "oscillation_index_20" in frame.columns else None,
            }
        )

        if idx % 1000 == 0:
            effective_logger.info("features_sanity.progress scanned=%s/%s", idx, len(files))

    null_rates = {
        column: (null_counts[column] / total_rows) if total_rows > 0 else None for column in key_features
    }
    summary: dict[str, Any] = {
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "symbol_count": symbol_count,
        "total_rows": total_rows,
        "global_min_trade_date": global_min_trade_date.isoformat() if global_min_trade_date is not None else None,
        "global_max_trade_date": global_max_trade_date.isoformat() if global_max_trade_date is not None else None,
        "null_rates": null_rates,
        "top_20_symbols_by_avg_flow_activity_20": sorted(
            (row for row in top_avg_flow_activity if row["avg_flow_activity_20"] is not None),
            key=lambda row: float(row["avg_flow_activity_20"]),
            reverse=True,
        )[:20],
        "top_20_symbols_by_max_abs_delta_flow_20": sorted(
            (row for row in top_max_abs_delta if row["max_abs_delta_flow_20"] is not None),
            key=lambda row: float(row["max_abs_delta_flow_20"]),
            reverse=True,
        )[:20],
        "top_20_symbols_by_avg_oscillation_index_20": sorted(
            (row for row in top_avg_oscillation if row["avg_oscillation_index_20"] is not None),
            key=lambda row: float(row["avg_oscillation_index_20"]),
            reverse=True,
        )[:20],
    }
    summary_path = artifacts_root / "gold_feature_qa" / "features_sanity_summary.json"
    _write_json_atomically(summary, summary_path)

    return GoldFeatureSanityResult(
        summary=summary,
        feature_file_count=len(files),
        read_errors=read_errors,
        summary_path=summary_path,
    )


def export_ml_dataset(
    settings: AppSettings,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    symbols_limit: int | None = None,
    sample_frac: float | None = None,
    logger: logging.Logger | None = None,
) -> GoldMLDatasetExportResult:
    """Export stacked ML dataset from per-symbol Gold feature parquet files."""

    effective_logger = logger or LOGGER
    if sample_frac is not None and (sample_frac <= 0 or sample_frac > 1):
        raise ValueError("sample_frac must be within (0, 1].")

    files = discover_feature_files(settings.paths.gold_root)
    if symbols_limit is not None:
        files = files[: max(0, symbols_limit)]

    selected_columns = [
        "ticker",
        "exchange",
        "trade_date",
        "trade_dt",
        "flow_state_code",
        "tmf_21",
        "tmf_abs",
        "tmf_slope_1",
        "tmf_slope_5",
        "tmf_slope_10",
        "tmf_curvature_1",
        "tti_proxy_v1_21",
        "tti_proxy_slope_1",
        "tti_proxy_slope_5",
        "tmf_tti_sign_agree",
        "tmf_tti_divergence",
        "long_flow_score_20",
        "short_flow_score_20",
        "delta_flow_20",
        "flow_activity_20",
        "flow_bias_20",
        "long_burst_20",
        "short_burst_20",
        "persistence_pos_20",
        "persistence_neg_20",
        "oscillation_index_20",
        "respect_fail_balance_20",
        "rec_tmf_zero_up_20",
        "rec_tmf_zero_down_20",
        "rec_tmf_burst_up_20",
        "rec_tmf_burst_down_20",
        "state_prev",
        "state_changed",
        "state_run_length",
        "state_transition_code",
        "bs_state_change",
        "close",
        "volume",
        "atr_14",
        "atr_pct_14",
        "dollar_volume",
        "quality_warn_count",
        "feature_calc_version",
        "feature_schema_version",
    ]
    key_readiness_columns = ["tmf_21", "long_flow_score_20", "delta_flow_20", "flow_activity_20"]

    frames: list[pl.DataFrame] = []
    read_errors = 0
    for idx, path in enumerate(files, start=1):
        try:
            schema = pl.read_parquet_schema(path)
            columns = [column for column in selected_columns if column in schema]
            frame = pl.read_parquet(path, columns=columns)
        except Exception as exc:
            read_errors += 1
            effective_logger.warning("export_ml_dataset.read_failed path=%s error=%s", path, exc)
            continue

        if start_date is not None:
            frame = frame.filter(pl.col("trade_date") >= pl.lit(start_date))
        if end_date is not None:
            frame = frame.filter(pl.col("trade_date") <= pl.lit(end_date))

        if settings.gold_features.export.default_drop_null_key_features:
            existing_keys = [column for column in key_readiness_columns if column in frame.columns]
            if existing_keys:
                frame = frame.filter(pl.all_horizontal([pl.col(column).is_not_null() for column in existing_keys]))

        if frame.height > 0:
            frames.append(frame)

        if idx % 1000 == 0:
            effective_logger.info("export_ml_dataset.progress read=%s/%s", idx, len(files))

    if frames:
        dataset_df = pl.concat(frames, how="vertical_relaxed")
    else:
        dataset_df = pl.DataFrame(schema={column: pl.String for column in selected_columns})

    if sample_frac is not None and dataset_df.height > 0 and sample_frac < 1.0:
        dataset_df = dataset_df.sample(fraction=sample_frac, with_replacement=False, shuffle=True, seed=42)

    dataset_df = dataset_df.sort([column for column in ("ticker", "trade_date") if column in dataset_df.columns])

    run_id = f"ml-dataset-v1-{uuid4().hex[:12]}"
    dataset_dir = settings.paths.gold_root / "datasets" / "ml_dataset_v1" / run_id
    dataset_path = dataset_dir / "dataset.parquet"
    metadata_path = dataset_dir / "metadata.json"

    _write_parquet_atomically(
        dataset_df,
        dataset_path,
        compression=settings.parquet.compression,
        compression_level=settings.parquet.compression_level,
        statistics=settings.parquet.statistics,
    )

    date_bounds = _frame_date_bounds(dataset_df)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "row_count": dataset_df.height,
        "symbol_count": int(dataset_df.select(pl.col("ticker").n_unique()).item()) if "ticker" in dataset_df.columns and dataset_df.height > 0 else 0,
        "columns": dataset_df.columns,
        "global_min_trade_date": date_bounds[0].isoformat() if date_bounds[0] is not None else None,
        "global_max_trade_date": date_bounds[1].isoformat() if date_bounds[1] is not None else None,
        "feature_calc_version": FEATURE_CALC_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "filters": {
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "symbols_limit": symbols_limit,
            "sample_frac": sample_frac,
            "drop_null_key_features": settings.gold_features.export.default_drop_null_key_features,
        },
        "read_errors": read_errors,
        "dataset_path": str(dataset_path),
    }
    _write_json_atomically(metadata, metadata_path)

    return GoldMLDatasetExportResult(
        run_id=run_id,
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        row_count=dataset_df.height,
        symbol_count=metadata["symbol_count"],
    )
