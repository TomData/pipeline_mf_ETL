"""Gold Event Grammar v1 pipeline orchestration."""

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
from mf_etl.gold.event_grammar_v1 import FLOW_STATE_CODE_TO_LABEL, build_event_grammar_v1
from mf_etl.gold.writer import GoldEventWriteResult, write_gold_event_parquet
from mf_etl.transform.dtypes import dtype_for_precision_name

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GoldEventOneResult:
    """Result metadata for one indicator->events conversion."""

    ticker: str
    exchange: str
    indicator_file: Path
    output_path: Path
    rows_in: int
    rows_out: int
    min_trade_date: date | None
    max_trade_date: date | None
    state_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class GoldEventRunOptions:
    """Runtime options for Gold event batch pipeline."""

    limit: int | None = None
    progress_every: int = 100
    full: bool = False


@dataclass(frozen=True, slots=True)
class GoldEventRunResult:
    """Batch run output summary metadata."""

    run_id: str
    summary: dict[str, Any]
    summary_path: Path
    ticker_results_path: Path


@dataclass(frozen=True, slots=True)
class GoldEventSanityResult:
    """Result payload for Gold event sanity scan."""

    summary: dict[str, Any]
    gold_event_file_count: int
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


def _iso_utc_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def discover_indicator_files(silver_root: Path) -> list[Path]:
    """Discover per-symbol indicator parquet files."""

    base = silver_root / "indicators_by_symbol"
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*.parquet") if path.is_file())


def discover_indicator_inputs(silver_root: Path) -> pl.DataFrame:
    """Build ticker/exchange/path frame from indicator output files."""

    rows: list[dict[str, object]] = []
    for path in discover_indicator_files(silver_root):
        rows.append(
            {
                "ticker": (_partition_value(path, "ticker") or path.stem).strip().upper(),
                "exchange": (_partition_value(path, "exchange") or "UNKNOWN").strip().upper(),
                "indicator_path": str(path),
            }
        )

    if not rows:
        return pl.DataFrame(schema={"ticker": pl.String, "exchange": pl.String, "indicator_path": pl.String})
    return pl.DataFrame(rows, schema_overrides={"ticker": pl.String, "exchange": pl.String, "indicator_path": pl.String})


def resolve_indicator_file_for_ticker(silver_root: Path, ticker: str) -> Path:
    """Resolve one indicator file for a ticker with ambiguity checks."""

    normalized = ticker.strip().upper()
    if normalized == "":
        raise ValueError("Ticker must be non-empty.")

    inputs = discover_indicator_inputs(silver_root)
    matched = inputs.filter(pl.col("ticker") == normalized)
    if matched.height == 0:
        raise FileNotFoundError(f"No indicator parquet found for ticker {normalized}")
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
            "use --indicator-file to disambiguate."
        )
    output = Path(str(matched.select("indicator_path").to_dicts()[0]["indicator_path"]))
    if not output.exists():
        raise FileNotFoundError(f"Indicator file does not exist for ticker {normalized}: {output}")
    return output


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


def _empty_state_counts() -> dict[str, int]:
    return {label: 0 for label in FLOW_STATE_CODE_TO_LABEL.values()}


def _state_counts_from_df(df: pl.DataFrame) -> dict[str, int]:
    counts = _empty_state_counts()
    if "flow_state_label" not in df.columns or df.height == 0:
        return counts
    for row in df.group_by("flow_state_label").len(name="count").to_dicts():
        label = str(row["flow_state_label"])
        counts[label] = int(row["count"])
    return counts


def run_events_one_from_indicator_file(
    indicator_file: Path,
    settings: AppSettings,
    *,
    run_id: str,
    logger: logging.Logger | None = None,
) -> GoldEventOneResult:
    """Read one indicator file, build events, and write Gold output."""

    effective_logger = logger or LOGGER
    indicator_df = pl.read_parquet(indicator_file)
    rows_in = indicator_df.height
    if rows_in == 0:
        raise ValueError(f"Indicator file is empty: {indicator_file}")

    event_df = build_event_grammar_v1(
        indicator_df,
        pivot_mode=settings.event_grammar.pivot_mode,
        respect_fail_lookahead_bars=settings.event_grammar.respect_fail_lookahead_bars,
        hold_consecutive_bars=settings.event_grammar.hold_consecutive_bars,
        tmf_burst_abs_threshold=settings.event_grammar.tmf_burst_abs_threshold,
        tmf_burst_slope_threshold=settings.event_grammar.tmf_burst_slope_threshold,
        activity_windows=settings.event_grammar.activity_windows,
        eps=settings.event_grammar.eps,
        float_dtype=dtype_for_precision_name(settings.precision.gold_float),
        fallback_run_id=run_id,
    )
    write_result: GoldEventWriteResult = write_gold_event_parquet(
        event_df,
        gold_root=settings.paths.gold_root,
        compression=settings.parquet.compression,
        compression_level=settings.parquet.compression_level,
        statistics=settings.parquet.statistics,
    )

    min_trade_date, max_trade_date = _frame_date_bounds(event_df)
    state_counts = _state_counts_from_df(event_df)

    effective_logger.debug(
        "events_one.complete ticker=%s exchange=%s rows_in=%s rows_out=%s output=%s",
        write_result.ticker,
        write_result.exchange,
        rows_in,
        write_result.rows_out,
        write_result.output_path,
    )
    return GoldEventOneResult(
        ticker=write_result.ticker,
        exchange=write_result.exchange,
        indicator_file=indicator_file,
        output_path=write_result.output_path,
        rows_in=rows_in,
        rows_out=write_result.rows_out,
        min_trade_date=min_trade_date,
        max_trade_date=max_trade_date,
        state_counts=state_counts,
    )


def run_events_pipeline(
    settings: AppSettings,
    *,
    options: GoldEventRunOptions | None = None,
    logger: logging.Logger | None = None,
) -> GoldEventRunResult:
    """Run batch Gold Event Grammar v1 generation for indicator files."""

    effective_logger = logger or LOGGER
    run_options = options or GoldEventRunOptions()
    progress_every = max(1, run_options.progress_every)
    run_id = f"events-run-{uuid4().hex[:12]}"
    started_epoch = time.time()

    inputs = discover_indicator_inputs(settings.paths.silver_root).sort("ticker")
    if run_options.limit is not None:
        inputs = inputs.head(max(0, run_options.limit))

    selected_total = inputs.height
    effective_logger.info(
        "events_run.start run_id=%s selected_symbols=%s full=%s limit=%s",
        run_id,
        selected_total,
        run_options.full,
        run_options.limit,
    )

    success_count = 0
    failed_count = 0
    rows_total = 0
    state_counts_global = _empty_state_counts()
    global_min_trade_date: date | None = None
    global_max_trade_date: date | None = None
    failed_files: list[dict[str, str]] = []
    ticker_results_rows: list[dict[str, object]] = []

    for idx, row in enumerate(inputs.iter_rows(named=True), start=1):
        ticker = str(row["ticker"])
        exchange = str(row["exchange"])
        indicator_file = Path(str(row["indicator_path"]))
        try:
            result = run_events_one_from_indicator_file(
                indicator_file,
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
            for label, count in result.state_counts.items():
                state_counts_global[label] = state_counts_global.get(label, 0) + int(count)

            ticker_results_rows.append(
                {
                    "ticker": result.ticker,
                    "exchange": result.exchange,
                    "source_file": str(result.indicator_file),
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
            failed_files.append({"source_file": str(indicator_file), "error": message})
            ticker_results_rows.append(
                {
                    "ticker": ticker,
                    "exchange": exchange,
                    "source_file": str(indicator_file),
                    "rows_in": 0,
                    "rows_out": 0,
                    "output_path": None,
                    "success": False,
                    "error_message": message,
                }
            )
            effective_logger.exception("events_run.symbol_failed ticker=%s source_file=%s", ticker, indicator_file)

        if idx % progress_every == 0 or idx == selected_total:
            elapsed = time.time() - started_epoch
            effective_logger.info(
                "events_run.progress processed=%s/%s success=%s failure=%s rows_total=%s elapsed_sec=%.2f",
                idx,
                selected_total,
                success_count,
                failed_count,
                rows_total,
                elapsed,
            )

    finished_epoch = time.time()
    duration_sec = finished_epoch - started_epoch

    artifacts_root = settings.paths.artifacts_root / "gold_event_run_summaries"
    summary_path = artifacts_root / f"{run_id}_events_run_summary.json"
    ticker_results_path = artifacts_root / f"{run_id}_events_ticker_results.parquet"

    summary: dict[str, Any] = {
        "run_id": run_id,
        "started_ts": _iso_utc_from_epoch(started_epoch),
        "finished_ts": _iso_utc_from_epoch(finished_epoch),
        "duration_sec": round(duration_sec, 3),
        "symbols_total_selected": selected_total,
        "symbols_success": success_count,
        "symbols_failed": failed_count,
        "rows_total": rows_total,
        "state_counts_global": state_counts_global,
        "global_min_trade_date": global_min_trade_date.isoformat() if global_min_trade_date is not None else None,
        "global_max_trade_date": global_max_trade_date.isoformat() if global_max_trade_date is not None else None,
        "output_root": str(settings.paths.gold_root / "events_by_symbol"),
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
    _write_parquet_atomically(results_df, ticker_results_path)

    return GoldEventRunResult(
        run_id=run_id,
        summary=summary,
        summary_path=summary_path,
        ticker_results_path=ticker_results_path,
    )


def run_events_sanity(
    gold_root: Path,
    artifacts_root: Path,
    *,
    logger: logging.Logger | None = None,
) -> GoldEventSanityResult:
    """Scan Gold event outputs and report compact QA statistics."""

    effective_logger = logger or LOGGER
    event_root = gold_root / "events_by_symbol"
    files = sorted(path for path in event_root.rglob("*.parquet") if path.is_file()) if event_root.exists() else []

    total_rows = 0
    symbol_count = 0
    read_errors = 0
    global_min_trade_date: date | None = None
    global_max_trade_date: date | None = None

    key_event_cols = [
        "ev_tmf_zero_up",
        "ev_tmf_zero_down",
        "ev_tmf_respect_zero_up",
        "ev_tmf_respect_zero_down",
        "ev_tmf_burst_up",
        "ev_tmf_burst_down",
    ]
    key_event_counts = {column: 0 for column in key_event_cols}
    state_counts_global = _empty_state_counts()
    top_by_activity: list[dict[str, object]] = []
    top_by_fails: list[dict[str, object]] = []

    for idx, path in enumerate(files, start=1):
        try:
            schema = pl.read_parquet_schema(path)
        except Exception as exc:
            read_errors += 1
            effective_logger.warning("events_sanity.schema_read_failed path=%s error=%s", path, exc)
            continue

        requested = [
            column
            for column in (
                "ticker",
                "trade_date",
                "flow_state_label",
                "tmf_event_activity_20",
                "ev_tmf_respect_fail_up",
                "ev_tmf_respect_fail_down",
                *key_event_cols,
            )
            if column in schema
        ]
        try:
            frame = pl.read_parquet(path, columns=requested)
        except Exception as exc:
            read_errors += 1
            effective_logger.warning("events_sanity.file_read_failed path=%s error=%s", path, exc)
            continue

        symbol_count += 1
        rows = frame.height
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

        ticker = str(frame.select(pl.col("ticker").first()).item()) if rows > 0 and "ticker" in frame.columns else "UNKNOWN"
        for column in key_event_cols:
            if column in frame.columns:
                key_event_counts[column] += int(frame.select(pl.col(column).cast(pl.Int64).sum()).item())

        if "flow_state_label" in frame.columns:
            for row in frame.group_by("flow_state_label").len(name="count").to_dicts():
                label = str(row["flow_state_label"])
                state_counts_global[label] = state_counts_global.get(label, 0) + int(row["count"])

        activity_value = None
        if "tmf_event_activity_20" in frame.columns:
            activity_value = frame.select(pl.col("tmf_event_activity_20").max()).item()
        fail_total = 0
        if "ev_tmf_respect_fail_up" in frame.columns:
            fail_total += int(frame.select(pl.col("ev_tmf_respect_fail_up").cast(pl.Int64).sum()).item())
        if "ev_tmf_respect_fail_down" in frame.columns:
            fail_total += int(frame.select(pl.col("ev_tmf_respect_fail_down").cast(pl.Int64).sum()).item())

        top_by_activity.append(
            {
                "ticker": ticker,
                "max_tmf_event_activity_20": float(activity_value) if activity_value is not None else None,
            }
        )
        top_by_fails.append({"ticker": ticker, "tmf_respect_fail_count": fail_total})

        if idx % 1000 == 0:
            effective_logger.info("events_sanity.progress scanned=%s/%s", idx, len(files))

    top_activity = sorted(
        (row for row in top_by_activity if row["max_tmf_event_activity_20"] is not None),
        key=lambda row: float(row["max_tmf_event_activity_20"]),
        reverse=True,
    )[:20]
    top_fails = sorted(top_by_fails, key=lambda row: int(row["tmf_respect_fail_count"]), reverse=True)[:20]

    summary: dict[str, Any] = {
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "symbol_count": symbol_count,
        "total_rows": total_rows,
        "global_min_trade_date": global_min_trade_date.isoformat() if global_min_trade_date is not None else None,
        "global_max_trade_date": global_max_trade_date.isoformat() if global_max_trade_date is not None else None,
        "key_event_counts": key_event_counts,
        "state_distribution": state_counts_global,
        "top_20_symbols_by_max_tmf_event_activity_20": top_activity,
        "top_20_symbols_by_tmf_respect_fail_count": top_fails,
    }
    summary_path = artifacts_root / "gold_event_qa" / "events_sanity_summary.json"
    _write_json_atomically(summary, summary_path)

    return GoldEventSanityResult(
        summary=summary,
        gold_event_file_count=len(files),
        read_errors=read_errors,
        summary_path=summary_path,
    )
