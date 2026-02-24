"""Validation harness pipeline orchestration and artifact writers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from mf_etl.config import AppSettings
from mf_etl.validation.bootstrap import run_bootstrap_validation
from mf_etl.validation.dataset_adapters import AdaptedDataset, InputType, adapt_validation_dataset
from mf_etl.validation.event_studies import run_transition_event_study
from mf_etl.validation.scorecards import build_validation_scorecards
from mf_etl.validation.stability import build_rolling_stability_diagnostics

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ValidationRunResult:
    """Artifact locations for one validation harness run."""

    run_id: str
    output_dir: Path
    run_summary_path: Path
    validation_scorecard_path: Path


@dataclass(frozen=True, slots=True)
class ValidationCompareResult:
    """Artifact locations for one validation run comparison."""

    compare_id: str
    output_dir: Path
    summary_path: Path
    table_path: Path


def _atomic_temp_path(target_path: Path) -> Path:
    return target_path.parent / f".{target_path.name}.{uuid4().hex}.tmp"


def _write_json_atomically(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_temp_path(output_path)
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, output_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return output_path


def _write_parquet_atomically(df: pl.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_temp_path(output_path)
    try:
        df.write_parquet(tmp)
        os.replace(tmp, output_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return output_path


def _write_csv_atomically(df: pl.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_temp_path(output_path)
    try:
        df.write_csv(tmp)
        os.replace(tmp, output_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return output_path


def _frame_bounds(df: pl.DataFrame) -> dict[str, Any]:
    if df.height == 0 or "trade_date" not in df.columns:
        return {"rows": df.height, "min_trade_date": None, "max_trade_date": None}
    bounds = df.select(
        [
            pl.col("trade_date").min().alias("min_trade_date"),
            pl.col("trade_date").max().alias("max_trade_date"),
        ]
    ).to_dicts()[0]
    min_value = bounds["min_trade_date"]
    max_value = bounds["max_trade_date"]
    return {
        "rows": df.height,
        "min_trade_date": min_value.isoformat() if min_value is not None else None,
        "max_trade_date": max_value.isoformat() if max_value is not None else None,
    }


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def run_validation_harness(
    settings: AppSettings,
    *,
    input_file: Path,
    input_type: InputType = "auto",
    state_col: str | None = None,
    bootstrap_n: int | None = None,
    bootstrap_ci: float | None = None,
    bootstrap_mode: str | None = None,
    block_length: int | None = None,
    event_window_pre: int | None = None,
    event_window_post: int | None = None,
    min_events_per_transition: int | None = None,
    window_months: int | None = None,
    step_months: int | None = None,
    write_large_artifacts: bool | None = None,
    sample_frac: float | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    logger: logging.Logger | None = None,
) -> ValidationRunResult:
    """Run Validation Harness v1 and write scorecard/statistical artifacts."""

    effective_logger = logger or LOGGER
    started = datetime.now(timezone.utc)

    adapted: AdaptedDataset = adapt_validation_dataset(
        input_file,
        input_type=input_type,
        state_col=state_col,
        date_from=date_from,
        date_to=date_to,
        sample_frac=sample_frac,
        logger=effective_logger,
    )

    run_id = f"validation-{uuid4().hex[:12]}"
    state_tag = state_col or adapted.state_column
    output_dir = settings.paths.artifacts_root / "validation_runs" / f"{run_id}_{adapted.input_type}_{state_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_result = run_bootstrap_validation(
        adapted.frame,
        n_boot=bootstrap_n if bootstrap_n is not None else settings.validation.bootstrap.n_boot,
        ci=bootstrap_ci if bootstrap_ci is not None else settings.validation.bootstrap.ci,
        random_state=settings.validation.bootstrap.random_state,
        bootstrap_mode=(bootstrap_mode or settings.validation.bootstrap.mode),
        block_length=block_length if block_length is not None else settings.validation.bootstrap.block_length,
    )

    event_result = run_transition_event_study(
        adapted.frame,
        window_pre=event_window_pre if event_window_pre is not None else settings.validation.event_study.window_pre,
        window_post=event_window_post if event_window_post is not None else settings.validation.event_study.window_post,
        min_events_per_transition=(
            min_events_per_transition
            if min_events_per_transition is not None
            else settings.validation.event_study.min_events_per_transition
        ),
    )

    stability_result = build_rolling_stability_diagnostics(
        adapted.frame,
        window_months=window_months if window_months is not None else settings.validation.rolling_stability.window_months,
        step_months=step_months if step_months is not None else settings.validation.rolling_stability.step_months,
        eps=settings.validation.scorecard.eps,
        compute_transition_stability=(adapted.state_label_type == "hmm"),
    )

    scorecards = build_validation_scorecards(
        adapted_df=adapted.frame,
        state_label_type=adapted.state_label_type,
        bootstrap_state_summary=bootstrap_result.state_summary,
        bootstrap_pairwise_diff=bootstrap_result.pairwise_diff,
        state_stability_summary=stability_result.state_stability_summary,
        weights=settings.validation.scorecard.confidence_score_weights.model_dump(mode="python"),
        eps=settings.validation.scorecard.eps,
    )

    write_large = (
        write_large_artifacts
        if write_large_artifacts is not None
        else settings.validation.io.write_large_artifacts_default
    )

    run_summary_path = output_dir / "run_summary.json"
    adapter_summary_path = output_dir / "adapter_summary.json"
    bootstrap_summary_csv = output_dir / "bootstrap_state_summary.csv"
    bootstrap_summary_parquet = output_dir / "bootstrap_state_summary.parquet"
    bootstrap_pairwise_csv = output_dir / "bootstrap_pairwise_diff.csv"
    bootstrap_pairwise_parquet = output_dir / "bootstrap_pairwise_diff.parquet"
    bootstrap_meta_path = output_dir / "bootstrap_meta.json"
    transition_summary_csv = output_dir / "transition_event_summary.csv"
    transition_summary_parquet = output_dir / "transition_event_summary.parquet"
    transition_path_csv = output_dir / "transition_event_path_summary.csv"
    transition_path_parquet = output_dir / "transition_event_path_summary.parquet"
    transition_top_codes_path = output_dir / "transition_top_codes.json"
    rolling_csv = output_dir / "rolling_state_metrics.csv"
    rolling_parquet = output_dir / "rolling_state_metrics.parquet"
    state_stability_csv = output_dir / "state_stability_summary.csv"
    state_stability_parquet = output_dir / "state_stability_summary.parquet"
    transition_matrix_stability_csv = output_dir / "transition_matrix_stability.csv"
    state_scorecard_csv = output_dir / "state_scorecard.csv"
    state_scorecard_parquet = output_dir / "state_scorecard.parquet"
    validation_scorecard_path = output_dir / "validation_scorecard.json"

    _write_json_atomically(adapted.summary, adapter_summary_path)
    _write_csv_atomically(bootstrap_result.state_summary, bootstrap_summary_csv)
    _write_parquet_atomically(bootstrap_result.state_summary, bootstrap_summary_parquet)
    _write_csv_atomically(bootstrap_result.pairwise_diff, bootstrap_pairwise_csv)
    _write_parquet_atomically(bootstrap_result.pairwise_diff, bootstrap_pairwise_parquet)
    _write_json_atomically(bootstrap_result.meta, bootstrap_meta_path)
    _write_csv_atomically(event_result.transition_event_summary, transition_summary_csv)
    _write_parquet_atomically(event_result.transition_event_summary, transition_summary_parquet)
    _write_csv_atomically(event_result.transition_event_path_summary, transition_path_csv)
    _write_parquet_atomically(event_result.transition_event_path_summary, transition_path_parquet)
    _write_json_atomically(event_result.transition_top_codes, transition_top_codes_path)
    _write_csv_atomically(stability_result.rolling_state_metrics, rolling_csv)
    _write_parquet_atomically(stability_result.rolling_state_metrics, rolling_parquet)
    _write_csv_atomically(stability_result.state_stability_summary, state_stability_csv)
    _write_parquet_atomically(stability_result.state_stability_summary, state_stability_parquet)
    _write_csv_atomically(stability_result.transition_matrix_stability, transition_matrix_stability_csv)
    _write_csv_atomically(scorecards.state_scorecard, state_scorecard_csv)
    _write_parquet_atomically(scorecards.state_scorecard, state_scorecard_parquet)
    _write_json_atomically(scorecards.validation_scorecard, validation_scorecard_path)

    if write_large:
        _write_parquet_atomically(event_result.transition_events, output_dir / "transition_events.parquet")
        _write_parquet_atomically(adapted.frame.head(200_000), output_dir / "adapted_rows_sample.parquet")

    finished = datetime.now(timezone.utc)
    run_summary = {
        "run_id": run_id,
        "input_file": str(input_file),
        "input_type": adapted.input_type,
        "state_label_type": adapted.state_label_type,
        "state_column": adapted.state_column,
        "started_ts": started.isoformat(),
        "finished_ts": finished.isoformat(),
        "duration_sec": round((finished - started).total_seconds(), 3),
        "rows": adapted.frame.height,
        "state_count": _safe_int(adapted.frame.select(pl.col("state_id").n_unique()).item()) if adapted.frame.height > 0 else 0,
        "ticker_count": _safe_int(adapted.frame.select(pl.col("ticker").n_unique()).item()) if adapted.frame.height > 0 else 0,
        "bounds": _frame_bounds(adapted.frame),
        "bootstrap": bootstrap_result.meta,
        "event_study": {
            "event_rows": event_result.transition_events.height,
            "transition_summary_rows": event_result.transition_event_summary.height,
            "path_summary_rows": event_result.transition_event_path_summary.height,
        },
        "rolling_stability": {
            "rolling_rows": stability_result.rolling_state_metrics.height,
            "state_summary_rows": stability_result.state_stability_summary.height,
        },
        "write_large_artifacts": bool(write_large),
        "outputs": {
            "run_summary": str(run_summary_path),
            "adapter_summary": str(adapter_summary_path),
            "bootstrap_state_summary_csv": str(bootstrap_summary_csv),
            "bootstrap_pairwise_diff_csv": str(bootstrap_pairwise_csv),
            "transition_event_summary_csv": str(transition_summary_csv),
            "transition_event_path_summary_csv": str(transition_path_csv),
            "rolling_state_metrics_csv": str(rolling_csv),
            "state_stability_summary_csv": str(state_stability_csv),
            "state_scorecard_csv": str(state_scorecard_csv),
            "validation_scorecard_json": str(validation_scorecard_path),
        },
    }
    _write_json_atomically(run_summary, run_summary_path)

    effective_logger.info(
        "validation_run.complete run_id=%s rows=%s states=%s output=%s",
        run_id,
        run_summary["rows"],
        run_summary["state_count"],
        output_dir,
    )

    return ValidationRunResult(
        run_id=run_id,
        output_dir=output_dir,
        run_summary_path=run_summary_path,
        validation_scorecard_path=validation_scorecard_path,
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run_validation_compare(
    settings: AppSettings,
    *,
    run_dir_a: Path,
    run_dir_b: Path,
    logger: logging.Logger | None = None,
) -> ValidationCompareResult:
    """Compare two validation runs and write top-line comparison artifacts."""

    effective_logger = logger or LOGGER
    score_a = _load_json(run_dir_a / "validation_scorecard.json")
    score_b = _load_json(run_dir_b / "validation_scorecard.json")

    metrics = [
        "n_states",
        "total_rows",
        "forward_separation_score",
        "pairwise_diff_significant_share",
        "avg_ci_width_fwd_ret_10",
        "avg_state_sign_consistency",
        "avg_state_ret_cv",
    ]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        a_val = score_a.get(metric)
        b_val = score_b.get(metric)
        delta: float | None = None
        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
            delta = float(b_val - a_val)
        rows.append(
            {
                "metric": metric,
                "run_a": a_val,
                "run_b": b_val,
                "delta_b_minus_a": delta,
            }
        )

    table_df = pl.DataFrame(rows)
    compare_id = f"validation-compare-{uuid4().hex[:12]}"
    output_dir = settings.paths.artifacts_root / "validation_runs" / f"{compare_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "validation_compare_table.csv"
    summary_path = output_dir / "validation_compare_summary.json"

    _write_csv_atomically(table_df, table_path)
    summary_payload = {
        "compare_id": compare_id,
        "run_dir_a": str(run_dir_a),
        "run_dir_b": str(run_dir_b),
        "run_a_validation_grade": score_a.get("validation_grade"),
        "run_b_validation_grade": score_b.get("validation_grade"),
        "run_a_input_type": score_a.get("input_type"),
        "run_b_input_type": score_b.get("input_type"),
        "metric_diffs": rows,
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "summary": str(summary_path),
            "table": str(table_path),
        },
    }
    _write_json_atomically(summary_payload, summary_path)

    effective_logger.info(
        "validation_compare.complete compare_id=%s run_a=%s run_b=%s output=%s",
        compare_id,
        run_dir_a,
        run_dir_b,
        output_dir,
    )
    return ValidationCompareResult(
        compare_id=compare_id,
        output_dir=output_dir,
        summary_path=summary_path,
        table_path=table_path,
    )
