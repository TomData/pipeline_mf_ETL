"""Walk-forward orchestration for multi-split OOS validation packs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from mf_etl.config import AppSettings
from mf_etl.research.pipeline import run_research_cluster
from mf_etl.research_hmm.pipeline import run_hmm_baseline
from mf_etl.validation.pipeline import run_validation_compare, run_validation_harness
from mf_etl.validation.walkforward_reports import (
    WalkForwardReportPaths,
    collect_walkforward_outputs,
    write_walkforward_outputs,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WalkForwardRunResult:
    """Result artifact paths for one walk-forward execution."""

    wf_run_id: str
    output_dir: Path
    manifest_path: Path
    aggregate_summary_path: Path
    full_report_path: Path


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


def _build_wf_signature(payload: dict[str, Any]) -> str:
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return digest[:12]


def _normalize_train_ends(train_end_list: list[date]) -> list[str]:
    unique_sorted = sorted({entry.isoformat() for entry in train_end_list})
    return unique_sorted


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "wf_manifest.json"


def _load_or_init_manifest(
    *,
    output_dir: Path,
    wf_run_id: str,
    dataset_path: Path,
    config_payload: dict[str, Any],
    train_end_list: list[str],
) -> dict[str, Any]:
    path = _manifest_path(output_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    manifest = {
        "wf_run_id": wf_run_id,
        "dataset_path": str(dataset_path),
        "config": config_payload,
        "train_end_list": train_end_list,
        "started_ts": datetime.now(timezone.utc).isoformat(),
        "updated_ts": datetime.now(timezone.utc).isoformat(),
        "splits": [],
    }
    _write_json_atomically(manifest, path)
    return manifest


def _upsert_split_record(manifest: dict[str, Any], split_record: dict[str, Any]) -> None:
    train_end = split_record.get("train_end")
    splits = manifest.get("splits", [])
    for idx, entry in enumerate(splits):
        if entry.get("train_end") == train_end:
            splits[idx] = split_record
            manifest["splits"] = splits
            return
    splits.append(split_record)
    manifest["splits"] = splits


def _split_record_by_train_end(manifest: dict[str, Any], train_end: str) -> dict[str, Any] | None:
    for entry in manifest.get("splits", []):
        if entry.get("train_end") == train_end:
            return entry
    return None


def _split_outputs_valid(record: dict[str, Any]) -> bool:
    if record.get("status") != "SUCCESS":
        return False

    required_files = [
        Path(str(record.get("hmm_run_dir", ""))) / "run_summary.json",
        Path(str(record.get("cluster_run_dir", ""))) / "run_summary.json",
        Path(str(record.get("cluster_run_dir", ""))) / "clustered_dataset_full.parquet",
        Path(str(record.get("val_hmm_dir", ""))) / "validation_scorecard.json",
        Path(str(record.get("val_flow_dir", ""))) / "validation_scorecard.json",
        Path(str(record.get("val_cluster_dir", ""))) / "validation_scorecard.json",
        Path(str(record.get("cmp_hmm_flow_dir", ""))) / "validation_compare_summary.json",
        Path(str(record.get("cmp_hmm_cluster_dir", ""))) / "validation_compare_summary.json",
    ]
    return all(path.exists() for path in required_files)


def _resolve_cluster_full_path(cluster_run_dir: Path) -> Path:
    candidate = cluster_run_dir / "clustered_dataset_full.parquet"
    if candidate.exists():
        return candidate
    run_summary = json.loads((cluster_run_dir / "run_summary.json").read_text(encoding="utf-8"))
    path_from_summary = run_summary.get("outputs", {}).get("clustered_dataset_full_path")
    if path_from_summary:
        path_obj = Path(path_from_summary)
        if path_obj.exists():
            return path_obj
    raise FileNotFoundError(f"Full clustered dataset not found for run {cluster_run_dir}")


def run_validation_walkforward(
    settings: AppSettings,
    *,
    dataset_path: Path,
    train_end_list: list[date] | None = None,
    hmm_components: int | None = None,
    cluster_method: str | None = None,
    cluster_k: int | None = None,
    scaling_scope: str | None = None,
    bootstrap_n: int | None = None,
    bootstrap_mode: str | None = None,
    block_length: int | None = None,
    event_window_pre: int | None = None,
    event_window_post: int | None = None,
    min_events_per_transition: int | None = None,
    force: bool = False,
    force_splits: list[date] | None = None,
    stop_on_error: bool = False,
    max_splits: int | None = None,
    logger: logging.Logger | None = None,
) -> WalkForwardRunResult:
    """Run multi-split walk-forward OOS validation pack."""

    effective_logger = logger or LOGGER
    selected_train_ends = train_end_list or settings.validation_walkforward.train_end_list_default
    train_ends_iso = _normalize_train_ends(selected_train_ends)
    if max_splits is not None and max_splits > 0:
        train_ends_iso = train_ends_iso[:max_splits]

    force_split_set = {
        entry.isoformat() for entry in (force_splits or [])
    }

    resolved_hmm_components = hmm_components or settings.validation_walkforward.hmm_components_default
    resolved_cluster_method = (cluster_method or settings.validation_walkforward.cluster_method_default).lower().strip()
    resolved_cluster_k = cluster_k or settings.validation_walkforward.cluster_k_default
    resolved_scaling_scope = (scaling_scope or settings.validation_walkforward.scaling_scope_default).lower().strip()

    resolved_bootstrap_n = bootstrap_n if bootstrap_n is not None else settings.validation.bootstrap.n_boot
    resolved_bootstrap_mode = (bootstrap_mode or settings.validation.bootstrap.mode).lower().strip()
    resolved_block_length = block_length if block_length is not None else settings.validation.bootstrap.block_length
    resolved_event_window_pre = event_window_pre if event_window_pre is not None else settings.validation.event_study.window_pre
    resolved_event_window_post = event_window_post if event_window_post is not None else settings.validation.event_study.window_post
    resolved_min_events = (
        min_events_per_transition
        if min_events_per_transition is not None
        else settings.validation.event_study.min_events_per_transition
    )

    signature_payload = {
        "dataset_path": str(dataset_path),
        "train_end_list": train_ends_iso,
        "hmm_components": resolved_hmm_components,
        "cluster_method": resolved_cluster_method,
        "cluster_k": resolved_cluster_k,
        "scaling_scope": resolved_scaling_scope,
        "bootstrap_n": resolved_bootstrap_n,
        "bootstrap_mode": resolved_bootstrap_mode,
        "block_length": resolved_block_length,
        "event_window_pre": resolved_event_window_pre,
        "event_window_post": resolved_event_window_post,
        "min_events_per_transition": resolved_min_events,
    }
    wf_run_id = f"wf-{_build_wf_signature(signature_payload)}"
    output_dir = settings.paths.artifacts_root / "validation_walkforward" / wf_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_or_init_manifest(
        output_dir=output_dir,
        wf_run_id=wf_run_id,
        dataset_path=dataset_path,
        config_payload=signature_payload,
        train_end_list=train_ends_iso,
    )

    for train_end_iso in train_ends_iso:
        current = _split_record_by_train_end(manifest, train_end_iso)
        should_force_split = force or (train_end_iso in force_split_set)
        if not should_force_split and current is not None and _split_outputs_valid(current):
            current["skipped_existing"] = True
            current["updated_ts"] = datetime.now(timezone.utc).isoformat()
            _upsert_split_record(manifest, current)
            manifest["updated_ts"] = datetime.now(timezone.utc).isoformat()
            _write_json_atomically(manifest, _manifest_path(output_dir))
            effective_logger.info("validation_wf.split.skip train_end=%s reason=existing_success", train_end_iso)
            continue

        split_started = datetime.now(timezone.utc)
        split_record: dict[str, Any] = {
            "train_end": train_end_iso,
            "status": "RUNNING",
            "started_ts": split_started.isoformat(),
            "updated_ts": split_started.isoformat(),
            "skipped_existing": False,
            "error": None,
        }
        _upsert_split_record(manifest, split_record)
        manifest["updated_ts"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomically(manifest, _manifest_path(output_dir))

        try:
            train_end_date = date.fromisoformat(train_end_iso)

            hmm_result = run_hmm_baseline(
                settings,
                dataset_path=dataset_path,
                n_components=resolved_hmm_components,
                split_mode="time",
                train_end=train_end_date,
                fit_on="train",
                predict_on="test",
                scaling_scope=resolved_scaling_scope,
                scaler=settings.research_hmm.scaler,
                random_state=settings.research_hmm.hmm.random_state,
                logger=effective_logger,
            )
            hmm_decoded = hmm_result.decoded_rows_path

            cluster_result = run_research_cluster(
                settings,
                dataset_path=dataset_path,
                method=resolved_cluster_method,
                n_clusters=resolved_cluster_k,
                sample_frac=None,
                date_from=None,
                date_to=None,
                split_mode="time",
                train_end=train_end_date,
                fit_on="train",
                predict_on="test",
                scaling_scope=resolved_scaling_scope,
                scaler=settings.research_clustering.scaler,
                random_state=settings.research_clustering.random_state,
                write_full_clustered=True,
                logger=effective_logger,
            )
            cluster_full = _resolve_cluster_full_path(cluster_result.output_dir)

            val_hmm = run_validation_harness(
                settings,
                input_file=hmm_decoded,
                input_type="hmm",
                bootstrap_n=resolved_bootstrap_n,
                bootstrap_mode=resolved_bootstrap_mode,
                block_length=resolved_block_length,
                event_window_pre=resolved_event_window_pre,
                event_window_post=resolved_event_window_post,
                min_events_per_transition=resolved_min_events,
                logger=effective_logger,
            )
            val_flow = run_validation_harness(
                settings,
                input_file=hmm_decoded,
                input_type="generic",
                state_col="flow_state_code",
                bootstrap_n=resolved_bootstrap_n,
                bootstrap_mode=resolved_bootstrap_mode,
                block_length=resolved_block_length,
                event_window_pre=resolved_event_window_pre,
                event_window_post=resolved_event_window_post,
                min_events_per_transition=resolved_min_events,
                logger=effective_logger,
            )
            val_cluster = run_validation_harness(
                settings,
                input_file=cluster_full,
                input_type="cluster",
                bootstrap_n=resolved_bootstrap_n,
                bootstrap_mode=resolved_bootstrap_mode,
                block_length=resolved_block_length,
                event_window_pre=resolved_event_window_pre,
                event_window_post=resolved_event_window_post,
                min_events_per_transition=resolved_min_events,
                logger=effective_logger,
            )

            cmp_hf = run_validation_compare(
                settings,
                run_dir_a=val_hmm.output_dir,
                run_dir_b=val_flow.output_dir,
                logger=effective_logger,
            )
            cmp_hc = run_validation_compare(
                settings,
                run_dir_a=val_hmm.output_dir,
                run_dir_b=val_cluster.output_dir,
                logger=effective_logger,
            )

            split_finished = datetime.now(timezone.utc)
            split_record = {
                "train_end": train_end_iso,
                "status": "SUCCESS",
                "started_ts": split_started.isoformat(),
                "finished_ts": split_finished.isoformat(),
                "duration_sec": round((split_finished - split_started).total_seconds(), 3),
                "updated_ts": split_finished.isoformat(),
                "error": None,
                "hmm_run_dir": str(hmm_result.output_dir),
                "cluster_run_dir": str(cluster_result.output_dir),
                "val_hmm_dir": str(val_hmm.output_dir),
                "val_flow_dir": str(val_flow.output_dir),
                "val_cluster_dir": str(val_cluster.output_dir),
                "cmp_hmm_flow_dir": str(cmp_hf.output_dir),
                "cmp_hmm_cluster_dir": str(cmp_hc.output_dir),
                "skipped_existing": False,
            }
            _upsert_split_record(manifest, split_record)
            manifest["updated_ts"] = datetime.now(timezone.utc).isoformat()
            _write_json_atomically(manifest, _manifest_path(output_dir))
            effective_logger.info("validation_wf.split.success train_end=%s", train_end_iso)
        except Exception as exc:
            split_finished = datetime.now(timezone.utc)
            split_record = {
                "train_end": train_end_iso,
                "status": "FAILED",
                "started_ts": split_started.isoformat(),
                "finished_ts": split_finished.isoformat(),
                "duration_sec": round((split_finished - split_started).total_seconds(), 3),
                "updated_ts": split_finished.isoformat(),
                "error": str(exc),
            }
            _upsert_split_record(manifest, split_record)
            manifest["updated_ts"] = datetime.now(timezone.utc).isoformat()
            _write_json_atomically(manifest, _manifest_path(output_dir))
            effective_logger.exception("validation_wf.split.failed train_end=%s", train_end_iso)
            continue_on_error = settings.validation_walkforward.continue_on_error_default
            if stop_on_error or not continue_on_error:
                raise

    manifest["finished_ts"] = datetime.now(timezone.utc).isoformat()
    manifest["updated_ts"] = datetime.now(timezone.utc).isoformat()

    collected = collect_walkforward_outputs(manifest.get("splits", []))
    paths: WalkForwardReportPaths = write_walkforward_outputs(
        output_dir=output_dir,
        wf_manifest=manifest,
        dataset_path=dataset_path,
        train_end_list=train_ends_iso,
        collected=collected,
    )

    effective_logger.info(
        "validation_wf.complete wf_run_id=%s splits_total=%s splits_success=%s output=%s",
        wf_run_id,
        len(manifest.get("splits", [])),
        collected.aggregate_summary.get("splits_successful"),
        output_dir,
    )

    return WalkForwardRunResult(
        wf_run_id=wf_run_id,
        output_dir=output_dir,
        manifest_path=paths.manifest_path,
        aggregate_summary_path=paths.aggregate_summary_path,
        full_report_path=paths.full_report_path,
    )


def summarize_validation_walkforward_run(wf_run_dir: Path) -> dict[str, Any]:
    """Read a walk-forward run directory and return concise summary data."""

    manifest_path = wf_run_dir / "wf_manifest.json"
    aggregate_path = wf_run_dir / "wf_aggregate_summary.json"
    long_path = wf_run_dir / "wf_model_summary_long.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Walk-forward manifest not found: {manifest_path}")
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Walk-forward aggregate summary not found: {aggregate_path}")
    if not long_path.exists():
        raise FileNotFoundError(f"Walk-forward model summary not found: {long_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    long_df = pl.read_csv(long_path)

    split_status_rows = [
        {
            "train_end": split.get("train_end"),
            "status": split.get("status"),
            "error": split.get("error"),
        }
        for split in manifest.get("splits", [])
    ]

    return {
        "wf_run_dir": str(wf_run_dir),
        "wf_run_id": manifest.get("wf_run_id"),
        "dataset_path": manifest.get("dataset_path"),
        "splits_total": aggregate.get("splits_total"),
        "splits_successful": aggregate.get("splits_successful"),
        "splits_failed": aggregate.get("splits_failed"),
        "failed_splits": aggregate.get("failed_splits", []),
        "wins_by_metric": aggregate.get("wins_by_metric", {}),
        "aggregate_by_model": aggregate.get("aggregate_by_model", {}),
        "split_status_rows": split_status_rows,
        "model_summary_preview": long_df.head(30).to_dicts(),
    }
