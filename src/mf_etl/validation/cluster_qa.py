"""Cluster QA diagnostics for unstable state behavior in validation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mf_etl.config import AppSettings
from mf_etl.validation.cluster_qa_reports import (
    ClusterQASingleReportPaths,
    ClusterQAWalkForwardReportPaths,
    write_cluster_qa_single_reports,
    write_cluster_qa_walkforward_reports,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClusterQADiagnostics:
    """In-memory diagnostics for one cluster validation run."""

    summary: dict[str, Any]
    flagged_states: pl.DataFrame
    state_windows: pl.DataFrame


@dataclass(frozen=True, slots=True)
class ClusterQASingleResult:
    """Written artifacts for single-run cluster QA."""

    output_dir: Path
    summary_path: Path
    flagged_states_path: Path
    state_windows_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class ClusterQAWalkForwardResult:
    """Written artifacts for walk-forward cluster QA aggregation."""

    output_dir: Path
    summary_path: Path
    flagged_states_path: Path
    issue_frequency_path: Path
    report_path: Path


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _issue_counts(flagged_states: pl.DataFrame) -> dict[str, int]:
    if flagged_states.height == 0 or "issues" not in flagged_states.columns:
        return {}
    issues = (
        flagged_states.select(pl.col("issues").cast(pl.String).str.split(",").alias("issue_list"))
        .explode("issue_list")
        .with_columns(pl.col("issue_list").str.strip_chars().alias("issue"))
        .filter(pl.col("issue") != "")
    )
    if issues.height == 0:
        return {}
    grouped = issues.group_by("issue").len(name="count").sort("count", descending=True)
    return {row["issue"]: int(row["count"]) for row in grouped.to_dicts()}


def _transition_counts_by_state(transition_summary: pl.DataFrame) -> dict[int, int]:
    if transition_summary.height == 0 or "count_events" not in transition_summary.columns:
        return {}
    counts: dict[int, int] = {}
    for row in transition_summary.to_dicts():
        count = int(row.get("count_events", 0) or 0)
        prev_state = row.get("prev_state_id")
        next_state = row.get("next_state_id")
        if prev_state is not None:
            key = int(prev_state)
            counts[key] = counts.get(key, 0) + count
        if next_state is not None:
            key = int(next_state)
            counts[key] = counts.get(key, 0) + count
    return counts


def _state_window_diagnostics(
    state_id: int,
    rolling_metrics: pl.DataFrame,
) -> tuple[list[dict[str, Any]], bool]:
    subset = rolling_metrics.filter(pl.col("state_id") == state_id).sort("window_seq")
    if subset.height == 0:
        return [], False

    values = subset["fwd_ret_10_mean"].cast(pl.Float64, strict=False).to_numpy()
    finite_values = values[np.isfinite(values)]
    sign_flip_count = 0
    if finite_values.size > 1:
        prev_sign = 0
        for value in finite_values:
            sign = 1 if value > 0 else (-1 if value < 0 else 0)
            if prev_sign != 0 and sign != 0 and sign != prev_sign:
                sign_flip_count += 1
            if sign != 0:
                prev_sign = sign

    window_drift_high = False
    if finite_values.size >= 4:
        mean = float(np.mean(finite_values))
        std = float(np.std(finite_values, ddof=0))
        if std > 0:
            max_dev = float(np.max(np.abs(finite_values - mean)))
            window_drift_high = bool(max_dev > 2.0 * std)

    share_values = subset["state_share"].cast(pl.Float64, strict=False).to_numpy()
    finite_share = share_values[np.isfinite(share_values)]
    share_variance = float(np.var(finite_share, ddof=0)) if finite_share.size > 0 else None

    worst = subset.sort("fwd_ret_10_mean", descending=False, nulls_last=True).head(3)
    best = subset.sort("fwd_ret_10_mean", descending=True, nulls_last=True).head(3)

    rows: list[dict[str, Any]] = []
    for kind, frame in (("worst", worst), ("best", best)):
        for row in frame.to_dicts():
            rows.append(
                {
                    "state_id": state_id,
                    "window_kind": kind,
                    "window_seq": row.get("window_seq"),
                    "window_start": row.get("window_start"),
                    "window_end": row.get("window_end"),
                    "fwd_ret_10_mean": row.get("fwd_ret_10_mean"),
                    "fwd_ret_10_hit_rate": row.get("fwd_ret_10_hit_rate"),
                    "state_share": row.get("state_share"),
                    "sign_flip_count": sign_flip_count,
                    "share_variance": share_variance,
                }
            )

    return rows, window_drift_high


def analyze_cluster_validation_run(
    validation_run_dir: Path,
    *,
    ret_cv_threshold: float,
    min_n_rows: int,
    min_state_share: float,
    sign_consistency_threshold: float,
    ci_width_quantile_threshold: float,
    eps: float,
) -> ClusterQADiagnostics:
    """Analyze one cluster validation run and flag unstable states with root-cause labels."""

    state_scorecard = pl.read_csv(_require(validation_run_dir / "state_scorecard.csv"))
    state_stability = pl.read_csv(_require(validation_run_dir / "state_stability_summary.csv"))
    rolling_metrics = pl.read_csv(_require(validation_run_dir / "rolling_state_metrics.csv"))
    transition_summary = pl.read_csv(_require(validation_run_dir / "transition_event_summary.csv"))
    validation_scorecard = json.loads(_require(validation_run_dir / "validation_scorecard.json").read_text(encoding="utf-8"))

    if state_scorecard.height == 0:
        summary = {
            "generated_ts": datetime.now(timezone.utc).isoformat(),
            "validation_run_dir": str(validation_run_dir),
            "states_total": 0,
            "states_flagged": 0,
            "issue_counts": {},
            "validation_scorecard": validation_scorecard,
            "thresholds": {
                "ret_cv_threshold": ret_cv_threshold,
                "min_n_rows": min_n_rows,
                "min_state_share": min_state_share,
                "sign_consistency_threshold": sign_consistency_threshold,
                "ci_width_quantile_threshold": ci_width_quantile_threshold,
                "eps": eps,
            },
        }
        return ClusterQADiagnostics(
            summary=summary,
            flagged_states=pl.DataFrame(schema={"state_id": pl.Int32, "issues": pl.String}),
            state_windows=pl.DataFrame(schema={"state_id": pl.Int32}),
        )

    ci_quantile_raw = state_scorecard.select(
        pl.col("ci_width").cast(pl.Float64, strict=False).quantile(ci_width_quantile_threshold, interpolation="linear")
    ).item()
    ci_threshold = _safe_float(ci_quantile_raw)
    if ci_threshold is None:
        ci_threshold = float("inf")

    transition_counts = _transition_counts_by_state(transition_summary)
    transitions_sparse_threshold = max(10, int(min_n_rows // 5))
    near_zero_mean_threshold = max(0.001, 1000.0 * eps)

    state_windows_rows: list[dict[str, Any]] = []
    flagged_rows: list[dict[str, Any]] = []

    for row in state_scorecard.to_dicts():
        state_id = int(row.get("state_id"))
        n_rows = int(row.get("n_rows") or 0)
        state_share_mean = _safe_float(row.get("state_share_mean"))
        ret_mean_cv = _safe_float(row.get("ret_mean_cv"))
        ci_width = _safe_float(row.get("ci_width"))
        sign_consistency = _safe_float(row.get("stability_sign_consistency"))
        fwd_ret_10_mean = _safe_float(row.get("fwd_ret_10_mean"))

        issues: list[str] = []
        if n_rows < min_n_rows:
            issues.append("LOW_N")
        if state_share_mean is not None and state_share_mean < min_state_share:
            issues.append("LOW_OCCUPANCY")
        if ret_mean_cv is not None and ret_mean_cv > ret_cv_threshold:
            issues.append("WINDOW_DRIFT_HIGH")
        if (
            ret_mean_cv is not None
            and ret_mean_cv > ret_cv_threshold
            and fwd_ret_10_mean is not None
            and abs(fwd_ret_10_mean) <= near_zero_mean_threshold
        ):
            issues.append("MEAN_NEAR_ZERO_CV_INFLATION")
        if ci_width is not None and ci_width >= ci_threshold:
            issues.append("WIDE_CI")
        if sign_consistency is not None and sign_consistency < sign_consistency_threshold:
            issues.append("SIGN_FLIP_ACROSS_WINDOWS")

        state_transition_count = transition_counts.get(state_id, 0)
        if state_transition_count < transitions_sparse_threshold:
            issues.append("TRANSITIONS_TOO_SPARSE")

        window_rows, likely_outlier_window = _state_window_diagnostics(state_id, rolling_metrics)
        state_windows_rows.extend(window_rows)
        if likely_outlier_window:
            issues.append("LIKELY_OUTLIER_WINDOW")

        if issues:
            flagged_rows.append(
                {
                    **row,
                    "state_id": state_id,
                    "transition_event_count": state_transition_count,
                    "issues": ",".join(sorted(set(issues))),
                }
            )

    flagged_states = pl.DataFrame(flagged_rows) if flagged_rows else pl.DataFrame(schema={"state_id": pl.Int32, "issues": pl.String})
    state_windows = pl.DataFrame(state_windows_rows) if state_windows_rows else pl.DataFrame(schema={"state_id": pl.Int32})

    summary = {
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "validation_run_dir": str(validation_run_dir),
        "states_total": int(state_scorecard.height),
        "states_flagged": int(flagged_states.height),
        "issue_counts": _issue_counts(flagged_states),
        "validation_scorecard": validation_scorecard,
        "thresholds": {
            "ret_cv_threshold": ret_cv_threshold,
            "min_n_rows": min_n_rows,
            "min_state_share": min_state_share,
            "sign_consistency_threshold": sign_consistency_threshold,
            "ci_width_quantile_threshold": ci_width_quantile_threshold,
            "ci_width_threshold_value": (None if not np.isfinite(ci_threshold) else ci_threshold),
            "transitions_sparse_threshold": transitions_sparse_threshold,
            "near_zero_mean_threshold": near_zero_mean_threshold,
            "eps": eps,
        },
    }

    return ClusterQADiagnostics(
        summary=summary,
        flagged_states=flagged_states,
        state_windows=state_windows,
    )


def run_cluster_qa_single(
    settings: AppSettings,
    *,
    validation_run_dir: Path,
    ret_cv_threshold: float | None = None,
    min_n_rows: int | None = None,
    min_state_share: float | None = None,
    sign_consistency_threshold: float | None = None,
    ci_width_quantile_threshold: float | None = None,
    eps: float | None = None,
    logger: logging.Logger | None = None,
) -> ClusterQASingleResult:
    """Run cluster QA diagnostics for one validation run directory."""

    effective_logger = logger or LOGGER
    cfg = settings.cluster_qa
    diagnostics = analyze_cluster_validation_run(
        validation_run_dir,
        ret_cv_threshold=ret_cv_threshold if ret_cv_threshold is not None else cfg.ret_cv_threshold,
        min_n_rows=min_n_rows if min_n_rows is not None else cfg.min_n_rows,
        min_state_share=min_state_share if min_state_share is not None else cfg.min_state_share,
        sign_consistency_threshold=(
            sign_consistency_threshold if sign_consistency_threshold is not None else cfg.sign_consistency_threshold
        ),
        ci_width_quantile_threshold=(
            ci_width_quantile_threshold
            if ci_width_quantile_threshold is not None
            else cfg.ci_width_quantile_threshold
        ),
        eps=eps if eps is not None else cfg.eps,
    )

    output_dir = validation_run_dir / "cluster_qa"
    paths: ClusterQASingleReportPaths = write_cluster_qa_single_reports(
        output_dir=output_dir,
        validation_run_dir=validation_run_dir,
        summary=diagnostics.summary,
        flagged_states=diagnostics.flagged_states,
        state_windows=diagnostics.state_windows,
    )

    effective_logger.info(
        "cluster_qa.single.complete validation_run_dir=%s flagged_states=%s output=%s",
        validation_run_dir,
        diagnostics.flagged_states.height,
        output_dir,
    )

    return ClusterQASingleResult(
        output_dir=output_dir,
        summary_path=paths.summary_path,
        flagged_states_path=paths.flagged_states_path,
        state_windows_path=paths.state_windows_path,
        report_path=paths.report_path,
    )


def run_cluster_qa_walkforward(
    settings: AppSettings,
    *,
    wf_run_dir: Path,
    ret_cv_threshold: float | None = None,
    min_n_rows: int | None = None,
    min_state_share: float | None = None,
    sign_consistency_threshold: float | None = None,
    ci_width_quantile_threshold: float | None = None,
    eps: float | None = None,
    logger: logging.Logger | None = None,
) -> ClusterQAWalkForwardResult:
    """Run cluster QA across all successful cluster validations in a walk-forward run."""

    effective_logger = logger or LOGGER
    manifest_path = _require(wf_run_dir / "wf_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    cfg = settings.cluster_qa
    resolved_thresholds = {
        "ret_cv_threshold": ret_cv_threshold if ret_cv_threshold is not None else cfg.ret_cv_threshold,
        "min_n_rows": min_n_rows if min_n_rows is not None else cfg.min_n_rows,
        "min_state_share": min_state_share if min_state_share is not None else cfg.min_state_share,
        "sign_consistency_threshold": (
            sign_consistency_threshold if sign_consistency_threshold is not None else cfg.sign_consistency_threshold
        ),
        "ci_width_quantile_threshold": (
            ci_width_quantile_threshold
            if ci_width_quantile_threshold is not None
            else cfg.ci_width_quantile_threshold
        ),
        "eps": eps if eps is not None else cfg.eps,
    }

    flagged_frames: list[pl.DataFrame] = []
    split_issue_rows: list[dict[str, Any]] = []
    splits_analyzed = 0
    splits_with_flags = 0

    for split in manifest.get("splits", []):
        if split.get("status") != "SUCCESS":
            continue
        val_cluster_dir = split.get("val_cluster_dir")
        if val_cluster_dir is None:
            continue
        validation_run_dir = Path(str(val_cluster_dir))
        if not (validation_run_dir / "validation_scorecard.json").exists():
            continue

        splits_analyzed += 1
        diagnostics = analyze_cluster_validation_run(
            validation_run_dir,
            **resolved_thresholds,
        )
        flagged = diagnostics.flagged_states
        if flagged.height > 0:
            splits_with_flags += 1
            flagged = flagged.with_columns(
                [
                    pl.lit(str(split.get("train_end"))).alias("train_end"),
                    pl.lit(str(validation_run_dir)).alias("validation_run_dir"),
                ]
            )
            flagged_frames.append(flagged)

            issue_counts = diagnostics.summary.get("issue_counts", {})
            for issue, count in issue_counts.items():
                split_issue_rows.append(
                    {
                        "train_end": split.get("train_end"),
                        "issue": issue,
                        "state_count": int(count),
                    }
                )

    flagged_states = pl.concat(flagged_frames, how="diagonal_relaxed") if flagged_frames else pl.DataFrame(schema={"state_id": pl.Int32})

    if split_issue_rows:
        issue_frequency = (
            pl.DataFrame(split_issue_rows)
            .group_by("issue")
            .agg(
                [
                    pl.col("train_end").n_unique().alias("split_count"),
                    pl.col("state_count").sum().alias("state_count"),
                ]
            )
            .sort("split_count", descending=True)
        )
    else:
        issue_frequency = pl.DataFrame(schema={"issue": pl.String, "split_count": pl.Int64, "state_count": pl.Int64})

    summary = {
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "wf_run_dir": str(wf_run_dir),
        "wf_run_id": manifest.get("wf_run_id"),
        "splits_analyzed": splits_analyzed,
        "cluster_splits_flagged": splits_with_flags,
        "total_flagged_states": int(flagged_states.height),
        "issue_frequency": issue_frequency.to_dicts(),
        "thresholds": resolved_thresholds,
    }

    output_dir = wf_run_dir / "cluster_qa"
    paths: ClusterQAWalkForwardReportPaths = write_cluster_qa_walkforward_reports(
        output_dir=output_dir,
        wf_run_dir=wf_run_dir,
        summary=summary,
        flagged_states=flagged_states,
        issue_frequency=issue_frequency,
    )

    effective_logger.info(
        "cluster_qa.walkforward.complete wf_run_dir=%s splits_analyzed=%s flagged_states=%s output=%s",
        wf_run_dir,
        splits_analyzed,
        flagged_states.height,
        output_dir,
    )

    return ClusterQAWalkForwardResult(
        output_dir=output_dir,
        summary_path=paths.summary_path,
        flagged_states_path=paths.flagged_states_path,
        issue_frequency_path=paths.issue_frequency_path,
        report_path=paths.report_path,
    )
