"""Report writers for cluster hardening policies and summaries."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl


@dataclass(frozen=True, slots=True)
class ClusterHardeningSingleReportPaths:
    """Artifact paths for one single-run hardening result."""

    policy_path: Path
    state_table_path: Path
    summary_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class ClusterHardeningWFReportPaths:
    """Artifact paths for walk-forward hardening outputs."""

    wf_summary_path: Path
    wf_state_stats_path: Path
    split_counts_path: Path
    issue_frequency_path: Path
    threshold_recommendation_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class ClusterHardeningCompareReportPaths:
    """Artifact paths for hardening policy comparison outputs."""

    summary_path: Path
    table_path: Path


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


def _write_csv_atomically(df: pl.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(output_path)
    try:
        df.write_csv(temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return output_path


def _write_markdown_atomically(text: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(output_path)
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return output_path


def _render_single_report(
    *,
    validation_run_dir: Path,
    policy_payload: dict[str, Any],
    state_table: pl.DataFrame,
    export_summary: dict[str, Any] | None,
) -> str:
    summary = policy_payload.get("summary", {})
    thresholds = policy_payload.get("thresholds", {})
    lines: list[str] = []
    lines.append("# Cluster Hardening Report")
    lines.append("")
    lines.append(f"- validation_run_dir: `{validation_run_dir}`")
    lines.append(f"- total_states: {summary.get('total_states')}")
    lines.append(f"- ALLOW/WATCH/BLOCK: {summary.get('allow_count')}/{summary.get('watch_count')}/{summary.get('block_count')}")
    lines.append("")
    lines.append("## Thresholds")
    for key, value in thresholds.items():
        if key in {"weights", "penalties"}:
            continue
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## State Table")
    lines.append("| state_id | class | score | direction | n_rows | share | ret_cv | ci_width | sign_consistency | reasons |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|---:|---|")
    for row in state_table.sort(["class_label", "tradability_score"], descending=[False, True]).to_dicts():
        lines.append(
            f"| {row.get('state_id')} | {row.get('class_label')} | {row.get('tradability_score')} | "
            f"{row.get('allow_direction_hint')} | {row.get('n_rows')} | {row.get('state_share_mean')} | "
            f"{row.get('ret_mean_cv')} | {row.get('ci_width')} | {row.get('stability_sign_consistency')} | "
            f"{row.get('reasons')} |"
        )
    lines.append("")
    if export_summary is not None:
        lines.append("## Export Summary")
        lines.append(f"- source_rows: {export_summary.get('source_rows')}")
        lines.append(f"- tradable_rows: {export_summary.get('tradable_rows')}")
        lines.append(f"- watch_rows: {export_summary.get('watch_rows')}")
        lines.append(f"- class_counts: {export_summary.get('class_counts')}")
        lines.append("")
    lines.append("## Recommendation")
    lines.append("- Use `ALLOW` rows for tradable backtest baselines.")
    lines.append("- Treat `WATCH` rows as research-only until stability/CI improves.")
    lines.append("- Exclude `BLOCK` rows from tradable universes.")
    lines.append("")
    return "\n".join(lines)


def _render_wf_report(
    *,
    wf_run_dir: Path,
    wf_summary: dict[str, Any],
    split_counts: pl.DataFrame,
    issue_frequency: pl.DataFrame,
    threshold_recommendation: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Cluster Hardening Walk-Forward Report")
    lines.append("")
    lines.append(f"- wf_run_dir: `{wf_run_dir}`")
    lines.append(f"- splits_total: {wf_summary.get('splits_total')}")
    lines.append(f"- splits_successful: {wf_summary.get('splits_successful')}")
    lines.append("")
    lines.append("## Split Counts")
    lines.append("| train_end | allow_count | watch_count | block_count |")
    lines.append("|---|---:|---:|---:|")
    for row in split_counts.to_dicts():
        lines.append(
            f"| {row.get('train_end')} | {row.get('allow_count')} | {row.get('watch_count')} | {row.get('block_count')} |"
        )
    lines.append("")
    lines.append("## Issue Frequency")
    lines.append("| issue | split_count | state_count |")
    lines.append("|---|---:|---:|")
    for row in issue_frequency.to_dicts():
        lines.append(f"| {row.get('issue')} | {row.get('split_count')} | {row.get('state_count')} |")
    lines.append("")
    lines.append("## Threshold Recommendation")
    rec = threshold_recommendation.get("recommendations", {})
    for key, value in rec.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Practical Guidance")
    lines.append("- Recurrent WIDE_CI / SIGN_FLIP / DRIFT issues suggest stricter blocking.")
    lines.append("- If ALLOW counts are too low, use WATCH as analysis-only fallback.")
    lines.append("- Refit thresholds after major feature/event grammar revisions.")
    lines.append("")
    return "\n".join(lines)


def write_cluster_hardening_single_reports(
    *,
    output_dir: Path,
    validation_run_dir: Path,
    policy_payload: dict[str, Any],
    state_table: pl.DataFrame,
    export_summary: dict[str, Any] | None,
) -> ClusterHardeningSingleReportPaths:
    """Write single-run hardening policy artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "cluster_hardening_policy.json"
    state_table_path = output_dir / "cluster_hardening_state_table.csv"
    summary_path = output_dir / "cluster_hardening_summary.json"
    report_path = output_dir / "cluster_hardening_report.md"

    _write_json_atomically(policy_payload, policy_path)
    _write_csv_atomically(state_table, state_table_path)
    _write_json_atomically(policy_payload.get("summary", {}), summary_path)
    _write_markdown_atomically(
        _render_single_report(
            validation_run_dir=validation_run_dir,
            policy_payload=policy_payload,
            state_table=state_table,
            export_summary=export_summary,
        ),
        report_path,
    )
    return ClusterHardeningSingleReportPaths(
        policy_path=policy_path,
        state_table_path=state_table_path,
        summary_path=summary_path,
        report_path=report_path,
    )


def write_cluster_hardening_wf_reports(
    *,
    output_dir: Path,
    wf_run_dir: Path,
    wf_summary: dict[str, Any],
    wf_state_stats: pl.DataFrame,
    split_counts: pl.DataFrame,
    issue_frequency: pl.DataFrame,
    threshold_recommendation: dict[str, Any],
) -> ClusterHardeningWFReportPaths:
    """Write walk-forward hardening summary artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    wf_summary_path = output_dir / "cluster_hardening_wf_summary.json"
    wf_state_stats_path = output_dir / "cluster_hardening_wf_state_stats.csv"
    split_counts_path = output_dir / "cluster_hardening_wf_split_counts.csv"
    issue_frequency_path = output_dir / "cluster_hardening_wf_issue_frequency.csv"
    threshold_recommendation_path = output_dir / "cluster_hardening_threshold_recommendation.json"
    report_path = output_dir / "cluster_hardening_wf_report.md"

    _write_json_atomically(wf_summary, wf_summary_path)
    _write_csv_atomically(wf_state_stats, wf_state_stats_path)
    _write_csv_atomically(split_counts, split_counts_path)
    _write_csv_atomically(issue_frequency, issue_frequency_path)
    _write_json_atomically(threshold_recommendation, threshold_recommendation_path)
    _write_markdown_atomically(
        _render_wf_report(
            wf_run_dir=wf_run_dir,
            wf_summary=wf_summary,
            split_counts=split_counts,
            issue_frequency=issue_frequency,
            threshold_recommendation=threshold_recommendation,
        ),
        report_path,
    )
    return ClusterHardeningWFReportPaths(
        wf_summary_path=wf_summary_path,
        wf_state_stats_path=wf_state_stats_path,
        split_counts_path=split_counts_path,
        issue_frequency_path=issue_frequency_path,
        threshold_recommendation_path=threshold_recommendation_path,
        report_path=report_path,
    )


def write_cluster_hardening_compare_reports(
    *,
    output_dir: Path,
    summary_payload: dict[str, Any],
    compare_table: pl.DataFrame,
) -> ClusterHardeningCompareReportPaths:
    """Write compare artifacts for two hardening policies."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "cluster_hardening_compare_summary.json"
    table_path = output_dir / "cluster_hardening_compare_table.csv"
    _write_json_atomically(summary_payload, summary_path)
    _write_csv_atomically(compare_table, table_path)
    return ClusterHardeningCompareReportPaths(summary_path=summary_path, table_path=table_path)

