"""Report writers for cluster QA diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl


@dataclass(frozen=True, slots=True)
class ClusterQASingleReportPaths:
    """Artifacts generated for single-run cluster QA."""

    summary_path: Path
    flagged_states_path: Path
    state_windows_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class ClusterQAWalkForwardReportPaths:
    """Artifacts generated for walk-forward cluster QA."""

    summary_path: Path
    flagged_states_path: Path
    issue_frequency_path: Path
    report_path: Path


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


def _write_markdown_atomically(text: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_temp_path(output_path)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, output_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return output_path


def _render_single_report(
    *,
    validation_run_dir: Path,
    summary: dict[str, Any],
    flagged_states: pl.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Cluster QA Report (Single Validation Run)")
    lines.append("")
    lines.append(f"- validation_run_dir: `{validation_run_dir}`")
    lines.append(f"- states_total: {summary.get('states_total')}")
    lines.append(f"- states_flagged: {summary.get('states_flagged')}")
    lines.append(f"- issue_counts: {summary.get('issue_counts')}")
    lines.append("")
    lines.append("## Flagged States")
    lines.append("| state_id | n_rows | state_share_mean | ret_mean_cv | ci_width | sign_consistency | issues |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for row in flagged_states.to_dicts():
        lines.append(
            f"| {row.get('state_id')} | {row.get('n_rows')} | {row.get('state_share_mean')} | "
            f"{row.get('ret_mean_cv')} | {row.get('ci_width')} | {row.get('stability_sign_consistency')} | {row.get('issues')} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Flags identify cluster states likely responsible for unstable return CV behavior.")
    lines.append("- Root causes are heuristic labels intended for fast QA triage.")
    return "\n".join(lines) + "\n"


def _render_wf_report(
    *,
    wf_run_dir: Path,
    summary: dict[str, Any],
    issue_frequency: pl.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Cluster QA Walk-Forward Report")
    lines.append("")
    lines.append(f"- wf_run_dir: `{wf_run_dir}`")
    lines.append(f"- splits_analyzed: {summary.get('splits_analyzed')}")
    lines.append(f"- cluster_splits_flagged: {summary.get('cluster_splits_flagged')}")
    lines.append(f"- total_flagged_states: {summary.get('total_flagged_states')}")
    lines.append("")
    lines.append("## Issue Frequency Across Splits")
    lines.append("| issue | split_count | state_count |")
    lines.append("|---|---:|---:|")
    for row in issue_frequency.to_dicts():
        lines.append(f"| {row.get('issue')} | {row.get('split_count')} | {row.get('state_count')} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Cluster IDs can permute across splits, so aggregation is by issue type frequency.")
    lines.append("- Recurrent issue types point to model-level instability patterns.")
    return "\n".join(lines) + "\n"


def write_cluster_qa_single_reports(
    *,
    output_dir: Path,
    validation_run_dir: Path,
    summary: dict[str, Any],
    flagged_states: pl.DataFrame,
    state_windows: pl.DataFrame,
) -> ClusterQASingleReportPaths:
    """Write single-run cluster QA artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "cluster_qa_summary.json"
    flagged_states_path = output_dir / "cluster_qa_flagged_states.csv"
    state_windows_path = output_dir / "cluster_qa_state_windows.csv"
    report_path = output_dir / "cluster_qa_report.md"

    _write_json_atomically(summary, summary_path)
    _write_csv_atomically(flagged_states, flagged_states_path)
    _write_csv_atomically(state_windows, state_windows_path)
    _write_markdown_atomically(
        _render_single_report(
            validation_run_dir=validation_run_dir,
            summary=summary,
            flagged_states=flagged_states,
        ),
        report_path,
    )

    return ClusterQASingleReportPaths(
        summary_path=summary_path,
        flagged_states_path=flagged_states_path,
        state_windows_path=state_windows_path,
        report_path=report_path,
    )


def write_cluster_qa_walkforward_reports(
    *,
    output_dir: Path,
    wf_run_dir: Path,
    summary: dict[str, Any],
    flagged_states: pl.DataFrame,
    issue_frequency: pl.DataFrame,
) -> ClusterQAWalkForwardReportPaths:
    """Write walk-forward cluster QA artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "cluster_qa_wf_summary.json"
    flagged_states_path = output_dir / "cluster_qa_wf_flagged_states.csv"
    issue_frequency_path = output_dir / "cluster_qa_issue_frequency.csv"
    report_path = output_dir / "cluster_qa_wf_report.md"

    _write_json_atomically(summary, summary_path)
    _write_csv_atomically(flagged_states, flagged_states_path)
    _write_csv_atomically(issue_frequency, issue_frequency_path)
    _write_markdown_atomically(
        _render_wf_report(
            wf_run_dir=wf_run_dir,
            summary=summary,
            issue_frequency=issue_frequency,
        ),
        report_path,
    )

    return ClusterQAWalkForwardReportPaths(
        summary_path=summary_path,
        flagged_states_path=flagged_states_path,
        issue_frequency_path=issue_frequency_path,
        report_path=report_path,
    )
