"""Report writers for Candidate Re-run Pack v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from mf_etl.backtest.writer import (
    write_csv_atomically,
    write_json_atomically,
    write_markdown_atomically,
)


def _render_report(
    *,
    rerun_manifest: dict[str, Any],
    candidate_table: pl.DataFrame,
    summary: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Candidate Re-run Pack v1")
    lines.append("")
    lines.append("## Pack Info")
    lines.append(f"- rerun_id: `{rerun_manifest.get('rerun_id')}`")
    lines.append(f"- as_of_tag: `{rerun_manifest.get('as_of_tag')}`")
    lines.append(f"- source_pcp_pack_dir: `{rerun_manifest.get('pcp_pack_dir')}`")
    lines.append(f"- started_ts: `{rerun_manifest.get('started_ts')}`")
    lines.append(f"- finished_ts: `{rerun_manifest.get('finished_ts')}`")
    lines.append("")

    lines.append("## Candidate Drift Table")
    lines.append("| candidate | combo_id | expected_exp | observed_exp | delta_exp_pct | expected_pf | observed_pf | delta_pf_pct | expected_ret_cv | observed_ret_cv | trade_count_exp | trade_count_obs | drift_status |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in candidate_table.sort("candidate_label").to_dicts():
        lines.append(
            f"| {row.get('candidate_label')} | {row.get('combo_id')} | "
            f"{row.get('expected_expectancy')} | {row.get('observed_expectancy')} | {row.get('delta_expectancy_pct')} | "
            f"{row.get('expected_profit_factor')} | {row.get('observed_profit_factor')} | {row.get('delta_profit_factor_pct')} | "
            f"{row.get('expected_ret_cv')} | {row.get('observed_ret_cv')} | "
            f"{row.get('expected_trade_count')} | {row.get('observed_trade_count')} | "
            f"{row.get('drift_status')} |"
        )
    lines.append("")

    lines.append("## Drift Flags")
    for row in candidate_table.sort("candidate_label").to_dicts():
        reasons = row.get("drift_reasons")
        lines.append(f"- {row.get('candidate_label')}: {row.get('drift_status')} | reasons={reasons}")
    lines.append("")

    lines.append("## WF Results")
    wf = summary.get("wf", {})
    if wf:
        lines.append(f"- wf_enabled: {wf.get('enabled')}")
        lines.append(f"- candidates_with_wf: {wf.get('candidates_with_wf')}")
        lines.append(f"- wf_single_combo_summary_csv: `{wf.get('wf_single_combo_summary_csv')}`")
    else:
        lines.append("- wf: not requested")
    lines.append("")

    lines.append("## Student Note")
    lines.append("- Drift is the deviation between expected policy snapshot metrics and current rerun metrics.")
    lines.append("- Re-running candidates catches silent regime/data changes before promoting configs.")
    lines.append("")

    return "\n".join(lines)


def write_candidate_rerun_reports(
    *,
    output_dir: Path,
    rerun_manifest: dict[str, Any],
    candidate_table: pl.DataFrame,
    summary: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    """Write CRP v1 outputs atomically."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "rerun_manifest.json"
    table_path = output_dir / "rerun_candidates_table.csv"
    summary_path = output_dir / "rerun_summary.json"
    report_path = output_dir / "rerun_report.md"

    write_json_atomically(rerun_manifest, manifest_path)
    write_csv_atomically(candidate_table, table_path)
    write_json_atomically(summary, summary_path)
    write_markdown_atomically(
        _render_report(
            rerun_manifest=rerun_manifest,
            candidate_table=candidate_table,
            summary=summary,
        ),
        report_path,
    )
    return manifest_path, table_path, summary_path, report_path

