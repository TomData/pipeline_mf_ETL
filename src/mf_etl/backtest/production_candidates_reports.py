"""Report writers for Production Candidate Pack v1."""

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
    packet: dict[str, Any],
    table: pl.DataFrame,
    summary: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Production Candidate Pack v1")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- as_of: `{packet.get('as_of')}`")
    lines.append(f"- pcp_version: `{packet.get('pcp_version')}`")
    lines.append(f"- selected_candidates: {len(packet.get('candidates', {}))}")
    warnings = summary.get("warnings", [])
    lines.append(f"- warnings: {len(warnings)}")
    for warning in warnings[:10]:
        lines.append(f"  - {warning}")
    lines.append("")

    lines.append("## Locked Configs")
    lines.append("| label | combo_id | signal_mode | exit_mode | hold_bars | fee_bps | slippage_bps | overlay_mode | exec_profile | trade_count | expectancy | PF | robustness_v2 | ret_cv |")
    lines.append("|---|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|")
    for row in table.to_dicts():
        lines.append(
            f"| {row.get('label')} | {row.get('combo_id')} | {row.get('signal_mode')} | "
            f"{row.get('exit_mode')} | {row.get('hold_bars')} | {row.get('fee_bps_per_side')} | "
            f"{row.get('slippage_bps_per_side')} | {row.get('overlay_mode')} | {row.get('execution_profile')} | "
            f"{row.get('trade_count')} | {row.get('expectancy')} | {row.get('profit_factor')} | "
            f"{row.get('robustness_score_v2')} | {row.get('ret_cv')} |"
        )
    lines.append("")

    lines.append("## Walk-Forward Support")
    wf = summary.get("wf_consistency", {})
    if wf:
        for key, value in wf.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- unavailable")
    lines.append("")

    lines.append("## Operational Recommendation")
    lines.append("- `CANDIDATE_ALPHA` for raw alpha lens / benchmark behavior.")
    lines.append("- `CANDIDATE_EXEC` for execution-focused mode with overlay gating.")
    if "CANDIDATE_EXEC_2" in packet.get("candidates", {}):
        lines.append("- `CANDIDATE_EXEC_2` as realism control (baseline lite).")
    lines.append("")

    lines.append("## Student Note")
    lines.append("- Higher edge candidates can be less tradable under stricter realism.")
    lines.append("- Overlay gating often improves quality metrics while reducing trade count.")
    lines.append("")

    lines.append("## Re-Run Instructions")
    for label, candidate in packet.get("candidates", {}).items():
        strategy = candidate.get("strategy_params", {})
        overlay = candidate.get("overlay", {})
        execution = candidate.get("execution_realism", {})
        lines.append(f"### {label}")
        lines.append("```bash")
        cmd = [
            "python -m mf_etl.cli backtest-run",
            f"--input-type {candidate.get('input_type', 'hmm')}",
            f"--input-file {candidate.get('input_file')}",
            f"--signal-mode {strategy.get('signal_mode')}",
            f"--exit-mode {strategy.get('exit_mode')}",
            f"--hold-bars {strategy.get('hold_bars')}",
            f"--fee-bps-per-side {strategy.get('fee_bps_per_side')}",
            f"--slippage-bps-per-side {strategy.get('slippage_bps_per_side')}",
            f"--exec-profile {execution.get('profile')}",
        ]
        validation_run_dir = candidate.get("validation_run_dir")
        if validation_run_dir:
            cmd.append(f"--validation-run-dir {validation_run_dir}")
        if overlay.get("enabled"):
            cmd.extend(
                [
                    f"--overlay-cluster-file {overlay.get('overlay_cluster_file')}",
                    f"--overlay-cluster-hardening-dir {overlay.get('overlay_cluster_hardening_dir')}",
                    f"--overlay-mode {overlay.get('mode')}",
                ]
            )
        lines.append(" \\\n  ".join(cmd))
        lines.append("```")
    lines.append("")

    return "\n".join(lines)


def write_production_candidate_reports(
    *,
    output_dir: Path,
    packet: dict[str, Any],
    table: pl.DataFrame,
    summary: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    """Write PCP v1 JSON/CSV/markdown artifacts atomically."""

    output_dir.mkdir(parents=True, exist_ok=True)
    policy_packet_path = output_dir / "production_policy_packet_v1.json"
    candidates_table_path = output_dir / "production_candidates_table.csv"
    summary_path = output_dir / "production_candidates_summary.json"
    report_path = output_dir / "production_candidate_pack_report.md"

    write_json_atomically(packet, policy_packet_path)
    write_csv_atomically(table, candidates_table_path)
    write_json_atomically(summary, summary_path)
    write_markdown_atomically(
        _render_report(packet=packet, table=table, summary=summary),
        report_path,
    )
    return policy_packet_path, candidates_table_path, summary_path, report_path
