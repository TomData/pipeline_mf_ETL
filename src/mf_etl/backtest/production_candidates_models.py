"""Typed models for Production Candidate Pack v1 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProductionCandidatePackResult:
    """Artifact paths for one production candidate pack build."""

    run_id: str
    output_dir: Path
    policy_packet_path: Path
    candidates_table_path: Path
    summary_path: Path
    report_path: Path

