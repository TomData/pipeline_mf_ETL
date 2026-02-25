"""Typed models for Candidate Re-run Pack v1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CandidateRerunPackResult:
    """Artifact paths for one CRP v1 execution."""

    run_id: str
    output_dir: Path
    manifest_path: Path
    candidates_table_path: Path
    summary_path: Path
    report_path: Path

