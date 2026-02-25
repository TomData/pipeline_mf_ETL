"""Typed models for execution realism calibration workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CalibrationSourceType = Literal["hmm", "flow", "cluster", "auto"]
ResolvedCalibrationSourceType = Literal["hmm", "flow", "cluster"]


@dataclass(frozen=True, slots=True)
class ExecutionRealismCalibrationResult:
    """Artifact locations for one calibration run."""

    run_id: str
    output_dir: Path
    summary_path: Path
    distribution_path: Path
    waterfall_path: Path
    grid_path: Path
    recommendations_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class ExecutionRealismCalibrationReportResult:
    """Artifact locations for a standalone calibration report render."""

    calibration_dir: Path
    report_path: Path
    summary_path: Path
