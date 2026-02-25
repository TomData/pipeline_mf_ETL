"""Typed runtime models for backtest sensitivity orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mf_etl.backtest.models import EquityMode, ExitMode, InputType, OverlayMode, SignalMode

PolicyFilterMode = Literal["allow_only", "allow_watch", "all_states"]


@dataclass(frozen=True, slots=True)
class SourceInputSpec:
    """Source-specific input pointers and optional mapping artifacts."""

    source_type: InputType
    input_file: Path
    validation_run_dir: Path | None = None
    cluster_hardening_dir: Path | None = None
    state_map_file: Path | None = None
    policy_filter_mode: PolicyFilterMode = "allow_only"
    overlay_cluster_file: Path | None = None
    overlay_cluster_hardening_dir: Path | None = None
    overlay_mode: OverlayMode = "none"
    overlay_join_keys: list[str] | None = None


@dataclass(frozen=True, slots=True)
class GridDimensionValues:
    """Resolved value lists for grid dimensions."""

    hold_bars: list[int]
    signal_mode: list[SignalMode]
    exit_mode: list[ExitMode]
    fee_bps_per_side: list[float]
    slippage_bps_per_side: list[float]
    allow_overlap: list[bool]
    equity_mode: list[EquityMode]
    include_watch: list[bool]
    include_state_sets: list[list[int]]


@dataclass(frozen=True, slots=True)
class GridComboSpec:
    """One Cartesian combination of backtest parameters."""

    hold_bars: int
    signal_mode: SignalMode
    exit_mode: ExitMode
    fee_bps_per_side: float
    slippage_bps_per_side: float
    allow_overlap: bool
    equity_mode: EquityMode
    include_watch: bool
    include_state_ids: list[int]
    state_subset_key: str | None


@dataclass(frozen=True, slots=True)
class GridRunResult:
    """Artifact paths produced by a sensitivity grid run."""

    grid_run_id: str
    output_dir: Path
    config_path: Path
    manifest_path: Path
    metrics_table_path: Path
    summary_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class GridCompareResult:
    """Artifacts produced by grid-vs-grid comparison."""

    compare_id: str
    output_dir: Path
    summary_path: Path
    table_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class GridWalkForwardResult:
    """Artifacts produced by walk-forward grid sensitivity orchestration."""

    wf_grid_id: str
    output_dir: Path
    manifest_path: Path
    by_split_path: Path
    config_aggregate_path: Path
    source_summary_path: Path
    report_path: Path
