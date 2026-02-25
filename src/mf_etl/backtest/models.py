"""Typed models for backtest harness runtime configuration and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

InputType = Literal["flow", "hmm", "cluster"]
SignalMode = Literal["state_entry", "state_transition_entry", "state_persistence_confirm"]
ExitMode = Literal["horizon", "state_exit", "horizon_or_state"]
EquityMode = Literal["event_returns_only", "daily_equity_curve"]
DirectionHint = Literal["LONG_BIAS", "SHORT_BIAS", "UNCONFIRMED"]
PolicyFilterMode = Literal["allow_only", "allow_watch", "all_states"]
OverlayMode = Literal["none", "allow_only", "allow_watch", "block_veto", "allow_or_unknown"]
ExecutionProfileName = Literal["none", "lite", "strict"]


@dataclass(frozen=True, slots=True)
class BacktestRunConfig:
    """Resolved runtime config for one backtest run."""

    input_type: InputType
    input_file: Path
    signal_mode: SignalMode
    exit_mode: ExitMode
    hold_bars: int
    allow_overlap: bool
    allow_unconfirmed: bool
    include_watch: bool
    policy_filter_mode: PolicyFilterMode
    include_state_ids: list[int]
    overlay_cluster_file: Path | None
    overlay_cluster_hardening_dir: Path | None
    overlay_mode: OverlayMode
    overlay_join_keys: list[str]
    execution_profile: ExecutionProfileName
    exec_min_price: float | None
    exec_min_dollar_vol20: float | None
    exec_max_vol_pct: float | None
    exec_min_history_bars: int | None
    report_min_trades: int | None
    report_max_zero_trade_share: float | None
    report_max_ret_cv: float | None
    fee_bps_per_side: float
    slippage_bps_per_side: float
    equity_mode: EquityMode
    validation_run_dir: Path | None
    cluster_hardening_dir: Path | None
    state_map_file: Path | None
    export_joined_rows: bool
    tag: str | None


@dataclass(frozen=True, slots=True)
class NormalizedBacktestData:
    """Normalized input frame and adapter diagnostics."""

    frame: pl.DataFrame
    summary: dict[str, object]


@dataclass(frozen=True, slots=True)
class SignalResult:
    """Signal-enriched frame and diagnostics."""

    frame: pl.DataFrame
    diagnostics: dict[str, object]


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Per-symbol simulation result."""

    trades: pl.DataFrame
    signal_diagnostics: dict[str, object]


@dataclass(frozen=True, slots=True)
class BacktestRunResult:
    """Artifact paths for one backtest run."""

    run_id: str
    output_dir: Path
    summary_path: Path
    trades_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class BacktestCompareResult:
    """Artifact paths for compare output."""

    compare_id: str
    output_dir: Path
    summary_path: Path
    table_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class BacktestWalkForwardResult:
    """Artifact paths for walk-forward backtest output."""

    wf_bt_id: str
    output_dir: Path
    manifest_path: Path
    aggregate_summary_path: Path
    model_summary_path: Path
    report_path: Path
