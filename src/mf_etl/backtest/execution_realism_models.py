"""Typed models for execution realism filtering in backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

ExecutionProfileName = Literal["none", "lite", "strict"]
DollarVolRollingMethod = Literal["median", "mean"]
VolInputUnitMode = Literal["auto", "decimal", "percent_points"]


@dataclass(frozen=True, slots=True)
class ExecutionRealismParams:
    """Resolved execution realism parameters used for one run."""

    profile_name: ExecutionProfileName
    min_price: float | None
    min_dollar_vol_20: float | None
    max_vol_pct: float | None
    min_history_bars_for_execution: int | None
    dollar_vol_window: int
    dollar_vol_rolling_method: DollarVolRollingMethod
    vol_input_unit_mode: VolInputUnitMode
    report_min_trades: int
    report_max_zero_trade_share: float
    report_max_ret_cv: float


@dataclass(frozen=True, slots=True)
class ExecutionRealismResult:
    """Execution realism filter output frame and diagnostics artifacts."""

    frame: pl.DataFrame
    summary: dict[str, object]
    by_reason: pl.DataFrame
    by_year: pl.DataFrame
    filters_enabled: bool
