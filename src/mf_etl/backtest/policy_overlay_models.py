"""Typed models for policy overlay (hybrid gating) in backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

OverlayMode = Literal["none", "allow_only", "allow_watch", "block_veto", "allow_or_unknown"]


@dataclass(frozen=True, slots=True)
class PolicyOverlayResult:
    """Result bundle for row-level overlay join and gating."""

    frame: pl.DataFrame
    join_summary: dict[str, object]
    coverage_by_year: pl.DataFrame
    duplicate_keys: pl.DataFrame
    policy_mix_on_primary: pl.DataFrame

