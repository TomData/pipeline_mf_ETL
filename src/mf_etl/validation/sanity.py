"""Sanity utilities for completed validation harness runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _nan_count(df: pl.DataFrame) -> int:
    if df.height == 0:
        return 0
    numeric = [column for column, dtype in df.schema.items() if dtype.is_numeric()]
    if not numeric:
        return 0
    total = 0
    for column in numeric:
        total += int(df.select(pl.col(column).cast(pl.Float64, strict=False).is_nan().fill_null(False).sum()).item())
    return total


def summarize_validation_run(run_dir: Path) -> dict[str, Any]:
    """Read validation artifacts and return compact diagnostics summary."""

    run_summary = json.loads(_require(run_dir / "run_summary.json").read_text(encoding="utf-8"))
    validation_scorecard = json.loads(_require(run_dir / "validation_scorecard.json").read_text(encoding="utf-8"))
    state_scorecard = pl.read_csv(_require(run_dir / "state_scorecard.csv"))
    pairwise = pl.read_csv(_require(run_dir / "bootstrap_pairwise_diff.csv"))
    transition_summary = pl.read_csv(_require(run_dir / "transition_event_summary.csv"))
    stability = pl.read_csv(_require(run_dir / "state_stability_summary.csv"))

    top_states = []
    if state_scorecard.height > 0 and "fwd_ret_10_mean" in state_scorecard.columns:
        top_states = (
            state_scorecard.select(["state_id", "n_rows", "fwd_ret_10_mean", "fwd_ret_10_ci_lo", "fwd_ret_10_ci_hi", "confidence_score"])
            .sort("fwd_ret_10_mean", descending=True, nulls_last=True)
            .head(10)
            .to_dicts()
        )

    pairwise_share = None
    if pairwise.height > 0 and "diff_sign_consistent" in pairwise.columns:
        pairwise_share = float(pairwise.select(pl.col("diff_sign_consistent").cast(pl.Int32).mean()).item())

    top_transitions = []
    if transition_summary.height > 0 and "count_events" in transition_summary.columns:
        top_transitions = transition_summary.sort("count_events", descending=True, nulls_last=True).head(10).to_dicts()

    stability_highlights = []
    if stability.height > 0 and "ret_mean_cv" in stability.columns:
        stability_highlights = (
            stability.select(["state_id", "fwd_ret_10_mean_mean", "fwd_ret_10_sign_stability", "ret_mean_cv", "share_cv"])
            .sort("ret_mean_cv", descending=False, nulls_last=True)
            .head(10)
            .to_dicts()
        )

    nan_warnings = {
        "state_scorecard_nan_count": _nan_count(state_scorecard),
        "pairwise_nan_count": _nan_count(pairwise),
        "transition_summary_nan_count": _nan_count(transition_summary),
        "stability_nan_count": _nan_count(stability),
    }

    return {
        "run_dir": str(run_dir),
        "run_summary": run_summary,
        "validation_scorecard": validation_scorecard,
        "top_states_by_fwd_ret_10_mean": top_states,
        "pairwise_significant_diff_share": pairwise_share,
        "top_transition_codes": top_transitions,
        "state_stability_highlights": stability_highlights,
        "validation_grade": validation_scorecard.get("validation_grade"),
        "nan_warnings": nan_warnings,
    }
