"""Sanity helpers for completed HMM baseline runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

FORWARD_LABEL_COLUMNS: tuple[str, ...] = (
    "fwd_ret_5",
    "fwd_ret_10",
    "fwd_ret_20",
    "fwd_abs_ret_10",
    "fwd_vol_proxy_10",
)


def summarize_hmm_run(run_dir: Path) -> dict[str, Any]:
    """Read HMM run artifacts and return compact diagnostics summary."""

    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"HMM run directory not found: {run_dir}")

    run_summary_path = run_dir / "run_summary.json"
    state_profile_path = run_dir / "hmm_state_profile.parquet"
    transition_matrix_path = run_dir / "transition_matrix.csv"
    dwell_stats_path = run_dir / "dwell_stats.csv"
    flow_crosstab_path = run_dir / "hmm_vs_flow_state_crosstab.csv"
    if not run_summary_path.exists():
        raise FileNotFoundError(f"run_summary.json missing in {run_dir}")
    if not state_profile_path.exists():
        raise FileNotFoundError(f"hmm_state_profile.parquet missing in {run_dir}")
    if not transition_matrix_path.exists():
        raise FileNotFoundError(f"transition_matrix.csv missing in {run_dir}")
    if not dwell_stats_path.exists():
        raise FileNotFoundError(f"dwell_stats.csv missing in {run_dir}")

    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    profile_df = pl.read_parquet(state_profile_path)
    transition_df = pl.read_csv(transition_matrix_path)
    dwell_df = pl.read_csv(dwell_stats_path)
    flow_crosstab_df = (
        pl.read_csv(flow_crosstab_path)
        if flow_crosstab_path.exists()
        else pl.DataFrame(schema={"hmm_state": pl.Int16, "flow_state_label": pl.String, "count": pl.Int64, "share_in_hmm_state": pl.Float64})
    )

    top_states_by_fwd_ret_10 = []
    if "fwd_ret_10_mean" in profile_df.columns:
        top_states_by_fwd_ret_10 = (
            profile_df.select(["hmm_state", "row_count", "fwd_ret_10_mean"])
            .sort("fwd_ret_10_mean", descending=True, nulls_last=True)
            .head(10)
            .to_dicts()
        )

    top_self_transition_probs = []
    if {"hmm_state_prev", "hmm_state", "transition_probability"}.issubset(transition_df.columns):
        top_self_transition_probs = (
            transition_df.filter(pl.col("hmm_state_prev") == pl.col("hmm_state"))
            .select(
                [
                    pl.col("hmm_state").alias("state"),
                    "transition_probability",
                ]
            )
            .sort("transition_probability", descending=True, nulls_last=True)
            .head(10)
            .to_dicts()
        )

    overlap_highlights = []
    if flow_crosstab_df.height > 0 and "share_in_hmm_state" in flow_crosstab_df.columns:
        overlap_highlights = (
            flow_crosstab_df.sort("share_in_hmm_state", descending=True, nulls_last=True)
            .group_by("hmm_state")
            .first()
            .sort("hmm_state")
            .to_dicts()
        )

    forward_nan_summary: dict[str, dict[str, int]] = {}
    for base in FORWARD_LABEL_COLUMNS:
        for stat in ("mean", "median"):
            column = f"{base}_{stat}"
            if column not in profile_df.columns:
                continue
            forward_nan_summary[column] = {
                "nan_count": int(profile_df.select(pl.col(column).is_nan().fill_null(False).sum()).item()),
                "null_count": int(profile_df.select(pl.col(column).is_null().sum()).item()),
            }

    return {
        "run_dir": str(run_dir),
        "run_summary": run_summary,
        "state_count": profile_df.height,
        "state_frequency_total_rows": int(profile_df.select(pl.col("row_count").sum()).item()) if "row_count" in profile_df.columns else None,
        "top_states_by_fwd_ret_10_mean": top_states_by_fwd_ret_10,
        "top_self_transition_probs": top_self_transition_probs,
        "dwell_stats_highlights": dwell_df.sort("dwell_mean", descending=True, nulls_last=True).head(10).to_dicts(),
        "overlap_highlights_vs_flow_states": overlap_highlights,
        "forward_aggregate_nan_summary": forward_nan_summary,
    }

