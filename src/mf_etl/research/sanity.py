"""Sanity/inspection helpers for completed research clustering runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

FORWARD_AGGREGATE_BASES: tuple[str, ...] = (
    "fwd_ret_5",
    "fwd_ret_10",
    "fwd_ret_20",
    "fwd_abs_ret_10",
    "fwd_vol_proxy_10",
)


def summarize_research_run(run_dir: Path) -> dict[str, Any]:
    """Read a research run folder and produce concise diagnostics summary."""

    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Research run directory not found: {run_dir}")

    run_summary_path = run_dir / "run_summary.json"
    preprocess_summary_path = run_dir / "preprocess_summary.json"
    metrics_path = run_dir / "clustering_metrics.json"
    profile_path = run_dir / "cluster_profile.parquet"

    if not run_summary_path.exists():
        raise FileNotFoundError(f"run_summary.json missing in {run_dir}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"clustering_metrics.json missing in {run_dir}")
    if not profile_path.exists():
        raise FileNotFoundError(f"cluster_profile.parquet missing in {run_dir}")

    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    preprocess_summary = (
        json.loads(preprocess_summary_path.read_text(encoding="utf-8"))
        if preprocess_summary_path.exists()
        else {}
    )
    clustering_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    profile_df = pl.read_parquet(profile_path)
    top_fwd_ret_10: list[dict[str, Any]] = []
    if "fwd_ret_10_mean" in profile_df.columns:
        top_fwd_ret_10 = (
            profile_df.select(["cluster_id", "row_count", "fwd_ret_10_mean"])
            .sort("fwd_ret_10_mean", descending=True, nulls_last=True)
            .head(5)
            .to_dicts()
        )

    top_flow_activity: list[dict[str, Any]] = []
    if "flow_activity_20_mean" in profile_df.columns:
        top_flow_activity = (
            profile_df.select(["cluster_id", "row_count", "flow_activity_20_mean"])
            .sort("flow_activity_20_mean", descending=True, nulls_last=True)
            .head(5)
            .to_dicts()
        )

    forward_aggregate_nan_summary: dict[str, dict[str, int]] = {}
    for base in FORWARD_AGGREGATE_BASES:
        for stat in ("mean", "median"):
            column = f"{base}_{stat}"
            if column not in profile_df.columns:
                continue
            nan_count = int(
                profile_df.select(pl.col(column).is_nan().fill_null(False).sum()).item()
            )
            null_count = int(profile_df.select(pl.col(column).is_null().sum()).item())
            forward_aggregate_nan_summary[column] = {
                "nan_count": nan_count,
                "null_count": null_count,
            }

    return {
        "run_dir": str(run_dir),
        "run_summary": run_summary,
        "clustering_metrics": clustering_metrics,
        "preprocess_summary": preprocess_summary,
        "top_clusters_by_fwd_ret_10_mean": top_fwd_ret_10,
        "top_clusters_by_flow_activity_20_mean": top_flow_activity,
        "forward_aggregate_nan_summary": forward_aggregate_nan_summary,
    }
