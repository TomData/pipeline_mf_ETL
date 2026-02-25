"""Filtered export helpers for cluster hardening policy outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl


@dataclass(frozen=True, slots=True)
class ClusterHardeningExportResult:
    """Artifact paths and counts for filtered cluster-row exports."""

    output_dir: Path
    with_policy_path: Path
    tradable_path: Path
    watch_path: Path
    summary_path: Path
    by_state_path: Path
    source_rows: int
    joined_rows: int
    tradable_rows: int
    watch_rows: int


def _atomic_temp_path(target_path: Path) -> Path:
    return target_path.parent / f".{target_path.name}.{uuid4().hex}.tmp"


def _write_json_atomically(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(output_path)
    try:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return output_path


def _write_parquet_atomically(df: pl.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(output_path)
    try:
        df.write_parquet(temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return output_path


def _write_csv_atomically(df: pl.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(output_path)
    try:
        df.write_csv(temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return output_path


def _read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    raise ValueError(f"Unsupported clustered rows file format: {path}")


def export_cluster_rows_with_policy(
    *,
    clustered_rows_file: Path,
    state_policy_table: pl.DataFrame,
    output_dir: Path,
) -> ClusterHardeningExportResult:
    """Join clustered rows with policy and write ALLOW/WATCH filtered exports."""

    rows_df = _read_table(clustered_rows_file)
    if "cluster_id" not in rows_df.columns:
        raise ValueError(f"Clustered rows file missing required column cluster_id: {clustered_rows_file}")

    required_policy_cols = {"state_id", "class_label", "tradability_score", "allow_direction_hint", "reasons"}
    missing_policy = sorted(required_policy_cols - set(state_policy_table.columns))
    if missing_policy:
        rendered = ", ".join(missing_policy)
        raise ValueError(f"State policy table missing required columns: {rendered}")

    policy_join = state_policy_table.select(
        [
            pl.col("state_id").cast(pl.Int64).alias("cluster_id"),
            pl.col("class_label").alias("tradable_state_class"),
            pl.col("tradability_score").alias("tradable_state_score"),
            pl.col("allow_direction_hint").alias("tradable_direction_hint"),
            pl.col("reasons").alias("policy_reason_flags"),
        ]
    )
    joined = rows_df.with_columns(pl.col("cluster_id").cast(pl.Int64)).join(policy_join, on="cluster_id", how="left")
    joined = joined.with_columns(
        [
            pl.col("tradable_state_class").fill_null("UNMAPPED"),
            pl.col("tradable_direction_hint").fill_null("UNCONFIRMED"),
        ]
    )

    tradable = joined.filter(pl.col("tradable_state_class") == "ALLOW")
    watch = joined.filter(pl.col("tradable_state_class") == "WATCH")

    by_state = (
        joined.group_by(["cluster_id", "tradable_state_class", "tradable_direction_hint"])
        .len(name="row_count")
        .sort(["tradable_state_class", "row_count"], descending=[False, True])
    )
    by_class = (
        joined.group_by("tradable_state_class")
        .len(name="row_count")
        .sort("row_count", descending=True)
        .to_dicts()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with_policy_path = output_dir / "clustered_rows_with_policy.parquet"
    tradable_path = output_dir / "clustered_rows_tradable.parquet"
    watch_path = output_dir / "clustered_rows_watch.parquet"
    summary_path = output_dir / "cluster_hardening_export_summary.json"
    by_state_path = output_dir / "cluster_hardening_export_by_state.csv"

    _write_parquet_atomically(joined, with_policy_path)
    _write_parquet_atomically(tradable, tradable_path)
    _write_parquet_atomically(watch, watch_path)
    _write_csv_atomically(by_state, by_state_path)

    summary_payload = {
        "clustered_rows_file": str(clustered_rows_file),
        "source_rows": int(rows_df.height),
        "joined_rows": int(joined.height),
        "tradable_rows": int(tradable.height),
        "watch_rows": int(watch.height),
        "class_counts": by_class,
        "outputs": {
            "with_policy": str(with_policy_path),
            "tradable": str(tradable_path),
            "watch": str(watch_path),
            "by_state": str(by_state_path),
        },
    }
    _write_json_atomically(summary_payload, summary_path)

    return ClusterHardeningExportResult(
        output_dir=output_dir,
        with_policy_path=with_policy_path,
        tradable_path=tradable_path,
        watch_path=watch_path,
        summary_path=summary_path,
        by_state_path=by_state_path,
        source_rows=rows_df.height,
        joined_rows=joined.height,
        tradable_rows=tradable.height,
        watch_rows=watch.height,
    )

