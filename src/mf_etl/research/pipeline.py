"""Research baseline clustering pipelines and sweep orchestration."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from mf_etl.config import AppSettings
from mf_etl.research.clustering import run_clustering
from mf_etl.research.dataset_loader import LoadedDataset, load_research_dataset
from mf_etl.research.forward_labels import add_forward_outcomes
from mf_etl.research.preprocess import preprocess_for_clustering
from mf_etl.research.profiles import ClusterProfiles, build_cluster_profiles

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResearchClusterRunResult:
    """Research clustering run artifact outputs."""

    run_id: str
    output_dir: Path
    run_summary_path: Path
    profile_path: Path
    metrics_path: Path


@dataclass(frozen=True, slots=True)
class ResearchClusterSweepResult:
    """Cluster sweep artifact outputs."""

    run_id: str
    summary_json_path: Path
    summary_csv_path: Path
    rows: int


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


def _build_clustered_df(
    base_df: pl.DataFrame,
    labels: np.ndarray,
    extra_columns: dict[str, np.ndarray],
) -> pl.DataFrame:
    clustered = base_df.with_columns(pl.Series(name="cluster_id", values=labels.astype(np.int32)))
    for name, values in extra_columns.items():
        clustered = clustered.with_columns(pl.Series(name=name, values=values))
    return clustered


def _prepare_research_input(
    dataset_path: Path,
    *,
    date_from: date | None,
    date_to: date | None,
    sample_frac: float | None,
    settings: AppSettings,
    logger: logging.Logger,
) -> tuple[LoadedDataset, pl.DataFrame]:
    loaded = load_research_dataset(
        dataset_path,
        date_from=date_from,
        date_to=date_to,
        sample_frac=sample_frac,
        logger=logger,
    )
    if "close" not in loaded.frame.columns:
        raise ValueError("Dataset must include close for forward-return validation.")
    with_forward = add_forward_outcomes(loaded.frame, windows=settings.research_clustering.forward_windows)
    return loaded, with_forward


def run_research_cluster(
    settings: AppSettings,
    *,
    dataset_path: Path,
    method: str,
    n_clusters: int,
    sample_frac: float | None,
    date_from: date | None,
    date_to: date | None,
    random_state: int,
    write_full_clustered: bool,
    logger: logging.Logger | None = None,
) -> ResearchClusterRunResult:
    """Run one clustering pipeline and emit research artifacts."""

    effective_logger = logger or LOGGER
    started_ts = datetime.now(timezone.utc)
    loaded, with_forward = _prepare_research_input(
        dataset_path,
        date_from=date_from,
        date_to=date_to,
        sample_frac=sample_frac,
        settings=settings,
        logger=effective_logger,
    )

    preprocess_result = preprocess_for_clustering(
        with_forward,
        feature_list=settings.research_clustering.default_feature_list,
        scaler=settings.research_clustering.scaler,
        clip_zscore=settings.research_clustering.clip_zscore,
    )

    clustering_result = run_clustering(
        preprocess_result.X,
        method=method,
        n_clusters=n_clusters,
        random_state=random_state,
        silhouette_sample_max=settings.research_clustering.silhouette_sample_max,
        kmeans_n_init=settings.research_clustering.kmeans.n_init,
        kmeans_max_iter=settings.research_clustering.kmeans.max_iter,
        gmm_covariance_type=settings.research_clustering.gmm.covariance_type,
        gmm_reg_covar=settings.research_clustering.gmm.reg_covar,
        gmm_max_iter=settings.research_clustering.gmm.max_iter,
    )
    clustered_df = _build_clustered_df(
        preprocess_result.processed_df,
        clustering_result.labels,
        clustering_result.extra_columns,
    )
    profiles: ClusterProfiles = build_cluster_profiles(clustered_df)

    run_id = f"research-{uuid4().hex[:12]}"
    dataset_tag = dataset_path.parent.name.replace(" ", "_")
    output_dir = settings.paths.artifacts_root / "research_runs" / f"{run_id}_{method}_{dataset_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary_path = output_dir / "run_summary.json"
    preprocess_summary_path = output_dir / "preprocess_summary.json"
    metrics_path = output_dir / "clustering_metrics.json"
    profile_parquet_path = output_dir / "cluster_profile.parquet"
    profile_csv_path = output_dir / "cluster_profile.csv"
    state_distribution_path = output_dir / "cluster_state_distribution.parquet"
    forward_validation_path = output_dir / "cluster_forward_validation.parquet"
    sample_path = output_dir / "clustered_dataset_sample.parquet"
    scaler_params_path = output_dir / "scaler_params.json"
    feature_list_path = output_dir / "feature_list.json"

    _write_parquet_atomically(profiles.cluster_profile, profile_parquet_path)
    _write_csv_atomically(profiles.cluster_profile, profile_csv_path)
    _write_parquet_atomically(profiles.cluster_state_distribution, state_distribution_path)
    _write_parquet_atomically(profiles.cluster_forward_validation, forward_validation_path)
    _write_parquet_atomically(clustered_df.head(100_000), sample_path)
    _write_json_atomically(preprocess_result.preprocess_summary, preprocess_summary_path)
    _write_json_atomically(preprocess_result.scaler_params, scaler_params_path)
    _write_json_atomically({"feature_list": preprocess_result.feature_list}, feature_list_path)

    full_clustered_path: str | None = None
    if write_full_clustered:
        full_path = output_dir / "clustered_dataset_full.parquet"
        _write_parquet_atomically(clustered_df, full_path)
        full_clustered_path = str(full_path)

    metrics_payload = {
        "method": method,
        "n_clusters_requested": n_clusters,
        "metrics": clustering_result.metrics,
        "model_meta": clustering_result.model_meta,
    }
    _write_json_atomically(metrics_payload, metrics_path)

    finished_ts = datetime.now(timezone.utc)
    duration_sec = (finished_ts - started_ts).total_seconds()
    run_summary = {
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "dataset_metadata_path": loaded.stats.get("metadata_path"),
        "started_ts": started_ts.isoformat(),
        "finished_ts": finished_ts.isoformat(),
        "duration_sec": round(duration_sec, 3),
        "rows_loaded": loaded.frame.height,
        "rows_used": preprocess_result.processed_df.height,
        "features_count": len(preprocess_result.feature_list),
        "method": method,
        "n_clusters_requested": n_clusters,
        "random_state": random_state,
        "metrics": clustering_result.metrics,
        "output_dir": str(output_dir),
        "outputs": {
            "run_summary_path": str(run_summary_path),
            "preprocess_summary_path": str(preprocess_summary_path),
            "clustering_metrics_path": str(metrics_path),
            "cluster_profile_parquet": str(profile_parquet_path),
            "cluster_profile_csv": str(profile_csv_path),
            "cluster_state_distribution_parquet": str(state_distribution_path),
            "cluster_forward_validation_parquet": str(forward_validation_path),
            "clustered_dataset_sample_parquet": str(sample_path),
            "scaler_params_path": str(scaler_params_path),
            "feature_list_path": str(feature_list_path),
            "clustered_dataset_full_path": full_clustered_path,
        },
    }
    _write_json_atomically(run_summary, run_summary_path)
    effective_logger.info(
        "research_cluster_run.complete method=%s rows_used=%s features=%s output_dir=%s",
        method,
        preprocess_result.processed_df.height,
        len(preprocess_result.feature_list),
        output_dir,
    )
    return ResearchClusterRunResult(
        run_id=run_id,
        output_dir=output_dir,
        run_summary_path=run_summary_path,
        profile_path=profile_parquet_path,
        metrics_path=metrics_path,
    )


def run_research_cluster_sweep(
    settings: AppSettings,
    *,
    dataset_path: Path,
    methods: list[str],
    n_clusters_values: list[int],
    sample_frac: float | None,
    date_from: date | None,
    date_to: date | None,
    random_state: int,
    logger: logging.Logger | None = None,
) -> ResearchClusterSweepResult:
    """Run a lightweight clustering sweep and write comparison artifacts."""

    effective_logger = logger or LOGGER
    loaded, with_forward = _prepare_research_input(
        dataset_path,
        date_from=date_from,
        date_to=date_to,
        sample_frac=sample_frac,
        settings=settings,
        logger=effective_logger,
    )
    preprocess_result = preprocess_for_clustering(
        with_forward,
        feature_list=settings.research_clustering.default_feature_list,
        scaler=settings.research_clustering.scaler,
        clip_zscore=settings.research_clustering.clip_zscore,
    )

    rows: list[dict[str, Any]] = []
    run_id = f"research-sweep-{uuid4().hex[:12]}"
    for method in methods:
        method_norm = method.strip().lower()
        cluster_grid = [max(2, int(n)) for n in n_clusters_values] if method_norm in {"kmeans", "gmm"} else [max(2, int(n_clusters_values[0]))]
        for n_clusters in cluster_grid:
            try:
                clustering_result = run_clustering(
                    preprocess_result.X,
                    method=method_norm,
                    n_clusters=n_clusters,
                    random_state=random_state,
                    silhouette_sample_max=settings.research_clustering.silhouette_sample_max,
                    kmeans_n_init=settings.research_clustering.kmeans.n_init,
                    kmeans_max_iter=settings.research_clustering.kmeans.max_iter,
                    gmm_covariance_type=settings.research_clustering.gmm.covariance_type,
                    gmm_reg_covar=settings.research_clustering.gmm.reg_covar,
                    gmm_max_iter=settings.research_clustering.gmm.max_iter,
                )
                clustered_df = _build_clustered_df(
                    preprocess_result.processed_df,
                    clustering_result.labels,
                    clustering_result.extra_columns,
                )
                profiles = build_cluster_profiles(clustered_df)
                separation_score: float | None = None
                if "fwd_ret_10_mean" in profiles.cluster_profile.columns and profiles.cluster_profile.height > 0:
                    values = profiles.cluster_profile.select(pl.col("fwd_ret_10_mean")).to_series().drop_nulls()
                    if values.len() > 0:
                        separation_score = float(values.max() - values.min())

                row = {
                    "method": method_norm,
                    "n_clusters": n_clusters,
                    "rows_used": preprocess_result.processed_df.height,
                    "features_count": len(preprocess_result.feature_list),
                    "silhouette": clustering_result.metrics.get("silhouette"),
                    "davies_bouldin": clustering_result.metrics.get("davies_bouldin"),
                    "calinski_harabasz": clustering_result.metrics.get("calinski_harabasz"),
                    "bic": clustering_result.metrics.get("bic"),
                    "aic": clustering_result.metrics.get("aic"),
                    "n_clusters_detected": clustering_result.metrics.get("n_clusters_detected"),
                    "forward_return_separation_score": separation_score,
                    "status": "ok",
                    "error": None,
                }
            except Exception as exc:
                row = {
                    "method": method_norm,
                    "n_clusters": n_clusters,
                    "rows_used": preprocess_result.processed_df.height,
                    "features_count": len(preprocess_result.feature_list),
                    "silhouette": None,
                    "davies_bouldin": None,
                    "calinski_harabasz": None,
                    "bic": None,
                    "aic": None,
                    "n_clusters_detected": None,
                    "forward_return_separation_score": None,
                    "status": "failed",
                    "error": str(exc),
                }
            rows.append(row)

    summary_df = pl.DataFrame(rows)
    out_dir = settings.paths.artifacts_root / "research_runs"
    summary_json_path = out_dir / f"{run_id}_cluster_sweep_summary.json"
    summary_csv_path = out_dir / f"{run_id}_cluster_sweep_summary.csv"
    _write_json_atomically(
        {
            "run_id": run_id,
            "dataset_path": str(dataset_path),
            "rows_loaded": loaded.frame.height,
            "rows_used": preprocess_result.processed_df.height,
            "features_count": len(preprocess_result.feature_list),
            "results": summary_df.to_dicts(),
        },
        summary_json_path,
    )
    _write_csv_atomically(summary_df, summary_csv_path)
    return ResearchClusterSweepResult(
        run_id=run_id,
        summary_json_path=summary_json_path,
        summary_csv_path=summary_csv_path,
        rows=summary_df.height,
    )

