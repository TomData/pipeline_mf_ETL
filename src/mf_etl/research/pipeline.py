"""Research baseline clustering pipelines, split handling, and robustness sweeps."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from mf_etl.config import AppSettings
from mf_etl.research.clustering import run_clustering
from mf_etl.research.dataset_loader import LoadedDataset, load_research_dataset
from mf_etl.research.forward_labels import add_forward_outcomes
from mf_etl.research.preprocess import (
    PreprocessResult,
    fit_preprocess_model,
    transform_for_clustering,
)
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


@dataclass(frozen=True, slots=True)
class ResearchClusterStabilityResult:
    """Seed stability sweep artifact outputs."""

    run_id: str
    output_dir: Path
    stability_summary_path: Path
    stability_by_seed_path: Path
    pairwise_ari_path: Path


@dataclass(frozen=True, slots=True)
class PreparedResearchData:
    """Preprocessed matrices and metadata for clustering runs."""

    loaded: LoadedDataset
    with_forward: pl.DataFrame
    preprocess_fit: PreprocessResult
    preprocess_predict: PreprocessResult
    preprocess_train: PreprocessResult | None
    preprocess_test: PreprocessResult | None
    split_summary: dict[str, Any]
    split_mode: str
    fit_on: str
    predict_on: str
    scaler: str
    scaling_scope: str


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


def _frame_stats(df: pl.DataFrame) -> dict[str, Any]:
    if df.height == 0:
        return {
            "rows": 0,
            "tickers": 0,
            "min_trade_date": None,
            "max_trade_date": None,
        }
    ticker_count = int(df.select(pl.col("ticker").n_unique()).item()) if "ticker" in df.columns else 0
    min_trade_date = None
    max_trade_date = None
    if "trade_date" in df.columns:
        bounds = df.select(
            [
                pl.col("trade_date").min().alias("min_trade_date"),
                pl.col("trade_date").max().alias("max_trade_date"),
            ]
        ).to_dicts()[0]
        min_val = bounds.get("min_trade_date")
        max_val = bounds.get("max_trade_date")
        min_trade_date = min_val.isoformat() if min_val is not None else None
        max_trade_date = max_val.isoformat() if max_val is not None else None
    return {
        "rows": df.height,
        "tickers": ticker_count,
        "min_trade_date": min_trade_date,
        "max_trade_date": max_trade_date,
    }


def _resolve_split_and_scopes(
    *,
    split_mode: str,
    train_end: date | None,
    test_start: date | None,
    test_end: date | None,
    fit_on: str,
    predict_on: str | None,
) -> tuple[str, str, str, date | None, date | None]:
    split_mode_norm = split_mode.strip().lower()
    if split_mode_norm not in {"none", "time"}:
        raise ValueError("split_mode must be one of: none, time")

    fit_on_norm = fit_on.strip().lower()
    if fit_on_norm not in {"train", "all"}:
        raise ValueError("fit_on must be one of: train, all")

    predict_on_norm = predict_on.strip().lower() if predict_on is not None else ""
    if predict_on_norm and predict_on_norm not in {"test", "all"}:
        raise ValueError("predict_on must be one of: test, all")

    if split_mode_norm == "none":
        return split_mode_norm, "all", "all", None, None

    if train_end is None:
        raise ValueError("train_end is required when split_mode=time.")
    resolved_test_start = test_start or (train_end + timedelta(days=1))
    if test_end is not None and test_end < resolved_test_start:
        raise ValueError("test_end must be >= test_start.")
    if predict_on_norm == "":
        predict_on_norm = "test"
    return split_mode_norm, fit_on_norm, predict_on_norm, resolved_test_start, test_end


def _normalize_trade_date_column(df: pl.DataFrame) -> pl.DataFrame:
    if "trade_date" not in df.columns:
        raise ValueError("Dataset must include trade_date.")
    out = df.with_columns(pl.col("trade_date").cast(pl.Date, strict=False).alias("trade_date"))
    bad_dates = out.filter(pl.col("trade_date").is_null()).height
    if bad_dates > 0:
        raise ValueError(f"trade_date contains {bad_dates} unparsable rows.")
    return out


def _profile_forward_separation(profile: pl.DataFrame) -> float | None:
    if "fwd_ret_10_mean" not in profile.columns or profile.height == 0:
        return None
    values = profile.select(pl.col("fwd_ret_10_mean")).to_series().drop_nulls()
    if values.len() == 0:
        return None
    return float(values.max() - values.min())


def _label_distribution_metrics(labels: np.ndarray) -> dict[str, Any]:
    if labels.size == 0:
        return {
            "largest_cluster_share": None,
            "effective_cluster_count": None,
            "cluster_count_detected": 0,
        }
    unique, counts = np.unique(labels, return_counts=True)
    shares = counts / float(labels.shape[0])
    largest = float(np.max(shares))
    effective = float(1.0 / np.sum(np.square(shares)))
    valid_cluster_count = int(unique[unique >= 0].shape[0])
    return {
        "largest_cluster_share": largest,
        "effective_cluster_count": effective,
        "cluster_count_detected": valid_cluster_count,
    }


def _build_robustness_notes(
    *,
    silhouette: float | None,
    largest_cluster_share: float | None,
    forward_separation_score: float | None,
    ari_mean: float | None = None,
) -> list[str]:
    notes: list[str] = []
    if silhouette is None:
        notes.append("silhouette_unavailable")
    elif silhouette < 0.10:
        notes.append("weak_cluster_geometry")
    else:
        notes.append("cluster_geometry_ok")

    if largest_cluster_share is None:
        notes.append("cluster_concentration_unknown")
    elif largest_cluster_share > 0.70:
        notes.append("high_cluster_concentration")
    else:
        notes.append("cluster_concentration_ok")

    if forward_separation_score is None:
        notes.append("forward_separation_unavailable")
    elif abs(forward_separation_score) < 0.002:
        notes.append("weak_forward_separation")
    else:
        notes.append("forward_separation_ok")

    if ari_mean is not None:
        if ari_mean < 0.50:
            notes.append("low_seed_stability")
        else:
            notes.append("seed_stability_ok")

    return notes


def _prepare_preprocessed_data(
    settings: AppSettings,
    *,
    dataset_path: Path,
    sample_frac: float | None,
    date_from: date | None,
    date_to: date | None,
    split_mode: str | None,
    train_end: date | None,
    test_start: date | None,
    test_end: date | None,
    fit_on: str,
    predict_on: str | None,
    scaler: str | None,
    scaling_scope: str | None,
    logger: logging.Logger,
) -> PreparedResearchData:
    loaded, with_forward = _prepare_research_input(
        dataset_path,
        date_from=date_from,
        date_to=date_to,
        sample_frac=sample_frac,
        settings=settings,
        logger=logger,
    )
    with_forward = _normalize_trade_date_column(with_forward)

    split_mode_value = split_mode or settings.research_clustering.split_mode_default
    split_mode_norm, fit_on_norm, predict_on_norm, resolved_test_start, resolved_test_end = _resolve_split_and_scopes(
        split_mode=split_mode_value,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        fit_on=fit_on,
        predict_on=predict_on,
    )

    all_df = with_forward
    train_df: pl.DataFrame | None = None
    test_df: pl.DataFrame | None = None
    if split_mode_norm == "time":
        if train_end is None or resolved_test_start is None:
            raise ValueError("train_end/test_start resolution failed for split_mode=time.")
        train_df = all_df.filter(pl.col("trade_date") <= pl.lit(train_end))
        test_filter = pl.col("trade_date") >= pl.lit(resolved_test_start)
        if resolved_test_end is not None:
            test_filter = test_filter & (pl.col("trade_date") <= pl.lit(resolved_test_end))
        test_df = all_df.filter(test_filter)
        if train_df.height == 0:
            raise ValueError("Time split produced empty train set.")
        if test_df.height == 0:
            raise ValueError("Time split produced empty test set.")

    fit_source = all_df if fit_on_norm == "all" else (train_df if train_df is not None else all_df)
    predict_source = all_df if predict_on_norm == "all" else (test_df if test_df is not None else all_df)

    scaler_name = (scaler or settings.research_clustering.scaler).strip().lower()
    scaling_scope_name = (scaling_scope or settings.research_clustering.scaling_scope_default).strip().lower()
    preprocess_model = fit_preprocess_model(
        fit_source,
        feature_list=settings.research_clustering.default_feature_list,
        scaler=scaler_name,
        scaling_scope=scaling_scope_name,
    )
    preprocess_fit = transform_for_clustering(
        fit_source,
        model=preprocess_model,
        clip_zscore=settings.research_clustering.clip_zscore,
    )
    preprocess_predict = transform_for_clustering(
        predict_source,
        model=preprocess_model,
        clip_zscore=settings.research_clustering.clip_zscore,
    )
    preprocess_train = (
        transform_for_clustering(
            train_df,
            model=preprocess_model,
            clip_zscore=settings.research_clustering.clip_zscore,
        )
        if train_df is not None
        else None
    )
    preprocess_test = (
        transform_for_clustering(
            test_df,
            model=preprocess_model,
            clip_zscore=settings.research_clustering.clip_zscore,
        )
        if test_df is not None
        else None
    )

    split_summary = {
        "split_mode": split_mode_norm,
        "fit_on": fit_on_norm,
        "predict_on": predict_on_norm,
        "train_end": train_end.isoformat() if train_end is not None else None,
        "test_start": resolved_test_start.isoformat() if resolved_test_start is not None else None,
        "test_end": resolved_test_end.isoformat() if resolved_test_end is not None else None,
        "raw_stats": {
            "all": _frame_stats(all_df),
            "fit_source": _frame_stats(fit_source),
            "predict_source": _frame_stats(predict_source),
            "train": _frame_stats(train_df) if train_df is not None else None,
            "test": _frame_stats(test_df) if test_df is not None else None,
        },
        "processed_stats": {
            "fit_source": _frame_stats(preprocess_fit.processed_df),
            "predict_source": _frame_stats(preprocess_predict.processed_df),
            "train": _frame_stats(preprocess_train.processed_df) if preprocess_train is not None else None,
            "test": _frame_stats(preprocess_test.processed_df) if preprocess_test is not None else None,
        },
        "preprocess_scaler": preprocess_fit.preprocess_summary.get("scaler"),
        "preprocess_scaling_scope": preprocess_fit.preprocess_summary.get("scaling_scope"),
    }

    if preprocess_predict.preprocess_summary.get("rows_dropped_unseen_tickers", 0) > 0:
        logger.warning(
            "preprocess.transform dropped unseen tickers rows=%s sample=%s",
            preprocess_predict.preprocess_summary.get("rows_dropped_unseen_tickers"),
            preprocess_predict.preprocess_summary.get("unseen_tickers_sample"),
        )

    return PreparedResearchData(
        loaded=loaded,
        with_forward=with_forward,
        preprocess_fit=preprocess_fit,
        preprocess_predict=preprocess_predict,
        preprocess_train=preprocess_train,
        preprocess_test=preprocess_test,
        split_summary=split_summary,
        split_mode=split_mode_norm,
        fit_on=fit_on_norm,
        predict_on=predict_on_norm,
        scaler=scaler_name,
        scaling_scope=scaling_scope_name,
    )


def _build_cluster_profiles_for_set(
    preprocess_result: PreprocessResult,
    labels: np.ndarray,
    extra_columns: dict[str, np.ndarray],
) -> tuple[pl.DataFrame, ClusterProfiles]:
    clustered_df = _build_clustered_df(preprocess_result.processed_df, labels, extra_columns)
    profiles = build_cluster_profiles(clustered_df)
    return clustered_df, profiles


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
    split_mode: str | None = None,
    train_end: date | None = None,
    test_start: date | None = None,
    test_end: date | None = None,
    fit_on: str = "train",
    predict_on: str | None = None,
    scaler: str | None = None,
    scaling_scope: str | None = None,
    logger: logging.Logger | None = None,
) -> ResearchClusterRunResult:
    """Run one clustering pipeline and emit artifacts (supports optional time OOS split)."""

    effective_logger = logger or LOGGER
    started_ts = datetime.now(timezone.utc)
    prepared = _prepare_preprocessed_data(
        settings,
        dataset_path=dataset_path,
        sample_frac=sample_frac,
        date_from=date_from,
        date_to=date_to,
        split_mode=split_mode,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        fit_on=fit_on,
        predict_on=predict_on,
        scaler=scaler,
        scaling_scope=scaling_scope,
        logger=effective_logger,
    )

    predict_sets: dict[str, np.ndarray] = {"predict": prepared.preprocess_predict.X}
    if prepared.preprocess_train is not None:
        predict_sets["train"] = prepared.preprocess_train.X
    if prepared.preprocess_test is not None:
        predict_sets["test"] = prepared.preprocess_test.X

    clustering_result = run_clustering(
        prepared.preprocess_fit.X,
        method=method,
        n_clusters=n_clusters,
        random_state=random_state,
        silhouette_sample_max=settings.research_clustering.silhouette_sample_max,
        kmeans_n_init=settings.research_clustering.kmeans.n_init,
        kmeans_max_iter=settings.research_clustering.kmeans.max_iter,
        gmm_covariance_type=settings.research_clustering.gmm.covariance_type,
        gmm_reg_covar=settings.research_clustering.gmm.reg_covar,
        gmm_max_iter=settings.research_clustering.gmm.max_iter,
        predict_sets=predict_sets,
        primary_set="predict",
    )

    clustered_predict_df, profiles = _build_cluster_profiles_for_set(
        prepared.preprocess_predict,
        clustering_result.labels_by_set["predict"],
        clustering_result.extra_columns_by_set.get("predict", {}),
    )

    train_profiles: ClusterProfiles | None = None
    test_profiles: ClusterProfiles | None = None
    if prepared.preprocess_train is not None and "train" in clustering_result.labels_by_set:
        _, train_profiles = _build_cluster_profiles_for_set(
            prepared.preprocess_train,
            clustering_result.labels_by_set["train"],
            clustering_result.extra_columns_by_set.get("train", {}),
        )
    if prepared.preprocess_test is not None and "test" in clustering_result.labels_by_set:
        _, test_profiles = _build_cluster_profiles_for_set(
            prepared.preprocess_test,
            clustering_result.labels_by_set["test"],
            clustering_result.extra_columns_by_set.get("test", {}),
        )

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
    split_summary_path = output_dir / "split_summary.json"
    sample_path = output_dir / "clustered_dataset_sample.parquet"
    scaler_params_path = output_dir / "scaler_params.json"
    scaler_params_table_path = output_dir / "scaler_params_per_ticker.parquet"
    feature_list_path = output_dir / "feature_list.json"
    robustness_summary_path = output_dir / "robustness_summary.json"

    _write_parquet_atomically(profiles.cluster_profile, profile_parquet_path)
    _write_csv_atomically(profiles.cluster_profile, profile_csv_path)
    _write_parquet_atomically(profiles.cluster_state_distribution, state_distribution_path)
    _write_parquet_atomically(profiles.cluster_forward_validation, forward_validation_path)
    _write_parquet_atomically(clustered_predict_df.head(100_000), sample_path)
    _write_json_atomically(prepared.preprocess_predict.preprocess_summary, preprocess_summary_path)
    _write_json_atomically(prepared.preprocess_predict.scaler_params, scaler_params_path)
    _write_json_atomically({"feature_list": prepared.preprocess_predict.feature_list}, feature_list_path)
    _write_json_atomically(prepared.split_summary, split_summary_path)
    if prepared.preprocess_predict.scaler_params_table is not None:
        _write_parquet_atomically(prepared.preprocess_predict.scaler_params_table, scaler_params_table_path)

    split_profile_paths: dict[str, str | None] = {
        "cluster_profile_train_parquet": None,
        "cluster_profile_train_csv": None,
        "cluster_profile_test_parquet": None,
        "cluster_profile_test_csv": None,
    }
    if train_profiles is not None:
        train_profile_parquet_path = output_dir / "cluster_profile_train.parquet"
        train_profile_csv_path = output_dir / "cluster_profile_train.csv"
        _write_parquet_atomically(train_profiles.cluster_profile, train_profile_parquet_path)
        _write_csv_atomically(train_profiles.cluster_profile, train_profile_csv_path)
        split_profile_paths["cluster_profile_train_parquet"] = str(train_profile_parquet_path)
        split_profile_paths["cluster_profile_train_csv"] = str(train_profile_csv_path)
    if test_profiles is not None:
        test_profile_parquet_path = output_dir / "cluster_profile_test.parquet"
        test_profile_csv_path = output_dir / "cluster_profile_test.csv"
        _write_parquet_atomically(test_profiles.cluster_profile, test_profile_parquet_path)
        _write_csv_atomically(test_profiles.cluster_profile, test_profile_csv_path)
        split_profile_paths["cluster_profile_test_parquet"] = str(test_profile_parquet_path)
        split_profile_paths["cluster_profile_test_csv"] = str(test_profile_csv_path)

    full_clustered_path: str | None = None
    if write_full_clustered:
        full_path = output_dir / "clustered_dataset_full.parquet"
        _write_parquet_atomically(clustered_predict_df, full_path)
        full_clustered_path = str(full_path)

    metrics_payload = {
        "method": method,
        "n_clusters_requested": n_clusters,
        "metrics": clustering_result.metrics,
        "model_meta": clustering_result.model_meta,
        "split_mode": prepared.split_mode,
        "fit_on": prepared.fit_on,
        "predict_on": prepared.predict_on,
        "scaler": prepared.scaler,
        "scaling_scope": prepared.scaling_scope,
    }
    _write_json_atomically(metrics_payload, metrics_path)

    label_metrics = _label_distribution_metrics(clustering_result.labels_by_set["predict"])
    forward_separation_score = _profile_forward_separation(profiles.cluster_profile)
    robustness_summary = {
        "method": method,
        "n_clusters_requested": n_clusters,
        "largest_cluster_share": label_metrics["largest_cluster_share"],
        "effective_cluster_count": label_metrics["effective_cluster_count"],
        "forward_separation_score": forward_separation_score,
        "silhouette": clustering_result.metrics.get("silhouette"),
        "davies_bouldin": clustering_result.metrics.get("davies_bouldin"),
        "calinski_harabasz": clustering_result.metrics.get("calinski_harabasz"),
        "robustness_notes": _build_robustness_notes(
            silhouette=clustering_result.metrics.get("silhouette"),
            largest_cluster_share=label_metrics["largest_cluster_share"],
            forward_separation_score=forward_separation_score,
        ),
    }
    _write_json_atomically(robustness_summary, robustness_summary_path)

    finished_ts = datetime.now(timezone.utc)
    duration_sec = (finished_ts - started_ts).total_seconds()
    run_summary = {
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "dataset_metadata_path": prepared.loaded.stats.get("metadata_path"),
        "started_ts": started_ts.isoformat(),
        "finished_ts": finished_ts.isoformat(),
        "duration_sec": round(duration_sec, 3),
        "split_mode": prepared.split_mode,
        "fit_on": prepared.fit_on,
        "predict_on": prepared.predict_on,
        "scaler": prepared.scaler,
        "scaling_scope": prepared.scaling_scope,
        "rows_loaded": prepared.loaded.frame.height,
        "rows_fit": prepared.preprocess_fit.processed_df.height,
        "rows_used": prepared.preprocess_predict.processed_df.height,
        "features_count": len(prepared.preprocess_predict.feature_list),
        "method": method,
        "n_clusters_requested": n_clusters,
        "random_state": random_state,
        "metrics": clustering_result.metrics,
        "output_dir": str(output_dir),
        "outputs": {
            "run_summary_path": str(run_summary_path),
            "preprocess_summary_path": str(preprocess_summary_path),
            "split_summary_path": str(split_summary_path),
            "clustering_metrics_path": str(metrics_path),
            "cluster_profile_parquet": str(profile_parquet_path),
            "cluster_profile_csv": str(profile_csv_path),
            "cluster_state_distribution_parquet": str(state_distribution_path),
            "cluster_forward_validation_parquet": str(forward_validation_path),
            "clustered_dataset_sample_parquet": str(sample_path),
            "scaler_params_path": str(scaler_params_path),
            "scaler_params_table_path": str(scaler_params_table_path) if prepared.preprocess_predict.scaler_params_table is not None else None,
            "feature_list_path": str(feature_list_path),
            "clustered_dataset_full_path": full_clustered_path,
            "robustness_summary_path": str(robustness_summary_path),
            **split_profile_paths,
        },
    }
    _write_json_atomically(run_summary, run_summary_path)
    effective_logger.info(
        "research_cluster_run.complete method=%s split_mode=%s rows_fit=%s rows_used=%s features=%s output_dir=%s",
        method,
        prepared.split_mode,
        prepared.preprocess_fit.processed_df.height,
        prepared.preprocess_predict.processed_df.height,
        len(prepared.preprocess_predict.feature_list),
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
    split_mode: str | None = None,
    train_end: date | None = None,
    test_start: date | None = None,
    test_end: date | None = None,
    fit_on: str = "train",
    predict_on: str | None = None,
    scaler: str | None = None,
    scaling_scope: str | None = None,
    logger: logging.Logger | None = None,
) -> ResearchClusterSweepResult:
    """Run a lightweight clustering sweep and write comparison artifacts."""

    effective_logger = logger or LOGGER
    prepared = _prepare_preprocessed_data(
        settings,
        dataset_path=dataset_path,
        sample_frac=sample_frac,
        date_from=date_from,
        date_to=date_to,
        split_mode=split_mode,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        fit_on=fit_on,
        predict_on=predict_on,
        scaler=scaler,
        scaling_scope=scaling_scope,
        logger=effective_logger,
    )

    rows: list[dict[str, Any]] = []
    run_id = f"research-sweep-{uuid4().hex[:12]}"
    for method in methods:
        method_norm = method.strip().lower()
        if method_norm in {"kmeans", "gmm"}:
            cluster_grid = [max(2, int(n)) for n in n_clusters_values]
        else:
            cluster_grid = [max(2, int(n_clusters_values[0]))]
        for n_clusters in cluster_grid:
            try:
                clustering_result = run_clustering(
                    prepared.preprocess_fit.X,
                    method=method_norm,
                    n_clusters=n_clusters,
                    random_state=random_state,
                    silhouette_sample_max=settings.research_clustering.silhouette_sample_max,
                    kmeans_n_init=settings.research_clustering.kmeans.n_init,
                    kmeans_max_iter=settings.research_clustering.kmeans.max_iter,
                    gmm_covariance_type=settings.research_clustering.gmm.covariance_type,
                    gmm_reg_covar=settings.research_clustering.gmm.reg_covar,
                    gmm_max_iter=settings.research_clustering.gmm.max_iter,
                    predict_sets={"predict": prepared.preprocess_predict.X},
                    primary_set="predict",
                )
                clustered_df = _build_clustered_df(
                    prepared.preprocess_predict.processed_df,
                    clustering_result.labels_by_set["predict"],
                    clustering_result.extra_columns_by_set.get("predict", {}),
                )
                profiles = build_cluster_profiles(clustered_df)
                separation_score = _profile_forward_separation(profiles.cluster_profile)
                label_metrics = _label_distribution_metrics(clustering_result.labels_by_set["predict"])

                row = {
                    "method": method_norm,
                    "n_clusters": n_clusters,
                    "rows_fit": prepared.preprocess_fit.processed_df.height,
                    "rows_used": prepared.preprocess_predict.processed_df.height,
                    "features_count": len(prepared.preprocess_predict.feature_list),
                    "split_mode": prepared.split_mode,
                    "fit_on": prepared.fit_on,
                    "predict_on": prepared.predict_on,
                    "scaler": prepared.scaler,
                    "scaling_scope": prepared.scaling_scope,
                    "silhouette": clustering_result.metrics.get("silhouette"),
                    "davies_bouldin": clustering_result.metrics.get("davies_bouldin"),
                    "calinski_harabasz": clustering_result.metrics.get("calinski_harabasz"),
                    "bic": clustering_result.metrics.get("bic"),
                    "aic": clustering_result.metrics.get("aic"),
                    "n_clusters_detected": clustering_result.metrics.get("n_clusters_detected"),
                    "forward_return_separation_score": separation_score,
                    "largest_cluster_share": label_metrics["largest_cluster_share"],
                    "effective_cluster_count": label_metrics["effective_cluster_count"],
                    "status": "ok",
                    "error": None,
                }
            except Exception as exc:
                row = {
                    "method": method_norm,
                    "n_clusters": n_clusters,
                    "rows_fit": prepared.preprocess_fit.processed_df.height,
                    "rows_used": prepared.preprocess_predict.processed_df.height,
                    "features_count": len(prepared.preprocess_predict.feature_list),
                    "split_mode": prepared.split_mode,
                    "fit_on": prepared.fit_on,
                    "predict_on": prepared.predict_on,
                    "scaler": prepared.scaler,
                    "scaling_scope": prepared.scaling_scope,
                    "silhouette": None,
                    "davies_bouldin": None,
                    "calinski_harabasz": None,
                    "bic": None,
                    "aic": None,
                    "n_clusters_detected": None,
                    "forward_return_separation_score": None,
                    "largest_cluster_share": None,
                    "effective_cluster_count": None,
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
            "rows_loaded": prepared.loaded.frame.height,
            "rows_fit": prepared.preprocess_fit.processed_df.height,
            "rows_used": prepared.preprocess_predict.processed_df.height,
            "features_count": len(prepared.preprocess_predict.feature_list),
            "split_mode": prepared.split_mode,
            "fit_on": prepared.fit_on,
            "predict_on": prepared.predict_on,
            "scaler": prepared.scaler,
            "scaling_scope": prepared.scaling_scope,
            "split_summary": prepared.split_summary,
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


def _pairwise_ari_rows(labels_by_seed: dict[int, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    if len(labels_by_seed) < 2:
        return [], {"ari_mean": None, "ari_median": None, "ari_min": None, "ari_max": None}
    from sklearn.metrics import adjusted_rand_score

    seeds = sorted(labels_by_seed.keys())
    rows: list[dict[str, Any]] = []
    values: list[float] = []
    for idx, seed_a in enumerate(seeds):
        for seed_b in seeds[idx + 1 :]:
            ari = float(adjusted_rand_score(labels_by_seed[seed_a], labels_by_seed[seed_b]))
            rows.append({"seed_a": seed_a, "seed_b": seed_b, "ari": ari})
            values.append(ari)
    if not values:
        return rows, {"ari_mean": None, "ari_median": None, "ari_min": None, "ari_max": None}
    return rows, {
        "ari_mean": float(np.mean(values)),
        "ari_median": float(np.median(values)),
        "ari_min": float(np.min(values)),
        "ari_max": float(np.max(values)),
    }


def run_research_cluster_stability(
    settings: AppSettings,
    *,
    dataset_path: Path,
    method: str,
    n_clusters: int,
    seeds: int,
    seed_start: int,
    sample_frac: float | None,
    date_from: date | None,
    date_to: date | None,
    split_mode: str | None = None,
    train_end: date | None = None,
    test_start: date | None = None,
    test_end: date | None = None,
    fit_on: str = "train",
    predict_on: str | None = None,
    scaler: str | None = None,
    scaling_scope: str | None = None,
    logger: logging.Logger | None = None,
) -> ResearchClusterStabilityResult:
    """Run repeated clustering across seeds and compute ARI-based stability diagnostics."""

    effective_logger = logger or LOGGER
    if seeds < 1:
        raise ValueError("seeds must be >= 1.")
    method_norm = method.strip().lower()
    if method_norm not in {"kmeans", "gmm"}:
        raise ValueError("research-cluster-stability currently supports methods: kmeans, gmm")

    prepared = _prepare_preprocessed_data(
        settings,
        dataset_path=dataset_path,
        sample_frac=sample_frac,
        date_from=date_from,
        date_to=date_to,
        split_mode=split_mode,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        fit_on=fit_on,
        predict_on=predict_on,
        scaler=scaler,
        scaling_scope=scaling_scope,
        logger=effective_logger,
    )
    seeds_list = [seed_start + idx for idx in range(seeds)]
    seed_rows: list[dict[str, Any]] = []
    size_rows: list[dict[str, Any]] = []
    labels_by_seed: dict[int, np.ndarray] = {}

    started_ts = datetime.now(timezone.utc)
    for seed in seeds_list:
        clustering_result = run_clustering(
            prepared.preprocess_fit.X,
            method=method_norm,
            n_clusters=n_clusters,
            random_state=seed,
            silhouette_sample_max=settings.research_clustering.silhouette_sample_max,
            kmeans_n_init=settings.research_clustering.kmeans.n_init,
            kmeans_max_iter=settings.research_clustering.kmeans.max_iter,
            gmm_covariance_type=settings.research_clustering.gmm.covariance_type,
            gmm_reg_covar=settings.research_clustering.gmm.reg_covar,
            gmm_max_iter=settings.research_clustering.gmm.max_iter,
            predict_sets={"predict": prepared.preprocess_predict.X},
            primary_set="predict",
        )
        labels = clustering_result.labels_by_set["predict"]
        labels_by_seed[seed] = labels
        clustered_df = _build_clustered_df(
            prepared.preprocess_predict.processed_df,
            labels,
            clustering_result.extra_columns_by_set.get("predict", {}),
        )
        profiles = build_cluster_profiles(clustered_df)
        separation_score = _profile_forward_separation(profiles.cluster_profile)
        label_metrics = _label_distribution_metrics(labels)

        unique_labels, counts = np.unique(labels, return_counts=True)
        total = float(labels.shape[0])
        for label, count in zip(unique_labels.tolist(), counts.tolist(), strict=False):
            size_rows.append(
                {
                    "seed": seed,
                    "cluster_id": int(label),
                    "row_count": int(count),
                    "share": float(count / total),
                }
            )

        seed_rows.append(
            {
                "seed": seed,
                "method": method_norm,
                "n_clusters": n_clusters,
                "rows_fit": prepared.preprocess_fit.processed_df.height,
                "rows_used": prepared.preprocess_predict.processed_df.height,
                "split_mode": prepared.split_mode,
                "fit_on": prepared.fit_on,
                "predict_on": prepared.predict_on,
                "scaler": prepared.scaler,
                "scaling_scope": prepared.scaling_scope,
                "silhouette": clustering_result.metrics.get("silhouette"),
                "davies_bouldin": clustering_result.metrics.get("davies_bouldin"),
                "calinski_harabasz": clustering_result.metrics.get("calinski_harabasz"),
                "bic": clustering_result.metrics.get("bic"),
                "aic": clustering_result.metrics.get("aic"),
                "forward_return_separation_score": separation_score,
                "largest_cluster_share": label_metrics["largest_cluster_share"],
                "effective_cluster_count": label_metrics["effective_cluster_count"],
            }
        )

    pairwise_rows, ari_summary = _pairwise_ari_rows(labels_by_seed)
    finished_ts = datetime.now(timezone.utc)
    duration_sec = (finished_ts - started_ts).total_seconds()

    seed_df = pl.DataFrame(seed_rows).sort("seed")
    pairwise_df = pl.DataFrame(pairwise_rows).sort(["seed_a", "seed_b"]) if pairwise_rows else pl.DataFrame(
        schema={"seed_a": pl.Int64, "seed_b": pl.Int64, "ari": pl.Float64}
    )
    size_df = pl.DataFrame(size_rows).sort(["seed", "cluster_id"]) if size_rows else pl.DataFrame(
        schema={"seed": pl.Int64, "cluster_id": pl.Int32, "row_count": pl.Int64, "share": pl.Float64}
    )

    run_id = f"research-stability-{uuid4().hex[:12]}"
    dataset_tag = dataset_path.parent.name.replace(" ", "_")
    output_dir = settings.paths.artifacts_root / "research_runs" / f"{run_id}_{method_norm}_{dataset_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    stability_summary_path = output_dir / "stability_summary.json"
    stability_by_seed_parquet = output_dir / "stability_by_seed.parquet"
    stability_by_seed_csv = output_dir / "stability_by_seed.csv"
    pairwise_ari_csv = output_dir / "stability_pairwise_ari.csv"
    pairwise_ari_parquet = output_dir / "stability_pairwise_ari.parquet"
    cluster_sizes_csv = output_dir / "stability_cluster_sizes_by_seed.csv"
    robustness_summary_path = output_dir / "robustness_summary.json"
    split_summary_path = output_dir / "split_summary.json"

    _write_parquet_atomically(seed_df, stability_by_seed_parquet)
    _write_csv_atomically(seed_df, stability_by_seed_csv)
    _write_csv_atomically(pairwise_df, pairwise_ari_csv)
    _write_parquet_atomically(pairwise_df, pairwise_ari_parquet)
    _write_csv_atomically(size_df, cluster_sizes_csv)
    _write_json_atomically(prepared.split_summary, split_summary_path)

    avg_forward_sep = None
    if "forward_return_separation_score" in seed_df.columns and seed_df.height > 0:
        series = seed_df.select(pl.col("forward_return_separation_score")).to_series().drop_nulls()
        if series.len() > 0:
            avg_forward_sep = float(series.mean())
    avg_largest_share = None
    if "largest_cluster_share" in seed_df.columns and seed_df.height > 0:
        series = seed_df.select(pl.col("largest_cluster_share")).to_series().drop_nulls()
        if series.len() > 0:
            avg_largest_share = float(series.mean())
    silhouette_mean = None
    if "silhouette" in seed_df.columns and seed_df.height > 0:
        series = seed_df.select(pl.col("silhouette")).to_series().drop_nulls()
        if series.len() > 0:
            silhouette_mean = float(series.mean())

    robustness_summary = {
        "method": method_norm,
        "n_clusters": n_clusters,
        "seeds": seeds_list,
        "ari_summary": ari_summary,
        "largest_cluster_share_mean": avg_largest_share,
        "forward_separation_score_mean": avg_forward_sep,
        "silhouette_mean": silhouette_mean,
        "robustness_notes": _build_robustness_notes(
            silhouette=silhouette_mean,
            largest_cluster_share=avg_largest_share,
            forward_separation_score=avg_forward_sep,
            ari_mean=ari_summary["ari_mean"],
        ),
    }
    _write_json_atomically(robustness_summary, robustness_summary_path)

    stability_summary = {
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "started_ts": started_ts.isoformat(),
        "finished_ts": finished_ts.isoformat(),
        "duration_sec": round(duration_sec, 3),
        "method": method_norm,
        "n_clusters": n_clusters,
        "seed_start": seed_start,
        "seeds": seeds,
        "split_mode": prepared.split_mode,
        "fit_on": prepared.fit_on,
        "predict_on": prepared.predict_on,
        "scaler": prepared.scaler,
        "scaling_scope": prepared.scaling_scope,
        "rows_fit": prepared.preprocess_fit.processed_df.height,
        "rows_used": prepared.preprocess_predict.processed_df.height,
        "ari_summary": ari_summary,
        "outputs": {
            "stability_by_seed_parquet": str(stability_by_seed_parquet),
            "stability_by_seed_csv": str(stability_by_seed_csv),
            "stability_pairwise_ari_csv": str(pairwise_ari_csv),
            "stability_pairwise_ari_parquet": str(pairwise_ari_parquet),
            "stability_cluster_sizes_by_seed_csv": str(cluster_sizes_csv),
            "split_summary_path": str(split_summary_path),
            "robustness_summary_path": str(robustness_summary_path),
        },
    }
    _write_json_atomically(stability_summary, stability_summary_path)
    effective_logger.info(
        "research_cluster_stability.complete method=%s seeds=%s rows_used=%s output_dir=%s",
        method_norm,
        seeds,
        prepared.preprocess_predict.processed_df.height,
        output_dir,
    )
    return ResearchClusterStabilityResult(
        run_id=run_id,
        output_dir=output_dir,
        stability_summary_path=stability_summary_path,
        stability_by_seed_path=stability_by_seed_parquet,
        pairwise_ari_path=pairwise_ari_csv,
    )
