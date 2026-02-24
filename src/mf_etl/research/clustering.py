"""Clustering engines and metrics for research baselines."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    """Unified clustering result container."""

    labels: np.ndarray
    metrics: dict[str, Any]
    model_meta: dict[str, Any]
    extra_columns: dict[str, np.ndarray]
    labels_by_set: dict[str, np.ndarray] = field(default_factory=dict)
    extra_columns_by_set: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


def _require_sklearn() -> None:
    if importlib.util.find_spec("sklearn") is None:
        raise RuntimeError(
            "scikit-learn is required for research clustering. Install with: pip install scikit-learn"
        )


def _core_cluster_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    silhouette_sample_max: int = 200_000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute common clustering quality metrics."""

    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    unique = np.unique(labels)
    unique_valid = unique[unique >= 0]
    metrics: dict[str, Any] = {
        "n_clusters_detected": int(unique_valid.shape[0]),
        "has_noise_cluster": bool(np.any(labels < 0)),
    }
    if unique_valid.shape[0] < 2 or X.shape[0] < 3:
        metrics.update(
            {
                "silhouette": None,
                "davies_bouldin": None,
                "calinski_harabasz": None,
            }
        )
        return metrics

    sample_n = min(X.shape[0], silhouette_sample_max)
    if sample_n < X.shape[0]:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=sample_n, replace=False)
        X_eval = X[idx]
        labels_eval = labels[idx]
    else:
        X_eval = X
        labels_eval = labels

    metrics["silhouette"] = float(silhouette_score(X_eval, labels_eval))
    metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    return metrics


def run_clustering(
    X: np.ndarray,
    *,
    method: str,
    n_clusters: int = 5,
    random_state: int = 42,
    silhouette_sample_max: int = 200_000,
    kmeans_n_init: int = 20,
    kmeans_max_iter: int = 300,
    gmm_covariance_type: str = "diag",
    gmm_reg_covar: float = 1e-6,
    gmm_max_iter: int = 200,
    predict_sets: dict[str, np.ndarray] | None = None,
    primary_set: str = "predict",
) -> ClusteringResult:
    """Run selected clustering model and compute diagnostics."""

    _require_sklearn()
    method_norm = method.strip().lower()
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")
    if X.shape[0] == 0:
        raise ValueError("X has zero rows.")
    if X.shape[0] < 2:
        raise ValueError("X must have at least 2 rows.")

    if predict_sets is None:
        predict_sets = {primary_set: X}
    if primary_set not in predict_sets:
        raise ValueError("primary_set must be present in predict_sets.")
    for set_name, matrix in predict_sets.items():
        if matrix.ndim != 2:
            raise ValueError(f"predict set {set_name} must be a 2D matrix.")
        if matrix.shape[1] != X.shape[1]:
            raise ValueError(
                f"predict set {set_name} has {matrix.shape[1]} columns, expected {X.shape[1]}."
            )
        if matrix.shape[0] == 0:
            raise ValueError(f"predict set {set_name} has zero rows.")

    primary_X = predict_sets[primary_set]
    metrics: dict[str, Any] = {}
    model_meta: dict[str, Any] = {"method": method_norm}
    extra_columns: dict[str, np.ndarray] = {}
    labels_by_set: dict[str, np.ndarray] = {}
    extra_columns_by_set: dict[str, dict[str, np.ndarray]] = {}

    if method_norm == "kmeans":
        from sklearn.cluster import KMeans

        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=kmeans_n_init,
            max_iter=kmeans_max_iter,
        )
        model.fit(X)
        for set_name, matrix in predict_sets.items():
            labels_by_set[set_name] = model.predict(matrix).astype(np.int32)
            extra_columns_by_set[set_name] = {}
        labels = labels_by_set[primary_set]
        metrics.update(_core_cluster_metrics(primary_X, labels, silhouette_sample_max=silhouette_sample_max, random_state=random_state))
        metrics["inertia"] = float(model.inertia_)
        model_meta.update(
            {
                "n_clusters": n_clusters,
                "n_init": kmeans_n_init,
                "max_iter": kmeans_max_iter,
                "fit_rows": X.shape[0],
                "predict_rows": primary_X.shape[0],
            }
        )

    elif method_norm == "gmm":
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=gmm_covariance_type,
            reg_covar=gmm_reg_covar,
            max_iter=gmm_max_iter,
            random_state=random_state,
        )
        model.fit(X)
        for set_name, matrix in predict_sets.items():
            set_labels = model.predict(matrix).astype(np.int32)
            probs = model.predict_proba(matrix)
            prob_max = probs.max(axis=1)
            entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
            labels_by_set[set_name] = set_labels
            extra_columns_by_set[set_name] = {
                "cluster_prob_max": prob_max.astype(np.float32),
                "cluster_entropy": entropy.astype(np.float32),
            }
        labels = labels_by_set[primary_set]
        extra_columns = extra_columns_by_set[primary_set]
        metrics.update(_core_cluster_metrics(primary_X, labels, silhouette_sample_max=silhouette_sample_max, random_state=random_state))
        metrics["bic"] = float(model.bic(X))
        metrics["aic"] = float(model.aic(X))
        model_meta.update(
            {
                "n_components": n_clusters,
                "covariance_type": gmm_covariance_type,
                "reg_covar": gmm_reg_covar,
                "max_iter": gmm_max_iter,
                "fit_rows": X.shape[0],
                "predict_rows": primary_X.shape[0],
            }
        )

    elif method_norm == "hdbscan":
        if importlib.util.find_spec("hdbscan") is None:
            raise RuntimeError("hdbscan method requested but optional dependency is not installed.")
        if len(predict_sets) != 1:
            raise ValueError("hdbscan currently only supports prediction on the fit matrix.")
        import hdbscan  # type: ignore[import-not-found]

        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, n_clusters), min_samples=None)
        labels = clusterer.fit_predict(X).astype(np.int32)
        labels_by_set[primary_set] = labels
        extra_columns_by_set[primary_set] = {}
        metrics.update(_core_cluster_metrics(X, labels, silhouette_sample_max=silhouette_sample_max, random_state=random_state))
        model_meta.update(
            {
                "min_cluster_size": max(5, n_clusters),
                "n_clusters_requested_hint": n_clusters,
                "fit_rows": X.shape[0],
                "predict_rows": primary_X.shape[0],
            }
        )
    else:
        raise ValueError("method must be one of: kmeans, gmm, hdbscan")

    if not extra_columns and primary_set in extra_columns_by_set:
        extra_columns = extra_columns_by_set[primary_set]

    return ClusteringResult(
        labels=labels,
        metrics=metrics,
        model_meta=model_meta,
        extra_columns=extra_columns,
        labels_by_set=labels_by_set,
        extra_columns_by_set=extra_columns_by_set,
    )
