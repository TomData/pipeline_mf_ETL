"""Deterministic preprocessing for unsupervised clustering baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    """Preprocessed frame, model matrix, and metadata."""

    processed_df: pl.DataFrame
    X: np.ndarray
    feature_list: list[str]
    preprocess_summary: dict[str, Any]
    scaler_params: dict[str, Any]


def _default_identity_columns(df: pl.DataFrame) -> list[str]:
    candidates = [
        "ticker",
        "exchange",
        "trade_date",
        "trade_dt",
        "flow_state_code",
        "flow_state_label",
        "close",
        "volume",
        "quality_warn_count",
        "fwd_ret_5",
        "fwd_ret_10",
        "fwd_ret_20",
        "fwd_abs_ret_10",
        "fwd_vol_proxy_10",
    ]
    return [column for column in candidates if column in df.columns]


def preprocess_for_clustering(
    df: pl.DataFrame,
    *,
    feature_list: list[str],
    scaler: str = "standard",
    clip_zscore: float | None = 8.0,
) -> PreprocessResult:
    """Select, clean, scale, and clip features for clustering."""

    if not feature_list:
        raise ValueError("feature_list must not be empty.")

    available_features = [feature for feature in feature_list if feature in df.columns]
    missing_features = [feature for feature in feature_list if feature not in df.columns]
    if not available_features:
        raise ValueError("None of the requested clustering features exist in dataset.")

    identity_columns = _default_identity_columns(df)
    selected = df.select(identity_columns + available_features)

    null_counts = {
        feature: int(selected.select(pl.col(feature).is_null().sum()).item()) for feature in available_features
    }
    rows_before = selected.height
    selected = selected.filter(pl.all_horizontal([pl.col(feature).is_not_null() for feature in available_features]))
    rows_after = selected.height

    X_raw = selected.select(available_features).to_numpy().astype(np.float64, copy=False)
    if X_raw.size == 0:
        raise ValueError("No rows remain after null filtering for selected features.")

    scaler_name = scaler.strip().lower()
    if scaler_name not in {"standard", "robust"}:
        raise ValueError("scaler must be one of: standard, robust")

    if scaler_name == "standard":
        center = np.nanmean(X_raw, axis=0)
        spread = np.nanstd(X_raw, axis=0)
        spread[spread == 0] = 1.0
    else:
        center = np.nanmedian(X_raw, axis=0)
        q25 = np.nanpercentile(X_raw, 25, axis=0)
        q75 = np.nanpercentile(X_raw, 75, axis=0)
        spread = q75 - q25
        spread[spread == 0] = 1.0

    X = (X_raw - center) / spread
    clipping_applied = False
    if clip_zscore is not None and clip_zscore > 0:
        X = np.clip(X, -clip_zscore, clip_zscore)
        clipping_applied = True

    preprocess_summary = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_dropped_null_features": rows_before - rows_after,
        "selected_feature_count": len(available_features),
        "selected_features": available_features,
        "missing_requested_features": missing_features,
        "null_counts_per_feature": null_counts,
        "scaler": scaler_name,
        "clip_zscore": clip_zscore,
        "clipping_applied": clipping_applied,
    }
    scaler_params = {
        "scaler": scaler_name,
        "feature_list": available_features,
        "center": center.tolist(),
        "spread": spread.tolist(),
        "clip_zscore": clip_zscore,
    }
    return PreprocessResult(
        processed_df=selected,
        X=X,
        feature_list=available_features,
        preprocess_summary=preprocess_summary,
        scaler_params=scaler_params,
    )

