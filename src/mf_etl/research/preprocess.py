"""Deterministic preprocessing for unsupervised clustering baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

ScalerName = Literal["standard", "robust"]
ScalingScope = Literal["global", "per_ticker"]


@dataclass(frozen=True, slots=True)
class PreprocessModel:
    """Fitted preprocessing model used to transform feature matrices."""

    feature_list: list[str]
    missing_requested_features: list[str]
    scaler: ScalerName
    scaling_scope: ScalingScope
    center: np.ndarray | None
    spread: np.ndarray | None
    per_ticker_params: pl.DataFrame | None
    fit_rows_before: int
    fit_rows_after_null_filter: int
    fit_ticker_count: int
    fit_null_counts_per_feature: dict[str, int]


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    """Preprocessed frame, model matrix, and metadata."""

    processed_df: pl.DataFrame
    X: np.ndarray
    feature_list: list[str]
    preprocess_summary: dict[str, Any]
    scaler_params: dict[str, Any]
    scaler_params_table: pl.DataFrame | None = None


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


def _normalize_scaler_name(value: str) -> ScalerName:
    scaler_name = value.strip().lower()
    if scaler_name not in {"standard", "robust"}:
        raise ValueError("scaler must be one of: standard, robust")
    return scaler_name


def _normalize_scaling_scope(value: str) -> ScalingScope:
    scaling_scope = value.strip().lower()
    if scaling_scope not in {"global", "per_ticker"}:
        raise ValueError("scaling_scope must be one of: global, per_ticker")
    return scaling_scope


def _ensure_valid_feature_list(df: pl.DataFrame, feature_list: list[str]) -> tuple[list[str], list[str]]:
    if not feature_list:
        raise ValueError("feature_list must not be empty.")
    available_features = [feature for feature in feature_list if feature in df.columns]
    missing_features = [feature for feature in feature_list if feature not in df.columns]
    if not available_features:
        raise ValueError("None of the requested clustering features exist in dataset.")
    return available_features, missing_features


def _drop_null_feature_rows(df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    if not features:
        return df
    return df.filter(pl.all_horizontal([pl.col(feature).is_not_null() for feature in features]))


def _fit_global_parameters(X_raw: np.ndarray, scaler: ScalerName) -> tuple[np.ndarray, np.ndarray]:
    if scaler == "standard":
        center = np.nanmean(X_raw, axis=0)
        spread = np.nanstd(X_raw, axis=0)
    else:
        center = np.nanmedian(X_raw, axis=0)
        q25 = np.nanpercentile(X_raw, 25, axis=0)
        q75 = np.nanpercentile(X_raw, 75, axis=0)
        spread = q75 - q25

    center = np.where(np.isfinite(center), center, 0.0)
    spread = np.where(np.isfinite(spread) & (spread != 0.0), spread, 1.0)
    return center.astype(np.float64), spread.astype(np.float64)


def _fit_per_ticker_parameters(df: pl.DataFrame, features: list[str], scaler: ScalerName) -> pl.DataFrame:
    if "ticker" not in df.columns:
        raise ValueError("per_ticker scaling requires ticker column in dataframe.")

    agg_exprs: list[pl.Expr] = []
    for feature in features:
        if scaler == "standard":
            agg_exprs.extend(
                [
                    pl.col(feature).mean().alias(f"center__{feature}"),
                    pl.col(feature).std(ddof=0).alias(f"spread__{feature}"),
                ]
            )
        else:
            agg_exprs.extend(
                [
                    pl.col(feature).median().alias(f"center__{feature}"),
                    (
                        pl.col(feature).quantile(0.75, interpolation="linear")
                        - pl.col(feature).quantile(0.25, interpolation="linear")
                    ).alias(f"spread__{feature}"),
                ]
            )

    params = df.group_by("ticker").agg(agg_exprs).sort("ticker")
    clean_exprs: list[pl.Expr] = []
    for feature in features:
        center_col = f"center__{feature}"
        spread_col = f"spread__{feature}"
        clean_exprs.append(
            pl.when(pl.col(center_col).is_finite().fill_null(False))
            .then(pl.col(center_col))
            .otherwise(0.0)
            .alias(center_col)
        )
        clean_exprs.append(
            pl.when(
                pl.col(spread_col).is_finite().fill_null(False) & (pl.col(spread_col) != 0.0)
            )
            .then(pl.col(spread_col))
            .otherwise(1.0)
            .alias(spread_col)
        )
    return params.with_columns(clean_exprs)


def fit_preprocess_model(
    df: pl.DataFrame,
    *,
    feature_list: list[str],
    scaler: str = "standard",
    scaling_scope: str = "global",
) -> PreprocessModel:
    """Fit preprocessing parameters on a dataframe (typically training rows)."""

    scaler_name = _normalize_scaler_name(scaler)
    scope = _normalize_scaling_scope(scaling_scope)
    available_features, missing_features = _ensure_valid_feature_list(df, feature_list)

    base_columns = ["ticker", *available_features] if scope == "per_ticker" else available_features
    fit_df = df.select([column for column in base_columns if column in df.columns])

    null_counts = {
        feature: int(fit_df.select(pl.col(feature).is_null().sum()).item()) for feature in available_features
    }
    rows_before = fit_df.height
    filtered = _drop_null_feature_rows(fit_df, available_features)
    rows_after = filtered.height
    if rows_after == 0:
        raise ValueError("No rows remain after null filtering for selected features.")

    if scope == "global":
        X_raw = filtered.select(available_features).to_numpy().astype(np.float64, copy=False)
        center, spread = _fit_global_parameters(X_raw, scaler=scaler_name)
        ticker_count = int(filtered.select(pl.col("ticker").n_unique()).item()) if "ticker" in filtered.columns else 0
        return PreprocessModel(
            feature_list=available_features,
            missing_requested_features=missing_features,
            scaler=scaler_name,
            scaling_scope=scope,
            center=center,
            spread=spread,
            per_ticker_params=None,
            fit_rows_before=rows_before,
            fit_rows_after_null_filter=rows_after,
            fit_ticker_count=ticker_count,
            fit_null_counts_per_feature=null_counts,
        )

    per_ticker_params = _fit_per_ticker_parameters(filtered, available_features, scaler=scaler_name)
    ticker_count = per_ticker_params.height
    return PreprocessModel(
        feature_list=available_features,
        missing_requested_features=missing_features,
        scaler=scaler_name,
        scaling_scope=scope,
        center=None,
        spread=None,
        per_ticker_params=per_ticker_params,
        fit_rows_before=rows_before,
        fit_rows_after_null_filter=rows_after,
        fit_ticker_count=ticker_count,
        fit_null_counts_per_feature=null_counts,
    )


def _transform_global(selected: pl.DataFrame, model: PreprocessModel) -> tuple[pl.DataFrame, np.ndarray, dict[str, Any]]:
    if model.center is None or model.spread is None:
        raise ValueError("global preprocessing model is missing center/spread.")
    X_raw = selected.select(model.feature_list).to_numpy().astype(np.float64, copy=False)
    X_scaled = (X_raw - model.center) / model.spread
    extra = {
        "rows_dropped_unseen_tickers": 0,
        "unseen_ticker_count": 0,
        "unseen_tickers_sample": [],
    }
    return selected, X_scaled, extra


def _transform_per_ticker(selected: pl.DataFrame, model: PreprocessModel) -> tuple[pl.DataFrame, np.ndarray, dict[str, Any]]:
    if "ticker" not in selected.columns:
        raise ValueError("per_ticker scaling requires ticker column in dataframe.")
    if model.per_ticker_params is None:
        raise ValueError("per_ticker preprocessing model is missing per_ticker_params.")

    center_cols = [f"center__{feature}" for feature in model.feature_list]
    spread_cols = [f"spread__{feature}" for feature in model.feature_list]
    with_idx = selected.with_row_index("__row_idx")
    joined = with_idx.join(model.per_ticker_params, on="ticker", how="left").sort("__row_idx")

    center_matrix = joined.select(center_cols).to_numpy().astype(np.float64, copy=False)
    missing_mask = np.any(~np.isfinite(center_matrix), axis=1)
    missing_rows = int(np.sum(missing_mask))
    missing_tickers: list[str] = []
    if missing_rows > 0:
        missing_tickers = (
            joined.filter(pl.Series(name="_missing", values=missing_mask))
            .select(pl.col("ticker").cast(pl.String))
            .unique()
            .head(10)
            .to_series()
            .to_list()
        )

    keep_mask = ~missing_mask
    if not np.any(keep_mask):
        raise ValueError("No rows remain after per_ticker scaling due to unseen ticker params.")

    kept = joined.filter(pl.Series(name="_keep", values=keep_mask))
    X_raw = kept.select(model.feature_list).to_numpy().astype(np.float64, copy=False)
    centers = kept.select(center_cols).to_numpy().astype(np.float64, copy=False)
    spreads = kept.select(spread_cols).to_numpy().astype(np.float64, copy=False)
    X_scaled = (X_raw - centers) / spreads
    processed = kept.select(selected.columns)
    extra = {
        "rows_dropped_unseen_tickers": missing_rows,
        "unseen_ticker_count": len(missing_tickers),
        "unseen_tickers_sample": missing_tickers,
    }
    return processed, X_scaled, extra


def _scaler_params_payload(
    model: PreprocessModel,
    *,
    clip_zscore: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scaler": model.scaler,
        "scaling_scope": model.scaling_scope,
        "feature_list": model.feature_list,
        "missing_requested_features": model.missing_requested_features,
        "clip_zscore": clip_zscore,
        "fit_rows_before": model.fit_rows_before,
        "fit_rows_after_null_filter": model.fit_rows_after_null_filter,
        "fit_ticker_count": model.fit_ticker_count,
        "fit_null_counts_per_feature": model.fit_null_counts_per_feature,
    }
    if model.scaling_scope == "global":
        payload["center"] = [] if model.center is None else model.center.tolist()
        payload["spread"] = [] if model.spread is None else model.spread.tolist()
    else:
        payload["per_ticker_params_table"] = True
        payload["per_ticker_params_ticker_count"] = 0 if model.per_ticker_params is None else model.per_ticker_params.height
    return payload


def transform_for_clustering(
    df: pl.DataFrame,
    *,
    model: PreprocessModel,
    clip_zscore: float | None = 8.0,
) -> PreprocessResult:
    """Apply fitted preprocessing model to a dataframe and build clustering matrix."""

    identity_columns = _default_identity_columns(df)
    selected = df.select(identity_columns + model.feature_list)
    rows_before = selected.height
    null_counts = {
        feature: int(selected.select(pl.col(feature).is_null().sum()).item()) for feature in model.feature_list
    }
    selected = _drop_null_feature_rows(selected, model.feature_list)
    rows_after_null_filter = selected.height
    if rows_after_null_filter == 0:
        raise ValueError("No rows remain after null filtering for selected features.")

    if model.scaling_scope == "global":
        processed_df, X_scaled, transform_extra = _transform_global(selected, model)
    else:
        processed_df, X_scaled, transform_extra = _transform_per_ticker(selected, model)

    clipping_applied = False
    if clip_zscore is not None and clip_zscore > 0:
        X_scaled = np.clip(X_scaled, -clip_zscore, clip_zscore)
        clipping_applied = True

    preprocess_summary = {
        "rows_before": rows_before,
        "rows_after_null_filter": rows_after_null_filter,
        "rows_after": processed_df.height,
        "rows_dropped_null_features": rows_before - rows_after_null_filter,
        "selected_feature_count": len(model.feature_list),
        "selected_features": model.feature_list,
        "missing_requested_features": model.missing_requested_features,
        "null_counts_per_feature": null_counts,
        "scaler": model.scaler,
        "scaling_scope": model.scaling_scope,
        "clip_zscore": clip_zscore,
        "clipping_applied": clipping_applied,
        "fit_rows_before": model.fit_rows_before,
        "fit_rows_after_null_filter": model.fit_rows_after_null_filter,
        "fit_ticker_count": model.fit_ticker_count,
        "fit_null_counts_per_feature": model.fit_null_counts_per_feature,
        **transform_extra,
    }
    scaler_params = _scaler_params_payload(model, clip_zscore=clip_zscore)
    return PreprocessResult(
        processed_df=processed_df,
        X=X_scaled.astype(np.float64, copy=False),
        feature_list=model.feature_list,
        preprocess_summary=preprocess_summary,
        scaler_params=scaler_params,
        scaler_params_table=model.per_ticker_params,
    )


def preprocess_for_clustering(
    df: pl.DataFrame,
    *,
    feature_list: list[str],
    scaler: str = "standard",
    clip_zscore: float | None = 8.0,
    scaling_scope: str = "global",
) -> PreprocessResult:
    """Select, clean, scale, and clip features for clustering (fit+transform same rows)."""

    model = fit_preprocess_model(
        df,
        feature_list=feature_list,
        scaler=scaler,
        scaling_scope=scaling_scope,
    )
    return transform_for_clustering(df, model=model, clip_zscore=clip_zscore)

