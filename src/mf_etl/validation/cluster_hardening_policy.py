"""Policy configuration and threshold helpers for cluster hardening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from mf_etl.config import AppSettings


@dataclass(frozen=True, slots=True)
class ClusterHardeningThresholds:
    """Resolved thresholds and scoring controls for one hardening run."""

    min_n_rows_hard: int
    min_state_share_hard: float
    ret_cv_hard: float
    sign_consistency_hard: float
    ci_width_hard_quantile: float
    score_min_allow: float
    score_min_watch: float
    penalties: dict[str, float]
    weights: dict[str, float]
    eps: float


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def resolve_cluster_hardening_thresholds(
    settings: AppSettings,
    *,
    min_n_rows_hard: int | None = None,
    min_state_share_hard: float | None = None,
    ret_cv_hard: float | None = None,
    sign_consistency_hard: float | None = None,
    ci_width_hard_quantile: float | None = None,
    score_min_allow: float | None = None,
    score_min_watch: float | None = None,
) -> ClusterHardeningThresholds:
    """Resolve hardening thresholds from config with optional CLI overrides."""

    cfg = settings.cluster_hardening
    return ClusterHardeningThresholds(
        min_n_rows_hard=min_n_rows_hard if min_n_rows_hard is not None else cfg.min_n_rows_hard,
        min_state_share_hard=(
            min_state_share_hard if min_state_share_hard is not None else cfg.min_state_share_hard
        ),
        ret_cv_hard=ret_cv_hard if ret_cv_hard is not None else cfg.ret_cv_hard,
        sign_consistency_hard=(
            sign_consistency_hard if sign_consistency_hard is not None else cfg.sign_consistency_hard
        ),
        ci_width_hard_quantile=(
            ci_width_hard_quantile
            if ci_width_hard_quantile is not None
            else cfg.ci_width_hard_quantile
        ),
        score_min_allow=score_min_allow if score_min_allow is not None else cfg.score_min_allow,
        score_min_watch=score_min_watch if score_min_watch is not None else cfg.score_min_watch,
        penalties=cfg.penalties.model_dump(mode="python"),
        weights=cfg.weights.model_dump(mode="python"),
        eps=cfg.eps,
    )


def derive_ci_width_hard_value(state_scorecard: pl.DataFrame, quantile: float) -> float | None:
    """Derive per-run hard CI width threshold from state scorecard quantile."""

    if state_scorecard.height == 0 or "ci_width" not in state_scorecard.columns:
        return None
    raw = state_scorecard.select(
        pl.col("ci_width").cast(pl.Float64, strict=False).quantile(quantile, interpolation="linear")
    ).item()
    return _safe_float(raw)


def recommend_thresholds_from_state_stats(
    state_stats_long: pl.DataFrame,
    *,
    fallback: ClusterHardeningThresholds,
) -> dict[str, Any]:
    """Produce robust threshold recommendations from walk-forward state metrics."""

    if state_stats_long.height == 0:
        return {
            "method": "fallback_config_only",
            "recommendations": {
                "min_n_rows_hard": fallback.min_n_rows_hard,
                "min_state_share_hard": fallback.min_state_share_hard,
                "ret_cv_hard": fallback.ret_cv_hard,
                "sign_consistency_hard": fallback.sign_consistency_hard,
                "ci_width_hard_quantile": fallback.ci_width_hard_quantile,
            },
        }

    stable = state_stats_long.filter(pl.col("class_label") == "ALLOW")
    if stable.height == 0:
        stable = state_stats_long.filter(pl.col("class_label") == "WATCH")
    if stable.height == 0:
        stable = state_stats_long

    def q(col: str, p: float) -> float | None:
        if col not in stable.columns:
            return None
        raw = stable.select(pl.col(col).cast(pl.Float64, strict=False).quantile(p, interpolation="linear")).item()
        return _safe_float(raw)

    min_n_rows_rec = q("n_rows", 0.10)
    min_share_rec = q("state_share_mean", 0.10)
    ret_cv_rec = q("ret_mean_cv", 0.90)
    sign_cons_rec = q("stability_sign_consistency", 0.10)
    ci_width_value_rec = q("ci_width", 0.80)

    return {
        "method": "allow_watch_quantile_heuristic",
        "source_rows": int(stable.height),
        "recommendations": {
            "min_n_rows_hard": int(min_n_rows_rec) if min_n_rows_rec is not None else fallback.min_n_rows_hard,
            "min_state_share_hard": (
                float(min_share_rec) if min_share_rec is not None else fallback.min_state_share_hard
            ),
            "ret_cv_hard": float(ret_cv_rec) if ret_cv_rec is not None else fallback.ret_cv_hard,
            "sign_consistency_hard": (
                float(sign_cons_rec) if sign_cons_rec is not None else fallback.sign_consistency_hard
            ),
            "ci_width_hard_quantile": fallback.ci_width_hard_quantile,
            "ci_width_hard_value_recommended": ci_width_value_rec,
        },
    }

