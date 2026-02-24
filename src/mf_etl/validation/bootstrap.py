"""Bootstrap confidence interval utilities for state-level forward outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import polars as pl

BootstrapMode = Literal["iid", "block"]

RETURN_OUTCOMES: tuple[str, ...] = ("fwd_ret_5", "fwd_ret_10", "fwd_ret_20")
DEFAULT_OUTCOMES: tuple[str, ...] = (
    "fwd_ret_5",
    "fwd_ret_10",
    "fwd_ret_20",
    "fwd_abs_ret_10",
    "fwd_vol_proxy_10",
)


@dataclass(frozen=True, slots=True)
class BootstrapValidationResult:
    """Bootstrap artifact payloads for one validation run."""

    state_summary: pl.DataFrame
    pairwise_diff: pl.DataFrame
    meta: dict[str, Any]


def _normalize_values(values: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(values)
    return values[finite_mask].astype(np.float64, copy=False)


def _bootstrap_indices_iid(
    *,
    n: int,
    n_boot: int,
    rng: np.random.Generator,
    batch_size: int = 128,
) -> list[np.ndarray]:
    batches: list[np.ndarray] = []
    remaining = n_boot
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        batches.append(rng.integers(0, n, size=(this_batch, n), endpoint=False))
        remaining -= this_batch
    return batches


def _bootstrap_stats_iid(
    values: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
    compute_hit: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    means: list[np.ndarray] = []
    medians: list[np.ndarray] = []
    hit_rates: list[np.ndarray] = []
    for idx_batch in _bootstrap_indices_iid(n=values.shape[0], n_boot=n_boot, rng=rng):
        sampled = values[idx_batch]
        means.append(np.mean(sampled, axis=1))
        medians.append(np.median(sampled, axis=1))
        if compute_hit:
            hit_rates.append(np.mean(sampled > 0.0, axis=1))

    mean_dist = np.concatenate(means, axis=0).astype(np.float64, copy=False)
    median_dist = np.concatenate(medians, axis=0).astype(np.float64, copy=False)
    if compute_hit:
        hit_dist: np.ndarray | None = np.concatenate(hit_rates, axis=0).astype(np.float64, copy=False)
    else:
        hit_dist = None
    return mean_dist, median_dist, hit_dist


def _sample_block_bootstrap(
    values: np.ndarray,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = values.shape[0]
    if n == 0:
        return values
    if block_length <= 1 or block_length >= n:
        return values[rng.integers(0, n, size=n, endpoint=False)]

    starts_max = n - block_length + 1
    block_count = int(np.ceil(n / block_length))
    samples: list[np.ndarray] = []
    for _ in range(block_count):
        start = int(rng.integers(0, starts_max, size=1)[0])
        samples.append(values[start : start + block_length])
    combined = np.concatenate(samples, axis=0)
    return combined[:n]


def _bootstrap_stats_block(
    values: np.ndarray,
    *,
    n_boot: int,
    block_length: int,
    rng: np.random.Generator,
    compute_hit: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mean_dist = np.zeros(n_boot, dtype=np.float64)
    median_dist = np.zeros(n_boot, dtype=np.float64)
    hit_dist = np.zeros(n_boot, dtype=np.float64) if compute_hit else None
    for idx in range(n_boot):
        sampled = _sample_block_bootstrap(values, block_length=block_length, rng=rng)
        mean_dist[idx] = float(np.mean(sampled))
        median_dist[idx] = float(np.median(sampled))
        if hit_dist is not None:
            hit_dist[idx] = float(np.mean(sampled > 0.0))
    return mean_dist, median_dist, hit_dist


def _ci_bounds(distribution: np.ndarray, *, ci: float) -> tuple[float | None, float | None]:
    if distribution.size == 0:
        return None, None
    alpha = 1.0 - ci
    lo = float(np.quantile(distribution, alpha / 2.0))
    hi = float(np.quantile(distribution, 1.0 - alpha / 2.0))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None, None
    return lo, hi


def _safe_float(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    val = float(value)
    return val if np.isfinite(val) else None


def _assert_bootstrap_consistency(state_summary: pl.DataFrame) -> None:
    if state_summary.height == 0:
        return

    required = [
        "state_id",
        "outcome_name",
        "n",
        "mean_obs",
        "mean_ci_lo",
        "mean_ci_hi",
        "median_obs",
        "median_ci_lo",
        "median_ci_hi",
    ]
    missing = [column for column in required if column not in state_summary.columns]
    if missing:
        raise ValueError(f"Bootstrap QA consistency failed: missing columns {', '.join(missing)}")

    issues: list[str] = []
    for stat_col in ("mean_obs", "mean_ci_lo", "mean_ci_hi", "median_obs", "median_ci_lo", "median_ci_hi"):
        inconsistent = (
            ((pl.col("n") > 0) & (pl.col(stat_col).is_null() | pl.col(stat_col).is_nan().fill_null(False) | (~pl.col(stat_col).is_finite()).fill_null(False)))
            | ((pl.col("n") == 0) & pl.col(stat_col).is_not_null())
        )
        bad_rows = state_summary.filter(inconsistent).select(["state_id", "outcome_name"]).to_dicts()
        if bad_rows:
            issues.append(f"{stat_col} inconsistent for rows={bad_rows[:20]}")

    if issues:
        raise ValueError(f"Bootstrap QA consistency failed: {'; '.join(issues)}")


def run_bootstrap_validation(
    df: pl.DataFrame,
    *,
    outcomes: list[str] | tuple[str, ...] = DEFAULT_OUTCOMES,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
    bootstrap_mode: BootstrapMode = "iid",
    block_length: int = 10,
    pairwise_outcome: str = "fwd_ret_10",
    max_points_per_state: int | None = 50_000,
) -> BootstrapValidationResult:
    """Compute bootstrap summaries per state and pairwise state differences."""

    if "state_id" not in df.columns:
        raise ValueError("Input dataframe must include state_id.")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0,1)")
    if bootstrap_mode not in {"iid", "block"}:
        raise ValueError("bootstrap_mode must be one of: iid, block")

    available_outcomes = [outcome for outcome in outcomes if outcome in df.columns]
    if not available_outcomes:
        raise ValueError("No requested outcomes exist in dataframe.")

    rng = np.random.default_rng(random_state)
    sorted_df = df.sort([column for column in ("state_id", "ticker", "trade_date") if column in df.columns])
    state_ids = (
        sorted_df.select(pl.col("state_id").drop_nulls().unique().sort()).to_series().to_list()
        if sorted_df.height > 0
        else []
    )

    rows: list[dict[str, Any]] = []
    mean_distributions: dict[int, dict[str, np.ndarray]] = {}
    non_null_counts: dict[int, dict[str, int]] = {}

    for state_id in state_ids:
        state_df = sorted_df.filter(pl.col("state_id") == pl.lit(state_id))
        mean_distributions[int(state_id)] = {}
        non_null_counts[int(state_id)] = {}
        for outcome in available_outcomes:
            values = state_df.select(pl.col(outcome).cast(pl.Float64, strict=False)).to_series().to_numpy()
            values = _normalize_values(values)
            n = int(values.shape[0])
            non_null_counts[int(state_id)][outcome] = n
            if n == 0:
                row = {
                    "state_id": int(state_id),
                    "outcome_name": outcome,
                    "n": 0,
                    "mean_obs": None,
                    "mean_ci_lo": None,
                    "mean_ci_hi": None,
                    "median_obs": None,
                    "median_ci_lo": None,
                    "median_ci_hi": None,
                    "hit_rate_obs": None,
                    "hit_rate_ci_lo": None,
                    "hit_rate_ci_hi": None,
                    "bootstrap_method": "percentile",
                    "bootstrap_mode": bootstrap_mode,
                    "bootstrap_sample_n": 0,
                }
                rows.append(row)
                continue

            sampled_values = values
            if max_points_per_state is not None and n > max_points_per_state:
                sampled_values = rng.choice(values, size=max_points_per_state, replace=False)

            compute_hit = outcome in RETURN_OUTCOMES
            if bootstrap_mode == "iid":
                mean_dist, median_dist, hit_dist = _bootstrap_stats_iid(
                    sampled_values,
                    n_boot=n_boot,
                    rng=rng,
                    compute_hit=compute_hit,
                )
            else:
                mean_dist, median_dist, hit_dist = _bootstrap_stats_block(
                    sampled_values,
                    n_boot=n_boot,
                    block_length=block_length,
                    rng=rng,
                    compute_hit=compute_hit,
                )

            mean_distributions[int(state_id)][outcome] = mean_dist
            mean_lo, mean_hi = _ci_bounds(mean_dist, ci=ci)
            med_lo, med_hi = _ci_bounds(median_dist, ci=ci)
            hit_lo: float | None
            hit_hi: float | None
            if hit_dist is not None:
                hit_lo, hit_hi = _ci_bounds(hit_dist, ci=ci)
            else:
                hit_lo, hit_hi = None, None

            row = {
                "state_id": int(state_id),
                "outcome_name": outcome,
                "n": n,
                "mean_obs": _safe_float(np.mean(sampled_values)),
                "mean_ci_lo": _safe_float(mean_lo),
                "mean_ci_hi": _safe_float(mean_hi),
                "median_obs": _safe_float(np.median(sampled_values)),
                "median_ci_lo": _safe_float(med_lo),
                "median_ci_hi": _safe_float(med_hi),
                "hit_rate_obs": _safe_float(np.mean(sampled_values > 0.0)) if compute_hit else None,
                "hit_rate_ci_lo": _safe_float(hit_lo),
                "hit_rate_ci_hi": _safe_float(hit_hi),
                "bootstrap_method": "percentile",
                "bootstrap_mode": bootstrap_mode,
                "bootstrap_sample_n": int(sampled_values.shape[0]),
            }
            rows.append(row)

    state_summary = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "state_id": pl.Int32,
            "outcome_name": pl.String,
            "n": pl.Int64,
            "mean_obs": pl.Float64,
            "mean_ci_lo": pl.Float64,
            "mean_ci_hi": pl.Float64,
            "median_obs": pl.Float64,
            "median_ci_lo": pl.Float64,
            "median_ci_hi": pl.Float64,
            "hit_rate_obs": pl.Float64,
            "hit_rate_ci_lo": pl.Float64,
            "hit_rate_ci_hi": pl.Float64,
            "bootstrap_method": pl.String,
            "bootstrap_mode": pl.String,
            "bootstrap_sample_n": pl.Int64,
        }
    )
    if state_summary.height > 0:
        state_summary = state_summary.sort(["state_id", "outcome_name"])

    pairwise_rows: list[dict[str, Any]] = []
    if pairwise_outcome in available_outcomes:
        ordered_states = sorted(mean_distributions.keys())
        alpha_outcome = pairwise_outcome
        for idx, state_a in enumerate(ordered_states):
            for state_b in ordered_states[idx + 1 :]:
                dist_a = mean_distributions[state_a].get(alpha_outcome)
                dist_b = mean_distributions[state_b].get(alpha_outcome)
                if dist_a is None or dist_b is None or dist_a.size == 0 or dist_b.size == 0:
                    continue
                usable = min(dist_a.shape[0], dist_b.shape[0])
                diff = dist_a[:usable] - dist_b[:usable]
                diff_lo, diff_hi = _ci_bounds(diff, ci=ci)
                diff_obs = _safe_float(float(np.mean(diff)))
                pairwise_rows.append(
                    {
                        "state_a": int(state_a),
                        "state_b": int(state_b),
                        "n_a": int(non_null_counts[state_a].get(alpha_outcome, 0)),
                        "n_b": int(non_null_counts[state_b].get(alpha_outcome, 0)),
                        "outcome_name": alpha_outcome,
                        "diff_mean_obs": diff_obs,
                        "diff_ci_lo": _safe_float(diff_lo),
                        "diff_ci_hi": _safe_float(diff_hi),
                        "diff_sign_consistent": bool((diff_lo is not None and diff_hi is not None) and (diff_lo > 0.0 or diff_hi < 0.0)),
                    }
                )

    pairwise_df = pl.DataFrame(pairwise_rows).sort(["state_a", "state_b"]) if pairwise_rows else pl.DataFrame(
        schema={
            "state_a": pl.Int32,
            "state_b": pl.Int32,
            "n_a": pl.Int64,
            "n_b": pl.Int64,
            "outcome_name": pl.String,
            "diff_mean_obs": pl.Float64,
            "diff_ci_lo": pl.Float64,
            "diff_ci_hi": pl.Float64,
            "diff_sign_consistent": pl.Boolean,
        }
    )

    _assert_bootstrap_consistency(state_summary)

    meta = {
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "outcomes": available_outcomes,
        "n_boot": int(n_boot),
        "ci": float(ci),
        "bootstrap_mode": bootstrap_mode,
        "block_length": int(block_length),
        "random_state": int(random_state),
        "pairwise_outcome": pairwise_outcome,
        "state_count": len(state_ids),
    }
    return BootstrapValidationResult(
        state_summary=state_summary,
        pairwise_diff=pairwise_df,
        meta=meta,
    )
