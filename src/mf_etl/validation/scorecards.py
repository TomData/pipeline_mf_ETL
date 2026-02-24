"""Confidence and robustness scorecards for validation harness outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class ValidationScorecards:
    """Per-state and overall validation scorecards."""

    state_scorecard: pl.DataFrame
    validation_scorecard: dict[str, Any]


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _scaled_sample_score(n_rows: int, n_ref: int) -> float:
    if n_rows <= 0 or n_ref <= 0:
        return 0.0
    return float(min(1.0, np.log1p(n_rows) / np.log1p(n_ref)))


def _ci_width_score(ci_width: float | None) -> float:
    if ci_width is None or not np.isfinite(ci_width):
        return 0.0
    return float(max(0.0, 1.0 - min(ci_width, 0.2) / 0.2))


def _stability_score(sign_consistency: float | None, ret_cv: float | None) -> float:
    sign_component = 0.0 if sign_consistency is None else float(np.clip(sign_consistency, 0.0, 1.0))
    if ret_cv is None or not np.isfinite(ret_cv):
        cv_component = 0.0
    else:
        cv_component = float(max(0.0, 1.0 - min(ret_cv, 3.0) / 3.0))
    return 0.5 * sign_component + 0.5 * cv_component


def _validation_grade(
    *,
    pairwise_share: float | None,
    avg_sign_consistency: float | None,
    avg_ci_width: float | None,
) -> str:
    if pairwise_share is None or avg_sign_consistency is None or avg_ci_width is None:
        return "C"
    if pairwise_share >= 0.6 and avg_sign_consistency >= 0.65 and avg_ci_width <= 0.02:
        return "A"
    if pairwise_share >= 0.4 and avg_sign_consistency >= 0.55 and avg_ci_width <= 0.04:
        return "B"
    return "C"


def build_validation_scorecards(
    *,
    adapted_df: pl.DataFrame,
    state_label_type: str,
    bootstrap_state_summary: pl.DataFrame,
    bootstrap_pairwise_diff: pl.DataFrame,
    state_stability_summary: pl.DataFrame,
    weights: dict[str, float],
    eps: float = 1e-12,
) -> ValidationScorecards:
    """Build per-state and global validation scorecards from harness artifacts."""

    fwd_cols = {
        "n": "n",
        "mean": "mean_obs",
        "ci_lo": "mean_ci_lo",
        "ci_hi": "mean_ci_hi",
        "hit_rate": "hit_rate_obs",
    }
    required = ["state_id", "outcome_name", *fwd_cols.values()]
    missing = [column for column in required if column not in bootstrap_state_summary.columns]
    if missing:
        raise ValueError(f"bootstrap_state_summary missing required columns: {', '.join(missing)}")

    per_state = (
        bootstrap_state_summary.filter(pl.col("outcome_name") == "fwd_ret_10")
        .select(required)
        .drop("outcome_name")
        .sort("state_id")
    )
    if per_state.height == 0:
        raise ValueError("bootstrap_state_summary does not contain outcome_name=fwd_ret_10 rows.")
    per_state = per_state.with_columns(
        [
            (pl.col(fwd_cols["ci_hi"]) - pl.col(fwd_cols["ci_lo"])).alias("ci_width"),
            ((pl.col(fwd_cols["ci_lo"]) > 0.0) | (pl.col(fwd_cols["ci_hi"]) < 0.0)).alias("sign_confidence_flag"),
        ]
    )

    stability_cols = [
        "state_id",
        "fwd_ret_10_sign_stability",
        "ret_mean_cv",
        "state_share_mean",
        "share_cv",
    ]
    stability_present = [column for column in stability_cols if column in state_stability_summary.columns]
    if stability_present:
        per_state = per_state.join(
            state_stability_summary.select(stability_present),
            on="state_id",
            how="left",
        )

    pairwise_share_by_state: dict[int, float] = {}
    if bootstrap_pairwise_diff.height > 0 and {"state_a", "state_b", "diff_sign_consistent"}.issubset(
        bootstrap_pairwise_diff.columns
    ):
        pairwise_rows = bootstrap_pairwise_diff.select(["state_a", "state_b", "diff_sign_consistent"]).to_dicts()
        tally: dict[int, list[int]] = {}
        for row in pairwise_rows:
            for key in (int(row["state_a"]), int(row["state_b"])):
                if key not in tally:
                    tally[key] = [0, 0]
                tally[key][0] += 1
                if bool(row["diff_sign_consistent"]):
                    tally[key][1] += 1
        for key, (total, significant) in tally.items():
            pairwise_share_by_state[key] = float(significant / total) if total > 0 else 0.0

    n_ref = max(1, int(np.percentile(per_state[fwd_cols["n"]].to_numpy(), 90))) if per_state.height > 0 else 1
    rows: list[dict[str, Any]] = []
    for row in per_state.to_dicts():
        state_id = int(row["state_id"])
        n_rows = int(row[fwd_cols["n"]] or 0)
        mean_value = _safe_float(row.get(fwd_cols["mean"]))
        ci_lo = _safe_float(row.get(fwd_cols["ci_lo"]))
        ci_hi = _safe_float(row.get(fwd_cols["ci_hi"]))
        ci_width = _safe_float(row.get("ci_width"))
        sign_confidence = bool(row.get("sign_confidence_flag"))
        sign_consistency = _safe_float(row.get("fwd_ret_10_sign_stability"))
        ret_cv = _safe_float(row.get("ret_mean_cv"))
        state_share_mean = _safe_float(row.get("state_share_mean"))
        state_share_cv = _safe_float(row.get("share_cv"))
        separation_share = pairwise_share_by_state.get(state_id, 0.0)

        sample_component = _scaled_sample_score(n_rows, n_ref)
        ci_component = _ci_width_score(ci_width)
        sign_component = 1.0 if sign_confidence else 0.0
        stability_component = _stability_score(sign_consistency, ret_cv)
        separation_component = float(np.clip(separation_share, 0.0, 1.0))
        confidence = (
            weights.get("sample_size", 0.2) * sample_component
            + weights.get("ci_width", 0.25) * ci_component
            + weights.get("sign_confidence", 0.2) * sign_component
            + weights.get("stability", 0.25) * stability_component
            + weights.get("separation", 0.1) * separation_component
        )

        notes: list[str] = []
        if n_rows < 500:
            notes.append("low_sample")
        if ci_width is not None and ci_width > 0.04:
            notes.append("wide_ci")
        if not sign_confidence:
            notes.append("sign_uncertain")
        if ret_cv is not None and ret_cv > 1.5:
            notes.append("unstable_returns")

        rows.append(
            {
                "state_id": state_id,
                "n_rows": n_rows,
                "fwd_ret_10_mean": mean_value,
                "fwd_ret_10_ci_lo": ci_lo,
                "fwd_ret_10_ci_hi": ci_hi,
                "ci_width": ci_width,
                "hit_rate": _safe_float(row.get(fwd_cols["hit_rate"])),
                "sign_confidence_flag": sign_confidence,
                "stability_sign_consistency": sign_consistency,
                "ret_mean_cv": ret_cv,
                "state_share_mean": state_share_mean,
                "state_share_cv": state_share_cv,
                "pairwise_diff_significant_share": separation_share,
                "confidence_score": float(round(100.0 * confidence, 3)),
                "notes": ",".join(notes) if notes else "ok",
            }
        )

    state_scorecard = (
        pl.DataFrame(rows).sort(["confidence_score", "state_id"], descending=[True, False])
        if rows
        else pl.DataFrame(
            schema={
                "state_id": pl.Int32,
                "n_rows": pl.Int64,
                "fwd_ret_10_mean": pl.Float64,
                "fwd_ret_10_ci_lo": pl.Float64,
                "fwd_ret_10_ci_hi": pl.Float64,
                "ci_width": pl.Float64,
                "hit_rate": pl.Float64,
                "sign_confidence_flag": pl.Boolean,
                "stability_sign_consistency": pl.Float64,
                "ret_mean_cv": pl.Float64,
                "state_share_mean": pl.Float64,
                "state_share_cv": pl.Float64,
                "pairwise_diff_significant_share": pl.Float64,
                "confidence_score": pl.Float64,
                "notes": pl.String,
            }
        )
    )

    total_rows = adapted_df.height
    date_bounds = (
        adapted_df.select(
            [
                pl.col("trade_date").min().alias("date_min"),
                pl.col("trade_date").max().alias("date_max"),
            ]
        ).to_dicts()[0]
        if adapted_df.height > 0
        else {"date_min": None, "date_max": None}
    )

    pairwise_significant_share: float | None = None
    if bootstrap_pairwise_diff.height > 0 and "diff_sign_consistent" in bootstrap_pairwise_diff.columns:
        pairwise_significant_share = float(
            bootstrap_pairwise_diff.select(pl.col("diff_sign_consistent").cast(pl.Int32).mean()).item()
        )

    avg_ci_width: float | None = None
    avg_sign_consistency: float | None = None
    avg_ret_cv: float | None = None
    if state_scorecard.height > 0:
        avg_ci_width = _safe_float(state_scorecard.select(pl.col("ci_width").mean()).item())
        avg_sign_consistency = _safe_float(state_scorecard.select(pl.col("stability_sign_consistency").mean()).item())
        avg_ret_cv = _safe_float(state_scorecard.select(pl.col("ret_mean_cv").mean()).item())

    forward_separation_score: float | None = None
    if state_scorecard.height > 0:
        means = state_scorecard.select(pl.col("fwd_ret_10_mean").drop_nulls()).to_series()
        if means.len() > 0:
            forward_separation_score = _safe_float(float(means.max() - means.min()))

    top_state_id = None
    worst_state_id = None
    if state_scorecard.height > 0:
        sorted_by_ret = state_scorecard.sort("fwd_ret_10_mean", descending=True, nulls_last=True)
        top_state_id = int(sorted_by_ret["state_id"][0])
        worst_state_id = int(sorted_by_ret["state_id"][-1])

    grade = _validation_grade(
        pairwise_share=pairwise_significant_share,
        avg_sign_consistency=avg_sign_consistency,
        avg_ci_width=avg_ci_width,
    )

    validation_scorecard = {
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "input_type": state_label_type,
        "n_states": int(state_scorecard.height),
        "total_rows": int(total_rows),
        "date_min": date_bounds["date_min"].isoformat() if date_bounds["date_min"] is not None else None,
        "date_max": date_bounds["date_max"].isoformat() if date_bounds["date_max"] is not None else None,
        "forward_separation_score": forward_separation_score,
        "pairwise_diff_significant_share": pairwise_significant_share,
        "avg_ci_width_fwd_ret_10": avg_ci_width,
        "avg_state_sign_consistency": avg_sign_consistency,
        "avg_state_ret_cv": avg_ret_cv,
        "top_state_id": top_state_id,
        "worst_state_id": worst_state_id,
        "validation_grade": grade,
        "notes": [
            "finite_only_forward_aggregation",
            f"scorecard_eps={eps}",
        ],
    }

    return ValidationScorecards(
        state_scorecard=state_scorecard,
        validation_scorecard=validation_scorecard,
    )
