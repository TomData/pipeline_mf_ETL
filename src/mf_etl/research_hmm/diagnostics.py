"""Transition, dwell-time, and confidence diagnostics for decoded HMM states."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class HMMDiagnostics:
    """Diagnostics tables derived from decoded HMM rows."""

    transition_counts: pl.DataFrame
    transition_matrix: pl.DataFrame
    initial_state_distribution: pl.DataFrame
    dwell_stats: pl.DataFrame
    state_frequency: pl.DataFrame
    state_confidence_stats: pl.DataFrame


def build_hmm_diagnostics(decoded_df: pl.DataFrame) -> HMMDiagnostics:
    """Build transition/dwell/frequency diagnostics from decoded rows."""

    if "hmm_state" not in decoded_df.columns or "ticker" not in decoded_df.columns:
        raise ValueError("decoded_df must include ticker and hmm_state.")

    transition_counts = (
        decoded_df.filter(pl.col("hmm_state_prev").is_not_null())
        .group_by(["hmm_state_prev", "hmm_state"])
        .len(name="transition_count")
        .with_columns(
            [
                pl.col("hmm_state_prev").cast(pl.Int16, strict=False),
                pl.col("hmm_state").cast(pl.Int16, strict=False),
            ]
        )
        .sort(["hmm_state_prev", "hmm_state"])
    )
    transition_matrix = (
        transition_counts.join(
            transition_counts.group_by("hmm_state_prev")
            .agg(pl.col("transition_count").sum().alias("from_state_total")),
            on="hmm_state_prev",
            how="left",
        )
        .with_columns(
            (pl.col("transition_count") / pl.col("from_state_total")).alias("transition_probability")
        )
        .select(["hmm_state_prev", "hmm_state", "transition_count", "transition_probability"])
        .sort(["hmm_state_prev", "hmm_state"])
    )

    initial_state_distribution = (
        decoded_df.sort(["ticker", "trade_date"])
        .group_by("ticker", maintain_order=True)
        .first()
        .group_by("hmm_state")
        .len(name="start_count")
        .with_columns((pl.col("start_count") / pl.col("start_count").sum()).alias("start_share"))
        .sort("hmm_state")
    )

    runs = (
        decoded_df.sort(["ticker", "trade_date"])
        .with_columns(pl.col("hmm_state_changed").cast(pl.Int32).cum_sum().over("ticker").alias("__hmm_run_id"))
        .group_by(["ticker", "__hmm_run_id"])
        .agg(
            [
                pl.col("hmm_state").first().alias("hmm_state"),
                pl.col("hmm_state_run_length").max().alias("dwell_length"),
            ]
        )
    )
    dwell_stats = (
        runs.group_by("hmm_state")
        .agg(
            [
                pl.len().alias("dwell_count"),
                pl.col("dwell_length").mean().alias("dwell_mean"),
                pl.col("dwell_length").median().alias("dwell_median"),
                pl.col("dwell_length").quantile(0.10, interpolation="linear").alias("dwell_p10"),
                pl.col("dwell_length").quantile(0.90, interpolation="linear").alias("dwell_p90"),
                pl.col("dwell_length").max().alias("max_dwell"),
            ]
        )
        .sort("hmm_state")
    )

    state_frequency = (
        decoded_df.group_by("hmm_state")
        .len(name="row_count")
        .with_columns((pl.col("row_count") / pl.col("row_count").sum()).alias("share_of_rows"))
        .sort("hmm_state")
    )

    if "hmm_state_prob_max" in decoded_df.columns:
        state_confidence_stats = (
            decoded_df.group_by("hmm_state")
            .agg(
                [
                    pl.col("hmm_state_prob_max").mean().alias("prob_max_mean"),
                    pl.col("hmm_state_prob_max").max().alias("prob_max_max"),
                    pl.col("hmm_state_entropy").mean().alias("entropy_mean"),
                    pl.col("hmm_state_entropy").median().alias("entropy_median"),
                ]
            )
            .sort("hmm_state")
        )
    else:
        state_confidence_stats = pl.DataFrame(
            schema={
                "hmm_state": pl.Int16,
                "prob_max_mean": pl.Float64,
                "prob_max_max": pl.Float64,
                "entropy_mean": pl.Float64,
                "entropy_median": pl.Float64,
            }
        )

    return HMMDiagnostics(
        transition_counts=transition_counts,
        transition_matrix=transition_matrix,
        initial_state_distribution=initial_state_distribution,
        dwell_stats=dwell_stats,
        state_frequency=state_frequency,
        state_confidence_stats=state_confidence_stats,
    )

