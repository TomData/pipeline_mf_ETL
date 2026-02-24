"""HMM state profiling and NaN-safe forward-outcome aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import polars as pl

LOGGER = logging.getLogger(__name__)

FORWARD_LABEL_COLUMNS: tuple[str, ...] = (
    "fwd_ret_5",
    "fwd_ret_10",
    "fwd_ret_20",
    "fwd_abs_ret_10",
    "fwd_vol_proxy_10",
)


@dataclass(frozen=True, slots=True)
class HMMProfiles:
    """State profile tables for decoded HMM rows."""

    hmm_state_profile: pl.DataFrame
    hmm_vs_flow_state_crosstab: pl.DataFrame
    hmm_forward_validation: pl.DataFrame


def _sanitize_forward_columns(df: pl.DataFrame, forward_cols: list[str]) -> pl.DataFrame:
    if not forward_cols:
        return df
    exprs: list[pl.Expr] = []
    for column in forward_cols:
        value = pl.col(column).cast(pl.Float64, strict=False)
        exprs.append(
            pl.when(value.is_finite().fill_null(False))
            .then(value)
            .otherwise(None)
            .alias(column)
        )
    return df.with_columns(exprs)


def _forward_aggregations(forward_cols: list[str]) -> list[pl.Expr]:
    agg: list[pl.Expr] = []
    for column in forward_cols:
        value = pl.col(column)
        agg.extend(
            [
                value.is_not_null().sum().alias(f"{column}_n"),
                value.mean().alias(f"{column}_mean"),
                value.median().alias(f"{column}_median"),
                value.std(ddof=0).alias(f"{column}_std"),
                value.quantile(0.10, interpolation="linear").alias(f"{column}_p10"),
                value.quantile(0.90, interpolation="linear").alias(f"{column}_p90"),
            ]
        )
        if column in {"fwd_ret_5", "fwd_ret_10", "fwd_ret_20"}:
            agg.append(value.gt(0).mean().alias(f"{column}_hit_rate"))
    return agg


def _assert_forward_consistency(
    forward_validation: pl.DataFrame,
    forward_cols: list[str],
) -> None:
    if forward_validation.height == 0:
        return
    issues: list[str] = []
    for column in forward_cols:
        n_col = f"{column}_n"
        if n_col not in forward_validation.columns:
            continue
        for stat in ("mean", "median"):
            stat_col = f"{column}_{stat}"
            if stat_col not in forward_validation.columns:
                continue
            inconsistent = (
                ((pl.col(n_col) > 0) & (pl.col(stat_col).is_null() | pl.col(stat_col).is_nan().fill_null(False) | (~pl.col(stat_col).is_finite()).fill_null(False)))
                | ((pl.col(n_col) == 0) & pl.col(stat_col).is_not_null())
            )
            bad_states = (
                forward_validation.filter(inconsistent).select("hmm_state").to_series().to_list()
            )
            if bad_states:
                issues.append(f"{stat_col} inconsistent for hmm_state={bad_states}")
    if issues:
        message = "; ".join(issues)
        LOGGER.error("hmm_profiles.forward_consistency_failed %s", message)
        raise ValueError(f"HMM forward aggregate consistency check failed. {message}")


def build_hmm_state_profiles(decoded_df: pl.DataFrame) -> HMMProfiles:
    """Build HMM state feature/forward profiles and flow-state overlap table."""

    if "hmm_state" not in decoded_df.columns:
        raise ValueError("decoded_df must include hmm_state.")
    total_rows = decoded_df.height
    if total_rows == 0:
        raise ValueError("decoded_df has zero rows.")

    profile_features = [
        "tmf_21",
        "delta_flow_20",
        "flow_activity_20",
        "flow_bias_20",
        "oscillation_index_20",
        "state_run_length",
        "hmm_state_run_length",
    ]
    existing_features = [column for column in profile_features if column in decoded_df.columns]
    profile_agg: list[pl.Expr] = [
        pl.len().alias("row_count"),
        pl.col("ticker").n_unique().alias("ticker_count"),
        pl.col("trade_date").min().alias("date_min"),
        pl.col("trade_date").max().alias("date_max"),
    ]
    for feature in existing_features:
        profile_agg.extend(
            [
                pl.col(feature).mean().alias(f"{feature}_mean"),
                pl.col(feature).std(ddof=0).alias(f"{feature}_std"),
            ]
        )

    state_profile = (
        decoded_df.group_by("hmm_state")
        .agg(profile_agg)
        .with_columns((pl.col("row_count") / float(total_rows)).alias("share_of_dataset"))
        .sort("hmm_state")
    )

    flow_crosstab = pl.DataFrame(
        schema={
            "hmm_state": pl.Int16,
            "flow_state_label": pl.String,
            "count": pl.Int64,
            "share_in_hmm_state": pl.Float64,
        }
    )
    if "flow_state_label" in decoded_df.columns:
        flow_crosstab = (
            decoded_df.group_by(["hmm_state", "flow_state_label"])
            .len(name="count")
            .join(state_profile.select(["hmm_state", "row_count"]), on="hmm_state", how="left")
            .with_columns((pl.col("count") / pl.col("row_count")).alias("share_in_hmm_state"))
            .select(["hmm_state", "flow_state_label", "count", "share_in_hmm_state"])
            .sort(["hmm_state", "flow_state_label"])
        )

    forward_cols = [column for column in FORWARD_LABEL_COLUMNS if column in decoded_df.columns]
    forward_ready = _sanitize_forward_columns(decoded_df, forward_cols)
    forward_agg = _forward_aggregations(forward_cols)
    forward_validation = (
        forward_ready.group_by("hmm_state").agg(forward_agg).sort("hmm_state")
        if forward_agg
        else pl.DataFrame(schema={"hmm_state": pl.Int16})
    )
    _assert_forward_consistency(forward_validation, forward_cols)

    combined = state_profile.join(forward_validation, on="hmm_state", how="left")
    return HMMProfiles(
        hmm_state_profile=combined,
        hmm_vs_flow_state_crosstab=flow_crosstab,
        hmm_forward_validation=forward_validation,
    )


def build_hmm_vs_cluster_crosstab(decoded_df: pl.DataFrame) -> pl.DataFrame:
    """Build crosstab between HMM states and available clustering labels."""

    if "cluster_label" not in decoded_df.columns and "cluster_id" not in decoded_df.columns:
        raise ValueError("decoded_df must include cluster_label or cluster_id for crosstab.")
    cluster_column = "cluster_label" if "cluster_label" in decoded_df.columns else "cluster_id"
    return (
        decoded_df.filter(pl.col(cluster_column).is_not_null())
        .group_by(["hmm_state", cluster_column])
        .len(name="count")
        .sort(["hmm_state", cluster_column])
    )

