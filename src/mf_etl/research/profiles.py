"""Cluster profiling and forward-return validation helpers."""

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
class ClusterProfiles:
    """Profile artifacts for clustered datasets."""

    cluster_profile: pl.DataFrame
    cluster_state_distribution: pl.DataFrame
    cluster_forward_validation: pl.DataFrame


def _sanitize_forward_columns(df: pl.DataFrame, forward_cols: list[str]) -> pl.DataFrame:
    """Normalize forward columns so non-finite values become null."""

    if not forward_cols:
        return df
    expressions: list[pl.Expr] = []
    for column in forward_cols:
        value = pl.col(column).cast(pl.Float64, strict=False)
        expressions.append(
            pl.when(value.is_finite().fill_null(False))
            .then(value)
            .otherwise(None)
            .alias(column)
        )
    return df.with_columns(expressions)


def _build_forward_aggregations(forward_cols: list[str]) -> list[pl.Expr]:
    """Build NaN-safe forward aggregate expressions."""

    aggregations: list[pl.Expr] = []
    for column in forward_cols:
        value = pl.col(column)
        aggregations.extend(
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
            aggregations.append(value.gt(0).mean().alias(f"{column}_hit_rate"))
    return aggregations


def _assert_forward_aggregation_consistency(
    forward_validation: pl.DataFrame, forward_cols: list[str]
) -> None:
    """Fail fast on inconsistent forward aggregates."""

    if forward_validation.height == 0 or not forward_cols:
        return

    bad_messages: list[str] = []
    for column in forward_cols:
        n_col = f"{column}_n"
        if n_col not in forward_validation.columns:
            continue
        for stat in ("mean", "median"):
            stat_col = f"{column}_{stat}"
            if stat_col not in forward_validation.columns:
                continue
            stat_expr = pl.col(stat_col)
            n_expr = pl.col(n_col)
            inconsistent = (
                ((n_expr > 0) & (stat_expr.is_null() | stat_expr.is_nan().fill_null(False) | (~stat_expr.is_finite()).fill_null(False)))
                | ((n_expr == 0) & stat_expr.is_not_null())
            )
            bad_cluster_ids = (
                forward_validation.filter(inconsistent)
                .select("cluster_id")
                .to_series()
                .to_list()
            )
            if bad_cluster_ids:
                bad_messages.append(
                    f"{stat_col} inconsistent for cluster_id={bad_cluster_ids}"
                )

    if bad_messages:
        message = "; ".join(bad_messages)
        LOGGER.error("cluster_profiles.forward_aggregate_consistency_failed %s", message)
        raise ValueError(
            "Forward aggregate consistency check failed. "
            f"{message}"
        )


def build_cluster_profiles(clustered_df: pl.DataFrame) -> ClusterProfiles:
    """Build interpretable cluster profiles and forward-return validation tables."""

    if "cluster_id" not in clustered_df.columns:
        raise ValueError("clustered_df must include cluster_id.")

    total_rows = clustered_df.height
    profile_features = [
        "tmf_21",
        "delta_flow_20",
        "flow_activity_20",
        "flow_bias_20",
        "oscillation_index_20",
        "state_run_length",
        "long_flow_score_20",
        "short_flow_score_20",
        "persistence_pos_20",
        "persistence_neg_20",
    ]
    existing_profile_features = [feature for feature in profile_features if feature in clustered_df.columns]

    aggregations: list[pl.Expr] = [
        pl.len().alias("row_count"),
        pl.col("ticker").n_unique().alias("ticker_count"),
        pl.col("trade_date").min().alias("date_min"),
        pl.col("trade_date").max().alias("date_max"),
    ]
    for feature in existing_profile_features:
        aggregations.extend(
            [
                pl.col(feature).mean().alias(f"{feature}_mean"),
                pl.col(feature).std(ddof=0).alias(f"{feature}_std"),
            ]
        )

    cluster_profile = (
        clustered_df.group_by("cluster_id")
        .agg(aggregations)
        .sort("cluster_id")
        .with_columns((pl.col("row_count") / float(total_rows)).alias("share_of_dataset"))
    )

    state_distribution = pl.DataFrame(schema={"cluster_id": pl.Int32, "flow_state_label": pl.String, "count": pl.Int64, "share_in_cluster": pl.Float64})
    if "flow_state_label" in clustered_df.columns:
        state_distribution = (
            clustered_df.group_by(["cluster_id", "flow_state_label"])
            .len(name="count")
            .join(cluster_profile.select(["cluster_id", "row_count"]), on="cluster_id", how="left")
            .with_columns((pl.col("count") / pl.col("row_count")).alias("share_in_cluster"))
            .select(["cluster_id", "flow_state_label", "count", "share_in_cluster"])
            .sort(["cluster_id", "flow_state_label"])
        )

    forward_cols = [column for column in FORWARD_LABEL_COLUMNS if column in clustered_df.columns]
    sanitized_df = _sanitize_forward_columns(clustered_df, forward_cols)
    fwd_aggregations = _build_forward_aggregations(forward_cols)

    forward_validation = (
        sanitized_df.group_by("cluster_id").agg(fwd_aggregations).sort("cluster_id")
        if fwd_aggregations
        else pl.DataFrame(schema={"cluster_id": pl.Int32})
    )
    _assert_forward_aggregation_consistency(forward_validation, forward_cols)

    combined_profile = cluster_profile.join(forward_validation, on="cluster_id", how="left")
    return ClusterProfiles(
        cluster_profile=combined_profile,
        cluster_state_distribution=state_distribution,
        cluster_forward_validation=forward_validation,
    )
