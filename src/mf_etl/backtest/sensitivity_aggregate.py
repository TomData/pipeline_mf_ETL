"""Aggregation and robustness scoring utilities for sensitivity grid runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _rank_score(values: pl.Series, *, descending: bool) -> list[float]:
    """Map values to [0,1] rank score where 1 is best."""

    length = len(values)
    if length == 0:
        return []
    if length == 1:
        return [1.0]

    raw = values.to_list()
    indexed = [(idx, _as_float(val)) for idx, val in enumerate(raw)]
    valid = [(idx, val) for idx, val in indexed if val is not None]
    if not valid:
        return [0.5 for _ in raw]

    valid_sorted = sorted(valid, key=lambda x: x[1], reverse=descending)
    ranks: dict[int, int] = {}
    for rank, (idx, _) in enumerate(valid_sorted, start=1):
        ranks[idx] = rank

    worst_rank = len(valid_sorted)
    out: list[float] = []
    for idx, val in indexed:
        if val is None:
            out.append(0.0)
            continue
        rank = ranks.get(idx, worst_rank)
        out.append(1.0 - ((rank - 1) / (max(worst_rank - 1, 1))))
    return out


def _hygiene_score(trade_count: pl.Series, nan_warning_total: pl.Series) -> list[float]:
    out: list[float] = []
    for trades, nan_warn in zip(trade_count.to_list(), nan_warning_total.to_list(), strict=False):
        t = _as_float(trades) or 0.0
        n = _as_float(nan_warn) or 0.0
        trade_part = min(1.0, t / 100.0)
        warn_part = 1.0 if n <= 0 else 0.0
        out.append((0.7 * trade_part) + (0.3 * warn_part))
    return out


def _clip_0_100(values: list[float]) -> list[float]:
    return [float(max(0.0, min(100.0, v))) for v in values]


def compute_metrics_table(
    manifest: pl.DataFrame,
    *,
    weights: Any,
) -> pl.DataFrame:
    """Compute derived sensitivity metrics and robustness scores per successful combo."""

    if manifest.height == 0:
        return manifest
    success = manifest.filter(pl.col("status") == "SUCCESS")
    if success.height == 0:
        return success

    pieces: list[pl.DataFrame] = []
    for source in sorted(set(success.get_column("source_type").to_list())):
        sub = success.filter(pl.col("source_type") == source)
        expect_rank = _rank_score(sub.get_column("expectancy"), descending=True)
        pf_rank = _rank_score(sub.get_column("profit_factor"), descending=True)

        drawdown_abs = sub.select(pl.col("max_drawdown").cast(pl.Float64, strict=False).abs().alias("dd")).get_column("dd")
        drawdown_rank = _rank_score(drawdown_abs, descending=False)
        consistency_rank = _rank_score(sub.get_column("return_std"), descending=False)
        ret_cv_rank = _rank_score(sub.get_column("ret_cv"), descending=False)
        tail_rank = _rank_score(sub.get_column("downside_std"), descending=False)

        total_cost = (
            sub.get_column("fee_bps_per_side").cast(pl.Float64, strict=False)
            + sub.get_column("slippage_bps_per_side").cast(pl.Float64, strict=False)
        )
        cost_rank = _rank_score(total_cost, descending=False)
        hygiene = _hygiene_score(sub.get_column("trade_count"), sub.get_column("nan_warning_total"))

        scored = sub.with_columns(
            pl.Series("rank_expectancy", expect_rank),
            pl.Series("rank_profit_factor", pf_rank),
            pl.Series("drawdown_score", drawdown_rank),
            pl.Series("consistency_score", consistency_rank),
            pl.Series("cost_robustness_score", cost_rank),
            pl.Series("hygiene_score", hygiene),
            pl.Series("ret_cv_score", ret_cv_rank),
            pl.Series("tail_stability_score", tail_rank),
            total_cost.alias("total_cost_bps"),
        )

        v1 = (
            scored.get_column("rank_expectancy") * float(weights.expectancy_rank)
            + scored.get_column("rank_profit_factor") * float(weights.profit_factor_rank)
            + scored.get_column("drawdown_score") * float(weights.drawdown_score)
            + scored.get_column("consistency_score") * float(weights.consistency)
            + scored.get_column("cost_robustness_score") * float(weights.cost_robustness)
            + scored.get_column("hygiene_score") * float(weights.hygiene)
        ) * 100.0

        # v2 extends v1 with explicit fragility penalties/bonuses.
        zero_trade = scored.get_column("is_zero_trade_combo").cast(pl.Int8, strict=False).fill_null(1).to_list()
        v2_list: list[float] = []
        for base, ret_score, tail_score, zero_flag, cost_score in zip(
            v1.to_list(),
            scored.get_column("ret_cv_score").to_list(),
            scored.get_column("tail_stability_score").to_list(),
            zero_trade,
            scored.get_column("cost_robustness_score").to_list(),
            strict=False,
        ):
            base_val = _as_float(base) or 0.0
            ret_val = _as_float(ret_score) or 0.0
            tail_val = _as_float(tail_score) or 0.0
            cost_val = _as_float(cost_score) or 0.0
            adjusted = (0.70 * base_val) + (15.0 * ret_val) + (10.0 * tail_val) + (5.0 * cost_val)
            if int(zero_flag or 0) == 1:
                adjusted -= 20.0
            v2_list.append(adjusted)

        scored = scored.with_columns(
            pl.Series("robustness_score_v1", _clip_0_100([float(v) for v in v1.to_list()])),
            pl.Series("robustness_score_v2", _clip_0_100(v2_list)),
        ).with_columns(pl.col("robustness_score_v2").alias("robustness_score"))
        pieces.append(scored)

    table = pl.concat(pieces, how="vertical") if pieces else pl.DataFrame()
    if table.height == 0:
        return table

    # Convert any non-finite metrics to null for parquet/csv safety.
    numeric_cols = [name for name, dtype in table.schema.items() if dtype.is_numeric()]
    exprs: list[pl.Expr] = []
    for col in numeric_cols:
        exprs.append(
            pl.when(pl.col(col).cast(pl.Float64, strict=False).is_finite())
            .then(pl.col(col))
            .otherwise(None)
            .alias(col)
        )
    return table.with_columns(exprs)


def best_configs_by_metric(
    metrics_table: pl.DataFrame,
    *,
    top_n: int = 5,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Collect top-N configs for key metrics per source."""

    if metrics_table.height == 0:
        return {}
    requested = metrics or ["expectancy", "profit_factor", "robustness_score_v2"]
    out: dict[str, Any] = {}
    for source in sorted(set(metrics_table.get_column("source_type").to_list())):
        sub = metrics_table.filter(pl.col("source_type") == source)
        metric_payload: dict[str, Any] = {}
        for metric in requested:
            if metric not in sub.columns:
                continue
            descending = metric not in {"max_drawdown", "return_std", "ret_cv", "downside_std", "total_cost_bps"}
            metric_payload[metric] = (
                sub.sort(metric, descending=descending)
                .head(top_n)
                .select(
                    [
                        "combo_id",
                        "combo_index",
                        "source_type",
                        "hold_bars",
                        "signal_mode",
                        "exit_mode",
                        "fee_bps_per_side",
                        "slippage_bps_per_side",
                        "allow_overlap",
                        "equity_mode",
                        "include_watch",
                        "policy_filter_mode",
                        "overlay_mode",
                        "overlay_match_rate",
                        "overlay_vetoed_signal_share",
                        "state_subset_key",
                        "trade_count",
                        "expectancy",
                        "profit_factor",
                        "ret_cv",
                        "downside_std",
                        "max_drawdown",
                        "robustness_score_v1",
                        "robustness_score_v2",
                    ]
                )
                .to_dicts()
            )
        out[source] = metric_payload
    return out


def build_dimension_sensitivity(
    metrics_table: pl.DataFrame,
    *,
    manifest: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build grouped metric summaries per dimension value and source."""

    if metrics_table.height == 0:
        return pl.DataFrame()

    dims = [
        "hold_bars",
        "signal_mode",
        "exit_mode",
        "fee_bps_per_side",
        "slippage_bps_per_side",
        "allow_overlap",
        "equity_mode",
        "include_watch",
        "policy_filter_mode",
        "state_subset_key",
        "overlay_mode",
        "overlay_enabled",
        "execution_profile",
        "execution_filters_enabled",
    ]
    work = metrics_table.with_columns(
        pl.col("is_zero_trade_combo").cast(pl.Int8, strict=False).fill_null(1).alias("is_zero_trade_combo_int")
    )

    threshold_df = (
        work.group_by("source_type")
        .agg(pl.col("robustness_score_v2").cast(pl.Float64, strict=False).quantile(0.75).alias("r75"))
    )
    work = work.join(threshold_df, on="source_type", how="left").with_columns(
        (pl.col("robustness_score_v2") >= pl.col("r75")).cast(pl.Int8).alias("is_top_quartile")
    )

    frames: list[pl.DataFrame] = []
    for dim in dims:
        if dim not in work.columns:
            continue
        grouped = (
            work.group_by(["source_type", dim])
            .agg(
                pl.len().alias("combo_count"),
                pl.col("expectancy").cast(pl.Float64, strict=False).mean().alias("expectancy_mean"),
                pl.col("expectancy").cast(pl.Float64, strict=False).median().alias("expectancy_median"),
                pl.col("expectancy").cast(pl.Float64, strict=False).std().alias("expectancy_std"),
                pl.col("profit_factor").cast(pl.Float64, strict=False).mean().alias("profit_factor_mean"),
                pl.col("ret_cv").cast(pl.Float64, strict=False).mean().alias("ret_cv_mean"),
                pl.col("ret_cv").cast(pl.Float64, strict=False).median().alias("ret_cv_median"),
                pl.col("downside_std").cast(pl.Float64, strict=False).mean().alias("downside_std_mean"),
                pl.col("is_zero_trade_combo_int").cast(pl.Float64, strict=False).mean().alias("zero_trade_rate"),
                pl.col("is_top_quartile").cast(pl.Float64, strict=False).mean().alias("top_quartile_hit_rate"),
                pl.col("robustness_score_v2").cast(pl.Float64, strict=False).mean().alias("robustness_v2_mean"),
                pl.col("overlay_match_rate").cast(pl.Float64, strict=False).mean().alias("overlay_match_rate_mean"),
                pl.col("overlay_unknown_rate").cast(pl.Float64, strict=False).mean().alias("overlay_unknown_rate_mean"),
                pl.col("overlay_vetoed_signal_share")
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("overlay_vetoed_signal_share_mean"),
                pl.col("exec_eligibility_rate")
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("exec_eligibility_rate_mean"),
                pl.col("exec_suppressed_signal_share")
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("exec_suppressed_signal_share_mean"),
                pl.col("exec_trade_avg_dollar_vol_20")
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("exec_trade_avg_dollar_vol_20_mean"),
            )
            .with_columns(
                pl.lit(dim).alias("dimension"),
                pl.col(dim).cast(pl.String).alias("dimension_value"),
            )
            .drop(dim)
        )
        if manifest is not None and manifest.height > 0 and dim in manifest.columns:
            fail_counts = (
                manifest.group_by(["source_type", dim])
                .agg(
                    pl.len().alias("combo_total"),
                    (pl.col("status") == "FAILED").cast(pl.Int8).sum().alias("failed_count"),
                )
                .with_columns(pl.col(dim).cast(pl.String).alias("dimension_value"))
                .drop(dim)
                .with_columns(
                    (pl.col("failed_count") / pl.col("combo_total")).cast(pl.Float64).alias("failure_rate")
                )
            )
            grouped = grouped.join(fail_counts, on=["source_type", "dimension_value"], how="left")
        frames.append(grouped)

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical").sort(["source_type", "dimension", "dimension_value"])


def _fit_slope(costs: list[float], values: list[float]) -> float | None:
    finite_points = [
        (float(c), float(v))
        for c, v in zip(costs, values, strict=False)
        if c is not None and v is not None and np.isfinite(c) and np.isfinite(v)
    ]
    if len({c for c, _ in finite_points}) < 2:
        return None
    x = np.array([c for c, _ in finite_points], dtype=np.float64) / 10.0
    y = np.array([v for _, v in finite_points], dtype=np.float64)
    return float(np.polyfit(x, y, deg=1)[0])


def build_cost_fragility(metrics_table: pl.DataFrame) -> pl.DataFrame:
    """Estimate metric slopes against total cost bps per source and config family."""

    if metrics_table.height == 0:
        return pl.DataFrame()

    family_cols = [
        "source_type",
        "hold_bars",
        "signal_mode",
        "exit_mode",
        "allow_overlap",
        "equity_mode",
        "include_watch",
        "policy_filter_mode",
        "state_subset_key",
        "overlay_mode",
        "overlay_enabled",
        "execution_profile",
        "execution_filters_enabled",
    ]

    grouped = (
        metrics_table.group_by(family_cols + ["total_cost_bps"])
        .agg(
            pl.col("expectancy").cast(pl.Float64, strict=False).mean().alias("expectancy_mean"),
            pl.col("profit_factor").cast(pl.Float64, strict=False).mean().alias("profit_factor_mean"),
            pl.col("ret_cv").cast(pl.Float64, strict=False).mean().alias("ret_cv_mean"),
            pl.col("trade_count").cast(pl.Float64, strict=False).mean().alias("trade_count_mean"),
            pl.col("is_zero_trade_combo").cast(pl.Float64, strict=False).mean().alias("zero_trade_rate"),
        )
        .sort(family_cols + ["total_cost_bps"])
    )

    rows: list[dict[str, Any]] = []
    for family, sub in grouped.partition_by(family_cols, as_dict=True).items():
        costs = sub.get_column("total_cost_bps").cast(pl.Float64, strict=False).to_list()
        expect = sub.get_column("expectancy_mean").cast(pl.Float64, strict=False).to_list()
        pf = sub.get_column("profit_factor_mean").cast(pl.Float64, strict=False).to_list()
        ret_cv = sub.get_column("ret_cv_mean").cast(pl.Float64, strict=False).to_list()
        trade_count = sub.get_column("trade_count_mean").cast(pl.Float64, strict=False).to_list()
        zero_rate = sub.get_column("zero_trade_rate").cast(pl.Float64, strict=False).to_list()

        finite_costs = [float(c) for c in costs if c is not None and np.isfinite(c)]
        finite_zero = [float(v) for v in zero_rate if v is not None and np.isfinite(v)]

        record = {key: val for key, val in zip(family_cols, family, strict=True)}
        record.update(
            {
                "cost_points": len(costs),
                "min_cost_bps": float(min(finite_costs)) if finite_costs else None,
                "max_cost_bps": float(max(finite_costs)) if finite_costs else None,
                "expectancy_slope_per_10bps": _as_float(_fit_slope(costs, expect)),
                "profit_factor_slope_per_10bps": _as_float(_fit_slope(costs, pf)),
                "ret_cv_slope_per_10bps": _as_float(_fit_slope(costs, ret_cv)),
                "trade_count_slope_per_10bps": _as_float(_fit_slope(costs, trade_count)),
                "zero_trade_rate_delta_0_to_max_cost": (
                    _as_float(finite_zero[-1] - finite_zero[0]) if len(finite_zero) >= 2 else None
                ),
            }
        )
        rows.append(record)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def classify_hold_shape(holdbars_profile: pl.DataFrame, *, metric_col: str = "expectancy_mean") -> pl.DataFrame:
    """Classify hold-horizon sensitivity shape per source."""

    if holdbars_profile.height == 0:
        return pl.DataFrame()
    out_rows: list[dict[str, Any]] = []
    for source, sub in holdbars_profile.partition_by("source_type", as_dict=True).items():
        data = sub.sort("hold_bars")
        holds = data.get_column("hold_bars").cast(pl.Int64, strict=False).to_list()
        vals = data.get_column(metric_col).cast(pl.Float64, strict=False).to_list()
        finite = [(h, v) for h, v in zip(holds, vals, strict=False) if h is not None and v is not None and np.isfinite(v)]
        shape = "FLAT_OR_NOISY"
        best_hold = None
        if len(finite) >= 2:
            f_h = [int(h) for h, _ in finite]
            f_v = [float(v) for _, v in finite]
            diffs = [f_v[i] - f_v[i - 1] for i in range(1, len(f_v))]
            tol = max(1e-8, 0.05 * np.std(np.array(f_v, dtype=np.float64)))
            if all(d >= -tol for d in diffs) and any(d > tol for d in diffs):
                shape = "MONOTONIC_UP"
            elif all(d <= tol for d in diffs) and any(d < -tol for d in diffs):
                shape = "MONOTONIC_DOWN"
            else:
                idx_max = int(np.argmax(np.array(f_v, dtype=np.float64)))
                if 0 < idx_max < len(f_v) - 1:
                    left = all(f_v[i] <= f_v[i + 1] + tol for i in range(0, idx_max))
                    right = all(f_v[i] >= f_v[i + 1] - tol for i in range(idx_max, len(f_v) - 1))
                    if left and right:
                        shape = "HUMP_SHAPED"
            best_hold = int(f_h[int(np.argmax(np.array(f_v, dtype=np.float64)))])
        out_rows.append(
            {
                "source_type": str(source[0] if isinstance(source, tuple) else source),
                "hold_shape": shape,
                "best_hold_bars": best_hold,
                "metric_col": metric_col,
            }
        )
    return pl.DataFrame(out_rows)


def build_holdbars_profile(metrics_table: pl.DataFrame) -> pl.DataFrame:
    """Build hold-bars profile summaries by source."""

    if metrics_table.height == 0:
        return pl.DataFrame()
    return (
        metrics_table.group_by(["source_type", "hold_bars"])
        .agg(
            pl.len().alias("combo_count"),
            pl.col("expectancy").cast(pl.Float64, strict=False).mean().alias("expectancy_mean"),
            pl.col("expectancy").cast(pl.Float64, strict=False).median().alias("expectancy_median"),
            pl.col("profit_factor").cast(pl.Float64, strict=False).mean().alias("profit_factor_mean"),
            pl.col("ret_cv").cast(pl.Float64, strict=False).median().alias("ret_cv_median"),
            pl.col("downside_std").cast(pl.Float64, strict=False).mean().alias("downside_std_mean"),
            pl.col("is_zero_trade_combo").cast(pl.Float64, strict=False).mean().alias("zero_trade_rate"),
            pl.col("robustness_score_v2").cast(pl.Float64, strict=False).mean().alias("robustness_v2_mean"),
            pl.col("trade_count").cast(pl.Float64, strict=False).mean().alias("trade_count_mean"),
        )
        .sort(["source_type", "hold_bars"])
    )


def build_source_summary(metrics_table: pl.DataFrame, manifest: pl.DataFrame) -> pl.DataFrame:
    """Build top-line source summary across all combinations."""

    if manifest.height == 0:
        return pl.DataFrame()

    counts = (
        manifest.group_by(["source_type", "status"])
        .agg(pl.len().alias("n"))
        .pivot(index="source_type", on="status", values="n")
        .fill_null(0)
    )

    if metrics_table.height == 0:
        return counts

    stats = (
        metrics_table.group_by("source_type")
        .agg(
            pl.len().alias("success_count"),
            pl.col("robustness_score_v1").cast(pl.Float64, strict=False).mean().alias("robustness_v1_mean"),
            pl.col("robustness_score_v2").cast(pl.Float64, strict=False).mean().alias("robustness_v2_mean"),
            pl.col("robustness_score_v2").cast(pl.Float64, strict=False).median().alias("robustness_v2_median"),
            pl.col("expectancy").cast(pl.Float64, strict=False).mean().alias("expectancy_mean"),
            pl.col("expectancy").cast(pl.Float64, strict=False).median().alias("expectancy_median"),
            pl.col("profit_factor").cast(pl.Float64, strict=False).mean().alias("profit_factor_mean"),
            pl.col("ret_cv").cast(pl.Float64, strict=False).median().alias("ret_cv_median"),
            pl.col("downside_std").cast(pl.Float64, strict=False).mean().alias("downside_std_mean"),
            pl.col("is_zero_trade_combo").cast(pl.Float64, strict=False).mean().alias("zero_trade_combo_share"),
            pl.col("trade_count").cast(pl.Float64, strict=False).mean().alias("trade_count_mean"),
            pl.col("rows_input").cast(pl.Float64, strict=False).median().alias("rows_input_median"),
            pl.col("nan_warning_total").cast(pl.Float64, strict=False).sum().alias("nan_warning_total_sum"),
            pl.col("null_metric_count").cast(pl.Float64, strict=False).mean().alias("null_metric_cells_mean"),
            pl.col("overlay_enabled").cast(pl.Float64, strict=False).mean().alias("overlay_enabled_share"),
            pl.col("execution_filters_enabled")
            .cast(pl.Float64, strict=False)
            .mean()
            .alias("execution_filters_enabled_share"),
            pl.col("overlay_match_rate").cast(pl.Float64, strict=False).mean().alias("overlay_match_rate_mean"),
            pl.col("overlay_unknown_rate").cast(pl.Float64, strict=False).mean().alias("overlay_unknown_rate_mean"),
            pl.col("overlay_vetoed_signal_share")
            .cast(pl.Float64, strict=False)
            .mean()
            .alias("overlay_vetoed_signal_share_mean"),
            pl.col("overlay_direction_conflict_share")
            .cast(pl.Float64, strict=False)
            .mean()
            .alias("overlay_direction_conflict_share_mean"),
            pl.col("exec_eligibility_rate")
            .cast(pl.Float64, strict=False)
            .mean()
            .alias("exec_eligibility_rate_mean"),
            pl.col("exec_suppressed_signal_share")
            .cast(pl.Float64, strict=False)
            .mean()
            .alias("exec_suppressed_signal_share_mean"),
            pl.col("exec_trade_avg_dollar_vol_20")
            .cast(pl.Float64, strict=False)
            .mean()
            .alias("exec_trade_avg_dollar_vol_20_mean"),
        )
        .sort("source_type")
    )
    return counts.join(stats, on="source_type", how="outer_coalesce").sort("source_type")


def classify_universe_comparability(metrics_table: pl.DataFrame) -> str:
    """Classify row-universe comparability for multi-source runs."""

    if metrics_table.height == 0:
        return "DIFFERENT_UNIVERSE"
    if "source_type" not in metrics_table.columns:
        return "DIFFERENT_UNIVERSE"
    by_source = (
        metrics_table.group_by("source_type")
        .agg(pl.col("rows_input").cast(pl.Float64, strict=False).median().alias("rows_input_median"))
        .drop_nulls("rows_input_median")
    )
    if by_source.height < 2:
        return "ALIGNED"

    vals = by_source.get_column("rows_input_median").to_list()
    min_v = min(vals)
    max_v = max(vals)
    if min_v <= 0:
        return "DIFFERENT_UNIVERSE"
    rel = (max_v - min_v) / min_v
    if rel <= 0.0001:
        return "ALIGNED"
    if rel <= 0.05:
        return "NEAR_ALIGNED"
    return "DIFFERENT_UNIVERSE"
