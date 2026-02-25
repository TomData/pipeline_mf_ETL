"""Sanity and QA checks for backtest sensitivity grid artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


REQUIRED_GRID_FILES = [
    "grid_manifest.parquet",
    "grid_metrics_table.parquet",
    "grid_summary.json",
    "grid_robustness_v2_table.parquet",
]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _is_finite_or_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, bool)):
        return True
    if isinstance(value, float):
        return bool(np.isfinite(value))
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True
    return bool(np.isfinite(v))


def summarize_grid_run(grid_run_dir: Path) -> dict[str, Any]:
    """Validate one sensitivity grid run directory and return diagnostics."""

    for name in REQUIRED_GRID_FILES:
        _require(grid_run_dir / name)

    manifest = pl.read_parquet(grid_run_dir / "grid_manifest.parquet")
    metrics = pl.read_parquet(grid_run_dir / "grid_metrics_table.parquet")
    summary = json.loads((grid_run_dir / "grid_summary.json").read_text(encoding="utf-8"))

    errors: list[str] = []
    warnings: list[str] = []

    if manifest.height == 0:
        errors.append("empty_manifest")

    combo_unique = manifest.select(pl.col("combo_id").n_unique()).item() if manifest.height > 0 else 0
    if manifest.height != int(combo_unique):
        errors.append("duplicate_combo_id")

    status_counts = (
        manifest.group_by("status").agg(pl.len().alias("n")).to_dicts() if manifest.height > 0 else []
    )
    status_map = {row["status"]: int(row["n"]) for row in status_counts}
    total = manifest.height
    if total != status_map.get("SUCCESS", 0) + status_map.get("FAILED", 0) + status_map.get("SKIPPED", 0):
        errors.append("status_count_mismatch")

    bad_runs: list[str] = []
    for row in manifest.filter(pl.col("status") == "SUCCESS").select("backtest_run_dir").to_dicts():
        run_dir_raw = row.get("backtest_run_dir")
        if not run_dir_raw:
            bad_runs.append("<missing>")
            continue
        run_dir = Path(str(run_dir_raw))
        if not run_dir.exists() or not (run_dir / "backtest_summary.json").exists():
            bad_runs.append(str(run_dir))
    if bad_runs:
        errors.append("missing_linked_backtest_runs")

    non_finite_cells = 0
    null_metric_cells = 0
    for col, dtype in metrics.schema.items():
        if dtype.is_numeric():
            non_finite_cells += int(
                metrics.select(
                    (
                        (~pl.col(col).cast(pl.Float64, strict=False).is_finite())
                        & pl.col(col).cast(pl.Float64, strict=False).is_not_null()
                    ).sum()
                ).item()
            )
            null_metric_cells += int(metrics.select(pl.col(col).is_null().sum()).item())
    if non_finite_cells > 0:
        errors.append("non_finite_metrics_cells")

    if "robustness_score_v2" in metrics.columns and "is_zero_trade_combo" in metrics.columns:
        invalid_v2 = int(
            metrics.filter(
                (pl.col("is_zero_trade_combo") == False)
                & (
                    pl.col("robustness_score_v2").is_null()
                    | (~pl.col("robustness_score_v2").cast(pl.Float64, strict=False).is_finite())
                )
            ).height
        )
        if invalid_v2 > 0:
            errors.append("invalid_robustness_v2_for_non_zero_trades")

    if "overlay_enabled" in metrics.columns:
        overlay_enabled_count = int(metrics.filter(pl.col("overlay_enabled") == True).height)
        if overlay_enabled_count > 0:
            required_overlay_cols = [
                "overlay_mode",
                "overlay_coverage_mode",
                "overlay_coverage_status",
                "overlay_match_rate",
                "overlay_unknown_rate",
                "overlay_vetoed_signal_share",
            ]
            missing_overlay_cols = [c for c in required_overlay_cols if c not in metrics.columns]
            if missing_overlay_cols:
                errors.append(f"missing_overlay_metric_columns:{','.join(missing_overlay_cols)}")
            if "overlay_vetoed_signal_share" in metrics.columns:
                bad_veto = int(
                    metrics.filter(
                        pl.col("overlay_enabled")
                        & (
                            pl.col("overlay_vetoed_signal_share").cast(pl.Float64, strict=False).is_null()
                            | (~pl.col("overlay_vetoed_signal_share").cast(pl.Float64, strict=False).is_finite())
                            | (pl.col("overlay_vetoed_signal_share") < 0)
                            | (pl.col("overlay_vetoed_signal_share") > 1)
                        )
                    ).height
                )
                if bad_veto > 0:
                    errors.append("overlay_veto_share_invalid")
            bad_match = int(
                metrics.filter(
                    pl.col("overlay_enabled")
                    & (
                        pl.col("overlay_match_rate").cast(pl.Float64, strict=False).is_null()
                        | (~pl.col("overlay_match_rate").cast(pl.Float64, strict=False).is_finite())
                        | (pl.col("overlay_match_rate") < 0)
                        | (pl.col("overlay_match_rate") > 1)
                    )
                ).height
            )
            if bad_match > 0:
                errors.append("overlay_match_rate_invalid")
            bad_unknown = int(
                metrics.filter(
                    pl.col("overlay_enabled")
                    & (
                        pl.col("overlay_unknown_rate").cast(pl.Float64, strict=False).is_null()
                        | (~pl.col("overlay_unknown_rate").cast(pl.Float64, strict=False).is_finite())
                        | (pl.col("overlay_unknown_rate") < 0)
                        | (pl.col("overlay_unknown_rate") > 1)
                    )
                ).height
            )
            if bad_unknown > 0:
                errors.append("overlay_unknown_rate_invalid")

            missing_coverage_files = 0
            strict_fail_invalid = 0
            for row in metrics.filter(pl.col("overlay_enabled") == True).select(
                ["backtest_run_dir", "overlay_coverage_mode", "overlay_coverage_status"]
            ).to_dicts():
                run_dir_raw = row.get("backtest_run_dir")
                if not run_dir_raw:
                    continue
                run_dir = Path(str(run_dir_raw))
                verdict_path = run_dir / "overlay_coverage_verdict.json"
                join_summary_path = run_dir / "overlay_join_summary.json"
                if not verdict_path.exists():
                    missing_coverage_files += 1
                    continue
                if not join_summary_path.exists():
                    missing_coverage_files += 1
                    continue
                verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
                summary_payload = json.loads(join_summary_path.read_text(encoding="utf-8"))
                for key in ["match_rate", "unknown_rate"]:
                    val = summary_payload.get(key)
                    if val is None:
                        errors.append(f"overlay_{key}_missing_in_join_summary")
                        continue
                    val_f = float(val)
                    if val_f < 0 or val_f > 1:
                        errors.append(f"overlay_{key}_out_of_range_in_join_summary")
                mode = str(row.get("overlay_coverage_mode") or "warn_only")
                status = str(verdict.get("status", ""))
                if mode == "strict_fail" and status.startswith("FAIL"):
                    strict_fail_invalid += 1
            if missing_coverage_files > 0:
                errors.append(f"missing_overlay_coverage_verdict_artifacts:{missing_coverage_files}")
            if strict_fail_invalid > 0:
                errors.append(f"overlay_strict_fail_verdict_present:{strict_fail_invalid}")

    if "execution_filters_enabled" in metrics.columns:
        exec_enabled_count = int(metrics.filter(pl.col("execution_filters_enabled") == True).height)
        if exec_enabled_count > 0:
            required_exec_cols = [
                "execution_profile",
                "exec_eligibility_rate",
                "exec_suppressed_signal_share",
            ]
            missing_exec_cols = [c for c in required_exec_cols if c not in metrics.columns]
            if missing_exec_cols:
                errors.append(f"missing_execution_metric_columns:{','.join(missing_exec_cols)}")
            bad_exec_elig = int(
                metrics.filter(
                    pl.col("execution_filters_enabled")
                    & (
                        pl.col("exec_eligibility_rate").cast(pl.Float64, strict=False).is_null()
                        | (~pl.col("exec_eligibility_rate").cast(pl.Float64, strict=False).is_finite())
                        | (pl.col("exec_eligibility_rate") < 0)
                        | (pl.col("exec_eligibility_rate") > 1)
                    )
                ).height
            )
            if bad_exec_elig > 0:
                errors.append("exec_eligibility_rate_invalid")
            bad_exec_supp = int(
                metrics.filter(
                    pl.col("execution_filters_enabled")
                    & (
                        pl.col("exec_suppressed_signal_share").cast(pl.Float64, strict=False).is_null()
                        | (~pl.col("exec_suppressed_signal_share").cast(pl.Float64, strict=False).is_finite())
                        | (pl.col("exec_suppressed_signal_share") < 0)
                        | (pl.col("exec_suppressed_signal_share") > 1)
                    )
                ).height
            )
            if bad_exec_supp > 0:
                errors.append("exec_suppressed_signal_share_invalid")
            if all(
                col in metrics.columns
                for col in [
                    "exec_candidate_signals_before_filters",
                    "exec_candidate_signals_after_filters",
                    "exec_suppressed_signal_count",
                ]
            ):
                bad_order = int(
                    metrics.filter(
                        pl.col("execution_filters_enabled")
                        & (
                            pl.col("exec_candidate_signals_after_filters").cast(pl.Float64, strict=False)
                            > pl.col("exec_candidate_signals_before_filters").cast(pl.Float64, strict=False)
                        )
                    ).height
                )
                if bad_order > 0:
                    errors.append("execution_candidate_count_invalid_order")
                bad_mismatch = int(
                    metrics.filter(
                        pl.col("execution_filters_enabled")
                        & (
                            pl.col("exec_suppressed_signal_count").cast(pl.Float64, strict=False)
                            != (
                                pl.col("exec_candidate_signals_before_filters").cast(pl.Float64, strict=False)
                                - pl.col("exec_candidate_signals_after_filters").cast(pl.Float64, strict=False)
                            )
                        )
                    ).height
                )
                if bad_mismatch > 0:
                    errors.append("execution_suppressed_count_mismatch")
            for share_col in [
                "exec_suppressed_by_price_share",
                "exec_suppressed_by_liquidity_share",
                "exec_suppressed_by_vol_share",
                "exec_suppressed_by_warmup_share",
            ]:
                if share_col not in metrics.columns:
                    continue
                bad_share = int(
                    metrics.filter(
                        pl.col("execution_filters_enabled")
                        & (
                            pl.col(share_col).cast(pl.Float64, strict=False).is_null()
                            | (~pl.col(share_col).cast(pl.Float64, strict=False).is_finite())
                            | (pl.col(share_col).cast(pl.Float64, strict=False) < 0)
                            | (pl.col(share_col).cast(pl.Float64, strict=False) > 1)
                        )
                    ).height
                )
                if bad_share > 0:
                    errors.append(f"{share_col}_invalid")
            for count_col in [
                "exec_suppressed_by_price_count",
                "exec_suppressed_by_liquidity_count",
                "exec_suppressed_by_vol_count",
                "exec_suppressed_by_warmup_count",
            ]:
                if count_col not in metrics.columns:
                    continue
                bad_count = int(
                    metrics.filter(
                        pl.col("execution_filters_enabled")
                        & (
                            pl.col(count_col).cast(pl.Float64, strict=False).is_null()
                            | (~pl.col(count_col).cast(pl.Float64, strict=False).is_finite())
                            | (pl.col(count_col).cast(pl.Float64, strict=False) < 0)
                            | (
                                (
                                    pl.col(count_col).cast(pl.Float64, strict=False)
                                    - pl.col(count_col).cast(pl.Float64, strict=False).round(0)
                                ).abs()
                                > 1e-9
                            )
                        )
                    ).height
                )
                if bad_count > 0:
                    errors.append(f"{count_col}_invalid")
            if all(
                col in metrics.columns
                for col in [
                    "exec_suppressed_signal_count",
                    "exec_suppressed_by_price_count",
                    "exec_suppressed_by_liquidity_count",
                    "exec_suppressed_by_vol_count",
                    "exec_suppressed_by_warmup_count",
                ]
            ):
                bad_reason_sum = int(
                    metrics.filter(
                        pl.col("execution_filters_enabled")
                        & (
                            (
                                pl.col("exec_suppressed_by_price_count").cast(pl.Float64, strict=False)
                                + pl.col("exec_suppressed_by_liquidity_count").cast(pl.Float64, strict=False)
                                + pl.col("exec_suppressed_by_vol_count").cast(pl.Float64, strict=False)
                                + pl.col("exec_suppressed_by_warmup_count").cast(pl.Float64, strict=False)
                            )
                            > pl.col("exec_suppressed_signal_count").cast(pl.Float64, strict=False)
                        )
                    ).height
                )
                if bad_reason_sum > 0:
                    errors.append("execution_reason_count_sum_above_suppressed")

    if "realism_aware_verdict" in metrics.columns and "trade_count" in metrics.columns:
        bad_candidate = int(
            metrics.filter(
                pl.col("realism_aware_verdict").is_in(["PRIMARY_CANDIDATE", "SECONDARY_CANDIDATE"])
                & (pl.col("trade_count").cast(pl.Int64, strict=False) <= 0)
            ).height
        )
        if bad_candidate > 0:
            errors.append("candidate_verdict_with_zero_trades")

    sources = sorted(set(manifest.get_column("source_type").to_list())) if "source_type" in manifest.columns else []
    if str(summary.get("scope", "")).startswith("multi-"):
        for src in sources:
            succ = int(manifest.filter((pl.col("source_type") == src) & (pl.col("status") == "SUCCESS")).height)
            if succ == 0:
                warnings.append(f"source_no_success:{src}")

    # Cluster ALLOW-only sanity when include_watch is false.
    if "source_type" in manifest.columns:
        cluster_rows = manifest.filter((pl.col("source_type") == "cluster") & (pl.col("status") == "SUCCESS"))
        allow_only_issues = 0
        require_cluster_mix = int(cluster_rows.height) > 0 and int(
            cluster_rows.filter(pl.col("cluster_hardening_dir").is_not_null()).height
        ) > 0
        if require_cluster_mix and not (grid_run_dir / "cluster_policy_trade_mix.csv").exists():
            errors.append("missing_cluster_policy_trade_mix_artifact")
        for row in cluster_rows.select(["include_watch", "backtest_run_dir"]).to_dicts():
            include_watch = bool(row.get("include_watch"))
            run_dir_raw = row.get("backtest_run_dir")
            if include_watch or not run_dir_raw:
                continue
            trades_path = Path(str(run_dir_raw)) / "trades.parquet"
            if not trades_path.exists():
                continue
            trades = pl.read_parquet(trades_path)
            if "entry_state_class" in trades.columns:
                non_allow = int(trades.filter(pl.col("entry_state_class") != "ALLOW").height)
                if non_allow > 0:
                    allow_only_issues += 1
        if allow_only_issues > 0:
            warnings.append(f"cluster_allow_only_violations:{allow_only_issues}")

    top_expectancy = []
    if metrics.height > 0 and "expectancy" in metrics.columns:
        top_expectancy = (
            metrics.sort("expectancy", descending=True)
            .head(10)
            .select(
                [
                    "source_type",
                    "combo_id",
                    "hold_bars",
                    "signal_mode",
                    "exit_mode",
                    "fee_bps_per_side",
                    "slippage_bps_per_side",
                    "trade_count",
                    "expectancy",
                    "profit_factor",
                    "ret_cv",
                    "downside_std",
                    "overlay_mode",
                    "overlay_vetoed_signal_share",
                    "robustness_score_v2",
                ]
            )
            .to_dicts()
        )

    top_pf = []
    if metrics.height > 0 and "profit_factor" in metrics.columns:
        top_pf = (
            metrics.sort("profit_factor", descending=True)
            .head(10)
            .select(
                [
                    "source_type",
                    "combo_id",
                    "hold_bars",
                    "signal_mode",
                    "exit_mode",
                    "fee_bps_per_side",
                    "slippage_bps_per_side",
                    "trade_count",
                    "expectancy",
                    "profit_factor",
                    "ret_cv",
                    "downside_std",
                    "overlay_mode",
                    "overlay_vetoed_signal_share",
                    "robustness_score_v2",
                ]
            )
            .to_dicts()
        )

    zero_trade_combos = int(metrics.filter(pl.col("is_zero_trade_combo") == True).height) if "is_zero_trade_combo" in metrics.columns else 0
    if "zero_trade_combos" in summary and int(summary.get("zero_trade_combos", 0)) != zero_trade_combos:
        warnings.append("zero_trade_combo_count_mismatch")
    if bool(summary.get("realism_profile_broken_for_universe", False)):
        warnings.append("realism_profile_broken_for_universe")

    return {
        "grid_run_dir": str(grid_run_dir),
        "summary": summary,
        "status_counts": status_map,
        "sources": sources,
        "top_expectancy": top_expectancy,
        "top_profit_factor": top_pf,
        "errors": errors,
        "warnings": warnings,
        "non_finite_cells": non_finite_cells,
        "null_metric_cells": null_metric_cells,
        "zero_trade_combos": zero_trade_combos,
    }
