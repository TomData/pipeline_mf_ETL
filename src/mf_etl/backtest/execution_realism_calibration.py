"""Execution realism calibration diagnostics and threshold recommendation pass."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
import itertools
import json
import logging
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from mf_etl.backtest.adapters import normalize_backtest_input
from mf_etl.backtest.execution_realism import (
    apply_execution_realism_filter,
    resolve_execution_realism_params,
)
from mf_etl.backtest.execution_realism_calibration_models import (
    CalibrationSourceType,
    ExecutionRealismCalibrationReportResult,
    ExecutionRealismCalibrationResult,
    ResolvedCalibrationSourceType,
)
from mf_etl.backtest.policy_overlay import apply_policy_overlay
from mf_etl.backtest.state_mapping import (
    apply_cluster_policy_mapping,
    apply_flow_state_mapping,
    apply_hmm_state_mapping,
)
from mf_etl.backtest.writer import (
    write_csv_atomically,
    write_json_atomically,
    write_markdown_atomically,
)
from mf_etl.config import AppSettings

LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_json(payload: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        return value

    return convert(payload)


def _read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    raise ValueError(f"Unsupported source format: {path}")


def _detect_source_type(
    raw: pl.DataFrame,
    *,
    source_type: CalibrationSourceType,
    state_col: str | None,
) -> ResolvedCalibrationSourceType:
    if source_type != "auto":
        return source_type

    cols = set(raw.columns)
    if "hmm_state" in cols:
        return "hmm"
    if "flow_state_code" in cols:
        return "flow"
    if "cluster_id" in cols:
        return "cluster"
    if state_col and state_col in cols:
        # MVP simplification: auto+state-col defaults to HMM semantics.
        return "hmm"
    raise ValueError(
        "Could not auto-detect source type. Provide --source-type explicitly or --state-col for generic state input."
    )


def _maybe_alias_state_column(
    raw: pl.DataFrame,
    *,
    resolved_source_type: ResolvedCalibrationSourceType,
    state_col: str | None,
) -> pl.DataFrame:
    if state_col is None:
        return raw
    if state_col not in raw.columns:
        raise ValueError(f"state_col not found in source file: {state_col}")

    target = {
        "hmm": "hmm_state",
        "flow": "flow_state_code",
        "cluster": "cluster_id",
    }[resolved_source_type]
    if target in raw.columns:
        return raw
    return raw.with_columns(pl.col(state_col).alias(target))


def _prepare_normalized_input(
    *,
    source_file: Path,
    source_type: CalibrationSourceType,
    state_col: str | None,
    logger: logging.Logger,
) -> tuple[pl.DataFrame, ResolvedCalibrationSourceType, dict[str, Any]]:
    raw = _read_table(source_file)
    resolved = _detect_source_type(raw, source_type=source_type, state_col=state_col)
    aliased = _maybe_alias_state_column(raw, resolved_source_type=resolved, state_col=state_col)

    with tempfile.TemporaryDirectory(prefix="mf_etl_exec_calib_") as tmp_dir:
        temp_path = Path(tmp_dir) / "calibration_input.parquet"
        aliased.write_parquet(temp_path)
        normalized = normalize_backtest_input(temp_path, input_type=resolved, logger=logger)
    return normalized.frame, resolved, normalized.summary


def _infer_hmm_direction_map_from_frame(frame: pl.DataFrame) -> dict[int, str]:
    if "state_id" not in frame.columns or "fwd_ret_10" not in frame.columns:
        return {}
    stats = (
        frame.group_by("state_id")
        .agg(
            pl.when(pl.col("fwd_ret_10").cast(pl.Float64, strict=False).is_finite())
            .then(pl.col("fwd_ret_10").cast(pl.Float64, strict=False))
            .otherwise(None)
            .mean()
            .alias("fwd_ret_10_mean")
        )
        .sort("state_id")
    )
    out: dict[int, str] = {}
    for row in stats.to_dicts():
        sid = int(row["state_id"])
        mean_val = _safe_float(row.get("fwd_ret_10_mean"))
        if mean_val is None:
            out[sid] = "UNCONFIRMED"
        elif mean_val > 0:
            out[sid] = "LONG_BIAS"
        elif mean_val < 0:
            out[sid] = "SHORT_BIAS"
        else:
            out[sid] = "UNCONFIRMED"
    return out


def _apply_state_mapping_for_calibration(
    *,
    frame: pl.DataFrame,
    source_type: ResolvedCalibrationSourceType,
    settings: AppSettings,
    validation_run_dir: Path | None,
    state_map_file: Path | None,
    cluster_hardening_dir: Path | None,
    allow_unconfirmed: bool,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if source_type == "flow":
        mapped, summary, _ = apply_flow_state_mapping(
            frame,
            settings=settings,
            include_state_ids=[],
            allow_unconfirmed=allow_unconfirmed,
        )
        return mapped, summary

    if source_type == "cluster":
        if cluster_hardening_dir is None:
            raise ValueError(
                "Cluster calibration requires --cluster-hardening-dir (or --cluster-policy-file parent)."
            )
        mapped, summary, _ = apply_cluster_policy_mapping(
            frame,
            settings=settings,
            cluster_hardening_dir=cluster_hardening_dir,
            include_watch=False,
            policy_filter_mode="allow_only",
            include_state_ids=[],
            allow_unconfirmed=allow_unconfirmed,
        )
        return mapped, summary

    # HMM
    if state_map_file is not None or validation_run_dir is not None:
        mapped, summary, _ = apply_hmm_state_mapping(
            frame,
            settings=settings,
            validation_run_dir=validation_run_dir,
            state_map_file=state_map_file,
            include_state_ids=[],
            allow_unconfirmed=allow_unconfirmed,
        )
        return mapped, summary

    # Fallback for calibration ergonomics when explicit mapping artifacts are absent.
    inferred_map = _infer_hmm_direction_map_from_frame(frame)
    if not inferred_map:
        raise ValueError(
            "HMM calibration could not infer direction mapping. Provide --validation-run-dir or --state-map-file."
        )
    mapping_df = pl.DataFrame(
        [
            {"state_id": sid, "state_direction_hint_map": direction}
            for sid, direction in sorted(inferred_map.items())
        ]
    )
    mapped = frame.join(mapping_df, on="state_id", how="left").with_columns(
        pl.coalesce(
            [pl.col("state_direction_hint_map"), pl.col("state_direction_hint"), pl.lit("UNCONFIRMED")]
        ).alias("state_direction_hint"),
        pl.lit("NA").alias("state_class"),
        pl.lit(None).cast(pl.Float64).alias("state_score"),
    )
    if "state_direction_hint_map" in mapped.columns:
        mapped = mapped.drop("state_direction_hint_map")
    mapped = mapped.with_columns(
        (
            pl.col("state_direction_hint").is_in(["LONG_BIAS", "SHORT_BIAS"])
            | (pl.lit(allow_unconfirmed) & (pl.col("state_direction_hint") == "UNCONFIRMED"))
        ).alias("signal_eligible")
    )
    summary = {
        "mapping_source": "hmm_inferred_from_source_fwd_ret_10",
        "eligible_rows": int(mapped.filter(pl.col("signal_eligible")).height),
        "direction_counts": mapped.group_by("state_direction_hint").len(name="rows").to_dicts(),
    }
    return mapped, summary


def _apply_optional_overlay(
    *,
    frame: pl.DataFrame,
    settings: AppSettings,
    overlay_cluster_file: Path | None,
    overlay_cluster_hardening_dir: Path | None,
    overlay_mode: str,
    overlay_join_keys: list[str] | None,
    logger: logging.Logger,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if overlay_mode == "none" and overlay_cluster_file is None and overlay_cluster_hardening_dir is None:
        return frame, {
            "overlay_enabled": False,
            "overlay_mode": "none",
            "match_rate": None,
            "primary_rows_total": int(frame.height),
        }
    if overlay_cluster_file is None or overlay_cluster_hardening_dir is None:
        raise ValueError(
            "Overlay calibration requires both --overlay-cluster-file and --overlay-cluster-hardening-dir (or --cluster-policy-file)."
        )
    if overlay_mode == "none":
        # Explicit overlay inputs with mode none: still attach diagnostics columns and pass-through gating.
        mode = "none"
    else:
        mode = overlay_mode
    overlay = apply_policy_overlay(
        frame,
        overlay_cluster_file=overlay_cluster_file,
        overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,
        overlay_mode=mode,  # type: ignore[arg-type]
        join_keys=overlay_join_keys or list(settings.backtest_policy_overlay.join_keys),
        unknown_handling=settings.overlay_coverage_policy.unknown_handling,
        min_match_rate_warn=settings.overlay_coverage_policy.min_match_rate_warn,
        min_match_rate_fail=settings.overlay_coverage_policy.min_match_rate_fail,
        min_year_match_rate_warn=settings.overlay_coverage_policy.min_year_match_rate_warn,
        min_year_match_rate_fail=settings.overlay_coverage_policy.min_year_match_rate_fail,
        unknown_rate_warn=settings.overlay_coverage_policy.unknown_rate_warn,
        unknown_rate_fail=settings.overlay_coverage_policy.unknown_rate_fail,
        coverage_mode="warn_only",
        logger=logger,
    )
    return overlay.frame, overlay.join_summary


def _normalize_date_filters(
    frame: pl.DataFrame,
    *,
    start_date: date | None,
    end_date: date | None,
) -> pl.DataFrame:
    out = frame
    if start_date is not None:
        out = out.filter(pl.col("trade_date") >= pl.lit(start_date))
    if end_date is not None:
        out = out.filter(pl.col("trade_date") <= pl.lit(end_date))
    return out


def _apply_sample(frame: pl.DataFrame, *, sample_frac: float | None) -> pl.DataFrame:
    if sample_frac is None:
        return frame
    if sample_frac <= 0.0 or sample_frac > 1.0:
        raise ValueError("sample_frac must be within (0, 1].")
    if frame.height == 0:
        return frame
    return frame.sample(fraction=sample_frac, with_replacement=False, shuffle=True, seed=42)


def _stats_for_series(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    finite_n = int(finite.size)
    out: dict[str, Any] = {
        "finite_count": finite_n,
        "null_count": int(values.size - finite_n),
        "min": None,
        "p01": None,
        "p05": None,
        "p10": None,
        "p25": None,
        "p50": None,
        "p75": None,
        "p90": None,
        "p95": None,
        "p99": None,
        "max": None,
        "mean": None,
        "std": None,
    }
    if finite_n == 0:
        return out
    q = np.quantile(finite, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    out.update(
        {
            "min": _safe_float(np.min(finite)),
            "p01": _safe_float(q[0]),
            "p05": _safe_float(q[1]),
            "p10": _safe_float(q[2]),
            "p25": _safe_float(q[3]),
            "p50": _safe_float(q[4]),
            "p75": _safe_float(q[5]),
            "p90": _safe_float(q[6]),
            "p95": _safe_float(q[7]),
            "p99": _safe_float(q[8]),
            "max": _safe_float(np.max(finite)),
            "mean": _safe_float(np.mean(finite)),
            "std": _safe_float(np.std(finite, ddof=1)) if finite_n > 1 else 0.0,
        }
    )
    return out


def _distribution_rows_for_scope(
    frame: pl.DataFrame,
    *,
    year: int | None,
) -> list[dict[str, Any]]:
    metric_map = {
        "close": "close",
        "dollar_vol_20": "exec_dollar_vol_20",
        "vol_metric": "exec_vol_pct",
        "history_bars": "exec_history_bars",
    }
    subset_map: dict[str, pl.Expr] = {
        "candidate_all": pl.col("exec_candidate_before"),
        "candidate_eligible": pl.col("exec_candidate_after"),
        "candidate_suppressed": pl.col("exec_suppressed_signal"),
        "suppressed_price": pl.col("exec_suppressed_signal") & pl.col("exec_filter_price_fail"),
        "suppressed_liquidity": pl.col("exec_suppressed_signal") & pl.col("exec_filter_liquidity_fail"),
        "suppressed_vol": pl.col("exec_suppressed_signal") & pl.col("exec_filter_vol_fail"),
        "suppressed_warmup": pl.col("exec_suppressed_signal") & pl.col("exec_filter_warmup_fail"),
    }
    rows: list[dict[str, Any]] = []
    for subset_name, expr in subset_map.items():
        subset = frame.filter(expr)
        subset_count = int(subset.height)
        for metric_name, col in metric_map.items():
            if col not in subset.columns:
                rows.append(
                    {
                        "year": year,
                        "subset": subset_name,
                        "metric": metric_name,
                        "count": subset_count,
                        "finite_count": 0,
                        "null_count": subset_count,
                        "min": None,
                        "p01": None,
                        "p05": None,
                        "p10": None,
                        "p25": None,
                        "p50": None,
                        "p75": None,
                        "p90": None,
                        "p95": None,
                        "p99": None,
                        "max": None,
                        "mean": None,
                        "std": None,
                    }
                )
                continue
            arr = subset.get_column(col).cast(pl.Float64, strict=False).to_numpy()
            arr = np.asarray(arr, dtype=np.float64)
            stats = _stats_for_series(arr)
            rows.append(
                {
                    "year": year,
                    "subset": subset_name,
                    "metric": metric_name,
                    "count": subset_count,
                    **stats,
                }
            )
    return rows


def _build_distribution_table(
    frame: pl.DataFrame,
    *,
    by_year: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    overall_rows = _distribution_rows_for_scope(frame, year=None)
    overall = pl.DataFrame(overall_rows) if overall_rows else pl.DataFrame()

    yearly = pl.DataFrame()
    if by_year and frame.height > 0:
        rows: list[dict[str, Any]] = []
        years = (
            frame.with_columns(pl.col("trade_date").dt.year().alias("_year"))
            .select(pl.col("_year").drop_nulls().unique().sort())
            .get_column("_year")
            .to_list()
        )
        for year in years:
            sub = frame.filter(pl.col("trade_date").dt.year() == int(year))
            rows.extend(_distribution_rows_for_scope(sub, year=int(year)))
        yearly = pl.DataFrame(rows) if rows else pl.DataFrame()
    return overall, yearly


def _build_waterfall(frame: pl.DataFrame) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    candidate = frame.filter(pl.col("exec_candidate_before"))
    suppressed = candidate.filter(pl.col("exec_suppressed_signal"))
    candidate_total = int(candidate.height)
    eligible_total = int(candidate.filter(pl.col("exec_candidate_after")).height)
    suppressed_total = int(suppressed.height)

    price_cnt = int(suppressed.filter(pl.col("exec_filter_price_fail")).height)
    liq_cnt = int(suppressed.filter(pl.col("exec_filter_liquidity_fail")).height)
    vol_cnt = int(suppressed.filter(pl.col("exec_filter_vol_fail")).height)
    warm_cnt = int(suppressed.filter(pl.col("exec_filter_warmup_fail")).height)

    overlap = (
        suppressed.select(
            pl.col("exec_filter_price_fail").cast(pl.Boolean).alias("price_fail"),
            pl.col("exec_filter_liquidity_fail").cast(pl.Boolean).alias("liquidity_fail"),
            pl.col("exec_filter_vol_fail").cast(pl.Boolean).alias("vol_fail"),
            pl.col("exec_filter_warmup_fail").cast(pl.Boolean).alias("warmup_fail"),
        )
        .group_by(["price_fail", "liquidity_fail", "vol_fail", "warmup_fail"])
        .agg(pl.len().alias("count"))
        .with_columns(
            pl.when(pl.lit(suppressed_total) > 0)
            .then(pl.col("count").cast(pl.Float64) / pl.lit(float(suppressed_total)))
            .otherwise(0.0)
            .alias("share_suppressed")
        )
        .sort("count", descending=True)
    ) if suppressed_total > 0 else pl.DataFrame(
        schema={
            "price_fail": pl.Boolean,
            "liquidity_fail": pl.Boolean,
            "vol_fail": pl.Boolean,
            "warmup_fail": pl.Boolean,
            "count": pl.Int64,
            "share_suppressed": pl.Float64,
        }
    )

    first_fail = (
        suppressed.with_columns(
            # MVP simplification: first-fail uses deterministic precedence.
            pl.when(pl.col("exec_filter_price_fail"))
            .then(pl.lit("price_floor"))
            .when(pl.col("exec_filter_liquidity_fail"))
            .then(pl.lit("liquidity_floor"))
            .when(pl.col("exec_filter_vol_fail"))
            .then(pl.lit("vol_cap"))
            .when(pl.col("exec_filter_warmup_fail"))
            .then(pl.lit("warmup"))
            .otherwise(pl.lit("none"))
            .alias("first_fail_reason")
        )
        .group_by("first_fail_reason")
        .agg(pl.len().alias("count"))
        .with_columns(
            pl.when(pl.lit(suppressed_total) > 0)
            .then(pl.col("count").cast(pl.Float64) / pl.lit(float(suppressed_total)))
            .otherwise(0.0)
            .alias("share_suppressed")
        )
        .sort("count", descending=True)
    ) if suppressed_total > 0 else pl.DataFrame(
        schema={"first_fail_reason": pl.String, "count": pl.Int64, "share_suppressed": pl.Float64}
    )

    waterfall = {
        "candidate_signals_total": candidate_total,
        "eligible_signals_total": eligible_total,
        "suppressed_signals_total": suppressed_total,
        "eligible_rate": _safe_float((eligible_total / candidate_total) if candidate_total > 0 else None),
        "suppressed_rate": _safe_float((suppressed_total / candidate_total) if candidate_total > 0 else None),
        "suppressed_by_price_count": price_cnt,
        "suppressed_by_liquidity_count": liq_cnt,
        "suppressed_by_vol_count": vol_cnt,
        "suppressed_by_warmup_count": warm_cnt,
        "suppressed_by_price_share": _safe_float((price_cnt / suppressed_total) if suppressed_total > 0 else 0.0),
        "suppressed_by_liquidity_share": _safe_float((liq_cnt / suppressed_total) if suppressed_total > 0 else 0.0),
        "suppressed_by_vol_share": _safe_float((vol_cnt / suppressed_total) if suppressed_total > 0 else 0.0),
        "suppressed_by_warmup_share": _safe_float((warm_cnt / suppressed_total) if suppressed_total > 0 else 0.0),
        "first_fail_precedence": ["price_floor", "liquidity_floor", "vol_cap", "warmup"],
    }
    return waterfall, overlap, first_fail


def _collect_combo_row(result_frame: pl.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    eligible = result_frame.filter(pl.col("exec_candidate_after"))
    by_year = (
        result_frame.with_columns(pl.col("trade_date").dt.year().alias("year"))
        .filter(pl.col("exec_candidate_before"))
        .group_by("year")
        .agg(
            pl.col("exec_candidate_before").cast(pl.Int64).sum().alias("candidate_before"),
            pl.col("exec_candidate_after").cast(pl.Int64).sum().alias("candidate_after"),
        )
        .with_columns(
            pl.when(pl.col("candidate_before") > 0)
            .then(pl.col("candidate_after").cast(pl.Float64) / pl.col("candidate_before").cast(pl.Float64))
            .otherwise(None)
            .alias("eligibility_rate"),
        )
    )
    by_year_min = _safe_float(by_year.select(pl.col("eligibility_rate").min()).item()) if by_year.height > 0 else None
    by_year_max = _safe_float(by_year.select(pl.col("eligibility_rate").max()).item()) if by_year.height > 0 else None

    avg_close = _safe_float(eligible.select(pl.col("close").cast(pl.Float64, strict=False).mean()).item()) if eligible.height > 0 else None
    avg_dv20 = _safe_float(
        eligible.select(pl.col("exec_dollar_vol_20").cast(pl.Float64, strict=False).mean()).item()
    ) if eligible.height > 0 else None
    avg_vol = _safe_float(
        eligible.select(pl.col("exec_vol_pct").cast(pl.Float64, strict=False).mean()).item()
    ) if eligible.height > 0 else None

    return {
        "candidate_signals_before": int(summary.get("candidate_signals_before_filters", 0) or 0),
        "candidate_signals_after": int(summary.get("candidate_signals_after_filters", 0) or 0),
        "suppressed_signal_count": int(summary.get("suppressed_signal_count", 0) or 0),
        "eligibility_rate": _safe_float(summary.get("eligibility_rate")),
        "suppressed_signal_share": _safe_float(summary.get("suppressed_signal_share")),
        "suppressed_by_price_share": _safe_float(summary.get("exec_suppressed_by_price_share")) or 0.0,
        "suppressed_by_liquidity_share": _safe_float(summary.get("exec_suppressed_by_liquidity_share")) or 0.0,
        "suppressed_by_vol_share": _safe_float(summary.get("exec_suppressed_by_vol_share")) or 0.0,
        "suppressed_by_warmup_share": _safe_float(summary.get("exec_suppressed_by_warmup_share")) or 0.0,
        "max_reason_share": max(
            _safe_float(summary.get("exec_suppressed_by_price_share")) or 0.0,
            _safe_float(summary.get("exec_suppressed_by_liquidity_share")) or 0.0,
            _safe_float(summary.get("exec_suppressed_by_vol_share")) or 0.0,
            _safe_float(summary.get("exec_suppressed_by_warmup_share")) or 0.0,
        ),
        "eligible_avg_close": avg_close,
        "eligible_avg_dollar_vol_20": avg_dv20,
        "eligible_avg_vol_pct": avg_vol,
        "year_eligibility_min": by_year_min,
        "year_eligibility_max": by_year_max,
        "vol_metric_source": summary.get("vol_metric_source"),
        "vol_unit_detected": summary.get("vol_unit_detected"),
        "vol_threshold_input": _safe_float(summary.get("vol_threshold_input")),
        "vol_threshold_effective_decimal": _safe_float(summary.get("vol_threshold_effective_decimal")),
        "vol_threshold_effective_pct": _safe_float(summary.get("vol_threshold_effective_pct")),
    }


def _run_threshold_sweep(
    frame: pl.DataFrame,
    *,
    settings: AppSettings,
    base_profile: str,
    report_min_trades: int,
    report_max_zero_trade_share: float,
    report_max_ret_cv: float,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    sweep_logger = logging.getLogger("mf_etl.backtest.execution_realism_calibration.sweep")
    sweep_logger.setLevel(logging.ERROR)
    sweep_cfg = settings.backtest_execution_calibration.sweep
    prices = [float(v) for v in sweep_cfg.min_price]
    dollar_vols = [float(v) for v in sweep_cfg.min_dollar_vol20]
    max_vols = [None if v is None else float(v) for v in sweep_cfg.max_vol_pct]
    history_bars = [int(v) for v in sweep_cfg.min_history_bars]

    base_params = resolve_execution_realism_params(
        settings,
        exec_profile=base_profile,  # type: ignore[arg-type]
        exec_min_price=None,
        exec_min_dollar_vol20=None,
        exec_max_vol_pct=None,
        exec_min_history_bars=None,
        report_min_trades=report_min_trades,
        report_max_zero_trade_share=report_max_zero_trade_share,
        report_max_ret_cv=report_max_ret_cv,
    )

    rows: list[dict[str, Any]] = []
    for idx, (min_price, min_dv20, max_vol, min_hist) in enumerate(
        itertools.product(prices, dollar_vols, max_vols, history_bars),
        start=1,
    ):
        params = replace(
            base_params,
            min_price=min_price,
            min_dollar_vol_20=min_dv20,
            max_vol_pct=max_vol,
            min_history_bars_for_execution=min_hist,
        )
        result = apply_execution_realism_filter(frame, params=params, logger=sweep_logger)
        row = {
            "combo_id": f"c{idx:04d}",
            "min_price": min_price,
            "min_dollar_vol20": min_dv20,
            "max_vol_pct_input": max_vol,
            "min_history_bars": min_hist,
        }
        row.update(_collect_combo_row(result.frame, result.summary))
        rows.append(row)

    grid = pl.DataFrame(rows).sort(["min_price", "min_dollar_vol20", "max_vol_pct_input", "min_history_bars"])
    summary = {
        "combos_total": int(grid.height),
        "eligibility_rate_min": _safe_float(grid.select(pl.col("eligibility_rate").min()).item()) if grid.height > 0 else None,
        "eligibility_rate_p50": _safe_float(grid.select(pl.col("eligibility_rate").median()).item()) if grid.height > 0 else None,
        "eligibility_rate_max": _safe_float(grid.select(pl.col("eligibility_rate").max()).item()) if grid.height > 0 else None,
        "candidate_after_min": _safe_int(grid.select(pl.col("candidate_signals_after").min()).item()) if grid.height > 0 else None,
        "candidate_after_p50": _safe_int(grid.select(pl.col("candidate_signals_after").median()).item()) if grid.height > 0 else None,
        "candidate_after_max": _safe_int(grid.select(pl.col("candidate_signals_after").max()).item()) if grid.height > 0 else None,
    }
    return grid, summary


def _recommend_thresholds(settings: AppSettings, sweep: pl.DataFrame) -> dict[str, Any]:
    cfg = settings.backtest_execution_calibration
    if sweep.height == 0:
        return {
            "lite": {"top_candidates": [], "recommended": None},
            "strict": {"top_candidates": [], "recommended": None},
        }

    def pick(label: str, lo: float, hi: float) -> dict[str, Any]:
        mid = (lo + hi) / 2.0
        base = sweep.with_columns(
            (
                (pl.col("eligibility_rate").cast(pl.Float64, strict=False) - pl.lit(mid))
                .abs()
            ).alias("_distance_to_mid"),
            (
                pl.when(pl.col("candidate_signals_after").cast(pl.Float64, strict=False) > 0)
                .then(
                    (pl.col("candidate_signals_after").cast(pl.Float64, strict=False) / pl.lit(float(max(1, cfg.min_eligible_signals * 5))))
                    .clip(0.0, 1.0)
                )
                .otherwise(0.0)
            ).alias("_coverage_score"),
            (
                (
                    pl.col("max_reason_share").cast(pl.Float64, strict=False)
                    - pl.lit(cfg.max_single_reason_share)
                ).clip(lower_bound=0.0)
            ).alias("_dominance_penalty"),
        ).with_columns(
            (
                pl.lit(1.0) - pl.col("_distance_to_mid")
                + (pl.col("_coverage_score") * pl.lit(0.25))
                - (pl.col("_dominance_penalty") * pl.lit(0.5))
            ).alias("_score")
        )

        filtered = base.filter(
            (pl.col("eligibility_rate").cast(pl.Float64, strict=False) >= pl.lit(lo))
            & (pl.col("eligibility_rate").cast(pl.Float64, strict=False) <= pl.lit(hi))
            & (pl.col("candidate_signals_after").cast(pl.Int64, strict=False) >= pl.lit(cfg.min_eligible_signals))
            & (pl.col("max_reason_share").cast(pl.Float64, strict=False) <= pl.lit(cfg.max_single_reason_share))
        )
        if filtered.height == 0:
            filtered = base.filter(
                pl.col("candidate_signals_after").cast(pl.Int64, strict=False) >= pl.lit(cfg.min_eligible_signals)
            )
        if filtered.height == 0:
            filtered = base

        ranked = filtered.sort(
            [
                pl.col("_score").cast(pl.Float64, strict=False).fill_null(-1e9),
                pl.col("candidate_signals_after").cast(pl.Int64, strict=False).fill_null(0),
                pl.col("_distance_to_mid").cast(pl.Float64, strict=False).fill_null(1e9),
            ],
            descending=[True, True, False],
        ).head(int(cfg.top_k_recommendations))

        top = []
        for row in ranked.to_dicts():
            top.append(
                {
                    "combo_id": row.get("combo_id"),
                    "min_price": row.get("min_price"),
                    "min_dollar_vol20": row.get("min_dollar_vol20"),
                    "max_vol_pct_input": row.get("max_vol_pct_input"),
                    "min_history_bars": row.get("min_history_bars"),
                    "eligibility_rate": row.get("eligibility_rate"),
                    "candidate_signals_after": row.get("candidate_signals_after"),
                    "max_reason_share": row.get("max_reason_share"),
                    "suppressed_by_vol_share": row.get("suppressed_by_vol_share"),
                    "score": row.get("_score"),
                    "rationale": (
                        f"elig={row.get('eligibility_rate')} in/near target [{lo}, {hi}], "
                        f"eligible_signals={row.get('candidate_signals_after')}, "
                        f"max_reason_share={row.get('max_reason_share')}"
                    ),
                }
            )
        recommended = top[0] if top else None
        return {
            "target_eligibility_band": [lo, hi],
            "top_candidates": top,
            "recommended": recommended,
            "status": "ok" if recommended is not None else "no_candidate",
            "profile_label": label,
        }

    return {
        "lite": pick(
            "lite",
            float(cfg.target_lite_eligibility_min),
            float(cfg.target_lite_eligibility_max),
        ),
        "strict": pick(
            "strict",
            float(cfg.target_strict_eligibility_min),
            float(cfg.target_strict_eligibility_max),
        ),
    }


def _render_calibration_report(
    *,
    summary: dict[str, Any],
    recommendations: dict[str, Any],
    grid: pl.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Execution Realism Calibration Report v1")
    lines.append("")
    lines.append("## Run")
    lines.append(f"- run_id: {summary.get('run_id')}")
    lines.append(f"- source_type: {summary.get('source_type')}")
    lines.append(f"- source_file: {summary.get('source_file')}")
    lines.append(f"- rows_post_prep: {summary.get('rows_post_preparation')}")
    lines.append(f"- overlay_mode: {summary.get('overlay_mode')}")
    lines.append(f"- signal_mode: {summary.get('signal_mode')}")
    lines.append(f"- execution_profile: {summary.get('execution_profile')}")
    units = summary.get("units", {})
    if isinstance(units, dict):
        lines.append(f"- vol_metric_source: {units.get('vol_metric_source')}")
        lines.append(f"- vol_unit_detected: {units.get('vol_unit_detected')}")
        lines.append(
            f"- vol_threshold_input/effective: {units.get('vol_threshold_input')} / "
            f"{units.get('vol_threshold_effective_decimal')} (decimal)"
        )
        warnings = units.get("qa_warnings")
        lines.append(f"- unit_warnings: {warnings if warnings else 'none'}")
    lines.append("")
    lines.append("## Current Profile Waterfall")
    wf = summary.get("waterfall", {})
    lines.append(f"- candidate_signals_total: {wf.get('candidate_signals_total')}")
    lines.append(f"- eligible_signals_total: {wf.get('eligible_signals_total')}")
    lines.append(f"- suppressed_signals_total: {wf.get('suppressed_signals_total')}")
    lines.append(f"- eligible_rate: {wf.get('eligible_rate')}")
    lines.append(f"- suppressed_by_vol_share: {wf.get('suppressed_by_vol_share')}")
    lines.append(f"- suppressed_by_liquidity_share: {wf.get('suppressed_by_liquidity_share')}")
    lines.append("")
    lines.append("## Sweep Summary")
    sweep = summary.get("sweep_summary", {})
    lines.append(f"- combos_total: {sweep.get('combos_total')}")
    lines.append(
        f"- eligibility_rate[min/p50/max]: {sweep.get('eligibility_rate_min')} / "
        f"{sweep.get('eligibility_rate_p50')} / {sweep.get('eligibility_rate_max')}"
    )
    lines.append("")
    lines.append("## Recommendations")
    for label in ["lite", "strict"]:
        rec = recommendations.get(label, {})
        lines.append(f"### {label}")
        lines.append(f"- target_band: {rec.get('target_eligibility_band')}")
        best = rec.get("recommended")
        if isinstance(best, dict):
            lines.append(
                "- recommended: "
                f"min_price={best.get('min_price')}, min_dollar_vol20={best.get('min_dollar_vol20')}, "
                f"max_vol_pct_input={best.get('max_vol_pct_input')}, min_history_bars={best.get('min_history_bars')}, "
                f"eligibility_rate={best.get('eligibility_rate')}, eligible_signals={best.get('candidate_signals_after')}"
            )
        else:
            lines.append("- recommended: none")
    lines.append("")
    lines.append("## Top Grid Rows (by eligibility)")
    lines.append("| combo_id | min_price | min_dollar_vol20 | max_vol_pct_input | min_history_bars | eligibility_rate | candidate_after | max_reason_share |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    top = grid.sort("eligibility_rate", descending=True).head(10) if grid.height > 0 else pl.DataFrame()
    for row in top.to_dicts():
        lines.append(
            f"| {row.get('combo_id')} | {row.get('min_price')} | {row.get('min_dollar_vol20')} | "
            f"{row.get('max_vol_pct_input')} | {row.get('min_history_bars')} | {row.get('eligibility_rate')} | "
            f"{row.get('candidate_signals_after')} | {row.get('max_reason_share')} |"
        )
    lines.append("")
    lines.append("MVP simplification: first-fail decomposition uses deterministic precedence: price -> liquidity -> vol -> warmup.")
    return "\n".join(lines) + "\n"


def run_execution_realism_calibration(
    settings: AppSettings,
    *,
    source_file: Path,
    source_type: CalibrationSourceType,
    state_col: str | None,
    validation_run_dir: Path | None,
    cluster_hardening_dir: Path | None,
    state_map_file: Path | None,
    overlay_cluster_file: Path | None,
    overlay_cluster_hardening_dir: Path | None,
    cluster_policy_file: Path | None,
    overlay_mode: str,
    overlay_join_keys: list[str] | None,
    signal_mode: str,
    execution_profile: str,
    exec_min_price: float | None,
    exec_min_dollar_vol20: float | None,
    exec_max_vol_pct: float | None,
    exec_min_history_bars: int | None,
    report_min_trades: int | None,
    report_max_zero_trade_share: float | None,
    report_max_ret_cv: float | None,
    sample_frac: float | None,
    start_date: date | None,
    end_date: date | None,
    by_year: bool,
    out_dir: Path | None,
    logger: logging.Logger | None = None,
) -> ExecutionRealismCalibrationResult:
    """Run execution realism calibration diagnostics and sweep on one source file."""

    effective_logger = logger or LOGGER
    started = datetime.now(timezone.utc)

    overlay_hardening_dir = overlay_cluster_hardening_dir
    if overlay_hardening_dir is None and cluster_policy_file is not None:
        if cluster_policy_file.name != "cluster_hardening_policy.json":
            raise ValueError(
                "--cluster-policy-file must point to cluster_hardening_policy.json so parent dir can be used."
            )
        overlay_hardening_dir = cluster_policy_file.parent

    normalized, resolved_source_type, adapter_summary = _prepare_normalized_input(
        source_file=source_file,
        source_type=source_type,
        state_col=state_col,
        logger=effective_logger,
    )
    filtered = _normalize_date_filters(normalized, start_date=start_date, end_date=end_date)
    filtered = _apply_sample(filtered, sample_frac=sample_frac)

    mapped, mapping_summary = _apply_state_mapping_for_calibration(
        frame=filtered,
        source_type=resolved_source_type,
        settings=settings,
        validation_run_dir=validation_run_dir,
        state_map_file=state_map_file,
        cluster_hardening_dir=cluster_hardening_dir,
        allow_unconfirmed=settings.backtest.allow_unconfirmed,
    )
    mapped, overlay_summary = _apply_optional_overlay(
        frame=mapped,
        settings=settings,
        overlay_cluster_file=overlay_cluster_file,
        overlay_cluster_hardening_dir=overlay_hardening_dir,
        overlay_mode=overlay_mode,
        overlay_join_keys=overlay_join_keys,
        logger=effective_logger,
    )

    params = resolve_execution_realism_params(
        settings,
        exec_profile=execution_profile,  # type: ignore[arg-type]
        exec_min_price=exec_min_price,
        exec_min_dollar_vol20=exec_min_dollar_vol20,
        exec_max_vol_pct=exec_max_vol_pct,
        exec_min_history_bars=exec_min_history_bars,
        report_min_trades=report_min_trades,
        report_max_zero_trade_share=report_max_zero_trade_share,
        report_max_ret_cv=report_max_ret_cv,
    )
    execution = apply_execution_realism_filter(mapped, params=params, logger=effective_logger)

    distribution, distribution_by_year = _build_distribution_table(execution.frame, by_year=by_year)
    waterfall, overlap, first_fail = _build_waterfall(execution.frame)
    sweep_grid, sweep_summary = _run_threshold_sweep(
        mapped,
        settings=settings,
        base_profile=execution_profile,
        report_min_trades=params.report_min_trades,
        report_max_zero_trade_share=params.report_max_zero_trade_share,
        report_max_ret_cv=params.report_max_ret_cv,
    )
    recommendations = _recommend_thresholds(settings, sweep_grid)

    run_id = f"exec-calib-{uuid4().hex[:12]}"
    if out_dir is not None:
        output_dir = out_dir
    else:
        output_dir = settings.paths.artifacts_root / "execution_realism_calibration" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "execution_calibration_summary.json"
    distribution_path = output_dir / "execution_calibration_distribution.csv"
    distribution_by_year_path = output_dir / "execution_calibration_distribution_by_year.csv"
    waterfall_path = output_dir / "execution_calibration_waterfall.json"
    overlap_path = output_dir / "execution_calibration_reason_overlap.csv"
    first_fail_path = output_dir / "execution_calibration_first_fail.csv"
    units_path = output_dir / "execution_calibration_units.json"
    grid_path = output_dir / "execution_calibration_grid.csv"
    grid_summary_path = output_dir / "execution_calibration_grid_summary.json"
    recommendations_path = output_dir / "execution_calibration_recommendations.json"
    report_path = output_dir / "execution_calibration_report.md"

    units_payload = {
        "vol_metric_source": execution.summary.get("vol_metric_source"),
        "vol_unit_detected": execution.summary.get("vol_unit_detected"),
        "vol_input_unit_mode": execution.summary.get("params_used", {}).get("vol_input_unit_mode"),
        "vol_threshold_input": execution.summary.get("vol_threshold_input"),
        "vol_threshold_effective_decimal": execution.summary.get("vol_threshold_effective_decimal"),
        "vol_threshold_effective_pct": execution.summary.get("vol_threshold_effective_pct"),
        "qa_warnings": execution.summary.get("vol_unit_warnings", []),
    }

    finished = datetime.now(timezone.utc)
    summary_payload = {
        "run_id": run_id,
        "source_file": str(source_file),
        "source_type": resolved_source_type,
        "state_col": state_col,
        "overlay_mode": overlay_mode,
        "signal_mode": signal_mode,
        "execution_profile": execution_profile,
        "sample_frac": sample_frac,
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "by_year": bool(by_year),
        "started_ts": started.isoformat(),
        "finished_ts": finished.isoformat(),
        "duration_sec": round((finished - started).total_seconds(), 3),
        "rows_post_preparation": int(mapped.height),
        "adapter_summary": adapter_summary,
        "mapping_summary": mapping_summary,
        "overlay_summary": overlay_summary,
        "execution_summary": execution.summary,
        "waterfall": waterfall,
        "sweep_summary": sweep_summary,
        "units": units_payload,
        "outputs": {
            "distribution": str(distribution_path),
            "distribution_by_year": str(distribution_by_year_path) if by_year else None,
            "waterfall": str(waterfall_path),
            "reason_overlap": str(overlap_path),
            "first_fail": str(first_fail_path),
            "units": str(units_path),
            "grid": str(grid_path),
            "grid_summary": str(grid_summary_path),
            "recommendations": str(recommendations_path),
            "report": str(report_path),
        },
    }

    write_csv_atomically(distribution, distribution_path)
    if by_year:
        write_csv_atomically(distribution_by_year, distribution_by_year_path)
    write_json_atomically(_finite_json(waterfall), waterfall_path)
    write_csv_atomically(overlap, overlap_path)
    write_csv_atomically(first_fail, first_fail_path)
    write_json_atomically(_finite_json(units_payload), units_path)
    write_csv_atomically(sweep_grid, grid_path)
    write_json_atomically(_finite_json(sweep_summary), grid_summary_path)
    write_json_atomically(_finite_json(recommendations), recommendations_path)
    write_json_atomically(_finite_json(summary_payload), summary_path)
    write_markdown_atomically(
        _render_calibration_report(summary=summary_payload, recommendations=recommendations, grid=sweep_grid),
        report_path,
    )

    effective_logger.info(
        "backtest.execution_calibration.complete run_id=%s source_type=%s output=%s",
        run_id,
        resolved_source_type,
        output_dir,
    )
    return ExecutionRealismCalibrationResult(
        run_id=run_id,
        output_dir=output_dir,
        summary_path=summary_path,
        distribution_path=distribution_path,
        waterfall_path=waterfall_path,
        grid_path=grid_path,
        recommendations_path=recommendations_path,
        report_path=report_path,
    )


def run_execution_realism_calibration_report(
    settings: AppSettings,
    *,
    calibration_dir: Path,
    logger: logging.Logger | None = None,
) -> ExecutionRealismCalibrationReportResult:
    """Re-render calibration markdown report from existing calibration artifacts."""

    effective_logger = logger or LOGGER
    summary_path = calibration_dir / "execution_calibration_summary.json"
    recommendations_path = calibration_dir / "execution_calibration_recommendations.json"
    grid_path = calibration_dir / "execution_calibration_grid.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing calibration summary: {summary_path}")
    if not recommendations_path.exists():
        raise FileNotFoundError(f"Missing calibration recommendations: {recommendations_path}")
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing calibration grid CSV: {grid_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recommendations = json.loads(recommendations_path.read_text(encoding="utf-8"))
    grid = pl.read_csv(grid_path)

    report_path = calibration_dir / "execution_calibration_report.md"
    write_markdown_atomically(
        _render_calibration_report(summary=summary, recommendations=recommendations, grid=grid),
        report_path,
    )
    effective_logger.info("backtest.execution_calibration.report_written dir=%s", calibration_dir)
    return ExecutionRealismCalibrationReportResult(
        calibration_dir=calibration_dir,
        report_path=report_path,
        summary_path=summary_path,
    )
    sweep_logger = logging.getLogger("mf_etl.backtest.execution_realism_calibration.sweep")
    sweep_logger.setLevel(logging.ERROR)
