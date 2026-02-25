"""Execution realism filter layer for backtest and sensitivity runs."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl

from mf_etl.backtest.execution_realism_models import (
    DollarVolRollingMethod,
    ExecutionProfileName,
    ExecutionRealismParams,
    ExecutionRealismResult,
    VolInputUnitMode,
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


def _profile_payload(settings: AppSettings, profile: ExecutionProfileName) -> dict[str, Any]:
    cfg = settings.backtest_execution_realism
    if profile == "lite":
        p = cfg.profiles.lite
    elif profile == "strict":
        p = cfg.profiles.strict
    else:
        p = cfg.profiles.none
    return {
        "min_price": p.min_price,
        "min_dollar_vol_20": p.min_dollar_vol_20,
        "max_vol_pct": p.max_vol_pct,
        "min_history_bars_for_execution": p.min_history_bars_for_execution,
    }


def resolve_execution_realism_params(
    settings: AppSettings,
    *,
    exec_profile: ExecutionProfileName,
    exec_min_price: float | None,
    exec_min_dollar_vol20: float | None,
    exec_max_vol_pct: float | None,
    exec_min_history_bars: int | None,
    report_min_trades: int | None,
    report_max_zero_trade_share: float | None,
    report_max_ret_cv: float | None,
) -> ExecutionRealismParams:
    """Resolve profile + CLI overrides into a single execution realism params object."""

    base = _profile_payload(settings, exec_profile)
    min_price = exec_min_price if exec_min_price is not None else _safe_float(base.get("min_price"))
    min_dollar_vol_20 = (
        exec_min_dollar_vol20
        if exec_min_dollar_vol20 is not None
        else _safe_float(base.get("min_dollar_vol_20"))
    )
    max_vol_pct = exec_max_vol_pct if exec_max_vol_pct is not None else _safe_float(base.get("max_vol_pct"))
    min_history = (
        exec_min_history_bars
        if exec_min_history_bars is not None
        else _safe_int(base.get("min_history_bars_for_execution"))
    )
    return ExecutionRealismParams(
        profile_name=exec_profile,
        min_price=min_price,
        min_dollar_vol_20=min_dollar_vol_20,
        max_vol_pct=max_vol_pct,
        min_history_bars_for_execution=min_history,
        dollar_vol_window=int(settings.backtest_execution_realism.dollar_vol_window),
        dollar_vol_rolling_method=settings.backtest_execution_realism.dollar_vol_rolling_method,
        vol_input_unit_mode=settings.backtest_execution_realism.vol_input_unit_mode,
        report_min_trades=int(
            report_min_trades
            if report_min_trades is not None
            else settings.backtest_execution_realism.report_min_trades_default
        ),
        report_max_zero_trade_share=float(
            report_max_zero_trade_share
            if report_max_zero_trade_share is not None
            else settings.backtest_execution_realism.report_max_zero_trade_share_default
        ),
        report_max_ret_cv=float(
            report_max_ret_cv
            if report_max_ret_cv is not None
            else settings.backtest_execution_realism.report_max_ret_cv_default
        ),
    )


def _vol_raw_expr(frame: pl.DataFrame) -> tuple[pl.Expr, str]:
    if "atr_pct_14" in frame.columns:
        finite_n = int(
            frame.select(pl.col("atr_pct_14").cast(pl.Float64, strict=False).is_finite().sum()).item() or 0
        )
        if finite_n > 0:
            return pl.col("atr_pct_14").cast(pl.Float64, strict=False).alias("exec_vol_raw_pct"), "atr_pct_14"
    return (
        (pl.col("high").cast(pl.Float64, strict=False) - pl.col("low").cast(pl.Float64, strict=False))
        / pl.when(pl.col("close").cast(pl.Float64, strict=False) > 0)
        .then(pl.col("close").cast(pl.Float64, strict=False))
        .otherwise(None)
    ).alias("exec_vol_raw_pct"), "range_pct_fallback"


def _detect_vol_unit(
    values: np.ndarray,
    *,
    mode: VolInputUnitMode,
) -> tuple[VolInputUnitMode, float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ("decimal" if mode == "auto" else mode), None, None
    median = float(np.median(finite))
    p90 = float(np.quantile(finite, 0.90))
    if mode == "auto":
        detected = "percent_points" if (median > 1.0 or p90 > 1.0) else "decimal"
    else:
        detected = mode
    return detected, median, p90


def _normalize_vol_threshold(
    threshold_input: float | None,
    *,
    mode: VolInputUnitMode,
) -> tuple[float | None, float | None, str | None]:
    """Normalize user-input volatility threshold to decimal units used by exec_vol_pct."""

    if threshold_input is None:
        return None, None, None
    value = float(threshold_input)
    if mode == "percent_points":
        dec = value / 100.0
        return dec, dec * 100.0, "percent_points"
    if mode == "decimal":
        return value, value * 100.0, "decimal"
    # auto
    if value > 1.0:
        dec = value / 100.0
        return dec, dec * 100.0, "percent_points_auto"
    return value, value * 100.0, "decimal_auto"


def _dollar_vol_rolling_expr(
    *,
    method: DollarVolRollingMethod,
    window: int,
) -> pl.Expr:
    if method == "mean":
        return (
            pl.col("exec_dollar_vol")
            .rolling_mean(window_size=window, min_samples=window)
            .over("ticker")
            .alias("exec_dollar_vol_20")
        )
    return (
        pl.col("exec_dollar_vol")
        .rolling_median(window_size=window, min_samples=window)
        .over("ticker")
        .alias("exec_dollar_vol_20")
    )


def _reason_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.height == 0:
        return pl.DataFrame(schema={"reason": pl.String})
    suppressed_total = int(frame.select(pl.col("exec_suppressed_signal").cast(pl.Int64).sum()).item() or 0)
    rows = []
    for reason in ["price_floor", "liquidity_floor", "vol_cap", "warmup", "multiple_reasons"]:
        count = int(
            frame.filter(
                (pl.col("exec_suppressed_signal") == True) & (pl.col("execution_filter_reason") == reason)
            ).height
        )
        rows.append(
            {
                "reason": reason,
                "suppressed_signal_count": count,
                "suppressed_signal_share": float(count / suppressed_total) if suppressed_total > 0 else 0.0,
                "row_share_total": float(count / frame.height) if frame.height > 0 else 0.0,
            }
        )
    return pl.DataFrame(rows).sort("reason")


def apply_execution_realism_filter(
    frame: pl.DataFrame,
    *,
    params: ExecutionRealismParams,
    logger: logging.Logger | None = None,
) -> ExecutionRealismResult:
    """Attach execution realism eligibility columns and diagnostics."""

    effective_logger = logger or LOGGER
    if frame.height == 0:
        out = frame.with_columns(
            pl.lit(True).alias("execution_eligible"),
            pl.lit("none").alias("execution_profile"),
            pl.lit(False).alias("execution_filters_enabled"),
            pl.lit("none").alias("execution_filter_reason"),
            pl.lit(False).alias("exec_suppressed_signal"),
            pl.lit(False).alias("exec_candidate_before"),
            pl.lit(False).alias("exec_candidate_after"),
        )
        summary = {
            "profile_name": params.profile_name,
            "params_used": {
                "min_price": params.min_price,
                "min_dollar_vol_20": params.min_dollar_vol_20,
                "max_vol_pct": params.max_vol_pct,
                "min_history_bars_for_execution": params.min_history_bars_for_execution,
                "dollar_vol_window": params.dollar_vol_window,
                "dollar_vol_rolling_method": params.dollar_vol_rolling_method,
                "vol_input_unit_mode": params.vol_input_unit_mode,
            },
            "vol_metric_source": None,
            "vol_unit_detected": None,
            "vol_threshold_input": params.max_vol_pct,
            "vol_threshold_effective_decimal": None,
            "vol_threshold_effective_pct": None,
            "vol_threshold_input_interpretation": None,
            "vol_unit_warnings": [],
            "rows_total": 0,
            "rows_eligible": 0,
            "rows_ineligible": 0,
            "eligibility_rate": None,
            "candidate_signals_before_filters": 0,
            "candidate_signals_after_filters": 0,
            "suppressed_signal_count": 0,
            "suppressed_signal_share": 0.0,
            "exec_suppressed_by_price_count": 0,
            "exec_suppressed_by_liquidity_count": 0,
            "exec_suppressed_by_vol_count": 0,
            "exec_suppressed_by_warmup_count": 0,
            "exec_suppressed_by_price_share": 0.0,
            "exec_suppressed_by_liquidity_share": 0.0,
            "exec_suppressed_by_vol_share": 0.0,
            "exec_suppressed_by_warmup_share": 0.0,
        }
        return ExecutionRealismResult(
            frame=out,
            summary=summary,
            by_reason=pl.DataFrame(schema={"reason": pl.String}),
            by_year=pl.DataFrame(schema={"year": pl.Int32}),
            filters_enabled=False,
        )

    filters_enabled = any(
        value is not None
        for value in [
            params.min_price,
            params.min_dollar_vol_20,
            params.max_vol_pct,
            params.min_history_bars_for_execution,
        ]
    )

    base_exprs: list[pl.Expr] = [
        pl.col("close").cast(pl.Float64, strict=False).alias("close"),
        pl.col("signal_eligible").cast(pl.Boolean).fill_null(False).alias("signal_eligible"),
        (
            pl.col("overlay_allow_signal").cast(pl.Boolean).fill_null(True)
            if "overlay_allow_signal" in frame.columns
            else pl.lit(True)
        ).alias("overlay_allow_signal"),
        pl.col("state_direction_hint")
        .cast(pl.String)
        .fill_null("UNCONFIRMED")
        .alias("state_direction_hint"),
    ]
    if "volume" in frame.columns:
        base_exprs.append(pl.col("volume").cast(pl.Float64, strict=False).alias("volume"))
    else:
        base_exprs.append(pl.lit(None).cast(pl.Float64).alias("volume"))
    work = frame.sort(["ticker", "trade_date"]).with_columns(base_exprs)

    needs_liquidity = params.min_dollar_vol_20 is not None
    if needs_liquidity and ("volume" not in frame.columns or "close" not in frame.columns):
        raise ValueError(
            "Execution realism liquidity filter enabled, but input rows are missing close and/or volume. "
            "Provide volume column or disable min_dollar_vol_20."
        )

    vol_expr, vol_metric_source = _vol_raw_expr(work)
    work = work.with_columns(
        pl.when(pl.col("volume").is_not_null())
        .then(pl.col("close") * pl.col("volume"))
        .otherwise(None)
        .alias("exec_dollar_vol"),
        vol_expr,
        pl.cum_count("trade_date").over("ticker").cast(pl.Int64).alias("exec_history_bars"),
    ).with_columns(
        _dollar_vol_rolling_expr(
            method=params.dollar_vol_rolling_method,
            window=max(1, int(params.dollar_vol_window)),
        )
    )

    vol_raw = work.get_column("exec_vol_raw_pct").cast(pl.Float64, strict=False).to_numpy()
    vol_unit_detected, vol_raw_median, vol_raw_p90 = _detect_vol_unit(
        vol_raw,
        mode=params.vol_input_unit_mode,
    )
    vol_threshold_effective_decimal, vol_threshold_effective_pct, vol_threshold_input_interpretation = (
        _normalize_vol_threshold(
            _safe_float(params.max_vol_pct),
            mode=params.vol_input_unit_mode,
        )
    )
    vol_unit_warnings: list[str] = []
    if vol_threshold_effective_decimal is not None:
        if vol_threshold_effective_decimal <= 0:
            vol_unit_warnings.append("vol_threshold_non_positive")
        if vol_threshold_effective_decimal > 1.0:
            vol_unit_warnings.append("vol_threshold_above_100pct_decimal")
        if vol_unit_detected == "decimal" and vol_threshold_effective_decimal < 0.001:
            vol_unit_warnings.append("vol_threshold_too_low_for_decimal_series")
        if vol_unit_detected == "percent_points" and (vol_threshold_effective_pct or 0.0) < 0.1:
            vol_unit_warnings.append("vol_threshold_too_low_for_percent_points_series")
    vol_divisor = 100.0 if vol_unit_detected == "percent_points" else 1.0
    work = work.with_columns(
        pl.when(pl.col("exec_vol_raw_pct").cast(pl.Float64, strict=False).is_finite())
        .then(pl.col("exec_vol_raw_pct").cast(pl.Float64, strict=False) / pl.lit(vol_divisor))
        .otherwise(None)
        .alias("exec_vol_pct")
    )
    vol_norm = work.get_column("exec_vol_pct").cast(pl.Float64, strict=False).to_numpy()
    vol_norm_finite = vol_norm[np.isfinite(vol_norm)]
    if vol_norm_finite.size > 0:
        if float(np.max(vol_norm_finite)) == float(np.min(vol_norm_finite)):
            vol_unit_warnings.append("vol_series_constant")
        if bool(np.all(np.isclose(vol_norm_finite, 0.0))):
            vol_unit_warnings.append("vol_series_all_zero")

    price_fail = (
        pl.col("close") < float(params.min_price)
        if params.min_price is not None
        else pl.lit(False)
    )
    liq_fail = (
        (pl.col("exec_dollar_vol_20").is_null())
        | (pl.col("exec_dollar_vol_20") < float(params.min_dollar_vol_20))
        if params.min_dollar_vol_20 is not None
        else pl.lit(False)
    )
    vol_fail = (
        (pl.col("exec_vol_pct").is_null()) | (pl.col("exec_vol_pct") > float(vol_threshold_effective_decimal))
        if vol_threshold_effective_decimal is not None
        else pl.lit(False)
    )
    warmup_fail = (
        pl.col("exec_history_bars") < int(params.min_history_bars_for_execution)
        if params.min_history_bars_for_execution is not None
        else pl.lit(False)
    )

    work = work.with_columns(
        price_fail.alias("exec_filter_price_fail"),
        liq_fail.alias("exec_filter_liquidity_fail"),
        vol_fail.alias("exec_filter_vol_fail"),
        warmup_fail.alias("exec_filter_warmup_fail"),
    ).with_columns(
        pl.any_horizontal(
            [
                pl.col("exec_filter_price_fail"),
                pl.col("exec_filter_liquidity_fail"),
                pl.col("exec_filter_vol_fail"),
                pl.col("exec_filter_warmup_fail"),
            ]
        ).alias("exec_filter_any_fail")
    ).with_columns(
        (~pl.col("exec_filter_any_fail")).alias("execution_eligible"),
        pl.lit(params.profile_name).alias("execution_profile"),
        pl.lit(filters_enabled).alias("execution_filters_enabled"),
        # MVP simplification: one primary suppression reason chosen by fixed precedence.
        pl.when(
            (
                pl.col("exec_filter_price_fail").cast(pl.Int8)
                + pl.col("exec_filter_liquidity_fail").cast(pl.Int8)
                + pl.col("exec_filter_vol_fail").cast(pl.Int8)
                + pl.col("exec_filter_warmup_fail").cast(pl.Int8)
            )
            > 1
        )
        .then(pl.lit("multiple_reasons"))
        .when(pl.col("exec_filter_price_fail"))
        .then(pl.lit("price_floor"))
        .when(pl.col("exec_filter_liquidity_fail"))
        .then(pl.lit("liquidity_floor"))
        .when(pl.col("exec_filter_vol_fail"))
        .then(pl.lit("vol_cap"))
        .when(pl.col("exec_filter_warmup_fail"))
        .then(pl.lit("warmup"))
        .otherwise(pl.lit("none"))
        .alias("execution_filter_reason"),
    ).with_columns(
        (
            pl.col("signal_eligible")
            & pl.col("overlay_allow_signal")
            & pl.col("state_direction_hint").is_in(["LONG_BIAS", "SHORT_BIAS"])
        ).alias("exec_candidate_before"),
        (
            pl.col("signal_eligible")
            & pl.col("overlay_allow_signal")
            & pl.col("state_direction_hint").is_in(["LONG_BIAS", "SHORT_BIAS"])
            & pl.col("execution_eligible")
        ).alias("exec_candidate_after"),
    ).with_columns(
        (
            pl.col("exec_candidate_before") & (~pl.col("exec_candidate_after"))
        ).alias("exec_suppressed_signal")
    )

    rows_total = int(work.height)
    rows_eligible = int(work.filter(pl.col("execution_eligible")).height)
    rows_ineligible = rows_total - rows_eligible
    cand_before = int(work.select(pl.col("exec_candidate_before").cast(pl.Int64).sum()).item() or 0)
    cand_after = int(work.select(pl.col("exec_candidate_after").cast(pl.Int64).sum()).item() or 0)
    suppressed = cand_before - cand_after

    by_reason = _reason_table(work)
    by_year = (
        work.with_columns(pl.col("trade_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("rows_total"),
            pl.col("execution_eligible").cast(pl.Int64).sum().alias("rows_eligible"),
            pl.col("exec_candidate_before").cast(pl.Int64).sum().alias("candidate_signals_before"),
            pl.col("exec_candidate_after").cast(pl.Int64).sum().alias("candidate_signals_after"),
            pl.col("exec_suppressed_signal").cast(pl.Int64).sum().alias("suppressed_signal_count"),
            pl.col("exec_dollar_vol_20").cast(pl.Float64, strict=False).mean().alias("avg_dollar_vol_20"),
            pl.col("close").cast(pl.Float64, strict=False).median().alias("median_close"),
            pl.col("exec_vol_pct").cast(pl.Float64, strict=False).mean().alias("avg_vol_pct"),
        )
        .with_columns(
            (pl.col("rows_eligible") / pl.col("rows_total").cast(pl.Float64)).alias("eligibility_rate"),
            pl.when(pl.col("candidate_signals_before") > 0)
            .then(
                pl.col("suppressed_signal_count").cast(pl.Float64)
                / pl.col("candidate_signals_before").cast(pl.Float64)
            )
            .otherwise(0.0)
            .alias("suppressed_signal_share"),
        )
        .sort("year")
    )

    summary = {
        "profile_name": params.profile_name,
        "params_used": {
            "min_price": params.min_price,
            "min_dollar_vol_20": params.min_dollar_vol_20,
            "max_vol_pct": params.max_vol_pct,
            "min_history_bars_for_execution": params.min_history_bars_for_execution,
            "dollar_vol_window": params.dollar_vol_window,
            "dollar_vol_rolling_method": params.dollar_vol_rolling_method,
            "vol_input_unit_mode": params.vol_input_unit_mode,
        },
        "vol_metric_source": vol_metric_source,
        "vol_unit_detected": vol_unit_detected,
        "vol_raw_median": _safe_float(vol_raw_median),
        "vol_raw_p90": _safe_float(vol_raw_p90),
        "vol_threshold_input": _safe_float(params.max_vol_pct),
        "vol_threshold_effective_decimal": _safe_float(vol_threshold_effective_decimal),
        "vol_threshold_effective_pct": _safe_float(vol_threshold_effective_pct),
        "vol_threshold_input_interpretation": vol_threshold_input_interpretation,
        "vol_unit_warnings": vol_unit_warnings,
        "rows_total": rows_total,
        "rows_eligible": rows_eligible,
        "rows_ineligible": rows_ineligible,
        "eligibility_rate": _safe_float((rows_eligible / rows_total) if rows_total > 0 else None),
        "candidate_signals_before_filters": cand_before,
        "candidate_signals_after_filters": cand_after,
        "suppressed_signal_count": suppressed,
        "suppressed_signal_share": _safe_float((suppressed / cand_before) if cand_before > 0 else 0.0),
        "suppressed_by_reason": {
            row.get("reason"): row.get("suppressed_signal_share")
            for row in by_reason.to_dicts()
        },
        "suppressed_by_reason_count": {
            row.get("reason"): int(row.get("suppressed_signal_count", 0) or 0)
            for row in by_reason.to_dicts()
        },
        "exec_suppressed_by_price_count": int(
            by_reason.filter(pl.col("reason") == "price_floor").select("suppressed_signal_count").item() or 0
        ),
        "exec_suppressed_by_liquidity_count": int(
            by_reason.filter(pl.col("reason") == "liquidity_floor").select("suppressed_signal_count").item() or 0
        ),
        "exec_suppressed_by_vol_count": int(
            by_reason.filter(pl.col("reason") == "vol_cap").select("suppressed_signal_count").item() or 0
        ),
        "exec_suppressed_by_warmup_count": int(
            by_reason.filter(pl.col("reason") == "warmup").select("suppressed_signal_count").item() or 0
        ),
        "exec_suppressed_by_price_share": _safe_float(
            by_reason.filter(pl.col("reason") == "price_floor").select("suppressed_signal_share").item()
            if by_reason.height > 0
            else 0.0
        )
        or 0.0,
        "exec_suppressed_by_liquidity_share": _safe_float(
            by_reason.filter(pl.col("reason") == "liquidity_floor").select("suppressed_signal_share").item()
            if by_reason.height > 0
            else 0.0
        )
        or 0.0,
        "exec_suppressed_by_vol_share": _safe_float(
            by_reason.filter(pl.col("reason") == "vol_cap").select("suppressed_signal_share").item()
            if by_reason.height > 0
            else 0.0
        )
        or 0.0,
        "exec_suppressed_by_warmup_share": _safe_float(
            by_reason.filter(pl.col("reason") == "warmup").select("suppressed_signal_share").item()
            if by_reason.height > 0
            else 0.0
        )
        or 0.0,
        "filters_enabled": bool(filters_enabled),
    }
    effective_logger.info(
        "backtest.execution_realism profile=%s enabled=%s rows=%s eligible=%s suppressed_share=%s",
        params.profile_name,
        filters_enabled,
        rows_total,
        rows_eligible,
        summary["suppressed_signal_share"],
    )
    return ExecutionRealismResult(
        frame=work,
        summary=summary,
        by_reason=by_reason,
        by_year=by_year,
        filters_enabled=filters_enabled,
    )


def build_execution_trade_context_summary(
    *,
    trades: pl.DataFrame,
    signal_frame: pl.DataFrame,
) -> dict[str, Any]:
    """Build realized-trades context summary for execution realism diagnostics."""

    if trades.height == 0:
        return {
            "trade_count": 0,
            "avg_close": None,
            "median_close": None,
            "avg_dollar_vol_20": None,
            "p10_dollar_vol_20": None,
            "p50_dollar_vol_20": None,
            "p90_dollar_vol_20": None,
            "avg_vol_pct": None,
            "pct_trades_near_liquidity_threshold": None,
            "pct_trades_near_price_threshold": None,
            "pct_trades_near_vol_threshold": None,
        }

    if signal_frame.height == 0:
        return {
            "trade_count": int(trades.height),
            "avg_close": None,
            "median_close": None,
            "avg_dollar_vol_20": None,
            "p10_dollar_vol_20": None,
            "p50_dollar_vol_20": None,
            "p90_dollar_vol_20": None,
            "avg_vol_pct": None,
            "pct_trades_near_liquidity_threshold": None,
            "pct_trades_near_price_threshold": None,
            "pct_trades_near_vol_threshold": None,
        }

    ctx = signal_frame.select(
        pl.col("ticker").cast(pl.String).alias("_ticker"),
        pl.col("trade_date").cast(pl.Date).alias("_signal_date"),
        pl.col("close").cast(pl.Float64, strict=False).alias("_close"),
        pl.col("exec_dollar_vol_20").cast(pl.Float64, strict=False).alias("_dv20"),
        pl.col("exec_vol_pct").cast(pl.Float64, strict=False).alias("_vol_pct"),
        pl.col("exec_filter_price_fail").cast(pl.Boolean).alias("_price_fail"),
        pl.col("exec_filter_liquidity_fail").cast(pl.Boolean).alias("_liq_fail"),
        pl.col("exec_filter_vol_fail").cast(pl.Boolean).alias("_vol_fail"),
    )
    joined = trades.join(
        ctx,
        left_on=["ticker", "entry_signal_date"],
        right_on=["_ticker", "_signal_date"],
        how="left",
    )
    if joined.height == 0:
        return {
            "trade_count": 0,
            "avg_close": None,
            "median_close": None,
            "avg_dollar_vol_20": None,
            "p10_dollar_vol_20": None,
            "p50_dollar_vol_20": None,
            "p90_dollar_vol_20": None,
            "avg_vol_pct": None,
            "pct_trades_near_liquidity_threshold": None,
            "pct_trades_near_price_threshold": None,
            "pct_trades_near_vol_threshold": None,
        }

    trade_count = int(joined.height)
    return {
        "trade_count": trade_count,
        "avg_close": _safe_float(joined.select(pl.col("_close").mean()).item()),
        "median_close": _safe_float(joined.select(pl.col("_close").median()).item()),
        "avg_dollar_vol_20": _safe_float(joined.select(pl.col("_dv20").mean()).item()),
        "p10_dollar_vol_20": _safe_float(joined.select(pl.col("_dv20").quantile(0.10)).item()),
        "p50_dollar_vol_20": _safe_float(joined.select(pl.col("_dv20").quantile(0.50)).item()),
        "p90_dollar_vol_20": _safe_float(joined.select(pl.col("_dv20").quantile(0.90)).item()),
        "avg_vol_pct": _safe_float(joined.select(pl.col("_vol_pct").mean()).item()),
        "pct_trades_near_liquidity_threshold": _safe_float(
            joined.select(pl.col("_liq_fail").cast(pl.Float64, strict=False).mean()).item()
        ),
        "pct_trades_near_price_threshold": _safe_float(
            joined.select(pl.col("_price_fail").cast(pl.Float64, strict=False).mean()).item()
        ),
        "pct_trades_near_vol_threshold": _safe_float(
            joined.select(pl.col("_vol_fail").cast(pl.Float64, strict=False).mean()).item()
        ),
    }
