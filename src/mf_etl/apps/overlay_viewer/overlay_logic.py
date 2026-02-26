"""Overlay + execution realism + candidate marker logic for viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from mf_etl.backtest.execution_realism import (
    apply_execution_realism_filter,
    resolve_execution_realism_params,
)
from mf_etl.backtest.models import InputType, OverlayMode, SignalMode
from mf_etl.backtest.policy_overlay import apply_policy_overlay
from mf_etl.backtest.signals import generate_signals
from mf_etl.backtest.state_mapping import (
    apply_cluster_policy_mapping,
    apply_flow_state_mapping,
    apply_hmm_state_mapping,
)
from mf_etl.config import AppSettings

ManualCoverageMode = Literal["warn_only", "strict_fail"]


@dataclass(frozen=True, slots=True)
class ViewerRuntimeConfig:
    """Resolved runtime settings for one overlay viewer computation."""

    input_type: InputType
    signal_mode: SignalMode
    validation_run_dir: Path | None
    cluster_hardening_dir: Path | None
    state_map_file: Path | None
    include_watch: bool
    policy_filter_mode: str
    include_state_ids: list[int]
    allow_unconfirmed: bool
    overlay_cluster_file: Path | None
    overlay_cluster_hardening_dir: Path | None
    overlay_mode: OverlayMode
    overlay_join_keys: list[str]
    overlay_coverage_mode: ManualCoverageMode
    overlay_coverage_bypass: bool
    execution_profile: str
    exec_min_price: float | None
    exec_min_dollar_vol20: float | None
    exec_max_vol_pct: float | None
    exec_min_history_bars: int | None
    hmm_long_bias_states: tuple[int, ...] = ()
    hmm_short_bias_states: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ViewerOverlayResult:
    frame: pl.DataFrame
    mapping_summary: dict[str, Any]
    overlay_summary: dict[str, Any]
    execution_summary: dict[str, Any]
    signal_diagnostics: dict[str, Any]
    warnings: list[str]


def _safe_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def runtime_from_candidate(packet: dict[str, Any], candidate_name: str) -> ViewerRuntimeConfig:
    """Build runtime config from PCP packet candidate object."""

    candidates = packet.get("candidates", {})
    if not isinstance(candidates, dict) or candidate_name not in candidates:
        raise ValueError(f"Candidate {candidate_name} not found in PCP packet")
    cand = candidates[candidate_name]
    overlay = cand.get("overlay", {}) if isinstance(cand.get("overlay"), dict) else {}
    execution = (
        cand.get("execution_realism", {}) if isinstance(cand.get("execution_realism"), dict) else {}
    )
    thresholds = (
        execution.get("thresholds_used", {}) if isinstance(execution.get("thresholds_used"), dict) else {}
    )
    strategy = cand.get("strategy_params", {}) if isinstance(cand.get("strategy_params"), dict) else {}

    input_type = str(cand.get("input_type", "hmm")).strip().lower()
    if input_type not in {"hmm", "flow", "cluster"}:
        input_type = "hmm"

    mode = str(overlay.get("mode", "none")).strip().lower()
    if mode not in {"none", "allow_only", "allow_watch", "block_veto", "allow_or_unknown"}:
        mode = "none"

    signal_mode = str(strategy.get("signal_mode", "state_transition_entry")).strip().lower()
    if signal_mode not in {"state_entry", "state_transition_entry", "state_persistence_confirm"}:
        signal_mode = "state_transition_entry"

    execution_profile = str(execution.get("profile", "none")).strip().lower()
    if execution_profile not in {"none", "lite", "strict"}:
        execution_profile = "none"

    return ViewerRuntimeConfig(
        input_type=input_type,  # type: ignore[arg-type]
        signal_mode=signal_mode,  # type: ignore[arg-type]
        validation_run_dir=_safe_path(cand.get("validation_run_dir")),
        cluster_hardening_dir=None,
        state_map_file=None,
        include_watch=False,
        policy_filter_mode="allow_only",
        include_state_ids=[],
        allow_unconfirmed=bool(cand.get("input_type", "hmm") == "flow"),
        overlay_cluster_file=_safe_path(overlay.get("overlay_cluster_file")),
        overlay_cluster_hardening_dir=_safe_path(overlay.get("overlay_cluster_hardening_dir")),
        overlay_mode=mode,  # type: ignore[arg-type]
        overlay_join_keys=["ticker", "trade_date"],
        overlay_coverage_mode="warn_only",
        overlay_coverage_bypass=True,
        execution_profile=execution_profile,
        exec_min_price=(float(thresholds["min_price"]) if thresholds.get("min_price") is not None else None),
        exec_min_dollar_vol20=(
            float(thresholds["min_dollar_vol20"]) if thresholds.get("min_dollar_vol20") is not None else None
        ),
        exec_max_vol_pct=(
            float(thresholds["max_vol_pct"]) if thresholds.get("max_vol_pct") is not None else None
        ),
        exec_min_history_bars=(
            int(thresholds["min_history_bars"]) if thresholds.get("min_history_bars") is not None else None
        ),
    )


def _ensure_state_columns(frame: pl.DataFrame, input_type: InputType) -> pl.DataFrame:
    out = frame
    if "ticker" not in out.columns or "trade_date" not in out.columns:
        raise ValueError("Primary rows must include ticker and trade_date columns.")

    if input_type == "hmm":
        if "hmm_state" not in out.columns:
            if "state_id" in out.columns:
                out = out.with_columns(pl.col("state_id").cast(pl.Int32, strict=False).alias("hmm_state"))
            else:
                raise ValueError("HMM mode requires hmm_state column.")
    elif input_type == "flow":
        if "flow_state_code" not in out.columns:
            if "state_id" in out.columns:
                out = out.with_columns(pl.col("state_id").cast(pl.Int32, strict=False).alias("flow_state_code"))
            else:
                raise ValueError("FLOW mode requires flow_state_code column.")
    else:
        if "cluster_id" not in out.columns:
            if "state_id" in out.columns:
                out = out.with_columns(pl.col("state_id").cast(pl.Int32, strict=False).alias("cluster_id"))
            else:
                raise ValueError("CLUSTER mode requires cluster_id column.")

    with_defaults = out.with_columns(
        pl.col("ticker").cast(pl.String).str.to_uppercase().alias("ticker"),
        pl.col("trade_date").cast(pl.String, strict=False).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("trade_date"),
        pl.col("open").cast(pl.Float64, strict=False).alias("open") if "open" in out.columns else pl.col("close").cast(pl.Float64, strict=False).alias("open"),
        pl.col("high").cast(pl.Float64, strict=False).alias("high") if "high" in out.columns else pl.col("close").cast(pl.Float64, strict=False).alias("high"),
        pl.col("low").cast(pl.Float64, strict=False).alias("low") if "low" in out.columns else pl.col("close").cast(pl.Float64, strict=False).alias("low"),
        pl.col("close").cast(pl.Float64, strict=False).alias("close"),
        pl.col("volume").cast(pl.Float64, strict=False).alias("volume") if "volume" in out.columns else pl.lit(None).cast(pl.Float64).alias("volume"),
        pl.col("atr_pct_14").cast(pl.Float64, strict=False).alias("atr_pct_14") if "atr_pct_14" in out.columns else pl.lit(None).cast(pl.Float64).alias("atr_pct_14"),
        pl.col("trade_dt").cast(pl.Datetime("us"), strict=False).alias("trade_dt") if "trade_dt" in out.columns else pl.lit(None).cast(pl.Datetime("us")).alias("trade_dt"),
        pl.col("timeframe").cast(pl.String).alias("timeframe") if "timeframe" in out.columns else pl.lit(None).cast(pl.String).alias("timeframe"),
    )

    return with_defaults.sort("trade_date")


def _apply_state_mapping(
    frame: pl.DataFrame,
    *,
    runtime: ViewerRuntimeConfig,
    settings: AppSettings,
) -> tuple[pl.DataFrame, dict[str, Any], list[str]]:
    warnings: list[str] = []
    if runtime.input_type == "flow":
        mapped, summary, _snapshot = apply_flow_state_mapping(
            frame,
            settings=settings,
            include_state_ids=runtime.include_state_ids,
            allow_unconfirmed=runtime.allow_unconfirmed,
        )
        return mapped, summary, warnings

    if runtime.input_type == "hmm":
        if runtime.hmm_long_bias_states or runtime.hmm_short_bias_states:
            long_set = set(int(v) for v in runtime.hmm_long_bias_states)
            short_set = set(int(v) for v in runtime.hmm_short_bias_states)
            fallback = frame.with_columns(
                pl.when(pl.col("hmm_state").cast(pl.Int32, strict=False).is_in(sorted(long_set)))
                .then(pl.lit("LONG_BIAS"))
                .when(pl.col("hmm_state").cast(pl.Int32, strict=False).is_in(sorted(short_set)))
                .then(pl.lit("SHORT_BIAS"))
                .otherwise(pl.lit("UNCONFIRMED"))
                .alias("state_direction_hint"),
                pl.lit("NA").alias("state_class"),
                pl.lit(None).cast(pl.Float64).alias("state_score"),
                pl.lit("HMM_LOCAL_CACHE_FALLBACK").alias("policy_reason_flags"),
            ).with_columns(
                (
                    pl.col("state_direction_hint").is_in(["LONG_BIAS", "SHORT_BIAS"])
                    | (pl.lit(runtime.allow_unconfirmed) & (pl.col("state_direction_hint") == "UNCONFIRMED"))
                ).alias("signal_eligible"),
                pl.col("hmm_state").cast(pl.Int32, strict=False).alias("state_id"),
            )
            return (
                fallback,
                {
                    "mapping_source": "hmm_local_cache_fallback",
                    "long_bias_states": sorted(long_set),
                    "short_bias_states": sorted(short_set),
                },
                warnings,
            )
        try:
            mapped, summary, _snapshot = apply_hmm_state_mapping(
                frame,
                settings=settings,
                validation_run_dir=runtime.validation_run_dir,
                state_map_file=runtime.state_map_file,
                include_state_ids=runtime.include_state_ids,
                allow_unconfirmed=runtime.allow_unconfirmed,
            )
            return mapped, summary, warnings
        except Exception as exc:
            warnings.append(
                f"HMM direction mapping fallback to flow-state heuristic: {exc}"
            )
            if "flow_state_code" in frame.columns:
                fallback = frame.with_columns(
                    pl.when(pl.col("flow_state_code").cast(pl.Int32, strict=False).is_in([1, 2]))
                    .then(pl.lit("LONG_BIAS"))
                    .when(pl.col("flow_state_code").cast(pl.Int32, strict=False).is_in([3, 4]))
                    .then(pl.lit("SHORT_BIAS"))
                    .otherwise(pl.lit("UNCONFIRMED"))
                    .alias("state_direction_hint"),
                    pl.lit("NA").alias("state_class"),
                    pl.lit(None).cast(pl.Float64).alias("state_score"),
                    pl.lit("HMM_FLOW_FALLBACK").alias("policy_reason_flags"),
                ).with_columns(
                    (
                        pl.col("state_direction_hint").is_in(["LONG_BIAS", "SHORT_BIAS"])
                        | (pl.lit(runtime.allow_unconfirmed) & (pl.col("state_direction_hint") == "UNCONFIRMED"))
                    ).alias("signal_eligible")
                )
                return fallback, {"mapping_source": "hmm_flow_fallback"}, warnings
            raise

    if runtime.cluster_hardening_dir is None:
        raise ValueError("Cluster mode requires cluster_hardening_dir.")
    mapped, summary, _snapshot = apply_cluster_policy_mapping(
        frame,
        settings=settings,
        cluster_hardening_dir=runtime.cluster_hardening_dir,
        include_watch=runtime.include_watch,
        policy_filter_mode=runtime.policy_filter_mode,
        include_state_ids=runtime.include_state_ids,
        allow_unconfirmed=runtime.allow_unconfirmed,
    )
    return mapped, summary, warnings


def _apply_overlay(
    frame: pl.DataFrame,
    *,
    runtime: ViewerRuntimeConfig,
    settings: AppSettings,
) -> tuple[pl.DataFrame, dict[str, Any], list[str]]:
    warnings: list[str] = []

    def _allow_expr() -> pl.Expr:
        unknown_pass = settings.overlay_coverage_policy.unknown_handling == "treat_unknown_as_pass"
        if runtime.overlay_mode == "allow_only":
            return pl.col("overlay_policy_class") == "ALLOW"
        if runtime.overlay_mode == "allow_watch":
            return pl.col("overlay_policy_class").is_in(["ALLOW", "WATCH"])
        if runtime.overlay_mode == "block_veto":
            if unknown_pass:
                return pl.col("overlay_policy_class") != "BLOCK"
            return pl.col("overlay_policy_class").is_in(["ALLOW", "WATCH"])
        if runtime.overlay_mode == "allow_or_unknown":
            return pl.col("overlay_policy_class").is_in(["ALLOW", "UNKNOWN"])
        return pl.lit(True)

    def _summary_from_frame(df: pl.DataFrame, *, overlay_enabled: bool, mode: str) -> dict[str, Any]:
        rows = int(df.height)
        matched = int(df.filter(pl.col("overlay_join_status") == "MATCHED").height) if "overlay_join_status" in df.columns else 0
        unknown = int(df.filter(pl.col("overlay_policy_class") == "UNKNOWN").height) if "overlay_policy_class" in df.columns else rows
        allow = int(df.filter(pl.col("overlay_policy_class") == "ALLOW").height) if "overlay_policy_class" in df.columns else 0
        watch = int(df.filter(pl.col("overlay_policy_class") == "WATCH").height) if "overlay_policy_class" in df.columns else 0
        block = int(df.filter(pl.col("overlay_policy_class") == "BLOCK").height) if "overlay_policy_class" in df.columns else 0
        return {
            "overlay_enabled": overlay_enabled,
            "overlay_mode": mode,
            "rows_total": rows,
            "matched_rows": matched,
            "unmatched_rows": (rows - matched) if rows > 0 else 0,
            "match_rate": float(matched / rows) if rows > 0 else None,
            "unknown_rows": unknown,
            "unknown_rate": float(unknown / rows) if rows > 0 else None,
            "allow_rows": allow,
            "watch_rows": watch,
            "block_rows": block,
        }

    has_precomputed_overlay = "overlay_policy_class" in frame.columns
    if has_precomputed_overlay:
        precomputed = frame.with_columns(
            pl.col("overlay_policy_class")
            .cast(pl.String, strict=False)
            .str.to_uppercase()
            .fill_null("UNKNOWN")
            .alias("overlay_policy_class"),
            pl.col("overlay_cluster_state").cast(pl.Int32, strict=False).alias("overlay_cluster_state")
            if "overlay_cluster_state" in frame.columns
            else pl.lit(None).cast(pl.Int32).alias("overlay_cluster_state"),
            pl.col("overlay_direction_hint").cast(pl.String, strict=False).fill_null("UNCONFIRMED").alias("overlay_direction_hint")
            if "overlay_direction_hint" in frame.columns
            else pl.lit("UNCONFIRMED").alias("overlay_direction_hint"),
            pl.col("overlay_tradability_score").cast(pl.Float64, strict=False).alias("overlay_tradability_score")
            if "overlay_tradability_score" in frame.columns
            else pl.lit(None).cast(pl.Float64).alias("overlay_tradability_score"),
            pl.col("overlay_join_status").cast(pl.String, strict=False).fill_null("UNMATCHED_PRIMARY").alias("overlay_join_status")
            if "overlay_join_status" in frame.columns
            else pl.lit("UNMATCHED_PRIMARY").alias("overlay_join_status"),
        )
        if runtime.overlay_mode == "none":
            out = precomputed.with_columns(
                pl.lit("none").alias("overlay_mode"),
                pl.lit(True).alias("overlay_allow_signal"),
                pl.lit(False).alias("overlay_enabled"),
            )
            return out, _summary_from_frame(out, overlay_enabled=False, mode="none"), warnings

        out = precomputed.with_columns(
            pl.lit(runtime.overlay_mode).alias("overlay_mode"),
            _allow_expr().alias("overlay_allow_signal"),
            pl.lit(True).alias("overlay_enabled"),
        )
        return out, _summary_from_frame(out, overlay_enabled=True, mode=runtime.overlay_mode), warnings

    if runtime.overlay_mode == "none":
        out = frame.with_columns(
            pl.lit(None).cast(pl.Int32).alias("overlay_cluster_state"),
            pl.lit("UNKNOWN").alias("overlay_policy_class"),
            pl.lit("UNCONFIRMED").alias("overlay_direction_hint"),
            pl.lit(None).cast(pl.Float64).alias("overlay_tradability_score"),
            pl.lit("UNMATCHED_PRIMARY").alias("overlay_join_status"),
            pl.lit("none").alias("overlay_mode"),
            pl.lit(True).alias("overlay_allow_signal"),
            pl.lit(False).alias("overlay_enabled"),
        )
        return out, {"overlay_enabled": False, "overlay_mode": "none", "match_rate": None}, warnings

    if runtime.overlay_cluster_file is None or runtime.overlay_cluster_hardening_dir is None:
        warnings.append("Overlay requested but overlay files are missing; overlay gate disabled.")
        out = frame.with_columns(
            pl.lit(None).cast(pl.Int32).alias("overlay_cluster_state"),
            pl.lit("UNKNOWN").alias("overlay_policy_class"),
            pl.lit("UNCONFIRMED").alias("overlay_direction_hint"),
            pl.lit(None).cast(pl.Float64).alias("overlay_tradability_score"),
            pl.lit("UNMATCHED_PRIMARY").alias("overlay_join_status"),
            pl.lit("none").alias("overlay_mode"),
            pl.lit(True).alias("overlay_allow_signal"),
            pl.lit(False).alias("overlay_enabled"),
        )
        return out, {"overlay_enabled": False, "overlay_mode": "none", "match_rate": None}, warnings

    overlay_result = apply_policy_overlay(
        frame,
        overlay_cluster_file=runtime.overlay_cluster_file,
        overlay_cluster_hardening_dir=runtime.overlay_cluster_hardening_dir,
        overlay_mode=runtime.overlay_mode,
        join_keys=runtime.overlay_join_keys,
        unknown_handling=settings.overlay_coverage_policy.unknown_handling,
        min_match_rate_warn=settings.overlay_coverage_policy.min_match_rate_warn,
        min_match_rate_fail=settings.overlay_coverage_policy.min_match_rate_fail,
        min_year_match_rate_warn=settings.overlay_coverage_policy.min_year_match_rate_warn,
        min_year_match_rate_fail=settings.overlay_coverage_policy.min_year_match_rate_fail,
        unknown_rate_warn=settings.overlay_coverage_policy.unknown_rate_warn,
        unknown_rate_fail=settings.overlay_coverage_policy.unknown_rate_fail,
        coverage_mode=runtime.overlay_coverage_mode,
        logger=None,
    )

    if runtime.overlay_coverage_mode == "strict_fail" and str(overlay_result.coverage_verdict.get("status", "")).startswith("FAIL") and not runtime.overlay_coverage_bypass:
        raise ValueError(
            "Overlay coverage strict fail: "
            + ";".join(str(x) for x in (overlay_result.coverage_verdict.get("reasons") or []))
        )

    return overlay_result.frame, overlay_result.join_summary, warnings


def build_overlay_view(
    primary_rows: pl.DataFrame,
    *,
    runtime: ViewerRuntimeConfig,
    settings: AppSettings,
) -> ViewerOverlayResult:
    """Compute mapped state layers, overlay gating, execution realism and signal markers."""

    warnings: list[str] = []
    prepared = _ensure_state_columns(primary_rows, runtime.input_type)
    mapped, mapping_summary, map_warnings = _apply_state_mapping(prepared, runtime=runtime, settings=settings)
    warnings.extend(map_warnings)

    overlayed, overlay_summary, overlay_warnings = _apply_overlay(mapped, runtime=runtime, settings=settings)
    warnings.extend(overlay_warnings)

    exec_params = resolve_execution_realism_params(
        settings,
        exec_profile=runtime.execution_profile if runtime.execution_profile in {"none", "lite", "strict"} else "none",  # type: ignore[arg-type]
        exec_min_price=runtime.exec_min_price,
        exec_min_dollar_vol20=runtime.exec_min_dollar_vol20,
        exec_max_vol_pct=runtime.exec_max_vol_pct,
        exec_min_history_bars=runtime.exec_min_history_bars,
        report_min_trades=settings.backtest_execution_realism.report_min_trades_default,
        report_max_zero_trade_share=settings.backtest_execution_realism.report_max_zero_trade_share_default,
        report_max_ret_cv=settings.backtest_execution_realism.report_max_ret_cv_default,
    )
    exec_result = apply_execution_realism_filter(overlayed, params=exec_params)

    signal_result = generate_signals(
        exec_result.frame,
        signal_mode=runtime.signal_mode,
        confirm_bars=2,
    )

    frame = signal_result.frame.with_columns(
        pl.when(pl.col("entry_signal"))
        .then(pl.lit("candidate_signal"))
        .when(pl.col("overlay_vetoed_signal"))
        .then(pl.lit("blocked_overlay"))
        .when(pl.col("execution_suppressed_signal"))
        .then(pl.lit("blocked_execution"))
        .otherwise(pl.lit(None))
        .alias("candidate_marker_type"),
        pl.when(pl.col("overlay_vetoed_signal"))
        .then(pl.format("overlay:{}", pl.col("overlay_policy_class")))
        .when(pl.col("execution_suppressed_signal"))
        .then(pl.format("exec:{}", pl.col("execution_filter_reason")))
        .otherwise(pl.lit(None))
        .alias("blocked_reason"),
    )

    return ViewerOverlayResult(
        frame=frame,
        mapping_summary=mapping_summary,
        overlay_summary=overlay_summary,
        execution_summary=exec_result.summary,
        signal_diagnostics=signal_result.diagnostics,
        warnings=warnings,
    )
