"""On-demand per-ticker compute pipeline with persistent cache for overlay viewer v1.1.1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl

from mf_etl.apps.overlay_viewer.cache import (
    CACHE_SCHEMA_VERSION,
    build_cache_run_id,
    cache_run_dir,
    load_cache_bundle,
    run_exists,
    write_cache_run,
)
from mf_etl.apps.overlay_viewer.data_loader import (
    discover_default_paths,
    load_rows_for_ticker,
)
from mf_etl.apps.overlay_viewer.indicators_twiggs import TwiggsParams, compute_tmf_tti
from mf_etl.apps.overlay_viewer.flow_states_local import (
    LocalFlowStateParams,
    compute_local_flow_states,
)
from mf_etl.backtest.execution_realism import (
    apply_execution_realism_filter,
    resolve_execution_realism_params,
)
from mf_etl.backtest.policy_overlay import apply_policy_overlay
from mf_etl.config import AppSettings

LOGGER = logging.getLogger(__name__)

try:  # optional dependency safety
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover - depends on env
    GaussianHMM = None

from sklearn.preprocessing import RobustScaler, StandardScaler

ProgressCallback = Callable[[float, str], None]


COMPUTE_CODE_VERSION = "overlay_viewer_compute_v1_1_1"


@dataclass(frozen=True, slots=True)
class ComputeTickerParams:
    ticker: str
    source_file: Path | None
    full_history: bool
    date_from: date | None
    date_to: date | None
    compute_flow_states: bool
    compute_local_hmm_states: bool
    compute_exec_realism: bool
    attempt_global_overlay_join: bool
    hmm_n_components: int
    hmm_scaler: str
    exec_profile: str
    exec_min_price: float | None
    exec_min_dollar_vol20: float | None
    exec_max_vol_pct: float | None
    exec_min_history_bars: int | None
    force: bool


@dataclass(frozen=True, slots=True)
class ComputeTickerResult:
    ticker: str
    run_id: str
    run_dir: Path
    cache_hit: bool
    meta_path: Path
    summary_path: Path
    warnings: list[str]


def _emit(progress: ProgressCallback | None, frac: float, text: str) -> None:
    if progress is not None:
        progress(max(0.0, min(1.0, frac)), text)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _resolve_source_file(settings: AppSettings, source_file: Path | None) -> Path:
    if source_file is not None:
        if not source_file.exists():
            raise FileNotFoundError(f"Source file does not exist: {source_file}")
        return source_file
    defaults = discover_default_paths(settings)
    if defaults.get("ml_dataset") is not None:
        return defaults["ml_dataset"]  # type: ignore[return-value]
    raise FileNotFoundError("No source file available; pass --source-file or create gold ml_dataset export.")


def _load_rows_with_fallback(
    *,
    settings: AppSettings,
    ticker: str,
    source_file: Path,
    date_from: str | None,
    date_to: str | None,
    source_file_explicit: bool,
    warnings: list[str],
) -> tuple[pl.DataFrame, Path]:
    rows = load_rows_for_ticker(
        source_file,
        ticker=ticker,
        date_from=date_from,
        date_to=date_to,
    )
    if rows.height > 0:
        return rows, source_file

    gold_candidates = sorted(
        (settings.paths.data_root / "gold" / "datasets" / "ml_dataset_v1").glob("*/dataset.parquet"),
        key=lambda path: path.as_posix(),
        reverse=True,
    )
    for candidate in gold_candidates:
        if candidate == source_file:
            continue
        try:
            alt_rows = load_rows_for_ticker(
                candidate,
                ticker=ticker,
                date_from=date_from,
                date_to=date_to,
            )
        except Exception:
            continue
        if alt_rows.height > 0:
            warnings.append(f"Source fallback applied: {source_file} -> {candidate}")
            return alt_rows, candidate

    prefix = ticker[:1].upper()
    bronze_files = list(
        (settings.paths.bronze_root / "ohlcv_by_symbol").glob(
            f"exchange=*/prefix={prefix}/ticker={ticker}/part-*.parquet"
        )
    )
    for part in sorted(bronze_files):
        try:
            alt_rows = load_rows_for_ticker(
                part,
                ticker=ticker,
                date_from=date_from,
                date_to=date_to,
            )
        except Exception:
            continue
        if alt_rows.height > 0:
            warnings.append(f"Source fallback applied: {source_file} -> {part}")
            if source_file_explicit:
                warnings.append(
                    "Explicit source file had no rows for ticker; used bronze ticker parquet fallback."
                )
            return alt_rows, part

    return rows, source_file


def _ensure_ohlcv(frame: pl.DataFrame) -> pl.DataFrame:
    if "close" not in frame.columns:
        raise ValueError("Input rows must contain close column.")
    out = frame.with_columns(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        pl.col("trade_date").cast(pl.String, strict=False).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("trade_date"),
        pl.col("close").cast(pl.Float64, strict=False).alias("close"),
        pl.col("open").cast(pl.Float64, strict=False).alias("open") if "open" in frame.columns else pl.col("close").cast(pl.Float64, strict=False).alias("open"),
        pl.col("high").cast(pl.Float64, strict=False).alias("high") if "high" in frame.columns else pl.col("close").cast(pl.Float64, strict=False).alias("high"),
        pl.col("low").cast(pl.Float64, strict=False).alias("low") if "low" in frame.columns else pl.col("close").cast(pl.Float64, strict=False).alias("low"),
        pl.col("volume").cast(pl.Float64, strict=False).alias("volume") if "volume" in frame.columns else pl.lit(None).cast(pl.Float64).alias("volume"),
        pl.col("atr_pct_14").cast(pl.Float64, strict=False).alias("atr_pct_14") if "atr_pct_14" in frame.columns else pl.lit(None).cast(pl.Float64).alias("atr_pct_14"),
    ).sort("trade_date")
    return out


def _compute_indicators(ohlcv: pl.DataFrame) -> pl.DataFrame:
    pdf = ohlcv.select(["ticker", "trade_date", "open", "high", "low", "close", "volume"]).to_pandas()
    pdf["trade_date"] = pd.to_datetime(pdf["trade_date"], errors="coerce")
    ind = compute_tmf_tti(pdf, TwiggsParams(period=21, scale_pct=False, tti_scale=1.0))
    ind["ret_1"] = ind["close"].pct_change(1)
    ind["ret_5"] = ind["close"].pct_change(5)
    ind["tmf_slope_1"] = ind["tmf_raw"].diff(1)
    ind["tmf_slope_5"] = ind["tmf_raw"].diff(5)
    ind["tti_slope_1"] = ind["tti_raw"].diff(1)
    ind["tti_slope_5"] = ind["tti_raw"].diff(5)
    ind["range_pct"] = (ind["high"] - ind["low"]) / ind["close"].replace(0, np.nan)
    ind["dollar_vol"] = ind["close"] * ind["volume"]
    ind["dollar_vol_20"] = ind["dollar_vol"].rolling(20, min_periods=20).median()
    ind["dollar_vol_20_log"] = np.log1p(ind["dollar_vol_20"].clip(lower=0))

    keep = [
        "ticker",
        "trade_date",
        "prev_close",
        "trh",
        "trl",
        "tr",
        "adv",
        "advv",
        "tmf_raw",
        "tti_raw",
        "tmf",
        "tti",
        "tmf_zero_cross",
        "tti_zero_cross",
        "ret_1",
        "ret_5",
        "tmf_slope_1",
        "tmf_slope_5",
        "tti_slope_1",
        "tti_slope_5",
        "range_pct",
        "dollar_vol",
        "dollar_vol_20",
        "dollar_vol_20_log",
    ]
    out = pl.from_pandas(ind[keep]).with_columns(
        pl.col("trade_date").cast(pl.Datetime("us"), strict=False).dt.date().alias("trade_date"),
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
    )
    return out.sort("trade_date")


def _compute_flow_states(
    indicators: pl.DataFrame,
    *,
    settings: AppSettings,
    enabled: bool,
    warnings: list[str],
) -> tuple[pl.DataFrame | None, dict[str, Any]]:
    summary: dict[str, Any] = {
        "enabled": enabled,
        "states_flow_source": None,
        "ready_ratio": None,
        "state_counts": {},
    }
    if not enabled:
        return None, summary

    params = LocalFlowStateParams(
        hold_thr=float(settings.overlay_viewer.flow_states.hold_thr),
        burst_thr=float(settings.overlay_viewer.flow_states.burst_thr),
        persistence_window=int(settings.overlay_viewer.flow_states.persistence_window),
        persistent_min_hits=int(settings.overlay_viewer.flow_states.persistent_min_hits),
    )
    try:
        states, flow_summary = compute_local_flow_states(indicators, params=params)
        summary.update(flow_summary)
        summary["states_flow_source"] = "local_event_grammar_v1"
        return states, summary
    except Exception as exc:
        warnings.append(f"Local flow-state compute failed: {exc}")
        return None, summary


def _compute_local_hmm_states(
    ohlcv: pl.DataFrame,
    indicators: pl.DataFrame,
    *,
    enabled: bool,
    k: int,
    scaler_kind: str,
    warnings: list[str],
) -> tuple[pl.DataFrame | None, dict[str, Any]]:
    summary: dict[str, Any] = {
        "enabled": enabled,
        "fitted": False,
        "n_components": k,
        "scaler": scaler_kind,
        "long_bias_states": [],
        "feature_cols": [],
    }
    if not enabled:
        return None, summary
    if GaussianHMM is None:
        warnings.append("hmmlearn is not available; local HMM states skipped.")
        return None, summary

    merged = ohlcv.join(indicators, on=["ticker", "trade_date"], how="left")
    feature_cols = [
        "tmf_raw",
        "tmf_slope_1",
        "tmf_slope_5",
        "tti_raw",
        "tti_slope_1",
        "ret_1",
        "ret_5",
        "range_pct",
        "dollar_vol_20_log",
    ]
    available = [col for col in feature_cols if col in merged.columns]
    summary["feature_cols"] = available
    if len(available) < 3:
        warnings.append("Too few HMM features available; local HMM skipped.")
        return None, summary

    feature_pd = merged.select(["trade_date", "close", *available]).to_pandas()
    X_raw = feature_pd[available].to_numpy(dtype=float)
    mask = np.isfinite(X_raw).all(axis=1)
    valid_n = int(mask.sum())
    min_needed = max(120, int(k * 25))
    summary["fit_rows"] = valid_n
    summary["min_needed_rows"] = min_needed
    if valid_n < min_needed:
        warnings.append(
            f"Local HMM skipped: insufficient finite rows ({valid_n} < {min_needed})."
        )
        return None, summary

    X = X_raw[mask]
    scaler = StandardScaler() if scaler_kind == "standard" else RobustScaler()
    Xs = scaler.fit_transform(X)

    try:
        model = GaussianHMM(
            n_components=int(k),
            covariance_type="diag",
            n_iter=200,
            tol=1e-3,
            random_state=42,
        )
        model.fit(Xs)
        states = model.predict(Xs).astype(np.int32)
        try:
            probs = model.predict_proba(Xs)
            prob_max = probs.max(axis=1).astype(float)
            entropy = (-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)).astype(float)
        except Exception:
            prob_max = np.full(states.shape[0], np.nan, dtype=float)
            entropy = np.full(states.shape[0], np.nan, dtype=float)

        summary["fitted"] = True
        summary["train_loglik"] = _safe_float(model.score(Xs))
        summary["train_loglik_per_obs"] = _safe_float(model.score(Xs) / float(Xs.shape[0]))
    except Exception as exc:
        warnings.append(f"Local HMM fit failed: {exc}")
        return None, summary

    n = merged.height
    state_full = np.full(n, np.nan)
    prob_full = np.full(n, np.nan)
    ent_full = np.full(n, np.nan)
    state_full[mask] = states
    prob_full[mask] = prob_max
    ent_full[mask] = entropy

    prev_arr = np.full(n, np.nan)
    trans_arr = np.full(n, np.nan)
    run_arr = np.full(n, np.nan)
    bsch_arr = np.full(n, np.nan)

    prev_state: int | None = None
    run_len = 0
    bars_since = 0
    for idx in range(n):
        cur_raw = state_full[idx]
        if not np.isfinite(cur_raw):
            prev_state = None
            run_len = 0
            bars_since = 0
            continue
        cur = int(cur_raw)
        prev_arr[idx] = float(prev_state) if prev_state is not None else np.nan
        if prev_state is None:
            trans_arr[idx] = np.nan
            run_len = 1
            bars_since = 0
        else:
            trans_arr[idx] = float(prev_state * 100 + cur)
            if cur == prev_state:
                run_len += 1
                bars_since += 1
            else:
                run_len = 1
                bars_since = 0
        run_arr[idx] = float(run_len)
        bsch_arr[idx] = float(bars_since)
        prev_state = cur

    close = feature_pd["close"].to_numpy(dtype=float)
    fwd_ret_10 = (np.roll(close, -10) / close) - 1.0
    fwd_ret_10[-10:] = np.nan

    by_state: dict[int, list[float]] = {}
    for s_val, f_ret in zip(state_full, fwd_ret_10, strict=False):
        if not np.isfinite(s_val) or not np.isfinite(f_ret):
            continue
        by_state.setdefault(int(s_val), []).append(float(f_ret))

    state_means: list[tuple[int, float]] = []
    for sid, vals in by_state.items():
        if vals:
            state_means.append((sid, float(np.mean(vals))))
    state_means.sort(key=lambda item: item[1], reverse=True)
    long_bias_states = [state_means[0][0]] if state_means else []
    short_bias_states = [state_means[-1][0]] if state_means else []

    freq: dict[str, int] = {}
    for sid in states.tolist():
        key = str(int(sid))
        freq[key] = int(freq.get(key, 0) + 1)

    summary["state_frequency"] = freq
    summary["state_mean_fwd_ret_10"] = {str(k): v for k, v in state_means}
    summary["long_bias_states"] = long_bias_states
    summary["short_bias_states"] = short_bias_states

    out = merged.select("ticker", "trade_date").with_columns(
        pl.Series("hmm_state", state_full).cast(pl.Int32, strict=False),
        pl.Series("hmm_state_prev", prev_arr).cast(pl.Int32, strict=False),
        pl.Series("hmm_transition_code", trans_arr).cast(pl.Int32, strict=False),
        pl.Series("hmm_state_run_length", run_arr).cast(pl.Int32, strict=False),
        pl.Series("bs_hmm_state_change", bsch_arr).cast(pl.Int32, strict=False),
        pl.Series("hmm_state_prob_max", prob_full).cast(pl.Float64, strict=False),
        pl.Series("hmm_state_entropy", ent_full).cast(pl.Float64, strict=False),
        pl.Series("fwd_ret_10_proxy", fwd_ret_10).cast(pl.Float64, strict=False),
    )

    return out, summary


def _compute_exec_overlay(
    ohlcv: pl.DataFrame,
    *,
    settings: AppSettings,
    enabled: bool,
    exec_profile: str,
    exec_min_price: float | None,
    exec_min_dollar_vol20: float | None,
    exec_max_vol_pct: float | None,
    exec_min_history_bars: int | None,
    warnings: list[str],
) -> tuple[pl.DataFrame | None, dict[str, Any]]:
    summary: dict[str, Any] = {"enabled": enabled, "profile": exec_profile}
    if not enabled:
        return None, summary

    frame = ohlcv.select(
        "ticker",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        pl.col("atr_pct_14") if "atr_pct_14" in ohlcv.columns else pl.lit(None).cast(pl.Float64).alias("atr_pct_14"),
    ).with_columns(
        pl.lit(True).alias("signal_eligible"),
        pl.lit(True).alias("overlay_allow_signal"),
        pl.lit("LONG_BIAS").alias("state_direction_hint"),
    )

    try:
        params = resolve_execution_realism_params(
            settings,
            exec_profile=exec_profile if exec_profile in {"none", "lite", "strict"} else "none",  # type: ignore[arg-type]
            exec_min_price=exec_min_price,
            exec_min_dollar_vol20=exec_min_dollar_vol20,
            exec_max_vol_pct=exec_max_vol_pct,
            exec_min_history_bars=exec_min_history_bars,
            report_min_trades=settings.backtest_execution_realism.report_min_trades_default,
            report_max_zero_trade_share=settings.backtest_execution_realism.report_max_zero_trade_share_default,
            report_max_ret_cv=settings.backtest_execution_realism.report_max_ret_cv_default,
        )
        result = apply_execution_realism_filter(frame, params=params)
        summary.update(result.summary)

        out = result.frame.select(
            "ticker",
            "trade_date",
            "execution_profile",
            "execution_filters_enabled",
            "execution_eligible",
            "execution_filter_reason",
            "exec_candidate_before",
            "exec_candidate_after",
            "exec_suppressed_signal",
            "exec_dollar_vol_20",
            "exec_vol_pct",
            "exec_history_bars",
        )
        return out, summary
    except Exception as exc:
        warnings.append(f"Execution realism computation failed: {exc}")
        fallback = frame.select("ticker", "trade_date").with_columns(
            pl.lit(exec_profile).alias("execution_profile"),
            pl.lit(False).alias("execution_filters_enabled"),
            pl.lit(True).alias("execution_eligible"),
            pl.lit("none").alias("execution_filter_reason"),
            pl.lit(False).alias("exec_candidate_before"),
            pl.lit(False).alias("exec_candidate_after"),
            pl.lit(False).alias("exec_suppressed_signal"),
            pl.lit(None).cast(pl.Float64).alias("exec_dollar_vol_20"),
            pl.lit(None).cast(pl.Float64).alias("exec_vol_pct"),
            pl.lit(None).cast(pl.Int64).alias("exec_history_bars"),
        )
        return fallback, summary


def _compute_optional_overlay_policy(
    ohlcv: pl.DataFrame,
    *,
    settings: AppSettings,
    enabled: bool,
    warnings: list[str],
) -> tuple[pl.DataFrame | None, dict[str, Any]]:
    summary: dict[str, Any] = {"enabled": enabled}
    if not enabled:
        return None, summary

    defaults = discover_default_paths(settings)
    cluster_file = defaults.get("cluster_full")
    hardening_dir = defaults.get("cluster_hardening")
    if cluster_file is None or hardening_dir is None:
        warnings.append("Global overlay policy join skipped: clustered dataset or hardening dir not found.")
        return None, summary

    try:
        base = ohlcv.select("ticker", "trade_date", "open", "high", "low", "close", "volume").with_columns(
            pl.lit(True).alias("signal_eligible"),
            pl.lit(True).alias("overlay_allow_signal"),
            pl.lit("LONG_BIAS").alias("state_direction_hint"),
        )
        overlay = apply_policy_overlay(
            base,
            overlay_cluster_file=cluster_file,
            overlay_cluster_hardening_dir=hardening_dir,
            overlay_mode="none",
            join_keys=["ticker", "trade_date"],
            unknown_handling=settings.overlay_coverage_policy.unknown_handling,
            min_match_rate_warn=settings.overlay_coverage_policy.min_match_rate_warn,
            min_match_rate_fail=settings.overlay_coverage_policy.min_match_rate_fail,
            min_year_match_rate_warn=settings.overlay_coverage_policy.min_year_match_rate_warn,
            min_year_match_rate_fail=settings.overlay_coverage_policy.min_year_match_rate_fail,
            unknown_rate_warn=settings.overlay_coverage_policy.unknown_rate_warn,
            unknown_rate_fail=settings.overlay_coverage_policy.unknown_rate_fail,
            coverage_mode="warn_only",
            logger=None,
        )
        summary.update(overlay.join_summary)
        out = overlay.frame.select(
            "ticker",
            "trade_date",
            "overlay_cluster_state",
            "overlay_policy_class",
            "overlay_direction_hint",
            "overlay_tradability_score",
            "overlay_join_status",
            "overlay_allow_signal",
            "overlay_enabled",
        )
        return out, summary
    except Exception as exc:
        warnings.append(f"Global overlay policy join failed: {exc}")
        return None, summary


def _sanity_counts(base: pl.DataFrame, parts: dict[str, pl.DataFrame | None]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    base_n = int(base.height)
    base_dates = int(base.select(pl.col("trade_date").n_unique()).item() or 0)
    out["base_rows"] = base_n
    out["base_unique_dates"] = base_dates
    checks: list[dict[str, Any]] = []
    for name, part in parts.items():
        if part is None:
            continue
        rows = int(part.height)
        uniq = int(part.select(pl.col("trade_date").n_unique()).item() or 0) if "trade_date" in part.columns else None
        checks.append(
            {
                "name": name,
                "rows": rows,
                "unique_dates": uniq,
                "row_count_match_base": rows == base_n,
                "date_count_match_base": (uniq == base_dates) if uniq is not None else None,
            }
        )
    out["parts"] = checks
    return out


def compute_ticker_cache(
    settings: AppSettings,
    params: ComputeTickerParams,
    *,
    progress: ProgressCallback | None = None,
) -> ComputeTickerResult:
    """Compute per-ticker cached overlays and persist artifacts."""

    warnings: list[str] = []
    ticker = params.ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker is empty.")

    source_file = _resolve_source_file(settings, params.source_file)
    source_file_explicit = params.source_file is not None
    date_from = None if params.full_history else (params.date_from.isoformat() if params.date_from else None)
    date_to = None if params.full_history else (params.date_to.isoformat() if params.date_to else None)

    run_spec = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "code_version": COMPUTE_CODE_VERSION,
        "ticker": ticker,
        "source_file": str(source_file),
        "full_history": params.full_history,
        "date_from": date_from,
        "date_to": date_to,
        "compute_flow_states": params.compute_flow_states,
        "compute_local_hmm_states": params.compute_local_hmm_states,
        "compute_exec_realism": params.compute_exec_realism,
        "attempt_global_overlay_join": params.attempt_global_overlay_join,
        "hmm_n_components": int(params.hmm_n_components),
        "hmm_scaler": params.hmm_scaler,
        "exec_profile": params.exec_profile,
        "exec_min_price": params.exec_min_price,
        "exec_min_dollar_vol20": params.exec_min_dollar_vol20,
        "exec_max_vol_pct": params.exec_max_vol_pct,
        "exec_min_history_bars": params.exec_min_history_bars,
        "flow_state_hold_thr": float(settings.overlay_viewer.flow_states.hold_thr),
        "flow_state_burst_thr": float(settings.overlay_viewer.flow_states.burst_thr),
        "flow_state_persistence_window": int(settings.overlay_viewer.flow_states.persistence_window),
        "flow_state_persistent_min_hits": int(settings.overlay_viewer.flow_states.persistent_min_hits),
    }
    run_id = build_cache_run_id(run_spec)
    run_dir = cache_run_dir(settings, ticker, run_id)

    if run_exists(settings, ticker, run_id) and not params.force:
        _emit(progress, 1.0, f"Using cached run {run_id}")
        bundle = load_cache_bundle(run_dir)
        return ComputeTickerResult(
            ticker=ticker,
            run_id=run_id,
            run_dir=run_dir,
            cache_hit=True,
            meta_path=run_dir / "meta.json",
            summary_path=run_dir / "summary.json",
            warnings=[f"cache_hit:{run_id}"],
        )

    if params.force and run_dir.exists():
        shutil.rmtree(run_dir)

    _emit(progress, 0.05, "Loading ticker rows")
    rows, resolved_source_file = _load_rows_with_fallback(
        settings=settings,
        ticker=ticker,
        source_file=source_file,
        date_from=date_from,
        date_to=date_to,
        source_file_explicit=source_file_explicit,
        warnings=warnings,
    )
    if rows.height == 0:
        raise ValueError(f"No rows found for ticker {ticker} in source {source_file}")

    ohlcv = _ensure_ohlcv(rows)

    _emit(progress, 0.20, "Computing TMF/TTI indicators")
    indicators = _compute_indicators(ohlcv)

    _emit(progress, 0.35, "Computing flow states")
    states_flow, flow_summary = _compute_flow_states(
        indicators,
        settings=settings,
        enabled=params.compute_flow_states,
        warnings=warnings,
    )

    _emit(progress, 0.55, "Computing local HMM states")
    states_hmm, hmm_summary = _compute_local_hmm_states(
        ohlcv,
        indicators,
        enabled=params.compute_local_hmm_states,
        k=int(params.hmm_n_components),
        scaler_kind=params.hmm_scaler,
        warnings=warnings,
    )

    _emit(progress, 0.72, "Computing execution realism")
    overlay_exec, exec_summary = _compute_exec_overlay(
        ohlcv,
        settings=settings,
        enabled=params.compute_exec_realism,
        exec_profile=params.exec_profile,
        exec_min_price=params.exec_min_price,
        exec_min_dollar_vol20=params.exec_min_dollar_vol20,
        exec_max_vol_pct=params.exec_max_vol_pct,
        exec_min_history_bars=params.exec_min_history_bars,
        warnings=warnings,
    )

    _emit(progress, 0.84, "Joining optional global overlay policy")
    overlay_policy, overlay_policy_summary = _compute_optional_overlay_policy(
        ohlcv,
        settings=settings,
        enabled=params.attempt_global_overlay_join,
        warnings=warnings,
    )

    sanity = _sanity_counts(
        ohlcv,
        {
            "indicators": indicators,
            "states_flow": states_flow,
            "states_hmm": states_hmm,
            "overlay_exec": overlay_exec,
            "overlay_policy": overlay_policy,
        },
    )

    date_min = ohlcv.select(pl.col("trade_date").min()).item()
    date_max = ohlcv.select(pl.col("trade_date").max()).item()

    summary_payload: dict[str, Any] = {
        "ticker": ticker,
        "run_id": run_id,
        "row_count": int(ohlcv.height),
        "resolved_source_file": str(resolved_source_file),
        "date_min": date_min.isoformat() if date_min is not None else None,
        "date_max": date_max.isoformat() if date_max is not None else None,
        "hmm_state_count": int(states_hmm.select(pl.col("hmm_state").n_unique()).item() or 0) if states_hmm is not None and "hmm_state" in states_hmm.columns else 0,
        "long_bias_states": hmm_summary.get("long_bias_states", []),
        "states_flow_source": flow_summary.get("states_flow_source"),
        "flow_state_source": flow_summary.get("states_flow_source"),
        "flow_state_ready_ratio": flow_summary.get("ready_ratio"),
        "flow_state_counts": flow_summary.get("state_counts", {}),
        "exec_eligibility_rate": _safe_float(exec_summary.get("eligibility_rate")),
        "overlay_match_rate": _safe_float(overlay_policy_summary.get("match_rate")),
        "warnings": warnings,
    }

    meta_payload: dict[str, Any] = {
        "ticker": ticker,
        "run_id": run_id,
        "source_file": str(source_file),
        "resolved_source_file": str(resolved_source_file),
        "states_flow_source": flow_summary.get("states_flow_source"),
        "params": run_spec,
        "row_count": int(ohlcv.height),
        "date_min": date_min.isoformat() if date_min is not None else None,
        "date_max": date_max.isoformat() if date_max is not None else None,
        "warnings": warnings,
        "flow_states": flow_summary,
        "hmm": hmm_summary,
        "execution": exec_summary,
        "overlay_policy_join": overlay_policy_summary,
        "sanity": sanity,
    }

    _emit(progress, 0.94, "Writing cache artifacts")
    write_cache_run(
        run_dir=run_dir,
        ohlcv=ohlcv,
        indicators=indicators,
        states_flow=states_flow,
        states_hmm=states_hmm,
        overlay_exec=overlay_exec,
        overlay_policy=overlay_policy,
        meta=meta_payload,
        summary=summary_payload,
    )

    _emit(progress, 1.0, f"Done: {run_id}")

    LOGGER.info(
        "overlay_viewer.compute_ticker.complete ticker=%s run_id=%s rows=%s dir=%s",
        ticker,
        run_id,
        ohlcv.height,
        run_dir,
    )

    return ComputeTickerResult(
        ticker=ticker,
        run_id=run_id,
        run_dir=run_dir,
        cache_hit=False,
        meta_path=run_dir / "meta.json",
        summary_path=run_dir / "summary.json",
        warnings=warnings,
    )
