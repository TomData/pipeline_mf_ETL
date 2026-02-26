"""Streamlit app: Beta Expert Advisor Overlay Viewer v1.1.2."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Any

# Ensure local src/ is importable when run via `streamlit run .../app.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import polars as pl
import streamlit as st

from mf_etl.apps.overlay_viewer.compute_ticker import (
    ComputeTickerParams,
    compute_ticker_cache,
)
from mf_etl.apps.overlay_viewer.data_loader import (
    candidate_names_from_packet,
    discover_default_paths,
    latest_cached_run_for_ticker,
    list_cached_runs_for_ticker,
    list_cached_tickers_for_viewer,
    load_cached_bundle_for_viewer,
    load_pcp_packet,
    load_rows_for_ticker,
    load_symbol_master_tickers,
    normalize_ticker_input,
    resolve_latest_pcp_dir,
    source_debug_summary,
)
from mf_etl.apps.overlay_viewer.indicators_twiggs import (
    TwiggsParams,
    compute_reading_labels,
    compute_tmf_tti,
)
from mf_etl.apps.overlay_viewer.hmm_display import (
    HMMDisplayConfig,
    apply_hmm_display_transform,
)
from mf_etl.apps.overlay_viewer.overlay_logic import (
    ViewerRuntimeConfig,
    build_overlay_view,
    runtime_from_candidate,
)
from mf_etl.apps.overlay_viewer.plotting import PlotToggles, build_overlay_figure
from mf_etl.apps.overlay_viewer.utils import as_path_or_none
from mf_etl.config import AppSettings, load_settings


@st.cache_resource(show_spinner=False)
def _settings() -> AppSettings:
    return load_settings()


@st.cache_data(show_spinner=False)
def _symbol_master_tickers() -> list[str]:
    return load_symbol_master_tickers(_settings())


@st.cache_data(show_spinner=False)
def _default_paths() -> dict[str, str | None]:
    defaults = discover_default_paths(_settings())
    return {k: (str(v) if v is not None else None) for k, v in defaults.items()}


@st.cache_data(show_spinner=False)
def _latest_pcp_path() -> str | None:
    path = resolve_latest_pcp_dir(_settings())
    return str(path) if path is not None else None


@st.cache_data(show_spinner=False)
def _load_packet(path: str) -> dict[str, Any]:
    return load_pcp_packet(Path(path))


@st.cache_data(show_spinner=False)
def _load_rows(path: str, ticker: str) -> pl.DataFrame:
    return load_rows_for_ticker(Path(path), ticker=ticker, date_from=None, date_to=None)


@st.cache_data(show_spinner=False)
def _load_source_debug(path: str) -> dict[str, Any]:
    return source_debug_summary(Path(path))


def _filter_by_date(frame: pl.DataFrame, date_from: date, date_to: date) -> pl.DataFrame:
    return frame.filter(
        (pl.col("trade_date") >= pl.lit(date_from))
        & (pl.col("trade_date") <= pl.lit(date_to))
    )


def _bounds(frame: pl.DataFrame) -> tuple[date, date] | None:
    if "trade_date" not in frame.columns or frame.height == 0:
        return None
    min_dt = frame.select(pl.col("trade_date").min()).item()
    max_dt = frame.select(pl.col("trade_date").max()).item()
    if min_dt is None or max_dt is None:
        return None
    return min_dt, max_dt


def _manual_runtime(
    *,
    input_type: str,
    signal_mode: str,
    validation_run_dir: str,
    cluster_hardening_dir: str,
    overlay_cluster_file: str,
    overlay_cluster_hardening_dir: str,
    overlay_mode: str,
    execution_profile: str,
    exec_override: bool,
    exec_min_price: float,
    exec_min_dollar_vol20: float,
    exec_max_vol_pct: float,
    exec_min_history_bars: int,
) -> ViewerRuntimeConfig:
    return ViewerRuntimeConfig(
        input_type=input_type,  # type: ignore[arg-type]
        signal_mode=signal_mode,  # type: ignore[arg-type]
        validation_run_dir=as_path_or_none(validation_run_dir),
        cluster_hardening_dir=as_path_or_none(cluster_hardening_dir),
        state_map_file=None,
        include_watch=False,
        policy_filter_mode="allow_only",
        include_state_ids=[],
        allow_unconfirmed=(input_type == "flow"),
        overlay_cluster_file=as_path_or_none(overlay_cluster_file),
        overlay_cluster_hardening_dir=as_path_or_none(overlay_cluster_hardening_dir),
        overlay_mode=overlay_mode,  # type: ignore[arg-type]
        overlay_join_keys=["ticker", "trade_date"],
        overlay_coverage_mode="warn_only",
        overlay_coverage_bypass=True,
        execution_profile=execution_profile,
        exec_min_price=exec_min_price if exec_override else None,
        exec_min_dollar_vol20=exec_min_dollar_vol20 if exec_override else None,
        exec_max_vol_pct=exec_max_vol_pct if exec_override else None,
        exec_min_history_bars=exec_min_history_bars if exec_override else None,
    )


def _render_source_debug(debug: dict[str, Any]) -> None:
    if (
        str(debug.get("mode")) == "GLOBAL"
        and isinstance(debug.get("unique_tickers"), int)
        and int(debug["unique_tickers"]) <= 20
    ):
        st.sidebar.warning(
            f"ML dataset tickers: {debug['unique_tickers']} (sample). Bronze fallback is expected."
        )
    with st.sidebar.expander("Source Debug", expanded=False):
        st.json(debug)


def main() -> None:
    st.set_page_config(
        page_title="Beta Expert Advisor Overlay Viewer v1.1.2",
        layout="wide",
    )
    settings = _settings()

    st.title("Beta Expert Advisor Overlay Viewer v1.1.2")
    st.caption("Visualization-only beta expert advisor overlay. No order execution.")

    defaults = _default_paths()
    latest_pcp = _latest_pcp_path()
    symbol_universe = _symbol_master_tickers()

    data_source_mode = st.sidebar.selectbox(
        "Data Source",
        options=["CACHED (Compute Ticker)", "GLOBAL ARTIFACTS (Latest)"],
        index=0,
    )
    strategy_mode = st.sidebar.radio("Source mode", options=["PCP Candidate", "Manual"], index=0)

    runtime: ViewerRuntimeConfig | None = None
    primary_file_default = defaults.get("hmm_decoded") or defaults.get("ml_dataset") or ""
    primary_file = ""
    source_debug: dict[str, Any] = {}

    # Strategy/runtime controls.
    if strategy_mode == "PCP Candidate":
        pcp_dir_str = st.sidebar.text_input("PCP pack dir", value=latest_pcp or "")
        if not pcp_dir_str:
            st.error("No PCP pack path set. Switch to Manual mode or provide PCP pack dir.")
            st.stop()
        try:
            packet = _load_packet(pcp_dir_str)
        except Exception as exc:
            st.error(f"Failed to load PCP packet: {exc}")
            st.stop()

        names = candidate_names_from_packet(packet)
        if not names:
            st.error("No candidates found in PCP packet.")
            st.stop()
        candidate_name = st.sidebar.selectbox("Candidate", options=names)
        runtime = runtime_from_candidate(packet, candidate_name)

        candidate_obj = packet.get("candidates", {}).get(candidate_name, {})
        primary_file = str(candidate_obj.get("input_file") or "")

        with st.sidebar.expander("Locked config", expanded=False):
            st.json(
                {
                    "input_type": runtime.input_type,
                    "signal_mode": runtime.signal_mode,
                    "overlay_mode": runtime.overlay_mode,
                    "execution_profile": runtime.execution_profile,
                    "validation_run_dir": str(runtime.validation_run_dir) if runtime.validation_run_dir else None,
                }
            )
    else:
        input_type = st.sidebar.selectbox("Primary source type", options=["hmm", "flow", "cluster"], index=0)
        if input_type == "hmm":
            primary_file_default = defaults.get("hmm_decoded") or primary_file_default
        elif input_type == "cluster":
            primary_file_default = defaults.get("cluster_full") or primary_file_default
        else:
            primary_file_default = defaults.get("ml_dataset") or primary_file_default

        primary_file = st.sidebar.text_input("Primary file", value=primary_file_default or "")
        validation_run_dir = st.sidebar.text_input("HMM validation dir (optional)", value="")
        cluster_hardening_dir = st.sidebar.text_input(
            "Cluster hardening dir (cluster mode)",
            value=defaults.get("cluster_hardening") or "",
        )
        overlay_mode = st.sidebar.selectbox(
            "Overlay mode",
            options=["none", "allow_only", "allow_watch", "block_veto", "allow_or_unknown"],
            index=0,
        )
        overlay_cluster_file = st.sidebar.text_input(
            "Overlay cluster file",
            value=defaults.get("cluster_full") or "",
        )
        overlay_cluster_hardening_dir = st.sidebar.text_input(
            "Overlay hardening dir",
            value=defaults.get("cluster_hardening") or "",
        )
        execution_profile = st.sidebar.selectbox("Execution profile", options=["none", "lite", "strict"], index=1)
        signal_mode = st.sidebar.selectbox(
            "Signal mode",
            options=["state_entry", "state_transition_entry", "state_persistence_confirm"],
            index=1,
        )
        exec_override = st.sidebar.checkbox("Override execution thresholds", value=False)
        exec_min_price = st.sidebar.number_input("exec_min_price", min_value=0.0, value=2.0, step=0.5)
        exec_min_dollar_vol20 = st.sidebar.number_input(
            "exec_min_dollar_vol20", min_value=0.0, value=1_000_000.0, step=250_000.0
        )
        exec_max_vol_pct = st.sidebar.number_input("exec_max_vol_pct", min_value=0.0, value=0.12, step=0.01)
        exec_min_history_bars = st.sidebar.number_input("exec_min_history_bars", min_value=1, value=50, step=1)

        runtime = _manual_runtime(
            input_type=input_type,
            signal_mode=signal_mode,
            validation_run_dir=validation_run_dir,
            cluster_hardening_dir=cluster_hardening_dir,
            overlay_cluster_file=overlay_cluster_file,
            overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,
            overlay_mode=overlay_mode,
            execution_profile=execution_profile,
            exec_override=exec_override,
            exec_min_price=float(exec_min_price),
            exec_min_dollar_vol20=float(exec_min_dollar_vol20),
            exec_max_vol_pct=float(exec_max_vol_pct),
            exec_min_history_bars=int(exec_min_history_bars),
        )

    if runtime is None:
        st.error("Runtime configuration could not be resolved.")
        st.stop()

    rows_full: pl.DataFrame | None = None
    cache_meta: dict[str, Any] | None = None
    cache_meta_path: Path | None = None
    ticker = ""

    if data_source_mode == "GLOBAL ARTIFACTS (Latest)":
        if not primary_file:
            st.error("Primary input file is empty in GLOBAL mode.")
            st.stop()

        default_ticker = "AAPL.US" if "AAPL.US" in symbol_universe else (symbol_universe[0] if symbol_universe else "AAPL.US")
        typed = st.sidebar.text_input("Ticker", value=default_ticker)
        ticker, norm_warnings = normalize_ticker_input(typed, symbol_universe)
        for msg in norm_warnings:
            st.sidebar.info(msg)

        try:
            rows_full = _load_rows(primary_file, ticker)
        except Exception as exc:
            st.error(f"Failed to load rows: {exc}")
            st.stop()

        source_debug = _load_source_debug(primary_file)
        source_debug["mode"] = "GLOBAL"

    else:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Compute Ticker")

        compute_source_default = defaults.get("ml_dataset") or primary_file or ""
        compute_source_file_str = st.sidebar.text_input("Compute source file", value=compute_source_default)

        compute_ticker_raw = st.sidebar.text_input("Ticker to compute", value="AAPL.US")
        normalized_compute_ticker, norm_warnings = normalize_ticker_input(compute_ticker_raw, symbol_universe)
        for msg in norm_warnings:
            st.sidebar.info(msg)

        full_history = st.sidebar.toggle("Full history", value=True)
        custom_from = None
        custom_to = None
        if not full_history:
            today = date.today()
            custom_from = st.sidebar.date_input("Compute from", value=today - timedelta(days=365 * 5))
            custom_to = st.sidebar.date_input("Compute to", value=today)

        compute_flow = st.sidebar.toggle("Compute Flow states", value=True)
        compute_hmm = st.sidebar.toggle("Compute Local HMM states", value=True)
        compute_exec = st.sidebar.toggle("Compute exec realism", value=True)
        compute_overlay = st.sidebar.toggle("Attempt global overlay policy join", value=False)

        hmm_k = int(st.sidebar.number_input("HMM n_components", min_value=2, value=5, step=1))
        hmm_scaler = st.sidebar.selectbox("HMM scaling", options=["per_ticker", "global"], index=0)
        # Local compute uses one ticker; both map to robust/standard options for the scaler implementation.
        hmm_scaler_impl = "robust" if hmm_scaler == "per_ticker" else "standard"

        exec_profile_compute = st.sidebar.selectbox("Compute exec profile", options=["none", "lite", "strict"], index=1)
        exec_override_compute = st.sidebar.checkbox("Override compute exec thresholds", value=False)
        exec_min_price_compute = st.sidebar.number_input("compute_exec_min_price", min_value=0.0, value=2.0, step=0.5)
        exec_min_dv_compute = st.sidebar.number_input(
            "compute_exec_min_dollar_vol20", min_value=0.0, value=1_000_000.0, step=250_000.0
        )
        exec_max_vol_compute = st.sidebar.number_input("compute_exec_max_vol_pct", min_value=0.0, value=0.12, step=0.01)
        exec_min_hist_compute = int(st.sidebar.number_input("compute_exec_min_history_bars", min_value=1, value=50, step=1))

        force_compute = st.sidebar.checkbox("Force recompute", value=False)
        if st.sidebar.button("Compute ticker now", type="primary"):
            if not compute_source_file_str:
                st.sidebar.error("Compute source file is required.")
            else:
                progress_box = st.sidebar.progress(0.0)
                status_box = st.sidebar.empty()

                def _progress_cb(frac: float, text: str) -> None:
                    progress_box.progress(frac)
                    status_box.info(text)

                try:
                    result = compute_ticker_cache(
                        settings,
                        ComputeTickerParams(
                            ticker=normalized_compute_ticker,
                            source_file=Path(compute_source_file_str),
                            full_history=full_history,
                            date_from=custom_from,
                            date_to=custom_to,
                            compute_flow_states=compute_flow,
                            compute_local_hmm_states=compute_hmm,
                            compute_exec_realism=compute_exec,
                            attempt_global_overlay_join=compute_overlay,
                            hmm_n_components=hmm_k,
                            hmm_scaler=hmm_scaler_impl,
                            exec_profile=exec_profile_compute,
                            exec_min_price=float(exec_min_price_compute) if exec_override_compute else None,
                            exec_min_dollar_vol20=float(exec_min_dv_compute) if exec_override_compute else None,
                            exec_max_vol_pct=float(exec_max_vol_compute) if exec_override_compute else None,
                            exec_min_history_bars=int(exec_min_hist_compute) if exec_override_compute else None,
                            force=force_compute,
                        ),
                        progress=_progress_cb,
                    )
                    st.session_state["cached_selected_ticker"] = result.ticker
                    st.session_state["cached_selected_run_id"] = result.run_id
                    status_box.success(f"Compute complete: {result.run_id}")
                    st.sidebar.success(f"Cached run ready: {result.run_dir}")
                    if result.warnings:
                        for warning in result.warnings:
                            st.sidebar.warning(warning)
                except Exception as exc:
                    status_box.error(f"Compute failed: {exc}")

        cached_tickers = list_cached_tickers_for_viewer(settings)
        if not cached_tickers:
            st.warning("No cached tickers found. Use 'Compute ticker now' to create cache artifacts.")
            st.stop()

        default_cached_ticker = st.session_state.get("cached_selected_ticker")
        if not isinstance(default_cached_ticker, str) or default_cached_ticker not in cached_tickers:
            latest = latest_cached_run_for_ticker(settings, normalized_compute_ticker)
            default_cached_ticker = latest.ticker if latest is not None and latest.ticker in cached_tickers else cached_tickers[0]

        ticker = st.sidebar.selectbox(
            "Cached ticker",
            options=cached_tickers,
            index=cached_tickers.index(default_cached_ticker),
        )

        runs = list_cached_runs_for_ticker(settings, ticker)
        if not runs:
            st.warning(f"No cache runs found for {ticker}. Compute ticker first.")
            st.stop()

        run_labels: list[str] = []
        run_by_label: dict[str, Any] = {}
        for info in runs:
            label = f"{info.run_id} | ts={info.computed_ts or 'n/a'} | rows={info.row_count or 'n/a'}"
            run_labels.append(label)
            run_by_label[label] = info

        default_run_id = st.session_state.get("cached_selected_run_id")
        default_idx = 0
        if isinstance(default_run_id, str):
            for i, info in enumerate(runs):
                if info.run_id == default_run_id:
                    default_idx = i
                    break

        selected_label = st.sidebar.selectbox("Cache run", options=run_labels, index=default_idx)
        selected_run = run_by_label[selected_label]

        bundle = load_cached_bundle_for_viewer(selected_run.run_dir)
        rows_full = bundle.merged
        cache_meta = bundle.meta
        cache_meta_path = selected_run.run_dir / "meta.json"

        hmm_meta = bundle.meta.get("hmm", {}) if isinstance(bundle.meta.get("hmm"), dict) else {}
        long_bias_states = tuple(int(v) for v in hmm_meta.get("long_bias_states", []) if v is not None)
        short_bias_states = tuple(int(v) for v in hmm_meta.get("short_bias_states", []) if v is not None)
        runtime = replace(
            runtime,
            hmm_long_bias_states=long_bias_states,
            hmm_short_bias_states=short_bias_states,
        )

        source_debug = {
            "mode": "CACHED",
            "cache_run_dir": str(bundle.run_dir),
            "cache_run_id": bundle.run_id,
            "ticker": bundle.ticker,
            "rows": int(bundle.merged.height),
            "columns": bundle.merged.columns,
            "cached_ticker_count": len(cached_tickers),
            "sample_tickers": cached_tickers[:20],
            "meta": {
                "source_file": bundle.meta.get("source_file"),
                "resolved_source_file": bundle.meta.get("resolved_source_file"),
                "date_min": bundle.meta.get("date_min"),
                "date_max": bundle.meta.get("date_max"),
                "hmm_fitted": (bundle.meta.get("hmm", {}) or {}).get("fitted"),
                "states_flow_source": bundle.meta.get("states_flow_source")
                or (bundle.meta.get("flow_states", {}) or {}).get("states_flow_source"),
            },
        }

    _render_source_debug(source_debug)

    if rows_full is None or rows_full.height == 0:
        st.warning(f"No rows available for ticker {ticker}.")
        st.stop()

    bounds = _bounds(rows_full)
    if bounds is None:
        st.error("trade_date range is unavailable in selected data.")
        st.stop()
    min_dt, max_dt = bounds

    default_from = max(min_dt, max_dt - timedelta(days=365))
    from_col, to_col = st.sidebar.columns(2)
    date_from = from_col.date_input("From", value=default_from, min_value=min_dt, max_value=max_dt)
    date_to = to_col.date_input("To", value=max_dt, min_value=min_dt, max_value=max_dt)
    if date_from > date_to:
        st.error("Date range invalid: From > To")
        st.stop()

    rows_view = _filter_by_date(rows_full, date_from, date_to)
    if rows_view.height == 0:
        st.warning("No rows after date filter.")
        st.stop()

    try:
        overlay_result = build_overlay_view(rows_view, runtime=runtime, settings=settings)
    except Exception as exc:
        st.error(f"Failed to build overlay layers: {exc}")
        st.stop()

    if overlay_result.warnings:
        for warning in overlay_result.warnings:
            st.warning(warning)

    show_volume = st.sidebar.toggle("Show volume", value=True)
    volume_mode = st.sidebar.selectbox("Volume mode", options=["separate", "overlay"], index=0)
    show_tmf_tti = st.sidebar.toggle("Show TMF + TTI panel", value=True)
    show_flow_band = st.sidebar.toggle("Show Flow state band", value=True)
    show_hmm_band = st.sidebar.toggle("Show HMM state band", value=True)
    hmm_display_mode = st.sidebar.selectbox(
        "HMM Display Mode",
        options=["RAW", "SMOOTHED", "GROUPED", "SMOOTHED+GROUPED"],
        index=0,
    )
    smoothing_method = "mode"
    smoothing_window = 5
    grouping_scheme = "LONG_NEUTRAL_SHORT"
    long_states_top_k = 1
    short_states_bottom_k = 1
    persist_mapping_in_cache_meta = True
    if hmm_display_mode in {"SMOOTHED", "SMOOTHED+GROUPED"}:
        smoothing_method = st.sidebar.selectbox(
            "smoothing_method",
            options=["mode", "median"],
            index=0,
        )
        smoothing_window = int(
            st.sidebar.number_input("smoothing_window", min_value=3, max_value=21, value=5, step=2)
        )
    if hmm_display_mode in {"GROUPED", "SMOOTHED+GROUPED"}:
        grouping_scheme = st.sidebar.selectbox(
            "grouping_scheme",
            options=["LONG_NEUTRAL_SHORT", "LONG_OTHER"],
            index=0,
        )
        long_states_top_k = int(
            st.sidebar.number_input("long_states_top_k", min_value=1, max_value=8, value=1, step=1)
        )
        if grouping_scheme == "LONG_NEUTRAL_SHORT":
            short_states_bottom_k = int(
                st.sidebar.number_input("short_states_bottom_k", min_value=1, max_value=8, value=1, step=1)
            )
        persist_mapping_in_cache_meta = st.sidebar.checkbox(
            "Persist mapping in cache meta",
            value=True,
            disabled=(data_source_mode != "CACHED (Compute Ticker)"),
        )
    show_overlay_band = st.sidebar.toggle("Show overlay class ALLOW/WATCH/BLOCK band", value=True)
    show_exec_markers = st.sidebar.toggle("Show exec realism pass/fail markers", value=True)
    show_candidate_markers = st.sidebar.toggle("Show candidate/blocked markers", value=True)

    with st.sidebar.expander("TMF / TTI settings", expanded=False):
        tmf_period = st.number_input("TMF period", min_value=2, value=21, step=1)
        tmf_scale_pct = st.toggle("Scale TMF/TTI to %", value=False)
        tti_scale = st.number_input("TTI multiplier", min_value=0.1, value=1.0, step=0.1)
        reading_mode = st.toggle("Reading mode labels", value=False)
        pivot_left = st.number_input("pivotL", min_value=1, value=2, step=1)
        pivot_right = st.number_input("pivotR", min_value=1, value=2, step=1)
        zero_tol = st.number_input("Zero tolerance", min_value=0.0, value=0.01, step=0.001, format="%.3f")

    with st.sidebar.expander("Legend / help", expanded=False):
        st.markdown("- TMF is a sensor, not a strategy.")
        st.markdown("- `ALLOW` means cluster hardening considers state tradable under current policy.")
        st.markdown("- Execution realism gate marks bars that fail price/liquidity/vol/warmup constraints.")

    hmm_display_result = apply_hmm_display_transform(
        overlay_result.frame,
        config=HMMDisplayConfig(
            display_mode=hmm_display_mode,  # type: ignore[arg-type]
            smoothing_method=smoothing_method,  # type: ignore[arg-type]
            smoothing_window=int(smoothing_window),
            grouping_scheme=grouping_scheme,  # type: ignore[arg-type]
            long_states_top_k=int(long_states_top_k),
            short_states_bottom_k=int(short_states_bottom_k),
            persist_mapping_in_cache_meta=bool(persist_mapping_in_cache_meta),
        ),
        base_long_states=runtime.hmm_long_bias_states,
        base_short_states=runtime.hmm_short_bias_states,
        cached_meta=cache_meta,
        cache_meta_path=cache_meta_path if data_source_mode == "CACHED (Compute Ticker)" else None,
    )
    if hmm_display_result.warnings:
        for warning in hmm_display_result.warnings:
            st.warning(warning)

    plot_df = hmm_display_result.frame.to_pandas()
    plot_df["trade_date"] = pd.to_datetime(plot_df["trade_date"])

    reading_labels = None
    if show_tmf_tti:
        if "volume" not in plot_df.columns or plot_df["volume"].isna().all():
            st.warning("Volume missing; TMF/TTI panel disabled for this view.")
            show_tmf_tti = False
        else:
            twiggs = compute_tmf_tti(
                plot_df,
                TwiggsParams(
                    period=int(tmf_period),
                    scale_pct=bool(tmf_scale_pct),
                    tti_scale=float(tti_scale),
                ),
            )
            plot_df = twiggs
            if reading_mode:
                reading_labels = compute_reading_labels(
                    plot_df,
                    value_col="tmf",
                    pivot_left=int(pivot_left),
                    pivot_right=int(pivot_right),
                    zero_tolerance=float(zero_tol),
                )

    fig = build_overlay_figure(
        plot_df,
        toggles=PlotToggles(
            show_volume=show_volume,
            volume_mode=volume_mode,
            show_tmf_tti=show_tmf_tti,
            show_flow_band=show_flow_band,
            show_hmm_band=show_hmm_band,
            show_overlay_band=show_overlay_band,
            show_exec_markers=show_exec_markers,
            show_candidate_markers=show_candidate_markers,
            reading_mode=reading_mode,
            hmm_display_mode=hmm_display_mode,
        ),
        scale_pct=tmf_scale_pct,
        tti_level_signal=float(plot_df.get("tti_level_signal", pd.Series([0.20])).iloc[0]),
        tti_level_entry=float(plot_df.get("tti_level_entry", pd.Series([0.30])).iloc[0]),
        reading_labels=reading_labels,
        hmm_band_col=hmm_display_result.band_col,
        hmm_display_summary=hmm_display_result.summary,
    )

    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(plot_df):,}")
    c2.metric("Signals", str(int(plot_df.get("entry_signal", pd.Series(dtype=bool)).sum())))
    c3.metric(
        "Overlay match rate",
        f"{overlay_result.overlay_summary.get('match_rate')}" if overlay_result.overlay_summary else "n/a",
    )
    c4.metric(
        "Exec eligibility",
        f"{overlay_result.execution_summary.get('eligibility_rate')}" if overlay_result.execution_summary else "n/a",
    )
    flow_source = (
        source_debug.get("meta", {}).get("states_flow_source")
        if isinstance(source_debug.get("meta"), dict)
        else None
    )
    if flow_source:
        st.caption(f"Flow State Source: {flow_source}")
    if hmm_display_result.summary:
        st.caption(
            f"HMM Display: {hmm_display_result.summary.get('display_mode')} "
            f"(band={hmm_display_result.summary.get('band_col')})"
        )

    with st.expander("Diagnostics", expanded=False):
        st.subheader("Mapping summary")
        st.json(overlay_result.mapping_summary)
        st.subheader("Overlay summary")
        st.json(overlay_result.overlay_summary)
        st.subheader("Execution summary")
        st.json(overlay_result.execution_summary)
        st.subheader("Signal diagnostics")
        st.json(overlay_result.signal_diagnostics)
        st.subheader("HMM display summary")
        st.json(hmm_display_result.summary)

    with st.expander("Data preview", expanded=False):
        cols = [
            col
            for col in [
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "flow_state_code",
                "hmm_state",
                "hmm_state_smoothed",
                "hmm_group_code",
                "hmm_group_label",
                "overlay_policy_class",
                "execution_eligible",
                "execution_filter_reason",
                "entry_signal",
                "entry_side",
                "candidate_marker_type",
                "blocked_reason",
                "tmf",
                "tti",
            ]
            if col in plot_df.columns
        ]
        st.dataframe(plot_df[cols].tail(300), use_container_width=True)


if __name__ == "__main__":
    main()
