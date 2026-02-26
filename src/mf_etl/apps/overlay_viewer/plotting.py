"""Plotly chart builders for overlay viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True, slots=True)
class PlotToggles:
    show_volume: bool
    volume_mode: str
    show_tmf_tti: bool
    show_flow_band: bool
    show_hmm_band: bool
    show_overlay_band: bool
    show_exec_markers: bool
    show_candidate_markers: bool
    reading_mode: bool
    hmm_display_mode: str = "RAW"


_FLOW_COLORS = {
    0: "rgba(128,128,128,0.12)",
    1: "rgba(0,180,120,0.12)",
    2: "rgba(0,120,0,0.12)",
    3: "rgba(220,100,0,0.12)",
    4: "rgba(180,0,0,0.12)",
}

_OVERLAY_COLORS = {
    "ALLOW": "rgba(0,180,0,0.10)",
    "WATCH": "rgba(235,180,0,0.10)",
    "BLOCK": "rgba(200,0,0,0.14)",
    "UNKNOWN": "rgba(120,120,120,0.10)",
}

_HMM_COLORS = [
    "rgba(0,114,178,0.10)",
    "rgba(213,94,0,0.10)",
    "rgba(0,158,115,0.10)",
    "rgba(204,121,167,0.10)",
    "rgba(240,228,66,0.10)",
    "rgba(86,180,233,0.10)",
    "rgba(230,159,0,0.10)",
    "rgba(0,0,0,0.08)",
]

_HMM_GROUP_COLORS = {
    -1: "rgba(185,28,28,0.15)",  # SHORT
    0: "rgba(71,85,105,0.12)",   # NEUTRAL / OTHER
    1: "rgba(22,163,74,0.15)",   # LONG
}


def _segment_ranges(series: pd.Series) -> list[tuple[int, int, Any]]:
    if series.empty:
        return []
    values = series.to_list()
    out: list[tuple[int, int, Any]] = []
    start = 0
    cur = values[0]
    for idx in range(1, len(values)):
        val = values[idx]
        if val != cur:
            out.append((start, idx - 1, cur))
            start = idx
            cur = val
    out.append((start, len(values) - 1, cur))
    return out


def _add_band(
    fig: go.Figure,
    *,
    df: pd.DataFrame,
    value_col: str,
    colors: dict[Any, str],
    row: int = 1,
    col: int = 1,
) -> None:
    if value_col not in df.columns:
        return
    series = df[value_col]
    segments = _segment_ranges(series)
    for start, end, value in segments:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        color = colors.get(value)
        if color is None:
            continue
        x0 = df.iloc[start]["trade_date"]
        x1 = df.iloc[end]["trade_date"]
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color,
            opacity=1.0,
            line_width=0,
            row=row,
            col=col,
            layer="below",
        )


def build_overlay_figure(
    df: pd.DataFrame,
    *,
    toggles: PlotToggles,
    scale_pct: bool,
    tti_level_signal: float,
    tti_level_entry: float,
    reading_labels: pd.DataFrame | None,
    hmm_band_col: str = "hmm_state",
    hmm_display_summary: dict[str, Any] | None = None,
) -> go.Figure:
    """Create stacked candlestick + volume + indicator chart."""

    if df.empty:
        return go.Figure()

    rows = 1
    row_heights: list[float] = [0.62]
    if toggles.show_volume and toggles.volume_mode == "separate":
        rows += 1
        row_heights.append(0.18)
    if toggles.show_tmf_tti:
        rows += 1
        row_heights.append(0.20)

    specs: list[list[dict[str, bool]]] = []
    specs.append([{"secondary_y": toggles.show_volume and toggles.volume_mode == "overlay"}])
    if toggles.show_volume and toggles.volume_mode == "separate":
        specs.append([{"secondary_y": False}])
    if toggles.show_tmf_tti:
        specs.append([{"secondary_y": False}])

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
        specs=specs,
    )

    fig.add_trace(
        go.Candlestick(
            x=df["trade_date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#00b26f",
            decreasing_line_color="#d84a4a",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    if toggles.show_flow_band and "flow_state_code" in df.columns:
        _add_band(fig, df=df, value_col="flow_state_code", colors=_FLOW_COLORS, row=1, col=1)

    if toggles.show_hmm_band and hmm_band_col in df.columns:
        if hmm_band_col.startswith("hmm_group"):
            _add_band(fig, df=df, value_col=hmm_band_col, colors=_HMM_GROUP_COLORS, row=1, col=1)
        else:
            hmm_colors: dict[Any, str] = {}
            raw_vals = df[hmm_band_col].dropna().tolist()
            for state in sorted({int(v) for v in raw_vals if pd.notna(v)}):
                hmm_colors[state] = _HMM_COLORS[state % len(_HMM_COLORS)]
            _add_band(fig, df=df, value_col=hmm_band_col, colors=hmm_colors, row=1, col=1)

    if toggles.show_overlay_band and "overlay_policy_class" in df.columns:
        _add_band(fig, df=df, value_col="overlay_policy_class", colors=_OVERLAY_COLORS, row=1, col=1)

    volume_row = None
    if toggles.show_volume:
        if toggles.volume_mode == "overlay":
            fig.add_trace(
                go.Bar(
                    x=df["trade_date"],
                    y=df.get("volume"),
                    marker_color="rgba(90,90,180,0.35)",
                    name="Volume",
                    showlegend=True,
                ),
                row=1,
                col=1,
                secondary_y=True,
            )
        else:
            volume_row = 2
            fig.add_trace(
                go.Bar(
                    x=df["trade_date"],
                    y=df.get("volume"),
                    marker_color="rgba(90,90,180,0.50)",
                    name="Volume",
                    showlegend=True,
                ),
                row=volume_row,
                col=1,
            )

    indicator_row = None
    if toggles.show_tmf_tti:
        indicator_row = rows
        fig.add_trace(
            go.Scatter(
                x=df["trade_date"],
                y=df.get("tmf"),
                mode="lines",
                line={"color": "#0d6efd", "width": 1.8},
                name="TMF",
            ),
            row=indicator_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["trade_date"],
                y=df.get("tti"),
                mode="lines",
                line={"color": "#ff7f0e", "width": 1.6},
                name="TTI proxy",
            ),
            row=indicator_row,
            col=1,
        )

        zero = 0.0
        fig.add_hline(y=zero, line_width=1, line_dash="dot", line_color="#808080", row=indicator_row, col=1)
        fig.add_hline(y=tti_level_signal, line_width=1, line_dash="dot", line_color="#ffb347", row=indicator_row, col=1)
        fig.add_hline(y=-tti_level_signal, line_width=1, line_dash="dot", line_color="#ffb347", row=indicator_row, col=1)
        fig.add_hline(y=tti_level_entry, line_width=1, line_dash="dash", line_color="#ff6f61", row=indicator_row, col=1)
        fig.add_hline(y=-tti_level_entry, line_width=1, line_dash="dash", line_color="#ff6f61", row=indicator_row, col=1)

        if toggles.reading_mode and reading_labels is not None and not reading_labels.empty:
            fig.add_trace(
                go.Scatter(
                    x=reading_labels["trade_date"],
                    y=reading_labels["value"],
                    mode="markers+text",
                    marker={"size": 7, "color": "#111111"},
                    text=reading_labels["label"],
                    textposition="top center",
                    textfont={"size": 9},
                    name="Reading labels",
                    showlegend=False,
                    hovertemplate="%{x|%Y-%m-%d}<br>%{text}<br>value=%{y:.4f}<extra></extra>",
                ),
                row=indicator_row,
                col=1,
            )

    if toggles.show_candidate_markers and "candidate_marker_type" in df.columns:
        cand = df[df["candidate_marker_type"] == "candidate_signal"].copy()
        if not cand.empty:
            y = np.where(cand.get("entry_side", "").eq("SHORT"), cand["high"] * 1.005, cand["low"] * 0.995)
            symbol = np.where(cand.get("entry_side", "").eq("SHORT"), "triangle-down", "triangle-up")
            fig.add_trace(
                go.Scatter(
                    x=cand["trade_date"],
                    y=y,
                    mode="markers",
                    marker={"size": 10, "color": "#00aa55", "symbol": symbol, "line": {"width": 1, "color": "#065f46"}},
                    name="Candidate signal",
                    customdata=np.stack([
                        cand.get("entry_side", pd.Series([""] * len(cand))).astype(str),
                        cand.get("state_id", pd.Series([None] * len(cand))).astype(str),
                    ], axis=1),
                    hovertemplate="%{x|%Y-%m-%d}<br>candidate=%{customdata[0]}<br>state=%{customdata[1]}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        blocked = df[df["candidate_marker_type"].isin(["blocked_overlay", "blocked_execution"])].copy()
        if not blocked.empty:
            fig.add_trace(
                go.Scatter(
                    x=blocked["trade_date"],
                    y=blocked["close"],
                    mode="markers",
                    marker={"size": 9, "symbol": "x", "color": "#d62728"},
                    name="Blocked signal",
                    customdata=np.stack([
                        blocked.get("candidate_marker_type", pd.Series([""] * len(blocked))).astype(str),
                        blocked.get("blocked_reason", pd.Series([""] * len(blocked))).astype(str),
                    ], axis=1),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    if toggles.show_exec_markers and "execution_candidate_after" in df.columns:
        exec_pass = df[df["execution_candidate_after"] == True]
        if not exec_pass.empty:
            fig.add_trace(
                go.Scatter(
                    x=exec_pass["trade_date"],
                    y=exec_pass["low"] * 0.992,
                    mode="markers",
                    marker={"size": 6, "symbol": "circle", "color": "#2ca02c"},
                    name="Exec pass",
                    hovertemplate="%{x|%Y-%m-%d}<br>execution pass<extra></extra>",
                ),
                row=1,
                col=1,
            )
        exec_fail = df[df["execution_suppressed_signal"] == True]
        if not exec_fail.empty:
            fig.add_trace(
                go.Scatter(
                    x=exec_fail["trade_date"],
                    y=exec_fail["high"] * 1.008,
                    mode="markers",
                    marker={"size": 6, "symbol": "x", "color": "#b22222"},
                    name="Exec blocked",
                    customdata=np.stack([
                        exec_fail.get("execution_filter_reason", pd.Series([""] * len(exec_fail))).astype(str),
                    ], axis=1),
                    hovertemplate="%{x|%Y-%m-%d}<br>exec blocked: %{customdata[0]}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    fig.update_layout(
        template="plotly_white",
        height=850,
        margin={"l": 35, "r": 25, "t": 35, "b": 25},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if toggles.show_volume and toggles.volume_mode == "separate" and volume_row is not None:
        fig.update_yaxes(title_text="Volume", row=volume_row, col=1)
    if toggles.show_tmf_tti and indicator_row is not None:
        y_title = "TMF / TTI (%)" if scale_pct else "TMF / TTI"
        fig.update_yaxes(title_text=y_title, row=indicator_row, col=1)

    if toggles.show_hmm_band:
        mode_text = f"HMM Display: {toggles.hmm_display_mode}"
        if isinstance(hmm_display_summary, dict):
            window = hmm_display_summary.get("smoothing_window")
            if window is not None and str(toggles.hmm_display_mode).startswith("SMOOTHED"):
                mode_text = f"{mode_text} (w={window})"
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.995,
            y=1.06,
            text=mode_text,
            showarrow=False,
            align="right",
            font={"size": 11, "color": "#334155"},
        )

    return fig
