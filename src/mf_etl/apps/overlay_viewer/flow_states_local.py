"""Local per-ticker flow-state builder for overlay viewer cache compute.

This implements a lightweight deterministic event grammar over TMF to assign S0..S4
states without requiring global ML dataset artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

FLOW_LABELS: dict[int, str] = {
    0: "S0_QUIET",
    1: "S1_EARLY_DEMAND",
    2: "S2_PERSISTENT_DEMAND",
    3: "S3_EARLY_SUPPLY",
    4: "S4_PERSISTENT_SUPPLY",
}


@dataclass(frozen=True, slots=True)
class LocalFlowStateParams:
    hold_thr: float = 0.0
    burst_thr: float = 0.10
    persistence_window: int = 20
    persistent_min_hits: int = 10


def _state_label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("flow_state_code").is_null())
        .then(pl.lit("NA"))
        .when(pl.col("flow_state_code") == 0)
        .then(pl.lit("S0_QUIET"))
        .when(pl.col("flow_state_code") == 1)
        .then(pl.lit("S1_EARLY_DEMAND"))
        .when(pl.col("flow_state_code") == 2)
        .then(pl.lit("S2_PERSISTENT_DEMAND"))
        .when(pl.col("flow_state_code") == 3)
        .then(pl.lit("S3_EARLY_SUPPLY"))
        .when(pl.col("flow_state_code") == 4)
        .then(pl.lit("S4_PERSISTENT_SUPPLY"))
        .otherwise(pl.lit("NA"))
        .alias("flow_state_label")
    )


def compute_local_flow_states(
    indicators: pl.DataFrame,
    *,
    params: LocalFlowStateParams,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Compute local flow states and event flags from TMF features.

    Required columns: ticker, trade_date and either tmf_raw or tmf.
    """

    if "ticker" not in indicators.columns or "trade_date" not in indicators.columns:
        raise ValueError("Indicators frame must include ticker and trade_date columns.")

    if "tmf_raw" in indicators.columns:
        tmf_col = "tmf_raw"
    elif "tmf" in indicators.columns:
        tmf_col = "tmf"
    else:
        raise ValueError("Indicators frame must include tmf_raw or tmf column.")

    hold_thr = float(params.hold_thr)
    burst_thr = float(params.burst_thr)
    window = int(params.persistence_window)
    min_hits = int(params.persistent_min_hits)

    work = indicators.select("ticker", "trade_date", pl.col(tmf_col).cast(pl.Float64, strict=False).alias("tmf_21")).sort(
        ["ticker", "trade_date"]
    )

    work = work.with_columns(
        pl.col("tmf_21").is_finite().alias("tmf_ready_21"),
        pl.col("tmf_21").shift(1).over("ticker").alias("tmf_prev"),
        (pl.col("tmf_21") - pl.col("tmf_21").shift(5).over("ticker")).alias("tmf_slope_5"),
    )

    work = work.with_columns(
        (
            pl.col("tmf_ready_21")
            & pl.col("tmf_prev").is_not_null()
            & (pl.col("tmf_prev") <= 0.0)
            & (pl.col("tmf_21") > 0.0)
        ).alias("ev_tmf_zero_up"),
        (
            pl.col("tmf_ready_21")
            & pl.col("tmf_prev").is_not_null()
            & (pl.col("tmf_prev") >= 0.0)
            & (pl.col("tmf_21") < 0.0)
        ).alias("ev_tmf_zero_down"),
        (pl.col("tmf_ready_21") & (pl.col("tmf_slope_5") > burst_thr)).alias("ev_tmf_burst_up"),
        (pl.col("tmf_ready_21") & (pl.col("tmf_slope_5") < -burst_thr)).alias("ev_tmf_burst_down"),
        (pl.col("tmf_ready_21") & (pl.col("tmf_21") > hold_thr)).alias("ev_tmf_hold_pos"),
        (pl.col("tmf_ready_21") & (pl.col("tmf_21") < -hold_thr)).alias("ev_tmf_hold_neg"),
    )

    work = work.with_columns(
        pl.when(pl.col("tmf_ready_21") & (pl.col("tmf_21") > 0.0)).then(1).otherwise(0).alias("tmf_pos_hit"),
        pl.when(pl.col("tmf_ready_21") & (pl.col("tmf_21") < 0.0)).then(1).otherwise(0).alias("tmf_neg_hit"),
    )

    work = work.with_columns(
        pl.col("tmf_pos_hit").rolling_sum(window_size=window, min_samples=window).over("ticker").alias("tmf_pos_hits_w"),
        pl.col("tmf_neg_hit").rolling_sum(window_size=window, min_samples=window).over("ticker").alias("tmf_neg_hits_w"),
    )

    work = work.with_columns(
        (
            pl.col("ev_tmf_hold_pos") & (pl.col("tmf_pos_hits_w") >= min_hits)
        ).alias("tmf_persistent_pos"),
        (
            pl.col("ev_tmf_hold_neg") & (pl.col("tmf_neg_hits_w") >= min_hits)
        ).alias("tmf_persistent_neg"),
    )

    work = work.with_columns(
        pl.when(~pl.col("tmf_ready_21"))
        .then(pl.lit(None).cast(pl.Int8))
        .when(pl.col("tmf_persistent_pos"))
        .then(pl.lit(2).cast(pl.Int8))
        .when(pl.col("tmf_persistent_neg"))
        .then(pl.lit(4).cast(pl.Int8))
        .when(pl.col("tmf_21") > hold_thr)
        .then(pl.lit(1).cast(pl.Int8))
        .when(pl.col("tmf_21") < -hold_thr)
        .then(pl.lit(3).cast(pl.Int8))
        .otherwise(pl.lit(0).cast(pl.Int8))
        .alias("flow_state_code")
    ).with_columns(_state_label_expr())

    ready_rows = int(work.filter(pl.col("tmf_ready_21")).height)
    total_rows = int(work.height)

    state_counts: dict[str, int] = {}
    for state_code, state_label in FLOW_LABELS.items():
        count = int(
            work.filter(pl.col("flow_state_code") == state_code).height
        )
        state_counts[state_label] = count

    summary: dict[str, Any] = {
        "states_flow_source": "local_event_grammar_v1",
        "rows_total": total_rows,
        "rows_ready": ready_rows,
        "ready_ratio": float(ready_rows / total_rows) if total_rows > 0 else None,
        "state_counts": state_counts,
        "params": {
            "hold_thr": hold_thr,
            "burst_thr": burst_thr,
            "persistence_window": window,
            "persistent_min_hits": min_hits,
        },
        "event_counts": {
            "ev_tmf_zero_up": int(work.select(pl.col("ev_tmf_zero_up").cast(pl.Int64).sum()).item() or 0),
            "ev_tmf_zero_down": int(work.select(pl.col("ev_tmf_zero_down").cast(pl.Int64).sum()).item() or 0),
            "ev_tmf_burst_up": int(work.select(pl.col("ev_tmf_burst_up").cast(pl.Int64).sum()).item() or 0),
            "ev_tmf_burst_down": int(work.select(pl.col("ev_tmf_burst_down").cast(pl.Int64).sum()).item() or 0),
            "ev_tmf_hold_pos": int(work.select(pl.col("ev_tmf_hold_pos").cast(pl.Int64).sum()).item() or 0),
            "ev_tmf_hold_neg": int(work.select(pl.col("ev_tmf_hold_neg").cast(pl.Int64).sum()).item() or 0),
        },
    }

    out = work.select(
        "ticker",
        "trade_date",
        "tmf_21",
        "tmf_ready_21",
        "tmf_slope_5",
        "ev_tmf_zero_up",
        "ev_tmf_zero_down",
        "ev_tmf_burst_up",
        "ev_tmf_burst_down",
        "ev_tmf_hold_pos",
        "ev_tmf_hold_neg",
        "tmf_pos_hits_w",
        "tmf_neg_hits_w",
        "tmf_persistent_pos",
        "tmf_persistent_neg",
        "flow_state_code",
        "flow_state_label",
    )

    return out, summary
