"""Decoded-state dataframe builders for HMM research runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl


def _compute_state_change_features(group: pl.DataFrame) -> pl.DataFrame:
    states = group["hmm_state"].to_numpy().astype(np.int32, copy=False)
    n = states.shape[0]

    prev: list[int | None] = [None] * n
    changed = np.zeros(n, dtype=bool)
    transition: list[int | None] = [None] * n
    run_len = np.zeros(n, dtype=np.int32)
    bs_change = np.zeros(n, dtype=np.int32)

    current_run = 0
    for idx in range(n):
        if idx == 0:
            prev[idx] = None
            changed[idx] = False
            transition[idx] = None
            current_run = 1
            run_len[idx] = current_run
            bs_change[idx] = 0
            continue

        prev_state = int(states[idx - 1])
        prev[idx] = prev_state
        is_changed = int(states[idx]) != prev_state
        changed[idx] = is_changed
        transition[idx] = int(prev_state * 100 + int(states[idx]))
        if is_changed:
            current_run = 1
            bs_change[idx] = 0
        else:
            current_run += 1
            bs_change[idx] = bs_change[idx - 1] + 1
        run_len[idx] = current_run

    return group.with_columns(
        [
            pl.Series(name="hmm_state_prev", values=prev),
            pl.Series(name="hmm_state_changed", values=changed),
            pl.Series(name="hmm_transition_code", values=transition),
            pl.Series(name="hmm_state_run_length", values=run_len),
            pl.Series(name="bs_hmm_state_change", values=bs_change),
        ]
    )


def build_decoded_rows(
    base_frame: pl.DataFrame,
    *,
    decoded_states: np.ndarray,
    posterior_probs: np.ndarray | None,
    run_id: str,
    schema_version: str = "v1",
    calc_version: str = "hmm_baseline_v1",
    built_ts: datetime | None = None,
) -> pl.DataFrame:
    """Attach decoded states and transition helpers to aligned rows."""

    if "ticker" not in base_frame.columns or "trade_date" not in base_frame.columns:
        raise ValueError("base_frame must include ticker and trade_date.")
    if base_frame.height != decoded_states.shape[0]:
        raise ValueError("decoded_states length must match base_frame row count.")

    ts = built_ts or datetime.now(timezone.utc)
    decoded = base_frame.with_columns(
        pl.Series(name="hmm_state", values=decoded_states.astype(np.int16, copy=False))
    )
    if posterior_probs is not None:
        if posterior_probs.shape[0] != base_frame.height:
            raise ValueError("posterior_probs row count must match base_frame.")
        prob_max = posterior_probs.max(axis=1).astype(np.float32, copy=False)
        entropy = (
            -(posterior_probs * np.log(np.clip(posterior_probs, 1e-12, None))).sum(axis=1)
        ).astype(np.float32, copy=False)
        decoded = decoded.with_columns(
            [
                pl.Series(name="hmm_state_prob_max", values=prob_max),
                pl.Series(name="hmm_state_entropy", values=entropy),
            ]
        )
    else:
        decoded = decoded.with_columns(
            [
                pl.lit(None, dtype=pl.Float32).alias("hmm_state_prob_max"),
                pl.lit(None, dtype=pl.Float32).alias("hmm_state_entropy"),
            ]
        )

    decoded = (
        decoded.sort(["ticker", "trade_date"])
        .group_by("ticker", maintain_order=True)
        .map_groups(_compute_state_change_features)
        .with_columns(
            [
                pl.col("hmm_state_prev").cast(pl.Int16, strict=False),
                pl.col("hmm_transition_code").cast(pl.Int32, strict=False),
                pl.col("hmm_state_run_length").cast(pl.Int32, strict=False),
                pl.col("bs_hmm_state_change").cast(pl.Int32, strict=False),
                pl.lit(schema_version).alias("hmm_schema_version"),
                pl.lit(calc_version).alias("hmm_calc_version"),
                pl.lit(run_id).alias("run_id"),
                pl.lit(ts).alias("built_ts"),
            ]
        )
    )

    ordered = [
        "ticker",
        "trade_date",
        "trade_dt",
        "hmm_state",
        "hmm_state_prob_max",
        "hmm_state_entropy",
        "hmm_state_changed",
        "hmm_state_prev",
        "hmm_transition_code",
        "hmm_state_run_length",
        "bs_hmm_state_change",
        "flow_state_code",
        "flow_state_label",
        "cluster_id",
        "fwd_ret_5",
        "fwd_ret_10",
        "fwd_ret_20",
        "fwd_abs_ret_10",
        "fwd_vol_proxy_10",
        "hmm_schema_version",
        "hmm_calc_version",
        "run_id",
        "built_ts",
    ]
    trailing = [column for column in decoded.columns if column not in ordered]
    return decoded.select([column for column in ordered if column in decoded.columns] + trailing)


def decode_with_model(
    model: Any,
    X: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Decode Viterbi states and optional posterior probabilities."""

    states = model.predict(X, lengths=lengths.tolist()).astype(np.int16, copy=False)
    probs: np.ndarray | None = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X, lengths=lengths.tolist()).astype(np.float64, copy=False)
    return states, probs

