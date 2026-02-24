"""Sequence construction utilities for HMM input matrices."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SequenceBuildResult:
    """HMM sequence-matrix outputs with row alignment metadata."""

    frame: pl.DataFrame
    X: np.ndarray
    lengths: np.ndarray
    row_keys: pl.DataFrame
    tickers_dropped_short: list[str]


def build_hmm_sequences(
    df: pl.DataFrame,
    *,
    feature_list: list[str],
    min_sequence_length: int = 100,
    logger: logging.Logger | None = None,
) -> SequenceBuildResult:
    """Build stacked HMM matrix and lengths from per-ticker time series."""

    effective_logger = logger or LOGGER
    if min_sequence_length < 1:
        raise ValueError("min_sequence_length must be >= 1.")
    if "ticker" not in df.columns or "trade_date" not in df.columns:
        raise ValueError("Dataframe must include ticker and trade_date.")
    if not feature_list:
        raise ValueError("feature_list must not be empty.")

    missing = [column for column in feature_list if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for HMM sequence build: {missing}")

    sorted_df = df.sort(["ticker", "trade_date"])
    groups = sorted_df.partition_by("ticker", maintain_order=True)

    kept_groups: list[pl.DataFrame] = []
    lengths: list[int] = []
    dropped_tickers: list[str] = []
    for group in groups:
        ticker = str(group["ticker"][0])
        n_rows = group.height
        if n_rows < min_sequence_length:
            dropped_tickers.append(ticker)
            continue
        kept_groups.append(group)
        lengths.append(n_rows)

    if not kept_groups:
        raise ValueError(
            f"No ticker sequences remain after min_sequence_length={min_sequence_length}."
        )

    combined = pl.concat(kept_groups, how="vertical_relaxed").with_row_index("row_id")
    X = combined.select(feature_list).to_numpy().astype(np.float64, copy=False)
    lengths_arr = np.asarray(lengths, dtype=np.int32)
    if int(np.sum(lengths_arr)) != int(combined.height):
        raise ValueError("HMM sequence alignment mismatch: sum(lengths) != row count.")

    row_key_cols = [
        "row_id",
        "ticker",
        "trade_date",
        "trade_dt",
        "flow_state_code",
        "flow_state_label",
        "cluster_id",
        "close",
        "fwd_ret_5",
        "fwd_ret_10",
        "fwd_ret_20",
        "fwd_abs_ret_10",
        "fwd_vol_proxy_10",
    ]
    row_keys = combined.select([column for column in row_key_cols if column in combined.columns])

    if dropped_tickers:
        effective_logger.warning(
            "hmm.sequence_builder dropped_tickers_short count=%s min_sequence_length=%s sample=%s",
            len(dropped_tickers),
            min_sequence_length,
            dropped_tickers[:10],
        )
    effective_logger.info(
        "hmm.sequence_builder built_rows=%s sequences=%s min_sequence_length=%s",
        combined.height,
        len(lengths),
        min_sequence_length,
    )
    return SequenceBuildResult(
        frame=combined,
        X=X,
        lengths=lengths_arr,
        row_keys=row_keys,
        tickers_dropped_short=dropped_tickers,
    )

