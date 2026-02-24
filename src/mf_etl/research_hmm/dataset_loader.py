"""Dataset loader wrappers for HMM baseline research runs."""

from __future__ import annotations

from datetime import date
import logging
from pathlib import Path

import polars as pl

from mf_etl.research.dataset_loader import LoadedDataset, load_research_dataset
from mf_etl.research.forward_labels import add_forward_outcomes

LOGGER = logging.getLogger(__name__)


def load_hmm_dataset(
    dataset_path: Path,
    *,
    forward_windows: list[int],
    date_from: date | None = None,
    date_to: date | None = None,
    sample_frac: float | None = None,
    logger: logging.Logger | None = None,
) -> tuple[LoadedDataset, pl.DataFrame]:
    """Load dataset and attach forward labels for HMM profiling."""

    effective_logger = logger or LOGGER
    loaded = load_research_dataset(
        dataset_path,
        date_from=date_from,
        date_to=date_to,
        sample_frac=sample_frac,
        logger=effective_logger,
    )
    if "close" not in loaded.frame.columns:
        raise ValueError("Dataset must include close for forward-label generation.")
    with_forward = add_forward_outcomes(loaded.frame, windows=forward_windows)
    if "trade_date" not in with_forward.columns:
        raise ValueError("Dataset must include trade_date.")
    with_forward = with_forward.with_columns(pl.col("trade_date").cast(pl.Date, strict=False).alias("trade_date"))
    bad_dates = with_forward.filter(pl.col("trade_date").is_null()).height
    if bad_dates > 0:
        raise ValueError(f"trade_date contains {bad_dates} unparsable rows.")
    return loaded, with_forward

