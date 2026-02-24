"""Dataset loading helpers for research clustering workflows."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """Container for loaded dataset, metadata, and stats."""

    frame: pl.DataFrame
    metadata: dict[str, Any]
    stats: dict[str, Any]


def _metadata_path_for_dataset(dataset_path: Path) -> Path:
    return dataset_path.parent / "metadata.json"


def load_research_dataset(
    dataset_path: Path,
    *,
    date_from: date | None = None,
    date_to: date | None = None,
    tickers: list[str] | None = None,
    sample_frac: float | None = None,
    logger: logging.Logger | None = None,
) -> LoadedDataset:
    """Load exported Gold ML dataset with optional filters."""

    effective_logger = logger or LOGGER
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset parquet not found: {dataset_path}")
    if sample_frac is not None and (sample_frac <= 0 or sample_frac > 1):
        raise ValueError("sample_frac must be in (0, 1].")

    df = pl.read_parquet(dataset_path)
    metadata_path = _metadata_path_for_dataset(dataset_path)
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if date_from is not None:
        df = df.filter(pl.col("trade_date") >= pl.lit(date_from))
    if date_to is not None:
        df = df.filter(pl.col("trade_date") <= pl.lit(date_to))
    if tickers:
        normalized = [ticker.strip().upper() for ticker in tickers if ticker.strip() != ""]
        if normalized:
            df = df.filter(pl.col("ticker").cast(pl.String).str.to_uppercase().is_in(normalized))

    if sample_frac is not None and sample_frac < 1.0 and df.height > 0:
        df = df.sample(fraction=sample_frac, with_replacement=False, shuffle=True, seed=42)

    df = df.sort([column for column in ("ticker", "trade_date") if column in df.columns])

    min_date: str | None = None
    max_date: str | None = None
    if "trade_date" in df.columns and df.height > 0:
        bounds = df.select(
            [
                pl.col("trade_date").min().alias("min_trade_date"),
                pl.col("trade_date").max().alias("max_trade_date"),
            ]
        ).to_dicts()[0]
        min_val = bounds["min_trade_date"]
        max_val = bounds["max_trade_date"]
        min_date = min_val.isoformat() if min_val is not None else None
        max_date = max_val.isoformat() if max_val is not None else None

    ticker_count = int(df.select(pl.col("ticker").n_unique()).item()) if "ticker" in df.columns and df.height > 0 else 0
    stats = {
        "rows": df.height,
        "tickers": ticker_count,
        "min_trade_date": min_date,
        "max_trade_date": max_date,
        "columns": df.columns,
        "dataset_path": str(dataset_path),
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
    }
    effective_logger.info(
        "dataset_loader.loaded rows=%s tickers=%s min_date=%s max_date=%s path=%s",
        stats["rows"],
        stats["tickers"],
        stats["min_trade_date"],
        stats["max_trade_date"],
        dataset_path,
    )
    return LoadedDataset(frame=df, metadata=metadata, stats=stats)

