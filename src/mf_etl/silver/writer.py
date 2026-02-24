"""Silver parquet writer for base-series per symbol."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import polars as pl


@dataclass(frozen=True, slots=True)
class SilverWriteResult:
    """Result of writing one Silver per-symbol parquet output."""

    ticker: str
    exchange: str
    output_path: Path
    rows_out: int


def silver_output_path(silver_root: Path, ticker: str, exchange: str) -> Path:
    """Build canonical Silver partition path for one symbol."""

    normalized_ticker = ticker.strip().upper() or "UNKNOWN"
    normalized_exchange = exchange.strip().upper() or "UNKNOWN"
    prefix = normalized_ticker[0] if normalized_ticker else "_"
    if not prefix.strip():
        prefix = "_"
    return (
        silver_root
        / "base_series_by_symbol"
        / f"exchange={normalized_exchange}"
        / f"prefix={prefix}"
        / f"ticker={normalized_ticker}"
        / "part-000.parquet"
    )


def _atomic_temp_path(target_path: Path) -> Path:
    return target_path.parent / f".{target_path.name}.{uuid4().hex}.tmp"


def write_silver_parquet(
    silver_df: pl.DataFrame,
    *,
    silver_root: Path,
    compression: str = "zstd",
    compression_level: int | None = 3,
    statistics: bool = True,
) -> SilverWriteResult:
    """Write one per-symbol Silver dataset atomically."""

    if silver_df.height == 0:
        raise ValueError("Cannot write empty Silver dataframe for symbol output.")

    first = silver_df.select(
        [
            pl.col("ticker").cast(pl.String).first().alias("ticker"),
            pl.col("exchange").cast(pl.String).first().alias("exchange"),
        ]
    ).to_dicts()[0]
    ticker = str(first["ticker"] or "UNKNOWN").strip().upper()
    exchange = str(first["exchange"] or "UNKNOWN").strip().upper()
    output_path = silver_output_path(silver_root, ticker=ticker, exchange=exchange)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sort_cols = [col for col in ("ticker", "trade_date") if col in silver_df.columns]
    sorted_df = silver_df.sort(sort_cols) if sort_cols else silver_df

    temp_path = _atomic_temp_path(output_path)
    try:
        sorted_df.write_parquet(
            temp_path,
            compression=compression,
            compression_level=compression_level,
            statistics=statistics,
        )
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return SilverWriteResult(
        ticker=ticker,
        exchange=exchange,
        output_path=output_path,
        rows_out=sorted_df.height,
    )
