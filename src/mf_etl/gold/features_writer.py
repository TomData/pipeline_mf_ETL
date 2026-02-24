"""Atomic writer for Gold Features v1 per-symbol parquet outputs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import polars as pl


@dataclass(frozen=True, slots=True)
class GoldFeatureWriteResult:
    """Result metadata for one Gold feature parquet write."""

    ticker: str
    exchange: str
    output_path: Path
    rows_out: int


def gold_feature_output_path(gold_root: Path, ticker: str, exchange: str) -> Path:
    """Build canonical Gold feature output path for one symbol."""

    normalized_ticker = ticker.strip().upper() or "UNKNOWN"
    normalized_exchange = exchange.strip().upper() or "UNKNOWN"
    prefix = normalized_ticker[0] if normalized_ticker else "_"
    if not prefix.strip():
        prefix = "_"
    return (
        gold_root
        / "features_by_symbol"
        / f"exchange={normalized_exchange}"
        / f"prefix={prefix}"
        / f"ticker={normalized_ticker}"
        / "part-000.parquet"
    )


def _atomic_temp_path(target_path: Path) -> Path:
    return target_path.parent / f".{target_path.name}.{uuid4().hex}.tmp"


def write_gold_feature_parquet(
    feature_df: pl.DataFrame,
    *,
    gold_root: Path,
    compression: str = "zstd",
    compression_level: int | None = 3,
    statistics: bool = True,
) -> GoldFeatureWriteResult:
    """Write one Gold feature frame atomically and idempotently."""

    if feature_df.height == 0:
        raise ValueError("Cannot write empty Gold feature dataframe.")

    first = feature_df.select(
        [
            pl.col("ticker").cast(pl.String).first().alias("ticker"),
            pl.col("exchange").cast(pl.String).first().alias("exchange"),
        ]
    ).to_dicts()[0]
    ticker = str(first["ticker"] or "UNKNOWN").strip().upper()
    exchange = str(first["exchange"] or "UNKNOWN").strip().upper()
    output_path = gold_feature_output_path(gold_root, ticker=ticker, exchange=exchange)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sort_cols = [column for column in ("ticker", "trade_date") if column in feature_df.columns]
    sorted_df = feature_df.sort(sort_cols) if sort_cols else feature_df

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

    return GoldFeatureWriteResult(
        ticker=ticker,
        exchange=exchange,
        output_path=output_path,
        rows_out=sorted_df.height,
    )

