"""Normalization helpers for Bronze-ready OHLCV records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

RAW_COLUMNS: tuple[str, ...] = (
    "raw_ticker",
    "raw_per",
    "raw_date",
    "raw_time",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "raw_volume",
    "raw_openint",
    "source_line_no",
)

BRONZE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "exchange",
    "timeframe",
    "trade_date",
    "trade_dt",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "openint",
    "raw_ticker",
    "raw_per",
    "raw_date",
    "raw_time",
    "source_file",
    "source_file_name",
    "ingest_ts",
    "run_id",
    "source_line_no",
)


@dataclass(frozen=True, slots=True)
class BronzeNormalizeMetadata:
    """Static metadata for one-source normalization run."""

    source_file: Path
    exchange: str
    run_id: str
    ingest_ts: datetime

    @classmethod
    def build(
        cls,
        source_file: Path,
        exchange: str,
        run_id: str,
        ingest_ts: datetime | None = None,
    ) -> "BronzeNormalizeMetadata":
        """Create metadata with normalized defaults."""

        normalized_exchange = exchange.strip().upper() if exchange.strip() else "UNKNOWN"
        ts = ingest_ts or datetime.now(timezone.utc)
        return cls(
            source_file=source_file.resolve(strict=False),
            exchange=normalized_exchange,
            run_id=run_id.strip(),
            ingest_ts=ts,
        )


def _ensure_raw_columns(raw_df: pl.DataFrame) -> pl.DataFrame:
    """Ensure all required raw columns exist with nullable defaults."""

    out = raw_df
    for column in RAW_COLUMNS:
        if column in out.columns:
            continue
        if column == "source_line_no":
            out = out.with_columns(pl.lit(None, dtype=pl.Int64).alias(column))
        else:
            out = out.with_columns(pl.lit(None, dtype=pl.String).alias(column))
    return out


def normalize_bronze_rows(
    raw_df: pl.DataFrame,
    metadata: BronzeNormalizeMetadata,
) -> pl.DataFrame:
    """Normalize one raw ticker file into Bronze-friendly columns and dtypes."""

    out = _ensure_raw_columns(raw_df)

    raw_ticker_clean = pl.col("raw_ticker").cast(pl.String, strict=False).str.strip_chars()
    raw_per_clean = pl.col("raw_per").cast(pl.String, strict=False).str.strip_chars().str.to_uppercase()
    raw_date_clean = pl.col("raw_date").cast(pl.String, strict=False).str.strip_chars()
    raw_time_clean = (
        pl.col("raw_time")
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .str.pad_start(6, fill_char="0")
    )

    timeframe_expr = (
        pl.when(raw_per_clean == "D")
        .then(pl.lit("D1"))
        .when(raw_per_clean.is_null() | (raw_per_clean == ""))
        .then(pl.lit("UNKNOWN"))
        .otherwise(raw_per_clean)
    )

    normalized = out.with_columns(
        [
            raw_ticker_clean.str.to_uppercase().alias("ticker"),
            pl.lit(metadata.exchange).cast(pl.String).alias("exchange"),
            timeframe_expr.alias("timeframe"),
            raw_date_clean.str.strptime(pl.Date, format="%Y%m%d", strict=False).alias("trade_date"),
            pl.concat_str([raw_date_clean, raw_time_clean], separator="")
            .str.strptime(pl.Datetime, format="%Y%m%d%H%M%S", strict=False)
            .alias("trade_dt"),
            pl.col("raw_open").cast(pl.Float64, strict=False).alias("open"),
            pl.col("raw_high").cast(pl.Float64, strict=False).alias("high"),
            pl.col("raw_low").cast(pl.Float64, strict=False).alias("low"),
            pl.col("raw_close").cast(pl.Float64, strict=False).alias("close"),
            pl.col("raw_volume").cast(pl.Float64, strict=False).alias("volume"),
            pl.col("raw_openint").cast(pl.Int64, strict=False).alias("openint"),
            raw_ticker_clean.alias("raw_ticker"),
            raw_per_clean.alias("raw_per"),
            raw_date_clean.alias("raw_date"),
            raw_time_clean.alias("raw_time"),
            pl.lit(str(metadata.source_file)).cast(pl.String).alias("source_file"),
            pl.lit(metadata.source_file.name).cast(pl.String).alias("source_file_name"),
            pl.lit(metadata.ingest_ts).alias("ingest_ts"),
            pl.lit(metadata.run_id).cast(pl.String).alias("run_id"),
            pl.col("source_line_no").cast(pl.Int64, strict=False).alias("source_line_no"),
        ]
    )

    return normalized.select(list(BRONZE_COLUMNS))


def normalize_ohlcv(df: pl.DataFrame, float_dtype: pl.DataType = pl.Float64) -> pl.DataFrame:
    """Backward-compatible OHLCV cast helper for generic tabular inputs."""

    out = df
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        if column in out.columns:
            out = out.with_columns(pl.col(column).cast(float_dtype, strict=False).alias(column))
    if "date" in out.columns:
        out = out.with_columns(pl.col("date").str.to_date(strict=False).alias("date"))
    return out
