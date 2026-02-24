"""Canonical dtype policy and schema versions for ETL layers."""

from __future__ import annotations

import polars as pl

BRONZE_SCHEMA_VERSION = "v1"
SILVER_SCHEMA_VERSION = "v1"
GOLD_SCHEMA_VERSION = "v1"

BRONZE_FLOAT_DTYPE = pl.Float64
SILVER_FLOAT_DTYPE = pl.Float32
GOLD_FLOAT_DTYPE = pl.Float32

BRONZE_DTYPE_MAP: dict[str, pl.DataType] = {
    "open": BRONZE_FLOAT_DTYPE,
    "high": BRONZE_FLOAT_DTYPE,
    "low": BRONZE_FLOAT_DTYPE,
    "close": BRONZE_FLOAT_DTYPE,
    "adj_close": BRONZE_FLOAT_DTYPE,
    "volume": BRONZE_FLOAT_DTYPE,
}

SILVER_DTYPE_MAP: dict[str, pl.DataType] = {
    "open": SILVER_FLOAT_DTYPE,
    "high": SILVER_FLOAT_DTYPE,
    "low": SILVER_FLOAT_DTYPE,
    "close": SILVER_FLOAT_DTYPE,
    "adj_close": SILVER_FLOAT_DTYPE,
    "volume": SILVER_FLOAT_DTYPE,
}

GOLD_DTYPE_MAP: dict[str, pl.DataType] = {
    "open": GOLD_FLOAT_DTYPE,
    "high": GOLD_FLOAT_DTYPE,
    "low": GOLD_FLOAT_DTYPE,
    "close": GOLD_FLOAT_DTYPE,
    "adj_close": GOLD_FLOAT_DTYPE,
    "volume": GOLD_FLOAT_DTYPE,
}

PRECISION_NAME_TO_DTYPE: dict[str, pl.DataType] = {
    "float64": pl.Float64,
    "float32": pl.Float32,
}


def dtype_for_precision_name(precision_name: str) -> pl.DataType:
    """Resolve precision label to Polars dtype."""

    try:
        return PRECISION_NAME_TO_DTYPE[precision_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported precision name: {precision_name}") from exc
