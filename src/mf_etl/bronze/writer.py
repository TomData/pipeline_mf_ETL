"""Bronze parquet writer backed by PyArrow."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


def write_bronze_parquet(
    df: pl.DataFrame,
    output_path: Path,
    compression: str = "zstd",
    compression_level: int | None = 3,
    statistics: bool = True,
) -> Path:
    """Write a Polars DataFrame to parquet using PyArrow."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = df.to_arrow()
    pq.write_table(
        table,
        output_path,
        compression=compression,
        compression_level=compression_level,
        write_statistics=statistics,
    )
    return output_path
