"""Build a symbol master table from manifest entries."""

from __future__ import annotations

import polars as pl


def build_symbol_master(manifest: pl.DataFrame) -> pl.DataFrame:
    """Aggregate symbol-level metadata for downstream stages."""

    if manifest.is_empty():
        return pl.DataFrame(schema={"exchange": pl.String, "symbol": pl.String, "file_count": pl.Int64})

    return (
        manifest.group_by(["exchange", "symbol"])
        .agg(
            pl.len().alias("file_count"),
            pl.col("source_path").first().alias("sample_path"),
        )
        .sort(["exchange", "symbol"])
    )
