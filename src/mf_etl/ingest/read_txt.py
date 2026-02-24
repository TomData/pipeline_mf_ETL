"""Load raw stock text files into Polars DataFrames."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def _normalize_headers(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {column: column.strip().lower().replace(" ", "_") for column in df.columns}
    return df.rename(rename_map)


def read_stock_txt(path: Path, separator: str | None = None) -> pl.DataFrame:
    """Read a text file with fallback delimiters and normalized headers."""

    delimiters = [separator] if separator else [",", "\t", "|", ";"]
    last_error: Exception | None = None
    for delimiter in delimiters:
        try:
            frame = pl.read_csv(
                path,
                separator=delimiter,
                try_parse_dates=True,
                infer_schema_length=1000,
                ignore_errors=False,
            )
            if frame.width > 1:
                return _normalize_headers(frame)
        except Exception as exc:  # pragma: no cover - depends on source file format
            last_error = exc
    if last_error:
        raise ValueError(f"Unable to parse text file: {path}") from last_error
    raise ValueError(f"Unable to parse text file: {path}")
