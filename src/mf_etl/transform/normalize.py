"""Normalization helpers for canonical OHLCV shape."""

from __future__ import annotations

import polars as pl


ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "datetime", "timestamp"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c"),
    "adj_close": ("adj_close", "adjclose", "adjusted_close"),
    "volume": ("volume", "vol", "v"),
}


def _rename_to_canonical(df: pl.DataFrame) -> pl.DataFrame:
    rename_map: dict[str, str] = {}
    lower_to_original = {col.lower(): col for col in df.columns}

    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            original = lower_to_original.get(alias)
            if original and original != canonical:
                rename_map[original] = canonical
                break
    return df.rename(rename_map) if rename_map else df


def normalize_ohlcv(df: pl.DataFrame, float_dtype: pl.DataType = pl.Float64) -> pl.DataFrame:
    """Normalize common OHLCV columns and cast numeric data."""

    out = _rename_to_canonical(df)

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    cast_exprs: list[pl.Expr] = []
    for column in numeric_columns:
        if column in out.columns:
            cast_exprs.append(pl.col(column).cast(float_dtype, strict=False).alias(column))
    if "date" in out.columns:
        cast_exprs.append(pl.col("date").str.to_date(strict=False).alias("date"))
    return out.with_columns(cast_exprs) if cast_exprs else out
