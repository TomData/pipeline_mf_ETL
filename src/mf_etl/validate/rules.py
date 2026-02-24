"""Rule-based data quality checks."""

from __future__ import annotations

import polars as pl

REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def validate_ohlcv(df: pl.DataFrame) -> list[str]:
    """Return validation errors for a normalized OHLCV DataFrame."""

    errors: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return errors

    null_open = int(df.select(pl.col("open").is_null().sum()).item())
    if null_open > 0:
        errors.append(f"Column open has {null_open} null rows")

    negative_volume = int(df.select((pl.col("volume") < 0).sum()).item())
    if negative_volume > 0:
        errors.append(f"Column volume has {negative_volume} negative rows")

    if df.height == 0:
        errors.append("DataFrame contains zero rows")
    return errors
