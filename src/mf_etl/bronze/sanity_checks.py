"""Basic sanity checks for Bronze stage outputs."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def run_bronze_sanity_checks(df: pl.DataFrame, output_path: Path) -> list[str]:
    """Return non-fatal sanity issues detected in Bronze output."""

    issues: list[str] = []
    if df.height == 0:
        issues.append("Bronze output has zero rows")
    if not output_path.exists():
        issues.append(f"Bronze output file missing: {output_path}")
    return issues
