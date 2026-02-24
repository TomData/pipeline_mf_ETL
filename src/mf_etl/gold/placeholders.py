"""Gold stage placeholder helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb

from mf_etl.utils.paths import write_marker_file


def ensure_gold_placeholder(gold_root: Path) -> Path:
    """Create a gold stage marker README."""

    content = """# Gold Layer Placeholder

This directory is reserved for curated analytics-ready datasets and marts.
"""
    return write_marker_file(gold_root / "README.md", content)


def open_gold_duckdb(database_path: Path) -> duckdb.DuckDBPyConnection:
    """Open or create a DuckDB database for gold-layer prototypes."""

    database_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(database_path))
