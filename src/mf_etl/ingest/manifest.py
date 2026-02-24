"""Build source-file manifest tables."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from mf_etl.ingest.discover import DiscoveredFile, discover_txt_files


def build_manifest(raw_root: Path) -> pl.DataFrame:
    """Create a manifest DataFrame of discovered source files."""

    discovered: list[DiscoveredFile] = discover_txt_files(raw_root)
    rows: list[dict[str, object]] = []
    for item in discovered:
        stats = item.path.stat()
        rows.append(
            {
                "exchange": item.exchange,
                "symbol": item.symbol,
                "source_path": str(item.path),
                "file_size_bytes": stats.st_size,
                "modified_ts": stats.st_mtime,
            }
        )
    return pl.DataFrame(rows)
