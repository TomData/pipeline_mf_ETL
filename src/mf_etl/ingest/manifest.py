"""Build source-file manifest tables."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import polars as pl

from mf_etl.ingest.discover import DiscoveredFile
from mf_etl.utils.time_utils import now_utc

LOGGER = logging.getLogger(__name__)


def _empty_manifest() -> pl.DataFrame:
    """Return an empty manifest with a stable schema."""

    return pl.DataFrame(
        schema={
            "source_file": pl.String,
            "source_file_name": pl.String,
            "source_dir": pl.String,
            "exchange": pl.String,
            "ticker_hint": pl.String,
            "file_size_bytes": pl.Int64,
            "file_mtime": pl.Datetime(time_zone="UTC"),
            "fingerprint": pl.String,
            "discovered_ts": pl.Datetime(time_zone="UTC"),
            "file_ext": pl.String,
        }
    )


def build_manifest(
    discovered: Sequence[DiscoveredFile],
    discovered_ts: datetime | None = None,
    logger: logging.Logger | None = None,
) -> pl.DataFrame:
    """Create a Polars manifest DataFrame from discovered source files."""

    effective_logger = logger or LOGGER
    discovered_at = discovered_ts or now_utc()
    rows: list[dict[str, object]] = []
    for item in discovered:
        try:
            stats = item.path.stat()
        except OSError as exc:
            effective_logger.warning("manifest.stat_failed path=%s error=%s", item.path, exc)
            continue

        mtime_utc = datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc)
        source_file = str(item.path)
        rows.append(
            {
                "source_file": source_file,
                "source_file_name": item.path.name,
                "source_dir": str(item.path.parent),
                "exchange": item.exchange,
                "ticker_hint": item.ticker_hint,
                "file_size_bytes": stats.st_size,
                "file_mtime": mtime_utc,
                "fingerprint": f"{source_file}|{stats.st_size}|{stats.st_mtime_ns}",
                "discovered_ts": discovered_at,
                "file_ext": item.path.suffix.lower(),
            }
        )
    if not rows:
        return _empty_manifest()

    return pl.DataFrame(
        rows,
        schema_overrides={
            "source_file": pl.String,
            "source_file_name": pl.String,
            "source_dir": pl.String,
            "exchange": pl.String,
            "ticker_hint": pl.String,
            "file_size_bytes": pl.Int64,
            "file_mtime": pl.Datetime(time_zone="UTC"),
            "fingerprint": pl.String,
            "discovered_ts": pl.Datetime(time_zone="UTC"),
            "file_ext": pl.String,
        },
    )


def write_manifest_parquet(
    manifest: pl.DataFrame,
    output_path: Path,
    compression: str = "zstd",
    compression_level: int | None = 3,
    statistics: bool = True,
) -> Path:
    """Write manifest DataFrame to parquet and return the output path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(
        output_path,
        compression=compression,
        compression_level=compression_level,
        statistics=statistics,
    )
    return output_path
