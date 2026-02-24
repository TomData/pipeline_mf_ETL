"""Build, classify, and persist source-file manifests."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence
from uuid import uuid4

import polars as pl

from mf_etl.ingest.discover import DiscoveredFile
from mf_etl.utils.time_utils import now_utc

LOGGER = logging.getLogger(__name__)

ManifestStatus = Literal["NEW", "CHANGED", "UNCHANGED"]
MANIFEST_STATUS_VALUES: tuple[ManifestStatus, ...] = ("NEW", "CHANGED", "UNCHANGED")

CURRENT_MANIFEST_FILE = "file_manifest_current.parquet"
STABLE_MANIFEST_FILE = "file_manifest.parquet"


@dataclass(frozen=True, slots=True)
class ManifestPaths:
    """Resolved manifest locations under the Bronze root."""

    current_path: Path
    stable_path: Path


def get_manifest_paths(bronze_root: Path) -> ManifestPaths:
    """Return current and stable manifest paths for the Bronze layer."""

    manifest_dir = bronze_root / "manifests"
    return ManifestPaths(
        current_path=manifest_dir / CURRENT_MANIFEST_FILE,
        stable_path=manifest_dir / STABLE_MANIFEST_FILE,
    )


def _base_manifest_schema() -> dict[str, pl.DataType]:
    """Stable schema for file manifests (without status column)."""

    return {
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


def _manifest_schema_with_status() -> dict[str, pl.DataType]:
    """Stable schema for classified manifests."""

    schema = dict(_base_manifest_schema())
    schema["manifest_status"] = pl.String
    return schema


def empty_manifest(include_status: bool = False) -> pl.DataFrame:
    """Return an empty manifest frame with stable schema."""

    return pl.DataFrame(schema=_manifest_schema_with_status() if include_status else _base_manifest_schema())


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
        return empty_manifest(include_status=False)

    return pl.DataFrame(rows, schema_overrides=_base_manifest_schema())


def _atomic_temp_path(target_path: Path) -> Path:
    """Create a temp path in the same directory for atomic replacement."""

    return target_path.parent / f".{target_path.name}.{uuid4().hex}.tmp"


def write_manifest_parquet(
    manifest: pl.DataFrame,
    output_path: Path,
    compression: str = "zstd",
    compression_level: int | None = 3,
    statistics: bool = True,
) -> Path:
    """Write manifest DataFrame to parquet atomically and return output path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(output_path)
    try:
        manifest.write_parquet(
            temp_path,
            compression=compression,
            compression_level=compression_level,
            statistics=statistics,
        )
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return output_path


def load_manifest_parquet(path: Path) -> pl.DataFrame | None:
    """Load manifest parquet if it exists, otherwise return None."""

    if not path.exists():
        return None
    return pl.read_parquet(path)


def classify_manifest(
    current_manifest: pl.DataFrame,
    previous_manifest: pl.DataFrame | None,
    logger: logging.Logger | None = None,
) -> pl.DataFrame:
    """Classify current manifest rows as NEW, CHANGED, or UNCHANGED."""

    effective_logger = logger or LOGGER
    if current_manifest.height == 0:
        return empty_manifest(include_status=True)

    required = {"source_file", "fingerprint"}
    missing_current = required.difference(current_manifest.columns)
    if missing_current:
        raise ValueError(f"Current manifest missing columns: {', '.join(sorted(missing_current))}")

    if previous_manifest is None or previous_manifest.height == 0:
        return current_manifest.with_columns(pl.lit("NEW").alias("manifest_status"))

    missing_previous = required.difference(previous_manifest.columns)
    if missing_previous:
        effective_logger.warning(
            "manifest.classify_previous_missing_columns columns=%s; defaulting all current files to NEW",
            sorted(missing_previous),
        )
        return current_manifest.with_columns(pl.lit("NEW").alias("manifest_status"))

    previous_compact = (
        previous_manifest.select(["source_file", "fingerprint"])
        .unique(subset=["source_file"], keep="last")
        .rename({"fingerprint": "previous_fingerprint"})
    )
    classified = (
        current_manifest.join(previous_compact, on="source_file", how="left")
        .with_columns(
            pl.when(pl.col("previous_fingerprint").is_null())
            .then(pl.lit("NEW"))
            .when(pl.col("fingerprint") == pl.col("previous_fingerprint"))
            .then(pl.lit("UNCHANGED"))
            .otherwise(pl.lit("CHANGED"))
            .alias("manifest_status")
        )
        .drop("previous_fingerprint")
    )
    return classified


def manifest_status_counts(manifest: pl.DataFrame) -> dict[str, int]:
    """Return NEW/CHANGED/UNCHANGED counts from a classified manifest."""

    counts = {status: 0 for status in MANIFEST_STATUS_VALUES}
    if manifest.height == 0 or "manifest_status" not in manifest.columns:
        return counts

    for row in manifest.group_by("manifest_status").len(name="count").to_dicts():
        status = str(row["manifest_status"])
        if status in counts:
            counts[status] = int(row["count"])
    return counts


def exchange_counts(manifest: pl.DataFrame) -> dict[str, int]:
    """Return exchange counts from manifest rows."""

    if manifest.height == 0 or "exchange" not in manifest.columns:
        return {}
    result: dict[str, int] = {}
    for row in manifest.group_by("exchange").len(name="count").to_dicts():
        result[str(row["exchange"])] = int(row["count"])
    return dict(sorted(result.items()))


def persist_classified_current_manifest(
    *,
    classified_manifest: pl.DataFrame,
    bronze_root: Path,
    compression: str = "zstd",
    compression_level: int | None = 3,
    statistics: bool = True,
) -> Path:
    """Persist classified current manifest to the standard current path."""

    paths = get_manifest_paths(bronze_root)
    return write_manifest_parquet(
        classified_manifest,
        paths.current_path,
        compression=compression,
        compression_level=compression_level,
        statistics=statistics,
    )


def promote_current_manifest_to_stable(
    *,
    classified_manifest: pl.DataFrame,
    bronze_root: Path,
    compression: str = "zstd",
    compression_level: int | None = 3,
    statistics: bool = True,
) -> Path:
    """Promote classified current manifest to the stable manifest path."""

    paths = get_manifest_paths(bronze_root)
    return write_manifest_parquet(
        classified_manifest,
        paths.stable_path,
        compression=compression,
        compression_level=compression_level,
        statistics=statistics,
    )
