"""Ingestion package for source file discovery and parsing."""

from mf_etl.ingest.discover import (
    DiscoveredFile,
    discover_txt_files,
    extract_ticker_hint,
    infer_exchange_from_path,
)
from mf_etl.ingest.manifest import (
    CURRENT_MANIFEST_FILE,
    MANIFEST_STATUS_VALUES,
    STABLE_MANIFEST_FILE,
    ManifestPaths,
    build_manifest,
    classify_manifest,
    empty_manifest,
    exchange_counts,
    get_manifest_paths,
    load_manifest_parquet,
    manifest_status_counts,
    persist_classified_current_manifest,
    promote_current_manifest_to_stable,
    write_manifest_parquet,
)
from mf_etl.ingest.read_txt import TxtReadResult, read_stock_txt, read_stock_txt_with_rejects

__all__ = [
    "DiscoveredFile",
    "discover_txt_files",
    "infer_exchange_from_path",
    "extract_ticker_hint",
    "CURRENT_MANIFEST_FILE",
    "STABLE_MANIFEST_FILE",
    "MANIFEST_STATUS_VALUES",
    "ManifestPaths",
    "empty_manifest",
    "get_manifest_paths",
    "load_manifest_parquet",
    "classify_manifest",
    "manifest_status_counts",
    "exchange_counts",
    "persist_classified_current_manifest",
    "promote_current_manifest_to_stable",
    "build_manifest",
    "write_manifest_parquet",
    "TxtReadResult",
    "read_stock_txt",
    "read_stock_txt_with_rejects",
]
