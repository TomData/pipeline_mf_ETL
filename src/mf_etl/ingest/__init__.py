"""Ingestion package for source file discovery and parsing."""

from mf_etl.ingest.discover import (
    DiscoveredFile,
    discover_txt_files,
    extract_ticker_hint,
    infer_exchange_from_path,
)
from mf_etl.ingest.manifest import build_manifest, write_manifest_parquet
from mf_etl.ingest.read_txt import read_stock_txt

__all__ = [
    "DiscoveredFile",
    "discover_txt_files",
    "infer_exchange_from_path",
    "extract_ticker_hint",
    "build_manifest",
    "write_manifest_parquet",
    "read_stock_txt",
]
