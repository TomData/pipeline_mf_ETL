"""Ingestion package for source file discovery and parsing."""

from mf_etl.ingest.discover import DiscoveredFile, discover_txt_files
from mf_etl.ingest.manifest import build_manifest
from mf_etl.ingest.read_txt import read_stock_txt

__all__ = ["DiscoveredFile", "discover_txt_files", "build_manifest", "read_stock_txt"]
