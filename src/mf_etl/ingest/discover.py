"""Discover raw stock text files from NYSE and NASDAQ folders."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    """Metadata for a discovered source file."""

    exchange: str
    ticker_hint: str
    path: Path


def infer_exchange_from_path(path: Path, logger: logging.Logger | None = None) -> str:
    """Infer exchange from full path using case-insensitive folder matching."""

    effective_logger = logger or LOGGER
    path_text = str(path).lower()
    if "nasdaq stocks" in path_text:
        return "NASDAQ"
    if "nyse stocks" in path_text:
        return "NYSE"
    effective_logger.warning("discover.unknown_exchange path=%s", path)
    return "UNKNOWN"


def extract_ticker_hint(path: Path) -> str:
    """Extract uppercase ticker hint from filename while preserving dotted suffixes."""

    file_name = path.name.strip()
    if file_name.lower().endswith(".txt"):
        file_name = file_name[:-4]
    return file_name.strip().upper()


def discover_txt_files(raw_root: Path, logger: logging.Logger | None = None) -> list[DiscoveredFile]:
    """Recursively discover text files and derive exchange + ticker hint metadata."""

    effective_logger = logger or LOGGER
    files: list[DiscoveredFile] = []
    if not raw_root.exists():
        effective_logger.warning("discover.raw_root_missing raw_root=%s", raw_root)
        return files

    for file_path in sorted(raw_root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() != ".txt":
            continue
        files.append(
            DiscoveredFile(
                exchange=infer_exchange_from_path(file_path, logger=effective_logger),
                ticker_hint=extract_ticker_hint(file_path),
                path=file_path.resolve(strict=False),
            )
        )
    return files
