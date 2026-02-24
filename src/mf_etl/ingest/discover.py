"""Discover raw stock text files from NYSE and NASDAQ folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    """Metadata for a discovered source file."""

    exchange: str
    symbol: str
    path: Path


def _infer_exchange(path: Path) -> str:
    parent_name = path.parent.name.lower()
    if "nasdaq" in parent_name:
        return "NASDAQ"
    if "nyse" in parent_name:
        return "NYSE"
    return "UNKNOWN"


def discover_txt_files(raw_root: Path) -> list[DiscoveredFile]:
    """Recursively discover .txt files and infer symbol + exchange."""

    files: list[DiscoveredFile] = []
    for file_path in sorted(raw_root.rglob("*.txt")):
        if not file_path.is_file():
            continue
        files.append(
            DiscoveredFile(
                exchange=_infer_exchange(file_path),
                symbol=file_path.stem.upper(),
                path=file_path.resolve(),
            )
        )
    return files
