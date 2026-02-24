"""Path and filesystem helper functions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_directories(paths: Iterable[Path]) -> list[Path]:
    """Create all directories in the iterable if they do not exist."""

    created_or_existing: list[Path] = []
    for directory in paths:
        directory.mkdir(parents=True, exist_ok=True)
        created_or_existing.append(directory)
    return created_or_existing


def write_marker_file(path: Path, content: str) -> Path:
    """Write a marker file with deterministic UTF-8 content."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return path
