"""Silver stage placeholder helpers."""

from __future__ import annotations

from pathlib import Path

from mf_etl.utils.paths import write_marker_file


def ensure_silver_placeholder(silver_root: Path) -> Path:
    """Create a silver stage marker README."""

    content = """# Silver Layer Placeholder

This directory is reserved for cleaned and feature-ready datasets.
"""
    return write_marker_file(silver_root / "README.md", content)
