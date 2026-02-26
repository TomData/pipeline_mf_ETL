"""Utility helpers for overlay viewer app."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


def safe_float(value: Any) -> float | None:
    """Return finite float or None."""

    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def safe_int(value: Any) -> int | None:
    """Return int or None."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def discover_latest_path(pattern: str) -> Path | None:
    """Resolve latest existing path for a glob pattern by mtime then name."""

    candidates = sorted(Path().glob(pattern))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    existing.sort(key=lambda p: (p.stat().st_mtime, p.as_posix()), reverse=True)
    return existing[0]


def normalize_date(value: Any) -> date | None:
    """Parse ISO-like date values to date object."""

    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def as_path_or_none(value: Any) -> Path | None:
    """Return Path object when a non-empty path-like value is provided."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)
