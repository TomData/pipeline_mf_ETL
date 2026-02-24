"""Time utility helpers for UTC-safe timestamps."""

from __future__ import annotations

from datetime import datetime, timezone


def now_utc() -> datetime:
    """Return current timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def utc_date_string() -> str:
    """Return current UTC date in ISO format."""

    return now_utc().date().isoformat()
