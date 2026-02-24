"""Validation report formatting."""

from __future__ import annotations


def format_validation_report(symbol: str, errors: list[str]) -> str:
    """Render a deterministic plain-text validation report."""

    if not errors:
        return f"symbol={symbol} status=ok errors=0"
    joined = "; ".join(errors)
    return f"symbol={symbol} status=failed errors={len(errors)} details={joined}"
