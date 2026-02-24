"""Local development shim for src-layout imports."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

src_package_dir = Path(__file__).resolve().parent.parent / "src" / "mf_etl"
if src_package_dir.exists():
    __path__.append(str(src_package_dir))
