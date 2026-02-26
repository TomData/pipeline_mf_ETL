"""Persistent ticker cache helpers for overlay viewer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from mf_etl.backtest.writer import (
    write_json_atomically,
    write_parquet_atomically,
)
from mf_etl.config import AppSettings

CACHE_SCHEMA_VERSION = "overlay_viewer_cache_v1_1"


@dataclass(frozen=True, slots=True)
class TickerCacheRunInfo:
    ticker: str
    run_id: str
    run_dir: Path
    computed_ts: str | None
    date_min: str | None
    date_max: str | None
    row_count: int | None
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TickerCacheBundle:
    run_dir: Path
    ticker: str
    run_id: str
    meta: dict[str, Any]
    summary: dict[str, Any]
    merged: pl.DataFrame


def cache_root(settings: AppSettings) -> Path:
    return settings.paths.artifacts_root / "ticker_cache"


def ticker_cache_dir(settings: AppSettings, ticker: str) -> Path:
    return cache_root(settings) / ticker.upper()


def build_cache_run_id(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"cache-{digest}"


def cache_run_dir(settings: AppSettings, ticker: str, run_id: str) -> Path:
    return ticker_cache_dir(settings, ticker) / run_id


def run_exists(settings: AppSettings, ticker: str, run_id: str) -> bool:
    run_dir = cache_run_dir(settings, ticker, run_id)
    required = [
        run_dir / "ohlcv.parquet",
        run_dir / "indicators.parquet",
        run_dir / "meta.json",
        run_dir / "summary.json",
    ]
    return run_dir.exists() and all(path.exists() for path in required)


def list_cached_tickers(settings: AppSettings) -> list[str]:
    root = cache_root(settings)
    if not root.exists():
        return []
    out: list[str] = []
    for path in root.iterdir():
        if path.is_dir():
            out.append(path.name.upper())
    return sorted(set(out))


def list_cache_runs(settings: AppSettings, ticker: str) -> list[TickerCacheRunInfo]:
    base = ticker_cache_dir(settings, ticker)
    if not base.exists():
        return []
    runs: list[TickerCacheRunInfo] = []
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        summary_path = run_dir / "summary.json"
        if not meta_path.exists() or not summary_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        runs.append(
            TickerCacheRunInfo(
                ticker=ticker.upper(),
                run_id=run_dir.name,
                run_dir=run_dir,
                computed_ts=str(meta.get("computed_ts")) if meta.get("computed_ts") is not None else None,
                date_min=str(meta.get("date_min")) if meta.get("date_min") is not None else None,
                date_max=str(meta.get("date_max")) if meta.get("date_max") is not None else None,
                row_count=int(meta.get("row_count")) if meta.get("row_count") is not None else None,
                params=meta.get("params", {}),
            )
        )
    runs.sort(key=lambda item: (item.computed_ts or "", item.run_id), reverse=True)
    return runs


def latest_cache_run(settings: AppSettings, ticker: str) -> TickerCacheRunInfo | None:
    runs = list_cache_runs(settings, ticker)
    return runs[0] if runs else None


def _read_parquet_if_exists(path: Path, schema: dict[str, pl.DataType] | None = None) -> pl.DataFrame:
    if path.exists():
        return pl.read_parquet(path)
    if schema is None:
        return pl.DataFrame()
    return pl.DataFrame(schema=schema)


def _prepare_join_df(df: pl.DataFrame, required_cols: list[tuple[str, pl.DataType]]) -> pl.DataFrame:
    out = df
    for col_name, dtype in required_cols:
        if col_name not in out.columns:
            out = out.with_columns(pl.lit(None).cast(dtype).alias(col_name))
    return out


def load_cache_bundle(run_dir: Path) -> TickerCacheBundle:
    meta_path = run_dir / "meta.json"
    summary_path = run_dir / "summary.json"
    if not meta_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing cache metadata in {run_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ticker = str(meta.get("ticker") or "UNKNOWN").upper()
    run_id = str(meta.get("run_id") or run_dir.name)

    ohlcv = _read_parquet_if_exists(run_dir / "ohlcv.parquet")
    indicators = _read_parquet_if_exists(run_dir / "indicators.parquet")
    states_flow = _read_parquet_if_exists(run_dir / "states_flow.parquet")
    states_hmm = _read_parquet_if_exists(run_dir / "states_hmm.parquet")
    overlay_exec = _read_parquet_if_exists(run_dir / "overlay_exec.parquet")
    overlay_policy = _read_parquet_if_exists(run_dir / "overlay_policy.parquet")

    join_key_cols = [("ticker", pl.String), ("trade_date", pl.Date)]
    merged = _prepare_join_df(ohlcv, join_key_cols)

    for part in [indicators, states_flow, states_hmm, overlay_exec, overlay_policy]:
        if part.height == 0:
            continue
        part_ready = _prepare_join_df(part, join_key_cols)
        extra = [c for c in part_ready.columns if c not in {"ticker", "trade_date"}]
        if not extra:
            continue
        merged = merged.join(part_ready.select(["ticker", "trade_date", *extra]), on=["ticker", "trade_date"], how="left")

    if "trade_date" in merged.columns:
        merged = merged.with_columns(
            pl.col("trade_date").cast(pl.String, strict=False).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        ).sort("trade_date")

    return TickerCacheBundle(
        run_dir=run_dir,
        ticker=ticker,
        run_id=run_id,
        meta=meta,
        summary=summary,
        merged=merged,
    )


def write_cache_run(
    *,
    run_dir: Path,
    ohlcv: pl.DataFrame,
    indicators: pl.DataFrame,
    states_flow: pl.DataFrame | None,
    states_hmm: pl.DataFrame | None,
    overlay_exec: pl.DataFrame | None,
    overlay_policy: pl.DataFrame | None,
    meta: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    write_parquet_atomically(ohlcv, run_dir / "ohlcv.parquet")
    write_parquet_atomically(indicators, run_dir / "indicators.parquet")
    if states_flow is not None and states_flow.height > 0:
        write_parquet_atomically(states_flow, run_dir / "states_flow.parquet")
    if states_hmm is not None and states_hmm.height > 0:
        write_parquet_atomically(states_hmm, run_dir / "states_hmm.parquet")
    if overlay_exec is not None and overlay_exec.height > 0:
        write_parquet_atomically(overlay_exec, run_dir / "overlay_exec.parquet")
    if overlay_policy is not None and overlay_policy.height > 0:
        write_parquet_atomically(overlay_policy, run_dir / "overlay_policy.parquet")

    payload_meta = {
        **meta,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "computed_ts": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomically(payload_meta, run_dir / "meta.json")
    write_json_atomically(summary, run_dir / "summary.json")


def persist_hmm_display_mapping(meta_path: Path, mapping: dict[str, Any]) -> None:
    """Persist/merge HMM display mapping metadata into cache meta.json."""

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing cache meta file: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    display_mappings = meta.get("display_mappings", {})
    if not isinstance(display_mappings, dict):
        display_mappings = {}
    display_mappings["hmm_group_display_v1"] = mapping
    meta["display_mappings"] = display_mappings
    write_json_atomically(meta, meta_path)
