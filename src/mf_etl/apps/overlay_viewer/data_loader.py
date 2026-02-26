"""Data loading helpers for overlay viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from mf_etl.apps.overlay_viewer.cache import (
    TickerCacheBundle,
    TickerCacheRunInfo,
    latest_cache_run,
    list_cache_runs,
    list_cached_tickers,
    load_cache_bundle,
)
from mf_etl.config import AppSettings
from mf_etl.ops.nightly_ops_discovery import discover_latest_pcp_pack


def _read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    raise ValueError(f"Unsupported file format: {path}")


def _load_subset_parquet(
    path: Path,
    *,
    ticker: str,
    date_from: str | None,
    date_to: str | None,
) -> pl.DataFrame:
    scan = pl.scan_parquet(path)
    names = scan.collect_schema().names()
    if "ticker" in names:
        scan = scan.filter(pl.col("ticker").cast(pl.String).str.to_uppercase() == ticker.upper())
    if date_from is not None and "trade_date" in names:
        scan = scan.filter(
            pl.col("trade_date").cast(pl.String, strict=False).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            >= pl.lit(date_from).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        )
    if date_to is not None and "trade_date" in names:
        scan = scan.filter(
            pl.col("trade_date").cast(pl.String, strict=False).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            <= pl.lit(date_to).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        )
    return scan.collect()


def load_rows_for_ticker(
    file_path: Path,
    *,
    ticker: str,
    date_from: str | None,
    date_to: str | None,
) -> pl.DataFrame:
    """Load ticker/date filtered rows from local parquet/csv."""

    if not file_path.exists():
        raise FileNotFoundError(f"Missing input file: {file_path}")

    if file_path.suffix.lower() == ".parquet":
        frame = _load_subset_parquet(
            file_path,
            ticker=ticker,
            date_from=date_from,
            date_to=date_to,
        )
    else:
        frame = _read_table(file_path)
        if "ticker" in frame.columns:
            frame = frame.filter(pl.col("ticker").cast(pl.String).str.to_uppercase() == ticker.upper())
        if date_from is not None and "trade_date" in frame.columns:
            frame = frame.filter(
                pl.col("trade_date").cast(pl.String, strict=False).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                >= pl.lit(date_from).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            )
        if date_to is not None and "trade_date" in frame.columns:
            frame = frame.filter(
                pl.col("trade_date").cast(pl.String, strict=False).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                <= pl.lit(date_to).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            )

    if "trade_date" in frame.columns:
        frame = frame.with_columns(
            pl.col("trade_date")
            .cast(pl.String, strict=False)
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            .alias("trade_date")
        )
    if "ticker" in frame.columns:
        frame = frame.with_columns(pl.col("ticker").cast(pl.String).str.to_uppercase())
    if "trade_date" in frame.columns:
        frame = frame.sort("trade_date")
    return frame


def discover_default_paths(settings: AppSettings) -> dict[str, Path | None]:
    """Return best-effort default artifact paths for viewer controls."""

    artifacts_root = settings.paths.artifacts_root
    data_root = settings.paths.data_root

    hmm_candidates = sorted((artifacts_root / "hmm_runs").glob("*/decoded_rows.parquet"))
    cluster_candidates = sorted((artifacts_root / "research_runs").glob("*/clustered_dataset_full.parquet"))
    dataset_candidates = sorted((data_root / "gold" / "datasets" / "ml_dataset_v1").glob("*/dataset.parquet"))
    hardening_candidates = sorted((artifacts_root / "validation_runs").glob("*/cluster_hardening"))

    return {
        "hmm_decoded": hmm_candidates[-1] if hmm_candidates else None,
        "cluster_full": cluster_candidates[-1] if cluster_candidates else None,
        "ml_dataset": dataset_candidates[-1] if dataset_candidates else None,
        "cluster_hardening": hardening_candidates[-1] if hardening_candidates else None,
    }


def load_symbol_master_tickers(settings: AppSettings) -> list[str]:
    """Load ticker universe from Bronze symbol master if available."""

    path = settings.paths.bronze_root / "symbol_master" / "symbol_master.parquet"
    if not path.exists():
        return []
    try:
        frame = pl.read_parquet(path)
    except Exception:
        return []
    if "ticker" not in frame.columns:
        return []
    tickers = (
        frame.select(pl.col("ticker").cast(pl.String).str.to_uppercase().alias("ticker"))
        .drop_nulls()
        .unique()
        .sort("ticker")
        .get_column("ticker")
        .to_list()
    )
    return [str(t) for t in tickers]


def normalize_ticker_input(ticker_raw: str, symbol_universe: list[str]) -> tuple[str, list[str]]:
    """Normalize ticker with .US fallback mapping for user input."""

    warnings: list[str] = []
    token = ticker_raw.strip().upper()
    if not token:
        return token, ["Ticker is empty."]

    if token in symbol_universe:
        return token, warnings

    if not token.endswith(".US") and f"{token}.US" in symbol_universe:
        warnings.append(f"Ticker normalized from {token} to {token}.US")
        return f"{token}.US", warnings

    if token.endswith(".US") and token[:-3] in symbol_universe:
        warnings.append(f"Ticker normalized from {token} to {token[:-3]}")
        return token[:-3], warnings

    # Secondary heuristic: prefix match (e.g., AAPL -> AAPL.US)
    prefix = f"{token}."
    matches = [sym for sym in symbol_universe if sym.startswith(prefix)]
    if len(matches) == 1:
        warnings.append(f"Ticker normalized from {token} to {matches[0]}")
        return matches[0], warnings

    return token, warnings


def source_debug_summary(path: Path) -> dict[str, Any]:
    """Quick debug summary for selected source file."""

    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "rows": None,
            "unique_tickers": None,
            "sample_tickers": [],
        }

    scan = pl.scan_parquet(path) if path.suffix.lower() == ".parquet" else pl.scan_csv(path)
    names = scan.collect_schema().names()
    rows = int(scan.select(pl.len().alias("n")).collect().item())

    unique_tickers: int | None = None
    sample_tickers: list[str] = []
    if "ticker" in names:
        unique_tickers = int(
            scan.select(pl.col("ticker").cast(pl.String).str.to_uppercase().n_unique().alias("n")).collect().item()
        )
        sample_tickers = [
            str(v)
            for v in scan.select(pl.col("ticker").cast(pl.String).str.to_uppercase().unique().sort().head(20)).collect().get_column("ticker").to_list()
        ]

    return {
        "path": str(path),
        "exists": True,
        "rows": rows,
        "unique_tickers": unique_tickers,
        "sample_tickers": sample_tickers,
        "columns": names,
    }


def load_pcp_packet(pcp_pack_dir: Path) -> dict[str, Any]:
    """Read PCP policy packet JSON."""

    policy_path = pcp_pack_dir / "production_policy_packet_v1.json"
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing production policy packet: {policy_path}")
    return json.loads(policy_path.read_text(encoding="utf-8"))


def resolve_latest_pcp_dir(settings: AppSettings) -> Path | None:
    """Discover latest PCP pack dir; return None if unavailable."""

    try:
        return discover_latest_pcp_pack(settings.paths.artifacts_root)
    except Exception:
        return None


def candidate_names_from_packet(packet: dict[str, Any]) -> list[str]:
    """Return stable candidate list from PCP packet."""

    candidates = packet.get("candidates", {})
    if not isinstance(candidates, dict):
        return []
    return sorted(str(name) for name in candidates.keys())


def list_cached_tickers_for_viewer(settings: AppSettings) -> list[str]:
    return list_cached_tickers(settings)


def list_cached_runs_for_ticker(settings: AppSettings, ticker: str) -> list[TickerCacheRunInfo]:
    return list_cache_runs(settings, ticker)


def latest_cached_run_for_ticker(settings: AppSettings, ticker: str) -> TickerCacheRunInfo | None:
    return latest_cache_run(settings, ticker)


def load_cached_bundle_for_viewer(run_dir: Path) -> TickerCacheBundle:
    return load_cache_bundle(run_dir)
