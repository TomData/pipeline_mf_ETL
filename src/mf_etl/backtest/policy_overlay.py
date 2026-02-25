"""Row-level cluster policy overlay helpers for hybrid backtest gating."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mf_etl.backtest.policy_overlay_models import (
    OverlayCoverageMode,
    OverlayMode,
    OverlayUnknownHandling,
    PolicyOverlayResult,
)

LOGGER = logging.getLogger(__name__)

_ALLOWED_KEYS = {"ticker", "trade_date", "trade_dt", "timeframe"}


class OverlayCoveragePolicyError(ValueError):
    """Raised when strict overlay coverage policy fails."""


def _read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    raise ValueError(f"Unsupported overlay input format: {path}")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _normalize_join_keys(df: pl.DataFrame, *, join_keys: list[str]) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for key in join_keys:
        if key not in df.columns:
            raise ValueError(f"Missing required join key column: {key}")
        if key == "ticker":
            exprs.append(pl.col("ticker").cast(pl.String).str.strip_chars().str.to_uppercase().alias("ticker"))
        elif key == "trade_date":
            exprs.append(
                pl.col("trade_date").cast(pl.String, strict=False).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("trade_date")
            )
        elif key == "trade_dt":
            exprs.append(pl.col("trade_dt").cast(pl.Datetime("us"), strict=False).alias("trade_dt"))
        elif key == "timeframe":
            exprs.append(pl.col("timeframe").cast(pl.String).str.strip_chars().str.to_uppercase().alias("timeframe"))
        else:
            exprs.append(pl.col(key).cast(pl.String).alias(key))
    return df.with_columns(exprs)


def _duplicate_keys(df: pl.DataFrame, *, join_keys: list[str], dataset: str) -> pl.DataFrame:
    if df.height == 0:
        return pl.DataFrame(schema={"dataset": pl.String, "duplicate_count": pl.Int64, **{k: pl.String for k in join_keys}})

    dup = (
        df.group_by(join_keys)
        .agg(pl.len().alias("duplicate_count"))
        .filter(pl.col("duplicate_count") > 1)
    )
    if dup.height == 0:
        return pl.DataFrame(schema={"dataset": pl.String, "duplicate_count": pl.Int64, **{k: pl.String for k in join_keys}})

    return dup.with_columns(
        [
            pl.lit(dataset).alias("dataset"),
            *[pl.col(k).cast(pl.String).alias(k) for k in join_keys],
        ]
    ).select(["dataset", *join_keys, "duplicate_count"])


def _load_policy_table(cluster_hardening_dir: Path) -> pl.DataFrame:
    policy_path = cluster_hardening_dir / "cluster_hardening_policy.json"
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing overlay policy file: {policy_path}")
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    per_state = payload.get("per_state")
    if not isinstance(per_state, list):
        raise ValueError(f"Invalid cluster_hardening_policy.json at {policy_path}")

    rows: list[dict[str, Any]] = []
    for item in per_state:
        if not isinstance(item, dict):
            continue
        state_id = item.get("state_id")
        try:
            cid = int(state_id)
        except (TypeError, ValueError):
            continue
        policy_class = str(item.get("class_label") or "UNKNOWN").upper()
        direction_hint = str(item.get("allow_direction_hint") or "UNCONFIRMED").upper()
        rows.append(
            {
                "overlay_cluster_state": cid,
                "overlay_policy_class": policy_class,
                "overlay_direction_hint": direction_hint,
                "overlay_tradability_score": _safe_float(item.get("tradability_score")),
            }
        )

    out = pl.DataFrame(rows) if rows else pl.DataFrame(schema={"overlay_cluster_state": pl.Int32})
    if out.height == 0:
        raise ValueError(f"No per_state rows found in {policy_path}")
    known_classes = int(
        out.filter(pl.col("overlay_policy_class").cast(pl.String).is_in(["ALLOW", "WATCH", "BLOCK"])).height
    )
    if known_classes == 0:
        raise ValueError(
            f"Policy classes missing in {policy_path}; expected at least one ALLOW/WATCH/BLOCK class_label entry."
        )
    return out.with_columns(
        pl.col("overlay_cluster_state").cast(pl.Int32),
        pl.col("overlay_policy_class").cast(pl.String),
        pl.col("overlay_direction_hint").cast(pl.String),
        pl.col("overlay_tradability_score").cast(pl.Float64, strict=False),
    )


def _overlay_pass_expr(*, mode: OverlayMode, unknown_handling: OverlayUnknownHandling) -> pl.Expr:
    cls = pl.col("overlay_policy_class")
    allow_unknown = unknown_handling == "treat_unknown_as_pass"
    if mode == "none":
        return pl.lit(True)
    if mode == "allow_only":
        return cls == "ALLOW"
    if mode == "allow_watch":
        return cls.is_in(["ALLOW", "WATCH"])
    if mode == "allow_or_unknown":
        return cls.is_in(["ALLOW", "UNKNOWN"]) if allow_unknown else (cls == "ALLOW")

    # block_veto
    if allow_unknown:
        return cls != "BLOCK"
    return (~cls.is_in(["BLOCK", "UNKNOWN"]))


def _coverage_status_from_metrics(
    *,
    match_rate: float | None,
    unknown_rate: float | None,
    year_min_match_rate: float | None,
    duplicate_key_count_primary: int,
    duplicate_key_count_overlay: int,
    min_match_rate_warn: float,
    min_match_rate_fail: float,
    min_year_match_rate_warn: float,
    min_year_match_rate_fail: float,
    unknown_rate_warn: float,
    unknown_rate_fail: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if duplicate_key_count_primary > 0 or duplicate_key_count_overlay > 0:
        reasons.append(
            f"duplicate_keys primary={duplicate_key_count_primary} overlay={duplicate_key_count_overlay}"
        )
        return "FAIL_DUPLICATES", reasons

    if match_rate is not None and match_rate < min_match_rate_fail:
        reasons.append(
            f"match_rate_below_fail threshold={min_match_rate_fail:.4f} value={match_rate:.4f}"
        )
    if year_min_match_rate is not None and year_min_match_rate < min_year_match_rate_fail:
        reasons.append(
            f"year_min_match_rate_below_fail threshold={min_year_match_rate_fail:.4f} value={year_min_match_rate:.4f}"
        )
    if reasons:
        return "FAIL_LOW_MATCH", reasons

    if unknown_rate is not None and unknown_rate > unknown_rate_fail:
        reasons.append(
            f"unknown_rate_above_fail threshold={unknown_rate_fail:.4f} value={unknown_rate:.4f}"
        )
        return "FAIL_HIGH_UNKNOWN", reasons

    if match_rate is not None and match_rate < min_match_rate_warn:
        reasons.append(
            f"match_rate_below_warn threshold={min_match_rate_warn:.4f} value={match_rate:.4f}"
        )
    if year_min_match_rate is not None and year_min_match_rate < min_year_match_rate_warn:
        reasons.append(
            f"year_min_match_rate_below_warn threshold={min_year_match_rate_warn:.4f} value={year_min_match_rate:.4f}"
        )
    if reasons:
        return "WARN_LOW_MATCH", reasons

    if unknown_rate is not None and unknown_rate > unknown_rate_warn:
        reasons.append(
            f"unknown_rate_above_warn threshold={unknown_rate_warn:.4f} value={unknown_rate:.4f}"
        )
        return "WARN_HIGH_UNKNOWN", reasons

    return "OK", reasons


def apply_policy_overlay(
    frame: pl.DataFrame,
    *,
    overlay_cluster_file: Path,
    overlay_cluster_hardening_dir: Path,
    overlay_mode: OverlayMode,
    join_keys: list[str],
    unknown_handling: OverlayUnknownHandling,
    min_match_rate_warn: float,
    min_match_rate_fail: float,
    min_year_match_rate_warn: float,
    min_year_match_rate_fail: float,
    unknown_rate_warn: float,
    unknown_rate_fail: float,
    coverage_mode: OverlayCoverageMode,
    logger: logging.Logger | None = None,
) -> PolicyOverlayResult:
    """Attach cluster hardening policy to primary rows and compute gating columns."""

    effective_logger = logger or LOGGER
    keys = [k.strip() for k in join_keys if k.strip() != ""]
    if not keys:
        keys = ["ticker", "trade_date"]
    unsupported = [k for k in keys if k not in _ALLOWED_KEYS]
    if unsupported:
        raise ValueError(f"Unsupported overlay join key(s): {','.join(unsupported)}")

    if frame.height == 0:
        out = frame.with_columns(
            pl.lit(None).cast(pl.Int32).alias("overlay_cluster_state"),
            pl.lit("UNKNOWN").alias("overlay_policy_class"),
            pl.lit("UNCONFIRMED").alias("overlay_direction_hint"),
            pl.lit(None).cast(pl.Float64).alias("overlay_tradability_score"),
            pl.lit("UNMATCHED_PRIMARY").alias("overlay_join_status"),
            pl.lit(overlay_mode).alias("overlay_mode"),
            pl.lit(True).alias("overlay_allow_signal"),
            pl.lit(True).alias("overlay_enabled"),
        )
        return PolicyOverlayResult(
            frame=out,
            join_summary={
                "overlay_enabled": True,
                "overlay_mode": overlay_mode,
                "join_keys": keys,
                "primary_rows_total": 0,
                "primary_rows_matched": 0,
                "primary_rows_unmatched": 0,
                "match_rate": None,
                "overlay_rows_total": 0,
                "overlay_rows_used": 0,
                "duplicate_key_count_primary": 0,
                "duplicate_key_count_overlay": 0,
                "overlay_allow_rate": None,
                "overlay_watch_rate": None,
                "overlay_block_rate": None,
                "overlay_unknown_rate": None,
                "unknown_rate": None,
                "unknown_handling": unknown_handling,
                "year_min_match_rate": None,
                "year_p10_match_rate": None,
                "year_fail_years": [],
                "coverage_status": "OK",
            },
            coverage_verdict={
                "coverage_mode": coverage_mode,
                "thresholds": {
                    "min_match_rate_warn": min_match_rate_warn,
                    "min_match_rate_fail": min_match_rate_fail,
                    "min_year_match_rate_warn": min_year_match_rate_warn,
                    "min_year_match_rate_fail": min_year_match_rate_fail,
                    "unknown_rate_warn": unknown_rate_warn,
                    "unknown_rate_fail": unknown_rate_fail,
                },
                "summary": {
                    "match_rate": None,
                    "unknown_rate": None,
                    "year_min_match_rate": None,
                    "year_p10_match_rate": None,
                    "year_fail_years": [],
                },
                "status": "OK",
                "reasons": [],
                "strict_fail_triggered": False,
                "unknown_handling": unknown_handling,
            },
            coverage_by_year=pl.DataFrame(
                schema={
                    "year": pl.Int32,
                    "primary_rows": pl.Int64,
                    "matched_rows": pl.Int64,
                    "unmatched_rows": pl.Int64,
                    "unknown_rows": pl.Int64,
                    "match_rate": pl.Float64,
                    "unknown_rate": pl.Float64,
                }
            ),
            duplicate_keys=pl.DataFrame(schema={"dataset": pl.String}),
            policy_mix_on_primary=pl.DataFrame(schema={"overlay_policy_class": pl.String}),
        )

    primary = _normalize_join_keys(frame, join_keys=keys)
    overlay_raw = _normalize_join_keys(_read_table(overlay_cluster_file), join_keys=keys)
    if "cluster_id" not in overlay_raw.columns:
        raise ValueError("Overlay cluster file must contain cluster_id")

    policy_table = _load_policy_table(overlay_cluster_hardening_dir)
    overlay = overlay_raw.with_columns(
        pl.col("cluster_id").cast(pl.Int32, strict=False).alias("overlay_cluster_state")
    ).join(policy_table, on="overlay_cluster_state", how="left")

    overlay = overlay.with_columns(
        pl.coalesce([pl.col("overlay_policy_class"), pl.lit("UNKNOWN")]).alias("overlay_policy_class"),
        pl.coalesce([pl.col("overlay_direction_hint"), pl.lit("UNCONFIRMED")]).alias("overlay_direction_hint"),
        pl.col("overlay_tradability_score").cast(pl.Float64, strict=False).alias("overlay_tradability_score"),
    )

    primary_dup = _duplicate_keys(primary, join_keys=keys, dataset="primary")
    overlay_dup = _duplicate_keys(overlay, join_keys=keys, dataset="overlay")

    # MVP simplification: deterministic dedupe uses first row after key-sort.
    overlay_dedup = overlay.sort(keys).unique(subset=keys, keep="first", maintain_order=True)
    overlay_join_cols = [*keys, "overlay_cluster_state", "overlay_policy_class", "overlay_direction_hint", "overlay_tradability_score"]

    joined = primary.join(overlay_dedup.select(overlay_join_cols), on=keys, how="left")
    joined = joined.with_columns(
        pl.col("overlay_cluster_state").cast(pl.Int32, strict=False),
        pl.coalesce([pl.col("overlay_policy_class"), pl.lit("UNKNOWN")]).alias("overlay_policy_class"),
        pl.coalesce([pl.col("overlay_direction_hint"), pl.lit("UNCONFIRMED")]).alias("overlay_direction_hint"),
        pl.col("overlay_tradability_score").cast(pl.Float64, strict=False),
        pl.when(pl.col("overlay_cluster_state").is_null())
        .then(pl.lit("UNMATCHED_PRIMARY"))
        .otherwise(pl.lit("MATCHED"))
        .alias("overlay_join_status"),
        pl.lit(overlay_mode).alias("overlay_mode"),
        _overlay_pass_expr(mode=overlay_mode, unknown_handling=unknown_handling).alias(
            "overlay_allow_signal"
        ),
        pl.lit(True).alias("overlay_enabled"),
    )

    primary_rows_total = int(joined.height)
    matched = int(joined.filter(pl.col("overlay_join_status") == "MATCHED").height)
    unmatched = primary_rows_total - matched
    match_rate = _safe_float((matched / primary_rows_total) if primary_rows_total > 0 else None)

    if overlay_mode != "none" and (match_rate is None or match_rate <= 0.0):
        raise ValueError("Overlay mode requires non-zero join coverage, but overlay match_rate is 0.")

    overlay_rows_used = int(
        joined.filter(pl.col("overlay_join_status") == "MATCHED").select(keys).unique().height
    )
    overlay_rows_total = int(overlay_dedup.height)

    policy_mix = (
        joined.group_by("overlay_policy_class")
        .agg(pl.len().alias("rows"))
        .with_columns((pl.col("rows") / pl.lit(max(1, primary_rows_total))).alias("row_share"))
        .sort("rows", descending=True)
    )

    allow_rows = int(joined.filter(pl.col("overlay_policy_class") == "ALLOW").height)
    watch_rows = int(joined.filter(pl.col("overlay_policy_class") == "WATCH").height)
    block_rows = int(joined.filter(pl.col("overlay_policy_class") == "BLOCK").height)
    unknown_rows = int(joined.filter(pl.col("overlay_policy_class") == "UNKNOWN").height)

    coverage_by_year = (
        joined.with_columns(pl.col("trade_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("primary_rows"),
            (pl.col("overlay_join_status") == "MATCHED").cast(pl.Int64).sum().alias("matched_rows"),
            (pl.col("overlay_join_status") == "UNMATCHED_PRIMARY").cast(pl.Int64).sum().alias("unmatched_rows"),
            (pl.col("overlay_policy_class") == "UNKNOWN").cast(pl.Int64).sum().alias("unknown_rows"),
        )
        .with_columns(
            (pl.col("matched_rows") / pl.col("primary_rows").cast(pl.Float64)).alias("match_rate"),
            (pl.col("unknown_rows") / pl.col("primary_rows").cast(pl.Float64)).alias("unknown_rate"),
        )
        .sort("year")
    )

    duplicates = (
        pl.concat([primary_dup, overlay_dup], how="vertical") if (primary_dup.height > 0 or overlay_dup.height > 0) else pl.DataFrame(schema={"dataset": pl.String})
    )

    year_min_match_rate = (
        _safe_float(coverage_by_year.select(pl.col("match_rate").cast(pl.Float64, strict=False).min()).item())
        if coverage_by_year.height > 0
        else None
    )
    year_p10_match_rate = (
        _safe_float(
            coverage_by_year.select(
                pl.col("match_rate").cast(pl.Float64, strict=False).quantile(0.10)
            ).item()
        )
        if coverage_by_year.height > 0
        else None
    )
    year_fail_years = (
        coverage_by_year.filter(
            pl.col("match_rate").cast(pl.Float64, strict=False) < float(min_year_match_rate_fail)
        )
        .get_column("year")
        .cast(pl.Int64, strict=False)
        .drop_nulls()
        .to_list()
        if coverage_by_year.height > 0
        else []
    )

    duplicate_key_count_primary = (
        int(primary_dup.select(pl.col("duplicate_count").sum()).item() or 0) if primary_dup.height > 0 else 0
    )
    duplicate_key_count_overlay = (
        int(overlay_dup.select(pl.col("duplicate_count").sum()).item() or 0) if overlay_dup.height > 0 else 0
    )
    unknown_rate = _safe_float((unknown_rows / primary_rows_total) if primary_rows_total > 0 else None)
    coverage_status, coverage_reasons = _coverage_status_from_metrics(
        match_rate=match_rate,
        unknown_rate=unknown_rate,
        year_min_match_rate=year_min_match_rate,
        duplicate_key_count_primary=duplicate_key_count_primary,
        duplicate_key_count_overlay=duplicate_key_count_overlay,
        min_match_rate_warn=float(min_match_rate_warn),
        min_match_rate_fail=float(min_match_rate_fail),
        min_year_match_rate_warn=float(min_year_match_rate_warn),
        min_year_match_rate_fail=float(min_year_match_rate_fail),
        unknown_rate_warn=float(unknown_rate_warn),
        unknown_rate_fail=float(unknown_rate_fail),
    )
    strict_fail_triggered = bool(coverage_mode == "strict_fail" and coverage_status.startswith("FAIL"))

    if coverage_status.startswith("WARN"):
        effective_logger.warning(
            "backtest.overlay.coverage_warn mode=%s status=%s match_rate=%s unknown_rate=%s reasons=%s",
            overlay_mode,
            coverage_status,
            match_rate,
            unknown_rate,
            ";".join(coverage_reasons),
        )

    join_summary = {
        "overlay_enabled": True,
        "overlay_mode": overlay_mode,
        "join_keys": keys,
        "primary_rows_total": primary_rows_total,
        "primary_rows_matched": matched,
        "primary_rows_unmatched": unmatched,
        "match_rate": match_rate,
        "overlay_rows_total": overlay_rows_total,
        "overlay_rows_used": overlay_rows_used,
        "duplicate_key_count_primary": duplicate_key_count_primary,
        "duplicate_key_count_overlay": duplicate_key_count_overlay,
        "overlay_allow_rate": _safe_float((allow_rows / primary_rows_total) if primary_rows_total > 0 else None),
        "overlay_watch_rate": _safe_float((watch_rows / primary_rows_total) if primary_rows_total > 0 else None),
        "overlay_block_rate": _safe_float((block_rows / primary_rows_total) if primary_rows_total > 0 else None),
        "overlay_unknown_rate": unknown_rate,
        "unknown_rate": unknown_rate,
        "unknown_handling": unknown_handling,
        "year_min_match_rate": year_min_match_rate,
        "year_p10_match_rate": year_p10_match_rate,
        "year_fail_years": year_fail_years,
        "coverage_status": coverage_status,
        "coverage_reasons": coverage_reasons,
        "coverage_mode": coverage_mode,
        "min_match_rate_warn": float(min_match_rate_warn),
        "min_match_rate_fail": float(min_match_rate_fail),
        "min_year_match_rate_warn": float(min_year_match_rate_warn),
        "min_year_match_rate_fail": float(min_year_match_rate_fail),
        "unknown_rate_warn": float(unknown_rate_warn),
        "unknown_rate_fail": float(unknown_rate_fail),
    }

    coverage_verdict: dict[str, Any] = {
        "coverage_mode": coverage_mode,
        "unknown_handling": unknown_handling,
        "thresholds": {
            "min_match_rate_warn": float(min_match_rate_warn),
            "min_match_rate_fail": float(min_match_rate_fail),
            "min_year_match_rate_warn": float(min_year_match_rate_warn),
            "min_year_match_rate_fail": float(min_year_match_rate_fail),
            "unknown_rate_warn": float(unknown_rate_warn),
            "unknown_rate_fail": float(unknown_rate_fail),
        },
        "summary": {
            "match_rate": match_rate,
            "unknown_rate": unknown_rate,
            "year_min_match_rate": year_min_match_rate,
            "year_p10_match_rate": year_p10_match_rate,
            "year_fail_years": year_fail_years,
            "duplicate_key_count_primary": duplicate_key_count_primary,
            "duplicate_key_count_overlay": duplicate_key_count_overlay,
        },
        "status": coverage_status,
        "reasons": coverage_reasons,
        "strict_fail_triggered": strict_fail_triggered,
    }

    return PolicyOverlayResult(
        frame=joined,
        join_summary=join_summary,
        coverage_verdict=coverage_verdict,
        coverage_by_year=coverage_by_year,
        duplicate_keys=duplicates,
        policy_mix_on_primary=policy_mix,
    )


def build_overlay_performance_breakdown(trades: pl.DataFrame) -> pl.DataFrame:
    """Build overlay policy-class trade performance breakdown."""

    if trades.height == 0 or "entry_overlay_policy_class" not in trades.columns:
        return pl.DataFrame(schema={"entry_overlay_policy_class": pl.String})

    valid = trades.filter(pl.col("is_valid_trade") == True).with_columns(
        pl.col("net_return").cast(pl.Float64, strict=False)
    )
    if valid.height == 0:
        return pl.DataFrame(schema={"entry_overlay_policy_class": pl.String})

    grouped = valid.group_by("entry_overlay_policy_class").agg(
        pl.len().alias("trade_count"),
        pl.col("net_return").mean().alias("avg_return"),
        pl.col("net_return").median().alias("median_return"),
        pl.col("net_return").std(ddof=0).alias("return_std"),
        pl.col("net_return").quantile(0.10).alias("ret_p10"),
        pl.col("net_return").quantile(0.90).alias("ret_p90"),
        pl.col("net_return").min().alias("worst_trade_return"),
        pl.col("net_return").max().alias("best_trade_return"),
        pl.col("net_return").filter(pl.col("net_return") > 0).len().alias("win_count"),
        pl.col("net_return").filter(pl.col("net_return") > 0).sum().alias("gross_win"),
        pl.col("net_return").filter(pl.col("net_return") < 0).sum().alias("gross_loss"),
    )
    return grouped.with_columns(
        (pl.col("win_count") / pl.col("trade_count")).alias("win_rate"),
        pl.when(pl.col("gross_loss") == 0)
        .then(None)
        .otherwise(pl.col("gross_win") / pl.col("gross_loss").abs())
        .alias("profit_factor"),
        pl.when(pl.col("avg_return").abs() < 1e-12)
        .then(None)
        .otherwise(pl.col("return_std") / pl.col("avg_return").abs())
        .alias("ret_cv"),
    ).sort("trade_count", descending=True)
