"""Input adapters that normalize FLOW/HMM/CLUSTER rows to a common backtest schema."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mf_etl.backtest.models import InputType, NormalizedBacktestData

LOGGER = logging.getLogger(__name__)

FLOW_STATE_LABELS: dict[int, str] = {
    0: "S0_QUIET",
    1: "S1_EARLY_DEMAND",
    2: "S2_PERSISTENT_DEMAND",
    3: "S3_EARLY_SUPPLY",
    4: "S4_PERSISTENT_SUPPLY",
}


def _read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    raise ValueError(f"Unsupported input file format: {path}")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _normalize_trade_date(df: pl.DataFrame) -> pl.DataFrame:
    if "trade_date" not in df.columns:
        raise ValueError("Input is missing required column trade_date")
    dtype = df.schema.get("trade_date")
    if dtype == pl.Date:
        return df
    return df.with_columns(
        pl.col("trade_date").cast(pl.String, strict=False).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    )


def _ensure_ohlc(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Ensure open/high/low/close exist; derive missing OHLC from close when needed."""

    diagnostics = {
        "derived_open_from_close_rows": 0,
        "derived_high_from_close_rows": 0,
        "derived_low_from_close_rows": 0,
    }
    if "close" not in df.columns:
        raise ValueError("Input is missing required column close")

    out = df
    if "open" not in out.columns:
        out = out.with_columns(pl.col("close").alias("open"))
        diagnostics["derived_open_from_close_rows"] = out.height
    if "high" not in out.columns:
        out = out.with_columns(pl.col("close").alias("high"))
        diagnostics["derived_high_from_close_rows"] = out.height
    if "low" not in out.columns:
        out = out.with_columns(pl.col("close").alias("low"))
        diagnostics["derived_low_from_close_rows"] = out.height

    return out, diagnostics


def _normalize_state_columns(df: pl.DataFrame, *, input_type: InputType) -> pl.DataFrame:
    if input_type == "flow":
        if "flow_state_code" not in df.columns:
            raise ValueError("FLOW input requires flow_state_code")
        label_expr = (
            pl.col("flow_state_code")
            .cast(pl.Int32, strict=False)
            .replace_strict(FLOW_STATE_LABELS, default="FLOW_UNKNOWN")
        )
        return df.with_columns(
            [
                pl.lit("FLOW").alias("state_source"),
                pl.col("flow_state_code").cast(pl.Int32, strict=False).alias("state_id"),
                label_expr.alias("state_label"),
            ]
        )

    if input_type == "hmm":
        if "hmm_state" not in df.columns:
            raise ValueError("HMM input requires hmm_state")
        return df.with_columns(
            [
                pl.lit("HMM").alias("state_source"),
                pl.col("hmm_state").cast(pl.Int32, strict=False).alias("state_id"),
                pl.format("HMM_{}", pl.col("hmm_state").cast(pl.Int32, strict=False)).alias("state_label"),
            ]
        )

    if "cluster_id" not in df.columns:
        raise ValueError("CLUSTER input requires cluster_id")
    return df.with_columns(
        [
            pl.lit("CLUSTER").alias("state_source"),
            pl.col("cluster_id").cast(pl.Int32, strict=False).alias("state_id"),
            pl.format("CLUSTER_{}", pl.col("cluster_id").cast(pl.Int32, strict=False)).alias("state_label"),
        ]
    )


def _optional_confidence_column(df: pl.DataFrame) -> pl.Expr:
    if "hmm_state_prob_max" in df.columns:
        return pl.col("hmm_state_prob_max").cast(pl.Float64, strict=False).alias("confidence")
    if "cluster_prob_max" in df.columns:
        return pl.col("cluster_prob_max").cast(pl.Float64, strict=False).alias("confidence")
    return pl.lit(None).cast(pl.Float64).alias("confidence")


def normalize_backtest_input(
    input_file: Path,
    *,
    input_type: InputType,
    logger: logging.Logger | None = None,
) -> NormalizedBacktestData:
    """Normalize one source table to the common backtest schema."""

    effective_logger = logger or LOGGER
    raw = _read_table(input_file)
    rows_in = raw.height

    raw = _normalize_trade_date(raw)
    raw, ohlc_diag = _ensure_ohlc(raw)
    raw = _normalize_state_columns(raw, input_type=input_type)

    if "ticker" not in raw.columns:
        raise ValueError("Input is missing required column ticker")

    if "exchange" not in raw.columns:
        raw = raw.with_columns(pl.lit("UNKNOWN").alias("exchange"))
    if "trade_dt" not in raw.columns:
        raw = raw.with_columns(pl.lit(None).cast(pl.Datetime("us")).alias("trade_dt"))

    if "run_id" not in raw.columns:
        raw = raw.with_columns(pl.lit(None).cast(pl.String).alias("run_id"))

    essentials = ["ticker", "trade_date", "open", "high", "low", "close", "state_id"]
    with_missing = raw.with_columns(
        [
            pl.any_horizontal([pl.col(c).is_null() for c in essentials]).alias("_missing_essential"),
            pl.any_horizontal(
                [
                    (~pl.col("open").cast(pl.Float64, strict=False).is_finite()),
                    (~pl.col("high").cast(pl.Float64, strict=False).is_finite()),
                    (~pl.col("low").cast(pl.Float64, strict=False).is_finite()),
                    (~pl.col("close").cast(pl.Float64, strict=False).is_finite()),
                ]
            ).fill_null(True).alias("_bad_price"),
        ]
    )

    dropped_missing = int(with_missing.filter(pl.col("_missing_essential")).height)
    dropped_bad_price = int(with_missing.filter(pl.col("_bad_price")).height)

    cleaned = with_missing.filter((~pl.col("_missing_essential")) & (~pl.col("_bad_price"))).drop(
        ["_missing_essential", "_bad_price"]
    )

    before_dedupe = cleaned.height
    cleaned = cleaned.sort(["ticker", "trade_date", "trade_dt"]).unique(
        subset=["ticker", "trade_date"], keep="first", maintain_order=True
    )
    deduped_rows = before_dedupe - cleaned.height

    built_ts = datetime.now(timezone.utc)
    exprs: list[pl.Expr] = [
        pl.col("ticker").cast(pl.String).str.strip_chars().str.to_uppercase().alias("ticker"),
        pl.col("exchange").cast(pl.String).str.strip_chars().str.to_uppercase().alias("exchange"),
        pl.col("trade_date").cast(pl.Date),
        pl.col("trade_dt").cast(pl.Datetime("us"), strict=False),
        pl.col("open").cast(pl.Float64, strict=False).alias("open"),
        pl.col("high").cast(pl.Float64, strict=False).alias("high"),
        pl.col("low").cast(pl.Float64, strict=False).alias("low"),
        pl.col("close").cast(pl.Float64, strict=False).alias("close"),
        pl.col("state_source").cast(pl.String),
        pl.col("state_id").cast(pl.Int32, strict=False),
        pl.col("state_label").cast(pl.String),
        pl.lit("UNCONFIRMED").alias("state_direction_hint"),
        pl.lit("NA").alias("state_class"),
        pl.lit(None).cast(pl.Float64).alias("state_score"),
        pl.lit(False).alias("signal_eligible"),
        _optional_confidence_column(cleaned),
        pl.concat_str(
            [
                pl.col("ticker").cast(pl.String),
                pl.col("trade_date").cast(pl.String),
                pl.col("state_source").cast(pl.String),
                pl.col("state_id").cast(pl.String),
            ],
            separator="|",
        ).alias("row_id"),
        pl.col("run_id").cast(pl.String, strict=False),
        pl.lit(built_ts).cast(pl.Datetime("us")).alias("built_ts"),
    ]
    if "volume" in cleaned.columns:
        exprs.append(pl.col("volume").cast(pl.Float64, strict=False).alias("volume"))
    else:
        exprs.append(pl.lit(None).cast(pl.Float64).alias("volume"))
    if "atr_pct_14" in cleaned.columns:
        exprs.append(pl.col("atr_pct_14").cast(pl.Float64, strict=False).alias("atr_pct_14"))
    else:
        exprs.append(pl.lit(None).cast(pl.Float64).alias("atr_pct_14"))
    if "fwd_ret_10" in cleaned.columns:
        exprs.append(pl.col("fwd_ret_10").cast(pl.Float64, strict=False).alias("fwd_ret_10"))
    else:
        exprs.append(pl.lit(None).cast(pl.Float64).alias("fwd_ret_10"))
    if "timeframe" in cleaned.columns:
        exprs.append(pl.col("timeframe").cast(pl.String).alias("timeframe"))
    else:
        exprs.append(pl.lit(None).cast(pl.String).alias("timeframe"))

    normalized = cleaned.select(exprs).sort(["ticker", "trade_date"])

    summary = {
        "input_file": str(input_file),
        "input_type": input_type,
        "rows_in": rows_in,
        "rows_out": int(normalized.height),
        "rows_dropped_missing_essential": dropped_missing,
        "rows_dropped_bad_price": dropped_bad_price,
        "rows_deduped_ticker_date": deduped_rows,
        **ohlc_diag,
        "ticker_count": int(normalized.select(pl.col("ticker").n_unique()).item()) if normalized.height > 0 else 0,
        "state_count": int(normalized.select(pl.col("state_id").n_unique()).item()) if normalized.height > 0 else 0,
        "min_date": (
            normalized.select(pl.col("trade_date").min()).item().isoformat() if normalized.height > 0 else None
        ),
        "max_date": (
            normalized.select(pl.col("trade_date").max()).item().isoformat() if normalized.height > 0 else None
        ),
    }

    effective_logger.info(
        "backtest.adapter normalized input_type=%s rows_in=%s rows_out=%s dropped_missing=%s dropped_bad_price=%s deduped=%s",
        input_type,
        summary["rows_in"],
        summary["rows_out"],
        summary["rows_dropped_missing_essential"],
        summary["rows_dropped_bad_price"],
        summary["rows_deduped_ticker_date"],
    )
    return NormalizedBacktestData(frame=normalized, summary=summary)
