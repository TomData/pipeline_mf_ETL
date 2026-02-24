"""Adapters to normalize state-labeled research datasets into a common schema."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import logging
from pathlib import Path
from typing import Any, Literal

import polars as pl

from mf_etl.research.forward_labels import add_forward_outcomes

LOGGER = logging.getLogger(__name__)

InputType = Literal["auto", "hmm", "cluster", "generic"]
StateLabelType = Literal["hmm", "cluster", "flow"]

FORWARD_COLUMNS: tuple[str, ...] = (
    "fwd_ret_5",
    "fwd_ret_10",
    "fwd_ret_20",
    "fwd_abs_ret_10",
    "fwd_vol_proxy_10",
)
OPTIONAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "tmf_21",
    "delta_flow_20",
    "flow_activity_20",
    "flow_bias_20",
    "oscillation_index_20",
    "close",
)


@dataclass(frozen=True, slots=True)
class AdaptedDataset:
    """Normalized validation dataset and metadata summary."""

    frame: pl.DataFrame
    input_type: InputType
    state_label_type: StateLabelType
    state_column: str
    summary: dict[str, Any]


def _metadata_path_for_dataset(input_file: Path) -> Path:
    return input_file.parent / "metadata.json"


def _read_input_frame(input_file: Path) -> pl.DataFrame:
    suffix = input_file.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(input_file)
    if suffix == ".csv":
        return pl.read_csv(input_file)
    raise ValueError("input-file must be .parquet or .csv")


def _resolve_input_type(frame: pl.DataFrame, input_type: InputType, state_col: str | None) -> InputType:
    if input_type != "auto":
        return input_type

    if "hmm_state" in frame.columns:
        return "hmm"
    if "cluster_id" in frame.columns or "cluster_label" in frame.columns:
        return "cluster"
    if state_col is not None:
        return "generic"
    if "flow_state_code" in frame.columns:
        return "generic"
    raise ValueError(
        "Could not auto-detect input type. Provide --input-type and --state-col for generic datasets."
    )


def _state_mapping_for_input(
    frame: pl.DataFrame,
    *,
    input_type: InputType,
    state_col: str | None,
) -> tuple[StateLabelType, str, str | None]:
    if input_type == "hmm":
        if "hmm_state" not in frame.columns:
            raise ValueError("hmm input requires hmm_state column.")
        return "hmm", "hmm_state", None

    if input_type == "cluster":
        if "cluster_id" in frame.columns:
            return "cluster", "cluster_id", None
        if "cluster_label" in frame.columns:
            return "cluster", "cluster_label", None
        raise ValueError("cluster input requires cluster_id or cluster_label column.")

    if state_col is None:
        if "flow_state_code" in frame.columns:
            state_col = "flow_state_code"
        else:
            raise ValueError("generic input requires --state-col when flow_state_code is absent.")

    if state_col not in frame.columns:
        raise ValueError(f"State column not found: {state_col}")

    label_type: StateLabelType = "flow" if state_col.startswith("flow_state") else "flow"
    label_name_col = "flow_state_label" if "flow_state_label" in frame.columns else None
    return label_type, state_col, label_name_col


def normalize_forward_outcomes(df: pl.DataFrame, *, columns: tuple[str, ...] = FORWARD_COLUMNS) -> pl.DataFrame:
    """Replace non-finite forward-outcome values with nulls."""

    existing = [column for column in columns if column in df.columns]
    if not existing:
        return df

    exprs: list[pl.Expr] = []
    for column in existing:
        value = pl.col(column).cast(pl.Float64, strict=False)
        exprs.append(
            pl.when(value.is_finite().fill_null(False))
            .then(value)
            .otherwise(None)
            .alias(column)
        )
    return df.with_columns(exprs)


def adapt_validation_dataset(
    input_file: Path,
    *,
    input_type: InputType = "auto",
    state_col: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    sample_frac: float | None = None,
    logger: logging.Logger | None = None,
) -> AdaptedDataset:
    """Load and normalize a state-labeled dataset for validation harness processing."""

    effective_logger = logger or LOGGER
    if sample_frac is not None and (sample_frac <= 0 or sample_frac > 1):
        raise ValueError("sample_frac must be in (0, 1].")

    frame = _read_input_frame(input_file)
    resolved_type = _resolve_input_type(frame, input_type, state_col)
    state_label_type, resolved_state_col, label_name_col = _state_mapping_for_input(
        frame,
        input_type=resolved_type,
        state_col=state_col,
    )

    required_base = ["ticker", "trade_date", resolved_state_col]
    missing = [column for column in required_base if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns after type resolution: {', '.join(missing)}")

    metadata: dict[str, Any] = {}
    metadata_path = _metadata_path_for_dataset(input_file)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    mapped = frame.with_columns(
        [
            pl.col("ticker").cast(pl.String).str.strip_chars().str.to_uppercase().alias("ticker"),
            pl.col("trade_date").cast(pl.Date, strict=False).alias("trade_date"),
            pl.col(resolved_state_col).cast(pl.Int32, strict=False).alias("state_id"),
            pl.lit(state_label_type).alias("state_label_type"),
        ]
    )

    if label_name_col is not None and label_name_col in mapped.columns:
        mapped = mapped.with_columns(
            pl.col(label_name_col).cast(pl.String, strict=False).alias("state_label_name")
        )
    else:
        mapped = mapped.with_columns(
            pl.col("state_id").cast(pl.String).alias("state_label_name")
        )

    if "hmm_state" not in mapped.columns:
        mapped = mapped.with_columns(pl.lit(None, dtype=pl.Int32).alias("hmm_state"))
    if "cluster_id" not in mapped.columns:
        if "cluster_label" in mapped.columns:
            mapped = mapped.with_columns(pl.col("cluster_label").cast(pl.Int32, strict=False).alias("cluster_id"))
        else:
            mapped = mapped.with_columns(pl.lit(None, dtype=pl.Int32).alias("cluster_id"))
    if "flow_state_code" not in mapped.columns:
        mapped = mapped.with_columns(pl.lit(None, dtype=pl.Int32).alias("flow_state_code"))
    if "flow_state_label" not in mapped.columns:
        mapped = mapped.with_columns(pl.lit(None, dtype=pl.String).alias("flow_state_label"))
    if "source_run_id" not in mapped.columns:
        mapped = mapped.with_columns(pl.lit(None, dtype=pl.String).alias("source_run_id"))
    if "built_ts" not in mapped.columns:
        mapped = mapped.with_columns(pl.lit(None, dtype=pl.Datetime).alias("built_ts"))

    if date_from is not None:
        mapped = mapped.filter(pl.col("trade_date") >= pl.lit(date_from))
    if date_to is not None:
        mapped = mapped.filter(pl.col("trade_date") <= pl.lit(date_to))

    mapped = mapped.filter(pl.col("trade_date").is_not_null() & pl.col("state_id").is_not_null())
    missing_forward = [column for column in FORWARD_COLUMNS if column not in mapped.columns]
    if missing_forward and "close" in mapped.columns:
        mapped = add_forward_outcomes(mapped, windows=(5, 10, 20))
    mapped = normalize_forward_outcomes(mapped)

    selected_columns = [
        "ticker",
        "trade_date",
        "trade_dt",
        "state_label_type",
        "state_id",
        "state_label_name",
        "flow_state_code",
        "flow_state_label",
        "cluster_id",
        "hmm_state",
        *FORWARD_COLUMNS,
        *OPTIONAL_FEATURE_COLUMNS,
        "source_run_id",
        "built_ts",
    ]
    ordered = [column for column in selected_columns if column in mapped.columns]
    trailing = [column for column in mapped.columns if column not in ordered]
    normalized = mapped.select(ordered + trailing)

    if sample_frac is not None and sample_frac < 1.0 and normalized.height > 0:
        normalized = normalized.sample(
            fraction=sample_frac,
            with_replacement=False,
            shuffle=True,
            seed=42,
        )

    normalized = normalized.sort([column for column in ("ticker", "trade_date") if column in normalized.columns])

    ticker_count = (
        int(normalized.select(pl.col("ticker").n_unique()).item())
        if normalized.height > 0
        else 0
    )
    state_count = (
        int(normalized.select(pl.col("state_id").n_unique()).item())
        if normalized.height > 0
        else 0
    )
    if normalized.height > 0:
        bounds = normalized.select(
            [
                pl.col("trade_date").min().alias("min_trade_date"),
                pl.col("trade_date").max().alias("max_trade_date"),
            ]
        ).to_dicts()[0]
        min_trade_date = bounds["min_trade_date"]
        max_trade_date = bounds["max_trade_date"]
    else:
        min_trade_date = None
        max_trade_date = None

    summary = {
        "input_file": str(input_file),
        "metadata_file": str(metadata_path) if metadata_path.exists() else None,
        "metadata": metadata,
        "input_type_requested": input_type,
        "input_type_resolved": resolved_type,
        "state_label_type": state_label_type,
        "state_column": resolved_state_col,
        "rows": normalized.height,
        "ticker_count": ticker_count,
        "state_count": state_count,
        "min_trade_date": min_trade_date.isoformat() if min_trade_date is not None else None,
        "max_trade_date": max_trade_date.isoformat() if max_trade_date is not None else None,
        "sample_frac": sample_frac,
        "date_from": date_from.isoformat() if date_from is not None else None,
        "date_to": date_to.isoformat() if date_to is not None else None,
        "forward_columns_present": [column for column in FORWARD_COLUMNS if column in normalized.columns],
    }
    effective_logger.info(
        "validation.adapter rows=%s states=%s tickers=%s min_date=%s max_date=%s input_type=%s",
        summary["rows"],
        summary["state_count"],
        summary["ticker_count"],
        summary["min_trade_date"],
        summary["max_trade_date"],
        summary["input_type_resolved"],
    )

    return AdaptedDataset(
        frame=normalized,
        input_type=resolved_type,
        state_label_type=state_label_type,
        state_column=resolved_state_col,
        summary=summary,
    )
