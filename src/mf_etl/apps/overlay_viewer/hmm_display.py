"""HMM display transforms for overlay viewer (smoothing + grouping)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from mf_etl.apps.overlay_viewer.cache import persist_hmm_display_mapping

HMMDisplayMode = Literal["RAW", "SMOOTHED", "GROUPED", "SMOOTHED+GROUPED"]
HMMSmoothingMethod = Literal["mode", "median"]
HMMGroupingScheme = Literal["LONG_NEUTRAL_SHORT", "LONG_OTHER"]


@dataclass(frozen=True, slots=True)
class HMMDisplayConfig:
    display_mode: HMMDisplayMode = "RAW"
    smoothing_method: HMMSmoothingMethod = "mode"
    smoothing_window: int = 5
    grouping_scheme: HMMGroupingScheme = "LONG_NEUTRAL_SHORT"
    long_states_top_k: int = 1
    short_states_bottom_k: int = 1
    persist_mapping_in_cache_meta: bool = True


@dataclass(frozen=True, slots=True)
class HMMDisplayResult:
    frame: pl.DataFrame
    band_col: str
    summary: dict[str, Any]
    warnings: list[str]


def _as_state_array(series: pl.Series) -> np.ndarray:
    arr = series.cast(pl.Float64, strict=False).to_numpy()
    out = np.full(arr.shape[0], np.nan, dtype=float)
    mask = np.isfinite(arr)
    out[mask] = arr[mask]
    return out


def _rolling_mode_recent(values: np.ndarray, window: int) -> np.ndarray:
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - window + 1)
        vals = values[lo : i + 1]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        counts: dict[int, int] = {}
        seq = [int(v) for v in vals.tolist()]
        for v in seq:
            counts[v] = int(counts.get(v, 0) + 1)
        max_count = max(counts.values())
        tied = {k for k, c in counts.items() if c == max_count}
        chosen = None
        for v in reversed(seq):
            if v in tied:
                chosen = v
                break
        if chosen is not None:
            out[i] = float(chosen)
    return out


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - window + 1)
        vals = values[lo : i + 1]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[i] = float(int(np.rint(np.nanmedian(vals))))
    return out


def _compute_fwd_state_rank(frame: pl.DataFrame, state_col: str) -> list[int]:
    if state_col not in frame.columns or "close" not in frame.columns:
        return []

    ranked = (
        frame.sort(["ticker", "trade_date"])
        .with_columns(
            ((pl.col("close").shift(-10).over("ticker") / pl.col("close")) - 1.0)
            .cast(pl.Float64, strict=False)
            .alias("_fwd_ret_10")
        )
        .filter(
            pl.col(state_col).cast(pl.Float64, strict=False).is_finite()
            & pl.col("_fwd_ret_10").is_finite()
        )
        .group_by(state_col)
        .agg(pl.col("_fwd_ret_10").mean().alias("_mean_fwd"))
        .sort("_mean_fwd", descending=True)
    )
    if ranked.height == 0:
        return []
    return [int(v) for v in ranked.get_column(state_col).cast(pl.Int32, strict=False).to_list() if v is not None]


def _limit_unique(seq: list[int], limit: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for val in seq:
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
        if len(out) >= max(0, limit):
            break
    return out


def _persist_mapping(
    *,
    meta_path: Path,
    mapping: dict[str, Any],
) -> None:
    payload = {
        **mapping,
        "mapping_type": "display_only",
        "computed_ts": datetime.now(timezone.utc).isoformat(),
    }
    persist_hmm_display_mapping(meta_path, payload)


def apply_hmm_display_transform(
    frame: pl.DataFrame,
    *,
    config: HMMDisplayConfig,
    base_long_states: tuple[int, ...] = (),
    base_short_states: tuple[int, ...] = (),
    cached_meta: dict[str, Any] | None = None,
    cache_meta_path: Path | None = None,
) -> HMMDisplayResult:
    warnings: list[str] = []

    if "hmm_state" not in frame.columns:
        return HMMDisplayResult(
            frame=frame,
            band_col="hmm_state",
            summary={"display_mode": config.display_mode, "enabled": False, "reason": "missing_hmm_state"},
            warnings=["HMM display transform skipped: hmm_state column missing."],
        )

    if config.smoothing_window < 3:
        warnings.append("smoothing_window < 3 detected; clamped to 3.")
    window = max(3, int(config.smoothing_window))

    sorted_frame = frame.sort(["ticker", "trade_date"])
    out = sorted_frame

    state_arr = _as_state_array(sorted_frame.get_column("hmm_state"))
    if config.smoothing_method == "median":
        smoothed = _rolling_median(state_arr, window)
    else:
        smoothed = _rolling_mode_recent(state_arr, window)

    out = out.with_columns(
        pl.Series("hmm_state_smoothed", smoothed).cast(pl.Int32, strict=False)
    )

    mode_uses_smoothing = config.display_mode in {"SMOOTHED", "SMOOTHED+GROUPED"}
    mode_uses_grouping = config.display_mode in {"GROUPED", "SMOOTHED+GROUPED"}
    group_state_col = "hmm_state_smoothed" if config.display_mode == "SMOOTHED+GROUPED" else "hmm_state"

    ranking: list[int] = []
    long_states: list[int] = []
    short_states: list[int] = []
    mapping_source = "computed"

    if mode_uses_grouping:
        cached_display = None
        if isinstance(cached_meta, dict):
            cached_display = (
                cached_meta.get("display_mappings", {}) or {}
            ).get("hmm_group_display_v1")
        if isinstance(cached_display, dict):
            scheme_ok = str(cached_display.get("grouping_scheme")) == config.grouping_scheme
            if scheme_ok:
                long_cached = [int(v) for v in (cached_display.get("long_states") or [])]
                short_cached = [int(v) for v in (cached_display.get("short_states") or [])]
                if long_cached:
                    long_states = _limit_unique(long_cached, int(config.long_states_top_k))
                    short_states = _limit_unique(short_cached, int(config.short_states_bottom_k))
                    mapping_source = "cache_display_mapping"

        if not long_states:
            if base_long_states:
                long_states = _limit_unique([int(v) for v in base_long_states], int(config.long_states_top_k))
                mapping_source = "cache_hmm_long_bias"
            else:
                ranking = _compute_fwd_state_rank(out, group_state_col)
                long_states = _limit_unique(ranking, int(config.long_states_top_k))
                mapping_source = "computed_fwd_rank"

        if config.grouping_scheme == "LONG_NEUTRAL_SHORT":
            if not short_states:
                if base_short_states:
                    short_states = _limit_unique([int(v) for v in base_short_states], int(config.short_states_bottom_k))
                    if mapping_source == "cache_hmm_long_bias":
                        mapping_source = "cache_hmm_long_short_bias"
                else:
                    if not ranking:
                        ranking = _compute_fwd_state_rank(out, group_state_col)
                    rev = [v for v in reversed(ranking) if v not in set(long_states)]
                    short_states = _limit_unique(rev, int(config.short_states_bottom_k))

        if not long_states and config.grouping_scheme in {"LONG_NEUTRAL_SHORT", "LONG_OTHER"}:
            warnings.append("Grouped HMM display unavailable: unable to infer long states.")
            mode_uses_grouping = False

    if mode_uses_grouping:
        long_set = set(long_states)
        short_set = set(short_states)
        if config.grouping_scheme == "LONG_NEUTRAL_SHORT":
            out = out.with_columns(
                pl.when(pl.col(group_state_col).cast(pl.Int32, strict=False).is_in(sorted(long_set)))
                .then(pl.lit(1))
                .when(pl.col(group_state_col).cast(pl.Int32, strict=False).is_in(sorted(short_set)))
                .then(pl.lit(-1))
                .when(pl.col(group_state_col).is_null())
                .then(pl.lit(None).cast(pl.Int8))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias("hmm_group_code")
            ).with_columns(
                pl.when(pl.col("hmm_group_code") == 1)
                .then(pl.lit("LONG"))
                .when(pl.col("hmm_group_code") == -1)
                .then(pl.lit("SHORT"))
                .when(pl.col("hmm_group_code") == 0)
                .then(pl.lit("NEUTRAL"))
                .otherwise(pl.lit("NA"))
                .alias("hmm_group_label")
            )
        else:
            out = out.with_columns(
                pl.when(pl.col(group_state_col).cast(pl.Int32, strict=False).is_in(sorted(long_set)))
                .then(pl.lit(1))
                .when(pl.col(group_state_col).is_null())
                .then(pl.lit(None).cast(pl.Int8))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias("hmm_group_code")
            ).with_columns(
                pl.when(pl.col("hmm_group_code") == 1)
                .then(pl.lit("LONG"))
                .when(pl.col("hmm_group_code") == 0)
                .then(pl.lit("OTHER"))
                .otherwise(pl.lit("NA"))
                .alias("hmm_group_label")
            )

        out = out.with_columns(pl.col("hmm_group_code").alias("hmm_group_display"))

        if config.persist_mapping_in_cache_meta and cache_meta_path is not None:
            try:
                _persist_mapping(
                    meta_path=cache_meta_path,
                    mapping={
                        "grouping_scheme": config.grouping_scheme,
                        "long_states_top_k": int(config.long_states_top_k),
                        "short_states_bottom_k": int(config.short_states_bottom_k),
                        "long_states": long_states,
                        "short_states": short_states,
                        "state_col": group_state_col,
                        "mapping_source": mapping_source,
                    },
                )
            except Exception as exc:
                warnings.append(f"Failed to persist HMM display mapping to cache meta: {exc}")

    if config.display_mode == "RAW":
        band_col = "hmm_state"
    elif config.display_mode == "SMOOTHED":
        band_col = "hmm_state_smoothed"
    elif config.display_mode in {"GROUPED", "SMOOTHED+GROUPED"} and "hmm_group_display" in out.columns:
        band_col = "hmm_group_display"
    else:
        band_col = "hmm_state"

    summary: dict[str, Any] = {
        "enabled": True,
        "display_mode": config.display_mode,
        "smoothing_method": config.smoothing_method,
        "smoothing_window": int(window),
        "grouping_scheme": config.grouping_scheme,
        "band_col": band_col,
        "grouping_source": mapping_source if mode_uses_grouping else None,
        "long_states": long_states,
        "short_states": short_states,
    }

    return HMMDisplayResult(
        frame=out,
        band_col=band_col,
        summary=summary,
        warnings=warnings,
    )
