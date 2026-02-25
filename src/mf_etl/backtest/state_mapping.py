"""State mapping and tradability eligibility logic for backtest inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mf_etl.config import AppSettings

DIRECTION_VALUES = {"LONG_BIAS", "SHORT_BIAS", "UNCONFIRMED"}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _normalize_direction(value: Any) -> str:
    candidate = str(value).strip().upper()
    return candidate if candidate in DIRECTION_VALUES else "UNCONFIRMED"


def load_state_map_file(state_map_file: Path) -> dict[int, str]:
    """Load external state-direction map from JSON file."""

    payload = json.loads(state_map_file.read_text(encoding="utf-8"))
    raw_map: dict[str, Any]
    if isinstance(payload, dict) and "mapping" in payload and isinstance(payload["mapping"], dict):
        raw_map = payload["mapping"]
    elif isinstance(payload, dict):
        raw_map = {str(key): value for key, value in payload.items()}
    else:
        raise ValueError(f"Unsupported state map JSON format in {state_map_file}")

    out: dict[int, str] = {}
    for key, value in raw_map.items():
        try:
            state_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid state id in state map: {key}") from exc
        out[state_id] = _normalize_direction(value)
    return out


def apply_flow_state_mapping(
    frame: pl.DataFrame,
    *,
    settings: AppSettings,
    include_state_ids: list[int],
    allow_unconfirmed: bool,
) -> tuple[pl.DataFrame, dict[str, Any], dict[str, Any] | None]:
    """Apply deterministic flow-state direction mapping."""

    long_states = set(settings.backtest.flow_mapping.long_states)
    short_states = set(settings.backtest.flow_mapping.short_states)
    ignore_states = set(settings.backtest.flow_mapping.ignore_states)

    rows: list[dict[str, Any]] = []
    for state in sorted(set(long_states) | set(short_states) | set(ignore_states)):
        if state in long_states:
            direction = "LONG_BIAS"
        elif state in short_states:
            direction = "SHORT_BIAS"
        else:
            direction = "UNCONFIRMED"
        rows.append(
            {
                "state_id": state,
                "state_direction_hint": direction,
                "state_class": "NA",
                "state_score": None,
                "policy_reason_flags": "FLOW_DEFAULT_MAPPING",
            }
        )
    mapping_df = pl.DataFrame(rows) if rows else pl.DataFrame(schema={"state_id": pl.Int32})

    joined = frame.join(mapping_df, on="state_id", how="left", suffix="_map")
    reason_col = "policy_reason_flags_map" if "policy_reason_flags_map" in joined.columns else "policy_reason_flags"
    mapped = joined.with_columns(
        [
            pl.coalesce([pl.col("state_direction_hint_map"), pl.col("state_direction_hint")]).alias("state_direction_hint"),
            pl.coalesce([pl.col("state_class_map"), pl.col("state_class")]).alias("state_class"),
            pl.coalesce([pl.col("state_score_map"), pl.col("state_score")]).alias("state_score"),
            pl.coalesce([pl.col(reason_col), pl.lit("FLOW_UNMAPPED")]).alias("policy_reason_flags"),
        ]
    )
    drop_cols = [c for c in ["state_direction_hint_map", "state_class_map", "state_score_map", "policy_reason_flags_map"] if c in mapped.columns]
    if drop_cols:
        mapped = mapped.drop(drop_cols)

    state_filter = pl.lit(True)
    if include_state_ids:
        state_filter = pl.col("state_id").is_in(include_state_ids)

    eligible = (
        state_filter
        & (
            pl.col("state_direction_hint").is_in(["LONG_BIAS", "SHORT_BIAS"])
            | (pl.lit(allow_unconfirmed) & (pl.col("state_direction_hint") == "UNCONFIRMED"))
        )
    )
    mapped = mapped.with_columns(eligible.alias("signal_eligible"))

    summary = {
        "mapping_source": "flow_default",
        "eligible_rows": int(mapped.filter(pl.col("signal_eligible")).height),
        "direction_counts": mapped.group_by("state_direction_hint").len(name="rows").to_dicts(),
    }
    return mapped, summary, None


def infer_hmm_direction_map(
    *,
    validation_run_dir: Path,
    min_abs_forward_mean_for_direction: float,
) -> dict[int, str]:
    """Infer HMM state direction from validation state scorecard."""

    state_scorecard_path = validation_run_dir / "state_scorecard.csv"
    if not state_scorecard_path.exists():
        raise FileNotFoundError(f"Missing HMM validation state_scorecard.csv at {state_scorecard_path}")

    state_scorecard = pl.read_csv(state_scorecard_path)
    required = {"state_id", "fwd_ret_10_mean", "fwd_ret_10_ci_lo", "fwd_ret_10_ci_hi"}
    missing = sorted(required - set(state_scorecard.columns))
    if missing:
        rendered = ", ".join(missing)
        raise ValueError(f"HMM validation state_scorecard missing columns: {rendered}")

    mapping: dict[int, str] = {}
    for row in state_scorecard.to_dicts():
        state_id = int(row["state_id"])
        ci_lo = _safe_float(row.get("fwd_ret_10_ci_lo"))
        ci_hi = _safe_float(row.get("fwd_ret_10_ci_hi"))
        mean_val = _safe_float(row.get("fwd_ret_10_mean"))
        if ci_lo is not None and ci_hi is not None:
            if ci_lo > 0 and ci_hi > 0:
                mapping[state_id] = "LONG_BIAS"
                continue
            if ci_lo < 0 and ci_hi < 0:
                mapping[state_id] = "SHORT_BIAS"
                continue
        if mean_val is not None and abs(mean_val) >= min_abs_forward_mean_for_direction:
            mapping[state_id] = "LONG_BIAS" if mean_val > 0 else "SHORT_BIAS"
        else:
            mapping[state_id] = "UNCONFIRMED"
    return mapping


def apply_hmm_state_mapping(
    frame: pl.DataFrame,
    *,
    settings: AppSettings,
    validation_run_dir: Path | None,
    state_map_file: Path | None,
    include_state_ids: list[int],
    allow_unconfirmed: bool,
) -> tuple[pl.DataFrame, dict[str, Any], dict[str, Any] | None]:
    """Apply HMM state direction mapping from map file or validation scorecard inference."""

    mapping_source = None
    if state_map_file is not None:
        direction_map = load_state_map_file(state_map_file)
        mapping_source = "state_map"
    elif validation_run_dir is not None:
        direction_map = infer_hmm_direction_map(
            validation_run_dir=validation_run_dir,
            min_abs_forward_mean_for_direction=settings.backtest.hmm_direction_inference.min_abs_forward_mean_for_direction,
        )
        mapping_source = "validation_scorecard"
    else:
        direction_map = {}

    if not direction_map and not allow_unconfirmed:
        raise ValueError(
            "HMM direction mapping could not be inferred. Provide --validation-run-dir or --state-map-file, "
            "or set --allow-unconfirmed to continue."
        )

    rows = [
        {
            "state_id": int(state_id),
            "state_direction_hint": direction,
            "state_class": "NA",
            "state_score": None,
            "policy_reason_flags": f"HMM_MAPPING_{mapping_source or 'UNCONFIRMED_DEFAULT'}",
        }
        for state_id, direction in sorted(direction_map.items(), key=lambda x: x[0])
    ]
    mapping_df = pl.DataFrame(rows) if rows else pl.DataFrame(schema={"state_id": pl.Int32})

    joined = frame.join(mapping_df, on="state_id", how="left", suffix="_map")
    reason_col = "policy_reason_flags_map" if "policy_reason_flags_map" in joined.columns else "policy_reason_flags"
    mapped = joined.with_columns(
        [
            pl.coalesce([pl.col("state_direction_hint_map"), pl.lit("UNCONFIRMED")]).alias("state_direction_hint"),
            pl.coalesce([pl.col("state_class_map"), pl.lit("NA")]).alias("state_class"),
            pl.coalesce([pl.col("state_score_map"), pl.lit(None).cast(pl.Float64)]).alias("state_score"),
            pl.coalesce([pl.col(reason_col), pl.lit("HMM_UNMAPPED")]).alias("policy_reason_flags"),
        ]
    )
    drop_cols = [c for c in ["state_direction_hint_map", "state_class_map", "state_score_map", "policy_reason_flags_map"] if c in mapped.columns]
    if drop_cols:
        mapped = mapped.drop(drop_cols)

    state_filter = pl.lit(True)
    if include_state_ids:
        state_filter = pl.col("state_id").is_in(include_state_ids)

    eligible = (
        state_filter
        & (
            pl.col("state_direction_hint").is_in(["LONG_BIAS", "SHORT_BIAS"])
            | (pl.lit(allow_unconfirmed) & (pl.col("state_direction_hint") == "UNCONFIRMED"))
        )
    )
    mapped = mapped.with_columns(eligible.alias("signal_eligible"))

    summary = {
        "mapping_source": mapping_source or "none",
        "eligible_rows": int(mapped.filter(pl.col("signal_eligible")).height),
        "direction_counts": mapped.group_by("state_direction_hint").len(name="rows").to_dicts(),
    }
    snapshot = {"mapping_source": mapping_source or "none", "direction_map": direction_map}
    return mapped, summary, snapshot


def _cluster_policy_table(cluster_hardening_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    policy_path = cluster_hardening_dir / "cluster_hardening_policy.json"
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Missing cluster hardening policy at {policy_path}. Run cluster-hardening-run first."
        )
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    per_state = payload.get("per_state", [])
    if not isinstance(per_state, list):
        raise ValueError(f"Invalid cluster_hardening_policy.json structure at {policy_path}")

    rows: list[dict[str, Any]] = []
    for row in per_state:
        if not isinstance(row, dict):
            continue
        try:
            state_id = int(row.get("state_id"))
        except (TypeError, ValueError):
            continue
        reasons = row.get("reasons")
        if isinstance(reasons, list):
            reason_flags = ",".join(str(v) for v in reasons)
        else:
            reason_flags = str(reasons or "")
        rows.append(
            {
                "state_id": state_id,
                "state_class": str(row.get("class_label") or "BLOCK").upper(),
                "state_direction_hint": _normalize_direction(row.get("allow_direction_hint")),
                "state_score": _safe_float(row.get("tradability_score")),
                "policy_reason_flags": reason_flags,
            }
        )

    table = pl.DataFrame(rows) if rows else pl.DataFrame(schema={"state_id": pl.Int32})
    return table, payload


def apply_cluster_policy_mapping(
    frame: pl.DataFrame,
    *,
    settings: AppSettings,
    cluster_hardening_dir: Path,
    include_watch: bool,
    policy_filter_mode: str = "allow_only",
    include_state_ids: list[int],
    allow_unconfirmed: bool,
) -> tuple[pl.DataFrame, dict[str, Any], dict[str, Any] | None]:
    """Apply cluster hardening policy to cluster states and compute eligibility."""

    policy_table, policy_payload = _cluster_policy_table(cluster_hardening_dir)
    joined = frame.join(policy_table, on="state_id", how="left", suffix="_policy")
    reason_col = "policy_reason_flags_policy" if "policy_reason_flags_policy" in joined.columns else "policy_reason_flags"
    mapped = joined.with_columns(
        [
            pl.coalesce([pl.col("state_direction_hint_policy"), pl.lit("UNCONFIRMED")]).alias("state_direction_hint"),
            pl.coalesce([pl.col("state_class_policy"), pl.lit("BLOCK")]).alias("state_class"),
            pl.coalesce([pl.col("state_score_policy"), pl.lit(None).cast(pl.Float64)]).alias("state_score"),
            pl.coalesce([pl.col(reason_col), pl.lit("CLUSTER_UNMAPPED")]).alias("policy_reason_flags"),
        ]
    )
    drop_cols = [c for c in ["state_direction_hint_policy", "state_class_policy", "state_score_policy", "policy_reason_flags_policy"] if c in mapped.columns]
    if drop_cols:
        mapped = mapped.drop(drop_cols)

    filter_mode = str(policy_filter_mode).strip().lower()
    if filter_mode not in {"allow_only", "allow_watch", "all_states"}:
        raise ValueError(f"Unsupported policy_filter_mode: {policy_filter_mode}")

    if filter_mode == "all_states":
        allowed_classes = {"ALLOW", "WATCH", "BLOCK"}
    elif filter_mode == "allow_watch":
        allowed_classes = {"ALLOW", "WATCH"}
    else:
        allowed_classes = set(settings.backtest.cluster_policy.default_classes) or {"ALLOW"}
        if include_watch:
            allowed_classes.add("WATCH")

    class_filter = pl.col("state_class").is_in(sorted(allowed_classes))
    state_filter = pl.lit(True)
    if include_state_ids:
        state_filter = pl.col("state_id").is_in(include_state_ids)

    eligible = (
        class_filter
        & state_filter
        & (
            pl.col("state_direction_hint").is_in(["LONG_BIAS", "SHORT_BIAS"])
            | (pl.lit(allow_unconfirmed) & (pl.col("state_direction_hint") == "UNCONFIRMED"))
        )
    )
    mapped = mapped.with_columns(eligible.alias("signal_eligible"))

    summary = {
        "mapping_source": "cluster_hardening_policy",
        "policy_filter_mode": filter_mode,
        "eligible_rows": int(mapped.filter(pl.col("signal_eligible")).height),
        "class_counts": mapped.group_by("state_class").len(name="rows").to_dicts(),
        "direction_counts": mapped.group_by("state_direction_hint").len(name="rows").to_dicts(),
        "allowed_classes": sorted(allowed_classes),
    }
    return mapped, summary, policy_payload
