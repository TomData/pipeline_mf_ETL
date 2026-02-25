"""Hybrid overlay evaluation report builder for existing grid/WF artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from mf_etl.backtest.writer import (
    write_csv_atomically,
    write_json_atomically,
    write_markdown_atomically,
)
from mf_etl.config import AppSettings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridEvalReportResult:
    """Output paths for one hybrid overlay evaluation report run."""

    run_id: str
    output_dir: Path
    summary_path: Path
    table_path: Path
    wf_table_path: Path
    report_path: Path


@dataclass(frozen=True)
class _GridRunSpec:
    label: str
    run_dir: Path | None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"true", "1"}:
            return True
        if norm in {"false", "0"}:
            return False
    return None


def _normalize_metric(values: list[float | None], *, higher_better: bool) -> list[float]:
    finite = [val for val in values if val is not None and np.isfinite(val)]
    if not finite:
        return [0.5] * len(values)
    v_min = min(finite)
    v_max = max(finite)
    if np.isclose(v_max, v_min):
        return [0.5 if val is not None else 0.0 for val in values]

    out: list[float] = []
    denom = v_max - v_min
    for val in values:
        if val is None:
            out.append(0.0)
            continue
        scaled = (val - v_min) / denom
        out.append(float(scaled if higher_better else (1.0 - scaled)))
    return out


def _to_markdown_table(df: pl.DataFrame, *, max_rows: int = 20) -> str:
    if df.height == 0:
        return "(no rows)"
    head = df.head(max_rows)
    cols = head.columns
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in head.to_dicts():
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def _load_json(path: Path, *, loaded: list[str], missing: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        missing.append(str(path))
        return None
    loaded.append(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _load_table(base_path: Path, *, loaded: list[str], missing: list[str]) -> pl.DataFrame | None:
    parquet_path = base_path.with_suffix(".parquet")
    if parquet_path.exists():
        loaded.append(str(parquet_path))
        return pl.read_parquet(parquet_path)
    csv_path = base_path.with_suffix(".csv")
    if csv_path.exists():
        loaded.append(str(csv_path))
        return pl.read_csv(csv_path)
    missing.append(str(parquet_path))
    missing.append(str(csv_path))
    return None


def _count_non_finite_cells(df: pl.DataFrame) -> int:
    if df.height == 0:
        return 0
    total = 0
    for name, dtype in df.schema.items():
        if not dtype.is_numeric():
            continue
        finite_count = int(
            df.select(pl.col(name).cast(pl.Float64, strict=False).is_finite().sum()).item() or 0
        )
        null_count = int(df.select(pl.col(name).null_count()).item() or 0)
        total += max(df.height - finite_count - null_count, 0)
    return total


def _score_combo_rows(metrics_df: pl.DataFrame) -> pl.DataFrame:
    if metrics_df.height == 0:
        return metrics_df
    rows = metrics_df.to_dicts()
    expect_vals = [_safe_float(row.get("expectancy")) for row in rows]
    pf_vals = [_safe_float(row.get("profit_factor")) for row in rows]
    robust_vals = [_safe_float(row.get("robustness_score_v2")) for row in rows]
    retcv_vals = [_safe_float(row.get("ret_cv")) for row in rows]
    downside_vals = [_safe_float(row.get("downside_std")) for row in rows]
    trade_vals = [_safe_float(row.get("trade_count")) for row in rows]

    expect_norm = _normalize_metric(expect_vals, higher_better=True)
    pf_norm = _normalize_metric(pf_vals, higher_better=True)
    robust_norm = _normalize_metric(robust_vals, higher_better=True)
    retcv_norm = _normalize_metric(retcv_vals, higher_better=False)
    downside_norm = _normalize_metric(downside_vals, higher_better=False)
    trade_norm = _normalize_metric(trade_vals, higher_better=True)

    for idx, row in enumerate(rows):
        base = (
            (0.32 * expect_norm[idx])
            + (0.20 * pf_norm[idx])
            + (0.24 * robust_norm[idx])
            + (0.13 * retcv_norm[idx])
            + (0.06 * downside_norm[idx])
            + (0.05 * trade_norm[idx])
        )
        score = 100.0 * base
        if bool(row.get("is_zero_trade_combo")):
            score -= 20.0
        row["combo_candidate_score"] = float(max(0.0, min(100.0, score)))
    return pl.DataFrame(rows)


def _pick_top_combo(metrics_df: pl.DataFrame) -> dict[str, Any] | None:
    if metrics_df.height == 0:
        return None
    scored = _score_combo_rows(metrics_df)
    rows = scored.to_dicts()

    def key_func(row: dict[str, Any]) -> tuple[float, float, float, str]:
        candidate = _safe_float(row.get("combo_candidate_score"))
        expectancy = _safe_float(row.get("expectancy"))
        pf = _safe_float(row.get("profit_factor"))
        combo_id = str(row.get("combo_id") or "")
        return (
            -(candidate if candidate is not None else -1e9),
            -(expectancy if expectancy is not None else -1e9),
            -(pf if pf is not None else -1e9),
            combo_id,
        )

    rows.sort(key=key_func)
    return rows[0]


def _run_row_from_spec(
    spec: _GridRunSpec,
    *,
    loaded: list[str],
    missing: list[str],
) -> tuple[dict[str, Any], dict[str, Any] | None, int]:
    row: dict[str, Any] = {
        "run_label": spec.label,
        "run_dir": str(spec.run_dir) if spec.run_dir is not None else None,
        "overlay_mode": None,
        "overlay_enabled": None,
        "best_combo_id": None,
        "best_expectancy": None,
        "best_pf": None,
        "best_robustness_v2": None,
        "best_ret_cv": None,
        "best_zero_trade_share": None,
        "best_trade_count": None,
        "best_downside_std": None,
        "overlay_match_rate": None,
        "overlay_vetoed_signal_share": None,
        "overlay_unknown_rate": None,
        "grid_run_id": None,
        "successful_combos": None,
        "failed_combos": None,
        "status": "MISSING",
    }
    if spec.run_dir is None:
        return row, None, 0

    summary_payload = _load_json(spec.run_dir / "grid_summary.json", loaded=loaded, missing=missing)
    metrics_df = _load_table(spec.run_dir / "grid_metrics_table", loaded=loaded, missing=missing)
    non_finite = _count_non_finite_cells(metrics_df) if metrics_df is not None else 0
    if summary_payload is not None:
        row["grid_run_id"] = summary_payload.get("grid_run_id")
        row["successful_combos"] = _safe_int(summary_payload.get("successful_combos"))
        row["failed_combos"] = _safe_int(summary_payload.get("failed_combos"))
        row["best_zero_trade_share"] = _safe_float(summary_payload.get("zero_trade_combo_share"))
    if metrics_df is None:
        row["status"] = "MISSING_METRICS"
        return row, None, non_finite

    success_df = metrics_df
    if "status" in success_df.columns:
        success_df = success_df.filter(pl.col("status") == "SUCCESS")
    top = _pick_top_combo(success_df)
    if top is None:
        row["status"] = "NO_SUCCESS_ROWS"
        return row, None, non_finite

    row.update(
        {
            "overlay_mode": top.get("overlay_mode"),
            "overlay_enabled": _safe_bool(top.get("overlay_enabled")),
            "best_combo_id": top.get("combo_id"),
            "best_expectancy": _safe_float(top.get("expectancy")),
            "best_pf": _safe_float(top.get("profit_factor")),
            "best_robustness_v2": _safe_float(top.get("robustness_score_v2")),
            "best_ret_cv": _safe_float(top.get("ret_cv")),
            "best_trade_count": _safe_int(top.get("trade_count")),
            "best_downside_std": _safe_float(top.get("downside_std")),
            "overlay_match_rate": _safe_float(top.get("overlay_match_rate")),
            "overlay_vetoed_signal_share": _safe_float(top.get("overlay_vetoed_signal_share")),
            "overlay_unknown_rate": _safe_float(top.get("overlay_unknown_rate")),
            "status": "OK",
        }
    )
    if row["best_zero_trade_share"] is None and "is_zero_trade_combo" in success_df.columns:
        row["best_zero_trade_share"] = _safe_float(
            success_df.select(pl.col("is_zero_trade_combo").cast(pl.Float64, strict=False).mean()).item()
        )
    row["combo_candidate_score"] = _safe_float(top.get("combo_candidate_score"))
    return row, top, non_finite


def _append_single_run_scores(table: pl.DataFrame) -> pl.DataFrame:
    if table.height == 0:
        return table
    rows = table.to_dicts()
    expectancy = [_safe_float(row.get("best_expectancy")) for row in rows]
    pf = [_safe_float(row.get("best_pf")) for row in rows]
    robust = [_safe_float(row.get("best_robustness_v2")) for row in rows]
    ret_cv = [_safe_float(row.get("best_ret_cv")) for row in rows]
    downside = [_safe_float(row.get("best_downside_std")) for row in rows]
    zero_share = [_safe_float(row.get("best_zero_trade_share")) for row in rows]

    expectancy_n = _normalize_metric(expectancy, higher_better=True)
    pf_n = _normalize_metric(pf, higher_better=True)
    robust_n = _normalize_metric(robust, higher_better=True)
    retcv_n = _normalize_metric(ret_cv, higher_better=False)
    downside_n = _normalize_metric(downside, higher_better=False)
    zero_n = _normalize_metric(zero_share, higher_better=False)

    scored_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        score = 100.0 * (
            (0.30 * expectancy_n[idx])
            + (0.20 * pf_n[idx])
            + (0.25 * robust_n[idx])
            + (0.15 * retcv_n[idx])
            + (0.07 * downside_n[idx])
            + (0.03 * zero_n[idx])
        )
        veto_share = _safe_float(row.get("overlay_vetoed_signal_share"))
        if veto_share is not None and veto_share > 0.80:
            score -= 5.0
        row["single_candidate_score"] = float(max(0.0, min(100.0, score)))
        scored_rows.append(row)
    return pl.DataFrame(scored_rows)


def _classify_run_recommendations(table: pl.DataFrame) -> pl.DataFrame:
    if table.height == 0:
        return table
    rows = table.to_dicts()
    baseline_trade_count: int | None = None
    for row in rows:
        if row.get("run_label") == "HMM baseline":
            baseline_trade_count = _safe_int(row.get("best_trade_count"))
            break
    for row in rows:
        score = _safe_float(row.get("single_candidate_score")) or 0.0
        ret_cv = _safe_float(row.get("best_ret_cv"))
        zero_share = _safe_float(row.get("best_zero_trade_share")) or 0.0
        veto_share = _safe_float(row.get("overlay_vetoed_signal_share")) or 0.0
        trade_count = _safe_int(row.get("best_trade_count"))
        overlay_enabled = bool(row.get("overlay_enabled"))
        if (
            score >= 72.0
            and zero_share <= 0.25
            and (ret_cv is None or ret_cv <= 20.0)
            and row.get("status") == "OK"
        ):
            rec = "PROMOTE"
        elif (not overlay_enabled) and score >= 56.0 and row.get("status") == "OK":
            rec = "KEEP_AS_BENCH"
        elif (
            overlay_enabled
            and row.get("status") == "OK"
            and (
                veto_share >= 0.50
                or (
                    baseline_trade_count is not None
                    and trade_count is not None
                    and trade_count < (0.60 * baseline_trade_count)
                )
            )
        ):
            rec = "NICHE_FILTER"
        else:
            rec = "RESEARCH_ONLY"
        row["recommendation_label"] = rec
    return pl.DataFrame(rows)


def _extract_best_split_metrics(grid_run_dir: Path, *, loaded: list[str], missing: list[str]) -> dict[str, float | None]:
    metrics_df = _load_table(grid_run_dir / "grid_metrics_table", loaded=loaded, missing=missing)
    if metrics_df is None:
        return {
            "expectancy": None,
            "profit_factor": None,
            "robustness_score_v2": None,
            "ret_cv": None,
            "zero_trade_combo_share": None,
        }
    success_df = metrics_df
    if "status" in success_df.columns:
        success_df = success_df.filter(pl.col("status") == "SUCCESS")
    top = _pick_top_combo(success_df)
    if top is None:
        return {
            "expectancy": None,
            "profit_factor": None,
            "robustness_score_v2": None,
            "ret_cv": None,
            "zero_trade_combo_share": None,
        }
    zero_share = _safe_float(
        success_df.select(pl.col("is_zero_trade_combo").cast(pl.Float64, strict=False).mean()).item()
    ) if "is_zero_trade_combo" in success_df.columns and success_df.height > 0 else None
    return {
        "expectancy": _safe_float(top.get("expectancy")),
        "profit_factor": _safe_float(top.get("profit_factor")),
        "robustness_score_v2": _safe_float(top.get("robustness_score_v2")),
        "ret_cv": _safe_float(top.get("ret_cv")),
        "zero_trade_combo_share": zero_share,
    }


def _load_wf_split_metrics(
    wf_dir: Path | None,
    *,
    loaded: list[str],
    missing: list[str],
) -> dict[str, dict[str, float | None]]:
    if wf_dir is None:
        return {}
    by_split_df = _load_table(wf_dir / "wf_grid_by_split", loaded=loaded, missing=missing)
    if by_split_df is None or by_split_df.height == 0:
        return {}
    if "status" in by_split_df.columns:
        by_split_df = by_split_df.filter(pl.col("status") == "SUCCESS")
    out: dict[str, dict[str, float | None]] = {}
    for row in by_split_df.to_dicts():
        train_end = str(row.get("train_end") or "")
        grid_run_dir_raw = row.get("grid_run_dir")
        if train_end == "" or grid_run_dir_raw is None:
            continue
        grid_run_dir = Path(str(grid_run_dir_raw))
        out[train_end] = _extract_best_split_metrics(grid_run_dir, loaded=loaded, missing=missing)
    return out


def _load_wf_overlay_source_stats(
    wf_dir: Path | None,
    *,
    source: str,
    loaded: list[str],
    missing: list[str],
) -> dict[str, float | None]:
    if wf_dir is None:
        return {}
    overlay_df = _load_table(wf_dir / "wf_overlay_source_summary", loaded=loaded, missing=missing)
    if overlay_df is None or overlay_df.height == 0 or "source_type" not in overlay_df.columns:
        return {}
    source_df = overlay_df.filter(pl.col("source_type") == source)
    if source_df.height == 0:
        return {}
    row = source_df.head(1).to_dicts()[0]
    return {
        "overlay_match_rate": _safe_float(row.get("overlay_match_rate_mean")),
        "overlay_unknown_rate": _safe_float(row.get("overlay_unknown_rate_mean")),
        "overlay_vetoed_signal_share": _safe_float(row.get("overlay_vetoed_signal_share_mean")),
    }


def _build_wf_comparison_row(
    *,
    baseline_dir: Path | None,
    hybrid_dir: Path | None,
    loaded: list[str],
    missing: list[str],
) -> dict[str, Any]:
    baseline = _load_wf_split_metrics(baseline_dir, loaded=loaded, missing=missing)
    hybrid = _load_wf_split_metrics(hybrid_dir, loaded=loaded, missing=missing)
    common = sorted(set(baseline.keys()) & set(hybrid.keys()))
    split_count = len(common)
    deltas_exp: list[float] = []
    deltas_pf: list[float] = []
    deltas_rob: list[float] = []
    deltas_retcv: list[float] = []
    deltas_zero: list[float] = []
    wins_exp = 0
    wins_pf = 0
    wins_rob = 0
    wins_retcv = 0
    for split in common:
        b = baseline[split]
        h = hybrid[split]
        d_exp = _safe_float(h.get("expectancy")) - _safe_float(b.get("expectancy")) if (_safe_float(h.get("expectancy")) is not None and _safe_float(b.get("expectancy")) is not None) else None
        d_pf = _safe_float(h.get("profit_factor")) - _safe_float(b.get("profit_factor")) if (_safe_float(h.get("profit_factor")) is not None and _safe_float(b.get("profit_factor")) is not None) else None
        d_rob = _safe_float(h.get("robustness_score_v2")) - _safe_float(b.get("robustness_score_v2")) if (_safe_float(h.get("robustness_score_v2")) is not None and _safe_float(b.get("robustness_score_v2")) is not None) else None
        d_retcv = _safe_float(h.get("ret_cv")) - _safe_float(b.get("ret_cv")) if (_safe_float(h.get("ret_cv")) is not None and _safe_float(b.get("ret_cv")) is not None) else None
        d_zero = _safe_float(h.get("zero_trade_combo_share")) - _safe_float(b.get("zero_trade_combo_share")) if (_safe_float(h.get("zero_trade_combo_share")) is not None and _safe_float(b.get("zero_trade_combo_share")) is not None) else None
        if d_exp is not None:
            deltas_exp.append(d_exp)
            if d_exp > 0:
                wins_exp += 1
        if d_pf is not None:
            deltas_pf.append(d_pf)
            if d_pf > 0:
                wins_pf += 1
        if d_rob is not None:
            deltas_rob.append(d_rob)
            if d_rob > 0:
                wins_rob += 1
        if d_retcv is not None:
            deltas_retcv.append(d_retcv)
            if d_retcv < 0:
                wins_retcv += 1
        if d_zero is not None:
            deltas_zero.append(d_zero)

    split_count_nonzero = max(split_count, 1)
    wf_consistency = 100.0 * (
        (0.30 * (wins_exp / split_count_nonzero))
        + (0.20 * (wins_pf / split_count_nonzero))
        + (0.30 * (wins_rob / split_count_nonzero))
        + (0.20 * (wins_retcv / split_count_nonzero))
    )

    baseline_overlay = _load_wf_overlay_source_stats(
        baseline_dir, source="hmm", loaded=loaded, missing=missing
    )
    hybrid_overlay = _load_wf_overlay_source_stats(
        hybrid_dir, source="hmm", loaded=loaded, missing=missing
    )

    return {
        "source": "hmm",
        "comparison": "baseline_vs_hybrid",
        "split_count": split_count,
        "hybrid_wins_expectancy": wins_exp,
        "hybrid_wins_pf": wins_pf,
        "hybrid_wins_robustness_v2": wins_rob,
        "hybrid_wins_ret_cv": wins_retcv,
        "avg_delta_expectancy": _safe_float(float(np.mean(deltas_exp)) if deltas_exp else None),
        "avg_delta_pf": _safe_float(float(np.mean(deltas_pf)) if deltas_pf else None),
        "avg_delta_robustness_v2": _safe_float(float(np.mean(deltas_rob)) if deltas_rob else None),
        "avg_delta_ret_cv": _safe_float(float(np.mean(deltas_retcv)) if deltas_retcv else None),
        "avg_delta_zero_trade_share": _safe_float(float(np.mean(deltas_zero)) if deltas_zero else None),
        "baseline_overlay_match_rate": baseline_overlay.get("overlay_match_rate"),
        "hybrid_overlay_match_rate": hybrid_overlay.get("overlay_match_rate"),
        "baseline_overlay_unknown_rate": baseline_overlay.get("overlay_unknown_rate"),
        "hybrid_overlay_unknown_rate": hybrid_overlay.get("overlay_unknown_rate"),
        "baseline_overlay_vetoed_signal_share": baseline_overlay.get("overlay_vetoed_signal_share"),
        "hybrid_overlay_vetoed_signal_share": hybrid_overlay.get("overlay_vetoed_signal_share"),
        "wf_consistency_score": float(max(0.0, min(100.0, wf_consistency))),
    }


def _run_label_lookup(table: pl.DataFrame, label: str) -> dict[str, Any] | None:
    if table.height == 0:
        return None
    subset = table.filter(pl.col("run_label") == label)
    if subset.height == 0:
        return None
    return subset.head(1).to_dicts()[0]


def _build_key_delta(
    baseline_row: dict[str, Any] | None,
    other_row: dict[str, Any] | None,
) -> dict[str, Any]:
    if baseline_row is None or other_row is None:
        return {
            "delta_expectancy": None,
            "delta_pf": None,
            "delta_robustness_v2": None,
            "delta_ret_cv": None,
            "delta_zero_trade_share": None,
            "delta_trade_count": None,
        }
    return {
        "delta_expectancy": _safe_float(other_row.get("best_expectancy")) - _safe_float(baseline_row.get("best_expectancy"))
        if (_safe_float(other_row.get("best_expectancy")) is not None and _safe_float(baseline_row.get("best_expectancy")) is not None)
        else None,
        "delta_pf": _safe_float(other_row.get("best_pf")) - _safe_float(baseline_row.get("best_pf"))
        if (_safe_float(other_row.get("best_pf")) is not None and _safe_float(baseline_row.get("best_pf")) is not None)
        else None,
        "delta_robustness_v2": _safe_float(other_row.get("best_robustness_v2"))
        - _safe_float(baseline_row.get("best_robustness_v2"))
        if (
            _safe_float(other_row.get("best_robustness_v2")) is not None
            and _safe_float(baseline_row.get("best_robustness_v2")) is not None
        )
        else None,
        "delta_ret_cv": _safe_float(other_row.get("best_ret_cv")) - _safe_float(baseline_row.get("best_ret_cv"))
        if (_safe_float(other_row.get("best_ret_cv")) is not None and _safe_float(baseline_row.get("best_ret_cv")) is not None)
        else None,
        "delta_zero_trade_share": _safe_float(other_row.get("best_zero_trade_share"))
        - _safe_float(baseline_row.get("best_zero_trade_share"))
        if (
            _safe_float(other_row.get("best_zero_trade_share")) is not None
            and _safe_float(baseline_row.get("best_zero_trade_share")) is not None
        )
        else None,
        "delta_trade_count": _safe_float(other_row.get("best_trade_count")) - _safe_float(baseline_row.get("best_trade_count"))
        if (
            _safe_float(other_row.get("best_trade_count")) is not None
            and _safe_float(baseline_row.get("best_trade_count")) is not None
        )
        else None,
    }


def _build_final_recommendations(
    table: pl.DataFrame,
    *,
    wf_row: dict[str, Any],
) -> dict[str, Any]:
    if table.height == 0:
        return {
            "PRIMARY_CANDIDATE": None,
            "SECONDARY_CANDIDATE": None,
            "NEXT_EXPERIMENT": None,
            "run_recommendations": {},
        }
    rows = table.to_dicts()
    wf_consistency = _safe_float(wf_row.get("wf_consistency_score")) or 50.0
    ranking_rows: list[dict[str, Any]] = []
    for row in rows:
        base_score = _safe_float(row.get("single_candidate_score")) or 0.0
        run_label = str(row.get("run_label") or "")
        overlay_mode = str(row.get("overlay_mode") or "")
        wf_adjust = 0.0
        if run_label == "HMM + overlay allow_only":
            wf_adjust = (wf_consistency - 50.0) * 0.40
        elif run_label == "HMM baseline":
            wf_adjust = (50.0 - wf_consistency) * 0.25
        elif overlay_mode == "block_veto":
            wf_adjust = -2.5
        total = base_score + wf_adjust
        row["final_rank_score"] = float(total)
        ranking_rows.append(row)
    ranking_rows.sort(
        key=lambda r: (
            -(_safe_float(r.get("final_rank_score")) or -1e9),
            -(_safe_float(r.get("single_candidate_score")) or -1e9),
            str(r.get("run_label") or ""),
        )
    )
    primary = ranking_rows[0].get("run_label")
    secondary = ranking_rows[1].get("run_label") if len(ranking_rows) > 1 else None
    remaining = [r for r in ranking_rows if r.get("run_label") not in {primary, secondary}]
    next_experiment = None
    for row in remaining:
        if bool(row.get("overlay_enabled")):
            next_experiment = row.get("run_label")
            break
    if next_experiment is None and remaining:
        next_experiment = remaining[0].get("run_label")

    recs: dict[str, str] = {}
    for row in ranking_rows:
        label = str(row.get("run_label"))
        if label == primary:
            recs[label] = "PROMOTE"
        elif label == secondary:
            recs[label] = "KEEP_AS_BENCH"
        elif bool(row.get("overlay_enabled")):
            recs[label] = "NICHE_FILTER"
        else:
            recs[label] = "RESEARCH_ONLY"
    return {
        "PRIMARY_CANDIDATE": primary,
        "SECONDARY_CANDIDATE": secondary,
        "NEXT_EXPERIMENT": next_experiment,
        "run_recommendations": recs,
        "ranked_rows": ranking_rows,
    }


def _render_report_markdown(
    *,
    summary: dict[str, Any],
    run_table: pl.DataFrame,
    wf_table: pl.DataFrame,
) -> str:
    final = summary.get("final_verdicts", {})
    key_deltas = summary.get("key_deltas", {})
    sanity = summary.get("sanity", {})
    lines: list[str] = []
    lines.append("# Hybrid Overlay Evaluation Report v1")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- run_id: `{summary.get('run_id')}`")
    lines.append(f"- status: `{sanity.get('status')}`")
    lines.append(f"- primary candidate: `{final.get('PRIMARY_CANDIDATE')}`")
    lines.append(f"- secondary candidate: `{final.get('SECONDARY_CANDIDATE')}`")
    lines.append(f"- next experiment: `{final.get('NEXT_EXPERIMENT')}`")
    lines.append(
        "- decision logic: single-run weighted score (expectancy/PF/robustness/ret_cv/tails/zero-trade) + WF consistency overlay."
    )
    lines.append("")

    lines.append("## Single-Run Comparison")
    lines.append(_to_markdown_table(run_table, max_rows=20))
    lines.append("")

    lines.append("## Key Deltas")
    lines.append(f"- baseline vs allow_only: `{key_deltas.get('baseline_vs_allow_only')}`")
    lines.append(f"- baseline vs block_veto: `{key_deltas.get('baseline_vs_block_veto')}`")
    lines.append("")

    lines.append("## WF Comparison")
    lines.append(_to_markdown_table(wf_table, max_rows=10))
    lines.append("")

    lines.append("## Trade-offs (Edge vs Selectivity)")
    lines.append("- Overlay can improve selectivity but may suppress a large share of candidate signals.")
    lines.append("- Lower `ret_cv` is preferred only when edge metrics stay competitive.")
    lines.append("- WF win counts and delta consistency are treated as the main robustness check.")
    lines.append("")

    lines.append("## Recommendation")
    lines.append(f"- PRIMARY_CANDIDATE: `{final.get('PRIMARY_CANDIDATE')}`")
    lines.append(f"- SECONDARY_CANDIDATE: `{final.get('SECONDARY_CANDIDATE')}`")
    lines.append(f"- NEXT_EXPERIMENT: `{final.get('NEXT_EXPERIMENT')}`")
    lines.append("")

    lines.append("## QA")
    lines.append(f"- files_loaded: `{len(sanity.get('files_loaded', []))}`")
    lines.append(f"- missing_files: `{len(sanity.get('missing_files', []))}`")
    lines.append(f"- non_finite_cells_detected: `{sanity.get('non_finite_cells_detected')}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_hybrid_eval_report(
    settings: AppSettings,
    *,
    hmm_baseline_grid_dir: Path,
    hmm_allow_only_grid_dir: Path,
    hmm_block_veto_grid_dir: Path,
    flow_allow_only_grid_dir: Path | None,
    wf_hmm_baseline_dir: Path | None,
    wf_hmm_hybrid_dir: Path | None,
    compare_run_dir: Path | None,
    logger: logging.Logger | None = None,
) -> HybridEvalReportResult:
    """Build Hybrid Overlay Evaluation Report v1 from existing artifacts."""

    log = logger or LOGGER
    run_id = f"hybrid-eval-{uuid4().hex[:12]}"
    output_dir = settings.paths.artifacts_root / "hybrid_eval_reports" / f"{run_id}_hybrid_eval_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_files: list[str] = []
    missing_files: list[str] = []
    non_finite_cells = 0

    specs = [
        _GridRunSpec("HMM baseline", hmm_baseline_grid_dir),
        _GridRunSpec("HMM + overlay allow_only", hmm_allow_only_grid_dir),
        _GridRunSpec("HMM + overlay block_veto", hmm_block_veto_grid_dir),
    ]
    if flow_allow_only_grid_dir is not None:
        specs.append(_GridRunSpec("Flow + overlay allow_only", flow_allow_only_grid_dir))

    run_rows: list[dict[str, Any]] = []
    top_combo_per_run: dict[str, dict[str, Any]] = {}
    for spec in specs:
        row, top, non_finite = _run_row_from_spec(spec, loaded=loaded_files, missing=missing_files)
        non_finite_cells += non_finite
        run_rows.append(row)
        if top is not None:
            top_combo_per_run[spec.label] = top

    run_table = pl.DataFrame(run_rows) if run_rows else pl.DataFrame()
    if run_table.height > 0:
        run_table = _append_single_run_scores(run_table)
        run_table = _classify_run_recommendations(run_table)

    baseline_row = _run_label_lookup(run_table, "HMM baseline")
    allow_row = _run_label_lookup(run_table, "HMM + overlay allow_only")
    block_row = _run_label_lookup(run_table, "HMM + overlay block_veto")

    wf_row = _build_wf_comparison_row(
        baseline_dir=wf_hmm_baseline_dir,
        hybrid_dir=wf_hmm_hybrid_dir,
        loaded=loaded_files,
        missing=missing_files,
    )
    wf_table = pl.DataFrame([wf_row])

    compare_summary: dict[str, Any] | None = None
    if compare_run_dir is not None:
        compare_summary = _load_json(
            compare_run_dir / "grid_compare_summary.json", loaded=loaded_files, missing=missing_files
        )

    final_verdicts = _build_final_recommendations(run_table, wf_row=wf_row)
    if run_table.height > 0:
        ranking_by_label = {
            str(row.get("run_label")): _safe_float(row.get("single_candidate_score"))
            for row in run_table.to_dicts()
        }
    else:
        ranking_by_label = {}

    summary_payload: dict[str, Any] = {
        "run_id": run_id,
        "compared_runs": {
            "hmm_baseline": str(hmm_baseline_grid_dir),
            "hmm_allow_only": str(hmm_allow_only_grid_dir),
            "hmm_block_veto": str(hmm_block_veto_grid_dir),
            "flow_allow_only": str(flow_allow_only_grid_dir) if flow_allow_only_grid_dir else None,
            "wf_hmm_baseline": str(wf_hmm_baseline_dir) if wf_hmm_baseline_dir else None,
            "wf_hmm_hybrid": str(wf_hmm_hybrid_dir) if wf_hmm_hybrid_dir else None,
            "grid_compare_run": str(compare_run_dir) if compare_run_dir else None,
        },
        "top_combo_per_run": top_combo_per_run,
        "key_deltas": {
            "baseline_vs_allow_only": _build_key_delta(baseline_row, allow_row),
            "baseline_vs_block_veto": _build_key_delta(baseline_row, block_row),
        },
        "wf_consistency": wf_row,
        "single_candidate_scores": ranking_by_label,
        "final_verdicts": {
            "PRIMARY_CANDIDATE": final_verdicts.get("PRIMARY_CANDIDATE"),
            "SECONDARY_CANDIDATE": final_verdicts.get("SECONDARY_CANDIDATE"),
            "NEXT_EXPERIMENT": final_verdicts.get("NEXT_EXPERIMENT"),
            "run_recommendations": final_verdicts.get("run_recommendations"),
        },
        "decision_logic": {
            "single_run_candidate_score_weights": {
                "expectancy": 0.30,
                "profit_factor": 0.20,
                "robustness_v2": 0.25,
                "ret_cv_inverse": 0.15,
                "downside_inverse": 0.07,
                "zero_trade_inverse": 0.03,
            },
            "wf_consistency_score_weights": {
                "expectancy_wins": 0.30,
                "profit_factor_wins": 0.20,
                "robustness_v2_wins": 0.30,
                "ret_cv_wins_lower_is_better": 0.20,
            },
            "recommendation_labels": ["PROMOTE", "KEEP_AS_BENCH", "NICHE_FILTER", "RESEARCH_ONLY"],
        },
        "compare_run_summary": compare_summary,
        "sanity": {
            "files_loaded": loaded_files,
            "missing_files": missing_files,
            "non_finite_cells_detected": int(non_finite_cells),
            "status": "ok" if not missing_files else "partial",
        },
    }

    summary_path = write_json_atomically(summary_payload, output_dir / "hybrid_eval_summary.json")
    table_path = write_csv_atomically(run_table, output_dir / "hybrid_eval_table.csv")
    wf_table_path = write_csv_atomically(wf_table, output_dir / "hybrid_eval_wf_table.csv")
    report_markdown = _render_report_markdown(
        summary=summary_payload,
        run_table=run_table.sort("single_candidate_score", descending=True, nulls_last=True)
        if "single_candidate_score" in run_table.columns
        else run_table,
        wf_table=wf_table,
    )
    report_path = write_markdown_atomically(report_markdown, output_dir / "hybrid_eval_report.md")

    log.info(
        "Hybrid eval report created: output_dir=%s primary=%s secondary=%s",
        output_dir,
        summary_payload["final_verdicts"].get("PRIMARY_CANDIDATE"),
        summary_payload["final_verdicts"].get("SECONDARY_CANDIDATE"),
    )

    return HybridEvalReportResult(
        run_id=run_id,
        output_dir=output_dir,
        summary_path=summary_path,
        table_path=table_path,
        wf_table_path=wf_table_path,
        report_path=report_path,
    )

