"""Execution realism report builder from existing grid and walk-forward artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
class ExecutionRealismReportResult:
    """Artifact locations for one execution realism evaluation report."""

    run_id: str
    output_dir: Path
    summary_path: Path
    table_path: Path
    wf_table_path: Path
    report_path: Path


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


def _finite_json(payload: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        return value

    return convert(payload)


def _load_json(path: Path, *, loaded: list[str], missing: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        missing.append(str(path))
        return None
    loaded.append(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(run_dir: Path, *, loaded: list[str], missing: list[str]) -> pl.DataFrame | None:
    parquet = run_dir / "grid_metrics_table.parquet"
    csv = run_dir / "grid_metrics_table.csv"
    if parquet.exists():
        loaded.append(str(parquet))
        return pl.read_parquet(parquet)
    if csv.exists():
        loaded.append(str(csv))
        return pl.read_csv(csv)
    missing.append(str(parquet))
    missing.append(str(csv))
    return None


def _count_non_finite_cells(df: pl.DataFrame) -> int:
    if df.height == 0:
        return 0
    total = 0
    for col, dtype in df.schema.items():
        if not dtype.is_numeric():
            continue
        finite = int(df.select(pl.col(col).cast(pl.Float64, strict=False).is_finite().sum()).item() or 0)
        nulls = int(df.select(pl.col(col).null_count()).item() or 0)
        total += max(df.height - finite - nulls, 0)
    return total


def _infer_mode(summary: dict[str, Any] | None, metrics: pl.DataFrame | None) -> tuple[str | None, str | None, str | None]:
    source_type: str | None = None
    overlay_mode: str | None = None
    execution_profile: str | None = None

    if isinstance(summary, dict):
        sources = summary.get("sources")
        if isinstance(sources, list) and sources:
            first = sources[0]
            if isinstance(first, dict):
                source_type = str(first.get("source_type")) if first.get("source_type") is not None else None
                overlay_mode = str(first.get("overlay_mode")) if first.get("overlay_mode") is not None else None
        execution_profile = str(summary.get("execution_profile")) if summary.get("execution_profile") else None

    if metrics is not None and metrics.height > 0:
        if source_type is None and "source_type" in metrics.columns:
            source_type = str(metrics.get_column("source_type").drop_nulls().first())
        if overlay_mode is None and "overlay_mode" in metrics.columns:
            vals = [str(v) for v in metrics.get_column("overlay_mode").drop_nulls().to_list()]
            if vals:
                overlay_mode = vals[0]
        if execution_profile is None and "execution_profile" in metrics.columns:
            vals = [str(v) for v in metrics.get_column("execution_profile").drop_nulls().to_list()]
            if vals:
                # deterministic mode by frequency, then lexical.
                counts: dict[str, int] = {}
                for value in vals:
                    counts[value] = counts.get(value, 0) + 1
                execution_profile = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    overlay_mode = overlay_mode or "none"
    execution_profile = execution_profile or "none"
    return source_type, overlay_mode, execution_profile


def _discover_grid_run(
    *,
    root: Path,
    source_type: str,
    overlay_mode: str,
    execution_profile: str,
    loaded: list[str],
    missing: list[str],
) -> Path | None:
    if not root.exists():
        return None

    candidates: list[tuple[float, Path, str]] = []
    for run_dir in sorted(root.glob("grid-*")):
        summary_path = run_dir / "grid_summary.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path, loaded=loaded, missing=missing)
        if summary is None:
            continue
        scope = str(summary.get("scope") or "")
        if not scope.startswith(f"single-{source_type}"):
            continue
        mode = None
        profile = None
        sources = summary.get("sources")
        if isinstance(sources, list) and sources:
            first = sources[0]
            if isinstance(first, dict):
                mode = str(first.get("overlay_mode")) if first.get("overlay_mode") is not None else None
        if mode is None:
            mode = str(summary.get("overlay_mode")) if summary.get("overlay_mode") else "none"
        profile = str(summary.get("execution_profile")) if summary.get("execution_profile") else None

        if mode is None:
            mode = "none"
        if mode != overlay_mode:
            continue

        if profile is None:
            # fallback for older runs missing execution_profile in summary.
            metrics = _load_metrics(run_dir, loaded=loaded, missing=missing)
            _, _, inferred_profile = _infer_mode(summary, metrics)
            profile = inferred_profile

        if profile != execution_profile:
            continue

        mtime = run_dir.stat().st_mtime
        candidates.append((mtime, run_dir, profile))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _discover_wf_run(
    *,
    root: Path,
    source_type: str,
    overlay_mode: str,
    execution_profile: str,
    loaded: list[str],
    missing: list[str],
) -> Path | None:
    if not root.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    for run_dir in sorted(root.glob("wfgrid-*")):
        summary = _load_json(run_dir / "wf_grid_summary.json", loaded=loaded, missing=missing)
        if summary is None:
            continue

        sources = summary.get("sources")
        source_set = {str(v).lower() for v in sources} if isinstance(sources, list) else set()
        if source_type not in source_set:
            continue

        mode = str(summary.get("overlay_mode") or "none")
        profile = str(summary.get("execution_profile")) if summary.get("execution_profile") else None
        if profile is None:
            by_split_csv = run_dir / "wf_grid_by_split.csv"
            if by_split_csv.exists():
                loaded.append(str(by_split_csv))
                by_split = pl.read_csv(by_split_csv)
                if "execution_profile" in by_split.columns and by_split.height > 0:
                    vals = [str(v) for v in by_split.get_column("execution_profile").drop_nulls().to_list()]
                    if vals:
                        profile = vals[0]
        if profile is None:
            profile = "none"

        if mode != overlay_mode or profile != execution_profile:
            continue
        candidates.append((run_dir.stat().st_mtime, run_dir))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _pick_best_combo(metrics: pl.DataFrame) -> dict[str, Any] | None:
    if metrics.height == 0:
        return None
    frame = metrics
    if "status" in frame.columns:
        frame = frame.filter(pl.col("status") == "SUCCESS")
    if frame.height == 0:
        return None

    for col in ["robustness_score_v2", "expectancy", "profit_factor", "combo_id"]:
        if col not in frame.columns:
            if col == "combo_id":
                frame = frame.with_columns(pl.lit("").alias("combo_id"))
            else:
                frame = frame.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    best = frame.sort(
        [
            pl.col("robustness_score_v2").cast(pl.Float64, strict=False).fill_null(-1e18),
            pl.col("expectancy").cast(pl.Float64, strict=False).fill_null(-1e18),
            pl.col("profit_factor").cast(pl.Float64, strict=False).fill_null(-1e18),
            pl.col("combo_id").cast(pl.String).fill_null(""),
        ],
        descending=[True, True, True, False],
    ).head(1)
    return best.to_dicts()[0] if best.height > 0 else None


def _grid_row(
    *,
    run_label: str,
    run_dir: Path | None,
    loaded: list[str],
    missing: list[str],
) -> tuple[dict[str, Any], int]:
    row: dict[str, Any] = {
        "run_label": run_label,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "execution_profile": None,
        "overlay_mode": None,
        "overlay_enabled": None,
        "best_expectancy": None,
        "best_pf": None,
        "best_robustness_v2": None,
        "best_ret_cv": None,
        "best_zero_trade_share": None,
        "best_trade_count": None,
        "best_combo_id": None,
        "overlay_match_rate": None,
        "overlay_vetoed_signal_share": None,
        "overlay_unknown_rate": None,
        "exec_eligibility_rate": None,
        "exec_suppressed_signal_share": None,
        "exec_trade_avg_dollar_vol_20": None,
        "realism_aware_verdict": None,
        "status": "MISSING",
    }
    if run_dir is None:
        return row, 0

    summary = _load_json(run_dir / "grid_summary.json", loaded=loaded, missing=missing)
    metrics = _load_metrics(run_dir, loaded=loaded, missing=missing)
    if summary is None or metrics is None:
        row["status"] = "MISSING_ARTIFACTS"
        return row, _count_non_finite_cells(metrics) if metrics is not None else 0

    source_type, overlay_mode, execution_profile = _infer_mode(summary, metrics)
    top = _pick_best_combo(metrics)
    non_finite = _count_non_finite_cells(metrics)
    if top is None:
        row.update(
            {
                "overlay_mode": overlay_mode,
                "execution_profile": execution_profile,
                "overlay_enabled": bool(overlay_mode != "none"),
                "realism_aware_verdict": "NOT_TRADABLE",
                "status": "NO_SUCCESS_ROWS",
            }
        )
        return row, non_finite

    zero_share = _safe_float(summary.get("zero_trade_combo_share"))
    if zero_share is None and "is_zero_trade_combo" in metrics.columns:
        zero_share = _safe_float(
            metrics.select(pl.col("is_zero_trade_combo").cast(pl.Float64, strict=False).mean()).item()
        )

    row.update(
        {
            "execution_profile": execution_profile,
            "overlay_mode": overlay_mode,
            "overlay_enabled": bool(overlay_mode != "none"),
            "best_expectancy": _safe_float(top.get("expectancy")),
            "best_pf": _safe_float(top.get("profit_factor")),
            "best_robustness_v2": _safe_float(top.get("robustness_score_v2")),
            "best_ret_cv": _safe_float(top.get("ret_cv")),
            "best_zero_trade_share": zero_share,
            "best_trade_count": _safe_int(top.get("trade_count")),
            "best_combo_id": top.get("combo_id"),
            "overlay_match_rate": _safe_float(top.get("overlay_match_rate")),
            "overlay_vetoed_signal_share": _safe_float(top.get("overlay_vetoed_signal_share")),
            "overlay_unknown_rate": _safe_float(top.get("overlay_unknown_rate")),
            "exec_eligibility_rate": _safe_float(top.get("exec_eligibility_rate")),
            "exec_suppressed_signal_share": _safe_float(top.get("exec_suppressed_signal_share")),
            "exec_trade_avg_dollar_vol_20": _safe_float(top.get("exec_trade_avg_dollar_vol_20")),
            "status": "OK",
        }
    )
    trade_count = _safe_int(row.get("best_trade_count")) or 0
    zero_share = _safe_float(row.get("best_zero_trade_share")) or 0.0
    row["realism_aware_verdict"] = "NOT_TRADABLE" if (trade_count <= 0 or zero_share >= 1.0) else "TRADABLE"
    return row, non_finite


def _normalize_metric(values: list[float | None], *, higher_better: bool) -> list[float]:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return [0.5] * len(values)
    v_min = min(finite)
    v_max = max(finite)
    if np.isclose(v_max, v_min):
        return [0.5 if v is not None else 0.0 for v in values]
    out: list[float] = []
    denom = v_max - v_min
    for value in values:
        if value is None:
            out.append(0.0)
            continue
        scaled = (value - v_min) / denom
        out.append(float(scaled if higher_better else (1.0 - scaled)))
    return out


def _score_rows(table: pl.DataFrame) -> pl.DataFrame:
    if table.height == 0:
        return table

    rows = table.to_dicts()
    exp_vals = [_safe_float(r.get("best_expectancy")) for r in rows]
    pf_vals = [_safe_float(r.get("best_pf")) for r in rows]
    rob_vals = [_safe_float(r.get("best_robustness_v2")) for r in rows]
    cv_vals = [_safe_float(r.get("best_ret_cv")) for r in rows]
    zero_vals = [_safe_float(r.get("best_zero_trade_share")) for r in rows]

    exp_n = _normalize_metric(exp_vals, higher_better=True)
    pf_n = _normalize_metric(pf_vals, higher_better=True)
    rob_n = _normalize_metric(rob_vals, higher_better=True)
    cv_n = _normalize_metric(cv_vals, higher_better=False)
    zero_n = _normalize_metric(zero_vals, higher_better=False)

    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        trade_count = _safe_int(row.get("best_trade_count")) or 0
        zero_share = _safe_float(row.get("best_zero_trade_share")) or 0.0
        is_not_tradable = bool(
            str(row.get("status")) != "OK"
            or str(row.get("realism_aware_verdict")) == "NOT_TRADABLE"
            or trade_count <= 0
            or zero_share >= 1.0
        )
        score = 100.0 * (
            (0.30 * exp_n[idx])
            + (0.20 * pf_n[idx])
            + (0.25 * rob_n[idx])
            + (0.15 * cv_n[idx])
            + (0.10 * zero_n[idx])
        )
        if trade_count < 30:
            score -= 10.0
        if bool(row.get("best_zero_trade_share") is not None and zero_share > 0.5):
            score -= 10.0
        if is_not_tradable:
            row["candidate_score"] = 0.0
            row["realism_aware_verdict"] = "NOT_TRADABLE"
        else:
            row["candidate_score"] = _safe_float(max(0.0, min(100.0, score)))
            row["realism_aware_verdict"] = "TRADABLE"
        out_rows.append(row)
    return pl.DataFrame(out_rows)


def _build_wf_split_best(
    wf_dir: Path,
    *,
    loaded: list[str],
    missing: list[str],
) -> pl.DataFrame:
    by_split_path = wf_dir / "wf_grid_by_split.csv"
    if not by_split_path.exists():
        missing.append(str(by_split_path))
        return pl.DataFrame()
    loaded.append(str(by_split_path))

    by_split = pl.read_csv(by_split_path)
    if by_split.height == 0:
        return pl.DataFrame()
    if "status" in by_split.columns:
        by_split = by_split.filter(pl.col("status") == "SUCCESS")

    rows: list[dict[str, Any]] = []
    for row in by_split.to_dicts():
        train_end = str(row.get("train_end"))
        grid_run_dir_raw = row.get("grid_run_dir")
        if not grid_run_dir_raw:
            continue
        grid_run_dir = Path(str(grid_run_dir_raw))
        metrics = _load_metrics(grid_run_dir, loaded=loaded, missing=missing)
        if metrics is None or metrics.height == 0:
            continue
        if "source_type" in metrics.columns:
            hmm_only = metrics.filter(pl.col("source_type") == "hmm")
            if hmm_only.height > 0:
                metrics = hmm_only
        top = _pick_best_combo(metrics)
        if top is None:
            continue
        rows.append(
            {
                "train_end": train_end,
                "best_expectancy": _safe_float(top.get("expectancy")),
                "best_pf": _safe_float(top.get("profit_factor")),
                "best_robustness_v2": _safe_float(top.get("robustness_score_v2")),
                "best_ret_cv": _safe_float(top.get("ret_cv")),
                "best_zero_trade_share": _safe_float(row.get("zero_trade_combo_share")),
                "overlay_match_rate": _safe_float(top.get("overlay_match_rate")),
                "overlay_vetoed_signal_share": _safe_float(top.get("overlay_vetoed_signal_share")),
                "exec_eligibility_rate": _safe_float(top.get("exec_eligibility_rate")),
                "exec_suppressed_signal_share": _safe_float(top.get("exec_suppressed_signal_share")),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _mean_delta(base: list[float | None], hybrid: list[float | None]) -> float | None:
    vals: list[float] = []
    for b, h in zip(base, hybrid):
        if b is None or h is None:
            continue
        if not (np.isfinite(b) and np.isfinite(h)):
            continue
        vals.append(float(h - b))
    if not vals:
        return None
    return _safe_float(float(np.mean(vals)))


def _build_wf_comparison(
    *,
    baseline_dir: Path | None,
    hybrid_dir: Path | None,
    loaded: list[str],
    missing: list[str],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if baseline_dir is None or hybrid_dir is None:
        return pl.DataFrame(), {
            "status": "MISSING_WF_INPUTS",
            "split_count": 0,
        }

    base = _build_wf_split_best(baseline_dir, loaded=loaded, missing=missing)
    hyb = _build_wf_split_best(hybrid_dir, loaded=loaded, missing=missing)
    if base.height == 0 or hyb.height == 0:
        return pl.DataFrame(), {
            "status": "NO_WF_SPLITS",
            "split_count": 0,
        }

    merged = base.join(hyb, on="train_end", how="inner", suffix="_hybrid")
    if merged.height == 0:
        return pl.DataFrame(), {
            "status": "NO_SHARED_SPLITS",
            "split_count": 0,
        }

    rows = merged.to_dicts()
    wins_exp = 0
    wins_pf = 0
    wins_rob = 0
    wins_ret_cv = 0
    for row in rows:
        b_exp = _safe_float(row.get("best_expectancy"))
        h_exp = _safe_float(row.get("best_expectancy_hybrid"))
        b_pf = _safe_float(row.get("best_pf"))
        h_pf = _safe_float(row.get("best_pf_hybrid"))
        b_rob = _safe_float(row.get("best_robustness_v2"))
        h_rob = _safe_float(row.get("best_robustness_v2_hybrid"))
        b_cv = _safe_float(row.get("best_ret_cv"))
        h_cv = _safe_float(row.get("best_ret_cv_hybrid"))

        if b_exp is not None and h_exp is not None and h_exp > b_exp:
            wins_exp += 1
        if b_pf is not None and h_pf is not None and h_pf > b_pf:
            wins_pf += 1
        if b_rob is not None and h_rob is not None and h_rob > b_rob:
            wins_rob += 1
        if b_cv is not None and h_cv is not None and h_cv < b_cv:
            wins_ret_cv += 1

    wf_row = {
        "source": "hmm",
        "comparison": "baseline_vs_hybrid",
        "split_count": int(merged.height),
        "hybrid_wins_expectancy": int(wins_exp),
        "hybrid_wins_pf": int(wins_pf),
        "hybrid_wins_robustness_v2": int(wins_rob),
        "hybrid_wins_ret_cv": int(wins_ret_cv),
        "avg_delta_expectancy": _mean_delta(
            [_safe_float(v) for v in merged.get_column("best_expectancy").to_list()],
            [_safe_float(v) for v in merged.get_column("best_expectancy_hybrid").to_list()],
        ),
        "avg_delta_pf": _mean_delta(
            [_safe_float(v) for v in merged.get_column("best_pf").to_list()],
            [_safe_float(v) for v in merged.get_column("best_pf_hybrid").to_list()],
        ),
        "avg_delta_robustness_v2": _mean_delta(
            [_safe_float(v) for v in merged.get_column("best_robustness_v2").to_list()],
            [_safe_float(v) for v in merged.get_column("best_robustness_v2_hybrid").to_list()],
        ),
        "avg_delta_ret_cv": _mean_delta(
            [_safe_float(v) for v in merged.get_column("best_ret_cv").to_list()],
            [_safe_float(v) for v in merged.get_column("best_ret_cv_hybrid").to_list()],
        ),
        "avg_delta_zero_trade_share": _mean_delta(
            [_safe_float(v) for v in merged.get_column("best_zero_trade_share").to_list()],
            [_safe_float(v) for v in merged.get_column("best_zero_trade_share_hybrid").to_list()],
        ),
        "baseline_exec_eligibility_rate": _safe_float(
            merged.select(pl.col("exec_eligibility_rate").cast(pl.Float64, strict=False).mean()).item()
        ),
        "hybrid_exec_eligibility_rate": _safe_float(
            merged.select(pl.col("exec_eligibility_rate_hybrid").cast(pl.Float64, strict=False).mean()).item()
        ),
        "hybrid_overlay_match_rate": _safe_float(
            merged.select(pl.col("overlay_match_rate_hybrid").cast(pl.Float64, strict=False).mean()).item()
        ),
        "hybrid_overlay_vetoed_signal_share": _safe_float(
            merged.select(pl.col("overlay_vetoed_signal_share_hybrid").cast(pl.Float64, strict=False).mean()).item()
        ),
        "status": "OK",
    }

    wf_stats = {
        "status": "OK",
        "split_count": int(merged.height),
        "hybrid_wins_expectancy": int(wins_exp),
        "hybrid_wins_pf": int(wins_pf),
        "hybrid_wins_robustness_v2": int(wins_rob),
        "hybrid_wins_ret_cv": int(wins_ret_cv),
    }
    return pl.DataFrame([wf_row]), wf_stats


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


def _build_realism_note(table: pl.DataFrame) -> str:
    if table.height == 0:
        return "insufficient data"
    if "realism_aware_verdict" in table.columns:
        tradable = table.filter(pl.col("realism_aware_verdict") != "NOT_TRADABLE")
        if tradable.height == 0:
            return "NO_TRADABLE_CANDIDATE (likely over-restrictive execution filters)"

    rows = {str(r.get("run_label")): r for r in table.to_dicts()}
    none = rows.get("HMM baseline (none)")
    lite = rows.get("HMM baseline (lite)")
    strict = rows.get("HMM baseline (strict)")

    if none is None or lite is None:
        return "baseline/lite pair not available"

    d_exp_lite = _safe_float(lite.get("best_expectancy"))
    d_exp_none = _safe_float(none.get("best_expectancy"))
    d_pf_lite = _safe_float(lite.get("best_pf"))
    d_pf_none = _safe_float(none.get("best_pf"))
    d_cv_lite = _safe_float(lite.get("best_ret_cv"))
    d_cv_none = _safe_float(none.get("best_ret_cv"))
    if d_exp_lite is None or d_exp_none is None or d_pf_lite is None or d_pf_none is None:
        return "insufficient baseline metrics"

    exp_delta = d_exp_lite - d_exp_none
    pf_delta = d_pf_lite - d_pf_none
    cv_delta = None
    if d_cv_lite is not None and d_cv_none is not None:
        cv_delta = d_cv_lite - d_cv_none

    strict_msg = ""
    if strict is not None:
        s_exp = _safe_float(strict.get("best_expectancy"))
        s_pf = _safe_float(strict.get("best_pf"))
        if s_exp is not None and s_pf is not None:
            strict_msg = f" strict_delta_exp={_safe_float(s_exp - d_exp_none)} strict_delta_pf={_safe_float(s_pf - d_pf_none)}."

    if exp_delta >= -0.001 and pf_delta >= -0.05:
        return f"EDGE_RETAINED_UNDER_LITE (delta_exp={_safe_float(exp_delta)}, delta_pf={_safe_float(pf_delta)}, delta_ret_cv={_safe_float(cv_delta)}).{strict_msg}"
    if exp_delta < 0 and cv_delta is not None and cv_delta < 0:
        return f"EDGE_COMPROMISED_BUT_CLEANER_UNDER_LITE (delta_exp={_safe_float(exp_delta)}, delta_pf={_safe_float(pf_delta)}, delta_ret_cv={_safe_float(cv_delta)}).{strict_msg}"
    return f"TOO_FRAGILE_UNDER_REALISM (delta_exp={_safe_float(exp_delta)}, delta_pf={_safe_float(pf_delta)}, delta_ret_cv={_safe_float(cv_delta)}).{strict_msg}"


def run_execution_realism_report(
    settings: AppSettings,
    *,
    hmm_none_grid_dir: Path | None,
    hmm_lite_grid_dir: Path | None,
    hmm_strict_grid_dir: Path | None,
    hmm_overlay_allow_lite_grid_dir: Path | None,
    hmm_overlay_block_lite_grid_dir: Path | None,
    wf_hmm_baseline_lite_dir: Path | None,
    wf_hmm_hybrid_lite_dir: Path | None,
    logger: logging.Logger | None = None,
) -> ExecutionRealismReportResult:
    """Build execution realism comparison report from existing grid/WF artifacts."""

    effective_logger = logger or LOGGER
    loaded: list[str] = []
    missing: list[str] = []

    grid_root = settings.paths.artifacts_root / "backtest_sensitivity"
    wf_root = settings.paths.artifacts_root / "backtest_sensitivity_walkforward"

    if hmm_none_grid_dir is None:
        hmm_none_grid_dir = _discover_grid_run(
            root=grid_root,
            source_type="hmm",
            overlay_mode="none",
            execution_profile="none",
            loaded=loaded,
            missing=missing,
        )
    if hmm_lite_grid_dir is None:
        hmm_lite_grid_dir = _discover_grid_run(
            root=grid_root,
            source_type="hmm",
            overlay_mode="none",
            execution_profile="lite",
            loaded=loaded,
            missing=missing,
        )
    if hmm_strict_grid_dir is None:
        hmm_strict_grid_dir = _discover_grid_run(
            root=grid_root,
            source_type="hmm",
            overlay_mode="none",
            execution_profile="strict",
            loaded=loaded,
            missing=missing,
        )
    if hmm_overlay_allow_lite_grid_dir is None:
        hmm_overlay_allow_lite_grid_dir = _discover_grid_run(
            root=grid_root,
            source_type="hmm",
            overlay_mode="allow_only",
            execution_profile="lite",
            loaded=loaded,
            missing=missing,
        )
    if hmm_overlay_block_lite_grid_dir is None:
        hmm_overlay_block_lite_grid_dir = _discover_grid_run(
            root=grid_root,
            source_type="hmm",
            overlay_mode="block_veto",
            execution_profile="lite",
            loaded=loaded,
            missing=missing,
        )

    if wf_hmm_baseline_lite_dir is None:
        wf_hmm_baseline_lite_dir = _discover_wf_run(
            root=wf_root,
            source_type="hmm",
            overlay_mode="none",
            execution_profile="lite",
            loaded=loaded,
            missing=missing,
        )
    if wf_hmm_hybrid_lite_dir is None:
        wf_hmm_hybrid_lite_dir = _discover_wf_run(
            root=wf_root,
            source_type="hmm",
            overlay_mode="allow_only",
            execution_profile="lite",
            loaded=loaded,
            missing=missing,
        )

    run_specs = [
        ("HMM baseline (none)", hmm_none_grid_dir),
        ("HMM baseline (lite)", hmm_lite_grid_dir),
        ("HMM baseline (strict)", hmm_strict_grid_dir),
        ("HMM overlay allow_only (lite)", hmm_overlay_allow_lite_grid_dir),
        ("HMM overlay block_veto (lite)", hmm_overlay_block_lite_grid_dir),
    ]

    rows: list[dict[str, Any]] = []
    non_finite_cells = 0
    for label, run_dir in run_specs:
        row, non_finite = _grid_row(run_label=label, run_dir=run_dir, loaded=loaded, missing=missing)
        rows.append(row)
        non_finite_cells += non_finite

    table = pl.DataFrame(rows)
    if table.height > 0:
        table = _score_rows(table)
        table = table.sort(
            [
                pl.col("candidate_score").cast(pl.Float64, strict=False).fill_null(-1e18),
                pl.col("best_robustness_v2").cast(pl.Float64, strict=False).fill_null(-1e18),
                pl.col("best_expectancy").cast(pl.Float64, strict=False).fill_null(-1e18),
                pl.col("run_label").cast(pl.String),
            ],
            descending=[True, True, True, False],
        )

    wf_table, wf_stats = _build_wf_comparison(
        baseline_dir=wf_hmm_baseline_lite_dir,
        hybrid_dir=wf_hmm_hybrid_lite_dir,
        loaded=loaded,
        missing=missing,
    )

    top_ok = table.filter(pl.col("status") == "OK") if table.height > 0 else table
    tradable = (
        top_ok.filter(pl.col("realism_aware_verdict") != "NOT_TRADABLE")
        if top_ok.height > 0 and "realism_aware_verdict" in top_ok.columns
        else top_ok
    )
    primary = None
    secondary = None
    if tradable.height > 0:
        top_rows = tradable.sort(
            [
                pl.col("candidate_score").cast(pl.Float64, strict=False).fill_null(-1e18),
                pl.col("run_label").cast(pl.String),
            ],
            descending=[True, False],
        ).to_dicts()
        if top_rows:
            primary = top_rows[0].get("run_label")
        if len(top_rows) > 1:
            secondary = top_rows[1].get("run_label")
    elif top_ok.height > 0:
        primary = "NO_TRADABLE_CANDIDATE"
        secondary = None

    realism_note = _build_realism_note(table)

    key_deltas: dict[str, Any] = {}
    lookup = {str(r.get("run_label")): r for r in table.to_dicts()} if table.height > 0 else {}
    base = lookup.get("HMM baseline (none)")
    lite = lookup.get("HMM baseline (lite)")
    strict = lookup.get("HMM baseline (strict)")
    allow = lookup.get("HMM overlay allow_only (lite)")
    block = lookup.get("HMM overlay block_veto (lite)")

    def delta(a: dict[str, Any] | None, b: dict[str, Any] | None, key: str) -> float | None:
        av = _safe_float(a.get(key)) if a is not None else None
        bv = _safe_float(b.get(key)) if b is not None else None
        if av is None or bv is None:
            return None
        return _safe_float(av - bv)

    key_deltas["lite_minus_none"] = {
        "expectancy": delta(lite, base, "best_expectancy"),
        "pf": delta(lite, base, "best_pf"),
        "robustness_v2": delta(lite, base, "best_robustness_v2"),
        "ret_cv": delta(lite, base, "best_ret_cv"),
        "exec_eligibility_rate": delta(lite, base, "exec_eligibility_rate"),
        "exec_suppressed_signal_share": delta(lite, base, "exec_suppressed_signal_share"),
    }
    key_deltas["strict_minus_none"] = {
        "expectancy": delta(strict, base, "best_expectancy"),
        "pf": delta(strict, base, "best_pf"),
        "robustness_v2": delta(strict, base, "best_robustness_v2"),
        "ret_cv": delta(strict, base, "best_ret_cv"),
    }
    key_deltas["allow_only_lite_minus_baseline_lite"] = {
        "expectancy": delta(allow, lite, "best_expectancy"),
        "pf": delta(allow, lite, "best_pf"),
        "robustness_v2": delta(allow, lite, "best_robustness_v2"),
        "ret_cv": delta(allow, lite, "best_ret_cv"),
        "overlay_vetoed_signal_share": _safe_float(allow.get("overlay_vetoed_signal_share")) if allow else None,
    }
    key_deltas["block_veto_lite_minus_baseline_lite"] = {
        "expectancy": delta(block, lite, "best_expectancy"),
        "pf": delta(block, lite, "best_pf"),
        "robustness_v2": delta(block, lite, "best_robustness_v2"),
        "ret_cv": delta(block, lite, "best_ret_cv"),
        "overlay_vetoed_signal_share": _safe_float(block.get("overlay_vetoed_signal_share")) if block else None,
    }

    run_id = f"exec-realism-{uuid4().hex[:12]}"
    output_dir = settings.paths.artifacts_root / "execution_realism_reports" / f"{run_id}_execution_realism_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "execution_realism_summary.json"
    table_path = output_dir / "execution_realism_table.csv"
    wf_table_path = output_dir / "execution_realism_wf_table.csv"
    report_path = output_dir / "execution_realism_report.md"

    summary_payload = {
        "run_id": run_id,
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "compared_runs": {
            "hmm_none_grid_dir": str(hmm_none_grid_dir) if hmm_none_grid_dir else None,
            "hmm_lite_grid_dir": str(hmm_lite_grid_dir) if hmm_lite_grid_dir else None,
            "hmm_strict_grid_dir": str(hmm_strict_grid_dir) if hmm_strict_grid_dir else None,
            "hmm_overlay_allow_lite_grid_dir": (
                str(hmm_overlay_allow_lite_grid_dir) if hmm_overlay_allow_lite_grid_dir else None
            ),
            "hmm_overlay_block_lite_grid_dir": (
                str(hmm_overlay_block_lite_grid_dir) if hmm_overlay_block_lite_grid_dir else None
            ),
            "wf_hmm_baseline_lite_dir": str(wf_hmm_baseline_lite_dir) if wf_hmm_baseline_lite_dir else None,
            "wf_hmm_hybrid_lite_dir": str(wf_hmm_hybrid_lite_dir) if wf_hmm_hybrid_lite_dir else None,
        },
        "top_combo_per_run": {
            str(row.get("run_label")): {
                "combo_id": row.get("best_combo_id"),
                "expectancy": row.get("best_expectancy"),
                "profit_factor": row.get("best_pf"),
                "robustness_v2": row.get("best_robustness_v2"),
                "ret_cv": row.get("best_ret_cv"),
                "candidate_score": row.get("candidate_score"),
            }
            for row in table.to_dicts()
        },
        "key_deltas": key_deltas,
        "wf_consistency_stats": wf_stats,
        "final_verdicts": {
            "PRIMARY_CANDIDATE": primary,
            "SECONDARY_CANDIDATE": secondary,
            "REALISM_SENSITIVITY_NOTE": realism_note,
        },
        "sanity": {
            "files_loaded": sorted(set(loaded)),
            "missing_files": sorted(set(missing)),
            "non_finite_cells_detected": int(non_finite_cells),
            "status": (
                "OK"
                if (primary is not None and primary != "NO_TRADABLE_CANDIDATE" and non_finite_cells == 0)
                else "WARN"
            ),
        },
    }

    lines: list[str] = []
    lines.append("# Execution Realism Report v1")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- PRIMARY_CANDIDATE: `{primary}`")
    lines.append(f"- SECONDARY_CANDIDATE: `{secondary}`")
    lines.append(f"- REALISM_SENSITIVITY_NOTE: `{realism_note}`")
    lines.append("")
    lines.append("## Single-Run Comparison")
    lines.append(_to_markdown_table(table, max_rows=10))
    lines.append("")
    lines.append("## WF Comparison")
    lines.append(_to_markdown_table(wf_table, max_rows=10))
    lines.append("")
    lines.append("## Key Deltas")
    lines.append(f"- lite_minus_none: `{key_deltas.get('lite_minus_none')}`")
    lines.append(f"- strict_minus_none: `{key_deltas.get('strict_minus_none')}`")
    lines.append(
        f"- allow_only_lite_minus_baseline_lite: `{key_deltas.get('allow_only_lite_minus_baseline_lite')}`"
    )
    lines.append(
        f"- block_veto_lite_minus_baseline_lite: `{key_deltas.get('block_veto_lite_minus_baseline_lite')}`"
    )
    lines.append("")
    lines.append("## QA")
    lines.append(f"- non_finite_cells_detected: `{summary_payload['sanity']['non_finite_cells_detected']}`")
    lines.append(f"- missing_files_count: `{len(summary_payload['sanity']['missing_files'])}`")
    lines.append(f"- status: `{summary_payload['sanity']['status']}`")
    lines.append("")

    write_json_atomically(_finite_json(summary_payload), summary_path)
    write_csv_atomically(table, table_path)
    write_csv_atomically(wf_table, wf_table_path)
    write_markdown_atomically("\n".join(lines) + "\n", report_path)

    effective_logger.info(
        "backtest.execution_realism.report.complete run_id=%s output=%s",
        run_id,
        output_dir,
    )
    return ExecutionRealismReportResult(
        run_id=run_id,
        output_dir=output_dir,
        summary_path=summary_path,
        table_path=table_path,
        wf_table_path=wf_table_path,
        report_path=report_path,
    )
