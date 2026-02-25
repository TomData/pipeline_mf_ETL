"""Production Candidate Pack v1 builder and sanity checks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from mf_etl.backtest.production_candidates_models import ProductionCandidatePackResult
from mf_etl.backtest.production_candidates_reports import write_production_candidate_reports
from mf_etl.config import AppSettings

LOGGER = logging.getLogger(__name__)


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


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_json(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}")


def _load_grid_artifacts(grid_dir: Path) -> tuple[dict[str, Any], pl.DataFrame]:
    summary_path = grid_dir / "grid_summary.json"
    metrics_path = grid_dir / "grid_metrics_table.parquet"
    _require_file(summary_path)
    _require_file(metrics_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = pl.read_parquet(metrics_path)
    return summary, metrics


def _desc_key(value: Any) -> float:
    out = _safe_float(value)
    if out is None:
        return float("inf")
    return -out


def _asc_key(value: Any) -> float:
    out = _safe_float(value)
    if out is None:
        return float("inf")
    return out


def _select_best_row(
    *,
    metrics: pl.DataFrame,
    profile: str,
    min_trades: int,
    warnings: list[str],
) -> tuple[dict[str, Any], int]:
    success = metrics
    if "status" in success.columns:
        success = success.filter(pl.col("status") == "SUCCESS")
    if "is_zero_trade_combo" in success.columns:
        success = success.filter(~pl.col("is_zero_trade_combo").cast(pl.Boolean, strict=False).fill_null(False))
    if "trade_count" in success.columns:
        success = success.filter(pl.col("trade_count").cast(pl.Int64, strict=False).fill_null(0) > 0)

    if success.height == 0:
        raise ValueError(f"No non-zero successful combos available for {profile}.")

    trade_threshold = min_trades
    eligible = success.filter(pl.col("trade_count").cast(pl.Int64, strict=False).fill_null(0) >= trade_threshold)
    if eligible.height == 0 and min_trades > 10:
        trade_threshold = 10
        warnings.append(
            f"{profile}: no combos with trade_count >= {min_trades}; relaxed to trade_count >= 10."
        )
        eligible = success.filter(pl.col("trade_count").cast(pl.Int64, strict=False).fill_null(0) >= trade_threshold)
    if eligible.height == 0:
        raise ValueError(f"No combos satisfy trade_count >= 10 for {profile}.")

    rows = eligible.to_dicts()
    if profile == "alpha":
        rows = sorted(
            rows,
            key=lambda row: (
                _desc_key(row.get("robustness_score_v2")),
                _desc_key(row.get("expectancy")),
                _desc_key(row.get("profit_factor")),
                _desc_key(row.get("trade_count")),
                str(row.get("combo_id") or "~"),
            ),
        )
    else:
        rows = sorted(
            rows,
            key=lambda row: (
                _desc_key(row.get("profit_factor")),
                _desc_key(row.get("robustness_score_v2")),
                _asc_key(row.get("ret_cv")),
                _desc_key(row.get("trade_count")),
                str(row.get("combo_id") or "~"),
            ),
        )
    return rows[0], trade_threshold


def _load_backtest_run_config(backtest_run_dir: Path | None) -> dict[str, Any]:
    if backtest_run_dir is None:
        return {}
    config_path = backtest_run_dir / "backtest_run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_execution_filter_summary(backtest_run_dir: Path | None) -> dict[str, Any]:
    if backtest_run_dir is None:
        return {}
    summary_path = backtest_run_dir / "execution_filter_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _build_wf_consistency(
    *,
    wf_baseline_dir: Path | None,
    wf_hybrid_dir: Path | None,
    execution_realism_report_dir: Path | None,
) -> dict[str, Any]:
    if execution_realism_report_dir is not None:
        table_path = execution_realism_report_dir / "execution_realism_wf_table.csv"
        if table_path.exists():
            table = pl.read_csv(table_path)
            if table.height > 0:
                row = table.filter(pl.col("source") == "hmm").head(1).to_dicts()
                if row:
                    return row[0]

    if wf_baseline_dir is None or wf_hybrid_dir is None:
        return {}

    base_path = wf_baseline_dir / "wf_grid_source_summary.csv"
    hybrid_path = wf_hybrid_dir / "wf_grid_source_summary.csv"
    if not base_path.exists() or not hybrid_path.exists():
        return {}

    base = pl.read_csv(base_path).filter(pl.col("source_type") == "hmm").head(1).to_dicts()
    hybrid = pl.read_csv(hybrid_path).filter(pl.col("source_type") == "hmm").head(1).to_dicts()
    if not base or not hybrid:
        return {}

    b = base[0]
    h = hybrid[0]
    return {
        "source": "hmm",
        "split_count": _safe_int(h.get("splits_covered")),
        "avg_delta_expectancy": _safe_float(h.get("expectancy_mean")) - (_safe_float(b.get("expectancy_mean")) or 0.0),
        "avg_delta_pf": _safe_float(h.get("profit_factor_mean")) - (_safe_float(b.get("profit_factor_mean")) or 0.0),
        "avg_delta_robustness_v2": _safe_float(h.get("robustness_v2_mean")) - (_safe_float(b.get("robustness_v2_mean")) or 0.0),
        "avg_delta_ret_cv": _safe_float(h.get("ret_cv_median")) - (_safe_float(b.get("ret_cv_median")) or 0.0),
    }


def _candidate_payload(
    *,
    label: str,
    selected_row: dict[str, Any],
    selected_grid_dir: Path,
    min_trade_threshold_used: int,
    wf_consistency: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    backtest_run_dir = Path(str(selected_row.get("backtest_run_dir"))) if selected_row.get("backtest_run_dir") else None
    run_config = _load_backtest_run_config(backtest_run_dir)
    execution_summary = _load_execution_filter_summary(backtest_run_dir)

    candidate = {
        "label": label,
        "input_type": selected_row.get("source_type"),
        "input_file": run_config.get("input_file") or selected_row.get("input_file"),
        "validation_run_dir": run_config.get("validation_run_dir") or selected_row.get("validation_run_dir"),
        "selected_from_grid_dir": str(selected_grid_dir),
        "combo_id": selected_row.get("combo_id"),
        "backtest_run_dir": str(backtest_run_dir) if backtest_run_dir is not None else None,
        "selection_trade_threshold_used": min_trade_threshold_used,
        "overlay": {
            "enabled": bool(selected_row.get("overlay_enabled")),
            "mode": selected_row.get("overlay_mode"),
            "overlay_cluster_file": run_config.get("overlay_cluster_file"),
            "overlay_cluster_hardening_dir": run_config.get("overlay_cluster_hardening_dir"),
            "match_rate": _safe_float(selected_row.get("overlay_match_rate")),
            "veto_share": _safe_float(selected_row.get("overlay_vetoed_signal_share")),
            "unknown_rate": _safe_float(selected_row.get("overlay_unknown_rate")),
        },
        "execution_realism": {
            "profile": selected_row.get("execution_profile"),
            "thresholds_used": {
                "min_price": run_config.get("exec_min_price"),
                "min_dollar_vol20": run_config.get("exec_min_dollar_vol20"),
                "max_vol_pct": run_config.get("exec_max_vol_pct"),
                "min_history_bars": run_config.get("exec_min_history_bars"),
            },
            "eligibility_rate": _safe_float(selected_row.get("exec_eligibility_rate")),
            "suppression_breakdown": {
                "suppressed_signal_share": _safe_float(selected_row.get("exec_suppressed_signal_share")),
                "price_share": _safe_float(selected_row.get("exec_suppressed_by_price_share")),
                "liquidity_share": _safe_float(selected_row.get("exec_suppressed_by_liquidity_share")),
                "vol_share": _safe_float(selected_row.get("exec_suppressed_by_vol_share")),
                "warmup_share": _safe_float(selected_row.get("exec_suppressed_by_warmup_share")),
            },
            "vol_metric_source": selected_row.get("exec_vol_metric_source"),
            "vol_unit_detected": selected_row.get("exec_vol_unit_detected"),
        },
        "strategy_params": {
            "signal_mode": selected_row.get("signal_mode"),
            "exit_mode": selected_row.get("exit_mode"),
            "hold_bars": _safe_int(selected_row.get("hold_bars")),
            "fee_bps_per_side": _safe_float(selected_row.get("fee_bps_per_side")),
            "slippage_bps_per_side": _safe_float(selected_row.get("slippage_bps_per_side")),
            "allow_overlap": bool(selected_row.get("allow_overlap")),
        },
        "expected_behavior_snapshot": {
            "best_expectancy": _safe_float(selected_row.get("expectancy")),
            "PF": _safe_float(selected_row.get("profit_factor")),
            "robustness_v2": _safe_float(selected_row.get("robustness_score_v2")),
            "ret_cv": _safe_float(selected_row.get("ret_cv")),
            "trade_count": _safe_int(selected_row.get("trade_count")),
            "zero_trade_share": _safe_float(selected_row.get("is_zero_trade_combo")),
            "ret_p10": _safe_float(selected_row.get("ret_p10")),
            "downside_std": _safe_float(selected_row.get("downside_std")),
        },
        "wf_consistency": wf_consistency or {},
        "notes": execution_summary.get("warnings", []),
    }

    table_row = {
        "label": label,
        "selected_from_grid_dir": str(selected_grid_dir),
        "combo_id": selected_row.get("combo_id"),
        "signal_mode": selected_row.get("signal_mode"),
        "exit_mode": selected_row.get("exit_mode"),
        "hold_bars": _safe_int(selected_row.get("hold_bars")),
        "fee_bps_per_side": _safe_float(selected_row.get("fee_bps_per_side")),
        "slippage_bps_per_side": _safe_float(selected_row.get("slippage_bps_per_side")),
        "overlay_mode": selected_row.get("overlay_mode"),
        "execution_profile": selected_row.get("execution_profile"),
        "trade_count": _safe_int(selected_row.get("trade_count")),
        "expectancy": _safe_float(selected_row.get("expectancy")),
        "profit_factor": _safe_float(selected_row.get("profit_factor")),
        "robustness_score_v2": _safe_float(selected_row.get("robustness_score_v2")),
        "ret_cv": _safe_float(selected_row.get("ret_cv")),
        "zero_trade_combo": bool(selected_row.get("is_zero_trade_combo")),
        "overlay_match_rate": _safe_float(selected_row.get("overlay_match_rate")),
        "overlay_vetoed_signal_share": _safe_float(selected_row.get("overlay_vetoed_signal_share")),
        "exec_eligibility_rate": _safe_float(selected_row.get("exec_eligibility_rate")),
        "exec_suppressed_signal_share": _safe_float(selected_row.get("exec_suppressed_signal_share")),
        "selection_trade_threshold_used": min_trade_threshold_used,
    }
    return candidate, table_row


def run_production_candidates_build(
    settings: AppSettings,
    *,
    a1_grid_dir: Path,
    a2_grid_dir: Path,
    a3_grid_dir: Path,
    b_grid_dir: Path,
    c_grid_dir: Path,
    wf_baseline_dir: Path | None,
    wf_hybrid_dir: Path | None,
    execution_realism_report_dir: Path | None,
    out_dir: Path | None,
    min_trades: int,
    include_exec2: bool,
    logger: logging.Logger | None = None,
) -> ProductionCandidatePackResult:
    """Build deterministic Production Candidate Pack v1 artifacts."""

    effective_logger = logger or LOGGER
    warnings: list[str] = []

    _, a1_metrics = _load_grid_artifacts(a1_grid_dir)
    _, a2_metrics = _load_grid_artifacts(a2_grid_dir)
    _, _a3_metrics = _load_grid_artifacts(a3_grid_dir)
    _, b_metrics = _load_grid_artifacts(b_grid_dir)
    _, _c_metrics = _load_grid_artifacts(c_grid_dir)

    selected_alpha, alpha_threshold = _select_best_row(
        metrics=a1_metrics,
        profile="alpha",
        min_trades=min_trades,
        warnings=warnings,
    )
    selected_exec, exec_threshold = _select_best_row(
        metrics=b_metrics,
        profile="exec",
        min_trades=min_trades,
        warnings=warnings,
    )

    wf_consistency = _build_wf_consistency(
        wf_baseline_dir=wf_baseline_dir,
        wf_hybrid_dir=wf_hybrid_dir,
        execution_realism_report_dir=execution_realism_report_dir,
    )

    packet_candidates: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []

    alpha_payload, alpha_table = _candidate_payload(
        label="CANDIDATE_ALPHA",
        selected_row=selected_alpha,
        selected_grid_dir=a1_grid_dir,
        min_trade_threshold_used=alpha_threshold,
        wf_consistency={},
    )
    packet_candidates["CANDIDATE_ALPHA"] = alpha_payload
    table_rows.append(alpha_table)

    exec_payload, exec_table = _candidate_payload(
        label="CANDIDATE_EXEC",
        selected_row=selected_exec,
        selected_grid_dir=b_grid_dir,
        min_trade_threshold_used=exec_threshold,
        wf_consistency=wf_consistency,
    )
    packet_candidates["CANDIDATE_EXEC"] = exec_payload
    table_rows.append(exec_table)

    if include_exec2:
        selected_exec2, exec2_threshold = _select_best_row(
            metrics=a2_metrics,
            profile="exec",
            min_trades=min_trades,
            warnings=warnings,
        )
        exec2_payload, exec2_table = _candidate_payload(
            label="CANDIDATE_EXEC_2",
            selected_row=selected_exec2,
            selected_grid_dir=a2_grid_dir,
            min_trade_threshold_used=exec2_threshold,
            wf_consistency={},
        )
        packet_candidates["CANDIDATE_EXEC_2"] = exec2_payload
        table_rows.append(exec2_table)

    run_id = f"pcp-{uuid4().hex[:12]}"
    output_dir = (
        out_dir
        if out_dir is not None
        else settings.paths.artifacts_root / "production_candidates" / f"{run_id}_production_candidate_pack_v1"
    )

    packet = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "pcp_version": "production_candidate_pack_v1",
        "source_artifacts": {
            "a1_grid_dir": str(a1_grid_dir),
            "a2_grid_dir": str(a2_grid_dir),
            "a3_grid_dir": str(a3_grid_dir),
            "b_grid_dir": str(b_grid_dir),
            "c_grid_dir": str(c_grid_dir),
            "wf_baseline_dir": str(wf_baseline_dir) if wf_baseline_dir is not None else None,
            "wf_hybrid_dir": str(wf_hybrid_dir) if wf_hybrid_dir is not None else None,
            "execution_realism_report_dir": (
                str(execution_realism_report_dir)
                if execution_realism_report_dir is not None
                else None
            ),
        },
        "selection_policy": {
            "alpha_ranking": [
                "robustness_score_v2 desc",
                "expectancy desc",
                "profit_factor desc",
                "trade_count desc",
                "combo_id asc",
            ],
            "exec_ranking": [
                "profit_factor desc",
                "robustness_score_v2 desc",
                "ret_cv asc",
                "trade_count desc",
                "combo_id asc",
            ],
            "min_trades_requested": min_trades,
        },
        "candidates": packet_candidates,
    }

    table = pl.DataFrame(table_rows).sort("label")
    summary = {
        "run_id": run_id,
        "candidate_count": len(packet_candidates),
        "warnings": warnings,
        "wf_consistency": wf_consistency,
        "selected_labels": sorted(packet_candidates.keys()),
    }

    policy_packet_path, candidates_table_path, summary_path, report_path = (
        write_production_candidate_reports(
            output_dir=output_dir,
            packet=_finite_json(packet),
            table=table,
            summary=_finite_json(summary),
        )
    )

    effective_logger.info(
        "backtest.production_candidates.complete run_id=%s output=%s",
        run_id,
        output_dir,
    )
    return ProductionCandidatePackResult(
        run_id=run_id,
        output_dir=output_dir,
        policy_packet_path=policy_packet_path,
        candidates_table_path=candidates_table_path,
        summary_path=summary_path,
        report_path=report_path,
    )


def summarize_production_candidates_pack(pack_dir: Path) -> dict[str, Any]:
    """Validate and summarize a PCP v1 pack directory."""

    policy_path = pack_dir / "production_policy_packet_v1.json"
    table_path = pack_dir / "production_candidates_table.csv"
    summary_path = pack_dir / "production_candidates_summary.json"
    report_path = pack_dir / "production_candidate_pack_report.md"

    missing = [str(path) for path in [policy_path, table_path, summary_path, report_path] if not path.exists()]
    errors: list[str] = []
    warnings: list[str] = []
    if missing:
        errors.extend([f"missing_required_artifact: {path}" for path in missing])
        return {"pack_dir": str(pack_dir), "errors": errors, "warnings": warnings}

    packet = json.loads(policy_path.read_text(encoding="utf-8"))
    table = pl.read_csv(table_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    candidates = packet.get("candidates", {})
    if not isinstance(candidates, dict) or not candidates:
        errors.append("no_candidates_in_policy_packet")

    for label, candidate in candidates.items():
        trade_count = _safe_int(((candidate.get("expected_behavior_snapshot") or {}).get("trade_count")))
        if trade_count is None or trade_count <= 0:
            errors.append(f"{label}: trade_count must be > 0")
        combo_id = candidate.get("combo_id")
        if not combo_id:
            errors.append(f"{label}: combo_id missing")

        selected_grid_dir = candidate.get("selected_from_grid_dir")
        if selected_grid_dir and not Path(str(selected_grid_dir)).exists():
            errors.append(f"{label}: selected_from_grid_dir missing -> {selected_grid_dir}")

        overlay = candidate.get("overlay", {})
        if not isinstance(overlay, dict):
            errors.append(f"{label}: overlay block missing")
        else:
            if overlay.get("enabled") and not overlay.get("mode"):
                errors.append(f"{label}: overlay enabled but mode missing")

        execution = candidate.get("execution_realism", {})
        if not isinstance(execution, dict):
            errors.append(f"{label}: execution_realism block missing")
        else:
            if not execution.get("profile"):
                errors.append(f"{label}: execution profile missing")

    if table.height != len(candidates):
        warnings.append(
            f"table_row_count_mismatch: table={table.height} candidates={len(candidates)}"
        )

    if summary.get("warnings"):
        warnings.extend([str(item) for item in summary.get("warnings", [])])

    return {
        "pack_dir": str(pack_dir),
        "policy_path": str(policy_path),
        "table_path": str(table_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "candidate_count": len(candidates),
        "candidates": sorted(candidates.keys()),
        "errors": errors,
        "warnings": warnings,
    }

