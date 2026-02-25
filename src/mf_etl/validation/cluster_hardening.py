"""Cluster hardening policy/scoring orchestration and summaries."""

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

from mf_etl.config import AppSettings
from mf_etl.validation.cluster_hardening_export import (
    ClusterHardeningExportResult,
    export_cluster_rows_with_policy,
)
from mf_etl.validation.cluster_hardening_policy import (
    ClusterHardeningThresholds,
    derive_ci_width_hard_value,
    recommend_thresholds_from_state_stats,
    resolve_cluster_hardening_thresholds,
)
from mf_etl.validation.cluster_hardening_reports import (
    ClusterHardeningCompareReportPaths,
    ClusterHardeningSingleReportPaths,
    ClusterHardeningWFReportPaths,
    write_cluster_hardening_compare_reports,
    write_cluster_hardening_single_reports,
    write_cluster_hardening_wf_reports,
)

LOGGER = logging.getLogger(__name__)

SEVERE_QA_LABEL_SET = {"SIGN_FLIP_ACROSS_WINDOWS", "WIDE_CI", "LIKELY_OUTLIER_WINDOW"}


@dataclass(frozen=True, slots=True)
class ClusterHardeningSingleResult:
    """Artifact paths for one single-run hardening execution."""

    output_dir: Path
    policy_path: Path
    state_table_path: Path
    summary_path: Path
    report_path: Path
    export_summary_path: Path | None


@dataclass(frozen=True, slots=True)
class ClusterHardeningWalkForwardResult:
    """Artifact paths for walk-forward hardening aggregation."""

    output_dir: Path
    wf_summary_path: Path
    wf_state_stats_path: Path
    split_counts_path: Path
    issue_frequency_path: Path
    threshold_recommendation_path: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class ClusterHardeningCompareResult:
    """Artifact paths for comparing two hardening policies."""

    output_dir: Path
    summary_path: Path
    table_path: Path


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


def _require_files(base_dir: Path, relative_paths: list[str]) -> None:
    missing = [str(base_dir / rel) for rel in relative_paths if not (base_dir / rel).exists()]
    if missing:
        rendered = "\n".join(missing)
        raise FileNotFoundError(f"Missing required hardening artifacts under {base_dir}:\n{rendered}")


def _finite_or_none(value: Any) -> float | None:
    out = _safe_float(value)
    return out if out is None or np.isfinite(out) else None


def _parse_qa_issue_map(validation_run_dir: Path) -> dict[int, set[str]]:
    issue_map: dict[int, set[str]] = {}
    flagged_path = validation_run_dir / "cluster_qa" / "cluster_qa_flagged_states.csv"
    if not flagged_path.exists():
        return issue_map
    flagged = pl.read_csv(flagged_path)
    if flagged.height == 0 or "state_id" not in flagged.columns:
        return issue_map
    for row in flagged.to_dicts():
        state_id = _safe_int(row.get("state_id"))
        if state_id is None:
            continue
        labels = str(row.get("issues") or "")
        parsed = {part.strip() for part in labels.split(",") if part.strip() != ""}
        if parsed:
            issue_map[state_id] = parsed
    return issue_map


def _parse_wf_qa_issue_map(wf_run_dir: Path) -> dict[str, dict[int, set[str]]]:
    """Parse walk-forward aggregated QA flags as train_end -> state_id -> labels."""

    out: dict[str, dict[int, set[str]]] = {}
    flagged_path = wf_run_dir / "cluster_qa" / "cluster_qa_wf_flagged_states.csv"
    if not flagged_path.exists():
        return out
    flagged = pl.read_csv(flagged_path)
    required = {"train_end", "state_id", "issues"}
    if not required.issubset(set(flagged.columns)) or flagged.height == 0:
        return out
    for row in flagged.to_dicts():
        train_end = str(row.get("train_end") or "")
        state_id = _safe_int(row.get("state_id"))
        if train_end == "" or state_id is None:
            continue
        labels = {part.strip() for part in str(row.get("issues") or "").split(",") if part.strip() != ""}
        if not labels:
            continue
        by_split = out.setdefault(train_end, {})
        by_split[state_id] = set(by_split.get(state_id, set())) | labels
    return out


def _transition_counts_by_state(transition_summary: pl.DataFrame) -> dict[int, int]:
    counts: dict[int, int] = {}
    if transition_summary.height == 0:
        return counts
    required = {"count_events", "prev_state_id", "next_state_id"}
    if not required.issubset(set(transition_summary.columns)):
        return counts
    for row in transition_summary.to_dicts():
        event_count = _safe_int(row.get("count_events")) or 0
        prev_state = _safe_int(row.get("prev_state_id"))
        next_state = _safe_int(row.get("next_state_id"))
        if prev_state is not None:
            counts[prev_state] = counts.get(prev_state, 0) + event_count
        if next_state is not None:
            counts[next_state] = counts.get(next_state, 0) + event_count
    return counts


def _build_window_stats(rolling_metrics: pl.DataFrame) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {}
    if rolling_metrics.height == 0 or "state_id" not in rolling_metrics.columns:
        return stats
    for state_id in rolling_metrics["state_id"].unique().to_list():
        subset = rolling_metrics.filter(pl.col("state_id") == state_id).sort("window_seq")
        values = subset["fwd_ret_10_mean"].cast(pl.Float64, strict=False).to_numpy()
        finite_values = values[np.isfinite(values)]

        sign_flip_count = 0
        prev_sign = 0
        for value in finite_values:
            sign = 1 if value > 0 else (-1 if value < 0 else 0)
            if prev_sign != 0 and sign != 0 and sign != prev_sign:
                sign_flip_count += 1
            if sign != 0:
                prev_sign = sign

        best_window = float(np.max(finite_values)) if finite_values.size > 0 else None
        worst_window = float(np.min(finite_values)) if finite_values.size > 0 else None
        drift_span = float(best_window - worst_window) if best_window is not None and worst_window is not None else None

        stats[int(state_id)] = {
            "window_count": int(subset.height),
            "sign_flip_count": int(sign_flip_count),
            "best_window_fwd_ret_10": best_window,
            "worst_window_fwd_ret_10": worst_window,
            "drift_span": drift_span,
        }
    return stats


def _directional_edge_sign(ci_lo: float | None, ci_hi: float | None) -> int:
    if ci_lo is None or ci_hi is None:
        return 0
    if ci_lo > 0 and ci_hi > 0:
        return 1
    if ci_lo < 0 and ci_hi < 0:
        return -1
    return 0


def _direction_hint(edge_sign: int) -> str:
    if edge_sign > 0:
        return "LONG_BIAS"
    if edge_sign < 0:
        return "SHORT_BIAS"
    return "UNCONFIRMED"


def _score_state(
    *,
    row: dict[str, Any],
    qa_labels: set[str],
    thresholds: ClusterHardeningThresholds,
    ci_hard_value: float | None,
    max_n_rows: int,
    max_share: float,
) -> tuple[float, float, dict[str, float]]:
    n_rows = _safe_int(row.get("n_rows")) or 0
    state_share = _safe_float(row.get("state_share_mean")) or 0.0
    sign_confidence = bool(row.get("sign_confidence_flag"))
    ci_width = _safe_float(row.get("ci_width"))
    sign_consistency = _safe_float(row.get("stability_sign_consistency")) or 0.0
    ret_cv = _safe_float(row.get("ret_mean_cv"))
    confidence_score = _safe_float(row.get("confidence_score")) or 0.0

    sample_component = 0.0
    if max_n_rows > 0 and n_rows > 0:
        sample_component = float(min(1.0, np.log1p(n_rows) / np.log1p(max_n_rows)))

    occupancy_component = 0.0
    if max_share > 0 and state_share > 0:
        occupancy_component = float(min(1.0, state_share / max_share))

    sign_component = 1.0 if sign_confidence else 0.0

    ci_reference = ci_hard_value if ci_hard_value is not None and ci_hard_value > thresholds.eps else 0.2
    if ci_width is None:
        ci_component = 0.0
    else:
        ci_component = float(max(0.0, 1.0 - (ci_width / max(ci_reference, thresholds.eps))))

    sign_cons_component = float(np.clip(sign_consistency, 0.0, 1.0))

    if ret_cv is None:
        ret_cv_component = 0.0
    else:
        ret_cv_component = float(max(0.0, 1.0 - (ret_cv / max(thresholds.ret_cv_hard, thresholds.eps))))

    confidence_component = float(np.clip(confidence_score / 100.0, 0.0, 1.0))

    components = {
        "sample_size": sample_component,
        "occupancy": occupancy_component,
        "sign_confidence": sign_component,
        "ci_width": ci_component,
        "sign_consistency": sign_cons_component,
        "ret_cv": ret_cv_component,
        "confidence_score": confidence_component,
    }

    weight_sum = float(sum(max(0.0, thresholds.weights.get(name, 0.0)) for name in components))
    if weight_sum <= 0:
        base_score = 0.0
    else:
        weighted = sum(components[name] * max(0.0, thresholds.weights.get(name, 0.0)) for name in components)
        base_score = float(100.0 * (weighted / weight_sum))

    risk_penalty = float(sum(thresholds.penalties.get(label, 0.0) for label in qa_labels))
    final_score = float(np.clip(base_score - risk_penalty, 0.0, 100.0))
    return final_score, risk_penalty, components


def _build_state_policy_table(
    *,
    state_scorecard: pl.DataFrame,
    rolling_metrics: pl.DataFrame,
    transition_summary: pl.DataFrame,
    qa_issue_map: dict[int, set[str]],
    thresholds: ClusterHardeningThresholds,
) -> tuple[pl.DataFrame, float | None]:
    window_stats = _build_window_stats(rolling_metrics)
    transition_counts = _transition_counts_by_state(transition_summary)
    ci_hard_value = derive_ci_width_hard_value(state_scorecard, thresholds.ci_width_hard_quantile)

    max_n_rows = int(state_scorecard.select(pl.col("n_rows").max()).item()) if state_scorecard.height > 0 else 0
    max_share_value = state_scorecard.select(pl.col("state_share_mean").cast(pl.Float64, strict=False).max()).item()
    max_share = _safe_float(max_share_value) or 0.0

    rows: list[dict[str, Any]] = []
    for row in state_scorecard.to_dicts():
        state_id = _safe_int(row.get("state_id"))
        if state_id is None:
            continue
        qa_labels = set(qa_issue_map.get(state_id, set()))
        win = window_stats.get(state_id, {})

        n_rows = _safe_int(row.get("n_rows")) or 0
        state_share = _safe_float(row.get("state_share_mean"))
        ret_cv = _safe_float(row.get("ret_mean_cv"))
        ci_width = _safe_float(row.get("ci_width"))
        sign_consistency = _safe_float(row.get("stability_sign_consistency"))
        sign_conf = bool(row.get("sign_confidence_flag"))
        ci_lo = _safe_float(row.get("fwd_ret_10_ci_lo"))
        ci_hi = _safe_float(row.get("fwd_ret_10_ci_hi"))
        mean_val = _safe_float(row.get("fwd_ret_10_mean"))

        edge_sign = _directional_edge_sign(ci_lo, ci_hi)
        direction_hint = _direction_hint(edge_sign)
        transition_event_count = transition_counts.get(state_id, 0)

        score, risk_penalty, score_components = _score_state(
            row=row,
            qa_labels=qa_labels,
            thresholds=thresholds,
            ci_hard_value=ci_hard_value,
            max_n_rows=max_n_rows,
            max_share=max_share,
        )

        reasons: list[str] = []
        hard_fail = False
        if n_rows < thresholds.min_n_rows_hard:
            hard_fail = True
            reasons.append("HARD_LOW_N")
        if state_share is not None and state_share < thresholds.min_state_share_hard:
            hard_fail = True
            reasons.append("HARD_LOW_OCCUPANCY")
        if ret_cv is not None and ret_cv > thresholds.ret_cv_hard:
            hard_fail = True
            reasons.append("HARD_HIGH_RET_CV")
        if sign_consistency is not None and sign_consistency < thresholds.sign_consistency_hard:
            hard_fail = True
            reasons.append("HARD_LOW_SIGN_CONSISTENCY")
        if (
            ci_hard_value is not None
            and ci_width is not None
            and ci_width > ci_hard_value
            and not sign_conf
        ):
            hard_fail = True
            reasons.append("HARD_WIDE_CI_WITHOUT_SIGN_CONF")
        if SEVERE_QA_LABEL_SET.issubset(qa_labels):
            hard_fail = True
            reasons.append("HARD_SEVERE_QA_COMBINATION")

        if hard_fail or score < thresholds.score_min_watch:
            class_label = "BLOCK"
            if not hard_fail and score < thresholds.score_min_watch:
                reasons.append("BLOCK_LOW_SCORE")
        elif (
            score >= thresholds.score_min_allow
            and sign_conf
            and "WIDE_CI" not in qa_labels
            and "SIGN_FLIP_ACROSS_WINDOWS" not in qa_labels
        ):
            class_label = "ALLOW"
            reasons.append("ALLOW_SCORE_AND_STABILITY_OK")
        else:
            class_label = "WATCH"
            reasons.append("WATCH_MIXED_SIGNALS")

        for label in sorted(qa_labels):
            reasons.append(f"QA_{label}")

        rows.append(
            {
                "state_id": state_id,
                "n_rows": n_rows,
                "fwd_ret_10_mean": mean_val,
                "fwd_ret_10_ci_lo": ci_lo,
                "fwd_ret_10_ci_hi": ci_hi,
                "ci_width": ci_width,
                "hit_rate": _finite_or_none(row.get("hit_rate")),
                "sign_confidence_flag": sign_conf,
                "stability_sign_consistency": sign_consistency,
                "ret_mean_cv": ret_cv,
                "state_share_mean": state_share,
                "state_share_cv": _finite_or_none(row.get("state_share_cv")),
                "confidence_score": _finite_or_none(row.get("confidence_score")),
                "pairwise_diff_significant_share": _finite_or_none(row.get("pairwise_diff_significant_share")),
                "window_count": _safe_int(win.get("window_count")),
                "sign_flip_count": _safe_int(win.get("sign_flip_count")),
                "worst_window_fwd_ret_10": _finite_or_none(win.get("worst_window_fwd_ret_10")),
                "best_window_fwd_ret_10": _finite_or_none(win.get("best_window_fwd_ret_10")),
                "drift_span": _finite_or_none(win.get("drift_span")),
                "transition_event_count": transition_event_count,
                "directional_edge_sign": edge_sign,
                "allow_direction_hint": direction_hint,
                "risk_penalty": risk_penalty,
                "tradability_score_raw": score,
                "tradability_score": score,
                "class_label": class_label,
                "qa_labels": ",".join(sorted(qa_labels)),
                "reasons": ",".join(reasons),
                "score_component_sample_size": score_components["sample_size"],
                "score_component_occupancy": score_components["occupancy"],
                "score_component_sign_confidence": score_components["sign_confidence"],
                "score_component_ci_width": score_components["ci_width"],
                "score_component_sign_consistency": score_components["sign_consistency"],
                "score_component_ret_cv": score_components["ret_cv"],
                "score_component_confidence_score": score_components["confidence_score"],
            }
        )

    if not rows:
        return pl.DataFrame(schema={"state_id": pl.Int32}), ci_hard_value

    state_policy_table = pl.DataFrame(rows)
    state_policy_table = state_policy_table.with_columns(
        [
            pl.col("ci_width")
            .cast(pl.Float64, strict=False)
            .rank(method="dense")
            .over(pl.lit(1))
            .cast(pl.Int32)
            .alias("ci_relative_rank"),
            pl.col("state_share_mean")
            .cast(pl.Float64, strict=False)
            .rank(method="dense", descending=True)
            .over(pl.lit(1))
            .cast(pl.Int32)
            .alias("occupancy_rank"),
            pl.col("stability_sign_consistency")
            .cast(pl.Float64, strict=False)
            .rank(method="dense", descending=True)
            .over(pl.lit(1))
            .cast(pl.Int32)
            .alias("stability_rank"),
        ]
    ).sort(["class_label", "tradability_score"], descending=[False, True])
    return state_policy_table, ci_hard_value


def _policy_payload(
    *,
    validation_run_dir: Path,
    state_policy_table: pl.DataFrame,
    thresholds: ClusterHardeningThresholds,
    ci_hard_value: float | None,
) -> dict[str, Any]:
    allow_states = (
        state_policy_table.filter(pl.col("class_label") == "ALLOW")["state_id"].cast(pl.Int64).to_list()
        if state_policy_table.height > 0
        else []
    )
    watch_states = (
        state_policy_table.filter(pl.col("class_label") == "WATCH")["state_id"].cast(pl.Int64).to_list()
        if state_policy_table.height > 0
        else []
    )
    block_states = (
        state_policy_table.filter(pl.col("class_label") == "BLOCK")["state_id"].cast(pl.Int64).to_list()
        if state_policy_table.height > 0
        else []
    )
    per_state_rows: list[dict[str, Any]] = []
    for row in state_policy_table.to_dicts():
        per_state_rows.append(
            {
                "state_id": row.get("state_id"),
                "class_label": row.get("class_label"),
                "tradability_score": row.get("tradability_score"),
                "allow_direction_hint": row.get("allow_direction_hint"),
                "reasons": [part for part in str(row.get("reasons") or "").split(",") if part != ""],
                "qa_labels": [part for part in str(row.get("qa_labels") or "").split(",") if part != ""],
                "metrics": {
                    "n_rows": row.get("n_rows"),
                    "state_share_mean": row.get("state_share_mean"),
                    "fwd_ret_10_mean": row.get("fwd_ret_10_mean"),
                    "ci_width": row.get("ci_width"),
                    "stability_sign_consistency": row.get("stability_sign_consistency"),
                    "ret_mean_cv": row.get("ret_mean_cv"),
                    "confidence_score": row.get("confidence_score"),
                },
            }
        )

    return {
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "validation_run_dir": str(validation_run_dir),
        "thresholds": {
            "min_n_rows_hard": thresholds.min_n_rows_hard,
            "min_state_share_hard": thresholds.min_state_share_hard,
            "ret_cv_hard": thresholds.ret_cv_hard,
            "sign_consistency_hard": thresholds.sign_consistency_hard,
            "ci_width_hard_quantile": thresholds.ci_width_hard_quantile,
            "ci_width_hard_value": ci_hard_value,
            "score_min_allow": thresholds.score_min_allow,
            "score_min_watch": thresholds.score_min_watch,
            "weights": thresholds.weights,
            "penalties": thresholds.penalties,
            "eps": thresholds.eps,
        },
        "summary": {
            "total_states": int(state_policy_table.height),
            "allow_count": int(len(allow_states)),
            "watch_count": int(len(watch_states)),
            "block_count": int(len(block_states)),
            "allow_state_ids": allow_states,
            "watch_state_ids": watch_states,
            "block_state_ids": block_states,
        },
        "per_state": per_state_rows,
    }


def _resolve_clustered_rows_file(
    *,
    validation_run_dir: Path,
    explicit: Path | None,
) -> Path | None:
    if explicit is not None:
        return explicit
    run_summary_path = validation_run_dir / "run_summary.json"
    if not run_summary_path.exists():
        return None
    payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
    source = payload.get("input_file")
    if source is None:
        return None
    candidate = Path(str(source))
    return candidate if candidate.exists() else None


def run_cluster_hardening_single(
    settings: AppSettings,
    *,
    validation_run_dir: Path,
    clustered_rows_file: Path | None = None,
    export_filtered: bool = True,
    min_n_rows_hard: int | None = None,
    min_state_share_hard: float | None = None,
    ret_cv_hard: float | None = None,
    sign_consistency_hard: float | None = None,
    ci_width_hard_quantile: float | None = None,
    score_min_allow: float | None = None,
    score_min_watch: float | None = None,
    qa_issue_override: dict[int, set[str]] | None = None,
    output_dir: Path | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> ClusterHardeningSingleResult:
    """Build hardening policy for one cluster validation run and optionally export rows."""

    effective_logger = logger or LOGGER
    required = [
        "validation_scorecard.json",
        "state_scorecard.csv",
        "state_stability_summary.csv",
        "rolling_state_metrics.csv",
        "bootstrap_state_summary.csv",
    ]
    _require_files(validation_run_dir, required)

    target_dir = output_dir or (validation_run_dir / "cluster_hardening")
    target_dir.mkdir(parents=True, exist_ok=True)

    existing_policy = target_dir / "cluster_hardening_policy.json"
    existing_table = target_dir / "cluster_hardening_state_table.csv"
    existing_summary = target_dir / "cluster_hardening_summary.json"
    existing_report = target_dir / "cluster_hardening_report.md"
    existing_export_summary = target_dir / "exports" / "cluster_hardening_export_summary.json"
    clustered_source = _resolve_clustered_rows_file(
        validation_run_dir=validation_run_dir,
        explicit=clustered_rows_file,
    )
    export_requested = bool(export_filtered and clustered_source is not None)
    if (
        not force
        and existing_policy.exists()
        and existing_table.exists()
        and existing_summary.exists()
        and existing_report.exists()
        and (not export_requested or existing_export_summary.exists())
    ):
        effective_logger.info("cluster_hardening.single.skip output=%s reason=existing", target_dir)
        return ClusterHardeningSingleResult(
            output_dir=target_dir,
            policy_path=existing_policy,
            state_table_path=existing_table,
            summary_path=existing_summary,
            report_path=existing_report,
            export_summary_path=existing_export_summary if existing_export_summary.exists() else None,
        )

    state_scorecard = pl.read_csv(validation_run_dir / "state_scorecard.csv")
    rolling_metrics = pl.read_csv(validation_run_dir / "rolling_state_metrics.csv")
    transition_summary_path = validation_run_dir / "transition_event_summary.csv"
    transition_summary = pl.read_csv(transition_summary_path) if transition_summary_path.exists() else pl.DataFrame()
    qa_issue_map = _parse_qa_issue_map(validation_run_dir)
    if qa_issue_override:
        for state_id, labels in qa_issue_override.items():
            qa_issue_map[state_id] = set(qa_issue_map.get(state_id, set())) | set(labels)

    thresholds = resolve_cluster_hardening_thresholds(
        settings,
        min_n_rows_hard=min_n_rows_hard,
        min_state_share_hard=min_state_share_hard,
        ret_cv_hard=ret_cv_hard,
        sign_consistency_hard=sign_consistency_hard,
        ci_width_hard_quantile=ci_width_hard_quantile,
        score_min_allow=score_min_allow,
        score_min_watch=score_min_watch,
    )

    state_policy_table, ci_hard_value = _build_state_policy_table(
        state_scorecard=state_scorecard,
        rolling_metrics=rolling_metrics,
        transition_summary=transition_summary,
        qa_issue_map=qa_issue_map,
        thresholds=thresholds,
    )
    policy_payload = _policy_payload(
        validation_run_dir=validation_run_dir,
        state_policy_table=state_policy_table,
        thresholds=thresholds,
        ci_hard_value=ci_hard_value,
    )

    export_summary_payload: dict[str, Any] | None = None
    export_result: ClusterHardeningExportResult | None = None
    if export_requested and clustered_source is not None:
        export_result = export_cluster_rows_with_policy(
            clustered_rows_file=clustered_source,
            state_policy_table=state_policy_table,
            output_dir=target_dir / "exports",
        )
        export_summary_payload = json.loads(export_result.summary_path.read_text(encoding="utf-8"))

    paths: ClusterHardeningSingleReportPaths = write_cluster_hardening_single_reports(
        output_dir=target_dir,
        validation_run_dir=validation_run_dir,
        policy_payload=policy_payload,
        state_table=state_policy_table,
        export_summary=export_summary_payload,
    )

    effective_logger.info(
        "cluster_hardening.single.complete validation_run_dir=%s allow=%s watch=%s block=%s output=%s",
        validation_run_dir,
        policy_payload["summary"]["allow_count"],
        policy_payload["summary"]["watch_count"],
        policy_payload["summary"]["block_count"],
        target_dir,
    )

    return ClusterHardeningSingleResult(
        output_dir=target_dir,
        policy_path=paths.policy_path,
        state_table_path=paths.state_table_path,
        summary_path=paths.summary_path,
        report_path=paths.report_path,
        export_summary_path=(export_result.summary_path if export_result is not None else None),
    )


def run_cluster_hardening_walkforward(
    settings: AppSettings,
    *,
    wf_run_dir: Path,
    min_n_rows_hard: int | None = None,
    min_state_share_hard: float | None = None,
    ret_cv_hard: float | None = None,
    sign_consistency_hard: float | None = None,
    ci_width_hard_quantile: float | None = None,
    score_min_allow: float | None = None,
    score_min_watch: float | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> ClusterHardeningWalkForwardResult:
    """Run cluster hardening across successful walk-forward cluster validation splits."""

    effective_logger = logger or LOGGER
    manifest_path = wf_run_dir / "wf_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Walk-forward manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    base_thresholds = resolve_cluster_hardening_thresholds(
        settings,
        min_n_rows_hard=min_n_rows_hard,
        min_state_share_hard=min_state_share_hard,
        ret_cv_hard=ret_cv_hard,
        sign_consistency_hard=sign_consistency_hard,
        ci_width_hard_quantile=ci_width_hard_quantile,
        score_min_allow=score_min_allow,
        score_min_watch=score_min_watch,
    )

    output_dir = wf_run_dir / "cluster_hardening"
    split_root = output_dir / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    wf_qa_issue_map = _parse_wf_qa_issue_map(wf_run_dir)

    split_rows: list[dict[str, Any]] = []
    state_frames: list[pl.DataFrame] = []

    for split in manifest.get("splits", []):
        train_end = str(split.get("train_end"))
        status = str(split.get("status"))
        val_cluster_dir_raw = split.get("val_cluster_dir")
        if status != "SUCCESS" or val_cluster_dir_raw is None:
            split_rows.append(
                {
                    "train_end": train_end,
                    "status": status,
                    "allow_count": None,
                    "watch_count": None,
                    "block_count": None,
                    "policy_dir": None,
                    "error": split.get("error"),
                }
            )
            continue

        val_cluster_dir = Path(str(val_cluster_dir_raw))
        split_dir = split_root / train_end
        try:
            single = run_cluster_hardening_single(
                settings,
                validation_run_dir=val_cluster_dir,
                export_filtered=False,
                min_n_rows_hard=min_n_rows_hard,
                min_state_share_hard=min_state_share_hard,
                ret_cv_hard=ret_cv_hard,
                sign_consistency_hard=sign_consistency_hard,
                ci_width_hard_quantile=ci_width_hard_quantile,
                score_min_allow=score_min_allow,
                score_min_watch=score_min_watch,
                qa_issue_override=wf_qa_issue_map.get(train_end),
                output_dir=split_dir,
                force=force,
                logger=effective_logger,
            )
            policy = json.loads(single.policy_path.read_text(encoding="utf-8"))
            state_table = pl.read_csv(single.state_table_path).with_columns(
                [pl.lit(train_end).alias("train_end"), pl.lit(str(single.output_dir)).alias("policy_dir")]
            )
            state_frames.append(state_table)
            summary = policy.get("summary", {})
            split_rows.append(
                {
                    "train_end": train_end,
                    "status": "SUCCESS",
                    "allow_count": summary.get("allow_count"),
                    "watch_count": summary.get("watch_count"),
                    "block_count": summary.get("block_count"),
                    "policy_dir": str(single.output_dir),
                    "error": None,
                }
            )
        except Exception as exc:
            split_rows.append(
                {
                    "train_end": train_end,
                    "status": "FAILED",
                    "allow_count": None,
                    "watch_count": None,
                    "block_count": None,
                    "policy_dir": str(split_dir),
                    "error": str(exc),
                }
            )
            effective_logger.exception("cluster_hardening.wf.split_failed train_end=%s", train_end)

    split_counts = pl.DataFrame(split_rows) if split_rows else pl.DataFrame(schema={"train_end": pl.String})
    wf_state_stats = (
        pl.concat(state_frames, how="diagonal_relaxed")
        if state_frames
        else pl.DataFrame(schema={"train_end": pl.String, "state_id": pl.Int64})
    )

    issue_frequency = pl.DataFrame(schema={"issue": pl.String, "split_count": pl.Int64, "state_count": pl.Int64})
    if wf_state_stats.height > 0 and "qa_labels" in wf_state_stats.columns:
        issue_frequency = (
            wf_state_stats.select(["train_end", "qa_labels"])
            .with_columns(pl.col("qa_labels").cast(pl.String).str.split(",").alias("issue_list"))
            .explode("issue_list")
            .with_columns(pl.col("issue_list").str.strip_chars().alias("issue"))
            .filter(pl.col("issue") != "")
            .group_by("issue")
            .agg(
                [
                    pl.col("train_end").n_unique().alias("split_count"),
                    pl.len().alias("state_count"),
                ]
            )
            .sort("split_count", descending=True)
        )

    success_split_counts = split_counts.filter(pl.col("status") == "SUCCESS")
    wf_summary = {
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "wf_run_dir": str(wf_run_dir),
        "wf_run_id": manifest.get("wf_run_id"),
        "thresholds_used": {
            "min_n_rows_hard": base_thresholds.min_n_rows_hard,
            "min_state_share_hard": base_thresholds.min_state_share_hard,
            "ret_cv_hard": base_thresholds.ret_cv_hard,
            "sign_consistency_hard": base_thresholds.sign_consistency_hard,
            "ci_width_hard_quantile": base_thresholds.ci_width_hard_quantile,
            "score_min_allow": base_thresholds.score_min_allow,
            "score_min_watch": base_thresholds.score_min_watch,
        },
        "splits_total": int(split_counts.height),
        "splits_successful": int(success_split_counts.height),
        "splits_failed": int(split_counts.filter(pl.col("status") == "FAILED").height),
        "allow_count_total": int(success_split_counts["allow_count"].fill_null(0).sum()) if success_split_counts.height > 0 else 0,
        "watch_count_total": int(success_split_counts["watch_count"].fill_null(0).sum()) if success_split_counts.height > 0 else 0,
        "block_count_total": int(success_split_counts["block_count"].fill_null(0).sum()) if success_split_counts.height > 0 else 0,
        "failed_splits": split_counts.filter(pl.col("status") == "FAILED").to_dicts(),
    }
    threshold_recommendation = recommend_thresholds_from_state_stats(
        wf_state_stats,
        fallback=base_thresholds,
    )

    paths: ClusterHardeningWFReportPaths = write_cluster_hardening_wf_reports(
        output_dir=output_dir,
        wf_run_dir=wf_run_dir,
        wf_summary=wf_summary,
        wf_state_stats=wf_state_stats,
        split_counts=split_counts,
        issue_frequency=issue_frequency,
        threshold_recommendation=threshold_recommendation,
    )
    effective_logger.info(
        "cluster_hardening.wf.complete wf_run_dir=%s splits_success=%s output=%s",
        wf_run_dir,
        wf_summary["splits_successful"],
        output_dir,
    )
    return ClusterHardeningWalkForwardResult(
        output_dir=output_dir,
        wf_summary_path=paths.wf_summary_path,
        wf_state_stats_path=paths.wf_state_stats_path,
        split_counts_path=paths.split_counts_path,
        issue_frequency_path=paths.issue_frequency_path,
        threshold_recommendation_path=paths.threshold_recommendation_path,
        report_path=paths.report_path,
    )


def summarize_cluster_hardening(hardening_dir: Path) -> dict[str, Any]:
    """Read hardening outputs and return a concise sanity payload."""

    single_policy = hardening_dir / "cluster_hardening_policy.json"
    wf_summary = hardening_dir / "cluster_hardening_wf_summary.json"
    if single_policy.exists():
        payload = json.loads(single_policy.read_text(encoding="utf-8"))
        state_table = pl.read_csv(hardening_dir / "cluster_hardening_state_table.csv")
        export_summary_path = hardening_dir / "exports" / "cluster_hardening_export_summary.json"
        export_summary = (
            json.loads(export_summary_path.read_text(encoding="utf-8")) if export_summary_path.exists() else None
        )
        return {
            "mode": "single",
            "hardening_dir": str(hardening_dir),
            "thresholds": payload.get("thresholds", {}),
            "summary": payload.get("summary", {}),
            "state_rows": state_table.height,
            "state_table_preview": state_table.select(
                [
                    "state_id",
                    "class_label",
                    "tradability_score",
                    "allow_direction_hint",
                    "reasons",
                ]
            )
            .sort(["class_label", "tradability_score"], descending=[False, True])
            .head(20)
            .to_dicts(),
            "export_summary": export_summary,
        }
    if wf_summary.exists():
        summary = json.loads(wf_summary.read_text(encoding="utf-8"))
        split_counts = pl.read_csv(hardening_dir / "cluster_hardening_wf_split_counts.csv")
        issue_freq = pl.read_csv(hardening_dir / "cluster_hardening_wf_issue_frequency.csv")
        return {
            "mode": "walkforward",
            "hardening_dir": str(hardening_dir),
            "summary": summary,
            "split_counts_preview": split_counts.head(20).to_dicts(),
            "issue_frequency_preview": issue_freq.head(20).to_dicts(),
        }
    raise FileNotFoundError(
        f"No hardening artifacts found in {hardening_dir}. Expected cluster_hardening_policy.json or cluster_hardening_wf_summary.json."
    )


def _resolve_policy_inputs(hardening_dir: Path) -> tuple[dict[str, Any], pl.DataFrame, dict[str, Any] | None]:
    policy_path = hardening_dir / "cluster_hardening_policy.json"
    state_table_path = hardening_dir / "cluster_hardening_state_table.csv"
    if not policy_path.exists() or not state_table_path.exists():
        raise FileNotFoundError(
            f"Expected policy artifacts in {hardening_dir}: cluster_hardening_policy.json and cluster_hardening_state_table.csv."
        )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    state_table = pl.read_csv(state_table_path)
    export_summary_path = hardening_dir / "exports" / "cluster_hardening_export_summary.json"
    export_summary = (
        json.loads(export_summary_path.read_text(encoding="utf-8")) if export_summary_path.exists() else None
    )
    return policy, state_table, export_summary


def run_cluster_hardening_compare(
    settings: AppSettings,
    *,
    hardening_dir_a: Path,
    hardening_dir_b: Path,
    logger: logging.Logger | None = None,
) -> ClusterHardeningCompareResult:
    """Compare two single-run hardening policies and write diff artifacts."""

    effective_logger = logger or LOGGER
    policy_a, table_a, export_a = _resolve_policy_inputs(hardening_dir_a)
    policy_b, table_b, export_b = _resolve_policy_inputs(hardening_dir_b)

    joined = (
        table_a.select(
            [
                pl.col("state_id").cast(pl.Int64),
                pl.col("class_label").alias("class_a"),
                pl.col("tradability_score").alias("score_a"),
                pl.col("allow_direction_hint").alias("direction_a"),
                pl.col("reasons").alias("reasons_a"),
            ]
        )
        .join(
            table_b.select(
                [
                    pl.col("state_id").cast(pl.Int64),
                    pl.col("class_label").alias("class_b"),
                    pl.col("tradability_score").alias("score_b"),
                    pl.col("allow_direction_hint").alias("direction_b"),
                    pl.col("reasons").alias("reasons_b"),
                ]
            ),
            on="state_id",
            how="outer_coalesce",
        )
        .with_columns(
            [
                (pl.col("score_b").cast(pl.Float64, strict=False) - pl.col("score_a").cast(pl.Float64, strict=False)).alias("delta_score_b_minus_a"),
                (pl.col("class_a") != pl.col("class_b")).fill_null(True).alias("class_changed"),
            ]
        )
        .sort(["class_changed", "state_id"], descending=[True, False])
    )

    compare_id = f"cluster-hardening-compare-{uuid4().hex[:12]}"
    output_dir = settings.paths.artifacts_root / "validation_runs" / compare_id
    summary_payload = {
        "compare_id": compare_id,
        "hardening_dir_a": str(hardening_dir_a),
        "hardening_dir_b": str(hardening_dir_b),
        "run_a_allow_count": policy_a.get("summary", {}).get("allow_count"),
        "run_b_allow_count": policy_b.get("summary", {}).get("allow_count"),
        "run_a_watch_count": policy_a.get("summary", {}).get("watch_count"),
        "run_b_watch_count": policy_b.get("summary", {}).get("watch_count"),
        "run_a_block_count": policy_a.get("summary", {}).get("block_count"),
        "run_b_block_count": policy_b.get("summary", {}).get("block_count"),
        "class_changed_states": int(joined.filter(pl.col("class_changed") == True).height),  # noqa: E712
        "export_rows_a": (export_a or {}).get("tradable_rows"),
        "export_rows_b": (export_b or {}).get("tradable_rows"),
        "generated_ts": datetime.now(timezone.utc).isoformat(),
    }

    paths: ClusterHardeningCompareReportPaths = write_cluster_hardening_compare_reports(
        output_dir=output_dir,
        summary_payload=summary_payload,
        compare_table=joined,
    )
    effective_logger.info(
        "cluster_hardening.compare.complete compare_id=%s output=%s",
        compare_id,
        output_dir,
    )
    return ClusterHardeningCompareResult(
        output_dir=output_dir,
        summary_path=paths.summary_path,
        table_path=paths.table_path,
    )
