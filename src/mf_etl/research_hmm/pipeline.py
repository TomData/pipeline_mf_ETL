"""HMM baseline pipelines for sequential state modeling research."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from mf_etl.config import AppSettings
from mf_etl.research.preprocess import (
    PreprocessResult,
    fit_preprocess_model,
    transform_for_clustering,
)
from mf_etl.research_hmm.dataset_loader import load_hmm_dataset
from mf_etl.research_hmm.decode import build_decoded_rows, decode_with_model
from mf_etl.research_hmm.diagnostics import HMMDiagnostics, build_hmm_diagnostics
from mf_etl.research_hmm.hmm_model import HMMFitResult, fit_gaussian_hmm
from mf_etl.research_hmm.profiles import (
    HMMProfiles,
    build_hmm_state_profiles,
    build_hmm_vs_cluster_crosstab,
)
from mf_etl.research_hmm.sequence_builder import SequenceBuildResult, build_hmm_sequences

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HMMRunResult:
    """Artifact outputs for one HMM baseline run."""

    run_id: str
    output_dir: Path
    run_summary_path: Path
    decoded_rows_path: Path
    state_profile_path: Path


@dataclass(frozen=True, slots=True)
class HMMSweepResult:
    """Artifact outputs for HMM component-count sweep."""

    run_id: str
    output_dir: Path
    summary_json_path: Path
    summary_csv_path: Path
    rows: int


@dataclass(frozen=True, slots=True)
class HMMStabilityResult:
    """Artifact outputs for HMM seed-stability runs."""

    run_id: str
    output_dir: Path
    summary_json_path: Path
    by_seed_path: Path
    pairwise_ari_path: Path


@dataclass(frozen=True, slots=True)
class PreparedHMMData:
    """Split/preprocess outputs reused across HMM run/sweep/stability."""

    loaded_rows: int
    preprocess_fit: PreprocessResult
    preprocess_predict: PreprocessResult
    preprocess_train: PreprocessResult | None
    preprocess_test: PreprocessResult | None
    split_summary: dict[str, Any]
    split_mode: str
    fit_on: str
    predict_on: str
    scaler: str
    scaling_scope: str
    feature_list: list[str]


def _atomic_temp_path(target_path: Path) -> Path:
    return target_path.parent / f".{target_path.name}.{uuid4().hex}.tmp"


def _write_json_atomically(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_temp_path(output_path)
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, output_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return output_path


def _write_parquet_atomically(df: pl.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_temp_path(output_path)
    try:
        df.write_parquet(tmp)
        os.replace(tmp, output_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return output_path


def _write_csv_atomically(df: pl.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_temp_path(output_path)
    try:
        df.write_csv(tmp)
        os.replace(tmp, output_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return output_path


def _frame_stats(df: pl.DataFrame | None) -> dict[str, Any] | None:
    if df is None:
        return None
    if df.height == 0:
        return {"rows": 0, "tickers": 0, "min_trade_date": None, "max_trade_date": None}
    ticker_count = int(df.select(pl.col("ticker").n_unique()).item()) if "ticker" in df.columns else 0
    bounds = df.select(
        [
            pl.col("trade_date").min().alias("min_trade_date"),
            pl.col("trade_date").max().alias("max_trade_date"),
        ]
    ).to_dicts()[0]
    min_date = bounds["min_trade_date"]
    max_date = bounds["max_trade_date"]
    return {
        "rows": df.height,
        "tickers": ticker_count,
        "min_trade_date": min_date.isoformat() if min_date is not None else None,
        "max_trade_date": max_date.isoformat() if max_date is not None else None,
    }


def _resolve_split_and_scopes(
    *,
    split_mode: str,
    train_end: date | None,
    test_start: date | None,
    test_end: date | None,
    fit_on: str,
    predict_on: str | None,
) -> tuple[str, str, str, date | None, date | None]:
    split_mode_norm = split_mode.strip().lower()
    if split_mode_norm not in {"none", "time"}:
        raise ValueError("split_mode must be one of: none, time")
    fit_on_norm = fit_on.strip().lower()
    if fit_on_norm not in {"train", "all"}:
        raise ValueError("fit_on must be one of: train, all")
    predict_on_norm = predict_on.strip().lower() if predict_on is not None else ""
    if predict_on_norm and predict_on_norm not in {"train", "test", "all"}:
        raise ValueError("predict_on must be one of: train, test, all")

    if split_mode_norm == "none":
        return split_mode_norm, "all", "all", None, None

    if train_end is None:
        raise ValueError("train_end is required when split_mode=time.")
    resolved_test_start = test_start or (train_end + timedelta(days=1))
    if test_end is not None and test_end < resolved_test_start:
        raise ValueError("test_end must be >= test_start.")
    if predict_on_norm == "":
        predict_on_norm = "test"
    return split_mode_norm, fit_on_norm, predict_on_norm, resolved_test_start, test_end


def _resolve_feature_list(settings: AppSettings) -> list[str]:
    if settings.research_hmm.default_feature_list:
        return settings.research_hmm.default_feature_list
    return settings.research_clustering.default_feature_list


def _prepare_hmm_data(
    settings: AppSettings,
    *,
    dataset_path: Path,
    sample_frac: float | None,
    date_from: date | None,
    date_to: date | None,
    split_mode: str | None,
    train_end: date | None,
    test_start: date | None,
    test_end: date | None,
    fit_on: str,
    predict_on: str | None,
    scaler: str | None,
    scaling_scope: str | None,
    logger: logging.Logger,
) -> PreparedHMMData:
    loaded, with_forward = load_hmm_dataset(
        dataset_path,
        forward_windows=settings.research_clustering.forward_windows,
        date_from=date_from,
        date_to=date_to,
        sample_frac=sample_frac,
        logger=logger,
    )
    split_mode_value = split_mode or settings.research_hmm.split_mode_default
    split_mode_norm, fit_on_norm, predict_on_norm, resolved_test_start, resolved_test_end = _resolve_split_and_scopes(
        split_mode=split_mode_value,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        fit_on=fit_on,
        predict_on=predict_on,
    )

    all_df = with_forward
    train_df: pl.DataFrame | None = None
    test_df: pl.DataFrame | None = None
    if split_mode_norm == "time":
        if train_end is None or resolved_test_start is None:
            raise ValueError("train_end/test_start resolution failed for split mode time.")
        train_df = all_df.filter(pl.col("trade_date") <= pl.lit(train_end))
        test_filter = pl.col("trade_date") >= pl.lit(resolved_test_start)
        if resolved_test_end is not None:
            test_filter = test_filter & (pl.col("trade_date") <= pl.lit(resolved_test_end))
        test_df = all_df.filter(test_filter)
        if train_df.height == 0:
            raise ValueError("Time split produced empty train set.")
        if test_df.height == 0:
            raise ValueError("Time split produced empty test set.")

    fit_source = all_df if fit_on_norm == "all" else (train_df if train_df is not None else all_df)
    if predict_on_norm == "train":
        predict_source = train_df if train_df is not None else all_df
    elif predict_on_norm == "test":
        predict_source = test_df if test_df is not None else all_df
    else:
        predict_source = all_df

    scaler_name = (scaler or settings.research_hmm.scaler).strip().lower()
    scaling_scope_name = (scaling_scope or settings.research_hmm.scaling_scope_default).strip().lower()
    feature_list = _resolve_feature_list(settings)
    preprocess_model = fit_preprocess_model(
        fit_source,
        feature_list=feature_list,
        scaler=scaler_name,
        scaling_scope=scaling_scope_name,
    )
    preprocess_fit = transform_for_clustering(
        fit_source,
        model=preprocess_model,
        clip_zscore=settings.research_clustering.clip_zscore,
    )
    preprocess_predict = transform_for_clustering(
        predict_source,
        model=preprocess_model,
        clip_zscore=settings.research_clustering.clip_zscore,
    )
    preprocess_train = (
        transform_for_clustering(
            train_df,
            model=preprocess_model,
            clip_zscore=settings.research_clustering.clip_zscore,
        )
        if train_df is not None
        else None
    )
    preprocess_test = (
        transform_for_clustering(
            test_df,
            model=preprocess_model,
            clip_zscore=settings.research_clustering.clip_zscore,
        )
        if test_df is not None
        else None
    )

    split_summary = {
        "split_mode": split_mode_norm,
        "fit_on": fit_on_norm,
        "predict_on": predict_on_norm,
        "train_end": train_end.isoformat() if train_end is not None else None,
        "test_start": resolved_test_start.isoformat() if resolved_test_start is not None else None,
        "test_end": resolved_test_end.isoformat() if resolved_test_end is not None else None,
        "raw_stats": {
            "all": _frame_stats(all_df),
            "fit_source": _frame_stats(fit_source),
            "predict_source": _frame_stats(predict_source),
            "train": _frame_stats(train_df),
            "test": _frame_stats(test_df),
        },
        "processed_stats": {
            "fit_source": _frame_stats(preprocess_fit.processed_df),
            "predict_source": _frame_stats(preprocess_predict.processed_df),
            "train": _frame_stats(preprocess_train.processed_df) if preprocess_train is not None else None,
            "test": _frame_stats(preprocess_test.processed_df) if preprocess_test is not None else None,
        },
        "preprocess_scaler": preprocess_fit.preprocess_summary.get("scaler"),
        "preprocess_scaling_scope": preprocess_fit.preprocess_summary.get("scaling_scope"),
    }
    if preprocess_predict.preprocess_summary.get("rows_dropped_unseen_tickers", 0) > 0:
        logger.warning(
            "hmm.preprocess dropped_unseen_tickers rows=%s sample=%s",
            preprocess_predict.preprocess_summary.get("rows_dropped_unseen_tickers"),
            preprocess_predict.preprocess_summary.get("unseen_tickers_sample"),
        )

    return PreparedHMMData(
        loaded_rows=loaded.frame.height,
        preprocess_fit=preprocess_fit,
        preprocess_predict=preprocess_predict,
        preprocess_train=preprocess_train,
        preprocess_test=preprocess_test,
        split_summary=split_summary,
        split_mode=split_mode_norm,
        fit_on=fit_on_norm,
        predict_on=predict_on_norm,
        scaler=scaler_name,
        scaling_scope=scaling_scope_name,
        feature_list=preprocess_fit.feature_list,
    )


def _forward_separation_score(state_profile: pl.DataFrame) -> float | None:
    if "fwd_ret_10_mean" not in state_profile.columns or state_profile.height == 0:
        return None
    values = state_profile.select(pl.col("fwd_ret_10_mean")).to_series().drop_nulls()
    if values.len() == 0:
        return None
    return float(values.max() - values.min())


def _state_concentration_metrics(state_freq: pl.DataFrame) -> tuple[float | None, float | None]:
    if state_freq.height == 0 or "share_of_rows" not in state_freq.columns:
        return None, None
    shares = state_freq.select("share_of_rows").to_series().to_numpy()
    shares = shares[np.isfinite(shares)]
    if shares.size == 0:
        return None, None
    largest = float(np.max(shares))
    effective = float(1.0 / np.sum(np.square(shares)))
    return largest, effective


def _avg_self_transition_prob(transition_matrix: pl.DataFrame) -> float | None:
    if transition_matrix.height == 0:
        return None
    diag = transition_matrix.filter(pl.col("hmm_state_prev") == pl.col("hmm_state"))
    if diag.height == 0:
        return None
    return float(diag.select(pl.col("transition_probability").mean()).item())


def _avg_dwell_mean(dwell_stats: pl.DataFrame) -> float | None:
    if dwell_stats.height == 0 or "dwell_mean" not in dwell_stats.columns:
        return None
    return float(dwell_stats.select(pl.col("dwell_mean").mean()).item())


def _robustness_notes(
    *,
    largest_state_share: float | None,
    forward_separation_score: float | None,
    avg_self_transition_prob: float | None,
) -> list[str]:
    notes: list[str] = []
    if largest_state_share is None:
        notes.append("state_concentration_unknown")
    elif largest_state_share > 0.70:
        notes.append("high_state_concentration")
    else:
        notes.append("state_concentration_ok")

    if forward_separation_score is None:
        notes.append("forward_separation_unavailable")
    elif abs(forward_separation_score) < 0.002:
        notes.append("weak_forward_separation")
    else:
        notes.append("forward_separation_ok")

    if avg_self_transition_prob is None:
        notes.append("transition_persistence_unknown")
    elif avg_self_transition_prob < 0.35:
        notes.append("low_state_persistence")
    else:
        notes.append("state_persistence_ok")
    return notes


def _load_cluster_labels(cluster_labels_file: Path) -> pl.DataFrame:
    suffix = cluster_labels_file.suffix.lower()
    if suffix == ".parquet":
        labels = pl.read_parquet(cluster_labels_file)
    elif suffix == ".csv":
        labels = pl.read_csv(cluster_labels_file)
    else:
        raise ValueError("cluster-labels-file must be .parquet or .csv")

    cluster_column = None
    for candidate in ("cluster_label", "cluster_id"):
        if candidate in labels.columns:
            cluster_column = candidate
            break
    if cluster_column is None:
        raise ValueError("cluster-labels-file must include cluster_label or cluster_id.")
    if "ticker" not in labels.columns or "trade_date" not in labels.columns:
        raise ValueError("cluster-labels-file must include ticker and trade_date.")

    out = labels.with_columns(
        [
            pl.col("ticker").cast(pl.String).str.to_uppercase().alias("ticker"),
            pl.col("trade_date").cast(pl.Date, strict=False).alias("trade_date"),
            pl.col(cluster_column).alias("cluster_label"),
        ]
    ).select(["ticker", "trade_date", "cluster_label"])
    out = out.filter(pl.col("trade_date").is_not_null())
    return out.unique(subset=["ticker", "trade_date"], keep="first")


def _cluster_agreement_metrics(joined: pl.DataFrame) -> dict[str, Any]:
    subset = joined.filter(pl.col("cluster_label").is_not_null())
    if subset.height == 0:
        return {"rows_compared": 0, "ari": None, "nmi": None}
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    hmm_labels = subset.select(pl.col("hmm_state")).to_series().to_numpy()
    cluster_labels = subset.select(pl.col("cluster_label")).to_series().to_numpy()
    ari = float(adjusted_rand_score(cluster_labels, hmm_labels))
    nmi = float(normalized_mutual_info_score(cluster_labels, hmm_labels))
    return {"rows_compared": int(subset.height), "ari": ari, "nmi": nmi}


def _pairwise_ari(labels_by_seed: dict[int, np.ndarray]) -> tuple[pl.DataFrame, dict[str, float | None]]:
    if len(labels_by_seed) < 2:
        df = pl.DataFrame(schema={"seed_a": pl.Int64, "seed_b": pl.Int64, "ari": pl.Float64})
        return df, {"ari_mean": None, "ari_median": None, "ari_min": None, "ari_max": None}
    from sklearn.metrics import adjusted_rand_score

    rows: list[dict[str, Any]] = []
    values: list[float] = []
    seeds = sorted(labels_by_seed.keys())
    for idx, seed_a in enumerate(seeds):
        for seed_b in seeds[idx + 1 :]:
            ari = float(adjusted_rand_score(labels_by_seed[seed_a], labels_by_seed[seed_b]))
            rows.append({"seed_a": seed_a, "seed_b": seed_b, "ari": ari})
            values.append(ari)
    df = pl.DataFrame(rows).sort(["seed_a", "seed_b"])
    return df, {
        "ari_mean": float(np.mean(values)),
        "ari_median": float(np.median(values)),
        "ari_min": float(np.min(values)),
        "ari_max": float(np.max(values)),
    }


def _build_sequences_for_run(
    prepared: PreparedHMMData,
    *,
    min_sequence_length: int,
    logger: logging.Logger,
) -> tuple[SequenceBuildResult, SequenceBuildResult, SequenceBuildResult | None, SequenceBuildResult | None]:
    fit_seq = build_hmm_sequences(
        prepared.preprocess_fit.processed_df,
        feature_list=prepared.feature_list,
        min_sequence_length=min_sequence_length,
        logger=logger,
    )
    predict_seq = build_hmm_sequences(
        prepared.preprocess_predict.processed_df,
        feature_list=prepared.feature_list,
        min_sequence_length=min_sequence_length,
        logger=logger,
    )
    train_seq = (
        build_hmm_sequences(
            prepared.preprocess_train.processed_df,
            feature_list=prepared.feature_list,
            min_sequence_length=min_sequence_length,
            logger=logger,
        )
        if prepared.preprocess_train is not None
        else None
    )
    test_seq = (
        build_hmm_sequences(
            prepared.preprocess_test.processed_df,
            feature_list=prepared.feature_list,
            min_sequence_length=min_sequence_length,
            logger=logger,
        )
        if prepared.preprocess_test is not None
        else None
    )
    return fit_seq, predict_seq, train_seq, test_seq


def run_hmm_baseline(
    settings: AppSettings,
    *,
    dataset_path: Path,
    n_components: int,
    covariance_type: str | None = None,
    n_iter: int | None = None,
    tol: float | None = None,
    random_state: int | None = None,
    sample_frac: float | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    split_mode: str | None = None,
    train_end: date | None = None,
    test_start: date | None = None,
    test_end: date | None = None,
    fit_on: str = "train",
    predict_on: str | None = None,
    scaler: str | None = None,
    scaling_scope: str | None = None,
    cluster_labels_file: Path | None = None,
    write_full_decoded_csv: bool = False,
    logger: logging.Logger | None = None,
) -> HMMRunResult:
    """Run one HMM baseline fit/decode/profile pass and write artifacts."""

    effective_logger = logger or LOGGER
    started_ts = datetime.now(timezone.utc)
    prepared = _prepare_hmm_data(
        settings,
        dataset_path=dataset_path,
        sample_frac=sample_frac,
        date_from=date_from,
        date_to=date_to,
        split_mode=split_mode,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        fit_on=fit_on,
        predict_on=predict_on,
        scaler=scaler,
        scaling_scope=scaling_scope,
        logger=effective_logger,
    )

    min_seq_len = settings.research_hmm.min_sequence_length
    fit_seq, predict_seq, _train_seq, test_seq = _build_sequences_for_run(
        prepared,
        min_sequence_length=min_seq_len,
        logger=effective_logger,
    )
    model_result: HMMFitResult = fit_gaussian_hmm(
        fit_seq.X,
        fit_seq.lengths,
        n_components=n_components,
        covariance_type=(covariance_type or settings.research_hmm.hmm.covariance_type),
        n_iter=(n_iter if n_iter is not None else settings.research_hmm.hmm.n_iter),
        tol=(tol if tol is not None else settings.research_hmm.hmm.tol),
        random_state=(random_state if random_state is not None else settings.research_hmm.hmm.random_state),
        test_X=test_seq.X if test_seq is not None else None,
        test_lengths=test_seq.lengths if test_seq is not None else None,
    )

    decoded_states, posterior = decode_with_model(
        model_result.model,
        predict_seq.X,
        predict_seq.lengths,
    )
    run_id = f"hmm-{uuid4().hex[:12]}"
    decoded_rows = build_decoded_rows(
        predict_seq.frame,
        decoded_states=decoded_states,
        posterior_probs=posterior,
        run_id=run_id,
        schema_version="v1",
        calc_version="hmm_baseline_v1",
    )
    diagnostics: HMMDiagnostics = build_hmm_diagnostics(decoded_rows)
    profiles: HMMProfiles = build_hmm_state_profiles(decoded_rows)

    cluster_crosstab: pl.DataFrame | None = None
    cluster_metrics: dict[str, Any] | None = None
    if cluster_labels_file is not None:
        labels_df = _load_cluster_labels(cluster_labels_file)
        joined = decoded_rows.join(labels_df, on=["ticker", "trade_date"], how="left")
        cluster_crosstab = build_hmm_vs_cluster_crosstab(joined)
        cluster_metrics = _cluster_agreement_metrics(joined)

    largest_share, effective_count = _state_concentration_metrics(diagnostics.state_frequency)
    forward_sep = _forward_separation_score(profiles.hmm_state_profile)
    avg_self_prob = _avg_self_transition_prob(diagnostics.transition_matrix)
    avg_dwell = _avg_dwell_mean(diagnostics.dwell_stats)
    train_obs = int(fit_seq.X.shape[0])
    test_obs = int(test_seq.X.shape[0]) if test_seq is not None else 0
    robustness_summary = {
        "largest_state_share": largest_share,
        "effective_state_count": effective_count,
        "train_loglik_per_obs": (model_result.train_loglik / train_obs) if train_obs > 0 else None,
        "test_loglik_per_obs": (model_result.test_loglik / test_obs) if (model_result.test_loglik is not None and test_obs > 0) else None,
        "forward_separation_score": forward_sep,
        "avg_self_transition_prob": avg_self_prob,
        "avg_dwell_mean": avg_dwell,
        "notes": _robustness_notes(
            largest_state_share=largest_share,
            forward_separation_score=forward_sep,
            avg_self_transition_prob=avg_self_prob,
        ),
    }

    dataset_tag = dataset_path.parent.name.replace(" ", "_")
    output_dir = settings.paths.artifacts_root / "hmm_runs" / f"{run_id}_hmm_{dataset_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary_path = output_dir / "run_summary.json"
    split_summary_path = output_dir / "split_summary.json"
    preprocess_summary_path = output_dir / "preprocess_summary.json"
    model_meta_path = output_dir / "hmm_model_meta.json"
    decoded_rows_path = output_dir / "decoded_rows.parquet"
    decoded_sample_path = output_dir / "decoded_rows_sample.csv"
    state_profile_parquet = output_dir / "hmm_state_profile.parquet"
    state_profile_csv = output_dir / "hmm_state_profile.csv"
    flow_crosstab_parquet = output_dir / "hmm_vs_flow_state_crosstab.parquet"
    flow_crosstab_csv = output_dir / "hmm_vs_flow_state_crosstab.csv"
    forward_validation_path = output_dir / "hmm_forward_validation.parquet"
    transition_matrix_csv = output_dir / "transition_matrix.csv"
    transition_matrix_parquet = output_dir / "transition_matrix.parquet"
    transition_counts_csv = output_dir / "transition_counts.csv"
    transition_counts_parquet = output_dir / "transition_counts.parquet"
    dwell_stats_csv = output_dir / "dwell_stats.csv"
    dwell_stats_parquet = output_dir / "dwell_stats.parquet"
    state_frequency_csv = output_dir / "state_frequency.csv"
    state_frequency_parquet = output_dir / "state_frequency.parquet"
    initial_dist_csv = output_dir / "initial_state_distribution.csv"
    confidence_stats_csv = output_dir / "state_confidence_stats.csv"
    robustness_summary_path = output_dir / "robustness_summary.json"
    feature_list_path = output_dir / "feature_list.json"
    scaler_params_path = output_dir / "scaler_params.json"
    scaler_table_path = output_dir / "scaler_params_per_ticker.parquet"

    _write_json_atomically(prepared.split_summary, split_summary_path)
    _write_json_atomically(prepared.preprocess_predict.preprocess_summary, preprocess_summary_path)
    _write_json_atomically(model_result.model_meta, model_meta_path)
    _write_parquet_atomically(decoded_rows, decoded_rows_path)
    if write_full_decoded_csv:
        _write_csv_atomically(decoded_rows, output_dir / "decoded_rows.csv")
    _write_csv_atomically(decoded_rows.head(5_000), decoded_sample_path)
    _write_parquet_atomically(profiles.hmm_state_profile, state_profile_parquet)
    _write_csv_atomically(profiles.hmm_state_profile, state_profile_csv)
    _write_parquet_atomically(profiles.hmm_vs_flow_state_crosstab, flow_crosstab_parquet)
    _write_csv_atomically(profiles.hmm_vs_flow_state_crosstab, flow_crosstab_csv)
    _write_parquet_atomically(profiles.hmm_forward_validation, forward_validation_path)
    _write_csv_atomically(diagnostics.transition_matrix, transition_matrix_csv)
    _write_parquet_atomically(diagnostics.transition_matrix, transition_matrix_parquet)
    _write_csv_atomically(diagnostics.transition_counts, transition_counts_csv)
    _write_parquet_atomically(diagnostics.transition_counts, transition_counts_parquet)
    _write_csv_atomically(diagnostics.dwell_stats, dwell_stats_csv)
    _write_parquet_atomically(diagnostics.dwell_stats, dwell_stats_parquet)
    _write_csv_atomically(diagnostics.state_frequency, state_frequency_csv)
    _write_parquet_atomically(diagnostics.state_frequency, state_frequency_parquet)
    _write_csv_atomically(diagnostics.initial_state_distribution, initial_dist_csv)
    _write_csv_atomically(diagnostics.state_confidence_stats, confidence_stats_csv)
    _write_json_atomically(robustness_summary, robustness_summary_path)
    _write_json_atomically({"feature_list": prepared.feature_list}, feature_list_path)
    _write_json_atomically(prepared.preprocess_predict.scaler_params, scaler_params_path)
    if prepared.preprocess_predict.scaler_params_table is not None:
        _write_parquet_atomically(prepared.preprocess_predict.scaler_params_table, scaler_table_path)

    cluster_crosstab_path: str | None = None
    cluster_metrics_path: str | None = None
    if cluster_crosstab is not None and cluster_metrics is not None:
        crosstab_path = output_dir / "hmm_vs_cluster_crosstab.csv"
        metrics_path = output_dir / "hmm_vs_cluster_metrics.json"
        _write_csv_atomically(cluster_crosstab, crosstab_path)
        _write_json_atomically(cluster_metrics, metrics_path)
        cluster_crosstab_path = str(crosstab_path)
        cluster_metrics_path = str(metrics_path)

    finished_ts = datetime.now(timezone.utc)
    run_summary = {
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "started_ts": started_ts.isoformat(),
        "finished_ts": finished_ts.isoformat(),
        "duration_sec": round((finished_ts - started_ts).total_seconds(), 3),
        "method": "gaussian_hmm",
        "n_components": int(n_components),
        "split_mode": prepared.split_mode,
        "fit_on": prepared.fit_on,
        "predict_on": prepared.predict_on,
        "scaler": prepared.scaler,
        "scaling_scope": prepared.scaling_scope,
        "min_sequence_length": min_seq_len,
        "rows_loaded": prepared.loaded_rows,
        "rows_fit": int(fit_seq.X.shape[0]),
        "rows_decoded": int(decoded_rows.height),
        "sequences_fit": int(fit_seq.lengths.shape[0]),
        "sequences_decoded": int(predict_seq.lengths.shape[0]),
        "train_loglik": model_result.train_loglik,
        "test_loglik": model_result.test_loglik,
        "robustness_summary": robustness_summary,
        "outputs": {
            "run_summary": str(run_summary_path),
            "split_summary": str(split_summary_path),
            "preprocess_summary": str(preprocess_summary_path),
            "hmm_model_meta": str(model_meta_path),
            "decoded_rows": str(decoded_rows_path),
            "decoded_rows_sample": str(decoded_sample_path),
            "hmm_state_profile_parquet": str(state_profile_parquet),
            "hmm_state_profile_csv": str(state_profile_csv),
            "hmm_vs_flow_state_crosstab_csv": str(flow_crosstab_csv),
            "transition_matrix_csv": str(transition_matrix_csv),
            "transition_counts_csv": str(transition_counts_csv),
            "dwell_stats_csv": str(dwell_stats_csv),
            "state_frequency_csv": str(state_frequency_csv),
            "initial_state_distribution_csv": str(initial_dist_csv),
            "state_confidence_stats_csv": str(confidence_stats_csv),
            "robustness_summary_path": str(robustness_summary_path),
            "feature_list_path": str(feature_list_path),
            "scaler_params_path": str(scaler_params_path),
            "scaler_params_table_path": str(scaler_table_path) if prepared.preprocess_predict.scaler_params_table is not None else None,
            "hmm_vs_cluster_crosstab_csv": cluster_crosstab_path,
            "hmm_vs_cluster_metrics_json": cluster_metrics_path,
        },
    }
    _write_json_atomically(run_summary, run_summary_path)
    effective_logger.info(
        "hmm_run.complete run_id=%s n_components=%s split_mode=%s rows_decoded=%s output=%s",
        run_id,
        n_components,
        prepared.split_mode,
        decoded_rows.height,
        output_dir,
    )
    return HMMRunResult(
        run_id=run_id,
        output_dir=output_dir,
        run_summary_path=run_summary_path,
        decoded_rows_path=decoded_rows_path,
        state_profile_path=state_profile_parquet,
    )


def run_hmm_sweep(
    settings: AppSettings,
    *,
    dataset_path: Path,
    components: list[int],
    sample_frac: float | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    split_mode: str | None = None,
    train_end: date | None = None,
    test_start: date | None = None,
    test_end: date | None = None,
    fit_on: str = "train",
    predict_on: str | None = None,
    scaler: str | None = None,
    scaling_scope: str | None = None,
    logger: logging.Logger | None = None,
) -> HMMSweepResult:
    """Run HMM across component counts and write comparison summary artifacts."""

    effective_logger = logger or LOGGER
    prepared = _prepare_hmm_data(
        settings,
        dataset_path=dataset_path,
        sample_frac=sample_frac,
        date_from=date_from,
        date_to=date_to,
        split_mode=split_mode,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        fit_on=fit_on,
        predict_on=predict_on,
        scaler=scaler,
        scaling_scope=scaling_scope,
        logger=effective_logger,
    )
    min_seq_len = settings.research_hmm.min_sequence_length
    fit_seq, predict_seq, _train_seq, test_seq = _build_sequences_for_run(
        prepared,
        min_sequence_length=min_seq_len,
        logger=effective_logger,
    )

    rows: list[dict[str, Any]] = []
    for n_components in components:
        try:
            fit_result = fit_gaussian_hmm(
                fit_seq.X,
                fit_seq.lengths,
                n_components=int(n_components),
                covariance_type=settings.research_hmm.hmm.covariance_type,
                n_iter=settings.research_hmm.hmm.n_iter,
                tol=settings.research_hmm.hmm.tol,
                random_state=settings.research_hmm.hmm.random_state,
                test_X=test_seq.X if test_seq is not None else None,
                test_lengths=test_seq.lengths if test_seq is not None else None,
            )
            states, posterior = decode_with_model(fit_result.model, predict_seq.X, predict_seq.lengths)
            decoded = build_decoded_rows(
                predict_seq.frame,
                decoded_states=states,
                posterior_probs=posterior,
                run_id="hmm-sweep",
                schema_version="v1",
                calc_version="hmm_baseline_v1",
            )
            profiles = build_hmm_state_profiles(decoded)
            diagnostics = build_hmm_diagnostics(decoded)
            largest_share, effective_count = _state_concentration_metrics(diagnostics.state_frequency)
            forward_sep = _forward_separation_score(profiles.hmm_state_profile)
            avg_dwell = _avg_dwell_mean(diagnostics.dwell_stats)
            rows.append(
                {
                    "n_components": int(n_components),
                    "rows_fit": int(fit_seq.X.shape[0]),
                    "rows_decoded": int(decoded.height),
                    "sequences_fit": int(fit_seq.lengths.shape[0]),
                    "sequences_decoded": int(predict_seq.lengths.shape[0]),
                    "train_loglik": fit_result.train_loglik,
                    "test_loglik": fit_result.test_loglik,
                    "train_loglik_per_obs": fit_result.train_loglik / float(max(1, fit_seq.X.shape[0])),
                    "test_loglik_per_obs": (fit_result.test_loglik / float(max(1, test_seq.X.shape[0]))) if (fit_result.test_loglik is not None and test_seq is not None) else None,
                    "largest_state_share": largest_share,
                    "effective_state_count": effective_count,
                    "forward_separation_score": forward_sep,
                    "avg_dwell_mean": avg_dwell,
                    "status": "ok",
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "n_components": int(n_components),
                    "rows_fit": int(fit_seq.X.shape[0]),
                    "rows_decoded": 0,
                    "sequences_fit": int(fit_seq.lengths.shape[0]),
                    "sequences_decoded": int(predict_seq.lengths.shape[0]),
                    "train_loglik": None,
                    "test_loglik": None,
                    "train_loglik_per_obs": None,
                    "test_loglik_per_obs": None,
                    "largest_state_share": None,
                    "effective_state_count": None,
                    "forward_separation_score": None,
                    "avg_dwell_mean": None,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    summary_df = pl.DataFrame(rows).sort("n_components")
    run_id = f"hmm-sweep-{uuid4().hex[:12]}"
    dataset_tag = dataset_path.parent.name.replace(" ", "_")
    output_dir = settings.paths.artifacts_root / "hmm_runs" / f"{run_id}_hmm_sweep_{dataset_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_dir / "hmm_sweep_summary.json"
    summary_csv_path = output_dir / "hmm_sweep_summary.csv"
    payload = {
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "components": components,
        "split_mode": prepared.split_mode,
        "fit_on": prepared.fit_on,
        "predict_on": prepared.predict_on,
        "scaler": prepared.scaler,
        "scaling_scope": prepared.scaling_scope,
        "rows": summary_df.to_dicts(),
    }
    _write_json_atomically(payload, summary_json_path)
    _write_csv_atomically(summary_df, summary_csv_path)
    return HMMSweepResult(
        run_id=run_id,
        output_dir=output_dir,
        summary_json_path=summary_json_path,
        summary_csv_path=summary_csv_path,
        rows=summary_df.height,
    )


def run_hmm_stability(
    settings: AppSettings,
    *,
    dataset_path: Path,
    n_components: int,
    seeds: int,
    seed_start: int,
    sample_frac: float | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    split_mode: str | None = None,
    train_end: date | None = None,
    test_start: date | None = None,
    test_end: date | None = None,
    fit_on: str = "train",
    predict_on: str | None = None,
    scaler: str | None = None,
    scaling_scope: str | None = None,
    logger: logging.Logger | None = None,
) -> HMMStabilityResult:
    """Run HMM repeatedly over seeds and compute pairwise ARI stability."""

    effective_logger = logger or LOGGER
    if seeds < 1:
        raise ValueError("seeds must be >= 1.")
    prepared = _prepare_hmm_data(
        settings,
        dataset_path=dataset_path,
        sample_frac=sample_frac,
        date_from=date_from,
        date_to=date_to,
        split_mode=split_mode,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        fit_on=fit_on,
        predict_on=predict_on,
        scaler=scaler,
        scaling_scope=scaling_scope,
        logger=effective_logger,
    )
    min_seq_len = settings.research_hmm.min_sequence_length
    fit_seq, predict_seq, _train_seq, test_seq = _build_sequences_for_run(
        prepared,
        min_sequence_length=min_seq_len,
        logger=effective_logger,
    )

    seed_rows: list[dict[str, Any]] = []
    labels_by_seed: dict[int, np.ndarray] = {}
    for seed in [seed_start + idx for idx in range(seeds)]:
        fit_result = fit_gaussian_hmm(
            fit_seq.X,
            fit_seq.lengths,
            n_components=int(n_components),
            covariance_type=settings.research_hmm.hmm.covariance_type,
            n_iter=settings.research_hmm.hmm.n_iter,
            tol=settings.research_hmm.hmm.tol,
            random_state=seed,
            test_X=test_seq.X if test_seq is not None else None,
            test_lengths=test_seq.lengths if test_seq is not None else None,
        )
        states, posterior = decode_with_model(fit_result.model, predict_seq.X, predict_seq.lengths)
        labels_by_seed[seed] = states.astype(np.int32, copy=False)
        decoded = build_decoded_rows(
            predict_seq.frame,
            decoded_states=states,
            posterior_probs=posterior,
            run_id="hmm-stability",
            schema_version="v1",
            calc_version="hmm_baseline_v1",
        )
        profiles = build_hmm_state_profiles(decoded)
        diagnostics = build_hmm_diagnostics(decoded)
        largest_share, effective_count = _state_concentration_metrics(diagnostics.state_frequency)
        seed_rows.append(
            {
                "seed": seed,
                "n_components": int(n_components),
                "train_loglik": fit_result.train_loglik,
                "test_loglik": fit_result.test_loglik,
                "train_loglik_per_obs": fit_result.train_loglik / float(max(1, fit_seq.X.shape[0])),
                "test_loglik_per_obs": (fit_result.test_loglik / float(max(1, test_seq.X.shape[0]))) if (fit_result.test_loglik is not None and test_seq is not None) else None,
                "largest_state_share": largest_share,
                "effective_state_count": effective_count,
                "forward_separation_score": _forward_separation_score(profiles.hmm_state_profile),
            }
        )

    by_seed_df = pl.DataFrame(seed_rows).sort("seed")
    ari_df, ari_summary = _pairwise_ari(labels_by_seed)
    run_id = f"hmm-stability-{uuid4().hex[:12]}"
    dataset_tag = dataset_path.parent.name.replace(" ", "_")
    output_dir = settings.paths.artifacts_root / "hmm_runs" / f"{run_id}_hmm_stability_{dataset_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    by_seed_path = output_dir / "hmm_stability_by_seed.csv"
    by_seed_parquet = output_dir / "hmm_stability_by_seed.parquet"
    pairwise_csv = output_dir / "hmm_pairwise_ari.csv"
    pairwise_parquet = output_dir / "hmm_pairwise_ari.parquet"
    summary_json_path = output_dir / "hmm_stability_summary.json"

    _write_csv_atomically(by_seed_df, by_seed_path)
    _write_parquet_atomically(by_seed_df, by_seed_parquet)
    _write_csv_atomically(ari_df, pairwise_csv)
    _write_parquet_atomically(ari_df, pairwise_parquet)
    _write_json_atomically(
        {
            "run_id": run_id,
            "dataset_path": str(dataset_path),
            "n_components": int(n_components),
            "seeds": seeds,
            "seed_start": seed_start,
            "split_mode": prepared.split_mode,
            "fit_on": prepared.fit_on,
            "predict_on": prepared.predict_on,
            "scaler": prepared.scaler,
            "scaling_scope": prepared.scaling_scope,
            "rows_fit": int(fit_seq.X.shape[0]),
            "rows_decoded": int(predict_seq.X.shape[0]),
            "ari_summary": ari_summary,
            "outputs": {
                "hmm_stability_by_seed_csv": str(by_seed_path),
                "hmm_stability_by_seed_parquet": str(by_seed_parquet),
                "hmm_pairwise_ari_csv": str(pairwise_csv),
                "hmm_pairwise_ari_parquet": str(pairwise_parquet),
            },
        },
        summary_json_path,
    )
    return HMMStabilityResult(
        run_id=run_id,
        output_dir=output_dir,
        summary_json_path=summary_json_path,
        by_seed_path=by_seed_parquet,
        pairwise_ari_path=pairwise_csv,
    )

