"""Configuration models and loading logic."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DEFAULT_SETTINGS_FILE = Path("configs/settings.yaml")
SETTINGS_FILE_ENV = "MF_ETL_SETTINGS_FILE"


class ProjectConfig(BaseModel):
    """Project metadata settings."""

    name: str = "mf_etl"
    env: str = "dev"


class PathsConfig(BaseModel):
    """Filesystem paths used by each ETL stage."""

    raw_root: Path = Path("/media/tom/Hdd_240GB/data")
    data_root: Path = Path("./data")
    bronze_root: Path = Path("./data/bronze")
    silver_root: Path = Path("./data/silver")
    gold_root: Path = Path("./data/gold")
    artifacts_root: Path = Path("./artifacts")
    logs_root: Path = Path("./logs")

    def resolved(self, project_root: Path) -> "PathsConfig":
        """Return a copy with project-relative paths resolved to absolute paths."""

        updates: dict[str, Path] = {}
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            updates[field_name] = value if value.is_absolute() else (project_root / value).resolve()
        return self.model_copy(update=updates)


class PrecisionConfig(BaseModel):
    """Numeric precision policy per pipeline layer."""

    bronze_float: Literal["float64", "float32"] = "float64"
    silver_float: Literal["float64", "float32"] = "float32"
    gold_float: Literal["float64", "float32"] = "float32"


class ParquetConfig(BaseModel):
    """Parquet write settings."""

    compression: Literal["zstd", "snappy", "gzip", "brotli", "lz4", "none"] = "zstd"
    compression_level: int | None = 3
    statistics: bool = True


class ValidationBootstrapConfig(BaseModel):
    """Bootstrap parameterization for validation harness confidence intervals."""

    n_boot: int = Field(default=1000, ge=10)
    ci: float = Field(default=0.95, gt=0.0, lt=1.0)
    mode: Literal["iid", "block"] = "iid"
    block_length: int = Field(default=10, ge=1)
    random_state: int = 42


class ValidationEventStudyConfig(BaseModel):
    """Transition event-study defaults for validation harness runs."""

    window_pre: int = Field(default=10, ge=1)
    window_post: int = Field(default=20, ge=1)
    min_events_per_transition: int = Field(default=50, ge=1)


class ValidationRollingStabilityConfig(BaseModel):
    """Rolling-window stability defaults for validation harness runs."""

    window_months: int = Field(default=12, ge=1)
    step_months: int = Field(default=3, ge=1)


class ValidationConfidenceScoreWeightsConfig(BaseModel):
    """Weights for composing per-state validation confidence scores."""

    sample_size: float = 0.2
    ci_width: float = 0.25
    sign_confidence: float = 0.2
    stability: float = 0.25
    separation: float = 0.1


class ValidationScorecardConfig(BaseModel):
    """Scorecard-level controls for validation harness summaries."""

    eps: float = Field(default=1e-12, gt=0.0)
    confidence_score_weights: ValidationConfidenceScoreWeightsConfig = Field(
        default_factory=ValidationConfidenceScoreWeightsConfig
    )


class ValidationIOConfig(BaseModel):
    """I/O controls for optional large validation harness artifacts."""

    write_large_artifacts_default: bool = False


class ValidationConfig(BaseModel):
    """Validation thresholds and validation-harness controls."""

    suspicious_range_pct_threshold: float = 0.5
    suspicious_return_pct_threshold: float = 0.3
    gap_days_warn_threshold: int = 7
    bootstrap: ValidationBootstrapConfig = Field(default_factory=ValidationBootstrapConfig)
    event_study: ValidationEventStudyConfig = Field(default_factory=ValidationEventStudyConfig)
    rolling_stability: ValidationRollingStabilityConfig = Field(
        default_factory=ValidationRollingStabilityConfig
    )
    scorecard: ValidationScorecardConfig = Field(default_factory=ValidationScorecardConfig)
    io: ValidationIOConfig = Field(default_factory=ValidationIOConfig)


class ValidationWalkForwardConfig(BaseModel):
    """Walk-forward orchestration defaults for multi-split OOS validation packs."""

    train_end_list_default: list[date] = Field(
        default_factory=lambda: [
            date(2012, 12, 31),
            date(2014, 12, 31),
            date(2016, 12, 31),
            date(2018, 12, 31),
            date(2020, 12, 31),
        ],
        min_length=1,
    )
    hmm_components_default: int = Field(default=5, ge=2)
    cluster_method_default: Literal["gmm", "kmeans"] = "gmm"
    cluster_k_default: int = Field(default=5, ge=2)
    scaling_scope_default: Literal["global", "per_ticker"] = "per_ticker"
    continue_on_error_default: bool = True


class ClusterQAConfig(BaseModel):
    """Thresholds for diagnosing unstable cluster validation states."""

    ret_cv_threshold: float = Field(default=5.0, gt=0.0)
    min_n_rows: int = Field(default=200, ge=1)
    min_state_share: float = Field(default=0.03, ge=0.0, le=1.0)
    sign_consistency_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    ci_width_quantile_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    eps: float = Field(default=1e-12, gt=0.0)


class ClusterHardeningPenaltyConfig(BaseModel):
    """Penalty map for cluster hardening QA issue labels."""

    LOW_N: float = 20.0
    LOW_OCCUPANCY: float = 20.0
    MEAN_NEAR_ZERO_CV_INFLATION: float = 15.0
    WIDE_CI: float = 15.0
    SIGN_FLIP_ACROSS_WINDOWS: float = 20.0
    WINDOW_DRIFT_HIGH: float = 15.0
    LIKELY_OUTLIER_WINDOW: float = 10.0
    TRANSITIONS_TOO_SPARSE: float = 10.0


class ClusterHardeningWeightsConfig(BaseModel):
    """Component weights for tradability scoring."""

    sample_size: float = 0.15
    occupancy: float = 0.15
    sign_confidence: float = 0.20
    ci_width: float = 0.15
    sign_consistency: float = 0.15
    ret_cv: float = 0.10
    confidence_score: float = 0.10


class ClusterHardeningConfig(BaseModel):
    """State hardening thresholds and scoring controls."""

    min_n_rows_hard: int = Field(default=200, ge=1)
    min_state_share_hard: float = Field(default=0.03, ge=0.0, le=1.0)
    ret_cv_hard: float = Field(default=6.0, gt=0.0)
    sign_consistency_hard: float = Field(default=0.55, ge=0.0, le=1.0)
    ci_width_hard_quantile: float = Field(default=0.80, ge=0.0, le=1.0)
    score_min_allow: float = Field(default=70.0, ge=0.0, le=100.0)
    score_min_watch: float = Field(default=45.0, ge=0.0, le=100.0)
    penalties: ClusterHardeningPenaltyConfig = Field(default_factory=ClusterHardeningPenaltyConfig)
    weights: ClusterHardeningWeightsConfig = Field(default_factory=ClusterHardeningWeightsConfig)
    eps: float = Field(default=1e-12, gt=0.0)


class BacktestFlowMappingConfig(BaseModel):
    """Default deterministic mapping for flow_state_code direction classes."""

    long_states: list[int] = Field(default_factory=lambda: [1, 2])
    short_states: list[int] = Field(default_factory=lambda: [3, 4])
    ignore_states: list[int] = Field(default_factory=lambda: [0])


class BacktestHMMDirectionInferenceConfig(BaseModel):
    """Controls for HMM state-direction inference fallback order."""

    source_priority: list[Literal["state_map", "validation_scorecard", "profile"]] = Field(
        default_factory=lambda: ["state_map", "validation_scorecard", "profile"]
    )
    min_abs_forward_mean_for_direction: float = Field(default=0.0, ge=0.0)


class BacktestClusterPolicyConfig(BaseModel):
    """Cluster tradability policy defaults for backtesting."""

    default_classes: list[Literal["ALLOW", "WATCH", "BLOCK"]] = Field(default_factory=lambda: ["ALLOW"])
    include_watch_default: bool = False


class BacktestNanPolicyConfig(BaseModel):
    """Finite handling switches for backtest metrics and input sanitation."""

    strict_finite_prices: bool = True
    finite_aggregate_only: bool = True


class BacktestConfig(BaseModel):
    """Backtest harness defaults for state-driven execution research runs."""

    signal_mode: Literal["state_entry", "state_transition_entry", "state_persistence_confirm"] = (
        "state_transition_entry"
    )
    exit_mode: Literal["horizon", "state_exit", "horizon_or_state"] = "horizon_or_state"
    hold_bars: int = Field(default=10, ge=1)
    allow_overlap: bool = False
    allow_unconfirmed: bool = False
    equity_mode: Literal["event_returns_only", "daily_equity_curve"] = "event_returns_only"
    fee_bps_per_side: float = Field(default=0.0, ge=0.0)
    slippage_bps_per_side: float = Field(default=0.0, ge=0.0)
    capital_base: float = Field(default=1.0, gt=0.0)
    flow_mapping: BacktestFlowMappingConfig = Field(default_factory=BacktestFlowMappingConfig)
    hmm_direction_inference: BacktestHMMDirectionInferenceConfig = Field(
        default_factory=BacktestHMMDirectionInferenceConfig
    )
    cluster_policy: BacktestClusterPolicyConfig = Field(default_factory=BacktestClusterPolicyConfig)
    nan_policy: BacktestNanPolicyConfig = Field(default_factory=BacktestNanPolicyConfig)


class BacktestPolicyOverlayConfig(BaseModel):
    """Hybrid policy overlay defaults for gating primary state signals."""

    default_overlay_mode: Literal[
        "none", "allow_only", "allow_watch", "block_veto", "allow_or_unknown"
    ] = "none"
    join_keys: list[str] = Field(default_factory=lambda: ["ticker", "trade_date"], min_length=1)
    allow_unknown_for_block_veto: bool = True
    min_overlay_match_rate_warn: float = Field(default=0.80, ge=0.0, le=1.0)
    dedupe_rule: Literal["first"] = "first"
    enable_direction_conflict_metrics: bool = True


class BacktestExecutionRealismProfileConfig(BaseModel):
    """One execution realism profile definition."""

    min_price: float | None = Field(default=None, ge=0.0)
    min_dollar_vol_20: float | None = Field(default=None, ge=0.0)
    max_vol_pct: float | None = Field(default=None, ge=0.0)
    min_history_bars_for_execution: int | None = Field(default=None, ge=1)


class BacktestExecutionRealismProfilesConfig(BaseModel):
    """Named execution realism profiles."""

    none: BacktestExecutionRealismProfileConfig = Field(
        default_factory=BacktestExecutionRealismProfileConfig
    )
    lite: BacktestExecutionRealismProfileConfig = Field(
        default_factory=lambda: BacktestExecutionRealismProfileConfig(
            min_price=2.0,
            min_dollar_vol_20=1_000_000.0,
            max_vol_pct=0.12,
            min_history_bars_for_execution=50,
        )
    )
    strict: BacktestExecutionRealismProfileConfig = Field(
        default_factory=lambda: BacktestExecutionRealismProfileConfig(
            min_price=5.0,
            min_dollar_vol_20=5_000_000.0,
            max_vol_pct=0.08,
            min_history_bars_for_execution=100,
        )
    )


class BacktestExecutionRealismConfig(BaseModel):
    """Execution realism filter settings and report thresholds."""

    default_profile: Literal["none", "lite", "strict"] = "none"
    profiles: BacktestExecutionRealismProfilesConfig = Field(
        default_factory=BacktestExecutionRealismProfilesConfig
    )
    dollar_vol_window: int = Field(default=20, ge=1)
    dollar_vol_rolling_method: Literal["median", "mean"] = "median"
    vol_input_unit_mode: Literal["auto", "decimal", "percent_points"] = "auto"
    min_history_bars_default: int = Field(default=50, ge=1)
    report_min_trades_default: int = Field(default=30, ge=1)
    report_max_zero_trade_share_default: float = Field(default=0.50, ge=0.0, le=1.0)
    report_max_ret_cv_default: float = Field(default=20.0, gt=0.0)


class BacktestExecutionCalibrationSweepConfig(BaseModel):
    """Default threshold sweep ranges for execution realism calibration."""

    min_price: list[float] = Field(default_factory=lambda: [0.0, 1.0, 2.0, 5.0], min_length=1)
    min_dollar_vol20: list[float] = Field(
        default_factory=lambda: [0.0, 250_000.0, 500_000.0, 1_000_000.0, 2_000_000.0],
        min_length=1,
    )
    max_vol_pct: list[float | None] = Field(
        default_factory=lambda: [None, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0],
        min_length=1,
    )
    min_history_bars: list[int] = Field(default_factory=lambda: [20, 50], min_length=1)


class BacktestExecutionCalibrationConfig(BaseModel):
    """Calibration diagnostics and recommendation defaults for realism thresholds."""

    sweep: BacktestExecutionCalibrationSweepConfig = Field(default_factory=BacktestExecutionCalibrationSweepConfig)
    target_lite_eligibility_min: float = Field(default=0.20, ge=0.0, le=1.0)
    target_lite_eligibility_max: float = Field(default=0.60, ge=0.0, le=1.0)
    target_strict_eligibility_min: float = Field(default=0.05, ge=0.0, le=1.0)
    target_strict_eligibility_max: float = Field(default=0.30, ge=0.0, le=1.0)
    min_eligible_signals: int = Field(default=100, ge=1)
    max_single_reason_share: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k_recommendations: int = Field(default=5, ge=1)
    by_year_default: bool = True


class BacktestSensitivityDefaultGridConfig(BaseModel):
    """Default parameter grid for backtest sensitivity runs."""

    hold_bars: list[int] = Field(default_factory=lambda: [5, 10, 15, 20], min_length=1)
    signal_mode: list[Literal["state_entry", "state_transition_entry", "state_persistence_confirm"]] = (
        Field(default_factory=lambda: ["state_transition_entry"], min_length=1)
    )
    exit_mode: list[Literal["horizon", "state_exit", "horizon_or_state"]] = Field(
        default_factory=lambda: ["horizon_or_state"], min_length=1
    )
    fee_bps_per_side: list[float] = Field(default_factory=lambda: [0.0, 5.0, 10.0], min_length=1)
    slippage_bps_per_side: list[float] = Field(default_factory=lambda: [0.0], min_length=1)
    allow_overlap: list[bool] = Field(default_factory=lambda: [False], min_length=1)
    equity_mode: list[Literal["event_returns_only", "daily_equity_curve"]] = Field(
        default_factory=lambda: ["event_returns_only"],
        min_length=1,
    )
    include_watch: list[bool] = Field(default_factory=lambda: [False], min_length=1)
    include_state_sets: list[list[int]] = Field(default_factory=lambda: [[]], min_length=1)


class BacktestSensitivityRobustnessWeightsConfig(BaseModel):
    """Weights for backtest sensitivity robustness score composition."""

    expectancy_rank: float = 0.30
    profit_factor_rank: float = 0.20
    drawdown_score: float = 0.20
    consistency: float = 0.15
    cost_robustness: float = 0.10
    hygiene: float = 0.05


class BacktestSensitivityConfig(BaseModel):
    """Controls for backtest sensitivity grid orchestration and ranking."""

    progress_every: int = Field(default=10, ge=1)
    max_combos: int = Field(default=500, ge=1)
    stop_on_error: bool = False
    policy_filter_mode_default: Literal["allow_only", "allow_watch", "all_states"] = "allow_only"
    include_ret_cv_default: bool = True
    include_tail_metrics_default: bool = True
    report_top_n_default: int = Field(default=10, ge=1)
    min_successful_splits_default: int = Field(default=1, ge=1)
    default_grid: BacktestSensitivityDefaultGridConfig = Field(default_factory=BacktestSensitivityDefaultGridConfig)
    ranking_metric_default: Literal[
        "expectancy",
        "profit_factor",
        "max_drawdown",
        "sharpe_proxy",
        "robustness_score",
        "robustness_score_v2",
        "ret_cv",
        "downside_std",
    ] = "expectancy"
    robustness_score_weights: BacktestSensitivityRobustnessWeightsConfig = Field(
        default_factory=BacktestSensitivityRobustnessWeightsConfig
    )


class IndicatorsConfig(BaseModel):
    """Indicator-layer parameterization for Silver-derived signals."""

    tmf_period: int = Field(default=21, ge=1)
    eps: float = Field(default=1e-12, gt=0.0)
    proxy_period: int = Field(default=21, ge=1)
    float_dtype_override: Literal["float64", "float32"] | None = None


class EventGrammarConfig(BaseModel):
    """Gold event-grammar thresholds and behavior switches."""

    pivot_mode: Literal["3bar"] = "3bar"
    respect_fail_lookahead_bars: int = Field(default=10, ge=1)
    hold_consecutive_bars: int = Field(default=5, ge=1)
    tmf_burst_abs_threshold: float = Field(default=0.15, ge=0.0)
    tmf_burst_slope_threshold: float = Field(default=0.05, ge=0.0)
    activity_windows: list[int] = Field(default_factory=lambda: [5, 20], min_length=1)
    eps: float = Field(default=1e-12, gt=0.0)


class GoldFeatureScoreWeightsConfig(BaseModel):
    """Weights used to compose long/short flow event intensity scores."""

    zero: float = 1.0
    respect: float = 2.0
    burst: float = 2.0
    hold: float = 1.5


class GoldFeatureExportConfig(BaseModel):
    """Dataset export behavior flags for experiment-ready stacked outputs."""

    default_drop_null_key_features: bool = True


class GoldFeaturesConfig(BaseModel):
    """Gold features layer settings."""

    eps: float = Field(default=1e-12, gt=0.0)
    activity_windows: list[int] = Field(default_factory=lambda: [5, 20], min_length=1)
    score_weights: GoldFeatureScoreWeightsConfig = Field(default_factory=GoldFeatureScoreWeightsConfig)
    recency_clip_bars: int = Field(default=20, ge=1)
    float_dtype_override: Literal["float64", "float32"] | None = None
    export: GoldFeatureExportConfig = Field(default_factory=GoldFeatureExportConfig)


class ResearchKMeansConfig(BaseModel):
    """KMeans configuration defaults for research runs."""

    n_init: int = Field(default=20, ge=1)
    max_iter: int = Field(default=300, ge=1)


class ResearchGMMConfig(BaseModel):
    """Gaussian Mixture configuration defaults for research runs."""

    covariance_type: Literal["full", "tied", "diag", "spherical"] = "diag"
    reg_covar: float = Field(default=1e-6, ge=0.0)
    max_iter: int = Field(default=200, ge=1)


class ResearchTimeSplitConfig(BaseModel):
    """Default time split boundaries for OOS research runs."""

    train_end: date | None = None
    test_start: date | None = None
    test_end: date | None = None


class ResearchStabilityConfig(BaseModel):
    """Default settings for seed-based clustering stability sweeps."""

    seeds_default: int = Field(default=10, ge=1)
    seed_start_default: int = 42


class ResearchHMMModelConfig(BaseModel):
    """Core Gaussian HMM defaults."""

    n_components_default: int = Field(default=5, ge=2)
    covariance_type: Literal["diag", "full", "tied", "spherical"] = "diag"
    n_iter: int = Field(default=200, ge=1)
    tol: float = Field(default=1e-3, gt=0.0)
    random_state: int = 42


class ResearchHMMSweepConfig(BaseModel):
    """Default component grids for HMM sweep runs."""

    components_default: list[int] = Field(default_factory=lambda: [4, 5, 6, 8], min_length=1)


class ResearchHMMStabilityConfig(BaseModel):
    """Default seed settings for HMM stability runs."""

    seeds_default: int = Field(default=5, ge=1)
    seed_start_default: int = 42


class ResearchHMMConfig(BaseModel):
    """Sequential HMM baseline configuration."""

    default_feature_list: list[str] = Field(
        default_factory=lambda: [
            "tmf_21",
            "tmf_abs",
            "tmf_slope_1",
            "tmf_slope_5",
            "tmf_slope_10",
            "tmf_curvature_1",
            "tti_proxy_v1_21",
            "tti_proxy_slope_1",
            "tti_proxy_slope_5",
            "long_flow_score_20",
            "short_flow_score_20",
            "delta_flow_20",
            "flow_activity_20",
            "flow_bias_20",
            "long_burst_20",
            "short_burst_20",
            "persistence_pos_20",
            "persistence_neg_20",
            "oscillation_index_20",
            "respect_fail_balance_20",
            "rec_tmf_zero_up_20",
            "rec_tmf_zero_down_20",
            "rec_tmf_burst_up_20",
            "rec_tmf_burst_down_20",
            "state_run_length",
        ],
        min_length=1,
    )
    scaler: Literal["standard", "robust"] = "standard"
    scaling_scope_default: Literal["global", "per_ticker"] = "global"
    split_mode_default: Literal["none", "time"] = "none"
    min_sequence_length: int = Field(default=100, ge=1)
    hmm: ResearchHMMModelConfig = Field(default_factory=ResearchHMMModelConfig)
    sweep: ResearchHMMSweepConfig = Field(default_factory=ResearchHMMSweepConfig)
    stability: ResearchHMMStabilityConfig = Field(default_factory=ResearchHMMStabilityConfig)


class ResearchClusteringConfig(BaseModel):
    """Research clustering defaults for unsupervised baseline pipeline."""

    default_feature_list: list[str] = Field(
        default_factory=lambda: [
            "tmf_21",
            "tmf_abs",
            "tmf_slope_1",
            "tmf_slope_5",
            "tmf_slope_10",
            "tmf_curvature_1",
            "tti_proxy_v1_21",
            "tti_proxy_slope_1",
            "tti_proxy_slope_5",
            "long_flow_score_20",
            "short_flow_score_20",
            "delta_flow_20",
            "flow_activity_20",
            "flow_bias_20",
            "long_burst_20",
            "short_burst_20",
            "persistence_pos_20",
            "persistence_neg_20",
            "oscillation_index_20",
            "respect_fail_balance_20",
            "rec_tmf_zero_up_20",
            "rec_tmf_zero_down_20",
            "rec_tmf_burst_up_20",
            "rec_tmf_burst_down_20",
            "state_run_length",
        ],
        min_length=1,
    )
    scaler: Literal["standard", "robust"] = "standard"
    scaling_scope_default: Literal["global", "per_ticker"] = "global"
    split_mode_default: Literal["none", "time"] = "none"
    time_split: ResearchTimeSplitConfig = Field(default_factory=ResearchTimeSplitConfig)
    stability: ResearchStabilityConfig = Field(default_factory=ResearchStabilityConfig)
    clip_zscore: float | None = Field(default=8.0, gt=0.0)
    silhouette_sample_max: int = Field(default=200_000, ge=1000)
    random_state: int = 42
    kmeans: ResearchKMeansConfig = Field(default_factory=ResearchKMeansConfig)
    gmm: ResearchGMMConfig = Field(default_factory=ResearchGMMConfig)
    forward_windows: list[int] = Field(default_factory=lambda: [5, 10, 20], min_length=1)


class AppSettings(BaseSettings):
    """Top-level application settings."""

    _yaml_file_override: ClassVar[Path | None] = None

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
    parquet: ParquetConfig = Field(default_factory=ParquetConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    event_grammar: EventGrammarConfig = Field(default_factory=EventGrammarConfig)
    gold_features: GoldFeaturesConfig = Field(default_factory=GoldFeaturesConfig)
    validation_walkforward: ValidationWalkForwardConfig = Field(default_factory=ValidationWalkForwardConfig)
    cluster_qa: ClusterQAConfig = Field(default_factory=ClusterQAConfig)
    cluster_hardening: ClusterHardeningConfig = Field(default_factory=ClusterHardeningConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    backtest_policy_overlay: BacktestPolicyOverlayConfig = Field(default_factory=BacktestPolicyOverlayConfig)
    backtest_execution_realism: BacktestExecutionRealismConfig = Field(
        default_factory=BacktestExecutionRealismConfig
    )
    backtest_execution_calibration: BacktestExecutionCalibrationConfig = Field(
        default_factory=BacktestExecutionCalibrationConfig
    )
    backtest_sensitivity: BacktestSensitivityConfig = Field(default_factory=BacktestSensitivityConfig)
    research_clustering: ResearchClusteringConfig = Field(default_factory=ResearchClusteringConfig)
    research_hmm: ResearchHMMConfig = Field(default_factory=ResearchHMMConfig)

    model_config = SettingsConfigDict(
        env_prefix="MF_ETL_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Use YAML defaults while allowing env vars to override values."""

        yaml_file = resolve_settings_file(cls._yaml_file_override)
        yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file)
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_settings,
            file_secret_settings,
        )

    def as_dict(self) -> dict[str, object]:
        """Return settings as a standard nested dictionary."""

        return self.model_dump(mode="json")


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root by traversing upward for config markers."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs/settings.yaml").exists():
            return candidate
    return current


def resolve_settings_file(override: Path | None = None) -> Path:
    """Resolve settings file from explicit override, env var, or default."""

    chosen = override
    if chosen is None:
        env_value = os.getenv(SETTINGS_FILE_ENV)
        if env_value:
            chosen = Path(env_value)
    if chosen is None:
        chosen = DEFAULT_SETTINGS_FILE

    if not chosen.is_absolute():
        chosen = (find_project_root() / chosen).resolve()
    return chosen


def load_settings(config_file: Path | None = None) -> AppSettings:
    """Load settings with YAML defaults and environment variable overrides."""

    settings_file = resolve_settings_file(config_file)
    project_root = settings_file.parent.parent.resolve()
    AppSettings._yaml_file_override = settings_file
    try:
        settings = AppSettings()
    finally:
        AppSettings._yaml_file_override = None
    resolved_paths = settings.paths.resolved(project_root=project_root)
    return settings.model_copy(update={"paths": resolved_paths})
