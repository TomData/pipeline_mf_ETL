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
