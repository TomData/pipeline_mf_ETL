"""Configuration models and loading logic."""

from __future__ import annotations

import os
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


class ValidationConfig(BaseModel):
    """Validation thresholds for row-level quality checks."""

    suspicious_range_pct_threshold: float = 0.5
    suspicious_return_pct_threshold: float = 0.3
    gap_days_warn_threshold: int = 7


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
