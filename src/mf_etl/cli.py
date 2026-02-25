"""Typer CLI entrypoint for mf_etl."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import cast
from uuid import uuid4

import polars as pl
import typer
import yaml

from mf_etl.bronze.pipeline import BronzeRunOptions, run_bronze_pipeline
from mf_etl.bronze.sanity_checks import run_bronze_sanity_checks
from mf_etl.bronze.symbol_master import build_and_write_symbol_master, symbol_master_paths
from mf_etl.bronze.writer import write_bronze_artifacts
from mf_etl.config import AppSettings, load_settings
from mf_etl.ingest.discover import discover_txt_files, extract_ticker_hint, infer_exchange_from_path
from mf_etl.ingest.manifest import ManifestStatus, build_manifest, write_manifest_parquet
from mf_etl.ingest.read_txt import read_stock_txt_with_rejects
from mf_etl.logging_utils import configure_logging
from mf_etl.silver.pipeline import (
    SilverRunOptions,
    resolve_bronze_file_for_ticker,
    run_silver_one_from_bronze_file,
    run_silver_pipeline,
    run_silver_sanity,
)
from mf_etl.silver.indicators_pipeline import (
    IndicatorRunOptions,
    resolve_silver_base_file_for_ticker,
    run_indicators_one_from_silver_file,
    run_indicators_pipeline,
    run_indicators_sanity,
)
from mf_etl.gold.pipeline import (
    GoldEventRunOptions,
    resolve_indicator_file_for_ticker,
    run_events_one_from_indicator_file,
    run_events_pipeline,
    run_events_sanity,
)
from mf_etl.gold.features_pipeline import (
    GoldFeatureRunOptions,
    export_ml_dataset,
    resolve_events_file_for_ticker,
    run_features_one_from_events_file,
    run_features_pipeline,
    run_features_sanity,
)
from mf_etl.research.pipeline import (
    run_research_cluster,
    run_research_cluster_stability,
    run_research_cluster_sweep,
)
from mf_etl.research.sanity import summarize_research_run
from mf_etl.research_hmm.pipeline import (
    run_hmm_baseline,
    run_hmm_stability,
    run_hmm_sweep,
)
from mf_etl.research_hmm.sanity import summarize_hmm_run
from mf_etl.backtest.pipeline import (
    run_backtest_compare,
    run_backtest_run,
    run_backtest_walkforward,
)
from mf_etl.backtest.models import InputType
from mf_etl.backtest.sanity import summarize_backtest_run
from mf_etl.backtest.sensitivity_models import GridDimensionValues, SourceInputSpec
from mf_etl.backtest.hybrid_eval_report import run_hybrid_eval_report
from mf_etl.backtest.sensitivity_runner import (
    run_backtest_grid,
    run_backtest_grid_compare,
    run_backtest_grid_walkforward,
)
from mf_etl.backtest.sensitivity_sanity import summarize_grid_run
from mf_etl.validation.cluster_qa import run_cluster_qa_single, run_cluster_qa_walkforward
from mf_etl.validation.cluster_hardening import (
    run_cluster_hardening_compare,
    run_cluster_hardening_single,
    run_cluster_hardening_walkforward,
    summarize_cluster_hardening,
)
from mf_etl.validation.pipeline import run_validation_compare, run_validation_harness
from mf_etl.validation.sanity import summarize_validation_run
from mf_etl.validation.walkforward import run_validation_walkforward, summarize_validation_walkforward_run
from mf_etl.silver.placeholders import ensure_silver_placeholder
from mf_etl.gold.placeholders import ensure_gold_placeholder
from mf_etl.transform.normalize import BronzeNormalizeMetadata, normalize_bronze_rows
from mf_etl.utils.paths import ensure_directories
from mf_etl.validate.reports import quality_flag_counts, validate_bronze_dataframe
from mf_etl.validate.rules import ValidationThresholds

app = typer.Typer(
    add_completion=False,
    help="mf_etl command line interface.",
    no_args_is_help=True,
)


def _load_and_optionally_configure_logger(
    config_file: Path | None,
    configure: bool,
) -> tuple[AppSettings, logging.Logger]:
    settings = load_settings(config_file=config_file)
    if configure:
        logger = configure_logging(settings.paths.logs_root / "etl.log")
    else:
        logger = logging.getLogger("mf_etl")
    return settings, logger


def _parse_iso_date(value: str | None, option_name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must be YYYY-MM-DD.") from exc


def _parse_int_csv(value: str, option_name: str) -> list[int]:
    items = [part.strip() for part in value.split(",") if part.strip() != ""]
    if not items:
        raise typer.BadParameter(f"{option_name} must contain at least one integer.")
    parsed: list[int] = []
    for item in items:
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise typer.BadParameter(f"{option_name} must be comma-separated integers.") from exc
    return parsed


def _parse_date_csv(value: str, option_name: str) -> list[date]:
    items = [part.strip() for part in value.split(",") if part.strip() != ""]
    if not items:
        raise typer.BadParameter(f"{option_name} must contain at least one YYYY-MM-DD date.")
    parsed: list[date] = []
    for item in items:
        try:
            parsed.append(date.fromisoformat(item))
        except ValueError as exc:
            raise typer.BadParameter(
                f"{option_name} must be comma-separated dates in YYYY-MM-DD format."
            ) from exc
    return parsed


def _parse_method_csv(value: str) -> list[str]:
    methods = [part.strip().lower() for part in value.split(",") if part.strip() != ""]
    if not methods:
        raise typer.BadParameter("methods must contain at least one method.")
    allowed = {"kmeans", "gmm", "hdbscan"}
    for method in methods:
        if method not in allowed:
            raise typer.BadParameter("methods must be comma-separated values from: kmeans,gmm,hdbscan")
    return methods


def _parse_float_csv(value: str, option_name: str) -> list[float]:
    items = [part.strip() for part in value.split(",") if part.strip() != ""]
    if not items:
        raise typer.BadParameter(f"{option_name} must contain at least one numeric value.")
    parsed: list[float] = []
    for item in items:
        try:
            parsed.append(float(item))
        except ValueError as exc:
            raise typer.BadParameter(f"{option_name} must be comma-separated numbers.") from exc
    return parsed


def _parse_bool_csv(value: str, option_name: str) -> list[bool]:
    items = [part.strip().lower() for part in value.split(",") if part.strip() != ""]
    if not items:
        raise typer.BadParameter(f"{option_name} must contain at least one boolean value.")
    out: list[bool] = []
    mapping = {"true": True, "false": False}
    for item in items:
        if item not in mapping:
            raise typer.BadParameter(f"{option_name} must contain comma-separated true/false values.")
        out.append(mapping[item])
    return out


def _parse_choice_csv(value: str, option_name: str, allowed: set[str]) -> list[str]:
    items = [part.strip().lower() for part in value.split(",") if part.strip() != ""]
    if not items:
        raise typer.BadParameter(f"{option_name} must contain at least one value.")
    for item in items:
        if item not in allowed:
            allowed_rendered = ",".join(sorted(allowed))
            raise typer.BadParameter(f"{option_name} values must be from: {allowed_rendered}")
    return items


def _parse_state_set_grid(value: str, option_name: str) -> list[list[int]]:
    groups = [part.strip() for part in value.split(";") if part.strip() != ""]
    if not groups:
        raise typer.BadParameter(f"{option_name} must contain at least one state set.")
    out: list[list[int]] = []
    for group in groups:
        state_items = [p.strip() for p in group.split("|") if p.strip() != ""]
        if not state_items:
            raise typer.BadParameter(f"{option_name} has an empty state subset.")
        subset: list[int] = []
        for item in state_items:
            try:
                subset.append(int(item))
            except ValueError as exc:
                raise typer.BadParameter(f"{option_name} supports integer states only.") from exc
        out.append(sorted(set(subset)))
    return out


def _normalize_choice(value: str | None, *, allowed: set[str], option_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in allowed:
        allowed_rendered = ",".join(sorted(allowed))
        raise typer.BadParameter(f"{option_name} must be one of: {allowed_rendered}")
    return normalized


@app.command("show-config")
def show_config(
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Print the effective configuration after env overrides."""

    settings, _ = _load_and_optionally_configure_logger(config_file, configure=False)
    rendered = yaml.safe_dump(settings.as_dict(), sort_keys=False)
    typer.echo(rendered)


@app.command("bronze-run")
def bronze_run(
    full: bool = typer.Option(
        False,
        "--full",
        help="Process all files (including UNCHANGED).",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Process at most N selected files.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Discover and classify files without processing.",
    ),
    progress_every: int = typer.Option(
        100,
        "--progress-every",
        min=1,
        help="Log progress every N processed files.",
    ),
    only_status: str | None = typer.Option(
        None,
        "--only-status",
        help="Optional status filter: NEW, CHANGED, or UNCHANGED.",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run incremental Bronze pipeline over discovered source files."""

    normalized_only_status: str | None = None
    if only_status is not None:
        candidate = only_status.strip().upper()
        if candidate not in {"NEW", "CHANGED", "UNCHANGED"}:
            raise typer.BadParameter("only-status must be one of: NEW, CHANGED, UNCHANGED")
        normalized_only_status = candidate

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    options = BronzeRunOptions(
        full=full,
        limit=limit,
        dry_run=dry_run,
        progress_every=progress_every,
        only_status=cast(ManifestStatus | None, normalized_only_status),
    )
    result = run_bronze_pipeline(settings, options=options, logger=logger)

    summary = result.summary
    typer.echo(f"run_id: {summary['run_id']}")
    typer.echo(f"files_discovered_total: {summary['files_discovered_total']}")
    typer.echo(f"files_selected_total: {summary['files_selected_total']}")
    typer.echo(f"files_processed_success: {summary['files_processed_success']}")
    typer.echo(f"files_processed_failed: {summary['files_processed_failed']}")
    typer.echo(f"files_skipped_unchanged: {summary['files_skipped_unchanged']}")
    typer.echo(f"rows_total: {summary['rows_total']}")
    typer.echo(f"rows_valid: {summary['rows_valid']}")
    typer.echo(f"rows_invalid: {summary['rows_invalid']}")
    typer.echo(f"manifest_current_path: {summary['outputs']['manifest_current_path']}")
    typer.echo(f"manifest_stable_path: {summary['outputs']['manifest_stable_path']}")
    typer.echo(f"summary_path: {result.summary_path}")


@app.command("init-placeholders")
def init_placeholders(
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Create base data/artifacts/log folders and stage marker files."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    created_dirs = ensure_directories(
        [
            settings.paths.data_root,
            settings.paths.bronze_root,
            settings.paths.silver_root,
            settings.paths.gold_root,
            settings.paths.artifacts_root,
            settings.paths.logs_root,
        ]
    )
    silver_readme = ensure_silver_placeholder(settings.paths.silver_root)
    gold_readme = ensure_gold_placeholder(settings.paths.gold_root)
    logger.info("init_placeholders.created_dirs count=%s", len(created_dirs))
    logger.info("init_placeholders.marker_files silver=%s gold=%s", silver_readme, gold_readme)
    typer.echo("Initialized data/artifacts/logs folders and silver/gold marker files.")


@app.command("build-symbol-master")
def build_symbol_master_cmd(
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Build symbol-master artifacts from Bronze parquet outputs."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    result = build_and_write_symbol_master(settings.paths.bronze_root, logger=logger)

    logger.info(
        "build_symbol_master.summary rows=%s bronze_files=%s read_errors=%s quality_reports_total_files=%s parquet=%s csv=%s",
        result.symbol_master_df.height,
        result.bronze_file_count,
        result.bronze_files_read_errors,
        result.quality_reports_total_files,
        result.parquet_path,
        result.csv_path,
    )

    typer.echo(f"symbol_count: {result.symbol_master_df.height}")
    typer.echo(f"bronze_file_count: {result.bronze_file_count}")
    typer.echo(f"bronze_read_errors: {result.bronze_files_read_errors}")
    typer.echo(f"quality_reports_total_files: {result.quality_reports_total_files}")
    typer.echo(f"symbol_master_parquet: {result.parquet_path}")
    typer.echo(f"symbol_master_csv: {result.csv_path}")


@app.command("sanity-checks")
def sanity_checks_cmd(
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run global Bronze QA sanity checks and write QA artifacts."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    symbol_master_parquet, _ = symbol_master_paths(settings.paths.bronze_root)
    symbol_master_df: pl.DataFrame | None = None

    if not symbol_master_parquet.exists():
        logger.info("sanity_checks.symbol_master_missing path=%s; building symbol master first", symbol_master_parquet)
        build_result = build_and_write_symbol_master(settings.paths.bronze_root, logger=logger)
        symbol_master_df = build_result.symbol_master_df

    sanity_result = run_bronze_sanity_checks(
        settings.paths.bronze_root,
        settings.paths.artifacts_root,
        symbol_master_df=symbol_master_df,
        logger=logger,
    )
    summary = sanity_result.summary

    logger.info(
        "sanity_checks.summary ticker_count=%s total_rows=%s total_warn_rows=%s total_invalid_rows=%s summary_path=%s",
        summary["ticker_count"],
        summary["total_rows"],
        summary["total_warn_rows"],
        summary["total_invalid_rows"],
        sanity_result.summary_path,
    )

    typer.echo(f"ticker_count: {summary['ticker_count']}")
    typer.echo(f"total_rows: {summary['total_rows']}")
    typer.echo(f"global_min_trade_date: {summary['global_min_trade_date']}")
    typer.echo(f"global_max_trade_date: {summary['global_max_trade_date']}")
    typer.echo(f"total_warn_rows: {summary['total_warn_rows']}")
    typer.echo(f"total_invalid_rows: {summary['total_invalid_rows']}")
    typer.echo("top_warn_tickers_top10:")
    for row in summary["top_tickers_by_warn_rows"][:10]:
        typer.echo(
            f"{row.get('ticker')} | exch={row.get('exchange')} | warn_rows={row.get('warn_row_count')} | rows={row.get('row_count')}"
        )
    typer.echo(f"summary_json: {sanity_result.summary_path}")
    typer.echo(f"by_exchange_parquet: {sanity_result.by_exchange_path}")
    typer.echo(f"rows_by_year_parquet: {sanity_result.rows_by_year_path}")


@app.command("list-problem-tickers")
def list_problem_tickers_cmd(
    limit: int = typer.Option(50, "--limit", min=1, help="Maximum number of tickers to show."),
    only_invalid: bool = typer.Option(False, "--only-invalid", help="Show only symbols with invalid rows > 0."),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """List tickers with warnings and/or invalid rows from symbol master."""

    settings, _ = _load_and_optionally_configure_logger(config_file, configure=False)
    symbol_master_parquet, _ = symbol_master_paths(settings.paths.bronze_root)
    if not symbol_master_parquet.exists():
        raise typer.BadParameter(
            f"Symbol master not found at {symbol_master_parquet}. Run build-symbol-master first."
        )

    df = pl.read_parquet(symbol_master_parquet)
    if only_invalid:
        problems = df.filter(pl.col("invalid_row_count") > 0)
    else:
        problems = df.filter((pl.col("invalid_row_count") > 0) | (pl.col("warn_row_count") > 0))

    if problems.height == 0:
        typer.echo("No problem tickers found.")
        return

    preview = (
        problems.select(["ticker", "exchange", "row_count", "warn_row_count", "invalid_row_count", "first_date", "last_date"])
        .sort(["invalid_row_count", "warn_row_count", "row_count"], descending=[True, True, True])
        .head(limit)
    )
    typer.echo(str(preview))


@app.command("discover-files")
def discover_files(
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Discover source text files and write a Bronze manifest parquet."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    manifest_path = settings.paths.bronze_root / "manifests" / "file_manifest_current.parquet"

    discovered = discover_txt_files(settings.paths.raw_root, logger=logger)
    manifest = build_manifest(discovered, logger=logger)
    write_manifest_parquet(
        manifest,
        manifest_path,
        compression=settings.parquet.compression,
        compression_level=settings.parquet.compression_level,
        statistics=settings.parquet.statistics,
    )

    logger.info("discover_files.total_files_found discovered=%s manifest_rows=%s", len(discovered), manifest.height)

    if manifest.height > 0:
        exchange_counts = (
            manifest.group_by("exchange")
            .len(name="file_count")
            .sort("exchange")
            .to_dicts()
        )
    else:
        exchange_counts = []
    logger.info("discover_files.exchange_counts %s", exchange_counts)
    logger.info("discover_files.preview_first_10\n%s", manifest.head(10))
    logger.info("discover_files.output_path %s", manifest_path)
    typer.echo(f"Manifest written: {manifest_path}")


@app.command("parse-sample")
def parse_sample(
    file: Path = typer.Option(
        ...,
        "--file",
        help="Path to one source TXT file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    exchange: str | None = typer.Option(
        None,
        "--exchange",
        help="Exchange label override (for example: NASDAQ, NYSE).",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Parse and normalize one source TXT file for Bronze readiness checks."""

    _, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    inferred_exchange = infer_exchange_from_path(file, logger=logger)
    selected_exchange = exchange.strip().upper() if exchange else inferred_exchange
    run_id = f"parse-sample-{uuid4().hex[:12]}"
    metadata = BronzeNormalizeMetadata.build(
        source_file=file,
        exchange=selected_exchange,
        run_id=run_id,
    )

    raw_result = read_stock_txt_with_rejects(file, logger=logger)
    normalized = normalize_bronze_rows(raw_result.data, metadata=metadata)

    min_trade_date: str | None = None
    max_trade_date: str | None = None
    if normalized.height > 0:
        date_bounds = normalized.select(
            [
                pl.col("trade_date").min().alias("min_trade_date"),
                pl.col("trade_date").max().alias("max_trade_date"),
            ]
        ).to_dicts()[0]
        min_value = date_bounds["min_trade_date"]
        max_value = date_bounds["max_trade_date"]
        min_trade_date = min_value.isoformat() if min_value is not None else None
        max_trade_date = max_value.isoformat() if max_value is not None else None

    logger.info(
        "parse_sample.summary file=%s exchange=%s raw_rows=%s normalized_rows=%s rejected_rows=%s header_skipped=%s delimiter=%s",
        file,
        selected_exchange,
        raw_result.data.height,
        normalized.height,
        raw_result.rejects.height,
        raw_result.skipped_header,
        raw_result.delimiter,
    )
    if raw_result.rejects.height > 0:
        logger.warning("parse_sample.reject_preview\n%s", raw_result.rejects.head(5))

    typer.echo(f"raw_row_count: {raw_result.data.height}")
    typer.echo(f"normalized_row_count: {normalized.height}")
    typer.echo(f"schema: {normalized.schema}")
    typer.echo("first_5_rows:")
    typer.echo(str(normalized.head(5)))
    typer.echo(f"min_trade_date: {min_trade_date}")
    typer.echo(f"max_trade_date: {max_trade_date}")


@app.command("validate-sample")
def validate_sample(
    file: Path = typer.Option(
        ...,
        "--file",
        help="Path to one source TXT file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    exchange: str | None = typer.Option(
        None,
        "--exchange",
        help="Exchange label override (for example: NASDAQ, NYSE).",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Parse, normalize, and validate one source TXT file."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    inferred_exchange = infer_exchange_from_path(file, logger=logger)
    selected_exchange = exchange.strip().upper() if exchange else inferred_exchange
    run_id = f"validate-sample-{uuid4().hex[:12]}"

    raw_result = read_stock_txt_with_rejects(file, logger=logger)
    metadata = BronzeNormalizeMetadata.build(
        source_file=file,
        exchange=selected_exchange,
        run_id=run_id,
    )
    normalized = normalize_bronze_rows(raw_result.data, metadata=metadata)

    thresholds = ValidationThresholds(
        suspicious_range_pct_threshold=settings.validation.suspicious_range_pct_threshold,
        suspicious_return_pct_threshold=settings.validation.suspicious_return_pct_threshold,
        gap_days_warn_threshold=settings.validation.gap_days_warn_threshold,
    )
    validation_result = validate_bronze_dataframe(
        normalized,
        thresholds=thresholds,
        header_skipped=raw_result.skipped_header,
        malformed_raw_rows_count=raw_result.rejects.height,
    )
    flag_counts = quality_flag_counts(validation_result.validated_df)

    date_bounds = validation_result.validated_df.select(
        [
            pl.col("trade_date").min().alias("min_trade_date"),
            pl.col("trade_date").max().alias("max_trade_date"),
        ]
    ).to_dicts()[0]
    min_trade_date = date_bounds["min_trade_date"]
    max_trade_date = date_bounds["max_trade_date"]

    logger.info(
        "validate_sample.summary file=%s exchange=%s raw_rows=%s normalized_rows=%s valid_rows=%s reject_rows=%s malformed_raw_rows=%s",
        file,
        selected_exchange,
        raw_result.data.height,
        normalized.height,
        validation_result.valid_rows.height,
        validation_result.reject_rows.height,
        raw_result.rejects.height,
    )
    logger.info("validate_sample.flag_counts %s", flag_counts)
    logger.info("validate_sample.report %s", validation_result.quality_report)

    typer.echo(f"raw_row_count: {raw_result.data.height}")
    typer.echo(f"normalized_row_count: {normalized.height}")
    typer.echo(f"valid_row_count: {validation_result.valid_rows.height}")
    typer.echo(f"reject_row_count: {validation_result.reject_rows.height}")
    typer.echo(f"quality_flag_counts: {flag_counts}")
    typer.echo(f"min_trade_date: {min_trade_date.isoformat() if min_trade_date is not None else None}")
    typer.echo(f"max_trade_date: {max_trade_date.isoformat() if max_trade_date is not None else None}")
    typer.echo("quality_report_preview:")
    typer.echo(json.dumps(validation_result.quality_report, indent=2, sort_keys=True, default=str))


@app.command("bronze-one")
def bronze_one(
    file: Path = typer.Option(
        ...,
        "--file",
        help="Path to one source TXT file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    exchange: str | None = typer.Option(
        None,
        "--exchange",
        help="Exchange label override (for example: NASDAQ, NYSE).",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run single-file Bronze parse, validate, and atomic artifact writes."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    inferred_exchange = infer_exchange_from_path(file, logger=logger)
    selected_exchange = exchange.strip().upper() if exchange else inferred_exchange
    run_id = f"bronze-one-{uuid4().hex[:12]}"

    try:
        raw_result = read_stock_txt_with_rejects(file, logger=logger)
        metadata = BronzeNormalizeMetadata.build(
            source_file=file,
            exchange=selected_exchange,
            run_id=run_id,
        )
        normalized = normalize_bronze_rows(raw_result.data, metadata=metadata)

        thresholds = ValidationThresholds(
            suspicious_range_pct_threshold=settings.validation.suspicious_range_pct_threshold,
            suspicious_return_pct_threshold=settings.validation.suspicious_return_pct_threshold,
            gap_days_warn_threshold=settings.validation.gap_days_warn_threshold,
        )
        validation_result = validate_bronze_dataframe(
            normalized,
            thresholds=thresholds,
            header_skipped=raw_result.skipped_header,
            malformed_raw_rows_count=raw_result.rejects.height,
        )

        fallback_ticker = extract_ticker_hint(file)
        write_result = write_bronze_artifacts(
            bronze_root=settings.paths.bronze_root,
            validation_result=validation_result,
            quality_report=validation_result.quality_report,
            fallback_ticker=fallback_ticker,
            fallback_exchange=selected_exchange,
            fallback_run_id=run_id,
            compression=settings.parquet.compression,
            compression_level=settings.parquet.compression_level,
            statistics=settings.parquet.statistics,
            malformed_rows=raw_result.rejects if raw_result.rejects.height > 0 else None,
        )
    except Exception as exc:
        logger.exception("bronze_one.failed file=%s exchange=%s", file, selected_exchange)
        raise RuntimeError(f"bronze-one failed for file {file}: {exc}") from exc

    logger.info(
        "bronze_one.summary file=%s ticker=%s raw_rows=%s valid_rows=%s reject_rows=%s bronze_path=%s rejects_path=%s quality_report_path=%s",
        file,
        write_result.ticker,
        raw_result.data.height,
        write_result.rows_valid,
        write_result.rows_invalid,
        write_result.bronze_path,
        write_result.rejects_path,
        write_result.quality_report_path,
    )

    typer.echo(f"file: {file}")
    typer.echo(f"ticker: {write_result.ticker}")
    typer.echo(f"raw_rows: {raw_result.data.height}")
    typer.echo(f"valid_rows: {write_result.rows_valid}")
    typer.echo(f"reject_rows: {write_result.rows_invalid}")
    typer.echo(f"bronze_parquet: {write_result.bronze_path}")
    typer.echo(f"rejects_parquet: {write_result.rejects_path if write_result.rejects_path else 'none'}")
    typer.echo(f"quality_report_json: {write_result.quality_report_path}")


@app.command("silver-one")
def silver_one(
    ticker: str | None = typer.Option(
        None,
        "--ticker",
        help="Ticker symbol to resolve from Bronze outputs.",
    ),
    bronze_file: Path | None = typer.Option(
        None,
        "--bronze-file",
        help="Direct path to a Bronze per-symbol parquet file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Build Silver base features for one symbol from Bronze parquet."""

    if (ticker is None and bronze_file is None) or (ticker is not None and bronze_file is not None):
        raise typer.BadParameter("Provide exactly one of --ticker or --bronze-file.")

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    source_bronze_file = bronze_file
    if source_bronze_file is None:
        source_bronze_file = resolve_bronze_file_for_ticker(settings.paths.bronze_root, ticker=ticker or "")

    run_id = f"silver-one-{uuid4().hex[:12]}"
    try:
        result = run_silver_one_from_bronze_file(
            source_bronze_file,
            settings,
            run_id=run_id,
            logger=logger,
        )
    except Exception as exc:
        logger.exception("silver_one.failed bronze_file=%s ticker=%s", source_bronze_file, ticker)
        raise RuntimeError(f"silver-one failed: {exc}") from exc

    logger.info(
        "silver_one.summary ticker=%s exchange=%s rows_in=%s rows_out=%s output=%s",
        result.ticker,
        result.exchange,
        result.rows_in,
        result.rows_out,
        result.silver_path,
    )
    typer.echo(f"ticker: {result.ticker}")
    typer.echo(f"exchange: {result.exchange}")
    typer.echo(f"rows_in: {result.rows_in}")
    typer.echo(f"rows_out: {result.rows_out}")
    typer.echo(f"min_trade_date: {result.min_trade_date.isoformat() if result.min_trade_date else None}")
    typer.echo(f"max_trade_date: {result.max_trade_date.isoformat() if result.max_trade_date else None}")
    typer.echo(f"output_path: {result.silver_path}")


@app.command("silver-run")
def silver_run(
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Process at most N symbols.",
    ),
    progress_every: int = typer.Option(
        100,
        "--progress-every",
        min=1,
        help="Log progress every N symbols.",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Reserved for future incremental mode; v1 still rebuilds selected symbols.",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run batch Silver base-series generation from Bronze valid outputs."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    options = SilverRunOptions(limit=limit, progress_every=progress_every, full=full)
    result = run_silver_pipeline(settings, options=options, logger=logger)

    summary = result.summary
    typer.echo(f"run_id: {summary['run_id']}")
    typer.echo(f"symbols_selected_total: {summary['symbols_selected_total']}")
    typer.echo(f"symbols_processed_success: {summary['symbols_processed_success']}")
    typer.echo(f"symbols_processed_failed: {summary['symbols_processed_failed']}")
    typer.echo(f"rows_in_total: {summary['rows_in_total']}")
    typer.echo(f"rows_out_total: {summary['rows_out_total']}")
    typer.echo(f"duration_sec: {summary['duration_sec']}")
    typer.echo(f"summary_path: {result.summary_path}")
    typer.echo(f"ticker_results_path: {result.ticker_results_path}")


@app.command("silver-sanity")
def silver_sanity(
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Scan Silver outputs and print compact sanity metrics."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    result = run_silver_sanity(settings.paths.silver_root, logger=logger)
    summary = result.summary

    logger.info(
        "silver_sanity.summary ticker_count=%s total_rows=%s min_date=%s max_date=%s read_errors=%s",
        summary["ticker_count"],
        summary["total_rows"],
        summary["global_min_trade_date"],
        summary["global_max_trade_date"],
        result.read_errors,
    )

    typer.echo(f"silver_file_count: {result.silver_file_count}")
    typer.echo(f"ticker_count: {summary['ticker_count']}")
    typer.echo(f"total_rows: {summary['total_rows']}")
    typer.echo(f"global_min_trade_date: {summary['global_min_trade_date']}")
    typer.echo(f"global_max_trade_date: {summary['global_max_trade_date']}")
    typer.echo(f"read_errors: {result.read_errors}")
    typer.echo("key_feature_null_rates:")
    for feature, rate in summary["key_feature_null_rates"].items():
        typer.echo(f"  {feature}: {rate}")
    typer.echo("feature_columns_present:")
    typer.echo(", ".join(summary["feature_columns_present"]))


@app.command("indicators-one")
def indicators_one(
    ticker: str | None = typer.Option(
        None,
        "--ticker",
        help="Ticker symbol to resolve from Silver base outputs.",
    ),
    silver_file: Path | None = typer.Option(
        None,
        "--silver-file",
        help="Direct path to a Silver base per-symbol parquet file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Build TMF and TTI-proxy indicators for one symbol."""

    if (ticker is None and silver_file is None) or (ticker is not None and silver_file is not None):
        raise typer.BadParameter("Provide exactly one of --ticker or --silver-file.")

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    source_silver_file = silver_file
    if source_silver_file is None:
        source_silver_file = resolve_silver_base_file_for_ticker(settings.paths.silver_root, ticker=ticker or "")

    run_id = f"indicators-one-{uuid4().hex[:12]}"
    try:
        result = run_indicators_one_from_silver_file(
            source_silver_file,
            settings,
            run_id=run_id,
            logger=logger,
        )
    except Exception as exc:
        logger.exception("indicators_one.failed silver_file=%s ticker=%s", source_silver_file, ticker)
        raise RuntimeError(f"indicators-one failed: {exc}") from exc

    logger.info(
        "indicators_one.summary ticker=%s exchange=%s rows_in=%s rows_out=%s output=%s",
        result.ticker,
        result.exchange,
        result.rows_in,
        result.rows_out,
        result.indicator_path,
    )
    typer.echo(f"ticker: {result.ticker}")
    typer.echo(f"exchange: {result.exchange}")
    typer.echo(f"rows_in: {result.rows_in}")
    typer.echo(f"rows_out: {result.rows_out}")
    typer.echo(f"min_trade_date: {result.min_trade_date.isoformat() if result.min_trade_date else None}")
    typer.echo(f"max_trade_date: {result.max_trade_date.isoformat() if result.max_trade_date else None}")
    typer.echo(f"output_path: {result.indicator_path}")


@app.command("indicators-run")
def indicators_run(
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Process at most N symbols.",
    ),
    progress_every: int = typer.Option(
        100,
        "--progress-every",
        min=1,
        help="Log progress every N symbols.",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Reserved for future incremental mode; v1 rebuilds selected symbols.",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run TMF + TTI-proxy indicators for selected Silver base symbols."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    options = IndicatorRunOptions(limit=limit, progress_every=progress_every, full=full)
    result = run_indicators_pipeline(settings, options=options, logger=logger)

    summary = result.summary
    typer.echo(f"run_id: {summary['run_id']}")
    typer.echo(f"symbols_total_selected: {summary['symbols_total_selected']}")
    typer.echo(f"symbols_success: {summary['symbols_success']}")
    typer.echo(f"symbols_failed: {summary['symbols_failed']}")
    typer.echo(f"rows_total: {summary['rows_total']}")
    typer.echo(f"duration_sec: {summary['duration_sec']}")
    typer.echo(f"summary_path: {result.summary_path}")
    typer.echo(f"ticker_results_path: {result.ticker_results_path}")


@app.command("indicators-sanity")
def indicators_sanity(
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Scan indicator outputs and report compact QA metrics."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    result = run_indicators_sanity(settings.paths.silver_root, settings.paths.artifacts_root, logger=logger)
    summary = result.summary

    logger.info(
        "indicators_sanity.summary symbol_count=%s total_rows=%s min_date=%s max_date=%s read_errors=%s summary_path=%s",
        summary["symbol_count"],
        summary["total_rows"],
        summary["global_min_trade_date"],
        summary["global_max_trade_date"],
        result.read_errors,
        result.summary_path,
    )

    typer.echo(f"indicator_file_count: {result.indicator_file_count}")
    typer.echo(f"symbol_count: {summary['symbol_count']}")
    typer.echo(f"total_rows: {summary['total_rows']}")
    typer.echo(f"global_min_trade_date: {summary['global_min_trade_date']}")
    typer.echo(f"global_max_trade_date: {summary['global_max_trade_date']}")
    typer.echo(f"tmf_21_null_rate: {summary['tmf_21_null_rate']}")
    typer.echo(f"tti_proxy_v1_21_null_rate: {summary['tti_proxy_v1_21_null_rate']}")
    typer.echo(f"tmf_zero_cross_up_count: {summary['tmf_zero_cross_up_count']}")
    typer.echo(f"tmf_zero_cross_down_count: {summary['tmf_zero_cross_down_count']}")
    typer.echo(f"tti_proxy_zero_cross_up_count: {summary['tti_proxy_zero_cross_up_count']}")
    typer.echo(f"tti_proxy_zero_cross_down_count: {summary['tti_proxy_zero_cross_down_count']}")
    typer.echo("top_20_symbols_by_max_abs_tmf_21:")
    for row in summary["top_20_symbols_by_max_abs_tmf_21"]:
        typer.echo(f"{row['ticker']} | max_abs_tmf_21={row['max_abs_tmf_21']}")
    typer.echo(f"summary_path: {result.summary_path}")


@app.command("events-one")
def events_one(
    ticker: str | None = typer.Option(
        None,
        "--ticker",
        help="Ticker symbol to resolve from indicator outputs.",
    ),
    indicator_file: Path | None = typer.Option(
        None,
        "--indicator-file",
        help="Direct path to one indicator parquet file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Build Gold Event Grammar v1 for one symbol."""

    if (ticker is None and indicator_file is None) or (ticker is not None and indicator_file is not None):
        raise typer.BadParameter("Provide exactly one of --ticker or --indicator-file.")

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    source_indicator_file = indicator_file
    if source_indicator_file is None:
        source_indicator_file = resolve_indicator_file_for_ticker(settings.paths.silver_root, ticker=ticker or "")

    run_id = f"events-one-{uuid4().hex[:12]}"
    try:
        result = run_events_one_from_indicator_file(
            source_indicator_file,
            settings,
            run_id=run_id,
            logger=logger,
        )
    except Exception as exc:
        logger.exception("events_one.failed indicator_file=%s ticker=%s", source_indicator_file, ticker)
        raise RuntimeError(f"events-one failed: {exc}") from exc

    logger.info(
        "events_one.summary ticker=%s exchange=%s rows_in=%s rows_out=%s output=%s",
        result.ticker,
        result.exchange,
        result.rows_in,
        result.rows_out,
        result.output_path,
    )
    typer.echo(f"ticker: {result.ticker}")
    typer.echo(f"exchange: {result.exchange}")
    typer.echo(f"rows_in: {result.rows_in}")
    typer.echo(f"rows_out: {result.rows_out}")
    typer.echo(f"min_trade_date: {result.min_trade_date.isoformat() if result.min_trade_date else None}")
    typer.echo(f"max_trade_date: {result.max_trade_date.isoformat() if result.max_trade_date else None}")
    typer.echo(f"output_path: {result.output_path}")


@app.command("events-run")
def events_run(
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Process at most N symbols.",
    ),
    progress_every: int = typer.Option(
        100,
        "--progress-every",
        min=1,
        help="Log progress every N symbols.",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Reserved for future incremental mode; v1 rebuilds selected symbols.",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run batch Gold Event Grammar v1 over indicator outputs."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    options = GoldEventRunOptions(limit=limit, progress_every=progress_every, full=full)
    result = run_events_pipeline(settings, options=options, logger=logger)

    summary = result.summary
    typer.echo(f"run_id: {summary['run_id']}")
    typer.echo(f"symbols_total_selected: {summary['symbols_total_selected']}")
    typer.echo(f"symbols_success: {summary['symbols_success']}")
    typer.echo(f"symbols_failed: {summary['symbols_failed']}")
    typer.echo(f"rows_total: {summary['rows_total']}")
    typer.echo(f"global_min_trade_date: {summary['global_min_trade_date']}")
    typer.echo(f"global_max_trade_date: {summary['global_max_trade_date']}")
    typer.echo(f"state_counts_global: {summary['state_counts_global']}")
    typer.echo(f"duration_sec: {summary['duration_sec']}")
    typer.echo(f"summary_path: {result.summary_path}")
    typer.echo(f"ticker_results_path: {result.ticker_results_path}")


@app.command("events-sanity")
def events_sanity(
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Scan Gold event outputs and print compact QA summary."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    result = run_events_sanity(settings.paths.gold_root, settings.paths.artifacts_root, logger=logger)
    summary = result.summary

    logger.info(
        "events_sanity.summary symbol_count=%s total_rows=%s min_date=%s max_date=%s read_errors=%s summary_path=%s",
        summary["symbol_count"],
        summary["total_rows"],
        summary["global_min_trade_date"],
        summary["global_max_trade_date"],
        result.read_errors,
        result.summary_path,
    )

    typer.echo(f"gold_event_file_count: {result.gold_event_file_count}")
    typer.echo(f"symbol_count: {summary['symbol_count']}")
    typer.echo(f"total_rows: {summary['total_rows']}")
    typer.echo(f"global_min_trade_date: {summary['global_min_trade_date']}")
    typer.echo(f"global_max_trade_date: {summary['global_max_trade_date']}")
    typer.echo(f"key_event_counts: {summary['key_event_counts']}")
    typer.echo(f"state_distribution: {summary['state_distribution']}")
    typer.echo("top_20_symbols_by_max_tmf_event_activity_20:")
    for row in summary["top_20_symbols_by_max_tmf_event_activity_20"]:
        typer.echo(f"{row['ticker']} | max_tmf_event_activity_20={row['max_tmf_event_activity_20']}")
    typer.echo("top_20_symbols_by_tmf_respect_fail_count:")
    for row in summary["top_20_symbols_by_tmf_respect_fail_count"]:
        typer.echo(f"{row['ticker']} | tmf_respect_fail_count={row['tmf_respect_fail_count']}")
    typer.echo(f"summary_path: {result.summary_path}")


@app.command("features-one")
def features_one(
    ticker: str | None = typer.Option(
        None,
        "--ticker",
        help="Ticker symbol to resolve from Gold event outputs.",
    ),
    events_file: Path | None = typer.Option(
        None,
        "--events-file",
        help="Direct path to one Gold event parquet file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Build Gold Features v1 for one symbol."""

    if (ticker is None and events_file is None) or (ticker is not None and events_file is not None):
        raise typer.BadParameter("Provide exactly one of --ticker or --events-file.")

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    source_events_file = events_file
    if source_events_file is None:
        source_events_file = resolve_events_file_for_ticker(settings.paths.gold_root, ticker=ticker or "")

    run_id = f"features-one-{uuid4().hex[:12]}"
    try:
        result = run_features_one_from_events_file(
            source_events_file,
            settings,
            run_id=run_id,
            logger=logger,
        )
    except Exception as exc:
        logger.exception("features_one.failed events_file=%s ticker=%s", source_events_file, ticker)
        raise RuntimeError(f"features-one failed: {exc}") from exc

    logger.info(
        "features_one.summary ticker=%s exchange=%s rows_in=%s rows_out=%s output=%s",
        result.ticker,
        result.exchange,
        result.rows_in,
        result.rows_out,
        result.output_path,
    )
    typer.echo(f"ticker: {result.ticker}")
    typer.echo(f"exchange: {result.exchange}")
    typer.echo(f"rows_in: {result.rows_in}")
    typer.echo(f"rows_out: {result.rows_out}")
    typer.echo(f"min_trade_date: {result.min_trade_date.isoformat() if result.min_trade_date else None}")
    typer.echo(f"max_trade_date: {result.max_trade_date.isoformat() if result.max_trade_date else None}")
    typer.echo(f"output_path: {result.output_path}")


@app.command("features-run")
def features_run(
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Process at most N symbols.",
    ),
    progress_every: int = typer.Option(
        100,
        "--progress-every",
        min=1,
        help="Log progress every N symbols.",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Reserved for future incremental mode; v1 rebuilds selected symbols.",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run batch Gold Features v1 over Gold event outputs."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    options = GoldFeatureRunOptions(limit=limit, progress_every=progress_every, full=full)
    result = run_features_pipeline(settings, options=options, logger=logger)

    summary = result.summary
    typer.echo(f"run_id: {summary['run_id']}")
    typer.echo(f"symbols_total_selected: {summary['symbols_total_selected']}")
    typer.echo(f"symbols_success: {summary['symbols_success']}")
    typer.echo(f"symbols_failed: {summary['symbols_failed']}")
    typer.echo(f"rows_total: {summary['rows_total']}")
    typer.echo(f"global_min_trade_date: {summary['global_min_trade_date']}")
    typer.echo(f"global_max_trade_date: {summary['global_max_trade_date']}")
    typer.echo(f"feature_calc_version: {summary['feature_calc_version']}")
    typer.echo(f"duration_sec: {summary['duration_sec']}")
    typer.echo(f"summary_path: {result.summary_path}")
    typer.echo(f"ticker_results_path: {result.ticker_results_path}")


@app.command("features-sanity")
def features_sanity(
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Scan Gold feature outputs and report compact QA metrics."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    result = run_features_sanity(settings.paths.gold_root, settings.paths.artifacts_root, logger=logger)
    summary = result.summary

    logger.info(
        "features_sanity.summary symbol_count=%s total_rows=%s min_date=%s max_date=%s read_errors=%s summary_path=%s",
        summary["symbol_count"],
        summary["total_rows"],
        summary["global_min_trade_date"],
        summary["global_max_trade_date"],
        result.read_errors,
        result.summary_path,
    )

    typer.echo(f"feature_file_count: {result.feature_file_count}")
    typer.echo(f"symbol_count: {summary['symbol_count']}")
    typer.echo(f"total_rows: {summary['total_rows']}")
    typer.echo(f"global_min_trade_date: {summary['global_min_trade_date']}")
    typer.echo(f"global_max_trade_date: {summary['global_max_trade_date']}")
    typer.echo(f"null_rates: {summary['null_rates']}")
    typer.echo("top_20_symbols_by_avg_flow_activity_20:")
    for row in summary["top_20_symbols_by_avg_flow_activity_20"]:
        typer.echo(f"{row['ticker']} | avg_flow_activity_20={row['avg_flow_activity_20']}")
    typer.echo("top_20_symbols_by_max_abs_delta_flow_20:")
    for row in summary["top_20_symbols_by_max_abs_delta_flow_20"]:
        typer.echo(f"{row['ticker']} | max_abs_delta_flow_20={row['max_abs_delta_flow_20']}")
    typer.echo("top_20_symbols_by_avg_oscillation_index_20:")
    for row in summary["top_20_symbols_by_avg_oscillation_index_20"]:
        typer.echo(f"{row['ticker']} | avg_oscillation_index_20={row['avg_oscillation_index_20']}")
    typer.echo(f"summary_path: {result.summary_path}")


@app.command("export-ml-dataset")
def export_ml_dataset_cmd(
    start_date: str | None = typer.Option(
        None,
        "--start-date",
        help="Optional start date filter YYYY-MM-DD.",
    ),
    end_date: str | None = typer.Option(
        None,
        "--end-date",
        help="Optional end date filter YYYY-MM-DD.",
    ),
    symbols_limit: int | None = typer.Option(
        None,
        "--symbols-limit",
        min=1,
        help="Optional max number of symbols to read.",
    ),
    sample_frac: float | None = typer.Option(
        None,
        "--sample-frac",
        help="Optional per-row sampling fraction in (0,1].",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Export stacked Gold feature dataset parquet + metadata for ML experiments."""

    parsed_start = _parse_iso_date(start_date, "start-date")
    parsed_end = _parse_iso_date(end_date, "end-date")
    if parsed_start is not None and parsed_end is not None and parsed_start > parsed_end:
        raise typer.BadParameter("start-date must be <= end-date.")

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    try:
        result = export_ml_dataset(
            settings,
            start_date=parsed_start,
            end_date=parsed_end,
            symbols_limit=symbols_limit,
            sample_frac=sample_frac,
            logger=logger,
        )
    except Exception as exc:
        logger.exception("export_ml_dataset.failed")
        raise RuntimeError(f"export-ml-dataset failed: {exc}") from exc

    logger.info(
        "export_ml_dataset.summary run_id=%s rows=%s symbols=%s dataset=%s metadata=%s",
        result.run_id,
        result.row_count,
        result.symbol_count,
        result.dataset_path,
        result.metadata_path,
    )
    typer.echo(f"run_id: {result.run_id}")
    typer.echo(f"row_count: {result.row_count}")
    typer.echo(f"symbol_count: {result.symbol_count}")
    typer.echo(f"dataset_path: {result.dataset_path}")
    typer.echo(f"metadata_path: {result.metadata_path}")


@app.command("research-cluster-run")
def research_cluster_run(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Path to exported ML dataset parquet.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    method: str = typer.Option(
        "kmeans",
        "--method",
        help="Clustering method: kmeans, gmm, hdbscan.",
    ),
    n_clusters: int = typer.Option(
        5,
        "--n-clusters",
        min=2,
        help="Number of clusters/components for kmeans/gmm.",
    ),
    sample_frac: float | None = typer.Option(
        None,
        "--sample-frac",
        help="Optional dataset sample fraction in (0,1].",
    ),
    date_from: str | None = typer.Option(
        None,
        "--date-from",
        help="Optional date lower bound YYYY-MM-DD.",
    ),
    date_to: str | None = typer.Option(
        None,
        "--date-to",
        help="Optional date upper bound YYYY-MM-DD.",
    ),
    split_mode: str = typer.Option(
        "none",
        "--split-mode",
        help="Split mode: none or time.",
    ),
    train_end: str | None = typer.Option(
        None,
        "--train-end",
        help="Train end date YYYY-MM-DD when split-mode=time.",
    ),
    test_start: str | None = typer.Option(
        None,
        "--test-start",
        help="Optional test start date YYYY-MM-DD (default: train-end + 1 day).",
    ),
    test_end: str | None = typer.Option(
        None,
        "--test-end",
        help="Optional test end date YYYY-MM-DD.",
    ),
    fit_on: str = typer.Option(
        "train",
        "--fit-on",
        help="Fit clustering model on train or all rows.",
    ),
    predict_on: str | None = typer.Option(
        None,
        "--predict-on",
        help="Predict/profile scope: test or all (default test when split-mode=time).",
    ),
    scaler: str | None = typer.Option(
        None,
        "--scaler",
        help="Optional scaler override: standard or robust.",
    ),
    scaling_scope: str | None = typer.Option(
        None,
        "--scaling-scope",
        help="Scaling scope override: global or per_ticker.",
    ),
    features_preset: str = typer.Option(
        "default",
        "--features-preset",
        help="Feature list preset (currently only 'default').",
    ),
    write_full_clustered: bool = typer.Option(
        False,
        "--write-full-clustered",
        help="Write full clustered dataset parquet in run output directory.",
    ),
    random_state: int = typer.Option(
        42,
        "--random-state",
        help="Random seed for deterministic clustering.",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run one unsupervised clustering baseline on exported Gold features dataset."""

    if features_preset.strip().lower() != "default":
        raise typer.BadParameter("Only features-preset=default is supported in v1.")
    method_norm = method.strip().lower()
    if method_norm not in {"kmeans", "gmm", "hdbscan"}:
        raise typer.BadParameter("method must be one of: kmeans, gmm, hdbscan")

    parsed_from = _parse_iso_date(date_from, "date-from")
    parsed_to = _parse_iso_date(date_to, "date-to")
    if parsed_from is not None and parsed_to is not None and parsed_from > parsed_to:
        raise typer.BadParameter("date-from must be <= date-to.")
    parsed_train_end = _parse_iso_date(train_end, "train-end")
    parsed_test_start = _parse_iso_date(test_start, "test-start")
    parsed_test_end = _parse_iso_date(test_end, "test-end")
    split_mode_norm = _normalize_choice(split_mode, allowed={"none", "time"}, option_name="split-mode")
    fit_on_norm = _normalize_choice(fit_on, allowed={"train", "all"}, option_name="fit-on")
    predict_on_norm = _normalize_choice(predict_on, allowed={"test", "all"}, option_name="predict-on")
    scaler_norm = _normalize_choice(scaler, allowed={"standard", "robust"}, option_name="scaler")
    scaling_scope_norm = _normalize_choice(
        scaling_scope,
        allowed={"global", "per_ticker"},
        option_name="scaling-scope",
    )
    if split_mode_norm == "time" and parsed_train_end is None:
        raise typer.BadParameter("train-end is required when split-mode=time.")
    if parsed_test_start is not None and parsed_test_end is not None and parsed_test_start > parsed_test_end:
        raise typer.BadParameter("test-start must be <= test-end.")

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    result = run_research_cluster(
        settings,
        dataset_path=dataset,
        method=method_norm,
        n_clusters=n_clusters,
        sample_frac=sample_frac,
        date_from=parsed_from,
        date_to=parsed_to,
        random_state=random_state,
        write_full_clustered=write_full_clustered,
        split_mode=split_mode_norm or "none",
        train_end=parsed_train_end,
        test_start=parsed_test_start,
        test_end=parsed_test_end,
        fit_on=fit_on_norm or "train",
        predict_on=predict_on_norm,
        scaler=scaler_norm,
        scaling_scope=scaling_scope_norm,
        logger=logger,
    )
    run_summary = json.loads(result.run_summary_path.read_text(encoding="utf-8"))
    metrics_payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    metrics = metrics_payload.get("metrics", {})

    typer.echo(f"run_id: {run_summary['run_id']}")
    typer.echo(f"rows_used: {run_summary['rows_used']}")
    typer.echo(f"features_count: {run_summary['features_count']}")
    typer.echo(f"method: {run_summary['method']}")
    typer.echo(f"n_clusters_requested: {run_summary['n_clusters_requested']}")
    typer.echo(f"split_mode: {run_summary.get('split_mode')}")
    typer.echo(f"fit_on: {run_summary.get('fit_on')}")
    typer.echo(f"predict_on: {run_summary.get('predict_on')}")
    typer.echo(f"scaler: {run_summary.get('scaler')}")
    typer.echo(f"scaling_scope: {run_summary.get('scaling_scope')}")
    typer.echo(f"silhouette: {metrics.get('silhouette')}")
    typer.echo(f"davies_bouldin: {metrics.get('davies_bouldin')}")
    typer.echo(f"calinski_harabasz: {metrics.get('calinski_harabasz')}")
    typer.echo(f"bic: {metrics.get('bic')}")
    typer.echo(f"aic: {metrics.get('aic')}")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"run_summary_path: {result.run_summary_path}")


@app.command("research-cluster-sweep")
def research_cluster_sweep(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Path to exported ML dataset parquet.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    methods: str = typer.Option(
        "kmeans,gmm",
        "--methods",
        help="Comma-separated methods from kmeans,gmm,hdbscan.",
    ),
    n_clusters_values: str = typer.Option(
        "4,5,6,8",
        "--n-clusters-values",
        help="Comma-separated cluster counts for kmeans/gmm.",
    ),
    sample_frac: float | None = typer.Option(
        None,
        "--sample-frac",
        help="Optional dataset sample fraction in (0,1].",
    ),
    date_from: str | None = typer.Option(
        None,
        "--date-from",
        help="Optional date lower bound YYYY-MM-DD.",
    ),
    date_to: str | None = typer.Option(
        None,
        "--date-to",
        help="Optional date upper bound YYYY-MM-DD.",
    ),
    split_mode: str = typer.Option(
        "none",
        "--split-mode",
        help="Split mode: none or time.",
    ),
    train_end: str | None = typer.Option(
        None,
        "--train-end",
        help="Train end date YYYY-MM-DD when split-mode=time.",
    ),
    test_start: str | None = typer.Option(
        None,
        "--test-start",
        help="Optional test start date YYYY-MM-DD (default: train-end + 1 day).",
    ),
    test_end: str | None = typer.Option(
        None,
        "--test-end",
        help="Optional test end date YYYY-MM-DD.",
    ),
    fit_on: str = typer.Option(
        "train",
        "--fit-on",
        help="Fit scope: train or all.",
    ),
    predict_on: str | None = typer.Option(
        None,
        "--predict-on",
        help="Predict/profile scope: test or all.",
    ),
    scaler: str | None = typer.Option(
        None,
        "--scaler",
        help="Optional scaler override: standard or robust.",
    ),
    scaling_scope: str | None = typer.Option(
        None,
        "--scaling-scope",
        help="Scaling scope override: global or per_ticker.",
    ),
    random_state: int = typer.Option(
        42,
        "--random-state",
        help="Random seed for deterministic clustering.",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run a clustering sweep and write cross-model comparison artifacts."""

    parsed_from = _parse_iso_date(date_from, "date-from")
    parsed_to = _parse_iso_date(date_to, "date-to")
    if parsed_from is not None and parsed_to is not None and parsed_from > parsed_to:
        raise typer.BadParameter("date-from must be <= date-to.")
    parsed_train_end = _parse_iso_date(train_end, "train-end")
    parsed_test_start = _parse_iso_date(test_start, "test-start")
    parsed_test_end = _parse_iso_date(test_end, "test-end")
    split_mode_norm = _normalize_choice(split_mode, allowed={"none", "time"}, option_name="split-mode")
    fit_on_norm = _normalize_choice(fit_on, allowed={"train", "all"}, option_name="fit-on")
    predict_on_norm = _normalize_choice(predict_on, allowed={"test", "all"}, option_name="predict-on")
    scaler_norm = _normalize_choice(scaler, allowed={"standard", "robust"}, option_name="scaler")
    scaling_scope_norm = _normalize_choice(
        scaling_scope,
        allowed={"global", "per_ticker"},
        option_name="scaling-scope",
    )
    if split_mode_norm == "time" and parsed_train_end is None:
        raise typer.BadParameter("train-end is required when split-mode=time.")
    if parsed_test_start is not None and parsed_test_end is not None and parsed_test_start > parsed_test_end:
        raise typer.BadParameter("test-start must be <= test-end.")

    methods_list = _parse_method_csv(methods)
    n_values = _parse_int_csv(n_clusters_values, "n-clusters-values")

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    result = run_research_cluster_sweep(
        settings,
        dataset_path=dataset,
        methods=methods_list,
        n_clusters_values=n_values,
        sample_frac=sample_frac,
        date_from=parsed_from,
        date_to=parsed_to,
        random_state=random_state,
        split_mode=split_mode_norm or "none",
        train_end=parsed_train_end,
        test_start=parsed_test_start,
        test_end=parsed_test_end,
        fit_on=fit_on_norm or "train",
        predict_on=predict_on_norm,
        scaler=scaler_norm,
        scaling_scope=scaling_scope_norm,
        logger=logger,
    )

    typer.echo(f"run_id: {result.run_id}")
    typer.echo(f"rows: {result.rows}")
    typer.echo(f"summary_json_path: {result.summary_json_path}")
    typer.echo(f"summary_csv_path: {result.summary_csv_path}")


@app.command("research-cluster-sanity")
def research_cluster_sanity(
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        help="Path to a research run output directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """Inspect one research run directory and print concise diagnostics."""

    summary = summarize_research_run(run_dir)
    run_summary = summary["run_summary"]
    metrics = summary["clustering_metrics"].get("metrics", {})
    preprocess = summary.get("preprocess_summary", {})
    split = summary.get("split_summary", {})
    robustness = summary.get("robustness_summary", {})

    typer.echo(f"run_id: {run_summary.get('run_id')}")
    typer.echo(f"method: {run_summary.get('method')}")
    typer.echo(f"rows_used: {run_summary.get('rows_used')}")
    typer.echo(f"features_count: {run_summary.get('features_count')}")
    typer.echo(f"silhouette: {metrics.get('silhouette')}")
    typer.echo(f"davies_bouldin: {metrics.get('davies_bouldin')}")
    typer.echo(f"calinski_harabasz: {metrics.get('calinski_harabasz')}")
    typer.echo(f"rows_dropped_null_features: {preprocess.get('rows_dropped_null_features')}")
    if split:
        typer.echo(f"split_mode: {split.get('split_mode')}")
        typer.echo(f"fit_on: {split.get('fit_on')}")
        typer.echo(f"predict_on: {split.get('predict_on')}")
        typer.echo(f"preprocess_scaling_scope: {split.get('preprocess_scaling_scope')}")
    if robustness:
        typer.echo(f"largest_cluster_share: {robustness.get('largest_cluster_share')}")
        typer.echo(f"forward_separation_score: {robustness.get('forward_separation_score')}")
    nan_summary = summary.get("forward_aggregate_nan_summary", {})
    nan_total = sum(item.get("nan_count", 0) for item in nan_summary.values())
    typer.echo(f"forward_aggregate_nan_total: {nan_total}")
    typer.echo("top_clusters_by_fwd_ret_10_mean:")
    for row in summary["top_clusters_by_fwd_ret_10_mean"]:
        typer.echo(
            f"cluster={row.get('cluster_id')} | rows={row.get('row_count')} | fwd_ret_10_mean={row.get('fwd_ret_10_mean')}"
        )
    typer.echo("top_clusters_by_flow_activity_20_mean:")
    for row in summary["top_clusters_by_flow_activity_20_mean"]:
        typer.echo(
            f"cluster={row.get('cluster_id')} | rows={row.get('row_count')} | flow_activity_20_mean={row.get('flow_activity_20_mean')}"
        )


@app.command("research-cluster-stability")
def research_cluster_stability(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Path to exported ML dataset parquet.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    method: str = typer.Option(
        "kmeans",
        "--method",
        help="Stability method: kmeans or gmm.",
    ),
    n_clusters: int = typer.Option(
        5,
        "--n-clusters",
        min=2,
        help="Number of clusters/components for kmeans/gmm.",
    ),
    seeds: int | None = typer.Option(
        None,
        "--seeds",
        min=1,
        help="Number of random seeds to evaluate.",
    ),
    seed_start: int | None = typer.Option(
        None,
        "--seed-start",
        help="Starting random seed (sweep uses seed_start..seed_start+seeds-1).",
    ),
    sample_frac: float | None = typer.Option(
        None,
        "--sample-frac",
        help="Optional dataset sample fraction in (0,1].",
    ),
    date_from: str | None = typer.Option(
        None,
        "--date-from",
        help="Optional date lower bound YYYY-MM-DD.",
    ),
    date_to: str | None = typer.Option(
        None,
        "--date-to",
        help="Optional date upper bound YYYY-MM-DD.",
    ),
    scaler: str | None = typer.Option(
        None,
        "--scaler",
        help="Optional scaler override: standard or robust.",
    ),
    scaling_scope: str | None = typer.Option(
        None,
        "--scaling-scope",
        help="Scaling scope override: global or per_ticker.",
    ),
    split_mode: str = typer.Option(
        "none",
        "--split-mode",
        help="Split mode: none or time.",
    ),
    train_end: str | None = typer.Option(
        None,
        "--train-end",
        help="Train end date YYYY-MM-DD when split-mode=time.",
    ),
    test_start: str | None = typer.Option(
        None,
        "--test-start",
        help="Optional test start date YYYY-MM-DD (default: train-end + 1 day).",
    ),
    test_end: str | None = typer.Option(
        None,
        "--test-end",
        help="Optional test end date YYYY-MM-DD.",
    ),
    fit_on: str = typer.Option(
        "train",
        "--fit-on",
        help="Fit scope: train or all.",
    ),
    predict_on: str | None = typer.Option(
        None,
        "--predict-on",
        help="Predict/profile scope: test or all.",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run cluster stability sweep across random seeds and emit ARI artifacts."""

    method_norm = _normalize_choice(method, allowed={"kmeans", "gmm"}, option_name="method")
    parsed_from = _parse_iso_date(date_from, "date-from")
    parsed_to = _parse_iso_date(date_to, "date-to")
    if parsed_from is not None and parsed_to is not None and parsed_from > parsed_to:
        raise typer.BadParameter("date-from must be <= date-to.")
    parsed_train_end = _parse_iso_date(train_end, "train-end")
    parsed_test_start = _parse_iso_date(test_start, "test-start")
    parsed_test_end = _parse_iso_date(test_end, "test-end")
    if parsed_test_start is not None and parsed_test_end is not None and parsed_test_start > parsed_test_end:
        raise typer.BadParameter("test-start must be <= test-end.")

    split_mode_norm = _normalize_choice(split_mode, allowed={"none", "time"}, option_name="split-mode")
    fit_on_norm = _normalize_choice(fit_on, allowed={"train", "all"}, option_name="fit-on")
    predict_on_norm = _normalize_choice(predict_on, allowed={"test", "all"}, option_name="predict-on")
    scaler_norm = _normalize_choice(scaler, allowed={"standard", "robust"}, option_name="scaler")
    scaling_scope_norm = _normalize_choice(
        scaling_scope,
        allowed={"global", "per_ticker"},
        option_name="scaling-scope",
    )
    if split_mode_norm == "time" and parsed_train_end is None:
        raise typer.BadParameter("train-end is required when split-mode=time.")

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    resolved_seeds = seeds if seeds is not None else settings.research_clustering.stability.seeds_default
    resolved_seed_start = (
        seed_start if seed_start is not None else settings.research_clustering.stability.seed_start_default
    )
    result = run_research_cluster_stability(
        settings,
        dataset_path=dataset,
        method=method_norm or "kmeans",
        n_clusters=n_clusters,
        seeds=resolved_seeds,
        seed_start=resolved_seed_start,
        sample_frac=sample_frac,
        date_from=parsed_from,
        date_to=parsed_to,
        split_mode=split_mode_norm or "none",
        train_end=parsed_train_end,
        test_start=parsed_test_start,
        test_end=parsed_test_end,
        fit_on=fit_on_norm or "train",
        predict_on=predict_on_norm,
        scaler=scaler_norm,
        scaling_scope=scaling_scope_norm,
        logger=logger,
    )

    summary = json.loads(result.stability_summary_path.read_text(encoding="utf-8"))
    typer.echo(f"run_id: {summary.get('run_id')}")
    typer.echo(f"method: {summary.get('method')}")
    typer.echo(f"n_clusters: {summary.get('n_clusters')}")
    typer.echo(f"seeds: {summary.get('seeds')}")
    typer.echo(f"split_mode: {summary.get('split_mode')}")
    typer.echo(f"fit_on: {summary.get('fit_on')}")
    typer.echo(f"predict_on: {summary.get('predict_on')}")
    typer.echo(f"scaler: {summary.get('scaler')}")
    typer.echo(f"scaling_scope: {summary.get('scaling_scope')}")
    ari = summary.get("ari_summary", {})
    typer.echo(f"ari_mean: {ari.get('ari_mean')}")
    typer.echo(f"ari_median: {ari.get('ari_median')}")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"stability_summary_path: {result.stability_summary_path}")
    typer.echo(f"stability_by_seed_path: {result.stability_by_seed_path}")
    typer.echo(f"pairwise_ari_path: {result.pairwise_ari_path}")


@app.command("research-hmm-run")
def research_hmm_run(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Path to exported ML dataset parquet.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    n_components: int | None = typer.Option(
        None,
        "--n-components",
        min=2,
        help="Gaussian HMM component count (defaults to config).",
    ),
    covariance_type: str | None = typer.Option(
        None,
        "--covariance-type",
        help="Optional covariance type override: diag, full, tied, spherical.",
    ),
    n_iter: int | None = typer.Option(
        None,
        "--n-iter",
        min=1,
        help="Optional HMM max iterations override.",
    ),
    tol: float | None = typer.Option(
        None,
        "--tol",
        help="Optional HMM convergence tolerance override.",
    ),
    random_state: int | None = typer.Option(
        None,
        "--random-state",
        help="Optional random seed override.",
    ),
    sample_frac: float | None = typer.Option(
        None,
        "--sample-frac",
        help="Optional dataset sample fraction in (0,1].",
    ),
    date_from: str | None = typer.Option(
        None,
        "--date-from",
        help="Optional date lower bound YYYY-MM-DD.",
    ),
    date_to: str | None = typer.Option(
        None,
        "--date-to",
        help="Optional date upper bound YYYY-MM-DD.",
    ),
    split_mode: str | None = typer.Option(
        None,
        "--split-mode",
        help="Split mode override: none or time.",
    ),
    train_end: str | None = typer.Option(
        None,
        "--train-end",
        help="Train end date YYYY-MM-DD when split-mode=time.",
    ),
    test_start: str | None = typer.Option(
        None,
        "--test-start",
        help="Optional test start date YYYY-MM-DD.",
    ),
    test_end: str | None = typer.Option(
        None,
        "--test-end",
        help="Optional test end date YYYY-MM-DD.",
    ),
    fit_on: str = typer.Option(
        "train",
        "--fit-on",
        help="Fit scope: train or all.",
    ),
    predict_on: str | None = typer.Option(
        None,
        "--predict-on",
        help="Predict/decode scope: train, test, or all.",
    ),
    scaler: str | None = typer.Option(
        None,
        "--scaler",
        help="Optional scaler override: standard or robust.",
    ),
    scaling_scope: str | None = typer.Option(
        None,
        "--scaling-scope",
        help="Scaling scope override: global or per_ticker.",
    ),
    cluster_labels_file: Path | None = typer.Option(
        None,
        "--cluster-labels-file",
        help="Optional cluster labels parquet/csv for overlap metrics.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    write_full_decoded_csv: bool = typer.Option(
        False,
        "--write-full-decoded-csv",
        help="Also write full decoded rows CSV (can be large).",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run Gaussian HMM baseline with optional OOS split and diagnostics artifacts."""

    parsed_from = _parse_iso_date(date_from, "date-from")
    parsed_to = _parse_iso_date(date_to, "date-to")
    if parsed_from is not None and parsed_to is not None and parsed_from > parsed_to:
        raise typer.BadParameter("date-from must be <= date-to.")
    parsed_train_end = _parse_iso_date(train_end, "train-end")
    parsed_test_start = _parse_iso_date(test_start, "test-start")
    parsed_test_end = _parse_iso_date(test_end, "test-end")
    if parsed_test_start is not None and parsed_test_end is not None and parsed_test_start > parsed_test_end:
        raise typer.BadParameter("test-start must be <= test-end.")

    split_mode_norm = _normalize_choice(split_mode, allowed={"none", "time"}, option_name="split-mode")
    fit_on_norm = _normalize_choice(fit_on, allowed={"train", "all"}, option_name="fit-on")
    predict_on_norm = _normalize_choice(predict_on, allowed={"train", "test", "all"}, option_name="predict-on")
    scaler_norm = _normalize_choice(scaler, allowed={"standard", "robust"}, option_name="scaler")
    scaling_scope_norm = _normalize_choice(
        scaling_scope,
        allowed={"global", "per_ticker"},
        option_name="scaling-scope",
    )
    covariance_type_norm = _normalize_choice(
        covariance_type,
        allowed={"diag", "full", "tied", "spherical"},
        option_name="covariance-type",
    )
    if split_mode_norm == "time" and parsed_train_end is None:
        raise typer.BadParameter("train-end is required when split-mode=time.")

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    resolved_components = (
        n_components if n_components is not None else settings.research_hmm.hmm.n_components_default
    )
    result = run_hmm_baseline(
        settings,
        dataset_path=dataset,
        n_components=resolved_components,
        covariance_type=covariance_type_norm,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        sample_frac=sample_frac,
        date_from=parsed_from,
        date_to=parsed_to,
        split_mode=split_mode_norm,
        train_end=parsed_train_end,
        test_start=parsed_test_start,
        test_end=parsed_test_end,
        fit_on=fit_on_norm or "train",
        predict_on=predict_on_norm,
        scaler=scaler_norm,
        scaling_scope=scaling_scope_norm,
        cluster_labels_file=cluster_labels_file,
        write_full_decoded_csv=write_full_decoded_csv,
        logger=logger,
    )
    summary = json.loads(result.run_summary_path.read_text(encoding="utf-8"))
    typer.echo(f"run_id: {summary.get('run_id')}")
    typer.echo(f"rows_decoded: {summary.get('rows_decoded')}")
    typer.echo(f"n_components: {summary.get('n_components')}")
    typer.echo(f"split_mode: {summary.get('split_mode')}")
    typer.echo(f"fit_on: {summary.get('fit_on')}")
    typer.echo(f"predict_on: {summary.get('predict_on')}")
    typer.echo(f"scaler: {summary.get('scaler')}")
    typer.echo(f"scaling_scope: {summary.get('scaling_scope')}")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"run_summary_path: {result.run_summary_path}")


@app.command("research-hmm-sweep")
def research_hmm_sweep(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Path to exported ML dataset parquet.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    components: str | None = typer.Option(
        None,
        "--components",
        help="Comma-separated component counts, e.g. 4,5,6,8. Defaults to config.",
    ),
    sample_frac: float | None = typer.Option(
        None,
        "--sample-frac",
        help="Optional dataset sample fraction in (0,1].",
    ),
    date_from: str | None = typer.Option(
        None,
        "--date-from",
        help="Optional date lower bound YYYY-MM-DD.",
    ),
    date_to: str | None = typer.Option(
        None,
        "--date-to",
        help="Optional date upper bound YYYY-MM-DD.",
    ),
    split_mode: str | None = typer.Option(
        None,
        "--split-mode",
        help="Split mode override: none or time.",
    ),
    train_end: str | None = typer.Option(
        None,
        "--train-end",
        help="Train end date YYYY-MM-DD when split-mode=time.",
    ),
    test_start: str | None = typer.Option(
        None,
        "--test-start",
        help="Optional test start date YYYY-MM-DD.",
    ),
    test_end: str | None = typer.Option(
        None,
        "--test-end",
        help="Optional test end date YYYY-MM-DD.",
    ),
    fit_on: str = typer.Option(
        "train",
        "--fit-on",
        help="Fit scope: train or all.",
    ),
    predict_on: str | None = typer.Option(
        None,
        "--predict-on",
        help="Predict/decode scope: train, test, or all.",
    ),
    scaler: str | None = typer.Option(
        None,
        "--scaler",
        help="Optional scaler override: standard or robust.",
    ),
    scaling_scope: str | None = typer.Option(
        None,
        "--scaling-scope",
        help="Scaling scope override: global or per_ticker.",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run HMM across multiple component counts and write summary artifacts."""

    parsed_from = _parse_iso_date(date_from, "date-from")
    parsed_to = _parse_iso_date(date_to, "date-to")
    if parsed_from is not None and parsed_to is not None and parsed_from > parsed_to:
        raise typer.BadParameter("date-from must be <= date-to.")
    parsed_train_end = _parse_iso_date(train_end, "train-end")
    parsed_test_start = _parse_iso_date(test_start, "test-start")
    parsed_test_end = _parse_iso_date(test_end, "test-end")
    if parsed_test_start is not None and parsed_test_end is not None and parsed_test_start > parsed_test_end:
        raise typer.BadParameter("test-start must be <= test-end.")

    split_mode_norm = _normalize_choice(split_mode, allowed={"none", "time"}, option_name="split-mode")
    fit_on_norm = _normalize_choice(fit_on, allowed={"train", "all"}, option_name="fit-on")
    predict_on_norm = _normalize_choice(predict_on, allowed={"train", "test", "all"}, option_name="predict-on")
    scaler_norm = _normalize_choice(scaler, allowed={"standard", "robust"}, option_name="scaler")
    scaling_scope_norm = _normalize_choice(
        scaling_scope,
        allowed={"global", "per_ticker"},
        option_name="scaling-scope",
    )
    if split_mode_norm == "time" and parsed_train_end is None:
        raise typer.BadParameter("train-end is required when split-mode=time.")

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    if components is None:
        component_values = settings.research_hmm.sweep.components_default
    else:
        component_values = _parse_int_csv(components, "components")
    result = run_hmm_sweep(
        settings,
        dataset_path=dataset,
        components=component_values,
        sample_frac=sample_frac,
        date_from=parsed_from,
        date_to=parsed_to,
        split_mode=split_mode_norm,
        train_end=parsed_train_end,
        test_start=parsed_test_start,
        test_end=parsed_test_end,
        fit_on=fit_on_norm or "train",
        predict_on=predict_on_norm,
        scaler=scaler_norm,
        scaling_scope=scaling_scope_norm,
        logger=logger,
    )
    typer.echo(f"run_id: {result.run_id}")
    typer.echo(f"rows: {result.rows}")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"summary_json_path: {result.summary_json_path}")
    typer.echo(f"summary_csv_path: {result.summary_csv_path}")


@app.command("research-hmm-sanity")
def research_hmm_sanity(
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        help="Path to one HMM run output directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """Inspect a completed HMM run and print concise diagnostics."""

    summary = summarize_hmm_run(run_dir)
    run_summary = summary["run_summary"]
    typer.echo(f"run_id: {run_summary.get('run_id')}")
    typer.echo(f"rows_decoded: {run_summary.get('rows_decoded')}")
    typer.echo(f"n_components: {run_summary.get('n_components')}")
    typer.echo(f"split_mode: {run_summary.get('split_mode')}")
    typer.echo(f"state_count: {summary.get('state_count')}")
    nan_summary = summary.get("forward_aggregate_nan_summary", {})
    nan_total = sum(item.get("nan_count", 0) for item in nan_summary.values())
    typer.echo(f"forward_aggregate_nan_total: {nan_total}")
    typer.echo("top_states_by_fwd_ret_10_mean:")
    for row in summary.get("top_states_by_fwd_ret_10_mean", [])[:10]:
        typer.echo(
            f"state={row.get('hmm_state')} | rows={row.get('row_count')} | fwd_ret_10_mean={row.get('fwd_ret_10_mean')}"
        )
    typer.echo("top_self_transition_probs:")
    for row in summary.get("top_self_transition_probs", [])[:10]:
        typer.echo(f"state={row.get('state')} | self_prob={row.get('transition_probability')}")


@app.command("research-hmm-stability")
def research_hmm_stability(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Path to exported ML dataset parquet.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    n_components: int | None = typer.Option(
        None,
        "--n-components",
        min=2,
        help="Gaussian HMM component count (defaults to config).",
    ),
    seeds: int | None = typer.Option(
        None,
        "--seeds",
        min=1,
        help="Number of seeds to evaluate (defaults to config).",
    ),
    seed_start: int | None = typer.Option(
        None,
        "--seed-start",
        help="Seed range starting value (defaults to config).",
    ),
    sample_frac: float | None = typer.Option(
        None,
        "--sample-frac",
        help="Optional dataset sample fraction in (0,1].",
    ),
    date_from: str | None = typer.Option(
        None,
        "--date-from",
        help="Optional date lower bound YYYY-MM-DD.",
    ),
    date_to: str | None = typer.Option(
        None,
        "--date-to",
        help="Optional date upper bound YYYY-MM-DD.",
    ),
    split_mode: str | None = typer.Option(
        None,
        "--split-mode",
        help="Split mode override: none or time.",
    ),
    train_end: str | None = typer.Option(
        None,
        "--train-end",
        help="Train end date YYYY-MM-DD when split-mode=time.",
    ),
    test_start: str | None = typer.Option(
        None,
        "--test-start",
        help="Optional test start date YYYY-MM-DD.",
    ),
    test_end: str | None = typer.Option(
        None,
        "--test-end",
        help="Optional test end date YYYY-MM-DD.",
    ),
    fit_on: str = typer.Option(
        "train",
        "--fit-on",
        help="Fit scope: train or all.",
    ),
    predict_on: str | None = typer.Option(
        None,
        "--predict-on",
        help="Predict/decode scope: train, test, or all.",
    ),
    scaler: str | None = typer.Option(
        None,
        "--scaler",
        help="Optional scaler override: standard or robust.",
    ),
    scaling_scope: str | None = typer.Option(
        None,
        "--scaling-scope",
        help="Scaling scope override: global or per_ticker.",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run HMM across multiple seeds and write ARI-based stability artifacts."""

    parsed_from = _parse_iso_date(date_from, "date-from")
    parsed_to = _parse_iso_date(date_to, "date-to")
    if parsed_from is not None and parsed_to is not None and parsed_from > parsed_to:
        raise typer.BadParameter("date-from must be <= date-to.")
    parsed_train_end = _parse_iso_date(train_end, "train-end")
    parsed_test_start = _parse_iso_date(test_start, "test-start")
    parsed_test_end = _parse_iso_date(test_end, "test-end")
    if parsed_test_start is not None and parsed_test_end is not None and parsed_test_start > parsed_test_end:
        raise typer.BadParameter("test-start must be <= test-end.")

    split_mode_norm = _normalize_choice(split_mode, allowed={"none", "time"}, option_name="split-mode")
    fit_on_norm = _normalize_choice(fit_on, allowed={"train", "all"}, option_name="fit-on")
    predict_on_norm = _normalize_choice(predict_on, allowed={"train", "test", "all"}, option_name="predict-on")
    scaler_norm = _normalize_choice(scaler, allowed={"standard", "robust"}, option_name="scaler")
    scaling_scope_norm = _normalize_choice(
        scaling_scope,
        allowed={"global", "per_ticker"},
        option_name="scaling-scope",
    )
    if split_mode_norm == "time" and parsed_train_end is None:
        raise typer.BadParameter("train-end is required when split-mode=time.")

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    resolved_components = (
        n_components if n_components is not None else settings.research_hmm.hmm.n_components_default
    )
    resolved_seeds = seeds if seeds is not None else settings.research_hmm.stability.seeds_default
    resolved_seed_start = (
        seed_start if seed_start is not None else settings.research_hmm.stability.seed_start_default
    )
    result = run_hmm_stability(
        settings,
        dataset_path=dataset,
        n_components=resolved_components,
        seeds=resolved_seeds,
        seed_start=resolved_seed_start,
        sample_frac=sample_frac,
        date_from=parsed_from,
        date_to=parsed_to,
        split_mode=split_mode_norm,
        train_end=parsed_train_end,
        test_start=parsed_test_start,
        test_end=parsed_test_end,
        fit_on=fit_on_norm or "train",
        predict_on=predict_on_norm,
        scaler=scaler_norm,
        scaling_scope=scaling_scope_norm,
        logger=logger,
    )
    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    ari = summary.get("ari_summary", {})
    typer.echo(f"run_id: {summary.get('run_id')}")
    typer.echo(f"n_components: {summary.get('n_components')}")
    typer.echo(f"seeds: {summary.get('seeds')}")
    typer.echo(f"split_mode: {summary.get('split_mode')}")
    typer.echo(f"fit_on: {summary.get('fit_on')}")
    typer.echo(f"predict_on: {summary.get('predict_on')}")
    typer.echo(f"ari_mean: {ari.get('ari_mean')}")
    typer.echo(f"ari_median: {ari.get('ari_median')}")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"summary_json_path: {result.summary_json_path}")
    typer.echo(f"by_seed_path: {result.by_seed_path}")
    typer.echo(f"pairwise_ari_path: {result.pairwise_ari_path}")


@app.command("validation-run")
def validation_run(
    input_file: Path = typer.Option(
        ...,
        "--input-file",
        help="Input parquet/csv file (HMM decoded rows, clustered rows, or generic state-labeled rows).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    input_type: str = typer.Option(
        "auto",
        "--input-type",
        help="Input type: auto, hmm, cluster, generic.",
    ),
    state_col: str | None = typer.Option(
        None,
        "--state-col",
        help="State column name for generic input type.",
    ),
    bootstrap_n: int | None = typer.Option(
        None,
        "--bootstrap-n",
        min=10,
        help="Bootstrap iterations override.",
    ),
    bootstrap_ci: float | None = typer.Option(
        None,
        "--bootstrap-ci",
        help="Bootstrap confidence level override (0,1).",
    ),
    bootstrap_mode: str | None = typer.Option(
        None,
        "--bootstrap-mode",
        help="Bootstrap mode override: iid or block.",
    ),
    block_length: int | None = typer.Option(
        None,
        "--block-length",
        min=1,
        help="Block length for block bootstrap mode.",
    ),
    event_window_pre: int | None = typer.Option(
        None,
        "--event-window-pre",
        min=1,
        help="Transition event-study pre-window bars.",
    ),
    event_window_post: int | None = typer.Option(
        None,
        "--event-window-post",
        min=1,
        help="Transition event-study post-window bars.",
    ),
    min_events_per_transition: int | None = typer.Option(
        None,
        "--min-events-per-transition",
        min=1,
        help="Minimum events per transition code to include in transition summary.",
    ),
    window_months: int | None = typer.Option(
        None,
        "--window-months",
        min=1,
        help="Rolling stability window size in months.",
    ),
    step_months: int | None = typer.Option(
        None,
        "--step-months",
        min=1,
        help="Rolling stability step size in months.",
    ),
    write_large_artifacts: bool = typer.Option(
        False,
        "--write-large-artifacts",
        help="Write optional large artifacts (transition events, adapted sample).",
    ),
    sample_frac: float | None = typer.Option(
        None,
        "--sample-frac",
        help="Optional row sample fraction in (0,1] for quick tests.",
    ),
    date_from: str | None = typer.Option(
        None,
        "--date-from",
        help="Optional date lower bound YYYY-MM-DD.",
    ),
    date_to: str | None = typer.Option(
        None,
        "--date-to",
        help="Optional date upper bound YYYY-MM-DD.",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run Validation Harness v1 on state-labeled research outputs."""

    parsed_from = _parse_iso_date(date_from, "date-from")
    parsed_to = _parse_iso_date(date_to, "date-to")
    if parsed_from is not None and parsed_to is not None and parsed_from > parsed_to:
        raise typer.BadParameter("date-from must be <= date-to.")

    input_type_norm = _normalize_choice(
        input_type,
        allowed={"auto", "hmm", "cluster", "generic"},
        option_name="input-type",
    )
    bootstrap_mode_norm = _normalize_choice(
        bootstrap_mode,
        allowed={"iid", "block"},
        option_name="bootstrap-mode",
    )
    if bootstrap_ci is not None and not 0.0 < bootstrap_ci < 1.0:
        raise typer.BadParameter("bootstrap-ci must be in (0,1).")
    if input_type_norm == "generic" and (state_col is None or state_col.strip() == ""):
        raise typer.BadParameter("state-col is required when input-type=generic.")

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    result = run_validation_harness(
        settings,
        input_file=input_file,
        input_type=(input_type_norm or "auto"),
        state_col=state_col,
        bootstrap_n=bootstrap_n,
        bootstrap_ci=bootstrap_ci,
        bootstrap_mode=bootstrap_mode_norm,
        block_length=block_length,
        event_window_pre=event_window_pre,
        event_window_post=event_window_post,
        min_events_per_transition=min_events_per_transition,
        window_months=window_months,
        step_months=step_months,
        write_large_artifacts=write_large_artifacts,
        sample_frac=sample_frac,
        date_from=parsed_from,
        date_to=parsed_to,
        logger=logger,
    )
    summary = json.loads(result.run_summary_path.read_text(encoding="utf-8"))
    typer.echo(f"run_id: {summary.get('run_id')}")
    typer.echo(f"rows: {summary.get('rows')}")
    typer.echo(f"state_count: {summary.get('state_count')}")
    typer.echo(f"ticker_count: {summary.get('ticker_count')}")
    bounds = summary.get("bounds", {})
    typer.echo(f"min_trade_date: {bounds.get('min_trade_date')}")
    typer.echo(f"max_trade_date: {bounds.get('max_trade_date')}")
    typer.echo(
        f"transition_summary_rows: {summary.get('event_study', {}).get('transition_summary_rows')}"
    )
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"run_summary_path: {result.run_summary_path}")
    typer.echo(f"validation_scorecard_path: {result.validation_scorecard_path}")


@app.command("validation-sanity")
def validation_sanity(
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        help="Path to a completed validation run directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """Print concise sanity diagnostics from a completed validation run."""

    summary = summarize_validation_run(run_dir)
    run_summary = summary["run_summary"]
    typer.echo(f"run_id: {run_summary.get('run_id')}")
    typer.echo(f"input_type: {run_summary.get('input_type')}")
    typer.echo(f"rows: {run_summary.get('rows')}")
    typer.echo(f"state_count: {run_summary.get('state_count')}")
    typer.echo(f"validation_grade: {summary.get('validation_grade')}")
    typer.echo(f"pairwise_significant_diff_share: {summary.get('pairwise_significant_diff_share')}")
    nan_warnings = summary.get("nan_warnings", {})
    nan_total = sum(int(value) for value in nan_warnings.values())
    typer.echo(f"nan_warning_total: {nan_total}")
    typer.echo("top_states_by_fwd_ret_10_mean:")
    for row in summary.get("top_states_by_fwd_ret_10_mean", [])[:10]:
        typer.echo(
            f"state={row.get('state_id')} | n={row.get('n_rows')} | "
            f"mean={row.get('fwd_ret_10_mean')} | ci=[{row.get('fwd_ret_10_ci_lo')}, {row.get('fwd_ret_10_ci_hi')}]"
        )
    typer.echo("top_transition_codes:")
    for row in summary.get("top_transition_codes", [])[:10]:
        typer.echo(
            f"code={row.get('transition_code')} | count={row.get('count_events')} | "
            f"fwd_ret_10_mean={row.get('fwd_ret_10_mean')}"
        )


@app.command("validation-compare")
def validation_compare(
    run_dir_a: Path = typer.Option(
        ...,
        "--run-dir-a",
        help="First validation run directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    run_dir_b: Path = typer.Option(
        ...,
        "--run-dir-b",
        help="Second validation run directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Compare two validation runs and write comparison artifacts."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    result = run_validation_compare(
        settings,
        run_dir_a=run_dir_a,
        run_dir_b=run_dir_b,
        logger=logger,
    )
    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    typer.echo(f"compare_id: {payload.get('compare_id')}")
    typer.echo(f"run_a_validation_grade: {payload.get('run_a_validation_grade')}")
    typer.echo(f"run_b_validation_grade: {payload.get('run_b_validation_grade')}")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"summary_path: {result.summary_path}")
    typer.echo(f"table_path: {result.table_path}")


@app.command("validation-wf-run")
def validation_wf_run(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Path to exported ML dataset parquet.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    train_end_list: str | None = typer.Option(
        None,
        "--train-end-list",
        help="Optional comma-separated train-end dates YYYY-MM-DD.",
    ),
    hmm_components: int | None = typer.Option(
        None,
        "--hmm-components",
        min=2,
        help="HMM component count override.",
    ),
    cluster_method: str | None = typer.Option(
        None,
        "--cluster-method",
        help="Cluster method override: gmm or kmeans.",
    ),
    cluster_k: int | None = typer.Option(
        None,
        "--cluster-k",
        min=2,
        help="Cluster count override for kmeans/gmm.",
    ),
    scaling_scope: str | None = typer.Option(
        None,
        "--scaling-scope",
        help="Scaling scope override: global or per_ticker.",
    ),
    bootstrap_n: int | None = typer.Option(
        None,
        "--bootstrap-n",
        min=10,
        help="Bootstrap iterations override.",
    ),
    bootstrap_mode: str | None = typer.Option(
        None,
        "--bootstrap-mode",
        help="Bootstrap mode override: iid or block.",
    ),
    block_length: int | None = typer.Option(
        None,
        "--block-length",
        min=1,
        help="Block length when bootstrap-mode=block.",
    ),
    event_window_pre: int | None = typer.Option(
        None,
        "--event-window-pre",
        min=1,
        help="Event-study pre-window bars.",
    ),
    event_window_post: int | None = typer.Option(
        None,
        "--event-window-post",
        min=1,
        help="Event-study post-window bars.",
    ),
    min_events_per_transition: int | None = typer.Option(
        None,
        "--min-events-per-transition",
        min=1,
        help="Minimum events per transition code.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force rerun for all configured splits.",
    ),
    force_split: list[str] = typer.Option(
        [],
        "--force-split",
        help="One or more train-end split dates; can be repeated or comma-separated.",
    ),
    stop_on_error: bool = typer.Option(
        False,
        "--stop-on-error",
        help="Stop immediately on first split error.",
    ),
    max_splits: int | None = typer.Option(
        None,
        "--max-splits",
        min=1,
        help="Optional max split count (useful for smoke testing).",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run walk-forward OOS validation pack across multiple train-end splits."""

    parsed_train_ends = _parse_date_csv(train_end_list, "train-end-list") if train_end_list is not None else None
    parsed_force_splits: list[date] = []
    for raw_value in force_split:
        parsed_force_splits.extend(_parse_date_csv(raw_value, "force-split"))

    cluster_method_norm = _normalize_choice(
        cluster_method,
        allowed={"gmm", "kmeans"},
        option_name="cluster-method",
    )
    scaling_scope_norm = _normalize_choice(
        scaling_scope,
        allowed={"global", "per_ticker"},
        option_name="scaling-scope",
    )
    bootstrap_mode_norm = _normalize_choice(
        bootstrap_mode,
        allowed={"iid", "block"},
        option_name="bootstrap-mode",
    )

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    result = run_validation_walkforward(
        settings,
        dataset_path=dataset,
        train_end_list=parsed_train_ends,
        hmm_components=hmm_components,
        cluster_method=cluster_method_norm,
        cluster_k=cluster_k,
        scaling_scope=scaling_scope_norm,
        bootstrap_n=bootstrap_n,
        bootstrap_mode=bootstrap_mode_norm,
        block_length=block_length,
        event_window_pre=event_window_pre,
        event_window_post=event_window_post,
        min_events_per_transition=min_events_per_transition,
        force=force,
        force_splits=parsed_force_splits,
        stop_on_error=stop_on_error,
        max_splits=max_splits,
        logger=logger,
    )
    aggregate_summary = json.loads(result.aggregate_summary_path.read_text(encoding="utf-8"))
    typer.echo(f"wf_run_id: {result.wf_run_id}")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"splits_total: {aggregate_summary.get('splits_total')}")
    typer.echo(f"splits_successful: {aggregate_summary.get('splits_successful')}")
    typer.echo(f"splits_failed: {aggregate_summary.get('splits_failed')}")
    typer.echo(f"manifest_path: {result.manifest_path}")
    typer.echo(f"aggregate_summary_path: {result.aggregate_summary_path}")
    typer.echo(f"full_report_path: {result.full_report_path}")


@app.command("validation-wf-sanity")
def validation_wf_sanity(
    wf_run_dir: Path = typer.Option(
        ...,
        "--wf-run-dir",
        help="Path to a walk-forward validation run directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """Summarize a completed walk-forward validation run."""

    summary = summarize_validation_walkforward_run(wf_run_dir)
    typer.echo(f"wf_run_id: {summary.get('wf_run_id')}")
    typer.echo(f"dataset_path: {summary.get('dataset_path')}")
    typer.echo(f"splits_total: {summary.get('splits_total')}")
    typer.echo(f"splits_successful: {summary.get('splits_successful')}")
    typer.echo(f"splits_failed: {summary.get('splits_failed')}")
    failed_splits = summary.get("failed_splits", [])
    if failed_splits:
        typer.echo("failed_splits:")
        for row in failed_splits:
            typer.echo(f"{row.get('train_end')} | error={row.get('error')}")
    typer.echo("wins_by_metric:")
    for metric, payload in summary.get("wins_by_metric", {}).items():
        typer.echo(f"{metric}: {payload}")
    typer.echo("aggregate_by_model:")
    for model, payload in summary.get("aggregate_by_model", {}).items():
        typer.echo(
            f"{model} | sep_mean={payload.get('forward_separation_score_mean')} | "
            f"ci_mean={payload.get('avg_ci_width_fwd_ret_10_mean')} | "
            f"sign_mean={payload.get('avg_state_sign_consistency_mean')} | "
            f"ret_cv_mean={payload.get('avg_state_ret_cv_mean')}"
        )


@app.command("cluster-qa-run")
def cluster_qa_run(
    validation_run_dir: Path | None = typer.Option(
        None,
        "--validation-run-dir",
        help="Single cluster validation run directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    wf_run_dir: Path | None = typer.Option(
        None,
        "--wf-run-dir",
        help="Walk-forward run directory to analyze all cluster validation splits.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    ret_cv_threshold: float | None = typer.Option(
        None,
        "--ret-cv-threshold",
        help="Flag threshold for ret_mean_cv.",
    ),
    min_n_rows: int | None = typer.Option(
        None,
        "--min-n-rows",
        min=1,
        help="Minimum state rows threshold.",
    ),
    min_state_share: float | None = typer.Option(
        None,
        "--min-state-share",
        min=0.0,
        max=1.0,
        help="Minimum state share threshold.",
    ),
    ci_width_quantile_threshold: float | None = typer.Option(
        None,
        "--ci-width-quantile-threshold",
        min=0.0,
        max=1.0,
        help="CI width quantile threshold for WIDE_CI flagging.",
    ),
    sign_consistency_threshold: float | None = typer.Option(
        None,
        "--sign-consistency-threshold",
        min=0.0,
        max=1.0,
        help="Minimum stability sign consistency threshold.",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run cluster instability QA diagnostics for one run or a walk-forward pack."""

    if (validation_run_dir is None and wf_run_dir is None) or (
        validation_run_dir is not None and wf_run_dir is not None
    ):
        raise typer.BadParameter("Specify exactly one of --validation-run-dir or --wf-run-dir.")

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    if validation_run_dir is not None:
        result = run_cluster_qa_single(
            settings,
            validation_run_dir=validation_run_dir,
            ret_cv_threshold=ret_cv_threshold,
            min_n_rows=min_n_rows,
            min_state_share=min_state_share,
            sign_consistency_threshold=sign_consistency_threshold,
            ci_width_quantile_threshold=ci_width_quantile_threshold,
            logger=logger,
        )
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        typer.echo("mode: single")
        typer.echo(f"output_dir: {result.output_dir}")
        typer.echo(f"states_total: {summary.get('states_total')}")
        typer.echo(f"states_flagged: {summary.get('states_flagged')}")
        typer.echo(f"issue_counts: {summary.get('issue_counts')}")
        typer.echo(f"summary_path: {result.summary_path}")
        typer.echo(f"flagged_states_path: {result.flagged_states_path}")
        typer.echo(f"state_windows_path: {result.state_windows_path}")
        typer.echo(f"report_path: {result.report_path}")
        return

    result = run_cluster_qa_walkforward(
        settings,
        wf_run_dir=wf_run_dir,
        ret_cv_threshold=ret_cv_threshold,
        min_n_rows=min_n_rows,
        min_state_share=min_state_share,
        sign_consistency_threshold=sign_consistency_threshold,
        ci_width_quantile_threshold=ci_width_quantile_threshold,
        logger=logger,
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    typer.echo("mode: walkforward")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"splits_analyzed: {summary.get('splits_analyzed')}")
    typer.echo(f"cluster_splits_flagged: {summary.get('cluster_splits_flagged')}")
    typer.echo(f"total_flagged_states: {summary.get('total_flagged_states')}")
    typer.echo(f"summary_path: {result.summary_path}")
    typer.echo(f"flagged_states_path: {result.flagged_states_path}")
    typer.echo(f"issue_frequency_path: {result.issue_frequency_path}")
    typer.echo(f"report_path: {result.report_path}")


@app.command("cluster-hardening-run")
def cluster_hardening_run(
    validation_run_dir: Path | None = typer.Option(
        None,
        "--validation-run-dir",
        help="Single cluster validation run directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    wf_run_dir: Path | None = typer.Option(
        None,
        "--wf-run-dir",
        help="Walk-forward run directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    clustered_rows_file: Path | None = typer.Option(
        None,
        "--clustered-rows-file",
        help="Optional clustered rows parquet/csv for filtered exports in single-run mode.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    export_filtered: bool = typer.Option(
        True,
        "--export-filtered/--no-export-filtered",
        help="When single-run mode has clustered rows available, export ALLOW/WATCH/full joined rows.",
    ),
    min_n_rows_hard: int | None = typer.Option(None, "--min-n-rows-hard", min=1),
    min_state_share_hard: float | None = typer.Option(None, "--min-state-share-hard", min=0.0, max=1.0),
    ret_cv_hard: float | None = typer.Option(None, "--ret-cv-hard", min=0.0),
    sign_consistency_hard: float | None = typer.Option(None, "--sign-consistency-hard", min=0.0, max=1.0),
    ci_width_hard_quantile: float | None = typer.Option(
        None,
        "--ci-width-hard-quantile",
        min=0.0,
        max=1.0,
    ),
    score_min_allow: float | None = typer.Option(None, "--score-min-allow", min=0.0, max=100.0),
    score_min_watch: float | None = typer.Option(None, "--score-min-watch", min=0.0, max=100.0),
    force: bool = typer.Option(False, "--force", help="Force rebuild even if output artifacts already exist."),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run cluster hardening in single validation mode or walk-forward mode."""

    if (validation_run_dir is None and wf_run_dir is None) or (
        validation_run_dir is not None and wf_run_dir is not None
    ):
        raise typer.BadParameter("Specify exactly one of --validation-run-dir or --wf-run-dir.")

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    if validation_run_dir is not None:
        result = run_cluster_hardening_single(
            settings,
            validation_run_dir=validation_run_dir,
            clustered_rows_file=clustered_rows_file,
            export_filtered=export_filtered,
            min_n_rows_hard=min_n_rows_hard,
            min_state_share_hard=min_state_share_hard,
            ret_cv_hard=ret_cv_hard,
            sign_consistency_hard=sign_consistency_hard,
            ci_width_hard_quantile=ci_width_hard_quantile,
            score_min_allow=score_min_allow,
            score_min_watch=score_min_watch,
            force=force,
            logger=logger,
        )
        policy = json.loads(result.policy_path.read_text(encoding="utf-8"))
        summary = policy.get("summary", {})
        typer.echo("mode: single")
        typer.echo(f"output_dir: {result.output_dir}")
        typer.echo(f"allow_count: {summary.get('allow_count')}")
        typer.echo(f"watch_count: {summary.get('watch_count')}")
        typer.echo(f"block_count: {summary.get('block_count')}")
        typer.echo(f"policy_path: {result.policy_path}")
        typer.echo(f"state_table_path: {result.state_table_path}")
        typer.echo(f"report_path: {result.report_path}")
        if result.export_summary_path is not None and result.export_summary_path.exists():
            export_summary = json.loads(result.export_summary_path.read_text(encoding="utf-8"))
            typer.echo(f"tradable_rows: {export_summary.get('tradable_rows')}")
            typer.echo(f"watch_rows: {export_summary.get('watch_rows')}")
            typer.echo(f"export_summary_path: {result.export_summary_path}")
        return

    result = run_cluster_hardening_walkforward(
        settings,
        wf_run_dir=wf_run_dir,
        min_n_rows_hard=min_n_rows_hard,
        min_state_share_hard=min_state_share_hard,
        ret_cv_hard=ret_cv_hard,
        sign_consistency_hard=sign_consistency_hard,
        ci_width_hard_quantile=ci_width_hard_quantile,
        score_min_allow=score_min_allow,
        score_min_watch=score_min_watch,
        force=force,
        logger=logger,
    )
    wf_summary = json.loads(result.wf_summary_path.read_text(encoding="utf-8"))
    typer.echo("mode: walkforward")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"splits_total: {wf_summary.get('splits_total')}")
    typer.echo(f"splits_successful: {wf_summary.get('splits_successful')}")
    typer.echo(f"splits_failed: {wf_summary.get('splits_failed')}")
    typer.echo(f"wf_summary_path: {result.wf_summary_path}")
    typer.echo(f"wf_state_stats_path: {result.wf_state_stats_path}")
    typer.echo(f"split_counts_path: {result.split_counts_path}")
    typer.echo(f"issue_frequency_path: {result.issue_frequency_path}")
    typer.echo(f"threshold_recommendation_path: {result.threshold_recommendation_path}")
    typer.echo(f"report_path: {result.report_path}")


@app.command("cluster-hardening-sanity")
def cluster_hardening_sanity(
    hardening_dir: Path = typer.Option(
        ...,
        "--hardening-dir",
        help="Hardening output directory (single-run or walk-forward).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """Print concise summary for cluster hardening artifacts."""

    summary = summarize_cluster_hardening(hardening_dir)
    mode = summary.get("mode")
    typer.echo(f"mode: {mode}")
    typer.echo(f"hardening_dir: {summary.get('hardening_dir')}")
    if mode == "single":
        s = summary.get("summary", {})
        typer.echo(f"allow_count: {s.get('allow_count')}")
        typer.echo(f"watch_count: {s.get('watch_count')}")
        typer.echo(f"block_count: {s.get('block_count')}")
        typer.echo("state_table_preview:")
        for row in summary.get("state_table_preview", [])[:20]:
            typer.echo(
                f"state={row.get('state_id')} | class={row.get('class_label')} | "
                f"score={row.get('tradability_score')} | dir={row.get('allow_direction_hint')} | reasons={row.get('reasons')}"
            )
        export_summary = summary.get("export_summary")
        if export_summary is not None:
            typer.echo(
                f"export_rows source={export_summary.get('source_rows')} tradable={export_summary.get('tradable_rows')} watch={export_summary.get('watch_rows')}"
            )
        return

    wf = summary.get("summary", {})
    typer.echo(f"splits_total: {wf.get('splits_total')}")
    typer.echo(f"splits_successful: {wf.get('splits_successful')}")
    typer.echo(f"splits_failed: {wf.get('splits_failed')}")
    typer.echo("split_counts_preview:")
    for row in summary.get("split_counts_preview", [])[:20]:
        typer.echo(
            f"{row.get('train_end')} | status={row.get('status')} | allow={row.get('allow_count')} | watch={row.get('watch_count')} | block={row.get('block_count')}"
        )
    typer.echo("issue_frequency_preview:")
    for row in summary.get("issue_frequency_preview", [])[:20]:
        typer.echo(f"{row.get('issue')} | split_count={row.get('split_count')} | state_count={row.get('state_count')}")


@app.command("cluster-hardening-compare")
def cluster_hardening_compare(
    hardening_dir_a: Path = typer.Option(
        ...,
        "--hardening-dir-a",
        help="First single-run hardening output directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    hardening_dir_b: Path = typer.Option(
        ...,
        "--hardening-dir-b",
        help="Second single-run hardening output directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Compare two cluster hardening policies and write diff artifacts."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    result = run_cluster_hardening_compare(
        settings,
        hardening_dir_a=hardening_dir_a,
        hardening_dir_b=hardening_dir_b,
        logger=logger,
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    typer.echo(f"compare_id: {summary.get('compare_id')}")
    typer.echo(f"run_a_allow_count: {summary.get('run_a_allow_count')}")
    typer.echo(f"run_b_allow_count: {summary.get('run_b_allow_count')}")
    typer.echo(f"class_changed_states: {summary.get('class_changed_states')}")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"summary_path: {result.summary_path}")
    typer.echo(f"table_path: {result.table_path}")


def _resolve_sensitivity_dimensions(
    settings: AppSettings,
    *,
    hold_bars_grid: str | None,
    signal_mode_grid: str | None,
    exit_mode_grid: str | None,
    fee_bps_grid: str | None,
    slippage_bps_grid: str | None,
    allow_overlap_grid: str | None,
    equity_mode_grid: str | None,
    include_watch_grid: str | None,
    include_state_ids_grid: str | None,
) -> GridDimensionValues:
    cfg = settings.backtest_sensitivity.default_grid
    signal_allowed = {"state_entry", "state_transition_entry", "state_persistence_confirm"}
    exit_allowed = {"horizon", "state_exit", "horizon_or_state"}
    equity_allowed = {"event_returns_only", "daily_equity_curve"}

    hold_bars = _parse_int_csv(hold_bars_grid, "hold-bars-grid") if hold_bars_grid else [int(v) for v in cfg.hold_bars]
    signal_mode = (
        _parse_choice_csv(signal_mode_grid, "signal-mode-grid", signal_allowed)
        if signal_mode_grid
        else [str(v) for v in cfg.signal_mode]
    )
    exit_mode = (
        _parse_choice_csv(exit_mode_grid, "exit-mode-grid", exit_allowed)
        if exit_mode_grid
        else [str(v) for v in cfg.exit_mode]
    )
    fee_bps = _parse_float_csv(fee_bps_grid, "fee-bps-grid") if fee_bps_grid else [float(v) for v in cfg.fee_bps_per_side]
    slippage_bps = (
        _parse_float_csv(slippage_bps_grid, "slippage-bps-grid")
        if slippage_bps_grid
        else [float(v) for v in cfg.slippage_bps_per_side]
    )
    allow_overlap = (
        _parse_bool_csv(allow_overlap_grid, "allow-overlap-grid")
        if allow_overlap_grid
        else [bool(v) for v in cfg.allow_overlap]
    )
    equity_mode = (
        _parse_choice_csv(equity_mode_grid, "equity-mode-grid", equity_allowed)
        if equity_mode_grid
        else [str(v) for v in cfg.equity_mode]
    )
    include_watch = (
        _parse_bool_csv(include_watch_grid, "include-watch-grid")
        if include_watch_grid
        else [bool(v) for v in cfg.include_watch]
    )
    include_state_sets = (
        _parse_state_set_grid(include_state_ids_grid, "include-state-ids-grid")
        if include_state_ids_grid
        else [list(v) for v in cfg.include_state_sets]
    )
    if not include_state_sets:
        include_state_sets = [[]]

    return GridDimensionValues(
        hold_bars=hold_bars,
        signal_mode=cast(list[str], signal_mode),
        exit_mode=cast(list[str], exit_mode),
        fee_bps_per_side=fee_bps,
        slippage_bps_per_side=slippage_bps,
        allow_overlap=allow_overlap,
        equity_mode=cast(list[str], equity_mode),
        include_watch=include_watch,
        include_state_sets=include_state_sets,
    )


@app.command("backtest-run")
def backtest_run(
    input_type: str = typer.Option(
        ...,
        "--input-type",
        help="Input type: flow, hmm, cluster.",
    ),
    input_file: Path = typer.Option(
        ...,
        "--input-file",
        help="Input parquet/csv file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    validation_run_dir: Path | None = typer.Option(
        None,
        "--validation-run-dir",
        help="Optional validation run dir for HMM state-direction inference.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    cluster_hardening_dir: Path | None = typer.Option(
        None,
        "--cluster-hardening-dir",
        help="Cluster hardening directory for cluster ALLOW/WATCH/BLOCK policy.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    overlay_cluster_file: Path | None = typer.Option(
        None,
        "--overlay-cluster-file",
        help="Overlay cluster rows parquet/csv for hybrid gating.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    overlay_cluster_hardening_dir: Path | None = typer.Option(
        None,
        "--overlay-cluster-hardening-dir",
        help="Overlay cluster hardening directory containing cluster_hardening_policy.json.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    overlay_mode: str | None = typer.Option(
        None,
        "--overlay-mode",
        help="Overlay mode: none, allow_only, allow_watch, block_veto, allow_or_unknown.",
    ),
    overlay_join_keys: str | None = typer.Option(
        None,
        "--overlay-join-keys",
        help="Comma-separated join keys. Default: ticker,trade_date",
    ),
    state_map_file: Path | None = typer.Option(
        None,
        "--state-map-file",
        help="Optional state map JSON override.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    signal_mode: str | None = typer.Option(
        None,
        "--signal-mode",
        help="Signal mode: state_entry, state_transition_entry, state_persistence_confirm.",
    ),
    exit_mode: str | None = typer.Option(
        None,
        "--exit-mode",
        help="Exit mode: horizon, state_exit, horizon_or_state.",
    ),
    hold_bars: int | None = typer.Option(None, "--hold-bars", min=1),
    allow_overlap: bool | None = typer.Option(None, "--allow-overlap/--no-allow-overlap"),
    allow_unconfirmed: bool | None = typer.Option(None, "--allow-unconfirmed/--no-allow-unconfirmed"),
    include_watch: bool = typer.Option(False, "--include-watch", help="Cluster mode: include WATCH states."),
    include_state_ids: str | None = typer.Option(
        None,
        "--include-state-ids",
        help="Optional comma-separated state ids to include (e.g., 1,2,4).",
    ),
    fee_bps_per_side: float | None = typer.Option(None, "--fee-bps-per-side", min=0.0),
    slippage_bps_per_side: float | None = typer.Option(None, "--slippage-bps-per-side", min=0.0),
    equity_mode: str | None = typer.Option(
        None,
        "--equity-mode",
        help="Equity mode: event_returns_only or daily_equity_curve.",
    ),
    export_joined_rows: bool = typer.Option(False, "--export-joined-rows"),
    tag: str | None = typer.Option(None, "--tag"),
    force: bool = typer.Option(False, "--force"),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run one deterministic research backtest for FLOW/HMM/CLUSTER states."""

    input_type_norm = _normalize_choice(input_type, allowed={"flow", "hmm", "cluster"}, option_name="input-type")
    signal_mode_norm = _normalize_choice(
        signal_mode,
        allowed={"state_entry", "state_transition_entry", "state_persistence_confirm"},
        option_name="signal-mode",
    )
    exit_mode_norm = _normalize_choice(
        exit_mode,
        allowed={"horizon", "state_exit", "horizon_or_state"},
        option_name="exit-mode",
    )
    equity_mode_norm = _normalize_choice(
        equity_mode,
        allowed={"event_returns_only", "daily_equity_curve"},
        option_name="equity-mode",
    )
    include_ids = _parse_int_csv(include_state_ids, "include-state-ids") if include_state_ids else []
    overlay_mode_norm = _normalize_choice(
        overlay_mode,
        allowed={"none", "allow_only", "allow_watch", "block_veto", "allow_or_unknown"},
        option_name="overlay-mode",
    )

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    overlay_keys = (
        [part.strip() for part in overlay_join_keys.split(",") if part.strip() != ""]
        if overlay_join_keys
        else list(settings.backtest_policy_overlay.join_keys)
    )
    if (overlay_cluster_file is None) ^ (overlay_cluster_hardening_dir is None):
        raise typer.BadParameter(
            "Provide both --overlay-cluster-file and --overlay-cluster-hardening-dir, or neither."
        )
    if (overlay_mode_norm or settings.backtest_policy_overlay.default_overlay_mode) != "none" and (
        overlay_cluster_file is None or overlay_cluster_hardening_dir is None
    ):
        raise typer.BadParameter(
            "overlay-mode is not none but overlay inputs are missing. Provide both overlay inputs."
        )
    result = run_backtest_run(
        settings,
        input_type=input_type_norm or "flow",
        input_file=input_file,
        validation_run_dir=validation_run_dir,
        cluster_hardening_dir=cluster_hardening_dir,
        state_map_file=state_map_file,
        signal_mode=(signal_mode_norm or settings.backtest.signal_mode),
        exit_mode=(exit_mode_norm or settings.backtest.exit_mode),
        hold_bars=(hold_bars if hold_bars is not None else settings.backtest.hold_bars),
        allow_overlap=(allow_overlap if allow_overlap is not None else settings.backtest.allow_overlap),
        allow_unconfirmed=(
            allow_unconfirmed if allow_unconfirmed is not None else settings.backtest.allow_unconfirmed
        ),
        include_watch=include_watch,
        include_state_ids=include_ids,
        overlay_cluster_file=overlay_cluster_file,
        overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,
        overlay_mode=cast(
            str,
            overlay_mode_norm or settings.backtest_policy_overlay.default_overlay_mode,
        ),
        overlay_join_keys=overlay_keys,
        fee_bps_per_side=(
            fee_bps_per_side if fee_bps_per_side is not None else settings.backtest.fee_bps_per_side
        ),
        slippage_bps_per_side=(
            slippage_bps_per_side
            if slippage_bps_per_side is not None
            else settings.backtest.slippage_bps_per_side
        ),
        equity_mode=(equity_mode_norm or settings.backtest.equity_mode),
        export_joined_rows=export_joined_rows,
        tag=tag,
        force=force,
        logger=logger,
    )
    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    headline = payload.get("headline", {})
    typer.echo(f"run_id: {payload.get('run_id')}")
    typer.echo(f"input_type: {payload.get('input_type')}")
    typer.echo(f"trade_count: {headline.get('trade_count')}")
    typer.echo(f"win_rate: {headline.get('win_rate')}")
    typer.echo(f"avg_return: {headline.get('avg_return')}")
    typer.echo(f"profit_factor: {headline.get('profit_factor')}")
    typer.echo(f"expectancy: {headline.get('expectancy')}")
    overlay_payload = payload.get("overlay", {}) if isinstance(payload.get("overlay"), dict) else {}
    if overlay_payload:
        typer.echo(f"overlay_mode: {overlay_payload.get('overlay_mode')}")
        typer.echo(f"overlay_match_rate: {overlay_payload.get('overlay_match_rate')}")
        typer.echo(f"overlay_vetoed_signal_share: {overlay_payload.get('overlay_vetoed_signal_share')}")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"summary_path: {result.summary_path}")
    typer.echo(f"trades_path: {result.trades_path}")
    typer.echo(f"report_path: {result.report_path}")


@app.command("backtest-sanity")
def backtest_sanity(
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        help="Backtest run directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """Run sanity checks for one backtest run dir and print compact diagnostics."""

    summary = summarize_backtest_run(run_dir)
    payload = summary["summary"]
    headline = payload.get("headline", {})
    nan_warnings = summary.get("nan_warnings", {})
    nan_total = sum(int(v) for v in nan_warnings.values())
    typer.echo(f"run_id: {payload.get('run_id')}")
    typer.echo(f"input_type: {payload.get('input_type')}")
    typer.echo(f"trade_count: {headline.get('trade_count')}")
    typer.echo(f"win_rate: {headline.get('win_rate')}")
    typer.echo(f"avg_return: {headline.get('avg_return')}")
    typer.echo(f"profit_factor: {headline.get('profit_factor')}")
    typer.echo(f"expectancy: {headline.get('expectancy')}")
    typer.echo(f"nan_warning_total: {nan_total}")
    errors = summary.get("errors", [])
    typer.echo(f"errors: {errors if errors else 'none'}")
    policy_info = summary.get("policy_info")
    if policy_info is not None:
        typer.echo(
            f"policy allow/watch/block: {policy_info.get('allow_count')}/{policy_info.get('watch_count')}/{policy_info.get('block_count')}"
        )
    overlay_info = summary.get("overlay_info")
    if overlay_info is not None:
        typer.echo(
            f"overlay mode={overlay_info.get('overlay_mode')} match_rate={overlay_info.get('overlay_match_rate')} "
            f"veto_share={overlay_info.get('overlay_vetoed_signal_share')} conflict_share={overlay_info.get('overlay_direction_conflict_share')}"
        )
    typer.echo("top_states:")
    for row in summary.get("top_states", [])[:10]:
        typer.echo(
            f"state={row.get('entry_state_id')} class={row.get('entry_state_class')} trades={row.get('trade_count')} avg={row.get('avg_return')}"
        )


@app.command("backtest-compare")
def backtest_compare(
    run_dir: list[Path] = typer.Option(
        ...,
        "--run-dir",
        help="Backtest run directory (repeat option for multiple runs).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Compare two or more backtest runs and write compare artifacts."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    result = run_backtest_compare(settings, run_dirs=run_dir, logger=logger)
    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    typer.echo(f"compare_id: {payload.get('compare_id')}")
    typer.echo(f"runs_count: {len(payload.get('run_dirs', []))}")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"summary_path: {result.summary_path}")
    typer.echo(f"table_path: {result.table_path}")
    typer.echo(f"report_path: {result.report_path}")


@app.command("backtest-wf-run")
def backtest_wf_run(
    wf_run_dir: Path = typer.Option(
        ...,
        "--wf-run-dir",
        help="Validation walk-forward directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    flow_dataset_file: Path | None = typer.Option(
        None,
        "--flow-dataset-file",
        help="Optional flow dataset file fallback.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    overlay_cluster_file: Path | None = typer.Option(
        None,
        "--overlay-cluster-file",
        help="Overlay cluster rows parquet/csv for hybrid gating.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    overlay_cluster_hardening_dir: Path | None = typer.Option(
        None,
        "--overlay-cluster-hardening-dir",
        help="Overlay cluster hardening directory with cluster_hardening_policy.json.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    overlay_mode: str | None = typer.Option(
        None,
        "--overlay-mode",
        help="Overlay mode: none, allow_only, allow_watch, block_veto, allow_or_unknown.",
    ),
    overlay_join_keys: str | None = typer.Option(
        None,
        "--overlay-join-keys",
        help="Comma-separated join keys. Default: ticker,trade_date",
    ),
    signal_mode: str | None = typer.Option(None, "--signal-mode"),
    exit_mode: str | None = typer.Option(None, "--exit-mode"),
    hold_bars: int | None = typer.Option(None, "--hold-bars", min=1),
    fee_bps_per_side: float | None = typer.Option(None, "--fee-bps-per-side", min=0.0),
    slippage_bps_per_side: float | None = typer.Option(None, "--slippage-bps-per-side", min=0.0),
    allow_overlap: bool | None = typer.Option(None, "--allow-overlap/--no-allow-overlap"),
    allow_unconfirmed: bool | None = typer.Option(None, "--allow-unconfirmed/--no-allow-unconfirmed"),
    include_watch: bool = typer.Option(False, "--include-watch"),
    equity_mode: str | None = typer.Option(None, "--equity-mode"),
    force: bool = typer.Option(False, "--force"),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run HMM/FLOW/CLUSTER backtests across a validation walk-forward pack."""

    signal_mode_norm = _normalize_choice(
        signal_mode,
        allowed={"state_entry", "state_transition_entry", "state_persistence_confirm"},
        option_name="signal-mode",
    )
    exit_mode_norm = _normalize_choice(
        exit_mode,
        allowed={"horizon", "state_exit", "horizon_or_state"},
        option_name="exit-mode",
    )
    equity_mode_norm = _normalize_choice(
        equity_mode,
        allowed={"event_returns_only", "daily_equity_curve"},
        option_name="equity-mode",
    )
    overlay_mode_norm = _normalize_choice(
        overlay_mode,
        allowed={"none", "allow_only", "allow_watch", "block_veto", "allow_or_unknown"},
        option_name="overlay-mode",
    )
    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    if (overlay_cluster_file is None) ^ (overlay_cluster_hardening_dir is None):
        raise typer.BadParameter(
            "Provide both --overlay-cluster-file and --overlay-cluster-hardening-dir, or neither."
        )
    if (overlay_mode_norm or settings.backtest_policy_overlay.default_overlay_mode) != "none" and (
        overlay_cluster_file is None or overlay_cluster_hardening_dir is None
    ):
        raise typer.BadParameter(
            "overlay-mode is not none but overlay inputs are missing. Provide both overlay inputs."
        )
    overlay_keys = (
        [part.strip() for part in overlay_join_keys.split(",") if part.strip() != ""]
        if overlay_join_keys
        else list(settings.backtest_policy_overlay.join_keys)
    )
    result = run_backtest_walkforward(
        settings,
        wf_run_dir=wf_run_dir,
        flow_dataset_file=flow_dataset_file,
        overlay_cluster_file=overlay_cluster_file,
        overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,
        overlay_mode=cast(
            str,
            overlay_mode_norm or settings.backtest_policy_overlay.default_overlay_mode,
        ),
        overlay_join_keys=overlay_keys,
        signal_mode=(signal_mode_norm or settings.backtest.signal_mode),
        exit_mode=(exit_mode_norm or settings.backtest.exit_mode),
        hold_bars=(hold_bars if hold_bars is not None else settings.backtest.hold_bars),
        fee_bps_per_side=(
            fee_bps_per_side if fee_bps_per_side is not None else settings.backtest.fee_bps_per_side
        ),
        slippage_bps_per_side=(
            slippage_bps_per_side
            if slippage_bps_per_side is not None
            else settings.backtest.slippage_bps_per_side
        ),
        allow_overlap=(allow_overlap if allow_overlap is not None else settings.backtest.allow_overlap),
        allow_unconfirmed=(
            allow_unconfirmed if allow_unconfirmed is not None else settings.backtest.allow_unconfirmed
        ),
        include_watch=include_watch,
        equity_mode=(equity_mode_norm or settings.backtest.equity_mode),
        force=force,
        logger=logger,
    )
    payload = json.loads(result.aggregate_summary_path.read_text(encoding="utf-8"))
    typer.echo(f"wf_bt_id: {result.wf_bt_id}")
    typer.echo(f"splits_total: {payload.get('splits_total')}")
    typer.echo(f"splits_successful: {payload.get('splits_successful')}")
    typer.echo(f"splits_failed: {payload.get('splits_failed')}")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"manifest_path: {result.manifest_path}")
    typer.echo(f"aggregate_summary_path: {result.aggregate_summary_path}")
    typer.echo(f"model_summary_path: {result.model_summary_path}")
    typer.echo(f"report_path: {result.report_path}")


@app.command("backtest-grid-run")
def backtest_grid_run(
    multi_source: bool = typer.Option(False, "--multi-source", help="Run aligned grid across multiple sources."),
    input_type: str | None = typer.Option(None, "--input-type", help="Single-source mode: flow, hmm, cluster."),
    input_file: Path | None = typer.Option(
        None,
        "--input-file",
        help="Single-source mode input file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    flow_input_file: Path | None = typer.Option(
        None,
        "--flow-input-file",
        help="Multi-source FLOW input file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    hmm_input_file: Path | None = typer.Option(
        None,
        "--hmm-input-file",
        help="Multi-source HMM decoded rows file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    cluster_input_file: Path | None = typer.Option(
        None,
        "--cluster-input-file",
        help="Multi-source cluster rows file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    validation_run_dir: Path | None = typer.Option(
        None,
        "--validation-run-dir",
        help="Optional HMM validation run directory for direction inference.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    cluster_hardening_dir: Path | None = typer.Option(
        None,
        "--cluster-hardening-dir",
        help="Cluster hardening directory for cluster mapping.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    overlay_cluster_file: Path | None = typer.Option(
        None,
        "--overlay-cluster-file",
        help="Overlay cluster rows parquet/csv for hybrid gating.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    overlay_cluster_hardening_dir: Path | None = typer.Option(
        None,
        "--overlay-cluster-hardening-dir",
        help="Overlay cluster hardening directory containing cluster_hardening_policy.json.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    overlay_mode: str | None = typer.Option(
        None,
        "--overlay-mode",
        help="Overlay mode: none, allow_only, allow_watch, block_veto, allow_or_unknown.",
    ),
    overlay_join_keys: str | None = typer.Option(
        None,
        "--overlay-join-keys",
        help="Comma-separated join keys. Default: ticker,trade_date",
    ),
    state_map_file: Path | None = typer.Option(
        None,
        "--state-map-file",
        help="Optional JSON state map override.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    hold_bars_grid: str | None = typer.Option(None, "--hold-bars-grid"),
    signal_mode_grid: str | None = typer.Option(None, "--signal-mode-grid"),
    exit_mode_grid: str | None = typer.Option(None, "--exit-mode-grid"),
    fee_bps_grid: str | None = typer.Option(None, "--fee-bps-grid"),
    slippage_bps_grid: str | None = typer.Option(None, "--slippage-bps-grid"),
    allow_overlap_grid: str | None = typer.Option(None, "--allow-overlap-grid"),
    equity_mode_grid: str | None = typer.Option(None, "--equity-mode-grid"),
    include_watch_grid: str | None = typer.Option(None, "--include-watch-grid"),
    include_state_ids_grid: str | None = typer.Option(
        None,
        "--include-state-ids-grid",
        help="Semicolon-delimited state subsets, each subset pipe-delimited. Example: 1|2|4;2|3|4",
    ),
    policy_filter_mode: str | None = typer.Option(
        None,
        "--policy-filter-mode",
        help="Cluster policy filter mode: allow_only, allow_watch, all_states.",
    ),
    include_ret_cv: bool | None = typer.Option(None, "--include-ret-cv/--no-include-ret-cv"),
    include_tail_metrics: bool | None = typer.Option(None, "--include-tail-metrics/--no-include-tail-metrics"),
    report_top_n: int | None = typer.Option(None, "--report-top-n", min=1),
    max_combos: int | None = typer.Option(None, "--max-combos", min=1),
    shuffle_grid: bool = typer.Option(False, "--shuffle-grid"),
    seed: int = typer.Option(42, "--seed"),
    tag: str | None = typer.Option(None, "--tag"),
    force: bool = typer.Option(False, "--force"),
    progress_every: int | None = typer.Option(None, "--progress-every", min=1),
    stop_on_error: bool = typer.Option(False, "--stop-on-error"),
    write_run_manifest: bool = typer.Option(True, "--write-run-manifest/--no-write-run-manifest"),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run a parameter sensitivity grid for one source or aligned multi-source."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    policy_filter_mode_norm = _normalize_choice(
        policy_filter_mode,
        allowed={"allow_only", "allow_watch", "all_states"},
        option_name="policy-filter-mode",
    ) or settings.backtest_sensitivity.policy_filter_mode_default
    overlay_mode_norm = _normalize_choice(
        overlay_mode,
        allowed={"none", "allow_only", "allow_watch", "block_veto", "allow_or_unknown"},
        option_name="overlay-mode",
    ) or settings.backtest_policy_overlay.default_overlay_mode
    overlay_join_keys_norm = (
        [part.strip() for part in overlay_join_keys.split(",") if part.strip() != ""]
        if overlay_join_keys
        else list(settings.backtest_policy_overlay.join_keys)
    )
    if (overlay_cluster_file is None) ^ (overlay_cluster_hardening_dir is None):
        raise typer.BadParameter(
            "Provide both --overlay-cluster-file and --overlay-cluster-hardening-dir, or neither."
        )
    if overlay_mode_norm != "none" and (overlay_cluster_file is None or overlay_cluster_hardening_dir is None):
        raise typer.BadParameter(
            "overlay-mode is not none but overlay inputs are missing. Provide both overlay inputs."
        )
    dims = _resolve_sensitivity_dimensions(
        settings,
        hold_bars_grid=hold_bars_grid,
        signal_mode_grid=signal_mode_grid,
        exit_mode_grid=exit_mode_grid,
        fee_bps_grid=fee_bps_grid,
        slippage_bps_grid=slippage_bps_grid,
        allow_overlap_grid=allow_overlap_grid,
        equity_mode_grid=equity_mode_grid,
        include_watch_grid=include_watch_grid,
        include_state_ids_grid=include_state_ids_grid,
    )

    source_specs: list[SourceInputSpec] = []
    if multi_source:
        if flow_input_file is not None:
            source_specs.append(
                SourceInputSpec(
                    source_type="flow",
                    input_file=flow_input_file,
                    overlay_cluster_file=overlay_cluster_file,
                    overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,
                    overlay_mode=cast(str, overlay_mode_norm),
                    overlay_join_keys=overlay_join_keys_norm,
                )
            )
        if hmm_input_file is not None:
            source_specs.append(
                SourceInputSpec(
                    source_type="hmm",
                    input_file=hmm_input_file,
                    validation_run_dir=validation_run_dir,
                    state_map_file=state_map_file,
                    overlay_cluster_file=overlay_cluster_file,
                    overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,
                    overlay_mode=cast(str, overlay_mode_norm),
                    overlay_join_keys=overlay_join_keys_norm,
                )
            )
        if cluster_input_file is not None:
            source_specs.append(
                SourceInputSpec(
                    source_type="cluster",
                    input_file=cluster_input_file,
                    cluster_hardening_dir=cluster_hardening_dir,
                    policy_filter_mode=cast(str, policy_filter_mode_norm),
                    overlay_cluster_file=overlay_cluster_file,
                    overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,
                    overlay_mode=cast(str, overlay_mode_norm),
                    overlay_join_keys=overlay_join_keys_norm,
                )
            )
        if len(source_specs) < 2:
            raise typer.BadParameter("multi-source mode requires at least two source input files.")
    else:
        input_type_norm = _normalize_choice(input_type, allowed={"flow", "hmm", "cluster"}, option_name="input-type")
        if input_type_norm is None:
            raise typer.BadParameter("--input-type is required in single-source mode.")
        if input_file is None:
            raise typer.BadParameter("--input-file is required in single-source mode.")
        source_specs.append(
            SourceInputSpec(
                source_type=cast(InputType, input_type_norm),
                input_file=input_file,
                validation_run_dir=validation_run_dir if input_type_norm == "hmm" else None,
                cluster_hardening_dir=cluster_hardening_dir if input_type_norm == "cluster" else None,
                state_map_file=state_map_file if input_type_norm == "hmm" else None,
                policy_filter_mode=(cast(str, policy_filter_mode_norm) if input_type_norm == "cluster" else "allow_only"),
                overlay_cluster_file=overlay_cluster_file,
                overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,
                overlay_mode=cast(str, overlay_mode_norm),
                overlay_join_keys=overlay_join_keys_norm,
            )
        )

    result = run_backtest_grid(
        settings,
        source_specs=source_specs,
        dimensions=dims,
        tag=tag,
        max_combos=max_combos,
        shuffle_grid=shuffle_grid,
        seed=seed,
        progress_every=progress_every,
        stop_on_error=stop_on_error,
        force=force,
        write_run_manifest=write_run_manifest,
        include_ret_cv=(
            include_ret_cv if include_ret_cv is not None else settings.backtest_sensitivity.include_ret_cv_default
        ),
        include_tail_metrics=(
            include_tail_metrics
            if include_tail_metrics is not None
            else settings.backtest_sensitivity.include_tail_metrics_default
        ),
        report_top_n=(
            report_top_n if report_top_n is not None else settings.backtest_sensitivity.report_top_n_default
        ),
        logger=logger,
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    typer.echo(f"grid_run_id: {summary.get('grid_run_id')}")
    typer.echo(f"scope: {summary.get('scope')}")
    typer.echo(f"comparability: {summary.get('comparability')}")
    typer.echo(f"total_combos: {summary.get('total_combos')}")
    typer.echo(f"successful_combos: {summary.get('successful_combos')}")
    typer.echo(f"failed_combos: {summary.get('failed_combos')}")
    typer.echo(f"zero_trade_combos: {summary.get('zero_trade_combos')}")
    typer.echo(f"zero_trade_combo_share: {summary.get('zero_trade_combo_share')}")
    typer.echo(f"non_finite_cells: {summary.get('non_finite_cells')}")
    typer.echo(f"null_metric_cells: {summary.get('null_metric_cells')}")
    typer.echo(f"overlay_enabled: {any(bool((s or {}).get('overlay_cluster_file')) for s in summary.get('sources', []))}")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"manifest_path: {result.manifest_path}")
    typer.echo(f"metrics_table_path: {result.metrics_table_path}")
    typer.echo(f"summary_path: {result.summary_path}")
    typer.echo(f"report_path: {result.report_path}")


@app.command("backtest-grid-sanity")
def backtest_grid_sanity(
    grid_run_dir: Path = typer.Option(
        ...,
        "--grid-run-dir",
        help="Backtest sensitivity grid output directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    )
) -> None:
    """Run QA checks and print compact summary for one grid run."""

    summary = summarize_grid_run(grid_run_dir)
    payload = summary.get("summary", {})
    status_counts = summary.get("status_counts", {})
    typer.echo(f"grid_run_id: {payload.get('grid_run_id')}")
    typer.echo(f"scope: {payload.get('scope')}")
    typer.echo(f"comparability: {payload.get('comparability')}")
    typer.echo(f"total_combos: {payload.get('total_combos')}")
    typer.echo(
        f"status_counts: success={status_counts.get('SUCCESS',0)} failed={status_counts.get('FAILED',0)} skipped={status_counts.get('SKIPPED',0)}"
    )
    typer.echo(f"errors: {summary.get('errors') or 'none'}")
    typer.echo(f"warnings: {summary.get('warnings') or 'none'}")
    typer.echo(f"non_finite_cells: {summary.get('non_finite_cells')}")
    typer.echo(f"null_metric_cells: {summary.get('null_metric_cells')}")
    typer.echo(f"zero_trade_combos: {summary.get('zero_trade_combos')}")
    typer.echo("top_expectancy:")
    for row in summary.get("top_expectancy", [])[:10]:
        typer.echo(
            f"src={row.get('source_type')} combo={row.get('combo_id')} hb={row.get('hold_bars')} "
            f"sig={row.get('signal_mode')} fee={row.get('fee_bps_per_side')} exp={row.get('expectancy')} "
            f"pf={row.get('profit_factor')} ret_cv={row.get('ret_cv')} downside={row.get('downside_std')} "
            f"rob_v2={row.get('robustness_score_v2')} overlay_mode={row.get('overlay_mode')} "
            f"overlay_veto_share={row.get('overlay_vetoed_signal_share')}"
        )


@app.command("backtest-grid-compare")
def backtest_grid_compare(
    grid_run_dir: list[Path] = typer.Option(
        ...,
        "--grid-run-dir",
        help="Grid run directory (repeat option).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    metric: list[str] = typer.Option(
        [],
        "--metric",
        help="Optional prioritized metric(s), repeat option. Example: --metric expectancy --metric profit_factor",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Compare two or more sensitivity grid runs."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    allowed_metrics = {
        "expectancy",
        "profit_factor",
        "max_drawdown",
        "sharpe_proxy",
        "robustness_score",
        "robustness_score_v1",
        "robustness_score_v2",
        "ret_cv",
        "downside_std",
        "avg_return",
        "win_rate",
    }
    metric_norm = [m.strip().lower() for m in metric if m.strip() != ""]
    for value in metric_norm:
        if value not in allowed_metrics:
            raise typer.BadParameter(f"metric must be one of: {','.join(sorted(allowed_metrics))}")
    result = run_backtest_grid_compare(
        settings,
        grid_run_dirs=grid_run_dir,
        metrics=metric_norm or None,
        logger=logger,
    )
    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    typer.echo(f"compare_id: {payload.get('compare_id')}")
    typer.echo(f"primary_metric: {payload.get('primary_metric')}")
    typer.echo(f"grid_runs: {len(payload.get('grid_run_dirs', []))}")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"summary_path: {result.summary_path}")
    typer.echo(f"table_path: {result.table_path}")
    typer.echo(f"report_path: {result.report_path}")


@app.command("backtest-grid-wf-run")
def backtest_grid_wf_run(
    wf_run_dir: Path = typer.Option(
        ...,
        "--wf-run-dir",
        help="Validation walk-forward directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    flow_dataset_file: Path | None = typer.Option(
        None,
        "--flow-dataset-file",
        help="Optional flow dataset fallback path.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    overlay_cluster_file: Path | None = typer.Option(
        None,
        "--overlay-cluster-file",
        help="Overlay cluster rows parquet/csv for hybrid gating.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    overlay_cluster_hardening_dir: Path | None = typer.Option(
        None,
        "--overlay-cluster-hardening-dir",
        help="Overlay cluster hardening directory containing cluster_hardening_policy.json.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    overlay_mode: str | None = typer.Option(
        None,
        "--overlay-mode",
        help="Overlay mode: none, allow_only, allow_watch, block_veto, allow_or_unknown.",
    ),
    overlay_join_keys: str | None = typer.Option(
        None,
        "--overlay-join-keys",
        help="Comma-separated join keys. Default: ticker,trade_date",
    ),
    hold_bars_grid: str | None = typer.Option(None, "--hold-bars-grid"),
    signal_mode_grid: str | None = typer.Option(None, "--signal-mode-grid"),
    exit_mode_grid: str | None = typer.Option(None, "--exit-mode-grid"),
    fee_bps_grid: str | None = typer.Option(None, "--fee-bps-grid"),
    slippage_bps_grid: str | None = typer.Option(None, "--slippage-bps-grid"),
    allow_overlap_grid: str | None = typer.Option(None, "--allow-overlap-grid"),
    equity_mode_grid: str | None = typer.Option(None, "--equity-mode-grid"),
    include_watch_grid: str | None = typer.Option(None, "--include-watch-grid"),
    include_state_ids_grid: str | None = typer.Option(None, "--include-state-ids-grid"),
    train_ends: str | None = typer.Option(None, "--train-ends", help="Comma-separated train-end dates."),
    train_start: str | None = typer.Option(None, "--train-start", help="Schedule mode: start YYYY-MM-DD."),
    train_end_final: str | None = typer.Option(None, "--train-end-final", help="Schedule mode: end YYYY-MM-DD."),
    step_years: int | None = typer.Option(None, "--step-years", min=1),
    sources: str | None = typer.Option(None, "--sources", help="Comma-separated sources: hmm,flow,cluster"),
    policy_filter_mode: str | None = typer.Option(
        None,
        "--policy-filter-mode",
        help="Cluster policy filter mode: allow_only, allow_watch, all_states.",
    ),
    min_successful_splits: int | None = typer.Option(None, "--min-successful-splits", min=1),
    report_top_n: int | None = typer.Option(None, "--report-top-n", min=1),
    max_combos: int | None = typer.Option(None, "--max-combos", min=1),
    shuffle_grid: bool = typer.Option(False, "--shuffle-grid"),
    seed: int = typer.Option(42, "--seed"),
    progress_every: int | None = typer.Option(None, "--progress-every", min=1),
    stop_on_error: bool = typer.Option(False, "--stop-on-error"),
    force: bool = typer.Option(False, "--force"),
    tag: str | None = typer.Option(None, "--tag"),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run backtest sensitivity grid across a validation walk-forward pack."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    policy_filter_mode_norm = _normalize_choice(
        policy_filter_mode,
        allowed={"allow_only", "allow_watch", "all_states"},
        option_name="policy-filter-mode",
    ) or settings.backtest_sensitivity.policy_filter_mode_default
    overlay_mode_norm = _normalize_choice(
        overlay_mode,
        allowed={"none", "allow_only", "allow_watch", "block_veto", "allow_or_unknown"},
        option_name="overlay-mode",
    ) or settings.backtest_policy_overlay.default_overlay_mode
    overlay_join_keys_norm = (
        [part.strip() for part in overlay_join_keys.split(",") if part.strip() != ""]
        if overlay_join_keys
        else list(settings.backtest_policy_overlay.join_keys)
    )
    if (overlay_cluster_file is None) ^ (overlay_cluster_hardening_dir is None):
        raise typer.BadParameter(
            "Provide both --overlay-cluster-file and --overlay-cluster-hardening-dir, or neither."
        )
    if overlay_mode_norm != "none" and (overlay_cluster_file is None or overlay_cluster_hardening_dir is None):
        raise typer.BadParameter(
            "overlay-mode is not none but overlay inputs are missing. Provide both overlay inputs."
        )
    sources_norm: list[str] | None = None
    if sources:
        sources_norm = _parse_choice_csv(sources, "sources", {"hmm", "flow", "cluster"})
    train_ends_norm: list[str] | None = None
    if train_ends:
        train_ends_norm = [d.isoformat() for d in _parse_date_csv(train_ends, "train-ends")]
    dims = _resolve_sensitivity_dimensions(
        settings,
        hold_bars_grid=hold_bars_grid,
        signal_mode_grid=signal_mode_grid,
        exit_mode_grid=exit_mode_grid,
        fee_bps_grid=fee_bps_grid,
        slippage_bps_grid=slippage_bps_grid,
        allow_overlap_grid=allow_overlap_grid,
        equity_mode_grid=equity_mode_grid,
        include_watch_grid=include_watch_grid,
        include_state_ids_grid=include_state_ids_grid,
    )
    result = run_backtest_grid_walkforward(
        settings,
        wf_run_dir=wf_run_dir,
        flow_dataset_file=flow_dataset_file,
        overlay_cluster_file=overlay_cluster_file,
        overlay_cluster_hardening_dir=overlay_cluster_hardening_dir,
        overlay_mode=cast(str, overlay_mode_norm),
        overlay_join_keys=overlay_join_keys_norm,
        train_ends=train_ends_norm,
        train_start=train_start,
        train_end_final=train_end_final,
        step_years=step_years,
        sources=sources_norm,
        policy_filter_mode=cast(str, policy_filter_mode_norm),
        dimensions=dims,
        max_combos=max_combos,
        shuffle_grid=shuffle_grid,
        seed=seed,
        progress_every=progress_every,
        stop_on_error=stop_on_error,
        force=force,
        tag=tag,
        min_successful_splits=(
            min_successful_splits
            if min_successful_splits is not None
            else settings.backtest_sensitivity.min_successful_splits_default
        ),
        report_top_n=(
            report_top_n if report_top_n is not None else settings.backtest_sensitivity.report_top_n_default
        ),
        logger=logger,
    )
    summary = json.loads((result.output_dir / "wf_grid_summary.json").read_text(encoding="utf-8"))
    typer.echo(f"wf_grid_id: {summary.get('wf_grid_id')}")
    typer.echo(f"splits_total: {summary.get('splits_total')}")
    typer.echo(f"splits_successful: {summary.get('splits_successful')}")
    typer.echo(f"splits_failed: {summary.get('splits_failed')}")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"manifest_path: {result.manifest_path}")
    typer.echo(f"by_split_path: {result.by_split_path}")
    typer.echo(f"config_aggregate_path: {result.config_aggregate_path}")
    typer.echo(f"source_summary_path: {result.source_summary_path}")
    typer.echo(f"source_dimension_summary_path: {result.output_dir / 'wf_grid_source_dimension_summary.csv'}")
    typer.echo(f"config_consistency_path: {result.output_dir / 'wf_grid_config_consistency.csv'}")
    typer.echo(f"winner_stability_path: {result.output_dir / 'wf_grid_winner_stability.csv'}")
    typer.echo(f"cost_fragility_summary_path: {result.output_dir / 'wf_grid_cost_fragility_summary.csv'}")
    typer.echo(f"tail_risk_summary_path: {result.output_dir / 'wf_grid_tail_risk_summary.csv'}")
    typer.echo(f"overlay_split_summary_path: {result.output_dir / 'wf_overlay_split_summary.csv'}")
    typer.echo(f"overlay_source_summary_path: {result.output_dir / 'wf_overlay_source_summary.csv'}")
    typer.echo(f"overlay_effectiveness_summary_path: {result.output_dir / 'wf_overlay_effectiveness_summary.csv'}")
    typer.echo(f"report_path: {result.report_path}")


@app.command("hybrid-eval-report")
def hybrid_eval_report(
    hmm_baseline_grid_dir: Path = typer.Option(
        Path("/home/tom/projects/mf_etl/artifacts/backtest_sensitivity/grid-8e70f7c58ff4_single-hmm_default"),
        "--hmm-baseline-grid-dir",
        help="HMM baseline grid run directory.",
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    hmm_allow_only_grid_dir: Path = typer.Option(
        Path("/home/tom/projects/mf_etl/artifacts/backtest_sensitivity/grid-177aac2f15c8_single-hmm_default"),
        "--hmm-allow-only-grid-dir",
        help="HMM overlay allow_only grid run directory.",
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    hmm_block_veto_grid_dir: Path = typer.Option(
        Path("/home/tom/projects/mf_etl/artifacts/backtest_sensitivity/grid-50e9f3fede69_single-hmm_default"),
        "--hmm-block-veto-grid-dir",
        help="HMM overlay block_veto grid run directory.",
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    flow_allow_only_grid_dir: Path | None = typer.Option(
        Path("/home/tom/projects/mf_etl/artifacts/backtest_sensitivity/grid-36984dda7a77_single-flow_default"),
        "--flow-allow-only-grid-dir",
        help="Optional FLOW overlay allow_only grid run directory comparator.",
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    wf_hmm_baseline_dir: Path | None = typer.Option(
        Path("/home/tom/projects/mf_etl/artifacts/backtest_sensitivity_walkforward/wfgrid-34967fd30e99"),
        "--wf-hmm-baseline-dir",
        help="Optional HMM baseline walk-forward grid run directory.",
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    wf_hmm_hybrid_dir: Path | None = typer.Option(
        Path("/home/tom/projects/mf_etl/artifacts/backtest_sensitivity_walkforward/wfgrid-33285ab4fc2c"),
        "--wf-hmm-hybrid-dir",
        help="Optional HMM hybrid walk-forward grid run directory.",
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    compare_run_dir: Path | None = typer.Option(
        Path("/home/tom/projects/mf_etl/artifacts/backtest_sensitivity/backtest-grid-compare-4d0688d80ee8"),
        "--compare-run-dir",
        help="Optional existing backtest-grid-compare run directory.",
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="Optional settings YAML path.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Build Hybrid Overlay Evaluation Report v1 from existing grid/WF artifacts."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    result = run_hybrid_eval_report(
        settings,
        hmm_baseline_grid_dir=hmm_baseline_grid_dir,
        hmm_allow_only_grid_dir=hmm_allow_only_grid_dir,
        hmm_block_veto_grid_dir=hmm_block_veto_grid_dir,
        flow_allow_only_grid_dir=flow_allow_only_grid_dir,
        wf_hmm_baseline_dir=wf_hmm_baseline_dir,
        wf_hmm_hybrid_dir=wf_hmm_hybrid_dir,
        compare_run_dir=compare_run_dir,
        logger=logger,
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    table_df = pl.read_csv(result.table_path)
    if "single_candidate_score" in table_df.columns:
        table_df = table_df.sort("single_candidate_score", descending=True, nulls_last=True)
    compact_cols = [
        col
        for col in [
            "run_label",
            "overlay_mode",
            "single_candidate_score",
            "best_expectancy",
            "best_pf",
            "best_robustness_v2",
            "best_ret_cv",
            "best_trade_count",
            "overlay_vetoed_signal_share",
        ]
        if col in table_df.columns
    ]
    compact_df = table_df.select(compact_cols).head(6) if compact_cols else table_df.head(6)
    final = summary.get("final_verdicts", {})

    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(
        "recommendation: "
        f"PRIMARY={final.get('PRIMARY_CANDIDATE')} "
        f"SECONDARY={final.get('SECONDARY_CANDIDATE')} "
        f"NEXT_EXPERIMENT={final.get('NEXT_EXPERIMENT')}"
    )
    typer.echo("top_runs:")
    for row in compact_df.to_dicts():
        typer.echo(
            f"run={row.get('run_label')} mode={row.get('overlay_mode')} score={row.get('single_candidate_score')} "
            f"exp={row.get('best_expectancy')} pf={row.get('best_pf')} rob_v2={row.get('best_robustness_v2')} "
            f"ret_cv={row.get('best_ret_cv')} trades={row.get('best_trade_count')} "
            f"veto_share={row.get('overlay_vetoed_signal_share')}"
        )
    typer.echo(f"summary_path: {result.summary_path}")
    typer.echo(f"table_path: {result.table_path}")
    typer.echo(f"wf_table_path: {result.wf_table_path}")
    typer.echo(f"report_path: {result.report_path}")


def main() -> None:
    """CLI script entrypoint."""

    app()


if __name__ == "__main__":
    main()
