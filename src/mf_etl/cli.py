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
from mf_etl.validation.pipeline import run_validation_compare, run_validation_harness
from mf_etl.validation.sanity import summarize_validation_run
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


def _parse_method_csv(value: str) -> list[str]:
    methods = [part.strip().lower() for part in value.split(",") if part.strip() != ""]
    if not methods:
        raise typer.BadParameter("methods must contain at least one method.")
    allowed = {"kmeans", "gmm", "hdbscan"}
    for method in methods:
        if method not in allowed:
            raise typer.BadParameter("methods must be comma-separated values from: kmeans,gmm,hdbscan")
    return methods


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


def main() -> None:
    """CLI script entrypoint."""

    app()


if __name__ == "__main__":
    main()
