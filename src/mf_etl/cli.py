"""Typer CLI entrypoint for mf_etl."""

from __future__ import annotations

import json
import logging
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


def main() -> None:
    """CLI script entrypoint."""

    app()


if __name__ == "__main__":
    main()
