"""Typer CLI entrypoint for mf_etl."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

import polars as pl
import typer
import yaml

from mf_etl.config import AppSettings, load_settings
from mf_etl.ingest.discover import discover_txt_files, infer_exchange_from_path
from mf_etl.ingest.manifest import build_manifest, write_manifest_parquet
from mf_etl.ingest.read_txt import read_stock_txt_with_rejects
from mf_etl.logging_utils import configure_logging
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
    """Placeholder Bronze pipeline run."""

    settings, logger = _load_and_optionally_configure_logger(config_file, configure=True)
    logger.info("bronze_run.start project=%s env=%s", settings.project.name, settings.project.env)
    logger.info(
        "bronze_run.paths raw_root=%s bronze_root=%s logs_root=%s",
        settings.paths.raw_root,
        settings.paths.bronze_root,
        settings.paths.logs_root,
    )
    logger.info(
        "bronze_run.parquet compression=%s level=%s statistics=%s",
        settings.parquet.compression,
        settings.parquet.compression_level,
        settings.parquet.statistics,
    )
    logger.info("bronze_run.complete status=placeholder")
    typer.echo("Bronze run placeholder executed. See logs/etl.log for details.")


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


def main() -> None:
    """CLI script entrypoint."""

    app()


if __name__ == "__main__":
    main()
