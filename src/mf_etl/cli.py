"""Typer CLI entrypoint for mf_etl."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
import yaml

from mf_etl.config import AppSettings, load_settings
from mf_etl.ingest.discover import discover_txt_files
from mf_etl.ingest.manifest import build_manifest, write_manifest_parquet
from mf_etl.logging_utils import configure_logging
from mf_etl.silver.placeholders import ensure_silver_placeholder
from mf_etl.gold.placeholders import ensure_gold_placeholder
from mf_etl.utils.paths import ensure_directories

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


def main() -> None:
    """CLI script entrypoint."""

    app()


if __name__ == "__main__":
    main()
