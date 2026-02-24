# mf_etl

Production-style Python ETL skeleton for local daily stock data (NYSE + NASDAQ).

## Stack

- Python 3.11+
- Polars
- PyArrow
- DuckDB
- pydantic-settings
- Typer
- stdlib logging

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run CLI commands:

```bash
python -m mf_etl.cli show-config
python -m mf_etl.cli bronze-run --dry-run
python -m mf_etl.cli bronze-run --limit 10
python -m mf_etl.cli bronze-run
python -m mf_etl.cli init-placeholders
```

## Config

Default settings are defined in `configs/settings.yaml`.

Environment variable overrides use prefix `MF_ETL_` with nested keys via `__`.
Examples:

- `MF_ETL_PROJECT__ENV=prod`
- `MF_ETL_PATHS__RAW_ROOT=/media/tom/Hdd_240GB/data`
- `MF_ETL_PARQUET__COMPRESSION=zstd`

You can also override the settings file path with:

- `MF_ETL_SETTINGS_FILE=/path/to/settings.yaml`

## Data layout

After `init-placeholders`, the project layout includes:

- `data/bronze`
- `data/silver`
- `data/gold`
- `artifacts`
- `logs/etl.log`

## Bronze Incremental Behavior

- `bronze-run` writes a classified current manifest to `data/bronze/manifests/file_manifest_current.parquet`.
- Default mode processes only `NEW` and `CHANGED` files (by `source_file` + `fingerprint`).
- `--full` processes all files, including `UNCHANGED`.
- `--dry-run` classifies and writes current manifest but does not process files.
- Stable manifest promotion to `data/bronze/manifests/file_manifest.parquet` occurs only for non-dry, non-limited, non-filtered runs.
- Per-file processing failures are logged and captured in run summary artifacts, while the run continues.
- Stable manifest is still promoted when such a full run reaches completion, even with per-file failures.
