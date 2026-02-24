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

## Bronze QA Utilities

- Build symbol master:
  - `python -m mf_etl.cli build-symbol-master`
  - Writes:
    - `data/bronze/symbol_master/symbol_master.parquet`
    - `data/bronze/symbol_master/symbol_master.csv`

- Run global sanity checks:
  - `python -m mf_etl.cli sanity-checks`
  - Writes:
    - `artifacts/bronze_qa/bronze_sanity_summary.json`
    - `artifacts/bronze_qa/bronze_sanity_by_exchange.parquet` (and csv)
    - `artifacts/bronze_qa/bronze_rows_by_year.parquet` (and csv)

- List problematic tickers:
  - `python -m mf_etl.cli list-problem-tickers --limit 50`
  - `python -m mf_etl.cli list-problem-tickers --only-invalid`

## Silver Base Layer

The Silver base layer builds research-ready per-symbol helper series on top of Bronze valid rows.
It includes:

- Identity/context columns (`ticker`, `exchange`, `trade_date`, `trade_dt`, `source_file`, `run_id`)
- Core market columns (`open`, `high`, `low`, `close`, `volume`, `openint`)
- Base helper features for future TMF/TTI/event-grammar work:
  - price geometry
  - return/gap features
  - range/ATR helpers
  - volume/liquidity helpers
  - rolling context features
  - warmup/readiness flags

Silver outputs are written per ticker to:

- `data/silver/base_series_by_symbol/exchange=<EXCHANGE>/prefix=<LETTER>/ticker=<TICKER>/part-000.parquet`

Run commands:

- `python -m mf_etl.cli silver-one --ticker AAPL.US`
- `python -m mf_etl.cli silver-one --bronze-file /abs/path/to/data/bronze/ohlcv_by_symbol/.../part-000.parquet`
- `python -m mf_etl.cli silver-run --limit 10`
- `python -m mf_etl.cli silver-run`
- `python -m mf_etl.cli silver-sanity`

Silver run artifacts:

- `artifacts/silver_run_summaries/<run_id>_silver_run_summary.json`
- `artifacts/silver_run_summaries/<run_id>_silver_ticker_results.parquet`

This base layer is the foundation for future TMF/TTI, event grammar, and downstream research/ML pipelines.
