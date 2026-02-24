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

## Indicator Layer (TMF + TTI Proxy)

The first indicator layer builds on Silver base series and writes per-symbol indicator artifacts:

- TMF v1 uses the public Twiggs-style true-range AD formulation with Wilder-style smoothing.
- Twiggs Trend Index exact formula is proprietary; this project intentionally exposes a versioned proxy:
  - `tti_proxy_v1_21`
  - `tti_formula_status = PROXY_UNDISCLOSED_ORIGINAL`
  - `tti_proxy_version = v1`

Indicator outputs are written to:

- `data/silver/indicators_by_symbol/exchange=<EXCHANGE>/prefix=<LETTER>/ticker=<TICKER>/part-000.parquet`

Run commands:

- `python -m mf_etl.cli indicators-one --ticker AAPL.US`
- `python -m mf_etl.cli indicators-one --silver-file /abs/path/to/data/silver/base_series_by_symbol/.../part-000.parquet`
- `python -m mf_etl.cli indicators-run --limit 10`
- `python -m mf_etl.cli indicators-run`
- `python -m mf_etl.cli indicators-sanity`

Indicator artifacts:

- `artifacts/indicator_run_summaries/<run_id>_indicators_run_summary.json`
- `artifacts/indicator_run_summaries/<run_id>_indicators_ticker_results.parquet`
- `artifacts/indicator_qa/indicator_sanity_summary.json`

## Gold Event Grammar v1

Gold Event Grammar v1 converts indicator series into deterministic per-bar event/state columns for research:

- TMF zero-line, pivot, respect/failure, burst, and hold events
- TTI proxy zero/burst/hold events
- bars-since counters and rolling event activity/asymmetry features
- MVP deterministic state coding:
  - `S0_QUIET`
  - `S1_EARLY_DEMAND`
  - `S2_PERSISTENT_DEMAND`
  - `S3_EARLY_SUPPLY`
  - `S4_PERSISTENT_SUPPLY`

This S0-S4 coding is a seed state model for downstream clustering/HMM work, not a final ontology.

Gold outputs are written to:

- `data/gold/events_by_symbol/exchange=<EXCHANGE>/prefix=<LETTER>/ticker=<TICKER>/part-000.parquet`

Run commands:

- `python -m mf_etl.cli events-one --ticker AAPL.US`
- `python -m mf_etl.cli events-one --indicator-file /abs/path/to/data/silver/indicators_by_symbol/.../part-000.parquet`
- `python -m mf_etl.cli events-run --limit 10`
- `python -m mf_etl.cli events-run`
- `python -m mf_etl.cli events-sanity`

Gold event artifacts:

- `artifacts/gold_event_run_summaries/<run_id>_events_run_summary.json`
- `artifacts/gold_event_run_summaries/<run_id>_events_ticker_results.parquet`
- `artifacts/gold_event_qa/events_sanity_summary.json`

These outputs feed the next stage: Gold feature sets for regime clustering, HMM, and backtest research.
