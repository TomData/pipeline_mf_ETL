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

## Gold Features v1

Gold Features v1 transforms Event Grammar outputs into compact, numeric-heavy features for clustering/HMM and backtests.
It includes:

- TMF/TTI level and slope features
- weighted flow scores (`long_flow_score_*`, `short_flow_score_*`, `delta_flow_*`, `flow_bias_20`)
- burst/persistence/oscillation features
- bars-since recency transforms
- state transition features (`state_prev`, `state_changed`, `state_run_length`, `state_transition_code`, `bs_state_change`)

Per-symbol feature outputs:

- `data/gold/features_by_symbol/exchange=<EXCHANGE>/prefix=<LETTER>/ticker=<TICKER>/part-000.parquet`

Run commands:

- `python -m mf_etl.cli features-one --ticker AAPL.US`
- `python -m mf_etl.cli features-one --events-file /abs/path/to/data/gold/events_by_symbol/.../part-000.parquet`
- `python -m mf_etl.cli features-run --limit 10`
- `python -m mf_etl.cli features-run`
- `python -m mf_etl.cli features-sanity`

Feature artifacts:

- `artifacts/gold_feature_run_summaries/<run_id>_features_run_summary.json`
- `artifacts/gold_feature_run_summaries/<run_id>_features_ticker_results.parquet`
- `artifacts/gold_feature_qa/features_sanity_summary.json`

Dataset export helper:

- `python -m mf_etl.cli export-ml-dataset --symbols-limit 10`
- optional filters:
  - `--start-date YYYY-MM-DD`
  - `--end-date YYYY-MM-DD`
  - `--sample-frac F`

Exported datasets are written to:

- `data/gold/datasets/ml_dataset_v1/<run_id>/dataset.parquet`
- `data/gold/datasets/ml_dataset_v1/<run_id>/metadata.json`

## Research Baseline v1

Research Baseline v1 provides reproducible unsupervised state discovery on top of exported Gold feature datasets.
Pipeline components:

- dataset loading with optional filters/sampling
- preprocessing (feature selection, null filtering, scaling, clipping)
- clustering:
  - KMeans baseline
  - Gaussian Mixture baseline
  - optional HDBSCAN (if installed)
- cluster profiling and forward-return validation

Core commands:

- `python -m mf_etl.cli research-cluster-run --dataset /abs/path/dataset.parquet --method kmeans --n-clusters 5`
- `python -m mf_etl.cli research-cluster-run --dataset /abs/path/dataset.parquet --method gmm --n-clusters 5`
- `python -m mf_etl.cli research-cluster-run --dataset /abs/path/dataset.parquet --method kmeans --n-clusters 5 --split-mode time --train-end 2020-12-31`
- `python -m mf_etl.cli research-cluster-run --dataset /abs/path/dataset.parquet --method kmeans --n-clusters 5 --scaling-scope per_ticker`
- `python -m mf_etl.cli research-cluster-sweep --dataset /abs/path/dataset.parquet`
- `python -m mf_etl.cli research-cluster-stability --dataset /abs/path/dataset.parquet --method kmeans --n-clusters 5 --seeds 10`
- `python -m mf_etl.cli research-cluster-sanity --run-dir /abs/path/to/artifacts/research_runs/<run_dir>`

Run artifacts:

- `artifacts/research_runs/<run_id>_<method>_<dataset_tag>/run_summary.json`
- `artifacts/research_runs/<run_id>_<method>_<dataset_tag>/preprocess_summary.json`
- `artifacts/research_runs/<run_id>_<method>_<dataset_tag>/clustering_metrics.json`
- `artifacts/research_runs/<run_id>_<method>_<dataset_tag>/cluster_profile.parquet`
- `artifacts/research_runs/<run_id>_<method>_<dataset_tag>/cluster_profile.csv`
- `artifacts/research_runs/<run_id>_<method>_<dataset_tag>/clustered_dataset_sample.parquet`
- `artifacts/research_runs/<run_id>_<method>_<dataset_tag>/split_summary.json` (when split mode is used)
- `artifacts/research_runs/<run_id>_<method>_<dataset_tag>/robustness_summary.json`
- `artifacts/research_runs/<run_id>_cluster_sweep_summary.csv`
- `artifacts/research_runs/<run_id>_<method>_<dataset_tag>/stability_summary.json`
- `artifacts/research_runs/<run_id>_<method>_<dataset_tag>/stability_pairwise_ari.csv`

Cluster profiles include forward-return validation columns (`fwd_ret_5/10/20` means/medians/hit rates) to evaluate separation before moving to sequential/HMM modeling.
Forward validation aggregates are computed using finite-value-only inputs (NaN/inf are normalized to null before aggregation) for QA consistency.

### Robustness Workflow

- Time OOS split: use `--split-mode time --train-end YYYY-MM-DD` to fit scaler/model on train and evaluate profiles on test.
- Scaling scope:
  - `--scaling-scope global` applies one scaler to all symbols.
  - `--scaling-scope per_ticker` fits scaler parameters per ticker from fit rows and applies them to prediction rows.
- Seed stability: run `research-cluster-stability` to compute pairwise ARI across seeds.
- Recommended pre-HMM workflow:
  1. baseline `research-cluster-run`
  2. `research-cluster-sweep` for K/metric tradeoffs
  3. `research-cluster-stability` for ARI robustness
  4. OOS rerun with `--split-mode time`
  5. compare train/test profiles and forward-return validation

## HMM Baseline v1

HMM Baseline v1 adds sequential latent-state modeling on top of Gold feature datasets.
Unlike clustering, HMM explicitly models temporal transitions and persistence.

Core commands:

- `python -m mf_etl.cli research-hmm-run --dataset /abs/path/dataset.parquet --n-components 5`
- `python -m mf_etl.cli research-hmm-run --dataset /abs/path/dataset.parquet --n-components 5 --split-mode time --train-end 2018-12-31 --scaling-scope per_ticker`
- `python -m mf_etl.cli research-hmm-sweep --dataset /abs/path/dataset.parquet --components 4,5,6,8`
- `python -m mf_etl.cli research-hmm-sanity --run-dir /abs/path/to/artifacts/hmm_runs/<run_dir>`
- `python -m mf_etl.cli research-hmm-stability --dataset /abs/path/dataset.parquet --n-components 5 --seeds 5`

Run artifacts (per HMM run):

- `artifacts/hmm_runs/<run_id>_hmm_<dataset_tag>/run_summary.json`
- `artifacts/hmm_runs/<run_id>_hmm_<dataset_tag>/split_summary.json`
- `artifacts/hmm_runs/<run_id>_hmm_<dataset_tag>/hmm_model_meta.json`
- `artifacts/hmm_runs/<run_id>_hmm_<dataset_tag>/decoded_rows.parquet`
- `artifacts/hmm_runs/<run_id>_hmm_<dataset_tag>/hmm_state_profile.parquet`
- `artifacts/hmm_runs/<run_id>_hmm_<dataset_tag>/transition_matrix.csv`
- `artifacts/hmm_runs/<run_id>_hmm_<dataset_tag>/transition_counts.csv`
- `artifacts/hmm_runs/<run_id>_hmm_<dataset_tag>/dwell_stats.csv`
- `artifacts/hmm_runs/<run_id>_hmm_<dataset_tag>/state_frequency.csv`
- `artifacts/hmm_runs/<run_id>_hmm_<dataset_tag>/hmm_vs_flow_state_crosstab.csv`
- `artifacts/hmm_runs/<run_id>_hmm_<dataset_tag>/robustness_summary.json`

Recommended sequence before advanced sequential modeling:

1. clustering baseline + robustness sweep
2. HMM baseline run
3. compare HMM states vs deterministic flow states and cluster labels
4. iterate feature set/event grammar and rerun HMM sweep/stability
