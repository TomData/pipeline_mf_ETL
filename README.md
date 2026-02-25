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

## Validation Harness v1

Validation Harness v1 adds a reusable statistical validation layer for any state-labeled dataset:

- HMM decoded rows (`hmm_state`)
- cluster-labeled rows (`cluster_id` / `cluster_label`)
- generic state labels (for example `flow_state_code`)

It provides:

- bootstrap confidence intervals for forward outcomes
- pairwise state-difference confidence checks
- transition event studies around state changes
- rolling-window stability diagnostics
- per-state and overall validation scorecards

Core commands:

- `python -m mf_etl.cli validation-run --input-file /abs/path/to/decoded_rows.parquet --input-type hmm`
- `python -m mf_etl.cli validation-run --input-file /abs/path/to/dataset.parquet --input-type generic --state-col flow_state_code`
- `python -m mf_etl.cli validation-sanity --run-dir /abs/path/to/artifacts/validation_runs/<run_dir>`
- `python -m mf_etl.cli validation-compare --run-dir-a <run_a> --run-dir-b <run_b>`

Primary artifacts (per run):

- `artifacts/validation_runs/<run_id>_<input_type>_<state_tag>/run_summary.json`
- `artifacts/validation_runs/<run_id>_<input_type>_<state_tag>/adapter_summary.json`
- `artifacts/validation_runs/<run_id>_<input_type>_<state_tag>/bootstrap_state_summary.csv`
- `artifacts/validation_runs/<run_id>_<input_type>_<state_tag>/bootstrap_pairwise_diff.csv`
- `artifacts/validation_runs/<run_id>_<input_type>_<state_tag>/transition_event_summary.csv`
- `artifacts/validation_runs/<run_id>_<input_type>_<state_tag>/transition_event_path_summary.csv`
- `artifacts/validation_runs/<run_id>_<input_type>_<state_tag>/rolling_state_metrics.csv`
- `artifacts/validation_runs/<run_id>_<input_type>_<state_tag>/state_stability_summary.csv`
- `artifacts/validation_runs/<run_id>_<input_type>_<state_tag>/state_scorecard.csv`
- `artifacts/validation_runs/<run_id>_<input_type>_<state_tag>/validation_scorecard.json`

Forward-outcome aggregation is finite-only: non-finite values are normalized to null before aggregation, and QA guards enforce consistency for counts vs mean/median outputs.

## Validation Walk-Forward + Cluster QA

For higher statistical power, run multi-split OOS validation and aggregate results:

- `python -m mf_etl.cli validation-wf-run --dataset /abs/path/to/dataset.parquet`
- `python -m mf_etl.cli validation-wf-run --dataset /abs/path/to/dataset.parquet --train-end-list 2014-12-31,2016-12-31,2018-12-31,2020-12-31 --bootstrap-mode block --block-length 10 --min-events-per-transition 10`
- `python -m mf_etl.cli validation-wf-sanity --wf-run-dir /abs/path/to/artifacts/validation_walkforward/<wf_run_id>`

Walk-forward artifacts:

- `artifacts/validation_walkforward/<wf_run_id>/wf_manifest.json`
- `artifacts/validation_walkforward/<wf_run_id>/wf_split_runs.csv`
- `artifacts/validation_walkforward/<wf_run_id>/wf_model_summary_long.csv`
- `artifacts/validation_walkforward/<wf_run_id>/wf_model_summary_wide.csv`
- `artifacts/validation_walkforward/<wf_run_id>/wf_comparison_summary.csv`
- `artifacts/validation_walkforward/<wf_run_id>/wf_aggregate_summary.json`
- `artifacts/validation_walkforward/<wf_run_id>/wf_full_report.md`

For unstable cluster diagnostics (for example very high `ret_cv`), run:

- `python -m mf_etl.cli cluster-qa-run --validation-run-dir /abs/path/to/artifacts/validation_runs/<cluster_validation_run>`
- `python -m mf_etl.cli cluster-qa-run --wf-run-dir /abs/path/to/artifacts/validation_walkforward/<wf_run_id>`

Cluster QA artifacts:

- single-run: `cluster_qa_summary.json`, `cluster_qa_flagged_states.csv`, `cluster_qa_state_windows.csv`, `cluster_qa_report.md`
- walk-forward: `cluster_qa_wf_summary.json`, `cluster_qa_wf_flagged_states.csv`, `cluster_qa_issue_frequency.csv`, `cluster_qa_wf_report.md`

## Cluster Hardening & Tradable-State Filter v1

Cluster hardening turns cluster validation outputs into deterministic state policies:

- class labels per state: `ALLOW`, `WATCH`, `BLOCK`
- explicit reasons and QA-derived penalties
- optional filtered row exports for tradable-state backtests

Commands:

- single-run policy:
  - `python -m mf_etl.cli cluster-hardening-run --validation-run-dir /abs/path/to/artifacts/validation_runs/<cluster_validation_dir>`
- single-run with filtered exports:
  - `python -m mf_etl.cli cluster-hardening-run --validation-run-dir /abs/path/to/artifacts/validation_runs/<cluster_validation_dir> --clustered-rows-file /abs/path/to/clustered_dataset_full.parquet --export-filtered`
- walk-forward aggregation:
  - `python -m mf_etl.cli cluster-hardening-run --wf-run-dir /abs/path/to/artifacts/validation_walkforward/<wf_run_id>`
- sanity:
  - `python -m mf_etl.cli cluster-hardening-sanity --hardening-dir /abs/path/to/hardening_dir`
- compare two policies:
  - `python -m mf_etl.cli cluster-hardening-compare --hardening-dir-a /abs/path/to/hardening_a --hardening-dir-b /abs/path/to/hardening_b`

Single-run artifacts:

- `cluster_hardening_policy.json`
- `cluster_hardening_state_table.csv`
- `cluster_hardening_summary.json`
- `cluster_hardening_report.md`
- optional exports under `exports/`:
  - `clustered_rows_with_policy.parquet`
  - `clustered_rows_tradable.parquet`
  - `clustered_rows_watch.parquet`
  - `cluster_hardening_export_summary.json`
  - `cluster_hardening_export_by_state.csv`

Walk-forward artifacts:

- `cluster_hardening_wf_summary.json`
- `cluster_hardening_wf_state_stats.csv`
- `cluster_hardening_wf_split_counts.csv`
- `cluster_hardening_wf_issue_frequency.csv`
- `cluster_hardening_threshold_recommendation.json`
- `cluster_hardening_wf_report.md`

## Backtest Harness v1

Backtest Harness v1 provides deterministic, execution-oriented research baselines for:

- FLOW (`flow_state_code`)
- HMM (`hmm_state`)
- CLUSTER (`cluster_id`, with cluster hardening policy; ALLOW-only by default)

Core assumptions (MVP):

- D1 only
- EOD signal, next-bar execution
- no intraday fill modeling
- no portfolio optimizer
- no position sizing optimization
- slippage/fees default to zero unless configured

Commands:

- `python -m mf_etl.cli backtest-run --input-type flow --input-file /abs/path/dataset.parquet`
- `python -m mf_etl.cli backtest-run --input-type hmm --input-file /abs/path/decoded_rows.parquet --validation-run-dir /abs/path/to/hmm_validation_run`
- `python -m mf_etl.cli backtest-run --input-type cluster --input-file /abs/path/clustered_dataset_full.parquet --cluster-hardening-dir /abs/path/to/cluster_hardening`
- `python -m mf_etl.cli backtest-sanity --run-dir /abs/path/to/artifacts/backtest_runs/<run_dir>`
- `python -m mf_etl.cli backtest-compare --run-dir <run_a> --run-dir <run_b> --run-dir <run_c>`
- `python -m mf_etl.cli backtest-wf-run --wf-run-dir /abs/path/to/artifacts/validation_walkforward/<wf_run_id> --flow-dataset-file /abs/path/dataset.parquet`

Single-run artifacts:

- `artifacts/backtest_runs/<run_id>_<source>_<tag>/backtest_run_config.json`
- `artifacts/backtest_runs/<run_id>_<source>_<tag>/backtest_summary.json`
- `artifacts/backtest_runs/<run_id>_<source>_<tag>/trades.parquet`
- `artifacts/backtest_runs/<run_id>_<source>_<tag>/summary_by_state.csv`
- `artifacts/backtest_runs/<run_id>_<source>_<tag>/summary_by_symbol.csv`
- `artifacts/backtest_runs/<run_id>_<source>_<tag>/signal_diagnostics.json`
- `artifacts/backtest_runs/<run_id>_<source>_<tag>/backtest_report.md`

Walk-forward artifacts:

- `artifacts/backtest_walkforward/<wf_bt_id>/wf_backtest_manifest.json`
- `artifacts/backtest_walkforward/<wf_bt_id>/wf_backtest_aggregate_summary.json`
- `artifacts/backtest_walkforward/<wf_bt_id>/wf_backtest_by_split.csv`
- `artifacts/backtest_walkforward/<wf_bt_id>/wf_backtest_model_summary.csv`
- `artifacts/backtest_walkforward/<wf_bt_id>/wf_backtest_report.md`

## Backtest Sensitivity Pack v1

Backtest Sensitivity Pack v1 orchestrates parameter-grid experiments on top of `backtest-run` to test robustness across hold horizon, signal/exit modes, and cost assumptions.

Commands:

- `python -m mf_etl.cli backtest-grid-run --input-type flow --input-file /abs/path/dataset.parquet --hold-bars-grid \"5,10,20\" --fee-bps-grid \"0,10\"`
- `python -m mf_etl.cli backtest-grid-run --multi-source --flow-input-file /abs/path/dataset.parquet --hmm-input-file /abs/path/decoded_rows.parquet --cluster-input-file /abs/path/clustered_dataset_full.parquet --validation-run-dir /abs/path/hmm_validation --cluster-hardening-dir /abs/path/cluster_hardening`
- `python -m mf_etl.cli backtest-grid-sanity --grid-run-dir /abs/path/to/artifacts/backtest_sensitivity/<grid_run_dir>`
- `python -m mf_etl.cli backtest-grid-compare --grid-run-dir <grid_a> --grid-run-dir <grid_b> --grid-run-dir <grid_c>`
- `python -m mf_etl.cli backtest-grid-wf-run --wf-run-dir /abs/path/to/artifacts/validation_walkforward/<wf_run_id> --flow-dataset-file /abs/path/dataset.parquet`

Grid artifacts:

- `artifacts/backtest_sensitivity/grid-<id>_<scope>_<tag>/grid_run_config.json`
- `artifacts/backtest_sensitivity/grid-<id>_<scope>_<tag>/grid_manifest.parquet`
- `artifacts/backtest_sensitivity/grid-<id>_<scope>_<tag>/grid_metrics_table.parquet`
- `artifacts/backtest_sensitivity/grid-<id>_<scope>_<tag>/grid_dimension_sensitivity.parquet`
- `artifacts/backtest_sensitivity/grid-<id>_<scope>_<tag>/grid_summary.json`
- `artifacts/backtest_sensitivity/grid-<id>_<scope>_<tag>/grid_report.md`

Walk-forward grid artifacts:

- `artifacts/backtest_sensitivity_walkforward/wfgrid-<id>/wf_grid_manifest.json`
- `artifacts/backtest_sensitivity_walkforward/wfgrid-<id>/wf_grid_by_split.parquet`
- `artifacts/backtest_sensitivity_walkforward/wfgrid-<id>/wf_grid_config_aggregate.parquet`
- `artifacts/backtest_sensitivity_walkforward/wfgrid-<id>/wf_grid_source_summary.parquet`
- `artifacts/backtest_sensitivity_walkforward/wfgrid-<id>/wf_grid_report.md`

Robustness score (0-100) is a deterministic heuristic:

- expectancy rank (30%)
- profit factor rank (20%)
- drawdown score (20%)
- consistency via return dispersion (15%)
- cost robustness (10%)
- execution hygiene (5%)

This score is for ranking candidate configs and should not be treated as a statistical significance test.

## Backtest Sensitivity Pack v2

v2 extends v1 with:

- cluster policy filter modes:
  - `allow_only` (default)
  - `allow_watch`
  - `all_states`
- quality-of-edge metrics:
  - `ret_cv`
  - `ret_p10`, `ret_p90`
  - `downside_std`
  - `worst_trade_return`, `best_trade_return`
  - `trades_per_1000_rows`
  - `row_usage_rate` (MVP proxy)
  - `turnover_proxy` (MVP proxy)
  - `is_zero_trade_combo`
- extended robustness scoring:
  - `robustness_score_v1` and `robustness_score_v2`
- enriched walk-forward aggregation:
  - winner stability by metric
  - config consistency across splits
  - cost fragility summary
  - tail-risk summary

Important MVP simplifications:

- `row_usage_rate` is approximated from realized trade count vs eligible rows.
- `turnover_proxy` is approximated from trade count and average hold bars.
- These are deterministic research proxies, not exact microstructure turnover/utilization measures.

Examples:

- Cluster grid with policy filter:
  - `python -m mf_etl.cli backtest-grid-run --input-type cluster --input-file /abs/path/clustered_dataset_full.parquet --cluster-hardening-dir /abs/path/cluster_hardening --policy-filter-mode allow_only --hold-bars-grid "5,10,20" --fee-bps-grid "0,10,25"`
- WF grid with explicit split list:
  - `python -m mf_etl.cli backtest-grid-wf-run --wf-run-dir /abs/path/wf-run --flow-dataset-file /abs/path/dataset.parquet --train-ends "2014-12-31,2016-12-31,2018-12-31,2020-12-31" --sources "hmm,flow,cluster" --policy-filter-mode allow_only --hold-bars-grid "5,10,20" --fee-bps-grid "0,10"`

## Hybrid State Overlay v1

Hybrid overlay conditions a primary state engine (HMM/FLOW) with cluster hardening policy classes on aligned `ticker,trade_date` rows.

- overlay modes:
  - `none`
  - `allow_only`
  - `allow_watch`
  - `block_veto`
  - `allow_or_unknown`
- key CLI flags:
  - `--overlay-cluster-file`
  - `--overlay-cluster-hardening-dir`
  - `--overlay-mode`
  - `--overlay-join-keys` (default: `ticker,trade_date`)
- commands with overlay support:
  - `backtest-run`
  - `backtest-wf-run`
  - `backtest-grid-run`
  - `backtest-grid-wf-run`

Overlay artifacts (when enabled):

- `overlay_join_summary.json`
- `overlay_join_coverage_by_year.csv`
- `overlay_policy_mix_on_primary.csv`
- `overlay_signal_effect_summary.json`
- `overlay_performance_breakdown.csv`

MVP simplifications:

- Duplicate overlay keys are deduped deterministically by first row after key sort (`dedupe_rule=first`).
- Overlay gating is a hard pass/veto filter only (no weighting/blending of signals yet).

## Hybrid Overlay Evaluation Report v1

`hybrid-eval-report` is an analysis-only command that reads existing Hybrid Overlay grid/WF artifacts and writes a compact decision report.

- command:
  - `python -m mf_etl.cli hybrid-eval-report`
- default inputs are wired to current known run dirs (you can override with flags):
  - HMM baseline grid
  - HMM + overlay `allow_only` grid
  - HMM + overlay `block_veto` grid
  - optional FLOW + overlay `allow_only` comparator
  - HMM WF baseline vs HMM WF hybrid
  - optional prior `backtest-grid-compare` run
- outputs:
  - `artifacts/hybrid_eval_reports/<run_id>_hybrid_eval_v1/hybrid_eval_summary.json`
  - `artifacts/hybrid_eval_reports/<run_id>_hybrid_eval_v1/hybrid_eval_table.csv`
  - `artifacts/hybrid_eval_reports/<run_id>_hybrid_eval_v1/hybrid_eval_wf_table.csv`
  - `artifacts/hybrid_eval_reports/<run_id>_hybrid_eval_v1/hybrid_eval_report.md`

Decision heuristics (explicit):

- single-run candidate score combines: expectancy, PF, robustness_v2, ret_cv (lower better), downside risk (lower better), zero-trade penalty.
- WF consistency score combines hybrid split wins on: expectancy, PF, robustness_v2, ret_cv (lower better).
- final labels:
  - `PROMOTE`
  - `KEEP_AS_BENCH`
  - `NICHE_FILTER`
  - `RESEARCH_ONLY`
