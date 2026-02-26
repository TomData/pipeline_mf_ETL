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
  - `--overlay-coverage-mode` (`warn_only` or `strict_fail`)
  - `--overlay-min-match-rate-warn`, `--overlay-min-match-rate-fail`
  - `--overlay-min-year-match-rate-warn`, `--overlay-min-year-match-rate-fail`
  - `--overlay-unknown-rate-warn`, `--overlay-unknown-rate-fail`
  - `--overlay-coverage-bypass`
- commands with overlay support:
  - `backtest-run`
  - `backtest-wf-run`
  - `backtest-grid-run`
  - `backtest-grid-wf-run`

Overlay artifacts (when enabled):

- `overlay_join_summary.json`
- `overlay_join_coverage_by_year.csv`
- `overlay_coverage_verdict.json`
- `overlay_policy_mix_on_primary.csv`
- `overlay_signal_effect_summary.json`
- `overlay_performance_breakdown.csv`

MVP simplifications:

- Duplicate overlay keys are deduped deterministically by first row after key sort (`dedupe_rule=first`).
- Overlay gating is a hard pass/veto filter only (no weighting/blending of signals yet).
- Strict coverage checks evaluate join diagnostics before signal generation. In `strict_fail`, a FAIL verdict aborts the run unless `--overlay-coverage-bypass` is set.

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

## Execution Realism Pass v1

Execution Realism Pass v1 adds optional pre-signal execution eligibility filters without changing state engines or core signal semantics.

- realism controls:
  - `min_price` floor
  - `min_dollar_vol_20` floor using rolling `close * volume` (`median`, window=20)
  - `max_vol_pct` cap (uses `atr_pct_14` when present; fallback `(high-low)/close`)
  - `min_history_bars_for_execution` warmup
  - `vol_input_unit_mode` (`auto|decimal|percent_points`): in `auto`, volatility is normalized to decimal units if source appears to be percent-points (heuristic: median>1 or p90>1).
- named profiles:
  - `none` (disabled)
  - `lite`
  - `strict`
- CLI flags on `backtest-run`, `backtest-wf-run`, `backtest-grid-run`, `backtest-grid-wf-run`:
  - `--exec-profile {none,lite,strict}`
  - `--exec-min-price`
  - `--exec-min-dollar-vol20`
  - `--exec-max-vol-pct`
  - `--exec-min-history-bars`
  - report thresholds:
    - `--report-min-trades`
    - `--report-max-zero-trade-share`
    - `--report-max-ret-cv`

Execution realism artifacts (when enabled/profile not `none`):

- `execution_filter_summary.json`
- `execution_filter_by_reason.csv`
- `execution_filter_by_year.csv`
- `execution_trade_context_summary.json`

Suppression metrics semantics:

- `*_count` fields are integer counts.
- `*_share` fields are true shares in `[0,1]`, normalized by `suppressed_signal_count`.
- when `suppressed_signal_count == 0`, suppression shares are written as `0.0` (not null).

Execution metrics are propagated into:

- backtest summaries
- grid combo tables
- walk-forward split/source summaries
- grid-compare deltas and realism verdicts

`execution-realism-report` command:

- `python -m mf_etl.cli execution-realism-report`
- writes:
  - `artifacts/execution_realism_reports/<run_id>_execution_realism_v1/execution_realism_summary.json`
  - `artifacts/execution_realism_reports/<run_id>_execution_realism_v1/execution_realism_table.csv`
  - `artifacts/execution_realism_reports/<run_id>_execution_realism_v1/execution_realism_wf_table.csv`
  - `artifacts/execution_realism_reports/<run_id>_execution_realism_v1/execution_realism_report.md`

MVP simplification:

- If multiple realism rules suppress a row, one primary suppression reason is assigned by fixed precedence for reporting.
- Candidate ranking guard: runs with `trade_count == 0` or `zero_trade_share >= 1.0` are labeled `NOT_TRADABLE` and cannot be selected as primary/secondary candidates in `execution-realism-report`.

## Execution Realism Calibration Pass v1

Execution Realism Calibration Pass v1 adds a diagnostics/sweep layer to calibrate realism thresholds to the current universe before running full PnL backtests.

- main command:
  - `python -m mf_etl.cli execution-realism-calibrate --source-file /abs/path/decoded_rows.parquet --source-type hmm --overlay-mode none --by-year`
- report re-render:
  - `python -m mf_etl.cli execution-realism-calibration-report --calibration-dir /abs/path/artifacts/execution_realism_calibration/exec-calib-...`

What calibration produces:

- candidate/eligible/suppressed distributions for:
  - `close`
  - `dollar_vol_20`
  - volatility metric used by realism
  - history bars
- suppression decomposition:
  - waterfall JSON
  - overlap matrix CSV
  - first-fail CSV
- threshold sweep (fast diagnostics only, no PnL simulation):
  - `min_price`
  - `min_dollar_vol20`
  - `max_vol_pct`
  - `min_history_bars`
- auto recommendations for `lite` and `strict` target eligibility bands.

Calibration artifacts:

- `execution_calibration_summary.json`
- `execution_calibration_distribution.csv`
- `execution_calibration_distribution_by_year.csv` (when enabled)
- `execution_calibration_waterfall.json`
- `execution_calibration_reason_overlap.csv`
- `execution_calibration_first_fail.csv`
- `execution_calibration_units.json`
- `execution_calibration_grid.csv`
- `execution_calibration_grid_summary.json`
- `execution_calibration_recommendations.json`
- `execution_calibration_report.md`

Interpretation note:

- `ZERO_ELIGIBILITY` means threshold mismatch for the current universe/profile, not necessarily strategy failure. Grid summaries expose this with `realism_profile_broken_for_universe`.

## Production Candidate Pack v1

Production Candidate Pack (PCP v1) locks 1-2 deterministic candidate configs from existing grid/WF artifacts into a reusable operator packet.

- build command:
  - `python -m mf_etl.cli production-candidates-build`
- sanity command:
  - `python -m mf_etl.cli production-candidates-sanity --pack-dir <PACK_DIR>`

Default pack inputs point to current known runs:

- A1/A2/A3 HMM baseline grids (none/lite/strict)
- B HMM+overlay `allow_only` lite grid
- C HMM+overlay `block_veto` lite grid
- D1 baseline WF lite + D2 hybrid WF lite
- latest execution realism report (optional enrichment)

Selection policy (deterministic):

- `CANDIDATE_ALPHA` from A1 by:
  - robustness_v2 desc, expectancy desc, PF desc, trade_count desc, combo_id asc
- `CANDIDATE_EXEC` from B by:
  - PF desc, robustness_v2 desc, ret_cv asc, trade_count desc, combo_id asc
- optional `CANDIDATE_EXEC_2` from A2 using execution ranking above
- min trade threshold:
  - requested `--min-trades` (default `25`)
  - auto-relax to `10` if needed, with warning
  - zero-trade combos are never selected

PCP outputs:

- `artifacts/production_candidates/pcp-<id>_production_candidate_pack_v1/production_policy_packet_v1.json`
- `artifacts/production_candidates/pcp-<id>_production_candidate_pack_v1/production_candidates_table.csv`
- `artifacts/production_candidates/pcp-<id>_production_candidate_pack_v1/production_candidates_summary.json`
- `artifacts/production_candidates/pcp-<id>_production_candidate_pack_v1/production_candidate_pack_report.md`

## Candidate Re-run Pack v1

Candidate Re-run Pack (CRP v1) reruns locked PCP candidates and computes drift flags vs expected snapshots.

- run command:
  - `python -m mf_etl.cli candidate-rerun-run --pcp-pack-dir <PCP_DIR> [--as-of-tag TAG] [--wf-run-dir <WF_DIR>] [--override-input-file <FILE>]`
- sanity command:
  - `python -m mf_etl.cli candidate-rerun-sanity --rerun-dir <CRP_DIR>`

CRP behavior:

- reruns each PCP candidate via `backtest-run` using locked params
- optional micro-grid around locked config (small local sensitivity probe)
- optional WF single-combo rerun per candidate when `--wf-run-dir` is provided
- computes deltas and drift status (`OK`, `DRIFT_WARN`, `DRIFT_FAIL`)
- computes overlay coverage drift for overlay-enabled candidates:
  - `match_rate` absolute drop vs PCP baseline
  - `unknown_rate` absolute increase vs PCP baseline
  - escalates to `DRIFT_FAIL` on hard coverage breaches

CRP outputs:

- `artifacts/candidate_reruns/crp-<id>_candidate_rerun_pack_v1/rerun_manifest.json`
- `artifacts/candidate_reruns/crp-<id>_candidate_rerun_pack_v1/rerun_candidates_table.csv`
- `artifacts/candidate_reruns/crp-<id>_candidate_rerun_pack_v1/rerun_summary.json`
- `artifacts/candidate_reruns/crp-<id>_candidate_rerun_pack_v1/rerun_report.md`
- per-candidate subdirs:
  - `candidates/<CANDIDATE>/backtest_run_dir.txt`
  - `candidates/<CANDIDATE>/backtest_summary.json`
  - `candidates/<CANDIDATE>/drift_metrics.json`
  - `candidates/<CANDIDATE>/coverage_drift_metrics.json`
  - optional `micro_grid_dir.txt`

## Nightly Research Ops Pack v1

Nightly Research Ops Pack (NROP v1) automates daily research-ops checks on top of PCP + CRP + overlay coverage hardening.

- main command:
  - `python -m mf_etl.cli ops-nightly-run --pcp-pack-dir <PCP_DIR> --wf-run-dir <WF_DIR> --as-of-tag YYYY-MM-DD`
- sanity command:
  - `python -m mf_etl.cli ops-nightly-sanity --ops-run-dir <OPS_RUN_DIR>`
- ledger preview:
  - `python -m mf_etl.cli ops-ledger-view --last 10`

What nightly run does:

- auto-discovers latest PCP pack (unless `--pcp-pack-dir` is provided)
- executes CRP rerun with optional WF check and overlay coverage policy passthrough
- builds daily ops report (JSON/CSV/MD)
- appends one row to cumulative ledger CSV
- optionally cleans up old ops run dirs with `--keep-last-n`

NROP outputs:

- `artifacts/ops_runs/ops-<id>_nightly_ops_v1/ops_manifest.json`
- `artifacts/ops_runs/ops-<id>_nightly_ops_v1/ops_summary.json`
- `artifacts/ops_runs/ops-<id>_nightly_ops_v1/ops_report.md`
- `artifacts/ops_runs/ops-<id>_nightly_ops_v1/ops_report.csv`
- `artifacts/ops_runs/ops-<id>_nightly_ops_v1/ops_warnings.json`
- pointers:
  - `crp_rerun_dir.txt`
  - `wf_check_dir.txt` (when WF enabled)
- ledger:
  - `artifacts/ops_ledger/ops_ledger.csv`

Optional systemd timer template writer:

- `python -m mf_etl.cli ops-nightly-install-timer [--wf-run-dir <WF_DIR>] [--on-calendar 07:30]`

This writes templates to `configs/systemd/`:

- `mf_etl_ops_nightly.service`
- `mf_etl_ops_nightly.timer`

Install as user service:

- `mkdir -p ~/.config/systemd/user`
- `cp configs/systemd/mf_etl_ops_nightly.service ~/.config/systemd/user/`
- `cp configs/systemd/mf_etl_ops_nightly.timer ~/.config/systemd/user/`
- `systemctl --user daemon-reload`
- `systemctl --user enable --now mf_etl_ops_nightly.timer`
- `journalctl --user -u mf_etl_ops_nightly.service -n 200 --no-pager`

## Beta Expert Advisor Overlay Viewer v1.1.2

`overlay-viewer` is a local Streamlit visualization app for teaching/research.  
It overlays research layers on price without any execution logic.

Run:

- `python -m mf_etl.cli overlay-viewer`

Optional:

- `python -m mf_etl.cli overlay-viewer --server-port 8502 --server-address 0.0.0.0`
- `python -m mf_etl.cli overlay-compute-ticker --ticker AAPL.US --full-history --hmm-k 5 --exec-profile lite`

Data source modes:

- `CACHED (Compute Ticker)` (default): on-demand per-ticker compute pipeline with persistent cache under `artifacts/ticker_cache/<TICKER>/<run_id>/`
- `GLOBAL ARTIFACTS (Latest)`: loads latest discovered global artifacts (legacy behavior)

Cached compute artifacts per run:

- `ohlcv.parquet`
- `indicators.parquet` (TMF/TTI + helper series)
- `states_flow.parquet` (always generated via local event grammar in cached compute mode)
- `states_hmm.parquet` (local GaussianHMM decode + diagnostics columns)
- `overlay_exec.parquet` (execution realism pass/fail)
- `overlay_policy.parquet` (optional global overlay join)
- `meta.json`, `summary.json`

v1.1.1 patch:

- local flow states are now always computed from TMF event grammar for cached ticker runs (`states_flow_source=local_event_grammar_v1`), including Bronze fallback compute paths
- this keeps Flow band/S0..S4 teaching overlays available even when the latest ML dataset only contains a small ticker subset

v1.1.2 patch:

- HMM display controls for interpretability:
  - `HMM Display Mode`: `RAW`, `SMOOTHED`, `GROUPED`, `SMOOTHED+GROUPED`
  - smoothing controls: `method` (`mode`/`median`) + trailing `window`
  - grouping controls: `LONG_NEUTRAL_SHORT` or `LONG_OTHER`, with `top_k`/`bottom_k`
- grouped mapping can be persisted into cache `meta.json` under `display_mappings.hmm_group_display_v1`
- grouped/smoothed HMM band reduces state flicker for classroom interpretation

Viewer features:

- candlestick chart + volume (separate panel or overlay)
- TMF + TTI proxy indicator panel (Twiggs/Pine-consistent TRH/TRL + RMA logic)
- state overlays:
  - flow states
  - HMM states
  - cluster hardening policy class (`ALLOW`/`WATCH`/`BLOCK`)
- execution realism pass/fail markers and blocked reasons
- candidate signal markers from PCP candidate settings (or manual mode)

Data assumptions:

- minimum price columns: `trade_date`, `open`, `high`, `low`, `close`, `volume`
- optional layers use available columns (`hmm_state`, `flow_state_code`, `cluster_id`, etc.)
- if optional columns/files are missing, the app degrades gracefully and disables that layer with warnings
