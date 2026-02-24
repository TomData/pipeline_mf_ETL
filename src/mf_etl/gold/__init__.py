"""Gold stage package exports."""

from mf_etl.gold.event_grammar_v1 import (
    EVENT_CALC_VERSION,
    EVENT_SCHEMA_VERSION,
    FLOW_STATE_CODE_TO_LABEL,
    build_event_grammar_v1,
)
from mf_etl.gold.pipeline import (
    GoldEventOneResult,
    GoldEventRunOptions,
    GoldEventRunResult,
    GoldEventSanityResult,
    discover_indicator_files,
    discover_indicator_inputs,
    resolve_indicator_file_for_ticker,
    run_events_one_from_indicator_file,
    run_events_pipeline,
    run_events_sanity,
)
from mf_etl.gold.placeholders import ensure_gold_placeholder, open_gold_duckdb
from mf_etl.gold.writer import GoldEventWriteResult, gold_event_output_path, write_gold_event_parquet

__all__ = [
    "EVENT_CALC_VERSION",
    "EVENT_SCHEMA_VERSION",
    "FLOW_STATE_CODE_TO_LABEL",
    "GoldEventOneResult",
    "GoldEventRunOptions",
    "GoldEventRunResult",
    "GoldEventSanityResult",
    "GoldEventWriteResult",
    "build_event_grammar_v1",
    "discover_indicator_files",
    "discover_indicator_inputs",
    "ensure_gold_placeholder",
    "gold_event_output_path",
    "open_gold_duckdb",
    "resolve_indicator_file_for_ticker",
    "run_events_one_from_indicator_file",
    "run_events_pipeline",
    "run_events_sanity",
    "write_gold_event_parquet",
]
