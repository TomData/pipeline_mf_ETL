"""HMM research baseline package exports."""

from mf_etl.research_hmm.decode import build_decoded_rows, decode_with_model
from mf_etl.research_hmm.diagnostics import HMMDiagnostics, build_hmm_diagnostics
from mf_etl.research_hmm.hmm_model import HMMFitResult, fit_gaussian_hmm
from mf_etl.research_hmm.pipeline import (
    HMMRunResult,
    HMMStabilityResult,
    HMMSweepResult,
    run_hmm_baseline,
    run_hmm_stability,
    run_hmm_sweep,
)
from mf_etl.research_hmm.profiles import (
    HMMProfiles,
    build_hmm_state_profiles,
    build_hmm_vs_cluster_crosstab,
)
from mf_etl.research_hmm.sanity import summarize_hmm_run
from mf_etl.research_hmm.sequence_builder import SequenceBuildResult, build_hmm_sequences

__all__ = [
    "HMMFitResult",
    "HMMDiagnostics",
    "HMMProfiles",
    "HMMRunResult",
    "HMMStabilityResult",
    "HMMSweepResult",
    "SequenceBuildResult",
    "build_decoded_rows",
    "build_hmm_diagnostics",
    "build_hmm_sequences",
    "build_hmm_state_profiles",
    "build_hmm_vs_cluster_crosstab",
    "decode_with_model",
    "fit_gaussian_hmm",
    "run_hmm_baseline",
    "run_hmm_stability",
    "run_hmm_sweep",
    "summarize_hmm_run",
]

