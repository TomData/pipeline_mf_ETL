"""Gaussian HMM model wrappers for research baseline runs."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class HMMFitResult:
    """Container for fitted HMM model and fit diagnostics."""

    model: Any
    train_loglik: float
    test_loglik: float | None
    model_meta: dict[str, Any]


def _require_hmmlearn() -> None:
    if importlib.util.find_spec("hmmlearn") is None:
        raise RuntimeError(
            "hmmlearn is required for HMM baseline. Install with: pip install hmmlearn"
        )


def fit_gaussian_hmm(
    train_X: np.ndarray,
    train_lengths: np.ndarray,
    *,
    n_components: int,
    covariance_type: str,
    n_iter: int,
    tol: float,
    random_state: int,
    verbose: bool = False,
    test_X: np.ndarray | None = None,
    test_lengths: np.ndarray | None = None,
) -> HMMFitResult:
    """Fit a Gaussian HMM and return model + train/test log-likelihood."""

    _require_hmmlearn()
    if train_X.ndim != 2 or train_X.shape[0] == 0:
        raise ValueError("train_X must be a non-empty 2D matrix.")
    if train_lengths.ndim != 1 or train_lengths.shape[0] == 0:
        raise ValueError("train_lengths must be a non-empty 1D array.")
    if int(np.sum(train_lengths)) != int(train_X.shape[0]):
        raise ValueError("sum(train_lengths) must equal train_X row count.")

    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        verbose=verbose,
    )
    model.fit(train_X, lengths=train_lengths.tolist())
    train_loglik = float(model.score(train_X, lengths=train_lengths.tolist()))

    test_loglik: float | None = None
    if test_X is not None and test_lengths is not None and test_X.shape[0] > 0 and test_lengths.shape[0] > 0:
        if int(np.sum(test_lengths)) != int(test_X.shape[0]):
            raise ValueError("sum(test_lengths) must equal test_X row count.")
        test_loglik = float(model.score(test_X, lengths=test_lengths.tolist()))

    converged = None
    n_iter_used = None
    if hasattr(model, "monitor_") and model.monitor_ is not None:
        converged = bool(getattr(model.monitor_, "converged", False))
        n_iter_used = int(getattr(model.monitor_, "iter", 0))

    model_meta = {
        "model_type": "GaussianHMM",
        "n_components": int(n_components),
        "covariance_type": covariance_type,
        "n_iter_requested": int(n_iter),
        "tol": float(tol),
        "random_state": int(random_state),
        "train_loglik": train_loglik,
        "test_loglik": test_loglik,
        "converged": converged,
        "n_iter_used": n_iter_used,
        "train_rows": int(train_X.shape[0]),
        "train_sequences": int(train_lengths.shape[0]),
        "test_rows": int(test_X.shape[0]) if test_X is not None else 0,
        "test_sequences": int(test_lengths.shape[0]) if test_lengths is not None else 0,
    }
    return HMMFitResult(
        model=model,
        train_loglik=train_loglik,
        test_loglik=test_loglik,
        model_meta=model_meta,
    )

