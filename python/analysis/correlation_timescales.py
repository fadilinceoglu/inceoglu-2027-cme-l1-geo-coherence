"""Autocorrelation-based block durations for waveform uncertainty estimates."""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite_1d(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_integer(value: int, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def autocorrelation_block_diagnostics(
    x: np.ndarray,
    filter_floor_samples: int = 1,
) -> dict[str, Any]:
    """Return conservative serial-dependence and block-count diagnostics.

    Autocorrelation is evaluated through half the record using lag-specific
    covariance normalization.  ``first_1e_lag_samples`` is the first lag at
    or below ``1/e`` (half-record censored if no such lag is observed).
    ``integrated_positive_time_samples`` is ``1 + 2*sum(rho_k)`` over the
    initial strictly positive autocorrelation sequence.  The chosen block
    length is the ceiling of the maximum of those two scales and the declared
    filter-memory floor.  The independent-block count is a diagnostic
    ``floor(n / block_length)``, not a formal effective sample size.
    """

    sample = _finite_1d(x, name="x")
    floor = _positive_integer(
        filter_floor_samples,
        name="filter_floor_samples",
    )
    if len(sample) < 8:
        raise ValueError("x must contain at least eight samples")
    if np.ptp(sample) <= 0.0:
        raise ValueError("autocorrelation is undefined for a constant input")

    centered = sample - np.mean(sample)
    variance = float(np.mean(np.square(centered)))
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("autocorrelation requires positive finite variance")

    maximum_lag = max(1, len(sample) // 2)
    correlations = np.empty(maximum_lag, dtype=float)
    for lag in range(1, maximum_lag + 1):
        covariance = float(np.mean(centered[:-lag] * centered[lag:]))
        correlations[lag - 1] = np.clip(covariance / variance, -1.0, 1.0)

    below = np.flatnonzero(correlations <= np.exp(-1.0))
    first_1e_censored = len(below) == 0
    first_1e = int(below[0] + 1) if len(below) else int(maximum_lag)

    nonpositive = np.flatnonzero(correlations <= 0.0)
    positive_stop = int(nonpositive[0]) if len(nonpositive) else len(correlations)
    positive_sum = float(np.sum(correlations[:positive_stop]))
    integrated_positive = float(max(1.0, 1.0 + 2.0 * positive_sum))
    positive_sequence_censored = len(nonpositive) == 0

    block_length = int(
        min(
            len(sample),
            np.ceil(max(float(first_1e), integrated_positive, float(floor))),
        )
    )
    independent_blocks = int(len(sample) // block_length)
    return {
        "sample_count": int(len(sample)),
        "first_1e_lag_samples": first_1e,
        "first_1e_censored": bool(first_1e_censored),
        "integrated_positive_time_samples": integrated_positive,
        "positive_sequence_censored": bool(positive_sequence_censored),
        "filter_floor_samples": floor,
        "chosen_block_length_samples": block_length,
        "independent_block_count": independent_blocks,
        "autocorrelation_max_lag_samples": int(maximum_lag),
    }
