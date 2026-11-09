"""Band-energy intervals for the Figure 4 metrics.

Inside a search window, an interval runs from the minute where the cumulative
filtered band energy first reaches the lower fraction to the minute where it
reaches the upper fraction. Low-amplitude tails carry little energy, so they
barely move these edges.
"""

from __future__ import annotations

import numpy as np


def energy_edges(values: np.ndarray, lower: float, upper: float) -> tuple[int, int]:
    """Return half-open sample indices spanning the lower-to-upper energy fractions."""

    energy = np.square(np.asarray(values, dtype=float))
    if energy.ndim != 1 or not len(energy) or not np.isfinite(energy).all():
        raise ValueError("energy edges require a finite one-dimensional record")
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("energy fractions must satisfy 0 <= lower < upper <= 1")
    total = float(energy.sum())
    if total <= 0.0:
        raise ValueError("energy edges require nonzero band energy")
    cumulative = np.cumsum(energy) / total
    start = int(np.searchsorted(cumulative, lower))
    stop = int(np.searchsorted(cumulative, upper)) + 1
    return start, min(stop, len(energy))
