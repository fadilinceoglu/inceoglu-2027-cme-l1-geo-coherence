"""Shared reference direction for the sheath--ejecta boundary diagnostics."""

from __future__ import annotations

import numpy as np


def mean_field_direction(vectors: np.ndarray) -> np.ndarray:
    """Normalize each finite, nonzero field vector, then normalize their mean.

    Every sampled direction has equal weight, independent of field strength.
    Input rows contain the three magnetic components in a single frame.
    """
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected an (n, 3) array of magnetic vectors")
    lengths = np.linalg.norm(values, axis=1)
    valid = np.isfinite(values).all(axis=1) & np.isfinite(lengths) & (lengths > 0)
    if not valid.any():
        raise ValueError("Reference interval contains no finite, nonzero vectors")
    mean = (values[valid] / lengths[valid, None]).mean(axis=0)
    length = np.linalg.norm(mean)
    if length == 0:
        raise ValueError("Reference directions cancel; the mean direction is undefined")
    return mean / length
