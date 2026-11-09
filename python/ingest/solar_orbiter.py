"""Solar Orbiter in-situ readers for the 19 January 2026 CME study.

Solar Orbiter is a single far-upstream observation point, not a fifth member
of the compact L1 timing constellation.  MAG vectors are kept in the mission
RTN frame and SWA/PAS supplies plasma context and convected-distance speed.

Official event files are mirrored by NASA SPDF/CDAWeb from the ESA Solar
Orbiter Archive.  The readers operate on local CDF files so analyses remain
reproducible after download and do not hide network access inside a data read.

"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from ingest.cdf_helpers import _normalise_paths, _valid_numeric, _combine
from cdflib import CDF, cdfepoch


def _epoch_index(cdf: CDF, variable: str) -> pd.DatetimeIndex:
    values = cdfepoch.to_datetime(cdf.varget(variable))
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True), name="time_utc")


def read_solar_orbiter_mag(
    paths: Path | str | Iterable[Path | str],
) -> pd.DataFrame:
    """Read native-cadence Solar Orbiter MAG L2 RTN-normal CDF files.

    Args:
        paths: One or more local ``solo_l2_mag-rtn-normal`` CDF files.

    Returns:
        UTC-indexed DataFrame with ``br_rtn``, ``bt_rtn``, ``bn_rtn``, ``bt``
        (magnitude), ``quality_flag``, and ``quality_bitmask``.  The January
        2026 normal product is nominally 8 Hz (0.125 s median cadence).
    """
    frames: list[pd.DataFrame] = []
    for path in _normalise_paths(paths):
        cdf = CDF(str(path))
        try:
            index = _epoch_index(cdf, "EPOCH")
            vector = _valid_numeric(cdf, "B_RTN")
            quality = np.asarray(cdf.varget("QUALITY_FLAG"), dtype=np.uint8)
            bitmask = np.asarray(cdf.varget("QUALITY_BITMASK"), dtype=np.uint16)
        finally:
            close = getattr(cdf, "close", None)
            if close is not None:
                close()

        if vector.ndim != 2 or vector.shape[1] != 3:
            raise ValueError(f"B_RTN in {path} does not have shape (N, 3)")

        frame = pd.DataFrame(
            vector, index=index, columns=["br_rtn", "bt_rtn", "bn_rtn"]
        )
        frame["bt"] = np.linalg.norm(vector, axis=1)
        frame["quality_flag"] = quality
        frame["quality_bitmask"] = bitmask
        frames.append(frame)

    return _combine(
        frames,
        {
            "mission": "Solar Orbiter",
            "instrument": "MAG",
            "product": "L2 RTN normal",
            "frame": "RTN",
            "field_units": "nT",
            "role": "single far-upstream point",
            "source": "ESA Solar Orbiter Archive via NASA SPDF/CDAWeb",
            "quality_rule": "unmasked",
        },
    )


def read_solar_orbiter_swa_pas(
    paths: Path | str | Iterable[Path | str],
) -> pd.DataFrame:
    """Read Solar Orbiter SWA/PAS L2 ground moments at native cadence.

    The product supplies density and RTN bulk velocity for shock/structure
    context and for constructing the convected-distance coordinate.  PAS has
    data gaps, so callers must not interpolate across a shock by default.
    """
    frames: list[pd.DataFrame] = []
    for path in _normalise_paths(paths):
        cdf = CDF(str(path))
        try:
            index = _epoch_index(cdf, "Epoch")
            density = _valid_numeric(cdf, "N")
            velocity = _valid_numeric(cdf, "V_RTN")
            temperature = _valid_numeric(cdf, "T")
        finally:
            close = getattr(cdf, "close", None)
            if close is not None:
                close()

        if velocity.ndim != 2 or velocity.shape[1] != 3:
            raise ValueError(f"V_RTN in {path} does not have shape (N, 3)")
        frame = pd.DataFrame(
            velocity, index=index, columns=["vr_rtn", "vt_rtn", "vn_rtn"]
        )
        frame.insert(0, "density", density)
        frame["speed"] = np.linalg.norm(velocity, axis=1)
        frame["temperature"] = temperature
        frames.append(frame)

    return _combine(
        frames,
        {
            "mission": "Solar Orbiter",
            "instrument": "SWA/PAS",
            "product": "L2 ground moments",
            "frame": "RTN",
            "density_units": "cm^-3",
            "velocity_units": "km/s",
            "temperature_units": "eV",
            "role": "plasma context and convected-distance speed",
            "source": "ESA Solar Orbiter Archive via NASA SPDF/CDAWeb",
        },
    )
