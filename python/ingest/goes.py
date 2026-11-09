"""Canonical reader for NOAA GOES-R 10-Hz Level-2 MAG in GSM.

QC policy for the retained ``magn-l2-hires`` product:

* ``_FillValue`` / ``valid_min`` / ``valid_max`` masking via netCDF4
  automasking;
* DQF hard-invalid bits 0-14 (missing L0 input, sensor-axis failure or
  saturation, calibration manoeuvre, stale engineering data) reject the
  vector sample;
* the arcjet-firing bit (24) is accepted only when NOAA's
  valid-arcjet-correction bit (25) is also set;
* higher advisory bits (thermistor/electronics temperature, shadow) are
  retained but do not reject data;
* the raw ``dqf`` and ``b_quality`` columns stay in the output so alternative
  masks can be audited;
* time must be strictly increasing.

The publication uses the product's native ``b_gsm`` vector and unchanged
independent ``b_total`` scalar magnitude.
"""

from __future__ import annotations

import re
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

J2K_UTC = pd.Timestamp("2000-01-01T12:00:00Z")

# NOAA DQF bits, verified against the ``magn-l2-hires``
# flag_masks/flag_meanings attributes for the January 2026 files.
DQF_HARD_INVALID_MASK = np.uint32(0x00007FFF)   # bits 0-14
DQF_ARCJET_FIRING = np.uint32(0x01000000)       # bit 24
DQF_VALID_ARCJET_CORRECTION = np.uint32(0x02000000)  # bit 25
DQF_FILL = np.uint32(0xFFFFFFFF)


def spacecraft_from_path(path: Path | str) -> str:
    match = re.search(r"(?:^|[_-])g(\d{2})(?:[_\-.]|$)", Path(path).name.lower())
    return f"GOES-{match.group(1)}" if match else "GOES-unknown"


def _time_index(seconds_since_j2k: np.ndarray) -> pd.DatetimeIndex:
    values = np.asarray(seconds_since_j2k, dtype=float)
    index = pd.DatetimeIndex(
        J2K_UTC + pd.to_timedelta(values, unit="s"), name="time_utc")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("GOES time coordinate must be strictly increasing")
    return index


def _numeric(var) -> np.ndarray:
    """Float values with _FillValue/valid_min/valid_max applied as NaN."""
    data = var[:]
    if np.ma.isMaskedArray(data):
        return np.asarray(data.astype("float64").filled(np.nan))
    return np.asarray(data, dtype="float64")


def read_goes_mag(path: str | Path) -> pd.DataFrame:
    """Read one NOAA GOES-R ``magn-l2-hires`` file in GSM.

    Returns a DataFrame indexed by UTC time with columns
        bx, by, bz  [nT, GSM]   bt [nT, NOAA b_total]
        dqf, b_quality, dqf_usable          (audit columns)
    QC-rejected samples (see module docstring) are NaN in bx..bt; the raw
    flags stay untouched.
    """
    b_var = "b_gsm"

    with netCDF4.Dataset(str(path)) as ds:
        required = {"time", b_var, "b_total", "DQF"}
        missing = sorted(required.difference(ds.variables))
        if missing:
            raise KeyError(f"missing required GOES variables: {missing}")

        ds.variables["time"].set_auto_mask(False)
        full_time = np.asarray(ds.variables["time"][:], dtype=float)
        index = _time_index(full_time)

        vector = _numeric(ds.variables[b_var])
        if vector.shape != (len(index), 3):
            raise ValueError(f"{b_var} must have shape (time, 3)")
        b_total = _numeric(ds.variables["b_total"])

        dqf_var = ds.variables["DQF"]
        dqf_var.set_auto_mask(False)
        dqf = np.asarray(dqf_var[:], dtype=np.uint32)

        quality = np.full(len(index), 255, dtype=np.uint8)
        if "b_quality" in ds.variables:
            qv = ds.variables["b_quality"]
            qv.set_auto_mask(False)
            quality = np.asarray(qv[:], dtype=np.uint8)

    hard_invalid = (dqf & DQF_HARD_INVALID_MASK) != 0
    arcjet_bad = (((dqf & DQF_ARCJET_FIRING) != 0)
                  & ((dqf & DQF_VALID_ARCJET_CORRECTION) == 0))
    usable = (dqf != DQF_FILL) & ~hard_invalid & ~arcjet_bad

    vector[~usable] = np.nan
    b_total[~usable] = np.nan

    out = pd.DataFrame(vector, index=index, columns=["bx", "by", "bz"])
    out["bt"] = b_total
    out["dqf"] = dqf
    out["b_quality"] = quality
    out["dqf_usable"] = usable
    out.attrs.update({
        "spacecraft": spacecraft_from_path(path),
        "frame": "GSM",
        "vector_variable": b_var,
        "scalar_variable": "b_total",
        "source_path": str(Path(path).resolve()),
        "dqf_policy": ("hard bits 0-14 rejected; arcjet firing accepted only "
                       "with valid-correction bit; advisory bits retained"),
    })
    return out
