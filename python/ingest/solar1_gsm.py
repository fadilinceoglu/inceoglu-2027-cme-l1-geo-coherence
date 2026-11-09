"""Native Solar-1 GSM for magnetopause context, separate from waveform inputs.

Use the supplied provider-minute GSM field with complete accepted native-second
support. Proton correction provenance follows the existing mission reader.
"""
from __future__ import annotations

from pathlib import Path
import netCDF4
import numpy as np
import pandas as pd

from common.io import sha256
from config.constants import EPOCH_1958


def _times(variable):
    units = str(variable.getncattr("units")).lower()
    if "microseconds since 1958-01-01" not in units:
        raise ValueError("unexpected Solar-1 time units")
    values = np.ma.asarray(variable[:])
    if np.ma.getmaskarray(values).any() or not np.isfinite(values).all():
        raise ValueError("Solar-1 timestamps must be finite and unmasked")
    times = pd.DatetimeIndex(EPOCH_1958 + pd.to_timedelta(np.asarray(values, dtype="int64"), unit="us"))
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Solar-1 timestamps must be unique and ordered")
    return times


def read_native_solar1_gsm_files(files):
    parts, sources = [], []
    if not files:
        raise FileNotFoundError("supplied Solar-1 MAG files with native GSM are required")
    for path in files:
        with netCDF4.Dataset(path) as dataset:
            required = {"time_min", "time_sec", "b_gsm_min", "b_gsm_sec", "flags_summary"}
            if not required.issubset(dataset.variables):
                raise ValueError("native Solar-1 GSM/time/quality variables missing; no conversion fallback")
            for name in ("b_gsm_min", "b_gsm_sec"):
                if dataset.variables[name].getncattr("units") != "nT":
                    raise ValueError("native Solar-1 magnetic units must be nT")
            times = _times(dataset.variables["time_min"])
            second_times = _times(dataset.variables["time_sec"])
            if len(second_times) < 2 or not (times == times.floor("min")).all():
                raise ValueError("native MAG needs minute-start times and a measurable second cadence")
            cadence_s = float(np.median(np.diff(second_times.asi8))/1e9)
            expected_float = pd.Timedelta("1min").total_seconds()/cadence_s
            if cadence_s <= 0 or not np.isclose(expected_float, round(expected_float)):
                raise ValueError("native MAG cadence does not divide the selected minute bins")
            expected = int(round(expected_float))
            vectors = np.ma.filled(dataset.variables["b_gsm_min"][:], np.nan).astype(float)
            seconds = np.ma.filled(dataset.variables["b_gsm_sec"][:], np.nan).astype(float)
            flags = np.ma.filled(dataset.variables["flags_summary"][:], -1)
            if vectors.shape != (len(times), 3) or seconds.shape != (len(second_times), 3) or flags.shape != (len(second_times),):
                raise ValueError("inconsistent Solar-1 native vector/time/quality shapes")
            usable = (np.isfinite(seconds).all(axis=1) & (np.linalg.norm(seconds, axis=1) <= 500) & (flags == 1))
            support = pd.DataFrame({"usable": usable}, index=second_times.floor("min")).groupby(level=0).agg(
                observed_seconds=("usable", "size"), usable_seconds=("usable", "sum"))
            support = support.reindex(times)
            supported = (support.observed_seconds.eq(expected) & support.usable_seconds.eq(expected)).to_numpy()
            valid = np.isfinite(vectors).all(axis=1) & (np.linalg.norm(vectors, axis=1) <= 500) & supported
            vectors[~valid] = np.nan
            frame = pd.DataFrame(vectors, index=times, columns=["bx_gsm_nt", "by_gsm_nt", "bz_gsm_nt"])
            frame["magnetic_epoch_utc"] = [stamp.isoformat() for stamp in times]
            frame["magnetic_usable_seconds"] = support.usable_seconds.to_numpy()
            parts.append(frame)
            sources.append({"path": str(Path(path).resolve()), "sha256": sha256(path),
                            "native_cadence_s": cadence_s, "expected_native_records_per_minute": expected})
    combined = pd.concat(parts).sort_index()
    if combined.index.has_duplicates:
        raise ValueError("native Solar-1 minute files overlap")
    return combined, {"vector_variable": "b_gsm_min", "epoch_variable": "time_min", "units": "nT",
                      "time_convention": "native one-minute averaging-window start",
                      "quality_policy": "complete native-second support, finite norm <=500 nT, flags_summary==1 as in current pipeline",
                      "quality_note": "summary value 1 is labelled degraded by the supplied file, rather than good/flag-free",
                      "magnetic_inputs": sources}


def read_native_solar1_inputs(root):
    from ingest.missions import inventory, read_corrected_swips_moments
    mag_files, plasma_files = inventory("swfo", Path(root).resolve())
    native, provenance = read_native_solar1_gsm_files(mag_files)
    plasma_provenance = []
    for path in plasma_files:
        # Pressure provenance must validate exactly the same corrected moments
        # that supply the prepared plasma and speed records.
        read_corrected_swips_moments(path)
        plasma_provenance.append({"path": str(Path(path).resolve()), "sha256": sha256(path),
                                  "current_pipeline_selected_variables": {
                                      "density": "proton_n", "velocity": "proton_v_gse", "temperature": "proton_t"},
                                  "moment_status": "corrected replacement supplied October 2026"})
    if not plasma_provenance:
        raise ValueError("corrected Solar-1 proton moment inputs are required")
    provenance["plasma_inputs"] = plasma_provenance
    provenance["moment_status"] = "corrected replacement supplied October 2026"
    provenance["pressure_policy"] = "recompute from selected proton n and speed; supplied alpha/total pressure fields are not used"
    return native, provenance

