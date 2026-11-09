"""ARTEMIS lunar-orbit field and plasma readers.

For the 19 January 2026 event, ARTEMIS-P2 (THEMIS-C) was about 3,900 km
anti-sunward of the lunar centre and within the projected lunar disk.  It is
therefore selected as a *lunar-wake response* monitor, not as an undisturbed
upstream timing point.  ARTEMIS-P1 is the required geometry-qualified incident
control before a lunar-wake mechanism claim; neither probe is assumed to be in
pristine solar wind without event-time flow/IMF and selenocentric geometry.

The official THEMIS/ARTEMIS Level-2 files are available from NASA CDAWeb.
THEMIS CDFs store their usable epochs as Unix-second ``*_time`` variables;
these readers use those variables directly rather than relying on virtual CDF
epoch variables that some Python CDF libraries do not materialise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from ingest.cdf_helpers import _normalise_paths, _valid_numeric, _combine
from cdflib import CDF


def _unix_index(cdf: CDF, variable: str) -> pd.DatetimeIndex:
    seconds = np.asarray(cdf.varget(variable), dtype=float)
    return pd.DatetimeIndex(
        pd.to_datetime(seconds, unit="s", utc=True), name="time_utc"
    )


def _last_record(cdf: CDF, variable: str) -> int:
    """Return a CDF variable's last record using the pinned cdflib API."""

    return int(cdf.varinq(variable).Last_Rec)


def read_artemis_fgm(
    paths: Path | str | Iterable[Path | str],
    *,
    probe: str = "c",
    mode: str = "fgs",
    frame: str = "GSM",
) -> pd.DataFrame:
    """Read native ARTEMIS FGM vectors in the selected geocentric frame.

    Args:
        paths: One or more local ``th[b|c]_l2_fgm`` CDF files.
        probe: ``"b"`` for ARTEMIS-P1 or ``"c"`` for ARTEMIS-P2.
        mode: ``"fgs"`` (survey, about 4 s) or ``"fgh"`` (up to 128 Hz).
            Availability varies within each day.
        frame: ``"GSM"`` for the publication's magnetic components or ``"GSE"``.
            Scalar strength always retains the original native-GSE norm.

    Quality values are retained without masking.  In this event value 2
    denotes spacecraft shadow and is scientifically relevant to the lunar-wake
    interval.
    """
    probe = probe.lower()
    mode = mode.lower()
    coordinate = frame.lower()
    if probe not in {"b", "c"}:
        raise ValueError("ARTEMIS probe must be 'b' (P1) or 'c' (P2)")
    if mode not in {"fgs", "fgh"}:
        raise ValueError("FGM mode must be one of: fgs, fgh")
    if coordinate not in {"gse", "gsm"}:
        raise ValueError("FGM magnetic frame must be GSE or GSM")

    prefix = f"th{probe}_{mode}"
    quality_name = f"th{probe}_fgm_{mode}_quality"
    frames: list[pd.DataFrame] = []

    for path in _normalise_paths(paths):
        cdf = CDF(str(path))
        try:
            if _last_record(cdf, f"{prefix}_time") < 0:
                continue
            index = _unix_index(cdf, f"{prefix}_time")
            vector = _valid_numeric(cdf, f"{prefix}_{coordinate}")
            scalar_vector = vector if coordinate == "gse" else _valid_numeric(cdf, f"{prefix}_gse")
            quality = np.asarray(cdf.varget(quality_name), dtype=np.int32)
        finally:
            close = getattr(cdf, "close", None)
            if close is not None:
                close()

        if vector.ndim != 2 or vector.shape[1] != 3:
            raise ValueError(f"{prefix}_{coordinate} in {path} does not have shape (N, 3)")
        if scalar_vector.shape != vector.shape:
            raise ValueError(f"{prefix}_gse in {path} has inconsistent scalar-source shape")

        frame = pd.DataFrame(
            vector, index=index, columns=[f"b{axis}_{coordinate}" for axis in "xyz"]
        )
        frame["bt"] = np.linalg.norm(scalar_vector, axis=1)
        frame["quality"] = quality
        frames.append(frame)

    if not frames:
        raise ValueError(f"No {mode} records were present in the supplied CDF file(s)")

    return _combine(
        frames,
        {
            "mission": "ARTEMIS-P1" if probe == "b" else "ARTEMIS-P2",
            "instrument": "FGM",
            "product": f"L2 {mode}",
            "frame": coordinate.upper(),
            "vector_variable": f"{prefix}_{coordinate}",
            "scalar_source": f"norm({prefix}_gse)",
            "field_units": "nT",
            "target": "Moon",
            "role": "upstream lunar control" if probe == "b" else "lunar-wake response",
            "source": "NASA SPDF/CDAWeb",
            "quality_rule": "unmasked",
        },
    )


def read_artemis_esa_ions(
    paths: Path | str | Iterable[Path | str],
    *,
    probe: str = "c",
    mode: str = "peir",
) -> pd.DataFrame:
    """Read ARTEMIS ESA ion moments in GSE at native product cadence.

    ``peir`` provides reduced-mode ions at roughly 3--4 s cadence; ``peif``
    provides the full-mode spin snapshots.  Quality flags are retained and are
    not masked.
    """
    probe = probe.lower()
    mode = mode.lower()
    if probe not in {"b", "c"}:
        raise ValueError("ARTEMIS probe must be 'b' (P1) or 'c' (P2)")
    if mode not in {"peif", "peir"}:
        raise ValueError("ESA ion mode must be one of: peif, peir")

    prefix = f"th{probe}_{mode}"
    frames: list[pd.DataFrame] = []

    for path in _normalise_paths(paths):
        cdf = CDF(str(path))
        try:
            if _last_record(cdf, f"{prefix}_time") < 0:
                continue
            index = _unix_index(cdf, f"{prefix}_time")
            density = _valid_numeric(cdf, f"{prefix}_density")
            velocity = _valid_numeric(cdf, f"{prefix}_velocity_gse")
            temperature = _valid_numeric(cdf, f"{prefix}_avgtemp")
            quality = np.asarray(
                cdf.varget(f"{prefix}_data_quality"), dtype=np.int32
            )
        finally:
            close = getattr(cdf, "close", None)
            if close is not None:
                close()

        if velocity.ndim != 2 or velocity.shape[1] != 3:
            raise ValueError(
                f"{prefix}_velocity_gse in {path} does not have shape (N, 3)"
            )

        frame = pd.DataFrame(
            velocity, index=index, columns=["vx_gse", "vy_gse", "vz_gse"]
        )
        frame.insert(0, "density", density)
        frame["speed"] = np.linalg.norm(velocity, axis=1)
        frame["temperature"] = temperature
        frame["quality"] = quality
        frames.append(frame)

    if not frames:
        raise ValueError(f"No {mode} records were present in the supplied CDF file(s)")

    return _combine(
        frames,
        {
            "mission": "ARTEMIS-P1" if probe == "b" else "ARTEMIS-P2",
            "instrument": "ESA",
            "product": f"L2 {mode} ion moments",
            "frame": "GSE",
            "density_units": "cm^-3",
            "velocity_units": "km/s",
            "temperature_units": "eV",
            "target": "Moon",
            "role": "upstream lunar control" if probe == "b" else "lunar-wake response",
            "source": "NASA SPDF/CDAWeb",
            "quality_rule": "unmasked",
        },
    )
