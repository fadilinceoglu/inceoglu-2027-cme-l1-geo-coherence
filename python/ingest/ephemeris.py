"""Read the provenance-frozen SSCWeb GSE geometry cache for this study.

The cached Earth-local tracks are geocentric GSE Cartesian coordinates in
Earth radii (R_E = 6371.2 km, the SSCWeb/IRBEM conversion convention). The
publication pipeline never performs a live ephemeris request.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RE_KM = 6371.2  # Earth radius (km), IRBEM/spacepy convention

# SSCWeb id -> friendly label used throughout the study.
SSC_L1 = {
    "ace": "ACE",
    "dscovr": "DSCOVR",
    "wind": "Wind",
    "solar1": "Solar-1",
    "goes18": "GOES-18",
    "goes19": "GOES-19",
    "moon": "Moon",
    "themisb": "ARTEMIS-P1",
    "themisc": "ARTEMIS-P2",
}


def _cxf_find(node: object, key: str) -> object | None:
    """Walk SSCWeb's CXF-encoded JSON (nested [className, {...}] / [type, [...]])
    and return the value stored under ``key`` in the first dict that has it."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            found = _cxf_find(v, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _cxf_find(item, key)
            if found is not None:
                return found
    return None


def _cxf_list(node: object) -> list:
    """Unwrap a CXF list of the form ["java.util.ArrayList", [ ... ]]."""
    if isinstance(node, list) and len(node) == 2 and isinstance(node[1], list):
        return node[1]
    if isinstance(node, list):
        return node
    return []


def parse_ssc_gse_payload(payload: object) -> dict[str, pd.DataFrame]:
    """Parse one SSCWeb CXF-JSON locations payload without network access.

    The returned tracks use the study labels, GSE columns, Earth-radius units,
    and a timezone-naive UTC index. The parser consumes a provenance-stamped
    raw cache and never depends on a live service.
    """

    data_block = _cxf_find(payload, "Data")
    sats = _cxf_list(data_block)
    if not sats:
        raise ValueError("SSCWeb payload contains no satellite data")

    out: dict[str, pd.DataFrame] = {}
    for sat in sats:
        sat_id = _cxf_find(sat, "Id")
        coord = _cxf_list(_cxf_find(sat, "Coordinates"))
        if not coord:
            continue
        cdat = coord[0]
        coordinate_system = str(_cxf_find(cdat, "CoordinateSystem")).upper()
        if coordinate_system != "GSE":
            raise ValueError(
                f"SSCWeb payload for {sat_id!r} is {coordinate_system}, not GSE"
            )
        x = np.array(_cxf_list(_cxf_find(cdat, "X")), dtype=float)
        y = np.array(_cxf_list(_cxf_find(cdat, "Y")), dtype=float)
        z = np.array(_cxf_list(_cxf_find(cdat, "Z")), dtype=float)

        times_raw = _cxf_list(_cxf_find(sat, "Time"))
        times = pd.to_datetime(
            [t[1] if isinstance(t, list) else t for t in times_raw],
            utc=True,
        ).tz_convert(None)
        if not (len(times) == len(x) == len(y) == len(z)):
            raise ValueError(
                f"SSCWeb payload length mismatch for satellite {sat_id!r}"
            )

        frame = pd.DataFrame(
            {
                "x_gse": x / RE_KM,
                "y_gse": y / RE_KM,
                "z_gse": z / RE_KM,
            },
            index=times,
        )
        frame.index.name = "time_utc"
        frame["r"] = np.sqrt(
            frame.x_gse**2 + frame.y_gse**2 + frame.z_gse**2
        )
        label = SSC_L1.get(str(sat_id).lower(), str(sat_id).upper())
        out[label] = frame.sort_index()

    if not out:
        raise ValueError("SSCWeb payload contains no usable GSE tracks")
    return out


def read_ssc_gse_cache(path: Path | str) -> dict[str, pd.DataFrame]:
    """Read a cached SSCWeb CXF-JSON locations response."""

    with Path(path).open() as handle:
        payload = json.load(handle)
    return parse_ssc_gse_payload(payload)
