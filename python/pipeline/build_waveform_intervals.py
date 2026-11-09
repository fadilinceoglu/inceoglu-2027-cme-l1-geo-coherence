"""Figure 4 intervals from filtered band energy, for every spacecraft and family.

The search window runs from the minute containing each spacecraft's shock minus
the search margin to the minute containing its ejecta end plus the same margin.
Inside it, the interval spans the minutes where the cumulative saved one-minute
bt_f squared reaches the configured lower and upper energy fractions. The
padding is then added on each side. The mean-retaining trend uses the same rule;
its bt_f squared is essentially the magnetic energy |B|^2.

Output: <output-dir>/results/waveform_intervals.json

Run from python/: PYTHONPATH=. python pipeline/build_waveform_intervals.py
"""

from __future__ import annotations

from functools import partial
import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from analysis.product_io import boolean_values, utc_text
from common.io import display_path, read_json, utc_index, utc_timestamp
from analysis.waveform_intervals import energy_edges
from config import study
from config.paths import PROCESSED_DIR, PYTHON_ROOT, RESULTS_DIR
from config.waveform_bands import (
    INTERVAL_ENERGY_FRACTIONS,
    INTERVAL_PADDING_MINUTES,
    INTERVAL_SEARCH_MARGIN_MINUTES,
    WAVEFORM_FAMILIES,
)

SCHEMA_VERSION: Final = "waveform-intervals-v1.0.0"
OUTPUT: Final = RESULTS_DIR / "waveform_intervals.json"
BOUNDARIES: Final = RESULTS_DIR / "boundaries.json"
START: Final = pd.Timestamp(study.ANALYSIS_START)
STOP: Final = pd.Timestamp(study.ANALYSIS_STOP)
MINUTE: Final = pd.Timedelta(minutes=1)


def _utc_minute(value: object) -> pd.Timestamp:
    return utc_timestamp(value).floor("min")


_display_path = partial(display_path, root=PYTHON_ROOT)


def _band(craft: str, family: str) -> pd.DataFrame:
    path = PROCESSED_DIR / "bands" / f"{craft}__{family}.csv"
    frame = pd.read_csv(path, parse_dates=["time"], index_col="time")
    frame.index = utc_index(frame.index)
    if not frame.index.equals(pd.date_range(START, STOP, freq="1min", inclusive="left")):
        raise ValueError(f"{path.name} is not on the four-day one-minute grid")
    return frame


def interval_record(frame: pd.DataFrame, shock: object, ejecta_end: object) -> dict[str, Any]:
    """Energy-edge interval and padded support for one saved band record."""

    margin = pd.Timedelta(minutes=INTERVAL_SEARCH_MARGIN_MINUTES)
    padding = pd.Timedelta(minutes=INTERVAL_PADDING_MINUTES)
    search_start = max(START, _utc_minute(shock) - margin)
    search_stop = min(STOP, _utc_minute(ejecta_end) + margin)
    if search_stop <= search_start:
        raise ValueError("interval search window is empty")
    window = frame.loc[search_start:search_stop - MINUTE]
    values = window["bt_f"].to_numpy(dtype=float)
    if not (boolean_values(window["valid"], name="valid").all() and np.isfinite(values).all()):
        raise ValueError("interval search window contains invalid band rows")
    first, stop = energy_edges(values, *INTERVAL_ENERGY_FRACTIONS)
    energy_start = window.index[first]
    energy_stop = window.index[stop - 1] + MINUTE
    start = max(START, energy_start - padding)
    stop_exclusive = min(STOP, energy_stop + padding)
    return {
        "search_start_utc": utc_text(search_start),
        "search_stop_utc_exclusive": utc_text(search_stop),
        "energy_start_utc": utc_text(energy_start),
        "energy_stop_utc_exclusive": utc_text(energy_stop),
        "start_utc": utc_text(start),
        "stop_utc_exclusive": utc_text(stop_exclusive),
        "duration_minutes": int((stop_exclusive - start) / MINUTE),
    }


def build(output: Path = OUTPUT) -> Path:
    boundaries = read_json(BOUNDARIES)
    intervals: dict[str, dict[str, Any]] = {}
    for craft in study.UPSTREAM_SPACECRAFT:
        entry = boundaries.get(craft)
        if not isinstance(entry, dict) or not entry.get("shock") or not entry.get("ejecta_end"):
            raise ValueError(f"{craft} lacks a measured shock or ejecta end")
        intervals[craft] = {
            family: interval_record(_band(craft, family), entry["shock"], entry["ejecta_end"])
            for family in WAVEFORM_FAMILIES
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": {
            "energy": "cumulative sum of the squared saved one-minute bt_f",
            "energy_fractions": list(INTERVAL_ENERGY_FRACTIONS),
            "search_window": (
                "minute containing the shock minus the margin, to the minute "
                "containing the ejecta end plus the margin (half-open)"
            ),
            "search_margin_minutes": INTERVAL_SEARCH_MARGIN_MINUTES,
            "padding_minutes": INTERVAL_PADDING_MINUTES,
        },
        "boundaries_path": _display_path(BOUNDARIES),
        "spacecraft_order": list(study.UPSTREAM_SPACECRAFT),
        "family_order": list(WAVEFORM_FAMILIES),
        "intervals": intervals,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output


if __name__ == "__main__":
    print(build())
