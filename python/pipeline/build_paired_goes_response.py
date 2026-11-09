"""Build observed GOES quantities and two-day reference sensitivity.

Only newly processed one-minute GOES records are required. No orbit,
upstream-context, region-classification or response-model calculation is
part of this observational product.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.paired_goes_response import (
    DISPLAY_INDEX, DISPLAY_PADDING_MINUTES, DISPLAY_START,
    EVENT_INDEX, EVENT_START, EVENT_STOP, paired_response,
)
from config.paths import RESULTS_DIR, PROCESSED_DIR

OUT = RESULTS_DIR
PREFIX = "paired_goes_response"
RECORDS = {
    "g18": PROCESSED_DIR / "common_cadence_1min/goes18_gapfilled.csv",
    "g19": PROCESSED_DIR / "common_cadence_1min/goes19_gapfilled.csv",
}
REFERENCES = {"two_quiet_days": (1, 2), "one_day_before": (1,), "two_days_before": (2,)}


def _record(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("time"), utc=True), name="time_utc")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"unsorted or duplicate clocks in {path}")
    return frame


def build(output_dir: Path = OUT) -> dict[str, Any]:
    records = {key: _record(path) for key, path in RECORDS.items()}
    results = {}
    for reference, days in REFERENCES.items():
        frame = paired_response(records, event_index=DISPLAY_INDEX, control_days=days)
        frame.insert(0, "reference", reference)
        if not frame.index.equals(DISPLAY_INDEX):
            raise ValueError(f"the GOES response must contain all {len(DISPLAY_INDEX)} display minutes")
        results[reference] = frame
    tables = {
        "primary": results["two_quiet_days"],
        "reference_sensitivity": pd.concat([results["one_day_before"], results["two_days_before"]]),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "goes-magnetic-response-v3",
        "event": {"start_utc": EVENT_START.isoformat(), "stop_exclusive_utc": EVENT_STOP.isoformat(), "minutes_per_spacecraft": len(EVENT_INDEX)},
        "display": {"start_utc": DISPLAY_START.isoformat(), "stop_exclusive_utc": EVENT_STOP.isoformat(),
                    "pre_onset_minutes": DISPLAY_PADDING_MINUTES, "minutes_per_spacecraft": len(DISPLAY_INDEX)},
        "definitions": {
            "reference": "own-spacecraft arithmetic mean at the same UTC minute on the two preceding days; both controls required",
            "magnitude": "mean native scalar magnitude, averaged separately from GSM components",
            "angle_quiet_deg": "angle between the observed minute-mean GSM vector and its averaged quiet vector",
            "bz_std": "sample standard deviation of native Bz measurements in each minute; n-1 denominator",
            "reference_sensitivity": "pointwise range of departures using each preceding day separately; not a confidence interval",
        },
        "inputs": [{"spacecraft": key, "path": str(path.relative_to(PROCESSED_DIR)), "path_basis": "processed_directory"} for key, path in RECORDS.items()],
        "outputs": {},
    }
    for name, frame in tables.items():
        suffix = "" if name == "primary" else "_reference_sensitivity"
        path = output_dir / f"{PREFIX}{suffix}.csv"
        frame.to_csv(path, index=True, index_label="time_utc", date_format="%Y-%m-%dT%H:%M:%SZ")
        manifest["outputs"][name] = {"path": path.name, "row_count": len(frame)}
    (output_dir / f"{PREFIX}.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    result = build()
    print(json.dumps(result["event"], indent=2))
