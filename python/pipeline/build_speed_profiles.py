"""Construct the seven Methods speed profiles from this run's plasma records.

Observation preparation supplies one-minute median plasma records. This stage
applies the local and proxy speed policies used for convected-distance mapping.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config.event_boundaries import UPSTREAM_SPACECRAFT
from analysis.event_boundaries import shock_measurement, shock_delay
from config.paths import PROCESSED_DIR


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["time"], index_col="time")
    frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.has_duplicates:
        raise ValueError(f"duplicate plasma minute timestamps in {path}")
    return frame.sort_index()


def own_speed(frame: pd.DataFrame) -> pd.Series:
    """Preserve the original bounded preparation before final UTC interpolation.

    Remaining gaps, including the declared Solar Orbiter gap and record-edge
    proxy gaps, are filled by the spatial-filter builder's Methods mapping.
    """
    return frame.v.interpolate(method="time", limit=30, limit_direction="both")


def shifted_reference(reference: pd.Series, index: pd.DatetimeIndex, shift_s: float) -> pd.Series:
    shifted = reference.copy()
    shifted.index = shifted.index + pd.Timedelta(seconds=shift_s)
    return shifted.reindex(index, method="nearest", tolerance=pd.Timedelta("90s"))


def artemis_speed(own: pd.Series, proxy: pd.Series) -> tuple[pd.Series, dict]:
    own = own.where((own > 100) & (own < 3000))
    valid_deviation = (own / proxy - 1).abs() <= 0.30
    rejected = own.notna() & proxy.notna() & ~valid_deviation
    kept = own.where(valid_deviation | proxy.isna())
    return kept.combine_first(proxy), {
        "own_speed_fraction": float(kept.notna().mean()),
        "rejected_by_deviation": int(rejected.sum()),
    }


def build(directory: Path = PROCESSED_DIR) -> dict:
    records = {key: _read(directory / f"{key}_overview_1min.csv") for key in UPSTREAM_SPACECRAFT}
    output = {key: own_speed(records[key]) for key in ("swfo", "wind", "solar_orbiter")}
    provenance = {key: {"method": "own proton speed; bounded internal UTC interpolation"} for key in output}
    reference = output["swfo"]
    for key in ("ace", "dscovr", "artemis_p1", "artemis_p2"):
        frame = records[key]
        shift = shock_delay(key)
        proxy = shifted_reference(reference, frame.index, shift)
        details = {"solar1_timestamp_shift_s": shift, "nearest_tolerance_s": 90}
        if key == "ace":
            speed = proxy
            details["method"] = "shifted Solar-1 speed throughout the record"
        elif key == "dscovr":
            speed = proxy.copy()
            before = frame.index < shock_measurement(key)["t50"]
            speed.loc[before] = own_speed(frame).loc[before].combine_first(proxy.loc[before])
            details["method"] = "own speed before the adopted shock minute; shifted Solar-1 afterward"
        else:
            speed, qc = artemis_speed(frame.v, proxy)
            details.update(qc)
            details["method"] = "own merged ESA speed within (100,3000) km/s and 30% of available proxy; Solar-1 otherwise"
        output[key], provenance[key] = speed, details
    for key, speed in output.items():
        speed.rename("v_frozen").to_csv(directory / f"{key}_vt_frozen.csv", index_label="time")
        provenance[key]["source_plasma"] = str(directory / f"{key}_overview_1min.csv")
    (directory / "vt_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
