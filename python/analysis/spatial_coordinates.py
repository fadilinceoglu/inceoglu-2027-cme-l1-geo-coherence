"""Minute samples of the one-second coordinate shared by spectra and filters.

The exact shock coordinate is also retained for inverse interpolation. Readers
verify the current observations, frozen speed and boundaries before using this
interface; they never reconstruct a second production distance mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.product_io import utc_text
from analysis.waveform_scales import sha256
from config import study
from config.waveform_bands import SPATIAL_DX_RE


def sample_coordinates(
    time: pd.DatetimeIndex, distance_re: np.ndarray, anchor: str | pd.Timestamp,
) -> pd.Series:
    """Sample the existing mapping at UTC minutes and its measured zero."""
    anchor = pd.Timestamp(anchor)
    values = np.asarray(distance_re, dtype=float)
    if (time.tz is None or anchor.tz is None or not time.is_unique
            or not time.is_monotonic_increasing or len(time) < 2
            or values.shape != (len(time),) or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0)):
        raise ValueError("invalid prepared spatial coordinate")
    if not time[0] <= anchor <= time[-1]:
        raise ValueError("measured shock lies outside the prepared coordinate")
    minutes = pd.date_range(time[0].ceil("min"), time[-1].floor("min"), freq="min")
    samples = minutes.union(pd.DatetimeIndex([anchor])).sort_values()
    # Relative seconds avoid precision loss from floating-point epoch nanoseconds.
    seconds = (time - time[0]).total_seconds().to_numpy()
    sampled_x = np.interp((samples - time[0]).total_seconds(), seconds, values)
    coordinate = sampled_x - sampled_x[samples.get_loc(anchor)]
    return pd.Series(coordinate, index=samples, name="distance_re")


def write_coordinate_record(
    processed: Path, craft: str, coordinate: pd.Series, anchor: str | pd.Timestamp,
    native_path: Path, speed_path: Path, *, fingerprints: dict[str, str],
) -> tuple[Path, dict]:
    path = processed / "spatial_coordinates" / f"{craft}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinate.to_csv(path, index_label="time", float_format="%.17g", lineterminator="\n")

    def source_hash(source: Path) -> str:
        key = str(source.relative_to(processed.parent))
        return fingerprints[key] if key in fingerprints else sha256(source)

    return path, {
        "anchor_utc": utc_text(anchor), "row_count": len(coordinate),
        "native_sha256": source_hash(native_path), "speed_sha256": source_hash(speed_path),
        "coordinate_sha256": sha256(path),
    }


def read_coordinates(
    processed: Path, crafts: tuple[str, ...] = study.UPSTREAM_SPACECRAFT,
    *, verified_inputs: dict[str, str] | None = None,
) -> dict[str, pd.Series]:
    """Validate prerequisites, reusing fingerprints from a verified band design.

    Standalone readers hash every input. A stage that already called
    ``load_band_design`` can pass its verified input path/digest map.
    """
    checked = dict(verified_inputs or {})

    def current_hash(path: Path) -> str:
        key = str(path.relative_to(processed.parent))
        if key not in checked:
            checked[key] = sha256(path)
        return checked[key]
    metadata_path = processed / "spatial_coordinates.json"
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get("schema_version") != "shared-spatial-coordinates-v1"
            or metadata.get("spatial_grid_step_re") != SPATIAL_DX_RE):
        raise ValueError("invalid shared spatial-coordinate method/schema")
    boundary_path = processed.parent / "results" / "boundaries.json"
    if current_hash(boundary_path) != metadata["boundary_sha256"]:
        raise ValueError("stale spatial coordinates: boundaries changed; rebuild filters")
    boundaries = json.loads(boundary_path.read_text())
    coordinates = {}
    for craft in crafts:
        entry = metadata["crafts"][craft]
        anchor = pd.Timestamp(boundaries[craft]["shock"])
        if anchor.tz is None or pd.Timestamp(entry["anchor_utc"]) != anchor:
            raise ValueError(f"{craft}: spatial-coordinate shock anchor disagrees with boundaries")
        path = processed / "spatial_coordinates" / f"{craft}.csv"
        for source, key in (
            (processed / "native" / f"{craft}_1s.csv", "native_sha256"),
            (processed / f"{craft}_vt_frozen.csv", "speed_sha256"),
            (path, "coordinate_sha256"),
        ):
            if current_hash(source) != entry[key]:
                raise ValueError(f"stale spatial coordinates: {source.name} changed; rebuild filters")
        frame = pd.read_csv(path, float_precision="round_trip")
        if list(frame.columns) != ["time", "distance_re"]:
            raise ValueError(f"{craft}: invalid spatial-coordinate columns")
        time = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True, format="ISO8601"))
        values = pd.to_numeric(frame.distance_re, errors="raise").to_numpy(dtype=float)
        if len(frame) < 2:
            raise ValueError(f"{craft}: invalid shared spatial-coordinate grid")
        expected = pd.date_range(time[0].ceil("min"), time[-1].floor("min"), freq="min")
        expected = expected.union(pd.DatetimeIndex([anchor])).sort_values()
        if (len(frame) != entry["row_count"]
                or not time.equals(expected) or not np.isfinite(values).all()
                or np.any(np.diff(values) <= 0)
                or values[time.get_loc(anchor)] != 0.0):
            raise ValueError(f"{craft}: invalid shared spatial-coordinate grid")
        coordinates[craft] = pd.Series(values, index=time, name="distance_re")
    return coordinates


def coordinate_time_at(coordinate: pd.Series, distance_re: float) -> pd.Timestamp:
    """Inverse interpolation on the saved shared coordinate."""
    values = coordinate.to_numpy(dtype=float)
    if not np.isfinite(distance_re) or not values[0] <= distance_re <= values[-1]:
        raise ValueError("requested distance lies outside shared coordinate support")
    exact = np.searchsorted(values, distance_re)
    if values[exact] == distance_re:
        return coordinate.index[exact]
    seconds = (coordinate.index - coordinate.index[0]).total_seconds().to_numpy()
    offset = np.interp(distance_re, values, seconds)
    return coordinate.index[0] + pd.Timedelta(seconds=offset)
