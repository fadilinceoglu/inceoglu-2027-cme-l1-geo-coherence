"""Centered-dipole magnetic local time from supplied native GOES positions.

The orbit is the NOAA avg1m MAG product's geocentric latitude, longitude and
radius. SpacePy CTrans rotates each observed position from GEO to SM at its
native UTC epoch, using the same verified IGRF14/UTC--TAI backend as ingestion.
Crossing clocks are measurements of that orbit, never plotting constants.
"""
from __future__ import annotations

from pathlib import Path
import re
import warnings
from typing import Mapping

import netCDF4
import numpy as np
import pandas as pd

from analysis.paired_goes_response import DISPLAY_INDEX
from config.paths import OUTPUT_ROOT
from ingest import gsm_coordinates
from ingest.goes import J2K_UTC, _numeric

SPACECRAFT = ("g18", "g19")
CARDINAL_MLT = {"Dawn": 6.0, "Noon": 12.0, "Dusk": 18.0, "Midnight": 0.0}
# Physical radius convention used by the supplied SSCWeb/IRBEM geometry.
EARTH_RADIUS_M = 6_371_200.0
ORBIT_COLUMNS = ("latitude_geo_deg", "longitude_geo_deg", "radius_m")
MARKER_COLUMNS = ("spacecraft", "label", "mlt_hours", "time_utc")


def minute_clock(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Require an explicit, complete ordered UTC minute clock."""
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("GOES local-time clock must be a timezone-aware DatetimeIndex")
    index = index.tz_convert("UTC").as_unit("ns")
    if index.empty or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("GOES local-time clock must be nonempty, finite, unique and ordered")
    if np.any(index.asi8 % pd.Timedelta(minutes=1).value):
        raise ValueError("GOES local-time clock must contain whole UTC minutes")
    if len(index) > 1 and np.any(np.diff(index.asi8) != pd.Timedelta(minutes=1).value):
        raise ValueError("GOES local-time clock must contain every consecutive minute")
    return index


def read_orbit(path: Path | str, spacecraft: str, day: str) -> pd.DataFrame:
    """Decode a declared daily NOAA avg1m source, including renamed local files."""
    path = Path(path)
    with netCDF4.Dataset(path) as dataset:
        product = str(getattr(dataset, "id", ""))
        if not re.fullmatch(rf"dn_magn-l2-avg1m_{spacecraft}_d{day}_v[\w.-]+\.nc", product):
            raise ValueError(f"Expected NOAA avg1m {spacecraft} orbit for {day}: {path}")
        if getattr(dataset, "platform", None) != spacecraft:
            raise ValueError(f"GOES orbit platform does not match {spacecraft}: {path}")
        missing = {"time", "orbit_llr_geo"}.difference(dataset.variables)
        if missing:
            raise ValueError(f"Missing native GOES orbit variables {sorted(missing)}: {path}")
        clock, orbit = dataset["time"], dataset["orbit_llr_geo"]
        if getattr(clock, "units", None) != "seconds since 2000-01-01 12:00:00":
            raise ValueError(f"Unexpected native GOES orbit time units: {path}")
        if getattr(orbit, "units", None) != "degrees, degrees, meters":
            raise ValueError(f"Unexpected native GOES orbit coordinate units: {path}")
        seconds, values = _numeric(clock), _numeric(orbit)
    if seconds.ndim != 1 or not np.isfinite(seconds).all():
        raise ValueError(f"Native GOES orbit clocks must be finite vectors: {path}")
    index = minute_clock(pd.DatetimeIndex(J2K_UTC + pd.to_timedelta(seconds, unit="s")))
    if not (index.normalize() == pd.Timestamp(day, tz="UTC")).all():
        raise ValueError(f"Native GOES orbit clocks do not match declared day {day}: {path}")
    if values.shape != (len(index), 3):
        raise ValueError(f"Native GOES orbit must have shape (time, 3): {path}")
    if (not np.isfinite(values).all() or np.any(abs(values[:, 0]) > 90)
            or np.any(abs(values[:, 1]) > 360) or np.any(values[:, 2] <= 0)):
        raise ValueError(f"Native GOES orbit must contain valid finite latitude/longitude/radius: {path}")
    return pd.DataFrame(values, index=index, columns=ORBIT_COLUMNS)


def mlt_from_sm(position: np.ndarray) -> np.ndarray:
    """SM longitude clock: noon=12, dusk=18, midnight=0 and dawn=6 hours."""
    position = np.asarray(position, dtype=float)
    if (position.ndim != 2 or position.shape[1] != 3 or not np.isfinite(position).all()
            or np.any(np.hypot(position[:, 0], position[:, 1]) == 0)):
        raise ValueError("MLT requires finite SM positions with a defined magnetic longitude")
    return (12.0 + np.arctan2(position[:, 1], position[:, 0]) * 12.0 / np.pi) % 24.0


def _native_window(orbit: pd.DataFrame, display_index: pd.DatetimeIndex) -> pd.DataFrame:
    clock = minute_clock(orbit.index)
    stop = display_index[-1] + pd.Timedelta(minutes=1)
    left = clock.searchsorted(display_index[0], side="right") - 1
    right = clock.searchsorted(stop, side="left")
    if left < 0 or right >= len(clock):
        raise ValueError("Native GOES orbit must bracket the complete display window")
    selected = orbit.iloc[left:right + 1].copy()
    selected.index = clock[left:right + 1]
    if not np.isfinite(selected.loc[:, list(ORBIT_COLUMNS)].to_numpy(float)).all():
        raise ValueError("Native GOES orbit contains missing display-window coordinates")
    return selected


def cardinal_crossings(mlt: pd.Series, spacecraft: str, *,
                       display_index: pd.DatetimeIndex = DISPLAY_INDEX) -> pd.DataFrame:
    """Interpolate each crossing between its adjacent native unwrapped samples."""
    display_index = minute_clock(display_index)
    clock = minute_clock(mlt.index)
    values = mlt.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values >= 24)):
        raise ValueError("Native MLT must be finite in [0, 24) hours")
    if len(clock) < 2:
        raise ValueError("MLT crossings require at least two native samples")
    unwrapped = np.unwrap(values * np.pi / 12.0) * 12.0 / np.pi
    start, stop = display_index[0], display_index[-1] + pd.Timedelta(minutes=1)
    rows = []
    for label, cardinal in CARDINAL_MLT.items():
        first = int(np.floor((unwrapped.min() - cardinal) / 24.0))
        last = int(np.ceil((unwrapped.max() - cardinal) / 24.0))
        for turn in range(first, last + 1):
            target = cardinal + turn * 24.0
            for i in np.flatnonzero(((unwrapped[:-1] <= target) & (target < unwrapped[1:]))
                                    | ((unwrapped[1:] < target) & (target <= unwrapped[:-1]))):
                fraction = (target - unwrapped[i]) / (unwrapped[i + 1] - unwrapped[i])
                crossing = clock[i] + (clock[i + 1] - clock[i]) * fraction
                if start <= crossing < stop:
                    rows.append(dict(spacecraft=spacecraft, label=label, mlt_hours=cardinal,
                                     time_utc=crossing))
    result = pd.DataFrame(rows, columns=MARKER_COLUMNS)
    return result.drop_duplicates().sort_values("time_utc").reset_index(drop=True)


def derive_local_time(orbits: Mapping[str, pd.DataFrame], *,
                      display_index: pd.DatetimeIndex = DISPLAY_INDEX,
                      reference_directory: Path | str = gsm_coordinates.REFERENCE_DIRECTORY,
                      output_directory: Path | str = OUTPUT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Recompute minute MLT and native-bracket crossing measurements for both craft."""
    display_index = minute_clock(display_index)
    if set(orbits) != set(SPACECRAFT):
        raise ValueError("Local time requires native GOES-18 and GOES-19 orbits")
    selected = {craft: _native_window(orbits[craft], display_index) for craft in SPACECRAFT}
    # Verify on every run even when the process-local backend is already loaded.
    references = gsm_coordinates.reference_manifest(reference_directory)
    coord, spt, igrf, _ = gsm_coordinates._load_backend(
        str(Path(reference_directory).resolve(strict=True)), str(Path(output_directory).resolve()))
    minutes, markers = [], []
    for craft, native in selected.items():
        utc = native.index.tz_localize(None).to_pydatetime()
        if utc[0] < igrf.igrfcoeffs.datelow or utc[-1] > igrf.igrfcoeffs.datehigh:
            raise ValueError("Native GOES orbit epochs lie outside supplied IGRF14 validity")
        latitude = np.deg2rad(native.latitude_geo_deg.to_numpy(float))
        longitude = np.deg2rad(native.longitude_geo_deg.to_numpy(float))
        radius = native.radius_m.to_numpy(float) / EARTH_RADIUS_M
        geo = np.column_stack((radius * np.cos(latitude) * np.cos(longitude),
                               radius * np.cos(latitude) * np.sin(longitude), radius * np.sin(latitude)))
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=UserWarning, module=r"spacepy(\..*)?")
            sm = coord.Coords(geo, "GEO", "car", ticks=spt.Ticktock(utc, "UTC"),
                              use_irbem=False).convert("SM", "car").data
        if not np.allclose(np.linalg.norm(sm, axis=1), radius, rtol=1e-12, atol=1e-12):
            raise RuntimeError("GEO-to-SM orbit transformation failed radius preservation")
        native_mlt = pd.Series(mlt_from_sm(sm), index=native.index, name="mlt_hours")
        # The avg1m orbit epochs and display clock are exact UTC minutes. Require
        # every native sample, rather than inventing a missing orbital position.
        minute = native_mlt.reindex(display_index).to_frame()
        minute["mlt_next_hours"] = native_mlt.reindex(display_index + pd.Timedelta(minutes=1)).to_numpy(float)
        if not np.isfinite(minute[["mlt_hours", "mlt_next_hours"]]).all().all():
            raise ValueError("Native GOES orbit lacks a required display minute")
        minute.insert(0, "spacecraft", craft)
        minute.index.name = "time_utc"
        minutes.append(minute)
        markers.append(cardinal_crossings(native_mlt, craft, display_index=display_index))
    provenance = dict(
        method="SpacePy CTrans GEO-to-SM position rotation at each native observed UTC epoch",
        spacepy_version=gsm_coordinates.SPACEPY_VERSION, input_frame="GEO", output_frame="SM",
        backend="use_irbem=False; itol=0; no temporal rotation interpolation",
        geomagnetic_model="IGRF14 centered dipole from date-adjusted coefficients",
        model_valid_utc=[str(igrf.igrfcoeffs.datelow), str(igrf.igrfcoeffs.datehigh)],
        orbit_variable="NOAA MAG avg1m orbit_llr_geo: geocentric latitude/longitude/radius in degrees/degrees/meters",
        radius_convention_m_per_re=EARTH_RADIUS_M,
        mlt_definition="(12 + atan2(y_SM, x_SM) * 12/pi) modulo 24 hours",
        minute_sampling="native exact UTC-minute orbit samples selected on the shared GOES display clock",
        next_minute_column="mlt_next_hours retains the following native minute through the exclusive display stop for complete crossing validation",
        crossing_method="linear interpolation of unwrapped MLT between immediate previous and next native minute samples; display start inclusive and stop exclusive",
        cardinal_mlt_hours=CARDINAL_MLT,
        UT1_policy="SpacePy CTrans default UT1-UTC=0; frozen supplied UTC--TAI reference",
        network_access=False, reference_inputs=references["references"],
    )
    return pd.concat(minutes), pd.concat(markers, ignore_index=True), provenance


def load_local_time(minute_path: Path | str, markers_path: Path | str, *,
                    display_index: pd.DatetimeIndex = DISPLAY_INDEX) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the complete current stage interface, failing on invalid saved products."""
    display_index = minute_clock(display_index)
    minutes, markers = (pd.read_csv(path, float_precision="round_trip")
                        for path in (minute_path, markers_path))
    if not {"time_utc", "spacecraft", "mlt_hours", "mlt_next_hours"}.issubset(minutes.columns):
        raise ValueError("GOES local-time product lacks required columns")
    minutes.index = pd.DatetimeIndex(pd.to_datetime(minutes.pop("time_utc"), utc=True), name="time_utc")
    if set(minutes.spacecraft) != set(SPACECRAFT):
        raise ValueError("GOES local-time product requires both spacecraft")
    expected_markers = []
    for craft in SPACECRAFT:
        selected = minutes.loc[minutes.spacecraft.eq(craft)]
        if not selected.index.equals(display_index):
            raise ValueError(f"{craft} local time must contain the complete ordered display clock")
        values = selected[["mlt_hours", "mlt_next_hours"]].to_numpy(float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values >= 24)):
            raise ValueError("GOES local time must be finite in [0, 24) hours")
        if not np.array_equal(values[:-1, 1], values[1:, 0]):
            raise ValueError("GOES local-time next-minute brackets disagree with the minute clock")
        native_clock = display_index.append(pd.DatetimeIndex([display_index[-1] + pd.Timedelta(minutes=1)]))
        native = pd.Series(np.append(values[:, 0], values[-1, 1]), index=native_clock)
        expected_markers.append(cardinal_crossings(native, craft, display_index=display_index))
    if not set(MARKER_COLUMNS).issubset(markers.columns):
        raise ValueError("GOES local-time markers lack required columns")
    markers["time_utc"] = pd.to_datetime(markers.time_utc, utc=True)
    start, stop = display_index[0], display_index[-1] + pd.Timedelta(minutes=1)
    for row in markers.itertuples(index=False):
        if (row.spacecraft not in SPACECRAFT or row.label not in CARDINAL_MLT
                or row.mlt_hours != CARDINAL_MLT[row.label]
                or pd.isna(row.time_utc) or not start <= row.time_utc < stop):
            raise ValueError("Invalid GOES local-time crossing marker")
    if markers.duplicated(["spacecraft", "label", "time_utc"]).any():
        raise ValueError("Duplicate GOES local-time crossing markers")
    expected = pd.concat(expected_markers, ignore_index=True)
    order = ["spacecraft", "time_utc", "label"]
    expected = expected.sort_values(order).reset_index(drop=True)
    observed = markers.sort_values(order).reset_index(drop=True)
    if len(expected) != len(observed):
        raise ValueError("GOES local-time markers must contain every current MLT crossing")
    for actual, wanted in zip(observed.itertuples(index=False), expected.itertuples(index=False), strict=True):
        if (actual.spacecraft != wanted.spacecraft or actual.label != wanted.label
                or actual.mlt_hours != wanted.mlt_hours or actual.time_utc != wanted.time_utc):
            raise ValueError("GOES local-time markers disagree with current MLT crossing clocks")
    return minutes, markers
