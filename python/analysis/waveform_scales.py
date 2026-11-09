"""Measure reference scales and read the resulting current-run filter design."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from analysis.structure_bands import EARTH_RADIUS_KM
from common.io import read_json, sha256
from config import waveform_bands as settings
from config.paths import PROCESSED_DIR

FAMILIES = settings.WAVEFORM_FAMILIES


def _interval_coordinates_re(distance_re: pd.Series, start, stop) -> np.ndarray:
    """Evaluate the existing time-to-distance mapping at measured boundaries."""
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    if not distance_re.index[0] <= start < stop <= distance_re.index[-1]:
        raise ValueError("physical interval is unordered or outside the distance mapping")
    elapsed = (distance_re.index.asi8 - distance_re.index.asi8[0]) / 1e9
    endpoints = [(stamp.value - distance_re.index.asi8[0]) / 1e9 for stamp in (start, stop)]
    return np.interp(endpoints, elapsed, distance_re.to_numpy(dtype=float))


def interval_extent_re(distance_re: pd.Series, start, stop) -> float:
    """Subtract boundary coordinates from an already prepared distance mapping."""
    x_start, x_stop = _interval_coordinates_re(distance_re, start, stop)
    return float(x_stop - x_start)


def speed_at_time(speed: pd.Series, timestamp) -> float:
    """Interpolate the prepared speed to an exact, possibly fractional shock time."""
    timestamp = pd.Timestamp(timestamp)
    if not speed.index[0] <= timestamp <= speed.index[-1]:
        raise ValueError("shock time is outside the prepared speed record")
    elapsed = (speed.index.asi8 - speed.index.asi8[0]) / 1e9
    target = (timestamp.value - speed.index.asi8[0]) / 1e9
    value = float(np.interp(target, elapsed, speed.to_numpy(dtype=float)))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("speed at the shock must be positive and finite")
    return value


def sheath_spectral_scale(
    bt: pd.Series,
    spatial_record: tuple[np.ndarray, np.ndarray, np.ndarray],
    start,
    stop,
) -> dict:
    """Estimate the sheath spectrum from the already prepared spatial record.

    ``spatial_record`` is the output of ``prepare_spatial_record`` for ``bt``:
    coordinates at its one-second timestamps, uniform spatial grid, and B(x).
    Select the unfiltered sheath interval and use its full spatial record
    as one spectral segment.
    """
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    if not bt.index[0] <= start < stop <= bt.index[-1]:
        raise ValueError("sheath interval is unordered or outside the magnetic record")
    if not np.all(np.diff(bt.index.asi8) == 1_000_000_000):
        raise ValueError("sheath spectrum requires the prepared one-second magnetic grid")
    segment = bt.loc[(bt.index >= start) & (bt.index < stop)]
    if np.isfinite(segment).sum() < settings.PSD_MINIMUM_SAMPLES:
        raise ValueError("insufficient magnetic support for a sheath spatial spectrum")
    dx = settings.SPATIAL_DX_RE
    x_time, full_grid, spatial_field = spatial_record
    distance_re = pd.Series(x_time, index=bt.index)
    x_start, x_stop = _interval_coordinates_re(distance_re, start, stop)
    selected = (full_grid >= x_start) & (full_grid < x_stop)
    grid = full_grid[selected]
    field = spatial_field[selected].copy()
    if len(grid) < settings.PSD_MINIMUM_SAMPLES or not np.isfinite(field).all():
        raise ValueError("insufficient spatial samples for a sheath spectrum")
    local_grid = grid - grid[0]
    field -= np.polyval(np.polyfit(local_grid, field, 1), local_grid)
    nperseg = len(field)
    frequency, power = welch(
        field, fs=1.0 / dx, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, detrend="constant", scaling="density", average="mean",
    )
    sampled_span = float(grid[-1] - grid[0])
    eligible = (frequency > 1.0 / sampled_span) & np.isfinite(power)
    if not eligible.any() or np.max(power[eligible]) <= 0:
        raise ValueError("no positive, resolved sheath spectral peak")
    peak = float(frequency[eligible][np.argmax(power[eligible])])
    return {
        "wavelength_re": 1.0 / peak,
        "peak_frequency_cycles_per_re": peak,
        "spatial_grid_step_re": dx,
        "spatial_span_re": float(x_stop - x_start),
        "spectral_sampled_span_re": sampled_span,
        "input_second_count": len(segment),
        "missing_input_second_count": int((~np.isfinite(segment)).sum()),
        "magnetic_input": "unfiltered prepared one-second scalar magnitude",
        "spatial_preparation": "shared with waveform filtering; linear interpolation between finite magnetic samples",
        "spatial_interval_start_re": float(x_start),
        "spatial_interval_stop_re_exclusive": float(x_stop),
        "spatial_sample_count": len(grid),
        "welch_segment_samples": nperseg,
        "welch_overlap_samples": nperseg // 2,
        "welch_segment_count": 1,
        "welch_segment_policy": "full measured sheath; single-segment Hann spectrum",
    }


def derive_bands(references: dict[str, float], cadence_s: float) -> dict[str, dict]:
    ordered = np.asarray([
        references[name] for name in ("short", "substructure", "sheath", "ejecta")
    ], dtype=float)
    if (not np.isfinite(ordered).all() or np.any(ordered <= 0)
            or np.any(np.diff(ordered) <= 0)):
        raise ValueError("measured reference scales must be positive and strictly ordered")
    if not np.isfinite(cadence_s) or cadence_s <= 0:
        raise ValueError("native survey cadence must be positive and finite")
    lower = (settings.INFORMATION_FLOOR_MULTIPLIER * cadence_s
             * settings.INFORMATION_FLOOR_REFERENCE_SPEED_KMS / EARTH_RADIUS_KM)
    internal = np.sqrt(ordered[:-1] * ordered[1:])
    if lower >= internal[0]:
        raise ValueError("sampling floor leaves no resolved fine-scale band")
    edges = [lower, *internal, None]
    bands = {}
    for i, family in enumerate(FAMILIES):
        bands[family] = {
            "slug": family,
            "filter_kind": "lowpass" if family == "trend" else "bandpass",
            "min_re": float(edges[i]),
            "max_re": None if edges[i + 1] is None else float(edges[i + 1]),
            "lower_edge_basis": (
                "configured sampling multiplier times current-run worst ARTEMIS "
                "median survey cadence times the configured reference speed"
                if i == 0 else "geometric mean of adjacent current-run reference scales"
            ),
            "upper_edge_basis": (
                "mean-retaining low-pass" if family == "trend"
                else "geometric mean of adjacent current-run reference scales"
            ),
        }
    return bands


def load_band_design(path: Path | None = None, *, verify_inputs: bool = True) -> dict:
    """Reject missing or stale products; never substitute historical results."""
    path = Path(path) if path is not None else PROCESSED_DIR / "waveform_bands.json"
    if not path.is_file():
        raise FileNotFoundError(f"Recalculate waveform scales before filtering: {path}")
    design = read_json(path)
    if design.get("schema_version") != "waveform-band-design-v2":
        raise ValueError(f"unrecognized waveform scale product: {path}")
    expected = derive_bands(design["reference_scales_re"], design["worst_artemis_cadence_s"])
    numerical_keys = ("slug", "filter_kind", "min_re", "max_re")
    observed = design["bands"]
    if (set(observed) != set(expected) or any(
        {key: observed[family].get(key) for key in numerical_keys}
        != {key: band[key] for key in numerical_keys}
        for family, band in expected.items()
    )):
        raise ValueError("saved edges disagree with their measured reference scales")
    if design.get("method_parameters") != method_parameters():
        raise ValueError("waveform method settings changed; recalculate the scale product")
    if verify_inputs:
        root = path.parent.parent
        if not design.get("inputs"):
            raise ValueError("waveform scale product lacks input provenance")
        for record in design["inputs"]:
            source = root / record["path"]
            if not source.is_file() or sha256(source) != record["sha256"]:
                raise ValueError(f"waveform scales are stale; recompute after changing {source}")
    return design


def method_parameters() -> dict:
    """Machine settings only; descriptive provenance is not a schema contract."""
    return {
        "algorithm_revision": 1,
        "information_floor_reference_speed_kms": settings.INFORMATION_FLOOR_REFERENCE_SPEED_KMS,
        "information_floor_multiplier": settings.INFORMATION_FLOOR_MULTIPLIER,
        "psd_minimum_samples": settings.PSD_MINIMUM_SAMPLES,
        "psd_spatial_grid_step_re": settings.SPATIAL_DX_RE,
        "psd_window": "hann",
    }
