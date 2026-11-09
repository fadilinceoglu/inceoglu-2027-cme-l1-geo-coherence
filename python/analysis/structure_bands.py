"""Spatial filtering utilities for discrete solar-wind structures.

The functions in this module work in convected-distance coordinates rather
than directly in time.  That keeps a requested physical-size band fixed when
the solar-wind speed changes across the shock.  Filters are applied with
``sosfiltfilt`` so they do not introduce a phase delay into later timing work.
Every filtered product comes with boolean validity masks that excise filter
edge transients and bridged data gaps (see ``filter_prepared_spatial_record``).

The bands describe waveform envelope scales, rather than a turbulence-spectrum
decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt


from config.study import EARTH_RADIUS_KM


@dataclass(frozen=True)
class SpatialBand:
    """One physical-size band, expressed in Earth radii."""

    name: str
    minimum_re: float
    maximum_re: float | None = None

    def validate(self) -> None:
        if not np.isfinite(self.minimum_re) or self.minimum_re <= 0:
            raise ValueError("minimum_re must be positive and finite")
        if self.maximum_re is not None:
            if not np.isfinite(self.maximum_re) or self.maximum_re <= self.minimum_re:
                raise ValueError("maximum_re must exceed minimum_re")


def seconds_from_times(times: np.ndarray) -> np.ndarray:
    """Convert a monotonic datetime64 array to elapsed seconds."""

    t = np.asarray(times).astype("datetime64[ns]")
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("times must be a one-dimensional array with at least 2 values")
    elapsed = (t.astype("int64") - t.astype("int64")[0]) / 1e9
    if np.any(np.diff(elapsed) <= 0):
        raise ValueError("times must be strictly increasing")
    return elapsed.astype(float)


def convected_distance_re(
    times: np.ndarray,
    speed_km_s: np.ndarray,
    *,
    max_speed_gap_s: float = 1200.0,
) -> np.ndarray:
    """Integrate solar-wind speed to a convected-distance coordinate in R_E.

    A trapezoidal integral is used.  Short missing speed samples are linearly
    interpolated **in elapsed time** (not sample index), so a non-uniform time
    axis is handled correctly; leading/trailing missing values use the nearest
    finite value.  Material gaps are rejected so an analysis must split the
    interval or state an explicit speed model instead of silently bridging
    hours of missing data.
    """

    elapsed = seconds_from_times(times)
    speed = np.asarray(speed_km_s, dtype=float)
    if speed.shape != elapsed.shape:
        raise ValueError("speed_km_s must have the same shape as times")
    good = np.isfinite(speed) & (speed > 0)
    if good.sum() < 2:
        raise ValueError("at least two positive finite speed samples are required")
    bad = ~good
    if np.any(bad):
        edges = np.flatnonzero(np.diff(np.r_[False, bad, False]))
        for start, stop in edges.reshape(-1, 2):
            left = max(0, start - 1)
            right = min(len(elapsed) - 1, stop)
            gap_s = elapsed[right] - elapsed[left]
            if gap_s > max_speed_gap_s:
                raise ValueError(
                    f"speed gap of {gap_s:.1f} s exceeds max_speed_gap_s="
                    f"{max_speed_gap_s:.1f}; split the interval or provide a speed model"
                )
    speed_filled = np.interp(elapsed, elapsed[good], speed[good])
    dt = np.diff(elapsed)
    increments_km = 0.5 * (speed_filled[:-1] + speed_filled[1:]) * dt
    return np.concatenate(([0.0], np.cumsum(increments_km / EARTH_RADIUS_KM)))


def prepare_spatial_record(
    times: np.ndarray,
    values: np.ndarray,
    speed_km_s: np.ndarray,
    *,
    dx_re: float,
    max_speed_gap_s: float = 1200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map the unfiltered record to the shared uniform distance grid.

    Both reference-scale estimation and waveform filtering use this exact
    coordinate and magnetic interpolation. Missing values are bridged only
    in this numerical spatial representation; the input is not modified.
    """
    values = np.asarray(values, dtype=float)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != len(times):
        raise ValueError("values must have shape (N,) or (N, C) matching times")
    if not np.isfinite(dx_re) or dx_re <= 0:
        raise ValueError("dx_re must be positive and finite")
    x_time = convected_distance_re(times, speed_km_s, max_speed_gap_s=max_speed_gap_s)
    x_grid = np.arange(x_time[0], x_time[-1] + 0.5 * dx_re, dx_re)
    spatial_values = np.full((len(x_grid), values.shape[1]), np.nan, dtype=float)
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        if finite.sum() >= 20:
            spatial_values[:, column] = np.interp(
                x_grid, x_time[finite], values[finite, column],
            )
    return x_time, x_grid, spatial_values[:, 0] if squeeze else spatial_values


SETTLE_FACTOR = 1.5
"""Edge-transient settle length, in units of the band's longest passband
wavelength.  Applied from each end of the spatial record when building the
validity masks (the filter sees a hard edge there)."""

GAP_SETTLE_FACTOR = 0.5
"""Guard length around masked (over-tolerance) bridged gaps, in the same
units.  Milder than the record-end factor because a bridged interior gap is
a local perturbation rather than a hard record edge."""


def settle_distance_re(band: SpatialBand) -> float:
    """Settle distance in R_E for one band's filter edge transients.

    ``SETTLE_FACTOR`` times the longest passband wavelength: ``maximum_re``
    for a band-pass, ``minimum_re`` (the cutoff wavelength) for a low-pass.
    """

    band.validate()
    longest = band.minimum_re if band.maximum_re is None else band.maximum_re
    return SETTLE_FACTOR * longest


def _spatial_sos(
    dx_re: float,
    band: SpatialBand,
    order: int,
) -> np.ndarray:
    band.validate()
    if not np.isfinite(dx_re) or dx_re <= 0:
        raise ValueError("dx_re must be positive and finite")
    if order < 2:
        raise ValueError("order must be at least 2")

    nyquist = 0.5 / dx_re  # cycles per R_E
    if band.maximum_re is None:
        cutoff = (1.0 / band.minimum_re) / nyquist
        if not 0 < cutoff < 1:
            raise ValueError("low-pass cutoff is outside the sampled spatial band")
        return butter(order, cutoff, btype="lowpass", output="sos")

    low = (1.0 / band.maximum_re) / nyquist
    high = (1.0 / band.minimum_re) / nyquist
    if not 0 < low < high < 1:
        raise ValueError("band-pass cutoffs are outside the sampled spatial band")
    return butter(order, [low, high], btype="bandpass", output="sos")


def filter_prepared_spatial_record(
    values: np.ndarray,
    band: SpatialBand,
    spatial_record: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    dx_re: float,
    order: int = 4,
    max_gap_re: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Zero-phase filter one or more signals in convected-distance space.

    Parameters
    ----------
    spatial_record
        The shared output of ``prepare_spatial_record``.
    values
        Shape ``(N,)`` or ``(N, C)``.  Internal NaNs are interpolated for the
        filtering operation only; the returned time-grid result restores NaNs
        at the original missing samples.
    band
        A low-pass if ``maximum_re`` is ``None``; otherwise a size band-pass.
    dx_re
        Uniform spatial sampling step in Earth radii.
    max_gap_re
        Longest input-NaN run, measured in convected distance between the
        bounding finite samples, that may be silently bridged by the linear
        interpolation.  Longer gaps are still interpolated so the filter
        stays numerically stable, but the gap plus a settle margin on either
        side is marked invalid in the returned masks.

    Returns
    -------
    filtered_time, x_grid, filtered_x, valid_time, valid_x
        ``filtered_time`` is on the original time grid (NaN restored at the
        original missing samples).  ``x_grid`` is the uniform spatial grid in
        R_E and ``filtered_x`` contains its filtered samples.  ``valid_time``
        and ``valid_x`` are boolean masks matching ``filtered_time`` and
        ``filtered_x`` in shape; a sample is ``True`` only where the filtered
        estimate is trustworthy.  Masked (``False``) are:

        * edge transients — the first and last ``settle_distance_re(band)``
          (= ``SETTLE_FACTOR`` x the band's longest passband wavelength:
          ``maximum_re`` for a band-pass, ``minimum_re`` for a low-pass) of
          convected distance at each end of the record;
        * bridged gaps — every input-NaN run longer than ``max_gap_re``,
          extended on both sides by ``GAP_SETTLE_FACTOR`` (0.5) times the
          band's longest passband wavelength;
        * on the time grid, samples whose input value was not finite;
        * entire columns that had too few finite samples to filter.
    """
    values = np.asarray(values, dtype=float)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    x_time, x_grid, spatial_values = spatial_record
    spatial_values = np.asarray(spatial_values, dtype=float)
    if spatial_values.ndim == 1:
        spatial_values = spatial_values[:, None]
    if (values.ndim != 2 or len(values) != len(x_time)
            or spatial_values.shape != (len(x_grid), values.shape[1])):
        raise ValueError("prepared spatial record does not match the magnetic array")
    if not np.isfinite(max_gap_re) or max_gap_re <= 0:
        raise ValueError("max_gap_re must be positive and finite")

    sos = _spatial_sos(dx_re, band, order)
    filtered_x = np.full((len(x_grid), values.shape[1]), np.nan, dtype=float)
    filtered_t = np.full(values.shape, np.nan, dtype=float)
    valid_x = np.zeros((len(x_grid), values.shape[1]), dtype=bool)
    valid_t = np.zeros(values.shape, dtype=bool)

    settle_re = settle_distance_re(band)
    edge_valid_x = (x_grid - x_time[0] >= settle_re) & (x_time[-1] - x_grid >= settle_re)
    edge_valid_t = (x_time - x_time[0] >= settle_re) & (x_time[-1] - x_time >= settle_re)

    for column in range(values.shape[1]):
        y = values[:, column]
        finite = np.isfinite(y)
        if finite.sum() < 20:
            continue
        y_x = spatial_values[:, column]
        try:
            y_fx = sosfiltfilt(sos, y_x)
        except ValueError as exc:
            raise ValueError(
                f"spatial record is too short for band {band.name!r}; "
                "use a longer interval or a smaller filter order"
            ) from exc
        filtered_x[:, column] = y_fx
        filtered_t[:, column] = np.interp(x_time, x_grid, y_fx)
        filtered_t[~finite, column] = np.nan

        col_valid_x = edge_valid_x.copy()
        col_valid_t = edge_valid_t & finite
        gap_settle_re = settle_re * GAP_SETTLE_FACTOR / SETTLE_FACTOR
        for x_lo, x_hi in _long_gap_spans(x_time, finite, max_gap_re):
            lo = x_lo - gap_settle_re
            hi = x_hi + gap_settle_re
            col_valid_x &= (x_grid < lo) | (x_grid > hi)
            col_valid_t &= (x_time < lo) | (x_time > hi)
        valid_x[:, column] = col_valid_x
        valid_t[:, column] = col_valid_t

    if squeeze:
        return filtered_t[:, 0], x_grid, filtered_x[:, 0], valid_t[:, 0], valid_x[:, 0]
    return filtered_t, x_grid, filtered_x, valid_t, valid_x


def _long_gap_spans(
    x_time: np.ndarray,
    finite: np.ndarray,
    max_gap_re: float,
) -> list[tuple[float, float]]:
    """Convected-distance spans of input-NaN runs longer than ``max_gap_re``.

    Each span runs from the last finite sample before the run to the first
    finite sample after it (record edges when the run touches an end).
    """

    spans: list[tuple[float, float]] = []
    bad = ~finite
    if not np.any(bad):
        return spans
    edges = np.flatnonzero(np.diff(np.r_[False, bad, False]))
    for start, stop in edges.reshape(-1, 2):
        x_lo = x_time[start - 1] if start > 0 else x_time[0]
        x_hi = x_time[stop] if stop < len(x_time) else x_time[-1]
        if x_hi - x_lo > max_gap_re:
            spans.append((float(x_lo), float(x_hi)))
    return spans
