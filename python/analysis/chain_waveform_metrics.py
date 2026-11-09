"""Frame-safe metrics for the seven-spacecraft isolated-waveform chain.

This module is a read/registration/metric layer.  It never constructs a
filter and never changes a saved waveform.  The magnetic inputs are exactly
the saved Figure-3 columns

``bx_f, by_f, bz_f, bt_f, valid``

for Solar Orbiter, ACE, Wind, DSCOVR, Solar-1, and ARTEMIS P1/P2.  ARTEMIS
P2 is retained as the lunar-wake endpoint of the observational sequence.

Registration
------------
All four families use the spacecraft-specific, half-open intervals from the
current run's band-energy interval product.  Selection is
performed exactly in UTC on the saved one-minute rows.  Those rows use minute samples of the same one-second
frozen-speed distance mapping as the spectra and waveform filters.  The local native shock-ramp 50-percent clock remains
the zero of that coordinate; it is not an interval boundary or detector input.
No L1 distance union, physical-boundary crop, lag optimization, stretch, or
waveform warp is used.

Amplitude and support descriptors
---------------------------------
For the independently filtered scalar magnitude residual ``q = bt_f`` and
vector residual ``b_f = (bx_f, by_f, bz_f)``, all integrals use the common
all-component valid mask and convected distance ``x``. The selected-support
RMS values are

.. math::

    q_\\mathrm{rms} =
      \\left(\\frac{\\int_S q^2\\,dx}{\\int_S dx}\\right)^{1/2},

.. math::

    b_{v,\\mathrm{rms}} =
      \\left(\\frac{\\int_S |\\mathbf b_f|^2\\,dx}{\\int_S dx}\\right)^{1/2}.

Their identical-support centered variance ratio is

.. math::

    C_{\\mathrm{var}} =
      \\frac{\\int_S(q-\\bar q_x)^2\\,dx}
      {\\sum_j\\int_S(b_{f,j}-\\bar b_{f,j,x})^2\\,dx},

where each bar is that channel's spatially weighted mean over the complete
selected support. The ratio does not include the interval means and is not
bounded above by one: scalar magnitude and vector components were filtered
independently. RMS values retain their original uncentered definitions.

For a non-negative power density ``P``, the central 5--95 per cent power
support is

.. math::

    F(x_p)=p, \\qquad W_{90}=x_{95}-x_{05}.

These are descriptors inside each selected support. They never crop the RMS
calculation. ``bt_f`` is
independently filtered scalar ``|B|`` and is not the magnitude of the filtered
vector.

Solar Orbiter vectors remain RTN and near-Earth vectors use GSM. Native GSM
products are used except for DSCOVR, whose definitive GSE vectors are rotated
at their original epochs using SpacePy CTrans and the supplied IGRF14 model.
Scalar power, vector power, and width are invariant under a fixed
orthogonal frame rotation. Component signs and transverse azimuth are not
cross-frame quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.paths import PROCESSED_DIR
from config import study
from config.waveform_bands import WAVEFORM_FAMILIES
from analysis.product_io import boolean_values
from common.io import read_json as _read_json, utc_index, utc_timestamp as _utc
from analysis.spatial_coordinates import read_coordinates
from config.study import EARTH_RADIUS_KM


CRAFT_ORDER = study.UPSTREAM_SPACECRAFT
STRUCTURES = WAVEFORM_FAMILIES
FULL_WAVEFORM_CADENCE = pd.Timedelta(minutes=1)

DEFAULT_MAX_MATERIAL_GAP_RE = 25.0
BOUNDARIES_FILENAME = "boundaries.json"
TREND_MEAN_WARNING = (
    "trend is a low-pass at the current-run cutoff that retains the slowly varying "
    "mean magnetic field; RMS and width retain that mean, while the "
    "variance ratio subtracts each channel's weighted interval mean. These "
    "descriptors are not isolated-ejecta fluctuation metrics"
)


@dataclass(frozen=True)
class SignedSupport:
    """One fixed signed convected-distance support."""

    start_re: float
    stop_re: float

    @property
    def span_re(self) -> float:
        return float(self.stop_re - self.start_re)

    def validate(self) -> None:
        if (
            not np.isfinite(self.start_re)
            or not np.isfinite(self.stop_re)
            or self.stop_re <= self.start_re
        ):
            raise ValueError("signed support must have finite increasing edges")


@dataclass(frozen=True)
class WaveformRecord:
    """One exact saved component-plus-magnitude filtered product."""

    craft: str
    structure: str
    frame: str
    source_slug: str
    source_path: Path
    time: pd.DatetimeIndex
    vector_nt: np.ndarray
    scalar_bt_nt: np.ndarray
    source_interpolated: np.ndarray
    valid: np.ndarray
    stored_cadence_s: float
    effective_cadence_s: float


@dataclass(frozen=True)
class RegisteredWaveform:
    """One saved waveform mapped onto a fixed signed-distance window.

    ``window_stop_utc`` is the last included stored sample.  Figure-4 outputs
    serialize that value explicitly as ``registered_window_last_included_utc``
    and separately serialize the ledger's half-open stop.
    """

    record: WaveformRecord
    anchor_utc: pd.Timestamp
    support: SignedSupport
    window_start_utc: pd.Timestamp
    window_stop_utc: pd.Timestamp
    time: pd.DatetimeIndex
    x_re: np.ndarray
    vector_nt: np.ndarray
    scalar_bt_nt: np.ndarray
    source_interpolated: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class PowerSupport:
    """Central 5--95% support and integrated power of one power density."""

    x05_re: float
    x95_re: float
    width90_re: float
    integrated_power_nt2_re: float
    crosses_material_gap: bool


@dataclass(frozen=True)
class WaveformMetrics:
    """Full-window amplitude metrics and non-cropping support descriptors."""

    scalar_bt_rms_nt: float
    vector_rms_nt: float
    source_interpolated_row_count: int
    scalar_to_vector_variance_ratio: float
    scalar_power_support: PowerSupport
    declared_span_re: float
    valid_spatial_length_re: float
    spatial_coverage_fraction: float
    stored_valid_row_count: int
    effective_cadence_s: float
    interpretation_warning: str | None


def read_frozen_speed(processed: Path, craft: str) -> pd.Series:
    path = processed / f"{craft}_vt_frozen.csv"
    frame = pd.read_csv(path, parse_dates=["time"], index_col="time")
    if "v_frozen" not in frame:
        raise ValueError(f"{path.name} lacks v_frozen")
    frame.index = pd.DatetimeIndex(frame.index)
    frame.index = utc_index(frame.index)
    speed = frame["v_frozen"].astype(float).sort_index()
    finite = np.isfinite(speed.to_numpy()) & (speed.to_numpy() > 0.0)
    if finite.sum() < 2:
        raise ValueError(f"{path.name} has fewer than two valid speeds")
    return speed


def _stored_cadence_seconds(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        raise ValueError("waveform has fewer than two timestamps")
    differences = np.diff(index.asi8) / 1.0e9
    positive = differences[np.isfinite(differences) & (differences > 0.0)]
    if not len(positive):
        raise ValueError("waveform timestamps are not increasing")
    return float(np.median(positive))


def _magnetic_frame(craft: str) -> str:
    frame = str(study.SATELLITES[craft]["frame"]).upper()
    if frame not in {"RTN", "GSE", "GSM"}:
        raise ValueError(f"unsupported coordinate frame for {craft}: {frame}")
    return frame


def load_frame_anchors(boundaries_path: Path | str) -> dict[str, dict[str, Any]]:
    """Read exact current-run shock clocks and registry frames without a copy product."""
    boundaries = _read_json(Path(boundaries_path))
    anchors = {}
    for craft in CRAFT_ORDER:
        record = boundaries[craft]
        if not record.get("shock_t50_utc"):
            raise ValueError(f"{craft} lacks a current-run native shock_t50_utc")
        clock = pd.Timestamp(record["shock_t50_utc"])
        if clock.tzinfo is None:
            raise ValueError(f"{craft} native shock clock must be explicit UTC")
        clock = clock.tz_convert("UTC")
        if not study.ANALYSIS_START <= clock < study.ANALYSIS_STOP:
            raise ValueError(f"{craft} shock clock lies outside the event record")
        frame = _magnetic_frame(craft)
        anchors[craft] = {"frame": frame, "shock_t50_utc": clock}
    return anchors


def _anchor_times(processed: Path) -> dict[str, dict[str, Any]]:
    return load_frame_anchors(processed.parent / "results" / BOUNDARIES_FILENAME)


def load_chain_waveform(
    craft: str,
    structure: str,
    *,
    processed_directory: Path | str | None = None,
) -> WaveformRecord:
    """Load one exact saved seven-craft Figure-3 waveform.

    The returned validity mask is the saved mask intersected with finite
    ``bx_f/by_f/bz_f/bt_f``.  Every observer is retained, including ACE and
    the ARTEMIS-P2 lunar-wake endpoint.
    """

    if craft not in CRAFT_ORDER:
        raise ValueError(f"unsupported spacecraft: {craft}")
    if structure not in STRUCTURES:
        raise ValueError(f"unsupported structure: {structure}")
    processed = (
        Path(processed_directory)
        if processed_directory is not None
        else PROCESSED_DIR
    )
    slug = structure
    path = processed / "bands" / f"{craft}__{slug}.csv"
    frame = pd.read_csv(path, parse_dates=["time"], index_col="time")
    required = {"bx_f", "by_f", "bz_f", "bt_f", "valid", "source_interpolated"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {', '.join(sorted(missing))}"
        )
    frame.index = pd.DatetimeIndex(frame.index)
    frame.index = utc_index(frame.index)
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(f"{path.name} must have unique increasing UTC rows")

    frame_name = _magnetic_frame(craft)

    vector = frame[["bx_f", "by_f", "bz_f"]].to_numpy(dtype=float)
    scalar = frame["bt_f"].to_numpy(dtype=float)
    valid = (
        boolean_values(frame["valid"])
        & np.isfinite(vector).all(axis=1)
        & np.isfinite(scalar)
    )
    source_interpolated = boolean_values(frame["source_interpolated"], name=f"{path.name}/source_interpolated")
    stored_cadence = _stored_cadence_seconds(frame.index)
    if not np.isclose(stored_cadence, 60.0, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            f"{path.name} must be stored at the common one-minute cadence"
        )
    return WaveformRecord(
        craft=craft,
        structure=structure,
        frame=frame_name,
        source_slug=slug,
        source_path=path,
        time=frame.index,
        vector_nt=vector,
        scalar_bt_nt=scalar,
        source_interpolated=source_interpolated,
        valid=valid,
        stored_cadence_s=stored_cadence,
        effective_cadence_s=stored_cadence,
    )


def _finite_speed_arrays(speed: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    times = pd.DatetimeIndex(speed.index).asi8.astype(np.int64)
    values = speed.to_numpy(dtype=float)
    finite = np.isfinite(values) & (values > 0.0)
    times = times[finite]
    values = values[finite]
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("frozen speed must have increasing finite support")
    return times, values


def _coordinate_on_grid(
    speed: pd.Series,
    grid_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_ns, source_speed = _finite_speed_arrays(speed)
    grid = np.unique(np.asarray(grid_ns, dtype=np.int64))
    if grid[0] < source_ns[0] or grid[-1] > source_ns[-1]:
        raise ValueError("requested coordinate lies outside frozen-speed support")
    velocity = np.interp(
        grid.astype(float),
        source_ns.astype(float),
        source_speed,
    )
    dt_s = np.diff(grid.astype(float)) / 1.0e9
    increments = (
        0.5 * (velocity[:-1] + velocity[1:]) * dt_s / EARTH_RADIUS_KM
    )
    coordinate = np.r_[0.0, np.cumsum(increments)]
    return grid, coordinate


def signed_distances_re(
    speed: pd.Series,
    anchor: pd.Timestamp,
    times: pd.DatetimeIndex,
) -> np.ndarray:
    """Map ordered UTC samples to signed distance about ``anchor``."""

    target = pd.DatetimeIndex(times)
    target = utc_index(target)
    if not target.is_monotonic_increasing:
        raise ValueError("target times must be increasing")
    source_ns, _ = _finite_speed_arrays(speed)
    anchor_ns = np.int64(_utc(anchor).value)
    combined = np.unique(np.r_[source_ns, anchor_ns, target.asi8])
    grid, coordinate = _coordinate_on_grid(speed, combined)
    anchor_x = float(
        np.interp(float(anchor_ns), grid.astype(float), coordinate)
    )
    target_x = np.interp(
        target.asi8.astype(float),
        grid.astype(float),
        coordinate,
    )
    return target_x - anchor_x


def utc_support_time_grid(
    start_utc: str | pd.Timestamp,
    stop_utc_exclusive: str | pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return one exact, minute-aligned, half-open UTC support grid."""

    start = _utc(start_utc)
    stop = _utc(stop_utc_exclusive)
    if stop <= start:
        raise ValueError("UTC support stop must follow its start")
    minute_ns = int(FULL_WAVEFORM_CADENCE.value)
    if start.value % minute_ns or stop.value % minute_ns:
        raise ValueError("UTC support limits must be aligned to whole minutes")
    grid = pd.date_range(
        start,
        stop,
        freq=FULL_WAVEFORM_CADENCE,
        inclusive="left",
    )
    if len(grid) < 2:
        raise ValueError("UTC support must contain at least two minute rows")
    return grid


def register_waveform_utc(
    record: WaveformRecord,
    *,
    start_utc: str | pd.Timestamp,
    stop_utc_exclusive: str | pd.Timestamp,
    processed_directory: Path | str | None = None,
    anchor_utc: pd.Timestamp | None = None,
    coordinate: pd.Series | None = None,
) -> RegisteredWaveform:
    """Select an exact half-open UTC support and then map it to distance."""

    processed = (
        Path(processed_directory)
        if processed_directory is not None
        else PROCESSED_DIR
    )
    expected = utc_support_time_grid(start_utc, stop_utc_exclusive)
    indexer = record.time.get_indexer(expected)
    if np.any(indexer < 0):
        missing = expected[indexer < 0]
        raise ValueError(
            f"{record.craft}/{record.structure} is missing "
            f"{len(missing)} required support rows"
        )
    time = record.time[indexer]
    if not time.equals(expected):
        raise ValueError(
            f"{record.craft}/{record.structure} support clock is not exact"
        )
    anchor = (_anchor_times(processed)[record.craft]["shock_t50_utc"]
              if anchor_utc is None else _utc(anchor_utc))
    if coordinate is None:
        coordinate = read_coordinates(processed, (record.craft,))[record.craft]
    if anchor not in coordinate.index or coordinate.loc[anchor] != 0.0:
        raise ValueError("shared coordinate does not use the current measured shock")
    x = coordinate.reindex(time).to_numpy(dtype=float)
    if not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("shared coordinate is missing or invalid on waveform support")
    support = SignedSupport(float(x[0]), float(x[-1]))
    support.validate()
    return RegisteredWaveform(
        record=record,
        anchor_utc=anchor,
        support=support,
        window_start_utc=time[0],
        window_stop_utc=time[-1],
        time=time,
        x_re=x,
        vector_nt=record.vector_nt[indexer],
        scalar_bt_nt=record.scalar_bt_nt[indexer],
        source_interpolated=record.source_interpolated[indexer],
        valid=record.valid[indexer],
    )


def _connected_pairs(
    x_re: np.ndarray,
    valid: np.ndarray,
    *,
    max_material_gap_re: float,
) -> np.ndarray:
    x = np.asarray(x_re, dtype=float)
    good = np.asarray(valid, dtype=bool)
    if x.ndim != 1 or good.shape != x.shape or len(x) < 2:
        raise ValueError("x and valid must be equal one-dimensional arrays")
    dx = np.diff(x)
    if np.any(~np.isfinite(x)) or np.any(dx <= 0.0):
        raise ValueError("x must be finite and strictly increasing")
    if not np.isfinite(max_material_gap_re) or max_material_gap_re <= 0.0:
        raise ValueError("max_material_gap_re must be positive")
    return (
        good[:-1]
        & good[1:]
        & np.isfinite(dx)
        & (dx <= max_material_gap_re)
    )


def _trapezoid_integral(
    x_re: np.ndarray,
    values: np.ndarray,
    connected: np.ndarray,
) -> float:
    x = np.asarray(x_re, dtype=float)
    y = np.asarray(values, dtype=float)
    pair = np.asarray(connected, dtype=bool)
    finite_pair = pair & np.isfinite(y[:-1]) & np.isfinite(y[1:])
    dx = np.diff(x)[finite_pair]
    return float(np.sum(0.5 * (y[:-1][finite_pair] + y[1:][finite_pair]) * dx))


def _power_quantile_in_segment(
    x0: float,
    x1: float,
    p0: float,
    p1: float,
    target_energy: float,
) -> float:
    """Invert the integral of linearly interpolated non-negative power."""

    dx = float(x1 - x0)
    slope = float((p1 - p0) / dx)
    if abs(slope) <= 1.0e-14 * max(1.0, abs(p0), abs(p1)):
        if p0 <= 0.0:
            return x0
        z = target_energy / p0
    else:
        discriminant = p0 * p0 + 2.0 * slope * target_energy
        discriminant = max(0.0, discriminant)
        roots = [
            (-p0 + np.sqrt(discriminant)) / slope,
            (-p0 - np.sqrt(discriminant)) / slope,
        ]
        admissible = [root for root in roots if -1.0e-9 <= root <= dx + 1.0e-9]
        if not admissible:
            raise ValueError("power-quantile inversion left its segment")
        z = admissible[0]
    return float(x0 + np.clip(z, 0.0, dx))


def power_support_descriptor(
    x_re: np.ndarray,
    power_nt2: np.ndarray,
    valid: np.ndarray,
    *,
    max_material_gap_re: float = DEFAULT_MAX_MATERIAL_GAP_RE,
    bridge_invalid_rows: bool = True,
) -> PowerSupport:
    """Return central 5--95% power support without cropping anything."""

    x = np.asarray(x_re, dtype=float)
    power = np.asarray(power_nt2, dtype=float)
    good = np.asarray(valid, dtype=bool) & np.isfinite(power) & (power >= 0.0)
    if power.shape != x.shape:
        raise ValueError("power must match x")
    if int(good.sum()) < 2:
        raise ValueError("power has fewer than two valid spatial samples")
    if bridge_invalid_rows:
        # Short holes may be linearly spanned by their valid bounding samples.
        # A separation wider than max_material_gap_re remains disconnected.
        x = x[good]
        power = power[good]
        connected = _connected_pairs(
            x,
            np.ones(len(x), dtype=bool),
            max_material_gap_re=max_material_gap_re,
        )
    else:
        # Preserve the stored row topology: an invalid row disconnects the
        # adjacent segments even when its spatial gap is short.
        connected = _connected_pairs(
            x,
            good,
            max_material_gap_re=max_material_gap_re,
        )
    if not connected.any():
        raise ValueError("power has no connected valid spatial interval")
    dx = np.diff(x)
    pair_indices = np.flatnonzero(connected)
    segment_energy = (
        0.5
        * (power[pair_indices] + power[pair_indices + 1])
        * dx[pair_indices]
    )
    total = float(segment_energy.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("integrated power must be positive")

    def quantile(fraction: float) -> float:
        target = fraction * total
        cumulative = 0.0
        for index, energy in zip(pair_indices, segment_energy):
            if cumulative + energy >= target - 1.0e-14 * total:
                return _power_quantile_in_segment(
                    x[index],
                    x[index + 1],
                    power[index],
                    power[index + 1],
                    target - cumulative,
                )
            cumulative += float(energy)
        return float(x[pair_indices[-1] + 1])

    x05 = quantile(0.05)
    x95 = quantile(0.95)
    crossed_gap = bool(
        np.any(
            (~connected)
            & (x[:-1] < x95)
            & (x[1:] > x05)
        )
    )
    return PowerSupport(
        x05_re=x05,
        x95_re=x95,
        width90_re=float(x95 - x05),
        integrated_power_nt2_re=total,
        crosses_material_gap=crossed_gap,
    )


def compute_waveform_metrics(
    waveform: RegisteredWaveform,
    *,
    max_material_gap_re: float = DEFAULT_MAX_MATERIAL_GAP_RE,
    bridge_invalid_rows: bool = True,
) -> WaveformMetrics:
    """Compute agreed full-window amplitude and support descriptors."""

    x = np.asarray(waveform.x_re, dtype=float)
    vector = np.asarray(waveform.vector_nt, dtype=float)
    scalar = np.asarray(waveform.scalar_bt_nt, dtype=float)
    valid = (
        np.asarray(waveform.valid, dtype=bool)
        & np.isfinite(vector).all(axis=1)
        & np.isfinite(scalar)
    )
    if (
        vector.shape != (len(x), 3)
        or scalar.shape != x.shape
    ):
        raise ValueError("registered waveform arrays have inconsistent shapes")
    valid_count = int(valid.sum())
    if bridge_invalid_rows:
        x_valid = x[valid]
        vector_valid = vector[valid]
        scalar_valid = scalar[valid]
        topology_valid = np.ones(len(x_valid), dtype=bool)
    else:
        x_valid = x
        vector_valid = vector
        scalar_valid = scalar
        topology_valid = valid
    connected = _connected_pairs(
        x_valid,
        topology_valid,
        max_material_gap_re=max_material_gap_re,
    )
    valid_length = float(np.diff(x_valid)[connected].sum())
    if valid_length <= 0.0:
        raise ValueError("waveform has no valid connected spatial support")
    scalar_power = scalar_valid * scalar_valid
    vector_power = np.sum(vector_valid * vector_valid, axis=1)
    scalar_integral = _trapezoid_integral(
        x_valid, scalar_power, connected
    )
    vector_integral = _trapezoid_integral(
        x_valid, vector_power, connected
    )
    if vector_integral <= 0.0:
        raise ValueError("filtered vector power must be positive")
    scalar_rms = float(np.sqrt(max(0.0, scalar_integral / valid_length)))
    vector_rms = float(np.sqrt(vector_integral / valid_length))
    # Translate before calculating the weighted means to avoid cancellation
    # around large offsets. A constant decimal-valued channel then has exactly
    # zero centered differences, without a physical tolerance or fallback.
    first_connected = int(np.flatnonzero(connected)[0])
    scalar_delta = scalar_valid - scalar_valid[first_connected]
    vector_delta = vector_valid - vector_valid[first_connected]
    scalar_mean_delta = _trapezoid_integral(
        x_valid, scalar_delta, connected
    ) / valid_length
    vector_mean_delta = np.array([
        _trapezoid_integral(x_valid, vector_delta[:, j], connected) / valid_length
        for j in range(3)
    ])
    scalar_variance = _trapezoid_integral(
        x_valid, (scalar_delta - scalar_mean_delta) ** 2, connected
    ) / valid_length
    vector_variance = _trapezoid_integral(
        x_valid, np.sum((vector_delta - vector_mean_delta) ** 2, axis=1), connected
    ) / valid_length
    if not np.isfinite(vector_variance) or vector_variance <= 0.0:
        raise ValueError("filtered vector variance must be positive and finite")
    if not np.isfinite(scalar_variance) or scalar_variance < 0.0:
        raise ValueError("filtered scalar variance must be nonnegative and finite")
    ratio = float(scalar_variance / vector_variance)

    scalar_support = power_support_descriptor(
        x,
        scalar * scalar,
        valid,
        max_material_gap_re=max_material_gap_re,
        bridge_invalid_rows=bridge_invalid_rows,
    )
    declared_span = waveform.support.span_re
    coverage = float(valid_length / declared_span)
    warning = (
        TREND_MEAN_WARNING if waveform.record.structure == "trend" else None
    )
    return WaveformMetrics(
        scalar_bt_rms_nt=scalar_rms,
        vector_rms_nt=vector_rms,
        source_interpolated_row_count=int(
            np.asarray(waveform.source_interpolated, dtype=bool).sum()
        ),
        scalar_to_vector_variance_ratio=ratio,
        scalar_power_support=scalar_support,
        declared_span_re=declared_span,
        valid_spatial_length_re=valid_length,
        spatial_coverage_fraction=coverage,
        stored_valid_row_count=valid_count,
        effective_cadence_s=waveform.record.effective_cadence_s,
        interpretation_warning=warning,
    )
