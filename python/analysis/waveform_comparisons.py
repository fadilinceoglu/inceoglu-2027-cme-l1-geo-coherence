"""Full-record pairwise waveform alignment and morphology metrics.

All seven spacecraft and four waveform families share the same four-day
one-minute UTC grid. Pairwise lag searches and graph synchronization precede
the synchronized bootstrap and circular-shift inference in waveform_inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, least_squares
from scipy.signal import correlate

from config import study
from config.waveform_bands import WAVEFORM_FAMILIES
from analysis.product_io import boolean_values, utc_text
from analysis.geometry import EPHEMERIS_CACHE, reference_positions
from config.paths import PROCESSED_DIR
from ingest.ephemeris import RE_KM as SSCWEB_CACHE_RE_KM


PROCESSED: Final = PROCESSED_DIR
BANDS: Final = PROCESSED / "bands"
REFERENCE_EPOCH: Final = pd.Timestamp(study.GEOMETRY_EPOCH)
REPORTED_EARTH_RADIUS_KM: Final = float(study.EARTH_RADIUS_KM)

SCHEMA_VERSION: Final = 3
STRUCTURES = WAVEFORM_FAMILIES
CRAFT_ORDER = study.UPSTREAM_SPACECRAFT
EPHEMERIS_LABELS: Final = {
    craft: study.spacecraft_label(craft)
    for craft in (*study.L1_SPACECRAFT, *study.ARTEMIS_SPACECRAFT)
}
PAIR_ORDER: Final = tuple(itertools.combinations(CRAFT_ORDER, 2))
L1_CRAFT = study.L1_SPACECRAFT
PRIMARY_REGISTRATION_EDGES: Final = (
    *(("solar_orbiter", craft) for craft in L1_CRAFT),
    *itertools.combinations(L1_CRAFT, 2),
    *((craft, "artemis_p1") for craft in L1_CRAFT),
    ("artemis_p1", "artemis_p2"),
)
HELD_OUT_REGISTRATION_EDGES: Final = tuple(
    pair for pair in PAIR_ORDER if pair not in PRIMARY_REGISTRATION_EDGES
)


def registration_design_matrix() -> np.ndarray:
    """Build the shared chain graph with Solar Orbiter fixed at zero."""
    unknown = CRAFT_ORDER[1:]
    craft_index = {craft: index for index, craft in enumerate(unknown)}
    rows: list[np.ndarray] = []
    for source, target in PRIMARY_REGISTRATION_EDGES:
        row = np.zeros(len(unknown), dtype=float)
        if source != "solar_orbiter":
            row[craft_index[source]] -= 1.0
        if target != "solar_orbiter":
            row[craft_index[target]] += 1.0
        rows.append(row)
    return np.asarray(rows, dtype=float)


GRAPH_DESIGN: Final = registration_design_matrix()


def solve_registration_graph(observed: np.ndarray) -> OptimizeResult:
    """Fit primary-edge lags; callers handle convergence and offset storage."""
    initial, *_ = np.linalg.lstsq(GRAPH_DESIGN, observed, rcond=None)
    return least_squares(
        lambda offsets: GRAPH_DESIGN @ offsets - observed,
        initial,
        loss="soft_l1",
        f_scale=1.0,
    )


WINDOW_START: Final = pd.Timestamp(study.ANALYSIS_START)
WINDOW_STOP: Final = pd.Timestamp(study.ANALYSIS_STOP)
FULL_GRID: Final = pd.date_range(
    WINDOW_START,
    WINDOW_STOP,
    freq="1min",
    inclusive="left",
)
FULL_SAMPLE_COUNT: Final = len(FULL_GRID)
CADENCE_SECONDS: Final = 60

MAXIMUM_ABSOLUTE_LAG_MINUTES: Final = 720
LAG_GRID_MINUTES: Final = np.arange(
    -MAXIMUM_ABSOLUTE_LAG_MINUTES,
    MAXIMUM_ABSOLUTE_LAG_MINUTES + 1,
    dtype=int,
)
EXPANDED_LAG_GRID_MINUTES: Final = np.arange(-1440, 1441, dtype=int)
MINIMUM_OVERLAP_FRACTION: Final = 0.70
MINIMUM_PAIRED_SAMPLE_COUNT: Final = int(
    np.ceil(MINIMUM_OVERLAP_FRACTION * FULL_SAMPLE_COUNT)
)

RANDOM_SEED: Final = 20260731


@dataclass(frozen=True)
class WaveformSeries:
    """One exact saved waveform on the common four-day UTC grid."""

    craft: str
    structure: str
    path: Path
    values_nt: np.ndarray
    valid: np.ndarray
    source_interpolated: np.ndarray


def load_waveform(
    craft: str,
    structure: str,
    *,
    bands_directory: Path | str = BANDS,
) -> WaveformSeries:
    """Read one saved ``bt_f`` record and require all 5,760 minutes."""

    path = Path(bands_directory) / f"{craft}__{structure}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {"time", "bt_f", "valid", "source_interpolated"}
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing {sorted(missing)}")
    time = pd.to_datetime(frame["time"], utc=True)
    if not time.is_monotonic_increasing or time.duplicated().any():
        raise ValueError(f"{path.name} time must be sorted and unique")
    selected = (time >= WINDOW_START) & (time < WINDOW_STOP)
    window = frame.loc[selected].copy()
    window_time = pd.DatetimeIndex(time[selected])
    if not window_time.equals(FULL_GRID):
        raise ValueError(
            f"{path.name} must contain the exact half-open 5,760-minute "
            "four-day Figure-5 grid"
        )
    valid = boolean_values(window["valid"], name=f"{path.name}/valid")
    values = pd.to_numeric(window["bt_f"], errors="coerce").to_numpy(float)
    if not np.isfinite(values[valid]).all():
        raise ValueError(
            f"{path.name} has a non-finite bt_f value marked valid"
        )
    # Invalid saved rows remain on the common clock grid but are never read as
    # observations by any pair calculation.
    values = np.where(valid, values, np.nan)
    interpolated = boolean_values(
        window["source_interpolated"],
        name=f"{path.name}/source_interpolated",
    )
    if craft != "wind" and interpolated.any():
        raise ValueError(f"unexpected source-interpolated minute in {path.name}")
    return WaveformSeries(
        craft=craft,
        structure=structure,
        path=path,
        values_nt=values,
        valid=valid,
        source_interpolated=interpolated,
    )


def load_all_waveforms(
    *,
    bands_directory: Path | str = BANDS,
) -> dict[tuple[str, str], WaveformSeries]:
    records = {
        (structure, craft): load_waveform(
            craft,
            structure,
            bands_directory=bands_directory,
        )
        for structure in STRUCTURES
        for craft in CRAFT_ORDER
    }
    counts = {
        int(record.source_interpolated.sum())
        for (structure, craft), record in records.items()
        if craft == "wind"
    }
    if len(counts) != 1:
        raise ValueError("Wind interpolation flags differ among waveform families")
    return records


def load_positions() -> dict[str, np.ndarray]:
    """Load the seven frozen reference-epoch positions used by Figure 5."""

    positions = reference_positions(cache=EPHEMERIS_CACHE, epoch=REFERENCE_EPOCH)
    return {craft: positions[craft] for craft in CRAFT_ORDER}


def _baseline_class(source: str, target: str) -> str:
    l1 = set(L1_CRAFT)
    artemis = {"artemis_p1", "artemis_p2"}
    if source == "solar_orbiter":
        return "solar_orbiter_to_near_earth"
    if source in l1 and target in l1:
        return "within_l1"
    if source in l1 and target in artemis:
        return "l1_to_artemis"
    if source == "artemis_p1" and target == "artemis_p2":
        return "p1_to_p2_lunar_context"
    raise ValueError(f"unsupported baseline {source} -> {target}")


def pair_geometry(positions: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Return all 21 fixed-epoch baselines in geocentric GSE."""

    rows: list[dict[str, Any]] = []
    for source, target in PAIR_ORDER:
        delta = np.asarray(positions[target]) - np.asarray(positions[source])
        full = float(np.linalg.norm(delta))
        rows.append(
            {
                "source_craft": source,
                "target_craft": target,
                "baseline_class": _baseline_class(source, target),
                "delta_x_gse_re": float(delta[0]),
                "delta_y_gse_re": float(delta[1]),
                "delta_z_gse_re": float(delta[2]),
                "radial_abs_dx_re": float(abs(delta[0])),
                "transverse_to_gse_x_re": float(np.hypot(delta[1], delta[2])),
                "separation_3d_re": full,
                "separation_3d_km": full * REPORTED_EARTH_RADIUS_KM,
            }
        )
    table = pd.DataFrame(rows)
    if len(table) != len(PAIR_ORDER):
        raise AssertionError("expected 21 unique seven-spacecraft baselines")
    return table


def _geometry_lookup(
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Calculate Figure-5 geometry directly from its frozen source cache."""

    positions = load_positions()
    rows = pair_geometry(positions).to_dict(orient="records")
    lookup = {
        (str(row["source_craft"]), str(row["target_craft"])): row
        for row in rows
    }
    context = {
        "path": str(EPHEMERIS_CACHE),
        "coordinate_frame": "geocentric GSE",
        "reference_epoch_utc": utc_text(REFERENCE_EPOCH),
        "position_scope": (
            "fixed reference-epoch geometry; not a time-varying embedded-"
            "feature map"
        ),
        "position_provenance": {
            "near_earth_spacecraft": {
                "craft": list(EPHEMERIS_LABELS),
                "source": str(EPHEMERIS_CACHE),
                "method": (
                    "native SSCWeb GSE positions, linearly interpolated to "
                    "the reference epoch"
                ),
            },
            "solar_orbiter": {
                "source": "config/study.py SOLAR_ORBITER_POSITION_RE",
                "method": "JPL Horizons position rotated to GSE",
                "time_varying_gse_track_available": False,
            },
        },
        "distance_units": {
            "reported_separation_earth_radius_km": REPORTED_EARTH_RADIUS_KM,
            "sscweb_cache_coordinate_conversion_radius_km": SSCWEB_CACHE_RE_KM,
            "radius_convention_note": (
                "SSCWeb Cartesian kilometres were converted to cached R_E "
                "with the ingest convention; study separations convert those "
                "dimensionless coordinates with study.EARTH_RADIUS_KM"
            ),
            "positions": "R_E",
        },
    }
    return lookup, context


def lagged_normalized_cross_correlation(
    source: np.ndarray,
    target: np.ndarray,
    *,
    lags_minutes: np.ndarray = LAG_GRID_MINUTES,
    minimum_paired_count: int = MINIMUM_PAIRED_SAMPLE_COUNT,
    origin_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Pearson NCC for ``source(t)`` versus ``target(t + lag)``.

    The FFT path is exact up to floating-point roundoff.  Lag-specific means
    and variances are used, including when either input contains missing
    values in the Wind-exclusion sensitivity.
    """

    left = np.asarray(source, dtype=float)
    right = np.asarray(target, dtype=float)
    lags = np.asarray(lags_minutes, dtype=int)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("source and target must be equal-length 1-D arrays")
    n = len(left)
    if n < 3 or np.any(np.abs(lags) >= n):
        raise ValueError("lag grid must be strictly inside the record length")
    if minimum_paired_count < 3:
        raise ValueError("minimum_paired_count must be at least three")

    if origin_weights is None:
        weights = np.ones(n, dtype=float)
        weighted = False
    else:
        weights = np.asarray(origin_weights, dtype=float)
        if weights.shape != left.shape:
            raise ValueError("origin_weights must match the input shape")
        if not np.isfinite(weights).all() or (weights < 0.0).any():
            raise ValueError("origin_weights must be finite and nonnegative")
        weighted = True

    left_valid = np.isfinite(left)
    right_valid = np.isfinite(right)
    left_zero = np.where(left_valid, left, 0.0)
    right_zero = np.where(right_valid, right, 0.0)
    full_index = lags + (n - 1)

    if left_valid.all() and right_valid.all() and not weighted:
        counts = n - np.abs(lags)
        cross = correlate(right, left, mode="full", method="fft")[full_index]
        left_sum = np.empty(len(lags), dtype=float)
        right_sum = np.empty(len(lags), dtype=float)
        left_square_sum = np.empty(len(lags), dtype=float)
        right_square_sum = np.empty(len(lags), dtype=float)
        left_cumulative = np.r_[0.0, np.cumsum(left)]
        right_cumulative = np.r_[0.0, np.cumsum(right)]
        left_square_cumulative = np.r_[0.0, np.cumsum(np.square(left))]
        right_square_cumulative = np.r_[0.0, np.cumsum(np.square(right))]
        positive = lags >= 0
        k = lags[positive]
        left_sum[positive] = left_cumulative[n - k]
        right_sum[positive] = right_cumulative[n] - right_cumulative[k]
        left_square_sum[positive] = left_square_cumulative[n - k]
        right_square_sum[positive] = (
            right_square_cumulative[n] - right_square_cumulative[k]
        )
        negative = ~positive
        p = -lags[negative]
        left_sum[negative] = left_cumulative[n] - left_cumulative[p]
        right_sum[negative] = right_cumulative[n - p]
        left_square_sum[negative] = (
            left_square_cumulative[n] - left_square_cumulative[p]
        )
        right_square_sum[negative] = right_square_cumulative[n - p]
    else:
        left_mask = left_valid.astype(float)
        right_mask = right_valid.astype(float)
        counts = np.rint(
            correlate(
                right_mask,
                weights * left_mask,
                mode="full",
                method="fft",
            )[
                full_index
            ]
        ).astype(int)
        cross = correlate(
            right_zero,
            weights * left_zero,
            mode="full",
            method="fft",
        )[full_index]
        left_sum = correlate(
            right_mask,
            weights * left_zero,
            mode="full",
            method="fft",
        )[full_index]
        right_sum = correlate(
            right_zero,
            weights * left_mask,
            mode="full",
            method="fft",
        )[full_index]
        left_square_sum = correlate(
            right_mask,
            weights * np.square(left_zero),
            mode="full",
            method="fft",
        )[full_index]
        right_square_sum = correlate(
            np.square(right_zero),
            weights * left_mask,
            mode="full",
            method="fft",
        )[full_index]

    correlations = np.full(len(lags), np.nan, dtype=float)
    eligible = counts >= int(minimum_paired_count)
    count_float = counts[eligible].astype(float)
    covariance = (
        cross[eligible]
        - left_sum[eligible] * right_sum[eligible] / count_float
    )
    left_ss = (
        left_square_sum[eligible]
        - np.square(left_sum[eligible]) / count_float
    )
    right_ss = (
        right_square_sum[eligible]
        - np.square(right_sum[eligible]) / count_float
    )
    denominator = np.sqrt(np.maximum(0.0, left_ss * right_ss))
    finite = np.isfinite(covariance) & np.isfinite(denominator) & (denominator > 0)
    positions = np.flatnonzero(eligible)
    correlations[positions[finite]] = np.clip(
        covariance[finite] / denominator[finite],
        -1.0,
        1.0,
    )
    return correlations, counts


def select_peak_lag(
    correlations: np.ndarray,
    *,
    lags_minutes: np.ndarray = LAG_GRID_MINUTES,
) -> int:
    values = np.asarray(correlations, dtype=float)
    lags = np.asarray(lags_minutes, dtype=int)
    if values.shape != lags.shape:
        raise ValueError("correlations and lags must have equal shape")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("no lag satisfies the overlap and variance rules")
    maximum = float(np.nanmax(values))
    tied = np.flatnonzero(finite & np.isclose(values, maximum, atol=1e-12))
    minimum_absolute = np.min(np.abs(lags[tied]))
    tied = tied[np.abs(lags[tied]) == minimum_absolute]
    return int(tied[0])


def aligned_pair(
    source: np.ndarray,
    target: np.ndarray,
    lag_minutes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite samples for ``source(t)`` and ``target(t + lag)``."""

    left = np.asarray(source, dtype=float)
    right = np.asarray(target, dtype=float)
    lag = int(lag_minutes)
    if lag > 0:
        a, b = left[:-lag], right[lag:]
    elif lag < 0:
        a, b = left[-lag:], right[:lag]
    else:
        a, b = left, right
    paired = np.isfinite(a) & np.isfinite(b)
    return a[paired], b[paired]


def sampled_origin_pair(
    source: np.ndarray,
    target: np.ndarray,
    lag_minutes: int,
    origin_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample original ``source(t), target(t+lag)`` at bootstrap origins."""

    left = np.asarray(source, dtype=float)
    right = np.asarray(target, dtype=float)
    origins = np.asarray(origin_indices, dtype=int)
    target_indices = origins + int(lag_minutes)
    inside = (
        (origins >= 0)
        & (origins < len(left))
        & (target_indices >= 0)
        & (target_indices < len(right))
    )
    origins = origins[inside]
    target_indices = target_indices[inside]
    a = left[origins]
    b = right[target_indices]
    paired = np.isfinite(a) & np.isfinite(b)
    return a[paired], b[paired]


def _pair_metric_components(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.float64, float, float, float, float, float]:
    """Compute shared reductions without repeating bootstrap validation."""
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    centered_dot = np.dot(left_centered, right_centered)
    rms_source = float(np.sqrt(np.mean(np.square(left))))
    rms_target = float(np.sqrt(np.mean(np.square(right))))
    rmse = float(np.sqrt(np.mean(np.square(right - left))))
    symmetric_scale = float(
        np.sqrt(np.mean(0.5 * (np.square(left) + np.square(right))))
    )
    return centered_dot, denominator, rms_source, rms_target, rmse, symmetric_scale


def bootstrap_pair_metrics(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, float]:
    """Return NCC, RMS ratio and symmetric NRMSE for finite bootstrap pairs.

    The bootstrap already selects paired finite samples. Avoid the full
    product validation and dictionary allocation in its inner loop.
    """
    left = np.asarray(source, dtype=float)
    right = np.asarray(target, dtype=float)
    dot, denominator, rms_source, rms_target, rmse, scale = (
        _pair_metric_components(left, right)
    )
    return float(dot / denominator), float(rms_target / rms_source), float(rmse / scale)


def pair_metrics(source: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    left = np.asarray(source, dtype=float)
    right = np.asarray(target, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 3:
        raise ValueError("paired metric inputs must be equal 1-D samples")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired metric inputs must be finite")
    dot, denominator, rms_source, rms_target, rmse, symmetric_scale = (
        _pair_metric_components(left, right)
    )
    correlation = float(dot / denominator) if denominator > 0.0 else np.nan
    return {
        "paired_sample_count": int(len(left)),
        "overlap_fraction_of_full_window": float(len(left) / FULL_SAMPLE_COUNT),
        "normalized_cross_correlation": correlation,
        "rms_source_nt": rms_source,
        "rms_target_nt": rms_target,
        "rms_ratio_target_over_source": (
            float(rms_target / rms_source) if rms_source > 0.0 else np.nan
        ),
        "symmetric_normalized_rmse": (
            float(rmse / symmetric_scale) if symmetric_scale > 0.0 else np.nan
        ),
    }


def _optional_float(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _wind_exclusion_sensitivity(
    source: WaveformSeries,
    target: WaveformSeries,
    *,
    primary_lag_minutes: int,
    primary_metrics: Mapping[str, Any],
) -> dict[str, Any] | None:
    if source.craft != "wind" and target.craft != "wind":
        return None
    left = source.values_nt.copy()
    right = target.values_nt.copy()
    if source.craft == "wind":
        left[source.source_interpolated] = np.nan
        excluded = int(source.source_interpolated.sum())
    else:
        right[target.source_interpolated] = np.nan
        excluded = int(target.source_interpolated.sum())
    correlations, counts = lagged_normalized_cross_correlation(left, right)
    selected_index = select_peak_lag(correlations)
    lag = int(LAG_GRID_MINUTES[selected_index])
    aligned_source, aligned_target = aligned_pair(left, right, lag)
    metrics = pair_metrics(aligned_source, aligned_target)
    return {
        "method": (
            "repeat the complete lag search after masking Wind output samples "
            "whose one-minute source value was linearly interpolated"
        ),
        "scope_note": (
            "This masks the flagged output minutes; it does not reconstruct a "
            "different zero-phase filter across missing input samples."
        ),
        "excluded_wind_minute_count": excluded,
        "eligible": bool(
            int(counts[selected_index]) >= MINIMUM_PAIRED_SAMPLE_COUNT
        ),
        "lag_minutes": lag,
        "lag_change_minutes": int(lag - primary_lag_minutes),
        "normalized_cross_correlation": float(
            metrics["normalized_cross_correlation"]
        ),
        "normalized_cross_correlation_change": float(
            metrics["normalized_cross_correlation"]
            - float(primary_metrics["normalized_cross_correlation"])
        ),
        "rms_ratio_target_over_source": float(
            metrics["rms_ratio_target_over_source"]
        ),
        "symmetric_normalized_rmse": float(
            metrics["symmetric_normalized_rmse"]
        ),
        "paired_sample_count": int(metrics["paired_sample_count"]),
    }


def analyze_pair(
    structure: str,
    source: WaveformSeries,
    target: WaveformSeries,
    *,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    expanded_correlations, expanded_counts = lagged_normalized_cross_correlation(
        source.values_nt, target.values_nt, lags_minutes=EXPANDED_LAG_GRID_MINUTES,
    )
    central = np.isin(EXPANDED_LAG_GRID_MINUTES, LAG_GRID_MINUTES)
    correlations, counts = expanded_correlations[central], expanded_counts[central]
    selected_index = select_peak_lag(correlations)
    lag = int(LAG_GRID_MINUTES[selected_index])
    expanded_index = select_peak_lag(
        expanded_correlations,
        lags_minutes=EXPANDED_LAG_GRID_MINUTES,
    )
    expanded_lag = int(EXPANDED_LAG_GRID_MINUTES[expanded_index])
    aligned_source, aligned_target = aligned_pair(
        source.values_nt,
        target.values_nt,
        lag,
    )
    metrics = pair_metrics(aligned_source, aligned_target)
    eligible = bool(
        int(counts[selected_index]) >= MINIMUM_PAIRED_SAMPLE_COUNT
    )
    sensitivity = _wind_exclusion_sensitivity(
        source,
        target,
        primary_lag_minutes=lag,
        primary_metrics=metrics,
    )
    return {
        "structure": structure,
        "source_craft": source.craft,
        "target_craft": target.craft,
        "analysis_domain": "common_full_window_one_minute_utc",
        "eligible": eligible,
        "full_window_sample_count": FULL_SAMPLE_COUNT,
        "minimum_paired_sample_count": MINIMUM_PAIRED_SAMPLE_COUNT,
        "alignment": {
            "method": "normalized lagged cross-correlation",
            "lag_minutes": lag,
            "lag_seconds": int(lag * CADENCE_SECONDS),
            "lag_sign_convention": (
                "positive means target morphology is later than source; "
                "NCC compares source(t) with target(t + lag)"
            ),
            "lag_grid_minimum_minutes": int(LAG_GRID_MINUTES[0]),
            "lag_grid_maximum_minutes": int(LAG_GRID_MINUTES[-1]),
            "lag_grid_step_minutes": 1,
            "selected_lag_at_search_edge": bool(
                selected_index in {0, len(LAG_GRID_MINUTES) - 1}
            ),
            "expanded_search_sensitivity": {
                "lag_grid_minimum_minutes": int(
                    EXPANDED_LAG_GRID_MINUTES[0]
                ),
                "lag_grid_maximum_minutes": int(
                    EXPANDED_LAG_GRID_MINUTES[-1]
                ),
                "selected_lag_minutes": expanded_lag,
                "selected_lag_unchanged": bool(expanded_lag == lag),
                "selected_lag_at_expanded_search_edge": bool(
                    expanded_index
                    in {0, len(EXPANDED_LAG_GRID_MINUTES) - 1}
                ),
            },
            "paired_sample_count": int(counts[selected_index]),
            "overlap_fraction_of_full_window": float(
                counts[selected_index] / FULL_SAMPLE_COUNT
            ),
        },
        "metrics": {
            key: _optional_float(value) if key != "paired_sample_count" else int(value)
            for key, value in metrics.items()
        },
        "wind_interpolation_exclusion_sensitivity": sensitivity,
        "geometry": dict(geometry),
    }


def _solve_global_registration(
    structure: str,
    records: list[dict[str, Any]],
    waveforms: Mapping[tuple[str, str], WaveformSeries],
) -> dict[str, Any]:
    """Derive one cadence-resolved craft shift from the chain graph.

    The robust solve uses the 15 propagation-topology edges and fixes Solar
    Orbiter at zero.  Six shortcut pairs are retained only as closure checks,
    so P2 cannot bypass P1 and no L1 signal is aggregated.
    """

    relevant = [row for row in records if row["structure"] == structure]
    lookup = {
        (str(row["source_craft"]), str(row["target_craft"])): row
        for row in relevant
    }
    if set(lookup) != set(PAIR_ORDER):
        raise ValueError(f"{structure} lacks the complete 21-pair graph")

    unknown_craft = CRAFT_ORDER[1:]
    observed_lags: list[float] = []
    for source, target in PRIMARY_REGISTRATION_EDGES:
        observed_lags.append(
            float(lookup[(source, target)]["alignment"]["lag_minutes"])
        )
    observed = np.asarray(observed_lags, dtype=float)
    solution = solve_registration_graph(observed)
    continuous_offsets = {
        "solar_orbiter": 0.0,
        **{
            craft: float(solution.x[index])
            for index, craft in enumerate(unknown_craft)
        },
    }
    integer_offsets = {
        craft: int(np.rint(offset))
        for craft, offset in continuous_offsets.items()
    }

    edge_diagnostics: list[dict[str, Any]] = []
    for source, target in PAIR_ORDER:
        record = lookup[(source, target)]
        peak_lag = int(record["alignment"]["lag_minutes"])
        continuous_lag = float(
            continuous_offsets[target] - continuous_offsets[source]
        )
        registered_lag = int(integer_offsets[target] - integer_offsets[source])
        left, right = aligned_pair(
            waveforms[(structure, source)].values_nt,
            waveforms[(structure, target)].values_nt,
            registered_lag,
        )
        registered_metrics = pair_metrics(left, right)
        role = (
            "registration_edge"
            if (source, target) in PRIMARY_REGISTRATION_EDGES
            else "held_out_closure_check"
        )
        record["global_registration"] = {
            "role": role,
            "lag_minutes": registered_lag,
            "lag_sign_convention": (
                "positive means target is later than source; equal to the "
                "target craft offset minus the source craft offset"
            ),
            "continuous_graph_lag_minutes": continuous_lag,
            "pair_peak_lag_minus_registered_lag_minutes": int(
                peak_lag - registered_lag
            ),
            "paired_sample_count": int(
                registered_metrics["paired_sample_count"]
            ),
            "overlap_fraction_of_full_window": float(
                registered_metrics["overlap_fraction_of_full_window"]
            ),
            "normalized_cross_correlation": float(
                registered_metrics["normalized_cross_correlation"]
            ),
        }
        edge_diagnostics.append(
            {
                "source_craft": source,
                "target_craft": target,
                "role": role,
                "pair_peak_lag_minutes": peak_lag,
                "continuous_graph_lag_minutes": continuous_lag,
                "integer_registered_lag_minutes": registered_lag,
                "pair_peak_minus_continuous_graph_lag_minutes": float(
                    peak_lag - continuous_lag
                ),
                "pair_peak_ncc": float(
                    record["metrics"]["normalized_cross_correlation"]
                ),
                "registered_ncc": float(
                    registered_metrics["normalized_cross_correlation"]
                ),
            }
        )

    primary_residuals = [
        abs(float(row["pair_peak_minus_continuous_graph_lag_minutes"]))
        for row in edge_diagnostics
        if row["role"] == "registration_edge"
    ]
    held_out_residuals = [
        abs(float(row["pair_peak_minus_continuous_graph_lag_minutes"]))
        for row in edge_diagnostics
        if row["role"] == "held_out_closure_check"
    ]
    return {
        "method": (
            "equal-weight soft-L1 graph synchronization on 15 propagation-"
            "topology pair-peak lags with SO fixed at zero"
        ),
        "loss": "soft_l1",
        "loss_scale_minutes": 1.0,
        "loss_scale_basis": "one-minute input cadence",
        "display_shift_rule": "shift timestamp by negative craft offset",
        "signal_aggregation": "none",
        "amplitude_adjustment": "none",
        "primary_edges": [list(pair) for pair in PRIMARY_REGISTRATION_EDGES],
        "held_out_closure_edges": [
            list(pair) for pair in HELD_OUT_REGISTRATION_EDGES
        ],
        "continuous_offsets_from_so_minutes": continuous_offsets,
        "integer_offsets_from_so_minutes": integer_offsets,
        "integer_resolution_note": (
            "continuous graph solution rounded to the nearest minute; no "
            "sub-minute timing precision is claimed"
        ),
        "optimizer_success": bool(solution.success),
        "optimizer_message": str(solution.message),
        "maximum_absolute_primary_edge_residual_minutes": float(
            max(primary_residuals)
        ),
        "maximum_absolute_held_out_closure_residual_minutes": float(
            max(held_out_residuals)
        ),
        "edge_diagnostics": edge_diagnostics,
    }


def analyze_canonical_pairs(
    *,
    bands_directory: Path | str = BANDS,
) -> dict[str, Any]:
    waveforms = load_all_waveforms(bands_directory=bands_directory)
    geometry, geometry_context = _geometry_lookup()
    records: list[dict[str, Any]] = []
    for structure in STRUCTURES:
        for source, target in PAIR_ORDER:
            records.append(
                analyze_pair(
                    structure,
                    waveforms[(structure, source)],
                    waveforms[(structure, target)],
                    geometry=geometry[(source, target)],
                )
            )
    global_registration = {
        structure: _solve_global_registration(
            structure,
            records,
            waveforms,
        )
        for structure in STRUCTURES
    }

    inventory = [
        {
            "structure": structure,
            "craft": craft,
            "path": str(record.path),
            "sample_count": FULL_SAMPLE_COUNT,
            "valid_sample_count": int(record.valid.sum()),
            "invalid_sample_count": int((~record.valid).sum()),
            "source_interpolated_minute_count": int(
                record.source_interpolated.sum()
            ),
        }
        for (structure, craft), record in waveforms.items()
    ]
    result = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "scientific_question": (
                "How strongly does each saved isolated scalar-|B| waveform "
                "remain morphologically coherent across each spacecraft pair?"
            ),
            "analysis_domain": "uniform four-day one-minute UTC",
            "window_start_utc": utc_text(WINDOW_START),
            "window_stop_utc": utc_text(WINDOW_STOP),
            "window_endpoint_policy": "start inclusive, stop exclusive",
            "cadence_seconds": CADENCE_SECONDS,
            "full_window_sample_count": FULL_SAMPLE_COUNT,
            "input_policy": (
                "exact saved one-minute bt_f rows and saved valid masks; all "
                "four families and all seven spacecraft use the same 5,760-row "
                "UTC grid, and invalid rows are never treated as observations"
            ),
            "alignment_policy": (
                "identical total-lag NCC search on every family/pair, followed "
                "by equal-weight robust synchronization of the 15 chain-graph "
                "edges to obtain one display shift per spacecraft; no boundary "
                "anchor, crop, gain, stretch, warp, convected-distance mapping, "
                "or L1 signal aggregate"
            ),
            "lag_grid_minutes": [
                int(LAG_GRID_MINUTES[0]),
                int(LAG_GRID_MINUTES[-1]),
            ],
            "lag_step_minutes": 1,
            "lag_search_horizon_interpretation": (
                "common predeclared search horizon, not a velocity-derived "
                "physical propagation bound"
            ),
            "expanded_lag_sensitivity_grid_minutes": [
                int(EXPANDED_LAG_GRID_MINUTES[0]),
                int(EXPANDED_LAG_GRID_MINUTES[-1]),
            ],
            "lag_sign_convention": (
                "positive means target is later than source; compare source(t) "
                "with target(t + lag)"
            ),
            "minimum_overlap_fraction": MINIMUM_OVERLAP_FRACTION,
            "minimum_paired_sample_count": MINIMUM_PAIRED_SAMPLE_COUNT,
            "wind_interpolation_policy": (
                "primary analysis retains the approved filled minute values; "
                "every Wind pair also repeats the full lag search after masking "
                "the flagged output minutes"
            ),
            "craft_order": list(CRAFT_ORDER),
            "structure_order": list(STRUCTURES),
            "pair_count_per_structure": len(PAIR_ORDER),
            "record_count": len(records),
            "random_seed": RANDOM_SEED,
        },
        "geometry_context": geometry_context,
        "global_registration_by_structure": global_registration,
        "input_inventory": inventory,
        "scientific_caveats": [
            "Lag-optimized correlation is morphological coherence, not proof of causal transmission.",
            "Graph-registered correlation describes the synchronized display clock; pair-peak correlation describes optimized pairwise alignment.",
            "Solar Orbiter is a flank observer; its long baseline is geometric context and is excluded from incident-chain persistence summaries.",
            "P2 is a lunar-wake observer; differences involving P2 can contain lunar modification.",
            "The trend is a mean-retaining low-pass and is interpreted separately from fluctuation bands.",
            "Wind exclusion sensitivity masks the flagged minutes filled after filtering and averaging; it does not change the temporary spatial gap bridging used during filtering.",
            "All 28 saved inputs contain the complete 5,760-minute record; lag-specific overlap decreases only through the tested translation and any explicitly masked sensitivity samples.",
        ],
        "pair_records": records,
    }
    validate_result(result)
    return result


def records_to_table(result: Mapping[str, Any]) -> pd.DataFrame:
    validate_result(result)
    rows: list[dict[str, Any]] = []
    for record in result["pair_records"]:
        geometry = record["geometry"]
        alignment = record["alignment"]
        metrics = record["metrics"]
        global_alignment = record["global_registration"]
        sensitivity = record.get("wind_interpolation_exclusion_sensitivity")
        row = {
            "schema_version": result["meta"]["schema_version"],
            "window_start_utc": result["meta"]["window_start_utc"],
            "window_stop_utc": result["meta"]["window_stop_utc"],
            "full_window_sample_count": record["full_window_sample_count"],
            "structure": record["structure"],
            "source_craft": record["source_craft"],
            "target_craft": record["target_craft"],
            "eligible": record["eligible"],
            "minimum_paired_sample_count": record[
                "minimum_paired_sample_count"
            ],
            "lag_minutes": alignment["lag_minutes"],
            "total_delay_seconds": alignment["lag_seconds"],
            "selected_lag_at_search_edge": alignment[
                "selected_lag_at_search_edge"
            ],
            "expanded_search_selected_lag_minutes": alignment[
                "expanded_search_sensitivity"
            ]["selected_lag_minutes"],
            "expanded_search_lag_unchanged": alignment[
                "expanded_search_sensitivity"
            ]["selected_lag_unchanged"],
            "paired_sample_count": metrics["paired_sample_count"],
            "overlap_fraction_of_full_window": metrics[
                "overlap_fraction_of_full_window"
            ],
            "normalized_cross_correlation": metrics[
                "normalized_cross_correlation"
            ],
            "global_registration_role": global_alignment["role"],
            "globally_registered_lag_minutes": global_alignment[
                "lag_minutes"
            ],
            "pair_peak_lag_minus_registered_lag_minutes": global_alignment[
                "pair_peak_lag_minus_registered_lag_minutes"
            ],
            "globally_registered_normalized_cross_correlation": global_alignment[
                "normalized_cross_correlation"
            ],
            "rms_source_nt": metrics["rms_source_nt"],
            "rms_target_nt": metrics["rms_target_nt"],
            "rms_ratio_target_over_source": metrics[
                "rms_ratio_target_over_source"
            ],
            "symmetric_normalized_rmse": metrics[
                "symmetric_normalized_rmse"
            ],
            "baseline_class": geometry["baseline_class"],
            "delta_x_gse_re": geometry["delta_x_gse_re"],
            "delta_y_gse_re": geometry["delta_y_gse_re"],
            "delta_z_gse_re": geometry["delta_z_gse_re"],
            "radial_abs_dx_re": geometry["radial_abs_dx_re"],
            "transverse_to_gse_x_re": geometry[
                "transverse_to_gse_x_re"
            ],
            "separation_3d_re": geometry["separation_3d_re"],
            "separation_3d_km": geometry["separation_3d_km"],
            "wind_exclusion_sensitivity_available": sensitivity is not None,
            "wind_excluded_minute_count": (
                sensitivity["excluded_wind_minute_count"]
                if sensitivity is not None
                else None
            ),
            "wind_exclusion_lag_minutes": (
                sensitivity["lag_minutes"] if sensitivity is not None else None
            ),
            "wind_exclusion_lag_change_minutes": (
                sensitivity["lag_change_minutes"]
                if sensitivity is not None
                else None
            ),
            "wind_exclusion_normalized_cross_correlation": (
                sensitivity["normalized_cross_correlation"]
                if sensitivity is not None
                else None
            ),
            "wind_exclusion_normalized_cross_correlation_change": (
                sensitivity["normalized_cross_correlation_change"]
                if sensitivity is not None
                else None
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def validate_result(
    result: Mapping[str, Any],
) -> None:
    meta = result.get("meta") or {}
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported canonical pair schema")
    if meta.get("full_window_sample_count") != FULL_SAMPLE_COUNT:
        raise ValueError("canonical result does not use all 5,760 samples")
    if meta.get("window_start_utc") != utc_text(WINDOW_START) or meta.get(
        "window_stop_utc"
    ) != utc_text(WINDOW_STOP):
        raise ValueError("canonical result uses the wrong global UTC window")
    if meta.get("lag_grid_minutes") != [
        -MAXIMUM_ABSOLUTE_LAG_MINUTES,
        MAXIMUM_ABSOLUTE_LAG_MINUTES,
    ]:
        raise ValueError("canonical result uses the wrong total-lag grid")
    records = list(result.get("pair_records") or [])
    expected = {
        (structure, source, target)
        for structure in STRUCTURES
        for source, target in PAIR_ORDER
    }
    observed = {
        (row["structure"], row["source_craft"], row["target_craft"])
        for row in records
    }
    if len(records) != 84 or observed != expected:
        raise ValueError("canonical result requires exactly 84 unique records")
    registrations = result.get("global_registration_by_structure") or {}
    if set(registrations) != set(STRUCTURES):
        raise ValueError("canonical result lacks a graph registration per family")
    for structure, registration in registrations.items():
        offsets = registration.get("integer_offsets_from_so_minutes") or {}
        if set(offsets) != set(CRAFT_ORDER) or int(offsets["solar_orbiter"]) != 0:
            raise ValueError(f"{structure} has an incomplete SO-fixed registration")
        if not registration.get("optimizer_success"):
            raise ValueError(f"{structure} graph synchronization did not converge")
    for row in records:
        if row.get("analysis_domain") != "common_full_window_one_minute_utc":
            raise ValueError("every pair must use the common UTC analysis domain")
        if row.get("full_window_sample_count") != FULL_SAMPLE_COUNT:
            raise ValueError("pair record does not use all 5,760 samples")
        alignment = row["alignment"]
        lag = int(alignment["lag_minutes"])
        if lag < -720 or lag > 720:
            raise ValueError("selected total lag lies outside [-720,+720]")
        if row["metrics"]["paired_sample_count"] < MINIMUM_PAIRED_SAMPLE_COUNT:
            raise ValueError("selected pair violates the 70% overlap rule")
        if not alignment["expanded_search_sensitivity"][
            "selected_lag_unchanged"
        ]:
            raise ValueError("selected lag changes in the expanded-range sensitivity")
        graph = row.get("global_registration") or {}
        expected_graph_lag = int(
            registrations[row["structure"]]["integer_offsets_from_so_minutes"][
                row["target_craft"]
            ]
            - registrations[row["structure"]]["integer_offsets_from_so_minutes"][
                row["source_craft"]
            ]
        )
        if int(graph.get("lag_minutes", 10**9)) != expected_graph_lag:
            raise ValueError("pair graph lag is inconsistent with craft offsets")
        geometry = row["geometry"]
        if not np.isfinite(float(geometry["separation_3d_re"])):
            raise ValueError("pair geometry is incomplete")
