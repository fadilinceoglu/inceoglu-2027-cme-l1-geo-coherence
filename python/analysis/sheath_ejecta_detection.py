"""Detect sheath/ejecta entry from variability, fleet association and rotation.

Inputs are an observed one-minute magnetic record and its measured shock time.
No adopted sheath/ejecta timestamp enters this calculation.
"""

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from analysis.magnetic_directions import mean_field_direction
from analysis import angular_transition_fit


@dataclass(frozen=True)
class Settings:
    variability_threshold_nt: float = 2.5
    variability_window_min: int = 10
    persistence_duration_min: int = 10
    persistence_fraction: float = 0.8
    reference_buffer_min: int = 2
    reference_duration_min: int = 10


@dataclass(frozen=True)
class FleetSettings:
    """Declared association choices; not optimized against adopted boundaries."""

    agreement_span_min: float = 10.0
    search_half_width_min: float = 10.0
    minimum_agreeing: int = 3


from config.study import L1_SPACECRAFT as L1_CRAFT


def _forward_fraction(values, minutes):
    # Includes the timestamps at t and t + duration: 11 minute values for 10 min.
    n = minutes + 1
    return values.iloc[::-1].rolling(n, min_periods=n).mean().iloc[::-1]


def _directions(frame):
    values = frame[["bx", "by", "bz"]].to_numpy(dtype=float)
    lengths = np.linalg.norm(values, axis=1)
    valid = np.isfinite(values).all(axis=1) & (lengths > 0)
    unit = np.full_like(values, np.nan)
    unit[valid] = values[valid] / lengths[valid, None]
    return unit


def _angle(unit, reference, index):
    return pd.Series(
        np.degrees(np.arccos(np.clip(unit @ reference, -1.0, 1.0))), index=index,
    )


def _prepare_record(frame, shock_time, settings):
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is None:
        raise ValueError("Minute record must have timezone-aware timestamps")
    if len(frame) < 2 or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Minute record must be sorted, unique, and nonempty")
    if (np.diff(frame.index.asi8) != pd.Timedelta("1min").value).any():
        raise ValueError("Use a complete one-minute time grid, leaving missing values as NaN")
    if not frame.index.equals(frame.index.floor("1min")):
        raise ValueError("Minute values must be labelled at UTC minute boundaries")
    if not 0 < settings.persistence_fraction <= 1:
        raise ValueError("Persistence fraction must be in (0,1]")
    if settings.variability_window_min < 2 or settings.variability_threshold_nt <= 0:
        raise ValueError("Variability window needs at least two samples and a positive threshold")
    shock_time = pd.Timestamp(shock_time)
    if shock_time.tzinfo is None:
        raise ValueError("Shock time must have an explicit timezone")
    if not frame.index[0] <= shock_time < frame.index[-1]:
        raise ValueError("Shock time must lie within the supplied record")
    for name in ("bx", "by", "bz", "bt"):
        frame[name] = frame[name].where(np.isfinite(frame[name]))
        if f"{name}_samples" in frame:
            frame[name] = frame[name].where(frame[f"{name}_samples"] > 0)
    return frame, shock_time


def _variability_intervals(frame, shock_time, settings):
    """Find all local qualifying episodes, before any fleet or angular check.

    An episode is a consecutive run of eligible minute timestamps. Keep its
    first timestamp as the candidate marker, rather than moving a marker within
    a quiet episode to make it closer to another spacecraft's time.
    """
    std = frame.bt.rolling(
        f"{settings.variability_window_min}min", center=True,
        min_periods=settings.variability_window_min,
    ).std(ddof=1)
    below = std.lt(settings.variability_threshold_nt).astype(float).where(std.notna())
    sustained = _forward_fraction(below, settings.persistence_duration_min)
    diagnostics = pd.DataFrame({"bt": frame.bt, "moving_std_nt": std,
                                "low_variability_fraction": sustained})
    eligible = (frame.index > shock_time) & below.eq(1) & sustained.ge(settings.persistence_fraction)
    diagnostics["variability_eligible"] = eligible
    starts = np.flatnonzero((eligible & ~eligible.shift(fill_value=False)).to_numpy())
    stops = np.flatnonzero((eligible & ~eligible.shift(-1, fill_value=False)).to_numpy())
    intervals = [
        {"start_utc": frame.index[a], "last_qualifying_utc": frame.index[b],
         "qualifying_minutes": int(b - a + 1)}
        for a, b in zip(starts, stops)
    ]
    return intervals, diagnostics


def select_l1_intervals(candidate_times, fleet_settings=FleetSettings()):
    """Associate L1 candidates using only the independent first variability markers.

    A unique largest group of at least three L1 observers must span no more
    than the declared agreement interval. Keep that group's choices unchanged.
    For the other observer, choose its candidate nearest the group's median,
    within the declared search interval. An exact tie, missing local candidate,
    or non-unique consensus stays unresolved. Selected replacements never feed
    back into the consensus. Observers outside L1 retain independent selection.
    """
    if (fleet_settings.agreement_span_min <= 0 or fleet_settings.search_half_width_min <= 0
            or not 3 <= fleet_settings.minimum_agreeing <= len(L1_CRAFT)):
        raise ValueError("Fleet comparison needs positive time limits and at least three L1 members")
    candidates = {key: sorted(set(pd.Timestamp(t) for t in values))
                  for key, values in candidate_times.items()}
    decisions = {
        key: {"status": "independent" if times else "no_local_candidate",
              "first_marker_utc": times[0] if times else None,
              "selected_marker_utc": times[0] if times else None}
        for key, times in candidates.items()
    }
    first = {key: times[0] for key, times in candidates.items() if key in L1_CRAFT and times}
    comparison = {"settings": asdict(fleet_settings), "first_markers_utc": first,
                  "status": "insufficient_agreement", "consensus_members": []}
    groups = []
    for size in range(len(first), fleet_settings.minimum_agreeing - 1, -1):
        groups = [list(group) for group in combinations(sorted(first), size)
                  if max(first[k] for k in group) - min(first[k] for k in group)
                  <= pd.Timedelta(minutes=fleet_settings.agreement_span_min)]
        if groups:
            break
    if len(groups) != 1:
        if len(groups) > 1:
            comparison.update(status="ambiguous_agreement", competing_groups=groups)
        for key in decisions:
            if key in L1_CRAFT:
                decisions[key].update(status="fleet_" + comparison["status"], selected_marker_utc=None)
        return decisions, comparison
    members = groups[0]
    stamps = sorted(first[key] for key in members)
    middle = len(stamps) // 2
    median = stamps[middle] if len(stamps) % 2 else stamps[middle-1] + (stamps[middle] - stamps[middle-1]) / 2
    lo = median - pd.Timedelta(minutes=fleet_settings.search_half_width_min)
    hi = median + pd.Timedelta(minutes=fleet_settings.search_half_width_min)
    comparison.update(status="agreement_found", consensus_members=members,
                      consensus_median_utc=median, search_start_utc=lo, search_stop_utc=hi)
    for key, decision in decisions.items():
        if key not in L1_CRAFT:
            continue
        if key in members:
            decision["status"] = "fleet_agrees"
            continue
        nearby = [t for t in candidates[key] if lo <= t <= hi]
        decision.update(selected_marker_utc=None, status="no_candidate_near_fleet",
                        nearby_markers_utc=nearby)
        if not nearby:
            continue
        distance = min(abs(t - median) for t in nearby)
        nearest = [t for t in nearby if abs(t - median) == distance]
        if len(nearest) != 1:
            decision["status"] = "ambiguous_nearby_candidates"
            continue
        decision.update(status="fleet_reselected", selected_marker_utc=nearest[0])
    return decisions, comparison


def _detect_prepared(frame, shock_time, settings, intervals, diagnostics, variability_marker, fit_settings):
    """Fit the angular decrease using an actual fleet-selected local episode.

    Reference and persistence slices include both endpoint timestamps. Reference
    means use finite observed vectors; variability/persistence need full support.
    The fitted timing half-width has cadence/gap floors; it is not a confidence
    interval. All accepted fits remain in the report for diagnostic inspection.
    """
    report = {"settings": asdict(settings), "shock_utc": shock_time, "flags": []}
    marker = pd.Timestamp(variability_marker)
    if marker not in [interval["start_utc"] for interval in intervals]:
        raise ValueError("Selected marker is not a locally qualifying episode start")
    start = marker + pd.Timedelta(minutes=settings.reference_buffer_min)
    stop = start + pd.Timedelta(minutes=settings.reference_duration_min)
    report.update(variability_marker_utc=marker, reference_start_utc=start,
                  reference_stop_utc=stop, status="incomplete_reference_window")
    reference_frame = frame.loc[start:stop]
    reference_units = _directions(reference_frame)
    valid_reference = np.isfinite(reference_units).all(axis=1)
    if len(reference_frame) != settings.reference_duration_min + 1 or not valid_reference.any():
        return report, diagnostics
    if not valid_reference.all():
        report["flags"].append("missing_vectors_in_ejecta_reference")
    reference_units = reference_units[valid_reference]
    reference = mean_field_direction(reference_frame[["bx", "by", "bz"]])
    angles = _angle(_directions(frame), reference, frame.index)
    diagnostics["theta_ejecta_deg"] = angles
    report.update(
        reference_unit_vector=reference.tolist(),
        reference_sample_count=int(valid_reference.sum()),
        reference_mean_unit_length=float(np.linalg.norm(reference_units.mean(axis=0))),
        reference_angle_p90_deg=float(angles.loc[start:stop].quantile(.9)),
        reference_angle_max_deg=float(angles.loc[start:stop].max()),
        status="no_accepted_fit",
    )
    fit, candidates = angular_transition_fit.detect(
        angles, shock_time, start, stop, fit_settings,
    )
    report.update(fit)
    report["fit_settings"] = asdict(fit_settings)
    report["fit_candidates"] = candidates
    return report, diagnostics


def detect_fleet(frames, shock_times, settings=Settings(), fleet_settings=FleetSettings()):
    """Run local variability search -> L1 comparison -> directions -> boundaries."""
    intervals, diagnostics, prepared = {}, {}, {}
    for craft, frame in frames.items():
        prepared[craft], shock = _prepare_record(frame, shock_times[craft], settings)
        intervals[craft], diagnostics[craft] = _variability_intervals(prepared[craft], shock, settings)
    decisions, comparison = select_l1_intervals(
        {craft: [row["start_utc"] for row in rows] for craft, rows in intervals.items()},
        fleet_settings,
    )
    reports = {}
    for craft, frame in frames.items():
        decision = decisions[craft]
        marker = decision["selected_marker_utc"]
        if marker is None:
            report = {"status": "unresolved_selection", "settings": asdict(settings),
                      "shock_utc": pd.Timestamp(shock_times[craft]), "flags": [decision["status"]]}
        else:
            report, diagnostics[craft] = _detect_prepared(
                prepared[craft], pd.Timestamp(shock_times[craft]), settings,
                intervals[craft], diagnostics[craft], marker, angular_transition_fit.Settings(),
            )
        report["variability_candidates"] = intervals[craft]
        report["selection"] = decision
        reports[craft] = report
    return reports, diagnostics, comparison
