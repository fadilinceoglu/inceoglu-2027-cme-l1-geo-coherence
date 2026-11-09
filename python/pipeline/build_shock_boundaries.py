"""Detect magnetic transitions, associate the event shock, and register its clock.

The complete four-day scan uses mission observations only. Broad observational
association windows identify the studied event among the detected transitions;
no old numerical products or exact shock times enter the detector. The nearest
suitable observer within L1 or ARTEMIS may guide a failed local measurement.
"""
from __future__ import annotations

from dataclasses import asdict
import gc
import json

import numpy as np
import pandas as pd

from analysis.shock_detection import Settings, coarse_candidates, refine_candidate, separated_indices
from config import study
from config.event_boundaries import SHOCK_ASSOCIATION_WINDOWS
from config.paths import PROCESSED_DIR, RESULTS_DIR
from analysis.geometry import reference_positions
from analysis.shock_crossing import measure_crossing
from analysis.product_io import json_ready

CRAFT = study.UPSTREAM_SPACECRAFT
NEARBY_GROUPS = (study.L1_SPACECRAFT, study.ARTEMIS_SPACECRAFT)
START, STOP = pd.Timestamp(study.ANALYSIS_START), pd.Timestamp(study.ANALYSIS_STOP)


def native_record(craft):
    """Read this run's exact screened native scalar observations; never decode again."""
    with np.load(PROCESSED_DIR / "shock_inputs" / f"{craft}.npz", allow_pickle=False) as source:
        series = pd.Series(source["bt"], index=pd.to_datetime(source["time_ns"], unit="ns", utc=True))
    if len(series) < 2 or series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValueError(f"{craft}: invalid current-run native shock record")
    cadence = float(np.median(np.diff(series.index.as_unit("ns").asi8) / 1e9))
    return series, cadence


def detect_candidates(craft, native, cadence, settings=Settings()):
    minute = native.resample("1min").mean().reindex(
        pd.date_range(START, STOP, freq="1min", inclusive="left"))
    coarse = coarse_candidates(minute, settings)
    coarse.insert(0, "spacecraft", craft)
    coarse["coarse_id"] = [f"{craft}_m{i:04d}" for i in range(1, len(coarse) + 1)]
    native_rows = []
    for i, row in coarse.iterrows():
        fits = refine_candidate(native, row.coarse_time, cadence, settings)
        coarse.loc[i, "native_fits_before_deduplication"] = len(fits)
        native_rows.extend({"spacecraft": craft, "coarse_id": row.coarse_id, **fit} for fit in fits)
        if (i + 1) % 200 == 0:
            print(f"  {craft}: refined {i + 1}/{len(coarse)} minute candidates", flush=True)
    candidates = pd.DataFrame(native_rows)
    if candidates.empty:
        return coarse, candidates
    priority = (candidates.native_strength.to_numpy()
                + candidates.passes_native_r2.to_numpy() * (candidates.native_strength.max() + 1))
    keep = separated_indices((candidates.native_time - START).dt.total_seconds().to_numpy(),
                             priority, settings.native_window_s / 2, candidates.direction.to_numpy())
    candidates = candidates.iloc[keep].copy().reset_index(drop=True)
    candidates["candidate_id"] = [f"{craft}_c{i:04d}" for i in range(1, len(candidates) + 1)]
    return coarse, candidates


def select_event_candidate(candidates, window):
    """Associate the principal rise after scanning; keep a failed fit visible.

    Rank rising candidates by fitted increase times R² within the declared
    event interval. Do not substitute a weaker unrelated rise merely because
    the principal candidate fails the native fit-quality criterion.
    """
    if candidates.empty:
        return None
    selected = candidates[(candidates.direction == "rise") & candidates.native_time.between(
        pd.Timestamp(window[0]), pd.Timestamp(window[1]))]
    if selected.empty:
        return None
    return selected.sort_values(["native_strength", "native_time"], ascending=[False, True]).iloc[0]


def positions_km():
    return {key: value * study.EARTH_RADIUS_KM for key, value in reference_positions().items()}


def ballistic_delay(source_position, target_position, speed):
    """Antisunward bulk-flow travel estimate used only to guide a local search."""
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("A finite positive measured speed is required")
    return float((source_position[0] - target_position[0]) / speed)


def suitable_neighbours(target, available):
    group = next((g for g in NEARBY_GROUPS if target in g), ())
    candidates = [source for source in group if source != target and source in available]
    if len(candidates) < 2:
        return candidates
    pos = positions_km()
    return sorted(candidates, key=lambda source: np.linalg.norm(pos[source] - pos[target]))


def direct_measurement(candidate, native):
    t50 = pd.Timestamp(candidate.native_time)
    halfwidth = float(candidate.ramp_25_75_s) / 2
    good = native.dropna()
    bracket = good.index.searchsorted(t50)
    return {
        "method": "native logistic fit", "candidate_id": candidate.candidate_id,
        "t25": t50 - pd.Timedelta(seconds=halfwidth), "t50": t50,
        "t75": t50 + pd.Timedelta(seconds=halfwidth),
        "unc_s": float(candidate.timing_halfwidth_s), "ramp_s": float(candidate.ramp_25_75_s),
        "b_up": float(candidate.before_nT), "b_dn": float(candidate.after_nT),
        "r2": float(candidate.native_r2), "tau_s": float(candidate.tau_s),
        "cadence_s": float(candidate.cadence_s),
        "fit_window_start": candidate.fit_window_start, "fit_window_end": candidate.fit_window_end,
        "gap_limited": bool(candidate.gap_limited),
        "gap_start": good.index[bracket - 1] if candidate.gap_limited else None,
        "gap_stop": good.index[bracket] if candidate.gap_limited else None,
    }


def nearby_crossing(target, direct):
    result = {"method": "unresolved", "reason": "No suitable direct detection and measured speed in the permitted group"}
    for source in suitable_neighbours(target, direct):
        # These post-shock speed products are unreliable in this event.
        if source in ("ace", "dscovr"):
            continue
        source_time = direct[source]["t50"]
        context = pd.read_csv(PROCESSED_DIR / "common_cadence_1min" / f"{source}.csv", index_col=0)
        context.index = pd.to_datetime(context.index, utc=True)
        speed = context.v.where(context.v_samples > 0)
        speed = speed.loc[source_time + pd.Timedelta(minutes=1):source_time + pd.Timedelta(minutes=10)].dropna()
        speed = speed[(speed >= 100) & (speed <= 3000)]
        if len(speed):
            break
    else:
        return result
    v = float(speed.median())
    pos = positions_km()
    predicted = source_time + pd.Timedelta(seconds=ballistic_delay(pos[source], pos[target], v))
    result.update(source_spacecraft=source, source_time=source_time, predicted_time=predicted,
                  speed_km_s=v, speed_minutes=len(speed), speed_first_utc=speed.index[0], speed_last_utc=speed.index[-1],
                  separation_km=float(np.linalg.norm(pos[source] - pos[target])),
                  projected_separation_km=float(pos[source][0] - pos[target][0]))
    try:
        native, cadence = native_record(target)
        result.update(measure_crossing(target, native, cadence, predicted))
        result.pop("reason", None)
    except (ValueError, RuntimeError, StopIteration, IndexError, KeyError) as exc:
        result["reason"] = f"Local crossing failed: {type(exc).__name__}: {exc}"
    return result


def build():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    all_coarse, all_native, direct, selected = [], [], {}, {}
    for craft in CRAFT:
        print(f"{craft}: independent magnetic candidate scan", flush=True)
        native, cadence = native_record(craft)
        coarse, candidates = detect_candidates(craft, native, cadence, settings)
        candidate = select_event_candidate(candidates, SHOCK_ASSOCIATION_WINDOWS[craft])
        if candidate is not None:
            selected[craft] = candidate.to_dict()
            if candidate.passes_native_r2:
                direct[craft] = direct_measurement(candidate, native)
        all_coarse.append(coarse)
        all_native.append(candidates)
        del native
        gc.collect()
    pd.concat(all_coarse, ignore_index=True).to_csv(RESULTS_DIR / "shock_candidates_1min.csv", index=False)
    pd.concat(all_native, ignore_index=True).to_csv(RESULTS_DIR / "shock_candidates_native.csv", index=False)
    measurements = {craft: direct[craft] if craft in direct else nearby_crossing(craft, direct) for craft in CRAFT}
    path = RESULTS_DIR / "boundaries.json"
    records = json.loads(path.read_text())
    for craft, measurement in measurements.items():
        measurement["association_window"] = SHOCK_ASSOCIATION_WINDOWS[craft]
        if craft in selected:
            measurement["associated_candidate_id"] = selected[craft]["candidate_id"]
            measurement["associated_candidate_r2"] = selected[craft]["native_r2"]
        if measurement["method"] != "unresolved":
            stamp = measurement["t50"].isoformat().replace("+00:00", "Z")
            records[craft].update(shock=stamp, shock_t50_utc=stamp, shock_unc_s=float(measurement["unc_s"]))
        print(f"{craft}: {measurement['method']} {measurement.get('t50', '')}", flush=True)
    for craft, measurement in measurements.items():
        records[craft]["shock_detection"] = json_ready(measurement)
        records[craft]["shock_basis"] = measurement["method"]
    measured = [craft for craft, result in measurements.items() if result["method"] != "unresolved"]
    records["_meta"].update(native_shocks_measured=measured, shock_status="complete" if len(measured) == len(CRAFT) else "partial")
    records["_meta"]["shock_detection_settings"] = asdict(settings)
    records["_meta"]["shock_association"] = "principal rising candidate (increase times R²) in each observed event interval, after independent full-record detection"
    path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    failed = [craft for craft, m in measurements.items() if m["method"] == "unresolved"]
    if failed:
        raise RuntimeError("Unresolved event shock: " + ", ".join(failed))
    return measurements


if __name__ == "__main__":
    build()
