"""Build the four canonical magnetic waveform families.

This stage measures the spatial cutoffs, then filters the same prepared records.
It overwrites the design from current boundaries, spectra, speeds and native
sampling metadata and saves their shared distance coordinates for the metrics.

All four families start from the prepared one-second magnetic grids, use a
0.04-R_E convected-distance grid, and are filtered before minute reduction.
The grids preserve the existing mission-specific aggregation and short holds;
they do not imply independent one-second observations for every instrument.

Outputs are ``PROCESSED_DIR/bands/<craft>__<family>.csv`` with arithmetic
minute means, within-minute sample standard deviations, finite filtered-grid
counts, and validity/interpolation flags. Spreads describe physical variability,
not measurement errors, standard errors, or confidence intervals. Filtering uses
the six-day context; the outputs retain the four-day scientific record.

Run from ``python/`` with ``PYTHONPATH=.``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from analysis.structure_bands import (
    EARTH_RADIUS_KM,
    SpatialBand,
    prepare_spatial_record,
    filter_prepared_spatial_record,
)
from config import study
from config.paths import PROCESSED_DIR
from config.waveform_bands import SPATIAL_DX_RE, FILTER_ORDER, WAVEFORM_FAMILY_LABELS
from analysis.spatial_coordinates import sample_coordinates, write_coordinate_record


PROCESSED: Final = PROCESSED_DIR
COMMON: Final = PROCESSED / "common_cadence_1min"
NATIVE: Final = PROCESSED / "native"
BANDS: Final = PROCESSED / "bands"

CRAFT = study.UPSTREAM_SPACECRAFT
MAGNETIC_COLUMNS: Final = ("bx", "by", "bz", "bt")

# Every waveform family is filtered from the same six-day real-data context.
# Only afterward is the result cropped to the same four scientific UTC days.
FILTER_CONTEXT_START: Final = pd.Timestamp(study.FILTER_CONTEXT_START)
FILTER_CONTEXT_STOP: Final = pd.Timestamp(study.FILTER_CONTEXT_STOP)
FILTER_OUTPUT_START: Final = pd.Timestamp(study.FILTER_RECORD_START)
FILTER_OUTPUT_STOP: Final = pd.Timestamp(study.FILTER_RECORD_STOP)
FILTER_CONTEXT_MINUTE_GRID: Final = pd.date_range(
    FILTER_CONTEXT_START,
    FILTER_CONTEXT_STOP,
    freq="1min",
    inclusive="left",
    name="time",
)
FILTER_OUTPUT_MINUTE_GRID: Final = pd.date_range(
    FILTER_OUTPUT_START,
    FILTER_OUTPUT_STOP,
    freq="1min",
    inclusive="left",
    name="time",
)
# All families use the prepared one-second magnetic grids. Keep the spatial
# step finer than every prepared-grid convected step in the six-day context,
# checked separately for each spacecraft before filtering.
MIN_FILTERED_GRID_SAMPLES_FOR_STD: Final = 2
FILTER_APPLICATION: Final = (
    "scipy_sosfiltfilt_zero_phase_in_convected_distance"
)
MAX_UNDECLARED_SPEED_GAP_MINUTES: Final = 30
APPROVED_LONG_SPEED_GAPS: Final = {
    "solar_orbiter": (
        pd.Timestamp("2026-01-19T04:23:00Z"),
        pd.Timestamp("2026-01-19T08:00:00Z"),
    ),
}


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["time"], index_col="time")
    frame.index = (
        frame.index.tz_localize("UTC")
        if frame.index.tz is None
        else frame.index.tz_convert("UTC")
    )
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError(f"time index is not sorted and unique: {path}")
    return frame


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _missing_intervals(series: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return half-open intervals containing non-positive/non-finite speeds."""

    bad = ~np.isfinite(series.to_numpy(dtype=float)) | (
        series.to_numpy(dtype=float) <= 0.0
    )
    edges = np.flatnonzero(np.diff(np.r_[False, bad, False]))
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, stop in edges.reshape(-1, 2):
        interval_start = pd.Timestamp(series.index[start])
        interval_stop = pd.Timestamp(series.index[stop - 1]) + pd.Timedelta(
            minutes=1
        )
        intervals.append((interval_start, interval_stop))
    return intervals


def _validate_speed_gap_topology(
    source: pd.Series,
    craft: str,
) -> dict:
    """Fail on an undeclared material gap before interpolation can hide it."""

    context = source.reindex(FILTER_CONTEXT_MINUTE_GRID)
    intervals = _missing_intervals(context)
    approved = APPROVED_LONG_SPEED_GAPS.get(craft)
    records = []
    for start, stop in intervals:
        duration = int((stop - start) / pd.Timedelta(minutes=1))
        is_approved_long_gap = approved == (start, stop)
        if duration > MAX_UNDECLARED_SPEED_GAP_MINUTES and not is_approved_long_gap:
            raise ValueError(
                f"{craft} frozen speed has undeclared {duration}-minute gap "
                f"[{start.isoformat()}, {stop.isoformat()}); update the "
                "documented speed-gap policy before filtering"
            )
        records.append(
            {
                "start_utc": start.isoformat().replace("+00:00", "Z"),
                "stop_utc_exclusive": stop.isoformat().replace(
                    "+00:00", "Z"
                ),
                "duration_minutes": duration,
                "approved_material_gap": is_approved_long_gap,
            }
        )
    return {
        "interpolation": "piecewise linear in UTC; nearest endpoint at context edges",
        "maximum_undeclared_gap_minutes": MAX_UNDECLARED_SPEED_GAP_MINUTES,
        "missing_intervals": records,
        "missing_minute_count": int(
            sum(record["duration_minutes"] for record in records)
        ),
    }


def _speed_on(
    index: pd.DatetimeIndex,
    craft: str,
) -> tuple[pd.Series, Path, dict]:
    path = PROCESSED / f"{craft}_vt_frozen.csv"
    source = _read_csv(path)["v_frozen"]
    gap_policy = _validate_speed_gap_topology(source, craft)
    speed = source.reindex(index).interpolate(
        method="time",
        limit_direction="both",
    )
    return speed, path, gap_policy


def _nondecimating_grid_diagnostics(
    index: pd.DatetimeIndex,
    speed: pd.Series,
) -> dict:
    """Verify interpolation never decimates the prepared one-second grid."""

    elapsed_s = np.diff(index.asi8.astype(np.float64)) / 1.0e9
    values = speed.to_numpy(dtype=float)
    source_steps_re = (
        0.5 * (values[:-1] + values[1:]) * elapsed_s / EARTH_RADIUS_KM
    )
    if not np.isfinite(source_steps_re).all() or np.any(source_steps_re <= 0.0):
        raise ValueError("native speed coordinate contains an invalid source step")
    minimum = float(np.min(source_steps_re))
    maximum = float(np.max(source_steps_re))
    if SPATIAL_DX_RE > minimum + 1.0e-12:
        raise ValueError(
            "spatial grid would decimate the prepared one-second "
            f"coordinate: dx={SPATIAL_DX_RE:.6g} R_E exceeds the "
            f"minimum source step {minimum:.6g} R_E"
        )
    return {
        "policy": (
            "uniform spatial interpolation is non-decimating: dx_re is no "
            "larger than the minimum prepared-grid convected step"
        ),
        "minimum_source_step_re": minimum,
        "maximum_source_step_re": maximum,
        "internal_grid_step_re": SPATIAL_DX_RE,
        "explicit_predecimation_antialias_filter_required": False,
    }


def _native_segment(
    craft: str,
) -> tuple[pd.DataFrame, pd.Series, Path, Path, dict]:
    source_path = NATIVE / f"{craft}_1s.csv"
    frame = _read_csv(source_path)
    frame = frame.loc[
        (frame.index >= FILTER_CONTEXT_START)
        & (frame.index < FILTER_CONTEXT_STOP)
    ]
    speed, speed_path, speed_gap_policy = _speed_on(frame.index, craft)
    finite = (
        frame["bt"].notna().to_numpy()
        & np.isfinite(speed.to_numpy(dtype=float))
        & (speed.to_numpy(dtype=float) > 0.0)
    )
    if int(finite.sum()) < 20:
        raise ValueError(f"{craft} has too little prepared magnetic support")
    return frame, speed, source_path, speed_path, speed_gap_policy


def _gap_tolerance_re(minimum_re: float) -> float:
    return max(25.0, 0.25 * minimum_re)


def _wind_interpolation_flags(index: pd.DatetimeIndex, craft: str) -> pd.Series:
    """Return the already-approved Wind one-minute interpolation flags."""

    if craft != "wind":
        return pd.Series(False, index=index, dtype=bool)
    source = _read_csv(COMMON / "wind_gapfilled.csv")
    return (
        source["mag_interpolated"]
        .astype(bool)
        .reindex(index, fill_value=False)
    )


@dataclass(frozen=True)
class PreparedWaveform:
    frame: pd.DataFrame
    speed: pd.Series
    source_path: Path
    speed_path: Path
    speed_gap_policy: dict
    spatial_record: tuple[np.ndarray, np.ndarray, np.ndarray]
    grid_diagnostics: dict


def _prepare_waveform(craft: str) -> PreparedWaveform:
    frame, speed, source, speed_path, gap_policy = _native_segment(craft)
    diagnostics = _nondecimating_grid_diagnostics(frame.index, speed)
    spatial = prepare_spatial_record(
        frame.index.values, frame.loc[:, MAGNETIC_COLUMNS].to_numpy(dtype=float),
        speed.to_numpy(dtype=float), dx_re=SPATIAL_DX_RE,
    )
    return PreparedWaveform(frame, speed, source, speed_path, gap_policy, spatial, diagnostics)


def build_band(
    craft: str,
    definition: dict,
    prepared: PreparedWaveform,
    *,
    family: str,
    display_name: str,
) -> list[Path]:
    """Filter one shared spacecraft preparation, then form minute statistics."""
    frame = prepared.frame
    source_path, speed_path = prepared.source_path, prepared.speed_path
    speed_gap_policy = prepared.speed_gap_policy
    spatial_grid_diagnostics = prepared.grid_diagnostics
    maximum_re = definition["max_re"]
    band = SpatialBand(
        display_name,
        float(definition["min_re"]),
        None if maximum_re is None else float(maximum_re),
    )
    gap_tolerance = _gap_tolerance_re(band.minimum_re)
    outputs: list[Path] = []
    filtered, _x_grid, _filtered_x, valid, _valid_x = filter_prepared_spatial_record(
        frame.loc[:, MAGNETIC_COLUMNS].to_numpy(dtype=float), band,
        prepared.spatial_record, dx_re=SPATIAL_DX_RE,
        order=FILTER_ORDER, max_gap_re=gap_tolerance,
    )
    filtered_seconds = pd.DataFrame(
        np.where(valid, filtered, np.nan),
        index=frame.index,
        columns=[f"{column}_f" for column in MAGNETIC_COLUMNS],
    )
    grouped = filtered_seconds.resample(
        "1min",
        label="left",
        closed="left",
    )
    minute_mean = grouped.mean().reindex(FILTER_CONTEXT_MINUTE_GRID)
    minute_std = grouped.std(ddof=1).reindex(FILTER_CONTEXT_MINUTE_GRID)
    minute_count = grouped.count().reindex(
        FILTER_CONTEXT_MINUTE_GRID, fill_value=0
    )
    output = pd.DataFrame(index=minute_mean.index)
    for column in MAGNETIC_COLUMNS:
        filtered_column = f"{column}_f"
        output[filtered_column] = minute_mean[filtered_column]
        output[f"{column}_std_f"] = minute_std[filtered_column]
        output[f"{column}_filtered_grid_samples"] = minute_count[
            filtered_column
        ].astype("int64")
    counts = minute_count.to_numpy(dtype=int)
    observed_valid = pd.Series(
        (counts >= 1).all(axis=1),
        index=output.index,
    )
    observed_std_valid = observed_valid & np.isfinite(
        minute_std.to_numpy(dtype=float)
    ).all(axis=1) & (
        counts >= MIN_FILTERED_GRID_SAMPLES_FOR_STD
    ).all(axis=1)
    interpolation_flags = _wind_interpolation_flags(output.index, craft)
    fill_columns = [
        column
        for column in output.columns
        if column.endswith("_f")
    ]
    if interpolation_flags.any():
        estimates = output.loc[:, fill_columns].interpolate(
            method="time",
            limit_area="inside",
        )
        output.loc[interpolation_flags, fill_columns] = estimates.loc[
            interpolation_flags,
            fill_columns,
        ]
    filled_finite = np.isfinite(
        output.loc[:, fill_columns].to_numpy(dtype=float)
    ).all(axis=1)
    output["observed_valid"] = observed_valid.to_numpy(dtype=bool)
    output["valid"] = observed_valid.to_numpy(dtype=bool) | (
        interpolation_flags.to_numpy(dtype=bool) & filled_finite
    )
    output["std_valid"] = observed_std_valid.to_numpy(dtype=bool) | (
        interpolation_flags.to_numpy(dtype=bool) & filled_finite
    )
    output["source_interpolated"] = interpolation_flags.to_numpy(dtype=bool)
    context_interpolation_flags = interpolation_flags.copy()
    output = output.reindex(FILTER_OUTPUT_MINUTE_GRID)
    interpolation_flags = context_interpolation_flags.reindex(
        output.index, fill_value=False
    )

    path = BANDS / f"{craft}__{family}.csv"
    output.to_csv(
        path,
        index_label="time",
        date_format="%Y-%m-%dT%H:%M:%SZ",
        float_format="%.12g",
        na_rep="",
        lineterminator="\n",
    )
    metadata = {
        "canonical_family": family,
        "band_design_source": str(PROCESSED / "waveform_bands.json"),
        "display_name": display_name,
        "passband_re": [band.minimum_re, band.maximum_re],
        "passband_definition": {
            "lower_edge_basis": definition.get(
                "lower_edge_basis", "current-run measured boundary"
            ),
            "upper_edge_basis": definition.get(
                "upper_edge_basis",
                "mean-retaining low-pass" if maximum_re is None
                else "current-run measured boundary",
            ),
        },
        "filter_input_cadence": "prepared magnetic measurements on a 1-s grid",
        "output_cadence": "one-minute UTC bins",
        "input_source": str(source_path),
        "speed_source": str(speed_path),
        "speed_gap_policy": speed_gap_policy,
        "filter_application": FILTER_APPLICATION,
        "butterworth_order": FILTER_ORDER,
        "dx_re": SPATIAL_DX_RE,
        "spatial_resampling": spatial_grid_diagnostics,
        "gap_tolerance_re": gap_tolerance,
        "filter_input_window_utc": [
            FILTER_CONTEXT_START.isoformat().replace("+00:00", "Z"),
            FILTER_CONTEXT_STOP.isoformat().replace("+00:00", "Z"),
        ],
        "filter_input_stop_is_exclusive": True,
        "output_window_utc": [
            FILTER_OUTPUT_START.isoformat().replace("+00:00", "Z"),
            FILTER_OUTPUT_STOP.isoformat().replace("+00:00", "Z"),
        ],
        "output_stop_is_exclusive": True,
        "one_minute_reduction": (
            "after filtering the prepared one-second grid, arithmetic mean, "
            "sample "
            "standard deviation (ddof=1), and finite filtered-grid count "
            "are calculated in each left-closed UTC minute"
        ),
        "filtered_grid_count_interpretation": (
            "finite valid values on the prepared one-second grid; "
            "not a count of independent instrument observations"
        ),
        "processing_order": [
            "prepared_one_second_magnetic_grid",
            "trapezoidal_speed_integration_to_distance",
            "linear_interpolation_to_uniform_spatial_grid",
            "butterworth_forward_backward_filter",
            "linear_interpolation_back_to_one_second_times",
            "restore_missing_data_and_filter_validity_masks",
            "one_minute_mean_sample_std_and_grid_count",
            "approved_flagged_wind_minute_interpolation",
            "crop_to_four_day_record",
        ],
        "minimum_filtered_grid_samples_for_mean": 1,
        "minimum_filtered_grid_samples_for_std": (
            MIN_FILTERED_GRID_SAMPLES_FOR_STD
        ),
        "spread_interpretation": (
            "within-minute variability of the filtered one-second waveform; "
            "not standard error, measurement uncertainty, or a "
            "confidence interval"
        ),
        "post_filter_gap_fill": {
            "method": (
                "linear interpolation in UTC of the one-minute filtered "
                "mean and within-minute spread"
            ),
            "source_flag": "common_cadence_1min/wind_gapfilled.csv:mag_interpolated",
            "filled_minute_count": int(interpolation_flags.sum()),
            "filter_context_filled_minute_count": int(
                context_interpolation_flags.sum()
            ),
            "filled_minutes_utc": [
                timestamp.isoformat().replace("+00:00", "Z")
                for timestamp in output.index[interpolation_flags]
            ],
            "filtered_grid_counts_remain_zero": True,
        },
        "valid_fraction": float(output["valid"].mean()),
        "created_by": "pipeline/build_spatial_filters.py",
    }
    metadata_path = path.with_name(path.stem + "_meta.json")
    _write_json(metadata_path, metadata)
    outputs.extend((path, metadata_path))
    print(
        f"{craft:14s} {family:14s} {len(output):6d} rows  "
        f"valid {output['valid'].mean():.3f}",
        flush=True,
    )
    return outputs


def build() -> list[Path]:
    """Measure scales and filter each spacecraft with one shared preparation."""

    from pipeline.build_waveform_bands import build as build_scales

    BANDS.mkdir(parents=True, exist_ok=True)
    prepared_l1 = {craft: _prepare_waveform(craft) for craft in study.L1_SPACECRAFT}
    design = build_scales(prepared_l1)
    design_path = PROCESSED / "waveform_bands.json"
    definitions = design["bands"]
    fingerprints = {entry["path"]: entry["sha256"] for entry in design["inputs"]}
    boundary_path = PROCESSED.parent / "results" / "boundaries.json"
    boundaries = json.loads(boundary_path.read_text())
    coordinate_manifest = {
        "schema_version": "shared-spatial-coordinates-v1",
        "spatial_grid_step_re": SPATIAL_DX_RE,
        "boundary_sha256": fingerprints["results/boundaries.json"],
        "crafts": {},
    }
    names = {
        family: "mean-retaining trend" if family == "trend" else f"{label.lower()} residual"
        for family, label in WAVEFORM_FAMILY_LABELS.items()
    }
    outputs = [design_path, PROCESSED / "waveform_scales.csv"]
    for craft in CRAFT:
        prepared = prepared_l1.pop(craft) if craft in prepared_l1 else _prepare_waveform(craft)
        coordinate = sample_coordinates(
            prepared.frame.index, prepared.spatial_record[0], boundaries[craft]["shock"],
        )
        coordinate_path, provenance = write_coordinate_record(
            PROCESSED, craft, coordinate, boundaries[craft]["shock"],
            prepared.source_path, prepared.speed_path, fingerprints=fingerprints,
        )
        coordinate_manifest["crafts"][craft] = provenance
        outputs.append(coordinate_path)
        for family, definition in definitions.items():
            outputs.extend(build_band(
                craft, definition, prepared, family=family, display_name=names[family],
            ))
    coordinate_path = PROCESSED / "spatial_coordinates.json"
    _write_json(coordinate_path, coordinate_manifest)
    outputs.append(coordinate_path)
    return outputs


def main() -> None:
    outputs = build()
    print(f"wrote {len(outputs)} waveform data/provenance files", flush=True)


if __name__ == "__main__":
    main()
