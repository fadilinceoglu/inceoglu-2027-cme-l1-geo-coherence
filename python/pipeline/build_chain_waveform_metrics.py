"""Build the deterministic seven-spacecraft Figure-4 metric product.

All four families use each spacecraft's half-open interval from the current
run's band-energy interval product (pipeline/build_waveform_intervals.py).
Each observer's frozen speed maps the already selected UTC
rows to distance about its local native-cadence shock-t50 coordinate anchor.
That anchor does not select the interval.  The builder does not align by
correlation, crop by a physical event boundary or L1 distance union, fit
amplitudes, or compare spacecraft pairs.

The own/proxy/hybrid speed provenance is recorded with the product.
Missing frozen-speed samples use bounded piecewise-linear interpolation, with
previous- and next-value holds saved as coordinate-definition sensitivity.

Outputs
-------
<output-dir>/results/chain_waveform_metrics.csv
<output-dir>/results/chain_waveform_metrics.json

Run
---
cd python
PYTHONPATH=. python pipeline/build_chain_waveform_metrics.py
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from analysis.waveform_scales import load_band_design
from analysis.chain_waveform_product import (
    SCHEMA_VERSION, INTERVALS_SCHEMA_VERSION, SPEED_GAP_FILL_POLICY, ANCHOR_KIND, OBSERVATION_ROLE, CSV_COLUMNS, _comparison_status, _support_selection, _is_null, _validate_rows,
)
from analysis.product_io import utc_text, json_ready
from common.io import display_path, read_json, read_json_object as _read_json
from config.paths import RESULTS_DIR, PYTHON_ROOT, PROCESSED_DIR
from analysis.chain_waveform_metrics import (
    CRAFT_ORDER,
    DEFAULT_MAX_MATERIAL_GAP_RE,
    BOUNDARIES_FILENAME,
    STRUCTURES,
    RegisteredWaveform,
    WaveformMetrics,
    compute_waveform_metrics,
    load_chain_waveform,
    load_frame_anchors,
    read_frozen_speed,
    register_waveform_utc,
    signed_distances_re,
)
from analysis.spatial_coordinates import coordinate_time_at, read_coordinates
PROCESSED: Final = PROCESSED_DIR
OUTPUT_CSV: Final = RESULTS_DIR / "chain_waveform_metrics.csv"
OUTPUT_JSON: Final = RESULTS_DIR / "chain_waveform_metrics.json"
WAVEFORM_INTERVALS: Final = RESULTS_DIR / "waveform_intervals.json"
SPEED_PROVENANCE: Final = PROCESSED / "vt_provenance.json"


METRIC_UNITS: Final = {
    "power_support_width_re": "R_E",
    "scalar_integrated_power_nT2_re": "nT^2 R_E",
    "scalar_bt_rms_nT": "nT",
    "vector_rms_nT": "nT",
    "scalar_to_vector_variance_ratio": "dimensionless",
}

RATIO_CHANGE_METRICS: Final = frozenset(
    {
        "power_support_width_re",
        "scalar_integrated_power_nT2_re",
        "scalar_bt_rms_nT",
        "vector_rms_nT",
        "scalar_to_vector_variance_ratio",
    }
)

# The order is part of the machine-readable CSV contract.


_display_path = partial(display_path, root=PYTHON_ROOT)




def _speed_coordinate_diagnostics(
    speed: pd.Series,
    waveform: RegisteredWaveform,
    *,
    provenance_description: str,
) -> dict[str, Any]:
    """Audit frozen-speed gaps and bracket their coordinate effect.

    The production coordinate linearly interpolates between the nearest
    positive finite frozen-speed samples.  Previous-value and next-value
    holds across every missing run are retained as a definition-sensitivity
    bracket; they are not statistical confidence limits.
    """

    source = speed.copy().sort_index()
    if not source.index.is_unique:
        raise ValueError("frozen speed must have unique timestamps")
    positive = np.isfinite(source.to_numpy(dtype=float)) & (
        source.to_numpy(dtype=float) > 0.0
    )
    finite_source = source.where(positive)
    support = finite_source.reindex(waveform.time)
    missing = support.isna().to_numpy(dtype=bool)
    changes = np.diff(np.r_[False, missing, False].astype(int))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    intervals: list[dict[str, Any]] = []
    for start, stop in zip(starts, stops):
        stop_utc = (
            waveform.time[stop]
            if stop < len(waveform.time)
            else waveform.time[-1] + pd.Timedelta(minutes=1)
        )
        intervals.append(
            {
                "start_utc": utc_text(waveform.time[start]),
                "stop_utc_exclusive": utc_text(stop_utc),
                "duration_minutes": int(stop - start),
            }
        )

    previous_hold = finite_source.ffill().bfill()
    next_hold = finite_source.bfill().ffill()
    if previous_hold.isna().any() or next_hold.isna().any():
        raise ValueError("frozen-speed hold sensitivity could not fill support")
    production = np.asarray(waveform.x_re, dtype=float)
    linear_reference = signed_distances_re(source, waveform.anchor_utc, waveform.time)
    # Hold-policy deltas perturb the shared production mapping. Subtracting the
    # matching linear-speed integration isolates gaps from coordinate-origin
    # quadrature differences and gives exactly zero change without a gap.
    coordinates = [
        production,
        production + (signed_distances_re(
            previous_hold,
            waveform.anchor_utc,
            waveform.time,
        ) - linear_reference),
        production + (signed_distances_re(
            next_hold,
            waveform.anchor_utc,
            waveform.time,
        ) - linear_reference),
    ]
    max_deviation = float(
        max(
            np.max(np.abs(candidate - production))
            for candidate in coordinates
        )
    )
    spans = np.asarray(
        [candidate[-1] - candidate[0] for candidate in coordinates],
        dtype=float,
    )
    production_span = float(production[-1] - production[0])
    if production_span <= 0.0:
        raise ValueError("production speed coordinate has a non-positive span")
    return {
        "frozen_speed_provenance_description": provenance_description,
        "frozen_speed_gap_fill_policy": SPEED_GAP_FILL_POLICY,
        "frozen_speed_support_expected_row_count": int(len(support)),
        "frozen_speed_support_finite_positive_row_count": int(
            (~missing).sum()
        ),
        "frozen_speed_support_missing_row_count": int(missing.sum()),
        "frozen_speed_support_gap_count": int(len(intervals)),
        "frozen_speed_support_max_missing_run_minutes": int(
            max((item["duration_minutes"] for item in intervals), default=0)
        ),
        "frozen_speed_support_missing_intervals_json": json.dumps(
            intervals,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "frozen_speed_coordinate_hold_sensitivity_max_abs_re": max_deviation,
        "frozen_speed_coordinate_hold_sensitivity_over_span_percent": (
            100.0 * max_deviation / production_span
        ),
        "frozen_speed_support_span_hold_sensitivity_min_re": float(
            spans.min()
        ),
        "frozen_speed_support_span_hold_sensitivity_max_re": float(
            spans.max()
        ),
    }


def _waveform_metadata(record_path: Path) -> tuple[Path, dict[str, Any]]:
    path = record_path.with_name(record_path.stem + "_meta.json")
    if not path.exists():
        raise FileNotFoundError(f"missing waveform metadata: {path}")
    payload = read_json(path, encoding="utf-8")
    passband = payload.get("passband_re")
    if not isinstance(passband, list) or len(passband) != 2:
        raise ValueError(f"{path.name} lacks a two-edge passband_re")
    return path, payload


def _read_intervals(path: Path) -> tuple[Path, dict[str, Any]]:
    payload = read_json(path, encoding="utf-8")
    if payload.get("schema_version") != INTERVALS_SCHEMA_VERSION:
        raise ValueError("waveform interval schema is not supported")
    if payload.get("spacecraft_order") != list(CRAFT_ORDER):
        raise ValueError("waveform interval spacecraft order drifted")
    if payload.get("family_order") != list(STRUCTURES):
        raise ValueError("waveform interval family order drifted")
    return path, payload


def _read_speed_provenance(processed: Path) -> dict[str, Any]:
    """Load the speed-source provenance beside the current-run products."""

    path = processed / SPEED_PROVENANCE.name
    payload = _read_json(path)
    missing = set(CRAFT_ORDER).difference(payload)
    if missing:
        raise ValueError(
            "speed provenance lacks: " + ", ".join(sorted(missing))
        )
    return payload


def _base_row(
    waveform: RegisteredWaveform,
    speed_path: Path,
    support_selection: dict[str, Any],
) -> dict[str, Any]:
    record = waveform.record
    status = _comparison_status(record.craft)
    metadata_path, metadata = _waveform_metadata(record.source_path)
    if metadata.get("canonical_family") != record.structure:
        raise ValueError(
            f"{metadata_path.name} canonical_family does not match "
            f"{record.structure}"
        )
    passband = metadata["passband_re"]
    valid = np.asarray(waveform.valid, dtype=bool)
    invalid_time = waveform.time[~valid]
    return {
        "schema_version": SCHEMA_VERSION,
        "structure": record.structure,
        "craft": record.craft,
        "sequence_position": CRAFT_ORDER.index(record.craft),
        "observation_role": OBSERVATION_ROLE[record.craft],
        "frame": record.frame,
        "comparison_status": status,
        "eligible": True,
        "measurement_available": True,
        "flags": "",
        "waveform_source_slug": record.source_slug,
        "waveform_source_path": _display_path(record.source_path),
        "waveform_metadata_path": _display_path(metadata_path),
        "effective_support_policy": support_selection["policy"],
        "effective_support_status": support_selection["status"],
        "effective_support_start_utc": utc_text(
            support_selection["start_utc"]
        ),
        "effective_support_stop_utc_exclusive": utc_text(
            support_selection["stop_utc_exclusive"]
        ),
        "effective_support_duration_minutes": support_selection[
            "duration_minutes"
        ],
        "filter_minimum_re": float(passband[0]),
        "filter_maximum_re": (
            None if passband[1] is None else float(passband[1])
        ),
        "frozen_speed_source_path": _display_path(speed_path),
        "anchor_kind": ANCHOR_KIND[record.structure],
        "anchor_utc": utc_text(waveform.anchor_utc),
        "registered_support_start_re": waveform.support.start_re,
        "registered_support_stop_re": waveform.support.stop_re,
        "registered_support_span_re": waveform.support.span_re,
        "registered_window_start_utc": utc_text(
            waveform.window_start_utc
        ),
        "registered_window_last_included_utc": utc_text(
            waveform.window_stop_utc
        ),
        "registered_window_stop_utc_exclusive": utc_text(
            support_selection["stop_utc_exclusive"]
        ),
        "registered_stored_row_count": int(len(waveform.time)),
        "registered_valid_row_count": int(valid.sum()),
        "registered_invalid_row_count": int((~valid).sum()),
        "registered_first_invalid_utc": (
            utc_text(invalid_time[0]) if len(invalid_time) else None
        ),
        "registered_last_invalid_utc": (
            utc_text(invalid_time[-1]) if len(invalid_time) else None
        ),
        "stored_cadence_s": record.stored_cadence_s,
        "effective_cadence_s": record.effective_cadence_s,
    }


def _measurement_payload(
    metrics: WaveformMetrics,
    coordinate: pd.Series,
) -> dict[str, Any]:
    scalar = metrics.scalar_power_support
    return {
        **_metric_snapshot(metrics),
        "valid_spatial_length_re": metrics.valid_spatial_length_re,
        "spatial_coverage_fraction": metrics.spatial_coverage_fraction,
        "source_interpolated_row_count": metrics.source_interpolated_row_count,
        "power_support_start_re": scalar.x05_re,
        "power_support_start_utc": utc_text(coordinate_time_at(coordinate, scalar.x05_re)),
        "power_support_stop_re": scalar.x95_re,
        "power_support_stop_utc": utc_text(coordinate_time_at(coordinate, scalar.x95_re)),
        "scalar_power_support_crosses_material_gap": scalar.crosses_material_gap,
    }


def _flags_for_row(
    craft: str,
    structure: str,
    metrics: WaveformMetrics,
    speed_missing_row_count: int,
) -> str:
    flags: list[str] = []
    if craft == "solar_orbiter":
        flags.append("flank_context")
    if craft == "artemis_p2":
        flags.append("lunar_wake_context")
    if structure == "trend":
        flags.append("mean_retaining_lowpass_not_isolated_ejecta_residual")
    if speed_missing_row_count:
        flags.append("frozen_speed_gap_piecewise_linearly_interpolated")
    if metrics.source_interpolated_row_count:
        flags.append("contains_approved_linearly_interpolated_minutes")
    if metrics.scalar_power_support.crosses_material_gap:
        flags.append("scalar_power_support_crosses_material_gap")
    return "|".join(dict.fromkeys(flags))


def _metric_snapshot(metrics: WaveformMetrics) -> dict[str, float]:
    scalar = metrics.scalar_power_support
    return {
        "power_support_width_re": scalar.width90_re,
        "scalar_integrated_power_nT2_re": scalar.integrated_power_nt2_re,
        "scalar_bt_rms_nT": metrics.scalar_bt_rms_nt,
        "vector_rms_nT": metrics.vector_rms_nt,
        "scalar_to_vector_variance_ratio": metrics.scalar_to_vector_variance_ratio,
    }


def _interpolation_excluded_sensitivity(
    waveform: RegisteredWaveform,
) -> dict[str, Any] | None:
    interpolated = np.asarray(waveform.source_interpolated, dtype=bool)
    if not interpolated.any():
        return None
    gap_threshold = DEFAULT_MAX_MATERIAL_GAP_RE
    retained = ~interpolated
    sensitivity_waveform = replace(
        waveform,
        valid=np.asarray(waveform.valid, dtype=bool) & retained,
        source_interpolated=np.zeros(len(waveform.time), dtype=bool),
    )
    metrics = compute_waveform_metrics(
        sensitivity_waveform,
        max_material_gap_re=gap_threshold,
        bridge_invalid_rows=False,
    )
    return {
        "structure": waveform.record.structure,
        "craft": waveform.record.craft,
        "excluded_minutes_utc": [
            utc_text(timestamp) for timestamp in waveform.time[interpolated]
        ],
        "excluded_minute_count": int(interpolated.sum()),
        "gap_connection_threshold_re": gap_threshold,
        "policy": (
            "drop every flagged interpolated minute and preserve stored-row "
            "topology so an invalid row disconnects its adjacent integration "
            "segments instead of being silently bridged"
        ),
        "metrics": _metric_snapshot(metrics),
    }


def _build_rows(
    processed: Path,
    intervals: dict[str, Any],
    coordinates: dict[str, pd.Series],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rows: list[dict[str, Any]] = []
    interpolation_sensitivity_records: list[dict[str, Any]] = []
    speed_provenance = _read_speed_provenance(processed)
    anchors = load_frame_anchors(processed.parent / "results" / BOUNDARIES_FILENAME)
    speeds = {craft: read_frozen_speed(processed, craft) for craft in CRAFT_ORDER}
    for structure in STRUCTURES:
        for craft in CRAFT_ORDER:
            record = load_chain_waveform(
                craft,
                structure,
                processed_directory=processed,
            )
            speed_path = processed / f"{craft}_vt_frozen.csv"
            support_selection = _support_selection(
                intervals,
                craft=craft,
                structure=structure,
            )
            speed = speeds[craft]
            waveform = register_waveform_utc(
                record,
                start_utc=support_selection["start_utc"],
                stop_utc_exclusive=support_selection[
                    "stop_utc_exclusive"
                ],
                processed_directory=processed,
                anchor_utc=anchors[craft]["shock_t50_utc"],
                coordinate=coordinates[craft],
            )
            row = _base_row(waveform, speed_path, support_selection)
            metrics = compute_waveform_metrics(waveform)
            row.update(
                _speed_coordinate_diagnostics(
                    speed,
                    waveform,
                    provenance_description=str(speed_provenance[craft]),
                )
            )
            row.update(
                _measurement_payload(metrics, coordinates[craft])
            )
            row.update(
                {
                    "flags": _flags_for_row(
                        craft,
                        structure,
                        metrics,
                        int(row["frozen_speed_support_missing_row_count"]),
                    ),
                }
            )
            interpolation_sensitivity = _interpolation_excluded_sensitivity(waveform)
            if interpolation_sensitivity is not None:
                differences: dict[str, Any] = {}
                for metric, sensitivity_value in interpolation_sensitivity[
                    "metrics"
                ].items():
                    primary_value = row.get(metric)
                    if _is_null(primary_value) or _is_null(sensitivity_value):
                        difference = ratio = percent = None
                    else:
                        primary_float = float(primary_value)
                        sensitivity_float = float(sensitivity_value)
                        difference = sensitivity_float - primary_float
                        if (
                            metric in RATIO_CHANGE_METRICS
                            and primary_float != 0.0
                        ):
                            ratio = sensitivity_float / primary_float
                            percent = 100.0 * (ratio - 1.0)
                        else:
                            ratio = percent = None
                    differences[metric] = {
                        "primary_value": primary_value,
                        "interpolation_excluded_value": sensitivity_value,
                        "difference_excluded_minus_primary": difference,
                        "ratio_excluded_over_primary": ratio,
                        "percent_change": percent,
                    }
                interpolation_sensitivity[
                    "comparison_with_primary"
                ] = differences
                interpolation_sensitivity_records.append(
                    interpolation_sensitivity
                )
            rows.append({column: row.get(column) for column in CSV_COLUMNS})
    return (
        rows,
        interpolation_sensitivity_records,
    )


def _input_manifest(
    processed: Path,
    rows: list[dict[str, Any]],
    intervals_path: Path,
) -> list[dict[str, Any]]:
    """Collect each exact current input once, with all of its declared roles."""

    roles: dict[str, set[str]] = {}

    def add(path: Path, role: str) -> None:
        shown = _display_path(path)
        roles.setdefault(shown, set()).add(role)

    add(
        processed.parent / "results" / BOUNDARIES_FILENAME,
        "measured_shock_anchor",
    )
    add(
        PYTHON_ROOT / "config" / "study.py",
        "spacecraft_registry",
    )
    add(
        intervals_path,
        "computed_spacecraft_by_family_band_energy_intervals",
    )
    add(
        processed / SPEED_PROVENANCE.name,
        "frozen_speed_proxy_hybrid_and_gap_policy_provenance",
    )
    add(
        PYTHON_ROOT / "analysis" / "waveform_intervals.py",
        "band_energy_interval_implementation",
    )
    add(
        PYTHON_ROOT / "pipeline" / "build_waveform_intervals.py",
        "band_energy_interval_builder",
    )
    add(
        Path(__file__).resolve().parents[1]
        / "analysis"
        / "chain_waveform_metrics.py",
        "metric_and_registration_implementation",
    )
    add(Path(__file__).resolve(), "metric_product_builder_implementation")
    add(processed / "waveform_bands.json", "current_run_measured_filter_scales")
    add(processed / "spatial_coordinates.json", "shared_spatial_coordinate_provenance")
    add(PYTHON_ROOT / "analysis" / "spatial_coordinates.py", "shared_coordinate_interface")
    for craft in CRAFT_ORDER:
        add(processed / "spatial_coordinates" / f"{craft}.csv", "shared_filter_and_metric_coordinate")
    add(PYTHON_ROOT / "analysis" / "waveform_scales.py", "reference_scale_implementation")
    for row in rows:
        waveform = Path(str(row["waveform_source_path"]))
        if not waveform.is_absolute():
            waveform = PYTHON_ROOT / waveform
        add(waveform, "saved_figure3_filtered_waveform")
        metadata = Path(str(row["waveform_metadata_path"]))
        if not metadata.is_absolute():
            metadata = PYTHON_ROOT / metadata
        add(metadata, "saved_waveform_filter_provenance")
        speed = Path(str(row["frozen_speed_source_path"]))
        if not speed.is_absolute():
            speed = PYTHON_ROOT / speed
        add(speed, "observer_frozen_speed_coordinate")
    return [
        {
            "path": path,
            "roles": sorted(roles[path]),
        }
        for path in sorted(roles)
    ]


def _registration_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Serialize all 28 independently registered Figure-4 supports."""

    supports: dict[str, dict[str, dict[str, Any]]] = {}
    for structure in STRUCTURES:
        supports[structure] = {}
        for craft in CRAFT_ORDER:
            row = next(
                item
                for item in rows
                if item["structure"] == structure and item["craft"] == craft
            )
            supports[structure][craft] = {
                "support_policy": row["effective_support_policy"],
                "support_status": row["effective_support_status"],
                "start_utc": row["effective_support_start_utc"],
                "last_included_utc": row[
                    "registered_window_last_included_utc"
                ],
                "stop_utc_exclusive": row[
                    "effective_support_stop_utc_exclusive"
                ],
                "duration_minutes": int(
                    row["effective_support_duration_minutes"]
                ),
                "stored_row_count": int(row["registered_stored_row_count"]),
                "valid_row_count": int(row["registered_valid_row_count"]),
                "distance_start_re": float(
                    row["registered_support_start_re"]
                ),
                "distance_stop_re": float(
                    row["registered_support_stop_re"]
                ),
                "distance_span_re": float(
                    row["registered_support_span_re"]
                ),
            }
    return {
        "cadence_seconds": 60.0,
        "record_count": len(rows),
        "fully_valid_record_count": sum(
            int(row["registered_invalid_row_count"]) == 0 for row in rows
        ),
        "anchor_policy": {
            structure: ANCHOR_KIND[structure] for structure in STRUCTURES
        },
        "selection_policy": (
            "For each spacecraft and family, select the half-open interval "
            "from the band-energy interval product: the 5% and 95% points of "
            "cumulative bt_f^2 between the shock minus the search margin and "
            "the ejecta end plus the margin, padded on both sides. All "
            "selections use exact one-minute UTC rows."
        ),
        "coordinate_policy": (
            "Select UTC rows from minute samples of the prepared one-second "
            "distance mapping shared by reference spectra and filters. The "
            "measured local shock t50 sets x=0; inverse times interpolate the "
            "saved minute and exact-anchor coordinates. Distance reports "
            "location and never changes support; "
            "proxy/hybrid provenance and hold-definition sensitivity are "
            "serialized separately."
        ),
        "per_spacecraft_family_supports": supports,
    }


def _speed_coordinate_record(
    processed: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    provenance_path = processed / SPEED_PROVENANCE.name
    provenance = _read_json(provenance_path)
    supports = []
    for row in rows:
        supports.append(
            {
                "structure": row["structure"],
                "craft": row["craft"],
                "expected_row_count": int(
                    row["frozen_speed_support_expected_row_count"]
                ),
                "finite_positive_row_count": int(
                    row["frozen_speed_support_finite_positive_row_count"]
                ),
                "missing_row_count": int(
                    row["frozen_speed_support_missing_row_count"]
                ),
                "gap_count": int(row["frozen_speed_support_gap_count"]),
                "maximum_missing_run_minutes": int(
                    row["frozen_speed_support_max_missing_run_minutes"]
                ),
                "missing_intervals": json.loads(
                    str(row["frozen_speed_support_missing_intervals_json"])
                ),
                "hold_sensitivity_maximum_absolute_coordinate_change_re": (
                    float(
                        row[
                            "frozen_speed_coordinate_hold_sensitivity_max_abs_re"
                        ]
                    )
                ),
                "hold_sensitivity_maximum_coordinate_change_over_span_percent": (
                    float(
                        row[
                            "frozen_speed_coordinate_hold_sensitivity_over_span_percent"
                        ]
                    )
                ),
                "support_span_hold_sensitivity_min_re": float(
                    row[
                        "frozen_speed_support_span_hold_sensitivity_min_re"
                    ]
                ),
                "support_span_hold_sensitivity_max_re": float(
                    row[
                        "frozen_speed_support_span_hold_sensitivity_max_re"
                    ]
                ),
            }
        )
    return {
        "provenance_path": _display_path(provenance_path),
        "per_spacecraft_provenance": provenance,
        "production_gap_fill_policy": SPEED_GAP_FILL_POLICY,
        "sensitivity_policy": (
            "Recompute x(t) with previous-value and next-value holds across "
            "every missing frozen-speed run. Report the largest absolute "
            "coordinate change and support-span range across production, "
            "previous-hold, and next-hold definitions. These are definition-"
            "sensitivity bounds, not confidence intervals."
        ),
        "per_support_diagnostics": supports,
    }


def _product_record(
    *,
    processed: Path,
    output_csv: Path,
    intervals_path: Path,
    rows: list[dict[str, Any]],
    interpolation_sensitivity_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "event": "2026-01-19 CME",
            "row_order": {
                "structure": list(STRUCTURES),
                "craft": list(CRAFT_ORDER),
            },
            "observation_sequence": [
                {
                    "craft": craft,
                    "role": OBSERVATION_ROLE[craft],
                    "sequence_position": CRAFT_ORDER.index(craft),
                }
                for craft in CRAFT_ORDER
            ],
            "determinism": (
                "No wall-clock creation timestamp, random draw, or "
                "platform-dependent source discovery enters this product."
            ),
            "ace_discontinuity_policy": (
                "Included. The native-derived fine-scale product is "
                "filtered before one-minute reduction. A minute is retained "
                "when its filtered values are valid; finite "
                "sample counts are recorded in the waveform product."
            ),
            "solar_orbiter_policy": (
                "Included as the flank observer; SO-to-L1 change combines "
                "physical evolution with flank-to-nose sampling geometry."
            ),
            "artemis_p2_policy": (
                "Included as the lunar-wake endpoint; the P1-to-P2 stage is "
                "interpreted as the Moon's local modification."
            ),
            "trend_warning": (
                "The low-pass waveform at the current-run cutoff retains the "
                "mean field. RMS and scalar power support remain mean-retaining; "
                "the variance ratio subtracts one distance-weighted support "
                "mean from the scalar and each vector component. The waveform "
                "is not an isolated-ejecta fluctuation residual."
            ),
            "effective_support_policy": (
                "All four families use each observer's band-energy interval "
                "with padding. Every selected waveform row is saved-valid."
            ),
        },
        "waveform_intervals": {
            "path": _display_path(intervals_path),
            "schema_version": INTERVALS_SCHEMA_VERSION,
            "status": "computed",
        },
        "frozen_speed_coordinate": _speed_coordinate_record(
            processed,
            rows,
        ),
        "registration": _registration_record(rows),
        "metric_contract": {
            "metric_units": METRIC_UNITS,
            "waveform_inputs": ["bx_f", "by_f", "bz_f", "bt_f", "valid"],
            "scalar_field_note": (
                "bt_f is the independently filtered prepared scalar "
                "field-strength channel, not the magnitude of the filtered "
                "component vector"
            ),
            "effective_support_rms": (
                "sqrt(integral(q^2 dx)/valid_spatial_length) for bt_f and "
                "sqrt(integral(|b_f|^2 dx)/valid_spatial_length) for the "
                "filtered vector over the selected effective support, using "
                "identical connected valid rows"
            ),
            "frozen_speed_coordinate_sensitivity": (
                "Missing frozen-speed runs use piecewise-linear interpolation "
                "between their nearest positive finite samples. Previous- and "
                "next-value holds provide a coordinate-definition sensitivity "
                "bracket; it is not a confidence interval and does not alter "
                "the selected UTC waveform support."
            ),
            "scalar_to_vector_variance_ratio": (
                "Var_x(bt_f)/(Var_x(bx_f)+Var_x(by_f)+Var_x(bz_f)); "
                "Var_x(y)=integral((y-mean_x(y))^2 dx)/valid_spatial_length "
                "and mean_x(y)=integral(y dx)/valid_spatial_length. All means "
                "and variances use the same connected valid distance support "
                "as RMS. The summed vector-component variance must be positive; "
                "zero scalar variance is permitted. Independently processed "
                "scalar/vector channels can yield a ratio above one."
            ),
            "power_support": (
                "scalar bt_f^2 cumulative-power x05/x95; "
                "W90=x95-x05 is nested inside the selected effective support "
                "and never replaces or crops the RMS integration domain"
            ),
            "material_gap_threshold_re": DEFAULT_MAX_MATERIAL_GAP_RE,
            "frame_scope": (
                "Solar Orbiter remains RTN and near-Earth observers use "
                "GSM (native products except definitive DSCOVR converted by SpacePy CTrans/IGRF14). Scalar/vector norms and the vector variance trace are "
                "invariant under a fixed orthogonal frame rotation; "
                "component signs and azimuth are not compared across RTN and GSM. "
                "Time-dependent frame rotations precede filtering and need not "
                "preserve the filtered-vector metrics."
            ),
        },
        "records": rows,
        "interpolation_excluded_sensitivity_records": (
            interpolation_sensitivity_records
        ),
        "provenance": {
            "current_input_files": _input_manifest(processed, rows, intervals_path),
            "csv_output": {
                "path": _display_path(output_csv),
                "columns": list(CSV_COLUMNS),
                "row_count": len(rows),
            },
        },
    }


def build(
    *,
    processed_directory: Path | str = PROCESSED,
    intervals_path: Path | str = WAVEFORM_INTERVALS,
    output_csv: Path | str = OUTPUT_CSV,
    output_json: Path | str = OUTPUT_JSON,
) -> tuple[Path, Path]:
    """Build the canonical table and strict-JSON provenance product."""

    processed = Path(processed_directory)
    csv_path = Path(output_csv)
    json_path = Path(output_json)
    intervals_file, intervals = _read_intervals(Path(intervals_path))
    design = load_band_design(processed / "waveform_bands.json")
    coordinates = read_coordinates(processed, verified_inputs={
        entry["path"]: entry["sha256"] for entry in design["inputs"]
    })
    (
        rows,
        interpolation_sensitivity_records,
    ) = _build_rows(processed, intervals, coordinates)
    _validate_rows(rows, intervals)
    table = pd.DataFrame(rows, columns=CSV_COLUMNS)
    expected_rows = len(STRUCTURES) * len(CRAFT_ORDER)
    if len(table) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} metric rows, received {len(table)}"
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        csv_path,
        index=False,
        columns=CSV_COLUMNS,
        float_format="%.12g",
        na_rep="",
        lineterminator="\n",
    )
    product = _product_record(
        processed=processed,
        output_csv=csv_path,
        intervals_path=intervals_file,
        rows=rows,
        interpolation_sensitivity_records=interpolation_sensitivity_records,
    )
    json_path.write_text(
        json.dumps(
            json_ready(product),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build absolute seven-spacecraft physical metrics from the exact "
            "saved Figure-3 isolated waveforms."
        )
    )
    parser.add_argument(
        "--processed-directory",
        type=Path,
        default=PROCESSED,
        help="current-run processed-data directory",
    )
    parser.add_argument(
        "--intervals",
        type=Path,
        default=WAVEFORM_INTERVALS,
        help="current-run band-energy waveform intervals",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="metric table output path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_JSON,
        help="strict-JSON provenance output path",
    )
    args = parser.parse_args()
    outputs = build(
        processed_directory=args.processed_directory,
        intervals_path=args.intervals,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
