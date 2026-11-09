"""Current Figure-4 product contract shared by computation and standalone readers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from analysis.chain_waveform_metrics import CRAFT_ORDER, STRUCTURES
from analysis.product_io import boolean_values

SCHEMA_VERSION: Final = "chain-waveform-metrics-v9.0.0"


INTERVALS_SCHEMA_VERSION: Final = "waveform-intervals-v1.0.0"
INTERVAL_POLICY: Final = "band_energy_edges_with_padding"
INTERVAL_STATUS: Final = "computed"


SPEED_GAP_FILL_POLICY: Final = (
    "piecewise_linear_interpolation_between_nearest_positive_finite_"
    "frozen_speed_samples"
)


ANCHOR_KIND: Final = {
    structure: "local_native_shock_t50_common_coordinate_anchor"
    for structure in STRUCTURES
}


OBSERVATION_ROLE: Final = {
    "solar_orbiter": "flank_observer",
    "ace": "l1_cluster",
    "wind": "l1_cluster",
    "dscovr": "l1_cluster",
    "swfo": "l1_cluster",
    "artemis_p1": "incident_solar_wind_at_lunar_distance",
    "artemis_p2": "lunar_wake",
}


CSV_COLUMNS: Final = (
    "schema_version",
    "structure",
    "craft",
    "sequence_position",
    "observation_role",
    "frame",
    "comparison_status",
    "eligible",
    "measurement_available",
    "flags",
    "waveform_source_slug",
    "waveform_source_path",
    "waveform_metadata_path",
    "effective_support_policy",
    "effective_support_status",
    "effective_support_start_utc",
    "effective_support_stop_utc_exclusive",
    "effective_support_duration_minutes",
    "filter_minimum_re",
    "filter_maximum_re",
    "frozen_speed_source_path",
    "frozen_speed_provenance_description",
    "frozen_speed_gap_fill_policy",
    "frozen_speed_support_expected_row_count",
    "frozen_speed_support_finite_positive_row_count",
    "frozen_speed_support_missing_row_count",
    "frozen_speed_support_gap_count",
    "frozen_speed_support_max_missing_run_minutes",
    "frozen_speed_support_missing_intervals_json",
    "frozen_speed_coordinate_hold_sensitivity_max_abs_re",
    "frozen_speed_coordinate_hold_sensitivity_over_span_percent",
    "frozen_speed_support_span_hold_sensitivity_min_re",
    "frozen_speed_support_span_hold_sensitivity_max_re",
    "anchor_kind",
    "anchor_utc",
    "registered_support_start_re",
    "registered_support_stop_re",
    "registered_support_span_re",
    "registered_window_start_utc",
    "registered_window_last_included_utc",
    "registered_window_stop_utc_exclusive",
    "registered_stored_row_count",
    "registered_valid_row_count",
    "registered_invalid_row_count",
    "registered_first_invalid_utc",
    "registered_last_invalid_utc",
    "stored_cadence_s",
    "effective_cadence_s",
    "valid_spatial_length_re",
    "spatial_coverage_fraction",
    "source_interpolated_row_count",
    "power_support_start_re",
    "power_support_start_utc",
    "power_support_stop_re",
    "power_support_stop_utc",
    "power_support_width_re",
    "scalar_integrated_power_nT2_re",
    "scalar_power_support_crosses_material_gap",
    "scalar_bt_rms_nT",
    "vector_rms_nT",
    "scalar_to_vector_variance_ratio",
)


WAVEFORM_MEASUREMENT_COLUMNS: Final = (
    "valid_spatial_length_re",
    "spatial_coverage_fraction",
    "source_interpolated_row_count",
    "power_support_start_re",
    "power_support_start_utc",
    "power_support_stop_re",
    "power_support_stop_utc",
    "power_support_width_re",
    "scalar_integrated_power_nT2_re",
    "scalar_power_support_crosses_material_gap",
    "scalar_bt_rms_nT",
    "vector_rms_nT",
    "scalar_to_vector_variance_ratio",
)


def _comparison_status(craft: str) -> str:
    return f"included_{OBSERVATION_ROLE[craft]}"


def _support_selection(
    intervals: dict[str, Any],
    *,
    craft: str,
    structure: str,
) -> dict[str, Any]:
    """Return one spacecraft/family interval from the current-run interval product."""

    try:
        record = intervals["intervals"][craft][structure]
    except KeyError as error:
        raise ValueError(f"waveform intervals lack {craft}/{structure}") from error
    start = pd.Timestamp(record["start_utc"])
    stop = pd.Timestamp(record["stop_utc_exclusive"])
    duration = int(record["duration_minutes"])
    if stop <= start or duration != int((stop - start) / pd.Timedelta(minutes=1)):
        raise ValueError(f"{craft}/{structure} interval duration is inconsistent")
    return {
        "start_utc": start,
        "stop_utc_exclusive": stop,
        "duration_minutes": duration,
        "policy": INTERVAL_POLICY,
        "status": INTERVAL_STATUS,
    }


def _is_null(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def _validate_rows(
    rows: list[dict[str, Any]],
    intervals: dict[str, Any] | None = None,
) -> None:
    """Fail before writing if the frozen 4-by-7 product contract is broken."""

    for row in rows:
        missing = set(CSV_COLUMNS) - row.keys()
        if missing:
            raise ValueError("chain waveform table is missing columns: " + ", ".join(sorted(missing)))
        if row["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"Figure-4 metrics must use schema {SCHEMA_VERSION}")
    expected_keys = [
        (structure, craft)
        for structure in STRUCTURES
        for craft in CRAFT_ORDER
    ]
    observed_keys = [
        (str(row.get("structure")), str(row.get("craft"))) for row in rows
    ]
    if observed_keys != expected_keys:
        raise ValueError(
            "metric rows must follow the exact structure-major, craft-minor "
            "4-by-7 order"
        )
    anchor_by_craft: dict[str, str] = {}
    for row in rows:
        structure = str(row["structure"])
        craft = str(row["craft"])
        key = f"{structure}/{craft}"
        expected_status = _comparison_status(craft)
        if row.get("comparison_status") != expected_status:
            raise ValueError(f"{key} has an invalid comparison status")
        if not bool(row.get("eligible")):
            raise ValueError(f"{key} has an invalid eligible flag")
        if not bool(row.get("measurement_available")):
            raise ValueError(f"{key} has an invalid availability flag")
        if row.get("anchor_kind") != ANCHOR_KIND[structure]:
            raise ValueError(f"{key} does not use the common shock-t50 anchor")
        expected_support = _metric_support(row)
        if intervals is not None:
            measured = _support_selection(intervals, craft=craft, structure=structure)
            for field in ("start_utc", "stop_utc_exclusive", "duration_minutes", "policy", "status"):
                if expected_support[field] != measured[field]:
                    raise ValueError(f"{key} differs from its computed interval")
            expected_support = measured
        window_start = pd.Timestamp(row["registered_window_start_utc"])
        window_last = pd.Timestamp(
            row["registered_window_last_included_utc"]
        )
        window_stop_exclusive = pd.Timestamp(
            row["registered_window_stop_utc_exclusive"]
        )
        expected_start = pd.Timestamp(expected_support["start_utc"])
        expected_stop = pd.Timestamp(expected_support["stop_utc_exclusive"])
        expected_last = expected_stop - pd.Timedelta(minutes=1)
        expected_rows = int(expected_support["duration_minutes"])
        if not (
            window_start == expected_start
            and window_last == expected_last
            and window_stop_exclusive == expected_stop
        ):
            raise ValueError(f"{key} does not use its computed half-open support")
        if (
            pd.Timestamp(row["effective_support_start_utc"])
            != expected_start
            or pd.Timestamp(row["effective_support_stop_utc_exclusive"])
            != expected_stop
            or int(row["effective_support_duration_minutes"])
            != expected_rows
        ):
            raise ValueError(f"{key} support provenance differs from its clock")
        if row["effective_support_policy"] != expected_support["policy"]:
            raise ValueError(f"{key} serializes the wrong support policy")
        if row["effective_support_status"] != expected_support["status"]:
            raise ValueError(f"{key} serializes the wrong support status")
        if int(row["registered_stored_row_count"]) != expected_rows:
            raise ValueError(f"{key} stored-row count differs from its duration")
        if (
            int(row["registered_valid_row_count"]) != expected_rows
            or int(row["registered_invalid_row_count"]) != 0
        ):
            raise ValueError(f"{key} support contains an invalid saved row")
        if not (
            _is_null(row["registered_first_invalid_utc"])
            and _is_null(row["registered_last_invalid_utc"])
        ):
            raise ValueError(f"{key} serializes a spurious invalid row")
        if not (
            np.isclose(float(row["stored_cadence_s"]), 60.0)
            and np.isclose(float(row["effective_cadence_s"]), 60.0)
        ):
            raise ValueError(f"{key} is not on the common one-minute cadence")
        speed_expected = int(row["frozen_speed_support_expected_row_count"])
        speed_finite = int(
            row["frozen_speed_support_finite_positive_row_count"]
        )
        speed_missing = int(row["frozen_speed_support_missing_row_count"])
        speed_gap_count = int(row["frozen_speed_support_gap_count"])
        speed_max_run = int(
            row["frozen_speed_support_max_missing_run_minutes"]
        )
        try:
            speed_intervals = json.loads(
                str(row["frozen_speed_support_missing_intervals_json"])
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{key} has malformed frozen-speed gap provenance"
            ) from error
        if not (
            row["frozen_speed_gap_fill_policy"] == SPEED_GAP_FILL_POLICY
            and bool(str(row["frozen_speed_provenance_description"]).strip())
            and speed_expected == expected_rows
            and speed_finite + speed_missing == speed_expected
            and isinstance(speed_intervals, list)
            and len(speed_intervals) == speed_gap_count
            and sum(
                int(item["duration_minutes"]) for item in speed_intervals
            )
            == speed_missing
            and speed_max_run
            == max(
                (
                    int(item["duration_minutes"])
                    for item in speed_intervals
                ),
                default=0,
            )
        ):
            raise ValueError(f"{key} has inconsistent frozen-speed coverage")
        speed_sensitivity = float(
            row["frozen_speed_coordinate_hold_sensitivity_max_abs_re"]
        )
        speed_sensitivity_percent = float(
            row[
                "frozen_speed_coordinate_hold_sensitivity_over_span_percent"
            ]
        )
        speed_span_min = float(
            row["frozen_speed_support_span_hold_sensitivity_min_re"]
        )
        speed_span_max = float(
            row["frozen_speed_support_span_hold_sensitivity_max_re"]
        )
        if not (
            np.isfinite(speed_sensitivity)
            and speed_sensitivity >= 0.0
            and np.isfinite(speed_sensitivity_percent)
            and speed_sensitivity_percent >= 0.0
            and np.isfinite(speed_span_min)
            and np.isfinite(speed_span_max)
            and speed_span_min <= float(row["registered_support_span_re"])
            <= speed_span_max
        ):
            raise ValueError(
                f"{key} has invalid frozen-speed coordinate sensitivity"
            )
        if speed_missing == 0 and not (
            np.isclose(speed_sensitivity, 0.0, atol=1.0e-9)
            and np.isclose(speed_span_min, speed_span_max, atol=1.0e-9)
        ):
            raise ValueError(
                f"{key} has speed sensitivity without a missing speed row"
            )
        support_start = float(row["registered_support_start_re"])
        support_stop = float(row["registered_support_stop_re"])
        support_span = float(row["registered_support_span_re"])
        if not (
            support_start < support_stop
            and np.isclose(support_span, support_stop - support_start)
        ):
            raise ValueError(f"{key} has an invalid distance coordinate")
        anchor = str(row["anchor_utc"])
        if craft in anchor_by_craft and anchor != anchor_by_craft[craft]:
            raise ValueError(f"{craft} changes coordinate anchor by family")
        anchor_by_craft[craft] = anchor
        if not (support_start <= float(row["power_support_start_re"])
                < float(row["power_support_stop_re"]) <= support_stop):
            raise ValueError(f"{key} has unordered spatial support")
        if not np.isclose(float(row["power_support_width_re"]),
                          float(row["power_support_stop_re"]) - float(row["power_support_start_re"]),
                          rtol=1e-7, atol=1e-7):
            raise ValueError(f"{key} has inconsistent support width")
        coverage = float(row["spatial_coverage_fraction"])
        if not np.isclose(coverage, 1.0):
            raise ValueError(f"{key} does not cover its complete effective support")

        missing_waveform = [
            column
            for column in WAVEFORM_MEASUREMENT_COLUMNS
            if _is_null(row.get(column))
        ]
        if missing_waveform:
            raise ValueError(
                f"{key} has incomplete waveform metrics: "
                + ", ".join(missing_waveform)
            )
        numeric_waveform = [
            column for column in WAVEFORM_MEASUREMENT_COLUMNS
            if not column.endswith(("_utc", "_gap"))
        ]
        if not np.isfinite([float(row[column]) for column in numeric_waveform]).all():
            raise ValueError(f"{key} has nonfinite core metrics")
        start = pd.Timestamp(row["power_support_start_utc"])
        stop = pd.Timestamp(row["power_support_stop_utc"])
        if not window_start <= start < stop <= window_last:
            raise ValueError(f"{key} has unordered UTC power support")
        if not (
            float(row["power_support_width_re"]) > 0.0
            and float(row["scalar_bt_rms_nT"]) >= 0.0
            and float(row["vector_rms_nT"]) > 0.0
        ):
            raise ValueError(f"{key} has invalid waveform amplitudes")
        if float(row["scalar_to_vector_variance_ratio"]) < 0.0:
            raise ValueError(f"{key} has an invalid scalar-to-vector variance ratio")



def _metric_support(row: dict[str, Any]) -> dict[str, Any]:
    start = pd.Timestamp(row["effective_support_start_utc"])
    stop = pd.Timestamp(row["effective_support_stop_utc_exclusive"])
    duration = int(row["effective_support_duration_minutes"])
    if stop <= start or duration != (stop - start) / pd.Timedelta(minutes=1):
        raise ValueError("inconsistent half-open support")
    if row["effective_support_policy"] != INTERVAL_POLICY or row["effective_support_status"] != INTERVAL_STATUS:
        raise ValueError("metric row has an unsupported effective support policy/status")
    return {"start_utc": start, "stop_utc_exclusive": stop, "duration_minutes": duration,
            "policy": INTERVAL_POLICY, "status": INTERVAL_STATUS}


def read_metrics(path: Path | str) -> pd.DataFrame:
    """Read the current CSV contract and reject incomplete or invalid results."""
    frame = pd.read_csv(path)
    missing = set(CSV_COLUMNS) - set(frame)
    if missing:
        raise ValueError("chain waveform table is missing columns: " + ", ".join(sorted(missing)))
    unknown = set(frame["structure"]) - set(STRUCTURES)
    if unknown:
        raise ValueError("unsupported structures: " + ", ".join(sorted(unknown)))
    if frame.duplicated(["structure", "craft"]).any():
        raise ValueError("duplicate structure/craft rows")
    index = pd.MultiIndex.from_product((STRUCTURES, CRAFT_ORDER), names=("structure", "craft"))
    observed = pd.MultiIndex.from_frame(frame[["structure", "craft"]])
    if set(index) - set(observed):
        raise ValueError("missing structure/craft rows")
    if set(observed) - set(index):
        raise ValueError("unsupported structure/craft rows")
    frame = frame.set_index(["structure", "craft"]).loc[index].reset_index()
    for column in ("eligible", "measurement_available", "scalar_power_support_crosses_material_gap"):
        frame[column] = boolean_values(frame[column], name=column)
    _validate_rows(frame.to_dict(orient="records"))
    return frame.set_index(["structure", "craft"])
