"""Describe raw GOES magnitude using signed ARTEMIS-P1 component waveforms.

Twelve inputs (four families times three GSM components) retain their relative
source timing and receive one common measured shock-to-response shift. Fits
and native TreeSHAP values are descriptive, in-event model associations.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from analysis.goes_reconstruction import (
    SCALE_GROUPS, conservative_lightgbm_parameters, input_columns,
    prepare_waveform_predictors, scale_feature_groups,
)
from analysis.paired_goes_response import (
    EARLY_RESPONSE_STOP, EVENT_INDEX, EVENT_START, EVENT_STOP,
    RESPONSE_TRANSITION, load_response,
)
from analysis.event_boundaries import shock_measurement
from analysis.product_io import json_ready
from common.io import read_json, sha256
from config import study
from config.paths import PROCESSED_DIR, RESULTS_DIR
from config.waveform_bands import SPATIAL_DX_RE

BAND_DIR = PROCESSED_DIR / "bands"
SPACECRAFT = {"g18": "GOES-18", "g19": "GOES-19"}
SCHEMA_VERSION = "goes-event-reconstruction-v4"
MODEL_WINDOW_START_UTC = EVENT_INDEX[0]
MODEL_WINDOW_STOP_UTC = EVENT_INDEX[-1]


def _model_window(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the fixed model interval independently of observational padding."""
    return frame.loc[(frame.index >= MODEL_WINDOW_START_UTC)
                     & (frame.index <= MODEL_WINDOW_STOP_UTC)]


def _prepare_predictors(frames: Mapping[str, pd.DataFrame], *, shift_seconds: int,
                        input_kind: str) -> pd.DataFrame:
    return prepare_waveform_predictors(
        frames, propagation_shift=pd.Timedelta(seconds=shift_seconds), input_kind=input_kind,
    )


def _r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    if total <= 0:
        raise ValueError("GOES target must vary over the event interval")
    return float(1.0 - residual / total)


def _full_fit_case(
    predictors: pd.DataFrame, target: pd.Series, *, spacecraft: str,
    target_kind: str = "raw_bt", model_name: str = "all_scale_bands",
) -> tuple[Any, pd.Series, dict[str, Any]]:
    """Fit the fixed deterministic regressor to raw GOES magnitude only."""
    if target_kind != "raw_bt":
        raise ValueError("only raw GOES magnitude is modeled")
    if (not predictors.index.equals(target.index) or not len(predictors)
            or not np.isfinite(predictors.to_numpy(float)).all()
            or not np.isfinite(target.to_numpy(float)).all()):
        raise ValueError("model inputs and raw target require matching finite event minutes")
    from lightgbm import LGBMRegressor
    parameters = conservative_lightgbm_parameters()
    model = LGBMRegressor(**parameters)
    model.fit(predictors, target)
    prediction = pd.Series(np.asarray(model.predict(predictors), dtype=float),
                           index=predictors.index, name="full_fit_prediction")
    residual = target.to_numpy(float) - prediction.to_numpy(float)
    summary = {
        "spacecraft": spacecraft, "target_kind": target_kind, "model": model_name,
        "evaluation": "descriptive_complete_event_reconstruction",
        "row_count": int(len(predictors)), "feature_count": int(predictors.shape[1]),
        "n_estimators": int(parameters["n_estimators"]), "max_depth": int(parameters["max_depth"]),
        "r2": _r2_score(target.to_numpy(float), prediction.to_numpy(float)),
        "rmse_nT": float(np.sqrt(np.mean(residual ** 2))),
        "mae_nT": float(np.mean(np.abs(residual))),
    }
    return model, prediction, summary


def _term_summaries(terms: Mapping[str, np.ndarray], *, key: str,
                    spacecraft: str, shares: bool = True) -> pd.DataFrame:
    rows = [{"spacecraft": spacecraft, "target_kind": "raw_bt", key: name,
             "mean_absolute_model_contribution": float(np.mean(np.abs(values))),
             "units": "nT"} for name, values in terms.items()]
    summary = pd.DataFrame(rows)
    if shares:
        total = float(summary.mean_absolute_model_contribution.sum())
        summary["model_reliance_share_percent"] = (
            100.0 * summary.mean_absolute_model_contribution / total if total > 0 else np.nan)
        summary["share_definition"] = "share of mean absolute signed-sum group contributions"
    return summary


def _grouped_tree_contributions(
    model: Any, features: pd.DataFrame, target: pd.Series, prediction: pd.Series,
    *, spacecraft: str, target_kind: str = "raw_bt",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Keep individual TreeSHAP terms and signed sums by family and GSM axis."""
    if target_kind != "raw_bt":
        raise ValueError("only raw GOES magnitude is modeled")
    raw = np.asarray(model.predict(features, pred_contrib=True), dtype=float)
    if raw.shape != (len(features), features.shape[1] + 1) or not np.isfinite(raw).all():
        raise RuntimeError("LightGBM requires finite feature contributions and one expected-value column")
    positions = {str(column): j for j, column in enumerate(features.columns)}
    groups = scale_feature_groups(features.columns)
    family = {scale: raw[:, [positions[column] for column in groups[scale]]].sum(axis=1)
              for scale in SCALE_GROUPS}
    columns = {
        "time_utc": features.index, "spacecraft": spacecraft, "target_kind": target_kind,
        "observed": target.to_numpy(float), "full_fit_prediction": prediction.to_numpy(float),
        "expected_value": raw[:, -1],
        **{f"{scale}_model_contribution": values for scale, values in family.items()},
        **{f"{feature}_shap_nT": raw[:, j] for feature, j in positions.items()},
    }
    reconstructions = {"feature": raw[:, -1] + raw[:, :-1].sum(axis=1),
                       "family": raw[:, -1] + np.column_stack(list(family.values())).sum(axis=1)}
    if all(str(column).endswith(("__bx_f", "__by_f", "__bz_f")) for column in features.columns):
        axis = {name: raw[:, [j for feature, j in positions.items()
                             if feature.endswith(f"__b{name}_f")]].sum(axis=1)
                for name in ("x", "y", "z")}
        columns.update({f"axis_{name}_model_contribution": values for name, values in axis.items()})
        reconstructions["axis"] = raw[:, -1] + np.column_stack(list(axis.values())).sum(axis=1)
    errors = {name: float(np.max(np.abs(values - prediction.to_numpy(float))))
              for name, values in reconstructions.items()}
    if max(errors.values()) > 1e-8:
        raise RuntimeError("TreeSHAP feature/family/axis terms do not reconstruct the fitted prediction")
    audit = {
        "method": "LightGBM native TreeSHAP (pred_contrib=True); signed feature terms summed within each group",
        "maximum_prediction_reconstruction_error": max(errors.values()),
        "maximum_reconstruction_error_by_grouping": errors,
        "expected_value_is_separate": True,
        "interpretation": "model-attributed event association, not causal or physical-energy fractions",
        "groupings_are_alternative_partitions": True,
    }
    return pd.DataFrame(columns), _term_summaries(family, key="scale_band", spacecraft=spacecraft), audit

def _response_interval_summaries(contribution: pd.DataFrame) -> list[dict[str, Any]]:
    """Describe existing signed family terms over the declared response intervals."""
    index = pd.DatetimeIndex(pd.to_datetime(contribution["time_utc"], utc=True))
    if not index.equals(EVENT_INDEX):
        raise ValueError("response-interval summaries require the complete event clock")
    columns = [f"{scale}_model_contribution" for scale in SCALE_GROUPS]
    values = contribution[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("response-interval contributions must be finite")
    if not contribution["target_kind"].eq("raw_bt").all():
        raise ValueError("only raw GOES magnitude is modeled")
    units = "nT"
    intervals = (
        ("early_response", EVENT_START, EARLY_RESPONSE_STOP),
        ("sustained_disturbance", EARLY_RESPONSE_STOP, RESPONSE_TRANSITION),
        ("later_response", RESPONSE_TRANSITION, EVENT_STOP),
    )
    records = []
    for name, start, stop in intervals:
        selected = values[(index >= start) & (index < stop)]
        if not len(selected):
            raise ValueError(f"response interval {name} contains no samples")
        mean_absolute = np.mean(np.abs(selected), axis=0)
        total = float(mean_absolute.sum())
        shares = 100.0 * mean_absolute / total if total > 0.0 else None
        summary = []
        for position, scale in enumerate(SCALE_GROUPS):
            term = selected[:, position]
            summary.append({
                "scale_band": scale,
                "mean_signed_model_contribution": float(np.mean(term)),
                "minimum_signed_model_contribution": float(np.min(term)),
                "maximum_signed_model_contribution": float(np.max(term)),
                "signed_model_contribution_range": float(np.ptp(term)),
                "standard_deviation_model_contribution": float(np.std(term, ddof=0)),
                "mean_absolute_model_contribution": float(mean_absolute[position]),
                "model_reliance_share_percent": (
                    float(shares[position]) if shares is not None else None
                ),
                "units": units,
            })
        records.append({
            "interval": name,
            "start_utc": start.isoformat(),
            "stop_exclusive_utc": stop.isoformat(),
            "row_count": int(len(selected)),
            "standard_deviation_definition": "RMS departure from the interval mean (ddof=0)",
            "band_pass_model_reliance_share_percent": (
                float(sum(shares[j] for j, scale in enumerate(SCALE_GROUPS) if scale != "trend"))
                if shares is not None else None
            ),
            "summary": summary,
        })
    return records



def _read_source_frames(source: str, *, input_kind: str) -> dict[str, pd.DataFrame]:
    """Trace the saved minute waveforms to the actual prepared one-second source."""
    frames, records = {}, {}
    required_columns = input_columns(input_kind)
    for scale in SCALE_GROUPS:
        path = BAND_DIR / f"{source}__{scale}.csv"
        meta_path = path.with_name(path.stem + "_meta.json")
        metadata = read_json(meta_path)
        native_path = Path(metadata["input_source"])
        native_meta_path = native_path.with_name(native_path.stem + "_meta.json")
        native_metadata = read_json(native_meta_path)
        if metadata.get("canonical_family") != scale:
            raise ValueError(f"{source}/{scale}: wrong filtered-family provenance")
        if (native_path.resolve() != (PROCESSED_DIR / "native" / f"{source}_1s.csv").resolve()
                or native_metadata.get("cadence_s") != 1
                or metadata.get("dx_re") != SPATIAL_DX_RE
                or metadata.get("output_cadence") != "one-minute UTC bins"
                or not metadata.get("one_minute_reduction")):
            raise ValueError(f"{source}/{scale}: require the shared prepared one-second filtering branch")
        if input_kind == "components" and (
                native_metadata.get("frame") != "GSM" or study.SATELLITES[source]["frame"] != "GSM"):
            raise ValueError(f"{source}/{scale}: signed components require actual prepared GSM provenance")
        for file, role in ((path, "filtered_waveform"), (meta_path, "waveform_metadata"),
                           (native_path, "prepared_one_second_magnetic"),
                           (native_meta_path, "prepared_magnetic_metadata")):
            if str(file) not in records:
                records[str(file)] = {"path": str(file), "role": role,
                                      "sha256": sha256(file)}
        frame = pd.read_csv(path, usecols=["time", *required_columns, "valid"], float_precision="round_trip")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("time"), utc=True), name="time_utc")
        frame.attrs["provenance"] = [records[str(file)] for file in
                                     (path, meta_path, native_path, native_meta_path)]
        frame.attrs["prepared_frame"] = native_metadata.get("frame")
        frames[scale] = frame
    return frames


def _require_event_cases(predictors: pd.DataFrame, raw: pd.Series, *, input_kind: str
                         ) -> tuple[pd.DataFrame, pd.Series]:
    """Require the complete declared interval, never a shortened complete-case fit."""
    if not predictors.index.equals(EVENT_INDEX) or not raw.index.equals(EVENT_INDEX):
        raise ValueError(f"the response model requires all {len(EVENT_INDEX)} complete event minutes")
    expected = [f"{scale}__{column}" for scale in SCALE_GROUPS for column in input_columns(input_kind)]
    if list(predictors.columns) != expected:
        raise ValueError(f"the {input_kind} design requires exactly its {len(expected)} signed waveform inputs")
    if not np.isfinite(predictors.to_numpy(float)).all() or not np.isfinite(raw.to_numpy(float)).all():
        raise ValueError("the response model requires finite predictors and raw target at every event minute")
    return predictors, raw


def measured_response_delay(source: str) -> tuple[float, int]:
    """Derive one source-shock to GEO-onset delay, rounded only for minute inputs."""
    delay = (EVENT_INDEX[0] - shock_measurement(source, path=RESULTS_DIR / "boundaries.json")["t50"]).total_seconds()
    if not np.isfinite(delay) or delay < 0:
        raise ValueError(f"{source}: measured shock must precede the GEO response onset")
    return delay, int(round(delay / 60)) * 60


def build_source_analysis(*, source: str, output_prefix: str, input_kind: str) -> dict[str, Any]:
    """Recalculate one source/design from the current waveforms and GOES product."""
    measured_delay, shift_seconds = measured_response_delay(source)
    frames = _read_source_frames(source, input_kind=input_kind)
    predictors = _model_window(_prepare_predictors(frames, shift_seconds=shift_seconds, input_kind=input_kind))
    response_path = RESULTS_DIR / "paired_goes_response.csv"
    response = load_response(response_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "event-specific raw GOES magnitude; in-sample descriptive association",
        "registered_design": {
            "source_spacecraft": source, "input_kind": input_kind,
            "component_frame": "GSM" if input_kind == "components" else None,
            "feature_names": list(predictors.columns), "feature_count": predictors.shape[1],
            "frozen_model_window_utc": {"start": MODEL_WINDOW_START_UTC, "stop": MODEL_WINDOW_STOP_UTC},
            "measured_propagation_delay_s": measured_delay,
            "effective_model_shift_s": shift_seconds,
            "alignment": "one common measured shock-to-GEO-onset shift; source-family timing preserved",
            "model_parameters": conservative_lightgbm_parameters(),
        },
        "waveform_inputs": [
            {"waveform": scale, "path": f"bands/{source}__{scale}.csv", "path_basis": "processed_directory",
             "model_columns": list(input_columns(input_kind)), "row_count": len(frames[scale]),
             "provenance": frames[scale].attrs["provenance"],
             "prepared_frame": frames[scale].attrs["prepared_frame"]}
            for scale in SCALE_GROUPS],
        "boundary_input": {"path": "boundaries.json", "path_basis": "results_directory",
                           "sha256": sha256(RESULTS_DIR / "boundaries.json"),
                           "source_shock_t50_utc": EVENT_INDEX[0] - pd.Timedelta(seconds=measured_delay)},
        "response_input": {"path": response_path.name, "path_basis": "results_directory",
                           "sha256": sha256(response_path)},
        "spacecraft": {},
    }
    samples, feature_summaries = [], []
    for key, label in SPACECRAFT.items():
        features, raw = _require_event_cases(predictors, response[f"{key}_bt"], input_kind=input_kind)
        model, fitted, fit = _full_fit_case(features, raw, spacecraft=key)
        contribution, summary, audit = _grouped_tree_contributions(
            model, features, raw, fitted, spacecraft=key)
        feature_summary = _term_summaries(
            {feature: contribution[f"{feature}_shap_nT"].to_numpy(float) for feature in features},
            key="feature", spacecraft=key, shares=False)
        feature_summary["family"] = [feature.split("__")[0] for feature in feature_summary.feature]
        feature_summary["component"] = [feature.split("__")[1][1] if input_kind == "components" else None
                                        for feature in feature_summary.feature]
        target_result = {
            "full_fit": fit,
            "grouped_contributions": {"audit": audit, "summary": summary.to_dict(orient="records"),
                                      "response_intervals": _response_interval_summaries(contribution)},
            "feature_contributions": {"summary": feature_summary.to_dict(orient="records")},
        }
        if input_kind == "components":
            axis_summary = _term_summaries(
                {name: contribution[f"axis_{name}_model_contribution"].to_numpy(float)
                 for name in ("x", "y", "z")}, key="component", spacecraft=key)
            target_result["axis_contributions"] = {"summary": axis_summary.to_dict(orient="records")}
        result["spacecraft"][key] = {"label": label, "targets": {"raw_bt": target_result}}
        samples.append(contribution)
        feature_summaries.append(feature_summary)
    tables = {"event_samples": pd.concat(samples, ignore_index=True),
              "predictors": predictors.reset_index(),
              "feature_contributions": pd.concat(feature_summaries, ignore_index=True)}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result["outputs"] = {}
    for name, table in tables.items():
        path = RESULTS_DIR / f"{output_prefix}_{name}.csv"
        table.to_csv(path, index=False)
        result["outputs"][name] = path.name
    result = json_ready(result)
    (RESULTS_DIR / f"{output_prefix}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def build() -> dict[str, Any]:
    return build_source_analysis(source="artemis_p1", output_prefix="p1_goes_reconstruction", input_kind="components")


if __name__ == "__main__":
    result = build()
    print(json.dumps({key: value["targets"]["raw_bt"]["full_fit"]
                      for key, value in result["spacecraft"].items()}, indent=2))
