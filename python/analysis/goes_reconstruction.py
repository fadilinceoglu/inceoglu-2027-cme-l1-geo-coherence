"""Signed minute waveform inputs for descriptive GOES reconstruction.

All waveform families retain their relative timing. A single measured
propagation shift is applied to the source clock; no independent alignment,
vector norm, or extra history is introduced.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from analysis.product_io import boolean_values
from config.waveform_bands import WAVEFORM_FAMILIES

SCALE_GROUPS = WAVEFORM_FAMILIES
VECTOR_COLUMNS = ("bx_f", "by_f", "bz_f")
SIGNED_SCALAR_COLUMN = "bt_f"
CADENCE = pd.Timedelta(minutes=1)
MODEL_SEED = 20260119


def input_columns(input_kind: str) -> tuple[str, ...]:
    """The two separately fitted designs: signed components or signed |B| bands."""
    if input_kind == "components":
        return VECTOR_COLUMNS
    if input_kind == "scalar":
        return (SIGNED_SCALAR_COLUMN,)
    raise ValueError("input_kind must be components or scalar")


def _minute_delay(value: str | int | float | pd.Timedelta) -> pd.Timedelta:
    """Interpret numeric shifts as minutes; require a nonnegative whole minute."""
    shift = (pd.Timedelta(minutes=float(value))
             if isinstance(value, (int, float, np.integer, np.floating))
             else pd.Timedelta(value))
    if pd.isna(shift) or shift < pd.Timedelta(0) or shift.value % CADENCE.value:
        raise ValueError("propagation_shift must be a nonnegative exact multiple of one minute")
    return shift


def prepare_waveform_predictors(
    scale_frames: Mapping[str, pd.DataFrame],
    *,
    propagation_shift: str | int | float | pd.Timedelta,
    input_kind: str,
) -> pd.DataFrame:
    """Shift the four prepared one-minute waveforms together, without resampling.

    Invalid and missing source minutes remain NaN. The builder subsequently
    requires every input and target to be finite throughout its fixed interval.
    Component columns are signed coordinates, not a vector magnitude.
    """
    if set(scale_frames) != set(SCALE_GROUPS):
        raise ValueError(f"scale_frames must define exactly {', '.join(SCALE_GROUPS)}")
    columns = input_columns(input_kind)
    shift = _minute_delay(propagation_shift)
    pieces, metadata = [], {}
    for scale in SCALE_GROUPS:
        frame = scale_frames[scale]
        missing = sorted({*columns, "valid"} - set(frame.columns))
        if missing:
            raise ValueError(f"{scale} frame is missing columns: {missing}")
        index = frame.index
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
            raise ValueError(f"{scale} requires a UTC-aware DatetimeIndex")
        if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError(f"{scale} requires unique ordered timestamps")
        if (len(index) < 2 or np.any(index.asi8 % CADENCE.value)
                or np.any(np.diff(index.asi8) % CADENCE.value)):
            raise ValueError(f"{scale} input must contain the one-minute filtered records")
        numeric = frame.loc[:, columns].astype(float).copy()
        valid = boolean_values(frame["valid"], name=f"{scale} validity")
        if not np.isfinite(numeric.to_numpy()[valid]).all():
            raise ValueError(f"{scale} has nonfinite waveform values marked valid")
        numeric.loc[~valid] = np.nan
        # Reindexing exposes missing minutes; it does not interpolate or average.
        numeric = numeric.reindex(pd.date_range(index[0], index[-1], freq=CADENCE))
        numeric.index += shift
        names = {column: f"{scale}__{column}" for column in columns}
        pieces.append(numeric.rename(columns=names))
        for column, feature in names.items():
            metadata[feature] = {
                "family": scale, "source_column": column,
                "component": column[1] if input_kind == "components" else None,
                "propagation_shift_s": shift.total_seconds(),
            }
    predictors = pd.concat(pieces, axis=1).sort_index()
    predictors.index.name = "time_utc"
    predictors.attrs.update(
        input_kind=input_kind, scale_groups=list(SCALE_GROUPS),
        cadence_s=CADENCE.total_seconds(), propagation_shift_s=shift.total_seconds(),
        feature_metadata=metadata,
    )
    return predictors


def conservative_lightgbm_parameters() -> dict[str, Any]:
    """The fixed model settings, shared by all three separately fitted designs."""
    return {
        "objective": "regression_l2", "n_estimators": 400,
        "learning_rate": 0.02, "num_leaves": 7, "max_depth": 3,
        "min_child_samples": 15, "min_split_gain": 0.0,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "subsample": 1.0, "subsample_freq": 0, "colsample_bytree": 1.0,
        "random_state": MODEL_SEED, "n_jobs": 1, "deterministic": True,
        "force_col_wise": True, "verbosity": -1,
    }


def scale_feature_groups(columns: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Group the signed input channels by waveform family."""
    groups: dict[str, list[str]] = {scale: [] for scale in SCALE_GROUPS}
    for column in columns:
        scale = str(column).split("__", 1)[0]
        if scale not in groups:
            raise ValueError(f"unknown predictor scale: {column}")
        groups[scale].append(str(column))
    if any(not members for members in groups.values()):
        raise ValueError("all four scale groups require predictors")
    return {scale: tuple(members) for scale, members in groups.items()}
