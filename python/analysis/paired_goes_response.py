"""Observed GOES magnitude, direction and within-minute Bz variability.

Quiet scalar magnitude is averaged separately from the GSM vector; each
reference requires all requested same-clock controls to be finite.
"""
from __future__ import annotations
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import pandas as pd

from config import study

EVENT_START = pd.Timestamp(study.GEO_RESPONSE_ONSET)
RESPONSE_TRANSITION = pd.Timestamp(study.GEO_SHEATH_RESPONSE_STOP)
EARLY_RESPONSE_STOP = EVENT_START + pd.Timedelta(minutes=15)
EVENT_STOP = pd.Timestamp("2026-01-20T06:00:00Z")
EVENT_INDEX = pd.date_range(EVENT_START, EVENT_STOP, freq="1min", inclusive="left")
# Display context is a chosen plotting parameter, separate from the modeled event.
DISPLAY_PADDING_MINUTES = 17
DISPLAY_START = EVENT_START - pd.Timedelta(minutes=DISPLAY_PADDING_MINUTES)
DISPLAY_INDEX = pd.date_range(DISPLAY_START, EVENT_STOP, freq="1min", inclusive="left")
VECTOR_COLUMNS = ("bx", "by", "bz")

def vector_angle_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Angle between physical vectors; zero/missing vectors remain NaN."""
    a, b = np.broadcast_arrays(np.asarray(left, float), np.asarray(right, float))
    if a.ndim < 1 or a.shape[-1] != 3:
        raise ValueError("vectors must have final dimension three")
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.full(denominator.shape, np.nan)
    valid = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
    valid &= denominator > 0.0
    np.divide(np.sum(a * b, axis=-1), denominator, out=cosine, where=valid)
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _interpolation_flags(frame: pd.DataFrame) -> np.ndarray:
    if "mag_interpolated" not in frame:
        return np.zeros(len(frame), dtype=bool)
    return frame["mag_interpolated"].fillna(1).to_numpy(float) != 0.0


def spacecraft_response(
    record: pd.DataFrame,
    event_index: pd.DatetimeIndex = EVENT_INDEX,
    *,
    control_days: Sequence[int] = (1, 2),
) -> pd.DataFrame:
    """Subtract each craft's own exact same-UT quiet magnitude/vector.

    Every requested control must be finite. The two-day reference is an
    arithmetic vector mean and a separate arithmetic mean of |B|. These
    differ from the magnitude of the averaged vector and are not conflated.
    Existing bounded gap estimates are retained, explicitly flagged.
    """
    required = [*VECTOR_COLUMNS, "bt", "bz_std"]
    if not control_days or len(set(control_days)) != len(control_days):
        raise ValueError("require distinct positive quiet-day offsets")
    if any(day <= 0 for day in control_days):
        raise ValueError("quiet references must precede the event")
    if not isinstance(record.index, pd.DatetimeIndex) or record.index.tz is None:
        raise ValueError("record must have a timezone-aware DatetimeIndex")
    if record.index.has_duplicates or not record.index.is_monotonic_increasing:
        raise ValueError("record clock must be unique and sorted")
    event = record.reindex(event_index)
    if not np.isfinite(event[required].to_numpy(float)).all():
        raise ValueError("event requires all physical vectors, magnitudes and spreads")
    controls = [record.reindex(event_index - pd.Timedelta(days=d)) for d in control_days]
    reference_columns = [*VECTOR_COLUMNS, "bt"]
    quiet_values = np.stack([q[reference_columns].to_numpy(float) for q in controls])
    if not np.isfinite(quiet_values).all():
        raise ValueError("every requested same-UT quiet control must be finite")
    quiet = quiet_values.mean(axis=0)
    out = event[required].copy()
    for i, column in enumerate(reference_columns):
        out[f"quiet_{column}"] = quiet[:, i]
    vector = event[list(VECTOR_COLUMNS)].to_numpy(float)
    quiet_vector = quiet[:, :3]
    out["delta_bt_nt"] = event.bt.to_numpy(float) - out.quiet_bt.to_numpy(float)
    out["angle_quiet_deg"] = vector_angle_deg(vector, quiet_vector)
    out["event_interpolated"] = _interpolation_flags(event)
    out["quiet_interpolated"] = np.any(
        np.stack([_interpolation_flags(q) for q in controls]), axis=0
    )
    out["quiet_control_count"] = len(control_days)
    if not np.isfinite(out.angle_quiet_deg).all():
        raise ValueError("quiet/event physical direction is undefined at a zero vector")
    return out


def paired_response(
    records: Mapping[str, pd.DataFrame],
    *,
    event_index: pd.DatetimeIndex = EVENT_INDEX,
    control_days: Sequence[int] = (1, 2),
) -> pd.DataFrame:
    if set(records) != {"g18", "g19"}:
        raise ValueError("paired response requires exactly GOES-18 and GOES-19")
    craft = {
        key: spacecraft_response(value, event_index, control_days=control_days)
        for key, value in records.items()
    }
    out = pd.concat([craft[key].add_prefix(f"{key}_") for key in ("g18", "g19")], axis=1)
    out.index.name = "time_utc"
    out["uses_interpolated_sample"] = out[
        [f"{key}_{kind}_interpolated" for key in ("g18", "g19") for kind in ("event", "quiet")]
    ].any(axis=1)
    return out


def load_response(
    path: Path, *, event_index: pd.DatetimeIndex = EVENT_INDEX,
) -> pd.DataFrame:
    """Read the event slice, or explicitly request its pre-onset display context.

    The saved observational product must contain the full display clock. The
    default returned slice remains the original response-model interval.
    """
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("time_utc"), utc=True), name="time_utc")
    if not frame.index.equals(DISPLAY_INDEX):
        raise ValueError("GOES response must contain the complete ordered display clock")
    if not frame["reference"].eq("two_quiet_days").all():
        raise ValueError("the primary GOES response requires both quiet-day controls")
    return frame.loc[event_index].copy()
