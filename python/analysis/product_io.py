"""Small shared conversions at CSV and JSON product boundaries."""

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common.io import utc_text, utc_timestamp


def boolean_values(series: pd.Series, *, name: str = "value") -> np.ndarray:
    """Parse saved Boolean flags, rejecting missing or unrecognized values."""
    if series.isna().any():
        raise ValueError(f"{name} contains missing Boolean values")
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    parsed = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise ValueError(f"{name} contains values that are not Boolean")
    return parsed.to_numpy(dtype=bool)


def json_ready(value):
    """Convert analysis values to portable JSON, representing nonfinite values as null."""
    if value is pd.NA or value is pd.NaT:
        return None
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, pd.DataFrame):
        return json_ready(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return json_ready(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (pd.Index, np.ndarray)):
        return json_ready(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return utc_timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
