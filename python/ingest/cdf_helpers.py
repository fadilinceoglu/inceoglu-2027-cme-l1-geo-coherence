"""Small shared CDF input and numeric-screening helpers."""
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from cdflib import CDF

def _normalise_paths(paths: Path | str | Iterable[Path | str]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        result = [Path(paths)]
    else:
        result = [Path(path) for path in paths]
    if not result:
        raise ValueError("At least one CDF path is required")
    missing = [str(path) for path in result if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CDF file(s) not found: {missing}")
    return result

def _valid_numeric(cdf: CDF, variable: str) -> np.ndarray:
    values = np.asarray(cdf.varget(variable), dtype=float)
    attrs = cdf.varattsget(variable)
    fill = attrs.get("FILLVAL")
    if fill is not None and np.isfinite(float(fill)):
        values[np.isclose(values, float(fill))] = np.nan
    valid_min = attrs.get("VALIDMIN")
    valid_max = attrs.get("VALIDMAX")
    if valid_min is not None:
        values = np.where(values >= np.asarray(valid_min, dtype=float), values, np.nan)
    if valid_max is not None:
        values = np.where(values <= np.asarray(valid_max, dtype=float), values, np.nan)
    return values

def _combine(frames: list[pd.DataFrame], attrs: dict) -> pd.DataFrame:
    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    result.attrs.update(attrs)
    return result
