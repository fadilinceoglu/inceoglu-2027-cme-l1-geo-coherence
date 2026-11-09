"""Reference-epoch geometry derived from the supplied mission ephemeris."""

from pathlib import Path

import numpy as np
import pandas as pd

from config import study
from config.paths import DATA_ROOT
from ingest.ephemeris import read_ssc_gse_cache

EPHEMERIS_CACHE = DATA_ROOT / Path(study.L1_SSCWEB_CACHE).relative_to("data")
EPHEMERIS_LABELS = {
    key: meta["name"] for key, meta in study.SATELLITES.items()
    if key not in {"solar_orbiter", "goes18", "goes19"}
}
EPHEMERIS_LABELS["moon"] = "Moon"


def interpolate_position(track: pd.DataFrame, timestamp) -> np.ndarray:
    """Interpolate GSE coordinates without extrapolating beyond observed coverage."""
    index = pd.to_datetime(track.index, utc=True)
    target = pd.Timestamp(timestamp)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    values = track[["x_gse", "y_gse", "z_gse"]].to_numpy(float)
    if (len(index) < 2 or not index.is_monotonic_increasing or index.has_duplicates
            or not np.isfinite(values).all()):
        raise ValueError("ephemeris must contain finite, strictly ordered GSE coordinates")
    if target < index[0] or target > index[-1]:
        raise ValueError(f"position request {target} lies outside ephemeris")
    return np.array([
        np.interp(float(target.value), index.as_unit("ns").asi8.astype(float), component)
        for component in values.T
    ])


def reference_positions(*, cache=None, epoch=None) -> dict[str, np.ndarray]:
    """Return GSE positions in the documented SSCWeb Earth-radius convention.

    Solar Orbiter retains the recorded Horizons source: it has no track in
    this supplied SSCWeb record. Every available near-Earth track is measured
    at the chosen epoch.
    """
    tracks = read_ssc_gse_cache(EPHEMERIS_CACHE if cache is None else cache)
    epoch = study.GEOMETRY_EPOCH if epoch is None else epoch
    positions = {"solar_orbiter": np.asarray(study.SOLAR_ORBITER_POSITION_RE, dtype=float)}
    positions.update({
        key: interpolate_position(tracks[label], epoch)
        for key, label in EPHEMERIS_LABELS.items()
    })
    return positions
