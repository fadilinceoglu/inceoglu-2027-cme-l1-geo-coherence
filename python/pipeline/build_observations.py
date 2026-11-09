"""Prepare every mission record once, then derive the required cadence products.

Observed minute means/counts drive boundaries. Existing overview medians supply
plasma speeds and diagnostic context. Prepared one-second magnetic means
(with bounded mission-specific holding) drive both spectra and spatial filters.
Native scalar samples retain their exact timestamps for shock refinement.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import study
from config.event_boundaries import UPSTREAM_SPACECRAFT
from config.paths import DATA_ROOT, PROCESSED_DIR
from ingest import missions
from ingest.goes import read_goes_mag

CRAFT_ORDER = (*UPSTREAM_SPACECRAFT, *study.GOES_SPACECRAFT)
MAG_COLUMNS = missions.MAG_COLUMNS
PLASMA_COLUMNS = missions.PLASMA_COLUMNS
PHYSICAL_COLUMNS = (*MAG_COLUMNS, *PLASMA_COLUMNS)
GRID_START, GRID_STOP = pd.Timestamp(study.FILTER_CONTEXT_START), pd.Timestamp(study.FILTER_CONTEXT_STOP)
GRID = pd.date_range(GRID_START, GRID_STOP, freq="1min", inclusive="left", name="time")
GOES_VECTOR_SCALAR_VALIDITY_POLICY = (
    "After NOAA DQF screening, aggregate each finite GSM vector component "
    "independently. A missing/fill b_total invalidates only the scalar total; "
    "it does not discard a DQF-usable vector."
)


def minute_statistics(frame, columns):
    """Observed arithmetic means, sample deviations and counts on the shared grid."""
    source = frame.loc[(frame.index >= GRID_START) & (frame.index < GRID_STOP), list(columns)]
    if source.index.has_duplicates or not source.index.is_monotonic_increasing:
        raise ValueError("Source timestamps must be unique and ordered")
    grouped = source.apply(pd.to_numeric, errors="coerce").resample("1min", label="left", closed="left")
    result = pd.DataFrame(index=GRID)
    for column in columns:
        result[column] = grouped[column].mean().reindex(GRID)
        result[f"{column}_std"] = grouped[column].std(ddof=1).reindex(GRID)
        result[f"{column}_samples"] = grouped[column].count().reindex(GRID, fill_value=0).astype("int64")
    return result


def prepared_seconds(craft: str, magnetic: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Average magnetic observations in one-second bins, then apply bounded holds."""
    cadence = float(np.median(np.diff(magnetic.index.as_unit("ns").asi8) / 1e9))
    if not np.isfinite(cadence) or cadence <= 0:
        raise ValueError(f"{craft}: cannot measure native magnetic cadence")
    grid = magnetic.loc[:, list(MAG_COLUMNS)].resample("1s").mean()
    hold = 2 if craft == "wind" else 3 if craft in missions.ARTEMIS_PROBES else 0
    if craft == "wind":
        grid = grid.reindex(pd.date_range(pd.Timestamp(study.FETCH_DAYS_UPSTREAM[0], tz="UTC"),
                                         pd.Timestamp(study.FILTER_CONTEXT_STOP), freq="1s", inclusive="left"))
    if hold:
        grid = grid.ffill(limit=hold)
    # Keep the independently prepared scalar channel: ARTEMIS bt was formed
    # from its native GSE vector before selecting native GSM components.
    # Reconstructing it here from GSM would change the scalar input branch.
    metadata = {"frame": study.SATELLITES[craft]["frame"], "cadence_s": 1,
                "native_median_cadence_s": cadence, "ffill_limit_s": hold,
                "extent": [str(grid.index[0]), str(grid.index[-1])],
                "rows": len(grid), "rows_with_data": int(grid.bt.notna().sum())}
    if magnetic.attrs.get("coordinate_transform") is not None:
        metadata["coordinate_transform"] = magnetic.attrs["coordinate_transform"]
    if craft in missions.ARTEMIS_PROBES:
        metadata.update(native_fgs_median_cadence_s=cadence,
                        event_days=list(missions.ARTEMIS_EVENT_DAYS),
                        filter_padding_days=["20260122"])
    return grid, metadata


def write_native(craft, magnetic, directory):
    directory = Path(directory)
    (directory / "native").mkdir(parents=True, exist_ok=True)
    (directory / "shock_inputs").mkdir(parents=True, exist_ok=True)
    grid, metadata = prepared_seconds(craft, magnetic)
    path = directory / "native" / f"{craft}_1s.csv"
    grid.to_csv(path, index_label="time", float_format="%.4f" if craft in ("ace", "wind") else "%.3f")
    path.with_name(path.stem + "_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # No resampling or rounding enters the native shock-fit samples. Solar-1
    # already has whole-second times in the supplied commissioning product.
    scalar = magnetic.bt.loc[(magnetic.index >= study.ANALYSIS_START)
                             & (magnetic.index < study.ANALYSIS_STOP)]
    if len(scalar) < 2:
        raise ValueError(f"{craft}: no native observations cover the analysis window")
    np.savez(directory / "shock_inputs" / f"{craft}.npz",
             time_ns=scalar.index.as_unit("ns").asi8,
             bt=scalar.to_numpy(dtype=float))
    return path, metadata


def _apply_goes_total_field_policy(raw):
    raw.loc[raw.bt.notna() & ((raw.bt <= 0) | (raw.bt > 1024)), "bt"] = np.nan
    return raw


def goes_products(craft, root):
    """Reduce 10-Hz inputs daily, avoiding a multi-day high-rate allocation."""
    files, _ = missions.inventory(craft, root)
    parts, medians = [], []
    for path in files:
        raw = read_goes_mag(path)
        overview = raw.loc[:, list(MAG_COLUMNS)].resample("1min").median()
        overview.loc[(overview.bt <= 0) | (overview.bt > 1024)] = np.nan
        medians.append(overview)
        raw = _apply_goes_total_field_policy(raw)
        start, stop = raw.index[0].floor("1min"), raw.index[-1].floor("1min")
        parts.append(minute_statistics(raw, MAG_COLUMNS).loc[start:stop])
    minutes = pd.concat(parts).sort_index()
    if minutes.index.has_duplicates:
        raise ValueError(f"{craft}: selected daily files overlap")
    minutes = minutes.reindex(GRID)
    for column in MAG_COLUMNS:
        minutes[f"{column}_samples"] = minutes[f"{column}_samples"].fillna(0).astype("int64")
    for column in PLASMA_COLUMNS:
        minutes[column] = minutes[f"{column}_std"] = np.nan
        minutes[f"{column}_samples"] = 0
    return minutes, pd.concat(medians).sort_index(), files


def build(directory: Path = PROCESSED_DIR, root: Path = DATA_ROOT):
    directory, root = Path(directory), Path(root)
    minute_dir = directory / "common_cadence_1min"
    minute_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for craft in CRAFT_ORDER:
        print(f"{craft}: inventory, screening and cadence preparation", flush=True)
        if craft.startswith("goes"):
            minutes, overview, files = goes_products(craft, root)
            policy = GOES_VECTOR_SCALAR_VALIDITY_POLICY
        else:
            source = missions.load(craft, root)
            files, policy = source.files, source.quality_policy
            minutes = minute_statistics(source.magnetic, MAG_COLUMNS).join(minute_statistics(source.plasma, PLASMA_COLUMNS))
            mag = source.magnetic if source.overview_magnetic is None else source.overview_magnetic
            overview = mag.resample("1min").median().join(source.plasma.resample("1min").median(), how="outer")
            write_native(craft, source.magnetic, directory)
            del source, mag
        minute_path = minute_dir / f"{craft}.csv"
        minutes.to_csv(minute_path, index_label="time", date_format="%Y-%m-%dT%H:%M:%SZ", float_format="%.12g")
        overview.to_csv(directory / f"{craft}_overview_1min.csv", index_label="time")
        records[craft] = {
            "frame": study.SATELLITES[craft]["frame"],
            "magnetic_input_variables": missions.magnetic_input_variables(craft),
            "temperature_unit": None if craft.startswith("goes") else missions.TEMPERATURE_UNITS[craft],
            "quality_policy": policy,
            "inputs": [{"path": str(path), "size_bytes": path.stat().st_size} for path in files],
            "output": {"path": str(minute_path), "row_count": len(minutes)},
            "finite_minute_counts": {column: int(minutes[column].notna().sum()) for column in PHYSICAL_COLUMNS},
        }
        del minutes, overview
        gc.collect()
    manifest = {
        "schema_version": "mission-preparation-v2",
        "grid": {"start_utc": GRID_START.isoformat(), "stop_utc_exclusive": GRID_STOP.isoformat(), "cadence_s": 60},
        "aggregation": {"common_minutes": "arithmetic means, sample standard deviations (ddof=1), finite native counts",
                        "overview": "minute medians; provider minute magnetic variables for Wind and Solar-1",
                        "prepared_seconds": "one-second arithmetic means; Wind held at most 2 s, ARTEMIS at most 3 s",
                        "shock_inputs": "screened native scalar observations and exact integer-nanosecond UTC timestamps"},
        "spacecraft": records,
    }
    path = minute_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path


if __name__ == "__main__":
    print(build())
