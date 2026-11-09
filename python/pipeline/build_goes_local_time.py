"""Recompute GOES centered-dipole local-time clocks from supplied NOAA orbits."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.goes_local_time import SPACECRAFT, derive_local_time, minute_clock, read_orbit
from analysis.paired_goes_response import DISPLAY_INDEX
from common.io import sha256
from config.paths import DATA_ROOT, OUTPUT_ROOT, RESULTS_DIR
from ingest.missions import daily_files

PREFIX = "goes_local_time"


def orbit_inventory(root: Path, display_index: pd.DatetimeIndex) -> dict[str, list[tuple[str, Path]]]:
    """Select the latest daily source, or its supplied event-file alias.

    File identity and actual UTC day are checked while decoding, so a renamed
    event-day source cannot silently supply an absent next-day prerequisite.
    """
    display_index = minute_clock(display_index)
    days = pd.date_range(display_index[0].normalize(),
                         (display_index[-1] + pd.Timedelta(minutes=1)).normalize(), freq="1D")
    directory = Path(root) / "raw_event_20260119/earth_moon/goes"
    inventory = {}
    for craft in SPACECRAFT:
        stamps = [day.strftime("%Y%m%d") for day in days]
        paths = daily_files(directory, (f"dn_magn-l2-avg1m_{craft}_d{{day}}_v*.nc",
                                        f"{craft}_mag.nc"), stamps)
        inventory[craft] = list(zip(stamps, paths, strict=True))
    return inventory


def build(output_dir: Path = RESULTS_DIR, *, root: Path = DATA_ROOT,
          display_index: pd.DatetimeIndex = DISPLAY_INDEX,
          runtime_output_directory: Path = OUTPUT_ROOT) -> dict:
    """Overwrite all products from current raw orbit and verified reference inputs."""
    root, output_dir = Path(root), Path(output_dir)
    display_index = minute_clock(display_index)
    inventory = orbit_inventory(root, display_index)
    orbits = {craft: pd.concat([read_orbit(path, craft, day) for day, path in sources])
              for craft, sources in inventory.items()}
    minutes, markers, coordinate_method = derive_local_time(
        orbits, display_index=display_index, reference_directory=root / "coordinate_reference",
        output_directory=runtime_output_directory)
    inputs = []
    for craft, sources in inventory.items():
        for day, path in sources:
            inputs.append(dict(spacecraft=craft, day_utc=day,
                               path=str(path.resolve().relative_to(root.resolve())), path_basis="data_root",
                               size_bytes=path.stat().st_size,
                               sha256=sha256(path)))
    output_dir.mkdir(parents=True, exist_ok=True)
    minute_path, marker_path = output_dir / f"{PREFIX}.csv", output_dir / f"{PREFIX}_markers.csv"
    minutes.to_csv(minute_path, index=True, index_label="time_utc", date_format="%Y-%m-%dT%H:%M:%SZ")
    # Preserve subminute measurement precision rather than rounding crossing clocks.
    saved_markers = markers.copy()
    saved_markers["time_utc"] = saved_markers.time_utc.map(lambda stamp: stamp.isoformat())
    saved_markers.to_csv(marker_path, index=False)
    provenance = dict(
        schema_version="goes-local-time-v1",
        display=dict(start_utc=display_index[0].isoformat(),
                     stop_exclusive_utc=(display_index[-1] + pd.Timedelta(minutes=1)).isoformat(),
                     cadence_s=60, minutes_per_spacecraft=len(display_index)),
        inputs=inputs, coordinate_method=coordinate_method,
        reference_manifest=dict(path="coordinate_reference/manifest.json", path_basis="data_root",
                                sha256=sha256(root / "coordinate_reference/manifest.json")),
        outputs=dict(minutes=dict(path=minute_path.name, row_count=len(minutes)),
                     markers=dict(path=marker_path.name, row_count=len(markers))),
    )
    (output_dir / f"{PREFIX}.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return provenance


if __name__ == "__main__":
    result = build()
    print(json.dumps(result["outputs"], indent=2))
