"""Fill and flag the bounded short GOES-18/19 magnetic gaps."""

from __future__ import annotations

from config.paths import PROCESSED_DIR

import json
from pathlib import Path
from typing import Final

import pandas as pd

from pipeline.fill_wind_minute_gaps import (
    FIGURE2_START,
    FIGURE2_STOP,
    MAG_MEANS,
    MAG_STDS,
    fill_isolated_magnetic_gaps,
)


PYTHON_ROOT: Final = Path(__file__).resolve().parents[1]
DIRECTORY: Final = PROCESSED_DIR / "common_cadence_1min"
from config.study import GOES_SPACECRAFT as CRAFT
MANIFEST: Final = DIRECTORY / "goes_gapfill_manifest.json"


def build(
    directory: Path = DIRECTORY,
    manifest_path: Path = MANIFEST,
) -> tuple[list[Path], Path]:
    records = {}
    outputs = []
    for craft in CRAFT:
        input_path = directory / f"{craft}.csv"
        output_path = directory / f"{craft}_gapfilled.csv"
        frame = pd.read_csv(input_path, parse_dates=["time"], index_col="time")
        # GOES high-rate files are available through 21 January, while the
        # shared context grid also carries 22 January for the upstream
        # waveform filters.  Fill only bounded gaps inside each GOES
        # spacecraft's observed support and leave the unused trailing context
        # missing and unflagged.
        observed = frame.loc[:, list(MAG_MEANS)].notna().all(axis=1)
        if not observed.any():
            raise ValueError(f"{craft} has no observed magnetic support")
        first = observed[observed].index[0]
        last = observed[observed].index[-1]
        active = frame.loc[first:last]
        filled_active, audit = fill_isolated_magnetic_gaps(
            active,
            maximum_gap_minutes=2,
        )
        filled = frame.copy()
        filled["mag_interpolated"] = 0
        filled.loc[active.index, filled_active.columns] = filled_active
        filled.to_csv(
            output_path,
            index_label="time",
            date_format="%Y-%m-%dT%H:%M:%SZ",
            float_format="%.12g",
            na_rep="",
            lineterminator="\n",
        )
        records[craft] = {
            "input": {"path": str(input_path), "size_bytes": input_path.stat().st_size},
            "output": {
                "path": str(output_path),
                "size_bytes": output_path.stat().st_size,
            },
            "filled_minutes": audit,
        }
        outputs.append(output_path)

    manifest = {
        "spacecraft": list(CRAFT),
        "source_native_cadence_hz": 10,
        "method": "linear interpolation in UTC between adjacent observed minutes",
        "maximum_gap_minutes": 2,
        "columns_filled": list((*MAG_MEANS, *MAG_STDS)),
        "sample_counts_at_filled_minutes": 0,
        "plot_flag_column": "mag_interpolated",
        "figure2_window": {
            "start_utc": FIGURE2_START.isoformat().replace("+00:00", "Z"),
            "stop_utc_exclusive": FIGURE2_STOP.isoformat().replace(
                "+00:00", "Z"
            ),
        },
        "standard_deviation_note": (
            "Interpolated local within-minute spread; not a measured standard "
            "deviation and not a formal uncertainty of the interpolated mean."
        ),
        "records": records,
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return outputs, manifest_path


def main() -> None:
    outputs, manifest = build()
    for output in outputs:
        print(f"wrote {output}")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
