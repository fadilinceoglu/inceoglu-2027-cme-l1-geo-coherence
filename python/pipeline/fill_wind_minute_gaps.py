"""Fill isolated Wind magnetic gaps in the one-minute product.

The source file is never overwritten.  Only magnetic means and their
within-minute sample standard deviations are interpolated.  Native sample
counts remain zero at filled minutes, and ``mag_interpolated`` records the
locations for plotting and downstream exclusion/sensitivity checks.
"""

from __future__ import annotations

from config.paths import PROCESSED_DIR

import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from config import study
from analysis.product_io import utc_text as _utc_text


PYTHON_ROOT: Final = Path(__file__).resolve().parents[1]
INPUT: Final = PROCESSED_DIR / "common_cadence_1min/wind.csv"
OUTPUT: Final = (
    PROCESSED_DIR / "common_cadence_1min/wind_gapfilled.csv"
)
MANIFEST: Final = (
    PROCESSED_DIR
    / "common_cadence_1min/wind_gapfill_manifest.json"
)
MAG_MEANS: Final = ("bx", "by", "bz", "bt")
MAG_STDS: Final = tuple(f"{column}_std" for column in MAG_MEANS)
MAG_COUNTS: Final = tuple(f"{column}_samples" for column in MAG_MEANS)
FIGURE2_START: Final = pd.Timestamp(study.FIGURE2_START)
FIGURE2_STOP: Final = pd.Timestamp(study.FIGURE2_STOP)
MAGNETIC_INTERPOLATION_SEMANTICS: Final = (
    "Interpolate each stored minute statistic independently. bt remains the "
    "interpolated mean of source |B| and is not recomputed as the norm of the "
    "interpolated component means; those quantities need not be equal."
)


def _position_runs(mask: pd.Series) -> list[np.ndarray]:
    positions = np.flatnonzero(mask.to_numpy())
    if len(positions) == 0:
        return []
    return list(np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1))


def fill_isolated_magnetic_gaps(
    frame: pd.DataFrame,
    *,
    maximum_gap_minutes: int = 1,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Linearly fill bounded magnetic gaps and return their audit records.

    The common-cadence schema stores ``mean(|B|)`` separately from the mean
    vector. Each statistic is interpolated in its own time series, so the
    filled ``bt`` is deliberately not ``norm([bx, by, bz])``.
    """

    required = set((*MAG_MEANS, *MAG_STDS, *MAG_COUNTS))
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    source = frame.copy()
    source.index = pd.DatetimeIndex(source.index)
    source.index = (
        source.index.tz_localize("UTC")
        if source.index.tz is None
        else source.index.tz_convert("UTC")
    )
    if not source.index.is_monotonic_increasing or source.index.has_duplicates:
        raise ValueError("time index must be sorted and unique")
    cadence = source.index.to_series().diff().dropna()
    if not cadence.eq(pd.Timedelta("1min")).all():
        raise ValueError("input must use a complete one-minute UTC grid")

    missing_by_component = source.loc[:, list(MAG_MEANS)].isna()
    gap_mask = missing_by_component.iloc[:, 0]
    if not missing_by_component.eq(gap_mask, axis=0).all().all():
        raise ValueError("magnetic means do not share the same missing minutes")
    runs = _position_runs(gap_mask)
    if runs and max(map(len, runs)) > maximum_gap_minutes:
        raise ValueError(
            f"longest gap is {max(map(len, runs))} min; "
            f"limit is {maximum_gap_minutes} min"
        )
    if not source.loc[gap_mask, list(MAG_COUNTS)].eq(0).all().all():
        raise ValueError("a missing magnetic minute has a nonzero sample count")

    result = source.copy()
    audit: list[dict[str, object]] = []
    fill_columns = (*MAG_MEANS, *MAG_STDS)
    for run in runs:
        first = int(run[0])
        last = int(run[-1])
        if first == 0 or last == len(source) - 1:
            raise ValueError("cannot interpolate an unbounded edge gap")
        left = source.index[first - 1]
        right = source.index[last + 1]
        neighbors = source.iloc[[first - 1, last + 1]].loc[:, fill_columns]
        if not np.isfinite(neighbors.to_numpy(dtype=float)).all():
            raise ValueError(f"gap at {source.index[first]} lacks finite bounds")
        for position in run:
            time = source.index[int(position)]
            fraction = float((time - left) / (right - left))
            result.loc[time, list(fill_columns)] = (
                neighbors.iloc[0]
                + fraction * (neighbors.iloc[1] - neighbors.iloc[0])
            ).to_numpy()
            audit.append(
                {
                    "utc": _utc_text(time),
                    "left_observation_utc": _utc_text(left),
                    "right_observation_utc": _utc_text(right),
                    "interpolation_fraction": fraction,
                    "left_sample_counts": {
                        column: int(source.iloc[first - 1][f"{column}_samples"])
                        for column in MAG_MEANS
                    },
                    "right_sample_counts": {
                        column: int(source.iloc[last + 1][f"{column}_samples"])
                        for column in MAG_MEANS
                    },
                    "in_figure2_window": bool(
                        FIGURE2_START <= time < FIGURE2_STOP
                    ),
                }
            )

    result["mag_interpolated"] = gap_mask.astype("int8")
    if result.loc[gap_mask, list(fill_columns)].isna().any().any():
        raise AssertionError("interpolation left an unfilled magnetic statistic")
    if not result.loc[gap_mask, list(MAG_COUNTS)].eq(0).all().all():
        raise AssertionError("interpolation changed native sample counts")
    return result, audit


def build(
    input_path: Path = INPUT,
    output_path: Path = OUTPUT,
    manifest_path: Path = MANIFEST,
) -> tuple[Path, Path]:
    frame = pd.read_csv(input_path, parse_dates=["time"], index_col="time")
    filled, audit = fill_isolated_magnetic_gaps(
        frame,
        maximum_gap_minutes=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filled.to_csv(
        output_path,
        index_label="time",
        date_format="%Y-%m-%dT%H:%M:%SZ",
        float_format="%.12g",
        na_rep="",
        lineterminator="\n",
    )
    manifest = {
        "spacecraft": "Wind",
        "input": {"path": str(input_path), "size_bytes": input_path.stat().st_size},
        "output": {"path": str(output_path), "size_bytes": output_path.stat().st_size},
        "method": "linear interpolation in UTC between adjacent observed minutes",
        "magnetic_statistic_semantics": MAGNETIC_INTERPOLATION_SEMANTICS,
        "bt_definition": (
            "interpolated arithmetic minute mean of source |B|; not the norm "
            "of interpolated arithmetic component means"
        ),
        "maximum_gap_minutes": 1,
        "columns_filled": list((*MAG_MEANS, *MAG_STDS)),
        "sample_counts_at_filled_minutes": 0,
        "plot_flag_column": "mag_interpolated",
        "filled_minute_count": len(audit),
        "figure2_window": {
            "start_utc": _utc_text(FIGURE2_START),
            "stop_utc_exclusive": _utc_text(FIGURE2_STOP),
            "filled_minute_count": sum(
                bool(record["in_figure2_window"]) for record in audit
            ),
        },
        "standard_deviation_note": (
            "Interpolated local within-minute spread; not a measured standard "
            "deviation and not a formal uncertainty of the interpolated mean."
        ),
        "filled_minutes": audit,
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return output_path, manifest_path


def main() -> None:
    output, manifest = build()
    print(f"wrote {output}")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
