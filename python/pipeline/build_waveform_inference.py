"""Build waveform uncertainty, significance, persistence fits, and resampling draws."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from common.io import atomic_output_paths
from config.paths import RESULTS_DIR

from analysis.waveform_inference import (
    BOOTSTRAP_DRAWS,
    NULL_SURROGATES,
    analyze_joint_inference,
    validate_inference,
)


OUTPUT_JSON: Final = RESULTS_DIR / "waveform_inference.json"
OUTPUT_CSV: Final = OUTPUT_JSON.with_suffix(".csv")
OUTPUT_DRAWS: Final = RESULTS_DIR / "waveform_inference_draws.npz"
OUTPUT_FITS: Final = RESULTS_DIR / "waveform_persistence_fits.csv"


def build(
    *,
    output_json: Path | str = OUTPUT_JSON,
    output_csv: Path | str = OUTPUT_CSV,
    output_draws: Path | str = OUTPUT_DRAWS,
    output_fits: Path | str = OUTPUT_FITS,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    null_surrogates: int = NULL_SURROGATES,
) -> tuple[Path, Path, Path, Path]:
    result, raw = analyze_joint_inference(
        bootstrap_draws=bootstrap_draws,
        null_surrogates=null_surrogates,
    )

    draw_path = Path(output_draws)
    draw_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_paths(
        (draw_path,), temporary_paths=(draw_path.with_suffix(".tmp.npz"),)
    ) as (draw_temporary,):
        np.savez_compressed(
            draw_temporary,
            meta__schema_version=np.asarray(
                result["meta"]["schema_version"], dtype=np.int64
            ),
            meta__bootstrap_draws=np.asarray(bootstrap_draws, dtype=np.int64),
            meta__null_surrogates=np.asarray(
                null_surrogates, dtype=np.int64
            ),
            **raw,
        )
    result["meta"]["raw_draw_archive_path"] = str(draw_path)
    table = validate_inference(result)
    fits = pd.DataFrame(
        [
            {
                key: value
                for key, value in record.items()
                if not key.startswith("curve_")
                and not isinstance(value, (dict, list))
            }
            for record in result["coherence_length_records"]
        ]
    )
    json_path = Path(output_json)
    csv_path = Path(output_csv)
    fits_path = Path(output_fits)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_paths(
        (json_path, csv_path, fits_path),
        temporary_paths=(json_path.with_suffix(".json.tmp"),
                         csv_path.with_suffix(".csv.tmp"),
                         fits_path.with_suffix(".csv.tmp")),
    ) as (json_temporary, csv_temporary, fits_temporary):
        json_temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        table.to_csv(csv_temporary, index=False, float_format="%.12g")
        fits.to_csv(fits_temporary, index=False, float_format="%.12g")
    return json_path, csv_path, draw_path, fits_path


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
