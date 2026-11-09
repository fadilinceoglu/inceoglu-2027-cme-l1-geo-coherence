"""Recompute shock-aligned, common-activity wavelet coherence and persistence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.wavelet_coherence import Method, analyze_wavelet_coherence, validate_wavelet_coherence
from common.io import atomic_output_paths
from config.paths import RESULTS_DIR

OUTPUT_JSON = RESULTS_DIR / "wavelet_coherence.json"
OUTPUT_CSV = RESULTS_DIR / "wavelet_coherence.csv"
OUTPUT_DRAWS = RESULTS_DIR / "wavelet_coherence_draws.npz"
OUTPUT_FITS = RESULTS_DIR / "wavelet_coherence_persistence_fits.csv"


def build(*, output_json: Path | str = OUTPUT_JSON, output_csv: Path | str = OUTPUT_CSV,
          output_draws: Path | str = OUTPUT_DRAWS, output_fits: Path | str = OUTPUT_FITS,
          method: Method = Method(), cache_root: Path | str | None = None) -> tuple[Path, Path, Path, Path]:
    result, raw = analyze_wavelet_coherence(method=method, cache_root=cache_root)
    paths = tuple(Path(path) for path in (output_json, output_csv, output_draws, output_fits))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    result["meta"]["raw_draw_archive_path"] = str(paths[2])
    table = validate_wavelet_coherence(result)
    fits = pd.DataFrame([{key: value for key, value in row.items() if not key.startswith("curve_")}
                         for row in result["coherence_length_records"] + result["white_floor_sensitivity_records"]])
    with atomic_output_paths(paths, keep_suffix=True) as temporary:
        temporary[0].write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        table.to_csv(temporary[1], index=False, float_format="%.12g")
        np.savez_compressed(temporary[2], meta__schema_version=np.asarray(result["meta"]["schema_version"]), **raw)
        fits.to_csv(temporary[3], index=False, float_format="%.12g")
    return paths


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
