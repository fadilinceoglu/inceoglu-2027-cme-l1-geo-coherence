"""Recompute GSM Bz persistence within the scalar-aligned waveform families."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.bz_persistence import Method, analyze_bz_persistence, validate_bz_persistence
from common.io import atomic_output_paths
from config.paths import RESULTS_DIR


def build(*, method: Method = Method(), output_directory: Path | str = RESULTS_DIR) -> tuple[Path, ...]:
    result, raw = analyze_bz_persistence(method=method)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = tuple(output / name for name in ("bz_persistence.json", "bz_persistence.csv",
                                            "bz_persistence_draws.npz", "bz_persistence_fits.csv"))
    result["meta"]["raw_draw_archive_path"] = str(paths[2])
    table = validate_bz_persistence(result)
    fits = pd.DataFrame([{key: value for key, value in record.items()
                          if not key.startswith("curve_") and not isinstance(value, (list, dict))}
                         for record in result["coherence_length_records"]])
    with atomic_output_paths(paths, keep_suffix=True) as temporary:
        temporary[0].write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        table.to_csv(temporary[1], index=False, float_format="%.12g")
        np.savez_compressed(temporary[2], meta__schema_version=np.asarray(result["meta"]["schema_version"]), **raw)
        fits.to_csv(temporary[3], index=False, float_format="%.12g")
    return paths


if __name__ == "__main__":
    for path in build():
        print(path)
