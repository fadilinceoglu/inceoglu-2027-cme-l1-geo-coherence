"""Build the four-day pairwise waveform comparisons (JSON and CSV)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from common.io import atomic_output_paths
from config.paths import RESULTS_DIR

from analysis.waveform_comparisons import (
    analyze_canonical_pairs,
    records_to_table,
    validate_result,
)


OUTPUT_JSON: Final = RESULTS_DIR / "waveform_comparisons.json"
OUTPUT_CSV: Final = OUTPUT_JSON.with_suffix(".csv")


def build(
    *,
    output_json: Path | str = OUTPUT_JSON,
    output_csv: Path | str = OUTPUT_CSV,
) -> tuple[Path, Path]:
    """Calculate once, validate, and atomically replace both cache formats."""

    result = analyze_canonical_pairs()
    validate_result(result)
    table = records_to_table(result)
    json_path = Path(output_json)
    csv_path = Path(output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_paths((json_path, csv_path)) as (json_temporary, csv_temporary):
        json_temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        table.to_csv(csv_temporary, index=False, float_format="%.12g")
    return json_path, csv_path


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
