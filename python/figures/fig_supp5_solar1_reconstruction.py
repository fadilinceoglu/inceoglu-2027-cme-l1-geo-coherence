"""Supplementary Figure 5: SOLAR-1 four-scalar waveform reconstruction."""
from __future__ import annotations

from pathlib import Path

from config.paths import FIGURE_DIR, RESULTS_DIR
from figures.reconstruction_common import load_products, render_scalar_figure

RESULT_PATH = RESULTS_DIR / "solar1_goes_reconstruction.json"
SAMPLE_PATH = RESULTS_DIR / "solar1_goes_reconstruction_event_samples.csv"
OUTPUT_PDF = FIGURE_DIR / "SupFig5_solar1_reconstruction.pdf"


def build(output_path: str | Path = OUTPUT_PDF) -> str:
    result, samples = load_products(RESULT_PATH, SAMPLE_PATH)
    return render_scalar_figure(result=result, samples=samples, source="swfo", output_path=output_path)


if __name__ == "__main__":
    print(build())
