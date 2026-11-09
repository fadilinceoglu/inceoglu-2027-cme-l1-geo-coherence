"""Figure 8: signed component contributions within each P1 waveform family."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.paths import FIGURE_DIR, RESULTS_DIR
from figures import style
from figures.reconstruction_common import (
    AXES, AXIS_COLORS, DISPLAY_FAMILIES, FAMILY_LABELS, SPACECRAFT,
    _panel_letter, apply_style, draw_fits, format_time_axis, load_products,
    shared_signed_limits, validate_products,
)

RESULT_PATH = RESULTS_DIR / "p1_goes_reconstruction.json"
SAMPLE_PATH = RESULTS_DIR / "p1_goes_reconstruction_event_samples.csv"
OUTPUT_PDF = FIGURE_DIR / "Fig8_p1_goes_component_contributions.pdf"


def render_component_figure(*, result, samples, output_path) -> str:
    frames = validate_products(result, samples, source="artemis_p1", input_kind="components")
    apply_style()
    fig = plt.figure(figsize=(style.FIGURE_WIDTH_MM / style.MM_PER_INCH, 205 / style.MM_PER_INCH))
    grid = fig.add_gridspec(5, 2, height_ratios=(1.05, .83, .83, .83, .83),
                           left=.12, right=.98, top=.96, bottom=.08, hspace=.34, wspace=.24)
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(5)])
    draw_fits(axes[0], frames, result)
    for row, family in enumerate(DISPLAY_FAMILIES, 1):
        columns = [f"{family}__b{component}_f_shap_nT" for component in AXES]
        maximum = max(float(np.abs(frame[columns]).to_numpy().max()) for frame in frames.values())
        for col, craft in enumerate(SPACECRAFT):
            axis, frame = axes[row, col], frames[craft]
            for component, column in zip(AXES, columns, strict=True):
                axis.plot(frame.index, frame[column], color=AXIS_COLORS[component], ls="-", lw=.8, label=f"GSM {component.upper()}")
            axis.axhline(0, color=".35", lw=.5)
            format_time_axis(axis, frame.index, bottom=row == 4)
        shared_signed_limits(axes[row], [maximum])
        axes[row, 0].set_ylabel(FAMILY_LABELS[family] + "\nSHAP (nT)")
    axes[1, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3,
                      frameon=False, handlelength=2, borderaxespad=0)
    for index, axis in enumerate(axes.flat):
        _panel_letter(axis, chr(97 + index))
    return str(style.save_pdf(fig, output_path, strict_pdf=True))


def build(output_path: str | Path = OUTPUT_PDF) -> str:
    result, samples = load_products(RESULT_PATH, SAMPLE_PATH)
    return render_component_figure(result=result, samples=samples, output_path=Path(output_path))


if __name__ == "__main__":
    print(build())
