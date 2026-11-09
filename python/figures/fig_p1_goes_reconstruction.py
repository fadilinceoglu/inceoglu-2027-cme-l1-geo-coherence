"""Figure 7: P1 component-only fits, grouped TreeSHAP and feature heatmaps.

Reads current-run derived products without refitting. Family and axis stacks
are alternative partitions of the same signed SHAP terms.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.paths import FIGURE_DIR, RESULTS_DIR
from figures import style
from figures.reconstruction_common import (
    AXES, AXIS_COLORS, DISPLAY_FAMILIES, FAMILY_LABELS, SCALE_COLORS,
    SCALES, SPACECRAFT, _panel_letter,
    apply_style, draw_fits, draw_signed_stacks, format_time_axis,
    load_products, shared_signed_limits, stack_legend, validate_products,
)

RESULT_PATH = RESULTS_DIR / "p1_goes_reconstruction.json"
SAMPLE_PATH = RESULTS_DIR / "p1_goes_reconstruction_event_samples.csv"
OUTPUT_PDF = FIGURE_DIR / "Fig7_p1_goes_reconstruction.pdf"


def render_attribution_figure(*, result, samples, output_path: Path) -> str:
    """Render the eight-panel component-model view."""
    frames = validate_products(result, samples, source="artemis_p1", input_kind="components")
    apply_style()
    fig = plt.figure(figsize=(style.FIGURE_WIDTH_MM / style.MM_PER_INCH, 195 / style.MM_PER_INCH))
    grid = fig.add_gridspec(
        4, 2, height_ratios=(1.05, 1, 1, .94),
        left=.12, right=.98, top=.965, bottom=.06, hspace=.50, wspace=.24,
    )
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(4)])
    draw_fits(axes[0], frames, result)
    for row, grouping in enumerate(("family", "axis"), 1):
        keys = SCALES if grouping == "family" else AXES
        columns = ([f"{key}_model_contribution" for key in keys]
                   if grouping == "family" else [f"axis_{key}_model_contribution" for key in keys])
        colors = ([SCALE_COLORS[key] for key in keys]
                  if grouping == "family" else [AXIS_COLORS[key] for key in keys])
        labels = ([FAMILY_LABELS[key] for key in keys]
                  if grouping == "family" else [f"GSM {key.upper()}" for key in keys])
        limits = []
        for col, craft in enumerate(SPACECRAFT):
            limits.append(draw_signed_stacks(axes[row, col], frames[craft], columns, colors, labels))
            format_time_axis(axes[row, col], frames[craft].index, bottom=row == 2)
        shared_signed_limits(axes[row], limits)
        axes[row, 0].set_ylabel(f"{'Family' if grouping == 'family' else 'Axis'} contribution\n(nT)")
        stack_legend(axes[row, 0], count=len(keys) + 1)
    matrices = {
        craft: np.array([
            [np.abs(frame[f"{family}__b{axis}_f_shap_nT"]).mean() for axis in AXES]
            for family in DISPLAY_FAMILIES
        ]) for craft, frame in frames.items()
    }
    maximum = max(float(matrix.max()) for matrix in matrices.values())
    for col, craft in enumerate(SPACECRAFT):
        axis, matrix = axes[3, col], matrices[craft]
        axis.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=max(maximum, np.finfo(float).eps))
        axis.set_xticks(range(3), [f"GSM {key.upper()}" for key in AXES])
        axis.set_yticks(range(4), [FAMILY_LABELS[key] for key in DISPLAY_FAMILIES] if col == 0 else [])
        axis.tick_params(length=0, pad=3)
        axis.set_title("Mean |individual SHAP| (nT)", fontsize=7, pad=5)
        for row in range(4):
            for component in range(3):
                value = matrix[row, component]
                axis.text(component, row, f"{value:.1f}", ha="center", va="center", fontsize=7,
                          color="white" if value > .53 * maximum else "#173249")
    for index, axis in enumerate(axes.flat):
        _panel_letter(axis, chr(97 + index))
    return str(style.save_pdf(fig, output_path, strict_pdf=True))


def build(output_path: str | Path = OUTPUT_PDF) -> str:
    result, samples = load_products(RESULT_PATH, SAMPLE_PATH)
    return render_attribution_figure(result=result, samples=samples, output_path=Path(output_path))


if __name__ == "__main__":
    print(build())
