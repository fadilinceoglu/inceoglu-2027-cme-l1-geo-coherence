"""Supplementary Figure 2 — pairwise lag and waveform-coherence triangles.

Caption information (kept here rather than rendered inside the figure):

Single-triangle pairwise matrices of the joint Figure-5 inference.
For each waveform family, the 21 unique spacecraft pairs
are shown once: cell colour and the upper number are the peak full-record
normalized cross-correlation r; the lower number is the signed peak lag
Δt in minutes of the ROW spacecraft relative to the COLUMN spacecraft
(positive: the row spacecraft, which is the later of the pair in
propagation order, observes the structure later).  The synchronized
moving-block 95% lag intervals are validated here and retained in the
accompanying results table rather than crowded into the cells.  No pair
is discarded because its lag is uncertain.

Input
-----
<output-dir>/results/waveform_inference.csv

Outputs
-------
<output-dir>/figures/SupFig2_lag_coherence.pdf
"""

from __future__ import annotations

from pathlib import Path

from config.paths import RESULTS_DIR, FIGURE_DIR
from typing import Final

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import study
from config.waveform_bands import WAVEFORM_FAMILIES, WAVEFORM_FAMILY_LABELS
from analysis.waveform_inference import read_inference_table
from figures import style
from figures.style import FIGURE_WIDTH_MM, MM_PER_INCH, TEXT_PT


INPUT: Final = (
    RESULTS_DIR / "waveform_inference.csv"
)
OUT: Final = (
    FIGURE_DIR
    / "SupFig2_lag_coherence.pdf"
)

STRUCTURES = WAVEFORM_FAMILIES
STRUCTURE_LABELS: Final = WAVEFORM_FAMILY_LABELS
CRAFT = study.UPSTREAM_SPACECRAFT
CRAFT_LABELS: Final = study.CRAFT_LABELS

FIGURE_HEIGHT_MM: Final = 180.0
SMALL_TEXT_PT: Final = 6.4
CELL_TEXT_PT: Final = 6.0
INK: Final = study.INK


def _publication_style() -> None:
    style.apply_publication_style(text_size=TEXT_PT, tick_size=SMALL_TEXT_PT, overrides={
        'axes.titlesize': TEXT_PT,
        'axes.linewidth': 0.8,
        'axes.grid': False,
    })


def load_results(path: Path | str = INPUT) -> pd.DataFrame:
    table = read_inference_table(path)
    if not table["normalized_cross_correlation"].between(0.0, 1.0).all():
        raise ValueError("peak normalized correlations must lie in [0, 1]")
    return table


def _panel_arrays(
    table: pd.DataFrame,
    structure: str,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (len(CRAFT), len(CRAFT))
    correlation = np.full(shape, np.nan)
    lag = np.full(shape, np.nan)
    craft_index = {craft: index for index, craft in enumerate(CRAFT)}
    np.fill_diagonal(correlation, 1.0)
    np.fill_diagonal(lag, 0.0)
    subset = table.loc[table["structure"] == structure]
    for row in subset.itertuples(index=False):
        source = craft_index[row.source_craft]
        target = craft_index[row.target_craft]
        value = float(row.normalized_cross_correlation)
        selected_lag = float(row.lag_minutes)
        correlation[source, target] = value
        lag[source, target] = selected_lag
        correlation[target, source] = value
        lag[target, source] = -selected_lag
    return correlation, lag


def _text_color(value: float, norm, cmap) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.48 else INK


def plot(
    input_path: Path | str = INPUT,
    output_path: Path | str = OUT,
) -> Path:
    table = load_results(input_path)
    _publication_style()

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(
            FIGURE_WIDTH_MM / MM_PER_INCH,
            FIGURE_HEIGHT_MM / MM_PER_INCH,
        ),
        constrained_layout=True,
    )
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = matplotlib.colormaps["viridis"].copy()
    count = len(CRAFT)
    mesh = None
    for panel_index, (ax, structure) in enumerate(zip(axes.flat, STRUCTURES)):
        correlations, lags = _panel_arrays(table, structure)
        shown = np.full((count, count), np.nan)
        for row in range(count):
            for column in range(count):
                if row > column:
                    shown[row, column] = correlations[row, column]
        mesh = ax.pcolormesh(
            np.arange(count + 1),
            np.arange(count + 1),
            np.ma.masked_invalid(shown),
            cmap=cmap,
            norm=norm,
            edgecolors="white",
            linewidth=0.6,
        )
        for row in range(count):
            for column in range(count):
                if row <= column:
                    continue
                value = float(correlations[row, column])
                # Display the column->row lag so the row spacecraft (the
                # later of the pair in propagation order) reads positive
                # when it observes the structure later.
                delta = -float(lags[row, column]) + 0.0
                ax.text(
                    column + 0.5,
                    row + 0.5,
                    f"{value:.2f}\n{delta:+.0f}",
                    ha="center",
                    va="center",
                    fontsize=CELL_TEXT_PT,
                    linespacing=0.90,
                    color=_text_color(value, norm, cmap),
                )
        ax.set_xlim(0, count - 1)
        ax.set_ylim(count, 1)
        ax.set_xticks(np.arange(count - 1) + 0.5)
        ax.set_xticklabels(
            [CRAFT_LABELS[craft] for craft in CRAFT[:-1]],
            rotation=25,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_yticks(np.arange(1, count) + 0.5)
        ax.set_yticklabels([CRAFT_LABELS[craft] for craft in CRAFT[1:]])
        ax.text(
            0.02,
            1.02,
            f"({chr(ord('a') + panel_index)}) {STRUCTURE_LABELS[structure]}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=TEXT_PT,
            fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.tick_params(length=0, pad=2.0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    if mesh is None:
        raise RuntimeError("no triangle panels were drawn")
    colorbar = figure.colorbar(mesh, ax=axes, shrink=0.55, pad=0.02)
    colorbar.set_label("Peak full-record NCC, r")
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))

    pdf = style.save_pdf(figure, output_path)
    return pdf


def main() -> None:
    print(plot())


if __name__ == "__main__":
    main()
