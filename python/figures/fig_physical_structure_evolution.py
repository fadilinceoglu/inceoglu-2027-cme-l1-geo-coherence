"""Figure 4 — extent, amplitude and variance ratio of waveforms.

Columns show the four convected-size waveform families; rows show scalar
power-support width W90 (a-d), filtered-vector RMS amplitude (e-h),
scalar-to-vector centered variance ratio (i-l). All values come from the
current-run chain-waveform metric product and use spacecraft- and
family-specific intervals. Trend width and RMS retain the mean field;
the variance ratio centers each low-pass channel over its selected interval.

Solar Orbiter, the four L1 observers, ARTEMIS-P1 and ARTEMIS-P2 are shown
in observing order. Grey lines connect Solar Orbiter, the L1 mean and P1;
the dotted extension to P2 marks the lunar-wake comparison. These are
visual guides, not a fit to radial evolution. Each row uses a common
logarithmic y-axis range across its four panels, calculated from the saved
measurements with 14% padding in log space. The builder reads only the validated
metric CSV and does not recalculate waveforms.

Inputs:
    <output-dir>/results/chain_waveform_metrics.csv

Outputs:
    <output-dir>/figures/Fig4_physical_structure_evolution.pdf

Run:
    cd python
    PYTHONPATH=. <venv-python> figures/fig_physical_structure_evolution.py
"""

from __future__ import annotations

import string
from pathlib import Path

from config.paths import RESULTS_DIR, FIGURE_DIR
from typing import Final

import matplotlib

# Reproducible on headless CI and publication servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

from config import study
from config.waveform_bands import WAVEFORM_FAMILIES, WAVEFORM_FAMILY_LABELS
from analysis.chain_waveform_product import read_metrics
from figures import style
from figures.style import (
    FIGURE_WIDTH_MM,
    MM_PER_INCH,
    TEXT_PT as FIGURE_TEXT_PT,
    SMALL_TEXT_PT as FIGURE_SMALL_TEXT_PT,
)


METRICS: Final = (
    RESULTS_DIR / "chain_waveform_metrics.csv"
)
OUT_DIR: Final = FIGURE_DIR
OUTPUT_STEM: Final = "Fig4_physical_structure_evolution"

CRAFT_ORDER = study.UPSTREAM_SPACECRAFT
L1_CRAFT = study.L1_SPACECRAFT
STRUCTURE_ORDER = WAVEFORM_FAMILIES
FAMILY_TITLES: Final = WAVEFORM_FAMILY_LABELS


def family_title(family: str, rows: pd.DataFrame) -> str:
    edges = rows[["filter_minimum_re", "filter_maximum_re"]].drop_duplicates()
    if len(edges) != 1:
        raise ValueError(f"inconsistent spacecraft filter edges for {family}")
    minimum, maximum = edges.iloc[0]
    maximum = None if pd.isna(maximum) else float(maximum)

    def shown(value: float) -> str:
        return f"{value:.3f}" if value < 10.0 else f"{value:.2f}"

    span = (
        f">{shown(float(minimum))}"
        if maximum is None
        else f"{shown(float(minimum))}–{shown(maximum)}"
    )
    return f"{FAMILY_TITLES[family]}\n{span} R$_E$"


FIGURE_HEIGHT_MM: Final = 120.0

STATION_X: Final = {
    "solar_orbiter": 0.0,
    "l1": 1.0,
    "artemis_p1": 2.0,
    "artemis_p2": 3.0,
}
L1_JITTER: Final = {"ace": -0.18, "wind": -0.06, "dscovr": 0.06, "swfo": 0.18}
STATION_TICK_LABELS: Final = (
    "SO\nflank",
    "L1\ncluster",
    "P1\nincident",
    "P2\nwake",
)

METRIC_ROWS: Final = (
    # (ylabel, column)
    ("W90 [R$_E$]", "power_support_width_re"),
    ("Vector RMS [nT]", "vector_rms_nT"),
    (
        "$\\mathrm{Var}(b_{t,f})/\\sum_i\\mathrm{Var}(b_{i,f})$",
        "scalar_to_vector_variance_ratio",
    ),
)


def _publication_style() -> None:
    """Apply publication settings at the declared final artwork size."""

    style.apply_publication_style(text_size=FIGURE_TEXT_PT, tick_size=FIGURE_SMALL_TEXT_PT, overrides={
        'axes.titlesize': FIGURE_TEXT_PT,
        'axes.linewidth': 0.55,
        'axes.grid': False,
        'xtick.major.width': 0.55,
        'ytick.major.width': 0.55,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
    })


load_metrics = read_metrics


def _marker(craft: str) -> str:
    if craft == "solar_orbiter":
        return "s"
    if craft == "artemis_p1":
        return "^"
    if craft == "artemis_p2":
        return "X"
    return "o"


def _color(craft: str) -> str:
    return study.SATELLITES[craft]["color"]


def _station_x(craft: str) -> float:
    if craft in L1_CRAFT:
        return STATION_X["l1"] + L1_JITTER[craft]
    return STATION_X[craft]


def _panel_values(
    rows: pd.DataFrame,
    *,
    column: str,
) -> dict[str, float]:
    """Return each spacecraft's validated waveform metric."""

    return {craft: float(rows.loc[craft, column]) for craft in CRAFT_ORDER}


def _draw_panel(
    ax,
    rows: pd.DataFrame,
    *,
    column: str,
) -> None:
    values = _panel_values(rows, column=column)

    l1_values = [values[c] for c in L1_CRAFT if np.isfinite(values[c])]
    l1_mean = float(np.mean(l1_values)) if l1_values else float("nan")

    path_x, path_y = [], []
    for xk, yk in (
        (STATION_X["solar_orbiter"], values["solar_orbiter"]),
        (STATION_X["l1"], l1_mean),
        (STATION_X["artemis_p1"], values["artemis_p1"]),
    ):
        if np.isfinite(yk):
            path_x.append(xk)
            path_y.append(yk)
    ax.plot(path_x, path_y, color="#9a9a9a", lw=0.8, alpha=0.85, zorder=1)
    if np.isfinite(values["artemis_p1"]) and np.isfinite(values["artemis_p2"]):
        ax.plot(
            [STATION_X["artemis_p1"], STATION_X["artemis_p2"]],
            [values["artemis_p1"], values["artemis_p2"]],
            color="#9a9a9a",
            lw=0.8,
            ls=(0, (1.2, 1.6)),
            alpha=0.85,
            zorder=1,
        )
    if np.isfinite(l1_mean):
        ax.hlines(
            l1_mean,
            STATION_X["l1"] - 0.28,
            STATION_X["l1"] + 0.28,
            color="#555555",
            lw=0.7,
            alpha=0.6,
            zorder=2,
        )

    for craft in CRAFT_ORDER:
        value = values[craft]
        x = _station_x(craft)
        if not np.isfinite(value):
            continue
        ax.plot(
            x,
            value,
            marker=_marker(craft),
            ms=4.6,
            ls="none",
            mfc=_color(craft),
            mec="white",
            mew=0.4,
            zorder=4,
        )

    if column == "scalar_to_vector_variance_ratio":
        ax.axhline(1.0, color="#bbbbbb", lw=0.5, ls=(0, (2, 2)), zorder=0)


def build(
    out_dir: str | Path = OUT_DIR,
    *,
    data_path: str | Path = METRICS,
) -> dict[str, Path]:
    """Build the single propagation-ladder figure from one frozen CSV read."""

    _publication_style()
    metrics = load_metrics(data_path)

    fig, axes = plt.subplots(
        len(METRIC_ROWS),
        len(STRUCTURE_ORDER),
        figsize=(
            FIGURE_WIDTH_MM / MM_PER_INCH,
            FIGURE_HEIGHT_MM / MM_PER_INCH,
        ),
        sharex=True,
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )
    fig.get_layout_engine().set(
        w_pad=1.2 / MM_PER_INCH,
        h_pad=1.0 / MM_PER_INCH,
        wspace=0.02,
        hspace=0.02,
    )

    letters = iter(string.ascii_lowercase)
    for i, (ylabel, column) in enumerate(METRIC_ROWS):
        values = metrics[column].to_numpy(dtype=float)
        if np.any(values <= 0.0):
            plt.close(fig)
            if column == "scalar_to_vector_variance_ratio" and np.any(values == 0.0):
                raise ValueError(
                    "scalar_to_vector_variance_ratio contains zero, which is valid "
                    "but cannot be displayed on Figure 4's logarithmic axis"
                )
            raise ValueError(f"{column} requires positive values for a logarithmic axis")
        lower, upper = np.log10([values.min(), values.max()])
        padding = 0.14 * (upper - lower)
        row_axis = axes[i, 0]
        row_axis.set_yscale("log")
        row_axis.set_ylim(10 ** (lower - padding), 10 ** (upper + padding))
        row_axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=12))
        row_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
        row_axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9), numticks=100))
        row_axis.yaxis.set_minor_formatter(NullFormatter())
        for j, family in enumerate(STRUCTURE_ORDER):
            ax = axes[i, j]
            rows = metrics.loc[family].loc[list(CRAFT_ORDER)]
            _draw_panel(
                ax,
                rows,
                column=column,
            )
            # Keep all panel labels inside the upper-left corner of the axes.
            ax.text(
                0.02,
                0.98,
                f"({next(letters)})",
                transform=ax.transAxes,
                fontsize=FIGURE_TEXT_PT,
                fontweight="bold",
                ha="left",
                va="top",
            )
            if i == 0:
                ax.set_title(family_title(family, rows), fontsize=FIGURE_TEXT_PT, pad=8.0)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=FIGURE_TEXT_PT)
            ax.grid(axis="y", which="both", lw=0.3, alpha=0.22)
            ax.tick_params(length=2.2, pad=2.0)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.set_xlim(-0.55, 3.55)
    for ax in axes[-1, :]:
        ax.set_xticks(list(STATION_X.values()))
        ax.set_xticklabels(STATION_TICK_LABELS, fontsize=FIGURE_SMALL_TEXT_PT)

    handles = [
        plt.Line2D(
            [],
            [],
            marker=_marker(craft),
            ls="none",
            mfc=_color(craft),
            mec="white",
            mew=0.4,
            ms=4.6,
            label=study.SATELLITES[craft]["name"],
        )
        for craft in CRAFT_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=7,
        fontsize=FIGURE_SMALL_TEXT_PT,
        frameon=False,
        handletextpad=0.15,
        columnspacing=0.9,
        borderaxespad=0.1,
    )

    output_directory = Path(out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    pdf_output = output_directory / f"{OUTPUT_STEM}.pdf"
    style.save_pdf(fig, pdf_output, format="pdf")
    return {"physical_structure_evolution": pdf_output}


if __name__ == "__main__":
    print(build())
