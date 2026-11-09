"""Supplementary Figure 3 — pair-resolved waveform retention metrics.

This figure shows the 15 physical chain links individually:
four SO-to-L1 links, six within-L1 links, four L1-to-P1 links, and P1-to-P2.
It compares RMS amplitude ratios and aligned waveform differences for the
four scalar waveform families. Only these two measurements enter the figure.

Caption information: RMS-ratio and
NRMSE bars are 95% synchronized moving-block intervals (1,000 draws).

Input
-----
<output-dir>/results/waveform_inference.csv

Outputs
-------
<output-dir>/figures/SupFig3_information_retention.pdf
"""

from __future__ import annotations

from pathlib import Path

from config.paths import RESULTS_DIR, FIGURE_DIR
from typing import Final

import matplotlib

matplotlib.use("Agg")

from matplotlib.lines import Line2D
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
OUTPUT: Final = (
    FIGURE_DIR
    / "SupFig3_information_retention.pdf"
)

STRUCTURES = WAVEFORM_FAMILIES
STRUCTURE_LABELS: Final = WAVEFORM_FAMILY_LABELS
STRUCTURE_STYLE: Final = {
    "discontinuity": {"color": "#0072B2", "marker": "o"},
    "substructure": {"color": "#D55E00", "marker": "s"},
    "sheath": {"color": "#009E73", "marker": "^"},
    "trend": {"color": "#CC79A7", "marker": "D"},
}
STRUCTURE_OFFSETS: Final = {
    "discontinuity": 0.27,
    "substructure": 0.09,
    "sheath": -0.09,
    "trend": -0.27,
}

CRAFT_LABELS: Final = study.CRAFT_LABELS
L1 = study.L1_SPACECRAFT

# Within-L1 labels are undirected comparisons; arrows denote the physical
# ordering used to organize the remaining rows, not inferred information flow.
LINK_GROUPS: Final = (
    (
        "SO→L1",
        tuple(("solar_orbiter", target) for target in L1),
    ),
    (
        "Within L1",
        (
            ("ace", "wind"),
            ("ace", "dscovr"),
            ("ace", "swfo"),
            ("wind", "dscovr"),
            ("wind", "swfo"),
            ("dscovr", "swfo"),
        ),
    ),
    (
        "L1→P1",
        tuple((source, "artemis_p1") for source in L1),
    ),
    ("P1→P2", (("artemis_p1", "artemis_p2"),)),
)

METRICS: Final = (
    {
        "column": "rms_ratio_target_over_source",
        "low": "rms_ratio_target_over_source_bootstrap_ci95_low",
        "high": "rms_ratio_target_over_source_bootstrap_ci95_high",
        "title": "(a) RMS amplitude ratio",
        "label": "RMS(target) / RMS(source)",
        "reference": 1.0,
        "log": True,
    },
    {
        "column": "symmetric_normalized_rmse",
        "low": "symmetric_normalized_rmse_bootstrap_ci95_low",
        "high": "symmetric_normalized_rmse_bootstrap_ci95_high",
        "title": "(b) Aligned waveform difference",
        "label": "Symmetric normalized RMSE",
        "reference": 0.0,
        "log": False,
    },
)


FIGURE_HEIGHT_MM: Final = 195.0
SMALL_TEXT_PT: Final = 6.35
INK: Final = study.INK
GRID: Final = "#D8D8D8"


def _publication_style() -> None:
    style.apply_publication_style(text_size=TEXT_PT, tick_size=SMALL_TEXT_PT, overrides={
        'axes.titlesize': TEXT_PT,
        'axes.linewidth': 0.8,
        'axes.grid': False,
    })


def _chain_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(pair for _, links in LINK_GROUPS for pair in links)


def load_results(path: Path | str = INPUT) -> pd.DataFrame:
    """Select the 15 physical chain links from the validated inference product."""
    table = read_inference_table(path)
    pairs = set(_chain_pairs())
    selected = table.loc[
        [
            (str(source_craft), str(target_craft)) in pairs
            for source_craft, target_craft in zip(
                table["source_craft"], table["target_craft"]
            )
        ]
    ].copy()
    selected["pair"] = list(zip(selected["source_craft"], selected["target_craft"]))
    return selected


def _display_rows() -> tuple[list[dict[str, object]], dict[tuple[str, str], float]]:
    rows: list[dict[str, object]] = []
    pair_y: dict[tuple[str, str], float] = {}
    position = 0.0
    for group, links in LINK_GROUPS:
        rows.append({"kind": "header", "label": group, "y": position})
        position += 0.9
        for source, target in links:
            separator = "–" if source in L1 and target in L1 else "→"
            rows.append(
                {
                    "kind": "link",
                    "label": (
                        f"  {CRAFT_LABELS[source]}{separator}"
                        f"{CRAFT_LABELS[target]}"
                    ),
                    "y": position,
                }
            )
            pair_y[(source, target)] = position
            position += 1.0
        position += 0.35
    return rows, pair_y


def _metric_values(
    row: pd.Series,
    metric: dict[str, object],
) -> tuple[float, float, float]:
    value = float(row[str(metric["column"])])
    low = float(row[str(metric["low"])])
    high = float(row[str(metric["high"])])
    return value, low, high


def _metric_limits(
    table: pd.DataFrame,
    metric: dict[str, object],
) -> tuple[float, float]:
    values = np.asarray(
        [
            item
            for _, row in table.iterrows()
            for item in _metric_values(row, metric)
        ],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if not len(values):
        return (0.5, 2.0) if bool(metric["log"]) else (-0.1, 1.0)
    if bool(metric["log"]):
        positive = values[values > 0.0]
        if not len(positive):
            raise ValueError(f"{metric['column']} has no positive values")
        low = min(float(np.min(positive)), 1.0)
        high = max(float(np.max(positive)), 1.0)
        padding = 10.0 ** (0.08 * max(np.log10(high / low), 0.5))
        return low / padding, high * padding
    low = min(float(np.min(values)), float(metric["reference"]))
    high = max(float(np.max(values)), float(metric["reference"]))
    span = max(high - low, 0.1)
    if str(metric["column"]) == "symmetric_normalized_rmse":
        return 0.0, high + 0.08 * span
    return low - 0.08 * span, high + 0.08 * span


def _draw_interval(
    ax,
    *,
    low: float,
    high: float,
    y: float,
    color: str,
    log_axis: bool,
) -> None:
    """Draw an interval independently of the location of its point estimate."""

    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        return
    if log_axis and (low <= 0.0 or high <= 0.0):
        return
    ax.hlines(y, low, high, color=color, linewidth=0.65, alpha=0.62, zorder=2)
    cap = 0.055
    ax.vlines(
        [low, high],
        y - cap,
        y + cap,
        color=color,
        linewidth=0.65,
        alpha=0.62,
        zorder=2,
    )


def _draw_point(
    ax,
    row: pd.Series,
    *,
    y: float,
    metric: dict[str, object],
) -> None:
    structure = str(row["structure"])
    point_style = STRUCTURE_STYLE[structure]
    color = str(point_style["color"])
    marker = str(point_style["marker"])
    value, low, high = _metric_values(row, metric)
    _draw_interval(
        ax,
        low=low,
        high=high,
        y=y,
        color=color,
        log_axis=bool(metric["log"]),
    )

    ax.plot(
        value,
        y,
        marker=marker,
        markersize=4.0,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.8,
        linestyle="none",
        zorder=5,
    )


def plot(
    input_path: Path | str = INPUT,
    output_path: Path | str = OUTPUT,
) -> Path:
    table = load_results(input_path)
    rows, pair_y = _display_rows()
    _publication_style()

    figure, axes = plt.subplots(
        1,
        len(METRICS),
        figsize=(
            FIGURE_WIDTH_MM / MM_PER_INCH,
            FIGURE_HEIGHT_MM / MM_PER_INCH,
        ),
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.185,
        right=0.985,
        bottom=0.075,
        top=0.905,
        wspace=0.20,
    )

    y_ticks = [float(row["y"]) for row in rows]
    y_labels = [str(row["label"]) for row in rows]
    maximum_y = max(y_ticks) + 0.65
    minimum_y = min(y_ticks) - 0.55

    for panel_index, (ax, metric) in enumerate(zip(axes, METRICS)):
        xlim = _metric_limits(table, metric)
        if bool(metric["log"]):
            ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.axvline(
            float(metric["reference"]),
            color="#777777",
            linewidth=0.75,
            linestyle=(0, (3, 2)),
            zorder=0,
        )
        for display_row in rows:
            if display_row["kind"] == "header":
                ax.axhline(
                    float(display_row["y"]) + 0.43,
                    color=GRID,
                    lw=0.55,
                    zorder=0,
                )

        for record in table.itertuples(index=False):
            series = pd.Series(record._asdict())
            pair = tuple(series["pair"])
            y = pair_y[pair] + STRUCTURE_OFFSETS[str(series["structure"])]
            _draw_point(ax, series, y=y, metric=metric)

        ax.set_ylim(maximum_y, minimum_y)
        ax.set_title(str(metric["title"]), loc="left", pad=7.0)
        ax.set_xlabel(str(metric["label"]), labelpad=5.0)
        ax.tick_params(axis="y", length=0, pad=2.5)
        ax.tick_params(axis="x", width=0.7, length=3.0)
        ax.spines["left"].set_visible(panel_index == 0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, lw=0.45, alpha=0.75)
        if panel_index == 0:
            ax.set_yticks(y_ticks, y_labels)
            for tick, display_row in zip(ax.get_yticklabels(), rows):
                if display_row["kind"] == "header":
                    tick.set_fontweight("bold")
                    tick.set_color(INK)
                else:
                    tick.set_color("#4A4A4A")
        else:
            ax.tick_params(labelleft=False)

    structure_handles = [
        Line2D(
            [],
            [],
            marker=str(STRUCTURE_STYLE[structure]["marker"]),
            markerfacecolor=str(STRUCTURE_STYLE[structure]["color"]),
            markeredgecolor=str(STRUCTURE_STYLE[structure]["color"]),
            markeredgewidth=0.8,
            color="none",
            markersize=5.0,
            label=STRUCTURE_LABELS[structure],
        )
        for structure in STRUCTURES
    ]
    figure.legend(
        handles=structure_handles,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.985),
        ncol=4,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.35,
        fontsize=SMALL_TEXT_PT,
    )

    pdf = style.save_pdf(figure, output_path)
    return pdf


def main() -> None:
    print(plot())


if __name__ == "__main__":
    main()
