"""Supplementary Figure 1 — band-energy intervals used for the Figure 4 metrics.

Caption information (kept here rather than rendered inside the figure):
The four waveform families of Figure 3, aligned to Solar Orbiter with the
Figure 3 lags, with the interval each spacecraft and family contributes to
Figure 4.  Dark shading spans the minutes where the cumulative filtered band
energy (the squared saved one-minute signal) reaches 5% and 95%, searched from
2 h before the spacecraft's shock to 2 h after its ejecta end; light shading is
the 2 h padding added on each side.  For the mean-retaining trend the squared
signal is essentially the magnetic energy |B|^2.  Vertical lines mark the
shock, the sheath-ejecta boundary and the ejecta end.

Inputs: results/waveform_intervals.json, results/waveform_comparisons.json,
results/boundaries.json and processed/bands/*.csv.
Output: figures/SupFig1_extraction_intervals.pdf.

Run from python/: PYTHONPATH=. python figures/fig_supp1_extraction_intervals.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from analysis import event_boundaries
from config import study
from config.paths import FIGURE_DIR, RESULTS_DIR
from figures import fig_isolated_waveforms as fig3, style

INTERVALS: Final = RESULTS_DIR / "waveform_intervals.json"
OUT: Final = FIGURE_DIR / "SupFig1_extraction_intervals.pdf"
INTERVAL_COLOR: Final = "#c8d2de"
PADDING_COLOR: Final = "#e9edf2"


def _record_minutes(value: str) -> float:
    """Minutes since the aligned record start for one saved UTC stamp."""

    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return (stamp - pd.Timestamp(fig3.WINDOW_START)) / pd.Timedelta(minutes=1)


def build(
    out_path: str | Path = OUT,
    *,
    intervals_path: str | Path = INTERVALS,
) -> str:
    intervals = json.loads(Path(intervals_path).read_text())["intervals"]
    profiles = fig3._aligned_profiles(fig3.load_result())
    boundaries, _ = event_boundaries.load()
    fig3._publication_style()
    families = tuple(fig3.STRUCTURES)
    crafts = tuple(fig3.CRAFT_ORDER)
    figure, axes = plt.subplots(
        len(crafts), len(families),
        figsize=(fig3.FIGURE_WIDTH_MM / fig3.MM_PER_INCH, fig3.FIGURE_HEIGHT_MM / fig3.MM_PER_INCH),
        sharex="col",
    )
    figure.subplots_adjust(left=0.075, right=0.992, bottom=0.125, top=0.925, hspace=0.20, wspace=0.40)
    for col, family in enumerate(families):
        spans = {}
        for craft in crafts:
            record, lag = intervals[craft][family], profiles[family][craft][2]
            spans[craft] = tuple(
                (_record_minutes(record[key]) - lag) / 60.0
                for key in ("start_utc", "energy_start_utc", "energy_stop_utc_exclusive", "stop_utc_exclusive")
            )
        lo = min(span[0] for span in spans.values()) - 1.5
        hi = max(span[3] for span in spans.values()) + 1.5
        shown = np.concatenate([
            values[(time >= lo) & (time <= hi)]
            for time, values, _ in (profiles[family][craft] for craft in crafts)
        ])
        limits, ticks = fig3._amplitude_axis(shown, family)
        for row, craft in enumerate(crafts):
            ax = axes[row, col]
            time, values, lag = profiles[family][craft]
            selected = (time >= lo) & (time <= hi)
            padded_start, energy_start, energy_stop, padded_stop = spans[craft]
            ax.axvspan(padded_start, padded_stop, color=PADDING_COLOR, lw=0, zorder=0)
            ax.axvspan(energy_start, energy_stop, color=INTERVAL_COLOR, lw=0, zorder=0)
            ax.plot(time[selected], values[selected], color=study.SATELLITES[craft]["color"], lw=0.55)
            marks = fig3._aligned_boundary_hours(boundaries[craft], lag)
            for key, (_, color, dash) in fig3.BOUNDARY_STYLES.items():
                if lo <= marks[key] <= hi:
                    ax.axvline(marks[key], color=color, ls=dash, lw=0.65, zorder=3)
            ax.set_xlim(lo, hi)
            ax.set_ylim(*limits)
            ax.set_yticks(ticks)
            ax.tick_params(axis="both", length=2.2, pad=2)
            ax.text(
                0.035, 0.90, fig3.panel_letter(row * len(families) + col),
                transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
                fontweight="bold", color=study.INK,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.3},
                zorder=5,
            )
            if row < len(crafts) - 1:
                ax.tick_params(labelbottom=False)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if col == 0:
                ax.set_ylabel(fig3.CRAFT_LABELS[craft], fontsize=6.8, color=study.SATELLITES[craft]["color"])
        symbol = r"$|B|$" if family == "trend" else r"$\Delta|B|$"
        axes[0, col].set_title(f"{fig3.STRUCTURE_LABELS[family]}\n{symbol} [nT]", fontsize=fig3.TEXT_PT, pad=8)
    origin = pd.Timestamp(fig3.WINDOW_START).tz_convert("UTC").strftime("%d %b %Y, %H:%M UTC")
    figure.supxlabel(f"Time since {origin} [h], aligned to Solar Orbiter", fontsize=6.8, y=0.074)
    figure.legend(
        handles=[Patch(color=INTERVAL_COLOR, label="5–95% band energy"),
                 Patch(color=PADDING_COLOR, label="2 h padding")]
        + [Line2D([], [], color=color, ls=dash, lw=0.8, label=label)
           for label, color, dash in fig3.BOUNDARY_STYLES.values()],
        loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5,
        frameon=False, fontsize=7, handlelength=2.5,
    )
    output = style.save_pdf(figure, out_path)
    return str(output)


if __name__ == "__main__":
    print(build())
