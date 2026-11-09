"""Figure 6: magnitude, direction and rapid variability at GOES-18/19.

Both spacecraft are compared on the same axes. Panels show (a) measured
|B|, (b) measured GSM Bz, (c) the change in |B| from each spacecraft's own
same-clock quiet reference, (d) the angle from its own quiet-reference
vector, and (e) the sample standard deviation of native Bz measurements
within each minute. The reference uses the two preceding days.
Shading in (c, d) spans the
results obtained using either day separately, not a confidence interval.
Dashed green references show each spacecraft's quiet |B| and Bz in (a, b)
and zero departure in (c). The display includes 17 minutes before the observed
response onset, marked by the vertical dotted line. The vertical dashed line
marks the subsequent observed field transition.
Spacecraft-coloured dash-dot lines mark local-time passages derived from the
native GOES orbit records; their labels appear above the bottom panel.

Uses the paired-response products; no model is fitted here.
Run from python/: PYTHONPATH=. ../venv/bin/python figures/fig_goes_magnetic_response.py
"""

from __future__ import annotations

from datetime import timezone
from pathlib import Path

from config.paths import RESULTS_DIR, FIGURE_DIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from analysis.paired_goes_response import (
    DISPLAY_INDEX, DISPLAY_START, EVENT_START, EVENT_STOP,
    RESPONSE_TRANSITION, load_response,
)
from analysis.goes_local_time import load_local_time
from config import study
from figures import style


DATA = RESULTS_DIR
OUT = FIGURE_DIR / "Fig6_goes_magnetic_response"
SPACECRAFT = {
    alias: (study.spacecraft_label(alias), study.SATELLITES[study.spacecraft_key(alias)]["color"])
    for alias in ("g18", "g19")
}
QUIET_STYLES = {"g18": ("tab:green", (0, (6, 2))), "g19": ("#006400", (0, (2, 2)))}
TRANSITION_UTC = RESPONSE_TRANSITION
TRANSITION_COLOR = "#666666"
ONSET_COLOR = "#111111"


def _read_products() -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = load_response(DATA / "paired_goes_response.csv", event_index=DISPLAY_INDEX)
    sensitivity = pd.read_csv(
        DATA / "paired_goes_response_reference_sensitivity.csv",
        parse_dates=["time_utc"], float_precision="round_trip",
    ).set_index("time_utc")
    expected_references = {"one_day_before", "two_days_before"}
    if set(sensitivity.reference) != expected_references or any(
        not group.index.equals(DISPLAY_INDEX)
        for _, group in sensitivity.groupby("reference", sort=False)
    ):
        raise ValueError("quiet-reference comparisons require the complete display clock for each day")
    return primary, sensitivity


def _reference_band(axis, primary, sensitivity, column, color):
    alternatives = [group[column].reindex(primary.index).to_numpy(float)
                    for _, group in sensitivity.groupby("reference")]
    if len(alternatives) != 2:
        raise ValueError("Expected the two single-day reference comparisons")
    low, high = np.min(alternatives, axis=0), np.max(alternatives, axis=0)
    axis.fill_between(primary.index, low, high, color=color, alpha=.18, linewidth=0)


def build(output: Path = OUT) -> Path:
    primary, sensitivity = _read_products()
    _, local_time_markers = load_local_time(
        DATA / "goes_local_time.csv", DATA / "goes_local_time_markers.csv",
        display_index=DISPLAY_INDEX,
    )
    style.apply_publication_style(text_size=style.TEXT_PT, tick_size=style.SMALL_TEXT_PT, overrides={
        'axes.linewidth': 0.7,
        'legend.fontsize': 7,
        'figure.dpi': 110,
        'savefig.dpi': 200,
    })
    figure, axes = plt.subplots(
        5, 1, figsize=(style.FIGURE_WIDTH_MM / style.MM_PER_INCH, 210 / style.MM_PER_INCH),
        sharex=True, constrained_layout=True,
    )
    figure.get_layout_engine().set(w_pad=1 / style.MM_PER_INCH, h_pad=1 / style.MM_PER_INCH, hspace=.025)
    panels = (
        ("bt", r"$|B|$ [nT]", (0, 410)),
        ("bz", r"$B_z$ GSM [nT]", (-270, 390)),
        ("delta_bt_nt", r"$\Delta|B|$ [nT]", (-80, 310)),
        ("angle_quiet_deg", "Angle from quiet field [°]", (0, 205)),
        ("bz_std", r"Within-minute $\sigma(B_z)$ [nT]", (0, 165)),
    )
    for row, (axis, (quantity, ylabel, ylim)) in enumerate(zip(axes, panels)):
        axis.text(.012, .96, f"({chr(ord('a')+row)})", transform=axis.transAxes,
                  ha="left", va="top", fontsize=8, fontweight="bold", zorder=10)
        axis.axvline(TRANSITION_UTC, color=TRANSITION_COLOR, lw=.75, ls="--", zorder=2)
        axis.axvline(EVENT_START, color=ONSET_COLOR, lw=.85, ls=":", zorder=2)
        for marker in local_time_markers.itertuples(index=False):
            label, color = SPACECRAFT[marker.spacecraft]
            axis.axvline(marker.time_utc, color=color, lw=.7, ls="-.",
                         alpha=.8, zorder=2,
                         label=f"{label} {marker.label} MLT")
        for key, (label, color) in SPACECRAFT.items():
            column = f"{key}_{quantity}"
            if row in (2, 3):
                _reference_band(axis, primary, sensitivity, column, color)
            axis.plot(primary.index, primary[column], color=color, lw=.85, label=label, zorder=3)
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        axis.set_xlim(DISPLAY_START, EVENT_STOP)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", lw=.3, alpha=.22)
        axis.grid(axis="x", lw=.3, alpha=.12)
        axis.tick_params(axis="x", which="both", length=3)
    axes[2].axhline(0, color="tab:green", lw=1.7, ls=(0, (6, 2)),
                   label="Quiet reference", zorder=5)
    axes[1].axhline(0, color="#444444", lw=.6, zorder=2)
    for row, quantity in ((0, "bt"), (1, "bz")):
        quiet_handles = []
        for key, (label, _) in SPACECRAFT.items():
            color, linestyle = QUIET_STYLES[key]
            line, = axes[row].plot(primary.index, primary[f"{key}_quiet_{quantity}"],
                                  color=color, ls=linestyle, lw=1.5,
                                  label=f"Quiet {label}", zorder=5)
            quiet_handles.append(line)
        if row == 1:
            axes[row].legend(handles=quiet_handles, loc="upper right", ncol=2,
                             frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    axes[3].set_yticks([0, 90, 180])
    axes[3].axhline(90, color="#555555", lw=.6, zorder=2)
    handles = [Line2D([], [], color=color, lw=1, label=label)
               for label, color in SPACECRAFT.values()]
    handles.extend(quiet_handles)
    handles.append(Line2D([], [], color=ONSET_COLOR, lw=.85, ls=":",
                          label=f"Response onset {EVENT_START:%H:%M}"))
    handles.append(Line2D([], [], color=TRANSITION_COLOR, lw=.75, ls="--",
                          label=f"Field transition {TRANSITION_UTC:%H:%M}"))
    axes[0].legend(handles=handles, loc="upper right", ncol=3,
                   frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    axes[2].legend(handles=[Line2D([], [], color="tab:green", lw=1.7, ls=(0, (6, 2)),
                                   label="Quiet reference")],
                   loc="upper right", frameon=True, facecolor="white",
                   edgecolor="none", framealpha=1)
    axes[-1].set_xticks(pd.date_range(DISPLAY_START, EVENT_STOP, freq="2h"))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    axes[-1].set_xlabel("19-20 January 2026 [UTC]")
    for marker in local_time_markers.itertuples(index=False):
        label, color = SPACECRAFT[marker.spacecraft]
        displayed_time = marker.time_utc.round("min")
        axes[-1].annotate(
            f"{label} {marker.label.lower()}\n{displayed_time:%H:%M} UT",
            xy=(marker.time_utc, 1.015), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", color=color, fontsize=6.8,
            annotation_clip=False,
        )
    output = style.save_pdf(figure, output)
    return output


if __name__ == "__main__":
    print(build())
