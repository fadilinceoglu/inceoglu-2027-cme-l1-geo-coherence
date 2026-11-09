"""Figure 2 — |B| along the Solar Orbiter-to-GEO observing chain.

The seven aligned panels show the four complete UTC days from 18 through
21 January 2026 and retain
encounter order: Solar Orbiter; ACE, Wind, DSCOVR, and Solar-1 at L1; the
ARTEMIS-P1/P2 pair near the Moon; and GOES-18/19 at geosynchronous orbit.

Caption information (kept here rather than rendered inside the figure):
Arithmetic one-minute mean magnetic-field magnitude at all nine observers;
the same-color envelope is +/- one within-minute sample standard deviation.
Hollow circles mark linearly interpolated minute means and local spreads;
the interpolated spread is not a measured uncertainty, and native sample
counts remain zero at those marked times.  Shaded intervals are the
registered sheath (orange) and ejecta (violet) intervals from
<output-dir>/results/boundaries.json. Dashed upstream lines are each spacecraft's
registered shock.  ARTEMIS-P1 and P2 have separately measured shock clocks
from this run; their registered sheath and ejecta edges are
coincident at the displayed scale.  The GOES mark is the directly observed
local magnetic-response onset at 19:17 UTC, not a propagated shock arrival.
Solar Orbiter was at 0.736 AU and led L1 by about seven hours; ARTEMIS was
near the Moon about 60 R_E upstream of Earth; GOES was at 6.6 R_E.

Data provenance: Solar Orbiter MAG L2 RTN-normal; ACE AC_H3_MFI; Wind
WI_H0_MFI; DSCOVR H0 MAG; Solar-1 MAG L3; ARTEMIS L2 FGM fgs; and NOAA
GOES-R L2 MAG hires.  Source-specific quality screening, arithmetic means,
sample standard deviations, sample counts, and interpolation flags are
recorded in the common-cadence products and manifests.

Run:  cd python && PYTHONPATH=. <venv-python> figures/fig_upstream_stack.py
"""

from __future__ import annotations

from pathlib import Path

from config.paths import PROCESSED_DIR, FIGURE_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from config import study
from figures import style
from figures.style import (
    FIGURE_WIDTH_MM,
    MM_PER_INCH,
    TEXT_PT as FIGURE_TEXT_PT,
    SMALL_TEXT_PT as FIGURE_SMALL_TEXT_PT,
)

UPSTREAM = ["solar_orbiter", *study.L1_SPACECRAFT]
GROUP_PANELS = [
    ("artemis", study.ARTEMIS_SPACECRAFT),
    ("goes", study.GOES_SPACECRAFT),
]
COMMON_CADENCE = PROCESSED_DIR / "common_cadence_1min"
CACHE_FILES = {
    "solar_orbiter": "solar_orbiter.csv",
    "ace": "ace.csv",
    "wind": "wind_gapfilled.csv",
    "dscovr": "dscovr.csv",
    "swfo": "swfo.csv",
    "artemis_p1": "artemis_p1.csv",
    "artemis_p2": "artemis_p2.csv",
    "goes18": "goes18_gapfilled.csv",
    "goes19": "goes19_gapfilled.csv",
}
GAPFILLED_KEYS = {"wind", "goes18", "goes19"}
OUT = FIGURE_DIR / "Fig2_upstream_bt_stack.pdf"

FIGURE_HEIGHT_MM = 170.0
TRACE_WIDTH_PT = 1.0
BOUNDARY_WIDTH_PT = 1.1
RESPONSE_COLOR = "#4D4D4D"
STD_ALPHA = 0.16
INTERPOLATED_MARKER_SIZE = 16.0


def _load(key: str) -> pd.DataFrame:
    path = COMMON_CADENCE / CACHE_FILES[key]
    if not path.is_file():
        raise FileNotFoundError(f"required Figure 2 dataset not found: {path}")
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index = (df.index.tz_localize("UTC") if df.index.tz is None
                else df.index.tz_convert("UTC"))
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError(f"{path} time index must be sorted and unique")
    required = {"bt", "bt_std", "bt_samples"}
    if key in GAPFILLED_KEYS:
        required.add("mag_interpolated")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns {missing}")
    if "mag_interpolated" not in df:
        df["mag_interpolated"] = 0
    flagged = pd.to_numeric(
        df["mag_interpolated"], errors="coerce"
    ).fillna(0).astype(bool)
    if flagged.any():
        if df.loc[flagged, ["bt", "bt_std"]].isna().any().any():
            raise ValueError(f"{path} has an unfilled flagged minute")
        if not pd.to_numeric(
            df.loc[flagged, "bt_samples"], errors="coerce"
        ).eq(0).all():
            raise ValueError(f"{path} gives native samples to interpolated data")
    return df


def _plot_bt(ax, frame: pd.DataFrame, key: str) -> None:
    """Plot one-minute |B| mean, within-minute spread, and filled minutes."""

    meta = study.SATELLITES[key]
    mean = pd.to_numeric(frame["bt"], errors="coerce")
    spread = pd.to_numeric(frame["bt_std"], errors="coerce")
    lower = (mean - spread).clip(lower=0)
    upper = mean + spread
    ax.fill_between(
        frame.index,
        lower.to_numpy(dtype=float),
        upper.to_numpy(dtype=float),
        color=meta["color"],
        alpha=STD_ALPHA,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        frame.index,
        mean,
        lw=TRACE_WIDTH_PT,
        color=meta["color"],
        zorder=2,
    )
    interpolated = pd.to_numeric(
        frame["mag_interpolated"], errors="coerce"
    ).fillna(0).astype(bool)
    if interpolated.any():
        ax.scatter(
            frame.index[interpolated],
            mean.loc[interpolated],
            s=INTERPOLATED_MARKER_SIZE,
            marker="o",
            facecolors="white",
            edgecolors=meta["color"],
            linewidths=0.9,
            zorder=5,
        )


def _marks(key: str) -> dict:
    from analysis import event_boundaries
    return event_boundaries.marks_for(key)


def _panel_labels(ax, letter: str, keys: list[str]) -> None:
    """Put the panel letter and color-coded observer labels inside the axes."""
    ax.text(
        0.006, 0.94, f"({letter})", transform=ax.transAxes,
        ha="left", va="top", fontsize=FIGURE_TEXT_PT, fontweight="bold",
        color=study.INK, zorder=10,
    )
    x = 0.045
    for key in keys:
        meta = study.SATELLITES[key]
        ax.text(
            x, 0.94, meta["name"], transform=ax.transAxes,
            ha="left", va="top", fontsize=FIGURE_TEXT_PT,
            fontweight="bold", color=meta["color"], zorder=10,
        )
        x += 0.020 + 0.0125 * len(meta["name"])


def _structure_legend(ax) -> None:
    """Explain the registered-boundary encodings without a rendered footer."""
    handles = [
        Line2D(
            [0], [0], color=study.SHOCK_COLOR, lw=BOUNDARY_WIDTH_PT,
            ls=(0, (4, 2)), label="Shock",
        ),
        Patch(
            facecolor=study.SPAN_COLORS["ejecta"][0],
            alpha=study.SPAN_COLORS["ejecta"][1], edgecolor="none",
            label="Ejecta",
        ),
        Line2D(
            [0], [0], marker="o", ls="none", markersize=4.0,
            markerfacecolor="white", markeredgecolor="#555555",
            markeredgewidth=0.9, label="Interpolated minute",
        ),
        Patch(
            facecolor=study.SPAN_COLORS["sheath"][0],
            alpha=study.SPAN_COLORS["sheath"][1], edgecolor="none",
            label="Sheath",
        ),
        Patch(
            facecolor="#777777", alpha=STD_ALPHA, edgecolor="none",
            label=r"$\pm1$ within-min SD",
        ),
    ]
    ax.legend(
        handles=handles, loc="upper right", ncol=2, frameon=False,
        fontsize=FIGURE_SMALL_TEXT_PT, handlelength=1.15,
        handletextpad=0.35, columnspacing=0.75, borderaxespad=0.2,
    )


def _mark_artemis(ax, axis_end) -> None:
    """Render registered P1/P2 structures without implying one lunar clock."""
    marks = {key: _marks(key) for key in ("artemis_p1", "artemis_p2")}

    # Close P1/P2 spans can overlap at the figure resolution.
    # Three minutes is approximately one rendered x pixel at final size, so
    # draw spans that agree within that tolerance once (using their union) to
    # avoid artificially doubling the alpha.  The exact shock clocks remain
    # separately encoded and printed in-panel.  Future resolvable span differences
    # are drawn independently.
    span_sets = [marks[key]["spans"] for key in marks]
    display_tolerance = pd.Timedelta("3min")

    def close_at_scale(left, right) -> bool:
        if len(left) != len(right):
            return False
        for (l0, l1, lk), (r0, r1, rk) in zip(left, right):
            if lk != rk or abs(l0 - r0) > display_tolerance:
                return False
            if (l1 is None) != (r1 is None):
                return False
            if l1 is not None and abs(l1 - r1) > display_tolerance:
                return False
        return True

    if close_at_scale(span_sets[0], span_sets[1]):
        shared_spans = []
        for left, right in zip(span_sets[0], span_sets[1]):
            t0 = min(left[0], right[0])
            t1 = (None if left[1] is None or right[1] is None
                  else max(left[1], right[1]))
            shared_spans.append((t0, t1, left[2]))
        style.mark_structures(
            ax, shock=None, spans=shared_spans, axis_end=axis_end,
        )
    else:
        for spans in span_sets:
            style.mark_structures(
                ax, shock=None, spans=spans, axis_end=axis_end,
            )

    for key, dash in (
        ("artemis_p1", (0, (4, 2))),
        ("artemis_p2", (0, (1.5, 1.5))),
    ):
        shock = marks[key]["shock"]
        if shock is not None:
            ax.axvline(
                shock, color=study.SATELLITES[key]["color"],
                lw=BOUNDARY_WIDTH_PT, ls=dash, zorder=4,
            )

    p1 = marks["artemis_p1"]["shock"]
    p2 = marks["artemis_p2"]["shock"]
    if p1 is not None and p2 is not None:
        ax.text(
            0.995, 0.94,
            f"P1 shock {pd.Timestamp(p1).round('1s'):%H:%M:%S}   P2 shock {pd.Timestamp(p2).round('1s'):%H:%M:%S}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FIGURE_SMALL_TEXT_PT, color="#555555", zorder=10,
        )


def _mark_goes_response(ax) -> None:
    """Mark the observed local GEO response, explicitly not a shock clock."""
    onset = study.GEO_RESPONSE_ONSET
    ax.axvline(
        onset, color=RESPONSE_COLOR, lw=BOUNDARY_WIDTH_PT,
        ls=(0, (3, 2)), zorder=4,
    )
    ax.text(
        0.995, 0.94,
        f"response onset {onset:%H:%M} UTC",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=FIGURE_SMALL_TEXT_PT,
        color=RESPONSE_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82,
              "pad": 0.8},
        zorder=10,
    )


def _publication_style() -> None:
    """Apply Figure-2-specific final-artwork settings after shared defaults."""
    style.apply_publication_style(text_size=FIGURE_TEXT_PT, tick_size=FIGURE_TEXT_PT, overrides={
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.08,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
    })


def build(out_path: str | Path = OUT) -> str:
    _publication_style()
    start, stop = study.FIGURE2_START, study.FIGURE2_STOP

    frames = {k: _load(k) for k in UPSTREAM}
    present = list(UPSTREAM)
    groups = []
    for kind, keys in GROUP_PANELS:
        groups.append((kind, [(k, _load(k)) for k in keys]))
    n_panels = len(present) + len(groups)
    if n_panels != 7:
        raise AssertionError(f"Figure 2 requires seven panels; found {n_panels}")

    fig, axes = plt.subplots(
        n_panels, 1, sharex=True,
        figsize=(
            FIGURE_WIDTH_MM / MM_PER_INCH,
            FIGURE_HEIGHT_MM / MM_PER_INCH,
        ),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    panel_letters = iter("abcdefghijklmnopqrstuvwxyz")

    for i, (ax, key) in enumerate(zip(axes, present)):
        df = frames[key]
        win = df.loc[(df.index >= start) & (df.index < stop)]
        _plot_bt(ax, win, key)
        marks = _marks(key)
        style.mark_structures(ax, shock=marks["shock"], spans=marks["spans"],
                              label=False, axis_end=stop)
        ax.set_ylabel("|B| [nT]")
        _panel_labels(ax, next(panel_letters), [key])
        if i == 0:
            _structure_legend(ax)

    for ax, (kind, have) in zip(axes[len(present):], groups):
        for k, df in have:
            win = df.loc[(df.index >= start) & (df.index < stop)]
            _plot_bt(ax, win, k)
        if kind == "artemis":
            _mark_artemis(ax, stop)
        elif kind == "goes":
            _mark_goes_response(ax)
        ax.set_ylabel("|B| [nT]")
        _panel_labels(ax, next(panel_letters), [k for k, _ in have])

    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.tick_params(axis="both", which="major", length=3.5)
        ax.tick_params(axis="both", which="minor", length=2.0)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
    style.format_time_axis(axes[-1], start, stop)
    for secondary_axis in axes[-1].child_axes:
        secondary_axis.tick_params(labelsize=FIGURE_TEXT_PT)
    axes[-1].set_xlabel("time (UTC)", labelpad=16)
    fig.align_ylabels(axes)

    output = style.save_pdf(fig, out_path)
    return str(output)


if __name__ == "__main__":
    print(build())
