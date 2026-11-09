"""Shared matplotlib style for all study figures.

Every figure module calls ``apply_style()`` once, then uses the helpers here
so satellite colors, structure shading, and time axes look identical across
the whole figure set (fixed color assignment — see config/study.py).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from config import study


# Shared artwork dimensions and type sizes, specified at the final physical size.
FIGURE_WIDTH_MM = 180.0
MM_PER_INCH = 25.4
TEXT_PT = 8.0
SMALL_TEXT_PT = 7.0
SANS_SERIF_FONTS = ["Arial", "Helvetica", "DejaVu Sans"]


OVERVIEW_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": SANS_SERIF_FONTS,
    "font.size": TEXT_PT,
    "axes.labelsize": TEXT_PT,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#555555",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#000000",
    "grid.alpha": 0.10,
    "grid.linewidth": 0.45,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "legend.frameon": False,
    "legend.fontsize": 7.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#000000",
        "grid.alpha": 0.10,
        "grid.linewidth": 0.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def apply_publication_style(*, text_size=TEXT_PT, tick_size=None, overrides=None) -> None:
    """Shared vector-artwork settings; figures supply only their layout differences."""
    apply_style()
    tick_size = text_size if tick_size is None else tick_size
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": None,
        "font.family": "sans-serif",
        "font.sans-serif": SANS_SERIF_FONTS,
        "font.size": text_size, "axes.labelsize": text_size,
        "xtick.labelsize": tick_size, "ytick.labelsize": tick_size,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        **(overrides or {}),
    })


def format_time_axis(ax, start: datetime, stop: datetime) -> None:
    """UTC axis: 6-hour major ticks, hourly minors, day labels."""
    ax.set_xlim(start, stop)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 6, 12, 18)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    sec = ax.secondary_xaxis(-0.18)
    sec.xaxis.set_major_locator(mdates.DayLocator())
    sec.xaxis.set_major_formatter(mdates.DateFormatter("%d Jan %Y"))
    sec.tick_params(length=0, labelsize=9)
    for spine in sec.spines.values():
        spine.set_visible(False)


def mark_structures(ax, *, shock=None, spans=(), label: bool = False,
                    axis_end=None) -> None:
    """Draw the registered shock line and structure spans on one axis.

    spans: iterable of (t0, t1_or_None, kind) with kind in study.SPAN_COLORS.
    A ``None`` end means right-censored; the span extends to ``axis_end``
    and, when labeling, gains an 'ongoing' arrow.
    """
    for t0, t1, kind in spans:
        color, alpha = study.SPAN_COLORS[kind]
        open_ended = t1 is None
        t1 = t1 or axis_end
        if t1 is None:
            continue
        ax.axvspan(t0, t1, color=color, alpha=alpha, lw=0, zorder=0)
        if label:
            mid = t0 + (t1 - t0) / 2
            text = kind + (" →" if open_ended else "")
            ax.annotate(
                text, xy=(mid, 1.015), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=8.5, color=color,
                fontweight="bold", annotation_clip=False,
            )
    if shock is not None:
        ax.axvline(shock, color=study.SHOCK_COLOR, lw=1.1, ls=(0, (4, 2)),
                   zorder=3)
        if label:
            ax.annotate(
                "shock " + shock.strftime("%H:%M:%S"),
                xy=(shock, 1.015), xycoords=("data", "axes fraction"),
                ha="right", va="bottom", fontsize=8.5,
                color=study.SHOCK_COLOR, fontweight="bold", rotation=0,
                xytext=(-4, 0), textcoords="offset points",
                annotation_clip=False,
            )


def save_pdf(fig, output_path: str | Path, *, strict_pdf: bool = False, **savefig_kwargs) -> Path:
    """Save a PDF on its declared canvas, close it and return the output path.

    No bounding-box override is applied: the figure's active save settings
    control its canvas. Reconstruction callers require an explicit PDF suffix;
    other callers may supply a filename stem or replace a different suffix.
    """
    output = Path(output_path)
    if strict_pdf:
        if output.suffix.lower() != ".pdf":
            raise ValueError("publication reconstruction figures must be saved as PDF")
    else:
        output = output.with_suffix(".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, **savefig_kwargs)
    plt.close(fig)
    return output
