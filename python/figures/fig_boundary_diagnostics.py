"""Native magnetic shock fits and sheath-direction diagnostics.

Shock panels read this run's registered logistic fits or neighbour-guided
level crossings. They show observed samples, the fitted curve (or persistence
median), 25/50/75% crossings and timing uncertainty. ACE's unresolved gap is
marked explicitly. Sheath panels show the shared variability selection and
logistic angular transition from this same run.
"""

from __future__ import annotations

import gc
from pathlib import Path

from config.paths import DIAGNOSTIC_DIR
import textwrap

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from config import study
from analysis.event_boundaries import shock_measurement
from analysis.shock_detection import logistic
from figures import style
from figures.style import FIGURE_WIDTH_MM, MM_PER_INCH, TEXT_PT, SMALL_TEXT_PT as SMALL_PT


SHOCK_OUT = DIAGNOSTIC_DIR / (
    "shock_boundaries.pdf"
)
SHEATH_OUT = DIAGNOSTIC_DIR / (
    "sheath_ejecta_boundaries.pdf"
)

FIGURE_HEIGHT_MM = 170.0
TRACE_PT = 1.0
EMPHASIS_PT = 1.2

SHOCK_XLIM_S = (-180.0, 180.0)

LEVEL_COLORS = {
    25: "#777777",
    50: study.SHOCK_COLOR,
    75: "#333333",
}


def _seconds(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    return (index - origin).total_seconds().to_numpy(dtype=float)


def _relative_seconds(time: pd.Timestamp, origin: pd.Timestamp) -> float:
    return float((time - origin).total_seconds())


def _fmt_clock(time: pd.Timestamp, decimals: int = 3) -> str:
    time = pd.Timestamp(time).round(pd.Timedelta(seconds=10.0 ** -max(0, decimals)))
    base = time.strftime("%H:%M:%S")
    if decimals <= 0:
        return base
    fraction = f"{time.microsecond:06d}"[:decimals]
    return f"{base}.{fraction}"


def _publication_style() -> None:
    style.apply_publication_style(text_size=TEXT_PT, tick_size=TEXT_PT, overrides={
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.08,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
    })



def _wrap_annotation(text: str, width: int = 43) -> str:
    return "\n".join(
        textwrap.fill(
            line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for line in text.splitlines()
    )


def _annotation_axis(
    ax,
    *,
    clock_text: str,
    method_text: str,
) -> None:
    ax.set_axis_off()
    ax.text(
        0.0,
        0.96,
        clock_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=SMALL_PT,
        color="#444444",
        linespacing=0.94,
    )
    ax.text(
        0.0,
        0.04,
        _wrap_annotation(method_text),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=SMALL_PT,
        color="#555555",
        linespacing=0.94,
    )


def _panel_label(ax, letter: str, keys: list[str]) -> None:
    ax.text(
        0.006,
        0.95,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TEXT_PT,
        fontweight="bold",
        color=study.INK,
        zorder=20,
    )
    x = 0.045
    for key in keys:
        meta = study.SATELLITES[key]
        ax.text(
            x,
            0.95,
            meta["name"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=TEXT_PT,
            fontweight="bold",
            color=meta["color"],
            zorder=20,
        )
        x += 0.020 + 0.0125 * len(meta["name"])


def _levels(b_up: float, b_down: float) -> dict[int, float]:
    jump = float(b_down - b_up)
    return {
        percent: float(b_up + percent / 100.0 * jump)
        for percent in (25, 50, 75)
    }


def _shock_entry(
    *,
    key: str,
    raw: pd.Series,
    smooth: pd.Series | None,
    t25: pd.Timestamp,
    t50: pd.Timestamp,
    t75: pd.Timestamp,
    uncertainty_s: float,
    b_up: float,
    b_down: float,
    up_window: tuple[pd.Timestamp, pd.Timestamp],
    down_window: tuple[pd.Timestamp, pd.Timestamp],
    cadence_text: str,
    gap: tuple[pd.Timestamp, pd.Timestamp, str] | None = None,
) -> dict:
    return {
        "key": key,
        "raw": raw,
        "smooth": smooth,
        "t25": pd.Timestamp(t25),
        "t50": pd.Timestamp(t50),
        "t75": pd.Timestamp(t75),
        "uncertainty_s": float(uncertainty_s),
        "b_up": float(b_up),
        "b_down": float(b_down),
        "levels": _levels(float(b_up), float(b_down)),
        "up_window": up_window,
        "down_window": down_window,
        "cadence_text": cadence_text,
        "gap": gap,
    }


def _registered_shock_entry(key: str, series: pd.Series) -> dict:
    shock = shock_measurement(key)
    t50 = shock["t50"]
    raw = series.loc[t50 - pd.Timedelta(seconds=180):t50 + pd.Timedelta(seconds=180)].copy()
    if shock["method"] == "native logistic fit":
        grid = pd.date_range(shock["fit_window_start"], shock["fit_window_end"], periods=500)
        smooth = pd.Series(logistic((grid - t50).total_seconds(), shock["b_up"],
                                   shock["b_dn"], 0., np.log(shock["tau_s"])), index=grid)
        up_window = (shock["fit_window_start"], shock["t25"])
        down_window = (shock["t75"], shock["fit_window_end"])
        note = f"{shock['cadence_s']:g} s; native logistic fit; R²={shock['r2']:.3f}"
    else:
        smooth = raw.rolling(3, center=True, min_periods=1).median() if key.startswith("artemis") else None
        reference = shock["median_reference_time"]
        up_window = tuple(reference + pd.Timedelta(seconds=v) for v in shock["up_window_s"])
        down_window = tuple(reference + pd.Timedelta(seconds=v) for v in shock["down_window_s"])
        note = (f"{shock['cadence_s']:.2f} s; {study.SATELLITES[shock['source_spacecraft']]['name']} guide; "
                "native median levels; ≥80%/30 s persistence")
    gap = None
    if shock.get("gap_limited"):
        gap = (shock["gap_start"], shock["gap_stop"], "MAG gap")
        note += "; gap-limited uncertainty"
    return _shock_entry(key=key, raw=raw, smooth=smooth,
                        t25=shock["t25"], t50=t50, t75=shock["t75"],
                        uncertainty_s=shock["unc_s"], b_up=shock["b_up"], b_down=shock["b_dn"],
                        up_window=up_window, down_window=down_window, cadence_text=note, gap=gap)


def _collect_diagnostics() -> list[dict]:
    """Read the registered shocks and their native traces, without refitting."""
    from pipeline.build_shock_boundaries import CRAFT, native_record
    rows = []
    for key in CRAFT:
        series, _ = native_record(key)
        rows.append({"keys": [key], "entries": [_registered_shock_entry(key, series)]})
        del series
        gc.collect()
    return rows


def _shock_clock_text(entry: dict) -> str:
    return (f"t50 {_fmt_clock(entry['t50'])} "
            f"±{entry['uncertainty_s']:.2f} s")


def _plot_shock_row(ax, note_ax, row: dict, letter: str) -> None:
    _panel_label(ax, letter, row["keys"])
    entry, = row["entries"]
    key = entry["key"]
    color = study.SATELLITES[key]["color"]
    raw = entry["raw"]
    x_raw = _seconds(raw.index, entry["t50"])
    ax.plot(
        x_raw,
        raw.to_numpy(dtype=float),
        color=color,
        lw=TRACE_PT,
        alpha=0.48 if entry["smooth"] is not None else 1.0,
        zorder=3,
    )
    if entry["smooth"] is not None:
        smooth = entry["smooth"]
        ax.plot(
            _seconds(smooth.index, entry["t50"]),
            smooth.to_numpy(dtype=float),
            color=color,
            lw=EMPHASIS_PT,
            zorder=4,
        )

    ax.axvspan(
        -entry["uncertainty_s"],
        entry["uncertainty_s"],
        color=study.SHOCK_COLOR,
        alpha=0.10,
        lw=0,
        zorder=0,
    )
    crossing_times = {
        25: entry["t25"],
        50: entry["t50"],
        75: entry["t75"],
    }
    dash = {
        25: (0, (1.5, 1.5)),
        50: "-",
        75: (0, (4, 2)),
    }
    for percent, time in crossing_times.items():
        x = _relative_seconds(time, entry["t50"])
        ax.axvline(
            x,
            color=LEVEL_COLORS[percent],
            lw=EMPHASIS_PT if percent == 50 else TRACE_PT,
            ls=dash[percent],
            alpha=1.0 if percent == 50 else 0.85,
            zorder=5,
        )
        ax.hlines(
            entry["levels"][percent],
            -26,
            26,
            color=LEVEL_COLORS[percent],
            lw=TRACE_PT,
            ls=dash[percent],
            alpha=0.72,
            zorder=2,
        )

    for start, stop, value, label in (
        (
            entry["up_window"][0],
            entry["up_window"][1],
            entry["b_up"],
            "Bup",
        ),
        (
            entry["down_window"][0],
            entry["down_window"][1],
            entry["b_down"],
            "Bdown",
        ),
    ):
        x0 = _relative_seconds(start, entry["t50"])
        x1 = _relative_seconds(stop, entry["t50"])
        ax.hlines(
            value,
            x0,
            x1,
            color=color,
            lw=EMPHASIS_PT,
            zorder=6,
        )
        ax.text(
            0.5 * (x0 + x1),
            value,
            f"{label} {value:.1f}",
            ha="center",
            va="bottom" if label == "Bup" else "top",
            fontsize=SMALL_PT,
            color=color,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.76,
                "pad": 0.3,
            },
            zorder=10,
        )

    if entry["gap"] is not None:
        gap_start, gap_stop, gap_label = entry["gap"]
        gx0 = _relative_seconds(gap_start, entry["t50"])
        gx1 = _relative_seconds(gap_stop, entry["t50"])
        ax.axvspan(
            gx0,
            gx1,
            facecolor="#777777",
            edgecolor="#555555",
            hatch="////",
            alpha=0.20,
            lw=TRACE_PT,
            zorder=1,
        )
        ax.text(
            0.5 * (gx0 + gx1),
            0.05,
            gap_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=SMALL_PT,
            color="#555555",
            zorder=10,
        )

    for percent, dx in ((25, -14), (50, 0), (75, 14)):
        x = _relative_seconds(
            entry[f"t{percent}"], entry["t50"]
        )
        ax.annotate(
            f"t{percent}",
            xy=(x, 0.80),
            xycoords=("data", "axes fraction"),
            xytext=(dx, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=SMALL_PT,
            color=LEVEL_COLORS[percent],
            fontweight="bold" if percent == 50 else "normal",
            zorder=20,
        )

    _annotation_axis(
        note_ax,
        clock_text=_shock_clock_text(entry),
        method_text=entry["cadence_text"],
    )
    ax.set_ylabel("|B| [nT]")
    ax.set_xlim(*SHOCK_XLIM_S)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))


def build_shock_figure(rows: list[dict], output: Path = SHOCK_OUT) -> Path:
    """Render the native-cadence shock-detector diagnostic figure."""
    _publication_style()
    fig = plt.figure(
        figsize=(
            FIGURE_WIDTH_MM / MM_PER_INCH,
            FIGURE_HEIGHT_MM / MM_PER_INCH,
        ),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        len(rows),
        2,
        width_ratios=(2.45, 1.0),
        wspace=0.04,
        hspace=0.0,
    )
    axes = []
    note_axes = []
    for index in range(len(rows)):
        axes.append(
            fig.add_subplot(
                grid[index, 0],
                sharex=axes[0] if axes else None,
            )
        )
        note_axes.append(fig.add_subplot(grid[index, 1]))
    for ax, note_ax, row, letter in zip(
        axes, note_axes, rows, "abcdefg"
    ):
        _plot_shock_row(ax, note_ax, row, letter)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
    axes[-1].set_xlabel("Time from detected t50 [s]")
    fig.align_ylabels(axes)
    return style.save_pdf(fig, output)


def build() -> tuple[Path, Path]:
    """Collect exact detector diagnostics once and render both diagnostic figures."""
    from figures.sheath_boundary_diagnostics import build as build_sheath
    shock = build_shock_figure(_collect_diagnostics())
    sheath = build_sheath(SHEATH_OUT)
    return shock, sheath


if __name__ == "__main__":
    for path in build():
        print(path)
