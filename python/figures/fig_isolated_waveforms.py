"""Figure 3 - aligned isolated waveform families with physical amplitude axes.

The four columns stack the seven spacecraft in propagation order; every
subplot is labelled individually, (a)--(ab), in row-major order.
Each curve is its saved one-minute bt_f series, shifted by the current-run
SO-fixed, family-specific graph offset from waveform_comparisons.json. Saved
invalid rows remain gaps. No new filtering, averaging or lag estimation occurs.

Display windows use 2--98% cumulative energy of the
cross-craft median absolute amplitude on the common aligned grid, padded by
8%, and expand to include every spacecraft's shock and sheath-ejecta boundary.
Only large-scale and trend also show the ejecta end; the two faster families retain
a close view of their waveforms.
The fine-scale column starts at 35 aligned hours by display choice.
This interpolation selects the display window only; plotted values remain
the original saved minute means. Within each column all spacecraft share the
same physical nT limits, calculated to contain every visible peak. There is
no amplitude normalization or clipping and no within-minute spread shading.

The columns show fine-scale, intermediate-scale, large-scale and trend.
The first three show signed filtered scalar-magnitude residuals;
trend retains its mean. Aligned hours are measured from the analysis-record
start after subtracting the annotated offset (minutes) relative to SO.
The registered shock, sheath-ejecta transition and ejecta end receive the
same shift as their waveform. Sheath and ejecta shading matches Figure 2.

Inputs: results/waveform_comparisons.json, results/boundaries.json,
and processed/bands/*.csv.
Output: figures/Fig3_isolated_waveforms.pdf.
"""

from __future__ import annotations

import json
from pathlib import Path

from config.paths import RESULTS_DIR, FIGURE_DIR
from typing import Any, Final, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from analysis.waveform_comparisons import (
    CRAFT_ORDER,
    FULL_GRID,
    FULL_SAMPLE_COUNT,
    STRUCTURES,
    WINDOW_START,
    WINDOW_STOP,
    load_waveform,
)
from analysis.product_io import utc_text
from analysis import event_boundaries
from config import study
from config.waveform_bands import WAVEFORM_FAMILY_LABELS
from figures import style
from figures.style import FIGURE_WIDTH_MM, MM_PER_INCH, TEXT_PT, SMALL_TEXT_PT


INPUT_JSON: Final = (
    RESULTS_DIR / "waveform_comparisons.json"
)
OUT: Final = (
    FIGURE_DIR / "Fig3_isolated_waveforms.pdf"
)

SCHEMA_VERSION: Final = 3
STRUCTURE_LABELS: Final = WAVEFORM_FAMILY_LABELS
CRAFT_LABELS: Final = study.CRAFT_LABELS

FIGURE_HEIGHT_MM: Final = 170.0
BOUNDARY_STYLES: Final = {
    "shock": ("Shock", study.SHOCK_COLOR, (0, (4, 2))),
    "sheath_end": ("Sheath-ejecta", study.SPAN_COLORS["sheath"][0], (0, (1.5, 1.5))),
    "ejecta_end": ("Ejecta end", study.SPAN_COLORS["ejecta"][0], (0, (4, 2, 1, 2))),
}


def _publication_style() -> None:
    style.apply_publication_style(text_size=TEXT_PT, tick_size=SMALL_TEXT_PT, overrides={
        'axes.titlesize': TEXT_PT,
        'axes.linewidth': 0.8,
        'axes.grid': True,
    })


def load_result(path: Path | str = INPUT_JSON) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} is missing; build the current waveform comparison cache first"
        )
    result = json.loads(source.read_text())
    meta = result.get("meta") or {}
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported waveform comparison schema")
    if meta.get("window_start_utc") != utc_text(WINDOW_START):
        raise ValueError("canonical cache has the wrong waveform-window start")
    if meta.get("window_stop_utc") != utc_text(WINDOW_STOP):
        raise ValueError("canonical cache has the wrong waveform-window stop")
    if int(meta.get("full_window_sample_count", -1)) != FULL_SAMPLE_COUNT:
        raise ValueError("canonical cache does not use all 5,760 minutes")
    records = result.get("pair_records") or []
    if len(records) != 84:
        raise ValueError("canonical cache must contain 84 pair/family records")
    return result


def _global_registration_lags(
    result: Mapping[str, Any],
) -> dict[tuple[str, str], int]:
    registrations = result.get("global_registration_by_structure") or {}
    if set(registrations) != set(STRUCTURES):
        raise ValueError("canonical cache lacks the four graph registrations")
    lags = {
        (structure, craft): int(offset)
        for structure, registration in registrations.items()
        for craft, offset in registration[
            "integer_offsets_from_so_minutes"
        ].items()
    }
    expected = {
        (structure, craft)
        for structure in STRUCTURES
        for craft in CRAFT_ORDER
    }
    if set(lags) != expected:
        raise ValueError("canonical cache lacks a graph offset for a spacecraft")
    return lags


def _aligned_profiles(
    result: Mapping[str, Any],
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, int]]]:
    lags = _global_registration_lags(result)
    inventory = {
        (str(row["structure"]), str(row["craft"])): row
        for row in result.get("input_inventory", [])
    }
    expected_inventory = {
        (structure, craft)
        for structure in STRUCTURES
        for craft in CRAFT_ORDER
    }
    if set(inventory) != expected_inventory:
        raise ValueError("canonical cache has an incomplete waveform inventory")
    base_hours = (
        (FULL_GRID - WINDOW_START).total_seconds().to_numpy(dtype=float)
        / 3600.0
    )
    profiles: dict[str, dict[str, tuple[np.ndarray, np.ndarray, int]]] = {}
    for structure in STRUCTURES:
        profiles[structure] = {}
        for craft in CRAFT_ORDER:
            waveform = load_waveform(craft, structure)
            cached = inventory[(structure, craft)]
            if int(waveform.valid.sum()) != int(cached["valid_sample_count"]):
                raise ValueError(
                    f"{craft}/{structure} validity differs from the cache"
                )
            lag = lags[(structure, craft)]
            aligned_hours = base_hours - lag / 60.0
            profiles[structure][craft] = (
                aligned_hours,
                waveform.values_nt,
                lag,
            )
    return profiles


def _zoom_window(
    family: Mapping[str, tuple[np.ndarray, np.ndarray, int]],
) -> tuple[float, float]:
    """Family display window from cumulative cross-craft median energy.

    Each craft carries its own graph-shifted time grid, so every craft's
    |amplitude| is first interpolated onto one common aligned grid before
    the median; the window is the [2%, 98%] cumulative-energy interval of
    that median envelope, padded by 8%.
    """

    grids = family[CRAFT_ORDER[0]][0]
    envelope = np.nanmedian(
        np.column_stack(
            [
                np.interp(
                    grids,
                    family[craft][0],
                    np.nan_to_num(np.abs(family[craft][1]), nan=0.0),
                    left=0.0,
                    right=0.0,
                )
                for craft in CRAFT_ORDER
            ]
        ),
        axis=1,
    )
    energy = np.nan_to_num(envelope, nan=0.0) ** 2
    cumulative = np.cumsum(energy) / np.sum(energy)
    x_lo = float(grids[np.searchsorted(cumulative, 0.02)])
    x_hi = float(grids[np.searchsorted(cumulative, 0.98)])
    pad = 0.08 * (x_hi - x_lo)
    return x_lo - pad, x_hi + pad



def _amplitude_axis(values: np.ndarray, structure: str) -> tuple[tuple[float, float], list[float]]:
    """Shared physical limits that include all finite means in the display window."""
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError(f"{structure}: no finite waveform values in the display window")
    largest = max(float(np.max(np.abs(finite))), 1e-9)
    step = 10.0 ** np.floor(np.log10(largest))
    limit = float(np.ceil(1.05 * largest / step) * step)
    if structure == "trend":
        lower = min(0.0, float(np.min(finite)) * 1.05)
        return (lower, limit), [0.0, limit / 2, limit]
    return (-limit, limit), [-limit, 0.0, limit]


def panel_letter(index: int) -> str:
    """Subplot labels in row-major order, continuing after (z)."""
    if index < 0:
        raise ValueError("panel index must be non-negative")
    letters = ""
    while index >= 0:
        index, remainder = divmod(index, 26)
        letters = chr(ord("a") + remainder) + letters
        index -= 1
    return f"({letters})"


def _aligned_boundary_hours(entry: Mapping[str, Any], lag: int) -> dict[str, float]:
    """Put measured boundaries on the same shifted clock as their waveform."""
    if any(entry.get(key) is None for key in BOUNDARY_STYLES):
        raise ValueError("Figure 3 requires measured shock, sheath and ejecta-end boundaries")
    hours = {
        key: (entry[key] - WINDOW_START).total_seconds() / 3600.0 - lag / 60.0
        for key in BOUNDARY_STYLES
    }
    if not np.isfinite(list(hours.values())).all() or not np.all(np.diff(list(hours.values())) > 0):
        raise ValueError("Figure 3 boundaries must be finite and ordered")
    return hours


def build(
    out_path: str | Path = OUT,
    *,
    json_path: str | Path = INPUT_JSON,
) -> str:
    profiles = _aligned_profiles(load_result(json_path))
    boundaries, _ = event_boundaries.load()
    _publication_style()
    figure, axes = plt.subplots(
        len(CRAFT_ORDER), len(STRUCTURES),
        figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH),
    )
    figure.subplots_adjust(
        left=0.055, right=0.992, bottom=0.125, top=0.925, hspace=0.20, wspace=0.40,
    )
    for col, structure in enumerate(STRUCTURES):
        family = profiles[structure]
        lo, hi = _zoom_window(family)
        marks = {
            craft: _aligned_boundary_hours(boundaries[craft], family[craft][2])
            for craft in CRAFT_ORDER
        }
        boundary_keys = ("shock", "sheath_end") if structure in (
            "discontinuity", "substructure"
        ) else tuple(BOUNDARY_STYLES)
        boundary_lo = min(mark["shock"] for mark in marks.values())
        boundary_hi = max(mark[boundary_keys[-1]] for mark in marks.values())
        margin = 0.04 * (boundary_hi - boundary_lo)
        lo, hi = min(lo, boundary_lo - margin), max(hi, boundary_hi + margin)
        if structure == "discontinuity":
            lo = 35.0  # Display limit, not an analysis boundary.
        selected_values = np.concatenate([
            values[(time >= lo) & (time <= hi)]
            for time, values, _ in family.values()
        ])
        limits, ticks = _amplitude_axis(selected_values, structure)
        for row, craft in enumerate(CRAFT_ORDER):
            ax = axes[row, col]
            time, values, lag = family[craft]
            selected = (time >= lo) & (time <= hi)
            ax.plot(
                time[selected], values[selected],
                color=study.SATELLITES[craft]["color"], lw=0.55,
            )
            mark = marks[craft]
            for start, stop, kind in (
                (mark["shock"], mark["sheath_end"], "sheath"),
                (mark["sheath_end"], mark["ejecta_end"], "ejecta"),
            ):
                color, alpha = study.SPAN_COLORS[kind]
                ax.axvspan(start, stop, color=color, alpha=alpha, lw=0, zorder=0)
            for key in boundary_keys:
                _, color, dash = BOUNDARY_STYLES[key]
                ax.axvline(mark[key], color=color, ls=dash, lw=0.65, zorder=3, gid=key)
            ax.set_xlim(lo, hi)
            ax.set_ylim(*limits)
            ax.set_yticks(ticks)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(axis="both", length=2.2, pad=2)
            ax.axhline(0, lw=0.35, color="#999999", zorder=0)
            ax.text(
                0.035, 0.90, panel_letter(row * len(STRUCTURES) + col),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=6.8, fontweight="bold", color=study.INK,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.3},
                zorder=5,
            )
            ax.text(
                0.97, 0.16 if structure == "sheath" else 0.90,
                f"{lag:+d} min", transform=ax.transAxes,
                ha="right", va="top", fontsize=5.8, color="#666666",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.4},
            )
            if row < len(CRAFT_ORDER) - 1:
                ax.tick_params(labelbottom=False)
            for spine in ("top", "right", "bottom"):
                ax.spines[spine].set_visible(False)
        symbol = r"$|B|$" if structure == "trend" else r"$\Delta|B|$"
        axes[0, col].set_title(
            f"{STRUCTURE_LABELS[structure]}\n{symbol} [nT]",
            fontsize=TEXT_PT, pad=8,
        )
        axes[-1, col].spines["bottom"].set_visible(True)
    origin = WINDOW_START.tz_convert("UTC").strftime("%d %b %Y, %H:%M UTC")
    figure.supxlabel(
        f"Time since {origin} [h]",
        fontsize=6.8, y=0.074,
    )
    figure.legend(
        handles=[
            Line2D([], [], color=study.SATELLITES[craft]["color"],
                   label=CRAFT_LABELS[craft], linewidth=1.2)
            for craft in CRAFT_ORDER
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.035),
        ncol=len(CRAFT_ORDER), frameon=False, fontsize=7,
        labelcolor="linecolor", handlelength=1.4, handletextpad=0.4, columnspacing=0.9,
    )
    figure.legend(
        handles=[Line2D([], [], color=color, ls=dash, lw=0.8, label=label)
                 for label, color, dash in BOUNDARY_STYLES.values()],
        loc="lower center", bbox_to_anchor=(0.5, 0.002), ncol=3,
        frameon=False, fontsize=7, handlelength=2.5,
    )
    output = style.save_pdf(figure, out_path)
    return str(output)


if __name__ == "__main__":
    print(build())
