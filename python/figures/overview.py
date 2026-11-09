"""Per-spacecraft event overview diagnostics.

Each figure shows the quality-screened one-minute measurements available for
one observer.  Upstream and ARTEMIS views extend from 18 January through
21 January 08:00 UT so every registered magnetic-recovery endpoint and its
ten-minute confirmation interval is visible.  GOES retains the fixed
18--20 January response window.  Structure marks are supplied by the
boundary registry: Solar Orbiter, L1, and ARTEMIS use the measured
``boundaries.json`` clocks, while GOES uses the independently observed local
response interval.  The renderer never promotes a one-minute feature into a
new boundary.

Input is a standardized one-minute overview frame with columns among
``bx/by/bz/bt`` [nT], ``n`` [cm^-3], ``v`` [km/s], and ``t`` [K or eV].
Plot-only masking and QC layers identify measurements that fail the physical
health screen; the saved measurements remain available in the prepared records.

The angular and variability panels use the observed arithmetic-minute values
and the fitted boundary from build_sheath_boundaries. The renderer never repeats
boundary detection or calculates a different reference from the overview medians.

The one-minute shock zooms only orient the reader.  Their printed
``t25/t50/t75`` clocks come from the current-run native-cadence measurements and are
not re-derived from the overview traces.

Outputs are vector PDF files at 180-mm width.  Figures deliberately contain no title, subtitle, caption, or
provenance footer. Product provenance remains in the prepared-record manifests
and boundary products; measurement caveats annotate the affected panel.
"""

from __future__ import annotations

import string
from datetime import timezone

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from analysis import event_boundaries
from config.paths import RESULTS_DIR, PROCESSED_DIR
from analysis.shock_detection import logistic
from config import study
from figures import style

PUBLICATION_WIDTH_IN = style.FIGURE_WIDTH_MM / style.MM_PER_INCH
PUBLICATION_MAX_HEIGHT_IN = 170.0 / style.MM_PER_INCH
EXTENDED_OVERVIEW_STOP = pd.Timestamp("2026-01-21T08:00:00Z")
_DIAGNOSTIC_KEYS = {
    "solar_orbiter",
    "ace",
    "wind",
    "dscovr",
    "swfo",
    "artemis_p1",
    "artemis_p2",
}

_SPAN_STYLE = {
    "sheath": ("#E69F00", 0.13, "sheath"),
    "ejecta": ("#332288", 0.09, "ejecta"),
    "response": ("#777777", 0.10, "observed response"),
}


def _format_time_axis(ax, start, stop) -> None:
    ax.set_xlim(start, stop)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 12)))
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b\n%H:%M", tz=timezone.utc)
    )
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=(6, 18)))
    ax.set_xlabel("UTC")


def _mark_context(
    ax,
    marks: dict,
    *,
    label: bool,
    axis_end,
    data_handles: list | None = None,
    data_labels: list[str] | None = None,
) -> None:
    """Draw registered regions and one explicitly typed event line."""
    span_handles: list[Patch] = []
    span_labels: list[str] = []
    for t0, t1, kind in marks.get("spans", ()):
        color, alpha, text = _SPAN_STYLE.get(
            kind, ("#777777", 0.08, str(kind))
        )
        open_ended = t1 is None
        stop = t1 or axis_end
        if stop is None:
            continue
        ax.axvspan(t0, stop, color=color, alpha=alpha, lw=0, zorder=0)
        if label and text not in span_labels:
            span_handles.append(Patch(facecolor=color, alpha=max(alpha, 0.35),
                                      edgecolor="none"))
            span_labels.append(text + (" \u2192" if open_ended else ""))

    for boundary_time, boundary_kind in marks.get("boundary_lines", ()):
        if boundary_time is None:
            continue
        if boundary_kind == "sheath_end":
            color, linestyle = "#9A6700", (0, (2, 2))
        else:
            color, linestyle = "#555555", (0, (5, 2, 1, 2))
        ax.axvline(
            boundary_time,
            color=color,
            lw=1.0,
            ls=linestyle,
            zorder=4,
        )

    event_time = marks.get("line", marks.get("shock"))
    if event_time is None:
        if label and (span_handles or data_handles):
            handles = [*(data_handles or []), *span_handles]
            labels = [*(data_labels or []), *span_labels]
            ax.legend(handles, labels, loc="upper right", ncol=2)
        return
    event_label = marks.get("line_label", "shock")
    event_color = marks.get("line_color", study.SHOCK_COLOR)
    ax.axvline(
        event_time,
        color=event_color,
        lw=1.0,
        ls=(0, (4, 2)),
        zorder=4,
    )
    if label:
        handles = [
            *(data_handles or []),
            Line2D(
                [0],
                [0],
                color=event_color,
                lw=1.0,
                ls=(0, (4, 2)),
            ),
            *span_handles,
        ]
        labels = [
            *(data_labels or []),
            f"{event_label} {pd.Timestamp(event_time).round('1s'):%H:%M:%S}",
            *span_labels,
        ]
        ejecta_end = marks.get("ejecta_end")
        if ejecta_end is not None:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#555555",
                    lw=1.0,
                    ls=(0, (5, 2, 1, 2)),
                )
            )
            labels.append(f"ejecta end {ejecta_end:%d %b %H:%M}")
        ax.legend(
            handles,
            labels,
            loc="upper right",
            ncol=min(4, len(handles)),
            handlelength=1.5,
            columnspacing=0.8,
            frameon=True,
            framealpha=0.90,
            facecolor="white",
            edgecolor="none",
            fontsize=7.0,
        )



def _format_interval(start: pd.Timestamp, stop: pd.Timestamp) -> str:
    """Compact, unambiguous UTC interval for an in-panel method note."""
    start = pd.Timestamp(start).tz_convert("UTC")
    stop = pd.Timestamp(stop).tz_convert("UTC")
    if start.date() == stop.date():
        return (
            f"{start:%d %b %H:%M:%S}\u2013{stop:%H:%M:%S} UT"
        )
    return f"{start:%d %b %H:%M}\u2013{stop:%d %b %H:%M} UT"


def _half_open_rows(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    stop: pd.Timestamp,
) -> pd.DataFrame:
    """Select the canonical half-open display interval ``[start, stop)``."""

    return frame.loc[(frame.index >= start) & (frame.index < stop)]


def _registered_times(key: str) -> dict[str, pd.Timestamp | None]:
    boundaries, _ = event_boundaries.load()
    if key not in boundaries:
        raise KeyError(f"registered boundaries do not contain {key!r}")
    entry = boundaries[key]
    return {
        name: pd.Timestamp(value) if value is not None else None
        for name, value in entry.items()
    }




def _magnetic_diagnostics(
    key: str,
    plot: pd.DataFrame,
    labels: dict[str, str],
    notes: dict[str, str],
) -> dict:
    """Display the observed minute evidence and criteria used by each detector."""
    registered = _registered_times(key)
    sheath_end, shock, ejecta_end = (registered[k] for k in ("sheath_end", "shock", "ejecta_end"))
    is_artemis = key in {"artemis_p1", "artemis_p2"}
    measured = event_boundaries.sheath_measurement(key)
    ref_start, ref_stop = measured["reference_start_utc"], measured["reference_stop_utc"]
    collapse = measured["variability_marker_utc"]
    evidence = pd.read_csv(RESULTS_DIR / "sheath_boundary_diagnostics.csv", parse_dates=["time"])
    evidence = evidence.loc[evidence.spacecraft.eq(key)].set_index("time")
    evidence.index = pd.to_datetime(evidence.index, utc=True)
    # Use the actual observed arithmetic-minute angles and variability used by
    # the detector, not angles recomputed from the overview's minute medians.
    plot["theta_ej"] = evidence.theta_ejecta_deg.reindex(plot.index)
    plot["field_var"] = evidence.moving_std_nt.reindex(plot.index)
    field_var_threshold = measured["settings"]["variability_threshold_nt"]
    settings = measured["settings"]
    labels["field_var"] = rf"$s_{{|B|,{settings['variability_window_min']}}}$ [nT]"
    notes["field_var"] = (
        f"centered {settings['variability_window_min']}-min std; <{field_var_threshold:g} nT "
        f"in >={settings['persistence_fraction']:.0%} of next {settings['persistence_duration_min']} min"
    )
    labels["theta_ej"] = r"$\theta_{\rm ej}$ [deg]"
    notes["theta_ej"] = (f"mean unit-vector reference; {_format_interval(ref_start, ref_stop)}"
                          "\nobserved minute means; logistic angular transition")

    recovery = event_boundaries.ejecta_measurement(key)
    ambient_lower = recovery["background"]["lower"]
    ambient_upper = recovery["background"]["upper"]
    ambient_threshold = recovery["upper_threshold_nt"]
    observed = pd.read_csv(PROCESSED_DIR / f"common_cadence_1min/{key}.csv",
                           index_col=0, parse_dates=[0])
    observed.index = pd.to_datetime(observed.index, utc=True)
    bt = observed.bt.where((observed.bt_samples > 0) & np.isfinite(observed.bt))
    plot["bt_recovery"] = bt.reindex(plot.index)
    ambient_label = "background |B| 5-95%"
    edge_note = (f"{recovery['qualifying_minutes']}/{recovery['window_minutes']} min qualify"
                 f"\n{ejecta_end:%d Jan %H:%M} UT")

    return {
        "shock": shock,
        "sheath_fit": measured,
        "sheath_end": sheath_end,
        "ejecta_end": ejecta_end,
        "reference_start": ref_start,
        "reference_stop": ref_stop,
        "field_var_threshold": field_var_threshold,
        "collapse": collapse,
        "ambient_lower": ambient_lower,
        "ambient_upper": ambient_upper,
        "ambient_threshold": ambient_threshold,
        "ambient_label": ambient_label,
        "edge_note": edge_note,
        "confirmation_stop": recovery["confirmation_stop_utc_exclusive"],
        "native_shock": tuple(event_boundaries.shock_measurement(key)[f"t{p}"] for p in (25, 50, 75)),
        "is_artemis": is_artemis,
    }


def _add_panel_letter(ax, index: int, *, compact: bool = False) -> None:
    ax.text(
        0.006,
        0.965,
        f"({string.ascii_lowercase[index]})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0 if compact else 8.0,
        fontweight="bold",
        color="#222222",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 0.35 if compact else 0.5,
        },
        zorder=10,
    )


def _pre_event_note_anchor(
    marks: dict,
    start: pd.Timestamp,
    stop: pd.Timestamp,
) -> pd.Timestamp:
    """Place panel caveats just before the event instead of over its signal."""
    event = marks.get("shock", marks.get("line"))
    span = stop - start
    if event is None:
        return start + 0.48 * span
    anchor = pd.Timestamp(event) - 0.015 * span
    return max(anchor, start + 0.34 * span)


def _zoom_time_axis(ax, locator) -> None:
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M", tz=timezone.utc)
    )
    ax.tick_params(axis="both", which="major", labelsize=7.0, length=2.0)
    ax.locator_params(axis="y", nbins=3)
    ax.grid(True, linewidth=0.35, alpha=0.10)


def _draw_zoom_row(
    fig,
    slot,
    plot: pd.DataFrame,
    diagnostics: dict,
    *,
    first_letter: int,
    accent: str,
) -> list:
    """Compact evidence zooms; full variability evidence is in the dedicated PDF."""
    n_zoom = 4 if diagnostics["is_artemis"] else 3
    grid = slot.subgridspec(1, n_zoom, wspace=0.32)
    axes = [fig.add_subplot(grid[0, j]) for j in range(n_zoom)]
    if diagnostics["is_artemis"]:
        shock_ax, theta_ax, var_ax, end_ax = axes
    else:
        shock_ax, theta_ax, end_ax = axes
        var_ax = None

    # Shock: the trace is one-minute context only; exact clocks are frozen
    # native-cadence t25/t50/t75 values printed in the annotation.
    t25, t50, t75 = diagnostics["native_shock"]
    shock_pad = pd.Timedelta("12min")
    shock_ax.plot(plot.index, plot["bt"], color=study.INK, lw=1.0)
    shock_ax.axvspan(t25, t75, color=study.SHOCK_COLOR, alpha=0.22, lw=0)
    shock_ax.axvline(t50, color=study.SHOCK_COLOR, lw=1.0, ls=(0, (4, 2)))
    shock_ax.set_xlim(t25 - shock_pad, t75 + shock_pad)
    shock_ax.set_ylabel("|B| [nT]", fontsize=7.0)
    shock_ax.set_xlabel("shock zoom: 1-min trace (UT)", fontsize=7.0)
    native_times = [pd.Timestamp(t).round("1ms") for t in (t25, t50, t75)]
    same_minute = len(
        {(t.year, t.month, t.day, t.hour, t.minute) for t in native_times}
    ) == 1
    if same_minute:
        sec = [
            f"{t.second + t.microsecond / 1e6:06.3f}"
            for t in native_times
        ]
        clock_lines = (
            f"{native_times[0]:%H:%M}:{sec[0]} / {sec[1]}"
            f"\n/ {sec[2]} UT"
        )
    else:
        clock_lines = "\n".join(
            f"{t:%H:%M:%S}.{t.microsecond // 1000:03d}" for t in native_times
        )
    shock_ax.text(
        0.98,
        0.80,
        "native 25 / 50 / 75\n" + clock_lines,
        transform=shock_ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86,
              "pad": 0.6},
        zorder=8,
    )
    _zoom_time_axis(shock_ax, mdates.MinuteLocator(interval=10))

    # Sheath edge: observed minute angles and the registered logistic fit.
    sheath = diagnostics["sheath_end"]
    theta_ax.plot(plot.index, plot["theta_ej"], color=accent, lw=0.9)
    theta_ax.axvspan(
        diagnostics["reference_start"],
        diagnostics["reference_stop"],
        color="#56B4E9",
        alpha=0.13,
        lw=0,
    )
    fit = diagnostics["sheath_fit"]
    grid = pd.date_range(fit["fit_window_start_utc"], fit["fit_window_stop_utc"], periods=301)
    relative = (grid - fit["boundary_utc"]).total_seconds() / 60
    theta_ax.plot(grid, logistic(relative, fit["before_deg"], fit["after_deg"],
                                0., np.log(fit["tau_minutes"])), color="#D95F02", lw=1.0)
    theta_ax.axvline(sheath, color="#9A6700", lw=1.0, ls=(0, (2, 2)))
    theta_ax.set_xlim(sheath - pd.Timedelta("20min"),
                      sheath + pd.Timedelta("20min"))
    theta_ax.set_ylim(0, 180)
    theta_ax.set_yticks([0, 90, 180])
    theta_ax.set_ylabel(r"$\theta_{\rm ej}$ [deg]", fontsize=7.0)
    theta_ax.set_xlabel("sheath-edge angle zoom (UT)", fontsize=7.0)
    _zoom_time_axis(theta_ax, mdates.MinuteLocator(interval=20))

    # Variability collapse is shown only where it is an executable detector:
    # the ARTEMIS centred 10-min std(|B|) decision variable.
    if var_ax is not None:
        var_ax.plot(plot.index, plot["field_var"], color=accent, lw=0.9)
        var_threshold = diagnostics["field_var_threshold"]
        var_ax.axhline(var_threshold, color="#555555", lw=0.9, ls="--")
        collapse = diagnostics["collapse"]
        var_ax.axvline(collapse, color="#117733", lw=0.9, ls=(0, (3, 2)))
        var_ax.axvspan(
            collapse,
            collapse + pd.Timedelta("10min"),
            color="#117733",
            alpha=0.09,
            lw=0,
        )
        var_ax.axvline(sheath, color="#9A6700", lw=1.0, ls=(0, (2, 2)))
        var_ax.set_xlim(sheath - pd.Timedelta("45min"),
                        sheath + pd.Timedelta("45min"))
        var_ax.set_ylim(bottom=0)
        var_ax.set_ylabel(r"$s_{|B|,10}$ [nT]", fontsize=7.0)
        var_ax.set_xlabel("variability zoom (UT)", fontsize=7.0)
        var_ax.text(
            0.98,
            0.80,
            f"collapse {collapse:%H:%M}\n10-min hold",
            transform=var_ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.0,
            color="#333333",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
                "pad": 0.6,
            },
            zorder=8,
        )
        _zoom_time_axis(var_ax, mdates.MinuteLocator(interval=45))

    # Use the same observed minute means and threshold as the recovery detector.
    ejecta_end = diagnostics["ejecta_end"]
    end_ax.plot(plot.index, plot["bt_recovery"], color=accent, lw=1.0)
    end_ax.axhspan(
        diagnostics["ambient_lower"],
        diagnostics["ambient_upper"],
        color="#777777",
        alpha=0.14,
        lw=0,
    )
    ambient_threshold = diagnostics["ambient_threshold"]
    if ambient_threshold is not None:
        end_ax.axhline(
            ambient_threshold, color="#555555", lw=0.9, ls="--"
        )
        end_ax.axvspan(
            ejecta_end,
            diagnostics["confirmation_stop"],
            color="#117733",
            alpha=0.08,
            lw=0,
        )
        end_note = (
            f"$q_{{95}}={ambient_threshold:.1f}$ nT\n"
            + diagnostics["edge_note"]
        )
    else:
        end_note = diagnostics["edge_note"]
    end_ax.axvline(
        ejecta_end, color="#555555", lw=1.0, ls=(0, (5, 2, 1, 2))
    )
    end_ax.set_xlim(
        ejecta_end - pd.Timedelta("4h"),
        ejecta_end + pd.Timedelta("4h"),
    )
    end_ax.set_ylim(bottom=0)
    end_ax.set_ylabel("|B| [nT]", fontsize=7.0)
    end_ax.set_xlabel("ejecta-end zoom (UT)", fontsize=7.0)
    end_ax.text(
        0.98,
        0.80,
        end_note,
        transform=end_ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86,
              "pad": 0.6},
        zorder=8,
    )
    _zoom_time_axis(end_ax, mdates.HourLocator(interval=4))

    for j, ax in enumerate(axes):
        _add_panel_letter(ax, first_letter + j, compact=True)
    return axes


def build_overview(
    key: str,
    df,
    *,
    out_path: str,
    marks: dict,
    frame_label: str,
    t_unit: str = "K",
    panel_labels: dict[str, str] | None = None,
    panel_notes: dict[str, str] | None = None,
    diagnostics: dict | None = None,
) -> str:
    """Render PDF overview diagnostics.

    ``marks`` accepts ``spans=[(start, stop, kind)]`` plus either
    ``shock=<time>`` or the typed form ``line=<time>, line_label=<text>``.
    ``panel_labels`` and ``panel_notes`` apply only to this rendering and are
    used for instrument-health caveats without changing a source cache.
    ``diagnostics`` contains display evidence derived by
    :func:`_magnetic_diagnostics`; it never supplies a boundary.
    """
    meta = study.SATELLITES[key]
    accent = meta["color"]
    panel_labels = panel_labels or {}
    panel_notes = panel_notes or {}

    panels = []
    if "bt" in df and np.isfinite(df["bt"]).any():
        panels.append(("bt", "|B| [nT]"))
    comps = [c for c in ("bx", "by", "bz") if c in df and np.isfinite(df[c]).any()]
    if comps:
        panels.append(("components", f"B {frame_label} [nT]"))
    for col, lab in (
        ("theta_ej", r"$\theta_{\rm ej}$ [deg]"),
        ("field_var", r"$s_{|B|,10}$ [nT]"),
    ):
        if col in df and np.isfinite(df[col]).any():
            panels.append((col, panel_labels.get(col, lab)))
    for col, lab in (("n", "n [cm$^{-3}$]"), ("v", "V [km s$^{-1}$]"),
                     ("t", f"T [{t_unit}]")):
        if col in df and np.isfinite(df[col]).any():
            panels.append((col, panel_labels.get(col, lab)))
    if not panels:
        raise ValueError(f"{key}: no plottable columns in overview cache")

    start = pd.Timestamp(study.ANALYSIS_START)
    stop = (
        EXTENDED_OVERVIEW_STOP
        if key in _DIAGNOSTIC_KEYS
        else pd.Timestamp(study.ANALYSIS_STOP)
    )
    dfw = _half_open_rows(df, start, stop)

    height = (
        PUBLICATION_MAX_HEIGHT_IN
        if diagnostics is not None
        else max(3.0, 1.28 * len(panels) + 0.55)
    )
    with mpl.rc_context(style.OVERVIEW_STYLE):
        if diagnostics is not None:
            fig = plt.figure(
                figsize=(PUBLICATION_WIDTH_IN, height),
                constrained_layout=True,
            )
            grid = fig.add_gridspec(
                len(panels) + 1,
                1,
                height_ratios=[1.0] * len(panels) + [1.55],
            )
            axes = []
            for i in range(len(panels)):
                axes.append(
                    fig.add_subplot(
                        grid[i, 0],
                        sharex=axes[0] if axes else None,
                    )
                )
            zoom_slot = grid[-1, 0]
        else:
            fig, axes = plt.subplots(
                len(panels),
                1,
                sharex=True,
                figsize=(PUBLICATION_WIDTH_IN, height),
                constrained_layout=True,
            )
            axes = list(np.atleast_1d(axes))
            zoom_slot = None

        for i, (ax, (panel, ylabel)) in enumerate(zip(axes, panels)):
            data_handles: list = []
            data_labels: list[str] = []
            if panel == "components":
                names = {
                    "RTN": {"bx": "$B_R$", "by": "$B_T$", "bz": "$B_N$"}
                }.get(
                    frame_label,
                    {c: f"$B_{{{c[-1].upper()}}}$"
                     for c in ("bx", "by", "bz")},
                )
                for c in comps:
                    ax.plot(
                        dfw.index,
                        dfw[c],
                        lw=1.0,
                        color=study.COMPONENT_COLORS[c],
                        label=names[c],
                    )
                ax.axhline(0, color="#888888", lw=1.0, zorder=1)
                ax.legend(
                    loc="upper right",
                    ncol=3,
                    handlelength=1.2,
                    columnspacing=0.9,
                )
            elif panel == "bt":
                ax.plot(
                    dfw.index,
                    dfw["bt"],
                    lw=1.0,
                    color=study.INK,
                    zorder=2,
                )
                if diagnostics is not None:
                    ax.axhspan(
                        diagnostics["ambient_lower"],
                        diagnostics["ambient_upper"],
                        color="#777777",
                        alpha=0.14,
                        lw=0,
                        zorder=0.5,
                    )
                    ax.plot(
                        dfw.index,
                        dfw["bt_recovery"],
                        lw=1.1,
                        color=accent,
                        zorder=3,
                    )
                    data_handles.extend(
                        [
                            Line2D([0], [0], color=accent, lw=1.1),
                            Patch(
                                facecolor="#777777",
                                alpha=0.25,
                                edgecolor="none",
                            ),
                        ]
                    )
                    data_labels.extend(
                        ["observed minute mean", diagnostics["ambient_label"]]
                    )
                    threshold = diagnostics["ambient_threshold"]
                    if threshold is not None:
                        ax.axhline(
                            threshold,
                            color="#555555",
                            lw=1.0,
                            ls="--",
                            zorder=3,
                        )
                        data_handles.append(
                            Line2D(
                                [0], [0], color="#555555", lw=1.0, ls="--"
                            )
                        )
                        data_labels.append(r"$q_{95}$ threshold")
            elif panel == "theta_ej":
                ax.plot(dfw.index, dfw[panel], lw=1.0, color=accent)
                ax.set_ylim(0, 180)
                ax.set_yticks([0, 90, 180])
            elif panel == "field_var":
                ax.plot(dfw.index, dfw[panel], lw=1.0, color=accent)
                threshold = diagnostics["field_var_threshold"]
                ax.axhline(
                    threshold, color="#555555", lw=1.0, ls="--"
                )
                collapse = diagnostics["collapse"]
                ax.axvspan(
                    collapse,
                    collapse + pd.Timedelta("10min"),
                    color="#117733",
                    alpha=0.09,
                    lw=0,
                )
                ax.set_ylim(bottom=0)
            else:
                series = dfw[panel]
                rejected_col = f"{panel}_rejected"
                has_rejected = (
                    rejected_col in dfw
                    and np.isfinite(dfw[rejected_col]).any()
                )
                if has_rejected:
                    ax.plot(
                        dfw.index,
                        dfw[rejected_col],
                        lw=0.8,
                        color="#9A9A9A",
                        ls=(0, (3, 2)),
                        alpha=0.85,
                        zorder=1,
                    )
                ax.plot(dfw.index, series, lw=1.0, color=accent, zorder=2)
                if has_rejected:
                    ax.legend(
                        [
                            Line2D([0], [0], color=accent, lw=1.0),
                            Line2D(
                                [0],
                                [0],
                                color="#9A9A9A",
                                lw=0.8,
                                ls=(0, (3, 2)),
                            ),
                        ],
                        ["accepted", "QC-only / rejected"],
                        loc="lower right",
                        ncol=2,
                        handlelength=1.3,
                        columnspacing=0.8,
                        fontsize=7.0,
                    )
                if panel == "n":
                    positive = series[series > 0]
                    if (
                        len(positive)
                        and np.nanmax(positive) / max(np.nanmin(positive), 1e-3)
                        > 50
                    ):
                        ax.set_yscale("log")

            ax.set_ylabel(ylabel)
            _mark_context(
                ax,
                marks,
                label=(i == 0),
                axis_end=stop,
                data_handles=data_handles,
                data_labels=data_labels,
            )
            _add_panel_letter(ax, i)
            note = panel_notes.get(panel)
            if note:
                note_is_bottom = panel in {"bt", "theta_ej"}
                ax.text(
                    _pre_event_note_anchor(marks, start, stop),
                    0.055 if note_is_bottom else 0.94,
                    note,
                    transform=ax.get_xaxis_transform(),
                    ha="right",
                    va="bottom" if note_is_bottom else "top",
                    fontsize=7.0,
                    color="#444444",
                    bbox={"facecolor": "white", "edgecolor": "#BBBBBB",
                          "linewidth": 0.5, "alpha": 0.90, "pad": 1.4},
                    zorder=7,
                )

        _format_time_axis(axes[-1], start, stop)
        for ax in axes[:-1]:
            ax.tick_params(axis="x", which="both", labelbottom=False)
        if diagnostics is not None:
            _draw_zoom_row(
                fig,
                zoom_slot,
                dfw,
                diagnostics,
                first_letter=len(panels),
                accent=accent,
            )
        pdf = str(style.save_pdf(fig, out_path))
    return pdf


def _registered_marks(key: str) -> dict:
    """Return final physical boundaries or the observed GOES response window."""
    if key in {"goes18", "goes19"}:
        return {
            "line": study.GEO_RESPONSE_ONSET,
            "line_label": "response onset",
            "line_color": "#555555",
            "spans": [
                (
                    study.GEO_RESPONSE_ONSET,
                    study.GEO_SHEATH_RESPONSE_STOP,
                    "response",
                )
            ],
        }
    marks = event_boundaries.marks_for(key)
    registered = _registered_times(key)
    marks.update(
        {
            "sheath_end": registered["sheath_end"],
            "ejecta_end": registered["ejecta_end"],
            "boundary_lines": [
                (registered["sheath_end"], "sheath_end"),
                (registered["ejecta_end"], "ejecta_end"),
            ],
        }
    )
    return marks


def _scientific_plot_copy(
    key: str, df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], dict | None]:
    """Apply display-only health-screen decisions without changing a cache.

    Basic fill/range/quality screening happens in the spacecraft builders.
    These operations carry the later physical shock-health decisions into the
    supplementary figures:

    * ACE event plasma during the SWEPAM tracker lock and DSCOVR post-shock
      plasma are retained as thin gray ``QC-only / rejected`` layers, while
      accepted values remain in the craft color.
    * Solar-1 shows the supplied corrected SWiPS density in physical units.
    * Wind and ARTEMIS keep their measured gaps; no line is interpolated.
    """
    plot = df.copy()
    labels: dict[str, str] = {}
    notes: dict[str, str] = {}

    if key == "solar_orbiter":
        notes["n"] = "PAS gaps retained; no interpolation"
    elif key == "ace":
        bad = (
            (plot.index >= pd.Timestamp("2026-01-19T09:30:00Z"))
            & (plot.index <= pd.Timestamp("2026-01-19T21:16:00Z"))
        )
        for col in ("n", "v", "t"):
            plot[f"{col}_rejected"] = plot[col].where(bad)
            plot.loc[bad, col] = np.nan
        notes["n"] = (
            "gray dashed = QC-only / scientifically rejected"
            "\nSWEPAM tracker lock, 09:30\u201321:16 UT"
        )
    elif key == "wind":
        analysis = plot.loc[
            (plot.index >= pd.Timestamp(study.ANALYSIS_START))
            & (plot.index < EXTENDED_OVERVIEW_STOP),
            ["n", "v", "t"],
        ]
        finite = 100.0 * float(analysis.notna().to_numpy().mean())
        notes["n"] = f"SWE moments sparse; {finite:.0f}% of 1-min bins finite"
    elif key == "dscovr":
        shock = event_boundaries.marks_for(key)["shock"]
        bad = plot.index >= shock
        for col in ("n", "v", "t"):
            plot[f"{col}_rejected"] = plot[col].where(bad)
            plot.loc[bad, col] = np.nan
        notes["n"] = (
            "gray dashed = QC-only / scientifically rejected"
            "\nwrong-way post-shock moment jump"
        )
    elif key == "swfo":
        notes["n"] = "corrected SWiPS commissioning moments; gaps retained"
    elif key in {"artemis_p1", "artemis_p2"}:
        notes["v"] = "ESA V sparse; measured samples only"
        notes["t"] = "ESA T is measurement-mode dependent"
        if key == "artemis_p2":
            notes["n"] = "wake/shadow context retained; no interpolation"

    diagnostics = (
        _magnetic_diagnostics(key, plot, labels, notes)
        if key in _DIAGNOSTIC_KEYS
        else None
    )
    return plot, labels, notes, diagnostics


def build_registered_overview(
    key: str,
    df: pd.DataFrame,
    *,
    out_path: str,
    frame_label: str,
    t_unit: str = "K",
) -> str:
    """Render one overview using final marks and physical health-screen rules."""
    plot, labels, notes, diagnostics = _scientific_plot_copy(key, df)
    return build_overview(
        key,
        plot,
        out_path=out_path,
        marks=_registered_marks(key),
        frame_label=frame_label,
        t_unit=t_unit,
        panel_labels=labels,
        panel_notes=notes,
        diagnostics=diagnostics,
    )
