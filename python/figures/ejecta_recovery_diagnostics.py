"""Show the measured background, magnetic recovery and supporting plasma returns."""

import json
from datetime import timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.ejecta_recovery import observed_minutes
from config import study
from config.study import UPSTREAM_SPACECRAFT
from config.paths import DIAGNOSTIC_DIR, PROCESSED_DIR, RESULTS_DIR


def diagnostic_page(pdf, craft, frame, report, manifest):
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(4, 2, figsize=(11.7, 10), sharex="col",
                             gridspec_kw={"width_ratios": [1, 2.5]})
    fig.subplots_adjust(left=.085, right=.97, top=.87, bottom=.155, hspace=.28, wspace=.24)
    start, stop = report["background_start_utc"], report["background_stop_utc_exclusive"]
    bg = report["background"]["bt"]
    name = study.SATELLITES[craft]["name"]
    confirmation = report["magnetic_recovery"]
    fig.suptitle(f"{name}: confirmed magnetic recovery", y=.977, fontsize=14, fontweight="bold")
    fig.text(.5, .947, f"Magnetic rule: at least {confirmation['required_minutes']} of {confirmation['window_minutes']} minutes at or below the background upper limit; no extra averaging.",
             ha="center", fontsize=10)
    fig.text(.5, .92, f"Reference: {start:%d Jan %H:%M}-{stop:%H:%M} UT. "
             f"Magnetic background: {bg['lower']:.2f}-{bg['upper']:.2f} nT (5th-95th percentiles).",
             ha="center", fontsize=9)
    axes[0, 0].set_title("Selected background and adjacent measurements", fontsize=9)
    axes[0, 1].set_title("Magnetic recovery and supporting plasma first returns", fontsize=9)
    keys = ("bt", "theta_ej", "v", "t")
    labels = ("|B| [nT]", r"$\theta_{ej}$ [deg]", "Speed [km/s]",
              f"Temperature [{manifest['spacecraft'][craft]['temperature_unit']}]")
    limits = ((start - pd.Timedelta("30min"), stop + pd.Timedelta("30min")),
              (report["sheath_entry_utc"] - pd.Timedelta("1h"), pd.Timestamp(study.ANALYSIS_STOP)))
    for row, (key, label) in enumerate(zip(keys, labels)):
        measurement = report["returns"].get(key)
        excluded = measurement is not None and measurement["status"] == "excluded"
        for col, (left, right) in enumerate(limits):
            ax = axes[row, col]
            values = frame.loc[(frame.index >= left) & (frame.index < right), key].copy()
            if craft == "dscovr" and excluded and col == 1:
                values[:] = np.nan
                ax.set_yticks([])
            ax.plot(values.index, values, color="#999999" if excluded else "#374151",
                    lw=.6 if key in ("bt", "theta_ej") else 0,
                    marker=None if key in ("bt", "theta_ej") else ".", ms=1.6)
            if measurement is not None and not excluded:
                stats = report["background"][key]
                if stats["lower"] is not None:
                    ax.axhspan(stats["lower"], stats["upper"],
                               color="#2563eb", alpha=.12, lw=0)
                    if key == "bt":
                        ax.axhline(stats["upper"], color="#2563eb", lw=.6, ls=":")
            if col == 0:
                ax.axvspan(start, stop, color="#2563eb", alpha=.09, lw=0)
                ax.set_ylabel(label)
                ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
            else:
                ax.axvline(report["sheath_entry_utc"], color="#64748b", ls=":", lw=.8)
                if measurement is not None:
                    match = confirmation["time_utc"] if key == "bt" else measurement["first_in_range_utc"]
                    if match is not None:
                        ax.axvline(match, color="#047857", lw=1)
                        value = confirmation["value_nt"] if key == "bt" else measurement["first_in_range_value"]
                        ax.scatter([match], [value],
                                   s=24, edgecolor="#047857", facecolor="white", zorder=5)
                        prefix = ("Confirmed recovery" if key == "bt" else
                                  "Already inside" if measurement["status"] != "first_return" else "First return")
                        text = f"{prefix}: {match:%d Jan %H:%M} UT"
                        if key == "bt":
                            text += f" ({confirmation['qualifying_minutes']}/{confirmation['window_minutes']} minutes)"
                            ax.axvspan(match, confirmation["confirmation_stop_utc_exclusive"],
                                       color="#047857", alpha=.12, lw=0)
                    else:
                        text = ("No confirmed magnetic recovery" if key == "bt" else
                                {"excluded": "Plasma return excluded (instrument limitation)",
                                "no_return_in_record": "No return within 18-21 January",
                                "insufficient_background": "Insufficient background measurements",
                                "no_post_entry_observations": "No post-entry observations"}[measurement["status"]])
                    ax.text(.98, .90, text, transform=ax.transAxes, ha="right", va="top", fontsize=8,
                            bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 2})
                ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d Jan\n%H:%M", tz=timezone.utc))
            if key == "theta_ej":
                ax.set_ylim(0, 180)
            if key == "t" and (values > 0).any():
                ax.set_yscale("log")
            ax.set_xlim(left, right)
            ax.grid(alpha=.13)
    axes[-1, 0].set_xlabel(f"{start:%d January %Y}, UTC")
    axes[-1, 1].set_xlabel("UTC")
    fig.legend(handles=[Patch(facecolor="#2563eb", alpha=.15, label="Background range"),
                        Line2D([], [], color="#047857", label="Magnetic recovery / plasma first entry")],
               loc="lower center", bbox_to_anchor=(.5, .068), ncol=2, frameon=False)
    fig.text(.5, .047, report["plasma_quality_note"], ha="center", fontsize=8)
    fig.text(.5, .026, "The angle uses the ejecta reference for context. "
             "Plasma does not select the magnetic recovery time.", ha="center", fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    angles = pd.read_csv(RESULTS_DIR / "sheath_boundary_diagnostics.csv", parse_dates=["time"])
    angles["time"] = pd.to_datetime(angles.time, utc=True)
    reports = json.loads((RESULTS_DIR / "ejecta_recovery.json").read_text())["spacecraft"]
    manifest = json.loads((PROCESSED_DIR / "common_cadence_1min/manifest.json").read_text())
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    with PdfPages(DIAGNOSTIC_DIR / "ejecta_recovery.pdf") as pdf:
        for craft in UPSTREAM_SPACECRAFT:
            report = reports[craft]
            for key in ("background_start_utc", "background_stop_utc_exclusive", "sheath_entry_utc"):
                report[key] = pd.Timestamp(report[key])
            for key in ("time_utc", "confirmation_stop_utc_exclusive"):
                report["magnetic_recovery"][key] = pd.Timestamp(report["magnetic_recovery"][key])
            for result in report["returns"].values():
                if result["first_in_range_utc"] is not None:
                    result["first_in_range_utc"] = pd.Timestamp(result["first_in_range_utc"])
            frame = pd.read_csv(PROCESSED_DIR / f"common_cadence_1min/{craft}.csv", index_col=0, parse_dates=[0])
            frame = observed_minutes(frame)
            frame = frame.loc[(frame.index >= study.ANALYSIS_START) & (frame.index < study.ANALYSIS_STOP)]
            measured = angles.loc[angles.spacecraft.eq(craft)].set_index("time")
            if measured.index.has_duplicates or not frame.index.isin(measured.index).all():
                raise ValueError(f"Incomplete or duplicate sheath diagnostic clocks for {craft}")
            frame["theta_ej"] = measured.theta_ejecta_deg.reindex(frame.index)
            diagnostic_page(pdf, craft, frame, report, manifest)


if __name__ == "__main__":
    main()
