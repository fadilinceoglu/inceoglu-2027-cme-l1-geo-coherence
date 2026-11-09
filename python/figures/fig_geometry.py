"""Figure 1 — observer geometry with Shue 1998 context, 2x2.

Left column:  upstream — Solar Orbiter + the 4 L1 craft, GSE X-Y (top) and
X-Z (bottom), Earth at origin, with a zoom inset boxing the L1 cluster.
Right column: near-Earth/lunar geometry — GOES-18/19, ARTEMIS P1/P2 + Moon,
same planes, with the GEO ring.
Earth/GOES insets and two purple magnetopause outlines enlarge that comparison.
State times and model assumptions belong in the generated caption, not the art.

Positions, epoch 2026-01-19 19:00 UT:
  ACE/Wind/DSCOVR/Solar-1 — interpolated from the supplied SSCWeb cache;
  GOES-18/GOES-19 — fixed SSCWeb geometry;
  Moon/ARTEMIS — SSCWeb;
  Solar Orbiter — JPL Horizons vectors rotated to GSE.

Run:  cd python && PYTHONPATH=. <venv-python> figures/fig_geometry.py
"""

from __future__ import annotations

from pathlib import Path

from config.paths import FIGURE_DIR, RESULTS_DIR

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from analysis.geometry import reference_positions
from analysis.magnetopause import meridional_outline
from config import study
from figures import style
from figures.style import (
    FIGURE_WIDTH_MM,
    MM_PER_INCH,
    TEXT_PT as FIGURE_TEXT_PT,
    SMALL_TEXT_PT as FIGURE_SMALL_TEXT_PT,
)

OUT = FIGURE_DIR / "Fig1_geometry.pdf"

FIGURE_HEIGHT_MM = 128.0
PANEL_BOX_ASPECT = 5.0 / 7.0
UPSTREAM_X_LIMITS = (-450.0, 6600.0)
UPSTREAM_TRANSVERSE_LIMITS = {
    "xy": (-2700.0, 900.0),
    "xz": (-350.0, 1250.0),
}

# Recorded SSCWeb reference-epoch GOES coordinates; the supplied L1 cache
# contains no GOES tracks.
GOES_POSITIONS = {"goes18": (5.11, -3.29, 2.63), "goes19": (5.53, 3.38, 1.36)}
MAGNETOPAUSE_COLOR = "#7B3294"

_AX = {"xy": (0, 1, "Y"), "xz": (0, 2, "Z")}


def geo_equatorial_pole_gse() -> np.ndarray:
    """Return the fixed-epoch terrestrial equatorial pole in GSE.

    Both geostationary spacecraft lie in the terrestrial equatorial plane.
    Their non-collinear fixed-epoch GSE radius vectors therefore determine the
    plane normal without introducing a second astronomical-frame model.
    """

    goes18 = np.asarray(GOES_POSITIONS["goes18"], dtype=float)
    goes19 = np.asarray(GOES_POSITIONS["goes19"], dtype=float)
    pole = np.cross(goes18, goes19)
    norm = float(np.linalg.norm(pole))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("fixed GOES vectors do not define an equatorial plane")
    pole = pole / norm
    if pole[2] < 0.0:
        pole = -pole
    return pole


def geo_ring_gse(samples: int = 721) -> np.ndarray:
    """Return the nominal 6.6-R_E equatorial GEO ring in fixed-epoch GSE."""

    if samples < 4:
        raise ValueError("at least four samples are required for the GEO ring")
    pole = geo_equatorial_pole_gse()
    basis_x = np.asarray(GOES_POSITIONS["goes18"], dtype=float)
    basis_x = basis_x / np.linalg.norm(basis_x)
    basis_y = np.cross(pole, basis_x)
    basis_y = basis_y / np.linalg.norm(basis_y)
    phase = np.linspace(0.0, 2.0 * np.pi, samples)
    return study.GEO_ORBIT_RE * (
        np.cos(phase)[:, None] * basis_x
        + np.sin(phase)[:, None] * basis_y
    )


def _scatter(ax, key, pos, i, j, dx=0.0, dy=0.0, small=False):
    meta = study.SATELLITES[key]
    ax.scatter(pos[i], pos[j], s=26 if small else 48, color=meta["color"],
               zorder=5, edgecolors="white", linewidths=1.0)
    ax.annotate(meta["name"], xy=(pos[i], pos[j]),
                xytext=(pos[i] + dx, pos[j] + dy),
                fontsize=FIGURE_SMALL_TEXT_PT if small else FIGURE_TEXT_PT,
                color=meta["color"],
                fontweight="bold", zorder=6,
                ha="left" if dx >= 0 else "right")


def _upstream_panel(ax, plane, positions):
    i, j, name = _AX[plane]
    ax.axhline(0, color="#CCCCCC", lw=1.0, zorder=0)
    ax.scatter(0, 0, s=42, color="#2166AC", zorder=5, marker="o")
    ax.annotate("Earth", xy=(0, 0),
                xytext=(0, 120 if plane == "xy" else -120),
                fontsize=FIGURE_TEXT_PT, color="#2166AC", fontweight="bold",
                ha="center")
    so_dx, so_dy = (-260, -80) if plane == "xy" else (-260, 130)
    _scatter(ax, "solar_orbiter", positions["solar_orbiter"], i, j,
             dx=so_dx, dy=so_dy)
    for key in ("ace", "wind", "dscovr", "swfo"):
        pos = positions[key]
        ax.scatter(pos[i], pos[j], s=18, color=study.SATELLITES[key]["color"],
                   zorder=5)
    ax.annotate("L1 fleet", xy=(140, -320 if plane == "xy" else 130),
                fontsize=FIGURE_TEXT_PT, color="#444444",
                fontweight="bold")
    ax.annotate("", xy=(0.30, 0.90), xytext=(0.16, 0.90),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#E8A200", lw=1.2))
    ax.annotate("to Sun", xy=(0.23, 0.925), xycoords="axes fraction",
                fontsize=FIGURE_TEXT_PT, color="#E8A200", ha="center")
    ax.set_xlim(*UPSTREAM_X_LIMITS)
    ax.set_ylim(*UPSTREAM_TRANSVERSE_LIMITS[plane])
    ax.set_ylabel(f"GSE {name} [R$_E$]")

    # zoom inset on the L1 cluster, placed away from the SO marker
    ins = ax.inset_axes([0.50, 0.36, 0.46, 0.56] if plane == "xy"
                        else [0.52, 0.12, 0.44, 0.50])
    for key in ("ace", "wind", "dscovr", "swfo"):
        pos = positions[key]
        off = {"ace": (2, 4), "wind": (2, 6), "dscovr": (2, 4),
               "swfo": (-2, -12 if plane == "xy" else 5)}[key]
        _scatter(ins, key, pos, i, j, dx=off[0], dy=off[1], small=True)
    ins.set_xlim(195, 260)
    ins.set_ylim((-110, 60) if plane == "xy" else (-35, 20))
    ins.tick_params(labelsize=FIGURE_SMALL_TEXT_PT)
    ins.set_facecolor("#FBFBFB")
    for sp in ins.spines.values():
        sp.set_visible(True); sp.set_color("#999999")
    ax.indicate_inset_zoom(ins, edgecolor="#999999", alpha=0.7)


def _downstream_panel(ax, plane, positions):
    moon = positions["moon"]
    i, j, name = _AX[plane]
    ax.axhline(0, color="#CCCCCC", lw=1.0, zorder=0)
    earth = plt.Circle((0, 0), 1.0, color="#2166AC", zorder=4)
    ax.add_patch(earth)
    ax.annotate("Earth", xy=(0, 0), xytext=(-1.5, 3.2),
                fontsize=FIGURE_TEXT_PT,
                color="#2166AC", fontweight="bold", ha="right")
    geo = geo_ring_gse()
    ax.plot(geo[:, i], geo[:, j], lw=1.0, color="#888888", ls="-",
            alpha=0.7, zorder=2)
    ax.scatter(moon[i], moon[j], s=60, color=study.MOON_COLOR, zorder=4)
    ax.annotate("Moon", xy=(moon[i], moon[j]),
                xytext=(moon[i] - 1.5, moon[j] + 4.5),
                fontsize=FIGURE_TEXT_PT,
                color=study.MOON_COLOR, fontweight="bold", ha="right")
    _scatter(ax, "goes18", GOES_POSITIONS["goes18"], i, j, dx=1.0,
             dy=-4.5 if plane == "xy" else -4.0)
    _scatter(ax, "goes19", GOES_POSITIONS["goes19"], i, j, dx=1.0, dy=3.0)
    p1_dy = 2.8 if plane == "xy" else 3.4
    p2_dy = -5.2 if plane == "xy" else -6.0
    _scatter(ax, "artemis_p1", positions["artemis_p1"], i, j, dx=1.6,
             dy=p1_dy, small=True)
    _scatter(ax, "artemis_p2", positions["artemis_p2"], i, j, dx=1.2,
             dy=p2_dy, small=True)
    ay = 30 if plane == "xy" else 24
    ax.annotate("", xy=(78, ay), xytext=(68, ay),
                arrowprops=dict(arrowstyle="->", color="#E8A200", lw=1.2))
    ax.annotate(
        "to Sun",
        xy=(73, ay + 2.2),
        fontsize=FIGURE_TEXT_PT,
        color="#E8A200",
        ha="center",
    )
    ax.set_xlim(-32, 80)
    ax.set_ylim(-40, 40)
    ax.set_ylabel(f"GSE {name} [R$_E$]")


def _panel_indicator(ax, label):
    """Place a panel identifier inside the upper-left corner."""

    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FIGURE_TEXT_PT,
        fontweight="bold",
        zorder=20,
    )


def load_magnetopause_context(csv_path=None, metadata_path=None):
    """Read the two newly calculated states; missing products have no fallback."""
    csv_path = RESULTS_DIR / "fig1_magnetopause.csv" if csv_path is None else Path(csv_path)
    metadata_path = RESULTS_DIR / "fig1_magnetopause.json" if metadata_path is None else Path(metadata_path)
    table = pd.read_csv(csv_path)
    metadata = json.loads(metadata_path.read_text())
    required = {"case", "line_style", "r0_re", "flaring_alpha"}
    if (metadata.get("schema_version") != "fig1-magnetopause-v1" or not required.issubset(table.columns)
            or table.case.tolist() != ["response_onset", "snapshot"] or table.line_style.tolist() != ["-", "--"]):
        raise ValueError("current Figure 1 magnetopause context must contain the two declared states")
    values = table[["r0_re", "flaring_alpha"]].to_numpy(float)
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError("Figure 1 model parameters must be finite and positive")
    return table.to_dict("records")


def _magnetopause_curves(ax, states, *, inset=False):
    for state in states:
        curve = meridional_outline(state["r0_re"], state["flaring_alpha"], ax.get_xlim()[0])
        ax.plot(curve[:, 1], curve[:, 2], color=MAGNETOPAUSE_COLOR, ls=state["line_style"],
                lw=0.85 if inset else 1.1, zorder=3)


def _earth_geo_inset(ax, plane, states):
    """Earth/GOES inset matching the L1-inset style."""
    bounds = [0.48, 0.045, 0.50, 0.49] if plane == "xy" else [0.48, 0.51, 0.50, 0.45]
    ins = ax.inset_axes(bounds, zorder=10)
    ins.set(xlim=(-2, 14), ylim=(-5.5, 5.5))
    ins.set_aspect("equal", adjustable="box")
    ins.set_facecolor("#FBFBFB")
    ins.axhline(0, color="#CCCCCC", lw=0.5, zorder=0)
    j = 1 if plane == "xy" else 2
    geo = geo_ring_gse()
    ins.plot(geo[:, 0], geo[:, j], color="#888888", lw=0.75, zorder=2)
    _magnetopause_curves(ins, states, inset=True)
    ins.add_patch(plt.Circle((0, 0), 1, color="#2166AC", zorder=4))
    for craft, position in GOES_POSITIONS.items():
        meta = study.SATELLITES[craft]
        ins.scatter(position[0], position[j], s=17, color=meta["color"], edgecolors="white", linewidths=0.5, zorder=5)
        above = craft == "goes19" if plane == "xy" else craft == "goes18"
        ins.annotate(meta["name"], (position[0], position[j]), xytext=(3, 4 if above else -8),
                     textcoords="offset points", fontsize=6.2, color=meta["color"], fontweight="bold", zorder=6)
    ins.tick_params(labelsize=6, pad=1.5, length=2)
    ins.set_xticks((0, 5, 10))
    ins.set_yticks((-5, 0, 5))
    if plane == "xz":
        ins.tick_params(axis="x", labelbottom=False)
    for spine in ins.spines.values():
        spine.set_visible(True)
        spine.set_color("#999999")
    ax.indicate_inset_zoom(ins, edgecolor="#999999", alpha=0.7)
    return ins


def _downstream_context_panel(ax, plane, positions, states):
    _downstream_panel(ax, plane, positions)
    _magnetopause_curves(ax, states)
    for text in list(ax.texts):
        if text.get_text() in {"GOES-18", "GOES-19"}:
            text.remove()  # Identified in the inset; duplicate labels crowd Earth.
    if plane == "xz":
        for text in list(ax.texts):
            if text.get_text() == "to Sun" or getattr(text, "arrow_patch", None) is not None:
                text.remove()
        ax.annotate("", xy=(-9, 30), xytext=(-21, 30),
                    arrowprops={"arrowstyle": "->", "color": "#E8A200", "lw": 1.2})
        ax.text(-15, 32.2, "to Sun", fontsize=FIGURE_TEXT_PT, color="#E8A200", ha="center")
        for text in ax.texts:
            if text.get_text() == "Moon":
                text.set_position((positions["moon"][0]-3, -9))
                text.set_horizontalalignment("right")
            elif text.get_text() == "ARTEMIS-P1":
                text.set_position((positions["artemis_p1"][0]-1.5, -3.5))
                text.set_horizontalalignment("right")
    _earth_geo_inset(ax, plane, states)


def build(out_path: str | Path = OUT) -> str:
    """Build Figure 1.

    Caption information (kept here rather than rendered inside the figure):
    Observer geometry at 19:00 UT on 19 January 2026.  The left column shows
    Solar Orbiter and the four L1 spacecraft in GSE X-Y and X-Z, including an
    inset of the L1 fleet.  The right column shows GOES-18/19, the Moon, and
    ARTEMIS-P1/P2 with Earth and the nominal 6.6-R_E GEO reference.  Panel
    order is (a) upstream X-Y, (b) near-Earth/lunar X-Y, (c) upstream X-Z,
    and (d) near-Earth/lunar X-Z.  Each column shares its GSE-X scale, so the
    redundant upper-row X tick labels and axis labels are suppressed while
    the tick marks remain visible.

    Insets in (b,d) enlarge Earth and GOES. The solid/dashed purple Shue 1998
    outlines come from the current run's calculated magnetopause context, with
    their state times and assumptions in results/Fig1_caption.md. No model
    notes, timing legends, title or footer are rendered inside the artwork.

    Position provenance: SSCWeb for ACE, Wind, DSCOVR, GOES, Moon, and ARTEMIS;
    the L1, lunar and ARTEMIS coordinates are interpolated from the supplied cache;
    JPL Horizons supplied Solar Orbiter before rotation into GSE.  The GEO
    curve is the projection of the terrestrial equatorial ring whose plane is
    fixed by the two GOES radius vectors, not a circular X-Z proxy.  To retain
    Solar Orbiter and the L1 inset legibly, the upstream panels deliberately
    use unequal X/transverse scaling (displayed unit ratios Y:X = 1.40 and
    Z:X = 3.15); visual angles must not be read as physical directions.  This
    scale and GEO-projection information belongs in the caption rather than
    as additional in-panel notes.  The 180-mm-wide vector PDF embeds editable
    fonts.
    """

    style.apply_publication_style(text_size=FIGURE_TEXT_PT, tick_size=FIGURE_TEXT_PT, overrides={
        'axes.linewidth': 1.0,
        'grid.linewidth': 1.0,
        'grid.alpha': 0.08,
    })
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            FIGURE_WIDTH_MM / MM_PER_INCH,
            FIGURE_HEIGHT_MM / MM_PER_INCH,
        ),
        sharex="col",
        constrained_layout=True,
    )
    positions = reference_positions()
    _upstream_panel(axes[0, 0], "xy", positions)
    _upstream_panel(axes[1, 0], "xz", positions)
    states = load_magnetopause_context()
    _downstream_context_panel(axes[0, 1], "xy", positions, states)
    _downstream_context_panel(axes[1, 1], "xz", positions, states)
    # Give every main axes the same physical height-to-width ratio.  The
    # right-panel limits span 112 R_E by 80 R_E, so 5:7 retains equal physical
    # scaling there.  The intentionally unequal upstream scaling is disclosed
    # in the caption information above.
    for ax in axes.flat:
        ax.set_box_aspect(PANEL_BOX_ASPECT)
    for ax, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
        _panel_indicator(ax, label)
    for ax in axes[0, :]:
        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
    axes[1, 0].set_xlabel("GSE X [R$_E$]")
    axes[1, 1].set_xlabel("GSE X [R$_E$]")

    output = Path(out_path).with_suffix(".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    style.save_pdf(fig, output)
    return str(output)


if __name__ == "__main__":
    print(build())
