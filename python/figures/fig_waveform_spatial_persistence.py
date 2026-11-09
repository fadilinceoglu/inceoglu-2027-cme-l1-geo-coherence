"""Near-Earth scalar correlation, scalar coherence and GSM Bz correlation.

The figure reads seven-observer scalar inference and wavelet products. All three rows show the fifteen near-Earth pairs; the same
six L1--L1 plus four L1--P1 links enter each independent exponential fit. P2
links provide wake context. Solar Orbiter points and trend are not displayed.
The Bz row uses the scalar-derived pair lag, not an independently optimized Bz
lag. Its synchronized source-origin bootstrap repeats the scalar lag search.
Wavelet ranges remain conditional on fixed maps, shock registration and weights.
"""

from __future__ import annotations

from pathlib import Path

from config.paths import RESULTS_DIR, FIGURE_DIR
from typing import Any, Final, Mapping

import matplotlib

matplotlib.use("Agg")

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.waveform_inference import (
    BOOTSTRAP_DRAWS,
    NULL_SURROGATES,
    load_inference_product,
)
from analysis.bz_persistence import (
    load_bz_persistence, validate_bz_persistence,
)
from analysis.wavelet_coherence import (
    BOOTSTRAP_DRAWS as WAVELET_BOOTSTRAP_DRAWS,
    WHITE_DRAWS,
    load_wavelet_coherence,
    validate_wavelet_coherence,
)
from config import study
from config.waveform_bands import WAVEFORM_FAMILIES, WAVEFORM_FAMILY_LABELS
from analysis.geometry import reference_positions
from figures import style
from figures.style import FIGURE_WIDTH_MM, MM_PER_INCH, TEXT_PT


INPUT_JSON: Final = (
    RESULTS_DIR / "waveform_inference.json"
)
INPUT_CSV: Final = INPUT_JSON.with_suffix(".csv")
WAVELET_INPUT_JSON: Final = RESULTS_DIR / "wavelet_coherence.json"
BZ_INPUT_JSON: Final = RESULTS_DIR / "bz_persistence.json"
OUT: Final = (
    FIGURE_DIR
    / "Fig5_waveform_coherence_length_scales.pdf"
)

PLOTTED_STRUCTURES = WAVEFORM_FAMILIES[:-1]
L1_CRAFT = study.L1_SPACECRAFT
ONE_OVER_E: Final = float(np.exp(-1.0))
# Display limits are numerical layout choices, not inferred scales.
MAIN_X_LIMITS_RE: Final = (2.7, 350.0)
MAIN_X_TICKS_RE: Final = (3.0, 10.0, 30.0, 100.0, 300.0)

FIGURE_HEIGHT_MM: Final = 157.0
SMALL_TEXT_PT: Final = 6.0
GRID: Final = "#D8D8D8"
# Neutral fit colour keeps the model distinct from spacecraft pair classes.
FIT_COLOR: Final = "#1a1a1a"
FIT_BAND_COLOR: Final = "#9a9a9a"
EARTH_CONTEXT_COLOR: Final = "#A96A00"
WHITE_BENCHMARK_COLOR: Final = "#825A99"
FAMILY_TITLES: Final = WAVEFORM_FAMILY_LABELS


POINT_STYLES: Final = {
    "l1_l1": {
        "label": "L1–L1 (fit)",
        "marker": "o",
        "edge": "#333333",
        "face": "white",
        "size": 27,
        "zorder": 7,
    },
    "l1_p1": {
        "label": "L1→P1 incident (fit)",
        "marker": "^",
        "edge": study.SATELLITES["artemis_p1"]["color"],
        "face": study.SATELLITES["artemis_p1"]["color"],
        "size": 34,
        "zorder": 8,
    },
    "wake_context": {
        "label": "P2 wake context",
        "marker": "X",
        "edge": study.SATELLITES["artemis_p2"]["color"],
        "face": study.SATELLITES["artemis_p2"]["color"],
        "size": 31,
        "zorder": 6,
    },
}

def _publication_style() -> None:
    style.apply_publication_style(text_size=TEXT_PT, tick_size=SMALL_TEXT_PT, overrides={
        'axes.titlesize': TEXT_PT,
        'axes.linewidth': 0.8,
    })


def _point_class(source: str, target: str) -> str:
    if source == "solar_orbiter":
        return "so_context"
    if source in L1_CRAFT and target in L1_CRAFT:
        return "l1_l1"
    if source in L1_CRAFT and target == "artemis_p1":
        return "l1_p1"
    if target == "artemis_p2":
        return "wake_context"
    raise ValueError(f"unsupported canonical pair: {source} -> {target}")


def load_results(json_path: Path | str = INPUT_JSON, csv_path: Path | str = INPUT_CSV) -> tuple[dict[str, Any], pd.DataFrame]:
    result, table = load_inference_product(json_path, csv_path)
    if result["meta"]["bootstrap_draws"] != BOOTSTRAP_DRAWS or result["meta"]["null_surrogates"] != NULL_SURROGATES:
        raise ValueError("Figure 5 requires the publication resampling settings")
    if table["selected_lag_at_search_edge"].any() or not table["expanded_search_lag_unchanged"].all():
        raise ValueError("Figure 5 requires interior, expanded-range-stable lags")
    table["point_class"] = [_point_class(a, b) for a, b in zip(table["source_craft"], table["target_craft"])]
    return result, table


def _fit_lookup(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        record["structure"]: record
        for record in result["coherence_length_records"]
    }


def _plot_points(
    ax: plt.Axes,
    subset: pd.DataFrame,
    classes: tuple[str, ...],
    *,
    value_column: str = "normalized_cross_correlation",
    low_column: str = "normalized_cross_correlation_bootstrap_ci95_low",
    high_column: str = "normalized_cross_correlation_bootstrap_ci95_high",
) -> None:
    for point_class, point_style in POINT_STYLES.items():
        if point_class not in classes:
            continue
        group = subset.loc[subset["point_class"].eq(point_class)]
        group = group.sort_values("separation_3d_re")
        x = group["separation_3d_re"].to_numpy(float)
        y = group[value_column].to_numpy(float)
        # Capless intervals remain distinguishable where near-Earth pairs
        # have almost identical separations.
        low = group[low_column].to_numpy(float)
        high = group[high_column].to_numpy(float)
        finite = np.isfinite(low) & np.isfinite(high)
        ax.vlines(
            x[finite],
            low[finite],
            high[finite],
            color=point_style["edge"],
            linewidth=0.55,
            alpha=0.42,
            zorder=point_style["zorder"] - 1,
        )
        ax.scatter(
            x,
            y,
            marker=point_style["marker"],
            s=point_style["size"],
            facecolors=point_style["face"],
            edgecolors=point_style["edge"],
            linewidths=0.85,
            zorder=point_style["zorder"],
        )


def _format_axes(ax: plt.Axes, *, column: int, panel_index: int,
                 p1_to_earth_re: float, y_limits: tuple[float, float]) -> None:
    ax.axhline(ONE_OVER_E, color="#777777", lw=0.75, ls="--", zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.70, zorder=0)
    ax.set_ylim(*y_limits)
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    if y_limits[0] < -0.2:
        ticks = [-0.25, *ticks]
    if y_limits[0] < -0.5:
        ticks = [-0.75, -0.5, *ticks]
    ax.set_yticks([value for value in ticks if y_limits[0] <= value <= y_limits[1]])
    ax.tick_params(axis="x", pad=2, labelsize=SMALL_TEXT_PT)
    ax.axvline(p1_to_earth_re, color=EARTH_CONTEXT_COLOR, lw=0.75, ls=":", zorder=1)
    ax.text(0.09, 0.975, f"({chr(ord('a') + panel_index)})", transform=ax.transAxes,
            fontsize=TEXT_PT, fontweight="bold", ha="left", va="top", zorder=10)
    ax.set_xscale("log")
    ax.set_xlim(*MAIN_X_LIMITS_RE)
    ax.set_xticks(MAIN_X_TICKS_RE)
    ax.set_xticklabels(["3", "10", "30", "100", "300"])
    if column:
        ax.tick_params(axis="y", which="both", labelleft=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _plot_curve(ax: plt.Axes, fit: Mapping[str, Any], *, wavelet: bool) -> None:
    point_key, low_key, high_key = (
        ("curve_point_r2", "curve_conditional_low_r2", "curve_conditional_high_r2")
        if wavelet else ("curve_point_ncc", "curve_ci95_low_ncc", "curve_ci95_high_ncc")
    )
    distance = np.asarray(fit["curve_distance_re"], dtype=float)
    ax.fill_between(distance, np.asarray(fit[low_key], dtype=float),
                    np.asarray(fit[high_key], dtype=float),
                    color=FIT_BAND_COLOR, alpha=0.22, linewidth=0, zorder=2)
    ax.plot(distance, np.asarray(fit[point_key], dtype=float),
            color=FIT_COLOR, lw=1.4, zorder=6.5,
            ls="--" if wavelet and not fit["supported_decay_length"] else "-")


def _length_note(fit: Mapping[str, Any], *, wavelet: bool, bz: bool = False) -> str:
    symbol = r"L_{\mathrm{cor},z}" if bz else (r"L_{\mathrm{coh}}" if wavelet else r"L_{\mathrm{cor}}")
    length = fit["point_coherence_length_re"]
    if length is None:
        if fit.get("length_endpoint") == "infinity":
            return f"${symbol} = \\infty$ R$_E$"
        raise ValueError("A displayed fit must have a measured length or explicit endpoint")
    return f"${symbol}$ = {length:.0f} R$_E$"


def _caption_text(wavelet_table: pd.DataFrame,
                  wavelet_fits: Mapping[str, Mapping[str, Any]]) -> str:
    incident = wavelet_table.loc[wavelet_table["is_fit_pair"]]
    distance_low, distance_high = incident["separation_3d_re"].agg(["min", "max"])
    wavelet_status = {
        "no_supported_decay_length": "diagnostic without a supported decay length",
        "beyond_sampled_range": "beyond sampled baselines",
        "below_sampled_range": "below sampled baselines",
        "within_sampled_range": "within sampled baselines", "unresolved": "unresolved",
    }
    qualifications = [f"{FAMILY_TITLES[family].lower()}, {wavelet_status[wavelet_fits[family]['status']]}"
                      for family in PLOTTED_STRUCTURES]
    return "\n\n".join([
        "Figure 5. Near-Earth persistence of magnetic waveform families. Columns show "
        "fine-scale, intermediate-scale and large-scale bands. Rows show peak lagged scalar "
        "Pearson correlation (a-c), activity-weighted scalar magnitude-squared wavelet "
        "coherence (d-f), and GSM Bz Pearson correlation at the scalar-derived pair lag (g-i).",
        f"Six L1-L1 links (circles) and four L1-P1 links (triangles), spanning "
        f"{distance_low:.1f}-{distance_high:.1f} R_E, enter separate exponential fits; "
        "five P2 wake links (crosses) provide context. Labels give fitted 1/e lengths "
        r"\(L_{\rm cor}\), \(L_{\rm coh}\), and \(L_{\rm cor,z}\). "
        "Solar Orbiter's displaced heliocentric sampling remains in the underlying scalar "
        "calculations but is omitted from this near-Earth display. The trend is omitted; "
        "no trend decay length is fitted.",
        "Bars and gray bands show 95% moving-block bootstrap intervals in (a-c,g-i), "
        "conditional on the prepared data, and conditional spatial-block 2.5th-97.5th "
        "percentile ranges in (d-f). Bz draws repeat scalar lag selection, without "
        "independently maximizing Bz correlation. Wavelet ranges hold maps, measured-shock "
        "registration and four-L1 activity weights fixed; they are not calibrated full 95% "
        "confidence intervals. Purple dotted lines are white-pair 95th-percentile benchmarks, "
        "not formal significance thresholds. Horizontal dashed lines mark 1/e; vertical "
        "dotted lines give P1-Earth geometrical context, not demonstrated transmission to GEO.",
        "Wavelet-length assessments are: " + "; ".join(qualifications) + ". Dashed black "
        "curves indicate diagnostic wavelet fits. Bz fits are descriptive; scalar null-test "
        "acceptance criteria are not transferred to them. Lengths exceeding the sampled "
        "separations are extrapolations.",
    ]) + "\n"


def plot(
    json_path: Path | str = INPUT_JSON,
    csv_path: Path | str = INPUT_CSV,
    output_path: Path | str = OUT,
    *,
    wavelet_json_path: Path | str = WAVELET_INPUT_JSON,
    bz_json_path: Path | str = BZ_INPUT_JSON,
) -> Path:
    result, table = load_results(json_path, csv_path)
    wavelet_result = load_wavelet_coherence(wavelet_json_path)
    wavelet_table = validate_wavelet_coherence(wavelet_result)
    bz_result = load_bz_persistence(bz_json_path)
    bz_table = validate_bz_persistence(bz_result)
    if bz_result["meta"]["method_parameters"]["bootstrap_draws"] != BOOTSTRAP_DRAWS:
        raise ValueError("Figure 5 requires the publication Bz resampling settings")
    method = wavelet_result["meta"]["method_parameters"]
    if (method["bootstrap_draws"] != WAVELET_BOOTSTRAP_DRAWS
            or method["white_draws"] != WHITE_DRAWS):
        raise ValueError("Figure 5 requires the publication wavelet resampling settings")
    pair_keys = ["structure", "source_craft", "target_craft"]
    for current in (wavelet_table, bz_table):
        current["point_class"] = [_point_class(a, b) for a, b in
                                  zip(current.source_craft, current.target_craft)]
        expected = table.loc[table.structure.isin(current.structure.unique()) &
                             table.point_class.ne("so_context")] if current is bz_table else table
        geometry = expected[pair_keys + ["separation_3d_re"]].merge(
            current[pair_keys + ["separation_3d_re"]], on=pair_keys,
            how="outer", validate="one_to_one", suffixes=("_scalar", "_other"))
        if not np.allclose(geometry.separation_3d_re_scalar,
                           geometry.separation_3d_re_other, rtol=1e-9, atol=1e-9):
            raise ValueError("Figure 5 products use inconsistent pair geometry")
    visible_tables = [current.loc[current.point_class.ne("so_context") &
                                   current.structure.isin(PLOTTED_STRUCTURES)].copy()
                      for current in (table, wavelet_table, bz_table)]
    for current in visible_tables:
        if current.groupby("structure").size().to_dict() != {family: 15 for family in PLOTTED_STRUCTURES}:
            raise ValueError("Every Figure 5 panel requires all fifteen near-Earth links")
    columns_by_row = (
        ("normalized_cross_correlation", "normalized_cross_correlation_bootstrap_ci95_low",
         "normalized_cross_correlation_bootstrap_ci95_high"),
        ("activity_weighted_coherence", "conditional_range025", "conditional_range975"),
        ("bz_correlation", "bz_correlation_bootstrap_ci95_low", "bz_correlation_bootstrap_ci95_high"),
    )
    # Use common limits while retaining all signed point estimates and interval endpoints.
    minimum = min(float(current[list(columns_by_row[index])].min().min())
                  for index, current in enumerate(visible_tables))
    y_limits = (max(-1., min(-.04, minimum - .04)), 1.05)
    p1_to_earth_re = float(np.linalg.norm(reference_positions()["artemis_p1"]))
    fits_by_row = [_fit_lookup(current) for current in (result, wavelet_result, bz_result)]
    _publication_style()
    figure = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH,
                                 FIGURE_HEIGHT_MM / MM_PER_INCH))
    columns = len(PLOTTED_STRUCTURES)
    outer = figure.add_gridspec(3, columns, left=0.10, right=0.995,
                                bottom=0.10, top=0.88, wspace=0.15, hspace=0.22)
    first = None
    for row in range(3):
        for column, structure in enumerate(PLOTTED_STRUCTURES):
            wavelet = row == 1
            current = visible_tables[row]
            subset = current.loc[current.structure.eq(structure)]
            ax = figure.add_subplot(outer[row, column], sharey=first)
            if first is None:
                first = ax
            _format_axes(ax, column=column, panel_index=row * columns + column,
                         p1_to_earth_re=p1_to_earth_re, y_limits=y_limits)
            if not row:
                ax.set_title(FAMILY_TITLES[structure], fontsize=TEXT_PT)
            if wavelet:
                ax.axhline(float(subset.white_quantile95_r2.iloc[0]),
                           color=WHITE_BENCHMARK_COLOR, lw=.8, ls=":", zorder=1)
            fit = fits_by_row[row][structure]
            _plot_curve(ax, fit, wavelet=wavelet)
            rmse_key, mae_key = ("fit_rmse_r2", "fit_mae_r2") if wavelet else ("fit_rmse_ncc", "fit_mae_ncc")
            note = _length_note(fit, wavelet=wavelet, bz=row == 2)
            note += f"\nRMSE {fit[rmse_key]:.2f}; MAE {fit[mae_key]:.2f}"
            ax.text(.035, .035, note, transform=ax.transAxes, fontsize=SMALL_TEXT_PT,
                    color=FIT_COLOR, ha="left", va="bottom", zorder=10,
                    bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=1.))
            value, low, high = columns_by_row[row]
            _plot_points(ax, subset, ("l1_l1", "l1_p1", "wake_context"),
                         value_column=value, low_column=low, high_column=high)
    handles = [Line2D([0], [0], marker=spec["marker"], linestyle="none",
                      markerfacecolor=spec["face"], markeredgecolor=spec["edge"],
                      markeredgewidth=.85, markersize=5.2, label=spec["label"])
               for spec in POINT_STYLES.values()]
    handles.extend([
        Line2D([0], [0], color=FIT_COLOR, lw=1.4, label="Exponential fit"),
        Patch(facecolor=FIT_BAND_COLOR, alpha=.22, edgecolor="none", label="Resampling ranges"),
        Line2D([0], [0], color=FIT_COLOR, lw=1.4, ls="--", label="Diagnostic fit"),
        Line2D([0], [0], color=WHITE_BENCHMARK_COLOR, lw=.8, ls=":", label="White-noise benchmark (95th)"),
    ])
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .995),
                  ncol=4, frameon=False, fontsize=SMALL_TEXT_PT, handletextpad=.35,
                  columnspacing=1.2)
    for row, label in enumerate(("Scalar correlation, r", "Scalar coherence, R$^2$",
                                 "GSM B$_z$ correlation, r$_z$")):
        slot = outer[row, 0].get_position(figure)
        figure.text(.025, slot.y0 + slot.height / 2, label, rotation=90,
                    va="center", ha="center", fontsize=TEXT_PT)
    figure.supxlabel("Spacecraft-pair separation [R$_E$]", y=.022, fontsize=TEXT_PT)
    pdf = style.save_pdf(figure, output_path)
    caption = RESULTS_DIR / "Fig5_caption.md"
    caption.parent.mkdir(parents=True, exist_ok=True)
    temporary = caption.with_suffix(".tmp.md")
    temporary.write_text(_caption_text(wavelet_table, fits_by_row[1]))
    temporary.replace(caption)
    return pdf


def main() -> None:
    print(plot())


if __name__ == "__main__":
    main()
