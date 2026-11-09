"""Shared rendering and current-product validation for GOES reconstructions."""
from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.goes_reconstruction import SCALE_GROUPS
from analysis.paired_goes_response import (
    EARLY_RESPONSE_STOP, EVENT_INDEX, EVENT_START, EVENT_STOP, RESPONSE_TRANSITION,
)
from config.waveform_bands import WAVEFORM_FAMILY_LABELS
from figures import style

SCALES = tuple(SCALE_GROUPS)
DISPLAY_FAMILIES = tuple(reversed(SCALES))
AXES = ("x", "y", "z")
SPACECRAFT = ("g18", "g19")
SCALE_COLORS = dict(discontinuity="#882255", substructure="#0072B2", sheath="#E69F00", trend="#009E73")
FAMILY_LABELS = WAVEFORM_FAMILY_LABELS
AXIS_COLORS = dict(x="#0072B2", y="#D55E00", z="#CC79A7")
ADDITIVITY_TOLERANCE_NT = 1e-8


def load_products(result_path: Path, sample_path: Path) -> tuple[dict, pd.DataFrame]:
    """Load interfaces only, preserving stored float values and the UTC clock."""
    result = json.loads(Path(result_path).read_text())
    samples = pd.read_csv(sample_path, float_precision="round_trip")
    samples["time_utc"] = pd.to_datetime(samples["time_utc"], utc=True)
    return result, samples


def _close(actual, expected, description: str) -> None:
    if not np.allclose(actual, expected, rtol=0, atol=ADDITIVITY_TOLERANCE_NT):
        raise ValueError(f"{description} does not agree with current event samples")


def validate_products(result, samples, *, source: str, input_kind: str) -> dict[str, pd.DataFrame]:
    """Fail on incompatible models, clocks or nonadditive saved attributions."""
    if result.get("schema_version") != "goes-event-reconstruction-v4":
        raise ValueError("reconstruction figure requires current v4 products")
    design = result["registered_design"]
    if design["source_spacecraft"] != source or design["input_kind"] != input_kind:
        raise ValueError("reconstruction source or input kind does not match the figure")
    expected_features = [f"{family}__b{axis}_f" for family in SCALES for axis in AXES] if input_kind == "components" else [f"{family}__bt_f" for family in SCALES]
    if design["feature_names"] != expected_features or design["feature_count"] != len(expected_features):
        raise ValueError("reconstruction feature names or count do not match the registered design")
    if input_kind == "components" and design.get("component_frame") != "GSM":
        raise ValueError("component reconstruction requires signed GSM provenance")
    window = design["frozen_model_window_utc"]
    if pd.Timestamp(window["start"]) != EVENT_INDEX[0] or pd.Timestamp(window["stop"]) != EVENT_INDEX[-1]:
        raise ValueError("reconstruction window does not match the registered event clock")
    shift = float(design["effective_model_shift_s"])
    if not np.isfinite(shift) or shift % 60 != 0:
        raise ValueError("reconstruction requires a finite whole-minute model shift")
    if set(samples["target_kind"]) != {"raw_bt"} or set(samples["spacecraft"]) != set(SPACECRAFT):
        raise ValueError("reconstruction samples must contain only raw GOES-18 and GOES-19 targets")
    family_columns = [f"{family}_model_contribution" for family in SCALES]
    feature_columns = [f"{feature}_shap_nT" for feature in expected_features]
    axis_columns = [f"axis_{axis}_model_contribution" for axis in AXES] if input_kind == "components" else []
    required = ["observed", "full_fit_prediction", "expected_value", *family_columns, *axis_columns, *feature_columns]
    missing = set(required) - set(samples)
    if missing:
        raise ValueError(f"reconstruction samples are missing columns: {sorted(missing)}")
    frames = {}
    for craft in SPACECRAFT:
        frame = samples.loc[samples["spacecraft"].eq(craft)].set_index("time_utc")
        if not frame.index.equals(EVENT_INDEX):
            raise ValueError(f"{craft} reconstruction samples must cover the ordered model window")
        if not np.isfinite(frame[required].to_numpy(float)).all():
            raise ValueError(f"{craft} reconstruction contains nonfinite samples")
        net = frame["full_fit_prediction"] - frame["expected_value"]
        for group, columns in (("family", family_columns), ("feature", feature_columns)):
            _close(frame[columns].sum(axis=1), net, f"{craft} {group} contributions")
        target = result["spacecraft"][craft]["targets"]["raw_bt"]
        fit = target["full_fit"]
        if fit["row_count"] != len(EVENT_INDEX) or fit["feature_count"] != len(expected_features):
            raise ValueError("fitted model row or feature count does not match event samples")
        if not np.isfinite(float(fit["r2"])):
            raise ValueError("fitted R^2 is not finite")
        observed = frame["observed"].to_numpy(float)
        residual = observed - frame["full_fit_prediction"].to_numpy(float)
        total = float(np.sum((observed - observed.mean()) ** 2))
        if total <= 0:
            raise ValueError("observed reconstruction target has no variance")
        _close(float(fit["r2"]), 1 - float(np.sum(residual ** 2)) / total, f"{craft} fit R^2")
        grouped = pd.DataFrame(target["grouped_contributions"]["summary"]).set_index("scale_band")
        if set(grouped.index) != set(SCALES) or grouped.index.has_duplicates:
            raise ValueError("family SHAP summary must contain each family exactly once")
        means = np.array([np.abs(frame[column]).mean() for column in family_columns])
        _close(grouped.loc[list(SCALES), "mean_absolute_model_contribution"], means, f"{craft} family summary")
        shares = 100 * means / means.sum() if means.sum() else np.zeros_like(means)
        _close(grouped.loc[list(SCALES), "model_reliance_share_percent"], shares, f"{craft} family shares")
        features = pd.DataFrame(target["feature_contributions"]["summary"]).set_index("feature")
        if set(features.index) != set(expected_features) or features.index.has_duplicates:
            raise ValueError("individual SHAP summary does not match the registered features")
        _close(features.loc[expected_features, "mean_absolute_model_contribution"], np.abs(frame[feature_columns]).mean(axis=0), f"{craft} feature summary")
        for family in SCALES:
            columns = ([f"{family}__b{axis}_f_shap_nT" for axis in AXES] if input_kind == "components" else [f"{family}__bt_f_shap_nT"])
            _close(frame[columns].sum(axis=1), frame[f"{family}_model_contribution"], f"{craft} {family} grouping")
        if input_kind == "components":
            _close(frame[axis_columns].sum(axis=1), net, f"{craft} axis contributions")
            summary = pd.DataFrame(target["axis_contributions"]["summary"]).set_index("component")
            if set(summary.index) != set(AXES) or summary.index.has_duplicates:
                raise ValueError("axis SHAP summary must contain each component exactly once")
            axis_means = []
            for axis in AXES:
                _close(frame[[f"{family}__b{axis}_f_shap_nT" for family in SCALES]].sum(axis=1), frame[f"axis_{axis}_model_contribution"], f"{craft} {axis} grouping")
                axis_means.append(float(np.abs(frame[f"axis_{axis}_model_contribution"]).mean()))
            _close(summary.loc[list(AXES), "mean_absolute_model_contribution"], axis_means, f"{craft} axis summary")
        frames[craft] = frame
    return frames


def apply_style() -> None:
    style.apply_publication_style(text_size=7.5, tick_size=6.5, overrides={
        "axes.titlesize": 8.0, "axes.linewidth": .55,
        "legend.fontsize": 6.2, "axes.grid": False, "lines.linestyle": "-",
    })


def _panel_letter(axis, letter) -> None:
    axis.text(.015, .97, f"({letter})", transform=axis.transAxes, ha="left", va="top",
              fontsize=8, weight="bold", zorder=10,
              bbox=dict(facecolor="white", edgecolor="none", alpha=.75, pad=.5))


def _annotate_event_fit_r2(axis, value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("fitted R^2 is not finite")
    return axis.text(.975, .965, rf"$R^2={value:.3f}$", transform=axis.transAxes,
                     ha="right", va="top", fontsize=7.2,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=.86, pad=.8))


def format_time_axis(axis, index, *, bottom=False) -> None:
    for start, stop, color in ((EVENT_START, EARLY_RESPONSE_STOP, "#B2182B"),
                               (EARLY_RESPONSE_STOP, RESPONSE_TRANSITION, "#E69F00"),
                               (RESPONSE_TRANSITION, EVENT_STOP, "#332288")):
        axis.axvspan(start, stop, color=color, alpha=.045, lw=0, zorder=-6)
    for boundary in (EARLY_RESPONSE_STOP, RESPONSE_TRANSITION):
        axis.axvline(boundary, color=".45", ls=":", lw=.55)
    axis.set_xlim(index[0], index[-1])
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=timezone.utc))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    axis.tick_params(labelbottom=bottom, length=2.5)
    if bottom:
        axis.set_xlabel("Time (UTC; 19-20 January 2026)", labelpad=3)
    axis.grid(axis="y", color=".85", lw=.35, alpha=.6, zorder=-4)


def draw_fits(axes, frames, result) -> None:
    values = np.concatenate([frame[["observed", "full_fit_prediction"]].to_numpy().ravel() for frame in frames.values()])
    low, high = values.min(), values.max()
    span = max(high - low, 1.)
    for column, craft in enumerate(SPACECRAFT):
        axis, frame = axes[column], frames[craft]
        axis.plot(frame.index, frame.observed, color="#252525", lw=.7, ls="-", label="Observed")
        axis.plot(frame.index, frame.full_fit_prediction, color="#D55E00", lw=.85, ls="-", label="Fitted")
        axis.set_ylim(low - .06 * span, high + .11 * span)
        axis.set_title(f"GOES-{craft[1:]}", weight="bold", pad=5)
        _annotate_event_fit_r2(axis, result["spacecraft"][craft]["targets"]["raw_bt"]["full_fit"]["r2"])
        axis.legend(loc="upper right", bbox_to_anchor=(1, .82), ncol=2, handlelength=1.5,
                    frameon=True, facecolor="white", edgecolor="none", framealpha=.86)
        if column == 0:
            axis.set_ylabel(r"$|B|$ (nT)")
        format_time_axis(axis, frame.index)


def draw_signed_stacks(axis, frame, columns, colors, labels) -> float:
    positive, negative = np.zeros(len(frame)), np.zeros(len(frame))
    for column, color, label in zip(columns, colors, labels, strict=True):
        values = frame[column].to_numpy(float)
        above, below = np.maximum(values, 0), np.minimum(values, 0)
        axis.fill_between(frame.index, positive, positive + above, color=color, alpha=.83, lw=0, label=label)
        axis.fill_between(frame.index, negative, negative + below, color=color, alpha=.83, lw=0)
        positive += above
        negative += below
    net = frame[columns].sum(axis=1)
    _close(net, frame.full_fit_prediction - frame.expected_value, "net stacked contribution")
    axis.plot(frame.index, net, color="#202020", lw=.8, ls="-", label="Net", zorder=5)
    axis.axhline(0, color=".35", lw=.5)
    return max(float(np.abs(negative).max()), float(np.abs(positive).max()))


def shared_signed_limits(axes, limits) -> None:
    maximum = max(max(limits) * 1.18, 1)
    for axis in axes:
        axis.set_ylim(-maximum, maximum)


def stack_legend(axis, *, count) -> None:
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=count,
                frameon=False, handlelength=1.3, handletextpad=.3,
                columnspacing=.7, borderaxespad=0)



def render_scalar_figure(*, result, samples, source: str, output_path) -> str:
    """Identical six-panel P1/SOLAR-1 four-scalar supporting figure layout."""
    frames = validate_products(result, samples, source=source, input_kind="scalar")
    apply_style()
    fig, axes = plt.subplots(3, 2, figsize=(style.FIGURE_WIDTH_MM / style.MM_PER_INCH, 170 / style.MM_PER_INCH),
                             gridspec_kw={"height_ratios": (1.05, 1, .75)})
    draw_fits(axes[0], frames, result)
    columns = [f"{family}_model_contribution" for family in SCALES]
    colors = [SCALE_COLORS[family] for family in SCALES]
    means = {craft: np.abs(frame[columns]).mean(axis=0).to_numpy() for craft, frame in frames.items()}
    barmax = max(max(float(values.max()) for values in means.values()), np.finfo(float).eps)
    limits = []
    for col, craft in enumerate(SPACECRAFT):
        frame = frames[craft]
        limits.append(draw_signed_stacks(axes[1, col], frame, columns, colors, [FAMILY_LABELS[key] for key in SCALES]))
        format_time_axis(axes[1, col], frame.index, bottom=True)
        axis = axes[2, col]
        positions = np.arange(4)
        bars = axis.barh(positions, means[craft], color=colors, height=.6)
        axis.set_yticks(positions, [FAMILY_LABELS[key] for key in SCALES] if col == 0 else [])
        axis.set_ylim(-.6, 4.1)
        axis.set_xlim(0, barmax * 1.25)
        axis.set_xlabel("Mean |family SHAP| (nT)")
        axis.grid(axis="x", color=".85", lw=.4, alpha=.7, zorder=-1)
        denominator = means[craft].sum()
        shares = 100 * means[craft] / denominator if denominator else np.zeros(4)
        for bar, value, share in zip(bars, means[craft], shares, strict=True):
            axis.text(value + barmax * .025, bar.get_y() + bar.get_height() / 2,
                      f"{share:.1f}%", va="center", fontsize=6.8)
    axes[1, 0].set_ylabel("Family contribution\n(nT)")
    shared_signed_limits(axes[1], limits)
    stack_legend(axes[1, 0], count=5)
    for index, axis in enumerate(axes.flat):
        _panel_letter(axis, chr(97 + index))
    fig.subplots_adjust(left=.145, right=.98, top=.955, bottom=.085, hspace=.68, wspace=.24)
    return str(style.save_pdf(fig, output_path, strict_pdf=True))
