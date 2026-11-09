"""Shue (1998) surface and paired Figure 1 event contexts.

The published model coefficients and proton mass are fixed parameters.
Density, speed, GSM Bz, pressure, r0 and alpha are caller inputs or outputs.
The plotted surface axis is +X: solar-wind aberration is not applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from config.constants import PROTON_MASS_KG


def _finite(name, value):
    values = np.asarray(value, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values")
    return values


def _scalar_or_array(values):
    return float(values) if np.ndim(values) == 0 else values


def dynamic_pressure_npa(n_p_cm3, v_sw_kms, alpha_to_proton_ratio):
    """Ram pressure rho*V**2 in nPa, assuming co-moving protons and alphas.

    The helium mass contribution is 4*n_alpha/n_proton. Electron mass is
    neglected. The composition ratio must be supplied explicitly.
    """
    density = _finite("proton density", n_p_cm3)
    speed = _finite("solar-wind speed", v_sw_kms)
    ratio = _finite("alpha/proton number ratio", alpha_to_proton_ratio)
    if np.any(density <= 0) or np.any(speed <= 0) or np.any(ratio < 0):
        raise ValueError("density and speed must be positive; composition ratio nonnegative")
    pressure = PROTON_MASS_KG * density * 1e6 * (speed * 1e3) ** 2
    pressure = pressure * (1 + 4 * ratio) * 1e9
    return _scalar_or_array(_finite("derived pressure", pressure))


def shue_parameters(pressure_npa, bz_gsm_nt):
    """Published 1998 parameters, with pressure in nPa and GSM IMF Bz in nT."""
    pressure = _finite("dynamic pressure", pressure_npa)
    bz = _finite("GSM IMF Bz", bz_gsm_nt)
    if np.any(pressure <= 0):
        raise ValueError("dynamic pressure must be positive")
    r0 = (10.22 + 1.29 * np.tanh(0.184 * (bz + 8.14))) * pressure ** (-1 / 6.6)
    alpha = (0.58 - 0.007 * bz) * (1 + 0.024 * np.log(pressure))
    if np.any(alpha <= 0):
        raise ValueError("these inputs produce nonpositive flaring; no parameter clipping is applied")
    return _scalar_or_array(_finite("r0", r0)), _scalar_or_array(_finite("alpha", alpha))


def radius_re(theta_rad, r0_re, alpha):
    """Radial surface r(theta), with theta measured from +X.

    Signed theta permits a meridional plot. The tail pole |theta|=pi is
    excluded. cos(theta/2)**(-2*alpha) is algebraically identical to the
    published factor and avoids subtraction cancellation near that pole.
    """
    theta = _finite("solar-zenith angle", theta_rad)
    r0 = _finite("r0", r0_re)
    flaring = _finite("alpha", alpha)
    if np.any(np.abs(theta) >= np.pi) or np.any(r0 <= 0) or np.any(flaring <= 0):
        raise ValueError("require |theta| < pi and positive r0 and alpha")
    try:
        with np.errstate(over="raise", invalid="raise"):
            radius = np.exp(np.log(r0) - 2 * flaring * np.log(np.cos(theta / 2)))
    except FloatingPointError as exc:
        raise ValueError("surface radius is not representable for these inputs") from exc
    return _scalar_or_array(_finite("surface radius", radius))


def meridional_outline(r0_re, alpha, x_min_re, samples=2001):
    """Return (theta, X, transverse radius) covering an explicit tail X limit.

    For 0 < alpha < 2 the curve is single-valued in transverse radius at X.
    This is a plotting guard, not a published empirical validity range.
    The same outline is the XY/XZ silhouette when the symmetry axis is +X.
    """
    r0 = float(_finite("r0", r0_re))
    flaring = float(_finite("alpha", alpha))
    x_min = float(_finite("tail plotting limit", x_min_re))
    if r0 <= 0 or not 0 < flaring < 2 or x_min >= 0:
        raise ValueError("outline requires positive r0, 0 < alpha < 2, and negative tail X limit")
    if not isinstance(samples, int) or samples < 5 or samples % 2 == 0:
        raise ValueError("use an odd integer of at least five curve samples")
    endpoint = np.pi - 1e-10
    def residual(theta):
        return radius_re(theta, r0, flaring) * np.cos(theta) - x_min
    if residual(endpoint) >= 0:
        raise ValueError("requested tail limit cannot be reached with representable angles")
    theta_end = brentq(residual, np.pi / 2, endpoint, xtol=1e-13)
    theta = np.linspace(-theta_end, theta_end, samples)
    radius = radius_re(theta, r0, flaring)
    return np.column_stack((theta, radius * np.cos(theta), radius * np.sin(theta)))


def _utc(value, name):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must be finite with an explicit timezone")
    return stamp.tz_convert("UTC")


def select_figure1_states(overview, native_gsm, boundaries, response, reference_epoch,
                         alpha_to_proton_ratio=0.0):
    """Recalculate two paired contexts from current source and response clocks.

    The shock/onset difference associates upstream measurements with GEO response
    context; it is not an independently fitted physical travel time. The solid
    state uses the first complete post-shock source minute. The dashed state uses
    the source bin containing the reference epoch minus that clock difference.
    """
    if any(boundaries.get("_meta", {}).get(key) != "complete" for key in ("shock_status", "sheath_status")):
        raise ValueError("complete current shock and sheath prerequisites required")
    try:
        shock = _utc(boundaries["swfo"]["shock"], "Solar-1 shock")
        end = _utc(boundaries["swfo"]["sheath_end"], "Solar-1 sheath end")
        onset = _utc(response["event"]["start_utc"], "GOES response onset")
    except KeyError as exc:
        raise ValueError(f"missing current event-clock prerequisite: {exc.args[0]}") from exc
    epoch = _utc(reference_epoch, "reference geometry epoch")
    if not shock < end or not shock < onset or not epoch < onset:
        raise ValueError("ordered shock, sheath-end, response-onset and reference clocks required")
    for name, frame, columns in (("overview", overview, {"n", "v"}),
                                 ("native GSM", native_gsm, {"bz_gsm_nt"})):
        if (frame.empty or not columns.issubset(frame.columns)
                or not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None
                or frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing
                or not (frame.index == frame.index.floor("min")).all()):
            raise ValueError(f"{name} requires unique ordered timezone-aware minute bins and {sorted(columns)}")
    delay = onset - shock
    cases = (("response_onset", onset, shock.ceil("min"), "-"),
             ("snapshot", epoch, (epoch-delay).floor("min"), "--"))
    records = []
    for case, target, source, line_style in cases:
        source_end = source + pd.Timedelta(minutes=1)
        if case == "response_onset" and source_end > end:
            raise ValueError("no complete post-shock source minute before the sheath end")
        if source not in overview.index or source not in native_gsm.index:
            raise ValueError(f"exact required Solar-1 minute missing: {source}; no later fallback")
        n, v, bz = float(overview.loc[source, "n"]), float(overview.loc[source, "v"]), float(native_gsm.loc[source, "bz_gsm_nt"])
        pressure = dynamic_pressure_npa(n, v, alpha_to_proton_ratio)
        r0, alpha = shue_parameters(pressure, bz)
        records.append({"case": case, "line_style": line_style,
                        "target_geo_time_utc": target.isoformat(),
                        "source_window_start_utc": source.isoformat(), "source_window_end_utc": source_end.isoformat(),
                        "associated_geo_window_start_utc": (source+delay).isoformat(),
                        "associated_geo_window_end_utc": (source_end+delay).isoformat(),
                        "n_p_cm3": n, "v_sw_kms": v, "bz_gsm_nt": bz,
                        "alpha_to_proton_ratio": float(alpha_to_proton_ratio),
                        "pressure_npa": pressure, "r0_re": r0, "flaring_alpha": alpha})
    metadata = {"schema_version": "fig1-magnetopause-v1", "model": "Shue 1998",
                "solar1_shock_utc": shock.isoformat(), "goes_response_onset_utc": onset.isoformat(),
                "reference_geometry_epoch_utc": epoch.isoformat(), "shock_to_response_seconds": delay.value/1e9,
                "timing_policy": "measured Solar-1 shock / observed GOES response-onset clock association; not a fitted travel time",
                "selection": "first complete post-shock source minute; snapshot-containing source minute after clock association",
                "input_frame": "native provider GSM Bz", "symmetry_axis": "un-aberrated +GSE X",
                "composition": "explicit co-moving alpha/proton ratio; proton-only for the publication figure",
                "alpha_to_proton_ratio": float(alpha_to_proton_ratio),
                "interpretation": "empirical boundary context from minute means; no instantaneous shock equilibrium or observed crossing asserted"}
    return records, metadata
