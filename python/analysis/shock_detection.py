"""Minute-scale logistic candidates and native-cadence magnetic ramp fits.

Adapted from Pelkum Donahue & Inceoglu (2026), ApJ 1004, 146.
Candidate detection uses only the supplied magnetic record. Event association
and nearby-spacecraft guidance belong to the pipeline builder.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import least_squares
from scipy.special import expit

@dataclass(frozen=True)
class Settings:
    coarse_minutes: int = 5
    minimum_r2: float = .75
    coarse_separation_s: float = 90.
    native_window_s: float = 60.
    native_search_s: float = 90.
    native_search_step_s: float = 5.


def logistic(t, before, after, midpoint, log_tau):
    return before + (after - before) * expit((np.asarray(t) - midpoint) / np.exp(log_tau))


def logistic_jacobian(t, before, after, midpoint, log_tau):
    z = (np.asarray(t) - midpoint) / np.exp(log_tau)
    g = expit(z)
    slope = (after - before) * g * (1 - g)
    return np.column_stack((1 - g, g, -slope / np.exp(log_tau), -slope * z))


def bin_logistic(left, right, midpoint, tau):
    """Exact mean of a unit logistic curve over each time bin."""
    left, right = np.asarray(left), np.asarray(right)
    return tau * (np.logaddexp(0, (right - midpoint) / tau)
                  - np.logaddexp(0, (left - midpoint) / tau)) / (right - left)


def separated_indices(times, strengths, separation_s, directions=None):
    """Suppress repeated fits of a nearby step; retain distinct transitions."""
    selected = []
    for i in np.argsort(-np.asarray(strengths), kind="stable"):
        if not np.isfinite(strengths[i]):
            continue
        if all((directions is not None and directions[i] != directions[j])
               or abs(times[i] - times[j]) >= separation_s for j in selected):
            selected.append(int(i))
    return sorted(selected, key=lambda i: times[i])


def coarse_candidates(minute: pd.Series, settings=Settings()):
    """No event clocks, other spacecraft, or plasma enter this calculation."""
    minute = minute.sort_index().asfreq("1min")
    n = settings.coarse_minutes
    columns = ["coarse_time", "coarse_window_start", "coarse_r2", "coarse_before",
               "coarse_after", "coarse_tau_s", "coarse_strength", "direction"]
    if len(minute) < n:
        return pd.DataFrame(columns=columns)
    windows = sliding_window_view(minute.to_numpy(dtype=float), n)
    valid = np.isfinite(windows).all(axis=1)
    y = np.nan_to_num(windows)
    mean = y.mean(axis=1)
    yc = y - mean[:, None]
    total = (yc * yc).sum(axis=1)
    best = np.full(len(y), -np.inf)
    pars = np.full((len(y), 5), np.nan)
    edges = np.arange(n + 1) * 60.
    # One full minute of support on either side; grid only locates candidates.
    for mid in np.arange(60., n * 60. - 59., 15.):
        for tau in (.5, 2., 8., 30., 60.):
            g = bin_logistic(edges[:-1], edges[1:], mid, tau)
            gc = g - g.mean()
            dot = yc @ gc
            amplitude = dot / (gc @ gc)
            before = mean - amplitude * g.mean()
            after = before + amplitude
            r2 = np.divide(dot ** 2, (gc @ gc) * total,
                           out=np.zeros(len(y)), where=total > 1.e-20)
            ok = valid & (before > 0) & (after > 0) & (r2 >= settings.minimum_r2)
            # Best-fitting curve per window, not the largest extrapolated jump.
            take = ok & (r2 > best)
            best[take] = r2[take]
            pars[take] = np.column_stack((before, after, np.full(len(y), mid),
                                          np.full(len(y), tau), r2))[take]
    hits = np.flatnonzero(np.isfinite(best))
    if not len(hits):
        return pd.DataFrame(columns=columns)
    p = pars[hits]
    times = hits * 60. + p[:, 2]
    strength = np.abs(p[:, 1] - p[:, 0]) * p[:, 4]
    keep = separated_indices(times, strength, settings.coarse_separation_s,
                             np.sign(p[:, 1] - p[:, 0]))
    return pd.DataFrame([{
        "coarse_time": minute.index[0] + pd.Timedelta(seconds=times[j]),
        "coarse_window_start": minute.index[hits[j]],
        "coarse_r2": p[j, 4], "coarse_before": p[j, 0], "coarse_after": p[j, 1],
        "coarse_tau_s": p[j, 3], "coarse_strength": strength[j],
        "direction": "rise" if p[j, 1] > p[j, 0] else "fall",
    } for j in keep], columns=columns)


def fit_native_window(series: pd.Series, center, cadence, settings, initial):
    """Fit observed samples only; explicitly flag unresolved data gaps."""
    half = settings.native_window_s / 2
    s = series.loc[center - pd.Timedelta(seconds=half):
                   center + pd.Timedelta(seconds=half)].dropna()
    if len(s) < max(8, int(.75 * settings.native_window_s / cadence)):
        return None
    t = (s.index - center).total_seconds().to_numpy()
    if t[0] > -half + 2 * cadence or t[-1] < half - 2 * cadence:
        return None
    y = s.to_numpy()
    total = np.sum((y - y.mean()) ** 2)
    if total <= 1.e-20:
        return None
    low_tau, high_tau = cadence / 100., settings.native_window_s / 6.
    # Several width initializations avoid a flat-gradient solution between
    # sparsely spaced samples. They use identical rules for every spacecraft.
    attempts = []
    for initial_tau in sorted(set((initial[2], cadence / 4, min(3., high_tau)))):
        attempt = least_squares(
            lambda p: logistic(t, *p) - y,
            [initial[0], initial[1], 0., np.log(np.clip(initial_tau, low_tau, high_tau))],
            bounds=([0, 0, -half / 3, np.log(low_tau)],
                    [np.inf, np.inf, half / 3, np.log(high_tau)]),
            jac=lambda p: logistic_jacobian(t, *p), x_scale="jac", max_nfev=500,
        )
        if attempt.success:
            attempts.append(attempt)
    if not attempts:
        return None
    fit = min(attempts, key=lambda f: np.sum(f.fun ** 2))
    before, after, mid, ltau = fit.x
    tau = np.exp(ltau)
    r2 = 1 - np.sum(fit.fun ** 2) / total
    if not fit.success or r2 < 0:
        return None
    if np.sign(after - before) != np.sign(initial[1] - initial[0]):
        return None
    width = 2 * np.log(3) * tau
    if (t < mid - width / 2).sum() < 3 or (t > mid + width / 2).sum() < 3:
        return None
    pos = np.searchsorted(t, mid)
    if pos == 0 or pos == len(t):
        return None
    gap = float(t[pos] - t[pos - 1])
    gap_limited = gap > 1.5 * cadence
    return {
        "native_time": center + pd.Timedelta(seconds=mid),
        "fit_window_start": center - pd.Timedelta(seconds=half),
        "fit_window_end": center + pd.Timedelta(seconds=half),
        "native_r2": r2, "before_nT": before, "after_nT": after,
        "passes_native_r2": bool(r2 >= settings.minimum_r2),
        "jump_nT": after - before, "tau_s": tau, "ramp_25_75_s": width,
        # This is a descriptive width/cadence/gap floor, not a confidence interval.
        "timing_halfwidth_s": max(width / 2, cadence, gap / 2 if gap_limited else 0),
        "sample_bracket_s": gap, "gap_limited": gap_limited,
        "sampling_limited": width < cadence,
        "width_at_limit": bool(np.isclose(tau, high_tau, rtol=.001)),
        "native_samples": len(s), "cadence_s": cadence,
        "native_strength": abs(after - before) * r2,
        "direction": "rise" if after > before else "fall",
    }


def refine_candidate(series, coarse, cadence, settings=Settings()):
    """Scan around a coarse candidate and retain separate native steps."""
    half = settings.native_window_s / 2
    reach = settings.native_search_s
    section = series.loc[coarse - pd.Timedelta(seconds=reach + half + 1):
                         coarse + pd.Timedelta(seconds=reach + half + 1)].dropna()
    t = (section.index - coarse).total_seconds().to_numpy()
    y = section.to_numpy()
    seeds = []
    for center in np.arange(-reach, reach + .01, settings.native_search_step_s):
        i, j = np.searchsorted(t, [center - half, center + half], side="left")
        if j - i < max(8, int(.75 * settings.native_window_s / cadence)):
            continue
        x, v = t[i:j] - center, y[i:j]
        if x[0] > -half + 2 * cadence or x[-1] < half - 2 * cadence:
            continue
        yc = v - v.mean()
        total = yc @ yc
        if total < 1.e-20:
            continue
        best = None
        for tau in sorted(set((cadence / 4, cadence, 3., settings.native_window_s / 6))):
            g = expit(x / tau)
            gc = g - g.mean()
            amp = (yc @ gc) / (gc @ gc)
            before = v.mean() - amp * g.mean()
            after = before + amp
            r2 = (yc @ gc) ** 2 / (gc @ gc) / total
            strength = abs(amp) * r2
            # This is an initialization screen; final fits still require 0.75.
            if before > 0 and after > 0 and r2 >= min(.5, settings.minimum_r2):
                if best is None or strength > best["strength"]:
                    best = {"center": center, "before": before, "after": after,
                            "tau": tau, "strength": strength}
        if best is not None:
            seeds.append(best)
    if not seeds:
        return []
    indices = np.argsort([-s["strength"] for s in seeds], kind="stable")
    output = []
    for i in indices:
        p = seeds[i]
        center = coarse + pd.Timedelta(seconds=p["center"])
        direction = "rise" if p["after"] > p["before"] else "fall"
        # A failed/poor fit must not suppress an untried nearby window.
        if any(r["passes_native_r2"] and r["direction"] == direction
               and abs((r["native_time"] - center).total_seconds()) < half for r in output):
            continue
        result = fit_native_window(
            series, center, cadence, settings,
            (p["before"], p["after"], p["tau"]),
        )
        if result is not None:
            output.append(result)
    if not output:
        return []
    times = [(r["native_time"] - coarse).total_seconds() for r in output]
    # Prefer a qualifying fit when several windows describe the same step.
    strengths = [r["native_strength"] for r in output]
    bonus = max(strengths) + 1
    priorities = [s + bonus * r["passes_native_r2"] for s, r in zip(strengths, output)]
    keep = separated_indices(times, priorities, half, [r["direction"] for r in output])
    return [output[i] for i in keep]
