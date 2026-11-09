"""Logistic timing of a data-selected sheath/ejecta rotation.

Fit observed one-minute angles, not magnetic magnitude and not bin-averaged
angles. No adopted boundary or fixed angular crossing enters the search.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from analysis.shock_detection import logistic, logistic_jacobian, separated_indices


@dataclass(frozen=True)
class Settings:
    window_minutes: int = 10
    minimum_coverage: float = 0.8
    minimum_side_samples: int = 3
    minimum_r2: float = 0.75  # Screen poorly described angular transitions.
    separation_minutes: float = 5.0  # Merge repeated fits of the same decline.


def fit_window(angles, settings=Settings()):
    """Fit one window; return None when its data do not support a falling step.

    The returned timing half-width is descriptive (transition width, cadence,
    and data-gap floors). It is not a statistical confidence interval.
    """
    if len(angles) != settings.window_minutes + 1:
        raise ValueError("Use a complete minute grid spanning the fitting window")
    if not np.all(np.diff(angles.index.asi8) == pd.Timedelta('1min').value):
        raise ValueError("Missing timestamps must be explicit NaN values")
    finite = angles.where(np.isfinite(angles)).dropna()
    if len(finite) < max(2 * settings.minimum_side_samples + 1,
                         int(np.ceil(settings.minimum_coverage * len(angles)))):
        return None
    if ((finite < 0) | (finite > 180)).any():
        raise ValueError("Angles must be in degrees between zero and 180")
    center = angles.index[0] + pd.Timedelta(minutes=settings.window_minutes / 2)
    t = (finite.index - center).total_seconds().to_numpy(float) / 60
    y = finite.to_numpy(float)
    total = float(np.sum((y - y.mean()) ** 2))
    if total < 1.e-12:
        return None
    nside = settings.minimum_side_samples
    before, after = np.median(y[:nside]), np.median(y[-nside:])
    if before <= after:
        return None
    # Side support is checked explicitly after fitting; midpoint bounds keep
    # the optimizer away from extrapolated transitions outside the window.
    half = settings.window_minutes / 2
    min_tau, max_tau = .01, settings.window_minutes / 6
    bounds = ([0, 0, -half + 1, np.log(min_tau)],
              [180, 180, half - 1, np.log(max_tau)])
    fits = []
    for tau0 in (.1, .5, 1.):
        fit = least_squares(
            lambda p: logistic(t, *p) - y,
            [before, after, 0., np.log(min(tau0, max_tau))],
            jac=lambda p: logistic_jacobian(t, *p),
            bounds=bounds, x_scale='jac', max_nfev=500,
        )
        if fit.success:
            fits.append(fit)
    if not fits:
        return None
    fit = min(fits, key=lambda f: np.sum(f.fun ** 2))
    before, after, midpoint, log_tau = fit.x
    tau = float(np.exp(log_tau))
    r2 = float(1 - np.sum(fit.fun ** 2) / total)
    width = float(2 * np.log(3) * tau)
    if before <= after or r2 < settings.minimum_r2:
        return None
    if ((t < midpoint - width / 2).sum() < nside
            or (t > midpoint + width / 2).sum() < nside):
        return None
    mid = center + pd.Timedelta(minutes=midpoint)
    j = np.searchsorted(finite.index.asi8, mid.value)
    if j == 0 or j == len(finite):
        return None
    sample_gap_s = (finite.index[j] - finite.index[j-1]).total_seconds()
    return {
        'boundary_utc': mid, 'fit_window_start_utc': angles.index[0],
        'fit_window_stop_utc': angles.index[-1], 'fit_center_utc': center,
        'before_deg': float(before), 'after_deg': float(after),
        'drop_deg': float(before - after), 'r2': r2, 'tau_minutes': tau,
        'transition_25_75_seconds': width * 60,
        'transition_25_utc': mid - pd.Timedelta(minutes=width / 2),
        'transition_75_utc': mid + pd.Timedelta(minutes=width / 2),
        'timing_halfwidth_seconds': max(60., width * 30, sample_gap_s / 2),
        'last_sample_before_utc': finite.index[j-1],
        'first_sample_after_utc': finite.index[j],
        'sample_bracket_seconds': sample_gap_s,
        'sampling_limited': bool(width < 1),
        'gap_limited': bool(sample_gap_s > 60),
        'width_at_upper_limit': bool(np.isclose(tau, max_tau, rtol=.001)),
        'width_at_lower_limit': bool(np.isclose(tau, min_tau, rtol=.001)),
        'observed_samples': len(finite), 'window_samples': len(angles),
        'strength': float((before - after) * r2),
    }


def detect(angles, shock_time, reference_start, reference_stop, settings=Settings()):
    """Scan from the measured shock to the data-selected reference's end.

    Retain only fitted midpoints before the reference starts. Select the
    largest angular drop times R-squared, using no historical boundary or
    30/45-degree crossing. Keep every passing fit and distinct alternative.
    """
    if not isinstance(angles.index, pd.DatetimeIndex) or angles.index.tz is None:
        raise ValueError('Use a UTC-aware one-minute angle record')
    if angles.index.has_duplicates or not angles.index.is_monotonic_increasing:
        raise ValueError('Angle timestamps must be sorted and unique')
    if not np.all(np.diff(angles.index.asi8) == pd.Timedelta('1min').value):
        raise ValueError('Missing timestamps must be explicit NaN values')
    shock_time, reference_start, reference_stop = map(
        pd.Timestamp, (shock_time, reference_start, reference_stop))
    if not shock_time < reference_start < reference_stop:
        raise ValueError('Require shock < reference start < reference stop')
    part = angles.loc[(angles.index > shock_time) & (angles.index <= reference_stop)]
    fits = []
    n = settings.window_minutes + 1
    for start in range(len(part) - n + 1):
        fit = fit_window(part.iloc[start:start+n], settings)
        if fit is not None and shock_time < fit['boundary_utc'] <= reference_start:
            fits.append(fit)
    if not fits:
        return {'status': 'no_accepted_fit'}, []
    times = np.array([(f['boundary_utc'] - shock_time).total_seconds() for f in fits])
    strengths = np.array([f['strength'] for f in fits])
    distinct = separated_indices(times, strengths, settings.separation_minutes * 60)
    chosen = max(distinct, key=lambda i: fits[i]['strength'])
    for i, fit in enumerate(fits):
        fit['distinct_candidate'] = i in distinct
        fit['selected'] = i == chosen
    report = {'status': 'fit_detected', **fits[chosen],
              'accepted_window_count': len(fits), 'distinct_candidate_count': len(distinct)}
    return report, fits
