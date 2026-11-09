"""Confirmed magnetic recovery and supporting individual plasma returns.

The reference interval is selected using prior event inspection. The algorithm
does not discover quiet intervals. The pipeline uses confirmed magnetic recovery
as the operational trailing boundary, with plasma returns as supporting evidence.
All windows are half open. No gap filling or plasma substitution is performed.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecoverySettings:
    background_duration_min: int = 60
    background_gap_min: int = 120
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95
    magnetic_confirmation_min: int = 10
    magnetic_confirmation_fraction: float = 0.80

    def __post_init__(self):
        if self.background_duration_min <= 0:
            raise ValueError("Window durations must be positive")
        if self.background_gap_min < 0:
            raise ValueError("Background gap must be nonnegative")
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("Invalid background quantiles")
        if not isinstance(self.magnetic_confirmation_min, int) or self.magnetic_confirmation_min < 1:
            raise ValueError("Confirmation must contain a positive integer number of minutes")
        if not 0 < self.magnetic_confirmation_fraction <= 1:
            raise ValueError("Invalid magnetic confirmation fraction")


def observed_minutes(frame):
    """Keep only measured values, including rejection of count-zero fills."""
    result = frame.copy()
    result.index = pd.to_datetime(result.index, utc=True)
    if not result.index.is_unique or not result.index.is_monotonic_increasing:
        raise ValueError("Minute timestamps must be unique and ordered")
    if not (result.index == result.index.floor("min")).all():
        raise ValueError("Expected left-labelled minute bins")
    result = result.reindex(pd.date_range(result.index[0], result.index[-1], freq="min"))
    for column in ("bt", "bx", "by", "bz", "n", "v", "t"):
        result[column] = result[column].where(
            (result[f"{column}_samples"] > 0) & np.isfinite(result[column]))
    return result


def interval_stats(series, start, stop, settings):
    values = series.loc[(series.index >= start) & (series.index < stop)].dropna()
    return {
        "observed_minutes": len(values),
        "expected_minutes": int((stop - start).total_seconds() / 60),
        "mean": float(values.mean()) if len(values) else None,
        "std": float(values.std(ddof=1)) if len(values) >= 2 else None,
        "median": float(values.median()) if len(values) else None,
        "lower": float(values.quantile(settings.lower_quantile)) if len(values) else None,
        "upper": float(values.quantile(settings.upper_quantile)) if len(values) else None,
    }


def confirmed_magnetic_return(series, search_start, upper, settings):
    """First observed value <= upper with >=8 of its ten minute slots <= upper.

    The candidate belongs to the half-open interval [t, t+10 min). A candidate
    must itself be observed and below the upper limit. Missing values never
    contribute to the qualifying count; the denominator remains ten slots.
    Incomplete trailing windows cannot confirm a return. There is no lower
    threshold, extra averaging, interpolation, or imposed minimum ejecta duration.
    """
    duration = settings.magnetic_confirmation_min
    required = int(np.ceil(duration * settings.magnetic_confirmation_fraction))
    result = {"status": "no_confirmed_return", "time_utc": None, "value_nt": None,
              "confirmation_stop_utc_exclusive": None, "qualifying_minutes": None,
              "observed_minutes": None, "required_minutes": required,
              "window_minutes": duration, "upper_threshold_nt": upper}
    if upper is None or not np.isfinite(upper):
        return {**result, "status": "insufficient_background"}
    qualifying = series.notna() & (series <= upper)
    for i in np.flatnonzero(qualifying & (series.index >= search_start)):
        if i + duration > len(series):
            break
        window = series.iloc[i:i + duration]
        if ((window <= upper) & window.notna()).sum() >= required:
            return {**result, "status": "confirmed_return", "time_utc": series.index[i],
                    "value_nt": float(series.iloc[i]),
                    "confirmation_stop_utc_exclusive": series.index[i] + pd.Timedelta(minutes=duration),
                    "qualifying_minutes": int(qualifying.iloc[i:i + duration].sum()),
                    "observed_minutes": int(window.notna().sum())}
    return result


def background_returns(frame, shock, sheath_end, settings=RecoverySettings(),
                       excluded_parameters=None):
    """Report first in-range values and separately confirm magnetic recovery.

    First in-range values remain available for comparison and plasma context.
    Magnetic recovery uses the upper limit and the short confirmation above.
    If the first available post-entry value is already inside the
    range, report that rather than infer an earlier crossing. A reference with
    fewer than two observations is unresolved; always report the sample count.
    Instrument exclusions must be supplied explicitly by the caller.
    """
    frame = observed_minutes(frame)
    shock, sheath_end = pd.Timestamp(shock), pd.Timestamp(sheath_end)
    if shock.tzinfo is None or sheath_end.tzinfo is None:
        raise ValueError("Boundary times must include their UTC timezone")
    if sheath_end <= shock:
        raise ValueError("Sheath entry must follow the shock")
    stop = shock.floor("min") - pd.Timedelta(minutes=settings.background_gap_min)
    start = stop - pd.Timedelta(minutes=settings.background_duration_min)
    exclusions = excluded_parameters or {}
    background, returns = {}, {}
    for key in ("bt", "n", "v", "t"):
        stats = interval_stats(frame[key], start, stop, settings)
        background[key] = stats
        post = frame.loc[frame.index >= sheath_end, key].dropna()
        result = {"status": None, "first_in_range_utc": None,
                  "first_in_range_value": None,
                  "first_post_entry_observation_utc": post.index[0] if len(post) else None,
                  "post_entry_observed_minutes": len(post),
                  "outside_range_seen_since_shock": None}
        if key in exclusions:
            result.update(status="excluded", reason=exclusions[key])
        elif stats["observed_minutes"] < 2:
            result["status"] = "insufficient_background"
        elif post.empty:
            result["status"] = "no_post_entry_observations"
        else:
            inside = post.between(stats["lower"], stats["upper"])
            matches = post.loc[inside]
            first = matches.index[0] if len(matches) else None
            event = frame.loc[(frame.index >= shock)
                              & ((frame.index <= first) if first is not None else True), key].dropna()
            departed = bool((~event.between(stats["lower"], stats["upper"])).any())
            result["outside_range_seen_since_shock"] = departed
            if first is None:
                result["status"] = "no_return_in_record"
            else:
                result.update(first_in_range_utc=first,
                              first_in_range_value=float(matches.iloc[0]),
                              status=("no_departure_observed" if not departed else
                                      "already_in_range_at_first_observation" if first == post.index[0]
                                      else "first_return"))
        returns[key] = result
    return {
        "settings": asdict(settings),
        "shock_utc": shock,
        "sheath_entry_utc": sheath_end,
        "background_start_utc": start,
        "background_stop_utc_exclusive": stop,
        "background": background,
        "returns": returns,
        "magnetic_recovery": confirmed_magnetic_return(
            frame.bt, sheath_end,
            background["bt"]["upper"] if background["bt"]["observed_minutes"] >= 2 else None,
            settings),
        "reference_selection": "Fixed shock-relative interval based on prior inspection of this event",
        "timing_note": "Magnetic time is the start of its accepted confirmation window, not its end. No extra averaging or crossing interpolation; plasma returns remain individual first entries.",
    }
