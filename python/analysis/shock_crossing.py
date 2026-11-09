"""Native level crossings guided by a nearby measured shock.

These retained mission methods are reached only when the principal native
logistic fit is unresolved. No historical clock enters any crossing search.
"""
import numpy as np
import pandas as pd

UP_WIN = (-72.0, -12.0)
DOWN_WIN = (12.0, 72.0)
HOLD_N = 5
SHOCK_SEARCH_S = 120
SMOOTH_S = 5
PRE_WIN_S = (150, 30)
POST_WIN_S = (30, 150)
HOLD_FRAC = 0.8


def _cross_time(s: pd.Series, level: float, seed: pd.Timestamp) -> tuple:
    """Upward crossing of `level` nearest the seed, on valid samples.

    Returns (t_interp, t_lo, t_hi): linear-interpolated crossing time and
    the bracketing valid samples (t_lo below the level, t_hi at/above it).
    """
    v = s.to_numpy()
    below = v[:-1] < level
    above = v[1:] >= level
    idx = np.flatnonzero(below & above)
    if idx.size == 0:
        raise ValueError(f"no upward crossing of {level:.1f} nT")
    times = s.index.view("int64") / 1e9
    mid = 0.5 * (times[idx] + times[idx + 1])
    i = idx[np.argmin(np.abs(mid - seed.timestamp()))]
    frac = (level - v[i]) / (v[i + 1] - v[i])
    t_interp = pd.Timestamp(
        (times[i] + frac * (times[i + 1] - times[i])) * 1e9, unit="ns",
        tz="UTC")
    return t_interp, s.index[i], s.index[i + 1]

def ace_crossing(df: pd.DataFrame, seed: pd.Timestamp, *, cadence_s: float,
                up_window_s: tuple[float, float] = (-65.0, -5.0),
                down_window_s: tuple[float, float] = (5.0, 65.0)) -> dict:
    s = df.bt.loc[seed - pd.Timedelta("10min"):
                  seed + pd.Timedelta("10min")].dropna()
    b_up = s.loc[seed + pd.Timedelta(seconds=up_window_s[0]):
                 seed + pd.Timedelta(seconds=up_window_s[1])
                 ].median()
    b_dn = s.loc[seed + pd.Timedelta(seconds=down_window_s[0]):
                 seed + pd.Timedelta(seconds=down_window_s[1])
                 ].median()
    jump = b_dn - b_up
    cross = {f: _cross_time(s, b_up + f * jump, seed)
             for f in (0.25, 0.50, 0.75)}
    t50, _, _ = cross[0.50]

    # Native-sample bracket containing the crossings (the flagged gap when
    # the ramp is flagged; one native step when it is resolved).
    lo = min(c[1] for c in cross.values())
    hi = max(c[2] for c in cross.values())
    bracket_s = (hi - lo).total_seconds()

    ramp_interp = (cross[0.75][0] - cross[0.25][0]).total_seconds()
    ramp_resolved = bracket_s <= 2 * cadence_s + 1e-9
    ramp_25_75 = ramp_interp if ramp_resolved else bracket_s
    unc = max(ramp_25_75 / 2.0, cadence_s)
    return {
        "t50": t50, "unc_s": unc, "ramp_25_75_s": ramp_25_75,
        "b_up": b_up, "b_dn": b_dn,
        "bracket": (lo, hi, bracket_s), "ramp_interp_s": ramp_interp,
        "ramp_resolved": ramp_resolved,
    }

def wind_crossing(df: pd.DataFrame, t0: pd.Timestamp, *, cadence_s: float,
                up_window_s: tuple[float, float] = UP_WIN,
                down_window_s: tuple[float, float] = DOWN_WIN) -> dict:
    """50% |B| crossing of the main ramp; 25/75% bracket -> uncertainty."""
    w = df.loc[t0 - pd.Timedelta(minutes=3):t0 + pd.Timedelta(minutes=3), "bt"]
    step = w.diff()
    t_b = step.idxmax()                            # steepest native step
    i_b = w.index.get_loc(t_b)
    t_steep = w.index[i_b - 1] + (t_b - w.index[i_b - 1]) / 2

    b_up = w.loc[t_steep + pd.Timedelta(seconds=up_window_s[0]):
                 t_steep + pd.Timedelta(seconds=up_window_s[1])].median()
    b_dn = w.loc[t_steep + pd.Timedelta(seconds=down_window_s[0]):
                 t_steep + pd.Timedelta(seconds=down_window_s[1])].median()

    def crossing(level: float) -> pd.Timestamp:
        """First sustained upward crossing, linearly interpolated."""
        v, idx = w.to_numpy(), w.index
        for i in range(1, len(v)):
            if v[i] >= level and v[i - 1] < level and \
                    np.nanmin(v[i:i + HOLD_N]) >= level:
                frac = (level - v[i - 1]) / (v[i] - v[i - 1])
                return idx[i - 1] + frac * (idx[i] - idx[i - 1])
        raise RuntimeError(f"no sustained crossing of {level:.2f} nT")

    t25 = crossing(b_up + 0.25 * (b_dn - b_up))
    t50 = crossing(b_up + 0.50 * (b_dn - b_up))
    t75 = crossing(b_up + 0.75 * (b_dn - b_up))
    ramp = (t75 - t25).total_seconds()
    return {"t50": t50, "t25": t25, "t75": t75, "ramp_s": ramp,
            "unc_s": max(cadence_s, ramp / 2.0),
            "b_up": float(b_up), "b_dn": float(b_dn), "t_steep": t_steep}

def cross_time(s: pd.Series, level: float, i: int, direction: int) -> pd.Timestamp:
    """Linearly interpolated upward crossing of `level`, scanning from sample i.

    direction=-1: last crossing at/before i;  +1: first crossing at/after i.
    """
    v, t = s.to_numpy(), s.index
    j = i
    while 0 < j < len(v) - 1:
        if v[j - 1] < level <= v[j]:
            frac = (level - v[j - 1]) / (v[j] - v[j - 1])
            return t[j - 1] + (t[j] - t[j - 1]) * frac
        j += direction
    raise RuntimeError(f"no crossing of {level}")

def dscovr_crossing(df: pd.DataFrame, t0: pd.Timestamp, *, cadence_s: float,
                up_window_s: tuple[float, float] = (-PRE_WIN_S[0], -PRE_WIN_S[1]),
                down_window_s: tuple[float, float] = POST_WIN_S) -> dict:
    w = df.bt.loc[t0 - pd.Timedelta(seconds=SHOCK_SEARCH_S):
                  t0 + pd.Timedelta(seconds=SHOCK_SEARCH_S)]
    sm = df.bt.loc[w.index[0] - pd.Timedelta(seconds=30):
                   w.index[-1] + pd.Timedelta(seconds=30)] \
        .rolling(f"{SMOOTH_S}s", center=True).median()
    grad = (sm.shift(-SMOOTH_S) - sm.shift(SMOOTH_S)).loc[w.index]  # 10-s rise
    t_steep = grad.idxmax()

    pre = df.bt.loc[t_steep + pd.Timedelta(seconds=up_window_s[0]):
                    t_steep + pd.Timedelta(seconds=up_window_s[1])].median()
    post = df.bt.loc[t_steep + pd.Timedelta(seconds=down_window_s[0]):
                     t_steep + pd.Timedelta(seconds=down_window_s[1])].median()
    lev = {f: pre + f / 100 * (post - pre) for f in (25, 50, 75)}

    seg = df.bt.loc[t_steep - pd.Timedelta(seconds=SHOCK_SEARCH_S):
                    t_steep + pd.Timedelta(seconds=SHOCK_SEARCH_S)].dropna()
    i0 = int(seg.index.get_indexer([t_steep], method="nearest")[0])
    # 50%: crossing nearest t_steep (scan back, else forward)
    try:
        t50 = cross_time(seg, lev[50], i0, -1)
    except RuntimeError:
        t50 = cross_time(seg, lev[50], i0, +1)
    i50 = int(seg.index.get_indexer([t50], method="nearest")[0])
    t25 = cross_time(seg, lev[25], i50, -1)          # last 25% crossing before
    t75 = cross_time(seg, lev[75], max(i50, 1), +1)  # first 75% crossing after

    ramp = (t75 - t25).total_seconds()
    return {"t50": t50, "t25": t25, "t75": t75, "ramp_s": ramp,
            "unc_s": max(cadence_s, ramp / 2),
            "pre_nT": pre, "post_nT": post, "t_steep": t_steep}


def solar1_crossing(bt: pd.Series, *, cadence_s: float,
                up_window_s: tuple[float, float] = (-90.0, -15.0),
                down_window_s: tuple[float, float] = (15.0, 60.0),
                reference_time: pd.Timestamp) -> dict:
    t0 = pd.Timestamp(reference_time)
    pre = bt.loc[t0 + pd.Timedelta(seconds=up_window_s[0]):
                 t0 + pd.Timedelta(seconds=up_window_s[1])].median()
    post = bt.loc[t0 + pd.Timedelta(seconds=down_window_s[0]):
                  t0 + pd.Timedelta(seconds=down_window_s[1])].median()
    jump = post - pre
    lev = {p: pre + jump * p / 100 for p in (25, 50, 75)}

    win = bt.loc[t0 - pd.Timedelta("300s"):
                 t0 + pd.Timedelta("300s")]
    t50 = next(t for t, v in win.items()
               if v >= lev[50] and _sustained(bt, lev[50], t, "30s"))
    below25 = win.loc[:t50][win.loc[:t50] < lev[25]]
    t25 = win.loc[below25.index[-1]:].index[1]  # first sample back above 25%
    t75 = next(t for t, v in win.loc[t50:].items()
               if v >= lev[75] and _sustained(bt, lev[75], t, "30s"))

    ramp = (t75 - t25).total_seconds()
    return {"shock_utc": t50, "t25": t25, "t75": t75,
            "ramp_25_75_s": ramp, "shock_unc_s": max(ramp / 2, cadence_s),
            "b_pre": float(pre), "b_post": float(post),
            "levels": {k: float(v) for k, v in lev.items()}}

def _despike(series: pd.Series) -> pd.Series:
    return series.rolling(3, center=True, min_periods=1).median()

def _sustained(
    series: pd.Series,
    level: float,
    timestamp: pd.Timestamp,
    hold: str,
) -> bool:
    window = series.loc[timestamp:timestamp + pd.Timedelta(hold)].dropna()
    return len(window) > 0 and (window >= level).mean() >= HOLD_FRAC

def artemis_crossing(bt: pd.Series, cadence_s: float, *,
                up_window_s: tuple[float, float] = (-90.0, -15.0),
                down_window_s: tuple[float, float] = (15.0, 60.0),
                reference_time: pd.Timestamp) -> dict:
    """Measure the sustained 50% main-ramp crossing at FGS cadence.

    The pre/post states are medians over [T0-90 s, T0-15 s] and
    [T0+15 s, T0+60 s]. Crossing persistence is evaluated on a three-sample
    median despike. Uncertainty is half the 25%-75% bracket, floored at one
    native sample.
    """

    t0 = pd.Timestamp(reference_time)
    pre = bt.loc[
        t0 + pd.Timedelta(seconds=up_window_s[0]):
        t0 + pd.Timedelta(seconds=up_window_s[1])
    ].median()
    post = bt.loc[
        t0 + pd.Timedelta(seconds=down_window_s[0]):
        t0 + pd.Timedelta(seconds=down_window_s[1])
    ].median()
    jump = post - pre
    levels = {percent: pre + jump * percent / 100 for percent in (25, 50, 75)}

    window = bt.loc[
        t0 - pd.Timedelta("300s"):t0 + pd.Timedelta("300s")
    ]
    smooth = _despike(window)
    t50 = next(
        timestamp
        for timestamp, value in window.items()
        if value >= levels[50]
        and _sustained(smooth, levels[50], timestamp, "30s")
    )
    below25 = window.loc[:t50][window.loc[:t50] < levels[25]]
    t25 = window.loc[below25.index[-1]:].index[1]
    t75 = next(
        timestamp
        for timestamp, value in window.loc[t50:].items()
        if value >= levels[75]
        and _sustained(smooth, levels[75], timestamp, "30s")
    )

    ramp_s = (t75 - t25).total_seconds()
    return {
        "shock_utc": t50,
        "t25": t25,
        "t75": t75,
        "ramp_25_75_s": ramp_s,
        "shock_unc_s": max(ramp_s / 2, cadence_s),
        "b_pre": float(pre),
        "b_post": float(post),
    }

def measure_crossing(craft, native, cadence, predicted):
    """Adapt the retained methods to one evidence schema for the registry."""
    if craft.startswith("artemis"):
        clock = artemis_crossing(native, cadence, reference_time=predicted)
    elif craft == "swfo":
        clock = solar1_crossing(native, cadence_s=cadence, reference_time=predicted)
    else:
        clock = {"ace": ace_crossing, "wind": wind_crossing,
                 "dscovr": dscovr_crossing}[craft](native.to_frame("bt"), predicted, cadence_s=cadence)
    measured = pd.Timestamp(clock.get("shock_utc", clock.get("t50")))
    uncertainty = float(clock.get("shock_unc_s", clock.get("unc_s")))
    if craft in ("ace", "wind"):
        upstream, downstream = clock["b_up"], clock["b_dn"]
    elif craft == "dscovr":
        upstream, downstream = clock["pre_nT"], clock["post_nT"]
    else:
        upstream, downstream = clock["b_pre"], clock["b_post"]
    if not np.isfinite([upstream, downstream, uncertainty]).all() or downstream <= upstream or uncertainty <= 0:
        raise ValueError("A finite positive field increase and timing uncertainty are required")
    if craft == "ace":
        local = native.loc[predicted - pd.Timedelta(minutes=10):predicted + pd.Timedelta(minutes=10)].dropna()
        t25 = _cross_time(local, upstream + .25 * (downstream - upstream), predicted)[0]
        t75 = _cross_time(local, upstream + .75 * (downstream - upstream), predicted)[0]
    else:
        t25, t75 = clock["t25"], clock["t75"]
    if abs((measured - predicted).total_seconds()) > 90 or not t25 <= measured <= t75:
        raise ValueError("No ordered local crossing within 90 s of the guide")
    windows = {"ace": ([-65., -5.], [5., 65.]), "wind": (list(UP_WIN), list(DOWN_WIN)),
               "dscovr": ([-PRE_WIN_S[0], -PRE_WIN_S[1]], list(POST_WIN_S))}
    up_window, down_window = windows.get(craft, ([-90., -15.], [15., 60.]))
    return dict(method="neighbour-guided native crossing", t25=t25, t50=measured, t75=t75,
                unc_s=uncertainty, ramp_s=(t75 - t25).total_seconds(),
                b_up=float(upstream), b_dn=float(downstream), cadence_s=cadence,
                median_reference_time=clock.get("t_steep", predicted),
                up_window_s=up_window, down_window_s=down_window)

