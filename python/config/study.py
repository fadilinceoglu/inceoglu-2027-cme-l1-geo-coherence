"""Study windows, observer registry, plotting colors and reference geometry.

Observed physical endpoints and shock-association intervals are in config.event_boundaries.
The pipeline measures native shock clocks from the supplied mission records.
"""

from __future__ import annotations

from datetime import datetime, timezone


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

# Complete common scientific record: four full UTC days, 18--21 January.
# The stop is exclusive.  Every common-cadence and filtered waveform product
# is generated from this same record; physical boundaries never crop it.
ANALYSIS_START = _utc(2026, 1, 18)
ANALYSIS_STOP = _utc(2026, 1, 22)
FILTER_RECORD_START = ANALYSIS_START
FILTER_RECORD_STOP = ANALYSIS_STOP

# Real-data context supplied to every spatial filter.  The filtered products
# are then cropped back to FILTER_RECORD_START/STOP for storage and display.
FILTER_CONTEXT_START = _utc(2026, 1, 17)
FILTER_CONTEXT_STOP = _utc(2026, 1, 23)

# Figure 2 is the broad four-day event overview, not one of the isolated-
# waveform comparison products.  Its stop is exclusive, so every minute of
# 18--21 January is included without confusing this display range with a
# physical boundary.
FIGURE2_START = ANALYSIS_START
FIGURE2_STOP = ANALYSIS_STOP

# Available native-data days, including the context used before cropping
# filtered products to the four-day record.
FETCH_DAYS_UPSTREAM = [f"202601{d:02d}" for d in range(16, 23)]    # 16-22 Jan
FETCH_DAYS_DOWNSTREAM = [f"202601{d:02d}" for d in range(17, 22)]  # 17-21 Jan


# ---------------------------------------------------------------------------
# Satellite registry with fixed colors (colorblind-safe: Okabe-Ito + Tol;
# fixed assignment, never cycled)
# ---------------------------------------------------------------------------

SATELLITES = {
    # upstream (driver side)
    "solar_orbiter": {
        "name": "Solar Orbiter", "frame": "RTN", "color": "#332288",
    },
    "ace": {
        "name": "ACE", "frame": "GSM", "color": "#0072B2",
    },
    "wind": {
        "name": "Wind", "frame": "GSM", "color": "#E69F00",
    },
    "dscovr": {
        "name": "DSCOVR", "frame": "GSM", "color": "#009E73",
    },
    "swfo": {
        "name": "Solar-1", "frame": "GSM", "color": "#D55E00",
    },
    # Downstream data coverage; ARTEMIS is driver-side by observing geometry.
    "goes18": {
        "name": "GOES-18", "frame": "GSM", "color": "#56B4E9",
    },
    "goes19": {
        "name": "GOES-19", "frame": "GSM", "color": "#CC79A7",
    },
    "artemis_p1": {
        "name": "ARTEMIS-P1", "frame": "GSM", "color": "#117733",
    },
    "artemis_p2": {
        "name": "ARTEMIS-P2", "frame": "GSM", "color": "#882255",
    },
}

L1_SPACECRAFT = ("ace", "wind", "dscovr", "swfo")
ARTEMIS_SPACECRAFT = ("artemis_p1", "artemis_p2")
GOES_SPACECRAFT = ("goes18", "goes19")
UPSTREAM_SPACECRAFT = ("solar_orbiter", *L1_SPACECRAFT, *ARTEMIS_SPACECRAFT)

CRAFT_LABELS = {
    key: {"solar_orbiter": "SO", "artemis_p1": "P1", "artemis_p2": "P2"}.get(
        key, entry["name"]
    )
    for key, entry in SATELLITES.items()
}


def spacecraft_key(craft: str) -> str:
    """Resolve provider GOES aliases to the study registry keys."""
    key = {"g18": "goes18", "g19": "goes19"}.get(craft, craft)
    if key not in SATELLITES:
        raise KeyError(f"unknown spacecraft: {craft}")
    return key


def spacecraft_label(craft: str, *, short: bool = False) -> str:
    """Return a full mission name or its publication abbreviation."""
    key = spacecraft_key(craft)
    return CRAFT_LABELS[key] if short else SATELLITES[key]["name"]


MOON_COLOR = "#888888"

# Magnetic-field component colors, identical in every figure.
COMPONENT_COLORS = {"bx": "#0072B2", "by": "#E69F00", "bz": "#009E73"}
INK = "#1A1A1A"

# Structure annotation colors.
SHOCK_COLOR = "#B2182B"
SPAN_COLORS = {"sheath": ("#E69F00", 0.15), "ejecta": ("#332288", 0.10)}


# ---------------------------------------------------------------------------
# Observed GOES magnetic response interval
# ---------------------------------------------------------------------------
# These clocks were selected directly from the unfiltered GOES-18/19
# magnetometer records. They are local response windows, not propagated
# solar-wind arrival estimates.
GEO_RESPONSE_ONSET = _utc(2026, 1, 19, 19, 17, 0)
GEO_SHEATH_RESPONSE_STOP = _utc(2026, 1, 19, 21, 26, 0)


# ---------------------------------------------------------------------------
# Reference geometry (GSE, R_E unless noted)
# ---------------------------------------------------------------------------

# One reference epoch is shared by Figure 1 and waveform-pair geometry.
GEOMETRY_EPOCH = _utc(2026, 1, 19, 19, 0, 0)
L1_SSCWEB_CACHE = (
    "data/raw_event_20260119/l1/"
    "sscweb_step4_l1_lunar_20260119_1840_20260120_0800_gse.json"
)
# Fixed non-L1 points used by more than one reference-epoch geometry consumer.
SOLAR_ORBITER_POSITION_RE = (6015.1, -2428.1, 995.0)

GEO_ORBIT_RE = 6.6

EARTH_RADIUS_KM = 6371.0
