"""Method settings; measured scales and edges are recalculated each run.

The short reference uses the current median prepared speed at the L1 shock
times. The sampling-floor speed is an explicit method setting. Derived
quantities belong in processed/waveform_bands.json.
"""

from config.study import L1_SPACECRAFT, ARTEMIS_SPACECRAFT

WAVEFORM_FAMILIES = ("discontinuity", "substructure", "sheath", "trend")
# Stable product identifiers above preserve the scientific interfaces. These
# display names describe spatial scales rather than identified plasma regions.
WAVEFORM_FAMILY_LABELS = {
    "discontinuity": "Fine-scale",
    "substructure": "Intermediate-scale",
    "sheath": "Large-scale",
    "trend": "Trend",
}
SPATIAL_DX_RE = 0.04
FILTER_ORDER = 4
INFORMATION_FLOOR_REFERENCE_SPEED_KMS = 1100.0
INFORMATION_FLOOR_MULTIPLIER = 2.5
PSD_MINIMUM_SAMPLES = 32

# Figure 4 intervals: for each spacecraft and family, the points holding these
# fractions of cumulative filtered band energy (saved one-minute bt_f squared),
# searched from the shock minus the margin to the ejecta end plus the margin,
# then padded on both sides.
INTERVAL_ENERGY_FRACTIONS = (0.05, 0.95)
INTERVAL_SEARCH_MARGIN_MINUTES = 120
INTERVAL_PADDING_MINUTES = 120
