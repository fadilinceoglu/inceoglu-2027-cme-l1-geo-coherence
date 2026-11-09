"""Spacecraft and event association for the January 2026 study.

Shock, sheath entry, and trailing magnetic recovery are measured on each run.
"""

from config.study import UPSTREAM_SPACECRAFT, L1_SPACECRAFT, ARTEMIS_SPACECRAFT

# Event association is separate from the full-record, unseeded candidate scan.
# These broad observed arrival intervals distinguish the studied shock from
# other transitions; they do not supply a native crossing time to the fitter.
SHOCK_ASSOCIATION_WINDOWS = {
    "solar_orbiter": ("2026-01-19T12:00Z", "2026-01-19T12:10Z"),
    **{key: ("2026-01-19T18:55Z", "2026-01-19T19:05Z")
       for key in L1_SPACECRAFT},
    **{key: ("2026-01-19T19:08Z", "2026-01-19T19:16Z")
       for key in ARTEMIS_SPACECRAFT},
}
