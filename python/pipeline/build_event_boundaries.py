"""Initialize an empty boundary registry for current-run measurements."""
from __future__ import annotations

import json
from pathlib import Path


from config.event_boundaries import UPSTREAM_SPACECRAFT
from config.paths import RESULTS_DIR, FIGURE_DIR

BOUNDARIES_PATH = RESULTS_DIR / "boundaries.json"


def initialize(path: Path = BOUNDARIES_PATH) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    result = {}
    for craft in UPSTREAM_SPACECRAFT:
        result[craft] = {
            "ejecta_end": None,
            "ejecta_end_unc_s": None,
            "boundary_basis": "pending measurement from mission inputs",
        }
    result["_meta"] = {
        "source": "config/event_boundaries.py",
        "shock_status": "pending magnetic candidate detection",
        "sheath_status": "pending variability and angular-fit detection",
        "ejecta_status": "pending confirmed magnetic recovery",
    }
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    initialize()
