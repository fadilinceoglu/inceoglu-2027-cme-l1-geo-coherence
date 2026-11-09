"""Access to the registered event structure boundaries.

The boundary-ID stage writes <output-dir>/results/boundaries.json:
    {craft: {"shock": iso, "shock_unc_s": float,
             "sheath_end": iso, "sheath_end_unc_s": float,
             "ejecta_end": iso|null, "ejecta_end_unc_s": float|null,
             "evidence": {...}}}
Figures and analysis stages read through this module and fail closed when the
registered boundary product is unavailable.
"""

from __future__ import annotations

from config.paths import RESULTS_DIR

import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
BOUNDARIES_PATH = RESULTS_DIR / "boundaries.json"


def _parse(iso: str | None) -> datetime | None:
    if iso is None:
        return None
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def measured_available() -> bool:
    return BOUNDARIES_PATH.is_file()


def load() -> tuple[dict, str]:
    """Return ``({craft: boundaries}, source_tag)``.

    Every consumer uses the registered boundary product.  Missing boundaries
    are therefore an error rather than a reason to substitute archive clocks.
    """
    if not measured_available():
        raise FileNotFoundError(
            "registered publication boundaries are required: "
            f"{BOUNDARIES_PATH}"
        )

    with BOUNDARIES_PATH.open() as handle:
        raw = json.load(handle)
    out = {}
    for craft, boundaries in raw.items():
        if craft.startswith("_"):
            continue
        out[craft] = {
            "shock": _parse(boundaries["shock"]),
            "sheath_end": _parse(boundaries["sheath_end"]),
            "ejecta_end": _parse(boundaries.get("ejecta_end")),
        }
    return out, "measured shock, sheath entry, and confirmed magnetic recovery"


def marks_for(craft: str) -> dict:
    """Return registered publication marks, failing closed if unavailable."""
    b, _ = load()
    if craft not in b:
        raise KeyError(f"registered boundaries do not contain {craft!r}")
    e = b[craft]
    return {"shock": e["shock"],
            "spans": [(e["shock"], e["sheath_end"], "sheath"),
                      (e["sheath_end"], e["ejecta_end"], "ejecta")]}


def shock_measurement(craft: str, path: Path | None = None) -> dict:
    """Read the measured ramp and its fit/fallback evidence from this run."""
    records = json.loads((path or BOUNDARIES_PATH).read_text())
    result = dict(records[craft]["shock_detection"])
    if result["method"] == "unresolved":
        raise ValueError(f"{craft}: the event shock remains unresolved")
    for key in ("t25", "t50", "t75", "fit_window_start", "fit_window_end",
                "predicted_time", "source_time", "median_reference_time", "gap_start", "gap_stop"):
        if result.get(key) is not None:
            result[key] = pd.Timestamp(result[key])
    return result


def shock_delay(target: str, source: str = "swfo", path: Path | None = None) -> float:
    """Measured magnetic arrival-time difference, without waveform alignment."""
    return (shock_measurement(target, path)["t50"]
            - shock_measurement(source, path)["t50"]).total_seconds()


def sheath_measurement(craft: str, path: Path | None = None) -> dict:
    """Return this run's data-selected reference and fitted angular boundary."""
    records = json.loads((path or BOUNDARIES_PATH).read_text())
    result = dict(records[craft]["sheath_detection"])
    if result["status"] != "fit_detected":
        raise ValueError(f"{craft}: the sheath/ejecta entry remains unresolved")
    for key, value in result.items():
        if key.endswith("_utc") and value is not None:
            result[key] = pd.Timestamp(value)
    return result


def ejecta_measurement(craft: str, path: Path | None = None) -> dict:
    """Read the common magnetic-recovery measurement and its background."""
    records = json.loads((path or BOUNDARIES_PATH).read_text())
    result = dict(records[craft]["ejecta_detection"])
    if result["status"] != "confirmed_return":
        raise ValueError(f"{craft}: trailing magnetic recovery remains unresolved")
    for key in ("time_utc", "confirmation_stop_utc_exclusive", "background_start_utc",
                "background_stop_utc_exclusive"):
        result[key] = pd.Timestamp(result[key])
    return result
