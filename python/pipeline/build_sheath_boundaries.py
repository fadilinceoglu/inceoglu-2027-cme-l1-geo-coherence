"""Measure sheath/ejecta entry from observed minute records and measured shocks.

This stage follows common-cadence preparation and shock detection. No stored
sheath time, exploratory result, or spacecraft-specific entry seed is required.
"""

import json
from pathlib import Path

import pandas as pd

from analysis.sheath_ejecta_detection import detect_fleet
from analysis.product_io import json_ready
from config import study
from config.event_boundaries import UPSTREAM_SPACECRAFT
from config.paths import PROCESSED_DIR, RESULTS_DIR

CRAFT = tuple(UPSTREAM_SPACECRAFT)


def build(boundary_path=RESULTS_DIR / "boundaries.json",
          minute_directory=PROCESSED_DIR / "common_cadence_1min"):
    boundary_path, minute_directory = Path(boundary_path), Path(minute_directory)
    registered = json.loads(boundary_path.read_text())
    frames, shock_times = {}, {}
    for craft in CRAFT:
        frame = pd.read_csv(minute_directory / f"{craft}.csv", index_col=0, parse_dates=[0])
        frame.index = pd.to_datetime(frame.index, utc=True)
        frames[craft] = frame.loc[(frame.index >= study.ANALYSIS_START)
                                 & (frame.index < study.ANALYSIS_STOP)]
        shock = registered[craft]["shock_detection"]
        if shock["method"] == "unresolved":
            raise ValueError(f"{craft}: measure the shock before sheath entry")
        shock_times[craft] = pd.Timestamp(shock["t50"])
    reports, diagnostics, comparison = detect_fleet(frames, shock_times)
    failures = {craft: report["status"] for craft, report in reports.items()
                if report["status"] != "fit_detected"}
    if failures:
        raise ValueError(f"Unresolved sheath/ejecta entries; no times substituted: {failures}")

    candidates = []
    for craft, report in reports.items():
        candidates.extend({"spacecraft": craft, **row} for row in report.pop("fit_candidates"))
        entry = registered[craft]
        entry.update(
            sheath_end=report["boundary_utc"],
            sheath_end_unc_s=report["timing_halfwidth_seconds"],
            sheath_detection=report,
            boundary_basis="measured shock and sheath entry; magnetic recovery pending",
        )
    registered.setdefault("_meta", {}).update(
        sheath_status="complete",
        sheath_l1_comparison=comparison,
        sheath_method="observed minute variability -> L1 association -> unit-vector reference -> logistic angular decrease",
        sheath_interval_convention="10 samples in centered std window (t-5 min, t+5 min]; persistence and reference include both endpoints (11 timestamps over 10 min)",
        sheath_uncertainty_note="descriptive half-width with cadence and gap floors, not a confidence interval",
    )
    output = boundary_path.parent
    pd.concat([frame.assign(spacecraft=craft) for craft, frame in diagnostics.items()]).to_csv(
        output / "sheath_boundary_diagnostics.csv", index_label="time")
    pd.DataFrame(candidates).to_csv(output / "sheath_boundary_candidates.csv", index=False)
    boundary_path.write_text(json.dumps(json_ready(registered), indent=2,
                                        allow_nan=False) + "\n")
    return reports


if __name__ == "__main__":
    for craft, result in build().items():
        print(f"{craft}: {result['boundary_utc']} +/- {result['timing_halfwidth_seconds']:g} s")
