"""Register trailing magnetic recovery from this run's observed minute records."""

import json
from pathlib import Path

import pandas as pd

from analysis.ejecta_recovery import background_returns
from analysis.product_io import json_ready
from config import study
from config.event_boundaries import UPSTREAM_SPACECRAFT
from config.paths import PROCESSED_DIR, RESULTS_DIR

QUALITY = {
    "solar_orbiter": "Local PAS measurements; temperature in eV.",
    "ace": "SWEPAM background lies in known tracking failure: plasma return times excluded.",
    "wind": "Local SWE moment fits; sparse observations retained without interpolation.",
    "dscovr": "Post-shock FC plasma excluded; no proxy values used.",
    "swfo": "Corrected SWiPS commissioning density, speed and temperature; plasma returns are descriptive only.",
    "artemis_p1": "ESA speed is sparse; density/temperature depend on viewing/mode. Plasma returns are descriptive only.",
    "artemis_p2": "ESA speed is sparse; mode effects and lunar wake affect plasma. Plasma returns are descriptive only.",
}


def build(boundary_path=RESULTS_DIR / "boundaries.json",
          minute_directory=PROCESSED_DIR / "common_cadence_1min"):
    boundary_path, minute_directory = Path(boundary_path), Path(minute_directory)
    registered = json.loads(boundary_path.read_text())
    reports, rows = {}, []
    for craft in UPSTREAM_SPACECRAFT:
        entry = registered[craft]
        if entry["shock_detection"]["method"] == "unresolved" or entry["sheath_detection"]["status"] != "fit_detected":
            raise ValueError(f"{craft}: measure shock and sheath entry before magnetic recovery")
        frame = pd.read_csv(minute_directory / f"{craft}.csv", index_col=0, parse_dates=[0])
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame = frame.loc[(frame.index >= study.ANALYSIS_START) & (frame.index < study.ANALYSIS_STOP)]
        exclusions = {key: QUALITY[craft] for key in ("n", "v", "t")} if craft in ("ace", "dscovr") else {}
        report = background_returns(frame, entry["shock"], entry["sheath_end"], excluded_parameters=exclusions)
        report["plasma_quality_note"] = QUALITY[craft]
        reports[craft] = report
        for variable, result in report["returns"].items():
            rows.append({"spacecraft": craft, "variable": variable,
                         "background_start_utc": report["background_start_utc"],
                         "background_stop_utc_exclusive": report["background_stop_utc_exclusive"],
                         **{f"background_{k}": v for k, v in report["background"][variable].items()},
                         **result,
                         **({f"magnetic_{k}": v for k, v in report["magnetic_recovery"].items()} if variable == "bt" else {}),
                         "quality_note": QUALITY[craft] if variable != "bt" else "Observed magnetic means"})
    failures = {craft: report["magnetic_recovery"]["status"] for craft, report in reports.items()
                if report["magnetic_recovery"]["status"] != "confirmed_return"}
    if failures:
        raise ValueError(f"Unresolved trailing magnetic recovery; no endpoints substituted: {failures}")

    for craft, report in reports.items():
        recovery = report["magnetic_recovery"]
        registered[craft].update(
            ejecta_end=recovery["time_utc"], ejecta_end_unc_s=None,
            ejecta_detection={**recovery, "settings": report["settings"],
                              "background": report["background"]["bt"],
                              "background_start_utc": report["background_start_utc"],
                              "background_stop_utc_exclusive": report["background_stop_utc_exclusive"],
                              "sampling_interval_s": 60,
                              "timing_note": "Start of accepted minute window; no statistical timing uncertainty assigned"},
            boundary_basis="measured shock, sheath entry, and confirmed magnetic recovery",
        )
    registered.setdefault("_meta", {}).update(
        ejecta_status="complete",
        ejecta_method="first observed post-entry |B| <= background Q95 confirmed in at least 8 of 10 minute bins",
        ejecta_reference="one-hour interval ending two hours before the shock minute, selected using prior event inspection",
        ejecta_interval_convention="half-open [t,t+10 min); candidate must qualify; missing bins do not pass; complete ten-bin window required",
        ejecta_uncertainty_note="minute timestamps; confirmation interval is not a confidence interval; sensitivity analysis is separate",
    )
    output = boundary_path.parent
    pd.DataFrame(rows).to_csv(output / "ejecta_recovery.csv", index=False)
    (output / "ejecta_recovery.json").write_text(json.dumps(json_ready({
        "interpretation": "Trailing magnetic recovery, with supporting independent plasma first returns",
        "spacecraft": reports,
    }), indent=2, allow_nan=False) + "\n")
    boundary_path.write_text(json.dumps(json_ready(registered), indent=2, allow_nan=False) + "\n")
    return reports


if __name__ == "__main__":
    for craft, report in build().items():
        recovery = report["magnetic_recovery"]
        print(f"{craft}: {recovery['time_utc']}; {recovery['qualifying_minutes']}/{recovery['window_minutes']} minutes qualify")
