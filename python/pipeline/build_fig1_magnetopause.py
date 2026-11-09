"""Recompute Figure 1 Shue contexts from the current run's observed inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.magnetopause import select_figure1_states
from common.io import read_json, sha256 as _sha256
from config import study
from config.paths import DATA_ROOT, PROCESSED_DIR, RESULTS_DIR
from ingest.solar1_gsm import read_native_solar1_inputs


def caption(records, metadata):
    onset, snapshot = records
    epoch = pd.Timestamp(metadata["reference_geometry_epoch_utc"])
    onset_time = pd.Timestamp(onset["target_geo_time_utc"])
    solid_start, solid_end = (pd.Timestamp(onset[key]) for key in ("source_window_start_utc", "source_window_end_utc"))
    dashed_start, dashed_end = (pd.Timestamp(snapshot[key]) for key in ("source_window_start_utc", "source_window_end_utc"))
    correction = metadata["native_input_provenance"]["moment_status"]
    scales = metadata["upstream_display_scales"]
    return (
        f"Observer geometry at {epoch:%H:%M} UTC on {epoch:%d %B %Y}. "
        "Panels (a,c) show Solar Orbiter and the four L1 spacecraft in GSE X–Y and X–Z, with L1-fleet insets. "
        "Panels (b,d) show GOES-18/19, the Moon and ARTEMIS-P1/P2 in the same planes; their insets enlarge Earth and GEO. "
        f"The grey curve is the projected nominal {study.GEO_ORBIT_RE:g}-R_E terrestrial equatorial orbit. "
        "Purple curves show the un-aberrated, axisymmetric Shue (1998) magnetopause using Solar-1 native GSM Bz "
        "and proton dynamic pressure calculated from its paired SWiPS proton density and speed, with no assumed helium contribution. "
        f"The solid curve uses {solid_start:%H:%M}–{solid_end:%H:%M} UTC upstream inputs as immediate context for the "
        f"{onset_time:%H:%M} UTC GOES magnetic-response onset; the dashed curve uses {dashed_start:%H:%M}–{dashed_end:%H:%M} UTC "
        f"inputs associated with the {epoch:%H:%M} UTC snapshot. "
        "The association uses the measured Solar-1 shock and manually selected GOES response clocks, not an independently fitted travel time. "
        f"SWiPS proton moments are commissioning data; their current correction status is {correction}. "
        "The empirical surfaces provide context rather than measured crossings. "
        f"Upstream display scales are unequal (Y:X = {scales['xy']:.2f}; Z:X = {scales['xz']:.2f}); "
        "displayed angles in (a,c) are not physical directions."
    )


def build(*, data_root=DATA_ROOT, processed_dir=PROCESSED_DIR, results_dir=RESULTS_DIR):
    data_root, processed_dir, results_dir = map(Path, (data_root, processed_dir, results_dir))
    boundary_path = results_dir / "boundaries.json"
    response_path = results_dir / "paired_goes_response.json"
    overview_path = processed_dir / "swfo_overview_1min.csv"
    boundaries = read_json(boundary_path)
    response = read_json(response_path)
    overview = pd.read_csv(overview_path, index_col="time")
    overview.index = pd.to_datetime(overview.index, utc=True)
    native, provenance = read_native_solar1_inputs(data_root)
    records, metadata = select_figure1_states(overview, native, boundaries, response, study.GEOMETRY_EPOCH)
    metadata["native_input_provenance"] = provenance
    from figures import fig_geometry
    metadata["upstream_display_scales"] = {
        plane: float(fig_geometry.PANEL_BOX_ASPECT * (fig_geometry.UPSTREAM_X_LIMITS[1]-fig_geometry.UPSTREAM_X_LIMITS[0])
                     / (limits[1]-limits[0]))
        for plane, limits in fig_geometry.UPSTREAM_TRANSVERSE_LIMITS.items()}
    metadata["inputs"] = {name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                          for name, path in (("boundaries", boundary_path), ("goes_response", response_path), ("overview", overview_path))}
    metadata["caption"] = caption(records, metadata)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "fig1_magnetopause.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    (results_dir / "fig1_magnetopause.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    (results_dir / "Fig1_caption.md").write_text(metadata["caption"]+"\n")
    return csv_path


if __name__ == "__main__":
    print(build())
