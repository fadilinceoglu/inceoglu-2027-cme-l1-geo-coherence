"""Overwrite spatial reference scales and band edges from this run's measurements."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from analysis.structure_bands import EARTH_RADIUS_KM
from analysis.waveform_scales import (
    derive_bands, interval_extent_re, method_parameters, sha256, sheath_spectral_scale,
    speed_at_time,
)
from config import waveform_bands as settings
from common.io import atomic_output_paths
from config.paths import PROCESSED_DIR, RESULTS_DIR
# Reuse coordinate preparation only; never read filtered waveform products here.
if TYPE_CHECKING:
    from pipeline.build_spatial_filters import PreparedWaveform


def build(prepared_records: dict[str, PreparedWaveform]) -> dict:
    boundary_path = RESULTS_DIR / "boundaries.json"
    boundaries = json.loads(boundary_path.read_text())
    speed_provenance_path = PROCESSED_DIR / "vt_provenance.json"
    speed_provenance = json.loads(speed_provenance_path.read_text())
    sources = [boundary_path, speed_provenance_path]
    per_craft = {}
    rows = []
    for craft in settings.L1_SPACECRAFT:
        boundary = boundaries[craft]
        if (boundary.get("shock_detection", {}).get("method") in (None, "unresolved")
                or boundary.get("sheath_detection", {}).get("status") != "fit_detected"
                or boundary.get("ejecta_detection", {}).get("status") != "confirmed_return"):
            raise ValueError(f"{craft}: all physical boundaries must be measured in this run")
        uncertainty = float(boundary["shock_unc_s"])
        if not np.isfinite(uncertainty) or uncertainty <= 0:
            raise ValueError(f"{craft}: invalid measured shock uncertainty")
        shock, sheath, ejecta = (
            pd.Timestamp(boundary[key]) for key in ("shock", "sheath_end", "ejecta_end")
        )
        prepared = prepared_records[craft]
        frame, speed = prepared.frame, prepared.speed
        source, speed_path = prepared.source_path, prepared.speed_path
        gap_policy, grid_diagnostics = prepared.speed_gap_policy, prepared.grid_diagnostics
        shock_speed = speed_at_time(speed, shock)
        x_time, x_grid, spatial_values = prepared.spatial_record
        spatial_record = (x_time, x_grid, spatial_values[:, 3])
        distance_re = pd.Series(spatial_record[0], index=frame.index)
        spectrum = sheath_spectral_scale(frame.bt, spatial_record, shock, sheath)
        row = {
            "craft": craft, "shock_unc_s": uncertainty,
            "shock_utc": shock.isoformat(), "sheath_end_utc": sheath.isoformat(),
            "ejecta_end_utc": ejecta.isoformat(),
            "speed_at_shock_kms": shock_speed,
            "sheath_re": interval_extent_re(distance_re, shock, sheath),
            "ejecta_re": interval_extent_re(distance_re, sheath, ejecta),
            "substructure_re": spectrum["wavelength_re"],
        }
        rows.append(row)
        per_craft[craft] = {
            **row, "spatial_spectrum": spectrum, "speed_gap_policy": gap_policy,
            "spatial_resampling": grid_diagnostics,
            "shock_speed_source": str(speed_path.relative_to(PROCESSED_DIR.parent)),
            "shock_speed_provenance": speed_provenance[craft],
        }
        sources.extend((source, speed_path))
    median_uncertainty = float(np.median([row["shock_unc_s"] for row in rows]))
    median_shock_speed = float(np.median([row["speed_at_shock_kms"] for row in rows]))
    references = {
        "short": median_uncertainty * median_shock_speed / EARTH_RADIUS_KM,
        **{key: float(np.median([row[f"{key}_re"] for row in rows]))
           for key in ("substructure", "sheath", "ejecta")},
    }
    cadences = {}
    for craft in settings.ARTEMIS_SPACECRAFT:
        source = PROCESSED_DIR / "native" / f"{craft}_1s_meta.json"
        cadence = float(json.loads(source.read_text())["native_fgs_median_cadence_s"])
        if not np.isfinite(cadence) or cadence <= 0:
            raise ValueError(f"{craft}: invalid measured survey cadence")
        cadences[craft] = cadence
        sources.append(source)
    worst_cadence = max(cadences.values())
    design = {
        "schema_version": "waveform-band-design-v2",
        "created_by": "pipeline/build_waveform_bands.py",
        "method_parameters": method_parameters(),
        "per_craft": per_craft,
        "l1_median_shock_unc_s": median_uncertainty,
        "l1_median_speed_at_shock_kms": median_shock_speed,
        "reference_scales_re": references,
        "artemis_native_cadences_s": cadences,
        "worst_artemis_cadence_s": worst_cadence,
        "bands": derive_bands(references, worst_cadence),
        "inputs": [
            {"path": str(path.relative_to(PROCESSED_DIR.parent)), "sha256": sha256(path)}
            for path in sources
        ],
        "interval_integration": "reuse the full-context trapezoidal distance mapping; linearly interpolate boundary coordinates and subtract",
        "fleet_policy": "all four L1 observers required; median of per-spacecraft measurements",
        "rebuild_policy": "recalculate and overwrite every run; no historic-value fallback",
    }
    path = PROCESSED_DIR / "waveform_bands.json"
    with atomic_output_paths((path,)) as (temporary,):
        temporary.write_text(json.dumps(design, indent=2, allow_nan=False) + "\n")
        pd.DataFrame(rows).to_csv(PROCESSED_DIR / "waveform_scales.csv", index=False)
    return design
