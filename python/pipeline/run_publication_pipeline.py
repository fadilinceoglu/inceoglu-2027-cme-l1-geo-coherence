"""Generate the paper results and supporting diagnostics from supplied mission data.

Run from the repository root:
    python python/pipeline/run_publication_pipeline.py

Mission files use the layout in README.md. By default, results and figures
are generated under python/output/. Use --data-dir and --output-dir for an
independent run. No products from a previous calculation are required.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence

PYTHON_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIRECTORY = PYTHON_ROOT / "data"
DEFAULT_OUTPUT_DIRECTORY = PYTHON_ROOT / "output"

@dataclass(frozen=True)
class PipelineStep:
    label: str
    argv: tuple[str, ...]

# Source choices and mission observations precede every derived quantity.
PUBLICATION_STEPS: tuple[PipelineStep, ...] = (
    PipelineStep("initialize boundary registry", ("pipeline/build_event_boundaries.py",)),
    PipelineStep("mission observations and required representations", ("pipeline/build_observations.py",)),
    PipelineStep("logistic shocks and nearby-spacecraft fallback", ("pipeline/build_shock_boundaries.py",)),
    PipelineStep("data-derived sheath/ejecta boundaries", ("pipeline/build_sheath_boundaries.py",)),
    PipelineStep("confirmed trailing magnetic recovery", ("pipeline/build_ejecta_boundaries.py",)),
    PipelineStep("mission overviews", ("pipeline/build_overviews.py",)),
    PipelineStep("bulk-speed profiles", ("pipeline/build_speed_profiles.py",)),
    PipelineStep("Wind minute gap fill", ("pipeline/fill_wind_minute_gaps.py",)),
    PipelineStep("GOES minute gap fill", ("pipeline/fill_goes_minute_gaps.py",)),
    PipelineStep("measure waveform scales and apply spatial filters", ("pipeline/build_spatial_filters.py",)),
    PipelineStep("band-energy waveform intervals", ("pipeline/build_waveform_intervals.py",)),
    PipelineStep("physical waveform metrics", ("pipeline/build_chain_waveform_metrics.py",)),
    PipelineStep("pairwise waveform comparisons", ("pipeline/build_waveform_comparisons.py",)),
    PipelineStep("waveform uncertainty and persistence", ("pipeline/build_waveform_inference.py",)),
    PipelineStep("GSM Bz persistence at scalar alignment", ("pipeline/build_bz_persistence.py",)),
    PipelineStep("activity-weighted wavelet coherence and spatial persistence", ("pipeline/build_wavelet_coherence.py",)),
    PipelineStep("observed GOES response", ("pipeline/build_paired_goes_response.py",)),
    PipelineStep("GOES magnetic local times from native orbits", ("pipeline/build_goes_local_time.py",)),
    PipelineStep("Figure 1 magnetopause context", ("pipeline/build_fig1_magnetopause.py",)),
    PipelineStep("Figure 1 geometry", ("figures/fig_geometry.py",)),
    PipelineStep("P1 GSM-component event reconstruction", ("pipeline/build_p1_goes_reconstruction.py",)),
    PipelineStep("P1 scalar-waveform event reconstruction", ("pipeline/build_p1_scalar_goes_reconstruction.py",)),
    PipelineStep("independent Solar-1 scalar-waveform reconstruction", ("pipeline/build_solar1_goes_reconstruction.py",)),
    PipelineStep("Figure 2 upstream stack", ("figures/fig_upstream_stack.py",)),
    PipelineStep("shock and sheath diagnostics", ("figures/fig_boundary_diagnostics.py",)),
    PipelineStep("ejecta recovery diagnostics", ("figures/ejecta_recovery_diagnostics.py",)),
    PipelineStep("Figure 3 aligned isolated waveforms", ("figures/fig_isolated_waveforms.py",)),
    PipelineStep("Figure 4 physical metrics", ("figures/fig_physical_structure_evolution.py",)),
    PipelineStep("Figure 5 spatial persistence", ("figures/fig_waveform_spatial_persistence.py",)),
    PipelineStep("Figure 6 GOES response", ("figures/fig_goes_magnetic_response.py",)),
    PipelineStep("Figure 7 P1 reconstruction", ("figures/fig_p1_goes_reconstruction.py",)),
    PipelineStep("Figure 8 P1 component contributions", ("figures/fig_p1_goes_component_contributions.py",)),
    PipelineStep("Supplementary Figure 1", ("figures/fig_supp1_extraction_intervals.py",)),
    PipelineStep("Supplementary Figure 2", ("figures/fig_supp2_lag_coherence.py",)),
    PipelineStep("Supplementary Figure 3", ("figures/fig_supp3_information_retention.py",)),
    PipelineStep("Supplementary Figure 4 P1 scalar reconstruction", ("figures/fig_supp4_p1_scalar_reconstruction.py",)),
    PipelineStep("Supplementary Figure 5 Solar-1 scalar reconstruction", ("figures/fig_supp5_solar1_reconstruction.py",)),
)

PUBLICATION_FIGURE_STEMS: tuple[str, ...] = (
    "Fig1_geometry",
    "Fig2_upstream_bt_stack",
    "Fig3_isolated_waveforms",
    "Fig4_physical_structure_evolution",
    "Fig5_waveform_coherence_length_scales",
    "Fig6_goes_magnetic_response",
    "Fig7_p1_goes_reconstruction",
    "Fig8_p1_goes_component_contributions",
    "SupFig1_extraction_intervals",
    "SupFig2_lag_coherence",
    "SupFig3_information_retention",
    "SupFig4_p1_scalar_reconstruction",
    "SupFig5_solar1_reconstruction",
)

DIAGNOSTIC_FIGURE_STEMS: tuple[str, ...] = (
    "ejecta_recovery",
    "sheath_ejecta_boundaries",
    "shock_boundaries",
    "overview_ace",
    "overview_artemis_p1",
    "overview_artemis_p2",
    "overview_dscovr",
    "overview_goes18",
    "overview_goes19",
    "overview_solar1",
    "overview_solar_orbiter",
    "overview_wind",
)


def _child_environment(data_directory: Path, output_directory: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PYTHON_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["L1_GEO_DATA_DIR"] = str(data_directory)
    environment["L1_GEO_OUTPUT_DIR"] = str(output_directory)
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(output_directory / ".cache" / "matplotlib")
    environment["XDG_CACHE_HOME"] = str(output_directory / ".cache")
    return environment


def run_steps(
    steps: Iterable[PipelineStep] = PUBLICATION_STEPS,
    *,
    data_directory: Path = DEFAULT_DATA_DIRECTORY,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> None:
    data_directory = Path(data_directory).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    if not data_directory.is_dir():
        raise FileNotFoundError(f"Mission-data directory does not exist: {data_directory}")
    if (data_directory == output_directory
            or data_directory.is_relative_to(output_directory)
            or output_directory.is_relative_to(data_directory)):
        raise ValueError("Output directory must be separate from the mission-data directory")
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
    environment = _child_environment(data_directory, output_directory)
    for index, step in enumerate(steps, start=1):
        print(f"[{index:02d}] {step.label}", flush=True)
        subprocess.run(
            (sys.executable, *step.argv),
            cwd=PYTHON_ROOT,
            env=environment,
            check=True,
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY,
                        help="supplied mission-data directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY,
                        help="directory for newly generated results and figures")
    args = parser.parse_args(argv)
    run_steps(data_directory=args.data_dir, output_directory=args.output_dir)
    output = args.output_dir.expanduser().resolve()
    missing = [
        directory / f"{stem}.pdf"
        for directory, stems in (
            (output / "figures", PUBLICATION_FIGURE_STEMS),
            (output / "figures" / "diag", DIAGNOSTIC_FIGURE_STEMS),
        )
        for stem in stems
        if not (directory / f"{stem}.pdf").is_file()
    ]
    if missing:
        raise RuntimeError("Missing figure outputs: " + ", ".join(map(str, missing)))
    print(f"Results: {output / 'results'}", flush=True)
    print(f"Prepared data: {output / 'processed'}", flush=True)
    print(f"Paper figures: {output / 'figures'} ({len(PUBLICATION_FIGURE_STEMS)} PDFs)", flush=True)
    print(f"Diagnostics: {output / 'figures' / 'diag'} ({len(DIAGNOSTIC_FIGURE_STEMS)} PDFs)", flush=True)


if __name__ == "__main__":
    main()
