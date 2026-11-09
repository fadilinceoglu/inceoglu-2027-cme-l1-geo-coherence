"""Rotate native DSCOVR GSE vectors to GSM at their exact observed UTC epochs.

SpacePy CTrans is used with IGRF14, no IRBEM, and no temporal approximation.
Supplied coefficient and UTC--TAI files are verified before being copied into
the current output directory's runtime cache. No network access or writes to
the user's global SpacePy configuration are performed.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

from common.io import atomic_output_paths, read_json, sha256 as _sha256
from config.paths import DATA_ROOT, OUTPUT_ROOT

REFERENCE_DIRECTORY = DATA_ROOT / "coordinate_reference"
SPACEPY_VERSION = "0.7.0"
REFERENCE_FILES = {"igrf14coeffs.txt": "igrfcoeffs.txt", "tai-utc.dat": "tai-utc.dat"}


def reference_manifest(directory: Path | str = REFERENCE_DIRECTORY) -> dict:
    """Fail on absent, changed or incomplete physical reference inputs."""
    directory = Path(directory).resolve(strict=True)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("spacepy_version") != SPACEPY_VERSION:
        raise ValueError("Unsupported GSM coordinate-reference manifest or SpacePy version")
    records = manifest.get("references", [])
    lookup = {record["filename"]: record for record in records}
    if len(records) != len(REFERENCE_FILES) or set(lookup) != set(REFERENCE_FILES):
        raise ValueError("GSM conversion requires supplied IGRF14 and UTC--TAI references")
    for name, runtime_name in REFERENCE_FILES.items():
        record = lookup[name]
        if record.get("runtime_filename") != runtime_name or not record.get("source_url"):
            raise ValueError(f"Invalid GSM reference provenance for {name}")
        if _sha256(directory / name) != record.get("sha256"):
            raise ValueError(f"GSM coordinate-reference checksum mismatch: {directory / name}")
    return manifest


@lru_cache(maxsize=4)
def _load_backend(reference_directory: str, output_directory: str):
    root = Path(reference_directory).resolve(strict=True)
    manifest = reference_manifest(root)
    # SpacePy rewrites its configuration on import. Separate process caches
    # prevent spawned workers from reading a configuration another is writing.
    runtime = Path(output_directory).resolve() / ".cache" / "spacepy" / str(os.getpid())
    data = runtime / ".spacepy" / "data"
    data.mkdir(parents=True, exist_ok=True)
    for record in manifest["references"]:
        destination = data / record["runtime_filename"]
        if not destination.is_file() or _sha256(destination) != record["sha256"]:
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            with atomic_output_paths((destination,), temporary_paths=(temporary,)) as (staged,):
                shutil.copyfile(root / record["filename"], staged)
        if _sha256(destination) != record["sha256"]:
            raise RuntimeError(f"GSM runtime reference checksum mismatch: {destination}")
    # Configuration is local to this run; freezing references replaces the
    # package's optional age warning and automatic-update advice.
    config = runtime / ".spacepy" / "spacepy.rc"
    if not config.is_file():
        temporary = config.with_name(f".{config.name}.{os.getpid()}.tmp")
        with atomic_output_paths((config,), temporary_paths=(temporary,)) as (staged,):
            staged.write_text("[spacepy]\nenable_old_data_warning = False\nsupport_notice = False\n")
    if "spacepy" in sys.modules:
        loaded = sys.modules["spacepy"]
        if Path(loaded.DOT_FLN).resolve() != runtime / ".spacepy":
            raise RuntimeError("SpacePy was initialized outside this run's verified coordinate cache")
    os.environ["SPACEPY"] = str(runtime)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UserWarning, module=r"spacepy(\..*)?")
        import spacepy
        from spacepy import coordinates as coord, time as spt, igrf
    if spacepy.__version__ != SPACEPY_VERSION:
        raise RuntimeError(f"GSM conversion requires SpacePy {SPACEPY_VERSION}; found {spacepy.__version__}")
    coord.DEFAULTS.set_values(use_irbem=False, itol=0)
    return coord, spt, igrf, manifest


def transformation_provenance(*, reference_directory: Path | str = REFERENCE_DIRECTORY,
                              output_directory: Path | str = OUTPUT_ROOT) -> dict:
    _, _, igrf, manifest = _load_backend(str(Path(reference_directory).resolve()),
                                         str(Path(output_directory).resolve()))
    return dict(
        method="SpacePy CTrans Cartesian GSE-to-GSM vector rotation",
        spacepy_version=SPACEPY_VERSION, input_frame="GSE", output_frame="GSM",
        backend="use_irbem=False; itol=0; exact native UTC epochs without rotation interpolation",
        geomagnetic_model="IGRF14 centered dipole from date-adjusted coefficients",
        model_valid_utc=[str(igrf.igrfcoeffs.datelow), str(igrf.igrfcoeffs.datehigh)],
        scalar_policy="supplied CDAWeb B1F1 and original quality masks retained independently",
        missing_policy="missing vectors remain missing; no NOAA source substitution or gap fill",
        UT1_policy="SpacePy CTrans default UT1-UTC=0; frozen supplied UTC--TAI reference",
        network_access=False, spacecraft_ephemeris_required=False,
        reference_inputs=manifest["references"],
    )


def _convert_chunk(task):
    vector, stamp_ns, reference_directory, output_directory = task
    coord, spt, igrf, _ = _load_backend(reference_directory, output_directory)
    utc = pd.to_datetime(stamp_ns, utc=True).tz_localize(None)
    if (utc[0].to_pydatetime() < igrf.igrfcoeffs.datelow
            or utc[-1].to_pydatetime() > igrf.igrfcoeffs.datehigh):
        raise ValueError("Native UTC epochs lie outside the supplied IGRF14 validity interval")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UserWarning, module=r"spacepy(\..*)?")
        result = coord.Coords(vector, "GSE", "car", ticks=spt.Ticktock(utc.to_pydatetime(), "UTC"),
                              use_irbem=False).convert("GSM", "car").data
    if not np.isfinite(result).all():
        raise RuntimeError("GSE-to-GSM conversion produced nonfinite components")
    scale = max(1., float(np.max(np.linalg.norm(vector, axis=1))))
    if np.max(abs(np.linalg.norm(result, axis=1) - np.linalg.norm(vector, axis=1))) > 1e-12 * scale:
        raise RuntimeError("GSE-to-GSM conversion failed vector-norm preservation")
    if np.max(abs(result[:, 0] - vector[:, 0])) > 1e-12 * scale:
        raise RuntimeError("GSE-to-GSM conversion changed the shared x component")
    return result


def gse_to_gsm(vectors, timestamps, *, reference_directory: Path | str = REFERENCE_DIRECTORY,
               output_directory: Path | str = OUTPUT_ROOT, workers: int = 4,
               chunk_size: int = 2000) -> np.ndarray:
    """Rotate finite vectors at native epochs; execution chunks do not average data.

    Scripts using multiple workers require the standard multiprocessing main
    guard. Partial/missing native vectors remain NaN, in their original rows.
    """
    vector = np.asarray(vectors, dtype=np.float64)
    utc = pd.DatetimeIndex(timestamps)
    if vector.ndim != 2 or vector.shape[1] != 3 or len(utc) != len(vector):
        raise ValueError("Expected matching (n,3) vectors and n UTC timestamps")
    if utc.tz is None or utc.hasnans or utc.has_duplicates or not utc.is_monotonic_increasing:
        raise ValueError("Expected unique, ordered, timezone-aware UTC timestamps")
    utc = utc.tz_convert("UTC").as_unit("ns")
    if np.any(utc.asi8 % 1000):
        raise ValueError("Submicrosecond epochs require a higher-precision time interface")
    if workers < 1 or chunk_size < 1:
        raise ValueError("Execution workers and chunk_size must be positive")
    root = str(Path(reference_directory).resolve(strict=True))
    output = str(Path(output_directory).resolve())
    _load_backend(root, output)
    selected = np.flatnonzero(np.isfinite(vector).all(axis=1))
    result = np.full_like(vector, np.nan)
    if not len(selected):
        return result
    tasks = [(vector[ix], utc.asi8[ix], root, output)
             for ix in (selected[start:start + chunk_size]
                        for start in range(0, len(selected), chunk_size))]
    if workers == 1 or len(tasks) == 1:
        converted = map(_convert_chunk, tasks)
        for start, block in zip(range(0, len(selected), chunk_size), converted):
            result[selected[start:start + len(block)]] = block
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            converted = pool.map(_convert_chunk, tasks, chunksize=1)
            for start, block in zip(range(0, len(selected), chunk_size), converted):
                result[selected[start:start + len(block)]] = block
    return result
