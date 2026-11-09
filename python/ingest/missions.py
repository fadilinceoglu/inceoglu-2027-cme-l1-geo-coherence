"""Mission inventories and screened records used by every publication stage.

Daily files are selected once, before decoding. Each mission's quality rule is
applied here so boundary means and prepared waveform grids share observations.
Provider minute magnetic products are retained only for overview medians.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path

import cdflib
import netCDF4
import numpy as np
import pandas as pd

from config import study
from config.constants import BOLTZMANN_J_K, EPOCH_1958, PROTON_MASS_KG
from config.paths import DATA_ROOT
from ingest.artemis import read_artemis_fgm, read_artemis_esa_ions
from ingest.solar_orbiter import read_solar_orbiter_mag, read_solar_orbiter_swa_pas
from ingest.gsm_coordinates import gse_to_gsm, transformation_provenance

MAG_COLUMNS = ("bx", "by", "bz", "bt")
PLASMA_COLUMNS = ("n", "v", "t")
ARTEMIS_PROBES = {"artemis_p1": "b", "artemis_p2": "c"}
ARTEMIS_EVENT_DAYS = tuple(study.FETCH_DAYS_DOWNSTREAM)
ARTEMIS_FGM_DAYS = (*ARTEMIS_EVENT_DAYS, "20260122")
TEMPERATURE_UNITS = {key: "eV" if key in ("solar_orbiter", *ARTEMIS_PROBES) else "K"
                     for key in study.SATELLITES}


@dataclass
class Observations:
    magnetic: pd.DataFrame
    plasma: pd.DataFrame
    files: list[Path]
    quality_policy: str
    overview_magnetic: pd.DataFrame | None = None


def magnetic_input_variables(craft: str) -> dict:
    """Provider channels supplying the generic bx/by/bz/bt product columns."""
    if craft == "solar_orbiter":
        return {"vector_frame": "RTN", "vector": "B_RTN", "scalar": "norm(native RTN vector)"}
    if craft == "ace":
        return {"vector_frame": "GSM", "vector": "BGSM", "scalar": "norm(BGSEc)"}
    if craft == "wind":
        return {"vector_frame": "GSM", "vector": "B3GSM", "scalar": "B3F1",
                "overview_vector": "BGSM", "overview_scalar": "BF1"}
    if craft == "dscovr":
        return {"vector_frame": "GSM", "native_vector": "B1GSE", "scalar": "B1F1",
                "product": "CDAWeb DSCOVR definitive MAG",
                "vector_transform": "SpacePy CTrans GSE-to-GSM at each native UTC epoch; IGRF14"}
    if craft == "swfo":
        return {"vector_frame": "GSM", "vector": "b_gsm_sec", "scalar": "norm(b_gse_sec)",
                "overview_vector": "b_gsm_min", "overview_scalar": "norm(b_gse_min)"}
    if craft in ARTEMIS_PROBES:
        prefix = f"th{ARTEMIS_PROBES[craft]}_fgs"
        return {"vector_frame": "GSM", "vector": f"{prefix}_gsm", "scalar": f"norm({prefix}_gse)"}
    if craft in study.GOES_SPACECRAFT:
        return {"vector_frame": "GSM", "vector": "b_gsm", "scalar": "b_total"}
    raise ValueError(f"Unknown magnetometer: {craft}")


def daily_files(directory: Path, patterns: tuple[str, ...], days) -> list[Path]:
    """One latest local version per day; alternative patterns are optional.

    Pattern order is preference order. All declared days must be present;
    an absent selected source never falls back to a different mission product.
    """
    selected = []
    for day in days:
        for pattern in patterns:
            hits = sorted(p for p in directory.glob(pattern.format(day=day))
                          if p.is_file() and p.stat().st_size > 0)
            if hits:
                selected.append(hits[-1])
                break
        else:
            raise FileNotFoundError(f"Supply mission input for {day} under {directory}: {patterns}")
    return selected


def inventory(craft: str, root: Path = DATA_ROOT) -> tuple[list[Path], list[Path]]:
    upstream = study.FETCH_DAYS_UPSTREAM
    l1 = root / "raw_event_20260119/l1"
    if craft == "solar_orbiter":
        directory = root / "raw_event_20260119/solar_orbiter"
        return (daily_files(directory, ("solar_mag_{day}.cdf",), upstream),
                daily_files(directory, ("solar_pas_{day}.cdf",), upstream))
    if craft == "ace":
        return (daily_files(l1, ("ac_h3_mfi_{day}_v*.cdf",), upstream),
                daily_files(root / "ace", ("{day}_ace_swepam_1m.txt",), upstream))
    if craft == "wind":
        monthly = root / "wind/wind_h1s_swe_20260101000132_20260131235813_cdaweb.cdf"
        plasma = ([monthly] if monthly.is_file() and monthly.stat().st_size else
                  daily_files(root / "wind", ("wi_h1_swe_{day}_v*.cdf",), upstream))
        return daily_files(l1, ("wi_h0_mfi_{day}_v*.cdf",), upstream), plasma
    if craft == "dscovr":
        return (daily_files(l1, ("dscovr_h0s_mag_{day}*.cdf", "dscovr_h0_mag_{day}_v*.cdf"), upstream),
                daily_files(l1, ("oe_f1m_dscovr_s{day}*.nc.gz",), upstream))
    if craft == "swfo":
        return (daily_files(root / "swfo_mag_l3", ("sci_mag-l3_solar1_s{day}T*.nc",), upstream),
                daily_files(root / "swfo_swips_l2", ("sci_swips-l2_*_s{day}T*.nc", "sci_swips-l2_s{day}T*.nc"), upstream))
    if craft in ARTEMIS_PROBES:
        probe = ARTEMIS_PROBES[craft]
        directory = root / "raw_event_20260119/earth_moon"
        if probe == "c":
            directory /= "artemis_p2"
        return (daily_files(directory, (f"th{probe}_l2_fgm_{{day}}_v*.cdf",), ARTEMIS_FGM_DAYS),
                daily_files(directory, (f"th{probe}_l2_esa_{{day}}_v*.cdf",), ARTEMIS_EVENT_DAYS))
    if craft.startswith("goes"):
        short = "g" + craft[-2:]
        return (daily_files(root / "raw_event_20260119/scale_effects/goes_hires",
                            (f"dn_magn-l2-hires_{short}_d{{day}}_v*.nc",), study.FETCH_DAYS_DOWNSTREAM), [])
    raise ValueError(f"Unknown spacecraft: {craft}")


def _combine(frames):
    result = pd.concat(frames).sort_index()
    return result[~result.index.duplicated(keep="last")]


def _cdf_time(cdf, variable):
    return pd.DatetimeIndex(pd.to_datetime(cdflib.cdfepoch.to_datetime(cdf.varget(variable)), utc=True))


def _mag_frame(vector, index, magnitude=None):
    frame = pd.DataFrame(vector, index=index, columns=MAG_COLUMNS[:3])
    frame["bt"] = np.linalg.norm(vector, axis=1) if magnitude is None else magnitude
    return frame


def _l1_magnetic(craft, paths, *, coordinate_reference_directory=None):
    parts, overviews = [], []
    conversion_options = ({} if coordinate_reference_directory is None else
                          {"reference_directory": coordinate_reference_directory})
    for path in paths:
        cdf = cdflib.CDF(str(path))
        if craft == "ace":
            b = np.asarray(cdf.varget("BGSM"), dtype=float)
            scalar_source = np.asarray(cdf.varget("BGSEc"), dtype=float)
            b[np.abs(b) > 9000] = np.nan
            scalar_source[np.abs(scalar_source) > 9000] = np.nan
            rejected = np.asarray(cdf.varget("Q_FLAG")).reshape(-1) != 0
            b[rejected] = scalar_source[rejected] = np.nan
            # Preserve the original coordinate-independent scalar input; do not
            # inject provider transformation rounding into the scalar branch.
            frame = _mag_frame(b, _cdf_time(cdf, "Epoch"), np.linalg.norm(scalar_source, axis=1))
        elif craft == "wind":
            def magnetic(epoch, vector, total):
                b = np.asarray(cdf.varget(vector), dtype=float)
                bt = np.asarray(cdf.varget(total), dtype=float)
                b[np.abs(b) > 1e4] = np.nan
                bt[(bt < 0) | (bt > 1e4)] = np.nan
                result = _mag_frame(b, _cdf_time(cdf, epoch), bt)
                result.loc[result.bt > 500] = np.nan
                return result
            frame = magnetic("Epoch3", "B3GSM", "B3F1")
            overviews.append(magnetic("Epoch", "BGSM", "BF1"))
        elif craft == "dscovr":
            frame = _mag_frame(np.asarray(cdf.varget("B1GSE"), dtype=float),
                               _cdf_time(cdf, "Epoch1"), np.asarray(cdf.varget("B1F1"), dtype=float))
            frame.loc[np.asarray(cdf.varget("FLAG1")) != 0] = np.nan
            frame[frame.abs() > 500] = np.nan
        else:
            raise ValueError(f"Unknown L1 magnetometer: {craft}")
        frame.loc[frame.bt > 500] = np.nan
        if craft == "dscovr":
            frame.loc[:, list(MAG_COLUMNS[:3])] = gse_to_gsm(
                frame.loc[:, list(MAG_COLUMNS[:3])].to_numpy(float), frame.index, **conversion_options)
        parts.append(frame)
    result = _combine(parts)
    if craft == "dscovr":
        result.attrs["coordinate_transform"] = transformation_provenance(**conversion_options)
    return result, _combine(overviews) if overviews else None


def _ace_plasma(paths):
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            f = line.split()
            if len(f) == 10 and f[0].isdigit():
                stamp = pd.Timestamp(year=int(f[0]), month=int(f[1]), day=int(f[2]),
                                     hour=int(f[3][:2]), minute=int(f[3][2:]), tz="UTC")
                rows.append((stamp, int(f[6]), float(f[7]), float(f[8]), float(f[9])))
    frame = pd.DataFrame(rows, columns=["time", "status", "n", "v", "t"]).set_index("time")
    frame.loc[frame.status != 0, list(PLASMA_COLUMNS)] = np.nan
    return frame.loc[:, list(PLASMA_COLUMNS)]


def _wind_plasma(paths):
    parts = []
    for path in paths:
        cdf = cdflib.CDF(str(path))
        def clean(name):
            values = np.asarray(cdf.varget(name), dtype=float)
            values[np.abs(values) >= 99999] = np.nan
            return values
        v = np.sqrt(sum(clean(f"Proton_V{axis}_moment") ** 2 for axis in "XYZ"))
        parts.append(pd.DataFrame({"n": clean("Proton_Np_moment"), "v": v,
                                   "t": PROTON_MASS_KG * (clean("Proton_W_moment") * 1e3) ** 2 / (2 * BOLTZMANN_J_K)},
                                  index=_cdf_time(cdf, "Epoch")))
    frame = _combine(parts)
    return frame.loc[(frame.index >= pd.Timestamp(study.FETCH_DAYS_UPSTREAM[0], tz="UTC"))
                     & (frame.index < pd.Timestamp(study.FILTER_CONTEXT_STOP))]


def _dscovr_plasma(paths):
    parts = []
    for path in paths:
        with gzip.open(path, "rb") as source:
            payload = source.read()
        with netCDF4.Dataset("inmemory.nc", memory=payload) as dataset:
            index = pd.to_datetime(np.asarray(dataset.variables["time"][:], dtype="int64"), unit="ms", utc=True)
            parts.append(pd.DataFrame({key: np.ma.filled(np.ma.masked_invalid(dataset.variables[name][:]), np.nan).astype(float)
                                       for key, name in zip(PLASMA_COLUMNS, ("proton_density", "proton_speed", "proton_temperature"))}, index=index))
    return _combine(parts)


def read_corrected_swips_moments(path):
    """Read SWiPS L2 proton moments in physical units.

    The provider's ``backfill`` flag describes delivery status, not science
    quality, so it does not mask moments. Required variables, units, shapes and
    timestamps are checked before the moments enter the analysis.
    """
    with netCDF4.Dataset(path) as dataset:
        required = {"sweep_mid_time", "proton_n", "proton_v_gse", "proton_t"}
        missing = required.difference(dataset.variables)
        if missing:
            raise ValueError(f"corrected SWiPS variables missing from {path}: {sorted(missing)}")
        for name, units in {"proton_n": "#/cm^3", "proton_v_gse": "km/s", "proton_t": "K"}.items():
            if getattr(dataset.variables[name], "units", None) != units:
                raise ValueError(f"unexpected corrected SWiPS units for {name} in {path}; expected {units}")
        time_variable = dataset.variables["sweep_mid_time"]
        units = str(getattr(time_variable, "units", "")).lower()
        if "microseconds since 1958-01-01" not in units:
            raise ValueError(f"unexpected corrected SWiPS time units in {path}")
        raw_times = np.ma.asarray(time_variable[:])
        if raw_times.ndim != 1 or np.ma.getmaskarray(raw_times).any() or not np.isfinite(raw_times).all():
            raise ValueError(f"corrected SWiPS timestamps must be a finite unmasked vector in {path}")
        index = pd.DatetimeIndex(EPOCH_1958 + pd.to_timedelta(np.asarray(raw_times, dtype="int64"), unit="us"))
        if index.empty or index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError(f"corrected SWiPS timestamps must be nonempty, unique and ordered in {path}")
        density, vector, temperature = [
            np.ma.filled(dataset.variables[name][:], np.nan).astype(float)
            for name in ("proton_n", "proton_v_gse", "proton_t")
        ]
        if density.shape != (len(index),) or temperature.shape != (len(index),) or vector.shape != (len(index), 3):
            raise ValueError(f"inconsistent corrected SWiPS moment shapes in {path}")
        return pd.DataFrame({"n": density, "v": np.linalg.norm(vector, axis=1), "t": temperature}, index=index)


def _swfo_records(mag_paths, plasma_paths):
    magnetic, overview, plasma = [], [], []
    for path in mag_paths:
        with netCDF4.Dataset(path) as dataset:
            def field(time, vector, scalar_vector):
                index = EPOCH_1958 + pd.to_timedelta(dataset.variables[time][:].filled(0).astype("int64"), unit="us")
                scalar = np.linalg.norm(dataset.variables[scalar_vector][:].filled(np.nan), axis=1)
                result = _mag_frame(dataset.variables[vector][:].filled(np.nan), index, scalar)
                result.loc[result.bt > 500] = np.nan
                return result
            frame = field("time_sec", "b_gsm_sec", "b_gse_sec")
            frame.loc[dataset.variables["flags_summary"][:].filled(0).astype(int) != 1] = np.nan
            magnetic.append(frame)
            overview.append(field("time_min", "b_gsm_min", "b_gse_min"))
    for path in plasma_paths:
        plasma.append(read_corrected_swips_moments(path))
    return _combine(magnetic), _combine(plasma), _combine(overview)


def load(craft: str, root: Path = DATA_ROOT) -> Observations:
    mag_paths, plasma_paths = inventory(craft, root)
    overview = None
    if craft == "solar_orbiter":
        raw = read_solar_orbiter_mag(mag_paths)
        magnetic = raw[["br_rtn", "bt_rtn", "bn_rtn", "bt"]].rename(columns={"br_rtn": "bx", "bt_rtn": "by", "bn_rtn": "bz"})
        raw = read_solar_orbiter_swa_pas(plasma_paths)
        plasma = raw[["density", "speed", "temperature"]].rename(columns={"density": "n", "speed": "v", "temperature": "t"})
        policy = "CDF fill/range screening; Solar Orbiter quality flags retained"
    elif craft in ("ace", "wind", "dscovr"):
        magnetic, overview = _l1_magnetic(craft, mag_paths,
            coordinate_reference_directory=Path(root) / "coordinate_reference" if craft == "dscovr" else None)
        plasma = {"ace": _ace_plasma, "wind": _wind_plasma, "dscovr": _dscovr_plasma}[craft](plasma_paths)
        policy = {"ace": "native BGSM vector; scalar norm(BGSEc) retained; Q_FLAG==0; fill/range screening",
                  "wind": "native B3GSM/BGSM vectors and unchanged B3F1/BF1 scalar; native 3-s fill/range screening",
                  "dscovr": "original CDAWeb B1GSE rotated at native Epoch1 to GSM using SpacePy CTrans/IGRF14; B1F1 retained; FLAG1==0 and original fill/range screening; no source substitution or long-gap fill"}[craft]
    elif craft == "swfo":
        magnetic, plasma, overview = _swfo_records(mag_paths, plasma_paths)
        policy = "native MAG b_gsm_sec/min vector; scalar norm(b_gse_sec/min) retained; flags_summary==1; corrected SWiPS proton_n/proton_v_gse/proton_t; fill/range screening; backfill is delivery status"
    elif craft in ARTEMIS_PROBES:
        probe = ARTEMIS_PROBES[craft]
        raw = read_artemis_fgm(mag_paths, probe=probe)
        magnetic = raw[["bx_gsm", "by_gsm", "bz_gsm", "bt"]].rename(columns={"bx_gsm": "bx", "by_gsm": "by", "bz_gsm": "bz"})
        magnetic.loc[magnetic.bt > 500] = np.nan
        reduced = read_artemis_esa_ions(plasma_paths, probe=probe, mode="peir")
        full = read_artemis_esa_ions(plasma_paths, probe=probe, mode="peif")
        speed = pd.concat([reduced.speed.dropna(), full.speed.dropna()]).sort_index()
        speed = speed[~speed.index.duplicated(keep="first")]
        plasma = reduced[["density", "temperature"]].rename(columns={"density": "n", "temperature": "t"}).join(speed.rename("v"), how="outer")
        policy = "native FGS GSM vector; scalar norm(native GSE vector) retained; FGS quality retained; |B| >500 nT rejected; ESA reduced/full speed merged"
    else:
        raise ValueError("GOES high-rate records are reduced daily by build_observations")
    plasma = _combine([plasma])
    plasma.loc[(plasma.n <= 0) | (plasma.n > (1e3 if craft in ("ace", "wind", "dscovr") else 1e4)), "n"] = np.nan
    plasma.loc[(plasma.v <= 0) | (plasma.v > 3000), "v"] = np.nan
    plasma.loc[plasma.t <= 0, "t"] = np.nan
    return Observations(magnetic, plasma, [*mag_paths, *plasma_paths], policy, overview)
