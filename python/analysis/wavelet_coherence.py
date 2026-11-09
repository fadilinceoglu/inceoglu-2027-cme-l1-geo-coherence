"""Event-specific, shock-registered scalar wavelet coherence and persistence.

The input is the exact prepared one-second |B| record and 0.04-R_E mapping
used to measure and filter the waveform families. Morlet coherence of that
record is summarized within the measured family wavelength bands. A single
physical auto-power field from the four L1 observers weights every pair.

Resampling ranges are conditional on the observed maps, activity field and
shock registration. They are not calibrated confidence intervals for other
events, nor do they include measurement or arrival-time uncertainty.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pycwt
from pycwt.helpers import rect
from scipy import fft
from scipy.ndimage import convolve1d
from scipy.optimize import minimize_scalar
from scipy.signal import correlate, resample_poly

from analysis.product_io import json_ready
from common.io import read_json, sha256
from analysis.waveform_scales import load_band_design
from analysis.waveform_comparisons import CRAFT_ORDER, PAIR_ORDER, EPHEMERIS_CACHE, REFERENCE_EPOCH, REPORTED_EARTH_RADIUS_KM, load_positions, pair_geometry
from config import study
from config.paths import OUTPUT_ROOT, PROCESSED_DIR, RESULTS_DIR
from config.waveform_bands import SPATIAL_DX_RE, WAVEFORM_FAMILIES
from pipeline.build_spatial_filters import _gap_tolerance_re, _prepare_waveform

SCHEMA_VERSION = 1
BOOTSTRAP_DRAWS = 1000
WHITE_DRAWS = 20
SENSITIVITY_DRAWS = 300
RESULT_PATH = RESULTS_DIR / "wavelet_coherence.json"
MOTHER = pycwt.Morlet(6)


@dataclass(frozen=True)
class Method:
    """Chosen numerical settings, kept separate from measured input quantities."""
    scales_per_octave: int = 12
    samples_per_minimum_wavelength: int = 12
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    white_draws: int = WHITE_DRAWS
    sensitivity_draws: int = SENSITIVITY_DRAWS
    seed: int = 20261016

    @property
    def dj(self) -> float:
        return 1.0 / self.scales_per_octave

    def validate(self) -> None:
        if min(self.scales_per_octave, self.samples_per_minimum_wavelength,
               self.bootstrap_draws, self.white_draws) < 1 or self.sensitivity_draws < 0:
            raise ValueError("Wavelet numerical settings require positive counts")


def _input_record(path: Path | str) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def prepare_sources() -> tuple[dict, dict, list[dict]]:
    """Reprepare current sources; never substitute a saved coherence product."""
    design_path = PROCESSED_DIR / "waveform_bands.json"
    boundary_path = RESULTS_DIR / "boundaries.json"
    design = load_band_design(design_path)
    boundaries = read_json(boundary_path)
    if set(design["bands"]) != set(WAVEFORM_FAMILIES):
        raise ValueError("Measured waveform design must contain the four families")
    inputs = [_input_record(design_path), _input_record(boundary_path)]
    sources = {}
    for craft in CRAFT_ORDER:
        prepared = _prepare_waveform(craft)
        time_x, x, values = prepared.spatial_record
        shock = pd.Timestamp(boundaries[craft]["shock"])
        elapsed = (prepared.frame.index - prepared.frame.index[0]).total_seconds().to_numpy()
        shock_elapsed = (shock - prepared.frame.index[0]).total_seconds()
        if not elapsed[0] <= shock_elapsed <= elapsed[-1]:
            raise ValueError(f"{craft}: measured shock outside prepared context")
        offset = float(np.interp(shock_elapsed, elapsed, time_x))
        time_x, x = time_x - offset, x - offset
        science = [float(np.interp((pd.Timestamp(stamp) - prepared.frame.index[0]).total_seconds(), elapsed, time_x))
                   for stamp in (study.ANALYSIS_START, study.ANALYSIS_STOP)]
        bad = ~np.isfinite(prepared.frame.bt.to_numpy(float))
        edges = np.flatnonzero(np.diff(np.r_[False, bad, False])).reshape(-1, 2)
        gaps = np.asarray([[time_x[max(0, a - 1)], time_x[min(len(time_x) - 1, b)]]
                           for a, b in edges], float).reshape(-1, 2)
        sources[craft] = dict(x_re=x.copy(), bt=values[:, 3].copy(),
                              scientific_bounds_re=science, gaps_re=gaps,
                              shock_utc=shock.isoformat())
        inputs.extend([dict(craft=craft, role="prepared_one_second_magnetic", **_input_record(prepared.source_path)),
                       dict(craft=craft, role="prepared_speed", **_input_record(prepared.speed_path))])
    return sources, design, inputs


def _distance_to_gaps(x: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    nearest = np.full(len(x), np.inf)
    for left, right in gaps:
        nearest = np.minimum(nearest, np.maximum(np.maximum(left - x, x - right), 0))
    return nearest


def make_grid(sources: dict, definition: dict, method: Method) -> dict:
    """Anti-alias the shared native spatial grid and derive all support limits."""
    lower = float(definition["min_re"])
    upper = definition["max_re"]
    context_left = max(float(item["x_re"][0]) for item in sources.values())
    context_right = min(float(item["x_re"][-1]) for item in sources.values())
    science_left = max(float(item["scientific_bounds_re"][0]) for item in sources.values())
    science_right = min(float(item["scientific_bounds_re"][1]) for item in sources.values())
    if not context_left < science_left < science_right < context_right:
        raise ValueError("No common scientific interval with real surrounding context")
    if upper is None:
        upper = min((science_right - science_left) / 2,
                    (context_right - context_left) * MOTHER.flambda() * MOTHER.coi() / 2 * (1 - 1e-6))
    upper = float(upper)
    if not 0 < lower < upper:
        raise ValueError("No supported finite wavelength interval")
    decimation = max(1, int(np.floor(lower / (method.samples_per_minimum_wavelength * SPATIAL_DX_RE))))
    dx = decimation * SPATIAL_DX_RE
    left = np.ceil(context_left / SPATIAL_DX_RE) * SPATIAL_DX_RE
    native_x = left + np.arange(int(np.floor((context_right - left) / SPATIAL_DX_RE)) + 1) * SPATIAL_DX_RE
    x = native_x[::decimation]
    # Extra scales protect the retained band from scale-smoothing edge effects.
    periods = (lower / 2) * 2 ** (np.arange(int(np.ceil(np.log2(4 * upper / lower) / method.dj)) + 1) * method.dj)
    keep = (periods >= lower * (1 - 1e-10)) & (periods < upper * (1 + 1e-10))
    scales = periods / MOTHER.flambda()
    science = (x >= science_left) & (x < science_right)
    coi = (len(x) / 2 - np.abs(np.arange(len(x)) - (len(x) - 1) / 2)) * dx * MOTHER.flambda() * MOTHER.coi()
    arrays, norms, masks = {}, {}, []
    for craft, source in sources.items():
        values = np.interp(native_x, source["x_re"], source["bt"])
        if not np.isfinite(values).all():
            raise ValueError(f"{craft}: unresolved prepared spatial interpolation")
        sampled = resample_poly(values, up=1, down=decimation, padtype="line") if decimation > 1 else values
        if sampled.shape != x.shape or not science.any():
            raise ValueError("Anti-alias coordinate/sample mismatch")
        center, std = float(sampled[science].mean()), float(sampled[science].std())
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"{craft}: zero or invalid scalar variation")
        arrays[craft] = (sampled - center) / std
        tolerance = _gap_tolerance_re(lower)
        gaps = source["gaps_re"]
        material = gaps[np.diff(gaps, axis=1).ravel() > tolerance]
        gap_coi = _distance_to_gaps(x, material) * MOTHER.flambda() * MOTHER.coi()
        masks.append(((periods[keep, None] <= np.minimum(coi, gap_coi)[None, :]) & science[None, :]).all(axis=0))
        norms[craft] = dict(spatial_mean_nt=center, spatial_std_nt=std,
                            material_gap_count=len(material), material_gap_tolerance_re=tolerance)
    common = np.logical_and.reduce(masks)
    if not common.any():
        raise ValueError("No complete-band cone-of-influence support shared by all observers")
    return dict(x=x, arrays=arrays, scales=scales, periods=periods, keep=keep,
                dx=dx, science=science, common=common, norms=norms,
                scientific_bounds_re=[science_left, science_right],
                context_bounds_re=[context_left, context_right], decimation=decimation,
                finite_maximum_wavelength_re=upper)


def scale_smooth(values: np.ndarray, dj: float) -> np.ndarray:
    window = rect(int(np.round(MOTHER.deltaj0 / dj * 2)), normalize=True)
    return convolve1d(values, window, axis=0, mode="constant", cval=0,
                      origin=-1 if len(window) % 2 == 0 else 0)


def cwt_rows(values: np.ndarray, dx: float, scales: np.ndarray,
             path: Path | None = None, *, dtype="complex64") -> np.ndarray:
    """Rowwise PyCWT Morlet transform; a memmap bounds peak resident memory."""
    count = len(values)
    fft_count = 2 ** int(np.ceil(np.log2(count)))
    omega = 2 * np.pi * fft.fftfreq(fft_count, dx)
    signal_fft = fft.fft(values, n=fft_count)
    shape = (len(scales), count)
    output = np.empty(shape, dtype=dtype) if path is None else np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    for index, scale in enumerate(scales):
        kernel = (scale * omega[1] * fft_count) ** .5 * np.conj(MOTHER.psi_ft(scale * omega))
        output[index] = fft.ifft(signal_fft * kernel, n=fft_count)[:count]
    if isinstance(output, np.memmap):
        output.flush()
    return output


def _gaussian_row(row: np.ndarray, dx: float, scale: float) -> np.ndarray:
    count = len(row)
    fft_count = 2 ** int(np.ceil(np.log2(count)))
    omega = 2 * np.pi * fft.fftfreq(fft_count)
    result = fft.ifft(fft.fft(row, n=fft_count) * np.exp(-.5 * (scale / dx) ** 2 * omega ** 2), n=fft_count)[:count]
    return result if np.iscomplexobj(row) else result.real


def autosmooth(wave: np.ndarray, dx: float, scales: np.ndarray, dj: float) -> np.ndarray:
    output = np.empty(wave.shape, dtype="float32")
    for index, scale in enumerate(scales):
        output[index] = _gaussian_row(abs(wave[index].astype("complex128")) ** 2 / scale, dx, scale)
    return scale_smooth(output, dj)


def wct_pair(left: np.ndarray, right: np.ndarray, left_auto: np.ndarray,
             right_auto: np.ndarray, dx: float, scales: np.ndarray,
             keep: np.ndarray, dj: float) -> tuple[np.ndarray, np.ndarray]:
    cross = np.empty(left.shape, dtype="complex64")
    for index, scale in enumerate(scales):
        cross[index] = _gaussian_row(left[index].astype("complex128") * np.conj(right[index]) / scale, dx, scale)
    cross = scale_smooth(cross, dj)[keep]
    denominator = left_auto[keep].astype(float) * right_auto[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.abs(cross.astype("complex128")) ** 2 / denominator
    return np.clip(r2, 0, 1), np.angle(cross)


def common_activity(autos: dict, norms: dict, keep: np.ndarray) -> np.ndarray:
    """Restore physical auto-power before pooling the four reference observers."""
    return sum(np.maximum(autos[craft][keep].astype(float), 0) * norms[craft]["spatial_std_nt"] ** 2
               for craft in study.L1_SPACECRAFT) / len(study.L1_SPACECRAFT)


def weighted_profiles(r2: np.ndarray, activity: np.ndarray, common: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if r2.shape != activity.shape or common.shape != (r2.shape[1],):
        raise ValueError("Coherence, common activity and support have incompatible shapes")
    numerator = np.where(common, np.sum(activity * r2, axis=0), 0.)
    denominator = np.where(common, np.sum(activity, axis=0), 0.)
    if not (np.isfinite(numerator).all() and np.isfinite(denominator).all()) or denominator.sum() <= 0:
        raise ValueError("Invalid activity-weighted coherence support")
    return numerator, denominator


def weighted_block_means(numerator: np.ndarray, denominator: np.ndarray,
                         block: int, draws: int, seed: int) -> np.ndarray:
    """One noncircular position resample shared by every pair and both sums."""
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    if numerator.ndim != 2 or denominator.shape != (numerator.shape[1],):
        raise ValueError("Expected pair numerator profiles and one common denominator")
    if not (np.isfinite(numerator).all() and np.isfinite(denominator).all()) or np.any(denominator < 0):
        raise ValueError("Profiles must zero unsupported positions and contain finite weights")
    count = numerator.shape[1]
    block = min(count, max(1, int(block)))
    full, tail = divmod(count, block)
    ns = np.pad(np.cumsum(numerator, axis=1, dtype=float), ((0, 0), (1, 0)))
    ds = np.r_[0., np.cumsum(denominator, dtype=float)]
    rng = np.random.default_rng(seed)
    output = np.full((draws, len(numerator)), np.nan)
    for draw in range(draws):
        starts = rng.integers(0, count - block + 1, size=full)
        numer = (ns[:, starts + block] - ns[:, starts]).sum(axis=1)
        denom = float((ds[starts + block] - ds[starts]).sum())
        if tail:
            start = int(rng.integers(0, count - block + 1))
            numer += ns[:, start + tail] - ns[:, start]
            denom += float(ds[start + tail] - ds[start])
        if denom > 0:
            output[draw] = numer / denom
    return output


def acf_efold(profile: np.ndarray, dx: float) -> float:
    finite = np.isfinite(profile)
    values = np.where(finite, profile, 0.).astype(float)
    if finite.sum() < 3:
        return 0.
    values[finite] -= values[finite].mean()
    if np.dot(values, values) <= np.finfo(float).eps * max(1, int(finite.sum())):
        return 0.
    count = len(values)
    numerator = correlate(values, values, mode="full", method="fft")[count - 1:count - 1 + count // 2 + 1]
    pairs = correlate(finite.astype(float), finite.astype(float), mode="full", method="fft")[count - 1:count - 1 + count // 2 + 1]
    acf = np.divide(numerator, pairs, out=np.zeros_like(numerator), where=pairs > .5)
    acf /= acf[0]
    found = np.flatnonzero((acf <= np.exp(-1)) & (np.arange(len(acf)) > 0))
    return float((found[0] if len(found) else len(acf) - 1) * dx)


def predict(distance: np.ndarray, rate: float, floor: float = 0.) -> np.ndarray:
    distance = np.asarray(distance, float)
    if np.isposinf(rate):
        return np.full(distance.shape, floor)
    return floor + (1 - floor) * np.exp(-distance * rate)


def fit_rate(distance: np.ndarray, values: np.ndarray, floor: float = 0.) -> tuple[float, float, float, float]:
    """Fit coherence-space least squares with L=0 and L=infinity endpoints."""
    distance, values = np.asarray(distance, float), np.asarray(values, float)
    if not (np.isfinite(distance).all() and np.isfinite(values).all()) or np.any(distance <= 0):
        raise ValueError("Decay fit requires finite coherence and positive distances")
    median = float(np.median(distance))
    def objective(u):
        return float(np.mean((values - predict(distance, u / median, floor)) ** 2))
    solution = minimize_scalar(objective, bounds=(0, 40 * median / float(distance.min())), method="bounded", options={"xatol": 1e-11})
    candidates = [(float(solution.fun), float(solution.x) / median), (objective(0.), 0.),
                  (float(np.mean((values - floor) ** 2)), np.inf)]
    loss, rate = min(candidates, key=lambda item: item[0])
    length = np.inf if rate == 0 else (0. if np.isinf(rate) else 1 / rate)
    return rate, length, float(np.sqrt(loss)), float(np.mean(abs(values - predict(distance, rate, floor))))


def _quantiles(values: np.ndarray) -> list[float]:
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    return np.quantile(values, [.025, .5, .975], method="inverted_cdf").tolist() if len(values) else [np.nan] * 3


def _length_status(lengths: np.ndarray, supported: bool, dmin: float, dmax: float) -> str:
    if not supported:
        return "no_supported_decay_length"
    if np.mean(lengths > dmax) >= .95:
        return "beyond_sampled_range"
    if np.mean(lengths < dmin) >= .95:
        return "below_sampled_range"
    if np.mean((lengths >= dmin) & (lengths <= dmax)) >= .95:
        return "within_sampled_range"
    return "unresolved"


def _activity_concentration(denominator: np.ndarray, block: int) -> list[dict]:
    count = len(denominator)
    cumulative = np.r_[0., np.cumsum(denominator)]
    result = []
    for offset in np.unique([0, block // 4, block // 2, 3 * block // 4]):
        edges = np.unique(np.r_[0, np.arange(int(offset), count, block), count])
        totals = np.diff(cumulative[edges])
        totals = totals[totals > 0]
        weights = totals / totals.sum()
        result.append(dict(offset_samples=int(offset), positive_weight_blocks=len(totals),
                           kish_weight_blocks=float(1 / np.sum(weights ** 2)),
                           largest_block_weight_fraction=float(weights.max())))
    return result


def infer_family(family: str, grid: dict, numerators: np.ndarray,
                 denominator: np.ndarray, white: np.ndarray, geometry: pd.DataFrame,
                 method: Method) -> tuple[list[dict], list[dict], dict, dict]:
    """Event-conditional intervals and unchanged ten-link scalar-distance model."""
    selected = np.flatnonzero(grid["common"])
    crop = slice(selected[0], selected[-1] + 1)
    common = grid["common"][crop]
    numerator, den = numerators[:, crop].astype(float), denominator[crop].astype(float)
    numerator[:, ~common], den[~common] = 0., 0.
    measured = numerator.sum(axis=1) / den.sum()
    local = np.divide(numerator, den[None, :], out=np.full_like(numerator, np.nan), where=den[None, :] > 0)
    dx = grid["dx"]
    local_acf = [acf_efold(row, dx) for row in local]
    numerator_acf = [acf_efold(np.where(common, row, np.nan), dx) for row in numerator]
    denominator_acf = acf_efold(np.where(common, den, np.nan), dx)
    wavelengths = grid["periods"][grid["keep"]]
    block = max(1, int(np.ceil(max(float(wavelengths.max()), *local_acf, *numerator_acf, denominator_acf) / dx)))
    fitmask = geometry.is_fit_pair.to_numpy(bool)
    distances = geometry.separation_3d_re.to_numpy(float)[fitmask]
    if fitmask.sum() != 10:
        raise ValueError("Wavelet persistence requires exactly six L1-L1 and four L1-P1 pairs")
    white_mean, white95 = float(white.mean()), float(np.quantile(white, .95))
    above = (measured > white95) & fitmask
    p1 = geometry.target_craft.eq("artemis_p1").to_numpy()
    seed = method.seed + WAVEFORM_FAMILIES.index(family) * 100
    points, fits, sensitivity, raw = [], [], [], {}
    for factor, draws in ((1., method.bootstrap_draws), (.5, method.sensitivity_draws), (2., method.sensitivity_draws)):
        if draws == 0:
            continue
        requested = max(1, int(np.ceil(block * factor)))
        boot = weighted_block_means(numerator, den, requested, draws, seed + int(factor * 10))
        slopes = np.sum((distances - distances.mean()) * (boot[:, fitmask] - white_mean) / (1 - white_mean), axis=1) / np.sum((distances - distances.mean()) ** 2)
        valid = np.isfinite(boot[:, fitmask]).all(axis=1)
        slope_probability = float(np.mean(slopes[valid] < 0)) if valid.any() else 0.
        benchmark_support = bool(above.sum() >= 3 and (above & ~p1).any() and (above & p1).any())
        supported = bool(benchmark_support and slope_probability >= .95 and valid.mean() >= .95 and family != "trend")
        if factor == 1:
            intervals = np.nanquantile(boot, [.025, .975], axis=0)
            for index, row in enumerate(geometry.to_dict(orient="records")):
                row.update(family=family, structure=family, distance_re=row["separation_3d_re"],
                           activity_weighted_coherence=float(measured[index]),
                           conditional_range025=float(intervals[0, index]) if family != "trend" else None,
                           conditional_range975=float(intervals[1, index]) if family != "trend" else None,
                           white_mean_r2=white_mean, white_quantile95_r2=white95,
                           above_white_benchmark=bool(measured[index] > white95))
                points.append(row)
            raw[family + "__activity_weighted_coherence"] = boot
            raw[family + "__white_pair_coherence"] = white
        for model, floor in (("raw", 0.), ("white_floor_sensitivity", white_mean)):
            record = dict(structure=family, family=family, model=model,
                          block_factor=factor, block_length_re=requested * dx,
                          resampled_block_length_re=min(requested, len(den)) * dx,
                          white_mean_r2=white_mean, white_quantile95_r2=white95,
                          supported_decay_length=supported,
                          negative_distance_slope_probability=slope_probability,
                          benchmark_support=benchmark_support,
                          valid_bootstrap_fraction=float(valid.mean()),
                          sampled_minimum_distance_re=float(distances.min()),
                          sampled_maximum_distance_re=float(distances.max()))
            if family == "trend":
                record.update(status="descriptive_only_insufficient_long_wave_support",
                              point_coherence_length_re=None, inverse_length_per_re=None,
                              conditional_length_quantile025_re=None, conditional_length_median_re=None,
                              conditional_length_quantile975_re=None, conditional_length_quantile025_endpoint="missing", conditional_length_median_endpoint="missing", conditional_length_quantile975_endpoint="missing", fit_rmse_r2=None, fit_mae_r2=None,
                              curve_distance_re=[], curve_point_r2=[], curve_conditional_low_r2=[], curve_conditional_high_r2=[])
            else:
                rate, length, rmse, mae = fit_rate(distances, measured[fitmask], floor)
                fitted = np.asarray([fit_rate(distances, row[fitmask], floor)[:2]
                                     if np.isfinite(row[fitmask]).all() else [np.nan, np.nan] for row in boot])
                rates, lengths = fitted.T
                quantile = _quantiles(lengths)
                curve_x = np.geomspace(distances.min(), distances.max(), 150)
                curves = np.asarray([predict(curve_x, value, floor) for value in rates if not np.isnan(value)])
                curve_ci = np.quantile(curves, [.025, .975], axis=0)
                prediction = predict(distances, rate, floor)
                point_ci = np.nanquantile(boot[:, fitmask], [.025, .975], axis=0)
                record.update(status=_length_status(lengths[~np.isnan(lengths)], supported, float(distances.min()), float(distances.max())),
                              point_coherence_length_re=length, inverse_length_per_re=rate,
                              length_endpoint="infinity" if np.isposinf(length) else ("zero" if length == 0 else "finite"),
                              conditional_length_quantile025_re=quantile[0], conditional_length_median_re=quantile[1],
                              conditional_length_quantile975_re=quantile[2],
                              **{name: ("missing" if np.isnan(value) else "infinity" if np.isposinf(value) else "zero" if value == 0 else "finite") for name, value in zip(("conditional_length_quantile025_endpoint", "conditional_length_median_endpoint", "conditional_length_quantile975_endpoint"), quantile)},
                              fit_rmse_r2=rmse, fit_mae_r2=mae,
                              curve_distance_re=curve_x, curve_point_r2=predict(curve_x, rate, floor),
                              curve_conditional_low_r2=curve_ci[0], curve_conditional_high_r2=curve_ci[1],
                              model_within_conditional_point_range_fraction=float(np.mean((prediction >= point_ci[0]) & (prediction <= point_ci[1]))))
                if factor == 1:
                    raw[family + "__" + model + "_rate"] = rates
                    raw[family + "__" + model + "_length_re"] = lengths
            sensitivity.append(record)
            if factor == 1:
                fits.append(record)
    metadata = dict(block_length_re=block * dx, spatial_dx_re=dx,
                    complete_band_support_samples=int(common.sum()),
                    scientific_support_fraction=float(grid["common"].sum() / grid["science"].sum()),
                    wavelength_min_re=float(wavelengths.min()), wavelength_max_re=float(wavelengths.max()),
                    effective_profile_blocks=float(common.sum() / block),
                    local_coherence_acf_efold_re=local_acf, weighted_numerator_acf_efold_re=numerator_acf,
                    common_weight_acf_efold_re=denominator_acf,
                    activity_concentration=_activity_concentration(den, block),
                    activity_concentration_is_eligibility_gate=False,
                    block_selection="maximum retained wavelength and first-1/e ACF lengths of every local coherence, numerator and common denominator profile",
                    sensitivity_records=sensitivity)
    return points, fits, metadata, raw


def analyze_family(family: str, sources: dict, definition: dict,
                   geometry: pd.DataFrame, method: Method, directory: Path) -> tuple:
    grid = make_grid(sources, definition, method)
    waves, autos = {}, {}
    print(f"Wavelet coherence: {family}, {len(grid['x'])} positions, {int(grid['keep'].sum())} wavelengths", flush=True)
    for craft in CRAFT_ORDER:
        wave_path, auto_path = directory / f"{craft}_wave.npy", directory / f"{craft}_auto.npy"
        wave = cwt_rows(grid["arrays"][craft], grid["dx"], grid["scales"], wave_path)
        auto = autosmooth(wave, grid["dx"], grid["scales"], method.dj)
        np.save(auto_path, auto)
        del wave, auto
        waves[craft], autos[craft] = np.load(wave_path, mmap_mode="r"), np.load(auto_path, mmap_mode="r")
    weight = common_activity(autos, grid["norms"], grid["keep"])
    if not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError("Invalid common physical auto-power field")
    numerators, denominator = [], None
    for left, right in PAIR_ORDER:
        r2, _ = wct_pair(waves[left], waves[right], autos[left], autos[right], grid["dx"], grid["scales"], grid["keep"], method.dj)
        numerator, current_denominator = weighted_profiles(r2, weight, grid["common"])
        if denominator is not None and not np.array_equal(denominator, current_denominator):
            raise AssertionError("Activity field differs between spacecraft pairs")
        denominator = current_denominator
        numerators.append(numerator)
        del r2
    white = []
    rng = np.random.default_rng(method.seed + WAVEFORM_FAMILIES.index(family))
    for draw in range(method.white_draws):
        left = cwt_rows(rng.normal(size=len(grid["x"])), grid["dx"], grid["scales"])
        right = cwt_rows(rng.normal(size=len(grid["x"])), grid["dx"], grid["scales"])
        sa, sb = autosmooth(left, grid["dx"], grid["scales"], method.dj), autosmooth(right, grid["dx"], grid["scales"], method.dj)
        r2, _ = wct_pair(left, right, sa, sb, grid["dx"], grid["scales"], grid["keep"], method.dj)
        num, den = weighted_profiles(r2, weight, grid["common"])
        white.append(float(num.sum() / den.sum()))
        del left, right, sa, sb, r2
        print(f"  {family} white benchmark {draw + 1}/{method.white_draws}", flush=True)
    numerators = np.asarray(numerators)
    points, fits, metadata, raw = infer_family(family, grid, numerators, denominator, np.asarray(white), geometry, method)
    metadata.update(normalizations=grid["norms"], retained_wavelengths_re=grid["periods"][grid["keep"]],
                    extended_wavelengths_re=grid["periods"], scientific_bounds_re=grid["scientific_bounds_re"],
                    context_bounds_re=grid["context_bounds_re"], finite_maximum_wavelength_re=grid["finite_maximum_wavelength_re"],
                    numerator_sha256=hashlib.sha256(numerators.tobytes()).hexdigest(),
                    denominator_sha256=hashlib.sha256(denominator.tobytes()).hexdigest())
    raw.update({family + "__x_re": grid["x"], family + "__weighted_numerator_profiles": numerators.astype("float32"),
                family + "__weighted_denominator_profile": denominator.astype("float32"),
                family + "__common_support_mask": grid["common"], family + "__wavelength_re": grid["periods"][grid["keep"]]})
    return points, fits, metadata, raw


def analyze_wavelet_coherence(*, method: Method = Method(), cache_root: Path | str | None = None) -> tuple[dict, dict]:
    method.validate()
    sources, design, inputs = prepare_sources()
    geometry = pair_geometry(load_positions())
    inputs.extend([dict(role="fixed_epoch_spacecraft_geometry", **_input_record(EPHEMERIS_CACHE)),
                   dict(role="wavelet_analysis_implementation", **_input_record(__file__)),
                   dict(role="study_method_configuration", **_input_record(study.__file__))])
    l1 = set(study.L1_SPACECRAFT)
    geometry["is_fit_pair"] = [(a in l1 and b in l1) or (a in l1 and b == "artemis_p1")
                               for a, b in zip(geometry.source_craft, geometry.target_craft)]
    cache = Path(cache_root) if cache_root is not None else OUTPUT_ROOT / ".cache" / "wavelet_coherence"
    cache.mkdir(parents=True, exist_ok=True)
    points, fits, family_metadata, raw = [], [], {}, {"pair_labels": np.asarray([a + "__" + b for a, b in PAIR_ORDER])}
    # TemporaryDirectory is deliberately created anew for every call. Runtime
    # transforms are never substitutes for a fresh complete publication run.
    with tempfile.TemporaryDirectory(prefix="run_", dir=cache) as runtime:
        runtime = Path(runtime)
        (runtime / "inputs.json").write_text(json.dumps(inputs, indent=2))
        for family in WAVEFORM_FAMILIES:
            with tempfile.TemporaryDirectory(prefix=family + "_", dir=runtime) as family_cache:
                rows, records, metadata, draws = analyze_family(family, sources, design["bands"][family], geometry, method, Path(family_cache))
                points.extend(rows)
                fits.extend(records)
                family_metadata[family] = metadata
                raw.update(draws)
    for item in inputs:
        if sha256(item["path"]) != item["sha256"]:
            raise RuntimeError(f"Wavelet prerequisite changed during calculation: {item['path']}")
    result = dict(meta=dict(schema_version=SCHEMA_VERSION, method_parameters=asdict(method),
                           estimator="Morlet-6 squared wavelet coherence, PyCWT smoothing", pycwt_version=pycwt.__version__,
                           native_spatial_dx_re=SPATIAL_DX_RE,
                           magnetic_input="original unfiltered prepared one-second |B|, same preparation and 0.04-R_E mapping as waveform filters",
                           alignment="each observer's current measured shock at x=0; no optimized family or pair lag",
                           weighting="one arithmetic-mean four-L1 smoothed physical auto-power field, identical for all 21 pairs",
                           weight_formula="w(x,lambda) = mean_L1[sigma_i^2 S(|W_i(normalized)|^2/s)]",
                           average_formula="sum_common_x,lambda(w R2_pair) / sum_common_x,lambda(w)",
                           support="identical all-seven-observer, all-retained-wavelength cone-of-influence support; scientific overlap in shock-relative distance",
                           intervals="2.5th-97.5th percentiles of synchronized noncircular spatial-profile block resamples, conditional on event, fixed maps, common weights and shock registration; not calibrated full 95% confidence intervals",
                           white_benchmark="independent white pairs on each final spatial grid, fixed observed common weight; descriptive finite-smoothing benchmark, no formal p-values",
                           trend_policy="finite observable wavelengths only, DC excluded; descriptive coherence without uncertainty interval or decay length",
                           primary_decay_model="R2(d)=exp(-d/L); equal-link least squares over unchanged six L1-L1 and four L1-P1 links",
                           geometry=dict(reference_epoch_utc=REFERENCE_EPOCH.isoformat(), coordinate_frame="geocentric GSE", earth_radius_km=REPORTED_EARTH_RADIUS_KM, separation="fixed reference-epoch three-dimensional baselines, same as existing correlation analysis"),
                           inputs=inputs),
                  pair_records=points,
                  coherence_length_records=[record for record in fits if record["model"] == "raw"],
                  white_floor_sensitivity_records=[record for record in fits if record["model"] != "raw"],
                  family_metadata=family_metadata)
    result = json_ready(result)
    validate_wavelet_coherence(result)
    return result, raw


def validate_wavelet_coherence(result: dict) -> pd.DataFrame:
    if result["meta"]["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported wavelet coherence schema")
    table = pd.DataFrame(result["pair_records"])
    expected = {(family, a, b) for family in WAVEFORM_FAMILIES for a, b in PAIR_ORDER}
    observed = set(zip(table.structure, table.source_craft, table.target_craft))
    if len(table) != len(expected) or observed != expected:
        raise ValueError("Wavelet product must contain all 84 unique pair-family measurements")
    values = table.activity_weighted_coherence.to_numpy(float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid wavelet coherence values")
    for family in WAVEFORM_FAMILIES:
        rows = table.loc[table.structure.eq(family)]
        if int(rows.is_fit_pair.sum()) != 10:
            raise ValueError("Unexpected wavelet fit geometry")
        if family != "trend":
            low, high = rows.conditional_range025.to_numpy(float), rows.conditional_range975.to_numpy(float)
            if not (np.isfinite(low).all() and np.isfinite(high).all()) or np.any((low < 0) | (high > 1) | (low > high)):
                raise ValueError("Invalid event-conditional coherence ranges")
        elif rows[["conditional_range025", "conditional_range975"]].notna().any().any():
            raise ValueError("Trend must not report unsupported conditional intervals")
    fits = {item["structure"]: item for item in result["coherence_length_records"]}
    if set(fits) != set(WAVEFORM_FAMILIES):
        raise ValueError("Missing primary wavelet persistence records")
    if fits["trend"]["point_coherence_length_re"] is not None:
        raise ValueError("Trend must not report a decay length")
    return table


def load_wavelet_coherence(path: Path | str = RESULT_PATH) -> dict[str, Any]:
    result = read_json(path)
    validate_wavelet_coherence(result)
    return result
