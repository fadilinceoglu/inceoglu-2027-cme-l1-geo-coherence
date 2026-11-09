"""GSM Bz persistence within the scalar-aligned magnetic structures.

This supplements, and does not modify, the seven-observer scalar analysis.
Each Bz correlation uses the scalar pair's own peak lag. Synchronized moving
blocks resample source origins on the original clock; every draw repeats the
scalar lag search before comparing the Bz values at those original-clock
pairs. Intervals are conditional on the prepared observations and filters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from analysis.correlation_timescales import autocorrelation_block_diagnostics
from analysis.product_io import boolean_values, json_ready, utc_text
from common.io import read_json, sha256
from analysis.waveform_comparisons import (
    FULL_GRID, FULL_SAMPLE_COUNT, LAG_GRID_MINUTES, MINIMUM_PAIRED_SAMPLE_COUNT,
    RANDOM_SEED, STRUCTURES, lagged_normalized_cross_correlation,
    load_all_waveforms, select_peak_lag,
)
from analysis.waveform_inference import (
    BOOTSTRAP_DRAWS, INCIDENT_PAIRS, _derived_seed, _family_block_diagnostics,
    _fit_fixed_unity, _moving_block_origins, load_inference_product,
)
from config import study
from config.paths import PROCESSED_DIR, RESULTS_DIR

SCHEMA_VERSION = 1
CRAFT = (*study.L1_SPACECRAFT, *study.ARTEMIS_SPACECRAFT)
PAIRS = tuple(itertools.combinations(CRAFT, 2))
FAMILIES = STRUCTURES[:-1]
RESULT_PATH = RESULTS_DIR / "bz_persistence.json"


@dataclass(frozen=True)
class Method:
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    seed: int = RANDOM_SEED
    n_jobs: int = 3


def _input_record(path: Path | str, role: str) -> dict:
    path = Path(path).resolve()
    return dict(path=str(path), role=role,
                sha256=sha256(path))


def load_gsm_bz(craft: str, family: str, bands_directory: Path) -> tuple[np.ndarray, list[dict]]:
    """Require actual prepared-source frame provenance, not column-name inference."""
    path = Path(bands_directory) / f"{craft}__{family}.csv"
    metadata_path = path.with_name(path.stem + "_meta.json")
    metadata = read_json(metadata_path)
    source = Path(metadata["input_source"])
    source_metadata_path = source.with_name(source.stem + "_meta.json")
    source_metadata = read_json(source_metadata_path)
    if (study.SATELLITES[craft]["frame"] != "GSM"
            or source_metadata.get("frame") != "GSM"):
        raise ValueError(f"{craft}: Bz persistence requires prepared GSM components")
    if metadata.get("canonical_family") != family:
        raise ValueError(f"{craft}: filtered-family provenance does not match {family}")
    frame = pd.read_csv(path)
    required = {"time", "bz_f", "valid"}
    if not required.issubset(frame):
        raise ValueError(f"{path}: missing Bz waveform columns {sorted(required - set(frame))}")
    if not pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True)).equals(FULL_GRID):
        raise ValueError(f"{path}: Bz requires the complete one-minute scientific grid")
    valid = boolean_values(frame.valid, name=f"{path.name}/valid")
    values = pd.to_numeric(frame.bz_f, errors="raise").to_numpy(float)
    if not np.isfinite(values[valid]).all():
        raise ValueError(f"{path}: nonfinite Bz marked valid")
    values = np.where(valid, values, np.nan)
    return values, [_input_record(path, "filtered_GSM_Bz"),
                    _input_record(metadata_path, "filtered_waveform_provenance"),
                    _input_record(source, "prepared_one_second_magnetic"),
                    _input_record(source_metadata_path, "prepared_magnetic_frame")]


def bz_correlation_at_lag(left: np.ndarray, right: np.ndarray,
                          weights: np.ndarray, lag: int,
                          minimum_paired_count: int = MINIMUM_PAIRED_SAMPLE_COUNT) -> tuple[float, int]:
    """Pearson correlation of original A[j], B[j+lag], weighted by origin counts."""
    left, right, weights = (np.asarray(x, dtype=float) for x in (left, right, weights))
    if (left.ndim != 1 or left.shape != right.shape or weights.shape != left.shape
            or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError("Bz inputs require equal one-dimensional shapes and finite nonnegative weights")
    origin = np.arange(len(left))
    target = origin + int(lag)
    inside = (target >= 0) & (target < len(right)) & (weights > 0)
    origin, target = origin[inside], target[inside]
    finite = np.isfinite(left[origin]) & np.isfinite(right[target])
    origin, target = origin[finite], target[finite]
    w = weights[origin]
    count = float(w.sum())
    if count < minimum_paired_count:
        return np.nan, int(round(count))
    a, b = left[origin], right[target]
    a = a - np.dot(w, a) / count
    b = b - np.dot(w, b) / count
    denominator = float(np.sqrt(np.dot(w, a * a) * np.dot(w, b * b)))
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan, int(round(count))
    return float(np.clip(np.dot(w, a * b) / denominator, -1, 1)), int(round(count))


def derive_block(family: str, bz: dict, scalar: dict, inference: dict) -> dict:
    scalar_floor = _family_block_diagnostics(family, scalar)
    published = inference["family_block_diagnostics"][family]["family_block_length_minutes"]
    if scalar_floor["family_block_length_minutes"] != published:
        raise ValueError(f"{family}: current scalar dependence differs from upstream inference")
    diagnostics = []
    for craft in CRAFT:
        for channel, values in (("bz", bz[craft]), ("bz_squared", bz[craft] ** 2)):
            if not np.isfinite(values).all():
                raise ValueError(f"{family}/{craft}: Bz ACF requires complete prepared minutes; no long-gap filling is allowed")
            diagnostics.append(dict(craft=craft, channel=channel,
                                    **autocorrelation_block_diagnostics(values)))
    length = max(scalar_floor["family_block_length_minutes"],
                 max(row["chosen_block_length_samples"] for row in diagnostics))
    return dict(block_length_minutes=int(length), scalar_publication_floor=scalar_floor,
                bz_channel_diagnostics=diagnostics)


def fit_bz_persistence(family: str, rows: list[dict], draws: np.ndarray) -> tuple[dict, np.ndarray]:
    """Fit the same unity-intercept model; do not borrow scalar significance gates."""
    lookup = {(row["source_craft"], row["target_craft"]): (i, row)
              for i, row in enumerate(rows)}
    if set(lookup) != set(PAIRS):
        raise ValueError("Bz fit needs all fifteen unique near-Earth pair records")
    indices = [lookup[pair][0] for pair in INCIDENT_PAIRS]
    incident = [lookup[pair][1] for pair in INCIDENT_PAIRS]
    distance = np.asarray([row["separation_3d_re"] for row in incident], float)
    observed = np.asarray([row["bz_correlation"] for row in incident], float)
    length = _fit_fixed_unity(distance, observed)
    if not np.isfinite(length):
        raise ValueError(f"{family}: no finite Bz exponential fit")
    lengths = np.asarray([_fit_fixed_unity(distance, values[indices]) for values in draws])
    finite = lengths[np.isfinite(lengths)]
    if len(finite) < .95 * len(lengths):
        raise ValueError(f"{family}: fewer than 95% finite Bz fit draws")
    low, median, high = np.quantile(finite, [.025, .5, .975])
    curve_distance = np.geomspace(distance.min(), distance.max(), 160)
    curve_draws = np.exp(-curve_distance[None, :] / finite[:, None])
    curve_low, curve_high = np.quantile(curve_draws, [.025, .975], axis=0)
    residual = observed - np.exp(-distance / length)
    return dict(
        structure=family, model="r_z(d) = exp(-d / L_cor,z)", amplitude_at_zero=1.,
        fit_scope="six L1-L1 plus four L1-P1 incident links",
        fit_pairs=[list(pair) for pair in INCIDENT_PAIRS], fit_link_count=len(incident),
        fit_loss="equal-link least squares in signed correlation space",
        point_coherence_length_re=float(length),
        bootstrap_length_ci95_low_re=float(low), bootstrap_length_median_re=float(median),
        bootstrap_length_ci95_high_re=float(high), finite_fit_draws=len(finite),
        minimum_sampled_incident_separation_re=float(distance.min()),
        maximum_sampled_incident_separation_re=float(distance.max()),
        probability_length_below_sampled_range=float(np.mean(finite < distance.min())),
        probability_length_above_sampled_range=float(np.mean(finite > distance.max())),
        fit_rmse_ncc=float(np.sqrt(np.mean(residual ** 2))),
        fit_mae_ncc=float(np.mean(abs(residual))),
        scalar_null_tests_applied=False,
        interpretation="descriptive conditional fit at scalar-peak alignment; not a separate null-tested persistence detection",
        curve_distance_re=curve_distance, curve_point_ncc=np.exp(-curve_distance / length),
        curve_ci95_low_ncc=curve_low, curve_ci95_high_ncc=curve_high,
    ), lengths


def _analyze_family(family: str, bz: dict, scalar: dict, comparison: pd.DataFrame,
                    block: dict, method: Method) -> tuple[list[dict], dict, dict]:
    selected = comparison.loc[comparison.structure.eq(family)].set_index(["source_craft", "target_craft"])
    ones = np.ones(FULL_SAMPLE_COUNT)
    rows = []
    for source, target in PAIRS:
        current = selected.loc[(source, target)]
        left, right = scalar[(family, source)].values_nt, scalar[(family, target)].values_nt
        correlations, _ = lagged_normalized_cross_correlation(left, right)
        lag = int(LAG_GRID_MINUTES[select_peak_lag(correlations)])
        if lag != int(current.lag_minutes):
            raise ValueError(f"{family}/{source}/{target}: current scalar alignment differs from inference")
        value, count = bz_correlation_at_lag(bz[source], bz[target], ones, lag)
        if not np.isfinite(value):
            raise ValueError(f"{family}/{source}/{target}: insufficient finite Bz support at scalar peak")
        rows.append(dict(structure=family, source_craft=source, target_craft=target,
                         separation_3d_re=float(current.separation_3d_re),
                         is_fit_pair=(source, target) in INCIDENT_PAIRS,
                         scalar_peak_lag_minutes=lag, bz_correlation=value,
                         paired_minutes=count, bootstrap_draws=method.bootstrap_draws,
                         block_length_minutes=block["block_length_minutes"]))
    length = block["block_length_minutes"]
    count = method.bootstrap_draws
    rng = np.random.default_rng(_derived_seed(method.seed, 530, list(FAMILIES).index(family)))
    draw_values = np.full((count, len(PAIRS)), np.nan)
    draw_lags = np.empty((count, len(PAIRS)), np.int16)
    block_starts = np.empty((count, int(np.ceil(FULL_SAMPLE_COUNT / length))), np.int16)
    for draw in range(count):
        starts, origins = _moving_block_origins(FULL_SAMPLE_COUNT, length, rng)
        block_starts[draw] = starts
        weights = np.bincount(origins, minlength=FULL_SAMPLE_COUNT).astype(float)
        for index, (source, target) in enumerate(PAIRS):
            correlations, _ = lagged_normalized_cross_correlation(
                scalar[(family, source)].values_nt, scalar[(family, target)].values_nt,
                origin_weights=weights)
            lag = int(LAG_GRID_MINUTES[select_peak_lag(correlations)])
            draw_lags[draw, index] = lag
            draw_values[draw, index], _ = bz_correlation_at_lag(bz[source], bz[target], weights, lag)
        if (draw + 1) % 200 == 0:
            print(f"{family}: GSM Bz bootstrap {draw + 1}/{count}", flush=True)
    for index, row in enumerate(rows):
        finite = draw_values[:, index][np.isfinite(draw_values[:, index])]
        if len(finite) < .95 * count:
            raise ValueError(f"{family}/{PAIRS[index]}: insufficient finite Bz bootstrap realizations")
        low, median, high = np.quantile(finite, [.025, .5, .975])
        row.update(bz_correlation_bootstrap_ci95_low=float(low),
                   bz_correlation_bootstrap_median=float(median),
                   bz_correlation_bootstrap_ci95_high=float(high), finite_draws=len(finite))
    fit, lengths = fit_bz_persistence(family, rows, draw_values)
    return rows, fit, {f"{family}__bz_correlation": draw_values.astype(np.float32),
                       f"{family}__scalar_lag_minutes": draw_lags,
                       f"{family}__block_starts": block_starts,
                       f"{family}__length_re": lengths}


def analyze_bz_persistence(*, method: Method = Method(),
                           bands_directory: Path | str = PROCESSED_DIR / "bands",
                           inference_path: Path | str = RESULTS_DIR / "waveform_inference.json") -> tuple[dict, dict]:
    if method.bootstrap_draws < 19 or method.n_jobs < 1:
        raise ValueError("Bz inference requires at least 19 draws and one worker")
    bands_directory, inference_path = Path(bands_directory), Path(inference_path)
    inference, comparison = load_inference_product(inference_path, inference_path.with_suffix(".csv"))
    scalar = load_all_waveforms(bands_directory=bands_directory)
    inputs = [_input_record(inference_path, "current_scalar_inference"),
              _input_record(inference_path.with_suffix(".csv"), "current_scalar_inference_table"),
              _input_record(__file__, "bz_analysis_implementation")]
    inputs.extend(_input_record(series.path, "current_scalar_waveform") for series in scalar.values())
    bz = {family: {} for family in FAMILIES}
    for family in FAMILIES:
        for craft in CRAFT:
            bz[family][craft], provenance = load_gsm_bz(craft, family, bands_directory)
            inputs.extend(provenance)
    blocks = {family: derive_block(family, bz[family], scalar, inference) for family in FAMILIES}
    products = Parallel(n_jobs=method.n_jobs, prefer="threads")(
        delayed(_analyze_family)(family, bz[family], scalar, comparison, blocks[family], method)
        for family in FAMILIES)
    pairs, fits, raw = [], [], {}
    for rows, fit, archive in products:
        pairs.extend(rows)
        fits.append(fit)
        raw.update(archive)
    result = json_ready(dict(
        meta=dict(schema_version=SCHEMA_VERSION, frame="GSM", component="bz_f",
                  window_start_utc=utc_text(study.ANALYSIS_START),
                  window_stop_utc=utc_text(study.ANALYSIS_STOP),
                  full_window_sample_count=FULL_SAMPLE_COUNT, method_parameters=asdict(method),
                  pair_order=[list(pair) for pair in PAIRS],
                  alignment="each pair's scalar-field-strength peak lag; repeated in every draw",
                  bootstrap="synchronized non-circular moving source-origin blocks; original-clock lagged pairs",
                  uncertainty_scope="conditional on prepared observations, frame realization, filters and fixed reference geometry; no calibration or boundary uncertainty",
                  scalar_estimator_unchanged=True,
                  trend="omitted; no trend Bz decay length is fitted"),
        family_block_diagnostics=blocks, pair_records=pairs,
        coherence_length_records=fits,
        inputs=list({(record["path"], record["role"]): record for record in inputs}.values()),
    ))
    validate_bz_persistence(result)
    return result, raw


def validate_bz_persistence(result: dict) -> pd.DataFrame:
    meta = result.get("meta", {})
    if meta.get("schema_version") != SCHEMA_VERSION or meta.get("frame") != "GSM":
        raise ValueError("Unsupported or non-GSM Bz persistence product")
    table = pd.DataFrame(result["pair_records"])
    required = {"structure", "source_craft", "target_craft", "is_fit_pair", "separation_3d_re",
                "bz_correlation", "bz_correlation_bootstrap_ci95_low",
                "bz_correlation_bootstrap_ci95_high", "finite_draws", "bootstrap_draws"}
    if not required.issubset(table):
        raise ValueError("Incomplete Bz persistence table")
    expected = {(family, a, b) for family in FAMILIES for a, b in PAIRS}
    actual = set(table[["structure", "source_craft", "target_craft"]].itertuples(index=False, name=None))
    if len(table) != len(expected) or actual != expected:
        raise ValueError("Bz persistence requires exactly 45 unique near-Earth comparisons")
    for column in ("bz_correlation", "bz_correlation_bootstrap_ci95_low", "bz_correlation_bootstrap_ci95_high"):
        if not table[column].between(-1, 1).all():
            raise ValueError(f"Invalid signed Bz correlation in {column}")
    if (table.bz_correlation_bootstrap_ci95_low > table.bz_correlation_bootstrap_ci95_high).any():
        raise ValueError("Reversed Bz uncertainty limits")
    flags = boolean_values(table.is_fit_pair, name="Bz/is_fit_pair")
    if not np.array_equal(flags, [(a, b) in INCIDENT_PAIRS for a, b in zip(table.source_craft, table.target_craft)]):
        raise ValueError("Bz fit roster must remain the ten incident links")
    fits = result["coherence_length_records"]
    if [row["structure"] for row in fits] != list(FAMILIES):
        raise ValueError("Bz persistence requires three ordered band-pass fits")
    for fit in fits:
        if fit["fit_link_count"] != len(INCIDENT_PAIRS) or fit["fit_pairs"] != [list(pair) for pair in INCIDENT_PAIRS]:
            raise ValueError("Bz fit roster does not match incident links")
        if fit.get("scalar_null_tests_applied") is not False:
            raise ValueError("Scalar significance gates must not be applied to Bz")
    return table


def load_bz_persistence(path: Path | str = RESULT_PATH) -> dict[str, Any]:
    result = read_json(path)
    validate_bz_persistence(result)
    return result
