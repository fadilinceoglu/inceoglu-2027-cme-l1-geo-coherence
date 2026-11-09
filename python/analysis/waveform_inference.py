"""Joint uncertainty, null tests, and spatial persistence of magnetic waveforms.

This module uses the full four-day one-minute records for each spacecraft
from the pairwise comparison stage. A synchronized moving-block bootstrap
preserves shared-node covariance across all seven spacecraft and repeats both
pairwise lag searches and the robust chain-graph registration in every draw.

The main-figure scale is the effective three-dimensional morphological-
correlation scale from ``C(d) = exp(-d / L_c)`` with ``C(0)=1`` fixed.  Only
the six L1--L1 and four L1--P1 incident links enter that fit.  Solar Orbiter
and P2 remain flank and lunar-wake context, respectively.
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar

from config.paths import RESULTS_DIR
from analysis.product_io import json_ready, boolean_values, utc_text
from config import study

from analysis.waveform_comparisons import (
    CRAFT_ORDER,
    FULL_SAMPLE_COUNT,
    L1_CRAFT,
    LAG_GRID_MINUTES,
    PAIR_ORDER,
    PRIMARY_REGISTRATION_EDGES,
    RANDOM_SEED,
    STRUCTURES,
    bootstrap_pair_metrics,
    lagged_normalized_cross_correlation,
    load_all_waveforms,
    sampled_origin_pair,
    select_peak_lag,
    solve_registration_graph,
    validate_result,
)
from analysis.correlation_timescales import (
    autocorrelation_block_diagnostics,
)


BASE_RESULT: Final = RESULTS_DIR / "waveform_comparisons.json"

SCHEMA_VERSION: Final = 1
BOOTSTRAP_DRAWS: Final = 1000
NULL_SURROGATES: Final = 1000
N_JOBS: Final = 4
ONE_OVER_E: Final = float(np.exp(-1.0))

INCIDENT_CRAFT: Final = (*L1_CRAFT, "artemis_p1")
INCIDENT_PAIRS: Final = tuple(itertools.combinations(INCIDENT_CRAFT, 2))
PLOTTED_STRUCTURES: Final = STRUCTURES[:-1]


def _derived_seed(seed: int, *tags: int) -> int:
    return int(
        np.random.SeedSequence([int(seed), *(int(tag) for tag in tags)])
        .generate_state(1, dtype=np.uint32)[0]
    )


def load_base_result(path: Path | str = BASE_RESULT) -> dict[str, Any]:
    source = Path(path)
    result = json.loads(source.read_text())
    validate_result(result)
    if int(result["meta"]["full_window_sample_count"]) != FULL_SAMPLE_COUNT:
        raise ValueError("Figure-5 inference requires all 5,760 minutes")
    return result


def _family_block_diagnostics(
    structure: str,
    waveforms: Mapping[tuple[str, str], Any],
) -> dict[str, Any]:
    craft_rows: list[dict[str, Any]] = []
    for craft in CRAFT_ORDER:
        values = np.asarray(waveforms[(structure, craft)].values_nt, dtype=float)
        raw = autocorrelation_block_diagnostics(values)
        energy = autocorrelation_block_diagnostics(np.square(values))
        chosen = int(
            max(
                raw["chosen_block_length_samples"],
                energy["chosen_block_length_samples"],
            )
        )
        craft_rows.append(
            {
                "craft": craft,
                "raw_signal": raw,
                "squared_signal_energy": energy,
                "chosen_block_length_minutes": chosen,
            }
        )
    family_block = max(
        int(row["chosen_block_length_minutes"]) for row in craft_rows
    )
    return {
        "structure": structure,
        "method": (
            "maximum autocorrelation block across all seven spacecraft and "
            "both x(t) and x(t)^2; x(t)^2 prevents oscillatory sign "
            "cancellation from erasing packet-envelope memory"
        ),
        "family_block_length_minutes": family_block,
        "independent_block_count_full_record": int(
            FULL_SAMPLE_COUNT // family_block
        ),
        "craft_diagnostics": craft_rows,
    }


def _moving_block_origins(
    sample_count: int,
    block_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    block_count = int(np.ceil(sample_count / block_length))
    starts = rng.integers(
        0,
        sample_count - block_length + 1,
        size=block_count,
        dtype=np.int64,
    )
    indices = np.concatenate(
        [start + np.arange(block_length, dtype=np.int64) for start in starts]
    )[:sample_count]
    return starts, indices


PAIR_INDEX: Final = {pair: index for index, pair in enumerate(PAIR_ORDER)}
PRIMARY_PAIR_INDICES: Final = np.asarray(
    [PAIR_INDEX[pair] for pair in PRIMARY_REGISTRATION_EDGES], dtype=int
)
INCIDENT_PAIR_INDICES: Final = np.asarray(
    [PAIR_INDEX[pair] for pair in INCIDENT_PAIRS], dtype=int
)


def _solve_graph_offsets(lags_minutes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(lags_minutes, dtype=float)[PRIMARY_PAIR_INDICES]
    solution = solve_registration_graph(observed)
    if not solution.success:
        raise RuntimeError("bootstrap graph synchronization did not converge")
    continuous = np.r_[0.0, solution.x]
    integer = np.rint(continuous).astype(np.int16)
    return continuous, integer


def _bootstrap_family(
    matrix: np.ndarray,
    block_length: int,
    draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pair_count = len(PAIR_ORDER)
    block_count = int(np.ceil(FULL_SAMPLE_COUNT / block_length))
    arrays: dict[str, np.ndarray] = {
        "block_starts": np.empty((draws, block_count), dtype=np.int16),
        "lag_minutes": np.empty((draws, pair_count), dtype=np.int16),
        "peak_ncc": np.empty((draws, pair_count), dtype=np.float32),
        "rms_ratio": np.empty((draws, pair_count), dtype=np.float32),
        "symmetric_nrmse": np.empty((draws, pair_count), dtype=np.float32),
        "graph_offsets_continuous_minutes": np.empty(
            (draws, len(CRAFT_ORDER)), dtype=np.float32
        ),
        "graph_offsets_integer_minutes": np.empty(
            (draws, len(CRAFT_ORDER)), dtype=np.int16
        ),
        "graph_registered_ncc": np.empty(
            (draws, pair_count), dtype=np.float32
        ),
    }
    craft_index = {craft: index for index, craft in enumerate(CRAFT_ORDER)}
    for draw in range(draws):
        starts, origins = _moving_block_origins(
            FULL_SAMPLE_COUNT,
            block_length,
            rng,
        )
        arrays["block_starts"][draw] = starts.astype(np.int16)
        weights = np.bincount(origins, minlength=FULL_SAMPLE_COUNT).astype(float)
        for pair_position, (source, target) in enumerate(PAIR_ORDER):
            left = matrix[:, craft_index[source]]
            right = matrix[:, craft_index[target]]
            correlations, _ = lagged_normalized_cross_correlation(
                left,
                right,
                origin_weights=weights,
            )
            selected = select_peak_lag(correlations)
            lag = int(LAG_GRID_MINUTES[selected])
            sampled_left, sampled_right = sampled_origin_pair(
                left,
                right,
                lag,
                origins,
            )
            correlation, rms_ratio, symmetric_nrmse = bootstrap_pair_metrics(
                sampled_left,
                sampled_right,
            )
            arrays["lag_minutes"][draw, pair_position] = lag
            arrays["peak_ncc"][draw, pair_position] = correlation
            arrays["rms_ratio"][draw, pair_position] = rms_ratio
            arrays["symmetric_nrmse"][draw, pair_position] = symmetric_nrmse

        continuous, integer = _solve_graph_offsets(
            arrays["lag_minutes"][draw]
        )
        arrays["graph_offsets_continuous_minutes"][draw] = continuous
        arrays["graph_offsets_integer_minutes"][draw] = integer
        for pair_position, (source, target) in enumerate(PAIR_ORDER):
            registered_lag = int(
                integer[craft_index[target]] - integer[craft_index[source]]
            )
            sampled_left, sampled_right = sampled_origin_pair(
                matrix[:, craft_index[source]],
                matrix[:, craft_index[target]],
                registered_lag,
                origins,
            )
            arrays["graph_registered_ncc"][draw, pair_position] = (
                bootstrap_pair_metrics(sampled_left, sampled_right)[0]
            )
    return arrays


def _summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "valid_draws": 0,
            "median": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    return {
        "valid_draws": int(len(finite)),
        "median": float(np.median(finite)),
        "ci95_low": float(np.quantile(finite, 0.025)),
        "ci95_high": float(np.quantile(finite, 0.975)),
    }


def _lag_summary(values: np.ndarray, point_lag: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=int)
    unique, counts = np.unique(array, return_counts=True)
    maximum = int(np.max(counts))
    candidates = unique[counts == maximum]
    mode = int(candidates[np.argmin(np.abs(candidates - int(point_lag)))])
    return {
        **_summary(array.astype(float)),
        "mode": mode,
        "point_lag_selection_probability": float(np.mean(array == point_lag)),
        "search_edge_probability": float(
            np.mean(np.abs(array) == int(np.max(np.abs(LAG_GRID_MINUTES))))
        ),
    }


def _null_pair(
    structure: str,
    source_craft: str,
    target_craft: str,
    source: np.ndarray,
    target: np.ndarray,
    point_lag: int,
    block_length: int,
    surrogates: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    minimum_shift = block_length + int(np.max(np.abs(LAG_GRID_MINUTES)))
    candidates = np.arange(1, FULL_SAMPLE_COUNT, dtype=int)
    circular_distance = np.minimum(candidates, FULL_SAMPLE_COUNT - candidates)
    admissible = candidates[circular_distance >= minimum_shift]
    if len(admissible) < surrogates:
        raise ValueError("too few guarded circular shifts for requested null")
    rng = np.random.default_rng(_derived_seed(seed, 2))
    shifts = np.sort(rng.choice(admissible, size=surrogates, replace=False))
    null_ncc = np.empty(surrogates, dtype=np.float32)
    null_lag = np.empty(surrogates, dtype=np.int16)
    for index, shift in enumerate(shifts):
        shifted_target = np.roll(target, int(shift))
        correlations, _ = lagged_normalized_cross_correlation(
            source,
            shifted_target,
        )
        selected = select_peak_lag(correlations)
        lag = int(LAG_GRID_MINUTES[selected])
        null_lag[index] = lag
        null_ncc[index] = float(correlations[selected])

    point_correlations, _ = lagged_normalized_cross_correlation(source, target)
    point_index = int(np.flatnonzero(LAG_GRID_MINUTES == point_lag)[0])
    observed_ncc = float(point_correlations[point_index])
    ncc_exceed = int(np.sum(null_ncc >= observed_ncc))
    record = {
        "structure": structure,
        "source_craft": source_craft,
        "target_craft": target_craft,
        "method": (
            "guarded circular shift of target; every surrogate repeats the "
            "complete NCC lag search"
        ),
        "surrogate_count": int(surrogates),
        "minimum_circular_shift_minutes": int(minimum_shift),
        "admissible_shift_count": int(len(admissible)),
        "observed_peak_ncc": observed_ncc,
        "ncc_null_median": float(np.median(null_ncc)),
        "ncc_null_ci95_low": float(np.quantile(null_ncc, 0.025)),
        "ncc_null_ci95_high": float(np.quantile(null_ncc, 0.975)),
        "ncc_empirical_p_value": float(
            (1 + ncc_exceed) / (1 + surrogates)
        ),
    }
    return record, null_lag, null_ncc, shifts


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _fit_fixed_unity(distance_re: np.ndarray, correlation: np.ndarray) -> float:
    distance = np.asarray(distance_re, dtype=float)
    values = np.asarray(correlation, dtype=float)
    if (
        distance.shape != values.shape
        or len(distance) != len(INCIDENT_PAIRS)
        or not np.isfinite(distance).all()
        or not np.isfinite(values).all()
    ):
        return np.nan

    def objective(log_length: float) -> float:
        length = float(np.exp(log_length))
        predicted = np.exp(-distance / length)
        return float(np.sum(np.square(values - predicted)))

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.1), np.log(1.0e7)),
        method="bounded",
    )
    return float(np.exp(result.x)) if result.success else np.nan


def _log_slope(distance_re: np.ndarray, correlation: np.ndarray) -> float:
    distance = np.asarray(distance_re, dtype=float)
    values = np.asarray(correlation, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        return np.nan
    return float(np.polyfit(distance, np.log(values), 1)[0])


def _coherence_fit_record(
    structure: str,
    distances: np.ndarray,
    point_ncc: np.ndarray,
    bootstrap_ncc: np.ndarray,
    point_graph_ncc: np.ndarray,
    bootstrap_graph_ncc: np.ndarray,
    incident_null_records: list[dict[str, Any]],
) -> dict[str, Any]:
    d_min = float(np.min(distances))
    d_max = float(np.max(distances))
    point_length = _fit_fixed_unity(distances, point_ncc)
    lengths = np.asarray(
        [_fit_fixed_unity(distances, row) for row in bootstrap_ncc],
        dtype=float,
    )
    slopes = np.asarray(
        [_log_slope(distances, row) for row in bootstrap_ncc],
        dtype=float,
    )
    finite = np.isfinite(lengths)
    finite_fraction = float(np.mean(finite))
    negative_slope_probability = float(np.mean(slopes[np.isfinite(slopes)] < 0.0))

    graph_positive_fraction = float(np.mean(point_graph_ncc > 0.0))
    point_survival = bool(np.all(point_ncc > ONE_OVER_E))
    graph_survival = bool(np.all(point_graph_ncc > ONE_OVER_E))
    graph_same_survival_class = point_survival == graph_survival

    significant = [
        row
        for row in incident_null_records
        if float(row["ncc_bh_q_value"]) <= 0.05
    ]
    significant_l1_l1 = sum(
        row["source_craft"] in L1_CRAFT
        and row["target_craft"] in L1_CRAFT
        for row in significant
    )
    significant_l1_p1 = sum(
        row["source_craft"] in L1_CRAFT
        and row["target_craft"] == "artemis_p1"
        for row in significant
    )
    null_gate = bool(
        len(significant) >= 3
        and significant_l1_l1 >= 1
        and significant_l1_p1 >= 1
    )
    graph_gate = bool(
        graph_positive_fraction >= 0.80 and graph_same_survival_class
    )
    decay_gate = bool(negative_slope_probability >= 0.95)
    valid_gate = bool(finite_fraction >= 0.95)
    gates_pass = null_gate and graph_gate and decay_gate and valid_gate

    valid_lengths = lengths[finite]
    probability_left = float(np.mean(valid_lengths < d_min))
    probability_resolved = float(
        np.mean((valid_lengths >= d_min) & (valid_lengths <= d_max))
    )
    probability_right = float(np.mean(valid_lengths > d_max))
    if not gates_pass:
        status = "unresolved_or_ineligible"
    elif probability_resolved >= 0.95:
        status = "finite_resolved"
    elif probability_right >= 0.95:
        status = "right_censored"
    elif probability_left >= 0.95:
        status = "left_censored"
    else:
        status = "unresolved_or_ineligible"

    length_summary = _summary(valid_lengths)
    point_residual = np.asarray(point_ncc, dtype=float) - np.exp(
        -np.asarray(distances, dtype=float) / point_length
    )
    curve_distance = np.geomspace(d_min, d_max, 160)
    curve_point = np.exp(-curve_distance / point_length)
    curve_draws = np.exp(
        -curve_distance[None, :] / valid_lengths[:, None]
    )
    curve_low, curve_high = np.quantile(curve_draws, [0.025, 0.975], axis=0)

    graph_lengths = np.asarray(
        [_fit_fixed_unity(distances, row) for row in bootstrap_graph_ncc],
        dtype=float,
    )
    graph_lengths = graph_lengths[np.isfinite(graph_lengths)]
    return {
        "structure": structure,
        "fit_scope": "six L1-L1 plus four L1-P1 incident links",
        "distance_definition": "fixed-epoch 3-D GSE separation",
        "interpretation": (
            "effective 3-D morphological-correlation scale over the sampled "
            "incident geometry; not an intrinsic isotropic plasma scale"
        ),
        "model": "C(d) = exp(-d / L_c)",
        "amplitude_at_zero": 1.0,
        "fit_loss": "equal-link least squares in correlation space",
        "point_coherence_length_re": float(point_length),
        "fit_link_count": int(len(point_residual)),
        "fit_rmse_ncc": float(np.sqrt(np.mean(np.square(point_residual)))),
        "fit_mae_ncc": float(np.mean(np.abs(point_residual))),
        "bootstrap_coherence_length_re": length_summary,
        "status": status,
        "minimum_sampled_incident_separation_re": d_min,
        "maximum_sampled_incident_separation_re": d_max,
        "probability_lc_below_sampled_range": probability_left,
        "probability_lc_inside_sampled_range": probability_resolved,
        "probability_lc_above_sampled_range": probability_right,
        "negative_log_slope_probability": negative_slope_probability,
        "valid_joint_draw_fraction": finite_fraction,
        "eligibility_gates": {
            "selection_aware_ncc_null": null_gate,
            "negative_distance_slope": decay_gate,
            "graph_registration_sensitivity": graph_gate,
            "valid_joint_draws": valid_gate,
        },
        "selection_aware_ncc_significant_incident_links": len(significant),
        "selection_aware_ncc_significant_l1_l1_links": significant_l1_l1,
        "selection_aware_ncc_significant_l1_p1_links": significant_l1_p1,
        "point_incident_links_above_one_over_e": int(
            np.sum(point_ncc > ONE_OVER_E)
        ),
        "graph_registered_positive_incident_fraction": graph_positive_fraction,
        "graph_registered_same_one_over_e_survival_class": graph_same_survival_class,
        "graph_registered_length_sensitivity_re": _summary(graph_lengths),
        "curve_distance_re": curve_distance,
        "curve_point_ncc": curve_point,
        "curve_ci95_low_ncc": curve_low,
        "curve_ci95_high_ncc": curve_high,
    }


def analyze_joint_inference(
    *,
    base_path: Path | str = BASE_RESULT,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    null_surrogates: int = NULL_SURROGATES,
    n_jobs: int = N_JOBS,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if bootstrap_draws < 19 or null_surrogates < 19:
        raise ValueError("at least 19 bootstrap draws and null surrogates required")
    base = load_base_result(base_path)
    waveforms = load_all_waveforms()
    point_lookup = {
        (row["structure"], row["source_craft"], row["target_craft"]): row
        for row in base["pair_records"]
    }
    blocks = {
        structure: _family_block_diagnostics(structure, waveforms)
        for structure in STRUCTURES
    }
    matrices = {
        structure: np.column_stack(
            [waveforms[(structure, craft)].values_nt for craft in CRAFT_ORDER]
        )
        for structure in STRUCTURES
    }
    bootstrap_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_bootstrap_family)(
            matrices[structure],
            int(blocks[structure]["family_block_length_minutes"]),
            bootstrap_draws,
            _derived_seed(RANDOM_SEED, 100, structure_index),
        )
        for structure_index, structure in enumerate(STRUCTURES)
    )
    bootstrap = dict(zip(STRUCTURES, bootstrap_results))

    null_tasks = []
    for structure_index, structure in enumerate(STRUCTURES):
        for pair_index, (source, target) in enumerate(PAIR_ORDER):
            point = point_lookup[(structure, source, target)]
            null_tasks.append(
                (
                    structure,
                    source,
                    target,
                    waveforms[(structure, source)].values_nt,
                    waveforms[(structure, target)].values_nt,
                    int(point["alignment"]["lag_minutes"]),
                    int(blocks[structure]["family_block_length_minutes"]),
                    null_surrogates,
                    _derived_seed(
                        RANDOM_SEED, 200, structure_index, pair_index
                    ),
                )
            )
    null_outputs = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_null_pair)(*task) for task in null_tasks
    )
    null_records = [item[0] for item in null_outputs]
    null_lookup = {
        (row["structure"], row["source_craft"], row["target_craft"]): row
        for row in null_records
    }
    for structure in STRUCTURES:
        rows = [row for row in null_records if row["structure"] == structure]
        ncc_q = _bh_adjust(
            np.asarray([row["ncc_empirical_p_value"] for row in rows])
        )
        for row, ncc_value in zip(rows, ncc_q):
            row["ncc_bh_q_value"] = float(ncc_value)
            row["multiplicity_family_size"] = len(rows)

    pair_records: list[dict[str, Any]] = []
    for structure in STRUCTURES:
        arrays = bootstrap[structure]
        for pair_position, (source, target) in enumerate(PAIR_ORDER):
            point = point_lookup[(structure, source, target)]
            null = null_lookup[(structure, source, target)]
            record = {
                "structure": structure,
                "source_craft": source,
                "target_craft": target,
                "baseline_class": point["geometry"]["baseline_class"],
                "separation_3d_re": float(
                    point["geometry"]["separation_3d_re"]
                ),
                "pair_peak": {
                    "lag_minutes": int(point["alignment"]["lag_minutes"]),
                    "normalized_cross_correlation": float(
                        point["metrics"]["normalized_cross_correlation"]
                    ),
                    "rms_ratio_target_over_source": float(
                        point["metrics"]["rms_ratio_target_over_source"]
                    ),
                    "symmetric_normalized_rmse": float(
                        point["metrics"]["symmetric_normalized_rmse"]
                    ),
                    "selected_lag_at_search_edge": bool(
                        point["alignment"]["selected_lag_at_search_edge"]
                    ),
                    "expanded_search_lag_unchanged": bool(
                        point["alignment"]["expanded_search_sensitivity"][
                            "selected_lag_unchanged"
                        ]
                    ),
                },
                "graph_registered": copy.deepcopy(
                    point["global_registration"]
                ),
                "joint_bootstrap": {
                    "method": (
                        "synchronized non-circular moving blocks across all "
                        "seven spacecraft; complete lag search and graph solve "
                        "repeated in every draw"
                    ),
                    "draws": int(bootstrap_draws),
                    "family_block_length_minutes": int(
                        blocks[structure]["family_block_length_minutes"]
                    ),
                    "lag_minutes": _lag_summary(
                        arrays["lag_minutes"][:, pair_position],
                        int(point["alignment"]["lag_minutes"]),
                    ),
                    "normalized_cross_correlation": _summary(
                        arrays["peak_ncc"][:, pair_position]
                    ),
                    "rms_ratio_target_over_source": _summary(
                        arrays["rms_ratio"][:, pair_position]
                    ),
                    "symmetric_normalized_rmse": _summary(
                        arrays["symmetric_nrmse"][:, pair_position]
                    ),
                    "graph_registered_ncc": _summary(
                        arrays["graph_registered_ncc"][:, pair_position]
                    ),
                },
                "selection_aware_null": copy.deepcopy(null),
            }
            pair_records.append(record)

    fit_records: list[dict[str, Any]] = []
    distances = np.asarray(
        [
            point_lookup[("discontinuity", source, target)]["geometry"][
                "separation_3d_re"
            ]
            for source, target in INCIDENT_PAIRS
        ],
        dtype=float,
    )
    for structure in PLOTTED_STRUCTURES:
        arrays = bootstrap[structure]
        point_ncc = np.asarray(
            [
                point_lookup[(structure, source, target)]["metrics"][
                    "normalized_cross_correlation"
                ]
                for source, target in INCIDENT_PAIRS
            ],
            dtype=float,
        )
        point_graph = np.asarray(
            [
                point_lookup[(structure, source, target)][
                    "global_registration"
                ]["normalized_cross_correlation"]
                for source, target in INCIDENT_PAIRS
            ],
            dtype=float,
        )
        incident_null = [
            null_lookup[(structure, source, target)]
            for source, target in INCIDENT_PAIRS
        ]
        fit_records.append(
            _coherence_fit_record(
                structure,
                distances,
                point_ncc,
                arrays["peak_ncc"][:, INCIDENT_PAIR_INDICES],
                point_graph,
                arrays["graph_registered_ncc"][:, INCIDENT_PAIR_INDICES],
                incident_null,
            )
        )

    trend_incident = [
        point_lookup[("trend", source, target)]["metrics"][
            "normalized_cross_correlation"
        ]
        for source, target in INCIDENT_PAIRS
    ]
    result = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "scientific_scope": (
                "joint uncertainty, selection-aware nulls, and effective "
                "coherence scales for the four-day Figure-5 waveforms"
            ),
            "base_result_path": str(Path(base_path)),
            "window_start_utc": base["meta"]["window_start_utc"],
            "window_stop_utc": base["meta"]["window_stop_utc"],
            "window_endpoint_policy": "start inclusive, stop exclusive",
            "full_window_sample_count": FULL_SAMPLE_COUNT,
            "cadence_seconds": 60,
            "bootstrap_draws": int(bootstrap_draws),
            "null_surrogates": int(null_surrogates),
            "random_seed": RANDOM_SEED,
            "pair_order": [list(pair) for pair in PAIR_ORDER],
            "structure_order": list(STRUCTURES),
            "incident_fit_pairs": [list(pair) for pair in INCIDENT_PAIRS],
        },
        "family_block_diagnostics": blocks,
        "pair_records": pair_records,
        "coherence_length_records": fit_records,
        "trend_context": {
            "fit_status": "not_fitted_mean_retaining_low_pass",
            "record_duration_hours": 96,
            "minimum_incident_pair_peak_ncc": float(min(trend_incident)),
            "maximum_incident_separation_re": float(np.max(distances)),
            "all_incident_pair_peak_ncc_above_one_over_e": bool(
                np.all(np.asarray(trend_incident) > ONE_OVER_E)
            ),
            "interpretation": (
                "no observed 1/e loss through the sampled incident baselines; "
                "right-censored descriptive context, not a numerical fit"
            ),
        },
    }

    raw: dict[str, np.ndarray] = {}
    for structure, arrays in bootstrap.items():
        for name, values in arrays.items():
            raw[f"bootstrap__{structure}__{name}"] = values
    for task, output in zip(null_tasks, null_outputs):
        structure, source, target = task[:3]
        raw[f"null__{structure}__{source}__{target}__lag_minutes"] = output[1]
        raw[f"null__{structure}__{source}__{target}__peak_ncc"] = output[2]
        raw[f"null__{structure}__{source}__{target}__shift_minutes"] = output[3]
    return json_ready(result), raw


def records_to_table(result: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in result["pair_records"]:
        point = record["pair_peak"]
        graph = record["graph_registered"]
        boot = record["joint_bootstrap"]
        null = record["selection_aware_null"]
        rows.append(
            {
                "schema_version": result["meta"]["schema_version"],
                "window_start_utc": result["meta"]["window_start_utc"],
                "window_stop_utc": result["meta"]["window_stop_utc"],
                "full_window_sample_count": result["meta"][
                    "full_window_sample_count"
                ],
                "structure": record["structure"],
                "source_craft": record["source_craft"],
                "target_craft": record["target_craft"],
                "baseline_class": record["baseline_class"],
                "separation_3d_re": record["separation_3d_re"],
                "lag_minutes": point["lag_minutes"],
                "normalized_cross_correlation": point[
                    "normalized_cross_correlation"
                ],
                "rms_ratio_target_over_source": point[
                    "rms_ratio_target_over_source"
                ],
                "symmetric_normalized_rmse": point[
                    "symmetric_normalized_rmse"
                ],
                "selected_lag_at_search_edge": point[
                    "selected_lag_at_search_edge"
                ],
                "expanded_search_lag_unchanged": point[
                    "expanded_search_lag_unchanged"
                ],
                "globally_registered_lag_minutes": graph["lag_minutes"],
                "globally_registered_normalized_cross_correlation": graph[
                    "normalized_cross_correlation"
                ],
                "bootstrap_draws": boot["draws"],
                "bootstrap_block_length_minutes": boot[
                    "family_block_length_minutes"
                ],
                "lag_bootstrap_mode_minutes": boot["lag_minutes"]["mode"],
                "lag_bootstrap_ci95_low_minutes": boot["lag_minutes"][
                    "ci95_low"
                ],
                "lag_bootstrap_ci95_high_minutes": boot["lag_minutes"][
                    "ci95_high"
                ],
                "lag_point_selection_probability": boot["lag_minutes"][
                    "point_lag_selection_probability"
                ],
                "lag_search_edge_probability": boot["lag_minutes"][
                    "search_edge_probability"
                ],
                "normalized_cross_correlation_bootstrap_ci95_low": boot[
                    "normalized_cross_correlation"
                ]["ci95_low"],
                "normalized_cross_correlation_bootstrap_ci95_high": boot[
                    "normalized_cross_correlation"
                ]["ci95_high"],
                "rms_ratio_target_over_source_bootstrap_ci95_low": boot[
                    "rms_ratio_target_over_source"
                ]["ci95_low"],
                "rms_ratio_target_over_source_bootstrap_ci95_high": boot[
                    "rms_ratio_target_over_source"
                ]["ci95_high"],
                "symmetric_normalized_rmse_bootstrap_ci95_low": boot[
                    "symmetric_normalized_rmse"
                ]["ci95_low"],
                "symmetric_normalized_rmse_bootstrap_ci95_high": boot[
                    "symmetric_normalized_rmse"
                ]["ci95_high"],
                "graph_registered_ncc_bootstrap_ci95_low": boot[
                    "graph_registered_ncc"
                ]["ci95_low"],
                "graph_registered_ncc_bootstrap_ci95_high": boot[
                    "graph_registered_ncc"
                ]["ci95_high"],
                "ncc_null_median": null["ncc_null_median"],
                "ncc_empirical_p_value": null["ncc_empirical_p_value"],
                "ncc_bh_q_value": null["ncc_bh_q_value"],
            }
        )
    return pd.DataFrame(rows)


CURVE_STATUSES = {"finite_resolved", "right_censored", "left_censored"}
INFERENCE_COLUMNS = (
    "schema_version", "window_start_utc", "window_stop_utc", "full_window_sample_count",
    "structure", "source_craft", "target_craft", "baseline_class", "separation_3d_re",
    "lag_minutes", "normalized_cross_correlation", "rms_ratio_target_over_source",
    "symmetric_normalized_rmse", "selected_lag_at_search_edge", "expanded_search_lag_unchanged",
    "globally_registered_lag_minutes", "globally_registered_normalized_cross_correlation",
    "bootstrap_draws", "bootstrap_block_length_minutes", "lag_bootstrap_mode_minutes",
    "lag_bootstrap_ci95_low_minutes", "lag_bootstrap_ci95_high_minutes",
    "lag_point_selection_probability", "lag_search_edge_probability",
    "normalized_cross_correlation_bootstrap_ci95_low", "normalized_cross_correlation_bootstrap_ci95_high",
    "rms_ratio_target_over_source_bootstrap_ci95_low", "rms_ratio_target_over_source_bootstrap_ci95_high",
    "symmetric_normalized_rmse_bootstrap_ci95_low", "symmetric_normalized_rmse_bootstrap_ci95_high",
    "graph_registered_ncc_bootstrap_ci95_low", "graph_registered_ncc_bootstrap_ci95_high",
    "ncc_null_median", "ncc_empirical_p_value", "ncc_bh_q_value",
)


def validate_inference_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate the shared inference table used by main and supplementary figures."""
    missing = set(INFERENCE_COLUMNS) - set(table)
    if missing:
        raise ValueError("joint inference table is missing: " + ", ".join(sorted(missing)))
    expected = {(family, a, b) for family in STRUCTURES for a, b in PAIR_ORDER}
    observed = set(table[["structure", "source_craft", "target_craft"]].itertuples(index=False, name=None))
    if len(table) != len(expected) or observed != expected:
        raise ValueError("joint inference requires all 84 unique pair/family records")
    constants = {"schema_version": SCHEMA_VERSION, "window_start_utc": utc_text(study.ANALYSIS_START),
                 "window_stop_utc": utc_text(study.ANALYSIS_STOP), "full_window_sample_count": FULL_SAMPLE_COUNT}
    for name, value in constants.items():
        if not table[name].eq(value).all():
            raise ValueError(f"joint inference table has invalid {name}")
    table = table.copy()
    flags = {"selected_lag_at_search_edge", "expanded_search_lag_unchanged"}
    for name in flags:
        table[name] = boolean_values(table[name], name=name)
    text = {"window_start_utc", "window_stop_utc", "structure", "source_craft", "target_craft", "baseline_class"}
    numeric = set(INFERENCE_COLUMNS) - flags - text
    for name in numeric:
        table[name] = pd.to_numeric(table[name], errors="raise")
    if not np.isfinite(table[list(numeric)].to_numpy(float)).all():
        raise ValueError("joint inference metrics must be finite for all records")
    if table["bootstrap_draws"].nunique() != 1 or table["bootstrap_draws"].iloc[0] < 19:
        raise ValueError("bootstrap draw count must be common and at least 19")
    for name in ("normalized_cross_correlation", "globally_registered_normalized_cross_correlation",
                 "normalized_cross_correlation_bootstrap_ci95_low", "normalized_cross_correlation_bootstrap_ci95_high",
                 "graph_registered_ncc_bootstrap_ci95_low", "graph_registered_ncc_bootstrap_ci95_high"):
        if not table[name].between(-1.0, 1.0).all():
            raise ValueError(f"{name} must lie in [-1, 1]")
    for name in ("rms_ratio_target_over_source", "rms_ratio_target_over_source_bootstrap_ci95_low",
                 "rms_ratio_target_over_source_bootstrap_ci95_high"):
        if not table[name].gt(0).all():
            raise ValueError("RMS ratios must be positive")
    for name in ("symmetric_normalized_rmse", "symmetric_normalized_rmse_bootstrap_ci95_low",
                 "symmetric_normalized_rmse_bootstrap_ci95_high"):
        if not table[name].ge(0).all():
            raise ValueError("symmetric normalized RMSE must be nonnegative")
    for low, high in (("lag_bootstrap_ci95_low_minutes", "lag_bootstrap_ci95_high_minutes"),
                      *((f"{name}_bootstrap_ci95_low", f"{name}_bootstrap_ci95_high") for name in
                        ("normalized_cross_correlation", "rms_ratio_target_over_source", "symmetric_normalized_rmse", "graph_registered_ncc"))):
        if (table[low] > table[high]).any():
            raise ValueError(f"{low} exceeds its upper interval bound")
    return table


def read_inference_table(path: Path | str) -> pd.DataFrame:
    return validate_inference_table(pd.read_csv(path))


def _validate_coherence_fits(result: Mapping[str, Any], table: pd.DataFrame) -> None:
    fits = result["coherence_length_records"]
    if [row.get("structure") for row in fits] != list(PLOTTED_STRUCTURES):
        raise ValueError("Figure 5 requires exactly three ordered fit records")

    incident = table.loc[
        table["structure"].isin(PLOTTED_STRUCTURES)
        & pd.Series([(a, b) in INCIDENT_PAIRS for a, b in zip(table["source_craft"], table["target_craft"])], index=table.index)
    ]
    if len(incident) != 30:
        raise ValueError("Figure 5 requires ten incident links per waveform family")
    expected_min = float(incident["separation_3d_re"].min())
    expected_max = float(incident["separation_3d_re"].max())

    for record in fits:
        status = record.get("status")
        if status not in CURVE_STATUSES | {"unresolved_or_ineligible"}:
            raise ValueError(f"unsupported coherence-scale status: {status!r}")
        if record.get("model") != "C(d) = exp(-d / L_c)":
            raise ValueError("Figure-5 model must be the fixed-unity exponential")
        if float(record.get("amplitude_at_zero", np.nan)) != 1.0:
            raise ValueError("Figure-5 model must fix C(0)=1")
        d_min = float(record["minimum_sampled_incident_separation_re"])
        d_max = float(record["maximum_sampled_incident_separation_re"])
        if not np.allclose([d_min, d_max], [expected_min, expected_max], atol=1e-9):
            raise ValueError("fit domain does not match the ten incident links")

        gates = record.get("eligibility_gates", {})
        gates_pass = all(bool(value) for value in gates.values())
        if status in CURVE_STATUSES and not gates_pass:
            raise ValueError("an eligible coherence-scale status has a failed gate")
        if status == "unresolved_or_ineligible" and gates_pass:
            probabilities = (
                float(record["probability_lc_below_sampled_range"]),
                float(record["probability_lc_inside_sampled_range"]),
                float(record["probability_lc_above_sampled_range"]),
            )
            if max(probabilities) >= 0.95:
                raise ValueError("unresolved fit conflicts with its censoring probabilities")

        probability_key = {
            "finite_resolved": "probability_lc_inside_sampled_range",
            "right_censored": "probability_lc_above_sampled_range",
            "left_censored": "probability_lc_below_sampled_range",
        }.get(status)
        if probability_key and float(record[probability_key]) < 0.95:
            raise ValueError(f"{status} fit does not meet its 0.95 probability gate")

        arrays = [
            np.asarray(record[name], dtype=float)
            for name in (
                "curve_distance_re",
                "curve_point_ncc",
                "curve_ci95_low_ncc",
                "curve_ci95_high_ncc",
            )
        ]
        if any(array.ndim != 1 or len(array) < 2 for array in arrays):
            raise ValueError("coherence-scale curve arrays must be one-dimensional")
        if len({len(array) for array in arrays}) != 1:
            raise ValueError("coherence-scale curve arrays have inconsistent lengths")
        distance, point, low, high = arrays
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("coherence-scale curve arrays must be finite")
        if not np.all(np.diff(distance) > 0.0):
            raise ValueError("coherence-scale curve distance must increase strictly")
        if not np.allclose([distance[0], distance[-1]], [d_min, d_max], atol=1e-9):
            raise ValueError("coherence-scale curve extends beyond the incident domain")
        if np.any(low > high):
            raise ValueError("coherence-scale interval lower bounds exceed upper bounds")
        if any(np.any((array < 0.0) | (array > 1.0)) for array in (point, low, high)):
            raise ValueError("coherence-scale curves must lie within [0, 1]")


def validate_inference(result: Mapping[str, Any]) -> pd.DataFrame:
    """Validate one scientific result and return its reusable flat table."""
    meta = result["meta"]
    expected = {"schema_version": SCHEMA_VERSION, "window_start_utc": utc_text(study.ANALYSIS_START),
                "window_stop_utc": utc_text(study.ANALYSIS_STOP), "full_window_sample_count": FULL_SAMPLE_COUNT,
                "cadence_seconds": 60}
    for name, value in expected.items():
        if meta.get(name) != value:
            raise ValueError(f"joint inference has invalid {name}")
    if [tuple(pair) for pair in meta.get("pair_order", [])] != list(PAIR_ORDER):
        raise ValueError("joint inference has a non-canonical pair order")
    if [tuple(pair) for pair in meta.get("incident_fit_pairs", [])] != list(INCIDENT_PAIRS):
        raise ValueError("joint inference has a non-canonical incident fit set")
    if int(meta.get("null_surrogates", 0)) < 19:
        raise ValueError("joint inference requires at least 19 null surrogates")
    table = validate_inference_table(records_to_table(result))
    if not table["bootstrap_draws"].eq(meta["bootstrap_draws"]).all():
        raise ValueError("inference metadata and pair bootstrap counts disagree")
    _validate_coherence_fits(result, table)
    trend = result.get("trend_context", {})
    if trend.get("fit_status") != "not_fitted_mean_retaining_low_pass":
        raise ValueError("trend must remain descriptive and unfitted")
    if trend.get("record_duration_hours") != (study.ANALYSIS_STOP - study.ANALYSIS_START) / pd.Timedelta(hours=1):
        raise ValueError("trend context must use the complete analysis record")
    return table


def load_inference_product(json_path: Path | str, csv_path: Path | str) -> tuple[dict[str, Any], pd.DataFrame]:
    result = json.loads(Path(json_path).read_text())
    expected = validate_inference(result)
    table = read_inference_table(csv_path)
    keys = ["structure", "source_craft", "target_craft"]
    expected, actual = expected.set_index(keys).sort_index(), table.set_index(keys).sort_index()
    for name in INFERENCE_COLUMNS:
        if name in keys:
            continue
        if pd.api.types.is_numeric_dtype(expected[name]) and not pd.api.types.is_bool_dtype(expected[name]):
            matches = np.allclose(expected[name], actual[name], rtol=1e-9, atol=1e-11, equal_nan=True)
        else:
            matches = expected[name].equals(actual[name])
        if not matches:
            raise ValueError(f"joint inference JSON and CSV disagree for {name}")
    return result, table
