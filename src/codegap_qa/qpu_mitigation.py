from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import beta

from .qpu_counts import counts_to_samples


@dataclass(frozen=True)
class CalibrationAnalysis:
    convention: str
    convention_scores: dict[str, float]
    convention_margin: float
    p01: tuple[float, ...]
    p10: tuple[float, ...]
    assignment_determinants: tuple[float, ...]
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "convention": self.convention,
            "convention_scores": self.convention_scores,
            "convention_margin": self.convention_margin,
            "p01": list(self.p01),
            "p10": list(self.p10),
            "assignment_determinants": list(self.assignment_determinants),
            "minimum_assignment_determinant": min(
                self.assignment_determinants,
                default=0.0,
            ),
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def short_label(label: object) -> str:
    return str(label).split("::")[-1]


def result_map(raw_counts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        short_label(item["label"]): item
        for item in raw_counts.get("results", [])
    }


def bitwise_pattern_accuracy(
    counts: dict[object, int],
    expected: Iterable[int],
    convention: str,
) -> float:
    target = np.asarray(tuple(int(bit) for bit in expected), dtype=np.uint8)
    samples = counts_to_samples(counts, int(target.size), convention)
    if samples.size == 0:
        return 0.0
    return float(np.mean(samples == target[None, :]))


def choose_bitwise_convention(
    labeled_counts: dict[str, dict[object, int]],
    expected_patterns: dict[str, tuple[int, ...]],
) -> tuple[str, dict[str, float], float]:
    scores: dict[str, float] = {}
    for convention in ("qiskit", "left_to_right"):
        values = [
            bitwise_pattern_accuracy(
                labeled_counts[label],
                expected_patterns[label],
                convention,
            )
            for label in sorted(expected_patterns)
        ]
        scores[convention] = float(np.mean(values)) if values else 0.0
    winner = max(
        ("qiskit", "left_to_right"),
        key=lambda name: (scores[name], name == "qiskit"),
    )
    loser = "left_to_right" if winner == "qiskit" else "qiskit"
    return winner, scores, float(scores[winner] - scores[loser])


def analyze_calibration(
    raw_counts: dict[str, Any],
    protocol: dict[str, Any],
) -> CalibrationAnalysis:
    results = result_map(raw_counts)
    calibration = protocol["calibration"]
    labels = calibration["labels"]
    expected_patterns = {
        str(label): tuple(int(bit) for bit in pattern)
        for label, pattern in calibration["expected_patterns"].items()
    }
    missing = [label for label in labels if label not in results]
    if missing:
        raise ValueError(f"Missing calibration results: {missing}")
    labeled_counts = {
        label: results[label]["counts"]
        for label in labels
    }
    convention, scores, margin = choose_bitwise_convention(
        labeled_counts,
        expected_patterns,
    )
    width = len(calibration["logical_qubits"])
    all0 = counts_to_samples(
        results[calibration["all0_label"]]["counts"],
        width,
        convention,
    )
    all1 = counts_to_samples(
        results[calibration["all1_label"]]["counts"],
        width,
        convention,
    )
    if all0.shape[0] == 0 or all1.shape[0] == 0:
        raise ValueError("Calibration counts are empty.")
    p01 = all0.mean(axis=0)
    p10 = 1.0 - all1.mean(axis=0)
    determinants = 1.0 - p01 - p10
    thresholds = protocol["thresholds"]
    reasons: list[str] = []
    if scores[convention] < float(
        thresholds["minimum_convention_bit_accuracy"]
    ):
        reasons.append("convention_bit_accuracy_below_limit")
    if margin < float(thresholds["minimum_convention_margin"]):
        reasons.append("convention_margin_below_limit")
    minimum_determinant = float(
        thresholds["minimum_assignment_determinant"]
    )
    if np.any(determinants < minimum_determinant):
        reasons.append("assignment_determinant_below_limit")
    return CalibrationAnalysis(
        convention=convention,
        convention_scores={key: float(value) for key, value in scores.items()},
        convention_margin=margin,
        p01=tuple(float(value) for value in p01),
        p10=tuple(float(value) for value in p10),
        assignment_determinants=tuple(float(value) for value in determinants),
        passed=not reasons,
        reasons=tuple(reasons),
    )


def clopper_pearson_interval(
    errors: int,
    shots: int,
    alpha: float,
) -> tuple[float, float]:
    if shots <= 0:
        raise ValueError("shots must be positive")
    if not 0 <= errors <= shots:
        raise ValueError("errors must lie between zero and shots")
    lower = (
        0.0
        if errors == 0
        else float(beta.ppf(alpha / 2.0, errors, shots - errors + 1))
    )
    upper = (
        1.0
        if errors == shots
        else float(
            beta.ppf(
                1.0 - alpha / 2.0,
                errors + 1,
                shots - errors,
            )
        )
    )
    return lower, upper


def assignment_matrix(p01: float, p10: float) -> np.ndarray:
    return np.asarray(
        [
            [1.0 - p01, p10],
            [p01, 1.0 - p10],
        ],
        dtype=np.float64,
    )


def inverse_observable_coefficients(
    p01: list[float],
    p10: list[float],
) -> np.ndarray:
    if len(p01) != len(p10) or not p01:
        raise ValueError("Readout parameter lists must be non-empty and aligned.")
    matrix = assignment_matrix(p01[0], p10[0])
    for index in range(1, len(p01)):
        matrix = np.kron(
            matrix,
            assignment_matrix(p01[index], p10[index]),
        )
    width = len(p01)
    parity = np.asarray(
        [
            1.0 if int(state).bit_count() % 2 == 0 else -1.0
            for state in range(1 << width)
        ],
        dtype=np.float64,
    )
    return np.linalg.solve(matrix.T, parity)


def empirical_bernstein_lcb(
    values: np.ndarray,
    alpha: float,
) -> tuple[float, float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2:
        return float("-inf"), float("nan"), float("nan"), float("nan")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    mean = float(data.mean())
    variance = float(data.var(ddof=1))
    bound = float(np.max(np.abs(data), initial=0.0))
    log_term = math.log(3.0 / alpha)
    radius = math.sqrt(2.0 * variance * log_term / data.size)
    radius += 3.0 * (2.0 * bound) * log_term / data.size
    return mean - radius, mean, variance, bound


def _calibration_samples(
    calibration_raw: dict[str, Any],
    protocol: dict[str, Any],
    convention: str,
) -> tuple[np.ndarray, np.ndarray]:
    results = result_map(calibration_raw)
    calibration = protocol["calibration"]
    width = len(calibration["logical_qubits"])
    all0 = counts_to_samples(
        results[calibration["all0_label"]]["counts"],
        width,
        convention,
    )
    all1 = counts_to_samples(
        results[calibration["all1_label"]]["counts"],
        width,
        convention,
    )
    return all0, all1


def _calibration_parameter_intervals(
    all0: np.ndarray,
    all1: np.ndarray,
    alpha_calibration: float,
) -> tuple[list[tuple[float, float]], list[dict[str, Any]]]:
    width = int(all0.shape[1])
    parameter_count = 2 * width
    per_parameter_alpha = alpha_calibration / parameter_count
    intervals: list[tuple[float, float]] = []
    details: list[dict[str, Any]] = []
    for position in range(width):
        k01 = int(all0[:, position].sum())
        k10 = int((1 - all1[:, position]).sum())
        p01_interval = clopper_pearson_interval(
            k01,
            int(all0.shape[0]),
            per_parameter_alpha,
        )
        p10_interval = clopper_pearson_interval(
            k10,
            int(all1.shape[0]),
            per_parameter_alpha,
        )
        intervals.extend([p01_interval, p10_interval])
        details.append(
            {
                "position": position,
                "p01_errors": k01,
                "p01_shots": int(all0.shape[0]),
                "p01_estimate": float(k01 / all0.shape[0]),
                "p01_interval": list(p01_interval),
                "p10_errors": k10,
                "p10_shots": int(all1.shape[0]),
                "p10_estimate": float(k10 / all1.shape[0]),
                "p10_interval": list(p10_interval),
            }
        )
    return intervals, details


def _feature_values(
    science_samples: np.ndarray,
    protocol: dict[str, Any],
    parameter_vector: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    calibration_qubits = [
        int(value) for value in protocol["calibration"]["logical_qubits"]
    ]
    calibration_positions = {
        logical: position
        for position, logical in enumerate(calibration_qubits)
    }
    p01 = parameter_vector[0::2]
    p10 = parameter_vector[1::2]
    witness_values = np.zeros(science_samples.shape[0], dtype=np.float64)
    feature_details: list[dict[str, Any]] = []
    for feature in protocol["witness"]["features"]:
        support = [int(value) for value in feature["support"]]
        positions = [calibration_positions[value] for value in support]
        local_p01 = [float(p01[position]) for position in positions]
        local_p10 = [float(p10[position]) for position in positions]
        coefficients = inverse_observable_coefficients(local_p01, local_p10)
        local_bits = science_samples[:, support].astype(np.int64)
        powers = 1 << np.arange(len(support) - 1, -1, -1, dtype=np.int64)
        indices = local_bits @ powers
        values = coefficients[indices]
        weight = float(feature["weight"])
        witness_values += weight * values
        raw_parity = 1.0 - 2.0 * (
            np.mod(local_bits.sum(axis=1), 2).astype(np.float64)
        )
        feature_details.append(
            {
                "feature_index": int(feature["feature_index"]),
                "feature_name": str(feature["feature_name"]),
                "support": support,
                "weight": weight,
                "raw_expectation": float(raw_parity.mean()),
                "mitigated_expectation": float(values.mean()),
                "inverse_coefficients": coefficients.tolist(),
                "coefficient_bound": float(
                    np.max(np.abs(coefficients), initial=0.0)
                ),
            }
        )
    return witness_values, feature_details


def certify_mitigated_counts(
    *,
    science_raw: dict[str, Any],
    calibration_raw: dict[str, Any],
    protocol: dict[str, Any],
    convention: str,
    label: str = "science",
) -> dict[str, Any]:
    science_results = result_map(science_raw)
    if label not in science_results:
        raise ValueError(f"Missing science result {label!r}.")
    n = int(protocol["n"])
    samples = counts_to_samples(
        science_results[label]["counts"],
        n,
        convention,
    )
    if samples.shape[0] == 0:
        raise ValueError("Science counts are empty.")
    all0, all1 = _calibration_samples(
        calibration_raw,
        protocol,
        convention,
    )
    alpha_calibration = float(protocol["confidence"]["alpha_calibration"])
    alpha_science = float(protocol["confidence"]["alpha_science"])
    intervals, interval_details = _calibration_parameter_intervals(
        all0,
        all1,
        alpha_calibration,
    )
    point_parameters = np.empty(2 * all0.shape[1], dtype=np.float64)
    for position in range(all0.shape[1]):
        point_parameters[2 * position] = float(all0[:, position].mean())
        point_parameters[2 * position + 1] = float(
            1.0 - all1[:, position].mean()
        )

    minimum_determinant = float(
        protocol["thresholds"]["minimum_assignment_determinant"]
    )

    def lower_bound(parameter_vector: np.ndarray) -> float:
        p01 = parameter_vector[0::2]
        p10 = parameter_vector[1::2]
        if np.any(1.0 - p01 - p10 < minimum_determinant):
            return -1.0e9
        try:
            values, _ = _feature_values(
                samples,
                protocol,
                parameter_vector,
            )
        except np.linalg.LinAlgError:
            return -1.0e9
        lcb, _, _, _ = empirical_bernstein_lcb(values, alpha_science)
        return float(lcb)

    point_values, point_features = _feature_values(
        samples,
        protocol,
        point_parameters,
    )
    point_lcb, point_mean, point_variance, point_bound = empirical_bernstein_lcb(
        point_values,
        alpha_science,
    )

    corner_values: list[float] = []
    corner_parameters: list[list[float]] = []
    for choices in product((0, 1), repeat=len(intervals)):
        parameters = [
            intervals[index][choice]
            for index, choice in enumerate(choices)
        ]
        value = lower_bound(np.asarray(parameters, dtype=np.float64))
        corner_values.append(value)
        corner_parameters.append(parameters)
    corner_index = int(np.argmin(corner_values))
    worst_lcb = float(corner_values[corner_index])
    worst_parameters = np.asarray(
        corner_parameters[corner_index],
        dtype=np.float64,
    )
    optimization = {
        "used": False,
        "success": None,
        "message": None,
        "fun": None,
        "x": None,
    }
    if bool(protocol["confidence"].get("continuous_box_optimization", True)):
        result = differential_evolution(
            lower_bound,
            intervals,
            seed=int(protocol["confidence"].get("optimization_seed", 20260725)),
            popsize=int(protocol["confidence"].get("optimization_popsize", 20)),
            maxiter=int(protocol["confidence"].get("optimization_maxiter", 500)),
            tol=float(protocol["confidence"].get("optimization_tolerance", 1.0e-10)),
            polish=True,
            workers=1,
            updating="immediate",
        )
        optimization = {
            "used": True,
            "success": bool(result.success),
            "message": str(result.message),
            "fun": float(result.fun),
            "x": [float(value) for value in result.x],
        }
        if float(result.fun) < worst_lcb:
            worst_lcb = float(result.fun)
            worst_parameters = np.asarray(result.x, dtype=np.float64)

    safety_epsilon = float(
        protocol["confidence"].get("optimization_safety_epsilon", 0.0)
    )
    worst_lcb -= safety_epsilon
    adversary_supremum = float(protocol["witness"]["adversary_supremum"])
    generalization_penalty = float(
        protocol["witness"]["adversary_generalization_penalty"]
    )
    margin_lcb = (
        worst_lcb
        - adversary_supremum
        - generalization_penalty
    )
    threshold = float(protocol["thresholds"]["minimum_margin_lcb"])
    required_shots = int(protocol["shots"][science_raw.get("stage", "diagnostic")])
    shots_received = int(science_results[label]["shots_received"])
    passed = bool(
        shots_received >= required_shots
        and margin_lcb > threshold
    )
    return {
        "schema": "codegap.qpu-mitigated-minimax-certificate.v1",
        "protocol": str(protocol["protocol"]),
        "stage": str(science_raw.get("stage")),
        "label": label,
        "convention": convention,
        "shots_received": shots_received,
        "required_shots": required_shots,
        "raw_counts_only": False,
        "readout_mitigation_used": True,
        "calibration_independent_of_science": True,
        "point_estimate": {
            "observed_mean": point_mean,
            "observed_lcb_conditional": point_lcb,
            "variance": point_variance,
            "estimator_bound": point_bound,
            "parameters": [float(value) for value in point_parameters],
            "features": point_features,
        },
        "calibration_confidence": {
            "alpha": alpha_calibration,
            "interval_method": "simultaneous Bonferroni Clopper-Pearson",
            "parameters": interval_details,
        },
        "science_confidence": {
            "alpha": alpha_science,
            "method": "empirical Bernstein with actual inverse-estimator bound",
        },
        "worst_case_calibration_box": {
            "corner_count": len(corner_values),
            "worst_lcb_before_adversary": worst_lcb,
            "worst_parameters": [float(value) for value in worst_parameters],
            "continuous_optimization": optimization,
            "optimization_safety_epsilon": safety_epsilon,
        },
        "observed_lcb": worst_lcb,
        "adversary_supremum_training": adversary_supremum,
        "adversary_generalization_penalty": generalization_penalty,
        "margin_lcb": float(margin_lcb),
        "minimum_margin_lcb": threshold,
        "acceptance_rule": "simultaneous-confidence margin_lcb > minimum_margin_lcb",
        "pass": passed,
        "claim_boundary": str(protocol["claim_boundary"]),
    }
