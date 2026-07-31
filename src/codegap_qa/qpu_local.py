from __future__ import annotations

from math import log, sqrt
from statistics import NormalDist
from typing import Any

import numpy as np

from .qpu_counts import counts_to_samples


LOCAL_PROTOCOL_MODE = "local_witness_v1"


def _result_map(payload: dict) -> dict[str, dict]:
    return {str(item["label"]): item for item in payload["results"]}


def _wilson_interval(successes: int, shots: int, alpha: float) -> tuple[float, float]:
    if shots <= 0:
        return 0.0, 1.0
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1).")
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p = successes / shots
    denominator = 1.0 + z * z / shots
    center = (p + z * z / (2.0 * shots)) / denominator
    radius = (
        z
        * sqrt(
            p * (1.0 - p) / shots
            + z * z / (4.0 * shots * shots)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _expected_pattern_bit_accuracy(
    counts: dict[object, int],
    expected: list[int] | tuple[int, ...],
    convention: str,
) -> float:
    expected_vector = np.asarray(expected, dtype=np.uint8)
    samples = counts_to_samples(counts, expected_vector.size, convention)
    if samples.size == 0:
        return 0.0
    return float(np.mean(samples == expected_vector[None, :]))


def _choose_bitwise_convention(
    labeled_counts: dict[str, dict[object, int]],
    expected_patterns: dict[str, list[int] | tuple[int, ...]],
) -> tuple[str, dict[str, float], float]:
    scores: dict[str, float] = {}
    for convention in ("qiskit", "left_to_right"):
        values = [
            _expected_pattern_bit_accuracy(
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
    margin = float(scores[winner] - scores[loser])
    return winner, scores, margin


def _all_zero_survival(samples: np.ndarray, positions: list[int] | None = None) -> tuple[int, int, float]:
    if samples.ndim != 2:
        raise ValueError("samples must be a matrix.")
    selected = samples if positions is None else samples[:, positions]
    shots = int(selected.shape[0])
    successes = int(np.count_nonzero(np.all(selected == 0, axis=1)))
    survival = successes / shots if shots else 0.0
    return successes, shots, float(survival)


def analyze_local_probes(
    manifest: dict,
    raw_counts: dict,
    *,
    protocol_config: dict,
) -> dict[str, Any]:
    if manifest.get("protocol") != LOCAL_PROTOCOL_MODE:
        raise ValueError("Manifest is not a local-witness probe bundle.")

    results = _result_map(raw_counts)
    calibration = manifest["calibration"]
    calibration_width = len(calibration["logical_qubits"])
    basis_items = [
        item
        for item in manifest["circuits"]
        if item["type"] == "local_basis"
    ]
    expected = {
        item["label"]: [int(bit) for bit in item["expected_pattern"]]
        for item in basis_items
    }
    labeled_counts = {
        label: results[label]["counts"]
        for label in expected
    }
    convention, convention_scores, convention_margin = _choose_bitwise_convention(
        labeled_counts,
        expected,
    )

    all0 = counts_to_samples(
        results["local_basis_all0"]["counts"],
        calibration_width,
        convention,
    )
    all1 = counts_to_samples(
        results["local_basis_all1"]["counts"],
        calibration_width,
        convention,
    )
    p01 = all0.mean(axis=0)
    p10 = 1.0 - all1.mean(axis=0)
    determinants = 1.0 - p01 - p10

    checkerboard_errors: dict[str, float] = {}
    for label in ("local_basis_even", "local_basis_odd"):
        samples = counts_to_samples(
            results[label]["counts"],
            calibration_width,
            convention,
        )
        target = np.asarray(expected[label], dtype=np.uint8)
        checkerboard_errors[label] = float(
            np.mean(samples != target[None, :])
        )

    thresholds = protocol_config["probe_acceptance"]
    alpha = float(thresholds["alpha"])
    feature_count = max(1, len(manifest["active_features"]))
    interval_count = max(1, 2 * calibration_width + 2 * feature_count)
    interval_alpha = alpha / interval_count

    feature_rows: list[dict[str, Any]] = []
    for feature in manifest["active_features"]:
        feature_index = int(feature["feature_index"])
        positions = [int(value) for value in feature["calibration_positions"]]
        baseline_successes, baseline_shots, baseline_survival = _all_zero_survival(
            all0,
            positions,
        )
        baseline_lcb, baseline_ucb = _wilson_interval(
            baseline_successes,
            baseline_shots,
            interval_alpha,
        )

        echoes: list[dict[str, Any]] = []
        for repetitions in manifest["echo_repetitions"]:
            label = f"local_echo_f{feature_index:02d}_x{int(repetitions)}"
            item = results[label]
            width = int(item.get("classical_bits") or len(feature["lightcone"]))
            samples = counts_to_samples(
                item["counts"],
                width,
                convention,
            )
            successes, shots, survival = _all_zero_survival(samples)
            lcb, ucb = _wilson_interval(successes, shots, interval_alpha)
            echoes.append(
                {
                    "label": label,
                    "repetitions": int(repetitions),
                    "shots": shots,
                    "survival": survival,
                    "survival_lcb": lcb,
                    "survival_ucb": ucb,
                }
            )

        terminal = max(echoes, key=lambda row: row["repetitions"])
        mirror_drop_ucb = max(
            0.0,
            float(baseline_ucb) - float(terminal["survival_lcb"]),
        )
        minimum_determinant = float(
            np.min(determinants[positions], initial=1.0)
        )
        feature_reasons: list[str] = []
        if minimum_determinant < float(
            thresholds["minimum_assignment_determinant"]
        ):
            feature_reasons.append("local_assignment_matrix_ill_conditioned")
        if float(terminal["survival_lcb"]) < float(
            thresholds["minimum_local_mirror_survival_lcb"]
        ):
            feature_reasons.append("local_mirror_survival_below_limit")
        if mirror_drop_ucb > float(
            thresholds["maximum_local_mirror_drop_ucb"]
        ):
            feature_reasons.append("local_mirror_drop_above_limit")

        # Diagnostic only. The acceptance rule deliberately does not assume
        # an exact exponential decay model.
        x = np.asarray([0.0] + [float(row["repetitions"]) for row in echoes])
        y = np.log(
            np.clip(
                np.asarray(
                    [baseline_survival]
                    + [float(row["survival"]) for row in echoes],
                    dtype=np.float64,
                ),
                1e-15,
                1.0,
            )
        )
        design = np.column_stack([np.ones_like(x), x])
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        fitted = np.exp(intercept + slope * x)
        model_mismatch = float(
            np.max(
                np.abs(
                    fitted
                    - np.asarray(
                        [baseline_survival]
                        + [float(row["survival"]) for row in echoes]
                    )
                )
            )
        )

        feature_rows.append(
            {
                "feature_index": feature_index,
                "feature_name": feature["feature_name"],
                "weight": float(feature["weight"]),
                "support": [int(value) for value in feature["support"]],
                "lightcone": [int(value) for value in feature["lightcone"]],
                "physical_qubits": [
                    int(value) for value in feature["physical_qubits"]
                ],
                "baseline_survival": baseline_survival,
                "baseline_survival_lcb": baseline_lcb,
                "baseline_survival_ucb": baseline_ucb,
                "minimum_assignment_determinant": minimum_determinant,
                "echoes": echoes,
                "terminal_mirror_survival_lcb": float(
                    terminal["survival_lcb"]
                ),
                "mirror_drop_ucb": mirror_drop_ucb,
                "exponential_fit_slope": float(slope),
                "exponential_fit_mismatch": model_mismatch,
                "passed": not feature_reasons,
                "reasons": feature_reasons,
            }
        )

    reasons: list[str] = []
    winner_score = float(convention_scores[convention])
    if winner_score < float(thresholds["minimum_convention_bit_accuracy"]):
        reasons.append("bitwise_convention_accuracy_below_limit")
    if convention_margin < float(thresholds["minimum_convention_margin"]):
        reasons.append("bitwise_convention_margin_below_limit")
    if any(not row["passed"] for row in feature_rows):
        reasons.append("local_feature_probe_failed")

    passed = not reasons
    local_noise_score = 1.0 - min(
        (
            float(row["terminal_mirror_survival_lcb"])
            for row in feature_rows
        ),
        default=0.0,
    )
    return {
        "schema": "codegap.qpu-local-probe-analysis.v1",
        "protocol": LOCAL_PROTOCOL_MODE,
        "candidate_id": str(manifest.get("candidate_id", "unknown")),
        "convention": convention,
        "convention_bit_accuracy": winner_score,
        "convention_scores": convention_scores,
        "convention_margin": convention_margin,
        "calibration": {
            "logical_qubits": [
                int(value) for value in calibration["logical_qubits"]
            ],
            "physical_qubits": [
                int(value) for value in calibration["physical_qubits"]
            ],
            "p01": [float(value) for value in p01],
            "p10": [float(value) for value in p10],
            "assignment_determinants": [
                float(value) for value in determinants
            ],
            "minimum_assignment_determinant": float(
                np.min(determinants, initial=1.0)
            ),
            "checkerboard_errors": checkerboard_errors,
        },
        "active_features": feature_rows,
        "local_noise_score": local_noise_score,
        "passed_integrity": passed,
        "qpu_noise_gate_pass": passed,
        "reasons": reasons,
        "acceptance_rule": (
            "bitwise convention accuracy and separation; stable local "
            "assignment matrices; and simultaneous-confidence lower bounds "
            "on isolated exact-lightcone mirror survival. No global 48-bit "
            "all-zero survival or full-distribution TV proxy is used."
        ),
        "claim_boundary": manifest["claim_boundary"],
        "thresholds": thresholds,
    }


