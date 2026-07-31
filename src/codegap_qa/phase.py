from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from .attacks import build_attack_suite
from .features import FeatureMap, build_feature_map
from .iqp import IQPSpec, exact_probabilities, sample_probabilities
from .lightcone import analytic_feature_means
from .noise import NoisePoint, apply_noise
from .progress import ProgressManager, default_progress
from .witness import (
    bounded_mean_lcb,
    certificate_margin_lcb,
    fit_configured_feature_bounds,
    fit_minimax_witness,
    tv_robust_radius,
)


def feature_names(feature_map: FeatureMap) -> tuple[str, ...]:
    names = [f"bit_z_{index}" for index in feature_map.resolved_bit_indices]
    names.extend(
        f"parity_{index}" for index in range(feature_map.parity_masks.shape[0])
    )
    names.extend(f"heavy_{index}" for index in feature_map.heavy_indices)
    if feature_map.include_centered_weight:
        names.append("centered_hamming_weight")
    return tuple(names)


def _write_phase(frame: pd.DataFrame, output: Path) -> Path:
    parquet_path = output / "hardware_phase_diagram.parquet"
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ValueError):
        csv_path = output / "hardware_phase_diagram.csv"
        frame.to_csv(csv_path, index=False)
        return csv_path


def _envelope_summary(frame: pd.DataFrame, config: dict) -> dict:
    envelope = config["gates"].get("noise_envelope", {})
    required_mask = np.ones(len(frame), dtype=bool)
    for column, maximum in envelope.items():
        if column not in frame.columns:
            raise ValueError(f"Unknown noise-envelope coordinate: {column}")
        required_mask &= frame[column].to_numpy(dtype=float) <= float(maximum)
    required_rows = frame[required_mask]
    return {
        "required_noise_envelope": envelope,
        "required_envelope_grid_points": int(len(required_rows)),
        "required_envelope_min_pass_probability": (
            float(required_rows["pass_probability"].min())
            if not required_rows.empty
            else 0.0
        ),
        "required_envelope_min_margin_lcb": (
            float(required_rows["mean_margin_lcb"].min())
            if not required_rows.empty
            else float("-inf")
        ),
    }


def _degree(circuit_spec: dict) -> np.ndarray:
    n = int(circuit_spec["n"])
    degree = np.zeros(n, dtype=np.float64)
    for gate in circuit_spec["gates"]:
        qubits = gate["qubits"]
        if len(qubits) == 2:
            degree[int(qubits[0])] += 1.0
            degree[int(qubits[1])] += 1.0
    return degree


def _point_error_probabilities(
    point: NoisePoint,
    degree: np.ndarray,
    *,
    multiplier: float = 1.0,
) -> np.ndarray:
    normalized = degree / max(1.0, degree.max(initial=1.0))
    coherent = math.sin(point.coherent_angle / 2.0) ** 2
    probability = (
        point.p_1q
        + point.p_measure
        + point.p_2q * normalized
        + coherent * normalized
        + 2.0 * point.crosstalk
        + point.drift
    )
    return np.clip(multiplier * probability, 0.0, 0.49)


def _analytic_adversary_bounds(
    fmap: FeatureMap,
    config: dict,
) -> dict[str, np.ndarray]:
    settings = config.get("certificate", {}).get("analytic_adversaries", {})
    bit_bias = float(settings.get("max_abs_bit_bias", 0.15))
    local_correlation = float(settings.get("max_abs_local_correlation", 0.20))
    parity_weights = fmap.parity_masks.sum(axis=1).astype(int)

    uniform = np.zeros(fmap.dimension, dtype=np.float64)
    product_parts = [
        np.full(len(fmap.resolved_bit_indices), bit_bias),
        np.asarray([bit_bias ** max(1, weight) for weight in parity_weights]),
    ]
    local_parts = [
        np.full(len(fmap.resolved_bit_indices), bit_bias),
        np.asarray(
            [
                local_correlation ** max(1, math.ceil(weight / 2))
                for weight in parity_weights
            ]
        ),
    ]
    if fmap.include_centered_weight:
        product_parts.append(np.asarray([bit_bias]))
        local_parts.append(np.asarray([bit_bias]))
    product_bounds = np.concatenate(product_parts)
    local_bounds = np.concatenate(local_parts)
    return {
        "uniform": uniform,
        "bounded_product": product_bounds,
        "bounded_local_correlation": local_bounds,
    }


def _evaluate_lightcone_phase(
    candidate: dict,
    config: dict,
    output: Path,
    manager: ProgressManager,
) -> dict:
    exact = candidate["exact_artifacts"]
    circuit_spec = exact["circuit_spec"]
    n = int(candidate["code"]["n"])
    selection = circuit_spec.get("verifier_selection") or {}
    selected_mode = bool(selection.get("selected_masks"))
    masks = np.asarray(
        selection.get("selected_masks", circuit_spec["verifier_masks"]),
        dtype=np.uint8,
    ).reshape(-1, n)
    maximum_parities = int(config["certificate"].get("max_parity_features", 64))
    masks = masks[:maximum_parities]
    max_cone = int(config["certificate"].get("max_lightcone_qubits", 18))
    manager.write(
        f"NoiseCert {candidate['candidate_id']}: exact selected light cones "
        f"(n={n}, masks={len(masks)}, max={max_cone})"
    )
    exact_backend = str(
        config.get("verifier_search", {}).get("exact_backend", "auto")
    )
    analytic = analytic_feature_means(
        circuit_spec,
        masks,
        max_lightcone_qubits=max_cone,
        bit_indices=() if selected_mode else None,
        include_centered_weight=not selected_mode,
        backend=exact_backend,
        require_all_bits=not selected_mode,
    )
    fmap = FeatureMap(
        parity_masks=analytic["parity_masks"],
        heavy_indices=(),
        n=n,
        bit_indices=tuple(analytic.get("bit_indices", [])) if selected_mode else None,
        include_centered_weight=bool(analytic.get("include_centered_weight", True)),
    )
    ideal_means = np.asarray(analytic["feature_means"], dtype=np.float64)
    adversary_bounds = _analytic_adversary_bounds(fmap, config)
    witness, witness_optimizer = fit_configured_feature_bounds(
        ideal_means,
        adversary_bounds,
        feature_names=feature_names(fmap),
        settings=config.get("verifier_search", {}),
    )
    penalty = float(config["certificate"]["adversary_generalization_penalty"])
    witness_payload = {
        "schema": "codegap.minimax-witness.v2-lightcone",
        "candidate_id": candidate["candidate_id"],
        "feature_map": {
            "parity_masks": fmap.parity_masks.tolist(),
            "heavy_indices": [],
            "n": fmap.n,
            "bit_indices": list(fmap.resolved_bit_indices),
            "include_centered_weight": fmap.include_centered_weight,
        },
        "witness": witness.to_dict(),
        "witness_optimizer": witness_optimizer,
        "adversary_generalization_penalty": penalty,
        "ideal_observables": {
            "method": analytic["method"],
            "maximum_lightcone_qubits": analytic["maximum_lightcone_qubits"],
            "feature_means": ideal_means.tolist(),
            "parity_results": analytic["parity_results"],
        },
        "analytic_adversary_bounds": {
            name: values.tolist() for name, values in adversary_bounds.items()
        },
        "verifier_selection": selection if selected_mode else None,
        "witness_optimizer": witness_optimizer,
        "unbounded_attack_warning": (
            "The analytic witness does not bound unrestricted affine-subspace "
            "or arbitrary tensor-network spoofers; those remain in Gate C's "
            "registered empirical/conditional attack scope."
        ),
    }
    (output / "robust_witness.json").write_text(
        json.dumps(witness_payload, indent=2), encoding="utf-8"
    )

    degree = _degree(circuit_spec)
    parity_weights = fmap.parity_masks.astype(np.float64)
    grid = config["noise"]["grid"]
    points = list(
        product(
            grid["p_1q"],
            grid["p_2q"],
            grid["p_measure"],
            grid["coherent_angle"],
            grid["crosstalk"],
            grid["drift"],
        )
    )
    repeats = int(config["noise"]["repeats"])
    shots = int(config["noise"]["test_shots"])
    alpha = float(config["certificate"]["alpha"])
    sigma = float(config["certificate"].get("analytic_noise_parameter_sigma", 0.05))
    rng = np.random.default_rng(int(config["seed"]) + 404)
    bar = manager.bar(
        total=len(points) * repeats,
        desc=f"NoiseCert {candidate['candidate_id']}: analytic phase diagram",
        unit="repeat",
        leave=True,
    )
    rows = []
    for values in points:
        point = NoisePoint(*map(float, values))
        outcomes = []
        lcbs = []
        for _ in range(repeats):
            multiplier = max(0.0, float(rng.normal(1.0, sigma)))
            p = _point_error_probabilities(point, degree, multiplier=multiplier)
            bit_attenuation = 1.0 - 2.0 * p
            parity_attenuation = (
                np.prod(
                    np.where(
                        parity_weights > 0,
                        bit_attenuation[None, :],
                        1.0,
                    ),
                    axis=1,
                )
                if fmap.parity_masks.shape[0]
                else np.empty(0, dtype=np.float64)
            )
            bit_count = len(fmap.resolved_bit_indices)
            bit_component = (
                ideal_means[:bit_count]
                * bit_attenuation[np.asarray(fmap.resolved_bit_indices, dtype=int)]
                if bit_count
                else np.empty(0, dtype=np.float64)
            )
            parity_start = bit_count
            parity_component = (
                ideal_means[
                    parity_start : parity_start + fmap.parity_masks.shape[0]
                ]
                * parity_attenuation
            )
            components = [bit_component, parity_component]
            if fmap.include_centered_weight:
                components.append(
                    np.asarray(
                        [
                            -float(np.mean(bit_component))
                            if len(bit_component)
                            else 0.0
                        ]
                    )
                )
            noisy_means = np.concatenate(components)
            witness_mean = float(noisy_means @ witness.weights)
            observed_lcb = bounded_mean_lcb(witness_mean, shots, alpha)
            margin_lcb = (
                observed_lcb
                - max(witness.adversary_means.values(), default=0.0)
                - penalty
            )
            outcomes.append(margin_lcb > 0.0)
            lcbs.append(margin_lcb)
            bar.update(1)
        probability = float(np.mean(outcomes))
        rows.append(
            point.to_dict()
            | {
                "pass_probability": probability,
                "mean_margin_lcb": float(np.mean(lcbs)),
                "region": (
                    "robust"
                    if probability >= 0.99
                    else "collapsed"
                    if probability <= 0.05
                    else "critical"
                ),
            }
        )
    bar.close()
    frame = pd.DataFrame(rows)
    phase_path = _write_phase(frame, output)
    ideal_lcb = bounded_mean_lcb(witness.ideal_mean, shots, alpha)
    ideal_certificate = {
        "shots": shots,
        "observed_mean": witness.ideal_mean,
        "observed_lcb": ideal_lcb,
        "adversary_supremum_training": max(
            witness.adversary_means.values(), default=0.0
        ),
        "adversary_generalization_penalty": penalty,
        "margin_lcb": (
            ideal_lcb
            - max(witness.adversary_means.values(), default=0.0)
            - penalty
        ),
    }
    ideal_certificate["pass"] = ideal_certificate["margin_lcb"] > 0.0
    robust_rows = frame[frame["pass_probability"] >= 0.99]
    result = {
        "schema": "codegap.noise-cert.v2-lightcone",
        "candidate_id": candidate["candidate_id"],
        "ideal_certificate": ideal_certificate,
        "training_margin": witness.training_margin,
        "tv_robust_radius": tv_robust_radius(witness.training_margin),
        "maximum_observed_pass_probability": (
            float(frame["pass_probability"].max()) if len(frame) else 0.0
        ),
        "minimum_observed_pass_probability": (
            float(frame["pass_probability"].min()) if len(frame) else 0.0
        ),
        "robust_grid_points": int((frame["pass_probability"] >= 0.99).sum()),
        "total_grid_points": int(len(frame)),
        "worst_robust_point": (
            robust_rows.sort_values("mean_margin_lcb").iloc[0].to_dict()
            if not robust_rows.empty
            else {}
        ),
        "phase_diagram": str(phase_path),
        "ideal_method": analytic["method"],
        "maximum_lightcone_qubits": analytic["maximum_lightcone_qubits"],
        "verifier_selection": selection if selected_mode else None,
        "coherent_noise_note": (
            "coherent_angle is converted to sin^2(delta/2) in a conservative "
            "local error envelope. Device-specific coherent claims require "
            "the QPU probe stage."
        ),
    }
    result.update(_envelope_summary(frame, config))
    (output / "finite_shot_bounds.json").write_text(
        json.dumps(result, indent=2, default=float), encoding="utf-8"
    )
    return result


def _evaluate_legacy_exact_phase(
    candidate: dict,
    config: dict,
    output: Path,
    manager: ProgressManager,
) -> dict:
    exact = candidate["exact_artifacts"]
    h_x = np.asarray(exact["h_x"], dtype=np.uint8)
    h_z = np.asarray(exact["h_z"], dtype=np.uint8)
    edges = tuple(tuple(edge) for edge in exact["logical_edges"])
    n = int(candidate["code"]["n"])
    spec = IQPSpec(
        n=n,
        edges=edges,
        theta_single=config["circuit"]["theta_single"],
        theta_pair=config["circuit"]["theta_pair"],
        seed=config["seed"],
    )
    probabilities = exact_probabilities(
        spec,
        max_qubits=config["circuit"]["max_exact_qubits"],
        progress=manager,
    )
    train_shots = int(config["noise"]["training_shots"])
    test_shots = int(config["noise"]["test_shots"])
    ideal_training = sample_probabilities(probabilities, train_shots, config["seed"] + 11)
    ideal_test = sample_probabilities(probabilities, test_shots, config["seed"] + 12)
    attacks = build_attack_suite(ideal_training, h_x, train_shots, config["seed"] + 100)
    fmap = build_feature_map(
        h_x,
        h_z,
        probabilities,
        heavy_count=config["certificate"]["heavy_features"],
        max_parities=config["certificate"]["max_parity_features"],
    )
    witness = fit_minimax_witness(
        fmap.transform(ideal_training),
        {
            attack.name: fmap.transform(attack.samples)
            for attack in attacks
            if attack.name != "corrupted_ideal_diagnostic"
        },
        feature_names=feature_names(fmap),
    )
    penalty = float(config["certificate"]["adversary_generalization_penalty"])
    (output / "robust_witness.json").write_text(
        json.dumps(
            {
                "schema": "codegap.minimax-witness.v1",
                "candidate_id": candidate["candidate_id"],
                "feature_map": {
                    "parity_masks": fmap.parity_masks.tolist(),
                    "heavy_indices": list(fmap.heavy_indices),
                    "n": fmap.n,
                },
                "witness": witness.to_dict(),
                "adversary_generalization_penalty": penalty,
                "attacks": [
                    {"name": attack.name, "metadata": attack.metadata}
                    for attack in attacks
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    degree = np.zeros(n, dtype=np.float64)
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    grid = config["noise"]["grid"]
    points = list(product(*[grid[key] for key in (
        "p_1q", "p_2q", "p_measure", "coherent_angle", "crosstalk", "drift"
    )]))
    repeats = int(config["noise"]["repeats"])
    rows = []
    bar = manager.bar(
        total=len(points) * repeats,
        desc=f"NoiseCert {candidate['candidate_id']}: exact phase diagram",
        unit="repeat",
        leave=True,
    )
    for values in points:
        point = NoisePoint(*map(float, values))
        outcomes, lcbs = [], []
        for repeat in range(repeats):
            noisy = apply_noise(
                ideal_test,
                point,
                degree,
                seed=config["seed"] + 10000 + len(rows) * 97 + repeat,
            )
            certificate = certificate_margin_lcb(
                witness,
                fmap.transform(noisy),
                alpha=config["certificate"]["alpha"],
                adversary_generalization_penalty=penalty,
            )
            outcomes.append(bool(certificate["pass"]))
            lcbs.append(float(certificate["margin_lcb"]))
            bar.update(1)
        probability = float(np.mean(outcomes))
        rows.append(point.to_dict() | {
            "pass_probability": probability,
            "mean_margin_lcb": float(np.mean(lcbs)),
            "region": "robust" if probability >= .99 else "collapsed" if probability <= .05 else "critical",
        })
    bar.close()
    frame = pd.DataFrame(rows)
    phase_path = _write_phase(frame, output)
    ideal_certificate = certificate_margin_lcb(
        witness,
        fmap.transform(ideal_test),
        alpha=config["certificate"]["alpha"],
        adversary_generalization_penalty=penalty,
    )
    result = {
        "schema": "codegap.noise-cert.v1",
        "candidate_id": candidate["candidate_id"],
        "ideal_certificate": ideal_certificate,
        "training_margin": witness.training_margin,
        "tv_robust_radius": tv_robust_radius(witness.training_margin),
        "maximum_observed_pass_probability": float(frame["pass_probability"].max()),
        "minimum_observed_pass_probability": float(frame["pass_probability"].min()),
        "robust_grid_points": int((frame["pass_probability"] >= .99).sum()),
        "total_grid_points": int(len(frame)),
        "phase_diagram": str(phase_path),
    }
    result.update(_envelope_summary(frame, config))
    (output / "finite_shot_bounds.json").write_text(
        json.dumps(result, indent=2, default=float), encoding="utf-8"
    )
    return result


def evaluate_noise_phase(
    candidate: dict,
    config: dict,
    output: Path,
    progress: ProgressManager | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manager = progress or default_progress(config)
    if "circuit_spec" in candidate.get("exact_artifacts", {}):
        return _evaluate_lightcone_phase(candidate, config, output, manager)
    return _evaluate_legacy_exact_phase(candidate, config, output, manager)
