from __future__ import annotations

from itertools import combinations
import math
from typing import Any

import numpy as np

from .gf2 import rank
from .lightcone import local_z_expectation
from .verifier_gpu import batch_backward_lightcones
from .witness import bounded_mean_lcb, fit_configured_feature_bounds


def _unique_masks(masks: list[np.ndarray], n: int) -> np.ndarray:
    unique: dict[bytes, np.ndarray] = {}
    for mask in masks:
        value = np.asarray(mask, dtype=np.uint8).reshape(n) & 1
        if not value.any():
            continue
        unique.setdefault(value.tobytes(), value)
    return (
        np.vstack(list(unique.values())).astype(np.uint8)
        if unique
        else np.empty((0, n), dtype=np.uint8)
    )


def _protected_templates(circuit_spec: dict[str, Any]) -> list[np.ndarray]:
    n = int(circuit_spec["n"])
    templates = []
    metadata = circuit_spec.get("schedule_metadata", {})
    protected = metadata.get("protected_masks") or []
    for mask in protected:
        templates.append(np.asarray(mask, dtype=np.uint8).reshape(n))
    schedule = circuit_spec.get("schedule", {})
    for mask in schedule.get("protected_masks", []):
        templates.append(np.asarray(mask, dtype=np.uint8).reshape(n))
    return templates


def generate_observable_candidates(
    circuit_spec: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    n = int(circuit_spec["n"])
    max_candidates = int(settings.get("max_candidate_masks", 512))
    max_weight = int(settings.get("candidate_max_weight", 8))
    base = np.asarray(circuit_spec.get("verifier_masks", []), dtype=np.uint8)
    if base.size == 0:
        base = np.empty((0, n), dtype=np.uint8)
    base = base.reshape(-1, n)
    masks: list[np.ndarray] = []
    sources: list[str] = []

    def add(mask: np.ndarray, source: str) -> None:
        value = np.asarray(mask, dtype=np.uint8).reshape(n) & 1
        weight = int(value.sum())
        if 0 < weight <= max_weight:
            masks.append(value)
            sources.append(source)

    for mask in _protected_templates(circuit_spec):
        add(mask, "protected_schedule_template")
    for row in base:
        add(row, "tracked_css_row")
    for qubit in range(n):
        singleton = np.zeros(n, dtype=np.uint8)
        singleton[qubit] = 1
        add(singleton, "single_bit")
    limited = base[: min(len(base), int(settings.get("combination_base_rows", 32)))]
    for left, right in combinations(range(len(limited)), 2):
        add(limited[left] ^ limited[right], "xor_pair")
        if len(masks) >= max_candidates * 2:
            break
    if bool(settings.get("include_matching_edge_masks", True)):
        seen_edges = set()
        for layer in circuit_spec.get("layers", []):
            for left, right in layer["matching"]["edges"]:
                edge = tuple(sorted((int(left), int(right))))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                mask = np.zeros(n, dtype=np.uint8)
                mask[list(edge)] = 1
                add(mask, "matching_edge")
    unique = _unique_masks(masks, n)
    # Reconstruct source labels conservatively after deduplication.
    source_map: dict[bytes, str] = {}
    for mask, source in zip(masks, sources):
        source_map.setdefault(mask.tobytes(), source)
    labels = [source_map.get(mask.tobytes(), "generated") for mask in unique]
    order = sorted(
        range(len(unique)),
        key=lambda index: (
            labels[index] != "protected_schedule_template",
            int(unique[index].sum()),
            labels[index],
            unique[index].tobytes(),
        ),
    )[:max_candidates]
    return {
        "masks": unique[order] if order else np.empty((0, n), dtype=np.uint8),
        "sources": [labels[index] for index in order],
        "generated": len(unique),
    }


def _adversary_bounds(mask_weights: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    settings = config.get("certificate", {}).get("analytic_adversaries", {})
    bit_bias = float(settings.get("max_abs_bit_bias", 0.15))
    local = float(settings.get("max_abs_local_correlation", 0.20))
    return {
        "uniform": np.zeros(len(mask_weights), dtype=np.float64),
        "bounded_product": np.asarray(
            [bit_bias ** max(1, int(weight)) for weight in mask_weights],
            dtype=np.float64,
        ),
        "bounded_local_correlation": np.asarray(
            [local ** max(1, math.ceil(int(weight) / 2)) for weight in mask_weights],
            dtype=np.float64,
        ),
    }


def _independent_selection(
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    maximum: int,
) -> list[int]:
    selected: list[int] = []
    current = np.empty((0, masks.shape[1]), dtype=np.uint8)
    current_rank = 0
    for index in np.argsort(scores)[::-1]:
        proposal = np.vstack([current, masks[int(index)]])
        proposal_rank = rank(proposal)
        if proposal_rank > current_rank:
            selected.append(int(index))
            current = proposal
            current_rank = proposal_rank
            if len(selected) >= maximum:
                break
    return selected


def select_verifiable_observables(
    circuit_spec: dict[str, Any],
    config: dict[str, Any],
    *,
    backend: str | None = None,
) -> dict[str, Any]:
    settings = config.get("verifier_search", {})
    n = int(circuit_spec["n"])
    maximum_cone = int(
        settings.get(
            "max_lightcone_qubits",
            config.get("certificate", {}).get("max_lightcone_qubits", 18),
        )
    )
    generated = generate_observable_candidates(circuit_spec, settings)
    masks = generated["masks"]
    sources = generated["sources"]
    if len(masks) == 0:
        return {
            "schema": "codegap.verifier-preflight.v1",
            "passed": False,
            "reason": "NO_OBSERVABLE_CANDIDATES",
            "selected_masks": [],
            "selected_results": [],
            "local_rank": 0,
            "local_observables": 0,
            "maximum_lightcone_qubits": 0,
            "witness_margin_lcb": float("-inf"),
            "verify_operations": float("inf"),
        }
    acceleration = config.get("acceleration", {})
    requested = backend or str(settings.get("gpu_backend", acceleration.get("backend", "auto")))
    cone_batch = batch_backward_lightcones(
        circuit_spec,
        masks,
        backend=requested,
        cuda_min_observables=int(settings.get("cuda_min_observables", 8)),
    )
    local_indices = np.flatnonzero(cone_batch["sizes"] <= maximum_cone)
    max_exact = int(settings.get("max_exact_observables_per_schedule", 48))
    # Prefer protected masks, smaller cones, and lower support weight.
    local_indices = sorted(
        local_indices.tolist(),
        key=lambda index: (
            sources[index] != "protected_schedule_template",
            int(cone_batch["sizes"][index]),
            int(masks[index].sum()),
        ),
    )[:max_exact]
    results = []
    exact_backend = str(settings.get("exact_backend", "auto"))
    for index in local_indices:
        support = tuple(int(value) for value in np.flatnonzero(masks[index]))
        result = local_z_expectation(
            circuit_spec,
            support,
            max_lightcone_qubits=maximum_cone,
            backend=exact_backend,
        )
        if result["status"] != "EXACT_LIGHTCONE":
            continue
        result["candidate_index"] = int(index)
        result["source"] = sources[index]
        results.append(result)
    min_abs = float(settings.get("min_abs_expectation", 0.02))
    viable = [
        item for item in results if abs(float(item["expectation"])) >= min_abs
    ]
    if not viable:
        return {
            "schema": "codegap.verifier-preflight.v1",
            "passed": False,
            "reason": "NO_LOCAL_OBSERVABLE_WITH_IDEAL_SIGNAL",
            "candidate_masks": int(len(masks)),
            "local_candidates": int(len(local_indices)),
            "selected_masks": [],
            "selected_results": [],
            "local_rank": 0,
            "local_observables": 0,
            "maximum_lightcone_qubits": 0,
            "witness_margin_lcb": float("-inf"),
            "verify_operations": float("inf"),
            "cone_backend": cone_batch["backend"],
        }
    viable_masks = np.vstack([masks[item["candidate_index"]] for item in viable])
    weights = viable_masks.sum(axis=1).astype(int)
    expectations = np.asarray([float(item["expectation"]) for item in viable])
    local_bounds = _adversary_bounds(weights, config)
    strongest_bound = np.max(np.vstack(list(local_bounds.values())), axis=0)
    scores = (
        np.abs(expectations)
        - strongest_bound
        - float(settings.get("cone_score_penalty", 0.002))
        * np.asarray([item["lightcone_qubits"] for item in viable], dtype=float)
    )
    if int(settings.get("min_protected_checks", 0)) > 0:
        scores += np.asarray(
            [
                1000.0 if item.get("source") == "protected_schedule_template" else 0.0
                for item in viable
            ],
            dtype=float,
        )
    selection = _independent_selection(
        viable_masks,
        scores,
        maximum=int(settings.get("max_selected_observables", 24)),
    )
    selected_masks = viable_masks[selection]
    selected_results = [viable[index] for index in selection]
    selected_expectations = expectations[selection]
    selected_weights = weights[selection]
    selected_bounds = _adversary_bounds(selected_weights, config)
    names = tuple(f"local_parity_{index}" for index in range(len(selection)))
    witness, witness_optimizer = fit_configured_feature_bounds(
        selected_expectations,
        selected_bounds,
        feature_names=names,
        settings=settings,
    )
    shots = int(config.get("noise", {}).get("test_shots", 10_000))
    alpha = float(config.get("certificate", {}).get("alpha", 0.01))
    penalty = float(
        config.get("certificate", {}).get("adversary_generalization_penalty", 0.0)
    )
    margin_lcb = (
        bounded_mean_lcb(witness.ideal_mean, shots, alpha)
        - max(witness.adversary_means.values(), default=0.0)
        - penalty
    )
    local_rank = rank(selected_masks) if len(selected_masks) else 0
    cone_union = np.zeros(n, dtype=np.uint8)
    for result in selected_results:
        cone_union[np.asarray(result["lightcone"], dtype=int)] = 1
    coverage = float(cone_union.mean()) if n else 0.0
    template_protected_count = sum(
        item.get("source") == "protected_schedule_template"
        for item in selected_results
    )
    exact_local_certified_count = sum(
        item.get("status") == "EXACT_LIGHTCONE" for item in selected_results
    )
    active_indices = np.flatnonzero(np.abs(witness.weights) > 1e-8)
    active_witness_features = int(len(active_indices))
    maximum_witness_coefficient = float(
        np.max(np.abs(witness.weights), initial=0.0)
    )
    linked_layers = sorted(
        {
            int(layer)
            for item in selected_results
            for layer in item.get("selected_two_qubit_layers", [])
        }
    )
    linked_axes = sorted(
        {
            str(axis)
            for item in selected_results
            for axis in item.get("selected_two_qubit_axes", [])
        }
    )
    active_verify_operations = float(
        max(1, int(selected_weights[active_indices].sum(initial=0)))
    )
    audited_verify_operations = float(
        max(1, int(selected_weights.sum(initial=0)))
    )
    verify_cost_mode = str(
        settings.get("verify_cost_mode", "full_selected_audit")
    ).lower()
    verify_operations = (
        active_verify_operations
        if verify_cost_mode == "active_features_only"
        else audited_verify_operations
    )
    requirements = {
        "minimum_local_observables": int(settings.get("min_local_observables", 4)),
        "minimum_local_rank": int(settings.get("min_local_rank", 4)),
        "minimum_witness_margin_lcb": float(
            settings.get("min_witness_margin_lcb", 0.0)
        ),
        "minimum_coverage_fraction": float(
            settings.get("min_coverage_fraction", 0.0)
        ),
        "minimum_exact_local_certified_checks": int(
            settings.get(
                "min_exact_local_certified_checks",
                settings.get("min_local_observables", 4),
            )
        ),
        "minimum_template_protected_checks": int(
            settings.get(
                "min_template_protected_checks",
                settings.get("min_protected_checks", 0),
            )
        ),
        "minimum_active_witness_features": int(
            settings.get("min_active_witness_features", 1)
        ),
        "maximum_absolute_witness_weight": float(
            settings.get("max_abs_witness_weight", 1.0)
        ),
        "minimum_linked_two_qubit_layers": int(
            settings.get("min_linked_two_qubit_layers", 1)
        ),
        "minimum_linked_axes": int(settings.get("min_linked_axes", 1)),
    }
    checks = {
        "local_observables": len(selected_masks)
        >= requirements["minimum_local_observables"],
        "local_rank": local_rank >= requirements["minimum_local_rank"],
        "witness_margin": margin_lcb
        > requirements["minimum_witness_margin_lcb"],
        "coverage": coverage >= requirements["minimum_coverage_fraction"],
        "exact_local_certified_checks": exact_local_certified_count
        >= requirements["minimum_exact_local_certified_checks"],
        "template_protected_checks": template_protected_count
        >= requirements["minimum_template_protected_checks"],
        "active_witness_features": active_witness_features
        >= requirements["minimum_active_witness_features"],
        "maximum_witness_weight": maximum_witness_coefficient
        <= requirements["maximum_absolute_witness_weight"] + 1e-12,
        "hardness_linked_layers": len(linked_layers)
        >= requirements["minimum_linked_two_qubit_layers"],
        "hardness_linked_axes": len(linked_axes)
        >= requirements["minimum_linked_axes"],
        "maximum_lightcone": max(
            [item["lightcone_qubits"] for item in selected_results], default=0
        )
        <= maximum_cone,
    }
    return {
        "schema": "codegap.verifier-preflight.v1",
        "passed": all(checks.values()),
        "reason": None if all(checks.values()) else "VERIFIER_CONSTRAINTS_FAILED",
        "checks": checks,
        "requirements": requirements,
        "candidate_masks": int(len(masks)),
        "local_candidates": int(len(local_indices)),
        "signal_candidates": int(len(viable)),
        "selected_masks": selected_masks.tolist(),
        "selected_results": selected_results,
        "selected_expectations": selected_expectations.tolist(),
        "selected_weights": selected_weights.tolist(),
        "local_rank": int(local_rank),
        "local_observables": int(len(selected_masks)),
        "protected_checks": int(template_protected_count),
        "template_protected_checks": int(template_protected_count),
        "exact_local_certified_checks": int(exact_local_certified_count),
        "active_witness_features": active_witness_features,
        "maximum_witness_coefficient": maximum_witness_coefficient,
        "linked_two_qubit_layers": linked_layers,
        "linked_two_qubit_axes": linked_axes,
        "coverage_fraction": coverage,
        "maximum_lightcone_qubits": max(
            [item["lightcone_qubits"] for item in selected_results], default=0
        ),
        "witness_margin_lcb": float(margin_lcb),
        "witness": witness.to_dict(),
        "witness_optimizer": witness_optimizer,
        "adversary_bounds": {
            key: value.tolist() for key, value in selected_bounds.items()
        },
        "verify_operations": verify_operations,
        "active_verify_operations": active_verify_operations,
        "audited_verify_operations": audited_verify_operations,
        "verify_cost_mode": verify_cost_mode,
        "cone_backend": cone_batch["backend"],
        "exact_backends": sorted(
            {str(item.get("backend", "numpy")) for item in selected_results}
        ),
        "claim_boundary": (
            "The selected local checks certify the registered protected/tracked "
            "observable family. They do not by themselves certify total-variation "
            "closeness of the complete output distribution."
        ),
    }


def apply_verifier_selection(circuit_spec: dict[str, Any], preflight: dict[str, Any]) -> None:
    circuit_spec["verifier_selection"] = preflight
    if preflight.get("selected_masks"):
        circuit_spec["verifier_masks"] = preflight["selected_masks"]
        circuit_spec["verifier_cost_model"] = (
            "sum of selected local parity weights with exact backward-lightcone "
            "expectations"
        )
