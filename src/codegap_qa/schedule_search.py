from __future__ import annotations

from hashlib import sha256
import json
import math
from itertools import combinations
from typing import Any, Iterable

import networkx as nx
import numpy as np

from .bicycle import BicycleFamilySpec
from .codesign import (
    build_tracked_frame_mixing_circuit,
    build_verifiable_mixing_circuit,
    css_automorphism_matchings,
)
from .hardware import (
    HardwareTopology,
    exact_zero_swap_embedding,
    zero_swap_hardware_metrics,
)
from .verifier_search import apply_verifier_selection, select_verifiable_observables
from .progress import ProgressManager
from .target_native import build_target_native_matching_pool
from .accel_bridge import accelerated_jaccard_matrix
from .spacetime import (
    cotengra_metrics_for_circuit,
    graph_metrics_for_circuit,
    spacetime_hardness_metrics,
)


def _edges(item: dict[str, Any]) -> frozenset[tuple[int, int]]:
    return frozenset(tuple(sorted(map(int, edge))) for edge in item["edges"])


def _jaccard_distance(
    left: frozenset[tuple[int, int]],
    right: frozenset[tuple[int, int]],
) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def _schedule_signature(payload: dict[str, Any]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _protected_schedule_templates(
    matching_pool: list[dict[str, Any]],
    settings: dict,
) -> list[dict[str, Any]]:
    if not matching_pool:
        return []
    n = len(matching_pool[0]["permutation"])
    min_unique = int(settings.get("min_unique_matchings", 3))
    max_weight = int(settings.get("protected_check_max_weight", 8))
    max_templates = int(settings.get("max_protected_templates", 32))
    candidates: dict[bytes, np.ndarray] = {}
    edge_pool = []
    for matching in matching_pool:
        edge_pool.extend(tuple(sorted(map(int, edge))) for edge in matching["edges"])
    for edge in sorted(set(edge_pool)):
        mask = np.zeros(n, dtype=np.uint8)
        mask[list(edge)] = 1
        candidates.setdefault(mask.tobytes(), mask)
    edges = sorted(set(edge_pool))[: min(48, len(set(edge_pool)))]
    for left, right in combinations(edges, 2):
        support = sorted(set(left) | set(right))
        if len(support) > max_weight:
            continue
        mask = np.zeros(n, dtype=np.uint8)
        mask[support] = 1
        candidates.setdefault(mask.tobytes(), mask)
    templates = []
    for mask in candidates.values():
        inside = set(np.flatnonzero(mask).astype(int).tolist())
        compatible = []
        for index, matching in enumerate(matching_pool):
            crosses = any(
                (int(left) in inside) != (int(right) in inside)
                for left, right in matching["edges"]
            )
            if not crosses:
                compatible.append(index)
        if len(compatible) >= min_unique:
            templates.append(
                {
                    "mask": mask.tolist(),
                    "compatible_matching_indices": compatible,
                    "weight": int(mask.sum()),
                    "compatible_count": len(compatible),
                }
            )
    templates.sort(
        key=lambda item: (-item["compatible_count"], item["weight"], item["mask"])
    )
    return templates[:max_templates]


def _generate_schedule_payloads(
    matching_pool: list[dict[str, Any]],
    settings: dict,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    count = int(settings.get("schedules_per_code", 256))
    depths = tuple(int(value) for value in settings.get("depths", [8, 10, 12]))
    axes_pool = tuple(
        str(value).lower() for value in settings.get("axes", ["zz", "xx"])
    )
    angle_scales = tuple(
        float(value) for value in settings.get("angle_scales", [0.75, 1.0, 1.25])
    )
    if not matching_pool:
        return []
    edge_sets = [_edges(item) for item in matching_pool]
    native_jaccard = accelerated_jaccard_matrix(edge_sets)
    protected_templates = _protected_schedule_templates(matching_pool, settings)
    protected_fraction = float(settings.get("protected_schedule_fraction", 0.65))
    generated: dict[str, dict[str, Any]] = {}

    def add(
        indices: list[int],
        axes: list[str],
        scales: list[float],
        mode: str,
        protected_masks: list[list[int]] | None = None,
    ) -> None:
        payload = {
            "matching_indices": indices,
            "axes": axes,
            "angle_scales": scales,
            "mode": mode,
            "protected_masks": protected_masks or [],
        }
        generated.setdefault(_schedule_signature(payload), payload)

    for depth in depths:
        indices = [layer % len(matching_pool) for layer in range(depth)]
        add(
            indices,
            [axes_pool[layer % len(axes_pool)] for layer in range(depth)],
            [1.0] * depth,
            "round_robin",
        )
        add(
            list(reversed(indices)),
            [axes_pool[(layer + 1) % len(axes_pool)] for layer in range(depth)],
            [1.0] * depth,
            "reverse_round_robin",
        )

    attempts = 0
    max_attempts = max(100, count * 20)
    while len(generated) < count and attempts < max_attempts:
        attempts += 1
        depth = int(rng.choice(depths))
        protected = (
            protected_templates[int(rng.integers(0, len(protected_templates)))]
            if protected_templates and rng.random() < protected_fraction
            else None
        )
        allowed = (
            np.asarray(protected["compatible_matching_indices"], dtype=int)
            if protected is not None
            else np.arange(len(matching_pool), dtype=int)
        )
        start = int(rng.choice(allowed))
        indices = [start]
        for _ in range(1, depth):
            previous = indices[-1]
            candidates = allowed
            if native_jaccard is not None:
                distances = np.asarray(
                    native_jaccard[previous, candidates],
                    dtype=float,
                )
            else:
                distances = np.asarray(
                    [
                        _jaccard_distance(edge_sets[previous], edge_sets[index])
                        for index in candidates
                    ],
                    dtype=float,
                )
            reuse_penalty = np.asarray(
                [indices.count(int(index)) / max(1, len(indices)) for index in candidates]
            )
            scores = distances - 0.35 * reuse_penalty + rng.random(len(candidates)) * 0.2
            if len(candidates) > 1:
                previous_positions = np.flatnonzero(candidates == previous)
                if len(previous_positions):
                    scores[int(previous_positions[0])] -= 1.0
            next_index = int(candidates[int(np.argmax(scores))])
            indices.append(next_index)
        offset = int(rng.integers(0, len(axes_pool)))
        axes = [axes_pool[(layer + offset) % len(axes_pool)] for layer in range(depth)]
        if len(axes_pool) > 1 and rng.random() < 0.5:
            rng.shuffle(axes)
        scales = [float(rng.choice(angle_scales)) for _ in range(depth)]
        add(
            indices,
            axes,
            scales,
            "protected_adversarial_randomized" if protected is not None else "adversarial_randomized",
            [protected["mask"]] if protected is not None else None,
        )
    return list(generated.values())[:count]


def schedule_diversity(
    matching_pool: list[dict[str, Any]], matching_indices: Iterable[int]
) -> dict[str, Any]:
    indices = tuple(int(value) for value in matching_indices)
    edge_sets = [_edges(matching_pool[index]) for index in indices]
    transitions = [
        _jaccard_distance(edge_sets[index], edge_sets[index + 1])
        for index in range(len(edge_sets) - 1)
    ]
    unique = len(set(indices))
    frequencies = np.asarray(
        [indices.count(index) for index in sorted(set(indices))], dtype=float
    )
    probabilities = frequencies / max(1.0, frequencies.sum())
    entropy = float(
        -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-300)))
    )
    max_entropy = math.log2(max(1, unique))
    return {
        "unique_matchings": unique,
        "unique_fraction": unique / max(1, len(indices)),
        "mean_transition_jaccard": float(np.mean(transitions)) if transitions else 0.0,
        "minimum_transition_jaccard": float(min(transitions, default=0.0)),
        "matching_entropy_bits": entropy,
        "normalized_matching_entropy": entropy / max(1.0, max_entropy),
        "immediate_repeats": sum(
            indices[index] == indices[index + 1]
            for index in range(len(indices) - 1)
        ),
    }


def _cutwidth_for_order(
    weighted_edges: list[tuple[int, int]], order: list[int]
) -> int:
    position = {node: index for index, node in enumerate(order)}
    maximum = 0
    for cut in range(1, len(order)):
        crossing = sum(
            (position[left] < cut <= position[right])
            or (position[right] < cut <= position[left])
            for left, right in weighted_edges
        )
        maximum = max(maximum, int(crossing))
    return maximum


def quick_cutwidth_metrics(circuit_spec: dict, *, seed: int) -> dict[str, Any]:
    n = int(circuit_spec["n"])
    weighted_edges = [
        tuple(map(int, gate["qubits"]))
        for gate in circuit_spec["gates"]
        if len(gate["qubits"]) == 2
    ]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(weighted_edges)
    orders: list[list[int]] = [list(range(n)), list(reversed(range(n)))]
    if graph.number_of_edges():
        orders.append(list(nx.utils.reverse_cuthill_mckee_ordering(graph)))
        degree_order = sorted(graph.nodes(), key=lambda node: (-graph.degree[node], node))
        orders.append(degree_order)
    rng = np.random.default_rng(seed)
    for _ in range(4):
        orders.append(rng.permutation(n).astype(int).tolist())
    widths = [_cutwidth_for_order(weighted_edges, order) for order in orders]

    partner_by_layer: list[dict[int, int]] = []
    for layer in circuit_spec["layers"]:
        partners: dict[int, int] = {}
        for left, right in layer["matching"]["edges"]:
            partners[int(left)] = int(right)
            partners[int(right)] = int(left)
        partner_by_layer.append(partners)
    frontiers = []
    for boundary in range(1, len(partner_by_layer)):
        past = partner_by_layer[:boundary]
        future = partner_by_layer[boundary:]
        active = 0
        for qubit in range(n):
            past_partners = {item.get(qubit) for item in past}
            future_partners = {item.get(qubit) for item in future}
            past_partners.discard(None)
            future_partners.discard(None)
            if past_partners and future_partners and past_partners != future_partners:
                active += 1
        frontiers.append(active)
    heuristic_min = min(widths, default=0)
    frontier_max = max(frontiers, default=0)
    return {
        "tested_order_cutwidths": widths,
        "heuristic_min_cutwidth": int(heuristic_min),
        "heuristic_max_cutwidth": int(max(widths, default=0)),
        "temporal_frontier_max": int(frontier_max),
        "temporal_frontier_mean": float(np.mean(frontiers)) if frontiers else 0.0,
        "ranking_score": float(heuristic_min + frontier_max),
    }


def verifier_operations(circuit_spec: dict) -> float:
    selection = circuit_spec.get("verifier_selection") or {}
    if selection.get("verify_operations") is not None:
        return float(selection["verify_operations"])
    masks = np.asarray(circuit_spec["verifier_masks"], dtype=np.uint8)
    return float(max(1, int(masks.sum(initial=0))))


def schedule_objective(
    *,
    cotengra_flops: float,
    verify_operations_value: float,
    two_qubit_depth: int,
    two_qubit_count: int,
    settings: dict,
    verifier_preflight: dict[str, Any] | None = None,
) -> float:
    weights = settings.get("objective", {})
    lambda_depth = float(weights.get("lambda_depth", 0.02))
    mu_twoq = float(weights.get("mu_twoq", 0.0005))
    rho_lightcone = float(weights.get("rho_lightcone", 0.02))
    eta_local_rank = float(weights.get("eta_local_rank", 0.05))
    eta_margin = float(weights.get("eta_witness_margin", 0.25))
    preflight = verifier_preflight or {}
    return float(
        math.log10(max(1.0, cotengra_flops))
        - math.log10(max(1.0, verify_operations_value))
        - lambda_depth * two_qubit_depth
        - mu_twoq * two_qubit_count
        - rho_lightcone * float(preflight.get("maximum_lightcone_qubits", 0.0))
        + eta_local_rank * float(preflight.get("local_rank", 0.0))
        + eta_margin * max(0.0, float(preflight.get("witness_margin_lcb", 0.0)))
    )


def _sort_and_keep(
    items: list[dict[str, Any]],
    *,
    key,
    count: int,
) -> list[dict[str, Any]]:
    return sorted(items, key=key, reverse=True)[: max(1, int(count))]


def search_adversarial_schedules(
    *,
    spec: BicycleFamilySpec,
    h_x: np.ndarray,
    h_z: np.ndarray,
    config: dict,
    topology: HardwareTopology,
    seed: int,
    progress: ProgressManager,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search schedules in the preregistered adversarial filter order."""

    settings = config.get("schedule_search", {})
    matching_source = str(
        settings.get("matching_source", "css_automorphism")
    ).lower()
    target_native_pool = None
    pinned_layout: tuple[int, ...] | None = None
    if matching_source == "target_native_tracked_frame":
        target_native_pool = build_target_native_matching_pool(
            topology=topology,
            n=spec.n,
            seed=int(config.get("seed", seed)) + 10007 * spec.n,
            layout_trials=int(settings.get("target_layout_trials", 24)),
            matching_trials=int(settings.get("target_matching_trials", 512)),
            minimum_matchings=int(
                settings.get(
                    "target_minimum_matchings",
                    settings.get("min_unique_matchings", 3),
                )
            ),
        )
        if target_native_pool.get("status") not in {"FOUND", "BEST_PARTIAL_POOL"}:
            report = {
                "schema": "codegap.adversarial-schedule-search.v3-target-native",
                "code_n": spec.n,
                "matching_source": matching_source,
                "status": "STOP_NO_TARGET_NATIVE_MATCHING_POOL",
                "target_native_pool": target_native_pool,
                "counts": {},
            }
            return [], report
        matching_pool = list(target_native_pool["matching_pool"])
        pinned_layout = tuple(
            int(value) for value in target_native_pool["layout"]
        )
    else:
        max_matchings = int(settings.get("max_automorphism_matchings", 64))
        matching_pool = css_automorphism_matchings(
            spec,
            h_x,
            h_z,
            max_matchings=max_matchings,
            include_pair_compositions=True,
        )
    payloads = _generate_schedule_payloads(matching_pool, settings, seed=seed)
    report: dict[str, Any] = {
        "schema": "codegap.adversarial-schedule-search.v2-live-target",
        "code_n": spec.n,
        "matching_pool_size": len(matching_pool),
        "matching_source": matching_source,
        "target_native_pool": target_native_pool,
        "generated_schedules": len(payloads),
        "hardware_source": topology.source,
        "target_structural_fingerprint": topology.structural_fingerprint,
        "filter_order": [
            (
                "exact_tracked_CSS_relation"
                if matching_source == "target_native_tracked_frame"
                else "css_relation_preservation"
            ),
            "live_target_zero_swap_embedding",
            "layer_diversity",
            "local_observable_construction",
            "gpu_lightcone_preflight",
            "ideal_signal_and_gf2_independence",
            "quick_cutwidth",
            "line_graph_treewidth",
            "cotengra_short",
            "cotengra_deep",
            "noisecert",
        ],
        "counts": {},
        "targets": {
            "cotengra_width_min": float(settings.get("target_width_min", 17.0)),
            "cotengra_flops_min": float(settings.get("min_cotengra_flops", 192_000_000.0)),
            "gamma_min": float(config["gates"]["min_gamma_log10"]),
        },
    }
    require_live = bool(settings.get("require_live_target_snapshot", True))
    if require_live and topology.source != "live_backend_target_snapshot":
        report["status"] = "STOP_TARGET_SNAPSHOT_REQUIRED"
        report["action"] = (
            "Capture the live BackendV2.target with codegap qpu-snapshot and "
            "set hardware.target_snapshot before rerunning the pipeline."
        )
        return [], report
    if not matching_pool or not payloads:
        report["status"] = "STOP_NO_VERIFIED_MATCHING_SCHEDULES"
        return [], report

    min_unique = int(settings.get("min_unique_matchings", 2))
    min_jaccard = float(settings.get("min_transition_jaccard", 0.25))
    max_states = int(settings.get("embedding_max_states", 2_000_000))
    timeout = float(settings.get("embedding_timeout_seconds", 10.0))
    heuristic_iterations = int(settings.get("embedding_heuristic_iterations", 10_000))
    embedding_cache: dict[tuple[tuple[int, int], ...], dict[str, Any]] = {}
    stage2: list[dict[str, Any]] = []
    bar = progress.bar(
        payloads,
        total=len(payloads),
        desc=f"ScheduleSearch n={spec.n}: relation + live zero-SWAP",
        unit="schedule",
        leave=progress.leave_nested,
    )
    relation_pass = 0
    degree_pass = 0
    zero_swap_pass = 0
    budget_exhausted = 0
    proved_no_embedding = 0
    diversity_pass = 0
    for index, payload in enumerate(bar):
        builder = (
            build_tracked_frame_mixing_circuit
            if matching_source == "target_native_tracked_frame"
            else build_verifiable_mixing_circuit
        )
        circuit_spec = builder(
            spec=spec,
            h_x=h_x,
            h_z=h_z,
            matching_pool=matching_pool,
            matching_indices=payload["matching_indices"],
            axes=payload["axes"],
            angle_scales=payload["angle_scales"],
            config=config,
            seed=seed + index * 1009,
            schedule_metadata={
                "generation_mode": payload["mode"],
                "matching_source": matching_source,
                "pinned_layout": list(pinned_layout) if pinned_layout else None,
                "protected_masks": payload.get("protected_masks", []),
            },
        )
        relation = circuit_spec["relation_preservation"]
        relation_ok = bool(
            relation.get("all_layers_involutive", False)
            and (
                relation.get("all_layers_css_rowspaces_preserved", False)
                or (
                    relation.get(
                        "all_layers_tracked_frame_transitions_exact", False
                    )
                    and relation.get(
                        "final_frame_matches_cumulative_permutation", False
                    )
                    and relation.get(
                        "verifier_masks_from_final_tracked_frame", False
                    )
                )
            )
        )
        if not relation_ok:
            continue
        relation_pass += 1
        union_edges = tuple(
            tuple(sorted(map(int, edge)))
            for edge in circuit_spec["union_two_qubit_edges"]
        )
        graph = nx.Graph()
        graph.add_nodes_from(range(spec.n))
        graph.add_edges_from(union_edges)
        if max(dict(graph.degree()).values(), default=0) > max(
            dict(topology.graph.degree()).values(), default=0
        ):
            proved_no_embedding += 1
            continue
        degree_pass += 1
        cache_key = tuple(sorted(set(union_edges)))
        if matching_source == "target_native_tracked_frame":
            if pinned_layout is None or len(pinned_layout) != spec.n:
                proved_no_embedding += 1
                continue
            physical_edges = tuple(
                tuple(sorted((pinned_layout[left], pinned_layout[right])))
                for left, right in cache_key
            )
            construction_valid = all(
                topology.graph.has_edge(left, right)
                for left, right in physical_edges
            )
            embedding = {
                "status": "FOUND" if construction_valid else "PROVED_NO_EMBEDDING",
                "reason": (
                    "target_native_construction_witness"
                    if construction_valid
                    else "target_native_construction_validation_failed"
                ),
                "layout": list(pinned_layout) if construction_valid else None,
                "states": 0,
                "elapsed_seconds": 0.0,
                "exact": True,
                "target_structural_fingerprint": topology.structural_fingerprint,
            }
            embedding_cache.setdefault(cache_key, embedding)
        else:
            embedding = embedding_cache.get(cache_key)
            if embedding is None:
                embedding = exact_zero_swap_embedding(
                    n=spec.n,
                    logical_edges=cache_key,
                    topology=topology,
                    seed=seed + index * 65537,
                    max_states=max_states,
                    timeout_seconds=timeout,
                    heuristic_iterations=heuristic_iterations,
                    prefer_native=bool(
                        config.get("acceleration", {}).get("native_layout", True)
                    ),
                )
                embedding_cache[cache_key] = embedding
        if embedding["status"] == "BUDGET_EXHAUSTED":
            budget_exhausted += 1
            continue
        if embedding["status"] != "FOUND":
            proved_no_embedding += 1
            continue
        zero_swap_pass += 1
        diversity = schedule_diversity(
            matching_pool, payload["matching_indices"]
        )
        if (
            diversity["unique_matchings"] < min_unique
            or diversity["mean_transition_jaccard"] < min_jaccard
            or diversity["immediate_repeats"] > 0
        ):
            continue
        diversity_pass += 1
        hardware = zero_swap_hardware_metrics(
            circuit_spec=circuit_spec,
            logical_edges=cache_key,
            topology=topology,
            embedding=embedding,
        )
        stage2.append(
            {
                "circuit_spec": circuit_spec,
                "hardware": hardware,
                "diversity": diversity,
                "embedding": embedding,
                "matching_source": matching_source,
            }
        )
        bar.set_postfix(
            relation=relation_pass,
            degree=degree_pass,
            zero_swap=zero_swap_pass,
            budget=budget_exhausted,
            diverse=diversity_pass,
            refresh=False,
        )
    bar.close()
    report["counts"].update(
        {
            "relation_preserved": relation_pass,
            "degree_compatible": degree_pass,
            "live_target_zero_swap": zero_swap_pass,
            "embedding_budget_exhausted": budget_exhausted,
            "proved_no_embedding": proved_no_embedding,
            "diversity_pass": diversity_pass,
            "unique_interaction_unions": len(embedding_cache),
            "target_native_matching_pool_size": (
                len(matching_pool)
                if matching_source == "target_native_tracked_frame"
                else 0
            ),
        }
    )
    if not stage2:
        if relation_pass == 0:
            report["status"] = "STOP_NO_RELATION_PRESERVING_SCHEDULE"
        elif zero_swap_pass == 0 and budget_exhausted > 0:
            report["status"] = "STOP_ZERO_SWAP_SEARCH_BUDGET_EXHAUSTED"
        elif zero_swap_pass == 0:
            report["status"] = "STOP_NO_ZERO_SWAP_TARGET_EMBEDDING"
        else:
            report["status"] = "STOP_NO_DIVERSE_ZERO_SWAP_SCHEDULE"
        return [], report

    verifier_bar = progress.bar(
        stage2,
        total=len(stage2),
        desc=f"ScheduleSearch n={spec.n}: GPU verifier preflight",
        unit="schedule",
        leave=True,
    )
    stage3: list[dict[str, Any]] = []
    for item in verifier_bar:
        preflight = select_verifiable_observables(
            item["circuit_spec"],
            config,
        )
        item["verifier_preflight"] = preflight
        if preflight.get("passed", False):
            apply_verifier_selection(item["circuit_spec"], preflight)
            stage3.append(item)
        verifier_bar.set_postfix(
            passed=preflight.get("passed", False),
            local=preflight.get("local_observables", 0),
            rank=preflight.get("local_rank", 0),
            cone=preflight.get("maximum_lightcone_qubits", 0),
            margin=f"{float(preflight.get('witness_margin_lcb', float('-inf'))):.3g}",
            backend=preflight.get("cone_backend"),
            refresh=False,
        )
    verifier_bar.close()
    report["counts"]["verifier_preflight_pass"] = len(stage3)
    report["counts"]["verifier_preflight_failed"] = len(stage2) - len(stage3)
    if not stage3:
        report["status"] = "STOP_NO_VERIFIABLE_SCHEDULE_BEFORE_COTENGRA"
        report["action"] = (
            "No schedule retained enough independent local observables with "
            "positive witness margin. Increase protected schedule generation or "
            "relax only preregistered verifier-search constraints."
        )
        return [], report

    cut_bar = progress.bar(
        stage3,
        total=len(stage3),
        desc=f"ScheduleSearch n={spec.n}: cutwidth",
        unit="schedule",
        leave=progress.leave_nested,
    )
    for index, item in enumerate(cut_bar):
        item["cutwidth"] = quick_cutwidth_metrics(
            item["circuit_spec"], seed=seed + 17 * index
        )
        cut_bar.set_postfix(
            score=f"{item['cutwidth']['ranking_score']:.1f}", refresh=False
        )
    cut_bar.close()
    stage4 = _sort_and_keep(
        stage3,
        key=lambda item: (
            item["cutwidth"]["ranking_score"],
            item["diversity"]["mean_transition_jaccard"],
        ),
        count=int(settings.get("keep_after_cutwidth", 48)),
    )
    report["counts"]["after_cutwidth"] = len(stage4)

    line_bar = progress.bar(
        stage4,
        total=len(stage4),
        desc=f"ScheduleSearch n={spec.n}: line-graph treewidth",
        unit="schedule",
        leave=progress.leave_nested,
    )
    for item in line_bar:
        graph_metrics = graph_metrics_for_circuit(item["circuit_spec"])
        item["graph_metrics"] = graph_metrics
        line_bar.set_postfix(
            lower=graph_metrics["line_graph_treewidth_lower"],
            upper=graph_metrics["line_graph_treewidth_upper"],
            refresh=False,
        )
    line_bar.close()
    stage5 = _sort_and_keep(
        stage4,
        key=lambda item: (
            item["graph_metrics"]["line_graph_treewidth_lower"],
            item["graph_metrics"]["line_graph_treewidth_upper"],
            item["cutwidth"]["ranking_score"],
        ),
        count=int(settings.get("keep_after_line_graph", 16)),
    )
    report["counts"]["after_line_graph"] = len(stage5)

    short_settings = dict(settings.get("cotengra_short", {}))
    short_settings.update(
        {
            "enabled": True,
            "mode": "short",
            "seed": seed,
        }
    )
    short_bar = progress.bar(
        stage5,
        total=len(stage5),
        desc=f"ScheduleSearch n={spec.n}: Cotengra short",
        unit="schedule",
        leave=True,
    )
    short_success = 0
    for index, item in enumerate(short_bar):
        current = dict(short_settings)
        current["seed"] = seed + index * 10007
        metrics = cotengra_metrics_for_circuit(item["circuit_spec"], current)
        item["cotengra_short"] = metrics
        flops = float(metrics.get("contraction_flops") or 0.0)
        verify = verifier_operations(item["circuit_spec"])
        item["short_objective"] = schedule_objective(
            cotengra_flops=flops,
            verify_operations_value=verify,
            two_qubit_depth=int(item["hardware"]["two_qubit_depth"]),
            two_qubit_count=int(item["hardware"]["two_qubit_count"]),
            settings=settings,
            verifier_preflight=item.get("verifier_preflight"),
        )
        short_success += metrics.get("status") == "PASS"
        short_bar.set_postfix(
            width=metrics.get("contraction_width_log2"),
            flops=f"{flops:.3g}",
            objective=f"{item['short_objective']:.3f}",
            refresh=False,
        )
    short_bar.close()
    require_cotengra = bool(settings.get("require_cotengra", True))
    if require_cotengra:
        stage5 = [
            item for item in stage5 if item["cotengra_short"].get("status") == "PASS"
        ]
    report["counts"]["cotengra_short_pass"] = short_success
    if not stage5:
        report["status"] = "STOP_COTENGRA_SHORT_UNAVAILABLE_OR_FAILED"
        return [], report
    stage6 = _sort_and_keep(
        stage5,
        key=lambda item: (
            item["short_objective"],
            float(item["cotengra_short"].get("contraction_width_log2") or 0.0),
        ),
        count=int(settings.get("deep_finalists", 4)),
    )

    deep_settings = dict(settings.get("cotengra_deep", {}))
    deep_settings.update({"enabled": True, "mode": "deep"})
    deep_bar = progress.bar(
        stage6,
        total=len(stage6),
        desc=f"ScheduleSearch n={spec.n}: Cotengra deep",
        unit="schedule",
        leave=True,
    )
    final: list[dict[str, Any]] = []
    for item in deep_bar:
        deep = cotengra_metrics_for_circuit(item["circuit_spec"], deep_settings)
        short = item["cotengra_short"]
        successful = [
            metrics
            for metrics in (short, deep)
            if metrics.get("status") == "PASS"
            and metrics.get("contraction_flops") is not None
        ]
        if require_cotengra and not successful:
            continue
        cotengra_adversary = (
            min(successful, key=lambda metrics: float(metrics["contraction_flops"]))
            if successful
            else deep
        )
        cotengra_adversary = dict(cotengra_adversary) | {
            "selection_rule": "minimum FLOPs across short and deep searches",
            "short_flops": short.get("contraction_flops"),
            "deep_flops": deep.get("contraction_flops"),
        }
        verify = verifier_operations(item["circuit_spec"])
        hardness_settings = dict(config["hardness"])
        hardness_settings["cotengra"] = deep_settings
        hardness = spacetime_hardness_metrics(
            circuit_spec=item["circuit_spec"],
            verify_operations=verify,
            assumptions=tuple(config["hardness"]["assumptions"]),
            settings=hardness_settings,
            graph_metrics=item["graph_metrics"],
            cotengra_override=cotengra_adversary,
        )
        flops = float(cotengra_adversary.get("contraction_flops") or 0.0)
        objective = schedule_objective(
            cotengra_flops=flops,
            verify_operations_value=verify,
            two_qubit_depth=int(item["hardware"]["two_qubit_depth"]),
            two_qubit_count=int(item["hardware"]["two_qubit_count"]),
            settings=settings,
            verifier_preflight=item.get("verifier_preflight"),
        )
        preflight = item["verifier_preflight"]
        target = {
            "cotengra_width": float(cotengra_adversary.get("contraction_width_log2") or 0.0)
            >= float(settings.get("target_width_min", 17.0)),
            "cotengra_flops": flops
            >= float(settings.get("min_cotengra_flops", 192_000_000.0)),
            "gamma": float(hardness["gamma_log10"])
            >= float(config["gates"]["min_gamma_log10"]),
            "verifier_preflight": bool(preflight.get("passed", False)),
            "live_target_zero_swap": int(item["hardware"]["swap_count"]) == 0
            and int(item["hardware"]["nonlocal_edges"]) == 0
            and bool(item["hardware"].get("target_snapshot_embedding")),
            "relation_certificate": bool(
                item["circuit_spec"]["relation_preservation"].get(
                    "all_layers_css_rowspaces_preserved", False
                )
                or (
                    item["circuit_spec"]["relation_preservation"].get(
                        "all_layers_tracked_frame_transitions_exact", False
                    )
                    and item["circuit_spec"]["relation_preservation"].get(
                        "final_frame_matches_cumulative_permutation", False
                    )
                )
            ),
        }
        final.append(
            item
            | {
                "cotengra_deep": deep,
                "cotengra_adversary": cotengra_adversary,
                "hardness": hardness,
                "objective": objective,
                "verifier_preflight": preflight,
                "lightcone_preflight": preflight,
                "target_checks": target,
                "target_pass": all(target.values()),
            }
        )
        deep_bar.set_postfix(
            width=cotengra_adversary.get("contraction_width_log2"),
            flops=f"{flops:.3g}",
            gamma=f"{hardness['gamma_log10']:.3f}",
            target=all(target.values()),
            refresh=False,
        )
    deep_bar.close()
    final.sort(
        key=lambda item: (
            item["target_pass"],
            item["objective"],
            item["hardness"]["gamma_log10"],
        ),
        reverse=True,
    )
    keep = int(settings.get("return_finalists_per_code", 1))
    selected = final[: max(1, keep)]
    report["counts"]["cotengra_deep_evaluated"] = len(final)
    report["counts"]["target_pass"] = sum(item["target_pass"] for item in final)
    report["status"] = "PASS_TARGET_FOUND" if any(
        item["target_pass"] for item in final
    ) else "BEST_PARTIAL_ONLY"
    report["best"] = (
        {
            "schedule_id": selected[0]["circuit_spec"]["schedule_id"],
            "objective": selected[0]["objective"],
            "gamma_log10": selected[0]["hardness"]["gamma_log10"],
            "cotengra": selected[0]["cotengra_adversary"],
            "cotengra_deep": selected[0]["cotengra_deep"],
            "target_checks": selected[0]["target_checks"],
            "verifier_preflight": selected[0]["verifier_preflight"],
            "lightcone_preflight": selected[0]["verifier_preflight"],
        }
        if selected
        else None
    )
    return selected, report
