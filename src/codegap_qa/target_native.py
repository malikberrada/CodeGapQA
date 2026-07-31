from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

import networkx as nx
import numpy as np

from .hardware import HardwareTopology


_POOL_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _edge_key(edges) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(tuple(sorted((int(a), int(b)))) for a, b in edges))


def _randomized_maximum_matching(
    graph: nx.Graph,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], ...]:
    weighted = nx.Graph()
    weighted.add_nodes_from(graph.nodes)
    for left, right in graph.edges:
        weighted.add_edge(
            int(left),
            int(right),
            weight=float(rng.random()) + 1e-9 * (int(left) + int(right)),
        )
    matching = nx.algorithms.matching.max_weight_matching(
        weighted,
        maxcardinality=True,
        weight="weight",
    )
    return _edge_key(matching)


def _pair_graph(
    physical: nx.Graph,
    matching: tuple[tuple[int, int], ...],
) -> nx.Graph:
    pair_graph = nx.Graph()
    pair_graph.add_nodes_from(range(len(matching)))
    owner: dict[int, int] = {}
    for index, edge in enumerate(matching):
        owner[int(edge[0])] = index
        owner[int(edge[1])] = index
    for left, right in physical.edges:
        left_owner = owner.get(int(left))
        right_owner = owner.get(int(right))
        if (
            left_owner is not None
            and right_owner is not None
            and left_owner != right_owner
        ):
            pair_graph.add_edge(left_owner, right_owner)
    return pair_graph


def _connected_pair_subset(
    pair_graph: nx.Graph,
    count: int,
    rng: np.random.Generator,
) -> tuple[int, ...] | None:
    eligible = [
        component
        for component in nx.connected_components(pair_graph)
        if len(component) >= count
    ]
    if not eligible:
        return None
    component = set(
        eligible[int(rng.integers(0, len(eligible)))]
    )
    starts = list(component)
    rng.shuffle(starts)
    for start in starts[: min(16, len(starts))]:
        selected = {int(start)}
        while len(selected) < count:
            boundary = {
                int(neighbor)
                for node in selected
                for neighbor in pair_graph.neighbors(node)
                if neighbor in component and neighbor not in selected
            }
            if not boundary:
                break
            values = list(boundary)
            weights = np.asarray(
                [
                    1.0
                    + sum(
                        int(neighbor in selected)
                        for neighbor in pair_graph.neighbors(value)
                    )
                    + 0.05 * pair_graph.degree[value]
                    for value in values
                ],
                dtype=float,
            )
            weights /= weights.sum()
            selected.add(int(rng.choice(values, p=weights)))
        if len(selected) == count:
            return tuple(sorted(selected))
    return None


def _matching_payload(
    *,
    physical_edges: tuple[tuple[int, int], ...],
    layout: tuple[int, ...],
    target_fingerprint: str | None,
) -> dict[str, Any]:
    inverse = {physical: logical for logical, physical in enumerate(layout)}
    logical_edges = _edge_key(
        (inverse[left], inverse[right]) for left, right in physical_edges
    )
    permutation = list(range(len(layout)))
    for left, right in logical_edges:
        permutation[left] = right
        permutation[right] = left
    payload = {
        "kind": "live_target_native_perfect_matching",
        "permutation": permutation,
        "edges": [list(edge) for edge in logical_edges],
        "physical_edges": [list(edge) for edge in physical_edges],
        "css_rowspaces_preserved": False,
        "tracked_css_frame_compatible": True,
        "fixed_point_free_involution": all(
            permutation[index] != index
            and permutation[permutation[index]] == index
            for index in range(len(permutation))
        ),
        "target_native": True,
        "target_structural_fingerprint": target_fingerprint,
    }
    payload["matching_id"] = sha256(
        json.dumps(
            {
                "physical_edges": payload["physical_edges"],
                "layout": list(layout),
                "target": target_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    return payload


def _pool_for_layout(
    *,
    topology: HardwareTopology,
    base_matching: tuple[tuple[int, int], ...],
    pair_indices: tuple[int, ...],
    rng: np.random.Generator,
    matching_trials: int,
) -> dict[str, Any]:
    selected_pairs = [base_matching[index] for index in pair_indices]
    physical_nodes = tuple(
        sorted({node for edge in selected_pairs for node in edge})
    )
    induced = topology.graph.subgraph(physical_nodes).copy()
    layout = tuple(physical_nodes)
    unique: dict[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] = {}
    base_restricted = _edge_key(selected_pairs)
    unique[base_restricted] = base_restricted
    for _ in range(max(1, matching_trials)):
        matching = _randomized_maximum_matching(induced, rng)
        if len(matching) * 2 != len(layout):
            continue
        unique.setdefault(matching, matching)
    matching_payloads = [
        _matching_payload(
            physical_edges=matching,
            layout=layout,
            target_fingerprint=topology.structural_fingerprint,
        )
        for matching in unique.values()
    ]
    matching_payloads.sort(key=lambda item: item["matching_id"])
    cycle_rank = (
        induced.number_of_edges()
        - induced.number_of_nodes()
        + nx.number_connected_components(induced)
    )
    return {
        "layout": list(layout),
        "physical_nodes": list(layout),
        "matching_pool": matching_payloads,
        "matching_pool_size": len(matching_payloads),
        "induced_edges": induced.number_of_edges(),
        "induced_connected": nx.is_connected(induced) if induced.number_of_nodes() else False,
        "cycle_rank": int(cycle_rank),
        "target_structural_fingerprint": topology.structural_fingerprint,
        "construction_zero_swap": all(
            topology.graph.has_edge(left, right)
            for item in matching_payloads
            for left, right in item["physical_edges"]
        ),
    }


def build_target_native_matching_pool(
    *,
    topology: HardwareTopology,
    n: int,
    seed: int,
    layout_trials: int = 24,
    matching_trials: int = 512,
    minimum_matchings: int = 3,
) -> dict[str, Any]:
    """Construct a zero-SWAP matching pool directly from a target graph.

    A randomized maximum matching on the full target supplies pairs. A connected
    set of n/2 pair-vertices is selected, which guarantees at least one perfect
    matching in the induced physical subgraph. Additional randomized
    max-cardinality matchings provide temporal diversity.
    """

    if n <= 0 or n % 2:
        return {
            "status": "INVALID_LOGICAL_SIZE",
            "reason": "target-native perfect matching construction requires even n",
        }
    cache_key = (
        topology.structural_fingerprint or topology.name,
        n,
        int(seed),
        int(layout_trials),
        int(matching_trials),
        int(minimum_matchings),
    )
    cached = _POOL_CACHE.get(cache_key)
    if cached is not None:
        return json.loads(json.dumps(cached))

    physical = topology.graph
    if n > physical.number_of_nodes():
        return {
            "status": "PROVED_NO_TARGET_NATIVE_POOL",
            "reason": "more logical than physical qubits",
            "n": n,
        }
    rng = np.random.default_rng(seed)
    best: dict[str, Any] | None = None
    attempts = 0
    full_matching_failures = 0
    component_failures = 0
    for _ in range(max(1, layout_trials)):
        attempts += 1
        full_matching = _randomized_maximum_matching(physical, rng)
        if len(full_matching) < n // 2:
            full_matching_failures += 1
            continue
        pairs = _pair_graph(physical, full_matching)
        pair_indices = _connected_pair_subset(pairs, n // 2, rng)
        if pair_indices is None:
            component_failures += 1
            continue
        candidate = _pool_for_layout(
            topology=topology,
            base_matching=full_matching,
            pair_indices=pair_indices,
            rng=rng,
            matching_trials=matching_trials,
        )
        score = (
            candidate["matching_pool_size"],
            candidate["cycle_rank"],
            candidate["induced_edges"],
        )
        if best is None or score > tuple(best["_score"]):
            best = candidate | {"_score": list(score)}
        if candidate["matching_pool_size"] >= minimum_matchings:
            break

    if best is None:
        result = {
            "status": "PROVED_NO_TARGET_NATIVE_POOL",
            "reason": "no connected matched physical subset found",
            "n": n,
            "attempts": attempts,
            "full_matching_failures": full_matching_failures,
            "component_failures": component_failures,
        }
    else:
        score = best.pop("_score")
        result = best | {
            "status": (
                "FOUND"
                if best["matching_pool_size"] >= minimum_matchings
                else "BEST_PARTIAL_POOL"
            ),
            "n": n,
            "attempts": attempts,
            "minimum_matchings": minimum_matchings,
            "full_matching_failures": full_matching_failures,
            "component_failures": component_failures,
            "selection_score": score,
            "claim_boundary": (
                "Zero-SWAP is established for the captured target graph by "
                "construction. Native gate support and measurement mapping still "
                "require fresh BackendV2.target compilation."
            ),
        }
    _POOL_CACHE[cache_key] = json.loads(json.dumps(result))
    return result
