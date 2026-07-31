from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from functools import cached_property
from typing import Iterable

import networkx as nx
import numpy as np

from .fast_backend import anneal_layout_native
from .accel_bridge import accelerated_apsp, accelerated_layout
from .progress import ProgressManager


@dataclass(frozen=True)
class HardwareTopology:
    graph: nx.Graph
    module_of: tuple[int, ...]
    name: str
    source: str = "synthetic_proxy"
    structural_fingerprint: str | None = None

    @classmethod
    def rectangular_grid(
        cls,
        rows: int,
        cols: int,
        module_rows: int = 0,
        module_cols: int = 0,
        name: str | None = None,
    ) -> "HardwareTopology":
        graph = nx.Graph()
        n = rows * cols
        graph.add_nodes_from(range(n))
        for row in range(rows):
            for col in range(cols):
                node = row * cols + col
                if row + 1 < rows:
                    graph.add_edge(node, (row + 1) * cols + col)
                if col + 1 < cols:
                    graph.add_edge(node, row * cols + col + 1)
        modules = []
        for row in range(rows):
            for col in range(cols):
                if module_rows > 0 and module_cols > 0:
                    module = (
                        (row // module_rows)
                        * ((cols + module_cols - 1) // module_cols)
                        + col // module_cols
                    )
                else:
                    module = 0
                modules.append(module)
        return cls(
            graph=graph,
            module_of=tuple(modules),
            name=name or f"grid-{rows}x{cols}",
            source="synthetic_proxy",
        )

    @classmethod
    def from_target_snapshot(cls, path: Path) -> "HardwareTopology":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        graph = nx.Graph()
        graph.add_nodes_from(range(int(payload["num_qubits"])))
        graph.add_edges_from(
            tuple(map(int, edge)) for edge in payload["coupling_edges"]
        )
        if not graph.number_of_edges():
            raise ValueError("Target snapshot contains no two-qubit coupling edges.")
        return cls(
            graph=graph,
            module_of=tuple(0 for _ in range(graph.number_of_nodes())),
            name=str(payload.get("backend", path.stem)),
            source="live_backend_target_snapshot",
            structural_fingerprint=payload.get("structural_fingerprint"),
        )


def distance_matrix(topology: HardwareTopology) -> np.ndarray:
    accelerated = accelerated_apsp(topology.graph)
    if accelerated is not None:
        if np.any(accelerated == 32767):
            raise ValueError("Hardware topology must be connected.")
        return accelerated
    n = topology.graph.number_of_nodes()
    matrix = np.full((n, n), 32767, dtype=np.int16)
    np.fill_diagonal(matrix, 0)
    for source, lengths in nx.all_pairs_shortest_path_length(topology.graph):
        for target, length in lengths.items():
            matrix[int(source), int(target)] = int(length)
    if np.any(matrix == 32767):
        raise ValueError("Hardware topology must be connected.")
    return matrix

def interaction_edges_from_checks(
    h_x: np.ndarray, h_z: np.ndarray
) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    for matrix in (h_x, h_z):
        for row in matrix:
            support = np.flatnonzero(row).tolist()
            if len(support) < 2:
                continue
            root = int(support[0])
            for other in support[1:]:
                edges.add(tuple(sorted((root, int(other)))))
    return tuple(sorted(edges))


def _layout_cost(
    interaction_edges: Iterable[tuple[int, int]],
    distances: np.ndarray,
    layout: tuple[int, ...] | list[int],
) -> int:
    return sum(
        int(distances[layout[left], layout[right]])
        for left, right in interaction_edges
    )


def optimize_layout(
    n: int,
    interaction_edges: tuple[tuple[int, int], ...],
    topology: HardwareTopology,
    seed: int,
    iterations: int = 3000,
    progress: ProgressManager | None = None,
    progress_desc: str | None = None,
    prefer_native: bool = True,
) -> tuple[int, ...]:
    if n > topology.graph.number_of_nodes():
        raise ValueError("Topology has fewer physical than logical qubits.")
    distances = distance_matrix(topology)
    edge_array = np.asarray(interaction_edges, dtype=np.int32).reshape(-1, 2)

    if prefer_native:
        accelerated = accelerated_layout(
            edge_array,
            distances,
            nlogical=n,
            iterations=iterations,
            seed=seed,
        )
        if accelerated is not None:
            return accelerated[0]

    if prefer_native:
        native_bar = (
            progress.bar(
                total=1,
                desc=progress_desc or "Native layout annealing",
                unit="layout",
                leave=progress.leave_nested,
            )
            if progress is not None
            else None
        )
        result = anneal_layout_native(
            edge_array,
            distances,
            nlogical=n,
            iterations=iterations,
            seed=seed,
        )
        if result is not None:
            if native_bar is not None:
                native_bar.update(1)
                native_bar.close()
            return result[0]
        if native_bar is not None:
            native_bar.close()

    rng = np.random.default_rng(seed)
    layout = tuple(range(n))
    best = layout
    best_cost = _layout_cost(interaction_edges, distances, best)
    current = list(best)
    current_cost = best_cost
    steps = range(max(1, iterations))
    if progress is not None:
        steps = progress.bar(
            steps,
            total=max(1, iterations),
            desc=progress_desc or "Layout annealing",
            unit="swap",
            leave=progress.leave_nested,
        )
    for step in steps:
        left, right = rng.choice(n, 2, replace=False)
        trial = current.copy()
        trial[int(left)], trial[int(right)] = trial[int(right)], trial[int(left)]
        cost = _layout_cost(interaction_edges, distances, trial)
        temperature = max(0.01, 1.0 - step / max(1, iterations))
        accept = cost <= current_cost or rng.random() < np.exp(
            -(cost - current_cost) / temperature
        )
        if accept:
            current, current_cost = trial, cost
        if cost < best_cost:
            best, best_cost = tuple(trial), cost
    return tuple(best)


def greedy_edge_depth(edges: Iterable[tuple[int, int]]) -> int:
    graph = nx.Graph()
    graph.add_edges_from(edges)
    color_of: dict[tuple[int, int], int] = {}
    for edge in sorted(
        graph.edges(),
        key=lambda item: -(graph.degree[item[0]] + graph.degree[item[1]]),
    ):
        forbidden = {
            color
            for neighbor_edge, color in color_of.items()
            if edge[0] in neighbor_edge or edge[1] in neighbor_edge
        }
        color = 0
        while color in forbidden:
            color += 1
        color_of[tuple(sorted(edge))] = color
    return max(color_of.values(), default=-1) + 1


def compile_metrics(
    h_x: np.ndarray,
    h_z: np.ndarray,
    topology: HardwareTopology,
    seed: int,
    layout_iterations: int = 3000,
    progress: ProgressManager | None = None,
    progress_desc: str | None = None,
    prefer_native: bool = True,
) -> dict:
    n = h_x.shape[1]
    logical_edges = interaction_edges_from_checks(h_x, h_z)
    layout = optimize_layout(
        n,
        logical_edges,
        topology,
        seed=seed,
        iterations=layout_iterations,
        progress=progress,
        progress_desc=progress_desc,
        prefer_native=prefer_native,
    )
    distances = distance_matrix(topology)
    routing_distance = 0
    swaps = 0
    nonlocal_edges = 0
    crossings = 0
    for left, right in logical_edges:
        p_left, p_right = layout[left], layout[right]
        distance = int(distances[p_left, p_right])
        routing_distance += distance
        if distance > 1:
            nonlocal_edges += 1
            swaps += max(0, distance - 1)
        if topology.module_of[p_left] != topology.module_of[p_right]:
            crossings += 1
    depth = greedy_edge_depth(logical_edges)
    return {
        "two_qubit_count": len(logical_edges),
        "two_qubit_depth": depth,
        "swap_count": swaps,
        "routing_distance": routing_distance,
        "nonlocal_edges": nonlocal_edges,
        "crossing_edges": crossings,
        "layout": layout,
        "logical_edges": logical_edges,
    }


def compile_edges_metrics(
    *,
    n: int,
    logical_edges: tuple[tuple[int, int], ...],
    two_qubit_count: int,
    logical_two_qubit_depth: int,
    topology: HardwareTopology,
    seed: int,
    layout_iterations: int = 3000,
    progress: ProgressManager | None = None,
    progress_desc: str | None = None,
    prefer_native: bool = True,
) -> dict:
    """Compile an explicitly co-designed interaction union on a proxy graph."""

    logical_edges = tuple(sorted({tuple(sorted(edge)) for edge in logical_edges}))
    layout = optimize_layout(
        n,
        logical_edges,
        topology,
        seed=seed,
        iterations=layout_iterations,
        progress=progress,
        progress_desc=progress_desc,
        prefer_native=prefer_native,
    )
    distances = distance_matrix(topology)
    routing_distance = 0
    swaps = 0
    nonlocal_edges = 0
    crossings = 0
    for left, right in logical_edges:
        p_left, p_right = layout[left], layout[right]
        distance = int(distances[p_left, p_right])
        routing_distance += distance
        if distance > 1:
            nonlocal_edges += 1
            swaps += max(0, distance - 1)
        if topology.module_of[p_left] != topology.module_of[p_right]:
            crossings += 1
    return {
        "two_qubit_count": int(two_qubit_count),
        "two_qubit_depth": int(logical_two_qubit_depth),
        "swap_count": swaps,
        "routing_distance": routing_distance,
        "nonlocal_edges": nonlocal_edges,
        "crossing_edges": crossings,
        "layout": layout,
        "logical_edges": logical_edges,
        "proxy_only": True,
        "qpu_native_gate_pending": True,
    }


def _embedding_domains(logical: nx.Graph, physical: nx.Graph) -> dict[int, set[int]]:
    return {
        int(node): {
            int(target)
            for target in physical.nodes
            if physical.degree[target] >= logical.degree[node]
        }
        for node in logical.nodes
    }


def exact_zero_swap_embedding(
    *,
    n: int,
    logical_edges: tuple[tuple[int, int], ...],
    topology: HardwareTopology,
    seed: int,
    max_states: int = 2_000_000,
    timeout_seconds: float = 10.0,
    heuristic_iterations: int = 10_000,
    prefer_native: bool = True,
) -> dict:
    """Find a subgraph monomorphism of the logical union into hardware.

    The annealer is used only as a fast witness finder. The bounded CSP search
    is exact when it terminates before the state/time budget. A budget-exhausted
    result is never interpreted as proof of non-embeddability.
    """

    logical_edges = tuple(sorted({tuple(sorted(map(int, edge))) for edge in logical_edges}))
    logical = nx.Graph()
    logical.add_nodes_from(range(n))
    logical.add_edges_from(logical_edges)
    physical = topology.graph
    started = time.perf_counter()
    if n > physical.number_of_nodes():
        return {
            "status": "PROVED_NO_EMBEDDING",
            "reason": "more_logical_than_physical_qubits",
            "layout": None,
            "states": 0,
            "elapsed_seconds": 0.0,
            "exact": True,
        }
    if max(dict(logical.degree()).values(), default=0) > max(
        dict(physical.degree()).values(), default=0
    ):
        return {
            "status": "PROVED_NO_EMBEDDING",
            "reason": "logical_max_degree_exceeds_hardware",
            "layout": None,
            "states": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "exact": True,
        }

    # Fast witness search. A zero-routing layout is a valid exact witness.
    try:
        trial = optimize_layout(
            n,
            logical_edges,
            topology,
            seed=seed,
            iterations=heuristic_iterations,
            prefer_native=prefer_native,
        )
        if all(physical.has_edge(trial[a], trial[b]) for a, b in logical_edges):
            return {
                "status": "FOUND",
                "reason": "heuristic_exact_witness",
                "layout": list(map(int, trial)),
                "states": 0,
                "elapsed_seconds": time.perf_counter() - started,
                "exact": True,
            }
    except Exception:
        pass

    domains = _embedding_domains(logical, physical)
    if any(not values for values in domains.values()):
        return {
            "status": "PROVED_NO_EMBEDDING",
            "reason": "empty_degree_domain",
            "layout": None,
            "states": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "exact": True,
        }
    rng = np.random.default_rng(seed)
    assignment: dict[int, int] = {}
    used: set[int] = set()
    states = 0
    exhausted = False

    def compatible(logical_node: int, physical_node: int) -> bool:
        for neighbor in logical.neighbors(logical_node):
            if neighbor in assignment and not physical.has_edge(
                physical_node, assignment[neighbor]
            ):
                return False
        return True

    def filtered_domain(logical_node: int) -> list[int]:
        values = [
            value
            for value in domains[logical_node]
            if value not in used and compatible(logical_node, value)
        ]
        # Prefer physical vertices with enough still-free neighbours.
        rng.shuffle(values)
        values.sort(
            key=lambda value: (
                -sum(neighbor not in used for neighbor in physical.neighbors(value)),
                value,
            )
        )
        return values

    def choose_node() -> tuple[int, list[int]] | None:
        best = None
        for node in logical.nodes:
            node = int(node)
            if node in assignment:
                continue
            values = filtered_domain(node)
            score = (
                len(values),
                -sum(int(neighbor in assignment) for neighbor in logical.neighbors(node)),
                -logical.degree[node],
                node,
            )
            if best is None or score < best[0]:
                best = (score, node, values)
        return None if best is None else (best[1], best[2])

    def forward_check(node: int) -> bool:
        for neighbor in logical.neighbors(node):
            neighbor = int(neighbor)
            if neighbor not in assignment and not filtered_domain(neighbor):
                return False
        return True

    def search() -> bool:
        nonlocal states, exhausted
        if len(assignment) == n:
            return True
        if states >= max_states or time.perf_counter() - started >= timeout_seconds:
            exhausted = True
            return False
        choice = choose_node()
        if choice is None:
            return True
        node, values = choice
        if not values:
            return False
        for value in values:
            states += 1
            assignment[node] = value
            used.add(value)
            if forward_check(node) and search():
                return True
            used.remove(value)
            del assignment[node]
            if exhausted:
                return False
        return False

    found = search()
    elapsed = time.perf_counter() - started
    if found:
        layout = [assignment[index] for index in range(n)]
        return {
            "status": "FOUND",
            "reason": "bounded_exact_csp_witness",
            "layout": layout,
            "states": states,
            "elapsed_seconds": elapsed,
            "exact": True,
        }
    if exhausted:
        return {
            "status": "BUDGET_EXHAUSTED",
            "reason": "embedding_search_budget_exhausted",
            "layout": None,
            "states": states,
            "elapsed_seconds": elapsed,
            "exact": False,
        }
    return {
        "status": "PROVED_NO_EMBEDDING",
        "reason": "exhaustive_csp_search",
        "layout": None,
        "states": states,
        "elapsed_seconds": elapsed,
        "exact": True,
    }


def zero_swap_hardware_metrics(
    *,
    circuit_spec: dict,
    logical_edges: tuple[tuple[int, int], ...],
    topology: HardwareTopology,
    embedding: dict,
) -> dict:
    if embedding.get("status") != "FOUND":
        raise ValueError("A FOUND embedding is required.")
    layout = tuple(int(value) for value in embedding["layout"])
    crossings = sum(
        topology.module_of[layout[a]] != topology.module_of[layout[b]]
        for a, b in logical_edges
    )
    return {
        "two_qubit_count": int(circuit_spec["two_qubit_count"]),
        "two_qubit_depth": int(circuit_spec["logical_two_qubit_depth"]),
        "swap_count": 0,
        "routing_distance": len(logical_edges),
        "nonlocal_edges": 0,
        "crossing_edges": int(crossings),
        "layout": layout,
        "logical_edges": logical_edges,
        "proxy_only": topology.source != "live_backend_target_snapshot",
        "target_snapshot_embedding": topology.source == "live_backend_target_snapshot",
        "target_structural_fingerprint": topology.structural_fingerprint,
        "embedding_search": embedding,
        "qpu_native_gate_pending": True,
    }
