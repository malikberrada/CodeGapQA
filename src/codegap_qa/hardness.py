from __future__ import annotations

import math
import networkx as nx


def treewidth_bounds(graph: nx.Graph) -> tuple[int, int]:
    if graph.number_of_nodes() == 0:
        return 0, 0
    upper, _ = nx.approximation.treewidth_min_fill_in(graph)
    # Degeneracy is a rigorous lower bound on treewidth.
    core_numbers = nx.core_number(graph) if graph.number_of_edges() else {0: 0}
    lower = max(core_numbers.values(), default=0)
    return int(lower), int(upper)


def simulator_costs(
    n: int,
    graph: nx.Graph,
    non_clifford_count: int,
    cut_size: int,
    verify_operations: float,
    assumptions: tuple[str, ...],
) -> dict:
    tw_lower, tw_upper = treewidth_bounds(graph)
    statevector = float(n * (2**n))
    tensor_proxy = float(max(1, graph.number_of_edges()) * (2 ** max(1, tw_upper)))
    mps_proxy = float(n * (2 ** min(n // 2, max(1, tw_upper))) ** 2)
    stabilizer_rank = float(
        max(1, n) * (2 ** min(non_clifford_count, 0.47 * non_clifford_count))
    )
    schrodinger_feynman = float(
        max(1, n) * (2 ** max(1, n - cut_size)) + (2 ** max(1, cut_size))
    )
    best = min(
        statevector,
        tensor_proxy,
        mps_proxy,
        stabilizer_rank,
        schrodinger_feynman,
    )
    gamma = math.log10(max(best, 1.0) / max(verify_operations, 1.0))
    return {
        "verify_operations": float(verify_operations),
        "statevector_operations": statevector,
        "tensor_proxy_operations": tensor_proxy,
        "mps_proxy_operations": mps_proxy,
        "stabilizer_rank_proxy_operations": stabilizer_rank,
        "schrodinger_feynman_proxy_operations": schrodinger_feynman,
        "best_attack_operations": float(best),
        "gamma_log10": float(gamma),
        "treewidth_upper_bound": tw_upper,
        "treewidth_lower_bound": tw_lower,
        "assumptions": assumptions,
        "warning": (
            "These are transparent cost models and implemented-attack proxies, "
            "not a lower bound against every classical algorithm."
        ),
    }


def interaction_graph(n: int, edges: tuple[tuple[int, int], ...]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    return graph
