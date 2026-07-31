from __future__ import annotations

import math
from typing import Any

import networkx as nx


def _degeneracy_lower_bound(graph: nx.Graph) -> int:
    if not graph.number_of_edges():
        return 0
    return int(max(nx.core_number(graph).values(), default=0))


def _vertex_separation_upper_bound(
    graph: nx.Graph, order: list[int]
) -> int:
    """Pathwidth/treewidth upper bound from a fixed vertex order."""

    position = {node: index for index, node in enumerate(order)}
    active: set[int] = set()
    maximum = 0
    for index, node in enumerate(order):
        if any(position[neighbor] > index for neighbor in graph.neighbors(node)):
            active.add(node)
        for neighbor in graph.neighbors(node):
            if position[neighbor] < index and not any(
                position[next_neighbor] > index
                for next_neighbor in graph.neighbors(neighbor)
            ):
                active.discard(neighbor)
        maximum = max(maximum, len(active))
    return max(0, int(maximum))


def treewidth_bounds(
    graph: nx.Graph,
    *,
    min_fill_node_limit: int = 1500,
    fallback_order: list[int] | None = None,
) -> tuple[int, int]:
    """Rigorous lower bound and valid upper bound without large-graph stalls."""

    if graph.number_of_nodes() == 0:
        return 0, 0
    lower = _degeneracy_lower_bound(graph)
    if graph.number_of_nodes() <= min_fill_node_limit:
        upper, _ = nx.approximation.treewidth_min_fill_in(graph)
        return lower, int(upper)
    orders: list[list[int]] = []
    if fallback_order is not None:
        orders.append(list(fallback_order))
    orders.append(list(graph.nodes()))
    if graph.number_of_edges():
        orders.append(list(nx.utils.reverse_cuthill_mckee_ordering(graph)))
    upper = min(_vertex_separation_upper_bound(graph, order) for order in orders)
    return lower, max(lower, int(upper))


def _tensor_index_pathwidth_upper(
    inputs: list[tuple[str, ...]],
) -> int:
    """Line-graph pathwidth upper bound from the chronological tensor order."""

    last_owner: dict[str, int] = {}
    for tensor_index, tensor in enumerate(inputs):
        for index in tensor:
            last_owner[index] = tensor_index
    active: set[str] = set()
    maximum = 0
    for tensor_index, tensor in enumerate(inputs):
        active.update(tensor)
        maximum = max(maximum, len(active) - 1)
        for index in tensor:
            if last_owner[index] == tensor_index:
                active.discard(index)
    return max(0, int(maximum))


def tensor_network_description(circuit_spec: dict) -> dict[str, Any]:
    """Build a doubled ket/bra tensor network for one output probability."""

    n = int(circuit_spec["n"])
    inputs: list[tuple[str, ...]] = []
    labels: list[str] = []
    current_ket = [f"q{q}_k_0" for q in range(n)]
    current_bra = [f"q{q}_b_0" for q in range(n)]
    clocks_ket = [0] * n
    clocks_bra = [0] * n

    for qubit in range(n):
        inputs.append((current_ket[qubit], current_bra[qubit]))
        labels.append(f"rho0_q{qubit}")

    for gate_index, gate in enumerate(circuit_spec["gates"]):
        name = str(gate["name"])
        if name == "measure":
            continue
        qubits = tuple(int(value) for value in gate["qubits"])
        ket_in = tuple(current_ket[q] for q in qubits)
        bra_in = tuple(current_bra[q] for q in qubits)
        ket_out = []
        bra_out = []
        for qubit in qubits:
            clocks_ket[qubit] += 1
            clocks_bra[qubit] += 1
            new_k = f"q{qubit}_k_{clocks_ket[qubit]}"
            new_b = f"q{qubit}_b_{clocks_bra[qubit]}"
            current_ket[qubit] = new_k
            current_bra[qubit] = new_b
            ket_out.append(new_k)
            bra_out.append(new_b)
        inputs.append(ket_in + tuple(ket_out))
        labels.append(f"ket_{gate_index}_{name}")
        inputs.append(bra_in + tuple(bra_out))
        labels.append(f"bra_{gate_index}_{name}")

    for qubit in range(n):
        inputs.append((current_ket[qubit], current_bra[qubit]))
        labels.append(f"project_q{qubit}")

    size_dict = {index: 2 for tensor in inputs for index in tensor}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(inputs)))
    owners: dict[str, list[int]] = {}
    for tensor_index, tensor in enumerate(inputs):
        for index in tensor:
            owners.setdefault(index, []).append(tensor_index)
    for index, tensors in owners.items():
        for left in range(len(tensors)):
            for right in range(left + 1, len(tensors)):
                graph.add_edge(tensors[left], tensors[right], index=index)

    return {
        "inputs": inputs,
        "output": (),
        "size_dict": size_dict,
        "labels": labels,
        "tensor_graph": graph,
    }


def _empty_cotengra(status: str, *, available: bool | None) -> dict[str, Any]:
    return {
        "available": available,
        "status": status,
        "mode": None,
        "contraction_flops": None,
        "contraction_width_log2": None,
        "max_intermediate_elements": None,
        "peak_memory_bytes_complex128": None,
        "path_length": None,
    }


def _tree_metrics(tree: Any) -> dict[str, Any]:
    width = float(tree.contraction_width())
    cost = float(tree.contraction_cost())
    maximum = float(2.0**width)
    peak = None
    for name in ("get_peak_size", "max_size", "max_contraction_size"):
        method = getattr(tree, name, None)
        if method is None:
            continue
        try:
            peak = float(method())
            break
        except Exception:
            continue
    if peak is None:
        peak = maximum
    path = tree.get_path() if hasattr(tree, "get_path") else []
    return {
        "contraction_flops": cost,
        "contraction_width_log2": width,
        "max_intermediate_elements": maximum,
        "peak_intermediate_elements": peak,
        "peak_memory_bytes_complex128": peak * 16.0,
        "path_length": len(path),
    }


def cotengra_metrics(
    inputs: list[tuple[str, ...]],
    output: tuple[str, ...],
    size_dict: dict[str, int],
    settings: dict,
) -> dict[str, Any]:
    """Find a low-cost contraction path under a preregistered search budget.

    The circuit optimizer maximizes the resulting minimized cost across circuit
    schedules. Cotengra itself always remains the adversary and minimizes cost.
    """

    if not bool(settings.get("enabled", True)):
        return _empty_cotengra(
            "DISABLED_FOR_BULK_SEARCH", available=None
        )
    try:
        import cotengra as ctg
    except ImportError:
        return _empty_cotengra("NOT_INSTALLED", available=False)

    mode = str(settings.get("mode", "deep")).lower()
    try:
        if mode == "short":
            repeats = int(settings.get("max_repeats", 32))
            optimizer = ctg.RandomGreedyOptimizer(
                max_repeats=repeats,
                temperature=tuple(settings.get("temperature", [0.01, 0.1])),
                seed=int(settings.get("seed", 0)),
            )
            tree = optimizer.search(inputs, output, size_dict)
        else:
            kwargs: dict[str, Any] = {
                "methods": list(settings.get("methods", ["greedy"])),
                "max_repeats": int(settings.get("max_repeats", 128)),
                "progbar": bool(settings.get("progbar", False)),
                "minimize": str(settings.get("minimize", "combo")),
                "optlib": str(settings.get("optlib", "random")),
                "parallel": settings.get("parallel", False),
            }
            if settings.get("max_time") is not None:
                kwargs["max_time"] = settings["max_time"]
            if bool(settings.get("reconfigure", True)):
                kwargs["reconf_opts"] = dict(settings.get("reconf_opts", {}))
            optimizer = ctg.HyperOptimizer(**kwargs)
            tree = optimizer.search(inputs, output, size_dict)
        return {
            "available": True,
            "status": "PASS",
            "mode": mode,
            **_tree_metrics(tree),
        }
    except Exception as error:
        payload = _empty_cotengra("FAILED", available=True)
        payload.update({"mode": mode, "error": str(error)})
        return payload


def cotengra_metrics_for_circuit(
    circuit_spec: dict,
    settings: dict,
) -> dict[str, Any]:
    description = tensor_network_description(circuit_spec)
    return cotengra_metrics(
        description["inputs"],
        description["output"],
        description["size_dict"],
        settings,
    )


def graph_metrics_for_circuit(
    circuit_spec: dict,
    *,
    min_fill_node_limit: int = 1500,
) -> dict[str, Any]:
    description = tensor_network_description(circuit_spec)
    tensor_graph: nx.Graph = description["tensor_graph"]
    line_graph = nx.line_graph(tensor_graph)
    tensor_lower, tensor_upper = treewidth_bounds(
        tensor_graph, min_fill_node_limit=min_fill_node_limit
    )
    chronological_upper = _tensor_index_pathwidth_upper(description["inputs"])
    if line_graph.number_of_nodes() <= min_fill_node_limit:
        line_lower, line_upper = treewidth_bounds(
            line_graph, min_fill_node_limit=min_fill_node_limit
        )
        line_method = "networkx_min_fill"
    else:
        line_lower = _degeneracy_lower_bound(line_graph)
        line_upper = max(line_lower, chronological_upper)
        line_method = "chronological_tensor_pathwidth_upper"
    return {
        "description": description,
        "tensor_graph_treewidth_lower": tensor_lower,
        "tensor_graph_treewidth_upper": tensor_upper,
        "line_graph_treewidth_lower": int(line_lower),
        "line_graph_treewidth_upper": int(line_upper),
        "line_graph_upper_method": line_method,
        "chronological_line_graph_pathwidth_upper": int(chronological_upper),
        "min_fill_node_limit": int(min_fill_node_limit),
        "tensor_count": tensor_graph.number_of_nodes(),
        "tensor_network_edges": tensor_graph.number_of_edges(),
    }


def _cut_metrics(circuit_spec: dict) -> dict[str, Any]:
    n = int(circuit_spec["n"])
    twoq = [
        tuple(int(v) for v in gate["qubits"])
        for gate in circuit_spec["gates"]
        if len(gate["qubits"]) == 2
    ]
    crossings = []
    for cut in range(1, n):
        crossing = sum(
            1
            for left, right in twoq
            if (left < cut <= right) or (right < cut <= left)
        )
        crossings.append(crossing)
    best_cut = min(crossings, default=0)
    worst_cut = max(crossings, default=0)
    mps_log2_bond = min(120, worst_cut)
    sf_log2 = min(1023, math.ceil(n / 2) + best_cut)
    return {
        "linear_cut_crossings_min": int(best_cut),
        "linear_cut_crossings_max": int(worst_cut),
        "mps_bond_dimension_proxy": float(2.0**mps_log2_bond),
        "mps_log2_bond_proxy": int(mps_log2_bond),
        "schrodinger_feynman_operations_proxy": float(2.0**sf_log2),
        "schrodinger_feynman_log2_proxy": int(sf_log2),
    }


def _non_clifford_metrics(circuit_spec: dict, settings: dict) -> dict[str, Any]:
    angles = []
    for gate in circuit_spec["gates"]:
        if gate["name"] not in {"rz", "rzz", "rxx"}:
            continue
        angle = abs(float(gate.get("angle") or 0.0))
        distance = abs(angle / (math.pi / 2) - round(angle / (math.pi / 2)))
        if distance > 1e-10:
            angles.append(angle)
    count = len(angles)
    stabilizer_log2 = sum(
        math.log2(max(1.0, 1.0 + abs(math.sin(angle)))) for angle in angles
    )
    truncation_order = int(settings.get("pauli_truncation_order", 6))
    branching_probabilities = [math.sin(angle / 2.0) ** 2 for angle in angles]
    mean_branch = sum(branching_probabilities)
    cumulative = 0.0
    for order in range(truncation_order + 1):
        cumulative += (
            math.exp(-mean_branch)
            * mean_branch**order
            / math.factorial(order)
        )
    retained = min(1.0, max(0.0, cumulative))
    return {
        "non_clifford_rotations": count,
        "stabilizer_decomposition_log2_cost": float(stabilizer_log2),
        "stabilizer_decomposition_cost": float(
            2.0 ** min(1023.0, stabilizer_log2)
        ),
        "pauli_path_truncation_order": truncation_order,
        "pauli_path_retained_mass_proxy": retained,
        "pauli_path_discarded_mass_proxy": 1.0 - retained,
    }


def spacetime_hardness_metrics(
    *,
    circuit_spec: dict,
    verify_operations: float,
    assumptions: tuple[str, ...],
    settings: dict | None = None,
    graph_metrics: dict[str, Any] | None = None,
    cotengra_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = settings or {}
    graph_metrics = graph_metrics or graph_metrics_for_circuit(circuit_spec)
    description = graph_metrics["description"]
    cotengra = cotengra_override or cotengra_metrics(
        description["inputs"],
        description["output"],
        description["size_dict"],
        settings.get("cotengra", {}),
    )
    cuts = _cut_metrics(circuit_spec)
    nonclifford = _non_clifford_metrics(circuit_spec, settings)
    n = int(circuit_spec["n"])
    statevector = float(n * (2.0**n))
    fallback_tensor = float(
        max(1, graph_metrics["tensor_count"])
        * 2.0 ** max(1, graph_metrics["line_graph_treewidth_upper"])
    )
    tensor_cost = (
        float(cotengra["contraction_flops"])
        if cotengra.get("contraction_flops") is not None
        else fallback_tensor
    )
    attack_costs = {
        "statevector": statevector,
        "cotengra_or_line_graph": tensor_cost,
        "mps": float(n * max(1.0, cuts["mps_bond_dimension_proxy"]) ** 2),
        "schrodinger_feynman": cuts["schrodinger_feynman_operations_proxy"],
        "stabilizer_decomposition": nonclifford[
            "stabilizer_decomposition_cost"
        ],
    }
    best_name = min(attack_costs, key=attack_costs.get)
    best = float(attack_costs[best_name])
    gamma = math.log10(max(best, 1.0) / max(float(verify_operations), 1.0))
    return {
        "method": "full_doubled_space_time_tensor_network",
        "verify_operations": float(verify_operations),
        "statevector_operations": statevector,
        "tensor_proxy_operations": tensor_cost,
        "mps_proxy_operations": attack_costs["mps"],
        "stabilizer_rank_proxy_operations": attack_costs[
            "stabilizer_decomposition"
        ],
        "schrodinger_feynman_proxy_operations": attack_costs[
            "schrodinger_feynman"
        ],
        "best_attack_name": best_name,
        "best_attack_operations": best,
        "gamma_log10": float(gamma),
        "treewidth_lower_bound": graph_metrics["line_graph_treewidth_lower"],
        "treewidth_upper_bound": graph_metrics["line_graph_treewidth_upper"],
        "tensor_graph_treewidth_lower": graph_metrics[
            "tensor_graph_treewidth_lower"
        ],
        "tensor_graph_treewidth_upper": graph_metrics[
            "tensor_graph_treewidth_upper"
        ],
        "line_graph_treewidth_lower": graph_metrics[
            "line_graph_treewidth_lower"
        ],
        "line_graph_treewidth_upper": graph_metrics[
            "line_graph_treewidth_upper"
        ],
        "tensor_count": graph_metrics["tensor_count"],
        "tensor_network_edges": graph_metrics["tensor_network_edges"],
        "cotengra": cotengra,
        "cuts": cuts,
        "non_clifford": nonclifford,
        "attack_costs": attack_costs,
        "assumptions": assumptions,
        "claim_scope": (
            "Conditional and limited to registered contraction, MPS, "
            "Schrodinger-Feynman, stabilizer and statevector attacks."
        ),
    }
