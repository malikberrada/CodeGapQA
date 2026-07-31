from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx

from .qpu_types import CompiledQPUCandidate


def _imports():
    try:
        from qiskit import qasm3, transpile
        from qiskit.circuit import QuantumCircuit
    except ImportError as error:
        raise RuntimeError('QPU compilation requires pip install -e ".[qpu]".') from error
    return qasm3, transpile, QuantumCircuit


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_science_circuit(spec: dict) -> Any:
    """Build the co-designed circuit, preserving explicit temporal layers."""

    _, _, QuantumCircuit = _imports()
    n = int(spec["n"])
    circuit = QuantumCircuit(n, n, name="codegap_science_v040")
    measured = False
    for gate in spec["gates"]:
        name = str(gate["name"])
        qubits = tuple(int(value) for value in gate["qubits"])
        angle = float(gate.get("angle") or 0.0)
        if name == "h":
            circuit.h(qubits[0])
        elif name == "rz":
            circuit.rz(angle, qubits[0])
        elif name == "rzz":
            circuit.rzz(angle, qubits[0], qubits[1])
        elif name == "rxx":
            circuit.rxx(angle, qubits[0], qubits[1])
        elif name == "barrier":
            # CODEGAP_V101_FOLD_BARRIERS
            circuit.barrier(*qubits)
        elif name == "measure":
            circuit.measure(qubits[0], qubits[0])
            measured = True
        else:
            raise ValueError(f"Unsupported co-designed gate: {name}")
    if not measured:
        circuit.measure(range(n), range(n))
    return circuit


def logical_interaction_graph(spec: dict) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(int(spec["n"])))
    graph.add_edges_from(
        (int(a), int(b)) for a, b in spec["union_two_qubit_edges"]
    )
    return graph


def physical_graph(target: Any) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(int(target.num_qubits)))
    for name in target.operation_names:
        try:
            operation_map = target[name]
        except Exception:
            continue
        if not hasattr(operation_map, "keys"):
            continue
        for qargs in operation_map.keys():
            if qargs is not None and len(qargs) == 2:
                graph.add_edge(int(qargs[0]), int(qargs[1]))
    return graph


def enumerate_layouts(
    logical: nx.Graph,
    physical: nx.Graph,
    max_layouts: int,
) -> list[tuple[int, ...]]:
    matcher = nx.algorithms.isomorphism.GraphMatcher(physical, logical)
    layouts: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for physical_to_logical in matcher.subgraph_monomorphisms_iter():
        inverse = {
            int(logical_node): int(physical_node)
            for physical_node, logical_node in physical_to_logical.items()
        }
        if set(inverse) != set(logical.nodes):
            continue
        layout = tuple(inverse[index] for index in range(logical.number_of_nodes()))
        if layout not in seen:
            seen.add(layout)
            layouts.append(layout)
        if len(layouts) >= max_layouts:
            break
    return layouts


def _supported(target: Any, name: str, qargs: tuple[int, ...]) -> bool:
    try:
        return bool(target.instruction_supported(operation_name=name, qargs=qargs))
    except Exception:
        return False


def _instruction_property(
    target: Any,
    name: str,
    qargs: tuple[int, ...],
    field: str,
) -> float | None:
    try:
        properties = target[name].get(qargs)
    except Exception:
        return None
    value = getattr(properties, field, None) if properties is not None else None
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _measurement_map(circuit: Any) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for instruction in circuit.data:
        if str(instruction.operation.name) != "measure":
            continue
        physical = int(circuit.find_bit(instruction.qubits[0]).index)
        classical = int(circuit.find_bit(instruction.clbits[0]).index)
        mapping[classical] = physical
    return mapping


def _twoq_depth(circuit: Any) -> int:
    return int(
        circuit.depth(
            filter_function=lambda instruction: len(instruction.qubits) == 2
        )
        or 0
    )


def compile_layout(
    logical_circuit: Any,
    backend: Any,
    layout: tuple[int, ...],
    logical_union_edges: tuple[tuple[int, int], ...],
    output_qasm: Path,
    seed: int,
) -> CompiledQPUCandidate:
    qasm3, transpile, _ = _imports()
    target = backend.target
    compiled = transpile(
        logical_circuit,
        target=target,
        initial_layout=list(layout),
        optimization_level=3,
        routing_method="none",
        seed_transpiler=seed,
        approximation_degree=1.0,
        ignore_backend_supplied_default_methods=True,
    )
    operations = {
        str(name): int(count) for name, count in compiled.count_ops().items()
    }
    violations = []
    calibrated_log_error = 0.0
    calibrated = 0
    considered = 0
    twoq_count = 0
    nonlocal_edges = 0
    native_twoq_names: set[str] = set()
    physical = physical_graph(target)
    for instruction in compiled.data:
        name = str(instruction.operation.name)
        if name == "barrier":
            continue
        qargs = tuple(
            int(compiled.find_bit(qubit).index) for qubit in instruction.qubits
        )
        supported = _supported(target, name, qargs)
        if not supported:
            violations.append({"operation": name, "qargs": list(qargs)})
        if name not in {"measure", "delay"}:
            considered += 1
            error = _instruction_property(target, name, qargs, "error")
            if error is not None and 0.0 <= error < 1.0:
                calibrated += 1
                calibrated_log_error += -math.log1p(-error)
        if len(qargs) == 2:
            twoq_count += 1
            native_twoq_names.add(name)
            if not supported or not physical.has_edge(qargs[0], qargs[1]):
                nonlocal_edges += 1
    measurement_map = _measurement_map(compiled)
    expected = {index: physical for index, physical in enumerate(layout)}
    output_qasm.parent.mkdir(parents=True, exist_ok=True)
    output_qasm.write_text(qasm3.dumps(compiled), encoding="utf-8")
    swap_count = int(operations.get("swap", 0))
    calibrated_fraction = calibrated / considered if considered else 0.0
    embedding_valid = all(
        physical.has_edge(layout[left], layout[right])
        for left, right in logical_union_edges
    )
    native_twoq_valid = bool(
        twoq_count > 0
        and nonlocal_edges == 0
        and all(name in target.operation_names for name in native_twoq_names)
    )
    score = (
        len(violations),
        swap_count,
        nonlocal_edges,
        not embedding_valid,
        not native_twoq_valid,
        not (measurement_map == expected),
        -calibrated_fraction,
        calibrated_log_error,
        _twoq_depth(compiled),
        twoq_count,
        int(compiled.depth()),
        layout,
    )
    return CompiledQPUCandidate(
        candidate_id="",
        logical_qubits=logical_circuit.num_qubits,
        layout=layout,
        qasm_path=str(output_qasm),
        qasm_sha256=sha256_file(output_qasm),
        swap_count=swap_count,
        two_qubit_count=twoq_count,
        two_qubit_depth=_twoq_depth(compiled),
        compiled_depth=int(compiled.depth()),
        target_violations=tuple(violations),
        calibrated_log_error=calibrated_log_error,
        calibrated_fraction=calibrated_fraction,
        estimated_success=(
            math.exp(-calibrated_log_error) if calibrated else None
        ),
        measurement_map=measurement_map,
        expected_measurement_map=expected,
        measurement_map_valid=measurement_map == expected,
        nonlocal_edges=nonlocal_edges,
        target_embedding_valid=embedding_valid,
        native_two_qubit_gates_valid=native_twoq_valid,
        native_two_qubit_gate_names=tuple(sorted(native_twoq_names)),
        score=score,
    )


def compile_candidate(
    candidate: dict,
    backend: Any,
    output: Path,
    max_layouts: int,
    seed: int,
    excluded_physical_qubits: tuple[int, ...] = (),
) -> dict:
    spec = candidate["exact_artifacts"].get("circuit_spec")
    if spec is None:
        raise ValueError(
            "QPU-native v0.4 compilation requires exact_artifacts.circuit_spec."
        )
    logical_circuit = build_science_circuit(spec)
    qasm3, _, _ = _imports()
    output.mkdir(parents=True, exist_ok=True)
    logical_qasm = output / "logical_science.qasm3"
    logical_qasm.write_text(qasm3.dumps(logical_circuit), encoding="utf-8")
    logical_graph = logical_interaction_graph(spec)
    # CODEGAP_V068_LAYOUT_BLACKLIST
    excluded = tuple(
        sorted(
            {
                int(value)
                for value in excluded_physical_qubits
            }
        )
    )

    invalid_excluded = [
        value
        for value in excluded
        if value < 0
        or value >= int(backend.target.num_qubits)
    ]

    if invalid_excluded:
        raise ValueError(
            "Excluded physical qubits outside backend range: "
            f"{invalid_excluded}"
        )

    physical = physical_graph(backend.target)
    physical.remove_nodes_from(excluded)

    schedule_evidence = candidate.get("exact_artifacts", {}).get(
        "schedule_search", {}
    )
    pinned = schedule_evidence.get("pinned_layout")

    if pinned is not None:
        pinned_layout = tuple(
            int(value)
            for value in pinned
        )

        pinned_forbidden = bool(
            set(pinned_layout).intersection(excluded)
        )

        pinned_valid = bool(
            not pinned_forbidden
            and len(pinned_layout)
            == logical_graph.number_of_nodes()
            and len(set(pinned_layout))
            == len(pinned_layout)
            and all(
                physical.has_edge(
                    pinned_layout[left],
                    pinned_layout[right],
                )
                for left, right in logical_graph.edges
            )
        )

        if pinned_valid:
            layouts = [pinned_layout]
            layout_source = (
                "target_native_pinned_layout"
            )
        elif pinned_forbidden:
            layouts = enumerate_layouts(
                logical_graph,
                physical,
                max_layouts,
            )
            layout_source = (
                "enumerated_after_excluded_pinned_layout"
            )
        else:
            layouts = []
            layout_source = (
                "invalid_target_native_pinned_layout"
            )
    else:
        layouts = enumerate_layouts(
            logical_graph,
            physical,
            max_layouts,
        )
        layout_source = (
            "enumerated_subgraph_monomorphisms"
        )
    backend_name = str(getattr(backend, "name", "unknown"))
    if not layouts:
        return {
            "candidate_id": candidate["candidate_id"],
            "backend": backend_name,
            "status": "NO_ZERO_SWAP_TARGET_EMBEDDING",
            "layouts_examined": 0,
            "layout_source": layout_source,
            "logical_qasm_path": str(logical_qasm),
            "requirements": {
                "target_embedding": True,
                "zero_swap": True,
                "native_two_qubit_gates": True,
                "measurement_map": True,
            },
        }
    union_edges = tuple(tuple(edge) for edge in spec["union_two_qubit_edges"])
    compiled_items = []
    for index, layout in enumerate(layouts):
        try:
            result = compile_layout(
                logical_circuit,
                backend,
                layout,
                union_edges,
                output / f"layout_{index:04d}.qasm3",
                seed + index,
            )
        except Exception as error:
            compiled_items.append({"layout": list(layout), "error": str(error)})
            continue
        payload = result.to_dict()
        payload["candidate_id"] = candidate["candidate_id"]
        compiled_items.append(payload)
    valid = [
        item
        for item in compiled_items
        if "error" not in item
        and item["swap_count"] == 0
        and item["nonlocal_edges"] == 0
        and not item["target_violations"]
        and item["measurement_map_valid"]
        and item["target_embedding_valid"]
        and item["native_two_qubit_gates_valid"]
    ]
    if not valid:
        return {
            "candidate_id": candidate["candidate_id"],
            "backend": backend_name,
            "status": "NO_VALID_QPU_NATIVE_COMPILATION",
            "layouts_examined": len(layouts),
            "attempts": compiled_items,
            "logical_qasm_path": str(logical_qasm),
        }
    valid.sort(key=lambda item: tuple(item["score"]))
    best = valid[0]
    selected_qasm = output / "science.qasm3"
    source = Path(best["qasm_path"])
    selected_qasm.write_bytes(source.read_bytes())
    best["qasm_path"] = str(selected_qasm)
    best["qasm_sha256"] = sha256_file(selected_qasm)
    return {
        "candidate_id": candidate["candidate_id"],
        "backend": backend_name,
        "status": "PASS",
        "layouts_examined": len(layouts),
        "best": best,
        "valid_layouts": len(valid),
        "logical_qasm_path": str(logical_qasm),
        "circuit_spec_sha256": sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
