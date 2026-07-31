from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lightcone import backward_lightcone
from .qpu_provider import _validate_submission_qasm


def _imports():
    try:
        from qiskit import qasm3, transpile
        from qiskit.circuit import QuantumCircuit
    except ImportError as error:
        raise RuntimeError('QPU probes require pip install -e ".[qpu]".') from error
    return qasm3, transpile, QuantumCircuit



def _rewrite_parameterized_cp_qasm(source: str) -> tuple[str, int]:
    """Replace parameterized OpenQASM CP gates with RZ/CX instructions.

    For each instruction::

        cp(theta) control, target;

    the exporter writes an equivalent sequence, up to a global phase::

        rz(theta/2) control;
        cx control, target;
        rz(-theta/2) target;
        cx control, target;
        rz(theta/2) target;

    Global phase is irrelevant to measured probabilities and cancels between
    U and U† inside the echo probes.
    """
    import re

    operand = r"(?:\$\d+|[A-Za-z_]\w*\s*\[\s*\d+\s*\])"
    pattern = re.compile(
        rf"^(?P<indent>\s*)cp\s*\((?P<angle>.+)\)\s+"
        rf"(?P<control>{operand})\s*,\s*"
        rf"(?P<target>{operand})\s*;\s*(?://.*)?$",
        re.IGNORECASE,
    )

    output: list[str] = []
    replacements = 0

    for line in source.splitlines():
        match = pattern.match(line)
        if match is None:
            output.append(line)
            continue

        indent = match.group("indent")
        angle = match.group("angle").strip()
        control = match.group("control").strip()
        target = match.group("target").strip()
        half = f"(({angle})/2)"
        negative_half = f"-(({angle})/2)"

        output.extend(
            [
                f"{indent}rz({half}) {control};",
                f"{indent}cx {control}, {target};",
                f"{indent}rz({negative_half}) {target};",
                f"{indent}cx {control}, {target};",
                f"{indent}rz({half}) {target};",
            ]
        )
        replacements += 1

    suffix = "\n" if source.endswith("\n") else ""
    return "\n".join(output) + suffix, replacements


def _compile_probe(
    circuit: Any,
    backend: Any,
    layout: tuple[int, ...],
    path: Path,
    seed: int,
    *,
    optimization_level: int = 3,
    minimum_two_qubit_gates: int = 0,
    strip_barriers_from_qasm: bool = False,
    decompose_controlled_phase: bool = False,
) -> dict[str, int]:
    qasm3, transpile, _ = _imports()

    if decompose_controlled_phase:
        # Open Quantum's current QASM precompiler maps parameterized `cp`
        # to a fixed-arity `cphaseshift` form and rejects its angle. Expand
        # CP into one-qubit phase rotations and CX gates before target
        # transpilation. Qiskit then translates this decomposition to the
        # actual backend target without reintroducing CP at level 0.
        circuit = circuit.decompose(
            gates_to_decompose=[
                'cp',
                'cphase',
                'cphaseshift',
            ],
            reps=4,
        )

    compiled = transpile(
        circuit,
        target=backend.target,
        initial_layout=list(layout),
        optimization_level=optimization_level,
        routing_method='none',
        seed_transpiler=seed,
        approximation_degree=1.0,
        ignore_backend_supplied_default_methods=True,
    )
    swap_count = int(compiled.count_ops().get('swap', 0))
    if swap_count != 0:
        raise RuntimeError('Probe compilation inserted SWAPs.')

    two_qubit_count = sum(
        1
        for instruction in compiled.data
        if int(getattr(instruction.operation, 'num_qubits', 0)) == 2
    )
    if two_qubit_count < int(minimum_two_qubit_gates):
        raise RuntimeError(
            'Probe compilation erased the scientific circuit: '
            f'two_qubit_count={two_qubit_count}, '
            f'minimum_required={minimum_two_qubit_gates}.'
        )

    ensure_physical = getattr(compiled, 'ensure_physical', None)
    if callable(ensure_physical):
        ensure_physical()
    source = qasm3.dumps(compiled)
    if strip_barriers_from_qasm:
        source = '\n'.join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith('barrier ')
        ) + '\n'

    cp_rewrites = 0
    if decompose_controlled_phase:
        import re

        source, cp_rewrites = _rewrite_parameterized_cp_qasm(source)
        forbidden = re.findall(
            r'(?mi)^\s*(cp|cphase|cphaseshift)\s*\(',
            source,
        )
        if forbidden:
            raise RuntimeError(
                'Controlled-phase QASM rewrite failed; unsupported '
                f'gates remain: {sorted(set(forbidden))}.'
            )

    backend_width = int(getattr(backend, 'num_qubits', compiled.num_qubits))
    _validate_submission_qasm(source, backend_num_qubits=backend_width)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')
    return {
        'swap_count': swap_count,
        'two_qubit_count': int(two_qubit_count),
        'compiled_depth': int(compiled.depth()),
        'controlled_phase_qasm_rewrites': int(cp_rewrites),
    }


def _basis_probe(pattern: tuple[int, ...], name: str) -> Any:
    _, _, QuantumCircuit = _imports()
    n = len(pattern)
    circuit = QuantumCircuit(n, n, name=name)
    for qubit, bit in enumerate(pattern):
        if bit:
            circuit.x(qubit)

    if name in {'basis_all0', 'local_basis_all0'} and n:
        # Preserve the |0...0> target while ensuring that Open Quantum sees
        # an executable physical-qubit instruction before measurement.
        # This pair is compiled at optimization level 0 below.
        circuit.x(0)
        circuit.x(0)

    circuit.measure(range(n), range(n))
    return circuit


def _echo_probe(science_qasm: Path, repetitions: int) -> Any:
    qasm3, _, QuantumCircuit = _imports()
    science = qasm3.load(science_qasm)
    unitary = science.remove_final_measurements(inplace=False)
    n = unitary.num_qubits
    circuit = QuantumCircuit(n, n, name=f'echo_x{repetitions}')
    for _ in range(repetitions):
        circuit.compose(unitary, inplace=True)
        circuit.barrier()
        circuit.compose(unitary.inverse(), inplace=True)
        circuit.barrier()
    circuit.measure(range(n), range(n))
    return circuit



def _apply_spec_gate(circuit: Any, gate: dict, local_map: dict[int, int]) -> None:
    name = str(gate["name"])
    qubits = tuple(local_map[int(value)] for value in gate["qubits"])
    angle = float(gate.get("angle") or 0.0)
    if name == "h":
        circuit.h(qubits[0])
    elif name == "rz":
        circuit.rz(angle, qubits[0])
    elif name == "rzz":
        circuit.rzz(angle, qubits[0], qubits[1])
    elif name == "rxx":
        circuit.rxx(angle, qubits[0], qubits[1])
    else:
        raise ValueError(f"Unsupported local light-cone gate: {name}")


def _local_science_circuit(
    circuit_spec: dict,
    support: tuple[int, ...],
    name: str,
) -> tuple[Any, tuple[int, ...], list[dict]]:
    _, _, QuantumCircuit = _imports()
    cone, gates = backward_lightcone(circuit_spec["gates"], support)
    ordered = tuple(sorted(int(value) for value in cone))
    local_map = {logical: index for index, logical in enumerate(ordered)}
    circuit = QuantumCircuit(len(ordered), len(ordered), name=name)
    for gate in gates:
        _apply_spec_gate(circuit, gate, local_map)
    circuit.measure(range(len(ordered)), range(len(ordered)))
    return circuit, ordered, gates


def _mirror_probe(local_science: Any, repetitions: int, name: str) -> Any:
    _, _, QuantumCircuit = _imports()
    unitary = local_science.remove_final_measurements(inplace=False)
    circuit = QuantumCircuit(
        unitary.num_qubits,
        unitary.num_qubits,
        name=name,
    )
    for _ in range(repetitions):
        circuit.compose(unitary, inplace=True)
        circuit.barrier()
        circuit.compose(unitary.inverse(), inplace=True)
        circuit.barrier()
    circuit.measure(range(unitary.num_qubits), range(unitary.num_qubits))
    return circuit


def _active_local_features(
    candidate: dict,
    protocol_config: dict,
) -> tuple[dict, dict, list[dict]]:
    circuit_spec = candidate["exact_artifacts"]["circuit_spec"]
    selection = circuit_spec.get("verifier_selection") or {}
    witness = selection.get("witness") or {}
    weights = [float(value) for value in witness.get("weights", [])]
    masks = selection.get("selected_masks") or []
    stored_results = selection.get("selected_results") or []
    feature_names = witness.get("feature_names") or [
        f"local_parity_{index}" for index in range(len(weights))
    ]
    if not weights or len(masks) != len(weights):
        raise RuntimeError(
            "Local-witness probes require a stored verifier selection "
            "with witness weights and selected masks."
        )
    active_indices = [
        index for index, value in enumerate(weights) if abs(value) > 1e-8
    ]
    minimum_features = int(
        protocol_config.get("minimum_active_features", 2)
    )
    if len(active_indices) < minimum_features:
        raise RuntimeError(
            "The stored witness does not contain enough active features."
        )
    maximum_lightcone = int(
        protocol_config["probe_acceptance"][
            "maximum_active_lightcone_qubits"
        ]
    )
    features: list[dict] = []
    for feature_index in active_indices:
        mask = [int(value) for value in masks[feature_index]]
        support = tuple(index for index, value in enumerate(mask) if value)
        if not support:
            raise RuntimeError("An active witness feature has empty support.")
        local_science, ordered, gates = _local_science_circuit(
            circuit_spec,
            support,
            name=f"local_science_f{feature_index:02d}",
        )
        if len(ordered) > maximum_lightcone:
            raise RuntimeError(
                f"Active feature {feature_index} uses "
                f"{len(ordered)} light-cone qubits, exceeding "
                f"the preregistered maximum {maximum_lightcone}."
            )
        stored = (
            stored_results[feature_index]
            if feature_index < len(stored_results)
            else {}
        )
        stored_lightcone = tuple(
            int(value) for value in stored.get("lightcone", ordered)
        )
        if stored_lightcone and stored_lightcone != ordered:
            raise RuntimeError(
                "Recomputed exact backward light cone does not match "
                "the stored verifier result."
            )
        support_positions = [
            ordered.index(int(value)) for value in support
        ]
        features.append(
            {
                "feature_index": int(feature_index),
                "feature_name": str(feature_names[feature_index]),
                "weight": float(weights[feature_index]),
                "support": list(support),
                "lightcone": list(ordered),
                "support_positions": support_positions,
                "ideal_expectation": (
                    None
                    if stored.get("expectation") is None
                    else float(stored["expectation"])
                ),
                "selected_gates": len(gates),
                "selected_two_qubit_gates": sum(
                    len(gate["qubits"]) == 2 for gate in gates
                ),
                "_circuit": local_science,
            }
        )
    return circuit_spec, witness, features


def _build_local_witness_bundle(
    *,
    candidate: dict,
    backend: Any,
    layout: tuple[int, ...],
    output: Path,
    seed: int,
    protocol_config: dict,
) -> dict:
    _, witness, features = _active_local_features(
        candidate,
        protocol_config,
    )
    echo_repetitions = tuple(
        int(value) for value in protocol_config.get(
            "echo_repetitions", [1, 2, 3]
        )
    )
    if not echo_repetitions or any(value <= 0 for value in echo_repetitions):
        raise ValueError("echo_repetitions must contain positive integers.")
    if tuple(sorted(set(echo_repetitions))) != echo_repetitions:
        raise ValueError(
            "echo_repetitions must be strictly increasing and unique."
        )

    calibration_logical = tuple(
        sorted(
            {
                int(qubit)
                for feature in features
                for qubit in feature["lightcone"]
            }
        )
    )
    calibration_physical = tuple(
        int(layout[logical]) for logical in calibration_logical
    )
    calibration_position = {
        logical: index
        for index, logical in enumerate(calibration_logical)
    }
    patterns = {
        "local_basis_all0": tuple(0 for _ in calibration_logical),
        "local_basis_all1": tuple(1 for _ in calibration_logical),
        "local_basis_even": tuple(
            index % 2 for index in range(len(calibration_logical))
        ),
        "local_basis_odd": tuple(
            1 - index % 2 for index in range(len(calibration_logical))
        ),
    }
    circuits: list[dict] = []
    output.mkdir(parents=True, exist_ok=True)
    for index, (label, pattern) in enumerate(patterns.items()):
        path = output / f"{label}.qasm3"
        metadata = _compile_probe(
            _basis_probe(pattern, label),
            backend,
            calibration_physical,
            path,
            seed + index,
            optimization_level=0 if label == "local_basis_all0" else 3,
        )
        circuits.append(
            {
                "label": label,
                "type": "local_basis",
                "expected_pattern": list(pattern),
                "logical_qubits": list(calibration_logical),
                "physical_qubits": list(calibration_physical),
                "classical_bits": len(calibration_logical),
                "qasm_path": str(path),
                "compiled_depth": metadata["compiled_depth"],
            }
        )

    science_root = output / "local_science"
    science_root.mkdir(parents=True, exist_ok=True)
    science_circuits: list[dict] = []
    public_features: list[dict] = []
    for feature_offset, feature in enumerate(features):
        feature_index = int(feature["feature_index"])
        lightcone = tuple(int(value) for value in feature["lightcone"])
        physical_qubits = tuple(int(layout[value]) for value in lightcone)
        feature["physical_qubits"] = list(physical_qubits)
        feature["calibration_positions"] = [
            calibration_position[int(value)] for value in lightcone
        ]

        local_science = feature.pop("_circuit")
        science_label = f"local_science_f{feature_index:02d}"
        science_path = science_root / f"{science_label}.qasm3"
        science_metadata = _compile_probe(
            local_science,
            backend,
            physical_qubits,
            science_path,
            seed + 1000 + feature_offset,
            optimization_level=3,
            minimum_two_qubit_gates=1,
        )
        science_circuits.append(
            {
                **feature,
                "label": science_label,
                "classical_bits": len(lightcone),
                "qasm_path": str(science_path),
                "compiled_two_qubit_count": science_metadata[
                    "two_qubit_count"
                ],
                "compiled_depth": science_metadata["compiled_depth"],
            }
        )

        echo_counts: list[int] = []
        for echo_offset, repetitions in enumerate(echo_repetitions):
            label = (
                f"local_echo_f{feature_index:02d}_x{repetitions}"
            )
            path = output / f"{label}.qasm3"
            metadata = _compile_probe(
                _mirror_probe(local_science, repetitions, label),
                backend,
                physical_qubits,
                path,
                seed + 100 + feature_offset * 100 + echo_offset,
                optimization_level=0,
                minimum_two_qubit_gates=1,
                strip_barriers_from_qasm=True,
                decompose_controlled_phase=True,
            )
            echo_counts.append(metadata["two_qubit_count"])
            circuits.append(
                {
                    "label": label,
                    "type": "local_echo",
                    "feature_index": feature_index,
                    "repetitions": int(repetitions),
                    "expected_pattern": [0] * len(lightcone),
                    "logical_qubits": list(lightcone),
                    "physical_qubits": list(physical_qubits),
                    "classical_bits": len(lightcone),
                    "qasm_path": str(path),
                    "compiled_two_qubit_count": metadata[
                        "two_qubit_count"
                    ],
                    "compiled_depth": metadata["compiled_depth"],
                    "controlled_phase_qasm_rewrites": metadata[
                        "controlled_phase_qasm_rewrites"
                    ],
                }
            )
        if any(
            left >= right
            for left, right in zip(echo_counts, echo_counts[1:])
        ):
            raise RuntimeError(
                "Local mirror probes did not preserve increasing workload: "
                f"feature={feature_index}, counts={echo_counts}."
            )
        public_features.append(feature)

    claim_boundary = str(
        protocol_config.get(
            "claim_boundary",
            "The QPU result certifies only the preregistered active local "
            "witness observables through exact backward light cones. It does "
            "not certify total-variation closeness of the complete 48-qubit "
            "output distribution.",
        )
    )
    manifest = {
        "schema": "codegap.qpu-local-witness-probe-bundle.v1",
        "protocol": "local_witness_v1",
        "logical_qubits": int(candidate["code"]["n"]),
        "full_layout": list(layout),
        "calibration": {
            "logical_qubits": list(calibration_logical),
            "physical_qubits": list(calibration_physical),
        },
        "echo_repetitions": list(echo_repetitions),
        "active_features": public_features,
        "witness": witness,
        "claim_boundary": claim_boundary,
        "circuits": circuits,
    }
    (output / "probe_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    science_manifest = {
        "schema": "codegap.qpu-local-witness-science-bundle.v1",
        "protocol": "local_witness_v1",
        "candidate_id": str(candidate["candidate_id"]),
        "witness": witness,
        "active_features": science_circuits,
        "claim_boundary": claim_boundary,
    }
    (output / "local_science_manifest.json").write_text(
        json.dumps(science_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_probe_bundle(
    *,
    science_qasm: Path,
    backend: Any,
    layout: tuple[int, ...],
    output: Path,
    seed: int,
    candidate: dict | None = None,
    protocol_config: dict | None = None,
) -> dict:
    if (
        protocol_config is not None
        and protocol_config.get("mode") == "local_witness_v1"
    ):
        if candidate is None:
            raise RuntimeError(
                "local_witness_v1 requires the prepared candidate payload."
            )
        return _build_local_witness_bundle(
            candidate=candidate,
            backend=backend,
            layout=layout,
            output=output,
            seed=seed,
            protocol_config=protocol_config,
        )
    n = len(layout)
    patterns = {
        'basis_all0': tuple(0 for _ in range(n)),
        'basis_all1': tuple(1 for _ in range(n)),
        'basis_even': tuple(index % 2 for index in range(n)),
        'basis_odd': tuple(1 - index % 2 for index in range(n)),
    }
    circuits: list[dict] = []
    for index, (label, pattern) in enumerate(patterns.items()):
        path = output / f'{label}.qasm3'
        _compile_probe(
            _basis_probe(pattern, label),
            backend,
            layout,
            path,
            seed + index,
            optimization_level=0 if label == 'basis_all0' else 3,
        )
        circuits.append(
            {
                'label': label,
                'type': 'basis',
                'expected_pattern': list(pattern),
                'qasm_path': str(path),
            }
        )
    echo_two_qubit_counts: list[int] = []
    for index, repetitions in enumerate((1, 2, 3)):
        label = f'echo_x{repetitions}'
        path = output / f'{label}.qasm3'
        compile_metadata = _compile_probe(
            _echo_probe(science_qasm, repetitions),
            backend,
            layout,
            path,
            seed + 100 + index,
            optimization_level=0,
            minimum_two_qubit_gates=1,
            strip_barriers_from_qasm=True,
            decompose_controlled_phase=True,
        )
        echo_two_qubit_counts.append(compile_metadata['two_qubit_count'])
        circuits.append(
            {
                'label': label,
                'type': 'echo',
                'repetitions': repetitions,
                'expected_pattern': [0] * n,
                'qasm_path': str(path),
                'compiled_two_qubit_count': compile_metadata['two_qubit_count'],
                'compiled_depth': compile_metadata['compiled_depth'],
                'controlled_phase_qasm_rewrites': compile_metadata[
                    'controlled_phase_qasm_rewrites'
                ],
            }
        )
    if not (
        echo_two_qubit_counts[0]
        < echo_two_qubit_counts[1]
        < echo_two_qubit_counts[2]
    ):
        raise RuntimeError(
            'Echo probes did not preserve increasing scientific workload: '
            f'{echo_two_qubit_counts}.'
        )
    manifest = {
        'schema': 'codegap.qpu-probe-bundle.v1',
        'logical_qubits': n,
        'layout': list(layout),
        'circuits': circuits,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'probe_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
