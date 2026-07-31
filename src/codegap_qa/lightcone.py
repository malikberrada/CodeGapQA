from __future__ import annotations

import math
from typing import Any

import numpy as np


H = np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)


def _rz(angle: float) -> np.ndarray:
    return np.diag(
        [np.exp(-0.5j * angle), np.exp(0.5j * angle)]
    ).astype(np.complex128)


def _rzz(angle: float) -> np.ndarray:
    eigenvalues = np.asarray([1.0, -1.0, -1.0, 1.0])
    return np.diag(np.exp(-0.5j * angle * eigenvalues)).astype(np.complex128)


def _rxx(angle: float) -> np.ndarray:
    xx = np.kron(X, X)
    return (
        math.cos(angle / 2.0) * np.eye(4, dtype=np.complex128)
        - 1j * math.sin(angle / 2.0) * xx
    )


def _apply_gate(
    state: np.ndarray,
    operator: np.ndarray,
    targets: tuple[int, ...],
    qubits: int,
) -> np.ndarray:
    axes = list(targets) + [axis for axis in range(qubits) if axis not in targets]
    inverse = np.argsort(axes)
    tensor = state.reshape((2,) * qubits).transpose(axes)
    leading = 2 ** len(targets)
    tensor = operator @ tensor.reshape(leading, -1)
    return tensor.reshape((2,) * qubits).transpose(inverse).reshape(-1)


def backward_lightcone(
    gates: list[dict[str, Any]],
    observable_support: tuple[int, ...],
) -> tuple[set[int], list[dict[str, Any]]]:
    support = set(int(value) for value in observable_support)
    selected_reverse: list[dict[str, Any]] = []
    for gate in reversed(gates):
        if gate["name"] == "measure":
            continue
        qubits = set(int(value) for value in gate["qubits"])
        if support.intersection(qubits):
            support.update(qubits)
            selected_reverse.append(gate)
    return support, list(reversed(selected_reverse))


def local_z_expectation(
    circuit_spec: dict,
    support: tuple[int, ...],
    *,
    max_lightcone_qubits: int,
    backend: str = "auto",
) -> dict[str, Any]:
    cone, gates = backward_lightcone(circuit_spec["gates"], support)
    ordered = tuple(sorted(cone))
    if len(ordered) > max_lightcone_qubits:
        return {
            "status": "LIGHTCONE_TOO_LARGE",
            "support": list(support),
            "lightcone": list(ordered),
            "lightcone_qubits": len(ordered),
            "expectation": None,
            "absolute_error_bound": 1.0,
        }
    local = {qubit: index for index, qubit in enumerate(ordered)}
    dimension = 1 << len(ordered)
    requested = str(backend).lower()
    use_cuda = requested == "cuda"
    if requested == "auto":
        try:
            import cupy as cp

            use_cuda = cp.cuda.runtime.getDeviceCount() > 0 and len(ordered) >= 8
        except Exception:
            use_cuda = False
    if use_cuda:
        try:
            import cupy as cp

            def apply_gpu(state, operator, targets):
                axes = list(targets) + [axis for axis in range(len(ordered)) if axis not in targets]
                inverse = np.argsort(axes)
                tensor = state.reshape((2,) * len(ordered)).transpose(axes)
                leading = 2 ** len(targets)
                tensor = operator @ tensor.reshape(leading, -1)
                return tensor.reshape((2,) * len(ordered)).transpose(tuple(inverse)).reshape(-1)

            state = cp.zeros(dimension, dtype=cp.complex128)
            state[0] = 1.0
            h_gpu = cp.asarray(H)
            x_gpu = cp.asarray(X)
            for gate in gates:
                name = str(gate["name"])
                targets = tuple(local[int(value)] for value in gate["qubits"])
                angle = float(gate.get("angle") or 0.0)
                if name == "h":
                    operator = h_gpu
                elif name == "rz":
                    operator = cp.diag(cp.asarray([cp.exp(-0.5j * angle), cp.exp(0.5j * angle)]))
                elif name == "rzz":
                    eigen = cp.asarray([1.0, -1.0, -1.0, 1.0])
                    operator = cp.diag(cp.exp(-0.5j * angle * eigen))
                elif name == "rxx":
                    xx = cp.kron(x_gpu, x_gpu)
                    operator = (
                        math.cos(angle / 2.0) * cp.eye(4, dtype=cp.complex128)
                        - 1j * math.sin(angle / 2.0) * xx
                    )
                else:
                    raise ValueError(f"Unsupported light-cone gate: {name}")
                state = apply_gpu(state, operator, targets)
            probabilities = cp.abs(state) ** 2
            indices = cp.arange(dimension, dtype=cp.uint64)
            parity = cp.zeros(dimension, dtype=cp.uint8)
            # NumPy/CuPy C-order flattening makes tensor axis 0 the most-significant bit.
            # The previous code shifted by the axis number directly, mirroring each
            # observable inside its light cone.
            for qubit in support:
                shift = len(ordered) - 1 - local[int(qubit)]
                parity ^= ((indices >> np.uint64(shift)) & 1).astype(cp.uint8)
            signs = 1.0 - 2.0 * parity.astype(cp.float64)
            expectation = float(cp.dot(probabilities, signs).get())
            used_backend = "cupy_statevector"
        except Exception:
            if requested == "cuda":
                raise
            use_cuda = False
    if not use_cuda:
        state = np.zeros(dimension, dtype=np.complex128)
        state[0] = 1.0
        for gate in gates:
            name = str(gate["name"])
            targets = tuple(local[int(value)] for value in gate["qubits"])
            angle = float(gate.get("angle") or 0.0)
            if name == "h":
                state = _apply_gate(state, H, targets, len(ordered))
            elif name == "rz":
                state = _apply_gate(state, _rz(angle), targets, len(ordered))
            elif name == "rzz":
                state = _apply_gate(state, _rzz(angle), targets, len(ordered))
            elif name == "rxx":
                state = _apply_gate(state, _rxx(angle), targets, len(ordered))
            else:
                raise ValueError(f"Unsupported light-cone gate: {name}")
        probabilities = np.abs(state) ** 2
        indices = np.arange(dimension, dtype=np.uint64)
        parity = np.zeros(dimension, dtype=np.uint8)
        # Tensor axis 0 is the most-significant flattened bit.
        for qubit in support:
            shift = len(ordered) - 1 - local[int(qubit)]
            parity ^= ((indices >> np.uint64(shift)) & 1).astype(np.uint8)
        signs = 1.0 - 2.0 * parity.astype(np.float64)
        expectation = float(np.dot(probabilities, signs))
        used_backend = "numpy_statevector"
    return {
        "status": "EXACT_LIGHTCONE",
        "support": list(support),
        "lightcone": list(ordered),
        "lightcone_qubits": len(ordered),
        "selected_gates": len(gates),
        "selected_two_qubit_gates": sum(len(gate["qubits"]) == 2 for gate in gates),
        "selected_two_qubit_layers": sorted(
            {
                int(gate.get("layer", -1))
                for gate in gates
                if len(gate["qubits"]) == 2 and int(gate.get("layer", -1)) >= 0
            }
        ),
        "selected_two_qubit_axes": sorted(
            {str(gate["name"]) for gate in gates if len(gate["qubits"]) == 2}
        ),
        "expectation": expectation,
        "backend": used_backend,
        "absolute_error_bound": 32.0 * np.finfo(float).eps * max(1, len(gates)),
    }


def analytic_feature_means(
    circuit_spec: dict,
    parity_masks: np.ndarray,
    *,
    max_lightcone_qubits: int,
    bit_indices: tuple[int, ...] | None = None,
    include_centered_weight: bool = True,
    backend: str = "auto",
    require_all_bits: bool = True,
) -> dict[str, Any]:
    n = int(circuit_spec["n"])
    requested_bits = tuple(range(n)) if bit_indices is None else tuple(int(v) for v in bit_indices)
    bit_results = [
        local_z_expectation(
            circuit_spec,
            (qubit,),
            max_lightcone_qubits=max_lightcone_qubits,
            backend=backend,
        )
        for qubit in requested_bits
    ]
    parity_results = []
    retained_masks = []
    for mask in np.asarray(parity_masks, dtype=np.uint8):
        support = tuple(int(value) for value in np.flatnonzero(mask))
        if not support:
            continue
        result = local_z_expectation(
            circuit_spec,
            support,
            max_lightcone_qubits=max_lightcone_qubits,
            backend=backend,
        )
        if result["status"] == "EXACT_LIGHTCONE":
            retained_masks.append(mask.copy())
            parity_results.append(result)
    if require_all_bits and any(item["status"] != "EXACT_LIGHTCONE" for item in bit_results):
        raise RuntimeError(
            "At least one requested single-bit verifier light cone exceeds the configured limit."
        )
    retained_bit_results = [item for item in bit_results if item["status"] == "EXACT_LIGHTCONE"]
    retained_bit_indices = [
        requested_bits[index]
        for index, item in enumerate(bit_results)
        if item["status"] == "EXACT_LIGHTCONE"
    ]
    bit_means = np.asarray(
        [float(item["expectation"]) for item in retained_bit_results],
        dtype=np.float64,
    )
    parity_means = np.asarray(
        [float(item["expectation"]) for item in parity_results],
        dtype=np.float64,
    )
    parts = [bit_means, parity_means]
    if include_centered_weight:
        centered_weight_mean = -float(bit_means.mean()) if len(bit_means) else 0.0
        parts.append(np.asarray([centered_weight_mean]))
    feature_means = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
    return {
        "feature_means": feature_means,
        "parity_masks": (
            np.asarray(retained_masks, dtype=np.uint8).reshape(-1, n)
            if retained_masks
            else np.empty((0, n), dtype=np.uint8)
        ),
        "bit_results": retained_bit_results,
        "requested_bit_results": bit_results,
        "bit_indices": retained_bit_indices,
        "include_centered_weight": include_centered_weight,
        "parity_results": parity_results,
        "maximum_lightcone_qubits": max(
            [item["lightcone_qubits"] for item in retained_bit_results + parity_results],
            default=0,
        ),
        "method": "exact_backward_lightcone_statevector",
    }
