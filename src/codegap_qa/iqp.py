from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .progress import ProgressManager


@dataclass(frozen=True)
class IQPSpec:
    n: int
    edges: tuple[tuple[int, int], ...]
    theta_single: float
    theta_pair: float
    seed: int

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "edges": [list(edge) for edge in self.edges],
            "theta_single": self.theta_single,
            "theta_pair": self.theta_pair,
            "seed": self.seed,
        }


def fwht(
    vector: np.ndarray,
    progress: ProgressManager | None = None,
) -> np.ndarray:
    result = np.asarray(vector, dtype=np.complex128).copy()
    h = 1
    n = result.shape[0]
    layers = int(np.log2(n))
    layer_bar = (
        progress.bar(
            total=layers,
            desc="Exact IQP: Walsh-Hadamard layers",
            unit="layer",
            leave=progress.leave_nested,
        )
        if progress is not None
        else None
    )
    while h < n:
        for start in range(0, n, h * 2):
            left = result[start : start + h].copy()
            right = result[start + h : start + 2 * h].copy()
            result[start : start + h] = left + right
            result[start + h : start + 2 * h] = left - right
        h *= 2
        if layer_bar is not None:
            layer_bar.update(1)
    if layer_bar is not None:
        layer_bar.close()
    return result


def exact_probabilities(
    spec: IQPSpec,
    max_qubits: int = 24,
    progress: ProgressManager | None = None,
) -> np.ndarray:
    if spec.n > max_qubits:
        raise ValueError(
            f"Exact state simulation disabled for n={spec.n}; max={max_qubits}."
        )
    size = 1 << spec.n
    indices = np.arange(size, dtype=np.uint64)
    phase = np.zeros(size, dtype=np.float64)
    rng = np.random.default_rng(spec.seed)
    signs = rng.choice([-1.0, 1.0], size=spec.n)
    z_values: list[np.ndarray] = []
    qubit_steps = range(spec.n)
    if progress is not None:
        qubit_steps = progress.bar(
            qubit_steps,
            total=spec.n,
            desc="Exact IQP: single-qubit phases",
            unit="qubit",
            leave=progress.leave_nested,
        )
    for qubit in qubit_steps:
        bit = ((indices >> np.uint64(qubit)) & np.uint64(1)).astype(np.int8)
        z = 1.0 - 2.0 * bit
        z_values.append(z)
        phase += -0.5 * spec.theta_single * signs[qubit] * z
    edge_steps = spec.edges
    if progress is not None:
        edge_steps = progress.bar(
            edge_steps,
            total=len(spec.edges),
            desc="Exact IQP: pair phases",
            unit="edge",
            leave=progress.leave_nested,
        )
    for left, right in edge_steps:
        phase += -0.5 * spec.theta_pair * z_values[left] * z_values[right]
    diagonal_state = np.exp(1j * phase) / np.sqrt(size)
    amplitudes = fwht(diagonal_state, progress=progress) / np.sqrt(size)
    probabilities = np.abs(amplitudes) ** 2
    probabilities /= probabilities.sum()
    return probabilities


def sample_probabilities(
    probabilities: np.ndarray, shots: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(np.log2(probabilities.shape[0]))
    indices = rng.choice(probabilities.shape[0], size=shots, p=probabilities)
    shifts = np.arange(n, dtype=np.uint64)
    return ((indices[:, None].astype(np.uint64) >> shifts) & 1).astype(np.uint8)


def qasm3(spec: IQPSpec) -> str:
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{spec.n}] q;",
        f"bit[{spec.n}] c;",
    ]
    for qubit in range(spec.n):
        lines.append(f"h q[{qubit}];")
    rng = np.random.default_rng(spec.seed)
    signs = rng.choice([-1, 1], size=spec.n)
    for qubit, sign in enumerate(signs):
        angle = spec.theta_single * int(sign)
        lines.append(f"rz({angle:.17g}) q[{qubit}];")
    for left, right in spec.edges:
        lines.append(f"rzz({spec.theta_pair:.17g}) q[{left}], q[{right}];")
    for qubit in range(spec.n):
        lines.append(f"h q[{qubit}];")
    for qubit in range(spec.n):
        lines.append(f"c[{qubit}] = measure q[{qubit}];")
    return "\n".join(lines) + "\n"


def write_qasm3(spec: IQPSpec, path: Path) -> None:
    path.write_text(qasm3(spec), encoding="utf-8")
