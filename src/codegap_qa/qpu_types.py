from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TargetInstruction:
    name: str
    qargs: tuple[int, ...]
    error: float | None
    duration: float | None


@dataclass(frozen=True)
class TargetSnapshot:
    backend: str
    captured_at: str
    num_qubits: int
    operation_names: tuple[str, ...]
    coupling_edges: tuple[tuple[int, int], ...]
    instructions: tuple[TargetInstruction, ...]
    accepting_jobs: bool | None = None
    queue_depth: int | None = None
    source: str = 'qiskit.BackendV2.target'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompiledQPUCandidate:
    candidate_id: str
    logical_qubits: int
    layout: tuple[int, ...]
    qasm_path: str
    qasm_sha256: str
    swap_count: int
    two_qubit_count: int
    two_qubit_depth: int
    compiled_depth: int
    target_violations: tuple[dict[str, Any], ...]
    calibrated_log_error: float
    calibrated_fraction: float
    estimated_success: float | None
    measurement_map: dict[int, int]
    expected_measurement_map: dict[int, int]
    measurement_map_valid: bool
    nonlocal_edges: int
    target_embedding_valid: bool
    native_two_qubit_gates_valid: bool
    native_two_qubit_gate_names: tuple[str, ...]
    score: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeAnalysis:
    candidate_id: str
    convention: str
    convention_accuracy: float
    readout_p01_mean: float
    readout_p10_mean: float
    readout_worst: float
    checkerboard_error: float
    echo_survival: dict[str, float]
    science_survival_proxy: float
    tv_noise_proxy: float
    model_mismatch: float
    passed_integrity: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
