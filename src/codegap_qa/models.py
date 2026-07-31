from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class ClaimLevel(str, Enum):
    EXACT_FINITE = "EXACT_FINITE"
    EMPIRICAL_ATTACK_SUITE = "EMPIRICAL_ATTACK_SUITE"
    CONDITIONAL_HARDNESS = "CONDITIONAL_HARDNESS"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


@dataclass(frozen=True)
class CodeMetrics:
    n: int
    k: int
    rank_x: int
    rank_z: int
    d_x: int | None
    d_z: int | None
    d_x_at_least: int
    d_z_at_least: int
    commutes: bool
    row_weight_x_max: int
    row_weight_z_max: int


@dataclass(frozen=True)
class HardwareMetrics:
    two_qubit_count: int
    two_qubit_depth: int
    swap_count: int
    routing_distance: int
    nonlocal_edges: int
    crossing_edges: int
    layout: tuple[int, ...]


@dataclass(frozen=True)
class HardnessMetrics:
    verify_operations: float
    statevector_operations: float
    tensor_proxy_operations: float
    mps_proxy_operations: float
    stabilizer_rank_proxy_operations: float
    schrodinger_feynman_proxy_operations: float
    best_attack_operations: float
    gamma_log10: float
    treewidth_upper_bound: int
    treewidth_lower_bound: int
    assumptions: tuple[str, ...]
    method: str = "legacy_static_interaction_graph"
    best_attack_name: str = "unknown"
    tensor_graph_treewidth_lower: int = 0
    tensor_graph_treewidth_upper: int = 0
    line_graph_treewidth_lower: int = 0
    line_graph_treewidth_upper: int = 0
    tensor_count: int = 0
    tensor_network_edges: int = 0
    cotengra: dict[str, Any] = field(default_factory=dict)
    cuts: dict[str, Any] = field(default_factory=dict)
    non_clifford: dict[str, Any] = field(default_factory=dict)
    attack_costs: dict[str, float] = field(default_factory=dict)
    claim_scope: str = ""


@dataclass(frozen=True)
class NoiseMetrics:
    ideal_margin: float
    tv_robust_radius: float
    predicted_pass_probability: float
    worst_noise_point: dict[str, float] = field(default_factory=dict)


@dataclass
class Candidate:
    candidate_id: str
    family: dict[str, Any]
    code: CodeMetrics
    hardware: HardwareMetrics
    hardness: HardnessMetrics
    noise: NoiseMetrics | None = None
    objective: float = float("-inf")
    exact_artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateDecision:
    gate: str
    passed: bool
    claim_level: ClaimLevel
    reasons: tuple[str, ...]
    evidence: dict[str, Any]
