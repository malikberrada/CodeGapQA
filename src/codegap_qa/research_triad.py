from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations, product
from math import cos, log, log10, prod, sqrt
from pathlib import Path
from typing import Any, Iterable, Iterator
import copy
import json

import networkx as nx
import numpy as np

from .codesign import build_tracked_frame_mixing_circuit
from .freeze import freeze, verify_freeze
from .gf2 import (
    commute,
    css_distance_at_least_fast_small,
    css_k,
    exact_css_distance,
    rank,
)
from .hardware import (
    HardwareTopology,
    compile_metrics,
    interaction_edges_from_checks,
    zero_swap_hardware_metrics,
)
from .hardness import interaction_graph, simulator_costs
from .progress import ProgressManager, default_progress
from .schedule_search import search_adversarial_schedules
from .target_native import build_target_native_matching_pool
from .verifier_search import apply_verifier_selection, select_verifiable_observables
from .accel_bridge import diagnostics as accel_diagnostics, screen_abelian_specs


@dataclass(frozen=True)
class AbelianBicycleSpec:
    """Generalized bicycle code over a finite abelian translation group.

    ``dimensions=(l,m)`` recovers a bivariate bicycle code. Three or more
    dimensions produce multivariate bicycle candidates. The CSS construction is
    the same commuting group-algebra construction used by generalized bicycle
    codes; only the translation group changes.
    """

    dimensions: tuple[int, ...]
    support_a: tuple[tuple[int, ...], ...]
    support_b: tuple[tuple[int, ...], ...]
    family_type: str = "abelian_bicycle"

    def __post_init__(self) -> None:
        if len(self.dimensions) < 2:
            raise ValueError("At least two cyclic dimensions are required.")
        if any(int(value) < 2 for value in self.dimensions):
            raise ValueError("Every cyclic dimension must be at least two.")
        rank = len(self.dimensions)
        for support in (self.support_a, self.support_b):
            if not support:
                raise ValueError("Both polynomial supports must be non-empty.")
            if any(len(item) != rank for item in support):
                raise ValueError("Support vectors must match the group rank.")

    @property
    def block_size(self) -> int:
        return int(prod(self.dimensions))

    @property
    def n(self) -> int:
        return 2 * self.block_size

    @property
    def rank(self) -> int:
        return len(self.dimensions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.family_type,
            "dimensions": list(self.dimensions),
            "support_a": [list(item) for item in self.support_a],
            "support_b": [list(item) for item in self.support_b],
            "check_weight": len(self.support_a) + len(self.support_b),
            "n": self.n,
        }


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _digest_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


def _flatten_index(coordinates: tuple[int, ...], dimensions: tuple[int, ...]) -> int:
    value = 0
    stride = 1
    for coordinate, dimension in zip(reversed(coordinates), reversed(dimensions)):
        value += int(coordinate) * stride
        stride *= int(dimension)
    return value


def _group_elements(dimensions: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(product(*(range(int(value)) for value in dimensions)))


def _canonical_support(
    support: Iterable[tuple[int, ...]],
    dimensions: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    values = tuple(sorted(set(tuple(map(int, item)) for item in support)))
    if not values:
        return values
    anchor = values[0]
    translated = {
        tuple(
            (int(value) - int(offset)) % int(dimension)
            for value, offset, dimension in zip(item, anchor, dimensions)
        )
        for item in values
    }
    return tuple(sorted(translated))


def translation_matrix(
    dimensions: tuple[int, ...],
    shift: tuple[int, ...],
) -> np.ndarray:
    size = int(prod(dimensions))
    matrix = np.zeros((size, size), dtype=np.uint8)
    for source_coordinates in _group_elements(dimensions):
        target_coordinates = tuple(
            (coordinate + delta) % dimension
            for coordinate, delta, dimension in zip(
                source_coordinates,
                shift,
                dimensions,
            )
        )
        source = _flatten_index(source_coordinates, dimensions)
        target = _flatten_index(target_coordinates, dimensions)
        matrix[target, source] = 1
    return matrix


def polynomial_matrix(
    dimensions: tuple[int, ...],
    support: Iterable[tuple[int, ...]],
) -> np.ndarray:
    size = int(prod(dimensions))
    matrix = np.zeros((size, size), dtype=np.uint8)
    for shift in support:
        matrix ^= translation_matrix(dimensions, tuple(map(int, shift)))
    return matrix


def build_abelian_bicycle_css(
    spec: AbelianBicycleSpec,
) -> tuple[np.ndarray, np.ndarray]:
    a = polynomial_matrix(spec.dimensions, spec.support_a)
    b = polynomial_matrix(spec.dimensions, spec.support_b)
    h_x = np.hstack([a, b]).astype(np.uint8)
    h_z = np.hstack([b.T, a.T]).astype(np.uint8)
    if not commute(h_x, h_z):
        raise AssertionError("Abelian bicycle construction must commute.")
    return h_x, h_z


def _canonical_support_pool(
    *,
    dimensions: tuple[int, ...],
    weight: int,
    anchor_identity: bool,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Enumerate the exact translation-canonical support space.

    # CODEGAP_V099_FINITE_FAMILY_SPACE

    The previous rejection sampler treated ``max_candidates`` as an exact
    target. That is impossible when the quotient space contains fewer unique
    support pairs than requested. This helper enumerates the finite quotient
    exactly, so candidate generation can be sampled without replacement and
    can become exhaustive when the registered cap exceeds the true capacity.
    """

    dimensions = tuple(int(value) for value in dimensions)
    weight = int(weight)
    elements = _group_elements(dimensions)
    zero = tuple(0 for _ in dimensions)
    if weight < 1:
        raise ValueError("Support weight must be positive.")
    if weight > len(elements):
        raise ValueError("Support weight exceeds the group size.")

    if anchor_identity:
        if weight == 1:
            raw_supports = ((zero,),)
        else:
            nonzero = tuple(item for item in elements if item != zero)
            raw_supports = (
                (zero, *selection)
                for selection in combinations(nonzero, weight - 1)
            )
    else:
        raw_supports = combinations(elements, weight)

    return tuple(
        sorted(
            {
                _canonical_support(support, dimensions)
                for support in raw_supports
            }
        )
    )


def random_abelian_specs(
    *,
    dimensions: tuple[int, ...],
    weight_a: int,
    weight_b: int,
    count: int,
    seed: int,
    family_type: str,
    anchor_identity: bool = True,
    audit: dict[str, Any] | None = None,
) -> Iterator[AbelianBicycleSpec]:
    """Sample unique canonical code specifications without replacement.

    ``count`` is a maximum, matching the configuration field
    ``max_candidates``. When the finite quotient space is smaller, every
    unique candidate is returned exactly once instead of raising an error.
    """

    dimensions = tuple(int(value) for value in dimensions)
    requested = max(0, int(count))
    supports_a = _canonical_support_pool(
        dimensions=dimensions,
        weight=int(weight_a),
        anchor_identity=bool(anchor_identity),
    )
    supports_b = _canonical_support_pool(
        dimensions=dimensions,
        weight=int(weight_b),
        anchor_identity=False,
    )

    candidate_pairs = tuple(
        sorted(
            {
                min((support_a, support_b), (support_b, support_a))
                for support_a in supports_a
                for support_b in supports_b
            }
        )
    )
    capacity = len(candidate_pairs)
    generated = min(requested, capacity)

    if audit is not None:
        audit.clear()
        audit.update(
            {
                "schema": "codegap.family-candidate-space.v1",
                "dimensions": list(dimensions),
                "weight_a": int(weight_a),
                "weight_b": int(weight_b),
                "anchor_identity": bool(anchor_identity),
                "canonical_supports_a": len(supports_a),
                "canonical_supports_b": len(supports_b),
                "unique_pair_capacity": capacity,
                "requested_max_candidates": requested,
                "generated_candidates": generated,
                "sampling_without_replacement": True,
                "exhaustive": generated == capacity,
                "capacity_limited": requested > capacity,
                "seed": int(seed),
            }
        )

    if generated == 0:
        return

    if generated == capacity:
        selected_pairs = candidate_pairs
    else:
        rng = np.random.default_rng(int(seed))
        selected_indices = np.sort(
            rng.choice(capacity, size=generated, replace=False)
        )
        selected_pairs = tuple(
            candidate_pairs[int(index)]
            for index in selected_indices
        )

    for support_a, support_b in selected_pairs:
        yield AbelianBicycleSpec(
            dimensions=dimensions,
            support_a=support_a,
            support_b=support_b,
            family_type=family_type,
        )

def _css_metrics_fast(
    h_x: np.ndarray,
    h_z: np.ndarray,
    *,
    min_d_x: int,
    min_d_z: int,
    max_exact_kernel_dimension: int,
) -> dict[str, Any]:
    h_x = np.asarray(h_x, dtype=np.uint8) & 1
    h_z = np.asarray(h_z, dtype=np.uint8) & 1
    if h_x.shape[1] != h_z.shape[1]:
        raise ValueError("H_X and H_Z must have the same number of columns.")
    commutes = commute(h_x, h_z)
    dx_ok, dx_cert = css_distance_at_least_fast_small(
        h_z, h_x, int(min_d_x)
    )
    dz_ok, dz_cert = css_distance_at_least_fast_small(
        h_x, h_z, int(min_d_z)
    )
    d_x = (
        exact_css_distance(
            h_z, h_x, int(max_exact_kernel_dimension)
        )
        if dx_ok
        else None
    )
    d_z = (
        exact_css_distance(
            h_x, h_z, int(max_exact_kernel_dimension)
        )
        if dz_ok
        else None
    )
    return {
        "n": int(h_x.shape[1]),
        "k": css_k(h_x, h_z),
        "rank_x": rank(h_x),
        "rank_z": rank(h_z),
        "commutes": bool(commutes),
        "d_x": d_x,
        "d_z": d_z,
        "d_x_at_least": int(min_d_x) if dx_ok else 0,
        "d_z_at_least": int(min_d_z) if dz_ok else 0,
        "d_x_certificate": dx_cert,
        "d_z_certificate": dz_cert,
        "row_weight_x_max": int(h_x.sum(axis=1).max(initial=0)),
        "row_weight_z_max": int(h_z.sum(axis=1).max(initial=0)),
    }


def _load_topology(config: dict[str, Any]) -> HardwareTopology:
    """Load a connected proxy view of the registered hardware target.

    # CODEGAP_V091_ACTIVE_COMPONENT_TOPOLOGY

    Live target snapshots may include disabled or isolated physical qubits. The
    classical layout/schedule search requires finite pairwise distances, so it
    runs on the largest connected component. The component graph is relabelled
    to contiguous proxy indices for the optimizers, while ``physical_qubits``
    records the exact proxy-to-device map. Export helpers below restore the
    original physical labels before artifacts are consumed by QPU compilation.
    """

    hardware = config["hardware"]
    snapshot_value = hardware.get("target_snapshot")
    if snapshot_value:
        snapshot_path = Path(snapshot_value)
        if not snapshot_path.is_absolute():
            snapshot_path = Path(config.get("_config_dir", ".")) / snapshot_path
        snapshot_path = snapshot_path.resolve()
        topology = HardwareTopology.from_target_snapshot(snapshot_path)
        components = sorted(
            (
                set(map(int, component))
                for component in nx.connected_components(topology.graph)
            ),
            key=lambda component: (-len(component), min(component)),
        )
        if not components:
            raise ValueError("Target snapshot contains no physical qubits.")
        component_sizes = [len(component) for component in components]
        selected = components[0]
        excluded = sorted(set(map(int, topology.graph.nodes)) - selected)
        policy = str(
            hardware.get(
                "disconnected_topology_policy",
                "largest_connected_component",
            )
        ).lower()
        if len(components) > 1 and policy == "error":
            raise ValueError(
                "Hardware target snapshot is disconnected: "
                f"component sizes={component_sizes}."
            )
        if len(components) > 1 and policy != "largest_connected_component":
            raise ValueError(
                "hardware.disconnected_topology_policy must be either "
                "'largest_connected_component' or 'error'."
            )
        if len(components) == 1:
            topology.graph.graph.update(
                {
                    "component_policy": "already_connected",
                    "original_num_qubits": topology.graph.number_of_nodes(),
                    "component_sizes": component_sizes,
                    "excluded_physical_qubits": [],
                    "physical_qubits": list(
                        range(topology.graph.number_of_nodes())
                    ),
                    "target_snapshot_path": str(snapshot_path),
                }
            )
            return topology

        physical_qubits = tuple(sorted(selected))
        physical_to_proxy = {
            physical: proxy
            for proxy, physical in enumerate(physical_qubits)
        }
        active_graph = nx.relabel_nodes(
            topology.graph.subgraph(physical_qubits).copy(),
            physical_to_proxy,
            copy=True,
        )
        if not nx.is_connected(active_graph):
            raise RuntimeError(
                "Largest target component relabelling is not connected."
            )
        active_graph.graph.update(
            {
                "component_policy": "largest_connected_component",
                "original_num_qubits": topology.graph.number_of_nodes(),
                "component_sizes": component_sizes,
                "excluded_physical_qubits": excluded,
                "physical_qubits": list(physical_qubits),
                "target_snapshot_path": str(snapshot_path),
            }
        )
        return HardwareTopology(
            graph=active_graph,
            module_of=tuple(0 for _ in physical_qubits),
            name=(
                f"{topology.name}:largest-component-"
                f"{len(physical_qubits)}"
            ),
            source=topology.source,
            structural_fingerprint=topology.structural_fingerprint,
        )

    topology = HardwareTopology.rectangular_grid(
        int(hardware["rows"]),
        int(hardware["cols"]),
        int(hardware.get("module_rows", 0)),
        int(hardware.get("module_cols", 0)),
        hardware.get("name"),
    )
    topology.graph.graph.update(
        {
            "component_policy": "synthetic_connected_proxy",
            "original_num_qubits": topology.graph.number_of_nodes(),
            "component_sizes": [topology.graph.number_of_nodes()],
            "excluded_physical_qubits": [],
            "physical_qubits": list(
                range(topology.graph.number_of_nodes())
            ),
        }
    )
    return topology


def _physical_qubit_map(
    topology: HardwareTopology,
) -> tuple[int, ...]:
    values = topology.graph.graph.get("physical_qubits")
    if values is None:
        return tuple(range(topology.graph.number_of_nodes()))
    mapping = tuple(int(value) for value in values)
    if len(mapping) != topology.graph.number_of_nodes():
        raise RuntimeError("Invalid proxy-to-physical qubit mapping.")
    return mapping


def _map_layout_to_physical(
    layout: Iterable[int],
    topology: HardwareTopology,
) -> list[int]:
    mapping = _physical_qubit_map(topology)
    values = [int(value) for value in layout]
    if any(
        value < 0 or value >= len(mapping)
        for value in values
    ):
        raise RuntimeError(
            "A proxy layout contains an out-of-range qubit index."
        )
    return [mapping[value] for value in values]


def _export_hardware_layout(
    hardware: dict[str, Any],
    topology: HardwareTopology,
) -> None:
    mapping = _physical_qubit_map(topology)
    if tuple(mapping) == tuple(range(len(mapping))):
        return
    layout = hardware.get("layout")
    if layout is not None:
        hardware["component_layout"] = [
            int(value)
            for value in layout
        ]
        hardware["layout"] = _map_layout_to_physical(
            layout,
            topology,
        )
    embedding = hardware.get("embedding_search")
    if (
        isinstance(embedding, dict)
        and embedding.get("layout") is not None
    ):
        embedding["component_layout"] = [
            int(value)
            for value in embedding["layout"]
        ]
        embedding["layout"] = _map_layout_to_physical(
            embedding["layout"],
            topology,
        )
    hardware["active_component_physical_qubits"] = list(mapping)
    hardware["excluded_physical_qubits"] = list(
        topology.graph.graph.get(
            "excluded_physical_qubits",
            [],
        )
    )


def _export_finalist_layouts(
    finalist: dict[str, Any],
    topology: HardwareTopology,
) -> None:
    mapping = _physical_qubit_map(topology)
    if tuple(mapping) == tuple(range(len(mapping))):
        return
    circuit_spec = finalist.get("circuit_spec", {})
    metadata = circuit_spec.get("schedule_metadata", {})
    pinned = metadata.get("pinned_layout")
    if pinned is not None:
        metadata["component_pinned_layout"] = [
            int(value)
            for value in pinned
        ]
        metadata["pinned_layout"] = _map_layout_to_physical(
            pinned,
            topology,
        )
        metadata["active_component_physical_qubits"] = list(mapping)
    hardware = finalist.get("hardware")
    if isinstance(hardware, dict):
        _export_hardware_layout(hardware, topology)


def _export_schedule_report_layouts(
    report: dict[str, Any],
    topology: HardwareTopology,
) -> None:
    mapping = _physical_qubit_map(topology)
    if tuple(mapping) == tuple(range(len(mapping))):
        return
    pool = report.get("target_native_pool")
    if (
        isinstance(pool, dict)
        and pool.get("layout") is not None
    ):
        pool["component_layout"] = [
            int(value)
            for value in pool["layout"]
        ]
        pool["layout"] = _map_layout_to_physical(
            pool["layout"],
            topology,
        )
        pool["active_component_physical_qubits"] = list(mapping)

def _family_configurations(config: dict[str, Any]) -> list[dict[str, Any]]:
    settings = config.get("research_triad", {}).get("family_search", {})
    families = list(settings.get("families", []))
    if not families:
        raise ValueError("research_triad.family_search.families must not be empty.")
    normalized = []
    for index, item in enumerate(families):
        dimensions = tuple(int(value) for value in item["dimensions"])
        family_type = str(
            item.get(
                "type",
                "bivariate_bicycle" if len(dimensions) == 2 else "multivariate_bicycle",
            )
        )
        normalized.append(
            {
                "index": index,
                "type": family_type,
                "dimensions": dimensions,
                "weight_a": int(item["weight_a"]),
                "weight_b": int(item["weight_b"]),
                "max_candidates": int(
                    item.get(
                        "max_candidates",
                        settings.get("samples_per_family", 128),
                    )
                ),
                "anchor_identity": bool(item.get("anchor_identity", True)),
            }
        )
    return normalized


def _code_score(
    metrics: dict[str, Any],
    hardware: dict[str, Any],
    settings: dict[str, Any],
) -> float:
    weights = settings.get("code_score", {})
    n = max(1, int(metrics["n"]))
    rate = float(metrics["k"]) / n
    distance = min(int(metrics["d_x_at_least"]), int(metrics["d_z_at_least"]))
    check_weight = max(
        int(metrics["row_weight_x_max"]),
        int(metrics["row_weight_z_max"]),
    )
    return float(
        float(weights.get("distance", 4.0)) * distance
        + float(weights.get("logical_rate", 12.0)) * rate
        + float(weights.get("logical_qubits", 0.2)) * int(metrics["k"])
        - float(weights.get("check_weight", 0.15)) * check_weight
        - float(weights.get("nonlocal_edges", 0.05))
        * int(hardware["nonlocal_edges"])
        - float(weights.get("swaps", 0.01)) * int(hardware["swap_count"])
        - float(weights.get("qubits", 0.002)) * n
    )


def _family_screen(
    *,
    config: dict[str, Any],
    topology: HardwareTopology,
    progress: ProgressManager,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Screen code families with a batched C++/CUDA prefilter.

    # CODEGAP_V098_DUAL_CUDA_ACCELERATION

    The native batch computes GF(2) ranks and exact low-weight logical tests for
    every family candidate. Detailed Python certificates and exact distances are
    rebuilt only for candidates that survive all registered gates. If the
    optional package is unavailable, the original Python path remains active.
    """

    triad = config.get("research_triad", {})
    settings = triad.get("family_search", {})
    constraints = config.get("constraints", {})
    acceleration = config.get("acceleration", {})
    min_dx = int(constraints.get("min_d_x", settings.get("min_d_x", 3)))
    min_dz = int(constraints.get("min_d_z", settings.get("min_d_z", 3)))
    min_k = int(constraints.get("min_k", settings.get("min_k", 2)))
    max_n = int(settings.get("max_qubits", topology.graph.number_of_nodes()))
    exact_kernel = int(constraints.get("max_exact_kernel_dimension", 20))
    layout_iterations = int(
        settings.get(
            "layout_iterations",
            config.get("hardware", {}).get("layout_iterations", 1000),
        )
    )
    records: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    family_configs = _family_configurations(config)
    family_bar = progress.bar(
        family_configs,
        total=len(family_configs),
        desc="TriadSearch: code families",
        unit="family",
        leave=True,
    )
    for family in family_bar:
        dimensions = tuple(family["dimensions"])
        n = 2 * int(prod(dimensions))
        if n > max_n or n > topology.graph.number_of_nodes():
            records.append(
                {
                    "family": family,
                    "accepted": False,
                    "reason": "qubit_budget_or_topology_too_small",
                    "n": n,
                }
            )
            continue
        generation_audit: dict[str, Any] = {}
        specs = list(
            random_abelian_specs(
                dimensions=dimensions,
                weight_a=int(family["weight_a"]),
                weight_b=int(family["weight_b"]),
                count=int(family["max_candidates"]),
                seed=int(config.get("seed", 0)) + 1009 * int(family["index"]),
                family_type=str(family["type"]),
                anchor_identity=bool(family["anchor_identity"]),
                audit=generation_audit,
            )
        )
        batch = screen_abelian_specs(
            specs,
            dimensions=dimensions,
            min_dx=min_dx,
            min_dz=min_dz,
            min_k=min_k,
            acceleration=acceleration,
        )
        native_batch = bool(batch and not batch.get("fallback", False))
        batch_backend = str(batch.get("backend", "python")) if batch else "python"
        candidate_bar = progress.bar(
            specs,
            total=len(specs),
            desc=f"{family['type']} {list(dimensions)} n={n}",
            unit="code",
            leave=progress.leave_nested,
        )
        for candidate_index, spec in enumerate(candidate_bar):
            candidate_payload = {
                "family": spec.to_dict(),
                "candidate_index": candidate_index,
            }
            code_id = _digest_payload(candidate_payload)[:16]
            h_x = None
            h_z = None
            if native_batch:
                flags = int(batch["flags"][candidate_index])
                witness_x_weight = int(batch["witness_x_weight"][candidate_index])
                witness_z_weight = int(batch["witness_z_weight"][candidate_index])
                witness_x = [
                    int(value)
                    for value in batch["witness_x_indices"][candidate_index][
                        :witness_x_weight
                    ]
                ]
                witness_z = [
                    int(value)
                    for value in batch["witness_z_indices"][candidate_index][
                        :witness_z_weight
                    ]
                ]
                metrics = {
                    "n": n,
                    "k": int(batch["k"][candidate_index]),
                    "rank_x": int(batch["rank_x"][candidate_index]),
                    "rank_z": int(batch["rank_z"][candidate_index]),
                    "commutes": True,
                    "d_x": None,
                    "d_z": None,
                    "d_x_at_least": min_dx if flags & 2 else 0,
                    "d_z_at_least": min_dz if flags & 4 else 0,
                    "d_x_certificate": {
                        "backend": batch_backend,
                        "witness_weight": witness_x_weight or None,
                        "witness_support": witness_x or None,
                    },
                    "d_z_certificate": {
                        "backend": batch_backend,
                        "witness_weight": witness_z_weight or None,
                        "witness_support": witness_z or None,
                    },
                    "row_weight_x_max": int(family["weight_a"]) + int(family["weight_b"]),
                    "row_weight_z_max": int(family["weight_a"]) + int(family["weight_b"]),
                }
            else:
                h_x, h_z = build_abelian_bicycle_css(spec)
                metrics = _css_metrics_fast(
                    h_x,
                    h_z,
                    min_d_x=min_dx,
                    min_d_z=min_dz,
                    max_exact_kernel_dimension=exact_kernel,
                )
                flags = (
                    (1 if int(metrics["k"]) >= min_k else 0)
                    | (2 if int(metrics["d_x_at_least"]) >= min_dx else 0)
                    | (4 if int(metrics["d_z_at_least"]) >= min_dz else 0)
                )
            record: dict[str, Any] = {
                "code_id": code_id,
                "family": spec.to_dict(),
                "code": metrics,
                "accepted": False,
                "screen_backend": batch_backend if native_batch else "python",
                "family_generation": generation_audit,
            }
            if not metrics["commutes"]:
                record["reason"] = "css_commutation_failed"
                records.append(record)
                continue
            if not (flags & 1):
                record["reason"] = "logical_dimension_below_threshold"
                records.append(record)
                continue
            if not (flags & 2) or not (flags & 4):
                record["reason"] = "distance_lower_bound_failed"
                records.append(record)
                continue
            if h_x is None or h_z is None:
                h_x, h_z = build_abelian_bicycle_css(spec)
                metrics = _css_metrics_fast(
                    h_x,
                    h_z,
                    min_d_x=min_dx,
                    min_d_z=min_dz,
                    max_exact_kernel_dimension=exact_kernel,
                )
                record["code"] = metrics
                if (
                    int(metrics["k"]) < min_k
                    or int(metrics["d_x_at_least"]) < min_dx
                    or int(metrics["d_z_at_least"]) < min_dz
                ):
                    raise RuntimeError(
                        "Native family screening disagreed with the exact Python audit."
                    )
            code_hardware = compile_metrics(
                h_x,
                h_z,
                topology,
                seed=int(config.get("seed", 0)) + candidate_index * 65537,
                layout_iterations=layout_iterations,
                progress=None,
                prefer_native=bool(acceleration.get("native_layout", True)),
            )
            if "_export_hardware_layout" in globals():
                _export_hardware_layout(code_hardware, topology)
            score = _code_score(metrics, code_hardware, settings)
            record.update(
                {
                    "accepted": True,
                    "hardware_code_proxy": code_hardware,
                    "code_score": score,
                    "h_x": h_x.tolist(),
                    "h_z": h_z.tolist(),
                }
            )
            records.append(record)
            accepted.append(
                {
                    "code_id": code_id,
                    "spec": spec,
                    "h_x": h_x,
                    "h_z": h_z,
                    "code": metrics,
                    "hardware_code_proxy": code_hardware,
                    "code_score": score,
                    "screen_backend": record["screen_backend"],
                    "family_generation": generation_audit,
                }
            )
            candidate_bar.set_postfix(
                accepted=len(accepted),
                backend=record["screen_backend"],
                k=int(metrics["k"]),
                d=min(int(metrics["d_x_at_least"]), int(metrics["d_z_at_least"])),
                refresh=False,
            )
        candidate_bar.close()
        family_bar.set_postfix(
            accepted=len(accepted),
            backend=batch_backend if native_batch else "python",
            generated=len(specs),
            capacity=generation_audit["unique_pair_capacity"],
            refresh=False,
        )
    family_bar.close()
    return records, accepted

def _select_code_representatives(
    accepted: list[dict[str, Any]],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    per_family_n = int(settings.get("representatives_per_family_size", 2))
    maximum = int(settings.get("max_code_representatives", 12))
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in accepted:
        key = (str(item["spec"].family_type), int(item["spec"].n))
        grouped.setdefault(key, []).append(item)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        selected.extend(
            sorted(
                grouped[key],
                key=lambda item: float(item["code_score"]),
                reverse=True,
            )[:per_family_n]
        )
    selected.sort(
        key=lambda item: (
            int(item["spec"].n),
            -float(item["code_score"]),
            str(item["spec"].family_type),
        )
    )
    return selected[:maximum]


def _random_schedule_indices(
    rng: np.random.Generator,
    pool_size: int,
    depth: int,
) -> tuple[int, ...]:
    if pool_size <= 0:
        raise ValueError("A non-empty matching pool is required.")
    values: list[int] = []
    previous: int | None = None
    for _ in range(depth):
        options = [index for index in range(pool_size) if index != previous]
        if not options:
            options = list(range(pool_size))
        value = int(rng.choice(options))
        values.append(value)
        previous = value
    return tuple(values)


def _balanced_cut_size(graph: nx.Graph, seed: int) -> int:
    if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
        return 0
    try:
        left, right = nx.algorithms.community.kernighan_lin_bisection(
            graph,
            seed=seed,
        )
        return int(nx.cut_size(graph, left, right))
    except Exception:
        return max(1, graph.number_of_nodes() // 2)


def _non_clifford_count(circuit_spec: dict[str, Any]) -> int:
    count = 0
    for gate in circuit_spec.get("gates", []):
        if str(gate.get("name")) not in {"rz", "rzz", "rxx"}:
            continue
        angle = float(gate.get("angle") or 0.0)
        quotient = angle / (np.pi / 2.0)
        if abs(quotient - round(quotient)) > 1e-9:
            count += 1
    return count


def _proxy_schedule_search(
    *,
    item: dict[str, Any],
    config: dict[str, Any],
    topology: HardwareTopology,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = config.get("schedule_search", {})
    spec: AbelianBicycleSpec = item["spec"]
    pool_result = build_target_native_matching_pool(
        topology=topology,
        n=spec.n,
        seed=seed,
        layout_trials=int(settings.get("target_layout_trials", 16)),
        matching_trials=int(settings.get("target_matching_trials", 128)),
        minimum_matchings=int(settings.get("target_minimum_matchings", 2)),
    )
    if pool_result.get("status") not in {"FOUND", "BEST_PARTIAL_POOL"}:
        return [], {
            "status": "STOP_NO_TARGET_NATIVE_MATCHING_POOL",
            "target_native_pool": pool_result,
        }
    matching_pool = list(pool_result["matching_pool"])
    pinned_layout = tuple(int(value) for value in pool_result["layout"])
    depths = [int(value) for value in settings.get("depths", [6, 8, 10])]
    scales = [float(value) for value in settings.get("angle_scales", [1.0])]
    axes_pool = [str(value).lower() for value in settings.get("axes", ["zz", "xx"])]
    schedule_count = int(settings.get("schedules_per_code", 32))
    rng = np.random.default_rng(seed)
    finalists: list[dict[str, Any]] = []
    generated = 0
    for schedule_index in range(schedule_count):
        depth = int(depths[schedule_index % len(depths)])
        matching_indices = _random_schedule_indices(
            rng,
            len(matching_pool),
            depth,
        )
        axes = tuple(
            axes_pool[(layer + schedule_index) % len(axes_pool)]
            for layer in range(depth)
        )
        angle_scales = tuple(float(rng.choice(scales)) for _ in range(depth))
        circuit_spec = build_tracked_frame_mixing_circuit(
            spec=spec,
            h_x=item["h_x"],
            h_z=item["h_z"],
            matching_pool=matching_pool,
            matching_indices=matching_indices,
            axes=axes,
            config=config,
            seed=seed + schedule_index * 1009,
            angle_scales=angle_scales,
            schedule_metadata={
                "generation_mode": "research_triad_proxy",
                "matching_source": "target_native_tracked_frame",
                "pinned_layout": list(pinned_layout),
            },
        )
        generated += 1
        preflight = select_verifiable_observables(circuit_spec, config)
        if not preflight.get("passed", False):
            continue
        apply_verifier_selection(circuit_spec, preflight)
        union_edges = tuple(
            tuple(sorted(map(int, edge)))
            for edge in circuit_spec["union_two_qubit_edges"]
        )
        embedding = {
            "status": "FOUND",
            "reason": "target_native_construction_witness",
            "layout": list(pinned_layout),
            "states": 0,
            "elapsed_seconds": 0.0,
            "exact": True,
            "target_structural_fingerprint": topology.structural_fingerprint,
        }
        hardware = zero_swap_hardware_metrics(
            circuit_spec=circuit_spec,
            logical_edges=union_edges,
            topology=topology,
            embedding=embedding,
        )
        graph = interaction_graph(spec.n, union_edges)
        cut_size = _balanced_cut_size(graph, seed + schedule_index)
        hardness = simulator_costs(
            spec.n,
            graph,
            _non_clifford_count(circuit_spec),
            cut_size,
            float(preflight["verify_operations"]),
            tuple(config.get("hardness", {}).get("assumptions", [])),
        )
        hardness.update(
            {
                "method": "research_triad_fast_attack_proxy",
                "best_attack_name": "minimum_registered_proxy",
                "claim_scope": (
                    "Fast ranking proxy only. Use research_triad.mode=deep for "
                    "the registered space-time/Cotengra attack suite."
                ),
            }
        )
        objective = (
            float(hardness["gamma_log10"])
            - float(settings.get("objective", {}).get("lambda_depth", 0.02))
            * int(hardware["two_qubit_depth"])
            - float(settings.get("objective", {}).get("mu_twoq", 0.0005))
            * int(hardware["two_qubit_count"])
        )
        finalists.append(
            {
                "circuit_spec": circuit_spec,
                "hardware": hardware,
                "hardness": hardness,
                "verifier_preflight": preflight,
                "objective": float(objective),
                "schedule_index": schedule_index,
                "matching_source": "target_native_tracked_frame",
            }
        )
    finalists.sort(key=lambda value: float(value["objective"]), reverse=True)
    keep = int(settings.get("return_finalists_per_code", 1))
    return finalists[:keep], {
        "status": "FOUND" if finalists else "STOP_NO_PROXY_SCHEDULE",
        "mode": "proxy",
        "generated_schedules": generated,
        "passing_schedules": len(finalists),
        "target_native_pool": pool_result,
    }


def _gate_counts_for_feature(
    result: dict[str, Any],
    circuit_spec: dict[str, Any],
) -> dict[str, int]:
    lightcone = set(int(value) for value in result.get("lightcone", []))
    selected = []
    for gate in circuit_spec.get("gates", []):
        if str(gate.get("name")) == "measure":
            continue
        qubits = set(int(value) for value in gate.get("qubits", []))
        if lightcone.intersection(qubits):
            selected.append(gate)
    return {
        "one_qubit": sum(len(gate.get("qubits", [])) == 1 for gate in selected),
        "two_qubit": sum(len(gate.get("qubits", [])) == 2 for gate in selected),
        "phase": sum(str(gate.get("name")) in {"rz", "rzz", "rxx"} for gate in selected),
    }


def _read_calibration_context(
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = config.get("research_triad", {}).get("robustness", {})
    value = settings.get("calibration_certificate")
    if not value:
        return {
            "source": None,
            "minimum_assignment_determinant": float(
                settings.get("minimum_assignment_determinant", 0.75)
            ),
            "maximum_assignment_error": float(
                settings.get("maximum_assignment_error", 0.08)
            ),
            "interval_half_width": float(
                settings.get("calibration_interval_half_width", 0.02)
            ),
        }
    path = Path(value)
    if not path.is_absolute():
        path = Path(config.get("_config_dir", ".")) / path
    if path.is_dir():
        candidates = [
            path / "final_certificate.json",
            path / "diagnostic_certificate.json",
        ]
        path = next((item for item in candidates if item.is_file()), path)
    if not path.is_file():
        return {
            "source": str(path.resolve()),
            "source_status": "MISSING_FALLBACK_TO_REGISTERED_BOUNDS",
            "minimum_assignment_determinant": float(
                settings.get("minimum_assignment_determinant", 0.75)
            ),
            "maximum_assignment_error": float(
                settings.get("maximum_assignment_error", 0.08)
            ),
            "interval_half_width": float(
                settings.get("calibration_interval_half_width", 0.02)
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    parameters = payload.get("calibration_confidence", {}).get("parameters", [])
    determinants = []
    maximum_error = 0.0
    half_widths = []
    point_parameters = payload.get("point_estimate", {}).get("parameters", [])
    for index in range(0, len(point_parameters), 2):
        p01 = float(point_parameters[index])
        p10 = float(point_parameters[index + 1])
        determinants.append(1.0 - p01 - p10)
        maximum_error = max(maximum_error, p01, p10)
    for item in parameters:
        for name in ("p01", "p10"):
            interval = item.get(f"{name}_interval")
            estimate = item.get(f"{name}_estimate")
            if interval is None or estimate is None:
                continue
            half_widths.append(
                max(
                    abs(float(estimate) - float(interval[0])),
                    abs(float(interval[1]) - float(estimate)),
                )
            )
    return {
        "source": str(path.resolve()),
        "source_status": "LOADED",
        "minimum_assignment_determinant": min(determinants, default=0.75),
        "maximum_assignment_error": maximum_error,
        "interval_half_width": max(half_widths, default=0.02),
    }


def _scaled_noise_point(envelope: dict[str, float], scale: float) -> dict[str, float]:
    return {
        key: float(value) * float(scale)
        for key, value in envelope.items()
    }


def _robust_margin_at_noise(
    *,
    preflight: dict[str, Any],
    circuit_spec: dict[str, Any],
    point: dict[str, float],
    settings: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    witness = preflight["witness"]
    weights = np.asarray(witness["weights"], dtype=np.float64)
    ideal = np.asarray(preflight["selected_expectations"], dtype=np.float64)
    results = list(preflight["selected_results"])
    active = np.flatnonzero(np.abs(weights) > 1e-10)
    c1 = float(settings.get("single_qubit_depolarizing_factor", 4.0 / 3.0))
    c2 = float(settings.get("two_qubit_depolarizing_factor", 16.0 / 15.0))
    p1 = max(0.0, float(point.get("p_1q", 0.0)))
    p2 = max(0.0, float(point.get("p_2q", 0.0)))
    pm = max(0.0, float(point.get("p_measure", 0.0)))
    coherent = abs(float(point.get("coherent_angle", 0.0)))
    readout_mitigation = bool(settings.get("readout_mitigation", True))
    residual_fraction = float(settings.get("residual_measurement_fraction", 0.25))
    effective_pm = pm * residual_fraction if readout_mitigation else pm
    minimum_det = max(
        1e-6,
        float(calibration["minimum_assignment_determinant"]),
    )
    interval_width = max(0.0, float(calibration["interval_half_width"]))
    lower_mean = 0.0
    feature_details = []
    estimator_bound = 0.0
    calibration_shift = 0.0
    minimum_signal_survival = 1.0
    for index in active:
        result = results[int(index)]
        counts = _gate_counts_for_feature(result, circuit_spec)
        support_weight = max(1, len(result.get("support", [])))
        one_factor = max(0.0, 1.0 - c1 * p1) ** counts["one_qubit"]
        two_factor = max(0.0, 1.0 - c2 * p2) ** counts["two_qubit"]
        measurement_factor = max(0.0, 1.0 - 2.0 * effective_pm) ** support_weight
        coherent_factor = max(0.0, abs(cos(coherent))) ** counts["phase"]
        attenuation = float(
            one_factor * two_factor * measurement_factor * coherent_factor
        )
        minimum_signal_survival = min(minimum_signal_survival, attenuation)
        signed_ideal = float(weights[index] * ideal[index])
        lower_term = signed_ideal * attenuation if signed_ideal >= 0.0 else signed_ideal
        lower_mean += lower_term
        inverse_bound = minimum_det ** (-support_weight) if readout_mitigation else 1.0
        estimator_bound += abs(float(weights[index])) * inverse_bound
        if readout_mitigation:
            calibration_shift += (
                abs(float(weights[index]))
                * support_weight
                * 2.0
                * interval_width
                / (minimum_det ** (support_weight + 1))
            )
        feature_details.append(
            {
                "feature_index": int(index),
                "support": result.get("support", []),
                "weight": float(weights[index]),
                "ideal_expectation": float(ideal[index]),
                "gate_counts": counts,
                "attenuation_lower": attenuation,
                "lower_contribution": lower_term,
                "inverse_estimator_bound": inverse_bound,
            }
        )
    crosstalk_tv = min(
        1.0,
        float(point.get("crosstalk", 0.0))
        * float(settings.get("crosstalk_tv_multiplier", 8.0)),
    )
    drift_tv = min(
        1.0,
        float(point.get("drift", 0.0))
        * float(settings.get("drift_tv_multiplier", 4.0)),
    )
    additive_shift = 2.0 * (crosstalk_tv + drift_tv)
    lower_mean -= additive_shift
    lower_mean -= calibration_shift
    shots = int(settings.get("shots", config_default(settings, "shots", 100_000)))
    alpha = float(settings.get("alpha", 0.01))
    estimator_bound = max(1.0, float(estimator_bound))
    statistical_radius = estimator_bound * sqrt(
        2.0 * log(max(1.0 / alpha, 1.0000001)) / max(1, shots)
    )
    observed_lcb_proxy = lower_mean - statistical_radius
    adversary_supremum = max(
        (float(value) for value in witness.get("adversary_means", {}).values()),
        default=0.0,
    )
    penalty = float(
        settings.get(
            "adversary_generalization_penalty",
            0.02,
        )
    )
    margin = observed_lcb_proxy - adversary_supremum - penalty
    return {
        "noise_point": point,
        "lower_witness_mean": float(lower_mean),
        "statistical_radius": float(statistical_radius),
        "observed_lcb_proxy": float(observed_lcb_proxy),
        "adversary_supremum": float(adversary_supremum),
        "adversary_generalization_penalty": penalty,
        "margin_lcb_proxy": float(margin),
        "pass_proxy": bool(margin > float(settings.get("minimum_margin_lcb", 0.0))),
        "minimum_signal_survival": float(minimum_signal_survival),
        "estimator_bound": float(estimator_bound),
        "calibration_uncertainty_shift": float(calibration_shift),
        "correlated_noise_shift": float(additive_shift),
        "features": feature_details,
    }


def config_default(settings: dict[str, Any], key: str, default: Any) -> Any:
    # Isolated helper keeps mypy/linters happy when a config value is absent.
    return settings[key] if key in settings else default


def evaluate_robustness_envelope(
    *,
    preflight: dict[str, Any],
    circuit_spec: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    settings = config.get("research_triad", {}).get("robustness", {})
    envelope = dict(
        settings.get(
            "envelope",
            config.get("gates", {}).get("noise_envelope", {}),
        )
    )
    for key in (
        "p_1q",
        "p_2q",
        "p_measure",
        "coherent_angle",
        "crosstalk",
        "drift",
    ):
        envelope.setdefault(key, 0.0)
    calibration = _read_calibration_context(config)
    grid_points = max(3, int(settings.get("grid_points", 21)))
    scales = np.linspace(0.0, 1.0, grid_points)
    phase_diagram = [
        _robust_margin_at_noise(
            preflight=preflight,
            circuit_spec=circuit_spec,
            point=_scaled_noise_point(envelope, float(scale)),
            settings=settings,
            calibration=calibration,
        )
        | {"scale": float(scale)}
        for scale in scales
    ]
    worst = phase_diagram[-1]
    low = 0.0
    high = 1.0
    if phase_diagram[0]["pass_proxy"]:
        for _ in range(48):
            middle = 0.5 * (low + high)
            result = _robust_margin_at_noise(
                preflight=preflight,
                circuit_spec=circuit_spec,
                point=_scaled_noise_point(envelope, middle),
                settings=settings,
                calibration=calibration,
            )
            if result["pass_proxy"]:
                low = middle
            else:
                high = middle
    critical_scale = low if phase_diagram[0]["pass_proxy"] else 0.0
    sensitivities = {}
    baseline = phase_diagram[0]["margin_lcb_proxy"]
    for key, maximum in envelope.items():
        if float(maximum) <= 0.0:
            sensitivities[key] = 0.0
            continue
        point = {name: 0.0 for name in envelope}
        point[key] = float(maximum)
        result = _robust_margin_at_noise(
            preflight=preflight,
            circuit_spec=circuit_spec,
            point=point,
            settings=settings,
            calibration=calibration,
        )
        sensitivities[key] = float(
            (result["margin_lcb_proxy"] - baseline) / float(maximum)
        )
    samples = max(1, int(settings.get("envelope_samples", 256)))
    rng = np.random.default_rng(seed)
    passing = 0
    sampled_margins = []
    for _ in range(samples):
        point = {
            key: float(rng.random()) * float(value)
            for key, value in envelope.items()
        }
        result = _robust_margin_at_noise(
            preflight=preflight,
            circuit_spec=circuit_spec,
            point=point,
            settings=settings,
            calibration=calibration,
        )
        sampled_margins.append(float(result["margin_lcb_proxy"]))
        passing += int(result["pass_proxy"])
    return {
        "schema": "codegap.research-triad-noise-envelope.v1",
        "model": (
            "Prospective registered attenuation and assignment-inversion stress "
            "model. It ranks experiments; it is not a replacement for a fresh "
            "QPU certificate."
        ),
        "envelope": envelope,
        "calibration_context": calibration,
        "zero_noise": phase_diagram[0],
        "worst_registered_noise": worst,
        "critical_common_noise_scale": float(critical_scale),
        "envelope_volume_pass_fraction": float(passing / samples),
        "sampled_margin_quantiles": {
            "q05": float(np.quantile(sampled_margins, 0.05)),
            "q50": float(np.quantile(sampled_margins, 0.50)),
            "q95": float(np.quantile(sampled_margins, 0.95)),
        },
        "one_at_a_time_margin_sensitivities": sensitivities,
        "phase_diagram": phase_diagram,
        "claim_boundary": (
            "The final experimental claim still requires a freshly calibrated "
            "independent QPU run under the preregistered certificate."
        ),
    }


def _structural_simulation_risk(
    *,
    circuit_spec: dict[str, Any],
    hardware: dict[str, Any],
) -> dict[str, Any]:
    """Register an anti-easiness proxy for shallow/peaked circuit attacks.

    This is deliberately labelled a structural risk proxy: it does not estimate
    the maximum output probability. It penalizes schedules with repeated
    matchings, one-axis structure, disconnected interaction graphs, small cuts,
    or a low density of genuinely non-Clifford gates. Deep mode separately runs
    the registered tensor-network/space-time attacks.
    """

    n = max(1, int(circuit_spec.get("n", 1)))
    schedule = circuit_spec.get("schedule", {})
    indices = tuple(int(value) for value in schedule.get("matching_indices", []))
    axes = tuple(str(value).lower() for value in schedule.get("axes", []))
    depth = max(1, len(indices))
    unique_fraction = len(set(indices)) / depth if indices else 0.0
    frequencies = np.asarray(
        [indices.count(value) for value in sorted(set(indices))],
        dtype=np.float64,
    )
    if frequencies.size:
        probabilities = frequencies / frequencies.sum()
        entropy = float(
            -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-300)))
        )
        maximum_entropy = log(max(2, len(frequencies)), 2)
        normalized_entropy = entropy / maximum_entropy
    else:
        normalized_entropy = 0.0
    axis_fraction = len(set(axes)) / 2.0

    edges = [
        tuple(sorted(map(int, edge)))
        for edge in circuit_spec.get("union_two_qubit_edges", [])
    ]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    largest_component_fraction = (
        max((len(value) for value in nx.connected_components(graph)), default=1) / n
    )
    cut = _balanced_cut_size(graph, int(circuit_spec.get("seed", 0)))
    cut_density = min(1.0, float(cut) / max(1.0, n / 4.0))

    non_clifford = _non_clifford_count(circuit_spec)
    phase_gates = sum(
        str(gate.get("name")) in {"rz", "rzz", "rxx"}
        for gate in circuit_spec.get("gates", [])
    )
    non_clifford_fraction = non_clifford / max(1, phase_gates)
    depth_diversity = min(1.0, depth / max(2.0, np.log2(n + 1.0)))

    easiness_risk = float(
        0.18 * (1.0 - unique_fraction)
        + 0.12 * (1.0 - normalized_entropy)
        + 0.15 * (1.0 - axis_fraction)
        + 0.20 * (1.0 - largest_component_fraction)
        + 0.15 * (1.0 - cut_density)
        + 0.12 * (1.0 - non_clifford_fraction)
        + 0.08 * (1.0 - depth_diversity)
    )
    return {
        "schema": "codegap.research-triad-structural-simulation-risk.v1",
        "shallow_peakedness_easiness_risk_proxy": easiness_risk,
        "unique_matching_fraction": float(unique_fraction),
        "normalized_matching_entropy": float(normalized_entropy),
        "axis_diversity_fraction": float(axis_fraction),
        "largest_interaction_component_fraction": float(largest_component_fraction),
        "balanced_cut": int(cut),
        "normalized_cut_density": float(cut_density),
        "non_clifford_gate_fraction": float(non_clifford_fraction),
        "depth_diversity": float(depth_diversity),
        "hardware_two_qubit_depth": int(hardware.get("two_qubit_depth", 0)),
        "claim_boundary": (
            "This is a preregistered structural attack-risk proxy, not an "
            "estimate of the full output peak probability. Deep mode must be "
            "used for the registered tensor-network and space-time attacks."
        ),
    }


def _triad_score(
    *,
    code: dict[str, Any],
    hardware: dict[str, Any],
    hardness: dict[str, Any],
    robustness: dict[str, Any],
    simulation_risk: dict[str, Any],
    settings: dict[str, Any],
) -> float:
    weights = settings.get("objective", {})
    n = max(1, int(code["n"]))
    rate = float(code["k"]) / n
    margin = float(robustness["worst_registered_noise"]["margin_lcb_proxy"])
    gamma = float(hardness["gamma_log10"])
    critical = float(robustness["critical_common_noise_scale"])
    structural_risk = float(
        simulation_risk["shallow_peakedness_easiness_risk_proxy"]
    )
    return float(
        float(weights.get("gamma", 1.0)) * gamma
        + float(weights.get("robust_margin", 4.0)) * margin
        + float(weights.get("critical_noise_scale", 1.0)) * critical
        + float(weights.get("logical_rate", 1.0)) * rate
        - float(weights.get("qubits", 0.03)) * n
        - float(weights.get("depth", 0.03)) * int(hardware["two_qubit_depth"])
        - float(weights.get("twoq", 0.0005)) * int(hardware["two_qubit_count"])
        - float(weights.get("swaps", 1.0)) * int(hardware["swap_count"])
        - float(weights.get("structural_simulation_risk", 2.0))
        * structural_risk
    )


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = (
        int(left["code"]["n"]),
        int(left["hardware"]["two_qubit_depth"]),
        int(left["hardware"]["two_qubit_count"]),
        float(
            left["simulation_risk"][
                "shallow_peakedness_easiness_risk_proxy"
            ]
        ),
        -float(left["hardness"]["gamma_log10"]),
        -float(left["robustness"]["worst_registered_noise"]["margin_lcb_proxy"]),
        -float(left["robustness"]["critical_common_noise_scale"]),
        -float(left["code"]["k"]) / max(1, int(left["code"]["n"])),
    )
    right_values = (
        int(right["code"]["n"]),
        int(right["hardware"]["two_qubit_depth"]),
        int(right["hardware"]["two_qubit_count"]),
        float(
            right["simulation_risk"][
                "shallow_peakedness_easiness_risk_proxy"
            ]
        ),
        -float(right["hardness"]["gamma_log10"]),
        -float(right["robustness"]["worst_registered_noise"]["margin_lcb_proxy"]),
        -float(right["robustness"]["critical_common_noise_scale"]),
        -float(right["code"]["k"]) / max(1, int(right["code"]["n"])),
    )
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def _pareto(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if not any(
            other["candidate_id"] != item["candidate_id"]
            and _dominates(other, item)
            for other in items
        )
    ]


def _minimal_qubit_certificate(
    items: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    gamma_targets = [
        float(value)
        for value in settings.get("gamma_targets", [4.0, 6.0, 8.0, 10.0])
    ]
    minimum_margin = float(settings.get("minimum_robust_margin", 0.0))
    distance = int(settings.get("minimum_distance", 3))
    maximum_structural_risk = float(
        settings.get("maximum_structural_simulation_risk", 1.0)
    )
    rows = []
    for target in gamma_targets:
        qualified = [
            item
            for item in items
            if float(item["hardness"]["gamma_log10"]) >= target
            and float(
                item["robustness"]["worst_registered_noise"]["margin_lcb_proxy"]
            )
            > minimum_margin
            and float(
                item["simulation_risk"][
                    "shallow_peakedness_easiness_risk_proxy"
                ]
            )
            <= maximum_structural_risk
            and min(
                int(item["code"]["d_x_at_least"]),
                int(item["code"]["d_z_at_least"]),
            )
            >= distance
        ]
        qualified.sort(
            key=lambda item: (
                int(item["code"]["n"]),
                int(item["hardware"]["two_qubit_depth"]),
                -float(item["robustness"]["worst_registered_noise"]["margin_lcb_proxy"]),
                -float(item["hardness"]["gamma_log10"]),
            )
        )
        winner = qualified[0] if qualified else None
        rows.append(
            {
                "gamma_target": target,
                "minimum_robust_margin": minimum_margin,
                "minimum_distance": distance,
                "maximum_structural_simulation_risk": maximum_structural_risk,
                "found": winner is not None,
                "candidate_id": winner["candidate_id"] if winner else None,
                "n": int(winner["code"]["n"]) if winner else None,
                "family": winner["family"] if winner else None,
                "gamma_log10": (
                    float(winner["hardness"]["gamma_log10"]) if winner else None
                ),
                "structural_simulation_risk": (
                    float(
                        winner["simulation_risk"][
                            "shallow_peakedness_easiness_risk_proxy"
                        ]
                    )
                    if winner
                    else None
                ),
                "robust_margin": (
                    float(
                        winner["robustness"]["worst_registered_noise"][
                            "margin_lcb_proxy"
                        ]
                    )
                    if winner
                    else None
                ),
            }
        )
    return {
        "schema": "codegap.research-triad-minimal-qubit-certificate.v1",
        "claim": "MINIMAL_WITHIN_EVALUATED_SEARCH_SPACE",
        "targets": rows,
        "warning": (
            "This is not a global lower bound on qubit count. It is an exact "
            "minimum only over the frozen evaluated candidate set."
        ),
    }


def _family_champions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        key = str(item["family"]["type"])
        grouped.setdefault(key, []).append(item)
    champions = []
    for family_type, values in sorted(grouped.items()):
        winner = max(values, key=lambda item: float(item["triad_score"]))
        champions.append(
            {
                "family_type": family_type,
                "candidate_id": winner["candidate_id"],
                "dimensions": winner["family"]["dimensions"],
                "n": winner["code"]["n"],
                "k": winner["code"]["k"],
                "distance_lower_bound": min(
                    winner["code"]["d_x_at_least"],
                    winner["code"]["d_z_at_least"],
                ),
                "gamma_log10": winner["hardness"]["gamma_log10"],
                "robust_margin": winner["robustness"]["worst_registered_noise"][
                    "margin_lcb_proxy"
                ],
                "critical_noise_scale": winner["robustness"][
                    "critical_common_noise_scale"
                ],
                "two_qubit_depth": winner["hardware"]["two_qubit_depth"],
                "structural_simulation_risk": winner["simulation_risk"][
                    "shallow_peakedness_easiness_risk_proxy"
                ],
                "triad_score": winner["triad_score"],
            }
        )
    return champions


def run_research_triad(
    config: dict[str, Any],
    output: Path,
    *,
    progress: ProgressManager | None = None,
) -> dict[str, Any]:
    """Jointly search code family, hardware robustness, and verification gap.

    The command solves the three research-design tasks as one frozen
    multi-objective search:

    1. compare bivariate and multivariate bicycle families under exact CSS and
       hardware-proxy constraints;
    2. stress each verifier under a registered gate/readout/crosstalk/drift
       envelope, including assignment-inversion amplification;
    3. identify the smallest evaluated circuits that retain a registered
       verification-versus-simulation gap.
    """

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manager = progress or default_progress(config)
    topology = _load_topology(config)
    triad = config.get("research_triad", {})
    mode = str(triad.get("mode", "deep")).lower()
    if mode not in {"proxy", "deep"}:
        raise ValueError("research_triad.mode must be 'proxy' or 'deep'.")

    family_records, accepted_codes = _family_screen(
        config=config,
        topology=topology,
        progress=manager,
    )
    representatives = _select_code_representatives(
        accepted_codes,
        triad.get("family_search", {}),
    )
    _json_write(output / "family_records.json", family_records)
    _json_write(
        output / "selected_code_representatives.json",
        [
            {
                "code_id": item["code_id"],
                "family": item["spec"].to_dict(),
                "code": item["code"],
                "hardware_code_proxy": item["hardware_code_proxy"],
                "code_score": item["code_score"],
            }
            for item in representatives
        ],
    )

    circuit_records: list[dict[str, Any]] = []
    schedule_reports = []
    circuit_bar = manager.bar(
        representatives,
        total=len(representatives),
        desc=f"TriadSearch: {mode} circuit co-design",
        unit="code",
        leave=True,
    )
    for code_index, item in enumerate(circuit_bar):
        seed = int(config.get("seed", 0)) + code_index * 1_000_003
        if mode == "deep":
            finalists, schedule_report = search_adversarial_schedules(
                spec=item["spec"],
                h_x=item["h_x"],
                h_z=item["h_z"],
                config=config,
                topology=topology,
                seed=seed,
                progress=manager,
            )
        else:
            finalists, schedule_report = _proxy_schedule_search(
                item=item,
                config=config,
                topology=topology,
                seed=seed,
            )
        for finalist in finalists:
            _export_finalist_layouts(finalist, topology)
        _export_schedule_report_layouts(schedule_report, topology)
        schedule_report["code_id"] = item["code_id"]
        schedule_report["family"] = item["spec"].to_dict()
        schedule_reports.append(schedule_report)
        for finalist_index, finalist in enumerate(finalists):
            preflight = finalist.get("verifier_preflight") or finalist.get(
                "circuit_spec", {}
            ).get("verifier_selection")
            if not preflight or not preflight.get("passed", False):
                continue
            robustness = evaluate_robustness_envelope(
                preflight=preflight,
                circuit_spec=finalist["circuit_spec"],
                config=config,
                seed=seed + finalist_index * 65537,
            )
            candidate_id = _digest_payload(
                {
                    "code_id": item["code_id"],
                    "schedule_id": finalist["circuit_spec"].get("schedule_id"),
                    "mode": mode,
                }
            )[:16]
            simulation_risk = _structural_simulation_risk(
                circuit_spec=finalist["circuit_spec"],
                hardware=finalist["hardware"],
            )
            score = _triad_score(
                code=item["code"],
                hardware=finalist["hardware"],
                hardness=finalist["hardness"],
                robustness=robustness,
                simulation_risk=simulation_risk,
                settings=triad,
            )
            record = {
                "candidate_id": candidate_id,
                "code_id": item["code_id"],
                "family": item["spec"].to_dict(),
                "code": item["code"],
                "hardware": finalist["hardware"],
                "hardness": finalist["hardness"],
                "robustness": robustness,
                "simulation_risk": simulation_risk,
                "verifier_preflight": preflight,
                "circuit_spec": finalist["circuit_spec"],
                "schedule_objective": finalist.get("objective"),
                "triad_score": score,
                "mode": mode,
            }
            circuit_records.append(record)
        circuit_bar.set_postfix(
            code=item["code_id"],
            schedules=len(finalists),
            retained=len(circuit_records),
            refresh=False,
        )
    circuit_bar.close()

    circuit_records.sort(key=lambda item: float(item["triad_score"]), reverse=True)
    frontier = _pareto(circuit_records)
    frontier.sort(
        key=lambda item: (
            int(item["code"]["n"]),
            -float(item["hardness"]["gamma_log10"]),
            -float(item["robustness"]["worst_registered_noise"]["margin_lcb_proxy"]),
        )
    )
    champions = _family_champions(circuit_records) if circuit_records else []
    minimal = _minimal_qubit_certificate(
        circuit_records,
        triad.get("minimal_qubit", {}),
    )
    recommended = {
        "best_overall": circuit_records[0]["candidate_id"] if circuit_records else None,
        "most_noise_robust": (
            max(
                circuit_records,
                key=lambda item: float(
                    item["robustness"]["worst_registered_noise"]["margin_lcb_proxy"]
                ),
            )["candidate_id"]
            if circuit_records
            else None
        ),
        "largest_gap": (
            max(
                circuit_records,
                key=lambda item: float(item["hardness"]["gamma_log10"]),
            )["candidate_id"]
            if circuit_records
            else None
        ),
        "smallest_qubit_qualified": next(
            (
                row["candidate_id"]
                for row in minimal["targets"]
                if row["found"]
            ),
            None,
        ),
    }

    _json_write(output / "schedule_reports.json", schedule_reports)
    _json_write(output / "circuit_records.json", circuit_records)
    _json_write(output / "triad_pareto_frontier.json", frontier)
    _json_write(output / "family_champions.json", champions)
    _json_write(output / "minimal_qubit_gap_certificate.json", minimal)
    _json_write(output / "recommended_experiments.json", recommended)

    status = "PASS" if circuit_records else "STOP_NO_TRIAD_CANDIDATE"
    report = {
        "schema": "codegap.research-triad.v1",
        "status": status,
        "mode": mode,
        "hardware": {
            "name": topology.name,
            "source": topology.source,
            "num_qubits": topology.graph.number_of_nodes(),
            "coupling_edges": topology.graph.number_of_edges(),
            "structural_fingerprint": topology.structural_fingerprint,
            "component_policy": topology.graph.graph.get("component_policy"),
            "original_num_qubits": topology.graph.graph.get(
                "original_num_qubits", topology.graph.number_of_nodes()
            ),
            "component_sizes": topology.graph.graph.get(
                "component_sizes", [topology.graph.number_of_nodes()]
            ),
            "excluded_physical_qubits": topology.graph.graph.get(
                "excluded_physical_qubits", []
            ),
            "active_component_physical_qubits": topology.graph.graph.get(
                "physical_qubits", list(range(topology.graph.number_of_nodes()))
            ),
        },
        "acceleration": accel_diagnostics(),
        "counts": {
            "family_records": len(family_records),
            "accepted_codes": len(accepted_codes),
            "schedule_searched_codes": len(representatives),
            "circuit_candidates": len(circuit_records),
            "pareto_candidates": len(frontier),
        },
        "recommended_experiments": recommended,
        "family_champions": champions,
        "minimal_qubit_gap": minimal,
        "research_questions": {
            "optimal_real_code_families": (
                "Answered by the frozen family/circuit Pareto frontier and family champions."
            ),
            "hardware_noise_robustness": (
                "Answered prospectively by the registered noise phase diagrams, critical "
                "common-noise scales, and sensitivity ranking; final claims still require QPU data."
            ),
            "few_qubits_large_verification_simulation_gap": (
                "Answered within the evaluated search space by the minimal-qubit gap certificate, with an explicit structural shallow/peaked-circuit easiness-risk penalty."
            ),
        },
        "claim_boundaries": [
            "Family optimality is only within the frozen evaluated search space.",
            "Proxy mode is a screening stage; deep mode is required for the registered attack suite.",
            "Classical costs are implemented attack/proxy results, not unconditional lower bounds.",
            "Noise robustness is prospective until confirmed by a fresh independent QPU run.",
            "Local witnesses do not certify total-variation closeness of the complete output distribution.",
        ],
    }
    _json_write(output / "research_triad_report.json", report)
    (output / "README.md").write_text(
        "# CodeGap Research Triad\n\n"
        "This frozen artifact jointly searches code families, prospective hardware-noise "
        "robustness, and low-qubit verification-versus-simulation gap. See "
        "`research_triad_report.json`, `triad_pareto_frontier.json`, and "
        "`minimal_qubit_gap_certificate.json`.\n",
        encoding="utf-8",
    )
    freeze_result = freeze(output)
    verification = verify_freeze(output / "freeze_manifest.json")
    report["freeze"] = {
        "manifest": str(output / "freeze_manifest.json"),
        "verified": bool(verification["ok"]),
        "files": freeze_result.get("files"),
    }
    return report
