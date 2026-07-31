from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Iterable

import numpy as np

from .bicycle import BicycleFamilySpec
from .gf2 import rank


@dataclass(frozen=True)
class CircuitGate:
    name: str
    qubits: tuple[int, ...]
    angle: float | None = None
    layer: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qubits": list(self.qubits),
            "angle": self.angle,
            "layer": self.layer,
        }


def _rowspace_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left, dtype=np.uint8) & 1
    right = np.asarray(right, dtype=np.uint8) & 1
    left_rank = rank(left)
    right_rank = rank(right)
    if left_rank != right_rank:
        return False
    return rank(np.vstack([left, right])) == left_rank


def permutation_preserves_css(
    h_x: np.ndarray,
    h_z: np.ndarray,
    permutation: tuple[int, ...],
) -> bool:
    return bool(
        _rowspace_equal(h_x, h_x[:, permutation])
        and _rowspace_equal(h_z, h_z[:, permutation])
    )


def _affine_permutation(
    spec: BicycleFamilySpec,
    *,
    sx: int,
    sy: int,
    dx: int,
    dy: int,
    swap_blocks: bool,
    swap_coordinates: bool = False,
) -> tuple[int, ...]:
    if swap_coordinates and spec.l != spec.m:
        raise ValueError("Coordinate swap requires l == m.")
    block = spec.block_size
    permutation = [0] * spec.n
    for source_block in (0, 1):
        target_block = 1 - source_block if swap_blocks else source_block
        for x in range(spec.l):
            for y in range(spec.m):
                source = source_block * block + x * spec.m + y
                if swap_coordinates:
                    raw_x, raw_y = y, x
                else:
                    raw_x, raw_y = x, y
                target_x = (sx * raw_x + dx) % spec.l
                target_y = (sy * raw_y + dy) % spec.m
                target = target_block * block + target_x * spec.m + target_y
                permutation[source] = target
    return tuple(permutation)


def _compose_permutations(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(left[right[index]] for index in range(len(left)))


def _is_fixed_point_free_involution(permutation: tuple[int, ...]) -> bool:
    return all(
        permutation[index] != index
        and permutation[permutation[index]] == index
        for index in range(len(permutation))
    )


def _matching_from_involution(
    permutation: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    edges = {
        tuple(sorted((index, permutation[index])))
        for index in range(len(permutation))
    }
    return tuple(sorted(edges))


def _matching_payload(
    permutation: tuple[int, ...],
    *,
    kind: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    edges = _matching_from_involution(permutation)
    payload = {
        "kind": kind,
        "permutation": list(permutation),
        "edges": [list(edge) for edge in edges],
        "css_rowspaces_preserved": True,
        "fixed_point_free_involution": True,
    }
    payload.update(metadata)
    payload["matching_id"] = sha256(
        json.dumps(payload["edges"], separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return payload


def css_automorphism_matchings(
    spec: BicycleFamilySpec,
    h_x: np.ndarray,
    h_z: np.ndarray,
    *,
    max_matchings: int | None = None,
    include_pair_compositions: bool = True,
) -> list[dict[str, Any]]:
    """Enumerate verified fixed-point-free CSS automorphism matchings.

    The pool includes translations, coordinate reflections, optional block swaps,
    and (when requested) pairwise compositions that remain involutions. Every
    candidate is independently checked against both CSS row spaces.
    """

    candidates: list[dict[str, Any]] = []
    signs_x = sorted({1 % spec.l, (-1) % spec.l})
    signs_y = sorted({1 % spec.m, (-1) % spec.m})
    coordinate_modes = (False, True) if spec.l == spec.m else (False,)
    for sx in signs_x:
        for sy in signs_y:
            for dx in range(spec.l):
                for dy in range(spec.m):
                    for swap_blocks in (False, True):
                        for swap_coordinates in coordinate_modes:
                            permutation = _affine_permutation(
                                spec,
                                sx=sx,
                                sy=sy,
                                dx=dx,
                                dy=dy,
                                swap_blocks=swap_blocks,
                                swap_coordinates=swap_coordinates,
                            )
                            if not _is_fixed_point_free_involution(permutation):
                                continue
                            if not permutation_preserves_css(
                                h_x, h_z, permutation
                            ):
                                continue
                            candidates.append(
                                _matching_payload(
                                    permutation,
                                    kind="affine_qc_css_automorphism",
                                    metadata={
                                        "affine": {
                                            "sx": int(sx),
                                            "sy": int(sy),
                                            "dx": int(dx),
                                            "dy": int(dy),
                                            "swap_blocks": bool(swap_blocks),
                                            "swap_coordinates": bool(
                                                swap_coordinates
                                            ),
                                        }
                                    },
                                )
                            )

    unique: dict[tuple[tuple[int, int], ...], dict[str, Any]] = {}
    for item in candidates:
        key = tuple(tuple(edge) for edge in item["edges"])
        unique.setdefault(key, item)

    if include_pair_compositions:
        base = list(unique.values())
        permutations = [tuple(item["permutation"]) for item in base]
        for left_index, left in enumerate(permutations):
            for right_index in range(left_index + 1, len(permutations)):
                composed = _compose_permutations(left, permutations[right_index])
                if not _is_fixed_point_free_involution(composed):
                    continue
                if not permutation_preserves_css(h_x, h_z, composed):
                    continue
                item = _matching_payload(
                    composed,
                    kind="composed_qc_css_automorphism",
                    metadata={
                        "composition": [
                            base[left_index]["matching_id"],
                            base[right_index]["matching_id"],
                        ]
                    },
                )
                key = tuple(tuple(edge) for edge in item["edges"])
                unique.setdefault(key, item)
                if max_matchings is not None and len(unique) >= max_matchings:
                    break
            if max_matchings is not None and len(unique) >= max_matchings:
                break

    result = sorted(unique.values(), key=lambda item: item["matching_id"])
    if max_matchings is not None:
        result = result[: int(max_matchings)]
    return result


def _phase_signs(
    h_x: np.ndarray,
    h_z: np.ndarray,
    layer: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed + 104729 * (layer + 1))
    rows = np.vstack([h_x, h_z])
    choose = max(1, min(4, rows.shape[0]))
    selected = rng.choice(rows.shape[0], size=choose, replace=False)
    secret = np.bitwise_xor.reduce(rows[selected], axis=0)
    random_mask = rng.integers(0, 2, size=secret.shape[0], dtype=np.uint8)
    secret ^= random_mask
    return [1 if bit == 0 else -1 for bit in secret.tolist()]


def build_verifiable_mixing_circuit(
    *,
    spec: BicycleFamilySpec,
    h_x: np.ndarray,
    h_z: np.ndarray,
    matching_pool: list[dict[str, Any]],
    matching_indices: Iterable[int],
    axes: Iterable[str],
    config: dict,
    seed: int,
    angle_scales: Iterable[float] | None = None,
    schedule_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    matching_indices = tuple(int(value) for value in matching_indices)
    axes = tuple(str(value).lower() for value in axes)
    if not matching_indices:
        raise ValueError("A circuit schedule must contain at least one layer.")
    if len(axes) != len(matching_indices):
        raise ValueError("One axis is required for every matching layer.")
    if any(axis not in {"zz", "xx"} for axis in axes):
        raise ValueError("Schedule axes must be 'zz' or 'xx'.")
    if any(index < 0 or index >= len(matching_pool) for index in matching_indices):
        raise ValueError("Schedule references an unknown matching.")
    scales = (
        tuple(float(value) for value in angle_scales)
        if angle_scales is not None
        else (1.0,) * len(matching_indices)
    )
    if len(scales) != len(matching_indices):
        raise ValueError("One angle scale is required for every layer.")

    theta_single = float(config["circuit"]["theta_single"])
    theta_pair = float(config["circuit"]["theta_pair"])
    layers: list[dict[str, Any]] = []
    gates: list[CircuitGate] = []
    for qubit in range(spec.n):
        gates.append(CircuitGate("h", (qubit,), layer=-1))

    for layer, (matching_index, axis, scale) in enumerate(
        zip(matching_indices, axes, scales)
    ):
        matching = matching_pool[matching_index]
        if not matching.get("css_rowspaces_preserved", False):
            raise ValueError("A schedule matching does not preserve CSS row spaces.")
        if not matching.get("fixed_point_free_involution", False):
            raise ValueError("A schedule matching is not a perfect involution.")
        signs = _phase_signs(h_x, h_z, layer, seed)
        for qubit, sign in enumerate(signs):
            gates.append(
                CircuitGate(
                    "rz",
                    (qubit,),
                    angle=theta_single * sign * scale,
                    layer=layer,
                )
            )
        gate_name = "rzz" if axis == "zz" else "rxx"
        for left, right in matching["edges"]:
            gates.append(
                CircuitGate(
                    gate_name,
                    (int(left), int(right)),
                    angle=theta_pair * scale,
                    layer=layer,
                )
            )
        layers.append(
            {
                "layer": layer,
                "axis": axis,
                "theta_pair": theta_pair * scale,
                "theta_scale": scale,
                "phase_signs": signs,
                "matching_index": matching_index,
                "matching": matching,
            }
        )

    depth = len(layers)
    for qubit in range(spec.n):
        gates.append(CircuitGate("h", (qubit,), layer=depth))
    for qubit in range(spec.n):
        gates.append(CircuitGate("measure", (qubit,), layer=depth + 1))

    union_edges = tuple(
        sorted(
            {
                tuple(sorted((int(left), int(right))))
                for layer in layers
                for left, right in layer["matching"]["edges"]
            }
        )
    )
    verifier_masks = np.vstack([h_x, h_z]).astype(np.uint8)
    order = np.argsort(verifier_masks.sum(axis=1), kind="stable")
    maximum_masks = int(config.get("codesign", {}).get("max_verifier_masks", 48))
    verifier_masks = verifier_masks[order[:maximum_masks]]
    schedule_payload = {
        "matching_indices": list(matching_indices),
        "matching_ids": [
            matching_pool[index]["matching_id"] for index in matching_indices
        ],
        "axes": list(axes),
        "angle_scales": list(scales),
    }
    schedule_id = sha256(
        json.dumps(schedule_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]

    return {
        "schema": "codegap.verifiable-mixing-circuit.v2-adversarial-schedule",
        "n": spec.n,
        "seed": seed,
        "theta_single": theta_single,
        "theta_pair": theta_pair,
        "schedule_id": schedule_id,
        "schedule": schedule_payload,
        "schedule_metadata": schedule_metadata or {},
        "layers": layers,
        "gates": [gate.to_dict() for gate in gates],
        "union_two_qubit_edges": [list(edge) for edge in union_edges],
        "two_qubit_count": sum(
            len(layer["matching"]["edges"]) for layer in layers
        ),
        "logical_two_qubit_depth": depth,
        "noncommuting_axes": sorted(set(axes)),
        "verifier_masks": verifier_masks.tolist(),
        "verifier_cost_model": "GF(2) parity evaluation O(n*w)",
        "relation_preservation": {
            "method": "verified_CSS_permutation_automorphisms",
            "all_layers_css_rowspaces_preserved": all(
                layer["matching"]["css_rowspaces_preserved"] for layer in layers
            ),
            "all_layers_involutive": all(
                layer["matching"]["fixed_point_free_involution"]
                for layer in layers
            ),
        },
    }



def build_tracked_frame_mixing_circuit(
    *,
    spec: BicycleFamilySpec,
    h_x: np.ndarray,
    h_z: np.ndarray,
    matching_pool: list[dict[str, Any]],
    matching_indices: Iterable[int],
    axes: Iterable[str],
    config: dict,
    seed: int,
    angle_scales: Iterable[float] | None = None,
    schedule_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a target-native circuit with an exactly tracked CSS wire frame.

    Matchings need not stabilize the original CSS row spaces. Instead, every
    involution updates the current column frame exactly. The final verifier masks
    are taken from the final tracked frame and are subsequently validated against
    the actual circuit by the light-cone certificate.
    """

    matching_indices = tuple(int(value) for value in matching_indices)
    axes = tuple(str(value).lower() for value in axes)
    if not matching_indices:
        raise ValueError("A circuit schedule must contain at least one layer.")
    if len(axes) != len(matching_indices):
        raise ValueError("One axis is required for every matching layer.")
    if any(axis not in {"zz", "xx"} for axis in axes):
        raise ValueError("Schedule axes must be 'zz' or 'xx'.")
    if any(index < 0 or index >= len(matching_pool) for index in matching_indices):
        raise ValueError("Schedule references an unknown matching.")
    scales = (
        tuple(float(value) for value in angle_scales)
        if angle_scales is not None
        else (1.0,) * len(matching_indices)
    )
    if len(scales) != len(matching_indices):
        raise ValueError("One angle scale is required for every layer.")

    theta_single = float(config["circuit"]["theta_single"])
    theta_pair = float(config["circuit"]["theta_pair"])
    current_x = np.asarray(h_x, dtype=np.uint8).copy() & 1
    current_z = np.asarray(h_z, dtype=np.uint8).copy() & 1
    cumulative = tuple(range(spec.n))
    transitions: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    gates: list[CircuitGate] = [
        CircuitGate("h", (qubit,), layer=-1) for qubit in range(spec.n)
    ]

    for layer, (matching_index, axis, scale) in enumerate(
        zip(matching_indices, axes, scales)
    ):
        matching = matching_pool[matching_index]
        if not matching.get("fixed_point_free_involution", False):
            raise ValueError("A target-native matching is not a perfect involution.")
        if not matching.get("tracked_css_frame_compatible", False):
            raise ValueError("A matching is not registered for tracked CSS frames.")
        permutation = tuple(int(value) for value in matching["permutation"])
        if sorted(permutation) != list(range(spec.n)):
            raise ValueError("Matching permutation is not bijective.")
        signs = _phase_signs(current_x, current_z, layer, seed)
        for qubit, sign in enumerate(signs):
            gates.append(
                CircuitGate(
                    "rz",
                    (qubit,),
                    angle=theta_single * sign * scale,
                    layer=layer,
                )
            )
        gate_name = "rzz" if axis == "zz" else "rxx"
        for left, right in matching["edges"]:
            gates.append(
                CircuitGate(
                    gate_name,
                    (int(left), int(right)),
                    angle=theta_pair * scale,
                    layer=layer,
                )
            )

        next_x = current_x[:, permutation]
        next_z = current_z[:, permutation]
        next_cumulative = tuple(
            cumulative[permutation[index]] for index in range(spec.n)
        )
        transition_exact = bool(
            np.array_equal(next_x, current_x[:, permutation])
            and np.array_equal(next_z, current_z[:, permutation])
            and np.array_equal(next_x, np.asarray(h_x, dtype=np.uint8)[:, next_cumulative])
            and np.array_equal(next_z, np.asarray(h_z, dtype=np.uint8)[:, next_cumulative])
        )
        transitions.append(
            {
                "layer": layer,
                "matching_id": matching["matching_id"],
                "transition_exact": transition_exact,
                "rank_x_before": rank(current_x),
                "rank_x_after": rank(next_x),
                "rank_z_before": rank(current_z),
                "rank_z_after": rank(next_z),
            }
        )
        current_x, current_z = next_x, next_z
        cumulative = next_cumulative
        layers.append(
            {
                "layer": layer,
                "axis": axis,
                "theta_pair": theta_pair * scale,
                "theta_scale": scale,
                "phase_signs": signs,
                "matching_index": matching_index,
                "matching": matching,
                "css_frame_transition": transitions[-1],
            }
        )

    depth = len(layers)
    gates.extend(CircuitGate("h", (qubit,), layer=depth) for qubit in range(spec.n))
    gates.extend(
        CircuitGate("measure", (qubit,), layer=depth + 1)
        for qubit in range(spec.n)
    )
    union_edges = tuple(
        sorted(
            {
                tuple(sorted((int(left), int(right))))
                for layer in layers
                for left, right in layer["matching"]["edges"]
            }
        )
    )
    verifier_masks = np.vstack([current_x, current_z]).astype(np.uint8)
    order = np.argsort(verifier_masks.sum(axis=1), kind="stable")
    maximum_masks = int(config.get("codesign", {}).get("max_verifier_masks", 48))
    verifier_masks = verifier_masks[order[:maximum_masks]]
    schedule_payload = {
        "matching_indices": list(matching_indices),
        "matching_ids": [
            matching_pool[index]["matching_id"] for index in matching_indices
        ],
        "axes": list(axes),
        "angle_scales": list(scales),
        "relation_mode": "tracked_css_wire_frame",
    }
    schedule_id = sha256(
        json.dumps(schedule_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    final_exact = bool(
        np.array_equal(current_x, np.asarray(h_x, dtype=np.uint8)[:, cumulative])
        and np.array_equal(current_z, np.asarray(h_z, dtype=np.uint8)[:, cumulative])
    )
    return {
        "schema": "codegap.verifiable-mixing-circuit.v3-target-native-frame",
        "n": spec.n,
        "seed": seed,
        "theta_single": theta_single,
        "theta_pair": theta_pair,
        "schedule_id": schedule_id,
        "schedule": schedule_payload,
        "schedule_metadata": schedule_metadata or {},
        "layers": layers,
        "gates": [gate.to_dict() for gate in gates],
        "union_two_qubit_edges": [list(edge) for edge in union_edges],
        "two_qubit_count": sum(
            len(layer["matching"]["edges"]) for layer in layers
        ),
        "logical_two_qubit_depth": depth,
        "noncommuting_axes": sorted(set(axes)),
        "verifier_masks": verifier_masks.tolist(),
        "verifier_cost_model": "GF(2) parity evaluation O(n*w)",
        "relation_preservation": {
            "method": "exact_tracked_CSS_wire_frame",
            "relation_mode": "tracked_css_wire_frame",
            "all_layers_css_rowspaces_preserved": all(
                layer["matching"].get("css_rowspaces_preserved", False)
                for layer in layers
            ),
            "all_layers_tracked_frame_transitions_exact": all(
                item["transition_exact"] for item in transitions
            ),
            "all_layers_involutive": all(
                layer["matching"]["fixed_point_free_involution"]
                for layer in layers
            ),
            "initial_final_ranks_preserved": bool(
                rank(current_x) == rank(h_x) and rank(current_z) == rank(h_z)
            ),
            "final_frame_matches_cumulative_permutation": final_exact,
            "verifier_masks_from_final_tracked_frame": True,
            "cumulative_permutation": list(cumulative),
            "transitions": transitions,
            "claim_boundary": (
                "This is an exact dynamic wire-frame certificate, not a claim "
                "that every physical matching stabilizes the original fixed CSS "
                "row spaces. Ideal witness expectations are independently checked "
                "on the actual circuit by exact backward light cones."
            ),
        },
    }

def design_verifiable_mixing_circuit(
    *,
    spec: BicycleFamilySpec,
    h_x: np.ndarray,
    h_z: np.ndarray,
    config: dict,
    seed: int,
) -> dict[str, Any]:
    settings = config.get("codesign", {})
    depth = int(settings.get("mixing_layers", 6))
    pool = css_automorphism_matchings(
        spec,
        h_x,
        h_z,
        max_matchings=int(settings.get("max_automorphism_matchings", 64)),
    )
    if not pool:
        raise ValueError("No fixed-point-free CSS automorphism matching was found.")
    axes_pool = tuple(
        str(value).lower() for value in settings.get("axes", ["zz", "xx"])
    )
    indices = tuple(layer % len(pool) for layer in range(depth))
    axes = tuple(axes_pool[layer % len(axes_pool)] for layer in range(depth))
    return build_verifiable_mixing_circuit(
        spec=spec,
        h_x=h_x,
        h_z=h_z,
        matching_pool=pool,
        matching_indices=indices,
        axes=axes,
        config=config,
        seed=seed,
        schedule_metadata={"mode": "deterministic_round_robin"},
    )


def circuit_qasm3(circuit_spec: dict) -> str:
    n = int(circuit_spec["n"])
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{n}] q;",
        f"bit[{n}] c;",
    ]
    for gate in circuit_spec["gates"]:
        name = str(gate["name"])
        qubits = [int(value) for value in gate["qubits"]]
        angle = float(gate.get("angle") or 0.0)
        if name in {"h", "rz"}:
            if name == "h":
                lines.append(f"h q[{qubits[0]}];")
            else:
                lines.append(f"rz({angle:.17g}) q[{qubits[0]}];")
        elif name in {"rzz", "rxx"}:
            lines.append(
                f"{name}({angle:.17g}) q[{qubits[0]}], q[{qubits[1]}];"
            )
        elif name == "measure":
            lines.append(f"c[{qubits[0]}] = measure q[{qubits[0]}];")
        else:
            raise ValueError(f"Unsupported QASM gate: {name}")
    return "\n".join(lines) + "\n"
