from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from itertools import islice
from math import comb
import json
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import numpy as np

from .bicycle import (
    BicycleFamilySpec,
    build_bicycle_css,
    enumerate_specs,
    random_specs,
)
from .dedup import IsomorphismRegistry, colored_interaction_graph
from .fast_backend import diagnostics as acceleration_diagnostics
from .fast_backend import screen_qc_batch
from .hardware import HardwareTopology, interaction_edges_from_checks
from .models import Candidate, CodeMetrics, HardwareMetrics, HardnessMetrics
from .progress import ProgressManager, default_progress
from .schedule_search import search_adversarial_schedules


T = TypeVar("T")


def candidate_id(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()[:16]


def dominates(left: Candidate, right: Candidate) -> bool:
    l = (
        -left.code.d_x_at_least,
        -left.code.d_z_at_least,
        -left.code.k,
        left.hardware.two_qubit_count,
        left.hardware.two_qubit_depth,
        left.hardware.swap_count,
        left.hardware.crossing_edges,
        -left.hardness.gamma_log10,
    )
    r = (
        -right.code.d_x_at_least,
        -right.code.d_z_at_least,
        -right.code.k,
        right.hardware.two_qubit_count,
        right.hardware.two_qubit_depth,
        right.hardware.swap_count,
        right.hardware.crossing_edges,
        -right.hardness.gamma_log10,
    )
    return all(a <= b for a, b in zip(l, r)) and any(
        a < b for a, b in zip(l, r)
    )


def pareto_front(
    candidates: Iterable[Candidate],
    progress: ProgressManager | None = None,
) -> list[Candidate]:
    items = list(candidates)
    manager = progress or default_progress()
    frontier: list[Candidate] = []
    iterator = manager.bar(
        items,
        total=len(items),
        desc="CodeForge: Pareto dominance",
        unit="candidate",
        leave=progress is None,
    )
    for candidate in iterator:
        if not any(
            other.candidate_id != candidate.candidate_id
            and dominates(other, candidate)
            for other in items
        ):
            frontier.append(candidate)
        iterator.set_postfix(frontier=len(frontier), refresh=False)
    return frontier


def co_design_objective(candidate: Candidate, weights: dict[str, float]) -> float:
    """Adversarial schedule objective requested by the preregistration.

    Cotengra remains the minimizing classical adversary. Across schedules we
    maximize its best found cost relative to verification, with depth and gate
    penalties. Gate C still uses the minimum across every registered attack.
    """

    cotengra_flops = candidate.hardness.cotengra.get("contraction_flops")
    classical = float(
        cotengra_flops
        if cotengra_flops is not None
        else candidate.hardness.best_attack_operations
    )
    lambda_depth = float(
        weights.get("lambda_depth", weights.get("depth", 0.02))
    )
    mu_twoq = float(weights.get("mu_twoq", weights.get("twoq", 0.0005)))
    return float(
        np.log10(max(classical, 1.0))
        - np.log10(max(candidate.hardness.verify_operations, 1.0))
        - lambda_depth * candidate.hardware.two_qubit_depth
        - mu_twoq * candidate.hardware.two_qubit_count
    )


def _batches(iterator: Iterator[T], batch_size: int) -> Iterator[list[T]]:
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def _support_arrays(
    specs: list[BicycleFamilySpec],
) -> tuple[np.ndarray, np.ndarray]:
    if not specs:
        raise ValueError("Cannot encode an empty candidate batch.")
    l, m = specs[0].l, specs[0].m
    weight_a = len(specs[0].support_a)
    weight_b = len(specs[0].support_b)
    a = np.empty((len(specs), weight_a), dtype=np.uint16)
    b = np.empty((len(specs), weight_b), dtype=np.uint16)
    for row, spec in enumerate(specs):
        if spec.l != l or spec.m != m:
            raise ValueError("A screening batch must contain one QC geometry.")
        a[row] = [x * m + y for x, y in spec.support_a]
        b[row] = [x * m + y for x, y in spec.support_b]
    return a, b


def _support_count(n: int, minimum: int) -> int:
    return sum(comb(n, weight) for weight in range(1, minimum))


def _mask_support(mask: int, n: int) -> list[int]:
    return [index for index in range(min(64, n)) if (mask >> index) & 1]


def _prefilter_code_record(
    *,
    spec: BicycleFamilySpec,
    rank_x: int,
    rank_z: int,
    logical: int,
    flags: int,
    witness_x: int,
    witness_z: int,
    min_dx: int,
    min_dz: int,
    backend: str,
) -> dict:
    dx_ok = bool(flags & 0x02)
    dz_ok = bool(flags & 0x04)
    n = spec.n
    return {
        "n": n,
        "k": logical,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "commutes": True,
        "d_x": None,
        "d_z": None,
        "d_x_at_least": min_dx if dx_ok else 0,
        "d_z_at_least": min_dz if dz_ok else 0,
        "d_x_certificate": {
            "method": "exact_low_weight_GF2_enumeration",
            "backend": backend,
            "inspected": _support_count(n, min_dx) if dx_ok else None,
            "witness_weight": int(witness_x).bit_count() if witness_x else None,
            "witness_support": _mask_support(int(witness_x), n) if witness_x else None,
        },
        "d_z_certificate": {
            "method": "exact_low_weight_GF2_enumeration",
            "backend": backend,
            "inspected": _support_count(n, min_dz) if dz_ok else None,
            "witness_weight": int(witness_z).bit_count() if witness_z else None,
            "witness_support": _mask_support(int(witness_z), n) if witness_z else None,
        },
        "row_weight_x_max": None,
        "row_weight_z_max": None,
    }


def _hardness_dataclass(payload: dict) -> HardnessMetrics:
    return HardnessMetrics(
        verify_operations=payload["verify_operations"],
        statevector_operations=payload["statevector_operations"],
        tensor_proxy_operations=payload["tensor_proxy_operations"],
        mps_proxy_operations=payload["mps_proxy_operations"],
        stabilizer_rank_proxy_operations=payload[
            "stabilizer_rank_proxy_operations"
        ],
        schrodinger_feynman_proxy_operations=payload[
            "schrodinger_feynman_proxy_operations"
        ],
        best_attack_operations=payload["best_attack_operations"],
        gamma_log10=payload["gamma_log10"],
        treewidth_upper_bound=payload["treewidth_upper_bound"],
        treewidth_lower_bound=payload["treewidth_lower_bound"],
        assumptions=tuple(payload["assumptions"]),
        method=payload["method"],
        best_attack_name=payload["best_attack_name"],
        tensor_graph_treewidth_lower=payload["tensor_graph_treewidth_lower"],
        tensor_graph_treewidth_upper=payload["tensor_graph_treewidth_upper"],
        line_graph_treewidth_lower=payload["line_graph_treewidth_lower"],
        line_graph_treewidth_upper=payload["line_graph_treewidth_upper"],
        tensor_count=payload["tensor_count"],
        tensor_network_edges=payload["tensor_network_edges"],
        cotengra=payload["cotengra"],
        cuts=payload["cuts"],
        non_clifford=payload["non_clifford"],
        attack_costs=payload["attack_costs"],
        claim_scope=payload["claim_scope"],
    )


def search_code_families(
    config: dict,
    output: Path,
    progress: ProgressManager | None = None,
) -> tuple[list[Candidate], dict]:
    """Search code representatives, then adversarial schedules per code.

    The expensive schedule stages are deliberately ordered as preregistered:
    relation preservation, live zero-SWAP embedding, diversity, GPU verifier
    preflight with exact local signals and GF(2) independence, cutwidth,
    line-graph treewidth, short Cotengra, deep Cotengra, then NoiseCert later in
    the pipeline.
    """

    output.mkdir(parents=True, exist_ok=True)
    manager = progress or default_progress(config)
    search = config["search"]
    constraints = config["constraints"]
    hardware_config = config["hardware"]
    acceleration = config.get("acceleration", {})
    schedule_settings = config.get("schedule_search", {})
    snapshot_value = hardware_config.get("target_snapshot")
    if snapshot_value:
        snapshot_path = Path(snapshot_value)
        if not snapshot_path.is_absolute():
            snapshot_path = Path(config.get("_config_dir", ".")) / snapshot_path
        topology = HardwareTopology.from_target_snapshot(snapshot_path.resolve())
    else:
        topology = HardwareTopology.rectangular_grid(
            hardware_config["rows"],
            hardware_config["cols"],
            hardware_config.get("module_rows", 0),
            hardware_config.get("module_cols", 0),
            hardware_config.get("name"),
        )
    batch_size = int(acceleration.get("batch_size", 8192))
    backend_requested = str(acceleration.get("backend", "auto"))
    cpu_threads = int(acceleration.get("cpu_threads", 0))
    cuda_min_batch = int(acceleration.get("cuda_min_batch", 2048))

    manager.write(
        "Acceleration: "
        + json.dumps(
            acceleration_diagnostics()
            | {
                "requested": backend_requested,
                "batch_size": batch_size,
                "cpu_threads": cpu_threads,
            },
            sort_keys=True,
        )
    )

    records: list[dict] = []
    code_representatives: list[dict] = []
    all_spaces_complete = True
    declared_space_size = 0
    backend_counts: dict[str, int] = {}
    dedup = IsomorphismRegistry()

    family_bar = manager.bar(
        enumerate(search["families"]),
        total=len(search["families"]),
        desc="CodeForge: code families",
        unit="family",
        leave=True,
    )
    for family_index, family in family_bar:
        l, m = int(family["l"]), int(family["m"])
        weight_a, weight_b = int(family["weight_a"]), int(family["weight_b"])
        max_candidates = int(family["max_candidates"])
        iterator, complete, total = enumerate_specs(
            l, m, weight_a, weight_b, max_candidates
        )
        if not complete and family.get("mode", "auto") != "exhaustive":
            iterator = random_specs(
                l,
                m,
                weight_a,
                weight_b,
                max_candidates,
                seed=config["seed"] + family_index * 1009,
            )
        all_spaces_complete &= complete
        declared_space_size += total
        evaluated_target = total if complete else max_candidates
        candidate_bar = manager.bar(
            total=evaluated_target,
            desc=(
                f"Family {family_index + 1}/{len(search['families'])} "
                f"QC-CSS {l}x{m} n={2*l*m}"
            ),
            unit="candidate",
            leave=manager.leave_nested,
        )
        counters = {
            "rank_or_k_reject": 0,
            "distance_reject": 0,
            "isomorphic_duplicate": 0,
            "code_representative": 0,
        }
        family_seen = 0
        for batch in _batches(iter(iterator), batch_size):
            support_a, support_b = _support_arrays(batch)
            screened, status = screen_qc_batch(
                support_a,
                support_b,
                l=l,
                m=m,
                min_dx=int(constraints["min_d_x"]),
                min_dz=int(constraints["min_d_z"]),
                min_k=int(constraints["min_k"]),
                backend=backend_requested,
                threads=cpu_threads,
                cuda_min_batch=cuda_min_batch,
            )
            backend_counts[status.selected] = (
                backend_counts.get(status.selected, 0) + len(batch)
            )
            for local_index, spec in enumerate(batch):
                flags = int(screened["flags"][local_index])
                rank_x = int(screened["rank_x"][local_index])
                rank_z = int(screened["rank_z"][local_index])
                logical = int(screened["k"][local_index])
                code_record = _prefilter_code_record(
                    spec=spec,
                    rank_x=rank_x,
                    rank_z=rank_z,
                    logical=logical,
                    flags=flags,
                    witness_x=int(screened["witness_x"][local_index]),
                    witness_z=int(screened["witness_z"][local_index]),
                    min_dx=int(constraints["min_d_x"]),
                    min_dz=int(constraints["min_d_z"]),
                    backend=status.selected,
                )
                record = {
                    "family": spec.to_dict(),
                    "code": code_record,
                    "screen_backend": status.selected,
                    "accepted": False,
                    "stage": "code_screen",
                }
                if not (flags & 0x01):
                    counters["rank_or_k_reject"] += 1
                    records.append(record)
                    continue
                if (flags & 0x06) != 0x06:
                    counters["distance_reject"] += 1
                    records.append(record)
                    continue
                if spec.n > topology.graph.number_of_nodes():
                    record["reason"] = "proxy_topology_too_small"
                    records.append(record)
                    continue

                h_x, h_z = build_bicycle_css(spec)
                code_record["row_weight_x_max"] = int(
                    h_x.sum(axis=1).max(initial=0)
                )
                code_record["row_weight_z_max"] = int(
                    h_z.sum(axis=1).max(initial=0)
                )
                code_edges = interaction_edges_from_checks(h_x, h_z)
                code_id = candidate_id({"family": spec.to_dict()})
                decision = dedup.register(
                    candidate_id=code_id,
                    spec=spec,
                    graph=colored_interaction_graph(spec.n, code_edges),
                )
                record["deduplication"] = {
                    "duplicate": decision.duplicate,
                    "representative_id": decision.representative_id,
                    "wl_hash": decision.wl_hash,
                    "reason": decision.reason,
                    "exact_isomorphism_checked": decision.exact_isomorphism_checked,
                }
                if decision.duplicate:
                    counters["isomorphic_duplicate"] += 1
                    records.append(record)
                    continue
                code_score = float(
                    100.0
                    * (code_record["d_x_at_least"] + code_record["d_z_at_least"])
                    + 10.0 * logical
                    - 0.1
                    * (
                        code_record["row_weight_x_max"]
                        + code_record["row_weight_z_max"]
                    )
                    - 0.001 * spec.n
                )
                code_representatives.append(
                    {
                        "code_id": code_id,
                        "spec": spec,
                        "h_x": h_x,
                        "h_z": h_z,
                        "rank_x": rank_x,
                        "rank_z": rank_z,
                        "logical": logical,
                        "code_record": code_record,
                        "deduplication": record["deduplication"],
                        "code_score": code_score,
                    }
                )
                counters["code_representative"] += 1
                record.update(
                    {
                        "accepted": True,
                        "stage": "code_representative",
                        "code_id": code_id,
                        "code_score": code_score,
                    }
                )
                records.append(record)

            family_seen += len(batch)
            candidate_bar.update(len(batch))
            candidate_bar.set_postfix(counters | {"backend": status.selected}, refresh=False)
        candidate_bar.close()
        family_bar.set_postfix(
            evaluated=len(records),
            unique=dedup.unique,
            backend=",".join(sorted(backend_counts)),
            refresh=False,
        )

    # Preserve every requested size while limiting expensive schedule searches.
    codes_per_size = int(schedule_settings.get("codes_per_size", 3))
    max_codes = int(schedule_settings.get("max_code_representatives", 24))
    grouped: dict[int, list[dict]] = {}
    for item in code_representatives:
        grouped.setdefault(item["spec"].n, []).append(item)
    preselected: list[dict] = []
    for n in sorted(grouped):
        preselected.extend(
            sorted(grouped[n], key=lambda item: item["code_score"], reverse=True)[
                :codes_per_size
            ]
        )
    preselected = sorted(
        preselected, key=lambda item: (item["spec"].n, -item["code_score"])
    )[:max_codes]

    schedule_root = output / "schedule_search"
    schedule_root.mkdir(parents=True, exist_ok=True)
    accepted: list[Candidate] = []
    schedule_summaries: list[dict] = []
    schedule_bar = manager.bar(
        preselected,
        total=len(preselected),
        desc="GapSearch: adversarial schedules",
        unit="code",
        leave=True,
    )
    for code_index, item in enumerate(schedule_bar):
        finalists, report = search_adversarial_schedules(
            spec=item["spec"],
            h_x=item["h_x"],
            h_z=item["h_z"],
            config=config,
            topology=topology,
            seed=config["seed"] + code_index * 1_000_003,
            progress=manager,
        )
        report["code_id"] = item["code_id"]
        report_path = schedule_root / f"{item['code_id']}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        schedule_summaries.append(
            {
                "code_id": item["code_id"],
                "n": item["spec"].n,
                "status": report.get("status"),
                "best": report.get("best"),
                "report": str(report_path),
            }
        )
        for finalist in finalists:
            circuit_spec = finalist["circuit_spec"]
            hardware = finalist["hardware"]
            hardness = finalist["hardness"]
            candidate_identifier = candidate_id(
                {
                    "code_id": item["code_id"],
                    "schedule_id": circuit_spec["schedule_id"],
                }
            )
            candidate = Candidate(
                candidate_id=candidate_identifier,
                family=item["spec"].to_dict(),
                code=CodeMetrics(
                    n=item["code_record"]["n"],
                    k=item["logical"],
                    rank_x=item["rank_x"],
                    rank_z=item["rank_z"],
                    d_x=None,
                    d_z=None,
                    d_x_at_least=item["code_record"]["d_x_at_least"],
                    d_z_at_least=item["code_record"]["d_z_at_least"],
                    commutes=True,
                    row_weight_x_max=item["code_record"]["row_weight_x_max"],
                    row_weight_z_max=item["code_record"]["row_weight_z_max"],
                ),
                hardware=HardwareMetrics(
                    two_qubit_count=int(hardware["two_qubit_count"]),
                    two_qubit_depth=int(hardware["two_qubit_depth"]),
                    swap_count=int(hardware["swap_count"]),
                    routing_distance=int(hardware["routing_distance"]),
                    nonlocal_edges=int(hardware["nonlocal_edges"]),
                    crossing_edges=int(hardware["crossing_edges"]),
                    layout=tuple(hardware["layout"]),
                ),
                hardness=_hardness_dataclass(hardness),
                exact_artifacts={
                    "d_x_certificate": item["code_record"]["d_x_certificate"],
                    "d_z_certificate": item["code_record"]["d_z_certificate"],
                    "h_x": item["h_x"].tolist(),
                    "h_z": item["h_z"].tolist(),
                    "logical_edges": circuit_spec["union_two_qubit_edges"],
                    "circuit_spec": circuit_spec,
                    "deduplication": item["deduplication"],
                    "schedule_search": {
                        "report_path": str(report_path),
                        "diversity": finalist["diversity"],
                        "cutwidth": finalist["cutwidth"],
                        "cotengra_short": finalist["cotengra_short"],
                        "cotengra_deep": finalist["cotengra_deep"],
                        "cotengra_adversary": finalist.get("cotengra_adversary"),
                        "verifier_preflight": finalist.get("verifier_preflight"),
                        "lightcone_preflight": finalist.get("verifier_preflight"),
                        "target_checks": finalist["target_checks"],
                        "target_pass": finalist["target_pass"],
                        "matching_source": finalist.get(
                            "matching_source",
                            circuit_spec.get("schedule_metadata", {}).get(
                                "matching_source"
                            ),
                        ),
                        "relation_mode": circuit_spec[
                            "relation_preservation"
                        ].get("relation_mode", "fixed_css_automorphism"),
                        "pinned_layout": list(hardware["layout"]),
                        "target_structural_fingerprint": hardware.get(
                            "target_structural_fingerprint"
                        ),
                    },
                },
            )
            candidate.objective = float(finalist["objective"])
            accepted.append(candidate)
        schedule_bar.set_postfix(
            n=item["spec"].n,
            status=report.get("status"),
            accepted=len(accepted),
            refresh=False,
        )
    schedule_bar.close()

    frontier = pareto_front(accepted, progress=manager)
    frontier.sort(key=lambda candidate: candidate.objective, reverse=True)
    complete_hash = sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    quotient = dedup.diagnostics()
    target_pass_count = sum(
        bool(
            candidate.exact_artifacts.get("schedule_search", {}).get(
                "target_pass", False
            )
        )
        for candidate in accepted
    )
    certificate = {
        "schema": "codegap.codeforge-optimality.v4-adversarial-schedules",
        "search_complete": False,
        "claim": "ADVERSARIAL_SCHEDULE_SEARCH_WITHIN_EVALUATED_QUOTIENT",
        "code_space_complete": all_spaces_complete,
        "declared_code_space_size": declared_space_size,
        "evaluated_code_records": len(records),
        "unique_code_representatives": len(code_representatives),
        "schedule_searched_code_representatives": len(preselected),
        "accepted_schedule_finalists": len(accepted),
        "target_passing_schedules": target_pass_count,
        "pareto_candidates": len(frontier),
        "records_sha256": complete_hash,
        "constraints": constraints,
        "schedule_search": schedule_settings,
        "schedule_summaries": schedule_summaries,
        "deduplication": quotient,
        "acceleration": {
            "configuration": acceleration,
            "backend_candidate_counts": backend_counts,
            "diagnostics": acceleration_diagnostics(),
        },
        "gate_a_boundary": (
            "Schedule search uses a signed live target snapshot for exact "
            "zero-SWAP embeddability. Gate A still cannot PASS until fresh "
            "BackendV2.target compilation validates zero SWAP, "
            "native two-qubit instructions and the measurement map."
        ),
        "hardness_boundary": (
            "Schedules maximize the best Cotengra cost found under registered "
            "short/deep budgets. Gate C still uses the minimum across all "
            "registered classical attacks and is not an unconditional lower bound."
        ),
    }
    frontier_payload = [candidate.to_dict() for candidate in frontier]
    (output / "hardware_code_frontier.json").write_text(
        json.dumps(frontier_payload, indent=2), encoding="utf-8"
    )
    (output / "deduplication_report.json").write_text(
        json.dumps(quotient, indent=2), encoding="utf-8"
    )
    (output / "schedule_search_summary.json").write_text(
        json.dumps(schedule_summaries, indent=2), encoding="utf-8"
    )
    families_root = output / "code_families"
    families_root.mkdir(parents=True, exist_ok=True)
    for candidate in frontier:
        (families_root / f"{candidate.candidate_id}.json").write_text(
            json.dumps(
                {
                    "schema": "codegap.code-family.v4-adversarial-schedule",
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "code": asdict(candidate.code),
                    "h_x": candidate.exact_artifacts["h_x"],
                    "h_z": candidate.exact_artifacts["h_z"],
                    "circuit_spec": candidate.exact_artifacts["circuit_spec"],
                    "schedule_search": candidate.exact_artifacts[
                        "schedule_search"
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    (output / "family_definition.md").write_text(
        (
            "# Bicycle-family and adversarial schedule validity\n\n"
            "Codes are quotiented by declared QC equivalences and exact colored "
            "graph isomorphism. Schedules use either fixed CSS automorphisms or "
            "target-native perfect matchings with an exactly tracked dynamic CSS "
            "wire frame. The final verifier is validated on the actual circuit "
            "through exact backward light cones. Schedule difficulty is evaluated "
            "on the full doubled space-time "
            "tensor network, with Cotengra acting as a minimizing adversary.\n"
        ),
        encoding="utf-8",
    )
    (output / "optimality_certificate.json").write_text(
        json.dumps(certificate, indent=2), encoding="utf-8"
    )
    (output / "evaluated_candidates.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return frontier, certificate

