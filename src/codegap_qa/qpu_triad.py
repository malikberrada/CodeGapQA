from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import secrets
import shutil
from typing import Any, Iterable

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import beta

from .codesign import circuit_qasm3
from .freeze import freeze, verify_freeze
from .qpu_compile import compile_candidate
from .qpu_counts import counts_to_samples
from .qpu_provider import OpenQuantumProvider
from .qpu_snapshot import snapshot_target, write_snapshot


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _dedupe(values: Iterable[str | None]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        token = str(value).strip()
        if token and token not in seen:
            seen.add(token)
            output.append(token)
    return output


def _candidate_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["candidate_id"]): item for item in records}


def select_qpu_panel(
    *,
    records: list[dict[str, Any]],
    recommended: dict[str, Any],
    champions: list[dict[str, Any]],
    minimal: dict[str, Any],
    maximum_candidates: int,
) -> list[dict[str, Any]]:
    if maximum_candidates < 2:
        raise ValueError("maximum_candidates must be at least two.")
    by_id = _candidate_by_id(records)

    # Family champions are mandatory for the cross-family question.
    family_ids = [str(item["candidate_id"]) for item in champions]
    minimal_ids = [
        str(item["candidate_id"])
        for item in minimal.get("targets", [])
        if item.get("found") and item.get("candidate_id")
    ]
    ordered = _dedupe(
        family_ids
        + [recommended.get("smallest_qubit_qualified")]
        + minimal_ids
        + [
            recommended.get("largest_gap"),
            recommended.get("most_noise_robust"),
            recommended.get("best_overall"),
        ]
    )

    selected: list[dict[str, Any]] = []
    for candidate_id in ordered:
        item = by_id.get(candidate_id)
        if item is not None:
            selected.append(item)
        if len(selected) >= maximum_candidates:
            break

    # Guarantee size diversity when possible without dropping a family.
    if len(selected) < maximum_candidates:
        remaining = [item for item in records if item not in selected]
        remaining.sort(
            key=lambda item: (
                int(item["code"]["n"]),
                -float(item["hardness"]["gamma_log10"]),
            )
        )
        for item in remaining:
            selected.append(item)
            if len(selected) >= maximum_candidates:
                break

    families = {str(item["family"]["type"]) for item in selected}
    available_families = {str(item["family"]["type"]) for item in records}
    if len(available_families) >= 2 and len(families) < 2:
        raise RuntimeError("The selected QPU panel does not cover two code families.")
    return selected


def fold_circuit_spec(circuit_spec: dict[str, Any], scale: int) -> dict[str, Any]:
    """Apply local unitary folding while preventing compiler cancellation.

    # CODEGAP_V101_FOLD_BARRIERS

    Every logical RXX/RZZ gate G is replaced by G (G^-1 G)^k. Barriers on
    the same qubit pair separate the three factors, so optimization-level 3
    cannot algebraically collapse the registered hardware-stress sequence.
    """

    scale = int(scale)
    if scale < 1 or scale % 2 == 0:
        raise ValueError("Fold scale must be a positive odd integer.")

    output = deepcopy(circuit_spec)
    base_foldable = sum(
        1
        for gate in circuit_spec["gates"]
        if str(gate["name"]) in {"rxx", "rzz"}
    )
    output["foldable_two_qubit_gates"] = int(base_foldable)
    output["expected_folded_two_qubit_gates"] = int(base_foldable * scale)

    if scale == 1:
        output["noise_fold_scale"] = 1
        output["noise_fold_semantics"] = "unmodified"
        return output

    folded: list[dict[str, Any]] = []
    for gate in circuit_spec["gates"]:
        gate_copy = deepcopy(gate)
        name = str(gate_copy["name"])
        if name not in {"rxx", "rzz"}:
            folded.append(gate_copy)
            continue

        qubits = [int(value) for value in gate_copy["qubits"]]
        angle = float(gate_copy.get("angle") or 0.0)
        folded.append(gate_copy)

        for pair_index in range((scale - 1) // 2):
            folded.append(
                {
                    "name": "barrier",
                    "qubits": qubits,
                    "fold_barrier": True,
                    "fold_pair_index": pair_index,
                }
            )
            inverse = deepcopy(gate_copy)
            inverse["angle"] = -angle
            folded.append(inverse)
            folded.append(
                {
                    "name": "barrier",
                    "qubits": qubits,
                    "fold_barrier": True,
                    "fold_pair_index": pair_index,
                }
            )
            folded.append(deepcopy(gate_copy))

    output["gates"] = folded
    output["two_qubit_count"] = int(base_foldable * scale)
    output["logical_two_qubit_depth"] = int(
        circuit_spec.get("logical_two_qubit_depth", 0)
    ) * scale
    output["schedule_id"] = (
        f"{circuit_spec.get('schedule_id', 'schedule')}-fold{scale}"
    )
    output["noise_fold_scale"] = scale
    output["noise_fold_semantics"] = (
        "Barrier-preserved local unitary folding G(G^-1 G)^k on every "
        "logical RXX/RZZ gate. The ideal circuit is unchanged while native "
        "two-qubit exposure must increase after compilation."
    )
    return output


def _adapt_candidate(record: dict[str, Any], circuit_spec: dict[str, Any]) -> dict[str, Any]:
    candidate = deepcopy(record)
    pinned = (
        circuit_spec.get("schedule_metadata", {}).get("pinned_layout")
        or record.get("hardware", {}).get("layout")
    )
    candidate["exact_artifacts"] = {
        "circuit_spec": circuit_spec,
        "schedule_search": {
            "pinned_layout": pinned,
            "verifier_preflight": record["verifier_preflight"],
            "target_pass": bool(
                int(record["hardware"].get("swap_count", 1)) == 0
                and int(record["hardware"].get("nonlocal_edges", 1)) == 0
            ),
            "target_checks": {
                "relation_certificate": bool(
                    circuit_spec.get("relation_preservation", {}).get("passed", True)
                ),
                "proxy_zero_swap": int(record["hardware"].get("swap_count", 1)) == 0,
            },
        },
    }
    candidate["objective"] = float(record.get("triad_score", 0.0))
    return candidate


def _active_features(record: dict[str, Any]) -> list[dict[str, Any]]:
    preflight = record["verifier_preflight"]
    masks = preflight["selected_masks"]
    expectations = preflight["selected_expectations"]
    witness = preflight["witness"]
    weights = witness["weights"]
    names = witness["feature_names"]
    active: list[dict[str, Any]] = []
    for index, weight in enumerate(weights):
        value = float(weight)
        if abs(value) <= 1.0e-12:
            continue
        support = [position for position, bit in enumerate(masks[index]) if int(bit)]
        if not support:
            raise RuntimeError("An active witness feature has empty support.")
        active.append(
            {
                "feature_index": index,
                "feature_name": str(names[index]),
                "support": support,
                "weight": value,
                "ideal_expectation": float(expectations[index]),
            }
        )
    if not active:
        raise RuntimeError("The QPU panel candidate has no active witness feature.")
    return active


def _calibration_qasm(
    *,
    physical_qubits: list[int],
    pattern: list[int],
    nonce: str,
    label: str,
) -> str:
    if len(physical_qubits) != len(pattern):
        raise ValueError("Calibration pattern width mismatch.")
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"bit[{len(pattern)}] c;",
        f"// CODEGAP_QPU_TRIAD_NONCE {nonce} {label}",
    ]
    # Symmetric SPAM anchors ensure physical-qubit declarations are accepted by
    # the Open Quantum precompiler even for the all-zero pattern.
    for physical in physical_qubits:
        lines.append(f"x ${physical};")
        lines.append(f"x ${physical};")
    for physical, bit in zip(physical_qubits, pattern):
        if int(bit):
            lines.append(f"x ${physical};")
    for position, physical in enumerate(physical_qubits):
        lines.append(f"c[{position}] = measure ${physical};")
    return "\n".join(lines) + "\n"


def _append_nonce(path: Path, marker: str) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    text += f"// CODEGAP_QPU_TRIAD_NONCE {marker}\n"
    path.write_text(text, encoding="utf-8")


def build_candidate_protocol(
    *,
    record: dict[str, Any],
    layout: list[int],
    settings: dict[str, Any],
) -> dict[str, Any]:
    active = _active_features(record)
    logical_support = sorted({q for feature in active for q in feature["support"]})
    maximum = int(settings["maximum_calibration_qubits"])
    if len(logical_support) > maximum:
        raise RuntimeError(
            f"Candidate {record['candidate_id']} needs {len(logical_support)} calibration "
            f"qubits, above the registered maximum {maximum}."
        )
    if max(logical_support, default=-1) >= len(layout):
        raise RuntimeError("Witness support is outside the compiled layout.")
    physical_support = [int(layout[index]) for index in logical_support]
    adversary_means = record["verifier_preflight"]["witness"].get(
        "adversary_means", {}
    )
    adversary_supremum = max(
        (float(value) for value in adversary_means.values()),
        default=0.0,
    )
    return {
        "schema": "codegap.qpu-triad-candidate-protocol.v1",
        "candidate_id": str(record["candidate_id"]),
        "family": record["family"],
        "code": record["code"],
        "hardware_proxy": record["hardware"],
        "hardness": record["hardness"],
        "n": int(record["code"]["n"]),
        "calibration": {
            "logical_qubits": logical_support,
            "physical_qubits": physical_support,
            "patterns": {
                "all0": [0] * len(logical_support),
                "all1": [1] * len(logical_support),
                "even": [1 if index % 2 == 0 else 0 for index in range(len(logical_support))],
                "odd": [0 if index % 2 == 0 else 1 for index in range(len(logical_support))],
            },
            "model": "independent single-qubit assignment/SPAM channels with pre/post drift hull",
        },
        "witness": {
            "features": active,
            "l1_norm": float(sum(abs(item["weight"]) for item in active)),
            "adversary_supremum": adversary_supremum,
            "adversary_generalization_penalty": float(
                settings["adversary_generalization_penalty"]
            ),
            "ideal_mean": float(
                sum(item["weight"] * item["ideal_expectation"] for item in active)
            ),
        },
        "thresholds": {
            "minimum_assignment_determinant": float(
                settings["minimum_assignment_determinant"]
            ),
            "minimum_convention_accuracy": float(
                settings["minimum_convention_accuracy"]
            ),
            "minimum_convention_margin": float(
                settings["minimum_convention_margin"]
            ),
            "minimum_margin_lcb": float(settings["minimum_margin_lcb"]),
        },
        "confidence": {
            "alpha_total": float(settings["alpha_total"]),
            "alpha_calibration_total": float(settings["alpha_calibration_total"]),
            "alpha_science_total": float(settings["alpha_science_total"]),
            "optimization_seed": int(settings["optimization_seed"]),
            "optimization_safety_epsilon": float(
                settings["optimization_safety_epsilon"]
            ),
        },
        "claim_boundary": (
            "QPU confirmation is simultaneous only over the frozen candidate panel, "
            "fold scales, witness family, assignment/SPAM model, and registered classical attacks. "
            "It does not prove global code-family optimality or full-output-distribution TV closeness."
        ),
    }


def prepare_qpu_triad_panel(
    *,
    deep_artifact: Path,
    config_path: Path,
    protocol_path: Path | None,
    credentials: Path | None,
    backend_name: str,
    output: Path,
    maximum_candidates: int,
    fold_scales: list[int],
    maximum_layouts: int,
) -> dict[str, Any]:
    deep_artifact = deep_artifact.resolve()
    config_path = config_path.resolve()
    protocol_path = protocol_path.resolve() if protocol_path is not None else None
    source_protocol_path = protocol_path
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    if not verify_freeze(deep_artifact / "freeze_manifest.json")["ok"]:
        raise RuntimeError("The V091 deep artifact freeze is invalid.")
    config = read_json(config_path)
    settings = dict(config.get("qpu_triad", {}))
    if protocol_path is not None:
        protocol_payload = read_json(protocol_path)
        settings.update(protocol_payload.get("qpu_triad", protocol_payload))
    required_settings = {
        "maximum_calibration_qubits": 6,
        "calibration_shots": 4000,
        "science_shots": 10000,
        "minimum_assignment_determinant": 0.75,
        "minimum_convention_accuracy": 0.9,
        "minimum_convention_margin": 0.2,
        "minimum_margin_lcb": 0.0,
        "adversary_generalization_penalty": 0.02,
        "alpha_total": 0.01,
        "alpha_calibration_total": 0.003,
        "alpha_science_total": 0.007,
        "optimization_seed": 20260725,
        "optimization_safety_epsilon": 0.0001,
        "gamma_target": 6.0,
    }
    settings = required_settings | settings
    scales = sorted({int(value) for value in fold_scales})
    if not scales or scales[0] != 1 or any(value % 2 == 0 or value < 1 for value in scales):
        raise ValueError("Fold scales must be positive odd integers and include 1.")

    records = read_json(deep_artifact / "circuit_records.json")
    recommended = read_json(deep_artifact / "recommended_experiments.json")
    champions = read_json(deep_artifact / "family_champions.json")
    minimal = read_json(deep_artifact / "minimal_qubit_gap_certificate.json")
    panel = select_qpu_panel(
        records=records,
        recommended=recommended,
        champions=champions,
        minimal=minimal,
        maximum_candidates=maximum_candidates,
    )
    output.mkdir(parents=True, exist_ok=True)
    nonce = utc_now().replace(":", "").replace("+00:00", "Z") + "-" + secrets.token_hex(6)

    with OpenQuantumProvider(credentials) as provider:
        metadata = provider.backend_metadata(backend_name)
        if metadata.get("accepting_jobs") is False:
            raise RuntimeError(f"Backend {backend_name!r} is not accepting jobs.")
        backend = provider.backend(backend_name, "public", "standard")
        snapshot = snapshot_target(
            backend,
            backend_name=backend_name,
            accepting_jobs=metadata.get("accepting_jobs"),
            queue_depth=metadata.get("queue_depth"),
        )
        snapshot_payload = write_snapshot(snapshot, output / "target_snapshot.json")
        expected_fingerprint = read_json(deep_artifact / "research_triad_report.json").get(
            "hardware", {}
        ).get("structural_fingerprint")
        if expected_fingerprint and expected_fingerprint != snapshot_payload.get(
            "structural_fingerprint"
        ):
            raise RuntimeError(
                "The live backend structure differs from the frozen V091 search target."
            )

        candidate_payloads: list[dict[str, Any]] = []
        calibration_pre: list[dict[str, Any]] = []
        science: list[dict[str, Any]] = []
        calibration_post: list[dict[str, Any]] = []

        for candidate_index, record in enumerate(panel):
            candidate_id = str(record["candidate_id"])
            candidate_root = output / "candidates" / candidate_id
            candidate_root.mkdir(parents=True, exist_ok=True)
            compiled: dict[str, Any] = {}
            base_layout: list[int] | None = None

            for scale in scales:
                spec = fold_circuit_spec(record["circuit_spec"], scale)
                adapted = _adapt_candidate(record, spec)
                scale_root = candidate_root / f"fold_{scale}"
                report = compile_candidate(
                    adapted,
                    backend,
                    scale_root / "layouts",
                    max_layouts=maximum_layouts,
                    seed=int(config.get("seed", 0))
                    + candidate_index * 100003
                    + scale * 1009,
                )
                write_json(scale_root / "compile_report.json", report)
                if report.get("status") != "PASS":
                    raise RuntimeError(
                        f"Candidate {candidate_id} fold {scale} failed live target compilation: "
                        f"{report.get('status')}"
                    )
                qasm_path = scale_root / "science.qasm3"
                shutil.copy2(report["best"]["qasm_path"], qasm_path)
                _append_nonce(qasm_path, f"{nonce} {candidate_id} SCIENCE FOLD {scale}")
                layout = [int(value) for value in report["best"]["layout"]]
                if base_layout is None:
                    base_layout = layout
                elif layout != base_layout:
                    raise RuntimeError(
                        f"Candidate {candidate_id} uses different layouts across fold scales."
                    )
                label = f"{candidate_id}::science_fold_{scale}"
                science.append(
                    {
                        "candidate_id": candidate_id,
                        "kind": "science",
                        "fold_scale": scale,
                        "label": label,
                        "qasm_path": str(qasm_path.relative_to(output)),
                        "sha256": file_sha256(qasm_path),
                        "classical_bits": int(record["code"]["n"]),
                    }
                )
                compiled[str(scale)] = {
                    "compile_report": str((scale_root / "compile_report.json").relative_to(output)),
                    "science_qasm": str(qasm_path.relative_to(output)),
                    "qasm_sha256": file_sha256(qasm_path),
                    "layout": layout,
                    "two_qubit_count": int(report["best"]["two_qubit_count"]),
                    "two_qubit_depth": int(report["best"]["two_qubit_depth"]),
                }

            fold_counts = {
                int(scale): int(compiled[str(scale)]["two_qubit_count"])
                for scale in scales
            }
            base_twoq = int(fold_counts[1])
            if base_twoq <= 0:
                raise RuntimeError(
                    f"Candidate {candidate_id} has no compiled two-qubit gates."
                )
            previous_count = 0
            for scale in scales:
                count = int(fold_counts[scale])
                if count <= previous_count:
                    raise RuntimeError(
                        f"Candidate {candidate_id} fold integrity failed: "
                        f"scale {scale} has {count} native two-qubit gates "
                        f"after {previous_count}. Compiler cancellation is suspected."
                    )
                if scale > 1 and count <= base_twoq:
                    raise RuntimeError(
                        f"Candidate {candidate_id} fold {scale} did not increase "
                        "native two-qubit exposure. Refusing QPU panel creation."
                    )
                previous_count = count
            fold_integrity = {
                "status": "PASS",
                "method": "barrier_preserved_logical_local_unitary_folding",
                "native_two_qubit_counts": {
                    str(scale): int(fold_counts[scale])
                    for scale in scales
                },
                "native_two_qubit_amplification": {
                    str(scale): float(fold_counts[scale] / base_twoq)
                    for scale in scales
                },
                "strictly_increasing": True,
                "compiler_cancellation_detected": False,
            }

            assert base_layout is not None
            protocol = build_candidate_protocol(
                record=record,
                layout=base_layout,
                settings=settings,
            )
            candidate_protocol_path = candidate_root / "qpu_triad_protocol.json"
            write_json(candidate_protocol_path, protocol)
            logical_support = protocol["calibration"]["logical_qubits"]
            physical_support = protocol["calibration"]["physical_qubits"]
            patterns = protocol["calibration"]["patterns"]
            for phase, target in (
                ("pre", calibration_pre),
                ("post", calibration_post),
            ):
                for pattern_name, pattern in patterns.items():
                    label = f"{candidate_id}::cal_{phase}_{pattern_name}"
                    path = candidate_root / "calibration" / phase / f"{pattern_name}.qasm3"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        _calibration_qasm(
                            physical_qubits=physical_support,
                            pattern=[int(value) for value in pattern],
                            nonce=nonce,
                            label=label,
                        ),
                        encoding="utf-8",
                    )
                    target.append(
                        {
                            "candidate_id": candidate_id,
                            "kind": f"calibration_{phase}",
                            "pattern": pattern_name,
                            "label": label,
                            "qasm_path": str(path.relative_to(output)),
                            "sha256": file_sha256(path),
                            "classical_bits": len(logical_support),
                        }
                    )

            candidate_payloads.append(
                {
                    "candidate_id": candidate_id,
                    "roles": [
                        key
                        for key, value in recommended.items()
                        if str(value or "") == candidate_id
                    ]
                    + [
                        f"family_champion:{item['family_type']}"
                        for item in champions
                        if str(item["candidate_id"]) == candidate_id
                    ],
                    "family": record["family"],
                    "code": record["code"],
                    "hardness": record["hardness"],
                    "triad_score": record["triad_score"],
                    "protocol_path": str(candidate_protocol_path.relative_to(output)),
                    "compiled": compiled,
                    "fold_integrity": fold_integrity,
                }
            )

    manifest = {
        "schema": "codegap.qpu-triad-panel.v1",
        "created_at": utc_now(),
        "status": "BOUND_BEFORE_QPU_SUBMISSION",
        "nonce": nonce,
        "backend": backend_name,
        "source_deep_artifact": str(deep_artifact),
        "source_deep_freeze_sha256": file_sha256(
            deep_artifact / "freeze_manifest.json"
        ),
        "source_config": str(config_path),
        "source_config_sha256": file_sha256(config_path),
        "source_qpu_protocol": str(source_protocol_path) if source_protocol_path is not None else None,
        "source_qpu_protocol_sha256": (
            file_sha256(source_protocol_path) if source_protocol_path is not None else None
        ),
        "fold_scales": scales,
        "fold_integrity_required": True,
        "fold_implementation": (
            "barrier_preserved_logical_local_unitary_folding"
        ),
        "candidate_count": len(candidate_payloads),
        "candidates": candidate_payloads,
        "bundles": {
            "calibration_pre": {
                "shots": int(settings["calibration_shots"]),
                "circuits": calibration_pre,
            },
            "science": {
                "shots": int(settings["science_shots"]),
                "circuits": science,
            },
            "calibration_post": {
                "shots": int(settings["calibration_shots"]),
                "circuits": calibration_post,
            },
        },
        "registered_analysis": {
            "simultaneous_alpha_total": float(settings["alpha_total"]),
            "alpha_calibration_total": float(settings["alpha_calibration_total"]),
            "alpha_science_total": float(settings["alpha_science_total"]),
            "ranking_rule": [
                "base_fold_pass",
                "number_of_fold_scales_passed",
                "minimum_margin_lcb_across_scales",
                "critical_fold_scale",
                "gamma_log10",
                "smaller_n",
            ],
            "gamma_target": float(settings["gamma_target"]),
            "robustness_interpretation": (
                "Local unitary folding is an empirical hardware-stress axis, not an exact scalar-noise model."
            ),
        },
        "claim_boundaries": [
            "Optimal code family means best within this frozen QPU panel and registered ranking rule.",
            "Hardware robustness is measured on the live backend with fresh pre/post calibration and fold stress.",
            "Verification-versus-simulation gap remains conditional on the registered classical attack suite.",
            "No full-distribution total-variation certificate is claimed.",
        ],
    }
    write_json(output / "qpu_triad_panel.json", manifest)
    shutil.copy2(config_path, output / "source_config.json")
    if source_protocol_path is not None:
        shutil.copy2(source_protocol_path, output / "source_qpu_protocol.json")
    freeze(output)
    if not verify_freeze(output / "freeze_manifest.json")["ok"]:
        raise RuntimeError("QPU triad panel freeze verification failed.")
    return manifest


def submit_qpu_triad_bundle(
    *,
    panel_root: Path,
    bundle_name: str,
    credentials: Path | None,
    output: Path,
    max_active: int,
    slot_poll_seconds: int,
    slot_timeout_seconds: int,
) -> dict[str, Any]:
    panel_root = panel_root.resolve()
    if not verify_freeze(panel_root / "freeze_manifest.json")["ok"]:
        raise RuntimeError("QPU triad panel freeze is invalid.")
    manifest = read_json(panel_root / "qpu_triad_panel.json")
    if bundle_name not in manifest["bundles"]:
        raise ValueError(f"Unknown bundle {bundle_name!r}.")
    bundle = manifest["bundles"][bundle_name]
    circuits = bundle["circuits"]
    qasm_files = [panel_root / item["qasm_path"] for item in circuits]
    labels = [str(item["label"]) for item in circuits]
    with OpenQuantumProvider(credentials) as provider:
        result = provider.submit_bundle_async(
            backend_name=str(manifest["backend"]),
            qasm_files=qasm_files,
            labels=labels,
            shots=int(bundle["shots"]),
            output=output.resolve(),
            stage=f"qpu_triad_{bundle_name}",
            execution_plan="public",
            queue_priority="standard",
            max_active=max_active,
            slot_poll_seconds=slot_poll_seconds,
            slot_timeout_seconds=slot_timeout_seconds,
        )
    return result


def collect_qpu_triad_bundle(
    *,
    panel_root: Path,
    bundle_name: str,
    credentials: Path | None,
    run_root: Path,
    wait: bool,
    poll_seconds: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    panel_root = panel_root.resolve()
    run_root = run_root.resolve()
    if not verify_freeze(panel_root / "freeze_manifest.json")["ok"]:
        raise RuntimeError("QPU triad panel freeze is invalid.")
    manifest = read_json(panel_root / "qpu_triad_panel.json")
    expected_stage = f"qpu_triad_{bundle_name}"
    with OpenQuantumProvider(credentials) as provider:
        result = provider.collect_async_bundle(
            output=run_root,
            wait=wait,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
        )
    if result.get("status") == "RESULTS_COLLECTED":
        raw = read_json(run_root / "raw_counts.json")
        if raw.get("stage") != expected_stage:
            raise RuntimeError(
                f"Collected stage {raw.get('stage')!r}, expected {expected_stage!r}."
            )
        expected_count = len(manifest["bundles"][bundle_name]["circuits"])
        if len(raw.get("results", [])) != expected_count:
            raise RuntimeError("Collected result count does not match the frozen panel.")
        freeze(run_root)
        if not verify_freeze(run_root / "freeze_manifest.json")["ok"]:
            raise RuntimeError("Collected QPU triad run freeze is invalid.")
    return result


def _short_label(value: object) -> str:
    return str(value).split("::")[-1]


def _result_map(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["label"]): item for item in raw.get("results", [])}


def _pattern_accuracy(
    counts: dict[object, int], expected: list[int], convention: str
) -> float:
    target = np.asarray(expected, dtype=np.uint8)
    samples = counts_to_samples(counts, len(expected), convention)
    if samples.size == 0:
        return 0.0
    return float(np.mean(samples == target[None, :]))


def _choose_convention(
    *,
    pre: dict[str, dict[str, Any]],
    post: dict[str, dict[str, Any]],
    candidate_id: str,
    patterns: dict[str, list[int]],
) -> tuple[str, dict[str, float], float]:
    scores: dict[str, float] = {}
    for convention in ("qiskit", "left_to_right"):
        values: list[float] = []
        for phase_map, phase in ((pre, "pre"), (post, "post")):
            for name, expected in patterns.items():
                label = f"{candidate_id}::cal_{phase}_{name}"
                values.append(
                    _pattern_accuracy(phase_map[label]["counts"], expected, convention)
                )
        scores[convention] = float(np.mean(values))
    winner = max(
        ("qiskit", "left_to_right"),
        key=lambda name: (scores[name], name == "qiskit"),
    )
    loser = "left_to_right" if winner == "qiskit" else "qiskit"
    return winner, scores, float(scores[winner] - scores[loser])


def _cp_interval(errors: int, shots: int, alpha: float) -> tuple[float, float]:
    lower = 0.0 if errors == 0 else float(beta.ppf(alpha / 2, errors, shots - errors + 1))
    upper = 1.0 if errors == shots else float(
        beta.ppf(1 - alpha / 2, errors + 1, shots - errors)
    )
    return lower, upper


def _assignment_matrix(p01: float, p10: float) -> np.ndarray:
    return np.asarray([[1 - p01, p10], [p01, 1 - p10]], dtype=np.float64)


def _inverse_parity_coefficients(p01: list[float], p10: list[float]) -> np.ndarray:
    matrix = _assignment_matrix(p01[0], p10[0])
    for index in range(1, len(p01)):
        matrix = np.kron(matrix, _assignment_matrix(p01[index], p10[index]))
    parity = np.asarray(
        [1.0 if state.bit_count() % 2 == 0 else -1.0 for state in range(1 << len(p01))],
        dtype=np.float64,
    )
    return np.linalg.solve(matrix.T, parity)


def _empirical_bernstein_from_histogram(
    values: np.ndarray,
    counts: np.ndarray,
    alpha: float,
) -> tuple[float, float, float, float]:
    shots = int(counts.sum())
    if shots < 2:
        return float("-inf"), float("nan"), float("nan"), float("nan")
    mean = float(np.dot(values, counts) / shots)
    variance = float(np.dot((values - mean) ** 2, counts) / (shots - 1))
    bound = float(np.max(np.abs(values), initial=0.0))
    log_term = math.log(3.0 / alpha)
    radius = math.sqrt(2.0 * variance * log_term / shots)
    radius += 6.0 * bound * log_term / shots
    return mean - radius, mean, variance, bound


def _joint_histogram(
    *,
    counts: dict[object, int],
    n: int,
    convention: str,
    logical_support: list[int],
) -> np.ndarray:
    samples = counts_to_samples(counts, n, convention)
    local = samples[:, logical_support].astype(np.int64)
    powers = 1 << np.arange(len(logical_support) - 1, -1, -1, dtype=np.int64)
    states = local @ powers
    return np.bincount(states, minlength=1 << len(logical_support)).astype(np.float64)


def _feature_state_map(
    logical_support: list[int], feature_support: list[int]
) -> np.ndarray:
    positions = [logical_support.index(value) for value in feature_support]
    states = np.arange(1 << len(logical_support), dtype=np.int64)
    bits = ((states[:, None] >> np.arange(len(logical_support) - 1, -1, -1)) & 1)
    local = bits[:, positions]
    powers = 1 << np.arange(len(positions) - 1, -1, -1, dtype=np.int64)
    return local @ powers


def _witness_values_by_state(
    *,
    protocol: dict[str, Any],
    parameters: np.ndarray,
) -> np.ndarray:
    logical_support = [int(value) for value in protocol["calibration"]["logical_qubits"]]
    position = {value: index for index, value in enumerate(logical_support)}
    p01 = parameters[0::2]
    p10 = parameters[1::2]
    values = np.zeros(1 << len(logical_support), dtype=np.float64)
    for feature in protocol["witness"]["features"]:
        support = [int(value) for value in feature["support"]]
        local_positions = [position[value] for value in support]
        coefficients = _inverse_parity_coefficients(
            [float(p01[index]) for index in local_positions],
            [float(p10[index]) for index in local_positions],
        )
        state_map = _feature_state_map(logical_support, support)
        values += float(feature["weight"]) * coefficients[state_map]
    return values


def _calibration_box(
    *,
    pre_result: dict[str, dict[str, Any]],
    post_result: dict[str, dict[str, Any]],
    candidate_id: str,
    protocol: dict[str, Any],
    convention: str,
    alpha_per_parameter_period: float,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    width = len(protocol["calibration"]["logical_qubits"])
    period_details: dict[str, Any] = {}
    period_intervals: dict[str, list[tuple[float, float]]] = {}
    point_estimates: dict[str, list[float]] = {}
    for phase, result_map in (("pre", pre_result), ("post", post_result)):
        all0 = counts_to_samples(
            result_map[f"{candidate_id}::cal_{phase}_all0"]["counts"],
            width,
            convention,
        )
        all1 = counts_to_samples(
            result_map[f"{candidate_id}::cal_{phase}_all1"]["counts"],
            width,
            convention,
        )
        intervals: list[tuple[float, float]] = []
        points: list[float] = []
        details: list[dict[str, Any]] = []
        for index in range(width):
            k01 = int(all0[:, index].sum())
            k10 = int((1 - all1[:, index]).sum())
            i01 = _cp_interval(k01, len(all0), alpha_per_parameter_period)
            i10 = _cp_interval(k10, len(all1), alpha_per_parameter_period)
            intervals.extend([i01, i10])
            points.extend([k01 / len(all0), k10 / len(all1)])
            details.append(
                {
                    "position": index,
                    "p01_errors": k01,
                    "p10_errors": k10,
                    "shots": int(len(all0)),
                    "p01_interval": list(i01),
                    "p10_interval": list(i10),
                }
            )
        period_intervals[phase] = intervals
        point_estimates[phase] = points
        period_details[phase] = details
    hull = [
        (
            min(period_intervals["pre"][index][0], period_intervals["post"][index][0]),
            max(period_intervals["pre"][index][1], period_intervals["post"][index][1]),
        )
        for index in range(2 * width)
    ]
    drift = [
        abs(point_estimates["post"][index] - point_estimates["pre"][index])
        for index in range(2 * width)
    ]
    return hull, {
        "periods": period_details,
        "point_estimates": point_estimates,
        "maximum_parameter_drift": max(drift, default=0.0),
        "drift_vector": drift,
        "hull": [list(value) for value in hull],
    }


def _certificate_for_histogram(
    *,
    histogram: np.ndarray,
    protocol: dict[str, Any],
    calibration_box: list[tuple[float, float]],
    alpha_science: float,
    optimization_seed: int,
    safety_epsilon: float,
) -> dict[str, Any]:
    threshold_det = float(protocol["thresholds"]["minimum_assignment_determinant"])

    def evaluate(parameters: np.ndarray) -> tuple[float, float, float, float]:
        p01 = parameters[0::2]
        p10 = parameters[1::2]
        if np.any(1.0 - p01 - p10 < threshold_det):
            return -1.0e9, float("nan"), float("nan"), float("nan")
        try:
            values = _witness_values_by_state(protocol=protocol, parameters=parameters)
        except np.linalg.LinAlgError:
            return -1.0e9, float("nan"), float("nan"), float("nan")
        return _empirical_bernstein_from_histogram(values, histogram, alpha_science)

    point = np.asarray([(lower + upper) / 2 for lower, upper in calibration_box])
    point_lcb, point_mean, point_variance, point_bound = evaluate(point)
    dimension = len(calibration_box)
    corner_count = 1 << dimension
    worst_lcb = float("inf")
    worst_parameters: list[float] | None = None
    if dimension <= 12:
        for choices in product((0, 1), repeat=dimension):
            parameters = np.asarray(
                [calibration_box[index][choice] for index, choice in enumerate(choices)],
                dtype=np.float64,
            )
            lcb = evaluate(parameters)[0]
            if lcb < worst_lcb:
                worst_lcb = lcb
                worst_parameters = parameters.tolist()
    else:
        corner_count = 0
        rng = np.random.default_rng(optimization_seed)
        for _ in range(4096):
            parameters = np.asarray(
                [bounds[int(rng.integers(0, 2))] for bounds in calibration_box],
                dtype=np.float64,
            )
            lcb = evaluate(parameters)[0]
            if lcb < worst_lcb:
                worst_lcb = lcb
                worst_parameters = parameters.tolist()

    result = differential_evolution(
        lambda vector: evaluate(np.asarray(vector, dtype=np.float64))[0],
        calibration_box,
        seed=optimization_seed,
        popsize=16,
        maxiter=300,
        tol=1.0e-9,
        polish=True,
        workers=1,
        updating="immediate",
    )
    if float(result.fun) < worst_lcb:
        worst_lcb = float(result.fun)
        worst_parameters = [float(value) for value in result.x]
    worst_lcb -= safety_epsilon
    adversary = float(protocol["witness"]["adversary_supremum"])
    penalty = float(protocol["witness"]["adversary_generalization_penalty"])
    margin = worst_lcb - adversary - penalty
    threshold = float(protocol["thresholds"]["minimum_margin_lcb"])
    return {
        "point_estimate": {
            "mean": point_mean,
            "lcb": point_lcb,
            "variance": point_variance,
            "estimator_bound": point_bound,
        },
        "observed_lcb": worst_lcb,
        "adversary_supremum": adversary,
        "generalization_penalty": penalty,
        "margin_lcb": margin,
        "minimum_margin_lcb": threshold,
        "pass": bool(margin > threshold),
        "calibration_box": {
            "dimension": dimension,
            "corner_count": corner_count,
            "worst_parameters": worst_parameters,
            "continuous_optimization": {
                "success": bool(result.success),
                "message": str(result.message),
                "fun": float(result.fun),
                "x": [float(value) for value in result.x],
            },
            "safety_epsilon": safety_epsilon,
        },
    }


def _critical_fold_scale(scale_reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(scale_reports, key=lambda item: int(item["fold_scale"]))
    passes = [row for row in rows if row["certificate"]["pass"]]
    if not rows[0]["certificate"]["pass"]:
        return {"status": "BASE_FAIL", "lower_bound": 0.0, "estimate": None}
    failing = next((row for row in rows if not row["certificate"]["pass"]), None)
    if failing is None:
        return {
            "status": "AT_LEAST_MAX_TESTED",
            "lower_bound": float(rows[-1]["fold_scale"]),
            "estimate": None,
        }
    previous = max(
        (
            row
            for row in rows
            if int(row["fold_scale"]) < int(failing["fold_scale"])
        ),
        key=lambda row: int(row["fold_scale"]),
    )
    x0 = float(previous["fold_scale"])
    x1 = float(failing["fold_scale"])
    y0 = float(previous["certificate"]["margin_lcb"])
    y1 = float(failing["certificate"]["margin_lcb"])
    estimate = x0 if y1 == y0 else x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
    return {
        "status": "INTERPOLATED_ZERO_CROSSING",
        "lower_bound": x0,
        "estimate": float(estimate),
    }


def analyze_qpu_triad_panel(
    *,
    panel_root: Path,
    calibration_pre_root: Path,
    science_root: Path,
    calibration_post_root: Path,
    output: Path,
) -> dict[str, Any]:
    roots = [panel_root, calibration_pre_root, science_root, calibration_post_root]
    for root in roots:
        if not verify_freeze(root.resolve() / "freeze_manifest.json")["ok"]:
            raise RuntimeError(f"Invalid freeze: {root}")
    panel_root = panel_root.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    manifest = read_json(panel_root / "qpu_triad_panel.json")
    pre_raw = read_json(calibration_pre_root.resolve() / "raw_counts.json")
    science_raw = read_json(science_root.resolve() / "raw_counts.json")
    post_raw = read_json(calibration_post_root.resolve() / "raw_counts.json")
    pre_map = _result_map(pre_raw)
    science_map = _result_map(science_raw)
    post_map = _result_map(post_raw)
    candidate_count = int(manifest["candidate_count"])
    scales = [int(value) for value in manifest["fold_scales"]]
    alpha_cal_total = float(manifest["registered_analysis"]["alpha_calibration_total"])
    alpha_sci_total = float(manifest["registered_analysis"]["alpha_science_total"])
    alpha_science = alpha_sci_total / (candidate_count * len(scales))

    reports: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(manifest["candidates"]):
        candidate_id = str(candidate["candidate_id"])
        protocol = read_json(panel_root / candidate["protocol_path"])
        patterns = protocol["calibration"]["patterns"]
        convention, convention_scores, convention_margin = _choose_convention(
            pre=pre_map,
            post=post_map,
            candidate_id=candidate_id,
            patterns=patterns,
        )
        width = len(protocol["calibration"]["logical_qubits"])
        alpha_per_parameter_period = alpha_cal_total / (
            candidate_count * 2 * 2 * width
        )
        box, calibration_detail = _calibration_box(
            pre_result=pre_map,
            post_result=post_map,
            candidate_id=candidate_id,
            protocol=protocol,
            convention=convention,
            alpha_per_parameter_period=alpha_per_parameter_period,
        )
        worst_determinants = [
            1.0 - box[2 * index][1] - box[2 * index + 1][1]
            for index in range(width)
        ]
        calibration_reasons: list[str] = []
        if convention_scores[convention] < float(
            protocol["thresholds"]["minimum_convention_accuracy"]
        ):
            calibration_reasons.append("convention_accuracy_below_threshold")
        if convention_margin < float(
            protocol["thresholds"]["minimum_convention_margin"]
        ):
            calibration_reasons.append("convention_margin_below_threshold")
        if min(worst_determinants, default=0.0) < float(
            protocol["thresholds"]["minimum_assignment_determinant"]
        ):
            calibration_reasons.append("assignment_determinant_below_threshold")
        calibration_pass = not calibration_reasons

        scale_reports: list[dict[str, Any]] = []
        for scale_index, scale in enumerate(scales):
            label = f"{candidate_id}::science_fold_{scale}"
            result = science_map[label]
            histogram = _joint_histogram(
                counts=result["counts"],
                n=int(protocol["n"]),
                convention=convention,
                logical_support=[
                    int(value) for value in protocol["calibration"]["logical_qubits"]
                ],
            )
            certificate = _certificate_for_histogram(
                histogram=histogram,
                protocol=protocol,
                calibration_box=box,
                alpha_science=alpha_science,
                optimization_seed=int(protocol["confidence"]["optimization_seed"])
                + candidate_index * 1009
                + scale_index * 37,
                safety_epsilon=float(
                    protocol["confidence"]["optimization_safety_epsilon"]
                ),
            )
            certificate["pass"] = bool(calibration_pass and certificate["pass"])
            certificate["shots_received"] = int(result["shots_received"])
            scale_reports.append(
                {
                    "fold_scale": scale,
                    "job_id": result.get("job_id"),
                    "certificate": certificate,
                }
            )
        margins = [float(item["certificate"]["margin_lcb"]) for item in scale_reports]
        scale_passes = sum(bool(item["certificate"]["pass"]) for item in scale_reports)
        critical = _critical_fold_scale(scale_reports)
        reports.append(
            {
                "candidate_id": candidate_id,
                "roles": candidate["roles"],
                "family": candidate["family"],
                "code": candidate["code"],
                "hardness": candidate["hardness"],
                "triad_score_classical": candidate["triad_score"],
                "calibration": {
                    "pass": calibration_pass,
                    "reasons": calibration_reasons,
                    "convention": convention,
                    "convention_scores": convention_scores,
                    "convention_margin": convention_margin,
                    "worst_assignment_determinants": worst_determinants,
                    **calibration_detail,
                },
                "fold_scales": scale_reports,
                "base_pass": bool(scale_reports[0]["certificate"]["pass"]),
                "scales_passed": scale_passes,
                "minimum_margin_lcb": min(margins),
                "base_margin_lcb": margins[0],
                "critical_fold_scale": critical,
            }
        )

    def ranking_key(item: dict[str, Any]) -> tuple[Any, ...]:
        critical = item["critical_fold_scale"]
        critical_value = (
            float(critical["estimate"])
            if critical.get("estimate") is not None
            else float(critical.get("lower_bound", 0.0))
        )
        return (
            int(bool(item["base_pass"])),
            int(item["scales_passed"]),
            float(item["minimum_margin_lcb"]),
            critical_value,
            float(item["hardness"]["gamma_log10"]),
            -int(item["code"]["n"]),
        )

    ranked = sorted(reports, key=ranking_key, reverse=True)
    family_best: dict[str, str] = {}
    for item in ranked:
        family_best.setdefault(str(item["family"]["type"]), str(item["candidate_id"]))
    gamma_target = float(manifest["registered_analysis"]["gamma_target"])
    qualified = [
        item
        for item in reports
        if item["base_pass"] and float(item["hardness"]["gamma_log10"]) >= gamma_target
    ]
    smallest = min(
        qualified,
        key=lambda item: (int(item["code"]["n"]), -float(item["base_margin_lcb"])),
        default=None,
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "candidate_qpu_reports.json", reports)
    write_json(output / "qpu_family_champions.json", family_best)
    write_json(
        output / "qpu_minimal_qubit_gap.json",
        {
            "schema": "codegap.qpu-triad-minimal-qubit-gap.v1",
            "gamma_target": gamma_target,
            "status": "FOUND" if smallest else "NOT_FOUND",
            "candidate_id": smallest["candidate_id"] if smallest else None,
            "n": smallest["code"]["n"] if smallest else None,
            "gamma_log10": smallest["hardness"]["gamma_log10"] if smallest else None,
            "base_margin_lcb": smallest["base_margin_lcb"] if smallest else None,
            "scope": "Minimum only within the frozen QPU panel and registered attack suite.",
        },
    )
    report = {
        "schema": "codegap.qpu-triad-analysis.v1",
        "created_at": utc_now(),
        "status": "PASS" if ranked and ranked[0]["base_pass"] else "STOP_NO_QPU_PANEL_PASS",
        "candidate_count": len(reports),
        "best_overall_candidate": ranked[0]["candidate_id"] if ranked else None,
        "family_champions": family_best,
        "minimal_qubit_gap_candidate": smallest["candidate_id"] if smallest else None,
        "research_questions": {
            "optimal_real_code_families": {
                "answered": bool(family_best),
                "result": family_best,
                "scope": "Frozen QPU panel only.",
            },
            "hardware_noise_robustness": {
                "answered": True,
                "method": "fresh pre/post calibration drift hull plus local unitary fold scales",
                "candidate_curves": {
                    item["candidate_id"]: {
                        "base_margin_lcb": item["base_margin_lcb"],
                        "minimum_margin_lcb": item["minimum_margin_lcb"],
                        "critical_fold_scale": item["critical_fold_scale"],
                        "maximum_calibration_parameter_drift": item["calibration"][
                            "maximum_parameter_drift"
                        ],
                    }
                    for item in reports
                },
            },
            "few_qubits_large_gap": {
                "answered": smallest is not None,
                "candidate_id": smallest["candidate_id"] if smallest else None,
                "n": smallest["code"]["n"] if smallest else None,
                "gamma_target": gamma_target,
                "scope": "Frozen QPU panel and registered classical attack suite only.",
            },
        },
        "simultaneous_confidence": {
            "alpha_calibration_total": manifest["registered_analysis"][
                "alpha_calibration_total"
            ],
            "alpha_science_total": manifest["registered_analysis"][
                "alpha_science_total"
            ],
            "science_tests": candidate_count * len(scales),
            "alpha_per_science_test": alpha_science,
        },
        "claim_boundaries": manifest["claim_boundaries"],
        "source_hashes": {
            "panel": file_sha256(panel_root / "qpu_triad_panel.json"),
            "calibration_pre": file_sha256(
                calibration_pre_root.resolve() / "raw_counts.json"
            ),
            "science": file_sha256(science_root.resolve() / "raw_counts.json"),
            "calibration_post": file_sha256(
                calibration_post_root.resolve() / "raw_counts.json"
            ),
        },
    }
    write_json(output / "qpu_triad_report.json", report)
    freeze(output)
    if not verify_freeze(output / "freeze_manifest.json")["ok"]:
        raise RuntimeError("QPU triad analysis freeze verification failed.")
    return report
