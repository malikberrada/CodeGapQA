from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from typing import Any

from .codesign import circuit_qasm3
from .freeze import freeze, verify_freeze
from .gates import gate_a, gate_c, qpu_gate
from .progress import ProgressManager


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _resolve_target_snapshot(
    artifact: Path,
    config: dict[str, Any],
) -> Path:
    configured = config.get("hardware", {}).get("target_snapshot")
    candidates: list[Path] = []
    if configured:
        configured_path = Path(str(configured))
        if configured_path.is_absolute():
            candidates.append(configured_path)
        else:
            config_dir = Path(str(config.get("_config_dir", artifact)))
            candidates.extend(
                [
                    config_dir / configured_path,
                    artifact / configured_path,
                    artifact.parent / configured_path,
                ]
            )
    candidates.extend(
        [
            artifact / "target_snapshot.json",
            artifact / "codeforge" / "target_snapshot.json",
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    attempted = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "The preregistered Cepheus target snapshot could not be located. "
        f"Attempted:\n{attempted}"
    )


def _record_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record["candidate_id"]): record
        for record in report.get("records", [])
    }


def _normalize_target_checks(
    candidate: dict[str, Any],
    record: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    schedule = candidate.setdefault("exact_artifacts", {}).setdefault(
        "schedule_search", {}
    )
    preflight = record["verifier_preflight"]
    schedule["verifier_preflight"] = preflight
    schedule["lightcone_preflight"] = preflight
    checks = schedule.setdefault("target_checks", {})

    cotengra = candidate.get("hardness", {}).get("cotengra", {})
    schedule_settings = config.get("schedule_search", {})
    require_cotengra = bool(schedule_settings.get("require_cotengra", False))
    width = float(cotengra.get("contraction_width_log2") or 0.0)
    flops = float(cotengra.get("contraction_flops") or 0.0)
    gamma = float(candidate.get("hardness", {}).get("gamma_log10", -math.inf))

    zero_swap = bool(
        checks.get("proxy_zero_swap", False)
        or checks.get("live_target_zero_swap", False)
    )
    checks["cotengra_width"] = (
        width >= float(schedule_settings.get("target_width_min", 0.0))
        if require_cotengra
        else True
    )
    checks["cotengra_flops"] = (
        flops >= float(schedule_settings.get("min_cotengra_flops", 0.0))
        if require_cotengra
        else True
    )
    checks["gamma"] = gamma >= float(config["gates"]["min_gamma_log10"])
    checks["verifier_preflight"] = bool(preflight.get("passed", False))
    checks["relation_certificate"] = bool(
        checks.get("relation_certificate", False)
    )
    checks["proxy_zero_swap"] = zero_swap
    checks["live_target_zero_swap"] = zero_swap
    checks.pop("lightcone", None)

    required = (
        "cotengra_width",
        "cotengra_flops",
        "gamma",
        "verifier_preflight",
        "relation_certificate",
        "proxy_zero_swap",
    )
    target_pass = all(bool(checks.get(name, False)) for name in required)
    schedule["target_pass"] = target_pass
    schedule["target_pass_recomputed_by"] = (
        "codegap.promote-recovery.v0.5.5"
    )
    schedule["target_pass_required_checks"] = list(required)
    return target_pass


def _candidate_by_id(
    frontier: list[dict[str, Any]],
    candidate_id: str,
) -> dict[str, Any]:
    for candidate in frontier:
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    raise KeyError(
        f"Recovered candidate {candidate_id!r} is missing from the recovered frontier."
    )


def _validate_snapshot_fingerprint(
    snapshot: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> None:
    actual = snapshot.get("structural_fingerprint")
    if not actual:
        raise ValueError("The target snapshot has no structural_fingerprint.")
    mismatches = []
    for candidate in candidates:
        schedule = candidate.get("exact_artifacts", {}).get("schedule_search", {})
        expected = schedule.get("target_structural_fingerprint")
        if expected and expected != actual:
            mismatches.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "expected": expected,
                    "actual": actual,
                }
            )
    if mismatches:
        raise RuntimeError(
            "The promoted candidates were searched against a different target "
            f"structure: {mismatches}"
        )


def _write_benchmark_views(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    output: Path,
) -> None:
    simulation = [
        {
            "candidate_id": item["candidate_id"],
            "n": item["code"]["n"],
            **item["hardness"],
        }
        for item in candidates
    ]
    verification = [
        {
            "candidate_id": item["candidate_id"],
            "n": item["code"]["n"],
            "verify_operations": item["hardness"]["verify_operations"],
            "active_verify_operations": item["exact_artifacts"]
            ["schedule_search"]["verifier_preflight"].get(
                "active_verify_operations"
            ),
            "audited_verify_operations": item["exact_artifacts"]
            ["schedule_search"]["verifier_preflight"].get(
                "audited_verify_operations"
            ),
            "local_observables": item["exact_artifacts"]
            ["schedule_search"]["verifier_preflight"].get(
                "local_observables"
            ),
            "local_rank": item["exact_artifacts"]
            ["schedule_search"]["verifier_preflight"].get("local_rank"),
        }
        for item in candidates
    ]
    _write(
        output / "simulation_benchmarks.json",
        {
            "schema": "codegap.simulation-benchmarks.v2-spacetime",
            "scope": (
                "Promoted candidates with reused registered classical attack costs."
            ),
            "candidates": simulation,
        },
    )
    _write(
        output / "verification_benchmarks.json",
        {
            "schema": "codegap.verification-benchmarks.v3-multifeature",
            "candidates": verification,
        },
    )
    threshold = float(config["gates"]["min_gamma_log10"])
    minimum = min(
        (
            item
            for item in candidates
            if float(item["hardness"]["gamma_log10"]) >= threshold
        ),
        key=lambda item: (
            int(item["code"]["n"]),
            -float(item.get("objective", 0.0)),
        ),
        default=None,
    )
    _write(
        output / "minimum_hard_instance.json",
        {
            "schema": "codegap.minimum-hard-instance.v3-promoted-recovery",
            "gamma_threshold": threshold,
            "status": "FOUND" if minimum else "NOT_FOUND",
            "candidate": minimum,
            "scope": (
                "Minimum only among the promoted recovered candidates and the "
                "registered attack suite."
            ),
        },
    )


def promote_recovered_artifact(
    *,
    artifact: Path,
    recovery: Path,
    output: Path,
) -> dict[str, Any]:
    artifact = artifact.resolve()
    recovery = recovery.resolve()
    output = output.resolve()
    if not artifact.is_dir():
        raise NotADirectoryError(artifact)
    if not recovery.is_dir():
        raise NotADirectoryError(recovery)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Promotion output must be absent or empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    source_manifest = artifact / "freeze_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(
            "The source classical artifact has no freeze_manifest.json."
        )
    source_freeze = verify_freeze(source_manifest)
    if not source_freeze["ok"]:
        raise RuntimeError(
            "The source classical artifact freeze is invalid: "
            f"{source_freeze['errors']}"
        )

    config_path = artifact / "resolved_config.json"
    optimality_path = artifact / "codeforge" / "optimality_certificate.json"
    recovery_report_path = recovery / "verifier_recovery_report.json"
    recovery_frontier_path = recovery / "recovered_hardware_code_frontier.json"
    config = _load(config_path)
    optimality = _load(optimality_path)
    report = _load(recovery_report_path)
    recovered_frontier = _load(recovery_frontier_path)

    if report.get("status") != "RECOVERED_CANDIDATE_FOUND":
        raise RuntimeError(
            "Recovery cannot be promoted because its status is not "
            "RECOVERED_CANDIDATE_FOUND."
        )
    source_reference = report.get("source_artifact")
    if source_reference:
        referenced = Path(str(source_reference)).resolve()
        if referenced != artifact:
            raise RuntimeError(
                "Recovery provenance does not match the supplied source artifact: "
                f"{referenced} != {artifact}"
            )

    recommended_id = report.get("recommended_candidate_id")
    backup_id = report.get("backup_candidate_id")
    if not recommended_id:
        raise RuntimeError("Recovery report has no recommended_candidate_id.")
    ordered_ids = [recommended_id]
    if backup_id and backup_id != recommended_id:
        ordered_ids.append(backup_id)

    records = _record_map(report)
    candidates: list[dict[str, Any]] = []
    gate_records: dict[str, dict[str, Any]] = {}
    for candidate_id in ordered_ids:
        record = records.get(candidate_id)
        if not record or not record.get("fully_qualified_classically", False):
            raise RuntimeError(
                f"Candidate {candidate_id} is not fully qualified classically."
            )
        candidate = _candidate_by_id(recovered_frontier, candidate_id)
        target_pass = _normalize_target_checks(candidate, record, config)
        if not target_pass:
            raise RuntimeError(
                f"Candidate {candidate_id} failed recomputed target checks."
            )
        c = gate_c(candidate, config)
        if not c.passed:
            raise RuntimeError(
                f"Candidate {candidate_id} failed recomputed Gate C: {c.reasons}"
            )
        if not record["B_NOISE"].get("passed", False):
            raise RuntimeError(f"Candidate {candidate_id} failed Gate B.")
        candidates.append(candidate)
        gate_records[candidate_id] = {
            "B_NOISE": record["B_NOISE"],
            "C_HARDNESS": asdict(c),
        }

    target_snapshot_path = _resolve_target_snapshot(artifact, config)
    target_snapshot = _load(target_snapshot_path)
    _validate_snapshot_fingerprint(target_snapshot, candidates)

    source_codeforge = artifact / "codeforge"
    target_codeforge = output / "codeforge"
    shutil.copytree(source_codeforge, target_codeforge, dirs_exist_ok=True)
    _write(target_codeforge / "hardware_code_frontier.json", candidates)
    _copy_file(optimality_path, target_codeforge / "optimality_certificate.json")

    _copy_file(target_snapshot_path, output / "target_snapshot.json")
    promoted_config = dict(config)
    promoted_config["hardware"] = dict(config.get("hardware", {}))
    promoted_config["hardware"]["target_snapshot"] = "target_snapshot.json"
    promoted_config["_config_path"] = str(output / "resolved_config.json")
    promoted_config["_config_dir"] = str(output)
    promoted_config["promotion"] = {
        "schema": "codegap.recovery-promotion.v1",
        "source_artifact": str(artifact),
        "recovery_artifact": str(recovery),
        "selected_candidate_id": recommended_id,
        "backup_candidate_id": backup_id,
        "cotengra_recomputed": False,
    }
    _write(output / "resolved_config.json", promoted_config)

    _copy_file(
        recovery_report_path,
        output / "verifier_recovery_report.json",
    )
    _copy_file(
        recovery_frontier_path,
        output / "recovered_hardware_code_frontier.json",
    )
    for candidate_id in ordered_ids:
        source_noise = recovery / "noisecert" / candidate_id
        if source_noise.is_dir():
            shutil.copytree(
                source_noise,
                output / "noisecert" / candidate_id,
                dirs_exist_ok=True,
            )

    selected = candidates[0]
    backup = candidates[1] if len(candidates) > 1 else None
    _write(output / "fully_qualified_candidate.json", selected)
    _write(output / "selected_candidate.json", selected)
    if backup is not None:
        _write(output / "backup_candidate.json", backup)
    circuit = selected["exact_artifacts"]["circuit_spec"]
    _write(output / "selected_circuit.json", circuit)
    (output / "selected_circuit.qasm3").write_text(
        circuit_qasm3(circuit), encoding="utf-8"
    )

    static_a = gate_a(selected, optimality, promoted_config)
    selected_b = gate_records[recommended_id]["B_NOISE"]
    selected_c = gate_records[recommended_id]["C_HARDNESS"]
    qpu_a = qpu_gate(
        "A_CODE_QPU",
        False,
        {
            "status": "PENDING_LIVE_BACKEND_TARGET_COMPILATION",
            "required_command": "codegap qpu-prepare",
            "static_gate": asdict(static_a),
            "promoted_candidate_id": recommended_id,
        },
    )
    ready = bool(
        static_a.passed
        and selected_b.get("passed", False)
        and selected_c.get("passed", False)
        and selected["exact_artifacts"]["schedule_search"].get(
            "target_pass", False
        )
    )
    if not ready:
        raise RuntimeError(
            "The recommended candidate did not satisfy all promotion gates."
        )

    candidate_screen = []
    for index, candidate in enumerate(candidates):
        cid = candidate["candidate_id"]
        candidate_screen.append(
            {
                "candidate_id": cid,
                "priority_index": index,
                "n": candidate["code"]["n"],
                "objective": candidate.get("objective"),
                "gamma_log10": candidate["hardness"]["gamma_log10"],
                "target_pass": candidate["exact_artifacts"]
                ["schedule_search"]["target_pass"],
                "gates": [
                    asdict(gate_a(candidate, optimality, promoted_config)),
                    gate_records[cid]["B_NOISE"],
                    gate_records[cid]["C_HARDNESS"],
                ],
            }
        )
    _write(output / "candidate_gate_screen.json", candidate_screen)
    _write_benchmark_views(candidates, promoted_config, output)

    assumptions = promoted_config.get("hardness", {}).get("assumptions", [])
    (output / "complexity_assumptions.md").write_text(
        "# Registered complexity assumptions\n\n"
        + "\n".join(f"- {item}" for item in assumptions)
        + "\n\nGate C remains conditional and scoped to the registered attack suite.\n",
        encoding="utf-8",
    )

    gate_report = {
        "schema": "codegap.pipeline.v3.4-promoted-multifeature-recovery",
        "status": "READY_FOR_QPU_NATIVE_PREPARATION",
        "selection_rule": (
            "promote the v0.5.4 multifeature recovery recommendation after "
            "recomputing target checks, Gate B and Gate C; preserve the backup "
            "candidate and require fresh live BackendV2.target compilation"
        ),
        "fully_qualified_candidate_id": recommended_id,
        "best_partial_candidate_id": None,
        "selected_candidate_id": recommended_id,
        "backup_candidate_id": backup_id,
        "display_candidate_id": recommended_id,
        "display_candidate_status": "FULLY_QUALIFIED_RECOVERED",
        "selected_n": selected["code"]["n"],
        "best_partial_n": None,
        "gates": [
            asdict(static_a),
            asdict(qpu_a),
            selected_b,
            selected_c,
            asdict(qpu_gate("D_QPU_DIAGNOSTIC", False, {"status": "NOT_RUN"})),
            asdict(qpu_gate("E_QPU_FINAL", False, {"status": "NOT_RUN"})),
        ],
        "qpu_preparation_authorized": True,
        "qpu_submission_authorized": False,
        "promotion": {
            "source_artifact": str(artifact),
            "source_freeze_verified": True,
            "source_freeze_manifest_sha256": _digest(source_manifest),
            "recovery_artifact": str(recovery),
            "recovery_report_sha256": _digest(recovery_report_path),
            "recovered_frontier_sha256": _digest(recovery_frontier_path),
            "optimality_certificate_sha256": _digest(optimality_path),
            "target_snapshot_sha256": _digest(target_snapshot_path),
            "target_structural_fingerprint": target_snapshot[
                "structural_fingerprint"
            ],
            "cotengra_recomputed": False,
            "target_pass_recomputed": True,
        },
        "claim_boundary": {
            "code": optimality["claim"],
            "schedule": (
                "promoted from the explicitly evaluated adversarial schedule set; "
                "not a global optimum over all circuits"
            ),
            "verifier": (
                "independent exact-local multifeature checks; not a certificate of "
                "total-variation closeness for the complete output distribution"
            ),
            "qpu_native": "pending fresh live BackendV2.target compilation",
            "noise": (
                "exact local-lightcone observables with finite-shot and registered "
                "analytic adversary bounds"
            ),
            "hardness": (
                "conditional and attack-suite scoped on the full doubled "
                "space-time tensor network"
            ),
        },
    }
    _write(output / "gate_report.json", gate_report)
    _write(
        output / "promotion_report.json",
        {
            "schema": "codegap.recovery-promotion.v1",
            "status": "PASS",
            "selected_candidate_id": recommended_id,
            "backup_candidate_id": backup_id,
            "promoted_candidates": ordered_ids,
            "target_pass": {
                candidate["candidate_id"]: candidate["exact_artifacts"]
                ["schedule_search"]["target_pass"]
                for candidate in candidates
            },
            "qpu_preparation_authorized": True,
            "qpu_submission_authorized": False,
            "source_artifact": str(artifact),
            "recovery_artifact": str(recovery),
            "target_snapshot": str(output / "target_snapshot.json"),
            "optimality_certificate": str(
                target_codeforge / "optimality_certificate.json"
            ),
            "next_command": "codegap qpu-prepare",
        },
    )

    manager = ProgressManager.from_config(promoted_config)
    manifest = freeze(output, progress=manager)
    verification = verify_freeze(
        output / "freeze_manifest.json",
        progress=manager,
    )
    if not verification["ok"]:
        raise RuntimeError(
            "Generated promotion freeze failed verification: "
            f"{verification['errors']}"
        )
    return {
        "schema": "codegap.recovery-promotion-command.v1",
        "status": "READY_FOR_QPU_NATIVE_PREPARATION",
        "artifact": str(output),
        "selected_candidate_id": recommended_id,
        "backup_candidate_id": backup_id,
        "promoted_candidates": len(candidates),
        "target_pass_recomputed": True,
        "target_structural_fingerprint": target_snapshot[
            "structural_fingerprint"
        ],
        "optimality_certificate_preserved": True,
        "qpu_preparation_authorized": True,
        "qpu_submission_authorized": False,
        "freeze_manifest": str(output / "freeze_manifest.json"),
        "freeze_files": len(manifest["files"]),
        "freeze_verified": verification["ok"],
        "cotengra_recomputed": False,
    }
