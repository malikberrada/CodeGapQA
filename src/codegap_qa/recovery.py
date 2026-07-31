from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

from .gates import gate_b, gate_c, qpu_gate
from .phase import evaluate_noise_phase
from .progress import ProgressManager
from .verifier_search import apply_verifier_selection, select_verifiable_observables


def _load_frontier(artifact: Path) -> list[dict[str, Any]]:
    paths = [
        artifact / "codeforge" / "hardware_code_frontier.json",
        artifact / "hardware_code_frontier.json",
    ]
    for path in paths:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8-sig"))
    raise FileNotFoundError(
        "hardware_code_frontier.json was not found under the supplied artifact."
    )


def _update_candidate(candidate: dict[str, Any], preflight: dict[str, Any]) -> None:
    circuit = candidate["exact_artifacts"]["circuit_spec"]
    apply_verifier_selection(circuit, preflight)
    schedule = candidate["exact_artifacts"].setdefault("schedule_search", {})
    schedule["verifier_preflight"] = preflight
    schedule["lightcone_preflight"] = preflight
    checks = schedule.setdefault("target_checks", {})
    checks["verifier_preflight"] = bool(preflight.get("passed", False))
    checks.pop("lightcone", None)
    best = float(candidate["hardness"]["best_attack_operations"])
    verify = float(preflight.get("verify_operations") or float("inf"))
    candidate["hardness"]["verify_operations"] = verify
    candidate["hardness"]["gamma_log10"] = (
        math.log10(max(best, 1.0) / max(verify, 1.0))
        if math.isfinite(verify)
        else float("-inf")
    )


def recover_verifier_from_artifacts(
    artifact: Path,
    config: dict[str, Any],
    output: Path,
    candidate_ids: list[str] | None = None,
) -> dict[str, Any]:
    artifact = artifact.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manager = ProgressManager.from_config(config)
    candidates = _load_frontier(artifact)
    configured_priority = list(
        config.get("recovery", {}).get("candidate_priority", [])
    )
    priority = list(candidate_ids or configured_priority)
    if priority:
        by_id = {item["candidate_id"]: item for item in candidates}
        missing = [value for value in priority if value not in by_id]
        if missing:
            raise ValueError(
                "Requested recovery candidates were not found: "
                + ", ".join(missing)
            )
        candidates = [by_id[value] for value in priority]
    bar = manager.bar(
        candidates,
        total=len(candidates),
        desc="RecoverVerifier: GPU local observables",
        unit="candidate",
        leave=True,
    )
    records = []
    recovered = []
    for candidate in bar:
        circuit = candidate["exact_artifacts"]["circuit_spec"]
        preflight = select_verifiable_observables(circuit, config)
        _update_candidate(candidate, preflight)
        noise_result = None
        if preflight.get("passed", False):
            try:
                noise_result = evaluate_noise_phase(
                    candidate,
                    config,
                    output / "noisecert" / candidate["candidate_id"],
                    progress=manager,
                )
                b = gate_b(noise_result, config)
            except Exception as error:
                b = qpu_gate(
                    "B_NOISE",
                    False,
                    {"status": "EVALUATION_FAILED", "error": str(error)},
                )
        else:
            b = qpu_gate(
                "B_NOISE",
                False,
                {"status": "VERIFIER_PREFLIGHT_FAILED", "preflight": preflight},
            )
        c = gate_c(candidate, config)
        passed = bool(preflight.get("passed") and b.passed and c.passed)
        record = {
            "candidate_id": candidate["candidate_id"],
            "n": candidate["code"]["n"],
            "verifier_preflight": preflight,
            "B_NOISE": asdict(b),
            "C_HARDNESS": asdict(c),
            "fully_qualified_classically": passed,
            "cotengra_reused": True,
            "new_gamma_log10": float(candidate["hardness"]["gamma_log10"]),
            "new_verify_operations": float(
                candidate["hardness"]["verify_operations"]
            ),
            "active_witness_features": int(
                preflight.get("active_witness_features", 0)
            ),
            "maximum_witness_coefficient": float(
                preflight.get("maximum_witness_coefficient", 0.0)
            ),
            "priority_index": (
                priority.index(candidate["candidate_id"])
                if candidate["candidate_id"] in priority
                else len(priority)
            ),
        }
        records.append(record)
        if preflight.get("passed"):
            recovered.append(candidate)
        bar.set_postfix(
            n=candidate["code"]["n"],
            verifier=preflight.get("passed", False),
            local=preflight.get("local_observables", 0),
            rank=preflight.get("local_rank", 0),
            B=b.passed,
            C=c.passed,
            refresh=False,
        )
    bar.close()
    qualified = [item for item in records if item["fully_qualified_classically"]]

    def record_key(item: dict[str, Any]) -> tuple:
        noise = item.get("B_NOISE", {}).get("evidence", {})
        minimum_probability = float(
            noise.get("required_envelope_min_pass_probability", 0.0)
        )
        return (
            bool(item.get("fully_qualified_classically")),
            bool(item.get("B_NOISE", {}).get("passed")),
            bool(item.get("C_HARDNESS", {}).get("passed")),
            minimum_probability,
            float(item["verifier_preflight"].get("witness_margin_lcb", float("-inf"))),
            float(item.get("new_gamma_log10", float("-inf"))),
            -int(item.get("priority_index", 10**9)),
        )

    ranked_records = sorted(records, key=record_key, reverse=True)
    recovered.sort(
        key=lambda item: (
            item["hardness"]["gamma_log10"],
            item.get("objective", float("-inf")),
        ),
        reverse=True,
    )
    (output / "recovered_hardware_code_frontier.json").write_text(
        json.dumps(recovered, indent=2), encoding="utf-8"
    )
    report = {
        "schema": "codegap.verifier-recovery.v2-multifeature",
        "source_artifact": str(artifact),
        "status": "RECOVERED_CANDIDATE_FOUND" if qualified else "NO_RECOVERABLE_VERIFIER",
        "cotengra_recomputed": False,
        "evaluated_candidates": len(records),
        "verifier_preflight_pass": sum(
            bool(item["verifier_preflight"].get("passed")) for item in records
        ),
        "fully_qualified_classically": len(qualified),
        "candidate_priority": priority,
        "recommended_candidate_id": (
            ranked_records[0]["candidate_id"] if ranked_records else None
        ),
        "backup_candidate_id": (
            ranked_records[1]["candidate_id"] if len(ranked_records) > 1 else None
        ),
        "best_candidate_id": (
            next(
                (item["candidate_id"] for item in ranked_records if item["fully_qualified_classically"]),
                None,
            )
        ),
        "comparison": [
            {
                "candidate_id": item["candidate_id"],
                "priority_index": item["priority_index"],
                "fully_qualified_classically": item["fully_qualified_classically"],
                "B_NOISE": item["B_NOISE"]["passed"],
                "C_HARDNESS": item["C_HARDNESS"]["passed"],
                "witness_margin_lcb": item["verifier_preflight"].get("witness_margin_lcb"),
                "active_witness_features": item.get("active_witness_features"),
                "maximum_witness_coefficient": item.get("maximum_witness_coefficient"),
                "verify_operations": item.get("new_verify_operations"),
                "gamma_log10": item.get("new_gamma_log10"),
            }
            for item in ranked_records
        ],
        "records": records,
        "claim_boundary": (
            "This command reuses existing registered classical attack costs and "
            "does not rerun Cotengra. A recovered candidate still requires fresh "
            "Gate A QPU compilation and QPU diagnostics."
        ),
    }
    (output / "verifier_recovery_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
