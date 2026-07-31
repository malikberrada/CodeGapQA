from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .codesign import circuit_qasm3
from .freeze import freeze
from .gates import gate_a, gate_b, gate_c, qpu_gate
from .optimizer import search_code_families
from .phase import evaluate_noise_phase
from .progress import ProgressManager


def load_config(path: Path) -> dict:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    payload["_config_path"] = str(resolved)
    payload["_config_dir"] = str(resolved.parent)
    return payload


def _write_benchmark_artifacts(frontier: list, config: dict, output: Path) -> None:
    payload = [candidate.to_dict() for candidate in frontier]
    simulation = [
        {
            "candidate_id": item["candidate_id"],
            "n": item["code"]["n"],
            **item["hardness"],
        }
        for item in payload
    ]
    verification = [
        {
            "candidate_id": item["candidate_id"],
            "n": item["code"]["n"],
            "verify_operations": item["hardness"]["verify_operations"],
            "verifier_masks": len(
                item["exact_artifacts"]["circuit_spec"]["verifier_masks"]
            ),
            "verifier_cost_model": item["exact_artifacts"]["circuit_spec"][
                "verifier_cost_model"
            ],
        }
        for item in payload
    ]
    (output / "simulation_benchmarks.json").write_text(
        json.dumps(
            {
                "schema": "codegap.simulation-benchmarks.v2-spacetime",
                "scope": (
                    "full doubled space-time tensor network and registered "
                    "classical attack cost models"
                ),
                "candidates": simulation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "verification_benchmarks.json").write_text(
        json.dumps(
            {
                "schema": "codegap.verification-benchmarks.v2-lightcone",
                "candidates": verification,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    threshold = float(config["gates"]["min_gamma_log10"])
    hard = sorted(
        (
            item
            for item in payload
            if item["hardness"]["gamma_log10"] >= threshold
        ),
        key=lambda item: (item["code"]["n"], -item["objective"]),
    )
    (output / "minimum_hard_instance.json").write_text(
        json.dumps(
            {
                "schema": "codegap.minimum-hard-instance.v2",
                "gamma_threshold": threshold,
                "status": "FOUND" if hard else "NOT_FOUND",
                "candidate": hard[0] if hard else None,
                "scope": (
                    "Minimum only within the searched quotient and registered "
                    "space-time attack models."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "complexity_assumptions.md").write_text(
        "# Registered complexity assumptions\n\n"
        + "\n".join(f"- {item}" for item in config["hardness"]["assumptions"])
        + "\n\nGate C is conditional and does not prove a lower bound against every "
        "possible classical algorithm.\n",
        encoding="utf-8",
    )


def _write_selected_circuit(candidate: dict, output: Path) -> None:
    spec = candidate["exact_artifacts"]["circuit_spec"]
    (output / "selected_circuit.qasm3").write_text(
        circuit_qasm3(spec), encoding="utf-8"
    )
    (output / "selected_circuit.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )


def run_pipeline(config_path: Path, output: Path) -> dict:
    config = load_config(config_path)
    manager = ProgressManager.from_config(config)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    pipeline_bar = manager.bar(
        total=7,
        desc="CodeGap classical pipeline",
        unit="stage",
        leave=True,
    )
    pipeline_bar.set_postfix(stage="CodeForge + adversarial schedules", refresh=True)
    frontier, optimality = search_code_families(
        config, output / "codeforge", progress=manager
    )
    pipeline_bar.update(1)

    pipeline_bar.set_postfix(stage="Benchmark artifacts", refresh=True)
    _write_benchmark_artifacts(frontier, config, output)
    pipeline_bar.update(1)
    if not frontier:
        summaries = optimality.get("schedule_summaries", [])
        statuses = [item.get("status") for item in summaries]
        if statuses and all(status == "STOP_TARGET_SNAPSHOT_REQUIRED" for status in statuses):
            stop_status = "STOP_TARGET_SNAPSHOT_REQUIRED"
            action = (
                "Run codegap qpu-snapshot, set hardware.target_snapshot in the "
                "configuration, then rerun the classical pipeline."
            )
        elif any(status == "STOP_NO_TARGET_NATIVE_MATCHING_POOL" for status in statuses):
            stop_status = "STOP_NO_TARGET_NATIVE_MATCHING_POOL"
            action = (
                "The captured target did not yield enough diverse perfect "
                "matchings for the requested n. Increase target layout/matching "
                "trials or inspect target-native pool diagnostics."
            )
        elif any(status == "STOP_ZERO_SWAP_SEARCH_BUDGET_EXHAUSTED" for status in statuses):
            stop_status = "STOP_ZERO_SWAP_SEARCH_BUDGET_EXHAUSTED"
            action = "Increase schedule_search embedding budgets or reduce schedule depth."
        elif statuses and all(status == "STOP_NO_ZERO_SWAP_TARGET_EMBEDDING" for status in statuses):
            stop_status = "STOP_NO_ZERO_SWAP_TARGET_EMBEDDING"
            action = "No generated schedule embeds on the captured target; expand the matching/schedule search."
        else:
            stop_status = "STOP_NO_SCHEDULE_CANDIDATE"
            action = "Inspect codeforge/schedule_search_summary.json for per-filter counts."
        report = {
            "schema": "codegap.pipeline.v3.2-target-native-frame-schedules",
            "status": stop_status,
            "fully_qualified_candidate_id": None,
            "best_partial_candidate_id": None,
            "selected_candidate_id": None,
            "qpu_preparation_authorized": False,
            "schedule_statuses": statuses,
            "action": action,
            "gates": [],
        }
        (output / "gate_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        freeze(output, progress=manager)
        pipeline_bar.update(5)
        pipeline_bar.close()
        return report

    # Verifier preflight, Cotengra and line-graph filtering are complete before
    # this point. NoiseCert is the final classical filter. Gate-C-pass schedules
    # are evaluated first; when none
    # pass, the strongest partial candidates are still characterized without
    # authorizing a QPU run.
    frontier_payload = [candidate.to_dict() for candidate in frontier]
    ordered = sorted(
        frontier_payload,
        key=lambda item: (
            item["hardness"]["gamma_log10"]
            < float(config["gates"]["min_gamma_log10"]),
            -item["hardness"]["gamma_log10"],
            item["code"]["n"],
            -item["objective"],
        ),
    )
    max_noise_candidates = int(config["gates"].get("max_noise_candidates", 8))
    screen_items = ordered[:max_noise_candidates]
    evaluated: list[dict] = []
    fully_qualified: dict | None = None
    fully_qualified_gates = None
    fully_qualified_noise = None
    screen_bar = manager.bar(
        screen_items,
        total=len(screen_items),
        desc="Filter 11: NoiseCert after verifier + Cotengra",
        unit="candidate",
        leave=True,
    )
    pipeline_bar.set_postfix(stage="NoiseCert after verifier-constrained Gate C", refresh=True)
    for candidate in screen_bar:
        static_a = gate_a(candidate, optimality, config)
        c = gate_c(candidate, config)
        noise_result = None
        if static_a.passed:
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
                {"status": "NOT_EVALUATED_STATIC_CODE_FAILED"},
            )
        evaluated.append(
            {
                "candidate_id": candidate["candidate_id"],
                "n": candidate["code"]["n"],
                "objective": candidate["objective"],
                "gamma_log10": candidate["hardness"]["gamma_log10"],
                "schedule_target_checks": candidate["exact_artifacts"]
                .get("schedule_search", {})
                .get("target_checks", {}),
                "gates": [asdict(static_a), asdict(b), asdict(c)],
            }
        )
        screen_bar.set_postfix(
            candidate=candidate["candidate_id"],
            n=candidate["code"]["n"],
            B=b.passed,
            C=c.passed,
            gamma=f"{candidate['hardness']['gamma_log10']:.3f}",
            refresh=False,
        )
        if static_a.passed and b.passed and c.passed:
            fully_qualified = candidate
            fully_qualified_gates = (static_a, b, c)
            fully_qualified_noise = noise_result
            break
    screen_bar.close()
    pipeline_bar.update(1)

    best_partial = max(
        frontier_payload,
        key=lambda item: (
            item["hardness"]["gamma_log10"],
            item["objective"],
            -item["code"]["n"],
        ),
    )
    display_candidate = fully_qualified or best_partial
    if fully_qualified is not None:
        selected_gates = fully_qualified_gates
    else:
        matching = next(
            (
                item
                for item in evaluated
                if item["candidate_id"] == best_partial["candidate_id"]
            ),
            None,
        )
        static_a = gate_a(best_partial, optimality, config)
        c = gate_c(best_partial, config)
        if matching is not None:
            b_payload = matching["gates"][1]
            b = qpu_gate(
                "B_NOISE",
                bool(b_payload["passed"]),
                b_payload.get("evidence", {}),
            )
        else:
            b = qpu_gate(
                "B_NOISE",
                False,
                {"status": "NOT_EVALUATED_OUTSIDE_NOISECERT_LIMIT"},
            )
        selected_gates = (static_a, b, c)

    pipeline_bar.set_postfix(stage="Candidate artifacts", refresh=True)
    (output / "candidate_gate_screen.json").write_text(
        json.dumps(evaluated, indent=2), encoding="utf-8"
    )
    (output / "best_partial_candidate.json").write_text(
        json.dumps(best_partial, indent=2), encoding="utf-8"
    )
    if fully_qualified is not None:
        (output / "fully_qualified_candidate.json").write_text(
            json.dumps(fully_qualified, indent=2), encoding="utf-8"
        )
    # Backward-compatible file, explicitly annotated in the report as either
    # fully qualified or partial.
    (output / "selected_candidate.json").write_text(
        json.dumps(display_candidate, indent=2), encoding="utf-8"
    )
    _write_selected_circuit(display_candidate, output)
    pipeline_bar.update(1)

    pipeline_bar.set_postfix(stage="Gate report", refresh=True)
    static_a, b, c = selected_gates
    qpu_a = qpu_gate(
        "A_CODE_QPU",
        False,
        {
            "status": "PENDING_LIVE_BACKEND_TARGET_COMPILATION"
            if fully_qualified is not None
            else "BLOCKED_UNTIL_B_AND_C_PASS",
            "required_command": "codegap qpu-prepare",
            "static_gate": asdict(static_a),
        },
    )
    ready_for_qpu_preparation = fully_qualified is not None
    report = {
        "schema": "codegap.pipeline.v3.3-verifier-constrained-schedules",
        "status": (
            "READY_FOR_QPU_NATIVE_PREPARATION"
            if ready_for_qpu_preparation
            else "STOP_BEFORE_QPU"
        ),
        "selection_rule": (
            "maximize adversarial schedule objective subject to local-observable "
            "count/rank/lightcone/witness constraints before cutwidth, line-graph "
            "and Cotengra; require B and C before live QPU Gate A"
        ),
        "fully_qualified_candidate_id": (
            fully_qualified["candidate_id"] if fully_qualified else None
        ),
        "best_partial_candidate_id": best_partial["candidate_id"],
        "selected_candidate_id": (
            fully_qualified["candidate_id"] if fully_qualified else None
        ),
        "display_candidate_id": display_candidate["candidate_id"],
        "display_candidate_status": (
            "FULLY_QUALIFIED" if fully_qualified else "BEST_PARTIAL_ONLY"
        ),
        "selected_n": (
            fully_qualified["code"]["n"] if fully_qualified else None
        ),
        "best_partial_n": best_partial["code"]["n"],
        "gates": [
            asdict(static_a),
            asdict(qpu_a),
            asdict(b),
            asdict(c),
            asdict(qpu_gate("D_QPU_DIAGNOSTIC", False, {"status": "NOT_RUN"})),
            asdict(qpu_gate("E_QPU_FINAL", False, {"status": "NOT_RUN"})),
        ],
        "qpu_preparation_authorized": ready_for_qpu_preparation,
        "qpu_submission_authorized": False,
        "claim_boundary": {
            "code": optimality["claim"],
            "schedule": (
                "adversarial optimization over the explicitly evaluated schedule "
                "set under preregistered local-verifier constraints; not a global "
                "optimum over all circuits"
            ),
            "verifier": (
                "selected independent local observables with exact backward-lightcone "
                "expectations; this does not establish full-distribution closeness"
            ),
            "qpu_native": "pending live BackendV2.target compilation",
            "noise": (
                "exact local-lightcone observables with finite-shot and "
                "registered analytic adversary bounds"
            ),
            "hardness": (
                "conditional and attack-suite scoped on the full doubled "
                "space-time tensor network"
            ),
        },
    }
    (output / "gate_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pipeline_bar.update(1)

    pipeline_bar.set_postfix(stage="Freeze artifacts", refresh=True)
    freeze(output, progress=manager)
    pipeline_bar.update(2)
    pipeline_bar.set_postfix(stage="Complete", refresh=True)
    pipeline_bar.close()
    return report

