from __future__ import annotations

from .models import ClaimLevel, GateDecision


def gate_a(candidate: dict, optimality: dict, config: dict) -> GateDecision:
    """Static algebraic/proxy screening only; never authorizes a QPU run."""

    constraints = config["constraints"]
    code = candidate["code"]
    hardware = candidate["hardware"]
    checks = {
        "commutes": code["commutes"],
        "k": code["k"] >= constraints["min_k"],
        "d_x": code["d_x_at_least"] >= constraints["min_d_x"],
        "d_z": code["d_z_at_least"] >= constraints["min_d_z"],
        "proxy_twoq": hardware["two_qubit_count"]
        <= constraints["max_two_qubit_count"],
        "proxy_depth": hardware["two_qubit_depth"]
        <= constraints["max_two_qubit_depth"],
        "proxy_swaps": hardware["swap_count"] <= constraints["max_swaps"],
        "proxy_nonlocal_edges": hardware["nonlocal_edges"]
        <= constraints.get("max_nonlocal_edges", 0),
    }
    passed = all(checks.values())
    level = (
        ClaimLevel.EXACT_FINITE
        if passed and optimality["search_complete"]
        else ClaimLevel.EMPIRICAL_ATTACK_SUITE
        if passed
        else ClaimLevel.NOT_ESTABLISHED
    )
    return GateDecision(
        gate="A_CODE_STATIC",
        passed=passed,
        claim_level=level,
        reasons=tuple(key for key, value in checks.items() if not value),
        evidence=checks
        | {
            "optimality_claim": optimality["claim"],
            "qpu_authorization": False,
            "boundary": (
                "Static PASS is not Gate A QPU. A live BackendV2.target "
                "compilation is mandatory."
            ),
        },
    )


def gate_a_qpu(
    candidate: dict,
    compile_report: dict,
    config: dict,
) -> GateDecision:
    settings = config.get("qpu", {}).get("gate_a_native", {})
    best = compile_report.get("best") or {}
    constraints = config["constraints"]
    code = candidate["code"]
    checks = {
        "static_code_pass": bool(
            code["commutes"]
            and code["k"] >= constraints["min_k"]
            and code["d_x_at_least"] >= constraints["min_d_x"]
            and code["d_z_at_least"] >= constraints["min_d_z"]
        ),
        "compile_status": compile_report.get("status") == "PASS",
        "max_swaps": int(best.get("swap_count", 10**9))
        <= int(settings.get("max_swaps", 0)),
        "max_nonlocal_edges": int(best.get("nonlocal_edges", 10**9))
        <= int(settings.get("max_nonlocal_edges", 0)),
        "backend_target_embedding": (
            bool(best.get("target_embedding_valid"))
            if settings.get("require_backend_target_embedding", True)
            else True
        ),
        "measurement_map_validation": (
            bool(best.get("measurement_map_valid"))
            if settings.get("require_measurement_map_validation", True)
            else True
        ),
        "native_two_qubit_gates": (
            bool(best.get("native_two_qubit_gates_valid"))
            if settings.get("require_native_two_qubit_gates", True)
            else True
        ),
        "target_violations_empty": not best.get("target_violations", ["missing"]),
    }
    passed = all(checks.values())
    return GateDecision(
        gate="A_CODE_QPU",
        passed=passed,
        claim_level=(
            ClaimLevel.EMPIRICAL_ATTACK_SUITE
            if passed
            else ClaimLevel.NOT_ESTABLISHED
        ),
        reasons=tuple(key for key, value in checks.items() if not value),
        evidence=checks
        | {
            "backend": compile_report.get("backend"),
            "compiled": best,
            "constraints": settings,
            "acceptance_rule": (
                "live target embedding AND zero-SWAP/native-two-qubit/"
                "measurement-map constraints"
            ),
        },
    )


def gate_b(noise_result: dict, config: dict) -> GateDecision:
    required = config["gates"]["min_noise_pass_probability"]
    probability = noise_result["required_envelope_min_pass_probability"]
    ideal_pass = noise_result["ideal_certificate"]["pass"]
    envelope_points = noise_result["required_envelope_grid_points"]
    passed = bool(ideal_pass and envelope_points > 0 and probability >= required)
    return GateDecision(
        gate="B_NOISE",
        passed=passed,
        claim_level=(
            ClaimLevel.EMPIRICAL_ATTACK_SUITE
            if passed
            else ClaimLevel.NOT_ESTABLISHED
        ),
        reasons=(() if passed else ("noise_robustness_below_target",)),
        evidence={
            "ideal_pass": ideal_pass,
            "required_envelope_min_pass_probability": probability,
            "required_envelope_grid_points": envelope_points,
            "required": required,
            "required_noise_envelope": noise_result["required_noise_envelope"],
            "tv_robust_radius": noise_result["tv_robust_radius"],
            "ideal_method": noise_result.get("ideal_method", "full_statevector"),
        },
    )


def gate_c(candidate: dict, config: dict) -> GateDecision:
    hardness = candidate["hardness"]
    threshold = float(config["gates"]["min_gamma_log10"])
    gamma = float(hardness["gamma_log10"])
    schedule_settings = config.get("schedule_search", {})
    schedule_evidence = candidate.get("exact_artifacts", {}).get(
        "schedule_search", {}
    )
    require_cotengra = bool(schedule_settings.get("require_cotengra", False))
    cotengra = hardness.get("cotengra", {})
    width = float(cotengra.get("contraction_width_log2") or 0.0)
    flops = float(cotengra.get("contraction_flops") or 0.0)
    checks = {
        "gamma": gamma >= threshold,
        "cotengra_available": (
            cotengra.get("status") == "PASS" if require_cotengra else True
        ),
        "cotengra_width": (
            width >= float(schedule_settings.get("target_width_min", 0.0))
            if require_cotengra
            else True
        ),
        "cotengra_flops": (
            flops
            >= float(schedule_settings.get("min_cotengra_flops", 0.0))
            if require_cotengra
            else True
        ),
        "schedule_verifier": (
            bool(
                schedule_evidence.get("verifier_preflight", {}).get(
                    "passed", False
                )
                or schedule_evidence.get("lightcone_preflight", {}).get(
                    "passed", False
                )
            )
            if require_cotengra
            else True
        ),
        "relation_certificate": (
            bool(
                schedule_evidence.get("target_checks", {}).get(
                    "relation_certificate", False
                )
            )
            if require_cotengra
            else True
        ),
        "proxy_zero_swap": (
            bool(
                schedule_evidence.get("target_checks", {}).get(
                    "proxy_zero_swap", False
                )
            )
            if bool(schedule_settings.get("require_proxy_zero_swap", False))
            else True
        ),
    }
    passed = all(checks.values())
    level = (
        ClaimLevel.CONDITIONAL_HARDNESS
        if passed and hardness["assumptions"]
        else ClaimLevel.EMPIRICAL_ATTACK_SUITE
        if passed
        else ClaimLevel.NOT_ESTABLISHED
    )
    return GateDecision(
        gate="C_HARDNESS",
        passed=passed,
        claim_level=level,
        reasons=tuple(key for key, value in checks.items() if not value),
        evidence={
            "checks": checks,
            "gamma_log10": gamma,
            "required": threshold,
            "best_attack_name": hardness.get("best_attack_name"),
            "best_attack_operations": hardness["best_attack_operations"],
            "verify_operations": hardness["verify_operations"],
            "method": hardness.get("method"),
            "cotengra": cotengra,
            "cotengra_targets": {
                "required": require_cotengra,
                "minimum_width_log2": float(
                    schedule_settings.get("target_width_min", 0.0)
                ),
                "minimum_flops": float(
                    schedule_settings.get("min_cotengra_flops", 0.0)
                ),
            },
            "line_graph_treewidth": {
                "lower": hardness.get("line_graph_treewidth_lower"),
                "upper": hardness.get("line_graph_treewidth_upper"),
            },
            "mps": hardness.get("cuts", {}),
            "non_clifford": hardness.get("non_clifford", {}),
            "schedule_search": schedule_evidence,
            "assumptions": hardness["assumptions"],
            "scope": hardness.get(
                "claim_scope",
                "implemented attack suite and declared cost models",
            ),
        },
    )


def qpu_gate(name: str, passed: bool, evidence: dict | None = None) -> GateDecision:
    return GateDecision(
        gate=name,
        passed=passed,
        claim_level=(
            ClaimLevel.EMPIRICAL_ATTACK_SUITE
            if passed
            else ClaimLevel.NOT_ESTABLISHED
        ),
        reasons=(() if passed else ("not_run_or_failed",)),
        evidence=evidence or {},
    )
