from __future__ import annotations

import json
from pathlib import Path

from codegap_qa.freeze import freeze, verify_freeze
from codegap_qa.promotion import promote_recovered_artifact


def _candidate(candidate_id: str, fingerprint: str, gamma: float) -> dict:
    preflight = {
        "schema": "codegap.verifier-preflight.v1",
        "passed": True,
        "local_observables": 4,
        "local_rank": 4,
        "exact_local_certified_checks": 4,
        "active_witness_features": 2,
        "maximum_witness_coefficient": 0.8,
        "maximum_lightcone_qubits": 4,
        "witness_margin_lcb": 0.25,
        "verify_operations": 8.0,
        "active_verify_operations": 2.0,
        "audited_verify_operations": 8.0,
        "witness": {
            "weights": [-0.8, 0.2, 0.0, 0.0],
            "feature_names": ["a", "b", "c", "d"],
            "l1_norm": 1.0,
        },
    }
    return {
        "candidate_id": candidate_id,
        "objective": gamma,
        "code": {
            "n": 4,
            "commutes": True,
            "k": 2,
            "d_x_at_least": 4,
            "d_z_at_least": 4,
        },
        "hardware": {
            "two_qubit_count": 2,
            "two_qubit_depth": 1,
            "swap_count": 0,
            "nonlocal_edges": 0,
        },
        "hardness": {
            "gamma_log10": gamma,
            "best_attack_name": "registered_test_attack",
            "best_attack_operations": 1.0e9,
            "verify_operations": 8.0,
            "method": "test",
            "assumptions": ["registered test assumption"],
            "claim_scope": "test attack suite",
            "cotengra": {
                "status": "PASS",
                "contraction_width_log2": 20.0,
                "contraction_flops": 1.0e10,
            },
            "cuts": {},
            "non_clifford": {},
        },
        "exact_artifacts": {
            "circuit_spec": {
                "n": 4,
                "gates": [
                    {"name": "h", "qubits": [0]},
                    {"name": "measure", "qubits": [0]},
                    {"name": "measure", "qubits": [1]},
                    {"name": "measure", "qubits": [2]},
                    {"name": "measure", "qubits": [3]},
                ],
                "verifier_masks": [[1, 0, 0, 0]],
                "verifier_cost_model": "test",
            },
            "schedule_search": {
                "target_structural_fingerprint": fingerprint,
                "target_checks": {
                    "relation_certificate": True,
                    "proxy_zero_swap": True,
                    "live_target_zero_swap": True,
                },
                "verifier_preflight": preflight,
                "lightcone_preflight": preflight,
            },
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _artifacts(tmp_path: Path) -> tuple[Path, Path, str, str]:
    source = tmp_path / "source"
    recovery = tmp_path / "recovery"
    source.mkdir()
    recovery.mkdir()
    fingerprint = "test-structural-fingerprint"
    primary_id = "primary"
    backup_id = "backup"
    config = {
        "seed": 7,
        "_config_dir": str(source),
        "_config_path": str(source / "resolved_config.json"),
        "constraints": {
            "min_k": 2,
            "min_d_x": 4,
            "min_d_z": 4,
            "max_two_qubit_count": 10,
            "max_two_qubit_depth": 4,
            "max_swaps": 0,
            "max_nonlocal_edges": 0,
        },
        "hardware": {"target_snapshot": "target_snapshot.json"},
        "gates": {"min_gamma_log10": 6.0},
        "schedule_search": {
            "require_cotengra": True,
            "target_width_min": 17.0,
            "min_cotengra_flops": 192000000.0,
            "require_proxy_zero_swap": True,
        },
        "hardness": {"assumptions": ["registered test assumption"]},
        "progress": {"enabled": False},
    }
    snapshot = {
        "schema": "codegap.target-snapshot.v1",
        "structural_fingerprint": fingerprint,
        "calibration_fingerprint": "calibration",
        "num_qubits": 4,
        "coupling_edges": [],
    }
    optimality = {
        "search_complete": False,
        "claim": "TEST_SEARCHED_QUOTIENT",
    }
    primary = _candidate(primary_id, fingerprint, 8.0)
    backup = _candidate(backup_id, fingerprint, 7.5)
    _write_json(source / "resolved_config.json", config)
    _write_json(source / "target_snapshot.json", snapshot)
    _write_json(source / "codeforge" / "optimality_certificate.json", optimality)
    _write_json(
        source / "codeforge" / "hardware_code_frontier.json",
        [primary, backup],
    )
    freeze(source)

    def record(candidate: dict, index: int) -> dict:
        return {
            "candidate_id": candidate["candidate_id"],
            "n": 4,
            "verifier_preflight": candidate["exact_artifacts"]
            ["schedule_search"]["verifier_preflight"],
            "B_NOISE": {
                "gate": "B_NOISE",
                "passed": True,
                "claim_level": "EMPIRICAL_ATTACK_SUITE",
                "reasons": [],
                "evidence": {
                    "required_envelope_min_pass_probability": 1.0
                },
            },
            "C_HARDNESS": {
                "gate": "C_HARDNESS",
                "passed": True,
                "claim_level": "CONDITIONAL_HARDNESS",
                "reasons": [],
                "evidence": {},
            },
            "fully_qualified_classically": True,
            "priority_index": index,
        }

    report = {
        "schema": "codegap.verifier-recovery.v2-multifeature",
        "source_artifact": str(source.resolve()),
        "status": "RECOVERED_CANDIDATE_FOUND",
        "recommended_candidate_id": primary_id,
        "backup_candidate_id": backup_id,
        "records": [record(primary, 0), record(backup, 1)],
    }
    _write_json(recovery / "verifier_recovery_report.json", report)
    _write_json(
        recovery / "recovered_hardware_code_frontier.json",
        [primary, backup],
    )
    return source, recovery, primary_id, backup_id


def test_promote_recovery_creates_qpu_ready_frozen_artifact(tmp_path: Path):
    source, recovery, primary_id, backup_id = _artifacts(tmp_path)
    output = tmp_path / "promoted"
    result = promote_recovered_artifact(
        artifact=source,
        recovery=recovery,
        output=output,
    )
    assert result["status"] == "READY_FOR_QPU_NATIVE_PREPARATION"
    assert result["selected_candidate_id"] == primary_id
    assert result["backup_candidate_id"] == backup_id
    assert result["qpu_preparation_authorized"] is True
    assert result["qpu_submission_authorized"] is False
    assert result["freeze_verified"] is True
    assert verify_freeze(output / "freeze_manifest.json")["ok"] is True

    report = json.loads((output / "gate_report.json").read_text())
    assert report["selected_candidate_id"] == primary_id
    assert report["qpu_preparation_authorized"] is True
    assert report["qpu_submission_authorized"] is False
    frontier = json.loads(
        (output / "codeforge" / "hardware_code_frontier.json").read_text()
    )
    assert [item["candidate_id"] for item in frontier] == [primary_id, backup_id]
    assert all(
        item["exact_artifacts"]["schedule_search"]["target_pass"]
        for item in frontier
    )


def test_promote_recovery_rejects_target_fingerprint_mismatch(tmp_path: Path):
    source, recovery, _, _ = _artifacts(tmp_path)
    snapshot_path = source / "target_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["structural_fingerprint"] = "different-target"
    _write_json(snapshot_path, snapshot)
    freeze(source)
    try:
        promote_recovered_artifact(
            artifact=source,
            recovery=recovery,
            output=tmp_path / "promoted",
        )
    except RuntimeError as error:
        assert "different target structure" in str(error)
    else:
        raise AssertionError("Expected target fingerprint mismatch rejection")


def test_qpu_prepare_order_respects_promoted_primary():
    from codegap_qa.qpu_workflow import _order_candidates_for_preparation

    frontier = [
        {
            "candidate_id": "backup",
            "objective": 100.0,
            "code": {"n": 48},
            "hardness": {"gamma_log10": 10.2},
        },
        {
            "candidate_id": "primary",
            "objective": 1.0,
            "code": {"n": 48},
            "hardness": {"gamma_log10": 10.0},
        },
    ]
    ordered = _order_candidates_for_preparation(
        frontier,
        {
            "selected_candidate_id": "primary",
            "backup_candidate_id": "backup",
        },
        6.0,
    )
    assert [item["candidate_id"] for item in ordered] == ["primary", "backup"]
