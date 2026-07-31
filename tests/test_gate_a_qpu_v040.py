from codegap_qa.gates import gate_a_qpu


def test_gate_a_qpu_requires_all_native_guards():
    candidate = {
        "code": {
            "commutes": True,
            "k": 2,
            "d_x_at_least": 4,
            "d_z_at_least": 4,
        }
    }
    config = {
        "constraints": {"min_k": 2, "min_d_x": 4, "min_d_z": 4},
        "qpu": {
            "gate_a_native": {
                "max_swaps": 0,
                "max_nonlocal_edges": 0,
                "require_backend_target_embedding": True,
                "require_measurement_map_validation": True,
                "require_native_two_qubit_gates": True,
            }
        },
    }
    best = {
        "swap_count": 0,
        "nonlocal_edges": 0,
        "target_embedding_valid": True,
        "measurement_map_valid": True,
        "native_two_qubit_gates_valid": True,
        "target_violations": [],
    }
    assert gate_a_qpu(candidate, {"status": "PASS", "best": best}, config).passed
    best["swap_count"] = 1
    assert not gate_a_qpu(candidate, {"status": "PASS", "best": best}, config).passed
