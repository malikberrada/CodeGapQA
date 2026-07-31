from codegap_qa.spacetime import spacetime_hardness_metrics


def test_space_time_network_metrics_are_present():
    circuit = {
        "n": 2,
        "gates": [
            {"name": "h", "qubits": [0], "angle": None},
            {"name": "h", "qubits": [1], "angle": None},
            {"name": "rzz", "qubits": [0, 1], "angle": 0.3},
            {"name": "h", "qubits": [0], "angle": None},
            {"name": "h", "qubits": [1], "angle": None},
            {"name": "measure", "qubits": [0], "angle": None},
            {"name": "measure", "qubits": [1], "angle": None},
        ],
    }
    metrics = spacetime_hardness_metrics(
        circuit_spec=circuit,
        verify_operations=8,
        assumptions=("test",),
        settings={"cotengra": {"enabled": False}},
    )
    assert metrics["method"] == "full_doubled_space_time_tensor_network"
    assert metrics["tensor_count"] > 0
    assert metrics["line_graph_treewidth_upper"] >= 0
    assert metrics["best_attack_operations"] > 0
