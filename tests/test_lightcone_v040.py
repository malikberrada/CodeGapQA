import numpy as np

from codegap_qa.lightcone import local_z_expectation


def test_exact_lightcone_simple_hadamard():
    circuit = {
        "n": 4,
        "gates": [
            {"name": "h", "qubits": [0], "angle": None},
            {"name": "measure", "qubits": [0], "angle": None},
            {"name": "measure", "qubits": [1], "angle": None},
            {"name": "measure", "qubits": [2], "angle": None},
            {"name": "measure", "qubits": [3], "angle": None},
        ],
    }
    result = local_z_expectation(circuit, (0,), max_lightcone_qubits=2)
    assert result["status"] == "EXACT_LIGHTCONE"
    assert abs(result["expectation"]) < 1e-12
    assert result["lightcone_qubits"] == 1
