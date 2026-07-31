from __future__ import annotations

from codegap_qa.lightcone import local_z_expectation


def test_local_z_expectation_uses_msb_tensor_axis_order() -> None:
    spec = {
        "n": 2,
        "gates": [
            {"name": "h", "qubits": [0], "layer": -1},
            {"name": "rzz", "qubits": [0, 1], "angle": 0.37, "layer": 0},
            {"name": "measure", "qubits": [0], "classical": 0, "layer": 1},
            {"name": "measure", "qubits": [1], "classical": 1, "layer": 1},
        ],
    }
    z0 = local_z_expectation(spec, (0,), max_lightcone_qubits=2, backend="numpy")
    z1 = local_z_expectation(spec, (1,), max_lightcone_qubits=2, backend="numpy")
    assert abs(float(z0["expectation"])) < 1.0e-12
    assert abs(float(z1["expectation"]) - 1.0) < 1.0e-12
