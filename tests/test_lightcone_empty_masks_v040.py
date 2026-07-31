import numpy as np

from codegap_qa.lightcone import analytic_feature_means


def test_analytic_feature_means_keeps_2d_empty_masks():
    circuit = {
        "n": 3,
        "gates": [
            {"name": "h", "qubits": [0], "angle": None},
            {"name": "h", "qubits": [1], "angle": None},
            {"name": "h", "qubits": [2], "angle": None},
            {"name": "measure", "qubits": [0], "angle": None},
            {"name": "measure", "qubits": [1], "angle": None},
            {"name": "measure", "qubits": [2], "angle": None},
        ],
    }
    # Every nonempty parity support is rejected by the deliberately tiny cone cap,
    # but single-qubit observables remain exact.
    masks = np.asarray([[1, 1, 0]], dtype=np.uint8)
    result = analytic_feature_means(
        circuit,
        masks,
        max_lightcone_qubits=1,
    )
    assert result["parity_masks"].shape == (0, 3)
    assert result["feature_means"].shape == (4,)
