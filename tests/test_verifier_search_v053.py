from __future__ import annotations

import numpy as np

from codegap_qa.features import FeatureMap
from codegap_qa.verifier_gpu import batch_backward_lightcones
from codegap_qa.verifier_search import select_verifiable_observables


def _simple_circuit() -> dict:
    gates = []
    for qubit in range(6):
        gates.append({"name": "h", "qubits": [qubit], "angle": None, "layer": -1})
    # Qubits 0-1 form a protected local sector; 2-5 scramble separately.
    for layer in range(4):
        gates.append({"name": "rzz", "qubits": [0, 1], "angle": 0.3, "layer": layer})
        gates.append({"name": "rxx", "qubits": [2, 3], "angle": 0.4, "layer": layer})
        gates.append({"name": "rxx", "qubits": [4, 5], "angle": 0.4, "layer": layer})
    return {
        "n": 6,
        "gates": gates,
        "layers": [],
        "verifier_masks": [[1, 1, 0, 0, 0, 0]],
        "schedule_metadata": {"protected_masks": [[1, 1, 0, 0, 0, 0]]},
    }


def test_packed_batch_lightcones_match_expected_sector():
    circuit = _simple_circuit()
    masks = np.asarray(
        [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=np.uint8
    )
    result = batch_backward_lightcones(circuit, masks, backend="auto")
    assert result["sizes"].tolist() == [2, 2]
    assert result["cones"][0].tolist() == [1, 1, 0, 0, 0, 0]


def test_feature_map_supports_parity_only_selected_observables():
    fmap = FeatureMap(
        parity_masks=np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
        heavy_indices=(),
        n=3,
        bit_indices=(),
        include_centered_weight=False,
    )
    transformed = fmap.transform(np.asarray([[0, 0, 0], [1, 0, 1]], dtype=np.uint8))
    assert transformed.shape == (2, 2)
    assert fmap.dimension == 2


def test_verifier_search_does_not_require_all_single_bits():
    circuit = _simple_circuit()
    config = {
        "acceleration": {"backend": "auto"},
        "verifier_search": {
            "max_lightcone_qubits": 2,
            "max_candidate_masks": 32,
            "candidate_max_weight": 2,
            "max_exact_observables_per_schedule": 16,
            "max_selected_observables": 4,
            "min_abs_expectation": 0.0,
            "min_local_observables": 1,
            "min_local_rank": 1,
            "min_witness_margin_lcb": -1.0,
            "min_coverage_fraction": 0.0,
            "min_protected_checks": 1,
            "gpu_backend": "auto",
            "exact_backend": "auto",
        },
        "certificate": {
            "max_lightcone_qubits": 2,
            "alpha": 0.01,
            "adversary_generalization_penalty": 0.0,
            "analytic_adversaries": {
                "max_abs_bit_bias": 0.15,
                "max_abs_local_correlation": 0.2,
            },
        },
        "noise": {"test_shots": 20000},
    }
    result = select_verifiable_observables(circuit, config)
    assert result["local_observables"] >= 1
    assert result["maximum_lightcone_qubits"] <= 2
    assert result["protected_checks"] >= 1
