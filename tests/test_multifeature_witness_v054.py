from __future__ import annotations

import numpy as np

from codegap_qa.witness import fit_multifeature_minimax_bounds


def test_multifeature_witness_has_exact_l1_two_active_and_cap():
    ideal = np.asarray([0.90, 0.72, 0.25, -0.10], dtype=float)
    bounds = {
        "uniform": np.zeros(4),
        "local": np.asarray([0.20, 0.20, 0.20, 0.20]),
    }
    witness, diagnostics = fit_multifeature_minimax_bounds(
        ideal,
        bounds,
        feature_names=("a", "b", "c", "d"),
        minimum_active_features=2,
        support_size=2,
        maximum_absolute_weight=0.8,
        candidate_limit=4,
    )
    absolute = np.abs(witness.weights)
    assert np.isclose(absolute.sum(), 1.0)
    assert np.count_nonzero(absolute > 1e-8) == 2
    assert absolute.max() <= 0.8 + 1e-12
    assert diagnostics["active_features"] == 2
    assert diagnostics["maximum_absolute_weight"] <= 0.8 + 1e-12


def test_multifeature_witness_rejects_infeasible_cap():
    ideal = np.asarray([0.9, 0.8], dtype=float)
    bounds = {"uniform": np.zeros(2)}
    try:
        fit_multifeature_minimax_bounds(
            ideal,
            bounds,
            support_size=2,
            maximum_absolute_weight=0.4,
        )
    except ValueError as error:
        assert "infeasible" in str(error)
    else:
        raise AssertionError("Expected an infeasible cap error")
