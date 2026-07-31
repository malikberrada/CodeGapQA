import numpy as np
from codegap_qa.witness import fit_minimax_witness


def test_minimax_witness_positive_margin():
    ideal = np.array([[1.0, 1.0], [0.8, 1.0], [1.0, 0.8]])
    attacks = {
        "a": np.array([[-1.0, -1.0], [-0.8, -1.0]]),
        "b": np.array([[-1.0, 0.0], [-0.5, 0.0]]),
    }
    witness = fit_minimax_witness(ideal, attacks)
    assert witness.training_margin > 0
    assert abs(witness.weights).sum() <= 1.000001
