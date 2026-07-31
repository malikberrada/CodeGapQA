from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .gf2 import row_basis


@dataclass(frozen=True)
class AttackSamples:
    name: str
    samples: np.ndarray
    metadata: dict


def uniform_attack(n: int, shots: int, seed: int) -> AttackSamples:
    rng = np.random.default_rng(seed)
    return AttackSamples(
        "uniform",
        rng.integers(0, 2, size=(shots, n), dtype=np.uint8),
        {"model": "independent unbiased bits"},
    )


def product_attack(
    training_samples: np.ndarray, shots: int, seed: int, smoothing: float = 1.0
) -> AttackSamples:
    rng = np.random.default_rng(seed)
    ones = training_samples.sum(axis=0) + smoothing
    probabilities = ones / (training_samples.shape[0] + 2.0 * smoothing)
    samples = (rng.random((shots, training_samples.shape[1])) < probabilities).astype(
        np.uint8
    )
    return AttackSamples(
        "product_marginals",
        samples,
        {"probabilities": probabilities.tolist(), "smoothing": smoothing},
    )


def markov_attack(
    training_samples: np.ndarray, shots: int, seed: int, order: tuple[int, ...] | None = None
) -> AttackSamples:
    rng = np.random.default_rng(seed)
    n = training_samples.shape[1]
    order = order or tuple(range(n))
    reordered = training_samples[:, order]
    initial = (reordered[:, 0].sum() + 1.0) / (reordered.shape[0] + 2.0)
    transitions = np.zeros((n - 1, 2, 2), dtype=np.float64)
    for position in range(1, n):
        previous = reordered[:, position - 1]
        current = reordered[:, position]
        for prev in (0, 1):
            mask = previous == prev
            denominator = int(mask.sum()) + 2
            for value in (0, 1):
                transitions[position - 1, prev, value] = (
                    int(np.sum(current[mask] == value)) + 1
                ) / denominator
    generated = np.zeros((shots, n), dtype=np.uint8)
    generated[:, 0] = (rng.random(shots) < initial).astype(np.uint8)
    for position in range(1, n):
        previous = generated[:, position - 1]
        p_one = transitions[position - 1, previous, 1]
        generated[:, position] = (rng.random(shots) < p_one).astype(np.uint8)
    inverse = np.argsort(np.asarray(order))
    generated = generated[:, inverse]
    return AttackSamples(
        "first_order_markov",
        generated,
        {"order": list(order), "initial_p1": float(initial)},
    )


def affine_subspace_attack(
    stabilizer_rows: np.ndarray, n: int, shots: int, seed: int
) -> AttackSamples:
    rng = np.random.default_rng(seed)
    basis = row_basis(stabilizer_rows)
    coefficients = rng.integers(
        0, 2, size=(shots, basis.shape[0]), dtype=np.uint8
    )
    samples = (coefficients @ basis) % 2
    if samples.shape[1] != n:
        raise ValueError("Affine subspace basis width mismatch.")
    shift = rng.integers(0, 2, size=n, dtype=np.uint8)
    samples ^= shift
    return AttackSamples(
        "affine_subspace",
        samples,
        {"dimension": int(basis.shape[0]), "shift": shift.tolist()},
    )


def corrupted_ideal_diagnostic(
    ideal_samples: np.ndarray, bit_error: float, seed: int
) -> AttackSamples:
    rng = np.random.default_rng(seed)
    errors = rng.random(ideal_samples.shape) < bit_error
    return AttackSamples(
        "corrupted_ideal_diagnostic",
        ideal_samples ^ errors.astype(np.uint8),
        {"bit_error": bit_error, "role": "diagnostic_not_classical_hardness"},
    )


def build_attack_suite(
    ideal_training: np.ndarray,
    h_x: np.ndarray,
    shots: int,
    seed: int,
) -> list[AttackSamples]:
    n = ideal_training.shape[1]
    return [
        uniform_attack(n, shots, seed + 1),
        product_attack(ideal_training, shots, seed + 2),
        markov_attack(ideal_training, shots, seed + 3),
        affine_subspace_attack(h_x, n, shots, seed + 4),
        corrupted_ideal_diagnostic(ideal_training[:shots], 0.15, seed + 5),
    ]
