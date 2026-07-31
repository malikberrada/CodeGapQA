from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NoisePoint:
    p_1q: float = 0.0
    p_2q: float = 0.0
    p_measure: float = 0.0
    coherent_angle: float = 0.0
    crosstalk: float = 0.0
    drift: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "p_1q": self.p_1q,
            "p_2q": self.p_2q,
            "p_measure": self.p_measure,
            "coherent_angle": self.coherent_angle,
            "crosstalk": self.crosstalk,
            "drift": self.drift,
        }


def apply_noise(
    samples: np.ndarray,
    point: NoisePoint,
    two_qubit_degree: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = np.asarray(samples, dtype=np.uint8).copy()
    shots, n = noisy.shape
    degree = np.asarray(two_qubit_degree, dtype=np.float64)
    if degree.shape != (n,):
        raise ValueError("two_qubit_degree must have shape (n,).")
    normalized_degree = degree / max(1.0, degree.max(initial=1.0))
    coherent_surrogate = np.sin(point.coherent_angle / 2.0) ** 2
    per_qubit = (
        point.p_1q
        + point.p_measure
        + point.p_2q * normalized_degree
        + coherent_surrogate * normalized_degree
    )
    per_qubit = np.clip(per_qubit, 0.0, 0.49)
    errors = rng.random((shots, n)) < per_qubit
    noisy ^= errors.astype(np.uint8)

    if point.crosstalk > 0.0:
        pair_events = rng.random((shots, n - 1)) < point.crosstalk
        for qubit in range(n - 1):
            mask = pair_events[:, qubit]
            noisy[mask, qubit] ^= 1
            noisy[mask, qubit + 1] ^= 1

    if point.drift > 0.0:
        drift_direction = rng.choice([-1.0, 1.0], size=n)
        time = np.linspace(-1.0, 1.0, shots)[:, None]
        drift_probability = np.clip(
            point.drift * (1.0 + time * drift_direction[None, :]),
            0.0,
            0.49,
        )
        drift_errors = rng.random((shots, n)) < drift_probability
        noisy ^= drift_errors.astype(np.uint8)
    return noisy
