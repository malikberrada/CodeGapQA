from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FeatureMap:
    parity_masks: np.ndarray
    heavy_indices: tuple[int, ...]
    n: int
    bit_indices: tuple[int, ...] | None = None
    include_centered_weight: bool = True

    @property
    def resolved_bit_indices(self) -> tuple[int, ...]:
        return tuple(range(self.n)) if self.bit_indices is None else tuple(self.bit_indices)

    @property
    def dimension(self) -> int:
        return int(
            len(self.resolved_bit_indices)
            + self.parity_masks.shape[0]
            + len(self.heavy_indices)
            + int(self.include_centered_weight)
        )

    def transform(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.uint8)
        if samples.ndim != 2 or samples.shape[1] != self.n:
            raise ValueError("Sample matrix has the wrong shape.")
        all_bit_features = 1.0 - 2.0 * samples.astype(np.float64)
        bit_features = all_bit_features[:, self.resolved_bit_indices]
        parity = (
            samples.astype(np.uint8) @ self.parity_masks.T.astype(np.uint8)
        ) % 2
        parity_features = 1.0 - 2.0 * parity.astype(np.float64)
        if self.heavy_indices:
            if self.n > 63:
                raise ValueError(
                    "Heavy-output indicator features require n <= 63. "
                    "Use parity/light-cone witnesses for larger circuits."
                )
            integer_indices = (
                samples.astype(np.uint64)
                * (np.uint64(1) << np.arange(self.n, dtype=np.uint64))
            ).sum(axis=1)
            heavy_features = np.column_stack(
                [
                    np.where(integer_indices == index, 1.0, -1.0)
                    for index in self.heavy_indices
                ]
            )
        else:
            heavy_features = np.empty((samples.shape[0], 0))
        centered_weight = (
            2.0 * samples.mean(axis=1, keepdims=True) - 1.0
            if self.include_centered_weight
            else np.empty((samples.shape[0], 0), dtype=np.float64)
        )
        return np.hstack(
            [bit_features, parity_features, heavy_features, centered_weight]
        )


def build_feature_map(
    h_x: np.ndarray,
    h_z: np.ndarray,
    probabilities: np.ndarray,
    heavy_count: int = 8,
    max_parities: int = 64,
) -> FeatureMap:
    masks = np.vstack([h_x, h_z]).astype(np.uint8)
    if masks.shape[0] > max_parities:
        weights = masks.sum(axis=1)
        order = np.argsort(weights, kind="stable")
        masks = masks[order[:max_parities]]
    heavy = tuple(
        int(index)
        for index in np.argsort(probabilities)[-heavy_count:][::-1]
    )
    return FeatureMap(
        parity_masks=masks,
        heavy_indices=heavy,
        n=h_x.shape[1],
    )
