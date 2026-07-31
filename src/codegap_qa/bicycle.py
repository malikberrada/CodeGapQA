from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterable, Iterator
import numpy as np

from .gf2 import commute


GroupElement = tuple[int, int]


@dataclass(frozen=True)
class BicycleFamilySpec:
    l: int
    m: int
    support_a: tuple[GroupElement, ...]
    support_b: tuple[GroupElement, ...]

    @property
    def block_size(self) -> int:
        return self.l * self.m

    @property
    def n(self) -> int:
        return 2 * self.block_size

    def to_dict(self) -> dict:
        return {
            "type": "bivariate_bicycle",
            "l": self.l,
            "m": self.m,
            "support_a": [list(item) for item in self.support_a],
            "support_b": [list(item) for item in self.support_b],
            "n": self.n,
        }


def group_elements(l: int, m: int) -> tuple[GroupElement, ...]:
    return tuple((x, y) for x in range(l) for y in range(m))


def translation_matrix(l: int, m: int, dx: int, dy: int) -> np.ndarray:
    size = l * m
    matrix = np.zeros((size, size), dtype=np.uint8)
    for x in range(l):
        for y in range(m):
            source = x * m + y
            target = ((x + dx) % l) * m + ((y + dy) % m)
            matrix[target, source] = 1
    return matrix


def polynomial_matrix(
    l: int, m: int, support: Iterable[GroupElement]
) -> np.ndarray:
    matrix = np.zeros((l * m, l * m), dtype=np.uint8)
    for dx, dy in support:
        matrix ^= translation_matrix(l, m, dx, dy)
    return matrix


def build_bicycle_css(spec: BicycleFamilySpec) -> tuple[np.ndarray, np.ndarray]:
    a = polynomial_matrix(spec.l, spec.m, spec.support_a)
    b = polynomial_matrix(spec.l, spec.m, spec.support_b)
    h_x = np.hstack([a, b]).astype(np.uint8)
    h_z = np.hstack([b.T, a.T]).astype(np.uint8)
    if not commute(h_x, h_z):
        raise AssertionError("Bicycle construction must satisfy CSS commutation.")
    return h_x, h_z


def canonical_support(
    support: Iterable[GroupElement],
) -> tuple[GroupElement, ...]:
    return tuple(sorted(set(support)))


def support_space_size(l: int, m: int, weight_a: int, weight_b: int) -> int:
    size = l * m
    return comb(size, weight_a) * comb(size, weight_b)


def enumerate_specs(
    l: int,
    m: int,
    weight_a: int,
    weight_b: int,
    max_candidates: int,
) -> tuple[Iterator[BicycleFamilySpec], bool, int]:
    elements = group_elements(l, m)
    total = support_space_size(l, m, weight_a, weight_b)
    complete = total <= max_candidates

    def iterator() -> Iterator[BicycleFamilySpec]:
        emitted = 0
        for support_a in combinations(elements, weight_a):
            for support_b in combinations(elements, weight_b):
                if emitted >= max_candidates:
                    return
                emitted += 1
                yield BicycleFamilySpec(
                    l=l,
                    m=m,
                    support_a=canonical_support(support_a),
                    support_b=canonical_support(support_b),
                )

    return iterator(), complete, total


def random_specs(
    l: int,
    m: int,
    weight_a: int,
    weight_b: int,
    count: int,
    seed: int,
) -> Iterator[BicycleFamilySpec]:
    rng = np.random.default_rng(seed)
    elements = group_elements(l, m)
    seen: set[tuple] = set()
    while len(seen) < count:
        indices_a = tuple(
            sorted(rng.choice(len(elements), weight_a, replace=False).tolist())
        )
        indices_b = tuple(
            sorted(rng.choice(len(elements), weight_b, replace=False).tolist())
        )
        key = (indices_a, indices_b)
        if key in seen:
            continue
        seen.add(key)
        yield BicycleFamilySpec(
            l=l,
            m=m,
            support_a=tuple(elements[index] for index in indices_a),
            support_b=tuple(elements[index] for index in indices_b),
        )
