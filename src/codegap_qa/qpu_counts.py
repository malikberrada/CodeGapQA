from __future__ import annotations

import re
from typing import Iterable
import numpy as np


def normalize_count_key(key: object, width: int) -> str:
    raw = str(key).replace(' ', '').replace('_', '')
    if raw.lower().startswith('0x'):
        value = int(raw, 16)
        return format(value, f'0{width}b')
    if raw.lower().startswith('0b'):
        raw = raw[2:]
    if not re.fullmatch(r'[01]+', raw):
        raise ValueError(f'Unsupported count key {key!r}.')
    if len(raw) > width:
        raise ValueError(f'Count key {raw!r} exceeds width {width}.')
    return raw.zfill(width)


def bitstring_to_vector(bitstring: str, convention: str) -> np.ndarray:
    if convention == 'qiskit':
        # Qiskit count strings display c[n-1] ... c[0].
        bits = bitstring[::-1]
    elif convention == 'left_to_right':
        bits = bitstring
    else:
        raise ValueError(f'Unknown convention {convention!r}.')
    return np.fromiter((int(bit) for bit in bits), dtype=np.uint8)


def counts_to_samples(
    counts: dict[object, int],
    width: int,
    convention: str,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for key, raw_count in counts.items():
        count = int(raw_count)
        if count < 0:
            raise ValueError('Counts must be non-negative.')
        bitstring = normalize_count_key(key, width)
        vector = bitstring_to_vector(bitstring, convention)
        if count:
            rows.append(np.repeat(vector[None, :], count, axis=0))
    if not rows:
        return np.empty((0, width), dtype=np.uint8)
    return np.vstack(rows)


def expected_pattern_accuracy(
    counts: dict[object, int],
    expected: Iterable[int],
    convention: str,
) -> float:
    expected_vector = np.asarray(tuple(expected), dtype=np.uint8)
    samples = counts_to_samples(counts, expected_vector.size, convention)
    if samples.size == 0:
        return 0.0
    return float(np.mean(np.all(samples == expected_vector[None, :], axis=1)))


def choose_count_convention(
    labeled_counts: dict[str, dict[object, int]],
    expected_patterns: dict[str, tuple[int, ...]],
) -> tuple[str, dict[str, float]]:
    scores: dict[str, float] = {}
    for convention in ('qiskit', 'left_to_right'):
        accuracies = [
            expected_pattern_accuracy(
                labeled_counts[label], expected_patterns[label], convention
            )
            for label in sorted(expected_patterns)
        ]
        scores[convention] = float(np.mean(accuracies)) if accuracies else 0.0
    # Deterministic tie-breaker: documented Qiskit convention.
    winner = max(('qiskit', 'left_to_right'), key=lambda name: (scores[name], name == 'qiskit'))
    return winner, scores
