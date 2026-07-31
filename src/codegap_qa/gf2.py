from __future__ import annotations

from itertools import combinations
from math import comb
import numpy as np


def binary(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.uint8) & 1


def rref(matrix: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    a = binary(matrix).copy()
    rows, cols = a.shape
    pivots: list[int] = []
    pivot_row = 0
    for col in range(cols):
        candidates = np.flatnonzero(a[pivot_row:, col])
        if candidates.size == 0:
            continue
        row = pivot_row + int(candidates[0])
        if row != pivot_row:
            a[[pivot_row, row]] = a[[row, pivot_row]]
        for other in range(rows):
            if other != pivot_row and a[other, col]:
                a[other] ^= a[pivot_row]
        pivots.append(col)
        pivot_row += 1
        if pivot_row == rows:
            break
    return a, tuple(pivots)


def rank(matrix: np.ndarray) -> int:
    return len(rref(matrix)[1])


def row_basis(matrix: np.ndarray) -> np.ndarray:
    reduced, pivots = rref(matrix)
    return reduced[: len(pivots)].copy()


def nullspace(matrix: np.ndarray) -> np.ndarray:
    a, pivots = rref(matrix)
    n = a.shape[1]
    free = [col for col in range(n) if col not in pivots]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    for index, free_col in enumerate(free):
        basis[index, free_col] = 1
        for pivot_row, pivot_col in enumerate(pivots):
            basis[index, pivot_col] = a[pivot_row, free_col]
    return basis


def in_rowspace(vector: np.ndarray, matrix: np.ndarray) -> bool:
    base = row_basis(matrix)
    return rank(np.vstack([base, binary(vector)])) == rank(base)


def commute(h_x: np.ndarray, h_z: np.ndarray) -> bool:
    return bool(np.all((binary(h_x) @ binary(h_z).T) % 2 == 0))


def css_k(h_x: np.ndarray, h_z: np.ndarray) -> int:
    return int(h_x.shape[1] - rank(h_x) - rank(h_z))


def _kernel_condition(check: np.ndarray, support: tuple[int, ...]) -> bool:
    if not support:
        return True
    return bool(np.all(check[:, support].sum(axis=1) % 2 == 0))


def css_distance_at_least(
    kernel_check: np.ndarray,
    stabilizer_rows: np.ndarray,
    minimum: int,
) -> tuple[bool, dict[str, int | list[int] | None]]:
    """Prove no nontrivial logical vector has weight below ``minimum``.

    Exhaustively checks every support of weight 1..minimum-1. This is often
    much cheaper than enumerating the full kernel and is an exact certificate.
    """
    n = kernel_check.shape[1]
    inspected = 0
    for weight in range(1, minimum):
        for support in combinations(range(n), weight):
            inspected += 1
            if not _kernel_condition(kernel_check, support):
                continue
            vector = np.zeros(n, dtype=np.uint8)
            vector[list(support)] = 1
            if not in_rowspace(vector, stabilizer_rows):
                return False, {
                    "inspected": inspected,
                    "witness_weight": weight,
                    "witness_support": list(support),
                }
    return True, {
        "inspected": inspected,
        "witness_weight": None,
        "witness_support": None,
    }


def exact_css_distance(
    kernel_check: np.ndarray,
    stabilizer_rows: np.ndarray,
    max_kernel_dimension: int = 22,
) -> int | None:
    basis = nullspace(kernel_check)
    dimension = basis.shape[0]
    if dimension > max_kernel_dimension:
        return None
    best = kernel_check.shape[1] + 1
    for mask in range(1, 1 << dimension):
        vector = np.zeros(kernel_check.shape[1], dtype=np.uint8)
        for bit in range(dimension):
            if (mask >> bit) & 1:
                vector ^= basis[bit]
        weight = int(vector.sum())
        if weight >= best or in_rowspace(vector, stabilizer_rows):
            continue
        best = weight
    return None if best > kernel_check.shape[1] else best


def css_metrics(
    h_x: np.ndarray,
    h_z: np.ndarray,
    min_d_x: int,
    min_d_z: int,
    max_exact_kernel_dimension: int = 22,
) -> dict:
    h_x = binary(h_x)
    h_z = binary(h_z)
    if h_x.shape[1] != h_z.shape[1]:
        raise ValueError("H_X and H_Z must have the same number of columns.")
    commutes = commute(h_x, h_z)
    dx_ok, dx_cert = css_distance_at_least(h_z, h_x, min_d_x)
    dz_ok, dz_cert = css_distance_at_least(h_x, h_z, min_d_z)
    d_x = exact_css_distance(h_z, h_x, max_exact_kernel_dimension) if dx_ok else None
    d_z = exact_css_distance(h_x, h_z, max_exact_kernel_dimension) if dz_ok else None
    return {
        "n": int(h_x.shape[1]),
        "k": css_k(h_x, h_z),
        "rank_x": rank(h_x),
        "rank_z": rank(h_z),
        "commutes": commutes,
        "d_x": d_x,
        "d_z": d_z,
        "d_x_at_least": min_d_x if dx_ok else 0,
        "d_z_at_least": min_d_z if dz_ok else 0,
        "d_x_certificate": dx_cert,
        "d_z_certificate": dz_cert,
        "row_weight_x_max": int(h_x.sum(axis=1).max(initial=0)),
        "row_weight_z_max": int(h_z.sum(axis=1).max(initial=0)),
    }


def _packed_row_basis(matrix: np.ndarray) -> list[int]:
    rows = []
    for row in binary(matrix):
        value = 0
        for index in np.flatnonzero(row):
            value |= 1 << int(index)
        rows.append(value)
    basis: dict[int, int] = {}
    for value in rows:
        current = value
        while current:
            pivot = current.bit_length() - 1
            if pivot in basis:
                current ^= basis[pivot]
            else:
                basis[pivot] = current
                for other_pivot, other in list(basis.items()):
                    if other_pivot != pivot and ((other >> pivot) & 1):
                        basis[other_pivot] = other ^ current
                break
    return [basis[pivot] for pivot in sorted(basis, reverse=True)]


def _packed_in_span(value: int, basis: list[int]) -> bool:
    current = int(value)
    for row in basis:
        pivot = row.bit_length() - 1
        if (current >> pivot) & 1:
            current ^= row
    return current == 0


def css_distance_at_least_fast_small(
    kernel_check: np.ndarray,
    stabilizer_rows: np.ndarray,
    minimum: int,
) -> tuple[bool, dict[str, int | list[int] | None]]:
    """Exact low-weight CSS test optimized for minimum <= 4 and n <= 128.

    The syndrome identities are checked with packed Python integers:
    zero columns detect weight 1, equal columns detect weight 2, and
    ``s_i xor s_j = s_k`` detects weight 3. Every zero-syndrome support is
    still checked against the stabilizer row space exactly.
    """

    if minimum > 4:
        return css_distance_at_least(kernel_check, stabilizer_rows, minimum)
    check = binary(kernel_check)
    n = check.shape[1]
    columns = []
    for column in range(n):
        syndrome = 0
        for row in np.flatnonzero(check[:, column]):
            syndrome |= 1 << int(row)
        columns.append(syndrome)
    stabilizer_basis = _packed_row_basis(stabilizer_rows)

    def logical(support: tuple[int, ...]) -> bool:
        value = sum(1 << index for index in support)
        return not _packed_in_span(value, stabilizer_basis)

    if minimum > 1:
        for index, syndrome in enumerate(columns):
            if syndrome == 0 and logical((index,)):
                return False, {
                    "inspected": sum(comb(n, w) for w in range(1, minimum)),
                    "witness_weight": 1,
                    "witness_support": [index],
                }
    if minimum > 2:
        groups: dict[int, list[int]] = {}
        for index, syndrome in enumerate(columns):
            groups.setdefault(syndrome, []).append(index)
        for indices in groups.values():
            if len(indices) < 2:
                continue
            for left_position in range(len(indices)):
                for right_position in range(left_position + 1, len(indices)):
                    support = (indices[left_position], indices[right_position])
                    if logical(support):
                        return False, {
                            "inspected": sum(comb(n, w) for w in range(1, minimum)),
                            "witness_weight": 2,
                            "witness_support": list(support),
                        }
    if minimum > 3:
        positions: dict[int, list[int]] = {}
        for index, syndrome in enumerate(columns):
            positions.setdefault(syndrome, []).append(index)
        for left in range(n):
            for middle in range(left + 1, n):
                target = columns[left] ^ columns[middle]
                for right in positions.get(target, ()):
                    if right <= middle:
                        continue
                    support = (left, middle, right)
                    if logical(support):
                        return False, {
                            "inspected": sum(comb(n, w) for w in range(1, minimum)),
                            "witness_weight": 3,
                            "witness_support": list(support),
                        }
    return True, {
        "inspected": sum(comb(n, w) for w in range(1, minimum)),
        "witness_weight": None,
        "witness_support": None,
    }
