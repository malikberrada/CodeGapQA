from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class MinimaxWitness:
    weights: np.ndarray
    feature_names: tuple[str, ...]
    training_margin: float
    adversary_means: dict[str, float]
    ideal_mean: float

    def evaluate_features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64) @ self.weights
        return np.clip(values, -1.0, 1.0)

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "feature_names": list(self.feature_names),
            "training_margin": self.training_margin,
            "adversary_means": self.adversary_means,
            "ideal_mean": self.ideal_mean,
            "l1_norm": float(np.abs(self.weights).sum()),
            "active_features": int(np.count_nonzero(np.abs(self.weights) > 1e-8)),
            "maximum_absolute_weight": float(np.max(np.abs(self.weights), initial=0.0)),
            "bounded_by": 1.0,
        }


def fit_minimax_witness(
    ideal_features: np.ndarray,
    adversary_features: dict[str, np.ndarray],
    feature_names: tuple[str, ...] | None = None,
) -> MinimaxWitness:
    ideal_mean_vector = ideal_features.mean(axis=0)
    adversary_mean_vectors = {
        name: features.mean(axis=0) for name, features in adversary_features.items()
    }
    dimension = ideal_mean_vector.shape[0]
    # Variables: w_plus[0:d], w_minus[d:2d], margin t.
    objective = np.zeros(2 * dimension + 1)
    objective[-1] = -1.0
    a_ub = []
    b_ub = []
    for mean in adversary_mean_vectors.values():
        difference = ideal_mean_vector - mean
        row = np.zeros(2 * dimension + 1)
        row[:dimension] = -difference
        row[dimension : 2 * dimension] = difference
        row[-1] = 1.0
        a_ub.append(row)
        b_ub.append(0.0)
    l1_row = np.zeros(2 * dimension + 1)
    l1_row[: 2 * dimension] = 1.0
    a_ub.append(l1_row)
    b_ub.append(1.0)
    bounds = [(0.0, None)] * (2 * dimension) + [(-2.0, 2.0)]
    result = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Minimax witness LP failed: {result.message}")
    weights = result.x[:dimension] - result.x[dimension : 2 * dimension]
    ideal_mean = float(ideal_mean_vector @ weights)
    adversary_means = {
        name: float(mean @ weights)
        for name, mean in adversary_mean_vectors.items()
    }
    margin = ideal_mean - max(adversary_means.values())
    names = feature_names or tuple(f"feature_{index}" for index in range(dimension))
    return MinimaxWitness(
        weights=weights,
        feature_names=names,
        training_margin=float(margin),
        adversary_means=adversary_means,
        ideal_mean=ideal_mean,
    )


def empirical_bernstein_lcb(
    values: np.ndarray, alpha: float, bound: float = 1.0
) -> float:
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n < 2:
        return float("-inf")
    variance = float(values.var(ddof=1))
    radius = math.sqrt(2.0 * variance * math.log(3.0 / alpha) / n)
    radius += 3.0 * (2.0 * bound) * math.log(3.0 / alpha) / n
    return float(values.mean() - radius)


def certificate_margin_lcb(
    witness: MinimaxWitness,
    observed_features: np.ndarray,
    alpha: float,
    adversary_generalization_penalty: float,
) -> dict:
    observed_values = witness.evaluate_features(observed_features)
    observed_lcb = empirical_bernstein_lcb(observed_values, alpha)
    adversary_supremum = max(witness.adversary_means.values())
    margin_lcb = (
        observed_lcb - adversary_supremum - adversary_generalization_penalty
    )
    return {
        "shots": int(observed_values.size),
        "observed_mean": float(observed_values.mean()),
        "observed_lcb": observed_lcb,
        "adversary_supremum_training": adversary_supremum,
        "adversary_generalization_penalty": adversary_generalization_penalty,
        "margin_lcb": float(margin_lcb),
        "pass": bool(margin_lcb > 0.0),
    }


def tv_robust_radius(ideal_margin: float) -> float:
    # For |f| <= 1, each expectation changes by at most 2*TV.
    return max(0.0, ideal_margin / 2.0)


def fit_minimax_feature_bounds(
    ideal_mean_vector: np.ndarray,
    adversary_absolute_bounds: dict[str, np.ndarray],
    feature_names: tuple[str, ...] | None = None,
) -> MinimaxWitness:
    """Fit a bounded linear witness against analytic feature-wise bounds.

    For adversary ``a`` with ``|E_a[phi_j]| <= b[a,j]``, the worst-case
    witness expectation is bounded by ``sum_j |w_j| b[a,j]``. The LP maximizes
    the ideal expectation minus every such registered upper bound while
    enforcing ``||w||_1 <= 1``, which guarantees ``|f(x)| <= 1`` whenever all
    features lie in [-1, 1].
    """

    ideal = np.asarray(ideal_mean_vector, dtype=np.float64)
    dimension = ideal.shape[0]
    objective = np.zeros(2 * dimension + 1, dtype=np.float64)
    objective[-1] = -1.0
    rows = []
    rhs = []
    for bounds in adversary_absolute_bounds.values():
        bound = np.asarray(bounds, dtype=np.float64)
        if bound.shape != ideal.shape:
            raise ValueError("Analytic adversary bound has the wrong dimension.")
        row = np.zeros(2 * dimension + 1, dtype=np.float64)
        # -(ideal @ (w+ - w-)) + bound @ (w+ + w-) + t <= 0
        row[:dimension] = -ideal + bound
        row[dimension : 2 * dimension] = ideal + bound
        row[-1] = 1.0
        rows.append(row)
        rhs.append(0.0)
    l1 = np.zeros(2 * dimension + 1, dtype=np.float64)
    l1[: 2 * dimension] = 1.0
    rows.append(l1)
    rhs.append(1.0)
    result = linprog(
        objective,
        A_ub=np.asarray(rows),
        b_ub=np.asarray(rhs),
        bounds=[(0.0, None)] * (2 * dimension) + [(-2.0, 2.0)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Analytic minimax witness LP failed: {result.message}")
    weights = result.x[:dimension] - result.x[dimension : 2 * dimension]
    ideal_mean = float(ideal @ weights)
    adversary_means = {
        name: float(np.asarray(bounds, dtype=np.float64) @ np.abs(weights))
        for name, bounds in adversary_absolute_bounds.items()
    }
    margin = ideal_mean - max(adversary_means.values(), default=0.0)
    names = feature_names or tuple(f"feature_{index}" for index in range(dimension))
    return MinimaxWitness(
        weights=weights,
        feature_names=names,
        training_margin=float(margin),
        adversary_means=adversary_means,
        ideal_mean=ideal_mean,
    )



def fit_multifeature_minimax_bounds(
    ideal_mean_vector: np.ndarray,
    adversary_absolute_bounds: dict[str, np.ndarray],
    feature_names: tuple[str, ...] | None = None,
    *,
    minimum_active_features: int = 2,
    support_size: int | None = None,
    maximum_absolute_weight: float = 0.8,
    candidate_limit: int = 16,
) -> tuple[MinimaxWitness, dict]:
    """Fit a sparse, non-degenerate analytic minimax witness.

    The support and signs are enumerated. For every signed support, a small LP
    optimizes non-negative magnitudes ``a`` subject to ``sum(a)=1`` and
    ``a_i <= maximum_absolute_weight``. The resulting witness therefore has
    exact L1 norm one, at least ``minimum_active_features`` non-zero entries,
    and no coefficient larger than the configured cap.
    """

    from itertools import combinations, product

    ideal = np.asarray(ideal_mean_vector, dtype=np.float64)
    dimension = int(ideal.shape[0])
    names = feature_names or tuple(f"feature_{index}" for index in range(dimension))
    if dimension == 0:
        raise ValueError("At least one feature is required.")
    minimum_active_features = int(minimum_active_features)
    support_size = int(support_size or minimum_active_features)
    if support_size < minimum_active_features:
        raise ValueError("support_size must be >= minimum_active_features.")
    if support_size > dimension:
        raise ValueError("support_size exceeds the available feature dimension.")
    cap = float(maximum_absolute_weight)
    if not (0.0 < cap <= 1.0):
        raise ValueError("maximum_absolute_weight must lie in (0, 1].")
    if support_size * cap < 1.0 - 1e-12:
        raise ValueError("The coefficient cap makes exact L1 norm one infeasible.")

    normalized_bounds: dict[str, np.ndarray] = {}
    for name, values in adversary_absolute_bounds.items():
        bound = np.asarray(values, dtype=np.float64)
        if bound.shape != ideal.shape:
            raise ValueError("Analytic adversary bound has the wrong dimension.")
        normalized_bounds[name] = bound
    strongest = (
        np.max(np.vstack(list(normalized_bounds.values())), axis=0)
        if normalized_bounds
        else np.zeros_like(ideal)
    )
    solo_scores = np.abs(ideal) - strongest
    limit = max(support_size, min(int(candidate_limit), dimension))
    candidate_indices = np.argsort(solo_scores)[::-1][:limit].tolist()

    best = None
    best_key = None
    lp_solves = 0
    supports_examined = 0
    lower = 1e-9
    for support_tuple in combinations(candidate_indices, support_size):
        support = np.asarray(support_tuple, dtype=int)
        supports_examined += 1
        for signs_tuple in product((-1.0, 1.0), repeat=support_size):
            signs = np.asarray(signs_tuple, dtype=np.float64)
            objective = np.zeros(support_size + 1, dtype=np.float64)
            objective[-1] = -1.0
            rows = []
            rhs = []
            for bound in normalized_bounds.values():
                # t <= sum_j a_j * (sign_j * ideal_j - bound_j)
                row = np.zeros(support_size + 1, dtype=np.float64)
                row[:support_size] = -(signs * ideal[support] - bound[support])
                row[-1] = 1.0
                rows.append(row)
                rhs.append(0.0)
            result = linprog(
                objective,
                A_ub=np.asarray(rows) if rows else None,
                b_ub=np.asarray(rhs) if rhs else None,
                A_eq=np.asarray([[1.0] * support_size + [0.0]]),
                b_eq=np.asarray([1.0]),
                bounds=[(lower, cap)] * support_size + [(-2.0, 2.0)],
                method="highs",
            )
            lp_solves += 1
            if not result.success:
                continue
            magnitudes = result.x[:support_size]
            weights = np.zeros(dimension, dtype=np.float64)
            weights[support] = signs * magnitudes
            active = int(np.count_nonzero(np.abs(weights) > 1e-8))
            max_abs = float(np.max(np.abs(weights), initial=0.0))
            ideal_mean = float(ideal @ weights)
            adversary_means = {
                name: float(bound @ np.abs(weights))
                for name, bound in normalized_bounds.items()
            }
            margin = ideal_mean - max(adversary_means.values(), default=0.0)
            # Prefer margin first, then less concentration, then deterministic support.
            key = (
                float(margin),
                -max_abs,
                -float(np.square(np.abs(weights)).sum()),
                tuple(-int(value) for value in support_tuple),
            )
            if best is None or key > best_key:
                best_key = key
                best = (weights, ideal_mean, adversary_means, margin, support_tuple)
    if best is None:
        raise RuntimeError("No feasible multifeature minimax witness was found.")

    weights, ideal_mean, adversary_means, margin, support_tuple = best
    witness = MinimaxWitness(
        weights=weights,
        feature_names=names,
        training_margin=float(margin),
        adversary_means=adversary_means,
        ideal_mean=float(ideal_mean),
    )
    diagnostics = {
        "method": "signed_support_enumeration_lp",
        "l1_norm": float(np.abs(weights).sum()),
        "active_features": int(np.count_nonzero(np.abs(weights) > 1e-8)),
        "maximum_absolute_weight": float(np.max(np.abs(weights), initial=0.0)),
        "support_indices": [int(value) for value in np.flatnonzero(np.abs(weights) > 1e-8)],
        "support_names": [names[int(value)] for value in np.flatnonzero(np.abs(weights) > 1e-8)],
        "requested_minimum_active_features": minimum_active_features,
        "requested_support_size": support_size,
        "requested_maximum_absolute_weight": cap,
        "candidate_limit": limit,
        "supports_examined": supports_examined,
        "lp_solves": lp_solves,
    }
    return witness, diagnostics


def fit_configured_feature_bounds(
    ideal_mean_vector: np.ndarray,
    adversary_absolute_bounds: dict[str, np.ndarray],
    feature_names: tuple[str, ...] | None,
    settings: dict,
) -> tuple[MinimaxWitness, dict]:
    mode = str(settings.get("witness_optimizer", "unconstrained_lp")).lower()
    if mode in {"multifeature", "multifeature_pair_lp", "signed_support_lp"}:
        return fit_multifeature_minimax_bounds(
            ideal_mean_vector,
            adversary_absolute_bounds,
            feature_names=feature_names,
            minimum_active_features=int(settings.get("min_active_witness_features", 2)),
            support_size=int(settings.get("witness_support_size", 2)),
            maximum_absolute_weight=float(settings.get("max_abs_witness_weight", 0.8)),
            candidate_limit=int(settings.get("witness_candidate_limit", 16)),
        )
    witness = fit_minimax_feature_bounds(
        ideal_mean_vector,
        adversary_absolute_bounds,
        feature_names=feature_names,
    )
    weights = np.abs(witness.weights)
    return witness, {
        "method": "unconstrained_lp",
        "l1_norm": float(weights.sum()),
        "active_features": int(np.count_nonzero(weights > 1e-8)),
        "maximum_absolute_weight": float(weights.max(initial=0.0)),
        "support_indices": [int(value) for value in np.flatnonzero(weights > 1e-8)],
        "support_names": [
            witness.feature_names[int(value)] for value in np.flatnonzero(weights > 1e-8)
        ],
    }

def bounded_mean_lcb(
    mean: float,
    shots: int,
    alpha: float,
) -> float:
    """Hoeffding lower bound for a random variable in [-1, 1]."""

    if shots <= 0:
        return float("-inf")
    radius = math.sqrt(2.0 * math.log(1.0 / alpha) / shots)
    return float(mean - radius)
