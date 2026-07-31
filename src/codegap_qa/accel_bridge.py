from __future__ import annotations

import os
from typing import Any, Iterable, Sequence

import numpy as np


MARKER = "CODEGAP_V100_RAGGED_CUDA_BATCHING"


def _accel_module():
    try:
        import codegap_accel

        return codegap_accel
    except Exception:
        return None


def available() -> bool:
    module = _accel_module()
    return bool(module is not None and module.native_available())


def diagnostics() -> dict[str, Any]:
    module = _accel_module()
    if module is None:
        return {
            "available": False,
            "reason": "codegap_accel_import_failed",
        }
    payload = dict(module.backend_diagnostics())
    payload["available"] = bool(payload.get("native_available"))
    return payload


def _flatten_index(
    coordinates: Sequence[int],
    dimensions: Sequence[int],
) -> int:
    value = 0
    for coordinate, dimension in zip(coordinates, dimensions):
        value = value * int(dimension) + int(coordinate)
    return int(value)


def screen_abelian_specs(
    specs: Sequence[Any],
    *,
    dimensions: Sequence[int],
    min_dx: int,
    min_dz: int,
    min_k: int,
    acceleration: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    module = _accel_module()
    if module is None or not specs:
        return None
    settings = acceleration or {}
    if not bool(settings.get("native_family_screen", True)):
        return None
    dims = tuple(int(value) for value in dimensions)
    supports_a = [
        tuple(_flatten_index(item, dims) for item in spec.support_a)
        for spec in specs
    ]
    supports_b = [
        tuple(_flatten_index(item, dims) for item in spec.support_b)
        for spec in specs
    ]
    ragged_screen = getattr(
        module,
        "screen_abelian_ragged_batch",
        None,
    )
    if ragged_screen is None:
        error = (
            "codegap-accelerate>=0.2.1 is required for variable-weight "
            "CUDA family batches."
        )
        if bool(settings.get("strict_native", False)):
            raise RuntimeError(error)
        return {
            "fallback": True,
            "error": error,
            "candidate_count": len(specs),
        }
    try:
        result = ragged_screen(
            supports_a,
            supports_b,
            dimensions=dims,
            min_dx=int(min_dx),
            min_dz=int(min_dz),
            min_k=int(min_k),
            backend=str(settings.get("backend", "auto")),
            threads=int(settings.get("cpu_threads", 0)),
            cuda_min_batch=int(settings.get("cuda_min_batch", 512)),
        )
        result["candidate_count"] = len(specs)
        return result
    except Exception as exception:
        if bool(settings.get("strict_native", False)):
            raise
        return {
            "fallback": True,
            "error": f"{type(exception).__name__}: {exception}",
            "candidate_count": len(specs),
        }

def accelerated_apsp(
    graph: Any,
    *,
    threads: int = 0,
) -> np.ndarray | None:
    module = _accel_module()
    if module is None:
        return None
    edges = np.asarray(list(graph.edges()), dtype=np.int32).reshape(-1, 2)
    try:
        return module.apsp_unweighted(
            int(graph.number_of_nodes()),
            edges,
            threads=int(threads),
        )
    except Exception:
        return None


def accelerated_layout(
    edges: np.ndarray,
    distances: np.ndarray,
    *,
    nlogical: int,
    iterations: int,
    seed: int,
) -> tuple[tuple[int, ...], int] | None:
    module = _accel_module()
    if module is None:
        return None
    restarts = int(os.getenv("CODEGAP_ACCEL_LAYOUT_RESTARTS", "8"))
    threads = int(os.getenv("CODEGAP_ACCEL_THREADS", "0"))
    try:
        return module.anneal_layout(
            edges,
            distances,
            nlogical=int(nlogical),
            iterations=int(iterations),
            seed=int(seed),
            restarts=max(1, restarts),
            threads=max(0, threads),
        )
    except Exception:
        return None


def accelerated_jaccard_matrix(
    edge_sets: Sequence[Iterable[tuple[int, int]]],
) -> np.ndarray | None:
    module = _accel_module()
    if module is None:
        return None
    threads = int(os.getenv("CODEGAP_ACCEL_THREADS", "0"))
    try:
        return module.jaccard_matrix(
            edge_sets,
            threads=max(0, threads),
        )
    except Exception:
        return None
