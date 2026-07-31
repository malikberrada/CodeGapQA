from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal
import numpy as np


BackendName = Literal["cuda", "native_cpu", "python"]


@dataclass(frozen=True)
class BackendStatus:
    selected: BackendName
    native_cpu_available: bool
    cuda_available: bool
    native_build: dict
    reason: str


def _native_module():
    try:
        from . import _fast_cpu

        return _fast_cpu
    except Exception:
        return None


def native_available() -> bool:
    return _native_module() is not None


def cuda_available() -> bool:
    try:
        from .cuda_backend import available

        return bool(available())
    except Exception:
        return False


def select_backend(
    requested: str,
    *,
    candidate_count: int,
    l: int,
    m: int,
    cuda_min_batch: int,
) -> BackendStatus:
    requested_argument = requested.lower()
    environment_override = os.getenv("CODEGAP_FAST_BACKEND")
    requested = (
        environment_override.lower()
        if requested_argument == "auto" and environment_override
        else requested_argument
    )
    native = _native_module()
    cuda = cuda_available()
    native_info = native.build_info() if native is not None else {}
    block = l * m
    cuda_compatible = block <= 16 and candidate_count >= cuda_min_batch
    native_compatible = block <= 32

    if requested == "cuda":
        if not cuda:
            raise RuntimeError(
                "CUDA backend requested but CuPy/CUDA is unavailable."
            )
        if not cuda_compatible:
            raise RuntimeError(
                "CUDA backend requires l*m <= 16 and a sufficiently large batch."
            )
        return BackendStatus("cuda", native is not None, True, native_info, "forced")
    if requested in {"native", "native_cpu", "cpu"}:
        if not native_compatible:
            raise RuntimeError(
                "Native CPU backend supports n <= 64; use backend=python for larger families."
            )
        if native is None:
            raise RuntimeError(
                "Native CPU backend requested but codegap_qa._fast_cpu is unavailable."
            )
        return BackendStatus(
            "native_cpu", True, cuda, native_info, "forced"
        )
    if requested == "python":
        return BackendStatus("python", native is not None, cuda, native_info, "forced")
    if requested != "auto":
        raise ValueError(f"Unknown acceleration backend: {requested}")

    if cuda and cuda_compatible:
        return BackendStatus(
            "cuda", native is not None, True, native_info, "auto_large_batch"
        )
    if native is not None and native_compatible:
        return BackendStatus(
            "native_cpu", True, cuda, native_info, "auto_native_fallback"
        )
    return BackendStatus(
        "python", False, cuda, native_info, "native_extension_missing"
    )


def _decode_native_result(raw: tuple[bytes, ...], count: int) -> dict[str, np.ndarray]:
    return {
        "rank_x": np.frombuffer(raw[0], dtype=np.uint8, count=count).copy(),
        "rank_z": np.frombuffer(raw[1], dtype=np.uint8, count=count).copy(),
        "k": np.frombuffer(raw[2], dtype=np.uint8, count=count).copy(),
        "flags": np.frombuffer(raw[3], dtype=np.uint8, count=count).copy(),
        "witness_x": np.frombuffer(raw[4], dtype=np.uint64, count=count).copy(),
        "witness_z": np.frombuffer(raw[5], dtype=np.uint64, count=count).copy(),
    }


def screen_qc_batch(
    supports_a: np.ndarray,
    supports_b: np.ndarray,
    *,
    l: int,
    m: int,
    min_dx: int,
    min_dz: int,
    min_k: int,
    backend: str = "auto",
    threads: int = 0,
    cuda_min_batch: int = 2048,
) -> tuple[dict[str, np.ndarray], BackendStatus]:
    a = np.ascontiguousarray(supports_a, dtype=np.uint16)
    b = np.ascontiguousarray(supports_b, dtype=np.uint16)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("Support batches must be 2D with matching candidate counts.")
    status = select_backend(
        backend,
        candidate_count=a.shape[0],
        l=l,
        m=m,
        cuda_min_batch=cuda_min_batch,
    )
    if status.selected == "cuda":
        from .cuda_backend import screen_qc_batch as cuda_screen

        return (
            cuda_screen(
                a,
                b,
                l=l,
                m=m,
                min_dx=min_dx,
                min_dz=min_dz,
                min_k=min_k,
            ),
            status,
        )
    if status.selected == "native_cpu":
        native = _native_module()
        assert native is not None
        raw = native.batch_qc_screen(
            memoryview(a),
            memoryview(b),
            int(a.shape[0]),
            int(a.shape[1]),
            int(b.shape[1]),
            int(l),
            int(m),
            int(min_dx),
            int(min_dz),
            int(min_k),
            int(threads),
        )
        return _decode_native_result(raw, int(a.shape[0])), status
    return _python_screen(
        a,
        b,
        l=l,
        m=m,
        min_dx=min_dx,
        min_dz=min_dz,
        min_k=min_k,
    ), status


def _python_screen(
    supports_a: np.ndarray,
    supports_b: np.ndarray,
    *,
    l: int,
    m: int,
    min_dx: int,
    min_dz: int,
    min_k: int,
) -> dict[str, np.ndarray]:
    from .bicycle import BicycleFamilySpec, build_bicycle_css
    from .gf2 import css_distance_at_least_fast_small, rank

    count = supports_a.shape[0]
    rank_x = np.zeros(count, dtype=np.uint8)
    rank_z = np.zeros(count, dtype=np.uint8)
    logical = np.zeros(count, dtype=np.uint8)
    flags = np.zeros(count, dtype=np.uint8)
    witness_x = np.zeros(count, dtype=np.uint64)
    witness_z = np.zeros(count, dtype=np.uint64)
    for index in range(count):
        spec = BicycleFamilySpec(
            l=l,
            m=m,
            support_a=tuple((int(v) // m, int(v) % m) for v in supports_a[index]),
            support_b=tuple((int(v) // m, int(v) % m) for v in supports_b[index]),
        )
        h_x, h_z = build_bicycle_css(spec)
        rx, rz = rank(h_x), rank(h_z)
        k = h_x.shape[1] - rx - rz
        rank_x[index], rank_z[index], logical[index] = rx, rz, max(0, k)
        if k < min_k:
            continue
        flags[index] |= 1
        dx_ok, dx_cert = css_distance_at_least_fast_small(h_z, h_x, min_dx)
        if not dx_ok:
            support = dx_cert.get("witness_support") or []
            witness_x[index] = (
                sum(np.uint64(1) << np.uint64(v) for v in support)
                if all(int(v) < 64 for v in support)
                else np.uint64(0)
            )
            continue
        flags[index] |= 2
        dz_ok, dz_cert = css_distance_at_least_fast_small(h_x, h_z, min_dz)
        if not dz_ok:
            support = dz_cert.get("witness_support") or []
            witness_z[index] = (
                sum(np.uint64(1) << np.uint64(v) for v in support)
                if all(int(v) < 64 for v in support)
                else np.uint64(0)
            )
            continue
        flags[index] |= 4
    return {
        "rank_x": rank_x,
        "rank_z": rank_z,
        "k": logical,
        "flags": flags,
        "witness_x": witness_x,
        "witness_z": witness_z,
    }


def anneal_layout_native(
    edges: np.ndarray,
    distance_matrix: np.ndarray,
    *,
    nlogical: int,
    iterations: int,
    seed: int,
) -> tuple[tuple[int, ...], int] | None:
    native = _native_module()
    if native is None:
        return None
    edge_array = np.ascontiguousarray(edges, dtype=np.int32).reshape(-1, 2)
    distances = np.ascontiguousarray(distance_matrix, dtype=np.int16)
    layout, cost = native.anneal_layout(
        memoryview(edge_array),
        memoryview(distances),
        int(edge_array.shape[0]),
        int(nlogical),
        int(distances.shape[0]),
        int(iterations),
        int(seed),
    )
    return tuple(int(v) for v in layout), int(cost)


def diagnostics() -> dict:
    native = _native_module()
    return {
        "native_cpu_available": native is not None,
        "native_build": native.build_info() if native is not None else {},
        "cuda_available": cuda_available(),
    }
