from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np


CUDA_SOURCE = r'''
extern "C" __global__ void propagate_cones(
    const unsigned long long* mask_lo,
    const unsigned long long* mask_hi,
    const unsigned short* left,
    const unsigned short* right,
    int observable_count,
    int edge_count,
    unsigned long long* out_lo,
    unsigned long long* out_hi,
    unsigned short* out_size
) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= observable_count) return;
    unsigned long long lo = mask_lo[index];
    unsigned long long hi = mask_hi[index];
    for (int edge = 0; edge < edge_count; ++edge) {
        int a = (int)left[edge];
        int b = (int)right[edge];
        bool hit_a = a < 64 ? ((lo >> a) & 1ULL) : ((hi >> (a - 64)) & 1ULL);
        bool hit_b = b < 64 ? ((lo >> b) & 1ULL) : ((hi >> (b - 64)) & 1ULL);
        if (hit_a || hit_b) {
            if (a < 64) lo |= 1ULL << a; else hi |= 1ULL << (a - 64);
            if (b < 64) lo |= 1ULL << b; else hi |= 1ULL << (b - 64);
        }
    }
    out_lo[index] = lo;
    out_hi[index] = hi;
    out_size[index] = (unsigned short)(__popcll(lo) + __popcll(hi));
}
'''


def cuda_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp

    return cp.RawKernel(
        CUDA_SOURCE,
        "propagate_cones",
        options=("--std=c++14",),
        backend="nvrtc",
    )


def _pack_masks(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masks = np.asarray(masks, dtype=np.uint8)
    if masks.ndim != 2 or masks.shape[1] > 128:
        raise ValueError("GPU light-cone masks require shape (count, n) with n <= 128.")
    lo = np.zeros(masks.shape[0], dtype=np.uint64)
    hi = np.zeros(masks.shape[0], dtype=np.uint64)
    for qubit in range(masks.shape[1]):
        values = masks[:, qubit].astype(np.uint64)
        if qubit < 64:
            lo |= values << np.uint64(qubit)
        else:
            hi |= values << np.uint64(qubit - 64)
    return lo, hi


def _unpack_masks(lo: np.ndarray, hi: np.ndarray, n: int) -> np.ndarray:
    output = np.zeros((len(lo), n), dtype=np.uint8)
    for qubit in range(n):
        if qubit < 64:
            output[:, qubit] = ((lo >> np.uint64(qubit)) & 1).astype(np.uint8)
        else:
            output[:, qubit] = ((hi >> np.uint64(qubit - 64)) & 1).astype(np.uint8)
    return output


def _reversed_two_qubit_edges(circuit_spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        tuple(int(value) for value in gate["qubits"])
        for gate in reversed(circuit_spec["gates"])
        if len(gate["qubits"]) == 2
    ]
    if not pairs:
        return np.empty(0, dtype=np.uint16), np.empty(0, dtype=np.uint16)
    return (
        np.asarray([pair[0] for pair in pairs], dtype=np.uint16),
        np.asarray([pair[1] for pair in pairs], dtype=np.uint16),
    )


def _cpu_batch(circuit_spec: dict[str, Any], masks: np.ndarray) -> dict[str, Any]:
    n = int(circuit_spec["n"])
    lo, hi = _pack_masks(masks)
    left, right = _reversed_two_qubit_edges(circuit_spec)
    out_lo = lo.copy()
    out_hi = hi.copy()
    for a, b in zip(left.tolist(), right.tolist()):
        if a < 64:
            hit_a = ((out_lo >> np.uint64(a)) & 1).astype(bool)
        else:
            hit_a = ((out_hi >> np.uint64(a - 64)) & 1).astype(bool)
        if b < 64:
            hit_b = ((out_lo >> np.uint64(b)) & 1).astype(bool)
        else:
            hit_b = ((out_hi >> np.uint64(b - 64)) & 1).astype(bool)
        hit = hit_a | hit_b
        if a < 64:
            out_lo[hit] |= np.uint64(1) << np.uint64(a)
        else:
            out_hi[hit] |= np.uint64(1) << np.uint64(a - 64)
        if b < 64:
            out_lo[hit] |= np.uint64(1) << np.uint64(b)
        else:
            out_hi[hit] |= np.uint64(1) << np.uint64(b - 64)
    sizes = np.asarray(
        [int(value).bit_count() + int(other).bit_count() for value, other in zip(out_lo, out_hi)],
        dtype=np.int32,
    )
    return {
        "backend": "numpy_packed_uint64",
        "sizes": sizes,
        "cones": _unpack_masks(out_lo, out_hi, n),
    }


def batch_backward_lightcones(
    circuit_spec: dict[str, Any],
    masks: np.ndarray,
    *,
    backend: str = "auto",
    cuda_min_observables: int = 8,
) -> dict[str, Any]:
    masks = np.asarray(masks, dtype=np.uint8)
    if masks.ndim != 2 or masks.shape[1] != int(circuit_spec["n"]):
        raise ValueError("Observable masks do not match circuit width.")
    requested = str(backend).lower()
    use_cuda = requested == "cuda" or (
        requested == "auto"
        and masks.shape[0] >= int(cuda_min_observables)
        and int(circuit_spec["n"]) <= 128
        and cuda_available()
    )
    if not use_cuda:
        return _cpu_batch(circuit_spec, masks)
    if not cuda_available():
        if requested == "cuda":
            raise RuntimeError("CUDA verifier backend requested but unavailable.")
        return _cpu_batch(circuit_spec, masks)

    import cupy as cp

    n = int(circuit_spec["n"])
    lo, hi = _pack_masks(masks)
    left, right = _reversed_two_qubit_edges(circuit_spec)
    d_lo = cp.asarray(lo)
    d_hi = cp.asarray(hi)
    d_left = cp.asarray(left)
    d_right = cp.asarray(right)
    out_lo = cp.zeros_like(d_lo)
    out_hi = cp.zeros_like(d_hi)
    out_size = cp.zeros(masks.shape[0], dtype=cp.uint16)
    threads = 128
    blocks = (masks.shape[0] + threads - 1) // threads
    _kernel()(
        (blocks,),
        (threads,),
        (
            d_lo,
            d_hi,
            d_left,
            d_right,
            np.int32(masks.shape[0]),
            np.int32(left.shape[0]),
            out_lo,
            out_hi,
            out_size,
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    host_lo = cp.asnumpy(out_lo)
    host_hi = cp.asnumpy(out_hi)
    return {
        "backend": "cuda_rawkernel_packed_uint64",
        "sizes": cp.asnumpy(out_size).astype(np.int32),
        "cones": _unpack_masks(host_lo, host_hi, n),
    }
