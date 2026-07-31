from __future__ import annotations

from functools import lru_cache
import numpy as np


CUDA_SOURCE = r'''
extern "C" {

#define MAX_ROWS 16

__device__ __forceinline__ int highest_bit(unsigned long long value) {
    return 63 - __clzll(value);
}

__device__ int make_basis(
    const unsigned long long* rows,
    int row_count,
    unsigned long long* basis
) {
    int rank = 0;
    for (int row = 0; row < row_count; ++row) {
        unsigned long long value = rows[row];
        for (int i = 0; i < rank && value; ++i) {
            int pivot = highest_bit(basis[i]);
            if ((value >> pivot) & 1ULL) value ^= basis[i];
        }
        if (!value) continue;
        int pivot = highest_bit(value);
        for (int i = 0; i < rank; ++i) {
            if ((basis[i] >> pivot) & 1ULL) basis[i] ^= value;
        }
        int insert = rank;
        while (insert > 0 && highest_bit(basis[insert - 1]) < pivot) {
            basis[insert] = basis[insert - 1];
            --insert;
        }
        basis[insert] = value;
        ++rank;
    }
    return rank;
}

__device__ __forceinline__ bool in_span(
    unsigned long long value,
    const unsigned long long* basis,
    int rank
) {
    for (int i = 0; i < rank && value; ++i) {
        int pivot = highest_bit(basis[i]);
        if ((value >> pivot) & 1ULL) value ^= basis[i];
    }
    return value == 0ULL;
}

__device__ __forceinline__ bool in_kernel(
    unsigned long long value,
    const unsigned long long* rows,
    int row_count
) {
    for (int row = 0; row < row_count; ++row) {
        if (__popcll(value & rows[row]) & 1) return false;
    }
    return true;
}

__device__ bool low_weight_logical_exists(
    const unsigned long long* check_rows,
    const unsigned long long* stabilizer_rows,
    int row_count,
    int n,
    int minimum,
    unsigned long long* witness
) {
    unsigned long long basis[MAX_ROWS];
    #pragma unroll
    for (int i = 0; i < MAX_ROWS; ++i) basis[i] = 0ULL;
    int rank = make_basis(stabilizer_rows, row_count, basis);

    if (minimum > 1) {
        for (int a = 0; a < n; ++a) {
            unsigned long long v = 1ULL << a;
            if (in_kernel(v, check_rows, row_count) && !in_span(v, basis, rank)) {
                *witness = v; return true;
            }
        }
    }
    if (minimum > 2) {
        for (int a = 0; a < n; ++a) for (int b = a + 1; b < n; ++b) {
            unsigned long long v = (1ULL << a) | (1ULL << b);
            if (in_kernel(v, check_rows, row_count) && !in_span(v, basis, rank)) {
                *witness = v; return true;
            }
        }
    }
    if (minimum > 3) {
        for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b)
        for (int c = b + 1; c < n; ++c) {
            unsigned long long v = (1ULL << a) | (1ULL << b) | (1ULL << c);
            if (in_kernel(v, check_rows, row_count) && !in_span(v, basis, rank)) {
                *witness = v; return true;
            }
        }
    }
    if (minimum > 4) {
        for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b)
        for (int c = b + 1; c < n; ++c)
        for (int d = c + 1; d < n; ++d) {
            unsigned long long v = (1ULL << a) | (1ULL << b)
                | (1ULL << c) | (1ULL << d);
            if (in_kernel(v, check_rows, row_count) && !in_span(v, basis, rank)) {
                *witness = v; return true;
            }
        }
    }
    *witness = 0ULL;
    return false;
}

__device__ __forceinline__ int wrap_index(int value, int modulus) {
    value %= modulus;
    return value < 0 ? value + modulus : value;
}

__global__ void qc_screen(
    const unsigned short* supports_a,
    const unsigned short* supports_b,
    int count,
    int weight_a,
    int weight_b,
    int l,
    int m,
    int min_dx,
    int min_dz,
    int min_k,
    unsigned char* rank_x_out,
    unsigned char* rank_z_out,
    unsigned char* k_out,
    unsigned char* flags_out,
    unsigned long long* witness_x_out,
    unsigned long long* witness_z_out
) {
    int candidate = blockDim.x * blockIdx.x + threadIdx.x;
    if (candidate >= count) return;
    int block = l * m;
    int n = 2 * block;
    if (block > MAX_ROWS || n > 64 || min_dx > 5 || min_dz > 5) {
        flags_out[candidate] = 0;
        return;
    }

    unsigned long long hx[MAX_ROWS];
    unsigned long long hz[MAX_ROWS];
    unsigned long long basis[MAX_ROWS];
    #pragma unroll
    for (int i = 0; i < MAX_ROWS; ++i) {
        hx[i] = 0ULL; hz[i] = 0ULL; basis[i] = 0ULL;
    }

    const unsigned short* a = supports_a + candidate * weight_a;
    const unsigned short* b = supports_b + candidate * weight_b;
    for (int row = 0; row < block; ++row) {
        int x = row / m;
        int y = row % m;
        unsigned long long xrow = 0ULL;
        unsigned long long zrow = 0ULL;
        for (int index = 0; index < weight_a; ++index) {
            int encoded = (int)a[index];
            int dx = encoded / m;
            int dy = encoded % m;
            int source = wrap_index(x - dx, l) * m + wrap_index(y - dy, m);
            int transpose = wrap_index(x + dx, l) * m + wrap_index(y + dy, m);
            xrow ^= 1ULL << source;
            zrow ^= 1ULL << (block + transpose);
        }
        for (int index = 0; index < weight_b; ++index) {
            int encoded = (int)b[index];
            int dx = encoded / m;
            int dy = encoded % m;
            int source = wrap_index(x - dx, l) * m + wrap_index(y - dy, m);
            int transpose = wrap_index(x + dx, l) * m + wrap_index(y + dy, m);
            xrow ^= 1ULL << (block + source);
            zrow ^= 1ULL << transpose;
        }
        hx[row] = xrow;
        hz[row] = zrow;
    }

    int rank_x = make_basis(hx, block, basis);
    #pragma unroll
    for (int i = 0; i < MAX_ROWS; ++i) basis[i] = 0ULL;
    int rank_z = make_basis(hz, block, basis);
    int logical = n - rank_x - rank_z;
    rank_x_out[candidate] = (unsigned char)rank_x;
    rank_z_out[candidate] = (unsigned char)rank_z;
    k_out[candidate] = (unsigned char)(logical > 0 ? logical : 0);

    unsigned char flags = 0;
    if (logical < min_k) {
        flags_out[candidate] = flags;
        return;
    }
    flags |= 1;

    unsigned long long witness_x = 0ULL;
    if (low_weight_logical_exists(hz, hx, block, n, min_dx, &witness_x)) {
        witness_x_out[candidate] = witness_x;
        flags_out[candidate] = flags;
        return;
    }
    flags |= 2;

    unsigned long long witness_z = 0ULL;
    if (!low_weight_logical_exists(hx, hz, block, n, min_dz, &witness_z)) {
        flags |= 4;
    }
    witness_x_out[candidate] = witness_x;
    witness_z_out[candidate] = witness_z;
    flags_out[candidate] = flags;
}

}
'''


def available() -> bool:
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
        "qc_screen",
        options=("--std=c++14",),
        backend="nvrtc",
    )


def screen_qc_batch(
    supports_a: np.ndarray,
    supports_b: np.ndarray,
    *,
    l: int,
    m: int,
    min_dx: int,
    min_dz: int,
    min_k: int,
) -> dict[str, np.ndarray]:
    import cupy as cp

    a = cp.asarray(np.ascontiguousarray(supports_a, dtype=np.uint16))
    b = cp.asarray(np.ascontiguousarray(supports_b, dtype=np.uint16))
    count = int(a.shape[0])
    rank_x = cp.zeros(count, dtype=cp.uint8)
    rank_z = cp.zeros(count, dtype=cp.uint8)
    logical = cp.zeros(count, dtype=cp.uint8)
    flags = cp.zeros(count, dtype=cp.uint8)
    witness_x = cp.zeros(count, dtype=cp.uint64)
    witness_z = cp.zeros(count, dtype=cp.uint64)
    threads = 128
    blocks = (count + threads - 1) // threads
    _kernel()(
        (blocks,),
        (threads,),
        (
            a,
            b,
            np.int32(count),
            np.int32(a.shape[1]),
            np.int32(b.shape[1]),
            np.int32(l),
            np.int32(m),
            np.int32(min_dx),
            np.int32(min_dz),
            np.int32(min_k),
            rank_x,
            rank_z,
            logical,
            flags,
            witness_x,
            witness_z,
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    return {
        "rank_x": cp.asnumpy(rank_x),
        "rank_z": cp.asnumpy(rank_z),
        "k": cp.asnumpy(logical),
        "flags": cp.asnumpy(flags),
        "witness_x": cp.asnumpy(witness_x),
        "witness_z": cp.asnumpy(witness_z),
    }
