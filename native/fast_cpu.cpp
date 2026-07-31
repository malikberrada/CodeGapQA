#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int MAX_ROWS = 32;
constexpr int MAX_N = 64;

inline int highest_bit(std::uint64_t value) {
#if defined(_MSC_VER)
    unsigned long index = 0;
#if defined(_M_X64) || defined(_M_ARM64)
    _BitScanReverse64(&index, value);
    return static_cast<int>(index);
#else
    if (value >> 32U) {
        _BitScanReverse(&index, static_cast<unsigned long>(value >> 32U));
        return static_cast<int>(index + 32U);
    }
    _BitScanReverse(&index, static_cast<unsigned long>(value));
    return static_cast<int>(index);
#endif
#else
    return 63 - __builtin_clzll(value);
#endif
}

int make_basis(
    const std::uint64_t* rows,
    int row_count,
    std::uint64_t* basis
) {
    int rank = 0;
    for (int row = 0; row < row_count; ++row) {
        std::uint64_t value = rows[row];
        for (int i = 0; i < rank && value; ++i) {
            const int pivot = highest_bit(basis[i]);
            if ((value >> pivot) & 1ULL) {
                value ^= basis[i];
            }
        }
        if (!value) {
            continue;
        }
        const int pivot = highest_bit(value);
        for (int i = 0; i < rank; ++i) {
            if ((basis[i] >> pivot) & 1ULL) {
                basis[i] ^= value;
            }
        }
        int insert = rank;
        while (
            insert > 0
            && highest_bit(basis[insert - 1]) < pivot
        ) {
            basis[insert] = basis[insert - 1];
            --insert;
        }
        basis[insert] = value;
        ++rank;
    }
    return rank;
}

inline bool in_span(
    std::uint64_t value,
    const std::uint64_t* basis,
    int rank
) {
    for (int i = 0; i < rank && value; ++i) {
        const int pivot = highest_bit(basis[i]);
        if ((value >> pivot) & 1ULL) {
            value ^= basis[i];
        }
    }
    return value == 0ULL;
}

inline bool in_kernel(
    std::uint64_t value,
    const std::uint64_t* check_rows,
    int row_count
) {
    for (int row = 0; row < row_count; ++row) {
#if defined(_MSC_VER)
        if (__popcnt64(value & check_rows[row]) & 1) {
#else
        if (__builtin_popcountll(value & check_rows[row]) & 1) {
#endif
            return false;
        }
    }
    return true;
}

bool low_weight_logical_exists(
    const std::uint64_t* check_rows,
    const std::uint64_t* stabilizer_rows,
    int row_count,
    int n,
    int minimum,
    std::uint64_t* witness
) {
    if (minimum <= 1) {
        *witness = 0ULL;
        return false;
    }
    if (minimum > 5) {
        throw std::invalid_argument(
            "Native distance prefilter supports minimum <= 5."
        );
    }
    std::uint64_t basis[MAX_ROWS]{};
    const int stabilizer_rank = make_basis(
        stabilizer_rows,
        row_count,
        basis
    );

    auto test = [&](std::uint64_t value) -> bool {
        if (!in_kernel(value, check_rows, row_count)) {
            return false;
        }
        if (in_span(value, basis, stabilizer_rank)) {
            return false;
        }
        *witness = value;
        return true;
    };

    if (minimum > 1) {
        for (int a = 0; a < n; ++a) {
            if (test(1ULL << a)) {
                return true;
            }
        }
    }
    if (minimum > 2) {
        for (int a = 0; a < n; ++a) {
            for (int b = a + 1; b < n; ++b) {
                if (test((1ULL << a) | (1ULL << b))) {
                    return true;
                }
            }
        }
    }
    if (minimum > 3) {
        for (int a = 0; a < n; ++a) {
            for (int b = a + 1; b < n; ++b) {
                for (int c = b + 1; c < n; ++c) {
                    if (test(
                        (1ULL << a)
                        | (1ULL << b)
                        | (1ULL << c)
                    )) {
                        return true;
                    }
                }
            }
        }
    }
    if (minimum > 4) {
        for (int a = 0; a < n; ++a) {
            for (int b = a + 1; b < n; ++b) {
                for (int c = b + 1; c < n; ++c) {
                    for (int d = c + 1; d < n; ++d) {
                        if (test(
                            (1ULL << a)
                            | (1ULL << b)
                            | (1ULL << c)
                            | (1ULL << d)
                        )) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    *witness = 0ULL;
    return false;
}

inline int wrap(int value, int modulus) {
    value %= modulus;
    return value < 0 ? value + modulus : value;
}

void build_qc_rows(
    const std::uint16_t* support_a,
    const std::uint16_t* support_b,
    int weight_a,
    int weight_b,
    int l,
    int m,
    std::uint64_t* h_x,
    std::uint64_t* h_z
) {
    const int block = l * m;
    for (int row = 0; row < block; ++row) {
        const int x = row / m;
        const int y = row % m;
        std::uint64_t hx = 0ULL;
        std::uint64_t hz = 0ULL;

        for (int index = 0; index < weight_a; ++index) {
            const int encoded = support_a[index];
            const int dx = encoded / m;
            const int dy = encoded % m;
            const int source = wrap(x - dx, l) * m + wrap(y - dy, m);
            const int transposed = wrap(x + dx, l) * m + wrap(y + dy, m);
            hx ^= 1ULL << source;
            hz ^= 1ULL << (block + transposed);
        }
        for (int index = 0; index < weight_b; ++index) {
            const int encoded = support_b[index];
            const int dx = encoded / m;
            const int dy = encoded % m;
            const int source = wrap(x - dx, l) * m + wrap(y - dy, m);
            const int transposed = wrap(x + dx, l) * m + wrap(y + dy, m);
            hx ^= 1ULL << (block + source);
            hz ^= 1ULL << transposed;
        }
        h_x[row] = hx;
        h_z[row] = hz;
    }
}

struct ScreenOutput {
    std::vector<std::uint8_t> rank_x;
    std::vector<std::uint8_t> rank_z;
    std::vector<std::uint8_t> k;
    std::vector<std::uint8_t> flags;
    std::vector<std::uint64_t> witness_x;
    std::vector<std::uint64_t> witness_z;
};

ScreenOutput screen_batch(
    const std::uint16_t* supports_a,
    const std::uint16_t* supports_b,
    int count,
    int weight_a,
    int weight_b,
    int l,
    int m,
    int min_dx,
    int min_dz,
    int min_k,
    int threads
) {
    const int block = l * m;
    const int n = 2 * block;
    if (block <= 0 || block > MAX_ROWS || n > MAX_N) {
        throw std::invalid_argument(
            "Native QC screening requires 1 <= l*m <= 32 and n <= 64."
        );
    }
    ScreenOutput output{
        std::vector<std::uint8_t>(count),
        std::vector<std::uint8_t>(count),
        std::vector<std::uint8_t>(count),
        std::vector<std::uint8_t>(count),
        std::vector<std::uint64_t>(count),
        std::vector<std::uint64_t>(count),
    };

#ifdef _OPENMP
    if (threads > 0) {
        omp_set_num_threads(threads);
    }
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int candidate = 0; candidate < count; ++candidate) {
        std::uint64_t h_x[MAX_ROWS]{};
        std::uint64_t h_z[MAX_ROWS]{};
        std::uint64_t basis[MAX_ROWS]{};
        build_qc_rows(
            supports_a + static_cast<std::size_t>(candidate) * weight_a,
            supports_b + static_cast<std::size_t>(candidate) * weight_b,
            weight_a,
            weight_b,
            l,
            m,
            h_x,
            h_z
        );
        const int rank_x = make_basis(h_x, block, basis);
        std::fill(std::begin(basis), std::end(basis), 0ULL);
        const int rank_z = make_basis(h_z, block, basis);
        const int logical = n - rank_x - rank_z;
        output.rank_x[candidate] = static_cast<std::uint8_t>(rank_x);
        output.rank_z[candidate] = static_cast<std::uint8_t>(rank_z);
        output.k[candidate] = static_cast<std::uint8_t>(std::max(0, logical));

        std::uint8_t flags = 0U;
        if (logical >= min_k) {
            flags |= 0x01U;
        } else {
            output.flags[candidate] = flags;
            continue;
        }

        std::uint64_t witness_x = 0ULL;
        const bool x_bad = low_weight_logical_exists(
            h_z,
            h_x,
            block,
            n,
            min_dx,
            &witness_x
        );
        output.witness_x[candidate] = witness_x;
        if (!x_bad) {
            flags |= 0x02U;
        } else {
            output.flags[candidate] = flags;
            continue;
        }

        std::uint64_t witness_z = 0ULL;
        const bool z_bad = low_weight_logical_exists(
            h_x,
            h_z,
            block,
            n,
            min_dz,
            &witness_z
        );
        output.witness_z[candidate] = witness_z;
        if (!z_bad) {
            flags |= 0x04U;
        }
        output.flags[candidate] = flags;
    }
    return output;
}

PyObject* bytes_from_vector(const std::vector<std::uint8_t>& values) {
    return PyBytes_FromStringAndSize(
        reinterpret_cast<const char*>(values.data()),
        static_cast<Py_ssize_t>(values.size())
    );
}

PyObject* bytes_from_vector_u64(const std::vector<std::uint64_t>& values) {
    return PyBytes_FromStringAndSize(
        reinterpret_cast<const char*>(values.data()),
        static_cast<Py_ssize_t>(values.size() * sizeof(std::uint64_t))
    );
}

PyObject* py_batch_qc_screen(PyObject*, PyObject* args) {
    PyObject* a_object = nullptr;
    PyObject* b_object = nullptr;
    int count = 0;
    int weight_a = 0;
    int weight_b = 0;
    int l = 0;
    int m = 0;
    int min_dx = 0;
    int min_dz = 0;
    int min_k = 0;
    int threads = 0;

    if (!PyArg_ParseTuple(
        args,
        "OOiiiiiiiii",
        &a_object,
        &b_object,
        &count,
        &weight_a,
        &weight_b,
        &l,
        &m,
        &min_dx,
        &min_dz,
        &min_k,
        &threads
    )) {
        return nullptr;
    }

    Py_buffer a_buffer{};
    Py_buffer b_buffer{};
    if (PyObject_GetBuffer(a_object, &a_buffer, PyBUF_CONTIG_RO) != 0) {
        return nullptr;
    }
    if (PyObject_GetBuffer(b_object, &b_buffer, PyBUF_CONTIG_RO) != 0) {
        PyBuffer_Release(&a_buffer);
        return nullptr;
    }

    const auto expected_a = static_cast<Py_ssize_t>(
        static_cast<std::size_t>(count) * weight_a * sizeof(std::uint16_t)
    );
    const auto expected_b = static_cast<Py_ssize_t>(
        static_cast<std::size_t>(count) * weight_b * sizeof(std::uint16_t)
    );
    if (a_buffer.len < expected_a || b_buffer.len < expected_b) {
        PyBuffer_Release(&a_buffer);
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_ValueError, "Support buffers are too small.");
        return nullptr;
    }

    ScreenOutput output;
    try {
        Py_BEGIN_ALLOW_THREADS
        output = screen_batch(
            static_cast<const std::uint16_t*>(a_buffer.buf),
            static_cast<const std::uint16_t*>(b_buffer.buf),
            count,
            weight_a,
            weight_b,
            l,
            m,
            min_dx,
            min_dz,
            min_k,
            threads
        );
        Py_END_ALLOW_THREADS
    } catch (const std::exception& exception) {
        PyBuffer_Release(&a_buffer);
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_RuntimeError, exception.what());
        return nullptr;
    }

    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);

    PyObject* result = PyTuple_New(6);
    if (!result) {
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, bytes_from_vector(output.rank_x));
    PyTuple_SET_ITEM(result, 1, bytes_from_vector(output.rank_z));
    PyTuple_SET_ITEM(result, 2, bytes_from_vector(output.k));
    PyTuple_SET_ITEM(result, 3, bytes_from_vector(output.flags));
    PyTuple_SET_ITEM(result, 4, bytes_from_vector_u64(output.witness_x));
    PyTuple_SET_ITEM(result, 5, bytes_from_vector_u64(output.witness_z));
    return result;
}

long long layout_cost(
    const std::vector<int>& layout,
    const std::int16_t* distances,
    int nphysical,
    const std::int32_t* edges,
    int edge_count
) {
    long long cost = 0;
    for (int edge = 0; edge < edge_count; ++edge) {
        const int left = edges[2 * edge];
        const int right = edges[2 * edge + 1];
        cost += distances[
            layout[left] * nphysical + layout[right]
        ];
    }
    return cost;
}

PyObject* py_anneal_layout(PyObject*, PyObject* args) {
    PyObject* edges_object = nullptr;
    PyObject* distance_object = nullptr;
    int edge_count = 0;
    int nlogical = 0;
    int nphysical = 0;
    int iterations = 0;
    unsigned long long seed = 0ULL;

    if (!PyArg_ParseTuple(
        args,
        "OOiiiiK",
        &edges_object,
        &distance_object,
        &edge_count,
        &nlogical,
        &nphysical,
        &iterations,
        &seed
    )) {
        return nullptr;
    }

    Py_buffer edge_buffer{};
    Py_buffer distance_buffer{};
    if (PyObject_GetBuffer(edges_object, &edge_buffer, PyBUF_CONTIG_RO) != 0) {
        return nullptr;
    }
    if (
        PyObject_GetBuffer(
            distance_object,
            &distance_buffer,
            PyBUF_CONTIG_RO
        ) != 0
    ) {
        PyBuffer_Release(&edge_buffer);
        return nullptr;
    }

    const auto* edges = static_cast<const std::int32_t*>(edge_buffer.buf);
    const auto* distances = static_cast<const std::int16_t*>(
        distance_buffer.buf
    );
    std::vector<int> current(nlogical);
    std::iota(current.begin(), current.end(), 0);
    std::vector<int> best = current;
    long long current_cost = layout_cost(
        current,
        distances,
        nphysical,
        edges,
        edge_count
    );
    long long best_cost = current_cost;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> qubit(0, nlogical - 1);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    Py_BEGIN_ALLOW_THREADS
    for (int step = 0; step < std::max(1, iterations); ++step) {
        int left = qubit(rng);
        int right = qubit(rng);
        while (right == left) {
            right = qubit(rng);
        }
        std::swap(current[left], current[right]);
        const long long trial_cost = layout_cost(
            current,
            distances,
            nphysical,
            edges,
            edge_count
        );
        const double temperature = std::max(
            0.01,
            1.0 - static_cast<double>(step)
                / static_cast<double>(std::max(1, iterations))
        );
        const bool accept = trial_cost <= current_cost
            || uniform(rng) < std::exp(
                -static_cast<double>(trial_cost - current_cost)
                / temperature
            );
        if (accept) {
            current_cost = trial_cost;
        } else {
            std::swap(current[left], current[right]);
        }
        if (current_cost < best_cost) {
            best = current;
            best_cost = current_cost;
        }
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&edge_buffer);
    PyBuffer_Release(&distance_buffer);

    PyObject* layout_list = PyList_New(nlogical);
    if (!layout_list) {
        return nullptr;
    }
    for (int index = 0; index < nlogical; ++index) {
        PyList_SET_ITEM(
            layout_list,
            index,
            PyLong_FromLong(best[index])
        );
    }
    PyObject* result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, layout_list);
    PyTuple_SET_ITEM(result, 1, PyLong_FromLongLong(best_cost));
    return result;
}

PyObject* py_build_info(PyObject*, PyObject*) {
    PyObject* result = PyDict_New();
#ifdef _OPENMP
    PyDict_SetItemString(result, "openmp", Py_True);
    PyDict_SetItemString(
        result,
        "max_threads",
        PyLong_FromLong(omp_get_max_threads())
    );
#else
    PyDict_SetItemString(result, "openmp", Py_False);
    PyDict_SetItemString(result, "max_threads", PyLong_FromLong(1));
#endif
    PyDict_SetItemString(result, "max_rows", PyLong_FromLong(MAX_ROWS));
    PyDict_SetItemString(result, "max_n", PyLong_FromLong(MAX_N));
    return result;
}

PyMethodDef METHODS[] = {
    {
        "batch_qc_screen",
        py_batch_qc_screen,
        METH_VARARGS,
        "Batch exact QC-CSS rank and low-weight-distance screening."
    },
    {
        "anneal_layout",
        py_anneal_layout,
        METH_VARARGS,
        "Native simulated-annealing layout optimizer."
    },
    {
        "build_info",
        py_build_info,
        METH_NOARGS,
        "Return native backend build information."
    },
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef MODULE = {
    PyModuleDef_HEAD_INIT,
    "_fast_cpu",
    "CodeGap-QA native C++ acceleration backend.",
    -1,
    METHODS,
};

}  // namespace

PyMODINIT_FUNC PyInit__fast_cpu() {
    return PyModule_Create(&MODULE);
}
