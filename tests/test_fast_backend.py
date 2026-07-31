from __future__ import annotations

import numpy as np
import pytest

from codegap_qa.fast_backend import native_available, screen_qc_batch


@pytest.mark.skipif(not native_available(), reason="native extension not built")
def test_native_matches_python_qc_screen():
    rng = np.random.default_rng(20260723)
    count = 32
    supports_a = np.vstack(
        [np.sort(rng.choice(12, 3, replace=False)) for _ in range(count)]
    ).astype(np.uint16)
    supports_b = np.vstack(
        [np.sort(rng.choice(12, 3, replace=False)) for _ in range(count)]
    ).astype(np.uint16)
    native, _ = screen_qc_batch(
        supports_a,
        supports_b,
        l=3,
        m=4,
        min_dx=4,
        min_dz=4,
        min_k=2,
        backend="native_cpu",
    )
    reference, _ = screen_qc_batch(
        supports_a,
        supports_b,
        l=3,
        m=4,
        min_dx=4,
        min_dz=4,
        min_k=2,
        backend="python",
    )
    for key in ("rank_x", "rank_z", "k", "flags", "witness_x", "witness_z"):
        np.testing.assert_array_equal(native[key], reference[key])


def test_explicit_backend_overrides_environment(monkeypatch):
    import codegap_qa.fast_backend as fast_backend

    class FakeNative:
        @staticmethod
        def build_info():
            return {"openmp": True, "max_n": 64, "max_rows": 32}

    monkeypatch.setenv("CODEGAP_FAST_BACKEND", "cuda")
    monkeypatch.setattr(fast_backend, "_native_module", lambda: FakeNative())
    monkeypatch.setattr(fast_backend, "cuda_available", lambda: True)

    status = fast_backend.select_backend(
        "native_cpu",
        candidate_count=32,
        l=3,
        m=4,
        cuda_min_batch=2048,
    )

    assert status.selected == "native_cpu"
    assert status.reason == "forced"


def test_environment_overrides_auto_backend(monkeypatch):
    import codegap_qa.fast_backend as fast_backend

    class FakeNative:
        @staticmethod
        def build_info():
            return {"openmp": True, "max_n": 64, "max_rows": 32}

    monkeypatch.setenv("CODEGAP_FAST_BACKEND", "native_cpu")
    monkeypatch.setattr(fast_backend, "_native_module", lambda: FakeNative())
    monkeypatch.setattr(fast_backend, "cuda_available", lambda: True)

    status = fast_backend.select_backend(
        "auto",
        candidate_count=8192,
        l=3,
        m=4,
        cuda_min_batch=2048,
    )

    assert status.selected == "native_cpu"
    assert status.reason == "forced"

