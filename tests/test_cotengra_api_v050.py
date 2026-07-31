from __future__ import annotations

import sys
import types

from codegap_qa.spacetime import cotengra_metrics


class _FakeTree:
    def contraction_width(self):
        return 18.0

    def contraction_cost(self):
        return 200_000_000.0

    def get_path(self):
        return [(0, 1), (0, 1)]

    def get_peak_size(self):
        return 1 << 18


class _FakeOptimizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def search(self, inputs, output, size_dict):
        assert inputs
        assert output == ()
        assert size_dict
        return _FakeTree()


def test_short_and_deep_cotengra_interfaces(monkeypatch):
    module = types.SimpleNamespace(
        RandomGreedyOptimizer=_FakeOptimizer,
        HyperOptimizer=_FakeOptimizer,
    )
    monkeypatch.setitem(sys.modules, "cotengra", module)
    inputs = [("a", "b"), ("b", "c"), ("a", "c")]
    size_dict = {"a": 2, "b": 2, "c": 2}
    short = cotengra_metrics(
        inputs,
        (),
        size_dict,
        {"enabled": True, "mode": "short", "max_repeats": 4, "seed": 7},
    )
    deep = cotengra_metrics(
        inputs,
        (),
        size_dict,
        {
            "enabled": True,
            "mode": "deep",
            "methods": ["greedy"],
            "max_repeats": 8,
            "parallel": False,
        },
    )
    assert short["status"] == "PASS"
    assert deep["status"] == "PASS"
    assert short["contraction_width_log2"] == 18.0
    assert deep["contraction_flops"] == 200_000_000.0
