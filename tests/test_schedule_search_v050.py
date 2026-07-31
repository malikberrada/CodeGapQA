from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from codegap_qa.bicycle import BicycleFamilySpec, build_bicycle_css
from codegap_qa.gates import gate_c
from codegap_qa.hardware import HardwareTopology
from codegap_qa.progress import ProgressManager
from codegap_qa.schedule_search import (
    schedule_objective,
    search_adversarial_schedules,
)


def test_schedule_objective_matches_preregistered_formula():
    settings = {
        "objective": {"lambda_depth": 0.02, "mu_twoq": 0.0005}
    }
    value = schedule_objective(
        cotengra_flops=192_000_000.0,
        verify_operations_value=192.0,
        two_qubit_depth=12,
        two_qubit_count=576,
        settings=settings,
    )
    expected = 6.0 - 0.02 * 12 - 0.0005 * 576
    assert abs(value - expected) < 1e-12


def test_gate_c_requires_registered_cotengra_targets():
    candidate = {
        "hardness": {
            "gamma_log10": 6.5,
            "best_attack_name": "cotengra_or_line_graph",
            "best_attack_operations": 1_000_000_000.0,
            "verify_operations": 192.0,
            "method": "full_doubled_space_time_tensor_network",
            "cotengra": {
                "status": "PASS",
                "contraction_width_log2": 8.0,
                "contraction_flops": 425_703.0,
            },
            "line_graph_treewidth_lower": 3,
            "line_graph_treewidth_upper": 11,
            "cuts": {},
            "non_clifford": {},
            "assumptions": ["registered assumption"],
            "claim_scope": "registered attacks",
        },
        "exact_artifacts": {
            "schedule_search": {
                "lightcone_preflight": {
                    "single_bit_feasible": True,
                    "at_least_one_parity_feasible": True,
                },
                "target_checks": {"proxy_zero_swap": True},
            }
        },
    }
    config = {
        "gates": {"min_gamma_log10": 6.0},
        "schedule_search": {
            "require_cotengra": True,
            "require_proxy_zero_swap": True,
            "target_width_min": 17.0,
            "min_cotengra_flops": 192_000_000.0,
        },
    }
    decision = gate_c(candidate, config)
    assert not decision.passed
    assert "cotengra_width" in decision.reasons
    assert "cotengra_flops" in decision.reasons


def test_schedule_search_records_preregistered_filter_order():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "smoke.json").read_text())
    config["progress"]["enabled"] = False
    spec = BicycleFamilySpec(
        l=2,
        m=3,
        support_a=((0, 0), (0, 1)),
        support_b=((0, 0), (0, 1)),
    )
    h_x, h_z = build_bicycle_css(spec)
    topology = HardwareTopology.rectangular_grid(3, 4)
    finalists, report = search_adversarial_schedules(
        spec=spec,
        h_x=h_x,
        h_z=h_z,
        config=config,
        topology=topology,
        seed=20260723,
        progress=ProgressManager.from_config(config),
    )
    assert report["filter_order"] == [
        "css_relation_preservation",
        "live_target_zero_swap_embedding",
        "layer_diversity",
        "local_observable_construction",
        "gpu_lightcone_preflight",
        "ideal_signal_and_gf2_independence",
        "quick_cutwidth",
        "line_graph_treewidth",
        "cotengra_short",
        "cotengra_deep",
        "noisecert",
    ]
    assert report["generated_schedules"] > 0
    assert finalists
    assert all(
        item["circuit_spec"]["relation_preservation"]
        ["all_layers_css_rowspaces_preserved"]
        for item in finalists
    )
