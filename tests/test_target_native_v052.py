from __future__ import annotations

import numpy as np

from codegap_qa.bicycle import BicycleFamilySpec, build_bicycle_css
from codegap_qa.codesign import build_tracked_frame_mixing_circuit
from codegap_qa.hardware import HardwareTopology
from codegap_qa.target_native import build_target_native_matching_pool


def test_target_native_pool_is_zero_swap_and_diverse():
    topology = HardwareTopology.rectangular_grid(4, 4)
    pool = build_target_native_matching_pool(
        topology=topology,
        n=12,
        seed=20260723,
        layout_trials=32,
        matching_trials=256,
        minimum_matchings=3,
    )
    assert pool["status"] == "FOUND"
    assert pool["matching_pool_size"] >= 3
    layout = tuple(pool["layout"])
    for matching in pool["matching_pool"]:
        assert matching["fixed_point_free_involution"]
        for left, right in matching["edges"]:
            assert topology.graph.has_edge(layout[left], layout[right])


def test_tracked_css_frame_is_exact_for_target_native_matchings():
    topology = HardwareTopology.rectangular_grid(3, 4)
    spec = BicycleFamilySpec(
        l=2,
        m=3,
        support_a=((0, 0), (0, 1)),
        support_b=((0, 0), (1, 1)),
    )
    h_x, h_z = build_bicycle_css(spec)
    pool = build_target_native_matching_pool(
        topology=topology,
        n=spec.n,
        seed=17,
        layout_trials=32,
        matching_trials=256,
        minimum_matchings=2,
    )
    assert pool["status"] in {"FOUND", "BEST_PARTIAL_POOL"}
    assert pool["matching_pool_size"] >= 2
    circuit = build_tracked_frame_mixing_circuit(
        spec=spec,
        h_x=h_x,
        h_z=h_z,
        matching_pool=pool["matching_pool"],
        matching_indices=(0, 1, 0, 1),
        axes=("zz", "xx", "zz", "xx"),
        config={
            "circuit": {
                "theta_single": np.pi / 8,
                "theta_pair": np.pi / 7,
            },
            "codesign": {"max_verifier_masks": 16},
        },
        seed=19,
        schedule_metadata={
            "matching_source": "target_native_tracked_frame",
            "pinned_layout": pool["layout"],
        },
    )
    relation = circuit["relation_preservation"]
    assert relation["all_layers_tracked_frame_transitions_exact"]
    assert relation["final_frame_matches_cumulative_permutation"]
    assert relation["verifier_masks_from_final_tracked_frame"]
    assert circuit["schedule"]["relation_mode"] == "tracked_css_wire_frame"
