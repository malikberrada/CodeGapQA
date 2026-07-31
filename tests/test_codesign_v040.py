from codegap_qa.bicycle import BicycleFamilySpec, build_bicycle_css
from codegap_qa.codesign import design_verifiable_mixing_circuit


def test_all_matching_layers_preserve_css():
    spec = BicycleFamilySpec(
        3,
        4,
        ((0, 0), (0, 1), (1, 0)),
        ((0, 2), (1, 1), (2, 3)),
    )
    h_x, h_z = build_bicycle_css(spec)
    config = {
        "circuit": {"theta_single": 0.3, "theta_pair": 0.4},
        "codesign": {
            "mixing_layers": 4,
            "axes": ["zz", "xx"],
            "require_css_automorphism_matchings": True,
            "max_verifier_masks": 8,
        },
    }
    circuit = design_verifiable_mixing_circuit(
        spec=spec, h_x=h_x, h_z=h_z, config=config, seed=7
    )
    assert circuit["logical_two_qubit_depth"] == 4
    assert circuit["relation_preservation"][
        "all_layers_css_rowspaces_preserved"
    ]
    assert circuit["relation_preservation"]["all_layers_involutive"]
