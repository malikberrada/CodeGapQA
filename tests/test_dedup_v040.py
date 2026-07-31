import networkx as nx

from codegap_qa.bicycle import BicycleFamilySpec
from codegap_qa.dedup import IsomorphismRegistry, colored_interaction_graph


def test_qc_translation_and_exact_isomorphism_dedup():
    left = BicycleFamilySpec(
        2, 4, ((0, 0), (0, 1)), ((1, 0), (1, 2))
    )
    right = BicycleFamilySpec(
        2, 4, ((0, 1), (0, 2)), ((1, 1), (1, 3))
    )
    edges = ((0, 1), (2, 3), (4, 5), (6, 7))
    registry = IsomorphismRegistry()
    first = registry.register(
        candidate_id="a",
        spec=left,
        graph=colored_interaction_graph(16, edges),
    )
    second = registry.register(
        candidate_id="b",
        spec=right,
        graph=colored_interaction_graph(16, edges),
    )
    assert not first.duplicate
    assert second.duplicate
    assert second.representative_id == "a"
