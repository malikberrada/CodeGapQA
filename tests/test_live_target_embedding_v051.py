from pathlib import Path
import json
import networkx as nx

from codegap_qa.hardware import HardwareTopology, exact_zero_swap_embedding


def test_exact_zero_swap_embedding_finds_path_in_grid():
    topology = HardwareTopology.rectangular_grid(3, 3)
    result = exact_zero_swap_embedding(
        n=4,
        logical_edges=((0, 1), (1, 2), (2, 3)),
        topology=topology,
        seed=7,
        max_states=10000,
        timeout_seconds=2.0,
        heuristic_iterations=100,
        prefer_native=False,
    )
    assert result["status"] == "FOUND"
    layout = result["layout"]
    assert all(topology.graph.has_edge(layout[a], layout[b]) for a, b in ((0,1),(1,2),(2,3)))


def test_degree_impossibility_is_proved():
    topology = HardwareTopology.rectangular_grid(2, 3)
    edges = tuple((0, node) for node in range(1, 6))
    result = exact_zero_swap_embedding(
        n=6,
        logical_edges=edges,
        topology=topology,
        seed=1,
        prefer_native=False,
    )
    assert result["status"] == "PROVED_NO_EMBEDDING"
    assert result["reason"] == "logical_max_degree_exceeds_hardware"


def test_target_snapshot_topology(tmp_path: Path):
    payload = {
        "backend": "fake:target",
        "num_qubits": 4,
        "coupling_edges": [[0,1],[1,2],[2,3]],
        "structural_fingerprint": "abc",
    }
    path = tmp_path / "target_snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    topology = HardwareTopology.from_target_snapshot(path)
    assert topology.source == "live_backend_target_snapshot"
    assert topology.structural_fingerprint == "abc"
    assert topology.graph.number_of_edges() == 3
