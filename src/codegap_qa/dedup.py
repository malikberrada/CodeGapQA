from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from .bicycle import BicycleFamilySpec


def _translate(
    support: tuple[tuple[int, int], ...],
    l: int,
    m: int,
    dx: int,
    dy: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted((((x - dx) % l, (y - dy) % m) for x, y in support))
    )


def canonical_qc_spec(spec: BicycleFamilySpec) -> tuple:
    """Canonicalize simultaneous QC translations and the A/B block swap.

    Multiplication of both QC polynomials by the same group monomial only
    permutes checks/qubits. Swapping the two bicycle blocks is also quotiented
    here, but graph isomorphism is still confirmed separately before a
    candidate is discarded.
    """

    variants: list[tuple] = []
    for dx in range(spec.l):
        for dy in range(spec.m):
            a = _translate(spec.support_a, spec.l, spec.m, dx, dy)
            b = _translate(spec.support_b, spec.l, spec.m, dx, dy)
            variants.append((spec.l, spec.m, a, b))
            variants.append((spec.l, spec.m, b, a))
    return min(variants)


def colored_interaction_graph(
    n: int,
    edges: tuple[tuple[int, int], ...],
) -> nx.Graph:
    graph = nx.Graph()
    block = n // 2
    for node in range(n):
        graph.add_node(
            node,
            block="left" if node < block else "right",
        )
    graph.add_edges_from(tuple(sorted((int(a), int(b)))) for a, b in edges)
    for node in graph:
        graph.nodes[node]["degree"] = int(graph.degree[node])
    return graph


def wl_hash(graph: nx.Graph) -> str:
    return nx.weisfeiler_lehman_graph_hash(
        graph,
        node_attr="block",
        iterations=4,
        digest_size=20,
    )


def exact_isomorphic(left: nx.Graph, right: nx.Graph) -> bool:
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        "block", None
    )
    return nx.is_isomorphic(left, right, node_match=node_match)


@dataclass(frozen=True)
class DedupDecision:
    duplicate: bool
    representative_id: str | None
    canonical_key: tuple
    wl_hash: str
    exact_isomorphism_checked: bool
    reason: str


class IsomorphismRegistry:
    """Deduplicate after algebraic canonicalization and before layout/hardness."""

    def __init__(self) -> None:
        self._canonical: dict[tuple, str] = {}
        self._buckets: dict[str, list[tuple[str, nx.Graph]]] = {}
        self.canonical_duplicates = 0
        self.wl_bucket_hits = 0
        self.exact_duplicates = 0
        self.unique = 0

    def register(
        self,
        *,
        candidate_id: str,
        spec: BicycleFamilySpec,
        graph: nx.Graph,
    ) -> DedupDecision:
        key = canonical_qc_spec(spec)
        graph_hash = wl_hash(graph)
        representative = self._canonical.get(key)
        if representative is not None:
            self.canonical_duplicates += 1
            return DedupDecision(
                duplicate=True,
                representative_id=representative,
                canonical_key=key,
                wl_hash=graph_hash,
                exact_isomorphism_checked=False,
                reason="qc_translation_or_block_swap",
            )

        bucket = self._buckets.setdefault(graph_hash, [])
        if bucket:
            self.wl_bucket_hits += 1
        for previous_id, previous_graph in bucket:
            if exact_isomorphic(previous_graph, graph):
                self.exact_duplicates += 1
                self._canonical[key] = previous_id
                return DedupDecision(
                    duplicate=True,
                    representative_id=previous_id,
                    canonical_key=key,
                    wl_hash=graph_hash,
                    exact_isomorphism_checked=True,
                    reason="wl_bucket_exact_graph_isomorphism",
                )

        self._canonical[key] = candidate_id
        bucket.append((candidate_id, graph.copy()))
        self.unique += 1
        return DedupDecision(
            duplicate=False,
            representative_id=None,
            canonical_key=key,
            wl_hash=graph_hash,
            exact_isomorphism_checked=bool(bucket[:-1]),
            reason="unique_representative",
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "unique_representatives": self.unique,
            "qc_canonical_duplicates": self.canonical_duplicates,
            "wl_bucket_hits": self.wl_bucket_hits,
            "exact_isomorphic_duplicates": self.exact_duplicates,
            "equivalence_relation": [
                "simultaneous_QC_translation",
                "A_B_block_swap",
                "colored_WL_hash_bucket",
                "exact_colored_graph_isomorphism",
            ],
        }
