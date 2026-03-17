"""Session subgraph using rustworkx for graph analytics."""

import rustworkx as rx

from quarry.store.csr import CSRGraph


class SessionSubgraph:
    """In-memory subgraph extracted from CSR for interactive analysis."""

    def __init__(self):
        self._graph = rx.PyDiGraph()
        self._id_to_node: dict[str, int] = {}
        self._node_to_id: dict[int, str] = {}

    def _ensure_node(self, pmid: str) -> int:
        if pmid not in self._id_to_node:
            node_idx = self._graph.add_node(pmid)
            self._id_to_node[pmid] = node_idx
            self._node_to_id[node_idx] = pmid
        return self._id_to_node[pmid]

    def add_nodes(self, ids: list[str]) -> None:
        for pmid in ids:
            self._ensure_node(pmid)

    def add_edges_from_csr(self, csr: CSRGraph, ids: list[str]) -> int:
        """Add edges between the given PMIDs using CSR graph data.

        Only adds edges where both endpoints are in the subgraph.
        Returns number of edges added.
        """
        id_set = set(ids)
        self.add_nodes(ids)
        added = 0

        for pmid in ids:
            for nb in csr.neighbors(pmid, direction="forward"):
                if nb in id_set:
                    src = self._id_to_node[pmid]
                    dst = self._id_to_node[nb]
                    self._graph.add_edge(src, dst, None)
                    added += 1

        return added

    @property
    def num_nodes(self) -> int:
        return self._graph.num_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.num_edges()

    def pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        """PageRank scores for nodes in the subgraph."""
        scores = rx.pagerank(self._graph, alpha=alpha)
        return {self._node_to_id[node]: score for node, score in scores.items()}

    def betweenness_centrality(self) -> dict[str, float]:
        """Betweenness centrality for nodes."""
        scores = rx.betweenness_centrality(self._graph)
        return {self._node_to_id[node]: score for node, score in scores.items()}

    def connected_components(self) -> list[set[str]]:
        """Weakly connected components."""
        components = rx.weakly_connected_components(self._graph)
        return [{self._node_to_id[n] for n in comp} for comp in components]

    def to_json(self) -> dict:
        """Export subgraph as JSON for visualization."""
        nodes = []
        for node_idx in self._graph.node_indices():
            pmid = self._node_to_id[node_idx]
            nodes.append({"id": pmid})

        edges = []
        for src, dst, _ in self._graph.weighted_edge_list():
            edges.append(
                {
                    "source": self._node_to_id[src],
                    "target": self._node_to_id[dst],
                }
            )

        return {"nodes": nodes, "edges": edges}
