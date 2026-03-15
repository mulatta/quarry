"""CSR mmap graph: load and query citation graph built by etl/csr.py."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np


class CSRGraph:
    """Memory-mapped CSR citation graph with bidirectional traversal."""

    def __init__(self, csr_dir: Path):
        meta = json.loads((csr_dir / "meta.json").read_text())
        self.num_nodes: int = meta["num_nodes"]
        self.num_edges: int = meta["num_edges"]

        # Load id_map: index → openalex_id, and reverse lookup
        lines = (csr_dir / "id_map.bin").read_text().split("\n")
        self._idx_to_id: list[str] = lines
        self._id_to_idx: dict[str, int] = {oa_id: i for i, oa_id in enumerate(lines)}

        # mmap CSR arrays
        self._fwd_indptr = np.memmap(
            csr_dir / "forward" / "indptr.bin", dtype=np.uint64, mode="r"
        )
        self._fwd_indices = np.memmap(
            csr_dir / "forward" / "indices.bin", dtype=np.uint32, mode="r"
        )
        self._rev_indptr = np.memmap(
            csr_dir / "reverse" / "indptr.bin", dtype=np.uint64, mode="r"
        )
        self._rev_indices = np.memmap(
            csr_dir / "reverse" / "indices.bin", dtype=np.uint32, mode="r"
        )

    def _resolve(self, node_id: str) -> int | None:
        return self._id_to_idx.get(node_id)

    def _neighbors_idx(self, idx: int, direction: str) -> np.ndarray:
        if direction == "forward":
            indptr, indices = self._fwd_indptr, self._fwd_indices
        else:
            indptr, indices = self._rev_indptr, self._rev_indices
        start, end = int(indptr[idx]), int(indptr[idx + 1])
        return indices[start:end]

    def neighbors(self, node_id: str, direction: str = "forward") -> list[str]:
        """Get direct neighbors. direction: 'forward' (cites) or 'reverse' (cited by)."""
        idx = self._resolve(node_id)
        if idx is None:
            return []
        return [self._idx_to_id[int(i)] for i in self._neighbors_idx(idx, direction)]

    def degree(self, node_id: str) -> tuple[int, int]:
        """Return (out_degree, in_degree) for a node."""
        idx = self._resolve(node_id)
        if idx is None:
            return (0, 0)
        out_d = int(self._fwd_indptr[idx + 1] - self._fwd_indptr[idx])
        in_d = int(self._rev_indptr[idx + 1] - self._rev_indptr[idx])
        return (out_d, in_d)

    def k_hop(
        self, node_id: str, k: int, direction: str = "both", max_nodes: int = 100_000
    ) -> set[str]:
        """BFS k-hop neighborhood. direction: 'forward', 'reverse', or 'both'."""
        idx = self._resolve(node_id)
        if idx is None:
            return set()

        visited: set[int] = {idx}
        frontier: set[int] = {idx}
        directions = ["forward", "reverse"] if direction == "both" else [direction]

        for _ in range(k):
            next_frontier: set[int] = set()
            for node in frontier:
                for d in directions:
                    for nb in self._neighbors_idx(node, d):
                        nb_int = int(nb)
                        if nb_int not in visited:
                            next_frontier.add(nb_int)
                            visited.add(nb_int)
                            if len(visited) >= max_nodes:
                                return {self._idx_to_id[i] for i in visited}
            frontier = next_frontier
            if not frontier:
                break

        return {self._idx_to_id[i] for i in visited}

    def shortest_path(
        self, src: str, dst: str, max_depth: int = 10
    ) -> list[str] | None:
        """Bidirectional BFS shortest path between two nodes.

        Returns list of openalex_ids from src to dst, or None if no path.
        """
        src_idx = self._resolve(src)
        dst_idx = self._resolve(dst)
        if src_idx is None or dst_idx is None:
            return None
        if src_idx == dst_idx:
            return [src]

        # Forward BFS from src, reverse BFS from dst
        fwd_parent: dict[int, int | None] = {src_idx: None}
        rev_parent: dict[int, int | None] = {dst_idx: None}
        fwd_frontier: deque[int] = deque([src_idx])
        rev_frontier: deque[int] = deque([dst_idx])

        for _ in range(max_depth):
            # Expand forward
            if fwd_frontier:
                next_fwd: deque[int] = deque()
                while fwd_frontier:
                    node = fwd_frontier.popleft()
                    for nb in self._neighbors_idx(node, "forward"):
                        nb_int = int(nb)
                        if nb_int in rev_parent:
                            # Found meeting point
                            return self._reconstruct_path(
                                fwd_parent, rev_parent, node, nb_int
                            )
                        if nb_int not in fwd_parent:
                            fwd_parent[nb_int] = node
                            next_fwd.append(nb_int)
                fwd_frontier = next_fwd

            # Expand reverse
            if rev_frontier:
                next_rev: deque[int] = deque()
                while rev_frontier:
                    node = rev_frontier.popleft()
                    for nb in self._neighbors_idx(node, "reverse"):
                        nb_int = int(nb)
                        if nb_int in fwd_parent:
                            return self._reconstruct_path(
                                fwd_parent, rev_parent, nb_int, node
                            )
                        if nb_int not in rev_parent:
                            rev_parent[nb_int] = node
                            next_rev.append(nb_int)
                rev_frontier = next_rev

            if not fwd_frontier and not rev_frontier:
                break

        return None

    def _reconstruct_path(
        self,
        fwd_parent: dict[int, int | None],
        rev_parent: dict[int, int | None],
        fwd_meet: int,
        rev_meet: int,
    ) -> list[str]:
        """Reconstruct path from bidirectional BFS parents."""
        # Forward part: src → fwd_meet
        fwd_path: list[int] = []
        node: int | None = fwd_meet
        while node is not None:
            fwd_path.append(node)
            node = fwd_parent[node]
        fwd_path.reverse()

        # Reverse part: rev_meet → dst
        rev_path: list[int] = []
        node = rev_meet
        while node is not None:
            rev_path.append(node)
            node = rev_parent[node]

        return [self._idx_to_id[i] for i in fwd_path + rev_path]
