"""Hybrid search pipeline: query encoding → ANN + BM25 → RRF fusion.

Combines JinaEncoder (query encoding) with LanceStore (hybrid search).
Enriches results with DuckDB metadata and supports MeSH expansion.
"""

from quarry.config import settings
from quarry.embed.jina import JinaEncoder
from quarry.store.duckdb import DuckDBStore
from quarry.store.lance import LanceStore


class HybridSearcher:
    """End-to-end search: encode query → hybrid search → metadata enrichment."""

    def __init__(
        self,
        lance_uri: str | None = None,
        db: DuckDBStore | None = None,
        encoder: JinaEncoder | None = None,
    ):
        self._lance = LanceStore(lance_uri or settings.lancedb_uri)
        self._db = db or DuckDBStore()
        self._encoder = encoder

    def _get_encoder(self) -> JinaEncoder:
        if self._encoder is None:
            self._encoder = JinaEncoder(dim=256)
        return self._encoder

    def search(
        self,
        query: str,
        limit: int = 20,
        mode: str = "hybrid",
        enrich: bool = True,
        mesh_expand: bool = False,
    ) -> list[dict]:
        """Search papers by natural language query.

        Args:
            query: Natural language search query.
            limit: Max results to return.
            mode: "hybrid" (ANN + BM25), "vector" (ANN only), "text" (BM25 only).
            enrich: If True, fetch full metadata from DuckDB.
            mesh_expand: If True, expand query using MeSH tree hierarchy.
        """
        if mode == "text":
            results = self._lance.text_search(query, limit=limit)
        elif mode == "vector":
            encoder = self._get_encoder()
            q_vec = encoder.encode_queries([query])[0]
            results = self._lance.vector_search(q_vec, limit=limit)
        else:
            encoder = self._get_encoder()
            q_vec = encoder.encode_queries([query])[0]
            results = self._lance.hybrid_search(q_vec, query, limit=limit)

        if enrich and results:
            pmids = [int(r["pmid"]) for r in results]
            papers = self._db.get_papers(pmids)
            paper_map = {p["pmid"]: p for p in papers}
            for r in results:
                meta = paper_map.get(int(r["pmid"]), {})
                r.update(
                    {k: v for k, v in meta.items() if k not in ("title", "abstract")}
                )

            # Add MeSH facets
            if mesh_expand:
                mesh_facets = self._db.top_mesh(pmids, limit=10)
                for r in results:
                    r["_mesh_facets"] = mesh_facets

        return results

    def similar(
        self,
        pmid: str | int,
        limit: int = 20,
        column: str = "vec_retrieval",
    ) -> list[dict]:
        """Find papers similar to a given paper (by its stored embedding)."""
        pmid_str = str(pmid)
        rows = (
            self._lance.table.search()
            .where(f"pmid = '{pmid_str}'")
            .select([column])
            .limit(1)
            .to_list()
        )
        if not rows:
            return []
        import numpy as np

        vec = np.array(rows[0][column], dtype=np.float32)
        results = self._lance.vector_search(vec, limit=limit + 1, column=column)
        return [r for r in results if r["pmid"] != pmid_str][:limit]

    def mesh_expanded_search(
        self,
        descriptor_ui: str,
        limit: int = 50,
    ) -> list[int]:
        """Search papers via MeSH tree expansion.

        Finds all descendant descriptors of the given UI, then returns
        PMIDs that have any of those MeSH headings.
        """
        # Get tree numbers for this descriptor
        tree_entries = self._db.query(
            f"SELECT tree_number FROM mesh_tree WHERE descriptor_ui = '{descriptor_ui}'"
        )
        if not tree_entries:
            return []

        # Expand to all descendant UIs
        all_uis: set[str] = {descriptor_ui}
        for entry in tree_entries:
            descendants = self._db.mesh_descendants(entry["tree_number"])
            for d in descendants:
                all_uis.add(d["descriptor_ui"])

        # Get PMIDs
        return self._db.mesh_expand_pmids(list(all_uis))[:limit]
