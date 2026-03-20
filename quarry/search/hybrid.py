"""Hybrid search pipeline: query encoding → ANN + BM25 → RRF fusion.

Combines JinaEncoder (query encoding) with LanceStore (hybrid search).
Enriches results with PG metadata and supports MeSH expansion.
"""

from quarry.config import settings
from quarry.embed.jina import JinaEncoder
from quarry.store.lance import LanceStore
from quarry.store.pg import PGStore


class HybridSearcher:
    """End-to-end search: encode query → hybrid search → metadata enrichment."""

    def __init__(
        self,
        lance_uri: str | None = None,
        db: PGStore | None = None,
        encoder: JinaEncoder | None = None,
    ):
        self._lance = LanceStore(lance_uri or settings.lancedb_uri)
        self._db = db or PGStore(settings.pg_conninfo)
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
        """Search papers by natural language query."""
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
            work_ids = [r["work_id"] for r in results if "work_id" in r]
            if work_ids:
                works = self._db.get_works(work_ids)
                work_map = {w["work_id"]: w for w in works}
                for r in results:
                    wid = r.get("work_id")
                    if wid:
                        meta = work_map.get(wid, {})
                        r.update(
                            {
                                k: v
                                for k, v in meta.items()
                                if k not in ("title", "abstract")
                            }
                        )

            if mesh_expand:
                pmids = [r.get("pmid") for r in results if r.get("pmid")]
                if pmids:
                    mesh_facets = self._db.top_mesh(pmids, limit=10)
                    for r in results:
                        r["_mesh_facets"] = mesh_facets

        return results

    def similar(
        self,
        work_id: str,
        limit: int = 20,
        column: str = "vec_retrieval",
    ) -> list[dict]:
        """Find works similar to a given work (by its stored embedding)."""
        rows = (
            self._lance.table.search()
            .where(f"work_id = '{work_id}'")
            .select([column])
            .limit(1)
            .to_list()
        )
        if not rows:
            return []
        import numpy as np

        vec = np.array(rows[0][column], dtype=np.float32)
        results = self._lance.vector_search(vec, limit=limit + 1, column=column)
        return [r for r in results if r.get("work_id") != work_id][:limit]

    def mesh_expanded_search(
        self,
        descriptor_ui: str,
        limit: int = 50,
    ) -> list[int]:
        """Search papers via MeSH tree expansion."""
        tree_entries = self._db.query(
            f"SELECT tree_number FROM mesh_tree WHERE descriptor_ui = '{descriptor_ui}'"
        )
        if not tree_entries:
            return []

        all_uis: set[str] = {descriptor_ui}
        for entry in tree_entries:
            descendants = self._db.mesh_descendants(entry["tree_number"])
            for d in descendants:
                all_uis.add(d["descriptor_ui"])

        return self._db.mesh_expand_pmids(list(all_uis))[:limit]
