"""Hybrid search pipeline: query encoding → ANN + BM25 → RRF fusion.

Combines JinaEncoder (query encoding) with LanceStore (hybrid search).
Enriches results with PG metadata and supports MeSH expansion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import numpy as np
except ImportError:
    raise ImportError("pip install quarry[server]") from None

from quarry.config import settings
from quarry.store.lance import LanceStore
from quarry.store.pg import PGStore

if TYPE_CHECKING:
    from quarry.embed.jina import JinaEncoder
    from quarry.embed.reranker import JinaReranker


class HybridSearcher:
    """End-to-end search: encode query → hybrid search → metadata enrichment."""

    def __init__(
        self,
        lance_uri: str | None = None,
        db: PGStore | None = None,
        encoder: JinaEncoder | None = None,
        reranker: JinaReranker | None = None,
    ):
        self._lance = LanceStore(lance_uri or settings.lancedb_uri)
        self._db = db or PGStore(settings.pg_conninfo)
        self._encoder = encoder
        self._reranker = reranker

    def _get_encoder(self) -> JinaEncoder:
        if self._encoder is None:
            from quarry.embed.jina import JinaEncoder

            self._encoder = JinaEncoder(dim=256)
        return self._encoder

    def _get_reranker(self) -> JinaReranker:
        if self._reranker is None:
            from quarry.embed.reranker import JinaReranker

            self._reranker = JinaReranker(model_name=settings.rerank_model)
        return self._reranker

    def search(
        self,
        query: str,
        limit: int = 20,
        mode: str = "hybrid",
        enrich: bool = True,
        mesh_expand: bool = False,
        rerank: bool | None = None,
    ) -> list[dict]:
        """Search papers by natural language query.

        rerank: None → use settings.rerank_enabled. True/False to override.
        """
        do_rerank = settings.rerank_enabled if rerank is None else rerank

        # Over-fetch when reranking, capped to model context (max_pairs).
        fetch_limit = (
            min(limit * settings.rerank_candidate_multiplier, settings.rerank_max_pairs)
            if do_rerank
            else limit
        )

        if mode == "text":
            results = self._lance.text_search(query, limit=fetch_limit)
        elif mode == "vector":
            encoder = self._get_encoder()
            q_vec = encoder.encode_queries([query])[0]
            results = self._lance.vector_search(q_vec, limit=fetch_limit)
        else:
            encoder = self._get_encoder()
            q_vec = encoder.encode_queries([query])[0]
            results = self._lance.hybrid_search(q_vec, query, limit=fetch_limit)

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

        if do_rerank and results:
            results = self._rerank_and_blend(query, results, limit)
        else:
            results = results[:limit]

        return results

    def _rerank_and_blend(
        self, query: str, candidates: list[dict], limit: int
    ) -> list[dict]:
        """Cross-encoder rerank with position-aware RRF/rerank blending.

        Position-aware weights protect top retrieval results from reranker
        disagreement (top ranks rarely benefit from reordering, tail ranks do).
        """
        cap = settings.rerank_truncate_chars
        docs = [
            (f"{c.get('title') or ''}. {c.get('abstract') or ''}")[:cap]
            for c in candidates
        ]
        rerank_results = self._get_reranker().rerank(
            query, docs, top_n=None, batch_size=settings.rerank_batch_size
        )
        idx_to_score = {r["index"]: r["relevance_score"] for r in rerank_results}

        for i, c in enumerate(candidates):
            rrf_rank = i + 1
            if rrf_rank <= 3:
                rrf_w = settings.rerank_blend_top
            elif rrf_rank <= 10:
                rrf_w = settings.rerank_blend_mid
            else:
                rrf_w = settings.rerank_blend_tail
            rrf_score = 1.0 / rrf_rank
            r_score = float(idx_to_score.get(i, 0.0))
            c["_rrf_rank"] = rrf_rank
            c["_rerank_score"] = round(r_score, 6)
            c["_blended_score"] = round(rrf_w * rrf_score + (1 - rrf_w) * r_score, 6)

        candidates.sort(key=lambda c: c["_blended_score"], reverse=True)
        if settings.rerank_min_score > 0:
            candidates = [
                c
                for c in candidates
                if c["_blended_score"] >= settings.rerank_min_score
            ]
        return candidates[:limit]

    def similar(
        self,
        work_id: str,
        limit: int = 20,
        column: str = "vec_retrieval",
    ) -> list[dict]:
        """Find works similar to a given work (by its stored embedding)."""
        from quarry.store.lance import _WORK_ID_RE

        if not _WORK_ID_RE.match(work_id):
            return []
        rows = (
            self._lance.table.search()
            .where(f"work_id = '{work_id}'")
            .select([column])
            .limit(1)
            .to_list()
        )
        if not rows:
            return []
        vec = np.array(rows[0][column], dtype=np.float32)
        results = self._lance.vector_search(vec, limit=limit + 1, column=column)
        return [r for r in results if r.get("work_id") != work_id][:limit]

    def mesh_expanded_search(
        self,
        descriptor_ui: str,
        limit: int = 50,
    ) -> list[int]:
        """Search papers via MeSH tree expansion."""
        tree_entries = self._db.mesh_by_ui(descriptor_ui)
        if not tree_entries:
            return []

        all_uis: set[str] = {descriptor_ui}
        for entry in tree_entries:
            descendants = self._db.mesh_descendants(entry["tree_number"])
            for d in descendants:
                all_uis.add(d["descriptor_ui"])

        return self._db.mesh_expand_pmids(list(all_uis))[:limit]
