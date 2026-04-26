"""Cross-encoder reranker wrapping jina-reranker-v3.

v3 uses a custom `rerank()` method (not sentence-transformers /
AutoModelForSequenceClassification). Up to 64 documents in one forward pass
via the 131K-token context.
"""

import logging
from collections.abc import Sequence

log = logging.getLogger(__name__)

DEFAULT_MODEL = "jinaai/jina-reranker-v3"


class JinaReranker:
    """Thin wrapper around jina-reranker-v3 (Qwen3-0.6B + projector).

    Loads on first use. Single forward pass for the entire candidate list
    (cap: 64 docs, 131K context).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        try:
            import torch
            from transformers import AutoModel
        except ImportError:
            raise ImportError("pip install quarry[elt]") from None

        # Auto-pick CUDA when available — AutoModel's dtype="auto" doesn't
        # move weights to GPU on its own.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info("Loading reranker: %s on %s", model_name, device)
        self._torch = torch
        self._device = device
        self._model = AutoModel.from_pretrained(
            model_name,
            dtype="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        self._model = self._model.to(device)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
        batch_size: int | None = None,
    ) -> list[dict]:
        """Score (query, doc) pairs.

        Returns list of {relevance_score, document, index} sorted by score.
        Indices reference the original `documents` list.

        Splits into batches when `batch_size` is set — Jina v3 packs all docs
        into a single forward pass; on a 48GB GPU, ~24 docs is the safe ceiling
        before O(seq²) attention OOMs.
        """
        docs = list(documents)
        if not batch_size or len(docs) <= batch_size:
            return self._model.rerank(
                query=query, documents=docs, top_n=top_n, return_embeddings=False
            )

        all_results: list[dict] = []
        for start in range(0, len(docs), batch_size):
            chunk = docs[start : start + batch_size]
            batch_out = self._model.rerank(
                query=query, documents=chunk, top_n=None, return_embeddings=False
            )
            # Restore original indices
            for r in batch_out:
                r = dict(r)
                r["index"] = start + r["index"]
                all_results.append(r)

        all_results.sort(key=lambda r: r["relevance_score"], reverse=True)
        return all_results if top_n is None else all_results[:top_n]

    def unload(self):
        del self._model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
