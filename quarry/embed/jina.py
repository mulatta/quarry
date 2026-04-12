"""Jina embeddings v5 nano encoder with matryoshka truncation.

Eval result: dim=256 is the sweet spot (98.4% R@10 retention vs full dim).
Two LoRA tasks: retrieval (search) and clustering (UMAP viz).
"""

import numpy as np

MODEL_NAME = "jinaai/jina-embeddings-v5-text-nano"
FULL_DIM = 768
DEFAULT_DIM = 256
DEFAULT_BATCH_SIZE = 32


def _truncate_norm(embs: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation: slice first `dim` dims, re-normalize."""
    if dim >= embs.shape[1]:
        return embs
    trunc = embs[:, :dim].copy()
    norms = np.linalg.norm(trunc, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return trunc / norms


class JinaEncoder:
    """Stateful encoder wrapping jina-v5 nano with dual LoRA."""

    def __init__(
        self,
        device: str | None = None,
        dim: int = DEFAULT_DIM,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_tokens: int | None = None,
    ):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install quarry[elt]") from None

        self.dim = dim
        self.batch_size = batch_size
        self.max_tokens = max_tokens

        def _default_device() -> str:
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        self.device = device or _default_device()
        # mem_efficient SDPA is 34% faster than flash on A6000 (cc 8.6) for this model
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.set_float32_matmul_precision("high")

        self._model = SentenceTransformer(
            MODEL_NAME,
            device=self.device,
            model_kwargs={"torch_dtype": torch.bfloat16},
            trust_remote_code=True,
        )
        if max_tokens:
            self._model.max_seq_length = max_tokens

        # torch.compile: triton kernel fusion (~45% speedup)
        # "default" mode: no CUDA graphs (model has 97 dynamic-shape partitions
        # from rotary embeddings; reduce-overhead causes 18GB graph pool → OOM)
        if self.device == "cuda":
            self._model[0].model = torch.compile(self._model[0].model)

        self._torch = torch

    def _encode(
        self, texts: list[str], task: str, prompt_name: str | None = None
    ) -> np.ndarray:
        kwargs = {
            "task": task,
            "batch_size": self.batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        embs = self._model.encode(texts, **kwargs)
        return _truncate_norm(embs, self.dim)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Encode documents for indexing (retrieval LoRA, document prompt)."""
        return self._encode(texts, task="retrieval", prompt_name="document")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Encode queries for search (retrieval LoRA, query prompt)."""
        return self._encode(texts, task="retrieval", prompt_name="query")

    def encode_clustering(self, texts: list[str]) -> np.ndarray:
        """Encode for clustering/UMAP (clustering LoRA)."""
        return self._encode(texts, task="clustering")

    def unload(self):
        """Free GPU/device memory."""
        del self._model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
