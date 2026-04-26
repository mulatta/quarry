"""Quarry configuration via environment variables / .env file.

Follows XDG Base Directory spec. Override with QUARRY_* env vars or .env file.
Defaults: $XDG_DATA_HOME/quarry (~/.local/share/quarry)
"""

import os
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings


def _xdg_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))


class Settings(BaseSettings):
    model_config = {"env_prefix": "QUARRY_", "env_file": ".env"}

    # Base data directory (XDG_DATA_HOME/quarry by default)
    data_dir: Path = _xdg_data_home() / "quarry"

    # PostgreSQL
    pg_conninfo: str = "host=/tmp/quarry-pg dbname=quarry"

    # ClickHouse
    ch_host: str = "localhost"
    ch_port: int = 9000
    ch_database: str = "quarry"
    ch_export_max_concurrent: int = 4
    ch_export_row_group_size: int = 100_000  # parquet row group size for works export

    # --- raw/ (sync stage: source downloads) ---
    pubmed_baseline_dir: Path = Path()
    pubmed_update_dir: Path = Path()
    pubmed_mesh_dir: Path = Path()
    icite_dir: Path = Path()
    oa_local_dir: Path = Path()

    # --- staging/ (parse + CH export) ---
    oa_s3_prefix: str = "s3://openalex/data/works"
    oa_parquet_dir: Path = Path()
    pm_parquet_dir: Path = Path()
    mesh_parquet_dir: Path = Path()
    parquet_dir: Path = Path()

    # --- serving/ (final consumers) ---
    csr_dir: Path = Path()
    lancedb_uri: str = ""

    # FTP
    pubmed_ftp_host: str = "ftp.ncbi.nlm.nih.gov"
    pubmed_ftp_baseline: str = "/pubmed/baseline/"
    pubmed_ftp_updates: str = "/pubmed/updatefiles/"

    # MeSH descriptor XML
    mesh_ftp_host: str = "nlmpubs.nlm.nih.gov"
    mesh_ftp_dir: str = "/online/mesh/MESH_FILES/xmlmesh/"

    # iCite figshare collection ID
    icite_figshare_collection: int = 4586573

    # R2 (Cloudflare)
    r2_endpoint: str = ""
    r2_bucket: str = "quarry-data"

    # Embeddings
    embed_batch_size: int = 32  # GPU forward pass batch; OOM-sensitive (VRAM dependent)
    embed_max_tokens: int = (
        1024  # tokenizer truncation; prevents GPU OOM from long texts
    )
    embed_encode_batch: int = 1024  # works per GPU encode batch (CH fetch → encode)
    embed_index_accelerator: str | None = (
        None  # "cuda" for GPU-accelerated IVF_PQ k-means, None for CPU
    )

    # Reranker (search-time cross-encoder, qmd-style multi-stage retrieval)
    rerank_model: str = "jinaai/jina-reranker-v3"  # CC BY-NC 4.0
    rerank_enabled: bool = True
    rerank_candidate_multiplier: int = 3  # over-fetch limit×N before rerank
    rerank_max_pairs: int = 24  # cap (Jina v3 60-doc OOMs at ~22GB on 48GB GPU)
    rerank_batch_size: int = 12  # split rerank calls; <=24 for headroom
    rerank_min_score: float = 0.0  # blended-score cutoff
    rerank_blend_top: float = 0.75  # rrf weight for rank 1-3
    rerank_blend_mid: float = 0.60  # rrf weight for rank 4-10
    rerank_blend_tail: float = 0.40  # rrf weight for rank 11+
    rerank_truncate_chars: int = 2000  # title+abstract char cap (~500 tokens)
    embed_allowed_types: list[str] = [
        "article",
        "preprint",
        "review",
        "letter",
        "editorial",
    ]

    # MCP HTTP Server
    mcp_host: str = "127.0.0.1"  # QUARRY_MCP_HOST — set to 0.0.0.0 for nginx/Tailscale
    mcp_port: int = 8000  # QUARRY_MCP_PORT — configurable to avoid conflicts

    # Download
    ftp_parallel: int = 4

    @model_validator(mode="before")
    @classmethod
    def _resolve_data_paths(cls, values: dict) -> dict:
        raw = values.get("data_dir")
        d = Path(str(raw)) if raw is not None else _xdg_data_home() / "quarry"
        raw = d / "raw"
        stg = d / "staging"
        srv = d / "serving"
        defaults = {
            # raw/
            "pubmed_baseline_dir": raw / "pubmed" / "baseline",
            "pubmed_update_dir": raw / "pubmed" / "updatefiles",
            "pubmed_mesh_dir": raw / "pubmed" / "mesh",
            "icite_dir": raw / "icite",
            "oa_local_dir": raw / "openalex" / "works",
            # staging/
            "oa_parquet_dir": stg / "oa",
            "pm_parquet_dir": stg / "pubmed",
            "mesh_parquet_dir": stg / "mesh",
            "parquet_dir": stg / "export",
            # serving/
            "csr_dir": srv / "csr",
            "lancedb_uri": str(srv / "lancedb"),
        }
        for k, v in defaults.items():
            values.setdefault(k, v)
        return values


settings = Settings()
