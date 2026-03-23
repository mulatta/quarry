"""Quarry configuration via environment variables / .env file.

Defaults use ~/quarry-data/. Override with QUARRY_* env vars for server deployment.
"""

from pathlib import Path

from pydantic_settings import BaseSettings

_DATA_DIR = Path.home() / "quarry-data"


class Settings(BaseSettings):
    model_config = {"env_prefix": "QUARRY_"}

    # PostgreSQL
    # Default for dev (overridden by QUARRY_PG_CONNINFO env var from nix shell)
    pg_conninfo: str = "host=/tmp/quarry-pg dbname=quarry"

    # PubMed FTP paths (baseline + updates downloaded here)
    pubmed_baseline_dir: Path = _DATA_DIR / "pubmed" / "baseline"
    pubmed_update_dir: Path = _DATA_DIR / "pubmed" / "updatefiles"
    pubmed_mesh_dir: Path = _DATA_DIR / "pubmed" / "mesh"

    # iCite (figshare monthly release)
    icite_dir: Path = _DATA_DIR / "icite"

    # CSR mmap output
    csr_dir: Path = _DATA_DIR / "csr"

    # LanceDB
    lancedb_uri: str = str(_DATA_DIR / "lancedb")

    # FTP
    pubmed_ftp_host: str = "ftp.ncbi.nlm.nih.gov"
    pubmed_ftp_baseline: str = "/pubmed/baseline/"
    pubmed_ftp_updates: str = "/pubmed/updatefiles/"

    # MeSH descriptor XML (auto-discovered from NLM FTP)
    mesh_ftp_host: str = "nlmpubs.nlm.nih.gov"
    mesh_ftp_dir: str = "/online/mesh/MESH_FILES/xmlmesh/"

    # iCite figshare collection ID (stable, auto-discovers latest monthly snapshot)
    icite_figshare_collection: int = 4586573

    # OpenAlex
    oa_s3_prefix: str = "s3://openalex/data/works"
    oa_s3_concurrency: int = 8
    oa_pg_writers: int = 4
    oa_channel_buffer: int = 0  # 0 = auto (s3_concurrency * 2)
    oa_fetch_max_retries: int = 3
    oa_fetch_initial_backoff_ms: int = 2000
    oa_fetch_max_backoff_ms: int = 30000

    # PubMed build tuning
    pubmed_parse_threads: int = 0  # 0 = auto (mem / 0.4GB, capped by CPUs)
    pubmed_pg_writers: int = 4
    pubmed_channel_buffer: int = 0  # 0 = auto (parse_threads * 2)

    # Download
    ftp_parallel: int = 4


settings = Settings()
