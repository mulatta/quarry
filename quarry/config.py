"""Quarry configuration via environment variables / .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "QUARRY_"}

    # DuckDB
    duckdb_path: Path = Path("/workspace/seungwon/quarry/quarry.duckdb")

    # PubMed FTP paths (baseline + updates downloaded here)
    pubmed_baseline_dir: Path = Path("/workspace/seungwon/quarry/pubmed/baseline")
    pubmed_update_dir: Path = Path("/workspace/seungwon/quarry/pubmed/updatefiles")
    pubmed_mesh_dir: Path = Path("/workspace/seungwon/quarry/pubmed/mesh")

    # iCite (figshare monthly release)
    icite_dir: Path = Path("/workspace/seungwon/quarry/icite")

    # CSR mmap output
    csr_dir: Path = Path("/workspace/seungwon/quarry/csr")

    # LanceDB
    lancedb_uri: str = "/workspace/seungwon/quarry/lancedb"

    # FTP URLs
    pubmed_ftp_baseline: str = "ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
    pubmed_ftp_updates: str = "ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"

    # Batch sizes
    duckdb_batch_size: int = 10_000


settings = Settings()
