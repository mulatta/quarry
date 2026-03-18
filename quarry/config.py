"""Quarry configuration via environment variables / .env file.

Defaults use ~/quarry-data/. Override with QUARRY_* env vars for server deployment.
"""

from pathlib import Path

from pydantic_settings import BaseSettings

_DATA_DIR = Path.home() / "quarry-data"


class Settings(BaseSettings):
    model_config = {"env_prefix": "QUARRY_"}

    # DuckDB
    duckdb_path: Path = _DATA_DIR / "quarry.duckdb"

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

    # FTP URLs
    pubmed_ftp_baseline: str = "ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
    pubmed_ftp_updates: str = "ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"

    # Batch sizes
    duckdb_batch_size: int = 10_000


settings = Settings()
