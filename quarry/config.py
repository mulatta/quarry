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

    # Staging (Arrow Feather intermediate files)
    staging_dir: Path = _DATA_DIR / "staging"

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
    oa_t2_domains: set[str] = {
        "Health Sciences",
        "Life Sciences",
        "Physical Sciences",
        "Engineering",
    }

    # Rust build output (quarry-build CLI)
    build_output_dir: Path = _DATA_DIR / "build_output"

    # Batch sizes
    duckdb_batch_size: int = 10_000

    # Download
    ftp_parallel: int = 4


settings = Settings()
