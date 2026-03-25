"""Quarry configuration via environment variables / .env file.

Defaults use ~/quarry-data/. Override with QUARRY_* env vars for server deployment.
"""

from pathlib import Path

from pydantic_settings import BaseSettings

_DATA_DIR = Path.home() / "quarry-data"


class Settings(BaseSettings):
    model_config = {"env_prefix": "QUARRY_"}

    # PostgreSQL
    pg_conninfo: str = "host=/tmp/quarry-pg dbname=quarry"

    # ClickHouse
    ch_host: str = "localhost"
    ch_port: int = 9001
    ch_database: str = "quarry"

    # PubMed FTP paths (baseline + updates downloaded here)
    pubmed_baseline_dir: Path = _DATA_DIR / "pubmed" / "baseline"
    pubmed_update_dir: Path = _DATA_DIR / "pubmed" / "updatefiles"
    pubmed_mesh_dir: Path = _DATA_DIR / "pubmed" / "mesh"

    # iCite (figshare monthly release)
    icite_dir: Path = _DATA_DIR / "icite"

    # OpenAlex
    oa_s3_prefix: str = "s3://openalex/data/works"
    oa_local_dir: Path = _DATA_DIR / "oa" / "works"

    # Parquet output directories
    oa_parquet_dir: Path = _DATA_DIR / "parsed" / "oa"
    pm_parquet_dir: Path = _DATA_DIR / "parsed" / "pubmed"
    mesh_parquet_dir: Path = _DATA_DIR / "parsed" / "mesh"

    # Parquet intermediate (CH export → downstream consumers)
    parquet_dir: Path = _DATA_DIR / "parquet"

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

    # R2 (Cloudflare)
    r2_endpoint: str = ""
    r2_bucket: str = "quarry-data"

    # AWS Batch
    batch_job_queue: str = "quarry-spot"
    batch_job_definition: str = "quarry-etl"

    # Download
    ftp_parallel: int = 4


settings = Settings()
