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
    ch_port: int = 9001
    ch_database: str = "quarry"

    # PubMed FTP paths
    pubmed_baseline_dir: Path = Path()
    pubmed_update_dir: Path = Path()
    pubmed_mesh_dir: Path = Path()

    # iCite
    icite_dir: Path = Path()

    # OpenAlex
    oa_s3_prefix: str = "s3://openalex/data/works"
    oa_local_dir: Path = Path()

    # Parquet output
    oa_parquet_dir: Path = Path()
    pm_parquet_dir: Path = Path()
    mesh_parquet_dir: Path = Path()

    # Parquet intermediate (CH export -> downstream)
    parquet_dir: Path = Path()

    # CSR mmap output
    csr_dir: Path = Path()

    # LanceDB
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

    # AWS Batch
    batch_job_queue: str = "quarry-spot"
    batch_job_definition: str = "quarry-etl"

    # Download
    ftp_parallel: int = 4

    @model_validator(mode="before")
    @classmethod
    def _resolve_data_paths(cls, values: dict) -> dict:
        raw = values.get("data_dir")
        d = Path(str(raw)) if raw is not None else _xdg_data_home() / "quarry"
        defaults = {
            "pubmed_baseline_dir": d / "pubmed" / "baseline",
            "pubmed_update_dir": d / "pubmed" / "updatefiles",
            "pubmed_mesh_dir": d / "pubmed" / "mesh",
            "icite_dir": d / "icite",
            "oa_local_dir": d / "oa" / "works",
            "oa_parquet_dir": d / "parsed" / "oa",
            "pm_parquet_dir": d / "parsed" / "pubmed",
            "mesh_parquet_dir": d / "parsed" / "mesh",
            "parquet_dir": d / "parquet",
            "csr_dir": d / "csr",
            "lancedb_uri": str(d / "lancedb"),
        }
        for k, v in defaults.items():
            values.setdefault(k, v)
        return values


settings = Settings()
