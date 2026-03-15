"""Quarry configuration via environment variables / .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "QUARRY_"}

    # OpenAlex snapshot paths (on psi server)
    openalex_works_dir: Path = Path("~/data/openalex/works").expanduser()
    openalex_manifest: Path = Path("~/data/openalex/works/manifest").expanduser()

    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_database: str = "quarry"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""

    # CSR mmap output (on RAID for space)
    csr_dir: Path = Path("/workspace/seungwon/quarry/csr")

    # LanceDB (on RAID)
    lancedb_uri: str = "/workspace/seungwon/quarry/lancedb"


settings = Settings()
