"""quarry-server configuration — independent of quarry ELT/CLI config.

Environment prefix: QUARRY_SERVER_
Override file:      .env.server (optional)

Only the settings the MCP server actually needs are exposed here.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


def _default_csr_dir() -> Path:
    import os

    xdg = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return xdg / "quarry" / "serving" / "csr"


class ServerSettings(BaseSettings):
    model_config = {"env_prefix": "QUARRY_SERVER_", "env_file": ".env.server"}

    # PostgreSQL connection string
    pg_conninfo: str = "host=/tmp/quarry-pg dbname=quarry"

    # Path to the mmap-backed CSR binary files (indptr.bin, indices.bin, id_map.bin)
    csr_dir: Path = _default_csr_dir()

    # HTTP bind address and port
    host: str = "127.0.0.1"
    port: int = 8000

    # Set True to require a valid API key on every request (Bearer token)
    require_auth: bool = False


server_settings = ServerSettings()
