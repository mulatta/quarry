"""quarry-server configuration.

Priority (highest → lowest):
  1. Environment variables  QUARRY_SERVER_<FIELD>
  2. Config file            ~/.config/quarry-server/config.toml  (non-secrets only)
  3. Defaults

Secrets (pg_conninfo) must not appear in the config file.
Use QUARRY_SERVER_PG_CONNINFO environment variable or systemd EnvironmentFile.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

# These fields contain credentials and must never be stored in a plaintext config file.
_SECRET_FIELDS: frozenset[str] = frozenset({"pg_conninfo"})


def config_path() -> Path:
    """Canonical path to the quarry-server config file (XDG-aware)."""
    xdg = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return xdg / "quarry-server" / "config.toml"


def _default_csr_dir() -> Path:
    xdg = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return xdg / "quarry" / "serving" / "csr"


class _TomlConfigSource(PydanticBaseSettingsSource):
    """Non-secret settings from ~/.config/quarry-server/config.toml."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        path = config_path()
        if path.is_file():
            with open(path, "rb") as f:
                raw = tomllib.load(f)
            secrets_found = _SECRET_FIELDS & raw.keys()
            if secrets_found:
                raise ValueError(
                    f"Secret fields must not appear in config file: {secrets_found}. "
                    "Use QUARRY_SERVER_PG_CONNINFO environment variable instead."
                )
            self._data: dict[str, Any] = raw
        else:
            self._data = {}

    def get_field_value(
        self,
        field: FieldInfo,
        field_name: str,  # noqa: ARG002
    ) -> tuple[Any, str, bool]:
        val = self._data.get(field_name)
        return val, field_name, val is not None

    def __call__(self) -> dict[str, Any]:
        return dict(self._data)


class ServerSettings(BaseSettings):
    # ── Secrets (env var only — never in config file) ────────────────────
    pg_conninfo: str = "host=/tmp/quarry-pg dbname=quarry"

    # ── Non-secrets (config file or env var) ─────────────────────────────
    csr_dir: Path = _default_csr_dir()
    host: str = "127.0.0.1"
    port: int = 8000
    require_auth: bool = False

    model_config = {"env_prefix": "QUARRY_SERVER_"}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG002
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG002
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,  # QUARRY_SERVER_* — highest priority
            _TomlConfigSource(settings_cls),  # ~/.config/quarry-server/config.toml
            # defaults follow implicitly
        )


server_settings = ServerSettings()
