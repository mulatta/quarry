"""Dagster resources: shared connections for DuckDB and LanceDB."""

from pathlib import Path

from dagster import ConfigurableResource

from quarry.config import settings
from quarry.store.duckdb import DuckDBStore
from quarry.store.lance import LanceStore


class DuckDBResource(ConfigurableResource):
    """DuckDB connection as a Dagster resource."""

    db_path: str = str(settings.duckdb_path)

    def get_store(self) -> DuckDBStore:
        db = DuckDBStore(db_path=Path(self.db_path))
        db.init_schema()
        return db


class LanceDBResource(ConfigurableResource):
    """LanceDB connection as a Dagster resource."""

    uri: str = settings.lancedb_uri

    def get_store(self) -> LanceStore:
        return LanceStore(self.uri)
