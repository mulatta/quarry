"""Dagster resources: shared connections for DuckDB and LanceDB.

DuckDBResource manages a single connection per run via yield_for_execution,
avoiding file-lock contention across assets.
"""

from contextlib import contextmanager
from pathlib import Path

from dagster import ConfigurableResource
from pydantic import PrivateAttr

from quarry.config import settings
from quarry.store.duckdb import DuckDBStore
from quarry.store.lance import LanceStore


class DuckDBResource(ConfigurableResource):
    """DuckDB connection as a Dagster resource.

    Lifecycle: single connection opened at run start, closed at run end.
    Assets access it via the `store` property.
    """

    db_path: str = str(settings.duckdb_path)
    _store: DuckDBStore | None = PrivateAttr(default=None)

    @contextmanager
    def yield_for_execution(self, context):
        store = DuckDBStore(db_path=Path(self.db_path))
        store.init_schema()
        self._store = store
        try:
            yield self
        finally:
            store.close()
            self._store = None

    @property
    def store(self) -> DuckDBStore:
        assert self._store is not None, "DuckDBResource not initialized"
        return self._store


class LanceDBResource(ConfigurableResource):
    """LanceDB connection as a Dagster resource."""

    uri: str = settings.lancedb_uri

    def get_store(self) -> LanceStore:
        return LanceStore(self.uri)
