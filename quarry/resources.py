"""Dagster resources: shared connections for PostgreSQL and LanceDB.

PGResource manages a single connection per run via yield_for_execution.
"""

from contextlib import contextmanager

from dagster import ConfigurableResource
from pydantic import PrivateAttr

from quarry.config import settings
from quarry.store.lance import LanceStore
from quarry.store.pg import PGStore


class PGResource(ConfigurableResource):
    """PostgreSQL connection as a Dagster resource.

    Lifecycle: single connection opened at run start, closed at run end.
    Assets access it via the `store` property.
    """

    conninfo: str = settings.pg_conninfo
    _store: PGStore | None = PrivateAttr(default=None)

    @contextmanager
    def yield_for_execution(self, context):
        store = PGStore(conninfo=self.conninfo)
        store.init_schema()
        self._store = store
        try:
            yield self
        finally:
            store.close()
            self._store = None

    @property
    def store(self) -> PGStore:
        assert self._store is not None, "PGResource not initialized"
        return self._store


class LanceDBResource(ConfigurableResource):
    """LanceDB connection as a Dagster resource."""

    uri: str = settings.lancedb_uri

    def get_store(self) -> LanceStore:
        return LanceStore(self.uri)
