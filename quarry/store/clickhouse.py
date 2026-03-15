"""ClickHouse query client for metadata retrieval and analytics."""

from __future__ import annotations

import re

import clickhouse_connect
from clickhouse_connect.driver.client import Client


# Only allow SELECT statements (prevent mutation via query interface)
_SELECT_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)


class ClickHouseStore:
    def __init__(
        self, host: str = "localhost", port: int = 8123, database: str = "quarry"
    ):
        self._host = host
        self._port = port
        self._database = database

    def _client(self) -> Client:
        return clickhouse_connect.get_client(
            host=self._host, port=self._port, database=self._database
        )

    def get_paper(self, openalex_id: str) -> dict | None:
        """Get a single paper by openalex_id."""
        client = self._client()
        try:
            result = client.query(
                f"SELECT * FROM {self._database}.papers WHERE openalex_id = {{id:String}}",
                parameters={"id": openalex_id},
            )
            if not result.result_rows:
                return None
            return dict(zip(result.column_names, result.result_rows[0]))
        finally:
            client.close()

    def get_papers(self, ids: list[str]) -> list[dict]:
        """Get multiple papers by openalex_id list."""
        if not ids:
            return []
        client = self._client()
        try:
            result = client.query(
                f"SELECT * FROM {self._database}.papers WHERE openalex_id IN {{ids:Array(String)}}",
                parameters={"ids": ids},
            )
            return [dict(zip(result.column_names, row)) for row in result.result_rows]
        finally:
            client.close()

    def query(self, sql: str) -> list[dict]:
        """Execute a read-only SQL query. Only SELECT allowed."""
        if not _SELECT_RE.match(sql):
            raise ValueError("Only SELECT queries are allowed")
        client = self._client()
        try:
            result = client.query(sql)
            return [dict(zip(result.column_names, row)) for row in result.result_rows]
        finally:
            client.close()

    def top_fields(self, ids: list[str], limit: int = 10) -> list[dict]:
        """Top research fields for a set of paper IDs."""
        if not ids:
            return []
        client = self._client()
        try:
            result = client.query(
                f"""SELECT field_name, count() AS cnt
                    FROM {self._database}.papers
                    WHERE openalex_id IN {{ids:Array(String)}} AND field_name != ''
                    GROUP BY field_name ORDER BY cnt DESC LIMIT {{limit:UInt32}}""",
                parameters={"ids": ids, "limit": limit},
            )
            return [dict(zip(result.column_names, row)) for row in result.result_rows]
        finally:
            client.close()

    def year_distribution(self, ids: list[str]) -> list[dict]:
        """Publication year distribution for a set of paper IDs."""
        if not ids:
            return []
        client = self._client()
        try:
            result = client.query(
                f"""SELECT pub_year, count() AS cnt
                    FROM {self._database}.papers
                    WHERE openalex_id IN {{ids:Array(String)}} AND pub_year > 0
                    GROUP BY pub_year ORDER BY pub_year""",
                parameters={"ids": ids},
            )
            return [dict(zip(result.column_names, row)) for row in result.result_rows]
        finally:
            client.close()

    def top_authors(self, ids: list[str], limit: int = 10) -> list[dict]:
        """Top authors for a set of paper IDs."""
        if not ids:
            return []
        client = self._client()
        try:
            result = client.query(
                f"""SELECT a.author_name, a.author_id, count() AS cnt
                    FROM {self._database}.authors a
                    WHERE a.openalex_id IN {{ids:Array(String)}} AND a.author_name != ''
                    GROUP BY a.author_name, a.author_id ORDER BY cnt DESC LIMIT {{limit:UInt32}}""",
                parameters={"ids": ids, "limit": limit},
            )
            return [dict(zip(result.column_names, row)) for row in result.result_rows]
        finally:
            client.close()
