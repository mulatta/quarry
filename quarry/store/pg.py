"""PostgreSQL store: schema creation and query interface.

Bulk loading is handled by Dagster assets (load.py) via COPY FROM.
This module provides schema DDL and read-only query methods.
"""

import re
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

# Only allow SELECT / WITH ... SELECT / EXPLAIN
_READ_ONLY_RE = re.compile(r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)

# Single source of truth: sql/schema.sql (shared with Rust include_str! and nix process-compose)
_SCHEMA_SQL = Path(__file__).resolve().parent.parent.parent / "sql" / "schema.sql"


def _split_sql(ddl: str) -> list[str]:
    """Split SQL on top-level semicolons, preserving $$ blocks intact."""
    stmts: list[str] = []
    current: list[str] = []
    in_dollar = False
    for line in ddl.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        if "$$" in line:
            in_dollar = not in_dollar
        current.append(line)
        if not in_dollar and stripped.endswith(";"):
            stmt = "\n".join(current).strip().rstrip(";").strip()
            if stmt:
                stmts.append(stmt)
            current = []
    if current:
        stmt = "\n".join(current).strip().rstrip(";").strip()
        if stmt:
            stmts.append(stmt)
    return stmts


class PGStore:
    """PostgreSQL store for PubMed + OpenAlex metadata."""

    def __init__(self, conninfo: str):
        self._conninfo = conninfo
        self._conn: psycopg.Connection | None = None

    @property
    def conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._conninfo, autocommit=True)
        return self._conn

    def close(self):
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def init_schema(self):
        """Create all tables if they don't exist.

        Reads from sql/schema.sql — the single source of truth shared with
        Rust (include_str!) and nix process-compose (initialDatabases).
        """
        ddl = _SCHEMA_SQL.read_text()
        for stmt in _split_sql(ddl):
            self.conn.execute(stmt)

    # -- Read queries --

    def query(self, sql: str) -> list[dict]:
        """Execute a read-only SQL query via quarry_ro role.

        Defence in depth: regex pre-check + PG role-level enforcement.
        """
        if not _READ_ONLY_RE.match(sql):
            raise ValueError("Only SELECT/WITH/EXPLAIN queries are allowed")
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SET ROLE quarry_ro")
            try:
                cur.execute(sql)
                return cur.fetchall()
            finally:
                cur.execute("RESET ROLE")

    def mesh_descendants(self, tree_prefix: str) -> list[dict]:
        """Get all MeSH descriptors under a tree number prefix."""
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT DISTINCT descriptor_ui, descriptor_name, tree_number "
                "FROM mesh_tree WHERE tree_number LIKE %s "
                "ORDER BY tree_number",
                (tree_prefix + "%",),
            )
            return cur.fetchall()

    def mesh_search_by_name(self, name: str, limit: int = 10) -> list[dict]:
        """Search MeSH descriptors by name (ILIKE, parameterized)."""
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT DISTINCT descriptor_ui, descriptor_name "
                "FROM mesh_tree WHERE descriptor_name ILIKE %s LIMIT %s",
                (f"%{name}%", limit),
            )
            return cur.fetchall()

    def mesh_by_ui(self, descriptor_ui: str) -> list[dict]:
        """Get all tree entries for a MeSH descriptor UI."""
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM mesh_tree WHERE descriptor_ui = %s",
                (descriptor_ui,),
            )
            return cur.fetchall()

    def mesh_expand_pmids(self, descriptor_uis: list[str]) -> list[int]:
        """Get PMIDs that have any of the given MeSH descriptor UIs."""
        if not descriptor_uis:
            return []
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT pmid FROM mesh_headings "
                "WHERE descriptor_ui = ANY(%s) "
                "AND pmid IN (SELECT pmid FROM papers WHERE NOT is_deleted)",
                (descriptor_uis,),
            )
            return [row[0] for row in cur.fetchall()]

    def top_mesh(self, pmids: list[int], limit: int = 10) -> list[dict]:
        """Top MeSH descriptors for a set of PMIDs."""
        if not pmids:
            return []
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT descriptor_ui, descriptor_name, count(*) AS cnt "
                "FROM mesh_headings WHERE pmid = ANY(%s) "
                "GROUP BY descriptor_ui, descriptor_name ORDER BY cnt DESC LIMIT %s",
                (pmids, limit),
            )
            return cur.fetchall()

    # -- v2 (OpenAlex) read queries --

    def get_work(self, work_id: str) -> dict | None:
        """Get a single work by OpenAlex work_id."""
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM works WHERE work_id = %s", (work_id,))
            return cur.fetchone()

    def get_works(self, work_ids: list[str]) -> list[dict]:
        """Get multiple works by work_id list."""
        if not work_ids:
            return []
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM works WHERE work_id = ANY(%s)",
                (work_ids,),
            )
            return cur.fetchall()

    def get_work_by_pmid(self, pmid: int) -> dict | None:
        """Get a single work by PMID (via works table)."""
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM works WHERE pmid = %s", (pmid,))
            return cur.fetchone()

    def get_work_by_doi(self, doi: str) -> dict | None:
        """Get a single work by DOI."""
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM works WHERE doi = %s", (doi,))
            return cur.fetchone()

    def resolve_pmid_to_work_id(self, pmid: int) -> str | None:
        """Resolve PMID → work_id via id_crosswalk."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT work_id FROM id_crosswalk WHERE pmid = %s", (pmid,))
            row = cur.fetchone()
            return row[0] if row else None
