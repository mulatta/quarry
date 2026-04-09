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
        """Search MeSH descriptors by synonym via mesh_lookup.

        Splits input into tokens and matches against the `term` column,
        which contains both official descriptor names and NLM entry terms
        (synonyms, abbreviations). Single query covers all sources:
        - Current vocabulary entry terms (from desc*.xml ConceptList)
        - Historical descriptors (from work_mesh, no longer in MeSH tree)

        Example: "post-translational modification" matches entry term for
        D011499 even though the official name is "Protein Processing,
        Post-Translational" and it has no tree_number in current MeSH.
        """
        tokens = name.split()
        if not tokens:
            return []
        conditions = " AND ".join("term ILIKE %s" for _ in tokens)
        params: list = [f"%{t}%" for t in tokens]
        params.append(limit)
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT DISTINCT descriptor_ui, descriptor_name "  # noqa: S608
                f"FROM mesh_lookup WHERE {conditions} LIMIT %s",
                params,
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
        """Get a single work by OpenAlex work_id, enriched with pmc_id."""
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT w.*, c.pmc_id "
                "FROM works w LEFT JOIN id_crosswalk c ON w.work_id = c.work_id "
                "WHERE w.work_id = %s",
                (work_id,),
            )
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
        """Get a single work by PMID (via works table), enriched with pmc_id."""
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT w.*, c.pmc_id "
                "FROM works w LEFT JOIN id_crosswalk c ON w.work_id = c.work_id "
                "WHERE w.pmid = %s",
                (pmid,),
            )
            return cur.fetchone()

    def get_work_by_doi(self, doi: str) -> dict | None:
        """Get a single work by DOI, enriched with pmc_id."""
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT w.*, c.pmc_id "
                "FROM works w LEFT JOIN id_crosswalk c ON w.work_id = c.work_id "
                "WHERE w.doi = %s",
                (doi,),
            )
            return cur.fetchone()

    def get_work_mesh(self, work_id: str) -> list[dict]:
        """Get MeSH descriptors for a work, major topics first."""
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT descriptor_ui, descriptor_name, qualifier_name, is_major_topic "
                "FROM work_mesh WHERE work_id = %s "
                "ORDER BY is_major_topic DESC, descriptor_name",
                (work_id,),
            )
            return cur.fetchall()

    def get_top_works_by_mesh(
        self, descriptor_ui: str, limit: int = 15, major_only: bool = True
    ) -> tuple[list[dict], int]:
        """Top cited works for a MeSH descriptor.

        Returns (works, total_count). Guards against high-cardinality
        descriptors by counting first.
        """
        with self.conn.cursor(row_factory=dict_row) as cur:
            major_filter = " AND wm.is_major_topic = true" if major_only else ""
            cur.execute(
                f"SELECT COUNT(DISTINCT wm.work_id) FROM work_mesh wm "  # noqa: S608
                f"WHERE wm.descriptor_ui = %s{major_filter}",
                (descriptor_ui,),
            )
            row = cur.fetchone()
            total = row["count"] if row else 0

            cur.execute(
                f"SELECT w.work_id, w.pub_year, w.cited_by_count, w.title "  # noqa: S608
                f"FROM work_mesh wm JOIN works w ON wm.work_id = w.work_id "
                f"WHERE wm.descriptor_ui = %s{major_filter} "
                f"ORDER BY w.cited_by_count DESC NULLS LAST LIMIT %s",
                (descriptor_ui, limit),
            )
            return cur.fetchall(), total

    def mesh_parent(self, tree_number: str) -> dict | None:
        """Get parent descriptor for a tree number."""
        parts = tree_number.rsplit(".", 1)
        if len(parts) < 2:
            return None
        parent_tn = parts[0]
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT descriptor_ui, descriptor_name, tree_number "
                "FROM mesh_tree WHERE tree_number = %s",
                (parent_tn,),
            )
            return cur.fetchone()

    def top_mesh_by_work_ids(
        self, work_ids: list[str], limit: int = 10, major_only: bool = True
    ) -> list[dict]:
        """Top MeSH descriptors across a set of work_ids."""
        if not work_ids:
            return []
        major_filter = " AND is_major_topic = true" if major_only else ""
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT descriptor_ui, descriptor_name, "  # noqa: S608
                f"COUNT(DISTINCT work_id) AS cnt "
                f"FROM work_mesh "
                f"WHERE work_id = ANY(%s){major_filter} "
                f"GROUP BY descriptor_ui, descriptor_name "
                f"ORDER BY cnt DESC LIMIT %s",
                (work_ids, limit),
            )
            return cur.fetchall()

    def resolve_pmid_to_work_id(self, pmid: int) -> str | None:
        """Resolve PMID → work_id via id_crosswalk."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT work_id FROM id_crosswalk WHERE pmid = %s", (pmid,))
            row = cur.fetchone()
            return row[0] if row else None
