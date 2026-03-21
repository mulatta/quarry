"""PostgreSQL store: schema creation and query interface.

Bulk loading is handled by Dagster assets (load.py) via COPY FROM.
This module provides schema DDL and read-only query methods.
"""

import re

import psycopg
from psycopg.rows import dict_row

# Only allow SELECT / WITH ... SELECT / EXPLAIN
_READ_ONLY_RE = re.compile(r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)

DDL = """
CREATE TABLE IF NOT EXISTS papers (
    pmid         INTEGER PRIMARY KEY,
    doi          TEXT,
    pmc_id       TEXT,
    title        TEXT,
    abstract     TEXT,
    pub_year     SMALLINT,
    pub_date     DATE,
    journal_title TEXT,
    journal_issn TEXT,
    journal_abbr TEXT,
    volume       TEXT,
    issue        TEXT,
    pages        TEXT,
    language     TEXT,
    pub_type     TEXT[],
    country      TEXT,
    medline_status TEXT,
    created_date DATE,
    revised_date DATE,
    indexed_date DATE,
    is_deleted   BOOLEAN DEFAULT FALSE,
    deleted_date DATE,
    rcr          REAL,
    nih_percentile REAL,
    apt          REAL,
    is_clinical  BOOLEAN,
    human        REAL,
    animal       REAL,
    molecular_cellular REAL,
    field_citation_rate REAL
);

CREATE TABLE IF NOT EXISTS cited_by_clin (
    pmid        INTEGER NOT NULL,
    citing_pmid INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cbc_pmid ON cited_by_clin(pmid);
CREATE INDEX IF NOT EXISTS idx_cbc_citing ON cited_by_clin(citing_pmid);

CREATE TABLE IF NOT EXISTS authors (
    pmid              INTEGER NOT NULL,
    author_position   SMALLINT,
    last_name         TEXT,
    fore_name         TEXT,
    initials          TEXT,
    orcid             TEXT,
    affiliation       TEXT,
    is_collective     BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_authors_pmid ON authors(pmid);

CREATE TABLE IF NOT EXISTS mesh_headings (
    pmid             INTEGER NOT NULL,
    descriptor_ui    TEXT NOT NULL,
    descriptor_name  TEXT NOT NULL,
    qualifier_ui     TEXT,
    qualifier_name   TEXT,
    is_major_topic   BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_mesh_pmid ON mesh_headings(pmid);
CREATE INDEX IF NOT EXISTS idx_mesh_descriptor ON mesh_headings(descriptor_ui);

CREATE TABLE IF NOT EXISTS mesh_tree (
    descriptor_ui    TEXT NOT NULL,
    descriptor_name  TEXT NOT NULL,
    tree_number      TEXT NOT NULL,
    PRIMARY KEY (descriptor_ui, tree_number)
);

CREATE TABLE IF NOT EXISTS grants (
    pmid       INTEGER NOT NULL,
    grant_id   TEXT,
    acronym    TEXT,
    agency     TEXT,
    country    TEXT
);
CREATE INDEX IF NOT EXISTS idx_grants_pmid ON grants(pmid);

CREATE TABLE IF NOT EXISTS chemicals (
    pmid              INTEGER NOT NULL,
    registry_number   TEXT,
    substance_ui      TEXT,
    substance_name    TEXT
);
CREATE INDEX IF NOT EXISTS idx_chemicals_pmid ON chemicals(pmid);

"""

DDL_V2 = """
-- v2: OpenAlex-primary tables

CREATE TABLE IF NOT EXISTS works (
    work_id        TEXT PRIMARY KEY,
    work_id_int    BIGINT NOT NULL,
    tier           TEXT NOT NULL,
    pmid           INTEGER,
    doi            TEXT,
    title          TEXT,
    abstract       TEXT,
    pub_year       SMALLINT,
    pub_date       DATE,
    type           TEXT,
    cited_by_count INTEGER,
    host_venue     TEXT,
    oa_status      TEXT,
    oa_url         TEXT,
    rcr            REAL,
    nih_percentile REAL,
    apt            REAL,
    is_clinical    BOOLEAN,
    is_retracted   BOOLEAN DEFAULT FALSE,
    updated_date   DATE,
    pm_journal_abbr    TEXT,
    pm_country         TEXT,
    pm_medline_status  TEXT,
    pm_pub_type        TEXT[],
    pm_created_date    DATE,
    pm_revised_date    DATE,
    pm_indexed_date    DATE
);

CREATE TABLE IF NOT EXISTS work_authors (
    work_id          TEXT NOT NULL,
    author_position  SMALLINT,
    display_name     TEXT,
    orcid            TEXT,
    institution_name TEXT,
    institution_ror  TEXT,
    raw_affiliation  TEXT
);

CREATE TABLE IF NOT EXISTS work_topics (
    work_id    TEXT NOT NULL,
    topic_id   TEXT NOT NULL,
    topic_name TEXT,
    subfield   TEXT,
    field      TEXT,
    domain     TEXT,
    score      REAL
);

CREATE TABLE IF NOT EXISTS work_mesh (
    work_id         TEXT NOT NULL,
    descriptor_ui   TEXT NOT NULL,
    descriptor_name TEXT NOT NULL,
    qualifier_ui    TEXT,
    qualifier_name  TEXT,
    is_major_topic  BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS work_citations (
    citing_id BIGINT NOT NULL,
    cited_id  BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS id_crosswalk (
    work_id TEXT PRIMARY KEY,
    pmid    INTEGER NOT NULL
);
"""

DDL_V2_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_works_pmid ON works(pmid);
CREATE INDEX IF NOT EXISTS idx_works_tier ON works(tier);
CREATE INDEX IF NOT EXISTS idx_works_pub_year ON works(pub_year);
CREATE INDEX IF NOT EXISTS idx_work_authors_wid ON work_authors(work_id);
CREATE INDEX IF NOT EXISTS idx_work_topics_wid ON work_topics(work_id);
CREATE INDEX IF NOT EXISTS idx_work_mesh_wid ON work_mesh(work_id);
CREATE INDEX IF NOT EXISTS idx_work_mesh_desc ON work_mesh(descriptor_ui);
CREATE INDEX IF NOT EXISTS idx_work_cit_citing ON work_citations(citing_id);
CREATE INDEX IF NOT EXISTS idx_work_cit_cited ON work_citations(cited_id);
CREATE INDEX IF NOT EXISTS idx_crosswalk_pmid ON id_crosswalk(pmid);
"""

DDL_BUILD_PROGRESS = """
CREATE TABLE IF NOT EXISTS _build_progress (
    source TEXT NOT NULL,
    filename TEXT NOT NULL,
    rows_inserted BIGINT,
    completed_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (source, filename)
);
"""


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
        """Create all tables if they don't exist (v1 + v2 + progress)."""
        for ddl in (DDL, DDL_V2, DDL_V2_INDEXES, DDL_BUILD_PROGRESS):
            for stmt in ddl.split(";"):
                stmt = stmt.strip()
                if stmt and not stmt.startswith("--"):
                    self.conn.execute(stmt)

    # -- Read queries --

    def query(self, sql: str) -> list[dict]:
        """Execute a read-only SQL query. Only SELECT/WITH/EXPLAIN allowed."""
        if not _READ_ONLY_RE.match(sql):
            raise ValueError("Only SELECT/WITH/EXPLAIN queries are allowed")
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql)
            return cur.fetchall()

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
