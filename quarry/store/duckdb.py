"""DuckDB store: schema creation, batch insert, and query interface.

All batch inserts use pyarrow table registration for columnar bulk loading.
This avoids executemany's per-row Python↔C overhead which hangs on large batches.
"""

import re
from pathlib import Path

import duckdb
import pyarrow as pa

from quarry.config import settings

# Only allow SELECT / WITH ... SELECT / EXPLAIN
_READ_ONLY_RE = re.compile(r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)

DDL = """
CREATE TABLE IF NOT EXISTS papers (
    pmid         INTEGER PRIMARY KEY,
    doi          VARCHAR,
    pmc_id       VARCHAR,
    title        VARCHAR,
    abstract     VARCHAR,
    pub_year     SMALLINT,
    pub_date     DATE,
    journal_title VARCHAR,
    journal_issn VARCHAR,
    journal_abbr VARCHAR,
    volume       VARCHAR,
    issue        VARCHAR,
    pages        VARCHAR,
    language     VARCHAR,
    pub_type     VARCHAR[],
    country      VARCHAR,
    medline_status VARCHAR,
    created_date DATE,
    revised_date DATE,
    indexed_date DATE,
    is_deleted   BOOLEAN DEFAULT FALSE,
    deleted_date DATE,
    rcr          FLOAT,
    nih_percentile FLOAT,
    apt          FLOAT,
    is_clinical  BOOLEAN,
    human        FLOAT,
    animal       FLOAT,
    molecular_cellular FLOAT,
    field_citation_rate FLOAT
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
    last_name         VARCHAR,
    fore_name         VARCHAR,
    initials          VARCHAR,
    orcid             VARCHAR,
    affiliation       VARCHAR,
    is_collective     BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_authors_pmid ON authors(pmid);

CREATE TABLE IF NOT EXISTS mesh_headings (
    pmid             INTEGER NOT NULL,
    descriptor_ui    VARCHAR NOT NULL,
    descriptor_name  VARCHAR NOT NULL,
    qualifier_ui     VARCHAR,
    qualifier_name   VARCHAR,
    is_major_topic   BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_mesh_pmid ON mesh_headings(pmid);
CREATE INDEX IF NOT EXISTS idx_mesh_descriptor ON mesh_headings(descriptor_ui);

CREATE TABLE IF NOT EXISTS mesh_tree (
    descriptor_ui    VARCHAR NOT NULL,
    descriptor_name  VARCHAR NOT NULL,
    tree_number      VARCHAR NOT NULL,
    PRIMARY KEY (descriptor_ui, tree_number)
);

CREATE TABLE IF NOT EXISTS grants (
    pmid       INTEGER NOT NULL,
    grant_id   VARCHAR,
    acronym    VARCHAR,
    agency     VARCHAR,
    country    VARCHAR
);
CREATE INDEX IF NOT EXISTS idx_grants_pmid ON grants(pmid);

CREATE TABLE IF NOT EXISTS chemicals (
    pmid              INTEGER NOT NULL,
    registry_number   VARCHAR,
    substance_ui      VARCHAR,
    substance_name    VARCHAR
);
CREATE INDEX IF NOT EXISTS idx_chemicals_pmid ON chemicals(pmid);

CREATE TABLE IF NOT EXISTS preprints (
    doi          VARCHAR PRIMARY KEY,
    title        VARCHAR,
    abstract     VARCHAR,
    date         DATE,
    server       VARCHAR,
    category     VARCHAR,
    version      SMALLINT,
    published_doi VARCHAR
);
"""

DDL_V2 = """
-- ── v2: OpenAlex-primary tables ──

CREATE TABLE IF NOT EXISTS works (
    work_id        VARCHAR PRIMARY KEY,
    work_id_int    BIGINT NOT NULL,
    tier           VARCHAR NOT NULL,
    pmid           INTEGER,
    doi            VARCHAR,
    title          VARCHAR NOT NULL,
    abstract       VARCHAR,
    pub_year       SMALLINT,
    pub_date       DATE,
    type           VARCHAR,
    cited_by_count INTEGER,
    host_venue     VARCHAR,
    oa_status      VARCHAR,
    oa_url         VARCHAR,
    rcr            FLOAT,
    nih_percentile FLOAT,
    apt            FLOAT,
    is_clinical    BOOLEAN,
    is_retracted   BOOLEAN DEFAULT FALSE,
    updated_date   DATE,
    pm_journal_abbr    VARCHAR,
    pm_country         VARCHAR,
    pm_medline_status  VARCHAR,
    pm_pub_type        VARCHAR[],
    pm_created_date    DATE,
    pm_revised_date    DATE,
    pm_indexed_date    DATE
);

CREATE TABLE IF NOT EXISTS work_authors (
    work_id          VARCHAR NOT NULL,
    author_position  SMALLINT,
    display_name     VARCHAR,
    orcid            VARCHAR,
    institution_name VARCHAR,
    institution_ror  VARCHAR,
    raw_affiliation  VARCHAR
);

CREATE TABLE IF NOT EXISTS work_topics (
    work_id    VARCHAR NOT NULL,
    topic_id   VARCHAR NOT NULL,
    topic_name VARCHAR,
    subfield   VARCHAR,
    field      VARCHAR,
    domain     VARCHAR,
    score      FLOAT
);

CREATE TABLE IF NOT EXISTS work_mesh (
    work_id         VARCHAR NOT NULL,
    descriptor_ui   VARCHAR NOT NULL,
    descriptor_name VARCHAR NOT NULL,
    qualifier_ui    VARCHAR,
    qualifier_name  VARCHAR,
    is_major_topic  BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS work_citations (
    citing_id BIGINT NOT NULL,
    cited_id  BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS id_crosswalk (
    work_id VARCHAR PRIMARY KEY,
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


class DuckDBStore:
    """DuckDB embedded store for PubMed metadata."""

    def __init__(self, db_path: Path | None = None):
        self._path = str(db_path or settings.duckdb_path)
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(self._path)
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def init_schema(self):
        """Create all tables if they don't exist (v1 + v2)."""
        # Run DDL first (idempotent CREATE IF NOT EXISTS)
        for ddl in (DDL, DDL_V2, DDL_V2_INDEXES):
            for stmt in ddl.split(";"):
                stmt = stmt.strip()
                if stmt:
                    self.conn.execute(stmt)

        # Migrations run outside transactions — ALTER TABLE in DuckDB
        # auto-commits and aborts outer transactions on failure.
        self._migrate_work_citations()
        self._migrate_works_pm_fields()
        self._migrate_nullable_title()
        self._migrate_cited_by_clin()

    def _migrate_works_pm_fields(self):
        """Add pm_ columns to works table if missing (backwards compat)."""
        try:
            cols = self.conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'works'"
            ).fetchall()
        except Exception:
            return
        existing = {r[0] for r in cols}
        pm_cols = [
            ("pm_journal_abbr", "VARCHAR"),
            ("pm_country", "VARCHAR"),
            ("pm_medline_status", "VARCHAR"),
            ("pm_pub_type", "VARCHAR[]"),
            ("pm_created_date", "DATE"),
            ("pm_revised_date", "DATE"),
            ("pm_indexed_date", "DATE"),
        ]
        for col_name, col_type in pm_cols:
            if col_name not in existing:
                self.conn.execute(f"ALTER TABLE works ADD COLUMN {col_name} {col_type}")

    def _migrate_nullable_title(self):
        """Drop NOT NULL on papers.title and works.title if present."""
        for table in ("papers", "works"):
            try:
                info = self.conn.execute(
                    f"SELECT column_name, is_nullable FROM information_schema.columns "
                    f"WHERE table_name = '{table}' AND column_name = 'title'"
                ).fetchone()
                if info and info[1] == "NO":
                    # DuckDB: ALTER COLUMN ... DROP NOT NULL
                    self.conn.execute(
                        f"ALTER TABLE {table} ALTER COLUMN title DROP NOT NULL"
                    )
            except Exception:
                pass

    def _migrate_cited_by_clin(self):
        """Drop cited_by_clin column from papers if it exists (moved to own table)."""
        try:
            cols = self.conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'papers' AND column_name = 'cited_by_clin'"
            ).fetchall()
        except Exception:
            return
        if cols:
            self.conn.execute("ALTER TABLE papers DROP COLUMN cited_by_clin")

    def _migrate_work_citations(self):
        """Drop work_citations if it has the old VARCHAR schema (citing_work_id)."""
        try:
            cols = self.conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'work_citations'"
            ).fetchall()
        except Exception:
            return
        col_names = {r[0] for r in cols}
        if "citing_work_id" in col_names:
            self.conn.execute("DROP TABLE work_citations")
            self.conn.execute("DROP INDEX IF EXISTS idx_work_cit_citing")
            self.conn.execute("DROP INDEX IF EXISTS idx_work_cit_cited")

    # -- Bulk insert helpers --

    def _bulk_insert(self, table: str, columns: list[str], arrow_table: pa.Table):
        """Register a pyarrow table and INSERT INTO target from it.

        Uses DuckDB's columnar scan of Arrow data — orders of magnitude
        faster than executemany's per-row Python↔C serialization.
        """
        tmp = f"_tmp_{table}"
        self.conn.register(tmp, arrow_table)
        cols = ", ".join(columns)
        self.conn.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM {tmp}")
        self.conn.unregister(tmp)

    def _delete_by_pmids(self, table: str, pmids: list[int]):
        """Delete rows from table where pmid is in the given list."""
        pmid_arr = pa.table({"pmid": pa.array(pmids, type=pa.int32())})
        self.conn.register("_tmp_del", pmid_arr)
        self.conn.execute(
            f"DELETE FROM {table} WHERE pmid IN (SELECT pmid FROM _tmp_del)"
        )
        self.conn.unregister("_tmp_del")

    def _delete_by_registered_pmids(self, table: str):
        """Delete rows using pre-registered _tmp_batch_pmids table."""
        self.conn.execute(
            f"DELETE FROM {table} WHERE pmid IN (SELECT pmid FROM _tmp_batch_pmids)"
        )

    def register_pmid_set(self, pmids: list[int]):
        """Register a PMID set for reuse across multiple child-table deletes."""
        pmid_arr = pa.table({"pmid": pa.array(pmids, type=pa.int32())})
        self.conn.register("_tmp_batch_pmids", pmid_arr)

    def unregister_pmid_set(self):
        """Unregister the shared PMID set."""
        self.conn.unregister("_tmp_batch_pmids")

    # -- Batch inserts --

    def upsert_papers(self, rows: list[dict]):
        """Insert or update papers batch via pyarrow bulk loading.

        Deduplicates within batch (last wins). Uses DELETE + INSERT
        for reliability with DuckDB's PK constraints.
        """
        if not rows:
            return
        # Deduplicate: keep last occurrence of each PMID
        seen: dict[int, dict] = {}
        for r in rows:
            seen[r["pmid"]] = r
        deduped = list(seen.values())

        columns = [
            "pmid",
            "doi",
            "pmc_id",
            "title",
            "abstract",
            "pub_year",
            "pub_date",
            "journal_title",
            "journal_issn",
            "journal_abbr",
            "volume",
            "issue",
            "pages",
            "language",
            "pub_type",
            "country",
            "medline_status",
            "created_date",
            "revised_date",
            "indexed_date",
        ]

        # Build pyarrow arrays with explicit types
        arrow_table = pa.table(
            {
                "pmid": pa.array([r["pmid"] for r in deduped], type=pa.int32()),
                "doi": pa.array([r.get("doi") for r in deduped], type=pa.string()),
                "pmc_id": pa.array(
                    [r.get("pmc_id") for r in deduped], type=pa.string()
                ),
                "title": pa.array([r["title"] for r in deduped], type=pa.string()),
                "abstract": pa.array(
                    [r.get("abstract") for r in deduped], type=pa.string()
                ),
                "pub_year": pa.array(
                    [r.get("pub_year") for r in deduped], type=pa.int16()
                ),
                "pub_date": pa.array(
                    [r.get("pub_date") for r in deduped], type=pa.date32()
                ),
                "journal_title": pa.array(
                    [r.get("journal_title") for r in deduped], type=pa.string()
                ),
                "journal_issn": pa.array(
                    [r.get("journal_issn") for r in deduped], type=pa.string()
                ),
                "journal_abbr": pa.array(
                    [r.get("journal_abbr") for r in deduped], type=pa.string()
                ),
                "volume": pa.array(
                    [r.get("volume") for r in deduped], type=pa.string()
                ),
                "issue": pa.array([r.get("issue") for r in deduped], type=pa.string()),
                "pages": pa.array([r.get("pages") for r in deduped], type=pa.string()),
                "language": pa.array(
                    [r.get("language") for r in deduped], type=pa.string()
                ),
                "pub_type": pa.array(
                    [r.get("pub_type") for r in deduped], type=pa.list_(pa.string())
                ),
                "country": pa.array(
                    [r.get("country") for r in deduped], type=pa.string()
                ),
                "medline_status": pa.array(
                    [r.get("medline_status") for r in deduped], type=pa.string()
                ),
                "created_date": pa.array(
                    [r.get("created_date") for r in deduped], type=pa.date32()
                ),
                "revised_date": pa.array(
                    [r.get("revised_date") for r in deduped], type=pa.date32()
                ),
                "indexed_date": pa.array(
                    [r.get("indexed_date") for r in deduped], type=pa.date32()
                ),
            }
        )

        # Delete existing PMIDs first (for upsert semantics)
        pmids = [r["pmid"] for r in deduped]
        self._delete_by_pmids("papers", pmids)
        self._bulk_insert("papers", columns, arrow_table)

    def insert_authors(self, rows: list[dict], pmids_registered: bool = False):
        """Batch insert authors via pyarrow (delete-then-insert by pmid set)."""
        if not rows:
            return
        if pmids_registered:
            self._delete_by_registered_pmids("authors")
        else:
            pmids = list({r["pmid"] for r in rows})
            self._delete_by_pmids("authors", pmids)

        columns = [
            "pmid",
            "author_position",
            "last_name",
            "fore_name",
            "initials",
            "orcid",
            "affiliation",
            "is_collective",
        ]
        arrow_table = pa.table(
            {
                "pmid": pa.array([r["pmid"] for r in rows], type=pa.int32()),
                "author_position": pa.array(
                    [r.get("author_position") for r in rows], type=pa.int16()
                ),
                "last_name": pa.array(
                    [r.get("last_name") for r in rows], type=pa.string()
                ),
                "fore_name": pa.array(
                    [r.get("fore_name") for r in rows], type=pa.string()
                ),
                "initials": pa.array(
                    [r.get("initials") for r in rows], type=pa.string()
                ),
                "orcid": pa.array([r.get("orcid") for r in rows], type=pa.string()),
                "affiliation": pa.array(
                    [r.get("affiliation") for r in rows], type=pa.string()
                ),
                "is_collective": pa.array(
                    [r.get("is_collective", False) for r in rows], type=pa.bool_()
                ),
            }
        )
        self._bulk_insert("authors", columns, arrow_table)

    def insert_mesh_headings(self, rows: list[dict], pmids_registered: bool = False):
        """Batch insert mesh_headings via pyarrow (delete-then-insert by pmid set)."""
        if not rows:
            return
        if pmids_registered:
            self._delete_by_registered_pmids("mesh_headings")
        else:
            pmids = list({r["pmid"] for r in rows})
            self._delete_by_pmids("mesh_headings", pmids)

        columns = [
            "pmid",
            "descriptor_ui",
            "descriptor_name",
            "qualifier_ui",
            "qualifier_name",
            "is_major_topic",
        ]
        arrow_table = pa.table(
            {
                "pmid": pa.array([r["pmid"] for r in rows], type=pa.int32()),
                "descriptor_ui": pa.array(
                    [r["descriptor_ui"] for r in rows], type=pa.string()
                ),
                "descriptor_name": pa.array(
                    [r["descriptor_name"] for r in rows], type=pa.string()
                ),
                "qualifier_ui": pa.array(
                    [r.get("qualifier_ui") for r in rows], type=pa.string()
                ),
                "qualifier_name": pa.array(
                    [r.get("qualifier_name") for r in rows], type=pa.string()
                ),
                "is_major_topic": pa.array(
                    [r.get("is_major_topic", False) for r in rows], type=pa.bool_()
                ),
            }
        )
        self._bulk_insert("mesh_headings", columns, arrow_table)

    def insert_grants(self, rows: list[dict], pmids_registered: bool = False):
        """Batch insert grants via pyarrow (delete-then-insert by pmid set)."""
        if not rows:
            return
        if pmids_registered:
            self._delete_by_registered_pmids("grants")
        else:
            pmids = list({r["pmid"] for r in rows})
            self._delete_by_pmids("grants", pmids)

        columns = ["pmid", "grant_id", "acronym", "agency", "country"]
        arrow_table = pa.table(
            {
                "pmid": pa.array([r["pmid"] for r in rows], type=pa.int32()),
                "grant_id": pa.array(
                    [r.get("grant_id") for r in rows], type=pa.string()
                ),
                "acronym": pa.array([r.get("acronym") for r in rows], type=pa.string()),
                "agency": pa.array([r.get("agency") for r in rows], type=pa.string()),
                "country": pa.array([r.get("country") for r in rows], type=pa.string()),
            }
        )
        self._bulk_insert("grants", columns, arrow_table)

    def insert_chemicals(self, rows: list[dict], pmids_registered: bool = False):
        """Batch insert chemicals via pyarrow (delete-then-insert by pmid set)."""
        if not rows:
            return
        if pmids_registered:
            self._delete_by_registered_pmids("chemicals")
        else:
            pmids = list({r["pmid"] for r in rows})
            self._delete_by_pmids("chemicals", pmids)

        columns = ["pmid", "registry_number", "substance_ui", "substance_name"]
        arrow_table = pa.table(
            {
                "pmid": pa.array([r["pmid"] for r in rows], type=pa.int32()),
                "registry_number": pa.array(
                    [r.get("registry_number") for r in rows], type=pa.string()
                ),
                "substance_ui": pa.array(
                    [r.get("substance_ui") for r in rows], type=pa.string()
                ),
                "substance_name": pa.array(
                    [r.get("substance_name") for r in rows], type=pa.string()
                ),
            }
        )
        self._bulk_insert("chemicals", columns, arrow_table)

    def soft_delete(self, pmids: list[int]):
        """Soft-delete papers by PMID list."""
        if not pmids:
            return
        pmid_arr = pa.table({"pmid": pa.array(pmids, type=pa.int32())})
        self.conn.register("_tmp_softdel", pmid_arr)
        self.conn.execute(
            "UPDATE papers SET is_deleted = TRUE, deleted_date = CURRENT_DATE "
            "WHERE pmid IN (SELECT pmid FROM _tmp_softdel)"
        )
        self.conn.unregister("_tmp_softdel")

    # -- Read queries --

    def get_paper(self, pmid: int) -> dict | None:
        """Get a single paper by PMID."""
        result = self.conn.execute(
            "SELECT * FROM papers WHERE pmid = ? AND NOT is_deleted", [pmid]
        ).fetchone()
        if not result:
            return None
        cols = [d[0] for d in self.conn.description]
        return dict(zip(cols, result))

    def get_paper_by_doi(self, doi: str) -> dict | None:
        """Get a single paper by DOI (parameterized, injection-safe)."""
        result = self.conn.execute(
            "SELECT * FROM papers WHERE doi = ? AND NOT is_deleted", [doi]
        ).fetchone()
        if not result:
            return None
        cols = [d[0] for d in self.conn.description]
        return dict(zip(cols, result))

    def get_papers(self, pmids: list[int]) -> list[dict]:
        """Get multiple papers by PMID list."""
        if not pmids:
            return []
        result = self.conn.execute(
            "SELECT * FROM papers WHERE pmid IN (SELECT unnest(?)) AND NOT is_deleted",
            [pmids],
        )
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def query(self, sql: str) -> list[dict]:
        """Execute a read-only SQL query. Only SELECT/WITH/EXPLAIN allowed."""
        if not _READ_ONLY_RE.match(sql):
            raise ValueError("Only SELECT/WITH/EXPLAIN queries are allowed")
        result = self.conn.execute(sql)
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def mesh_descendants(self, tree_prefix: str) -> list[dict]:
        """Get all MeSH descriptors under a tree number prefix."""
        result = self.conn.execute(
            "SELECT DISTINCT descriptor_ui, descriptor_name, tree_number "
            "FROM mesh_tree WHERE tree_number LIKE ? "
            "ORDER BY tree_number",
            [tree_prefix + "%"],
        )
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def mesh_search_by_name(self, name: str, limit: int = 10) -> list[dict]:
        """Search MeSH descriptors by name (ILIKE, parameterized)."""
        result = self.conn.execute(
            "SELECT DISTINCT descriptor_ui, descriptor_name "
            "FROM mesh_tree WHERE descriptor_name ILIKE ? LIMIT ?",
            [f"%{name}%", limit],
        )
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def mesh_by_ui(self, descriptor_ui: str) -> list[dict]:
        """Get all tree entries for a MeSH descriptor UI (parameterized)."""
        result = self.conn.execute(
            "SELECT * FROM mesh_tree WHERE descriptor_ui = ?",
            [descriptor_ui],
        )
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def mesh_expand_pmids(self, descriptor_uis: list[str]) -> list[int]:
        """Get PMIDs that have any of the given MeSH descriptor UIs."""
        if not descriptor_uis:
            return []
        result = self.conn.execute(
            "SELECT DISTINCT pmid FROM mesh_headings "
            "WHERE descriptor_ui IN (SELECT unnest(?)) "
            "AND pmid IN (SELECT pmid FROM papers WHERE NOT is_deleted)",
            [descriptor_uis],
        )
        return [row[0] for row in result.fetchall()]

    def top_mesh(self, pmids: list[int], limit: int = 10) -> list[dict]:
        """Top MeSH descriptors for a set of PMIDs."""
        if not pmids:
            return []
        result = self.conn.execute(
            "SELECT descriptor_ui, descriptor_name, count(*) AS cnt "
            "FROM mesh_headings WHERE pmid IN (SELECT unnest(?)) "
            "GROUP BY descriptor_ui, descriptor_name ORDER BY cnt DESC LIMIT ?",
            [pmids, limit],
        )
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def year_distribution(self, pmids: list[int]) -> list[dict]:
        """Publication year distribution for a set of PMIDs."""
        if not pmids:
            return []
        result = self.conn.execute(
            "SELECT pub_year, count(*) AS cnt FROM papers "
            "WHERE pmid IN (SELECT unnest(?)) AND pub_year > 0 AND NOT is_deleted "
            "GROUP BY pub_year ORDER BY pub_year",
            [pmids],
        )
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def top_authors(self, pmids: list[int], limit: int = 10) -> list[dict]:
        """Top authors for a set of PMIDs."""
        if not pmids:
            return []
        result = self.conn.execute(
            "SELECT last_name, fore_name, orcid, count(*) AS cnt "
            "FROM authors WHERE pmid IN (SELECT unnest(?)) "
            "GROUP BY last_name, fore_name, orcid ORDER BY cnt DESC LIMIT ?",
            [pmids, limit],
        )
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    # -- v2 (OpenAlex) read queries --

    def get_work(self, work_id: str) -> dict | None:
        """Get a single work by OpenAlex work_id."""
        result = self.conn.execute(
            "SELECT * FROM works WHERE work_id = ?", [work_id]
        ).fetchone()
        if not result:
            return None
        cols = [d[0] for d in self.conn.description]
        return dict(zip(cols, result))

    def get_works(self, work_ids: list[str]) -> list[dict]:
        """Get multiple works by work_id list."""
        if not work_ids:
            return []
        result = self.conn.execute(
            "SELECT * FROM works WHERE work_id IN (SELECT unnest(?))",
            [work_ids],
        )
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def get_work_by_pmid(self, pmid: int) -> dict | None:
        """Get a single work by PMID (via works table)."""
        result = self.conn.execute(
            "SELECT * FROM works WHERE pmid = ?", [pmid]
        ).fetchone()
        if not result:
            return None
        cols = [d[0] for d in self.conn.description]
        return dict(zip(cols, result))

    def resolve_pmid_to_work_id(self, pmid: int) -> str | None:
        """Resolve PMID → work_id via id_crosswalk."""
        row = self.conn.execute(
            "SELECT work_id FROM id_crosswalk WHERE pmid = ?", [pmid]
        ).fetchone()
        return row[0] if row else None

    def resolve_work_id_to_pmid(self, work_id: str) -> int | None:
        """Resolve work_id → PMID via id_crosswalk."""
        row = self.conn.execute(
            "SELECT pmid FROM id_crosswalk WHERE work_id = ?", [work_id]
        ).fetchone()
        return row[0] if row else None
