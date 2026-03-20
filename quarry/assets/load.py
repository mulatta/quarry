"""DuckDB bulk load asset: single sequential DB writer.

This is the ONLY asset that writes to DuckDB, eliminating lock contention.

Performance strategy:
- Baseline: COPY FROM Parquet (built by quarry-build CLI)
- Updates: parse XML directly → INSERT OR REPLACE (incremental, small volume)
- Indexes dropped before load, recreated after
- iCite: DuckDB read_csv() directly from CSV

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from pathlib import Path

from dagster import (
    AssetExecutionContext,
    AutomationCondition,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.download import (
    icite_metadata_sync,
    pubmed_updates_sync,
)
from quarry.assets.stage import enrich_build, mesh_stage, pubmed_parquet_build
from quarry.config import settings
from quarry.etl.feather import read_tables
from quarry.resources import DuckDBResource

import quarry_parse

# Column order matching Rust Arrow schema (used by updates path)
_PAPERS_COLS = [
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
_DATE_COLS = {"pub_date", "created_date", "revised_date", "indexed_date"}
_CHILD_TABLES = {
    "authors": [
        "pmid",
        "author_position",
        "last_name",
        "fore_name",
        "initials",
        "orcid",
        "affiliation",
        "is_collective",
    ],
    "mesh_headings": [
        "pmid",
        "descriptor_ui",
        "descriptor_name",
        "qualifier_ui",
        "qualifier_name",
        "is_major_topic",
    ],
    "grants": ["pmid", "grant_id", "acronym", "agency", "country"],
    "chemicals": ["pmid", "registry_number", "substance_ui", "substance_name"],
}
_INDEXES = [
    ("idx_authors_pmid", "authors", "pmid"),
    ("idx_mesh_pmid", "mesh_headings", "pmid"),
    ("idx_mesh_descriptor", "mesh_headings", "descriptor_ui"),
    ("idx_grants_pmid", "grants", "pmid"),
    ("idx_chemicals_pmid", "chemicals", "pmid"),
    ("idx_cbc_pmid", "cited_by_clin", "pmid"),
    ("idx_cbc_citing", "cited_by_clin", "citing_pmid"),
]

_COLS_INSERT = ", ".join(_PAPERS_COLS)
_COLS_SELECT = ", ".join(
    f"TRY_CAST({c} AS DATE)"
    if c in _DATE_COLS
    else f"COALESCE({c}, '')"
    if c == "title"
    else c
    for c in _PAPERS_COLS
)


@asset(
    group_name="load",
    deps=[
        pubmed_parquet_build,
        pubmed_updates_sync,
        mesh_stage,
        icite_metadata_sync,
    ],
    description="Bulk load PubMed data into DuckDB: Parquet baseline → updates → MeSH → iCite.",
    kinds={"duckdb"},
    automation_condition=AutomationCondition.eager(),
)
def duckdb_load(
    context: AssetExecutionContext,
    duckdb: DuckDBResource,
) -> MaterializeResult:
    db = duckdb.store
    conn = db.conn
    staging = settings.staging_dir
    build_dir = settings.build_output_dir
    stats: dict[str, int] = {}

    # Tune DuckDB for bulk load
    conn.execute("SET memory_limit='40GB'")
    conn.execute("SET threads=8")
    conn.execute("SET preserve_insertion_order=false")

    # Drop indexes for faster bulk insert
    context.log.info("Dropping indexes for bulk load")
    for idx_name, _, _ in _INDEXES:
        conn.execute(f"DROP INDEX IF EXISTS {idx_name}")

    # 1. PubMed baseline from Parquet (built by quarry-build build-pubmed)
    context.log.info("Loading PubMed baseline from Parquet")
    conn.execute("DELETE FROM papers")
    for table_name in _CHILD_TABLES:
        conn.execute(f"DELETE FROM {table_name}")

    _PARQUET_TABLES = [
        ("papers", "papers"),
        ("authors", "pubmed_authors"),
        ("mesh_headings", "mesh_headings"),
        ("grants", "grants"),
        ("chemicals", "chemicals"),
    ]
    for table, subdir in _PARQUET_TABLES:
        pq_dir = build_dir / subdir
        if not pq_dir.exists():
            context.log.warning(f"Parquet dir missing: {pq_dir}")
            continue
        conn.execute(
            f"INSERT INTO {table} BY NAME "
            f"SELECT * FROM read_parquet('{pq_dir}/*.parquet')"
        )
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        stats[f"baseline_{table}"] = n
        context.log.info(f"  {table}: {n:,} rows")

    # 2. Soft-delete retracted PMIDs from baseline
    del_dir = build_dir / "delete_pmids"
    if del_dir.exists() and any(del_dir.glob("*.parquet")):
        conn.execute(
            "UPDATE papers SET is_deleted = TRUE, deleted_date = CURRENT_DATE "
            f"WHERE pmid IN (SELECT pmid FROM read_parquet('{del_dir}/*.parquet'))"
        )
        n_del = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE is_deleted = TRUE"
        ).fetchone()[0]
        stats["baseline_deleted"] = n_del
        context.log.info(f"  Soft-deleted: {n_del:,} PMIDs")

    # 3. PubMed updates (parse XML directly — incremental, small volume)
    n = _load_updates_direct(conn, context)
    stats["update_papers"] = n
    context.log.info(f"Updates: {n:,} papers loaded")

    # 4. MeSH tree
    tables = read_tables(staging / "mesh")
    if "mesh_tree" in tables:
        t = tables["mesh_tree"]
        conn.execute("DELETE FROM mesh_tree")
        conn.register("_stg_mesh", t)
        conn.execute(
            "INSERT INTO mesh_tree (descriptor_ui, descriptor_name, tree_number) "
            "SELECT descriptor_ui, descriptor_name, tree_number FROM _stg_mesh"
        )
        conn.unregister("_stg_mesh")
        stats["mesh_entries"] = t.num_rows
        context.log.info(f"MeSH: {t.num_rows:,} entries loaded")

    # 5. iCite metrics (DuckDB reads CSV directly)
    meta_csv = settings.icite_dir / "icite_metadata.csv"
    if meta_csv.exists():
        context.log.info(f"iCite: loading from {meta_csv}")
        n = _load_icite_csv(conn, meta_csv, context)
        stats["icite_rows"] = n
        context.log.info(f"iCite: {n:,} rows updated")

    # Recreate indexes
    context.log.info("Recreating indexes")
    for idx_name, table_name, col in _INDEXES:
        conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}({col})")
    context.log.info("Indexes recreated")

    return MaterializeResult(
        metadata={k: MetadataValue.int(v) for k, v in stats.items()}
    )


def _register_and_insert(conn, result):
    """Register all Arrow tables, INSERT in a single transaction, then unregister.

    Used by _load_updates_direct for incremental updates.
    """
    registered = []

    papers = result["papers"]
    if papers is not None and papers.num_rows > 0:
        conn.register("_stg_papers", papers)
        registered.append("_stg_papers")

    for table_name in _CHILD_TABLES:
        t = result.get(table_name)
        if t is not None and t.num_rows > 0:
            tmp = f"_stg_{table_name}"
            conn.register(tmp, t)
            registered.append(tmp)

    conn.execute("BEGIN TRANSACTION")

    n_papers = 0
    if "_stg_papers" in registered:
        conn.execute(
            f"INSERT INTO papers ({_COLS_INSERT}) "
            f"SELECT {_COLS_SELECT} FROM _stg_papers"
        )
        n_papers = papers.num_rows

    for table_name, columns in _CHILD_TABLES.items():
        tmp = f"_stg_{table_name}"
        if tmp in registered:
            cols = ", ".join(columns)
            conn.execute(f"INSERT INTO {table_name} ({cols}) SELECT {cols} FROM {tmp}")

    conn.execute("COMMIT")

    for name in registered:
        conn.unregister(name)

    return n_papers


def _load_updates_direct(conn, context) -> int:
    """Parse update XML directly into DuckDB — pipelined.

    Same pipeline pattern as before but with DELETE+INSERT (incremental).
    Batches multiple update files for better throughput.
    """
    from concurrent.futures import ThreadPoolExecutor

    files = sorted(settings.pubmed_update_dir.glob("pubmed*.xml.gz"))
    if not files:
        return 0

    chunk_size = 10
    chunks = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]
    n_chunks = len(chunks)
    context.log.info(
        f"Updates: parsing {len(files)} XML files → DuckDB ({n_chunks} chunks, pipelined)"
    )

    def _parse(chunk_files):
        return quarry_parse.parse_pubmed_files([str(p) for p in chunk_files])

    total = 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_parse, chunks[0])

        for i in range(n_chunks):
            result = future.result()

            if i + 1 < n_chunks:
                future = pool.submit(_parse, chunks[i + 1])

            registered = []

            conn.execute("BEGIN TRANSACTION")

            # Upsert papers (INSERT OR REPLACE handles both existing rows
            # and duplicate PMIDs within the same batch)
            papers = result["papers"]
            if papers is not None and papers.num_rows > 0:
                conn.register("_stg_papers", papers)
                registered.append("_stg_papers")
                conn.execute(
                    f"INSERT OR REPLACE INTO papers ({_COLS_INSERT}) "
                    f"SELECT {_COLS_SELECT} FROM _stg_papers"
                )
                total += papers.num_rows

            # Child tables: DELETE by pmid set from papers staging, then INSERT
            for table_name, columns in _CHILD_TABLES.items():
                t = result.get(table_name)
                if t is not None and t.num_rows > 0:
                    tmp = f"_stg_{table_name}"
                    conn.register(tmp, t)
                    registered.append(tmp)
                    conn.execute(
                        f"DELETE FROM {table_name} "
                        f"WHERE pmid IN (SELECT DISTINCT pmid FROM {tmp})"
                    )
                    cols = ", ".join(columns)
                    conn.execute(
                        f"INSERT INTO {table_name} ({cols}) SELECT {cols} FROM {tmp}"
                    )

            # Soft-delete retracted PMIDs
            if result["delete_pmids"]:
                import pyarrow as pa

                del_t = pa.table(
                    {"pmid": pa.array(result["delete_pmids"], type=pa.int32())}
                )
                conn.register("_stg_del", del_t)
                registered.append("_stg_del")
                conn.execute(
                    "UPDATE papers SET is_deleted = TRUE, deleted_date = CURRENT_DATE "
                    "WHERE pmid IN (SELECT pmid FROM _stg_del)"
                )

            conn.execute("COMMIT")

            for name in registered:
                conn.unregister(name)

            context.log.info(
                f"  [{i + 1}/{n_chunks}] +{papers.num_rows if papers is not None else 0:,} papers "
                f"(total: {total:,})"
            )
    return total


def _load_icite_csv(conn, csv_path: Path, context) -> int:
    """Load iCite metrics directly from CSV via DuckDB's native CSV reader."""
    result = conn.execute(f"""
        UPDATE papers SET
            rcr = m.relative_citation_ratio,
            nih_percentile = m.nih_percentile,
            apt = m.apt,
            is_clinical = CAST(m.is_clinical AS BOOLEAN),
            human = m.human,
            animal = m.animal,
            molecular_cellular = m.molecular_cellular,
            field_citation_rate = m.field_citation_rate
        FROM read_csv('{csv_path}',
             auto_detect=true, ignore_errors=true, parallel=true) m
        WHERE papers.pmid = m.pmid
    """)
    n = result.fetchone()[0]

    # cited_by_clin: PMID list string → normalized rows
    conn.execute("DELETE FROM cited_by_clin")
    conn.execute(f"""
        INSERT INTO cited_by_clin (pmid, citing_pmid)
        SELECT m.pmid, CAST(unnest(string_split(m.cited_by_clin, ' ')) AS INTEGER)
        FROM read_csv('{csv_path}',
             auto_detect=true, ignore_errors=true, parallel=true) m
        WHERE m.cited_by_clin IS NOT NULL
          AND m.cited_by_clin != ''
    """)
    n_cbc = conn.execute("SELECT COUNT(*) FROM cited_by_clin").fetchone()[0]
    context.log.info(f"  cited_by_clin: {n_cbc:,} rows")

    return n


# -- v2 indexes (OpenAlex) --

_V2_INDEXES = [
    ("idx_works_pmid", "works", "pmid"),
    ("idx_works_tier", "works", "tier"),
    ("idx_works_pub_year", "works", "pub_year"),
    ("idx_work_authors_wid", "work_authors", "work_id"),
    ("idx_work_topics_wid", "work_topics", "work_id"),
    ("idx_work_mesh_wid", "work_mesh", "work_id"),
    ("idx_work_mesh_desc", "work_mesh", "descriptor_ui"),
    ("idx_work_cit_citing", "work_citations", "citing_id"),
    ("idx_work_cit_cited", "work_citations", "cited_id"),
    ("idx_crosswalk_pmid", "id_crosswalk", "pmid"),
]


@asset(
    group_name="openalex",
    description="Load OpenAlex Parquet → DuckDB works + child tables.",
    deps=[duckdb_load, enrich_build],
    kinds={"duckdb"},
)
def oa_snapshot_load(
    context: AssetExecutionContext,
    duckdb: DuckDBResource,
) -> MaterializeResult:
    """COPY FROM Parquet (built by quarry-build) → works + child tables."""
    db = duckdb.store
    conn = db.conn
    stats: dict[str, int] = {}
    build_dir = settings.build_output_dir

    # Tune DuckDB for bulk load
    conn.execute("SET memory_limit='40GB'")
    conn.execute("SET threads=8")
    conn.execute("SET preserve_insertion_order=false")

    # Drop v2 indexes for bulk load
    context.log.info("Dropping v2 indexes for bulk load")
    for idx_name, _, _ in _V2_INDEXES:
        conn.execute(f"DROP INDEX IF EXISTS {idx_name}")

    # Truncate v2 tables for full snapshot reload
    for table in (
        "work_citations",
        "work_topics",
        "work_authors",
        "work_mesh",
        "id_crosswalk",
    ):
        conn.execute(f"DELETE FROM {table}")
    conn.execute("DELETE FROM works")

    # Load from Parquet (enriched_works has pm_ fields from enrich step)
    _OA_PARQUET_TABLES = [
        ("works", "enriched_works"),
        ("work_authors", "work_authors"),
        ("work_topics", "work_topics"),
        ("work_citations", "work_citations"),
        ("id_crosswalk", "id_crosswalk"),
        ("work_mesh", "work_mesh"),
    ]
    for table, subdir in _OA_PARQUET_TABLES:
        pq_dir = build_dir / subdir
        if not pq_dir.exists():
            context.log.warning(f"Parquet dir missing: {pq_dir}")
            continue
        conn.execute(
            f"INSERT INTO {table} BY NAME "
            f"SELECT * FROM read_parquet('{pq_dir}/*.parquet')"
        )
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        stats[table] = n
        context.log.info(f"  {table}: {n:,} rows")

    # iCite RCR enrichment for works table (T1 only)
    meta_csv = settings.icite_dir / "icite_metadata.csv"
    if meta_csv.exists():
        context.log.info("iCite RCR enrichment → works")
        n = _load_icite_works(conn, meta_csv)
        stats["icite_works_updated"] = n
        context.log.info(f"iCite: {n:,} works updated")

    # Recreate v2 indexes
    context.log.info("Recreating v2 indexes")
    for idx_name, table_name, col in _V2_INDEXES:
        conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}({col})")
    context.log.info("v2 indexes recreated")

    # Tier stats
    tier_stats = conn.execute(
        "SELECT tier, COUNT(*) FROM works GROUP BY tier ORDER BY tier"
    ).fetchall()
    for tier, cnt in tier_stats:
        context.log.info(f"  {tier}: {cnt:,}")

    return MaterializeResult(
        metadata={k: MetadataValue.int(v) for k, v in stats.items()}
    )


def _load_icite_works(conn, csv_path: Path) -> int:
    """Update works table with iCite RCR metrics (T1 only)."""
    result = conn.execute(f"""
        UPDATE works SET
            rcr = m.relative_citation_ratio,
            nih_percentile = m.nih_percentile,
            apt = m.apt,
            is_clinical = CAST(m.is_clinical AS BOOLEAN)
        FROM read_csv('{csv_path}',
             auto_detect=true, ignore_errors=true, parallel=true) m
        WHERE works.pmid = m.pmid
    """)
    return result.fetchone()[0]
