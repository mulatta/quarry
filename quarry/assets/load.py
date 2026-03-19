"""DuckDB bulk load asset: single sequential DB writer.

This is the ONLY asset that writes to DuckDB, eliminating lock contention.

Performance strategy:
- Baseline: parse XML directly → DuckDB INSERT (no Feather staging, saves 66GB I/O)
- Updates: Feather staging → INSERT OR REPLACE (incremental, small volume)
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
    pubmed_baseline_sync,
    pubmed_updates_sync,
)
from quarry.assets.stage import mesh_stage
from quarry.config import settings
from quarry.etl.feather import read_tables
from quarry.resources import DuckDBResource

import quarry_parse

# Column order matching Rust Arrow schema
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
        pubmed_baseline_sync,
        pubmed_updates_sync,
        mesh_stage,
        icite_metadata_sync,
    ],
    description="Bulk load all data into DuckDB (single writer).",
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
    stats: dict[str, int] = {}

    # Tune DuckDB for bulk load
    conn.execute("SET memory_limit='40GB'")
    conn.execute("SET threads=8")
    conn.execute("SET preserve_insertion_order=false")

    # Drop indexes for faster bulk insert
    context.log.info("Dropping indexes for bulk load")
    for idx_name, _, _ in _INDEXES:
        conn.execute(f"DROP INDEX IF EXISTS {idx_name}")

    # 1. PubMed baseline (parse XML → Arrow → DuckDB directly, no Feather)
    n = _load_baseline_direct(conn, context)
    stats["baseline_papers"] = n
    context.log.info(f"Baseline: {n:,} papers loaded")

    # 2. PubMed updates (parse XML directly — same as baseline)
    n = _load_updates_direct(conn, context)
    stats["update_papers"] = n
    context.log.info(f"Updates: {n:,} papers loaded")

    # 3. MeSH tree
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

    # 4. iCite metrics (DuckDB reads CSV directly)
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


def _load_baseline_direct(conn, context) -> int:
    """Parse baseline XML directly into DuckDB — no Feather staging.

    XML.gz → [Rust/rayon parse] → Arrow RecordBatch → DuckDB INSERT.
    Saves ~66GB disk I/O (33GB write + 33GB read of Feather files).
    """
    files = sorted(settings.pubmed_baseline_dir.glob("pubmed*.xml.gz"))
    if not files:
        return 0

    chunk_size = 20
    n_chunks = (len(files) + chunk_size - 1) // chunk_size
    context.log.info(
        f"Baseline: parsing {len(files)} XML files → DuckDB directly ({n_chunks} chunks)"
    )

    # Truncate all PubMed tables
    conn.execute("DELETE FROM papers")
    for table_name in _CHILD_TABLES:
        conn.execute(f"DELETE FROM {table_name}")

    total = 0
    for i in range(0, len(files), chunk_size):
        chunk_files = files[i : i + chunk_size]
        result = quarry_parse.parse_pubmed_files([str(p) for p in chunk_files])

        # Papers
        papers = result["papers"]
        if papers is not None and papers.num_rows > 0:
            conn.register("_stg_papers", papers)
            conn.execute(
                f"INSERT INTO papers ({_COLS_INSERT}) "
                f"SELECT {_COLS_SELECT} FROM _stg_papers"
            )
            conn.unregister("_stg_papers")
            total += papers.num_rows

        # Child tables
        for table_name, columns in _CHILD_TABLES.items():
            t = result.get(table_name)
            if t is not None and t.num_rows > 0:
                tmp = f"_stg_{table_name}"
                conn.register(tmp, t)
                cols = ", ".join(columns)
                conn.execute(
                    f"INSERT INTO {table_name} ({cols}) SELECT {cols} FROM {tmp}"
                )
                conn.unregister(tmp)

        chunk_idx = i // chunk_size
        context.log.info(
            f"  [{chunk_idx + 1}/{n_chunks}] +{papers.num_rows if papers is not None else 0:,} papers "
            f"(total: {total:,})"
        )
    return total


def _load_updates_direct(conn, context) -> int:
    """Parse update XML directly into DuckDB — no Feather staging.

    Same pattern as _load_baseline_direct but with DELETE+INSERT (incremental).
    """
    files = sorted(settings.pubmed_update_dir.glob("pubmed*.xml.gz"))
    if not files:
        return 0

    context.log.info(f"Updates: parsing {len(files)} XML files → DuckDB directly")

    total = 0
    for i, f in enumerate(files):
        result = quarry_parse.parse_pubmed_file(str(f))

        papers = result["papers"]
        if papers is not None and papers.num_rows > 0:
            conn.register("_stg_papers", papers)
            conn.execute(
                "DELETE FROM papers WHERE pmid IN (SELECT pmid FROM _stg_papers)"
            )
            conn.execute(
                f"INSERT INTO papers ({_COLS_INSERT}) "
                f"SELECT {_COLS_SELECT} FROM _stg_papers"
            )
            conn.unregister("_stg_papers")
            total += papers.num_rows

        for table_name, columns in _CHILD_TABLES.items():
            t = result.get(table_name)
            if t is not None and t.num_rows > 0:
                tmp = f"_stg_{table_name}"
                conn.register(tmp, t)
                conn.execute(
                    f"DELETE FROM {table_name} "
                    f"WHERE pmid IN (SELECT DISTINCT pmid FROM {tmp})"
                )
                cols = ", ".join(columns)
                conn.execute(
                    f"INSERT INTO {table_name} ({cols}) SELECT {cols} FROM {tmp}"
                )
                conn.unregister(tmp)

        if result["delete_pmids"]:
            import pyarrow as pa

            del_t = pa.table(
                {"pmid": pa.array(result["delete_pmids"], type=pa.int32())}
            )
            conn.register("_stg_del", del_t)
            conn.execute(
                "UPDATE papers SET is_deleted = TRUE, deleted_date = CURRENT_DATE "
                "WHERE pmid IN (SELECT pmid FROM _stg_del)"
            )
            conn.unregister("_stg_del")

        context.log.info(
            f"  [{i + 1}/{len(files)}] {f.name}: papers={result['stats']['num_papers']}"
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
            cited_by_clin = m.cited_by_clin,
            field_citation_rate = m.field_citation_rate
        FROM read_csv('{csv_path}',
             auto_detect=true, ignore_errors=true, parallel=true) m
        WHERE papers.pmid = m.pmid
    """)
    n = result.fetchone()[0]
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
    description="Load OpenAlex S3 snapshot → DuckDB works + child tables via httpfs ELT.",
    deps=[duckdb_load],
    kinds={"duckdb", "s3"},
)
def oa_snapshot_load(
    context: AssetExecutionContext,
    duckdb: DuckDBResource,
) -> MaterializeResult:
    """DuckDB httpfs → S3 glob → works + child tables.

    ELT pattern: raw load → SQL transform → Python abstract post-processing.
    """
    import json
    from quarry.etl.openalex import reconstruct_abstract

    db = duckdb.store
    conn = db.conn
    stats: dict[str, int] = {}
    oa_prefix = settings.oa_s3_prefix
    t2_domains = settings.oa_t2_domains

    # Tune DuckDB for bulk load
    conn.execute("SET memory_limit='40GB'")
    conn.execute("SET threads=8")
    conn.execute("SET preserve_insertion_order=false")

    # Install and load httpfs for S3 access
    conn.execute("INSTALL httpfs")
    conn.execute("LOAD httpfs")
    conn.execute("SET s3_region = 'us-east-1'")
    conn.execute("SET s3_access_key_id = ''")
    conn.execute("SET s3_secret_access_key = ''")

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

    # List S3 partitions
    context.log.info(f"Listing S3 partitions from {oa_prefix}")
    partitions = conn.execute(f"""
        SELECT DISTINCT file
        FROM glob('{oa_prefix}/updated_date=*/*.gz')
        ORDER BY file
    """).fetchall()

    if not partitions:
        context.log.warning("No S3 partitions found — loading from single glob")
        partitions = [(f"{oa_prefix}/updated_date=*/*.gz",)]

    # Build domain list for tier SQL
    domain_list = ", ".join(f"'{d}'" for d in t2_domains)

    # ELT: raw load → transform
    total_works = 0
    glob_pattern = f"{oa_prefix}/updated_date=*/*.gz"
    context.log.info(f"Loading raw data from {glob_pattern}")

    # Create temp table with raw data
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE oa_raw AS
        SELECT id, ids, title, abstract_inverted_index,
               publication_year, publication_date, type,
               cited_by_count, is_retracted,
               primary_location, open_access, primary_topic,
               authorships, topics, referenced_works, updated_date
        FROM read_json(
            '{glob_pattern}',
            format = 'newline_delimited',
            compression = 'gzip',
            ignore_errors = true,
            columns = {{
                id: 'VARCHAR', ids: 'JSON', title: 'VARCHAR',
                abstract_inverted_index: 'JSON',
                publication_year: 'SMALLINT', publication_date: 'VARCHAR',
                type: 'VARCHAR', cited_by_count: 'INTEGER',
                is_retracted: 'BOOLEAN',
                primary_location: 'JSON', open_access: 'JSON',
                primary_topic: 'JSON', authorships: 'JSON',
                topics: 'JSON', referenced_works: 'VARCHAR[]',
                updated_date: 'VARCHAR'
            }}
        )
    """)

    raw_count = conn.execute("SELECT COUNT(*) FROM oa_raw").fetchone()[0]
    context.log.info(f"Raw load: {raw_count:,} rows")
    stats["raw_rows"] = raw_count

    # Transform: works table
    context.log.info("Transforming → works")
    conn.execute(f"""
        INSERT INTO works (
            work_id, work_id_int, tier, pmid, doi, title, abstract,
            pub_year, pub_date, type, cited_by_count,
            host_venue, oa_status, oa_url,
            is_retracted, updated_date
        )
        SELECT
            REPLACE(id, 'https://openalex.org/', '') AS work_id,
            CAST(REPLACE(id, 'https://openalex.org/W', '') AS BIGINT) AS work_id_int,
            CASE
                WHEN json_extract_string(ids, '$.pmid') IS NOT NULL THEN 't1'
                WHEN json_extract_string(
                    json_extract(primary_topic, '$.domain'), '$.display_name'
                ) IN ({domain_list}) THEN 't2'
                ELSE 't3'
            END AS tier,
            TRY_CAST(
                REPLACE(json_extract_string(ids, '$.pmid'), 'https://pubmed.ncbi.nlm.nih.gov/', '')
                AS INTEGER
            ) AS pmid,
            json_extract_string(ids, '$.doi') AS doi,
            COALESCE(title, '') AS title,
            NULL AS abstract,
            publication_year,
            TRY_CAST(publication_date AS DATE),
            type,
            cited_by_count,
            json_extract_string(
                json_extract(primary_location, '$.source'), '$.display_name'
            ),
            json_extract_string(open_access, '$.oa_status'),
            json_extract_string(open_access, '$.oa_url'),
            COALESCE(is_retracted, false),
            TRY_CAST(updated_date AS DATE)
        FROM oa_raw
    """)
    total_works = conn.execute("SELECT COUNT(*) FROM works").fetchone()[0]
    context.log.info(f"Works: {total_works:,} rows inserted")
    stats["works"] = total_works

    # Transform: work_authors (T1+T2 only)
    context.log.info("Transforming → work_authors")
    conn.execute("""
        INSERT INTO work_authors
        SELECT
            REPLACE(r.id, 'https://openalex.org/', ''),
            CAST(ROW_NUMBER() OVER (PARTITION BY r.id ORDER BY (SELECT NULL)) AS SMALLINT),
            json_extract_string(a, '$.author.display_name'),
            json_extract_string(a, '$.author.orcid'),
            json_extract_string(json_extract(a, '$.institutions[0]'), '$.display_name'),
            json_extract_string(json_extract(a, '$.institutions[0]'), '$.ror'),
            json_extract_string(a, '$.raw_affiliation_strings[0]')
        FROM oa_raw r, UNNEST(CAST(r.authorships AS JSON[])) AS t(a)
        WHERE EXISTS (SELECT 1 FROM works w
                      WHERE w.work_id = REPLACE(r.id, 'https://openalex.org/', '')
                      AND w.tier IN ('t1','t2'))
    """)
    n_authors = conn.execute("SELECT COUNT(*) FROM work_authors").fetchone()[0]
    context.log.info(f"Work authors: {n_authors:,} rows")
    stats["work_authors"] = n_authors

    # Transform: work_topics (T1+T2 only)
    context.log.info("Transforming → work_topics")
    conn.execute("""
        INSERT INTO work_topics
        SELECT
            REPLACE(r.id, 'https://openalex.org/', ''),
            REPLACE(json_extract_string(t, '$.id'), 'https://openalex.org/', ''),
            json_extract_string(t, '$.display_name'),
            json_extract_string(json_extract(t, '$.subfield'), '$.display_name'),
            json_extract_string(json_extract(t, '$.field'), '$.display_name'),
            json_extract_string(json_extract(t, '$.domain'), '$.display_name'),
            TRY_CAST(json_extract_string(t, '$.score') AS FLOAT)
        FROM oa_raw r, UNNEST(CAST(r.topics AS JSON[])) AS u(t)
        WHERE EXISTS (SELECT 1 FROM works w
                      WHERE w.work_id = REPLACE(r.id, 'https://openalex.org/', '')
                      AND w.tier IN ('t1','t2'))
    """)
    n_topics = conn.execute("SELECT COUNT(*) FROM work_topics").fetchone()[0]
    context.log.info(f"Work topics: {n_topics:,} rows")
    stats["work_topics"] = n_topics

    # Transform: work_citations (all tiers — graph completeness)
    context.log.info("Transforming → work_citations")
    conn.execute("""
        INSERT INTO work_citations (citing_id, cited_id)
        SELECT
            CAST(REPLACE(r.id, 'https://openalex.org/W', '') AS BIGINT),
            CAST(REPLACE(UNNEST(r.referenced_works), 'https://openalex.org/W', '') AS BIGINT)
        FROM oa_raw r
        WHERE r.referenced_works IS NOT NULL
    """)
    n_citations = conn.execute("SELECT COUNT(*) FROM work_citations").fetchone()[0]
    context.log.info(f"Work citations: {n_citations:,} rows")
    stats["work_citations"] = n_citations

    # id_crosswalk
    context.log.info("Building id_crosswalk")
    conn.execute("""
        INSERT INTO id_crosswalk (work_id, pmid)
        SELECT work_id, pmid FROM works WHERE pmid IS NOT NULL
    """)
    n_crosswalk = conn.execute("SELECT COUNT(*) FROM id_crosswalk").fetchone()[0]
    stats["id_crosswalk"] = n_crosswalk

    # work_mesh: JOIN mesh_headings (PMID-based) with id_crosswalk
    context.log.info("Building work_mesh from mesh_headings JOIN id_crosswalk")
    n_mesh_src = conn.execute("SELECT COUNT(*) FROM mesh_headings").fetchone()[0]
    if n_mesh_src > 0:
        conn.execute("""
            INSERT INTO work_mesh
            SELECT
                c.work_id,
                m.descriptor_ui, m.descriptor_name,
                m.qualifier_ui, m.qualifier_name,
                m.is_major_topic
            FROM mesh_headings m
            JOIN id_crosswalk c ON m.pmid = c.pmid
        """)
        n_work_mesh = conn.execute("SELECT COUNT(*) FROM work_mesh").fetchone()[0]
        stats["work_mesh"] = n_work_mesh
        context.log.info(f"Work mesh: {n_work_mesh:,} rows")

    # Abstract reconstruction (T1+T2 only, Python post-processing)
    context.log.info("Reconstructing abstracts (T1+T2)")
    n_abstracts = _reconstruct_abstracts(conn, context, reconstruct_abstract, json)
    stats["abstracts_reconstructed"] = n_abstracts

    # Drop temp table
    conn.execute("DROP TABLE IF EXISTS oa_raw")

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


def _reconstruct_abstracts(conn, context, reconstruct_fn, json_mod) -> int:
    """Reconstruct abstracts from abstract_inverted_index for T1+T2 works."""
    cursor = ""
    total = 0
    while True:
        rows = conn.execute(
            """
            SELECT r.id, r.abstract_inverted_index
            FROM oa_raw r
            JOIN works w ON REPLACE(r.id, 'https://openalex.org/', '') = w.work_id
            WHERE w.tier IN ('t1', 't2')
              AND r.abstract_inverted_index IS NOT NULL
              AND r.id > ?
            ORDER BY r.id LIMIT 10000
        """,
            [cursor],
        ).fetchall()
        if not rows:
            break
        updates = []
        for work_url, inv_idx_str in rows:
            if not inv_idx_str:
                continue
            try:
                inv_idx = json_mod.loads(inv_idx_str)
                abstract = reconstruct_fn(inv_idx)
                if abstract:
                    work_id = work_url.replace("https://openalex.org/", "")
                    updates.append((abstract, work_id))
            except Exception:
                continue
        if updates:
            conn.executemany("UPDATE works SET abstract = ? WHERE work_id = ?", updates)
        cursor = rows[-1][0]
        total += len(updates)
        if total % 100000 < 10000:
            context.log.info(f"  Abstracts: {total:,} reconstructed")
    return total


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
