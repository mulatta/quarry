"""PostgreSQL bulk load assets.

Each data source loads directly into PG via quarry_build PyO3 bindings:
- PubMed: quarry_build.build_pubmed_pg()
- OpenAlex: quarry_build.build_oa_s3_pg()
- iCite: psycopg COPY FROM CSV → UPDATE
- Enrichment: quarry_build.enrich_pg()

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from pathlib import Path

import psycopg
import quarry_build
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
from quarry.config import settings
from quarry.resources import PGResource

_SQL_DIR = Path(__file__).resolve().parent.parent.parent / "sql"


def _bulk_load_begin(conn: psycopg.Connection, context: AssetExecutionContext):
    """Drop indexes before bulk COPY load to eliminate WAL overhead."""
    drop_sql = (_SQL_DIR / "drop_indexes.sql").read_text()
    for stmt in drop_sql.split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            conn.execute(stmt)
    context.log.info("bulk_load_begin: indexes dropped")


def _bulk_load_end(conn: psycopg.Connection, context: AssetExecutionContext):
    """Recreate indexes and vacuum after bulk load."""
    schema_sql = (_SQL_DIR / "schema.sql").read_text()
    for stmt in schema_sql.split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            conn.execute(stmt)
    context.log.info("bulk_load_end: indexes created, starting vacuum")
    conn.execute("VACUUM ANALYZE")
    context.log.info("bulk_load_end: vacuum complete")


@asset(
    group_name="build",
    deps=[pubmed_baseline_sync, pubmed_updates_sync],
    description="Rust PubMed XML parser → PG direct load via PyO3.",
    kinds={"rust", "postgres"},
)
def pubmed_pg_load(context: AssetExecutionContext, pg: PGResource) -> MaterializeResult:
    _bulk_load_begin(pg.store.conn, context)

    update_dir = settings.pubmed_update_dir
    updates = (
        str(update_dir)
        if update_dir.exists() and any(update_dir.glob("pubmed*.xml.gz"))
        else None
    )

    stats = quarry_build.build_pubmed_pg(
        pg_conninfo=settings.pg_conninfo,
        xml_dir=str(settings.pubmed_baseline_dir),
        updates_dir=updates,
        threads=settings.pubmed_parse_threads or None,
        pg_writers=settings.pubmed_pg_writers,
        channel_buffer=settings.pubmed_channel_buffer,
    )
    return MaterializeResult(
        metadata={
            k: MetadataValue.int(v) if isinstance(v, int) else MetadataValue.float(v)
            for k, v in stats.items()
        },
    )


@asset(
    group_name="build",
    deps=[pubmed_pg_load],  # sequential: run after pubmed to avoid WAL contention
    description="Rust OpenAlex JSONL parser → PG direct load via PyO3.",
    kinds={"rust", "postgres"},
)
def oa_pg_load(context: AssetExecutionContext, pg: PGResource) -> MaterializeResult:
    _bulk_load_begin(pg.store.conn, context)

    stats = quarry_build.build_oa_s3_pg(
        pg_conninfo=settings.pg_conninfo,
        s3_prefix=settings.oa_s3_prefix,
        s3_concurrency=settings.oa_s3_concurrency,
        pg_writers=settings.oa_pg_writers,
        channel_buffer=settings.oa_channel_buffer,
        fetch_max_retries=settings.oa_fetch_max_retries,
        fetch_initial_backoff_ms=settings.oa_fetch_initial_backoff_ms,
        fetch_max_backoff_ms=settings.oa_fetch_max_backoff_ms,
    )

    _bulk_load_end(pg.store.conn, context)

    return MaterializeResult(
        metadata={
            k: MetadataValue.int(v) if isinstance(v, int) else MetadataValue.float(v)
            for k, v in stats.items()
        },
    )


@asset(
    group_name="load",
    deps=[pubmed_pg_load, oa_pg_load, icite_metadata_sync],
    description="iCite metrics: COPY CSV → temp table → UPDATE papers + works.",
    kinds={"postgres"},
    automation_condition=AutomationCondition.eager(),
)
def icite_pg_load(
    context: AssetExecutionContext,
    pg: PGResource,
) -> MaterializeResult:
    store = pg.store
    conn = store.conn
    stats: dict[str, int] = {}

    meta_csv = settings.icite_dir / "icite_metadata.csv"
    if not meta_csv.exists():
        context.log.warning(f"iCite CSV not found: {meta_csv}")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    context.log.info(f"iCite: loading from {meta_csv}")
    n = _load_icite_csv(conn, meta_csv, context)
    stats["icite_rows"] = n

    return MaterializeResult(
        metadata={k: MetadataValue.int(v) for k, v in stats.items()}
    )


def _load_icite_csv(conn, csv_path: Path, context) -> int:
    """Load iCite metrics from CSV → temp table → UPDATE papers atomically."""
    with conn.transaction():
        # 1. Load CSV into temp table via COPY
        conn.execute("DROP TABLE IF EXISTS _icite_tmp")
        conn.execute("""
            CREATE TEMPORARY TABLE _icite_tmp (
                pmid INTEGER,
                doi TEXT,
                title TEXT,
                authors TEXT,
                year SMALLINT,
                journal TEXT,
                is_research_article TEXT,
                citation_count INTEGER,
                field_citation_rate REAL,
                expected_citations_per_year REAL,
                citations_per_year REAL,
                relative_citation_ratio REAL,
                nih_percentile REAL,
                human REAL,
                animal REAL,
                molecular_cellular REAL,
                x_coord REAL,
                y_coord REAL,
                apt REAL,
                is_clinical TEXT,
                cited_by_clin TEXT,
                cited_by TEXT,
                "references" TEXT,
                provisional TEXT,
                last_modified TEXT
            )
        """)

        # COPY from CSV (HEADER true skips the first line automatically)
        with open(csv_path, "r") as f:
            with conn.cursor() as cur:
                with cur.copy(
                    "COPY _icite_tmp FROM STDIN WITH (FORMAT csv, HEADER true)"
                ) as copy:
                    while chunk := f.read(8 * 1024 * 1024):
                        copy.write(chunk)

        context.log.info("  iCite CSV loaded into temp table")

        # 2. UPDATE papers with iCite metrics
        result = conn.execute("""
            UPDATE papers SET
                rcr = m.relative_citation_ratio,
                nih_percentile = m.nih_percentile,
                apt = m.apt,
                is_clinical = (m.is_clinical = 'Yes'),
                human = m.human,
                animal = m.animal,
                molecular_cellular = m.molecular_cellular,
                field_citation_rate = m.field_citation_rate
            FROM _icite_tmp m
            WHERE papers.pmid = m.pmid
        """)
        total_updated = result.rowcount
        context.log.info(f"  iCite UPDATE papers: {total_updated:,} rows")

        # 3. Update works table too (T1 only — has PMID)
        result = conn.execute("""
            UPDATE works SET
                rcr = m.relative_citation_ratio,
                nih_percentile = m.nih_percentile,
                apt = m.apt,
                is_clinical = (m.is_clinical = 'Yes')
            FROM _icite_tmp m
            WHERE works.pmid = m.pmid
        """)
        context.log.info(f"  iCite UPDATE works: {result.rowcount:,} rows")

        # 4. cited_by_clin table
        conn.execute("TRUNCATE cited_by_clin")
        conn.execute("""
            INSERT INTO cited_by_clin (pmid, citing_pmid)
            SELECT m.pmid, unnest(
                string_to_array(m.cited_by_clin, ' ')
            )::integer
            FROM _icite_tmp m
            WHERE m.cited_by_clin IS NOT NULL AND m.cited_by_clin != ''
        """)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM cited_by_clin")
            n_cbc = cur.fetchone()[0]
        context.log.info(f"  cited_by_clin: {n_cbc:,} rows")

        conn.execute("DROP TABLE IF EXISTS _icite_tmp")
    return total_updated


@asset(
    group_name="load",
    deps=[pubmed_pg_load, oa_pg_load],
    description="Enrich: UPDATE works SET pm_* FROM papers; generate work_mesh via PyO3.",
    kinds={"rust", "postgres"},
)
def enrich_pg(context: AssetExecutionContext) -> MaterializeResult:
    stats = quarry_build.enrich_pg(pg_conninfo=settings.pg_conninfo)
    return MaterializeResult(
        metadata={k: MetadataValue.int(v) for k, v in stats.items()},
    )
