"""PostgreSQL bulk load assets.

Each data source loads directly into PG via quarry_rs PyO3 bindings:
- PubMed: quarry_rs.build_pubmed_pg()
- OpenAlex: quarry_rs.build_oa_s3_pg()
- iCite: psycopg COPY FROM CSV → UPDATE
- Enrichment: quarry_rs.enrich_pg()

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from pathlib import Path

import quarry_rs
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
from quarry.resources import PGResource


@asset(
    group_name="build",
    deps=[pubmed_baseline_sync, pubmed_updates_sync],
    description="Rust PubMed XML parser → PG direct load via PyO3.",
    kinds={"rust", "postgres"},
)
def pubmed_pg_load(context: AssetExecutionContext) -> MaterializeResult:
    update_dir = settings.pubmed_update_dir
    updates = (
        str(update_dir)
        if update_dir.exists() and any(update_dir.glob("pubmed*.xml.gz"))
        else None
    )

    stats = quarry_rs.build_pubmed_pg(
        pg_conninfo=settings.pg_conninfo,
        xml_dir=str(settings.pubmed_baseline_dir),
        updates_dir=updates,
    )
    return MaterializeResult(
        metadata={
            k: MetadataValue.int(v) if isinstance(v, int) else MetadataValue.float(v)
            for k, v in stats.items()
        },
    )


@asset(
    group_name="build",
    description="Rust OpenAlex JSONL parser → PG direct load via PyO3.",
    kinds={"rust", "postgres"},
)
def oa_pg_load(context: AssetExecutionContext) -> MaterializeResult:
    stats = quarry_rs.build_oa_s3_pg(
        pg_conninfo=settings.pg_conninfo,
        s3_prefix=settings.oa_s3_prefix,
    )
    return MaterializeResult(
        metadata={
            k: MetadataValue.int(v) if isinstance(v, int) else MetadataValue.float(v)
            for k, v in stats.items()
        },
    )


@asset(
    group_name="load",
    deps=[pubmed_pg_load, oa_pg_load, mesh_stage, icite_metadata_sync],
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
    """Load iCite metrics from CSV → temp table → UPDATE papers in chunks."""
    # 1. Load CSV into temp table via COPY
    conn.execute("DROP TABLE IF EXISTS _icite_tmp")
    conn.execute("""
        CREATE TEMPORARY TABLE _icite_tmp (
            pmid INTEGER,
            year SMALLINT,
            title TEXT,
            doi TEXT,
            relative_citation_ratio REAL,
            nih_percentile REAL,
            apt REAL,
            is_clinical TEXT,
            human REAL,
            animal REAL,
            molecular_cellular REAL,
            field_citation_rate REAL,
            citation_count INTEGER,
            citations_per_year REAL,
            expected_citations_per_year REAL,
            is_research_article TEXT,
            cited_by_clin TEXT,
            provisional TEXT,
            x_relative_citation_ratio REAL,
            x_nih_percentile REAL
        )
    """)

    # COPY from CSV
    with open(csv_path, "r") as f:
        # Skip header
        f.readline()
        with conn.cursor() as cur:
            with cur.copy("COPY _icite_tmp FROM STDIN WITH (FORMAT csv)") as copy:
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
    stats = quarry_rs.enrich_pg(pg_conninfo=settings.pg_conninfo)
    return MaterializeResult(
        metadata={k: MetadataValue.int(v) for k, v in stats.items()},
    )
