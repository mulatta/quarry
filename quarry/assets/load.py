"""ELT pipeline assets: parse → CH load → CH transform → serve.

Asset graph:
  oa_sync ──→ oa_parse ──────────┐
                                 │
  pm_sync ──→ pm_parse ──────────┼──→ ch_load ──→ ch_transform ──┬──→ pg_load
                                 │                               ├──→ paper_embeddings (CH pipe)
  mesh_sync ──→ mesh_stage ──────┤                               └──→ csr_graph (CH pipe)
                                 │
  icite_sync ────────────────────┘

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

import re
import subprocess
import time

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.download import (
    icite_metadata_sync,
    oa_sync,
    pubmed_baseline_sync,
    pubmed_updates_sync,
)
from quarry.assets.helpers import run, run_parse
from quarry.assets.stage import mesh_stage
from quarry.config import settings


def _ch_client_cmd() -> list[str]:
    """Base clickhouse-client command."""
    return [
        "clickhouse-client",
        "--host",
        settings.ch_host,
        "--port",
        str(settings.ch_port),
        "--database",
        settings.ch_database,
        "--receive_timeout",
        "7200",
        "--send_timeout",
        "7200",
        "--max_table_size_to_drop",
        "0",
        "--max_partition_size_to_drop",
        "0",
    ]


def _ch_query(
    query: str, context: AssetExecutionContext, label: str | None = None
) -> None:
    """Run a single CH query."""
    run(_ch_client_cmd() + ["--query", query], context, label=label or f"[CH] {query}")


def _psql(sql: str, context: AssetExecutionContext, label: str | None = None) -> None:
    """Run psql command."""
    run(
        [
            "psql",
            settings.pg_conninfo,
            "-c",
            sql,
        ],
        context,
        label=label or f"[PG] {sql}",
    )


def _ch_to_pg(
    ch_table: str,
    pg_table: str,
    pg_columns: str,
    context: AssetExecutionContext,
) -> tuple[str, subprocess.Popen, subprocess.Popen]:
    """Pipe CH query output (TabSeparated) to PG COPY FROM STDIN.

    Returns (pg_table, ch_proc, pg_proc) so caller can check both exit codes.
    """
    ch_query = f"SELECT {pg_columns} FROM {ch_table}"
    ch_cmd = _ch_client_cmd() + ["--query", ch_query, "--format", "TabSeparated"]
    pg_cmd = [
        "psql",
        settings.pg_conninfo,
        "-c",
        f"COPY {pg_table} ({pg_columns}) FROM STDIN",
    ]
    context.log.info(f"[PG] pipe: {ch_table} → {pg_table}")
    ch_proc = subprocess.Popen(ch_cmd, stdout=subprocess.PIPE)
    pg_proc = subprocess.Popen(pg_cmd, stdin=ch_proc.stdout)
    assert ch_proc.stdout is not None
    ch_proc.stdout.close()
    return pg_table, ch_proc, pg_proc


# ── Parse assets ──


@asset(
    group_name="parse",
    deps=[oa_sync],
    description="quarry-parse oa: gz JSONL → Parquet.",
    kinds={"rust", "parquet"},
)
def oa_parse(context: AssetExecutionContext) -> MaterializeResult:
    run_parse(
        [
            "oa",
            "--input-dir",
            str(settings.oa_local_dir),
            "--output-dir",
            str(settings.oa_parquet_dir),
        ],
        context,
    )
    return MaterializeResult(
        metadata={"status": MetadataValue.text("ok")},
    )


@asset(
    group_name="parse",
    deps=[pubmed_baseline_sync, pubmed_updates_sync],
    description="quarry-parse pubmed: XML → Parquet.",
    kinds={"rust", "parquet"},
)
def pm_parse(context: AssetExecutionContext) -> MaterializeResult:
    args = [
        "pubmed",
        "--input-dir",
        str(settings.pubmed_baseline_dir),
        "--output-dir",
        str(settings.pm_parquet_dir),
    ]
    update_dir = settings.pubmed_update_dir
    if update_dir.exists() and any(update_dir.glob("pubmed*.xml.gz")):
        args += ["--updates-dir", str(update_dir)]

    run_parse(args, context)
    return MaterializeResult(
        metadata={"status": MetadataValue.text("ok")},
    )


# ── CH load asset ──


def _ch_insert_async(
    ch_table: str,
    infile_glob: str,
    fmt: str,
    context: AssetExecutionContext,
) -> subprocess.Popen:
    """Start a CH INSERT FROM INFILE as a background process."""
    query = f"INSERT INTO {ch_table} FROM INFILE '{infile_glob}' FORMAT {fmt}"
    cmd = _ch_client_cmd() + ["--query", query]
    context.log.info(f"[CH] loading {ch_table}")
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


@asset(
    group_name="load",
    deps=[oa_parse, pm_parse, mesh_stage, icite_metadata_sync],
    description="Parquet + CSV → CH: parallel INSERT across all tables.",
    kinds={"clickhouse"},
)
def ch_load(context: AssetExecutionContext) -> MaterializeResult:
    oa_tables = [
        "works",
        "work_authors",
        "work_topics",
        "work_citations",
        "id_crosswalk",
    ]
    pm_tables = ["papers", "authors", "mesh_headings", "grants", "chemicals"]

    # TRUNCATE all tables
    for table in oa_tables:
        _ch_query(
            f"TRUNCATE TABLE oa_{table}", context, label=f"[CH] TRUNCATE oa_{table}"
        )
    for table in pm_tables:
        _ch_query(
            f"TRUNCATE TABLE pm_{table}", context, label=f"[CH] TRUNCATE pm_{table}"
        )
    _ch_query(
        "TRUNCATE TABLE pm_mesh_tree", context, label="[CH] TRUNCATE pm_mesh_tree"
    )
    _ch_query("TRUNCATE TABLE icite_raw", context, label="[CH] TRUNCATE icite_raw")

    # Parallel INSERT — launch all at once, wait for all
    procs: list[tuple[str, subprocess.Popen]] = []

    oa_dir = str(settings.oa_parquet_dir)
    for table in oa_tables:
        procs.append(
            (
                f"oa_{table}",
                _ch_insert_async(
                    f"oa_{table}", f"{oa_dir}/{table}/**/*.parquet", "Parquet", context
                ),
            )
        )

    pm_dir = str(settings.pm_parquet_dir)
    for table in pm_tables:
        procs.append(
            (
                f"pm_{table}",
                _ch_insert_async(
                    f"pm_{table}", f"{pm_dir}/{table}/**/*.parquet", "Parquet", context
                ),
            )
        )

    mesh_parquet = settings.mesh_parquet_dir / "mesh_tree.parquet"
    if mesh_parquet.exists():
        procs.append(
            (
                "pm_mesh_tree",
                _ch_insert_async("pm_mesh_tree", str(mesh_parquet), "Parquet", context),
            )
        )
    else:
        context.log.warning(f"MeSH Parquet not found: {mesh_parquet}, skipping")

    csv_path = settings.icite_dir / "icite_metadata.csv"
    if csv_path.exists():
        procs.append(
            (
                "icite_raw",
                _ch_insert_async("icite_raw", str(csv_path), "CSVWithNames", context),
            )
        )
    else:
        context.log.warning(f"iCite CSV not found: {csv_path}, skipping")

    # Wait for INSERTs — log in completion order
    n = len(procs)
    done = 0
    failed: list[str] = []
    remaining = list(procs)
    while remaining:
        for name, proc in remaining:
            if proc.poll() is not None:
                done += 1
                assert proc.stderr is not None
                stderr = proc.stderr.read()
                if stderr.strip():
                    for line in stderr.strip().split("\n"):
                        context.log.info(f"  [{name}] {line}")
                if proc.returncode != 0:
                    context.log.error(f"[CH] {name} failed (exit {proc.returncode})")
                    failed.append(name)
                else:
                    context.log.info(f"[CH] {name} done [{done}/{n}]")
        remaining = [(name, proc) for name, proc in remaining if proc.poll() is None]
        if remaining:
            time.sleep(1)

    if failed:
        raise RuntimeError(f"CH load failed for: {', '.join(failed)}")

    return MaterializeResult(
        metadata={"status": MetadataValue.text("ok")},
    )


# ── Transform asset ──


@asset(
    group_name="transform",
    deps=[ch_load],
    description="CH: OPTIMIZE FINAL + enriched export tables.",
    kinds={"clickhouse"},
)
def ch_transform(context: AssetExecutionContext) -> MaterializeResult:
    # 1. OPTIMIZE FINAL — dedup all raw tables
    optimize_tables = [
        "oa_works",
        "oa_work_authors",
        "oa_work_topics",
        "oa_work_citations",
        "oa_id_crosswalk",
        "pm_papers",
        "pm_authors",
        "pm_mesh_headings",
        "pm_grants",
        "pm_chemicals",
        "pm_mesh_tree",
        "icite_raw",
    ]
    for table in optimize_tables:
        _ch_query(
            f"OPTIMIZE TABLE {table} FINAL",
            context,
            label=f"[CH] OPTIMIZE {table} FINAL",
        )

    # 2-5. Enriched export tables
    export_sql = "sql/ch_transform.sql"
    context.log.info(f"[CH] reading {export_sql}")
    with open(export_sql) as f:
        content = f.read()

    # Extract CREATE OR REPLACE TABLE statements
    stmts = re.split(r"(?=CREATE OR REPLACE TABLE)", content)
    for stmt in stmts:
        stmt = stmt.strip()
        if not stmt.startswith("CREATE"):
            continue
        # Extract table name for logging
        match = re.match(r"CREATE OR REPLACE TABLE (\S+)", stmt)
        label = f"[CH] CREATE {match.group(1)}" if match else "[CH] CREATE TABLE"
        _ch_query(stmt, context, label=label)

    return MaterializeResult(
        metadata={"status": MetadataValue.text("ok")},
    )


# ── PG load asset ──


@asset(
    group_name="serve",
    deps=[ch_transform],
    description="CH export → PG TRUNCATE + COPY + CREATE INDEX.",
    kinds={"clickhouse", "postgres"},
)
def pg_load(context: AssetExecutionContext) -> MaterializeResult:
    _psql("\\i sql/drop_indexes.sql", context, label="[PG] dropping indexes")
    _psql(
        "TRUNCATE papers, authors, mesh_headings, grants, chemicals, "
        "cited_by_clin, works, work_authors, work_topics, work_mesh, "
        "work_citations, id_crosswalk, mesh_tree CASCADE",
        context,
        label="[PG] truncating all tables",
    )

    # Pipe CH export tables → PG COPY (parallel for large tables)
    pipes: list[tuple[str, subprocess.Popen, subprocess.Popen]] = []

    # Large tables: works, work_citations
    pipes.append(
        _ch_to_pg(
            "works_export",
            "works",
            "work_id, work_id_int, tier, pmid, doi, title, abstract, "
            "pub_year, pub_date, type, cited_by_count, host_venue, oa_status, oa_url, "
            "is_retracted, updated_date, pm_journal_abbr, pm_country, pm_medline_status, "
            "pm_pub_type, pm_created_date, pm_revised_date, pm_indexed_date, "
            "rcr, nih_percentile, apt, is_clinical",
            context,
        )
    )
    pipes.append(
        _ch_to_pg(
            "oa_work_citations",
            "work_citations",
            "citing_id, cited_id",
            context,
        )
    )

    # Medium tables
    pipes.append(
        _ch_to_pg(
            "papers_export",
            "papers",
            "pmid, doi, pmc_id, title, abstract, pub_year, pub_date, "
            "journal_title, journal_issn, journal_abbr, volume, issue, pages, "
            "language, pub_type, country, medline_status, created_date, revised_date, "
            "indexed_date, is_deleted, deleted_date, rcr, nih_percentile, apt, "
            "is_clinical, human, animal, molecular_cellular, field_citation_rate",
            context,
        )
    )
    pipes.append(
        _ch_to_pg(
            "oa_work_authors",
            "work_authors",
            "work_id, author_position, display_name, orcid, "
            "institution_name, institution_ror, raw_affiliation",
            context,
        )
    )
    pipes.append(
        _ch_to_pg(
            "oa_work_topics",
            "work_topics",
            "work_id, topic_id, topic_name, subfield, field, domain, score, is_primary",
            context,
        )
    )
    pipes.append(
        _ch_to_pg(
            "pm_authors",
            "authors",
            "pmid, author_position, last_name, fore_name, initials, "
            "orcid, affiliation, is_collective",
            context,
        )
    )
    pipes.append(
        _ch_to_pg(
            "pm_mesh_headings",
            "mesh_headings",
            "pmid, descriptor_ui, descriptor_name, qualifier_ui, "
            "qualifier_name, is_major_topic",
            context,
        )
    )
    pipes.append(
        _ch_to_pg(
            "pm_grants",
            "grants",
            "pmid, grant_id, acronym, agency, country",
            context,
        )
    )
    pipes.append(
        _ch_to_pg(
            "pm_chemicals",
            "chemicals",
            "pmid, registry_number, substance_ui, substance_name",
            context,
        )
    )
    pipes.append(
        _ch_to_pg(
            "oa_id_crosswalk",
            "id_crosswalk",
            "work_id, pmid",
            context,
        )
    )
    pipes.append(
        _ch_to_pg(
            "pm_mesh_tree",
            "mesh_tree",
            "descriptor_ui, descriptor_name, tree_number",
            context,
        )
    )
    pipes.append(
        _ch_to_pg(
            "work_mesh_export",
            "work_mesh",
            "work_id, descriptor_ui, descriptor_name, qualifier_ui, "
            "qualifier_name, is_major_topic",
            context,
        )
    )
    pipes.append(
        _ch_to_pg(
            "cited_by_clin_export",
            "cited_by_clin",
            "pmid, citing_pmid",
            context,
        )
    )

    n = len(pipes)
    done = 0
    failed: list[str] = []
    remaining: list[tuple[str, subprocess.Popen, subprocess.Popen]] = list(pipes)
    context.log.info(f"[PG] waiting for {n} pipes")
    while remaining:
        for name, ch_proc, pg_proc in remaining:
            if pg_proc.poll() is not None:
                done += 1
                ch_rc = ch_proc.wait()
                if ch_rc != 0:
                    context.log.error(f"[PG] {name}: CH export failed (exit {ch_rc})")
                    failed.append(name)
                elif pg_proc.returncode != 0:
                    context.log.error(
                        f"[PG] {name}: COPY failed (exit {pg_proc.returncode})"
                    )
                    failed.append(name)
                else:
                    context.log.info(f"[PG] {name} done [{done}/{n}]")
        remaining = [(nm, ch, pg) for nm, ch, pg in remaining if pg.poll() is None]
        if remaining:
            time.sleep(1)
    if failed:
        raise RuntimeError(f"PG load failed for: {', '.join(failed)}")

    _psql("\\i sql/schema.sql", context, label="[PG] recreating indexes")

    pg_tables = [
        "papers",
        "authors",
        "mesh_headings",
        "grants",
        "chemicals",
        "cited_by_clin",
        "works",
        "work_authors",
        "work_topics",
        "work_mesh",
        "work_citations",
        "id_crosswalk",
        "mesh_tree",
    ]
    vacuum_sql = "; ".join(f"VACUUM ANALYZE {t}" for t in pg_tables)
    _psql(vacuum_sql, context, label=f"[PG] VACUUM ANALYZE {len(pg_tables)} tables")

    return MaterializeResult(
        metadata={"status": MetadataValue.text("ok")},
    )
