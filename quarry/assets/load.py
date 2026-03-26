"""ELT pipeline assets: init → parse → CH load → CH transform → parquet → serve.

Asset graph:
                                                ch_init ────────┐
  oa_sync ──→ oa_parse ──────────┐              (DB + tables)   │
                                 │                              │
  pm_sync ──→ pm_parse ──────────┼──→ ch_load ──→ ch_transform ──→ parquet_export ──┬──→ pg_load
                                 │                                                  ├──→ paper_embeddings
  mesh_sync ──→ mesh_stage ──────┤                                                  └──→ csr_graph
                                 │
  icite_sync ────────────────────┘

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
        # Parquet files live in Hive-partitioned dirs (updated_date=.../) but
        # the column already exists in the file — disable auto-detection.
        "--use_hive_partitioning",
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


# ── CH init asset ──

_CH_SCHEMA_SQL = Path(__file__).resolve().parent.parent.parent / "sql" / "ch_schema.sql"


def _ch_query_no_db(
    query: str, context: AssetExecutionContext, label: str | None = None
) -> None:
    """Run a CH query without --database (for CREATE DATABASE)."""
    cmd = [
        "clickhouse-client",
        "--host",
        settings.ch_host,
        "--port",
        str(settings.ch_port),
        "--query",
        query,
    ]
    run(cmd, context, label=label or f"[CH] {query}")


@asset(
    group_name="init",
    description="Create CH database + tables from ch_schema.sql (idempotent).",
    kinds={"clickhouse"},
)
def ch_init(context: AssetExecutionContext) -> MaterializeResult:
    content = _CH_SCHEMA_SQL.read_text()
    # Strip block comments (/* ... */) before splitting
    cleaned = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    stmts = [s.strip() for s in cleaned.split(";") if s.strip()]

    for stmt in stmts:
        if stmt.startswith("--"):
            continue
        # CREATE DATABASE runs without --database flag
        if stmt.upper().startswith("CREATE DATABASE"):
            _ch_query_no_db(stmt, context, label=f"[CH] {stmt.split(chr(10))[0]}")
        elif stmt.upper().startswith("USE"):
            continue  # --database flag handles this
        else:
            _ch_query(stmt, context)

    return MaterializeResult(
        metadata={"status": MetadataValue.text("ok")},
    )


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
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


@asset(
    group_name="load",
    deps=[ch_init, oa_parse, pm_parse, mesh_stage, icite_metadata_sync],
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


# ── Parquet export ──

# CH table → (pg_table, columns) for Parquet export.
# Column list matches PG schema exactly; CH export SQL pre-formats types
# (e.g. pub_type Array → '{a,b}' PG text[] literal).
# works is exported separately with Hive partitioning by tier.
_WORKS_CH_TABLE = "works_export"
# Full column list (includes tier — reconstructed from Hive partition key on read).
_WORKS_COLUMNS = (
    "work_id, work_id_int, tier, pmid, doi, title, abstract, "
    "pub_year, pub_date, type, cited_by_count, host_venue, oa_status, oa_url, "
    "is_retracted, updated_date, pm_journal_abbr, pm_country, pm_medline_status, "
    "pm_pub_type, pm_created_date, pm_revised_date, pm_indexed_date, "
    "rcr, nih_percentile, apt, is_clinical"
)
# Export columns (tier excluded — encoded in Hive directory structure).
_WORKS_EXPORT_COLUMNS = (
    "work_id, work_id_int, pmid, doi, title, abstract, "
    "pub_year, pub_date, type, cited_by_count, host_venue, oa_status, oa_url, "
    "is_retracted, updated_date, pm_journal_abbr, pm_country, pm_medline_status, "
    "pm_pub_type, pm_created_date, pm_revised_date, pm_indexed_date, "
    "rcr, nih_percentile, apt, is_clinical"
)
_WORKS_TIERS = ["t1", "t2", "t3", "t4"]

_EXPORT_TABLES: list[tuple[str, str, str]] = [
    # (ch_table, pg_table, columns)
    ("oa_work_citations", "work_citations", "citing_id, cited_id"),
    (
        "papers_export",
        "papers",
        "pmid, doi, pmc_id, title, abstract, pub_year, pub_date, "
        "journal_title, journal_issn, journal_abbr, volume, issue, pages, "
        "language, pub_type, country, medline_status, created_date, revised_date, "
        "indexed_date, is_deleted, deleted_date, rcr, nih_percentile, apt, "
        "is_clinical, human, animal, molecular_cellular, field_citation_rate",
    ),
    (
        "oa_work_authors",
        "work_authors",
        "work_id, author_position, display_name, orcid, "
        "institution_name, institution_ror, raw_affiliation",
    ),
    (
        "oa_work_topics",
        "work_topics",
        "work_id, topic_id, topic_name, subfield, field, domain, score, is_primary",
    ),
    (
        "pm_authors",
        "authors",
        "pmid, author_position, last_name, fore_name, initials, "
        "orcid, affiliation, is_collective",
    ),
    (
        "pm_mesh_headings",
        "mesh_headings",
        "pmid, descriptor_ui, descriptor_name, qualifier_ui, "
        "qualifier_name, is_major_topic",
    ),
    ("pm_grants", "grants", "pmid, grant_id, acronym, agency, country"),
    (
        "pm_chemicals",
        "chemicals",
        "pmid, registry_number, substance_ui, substance_name",
    ),
    ("oa_id_crosswalk", "id_crosswalk", "work_id, pmid"),
    ("pm_mesh_tree", "mesh_tree", "descriptor_ui, descriptor_name, tree_number"),
    (
        "work_mesh_export",
        "work_mesh",
        "work_id, descriptor_ui, descriptor_name, qualifier_ui, "
        "qualifier_name, is_major_topic",
    ),
    ("cited_by_clin_export", "cited_by_clin", "pmid, citing_pmid"),
]


def _ch_export_one(
    ch_table: str, columns: str, out_path: Path
) -> subprocess.CompletedProcess:
    """Export one CH table to Parquet file via subprocess."""
    query = f"SELECT {columns} FROM {ch_table}"
    cmd = _ch_client_cmd() + ["--query", query + " FORMAT Parquet"]
    with open(out_path, "wb") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)


@asset(
    group_name="export",
    deps=[ch_transform],
    description="CH export tables → 13 Parquet files (parallel subprocess).",
    kinds={"clickhouse", "parquet"},
)
def parquet_export(context: AssetExecutionContext) -> MaterializeResult:
    out_dir = Path(settings.parquet_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    total = len(_EXPORT_TABLES) + len(_WORKS_TIERS)

    # --- works: Hive-partitioned by tier ---
    def _export_works_tier(tier: str) -> str:
        tier_dir = out_dir / "works" / f"tier={tier}"
        tier_dir.mkdir(parents=True, exist_ok=True)
        out_path = tier_dir / "data.parquet"
        query = f"SELECT {_WORKS_EXPORT_COLUMNS} FROM {_WORKS_CH_TABLE} WHERE tier = '{tier}'"
        cmd = _ch_client_cmd() + ["--query", query + " FORMAT Parquet"]
        context.log.info(f"[Parquet] exporting works tier={tier} → {out_path}")
        with open(out_path, "wb") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            context.log.error(
                f"[Parquet] works tier={tier} failed: {result.stderr.strip()}"
            )
            raise RuntimeError(
                f"Parquet export failed for works tier={tier}: {result.stderr}"
            )
        return f"works/tier={tier}"

    # --- flat tables ---
    def _export(ch_table: str, pg_table: str, columns: str) -> str:
        out_path = out_dir / f"{pg_table}.parquet"
        context.log.info(f"[Parquet] exporting {ch_table} → {out_path}")
        result = _ch_export_one(ch_table, columns, out_path)
        if result.returncode != 0:
            context.log.error(f"[Parquet] {pg_table} failed: {result.stderr.strip()}")
            raise RuntimeError(f"Parquet export failed for {pg_table}: {result.stderr}")
        return pg_table

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures: dict[object, str] = {}
        for tier in _WORKS_TIERS:
            futures[pool.submit(_export_works_tier, tier)] = f"works/tier={tier}"
        for ch, pg, cols in _EXPORT_TABLES:
            futures[pool.submit(_export, ch, pg, cols)] = pg

        done = 0
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                done += 1
                context.log.info(f"[Parquet] {name} done [{done}/{total}]")
            except RuntimeError:
                failed.append(name)

    if failed:
        raise RuntimeError(f"Parquet export failed for: {', '.join(failed)}")

    return MaterializeResult(
        metadata={
            "parquet_dir": MetadataValue.path(str(out_dir)),
            "num_tables": MetadataValue.int(total),
        }
    )


# ── PG load asset ──


# PG text[] columns stored as '{a,b}' VARCHAR in Parquet.
# DuckDB needs explicit conversion: strip braces → split → array.
_TEXT_ARRAY_COLUMNS = {"pm_pub_type", "pub_type"}


def _select_expr(columns: str) -> str:
    """Build SELECT expression, converting '{a,b}' VARCHAR → VARCHAR[] for text[] cols."""
    parts = []
    for col in (c.strip() for c in columns.split(",")):
        if col in _TEXT_ARRAY_COLUMNS:
            parts.append(f"string_split(trim('{{}}'  FROM {col}), ',') AS {col}")
        else:
            parts.append(col)
    return ", ".join(parts)


def _duckdb_pg_load(pg_table: str, columns: str, parquet_path: Path) -> None:
    """Insert Parquet → PG via DuckDB Python binding + postgres extension.

    parquet_path can be a single .parquet file or a Hive-partitioned directory.
    """
    import duckdb

    select = _select_expr(columns)
    # Hive directory: glob all parquet files; single file: use directly
    src = f"{parquet_path}/**/*.parquet" if parquet_path.is_dir() else str(parquet_path)
    conn = duckdb.connect()
    conn.execute("INSTALL postgres; LOAD postgres;")
    conn.execute(f"ATTACH '{settings.pg_conninfo}' AS pg (TYPE postgres)")
    conn.execute(
        f"INSERT INTO pg.{pg_table} ({columns}) "
        f"SELECT {select} FROM read_parquet('{src}', hive_partitioning=true)"
    )
    conn.close()


@asset(
    group_name="serve",
    deps=[parquet_export],
    description="Parquet → PG via DuckDB postgres extension (sequential).",
    kinds={"parquet", "postgres"},
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

    pq_dir = Path(settings.parquet_dir)

    # works: Hive-partitioned directory
    works_dir = pq_dir / "works"
    context.log.info(f"[PG] loading works from {works_dir}")
    _duckdb_pg_load("works", _WORKS_COLUMNS, works_dir)
    context.log.info("[PG] works done [1/{n}]".format(n=len(_EXPORT_TABLES) + 1))

    for i, (_, pg_table, columns) in enumerate(_EXPORT_TABLES, 2):
        pq_path = pq_dir / f"{pg_table}.parquet"
        context.log.info(f"[PG] loading {pg_table} from {pq_path}")
        _duckdb_pg_load(pg_table, columns, pq_path)
        context.log.info(f"[PG] {pg_table} done [{i}/{len(_EXPORT_TABLES) + 1}]")

    _psql("\\i sql/schema.sql", context, label="[PG] recreating indexes")

    _psql("VACUUM ANALYZE works", context, label="[PG] VACUUM ANALYZE works")
    for _, t, _ in _EXPORT_TABLES:
        _psql(f"VACUUM ANALYZE {t}", context, label=f"[PG] VACUUM ANALYZE {t}")

    return MaterializeResult(
        metadata={"status": MetadataValue.text("ok")},
    )
