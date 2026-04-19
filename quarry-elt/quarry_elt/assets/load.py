"""ELT pipeline assets: init → parse → CH load → CH transform → serve.

Asset graph:
                                                ch_init ────────┐
  oa_sync ──→ oa_parse ──────────┐              (DB + tables)   │
                                 │                              │
  pm_sync ──→ pm_parse ──────────┼──→ ch_load ──→ ch_transform ──┬──→ pg_load (CH → CSV → psql)
                                 │                               ├──→ paper_embeddings (CH → ArrowStream → GPU)
  mesh_sync ──→ mesh_stage ──────┤                               └──→ parquet_export ──→ csr_graph
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
    AutomationCondition,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry_elt.assets.download import (
    icite_metadata_sync,
    oa_sync,
    pubmed_baseline_sync,
    pubmed_updates_sync,
)
from quarry_elt.assets.helpers import run, run_parse
from quarry_elt.assets.init import ch_init
from quarry_elt.assets.stage import mesh_stage
from quarry.config import settings
from quarry_elt.resources import PGResource


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
        # Performance: parallel INSERT + large memory for JOINs
        "--max_insert_threads",
        "8",
        "--max_memory_usage",
        "64000000000",
    ]


_GRACE_HASH_SETTINGS = [
    "--join_algorithm",
    "grace_hash",
    "--max_bytes_in_join",
    "10000000000",
    "--grace_hash_join_initial_buckets",
    "4",
]


def _ch_query(
    query: str,
    context: AssetExecutionContext,
    label: str | None = None,
    *,
    grace_hash: bool = False,
    external_group_by: bool = False,
) -> None:
    """Run a single CH query."""
    cmd = _ch_client_cmd()
    if grace_hash:
        cmd += _GRACE_HASH_SETTINGS
    if external_group_by:
        cmd += ["--max_bytes_before_external_group_by", "32000000000"]
    run(cmd + ["--query", query], context, label=label or f"[CH] {query}")


def _psql(
    sql: str,
    context: AssetExecutionContext,
    label: str | None = None,
    *,
    pgoptions: str | None = None,
) -> None:
    """Run psql command. Optional PGOPTIONS for session-level GUC overrides."""
    import os

    cmd = ["psql", settings.pg_conninfo, "-c", sql]
    env = None
    if pgoptions:
        env = {**os.environ, "PGOPTIONS": pgoptions}
    run(cmd, context, label=label or f"[PG] {sql}", env=env)


_SQL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "sql"


# ── Parse assets ──


@asset(
    group_name="parse",
    deps=[oa_sync],
    description="quarry-parse oa: gz JSONL → Parquet.",
    kinds={"rust", "parquet"},
    automation_condition=AutomationCondition.eager(),
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
    automation_condition=AutomationCondition.eager(),
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


def _ch_load_tables(
    ch_tables: list[str],
    infile_specs: list[tuple[str, str, str]],
    context: AssetExecutionContext,
) -> None:
    """TRUNCATE + parallel INSERT for a set of CH tables.

    Args:
        ch_tables: table names to TRUNCATE.
        infile_specs: [(ch_table, infile_glob, format), ...] for INSERT.
    """
    for table in ch_tables:
        _ch_query(f"TRUNCATE TABLE {table}", context, label=f"[CH] TRUNCATE {table}")

    procs: list[tuple[str, subprocess.Popen]] = []
    for ch_table, infile_glob, fmt in infile_specs:
        procs.append((ch_table, _ch_insert_async(ch_table, infile_glob, fmt, context)))

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


_OA_TABLES = [
    "works",
    "work_authors",
    "work_topics",
    "work_citations",
    "id_crosswalk",
    "counts_by_year",
]
_PM_TABLES = ["papers", "authors", "mesh_headings", "grants", "chemicals"]


@asset(
    group_name="load",
    deps=[ch_init, oa_parse],
    description="OA Parquet → CH: oa_works + related tables.",
    kinds={"clickhouse"},
)
def ch_load_oa(context: AssetExecutionContext) -> MaterializeResult:
    oa_dir = str(settings.oa_parquet_dir)
    _ch_load_tables(
        [f"oa_{t}" for t in _OA_TABLES],
        [(f"oa_{t}", f"{oa_dir}/{t}/**/*.parquet", "Parquet") for t in _OA_TABLES],
        context,
    )
    return MaterializeResult(metadata={"status": MetadataValue.text("ok")})


@asset(
    group_name="load",
    deps=[ch_init, pm_parse],
    description="PubMed Parquet → CH: pm_papers + related tables.",
    kinds={"clickhouse"},
)
def ch_load_pm(context: AssetExecutionContext) -> MaterializeResult:
    pm_dir = str(settings.pm_parquet_dir)
    _ch_load_tables(
        [f"pm_{t}" for t in _PM_TABLES],
        [(f"pm_{t}", f"{pm_dir}/{t}/*.parquet", "Parquet") for t in _PM_TABLES],
        context,
    )
    return MaterializeResult(metadata={"status": MetadataValue.text("ok")})


@asset(
    group_name="load",
    deps=[ch_init, mesh_stage],
    description="MeSH Parquet → CH: pm_mesh_tree + pm_mesh_terms.",
    kinds={"clickhouse"},
)
def ch_load_mesh(context: AssetExecutionContext) -> MaterializeResult:
    mesh_tree_parquet = settings.mesh_parquet_dir / "mesh_tree.parquet"
    mesh_terms_parquet = settings.mesh_parquet_dir / "mesh_terms.parquet"

    if not mesh_tree_parquet.exists():
        context.log.warning(f"MeSH Parquet not found: {mesh_tree_parquet}, skipping")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    tables_to_truncate = ["pm_mesh_tree"]
    loads = [("pm_mesh_tree", str(mesh_tree_parquet), "Parquet")]

    if mesh_terms_parquet.exists():
        tables_to_truncate.append("pm_mesh_terms")
        loads.append(("pm_mesh_terms", str(mesh_terms_parquet), "Parquet"))
    else:
        context.log.warning(
            f"MeSH terms Parquet not found: {mesh_terms_parquet}, skipping terms"
        )

    _ch_load_tables(tables_to_truncate, loads, context)
    return MaterializeResult(metadata={"status": MetadataValue.text("ok")})


@asset(
    group_name="load",
    deps=[ch_init, icite_metadata_sync],
    description="iCite CSV → CH: icite_raw.",
    kinds={"clickhouse"},
)
def ch_load_icite(context: AssetExecutionContext) -> MaterializeResult:
    csv_path = settings.icite_dir / "icite_metadata.csv"
    if not csv_path.exists():
        context.log.warning(f"iCite CSV not found: {csv_path}, skipping")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})
    _ch_load_tables(
        ["icite_raw"],
        [("icite_raw", str(csv_path), "CSVWithNames")],
        context,
    )
    return MaterializeResult(metadata={"status": MetadataValue.text("ok")})


# ── Transform asset ──


@asset(
    group_name="transform",
    deps=[ch_load_oa, ch_load_pm, ch_load_mesh, ch_load_icite],
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
        "oa_counts_by_year",
        "pm_papers",
        "pm_authors",
        "pm_mesh_headings",
        "pm_grants",
        "pm_chemicals",
        "pm_mesh_tree",
        "pm_mesh_terms",
        "icite_raw",
    ]

    def _optimize(table: str) -> tuple[str, int, str]:
        query = f"OPTIMIZE TABLE {table} FINAL"
        proc = subprocess.run(
            _ch_client_cmd() + ["--query", query],
            capture_output=True,
            text=True,
        )
        return table, proc.returncode, proc.stderr

    with ThreadPoolExecutor(max_workers=len(optimize_tables)) as pool:
        futures = {pool.submit(_optimize, t): t for t in optimize_tables}
        for future in as_completed(futures):
            table, rc, stderr = future.result()
            if stderr.strip():
                context.log.info(f"  [OPTIMIZE {table}] {stderr.strip()}")
            if rc != 0:
                raise RuntimeError(
                    f"OPTIMIZE {table} FINAL failed (exit {rc}): {stderr}"
                )
            context.log.info(f"[CH] OPTIMIZE {table} FINAL done")

    # 2-5. Enriched export tables
    export_sql = _SQL_DIR / "ch_transform.sql"
    context.log.info(f"[CH] reading {export_sql}")
    content = export_sql.read_text()

    # Tables with large JOINs that need grace_hash to stay within 64GB.
    # Tables with large JOINs that need grace_hash to stay within 64GB.
    _GRACE_HASH_TABLES = {
        "_works_raw",  # 3-way LEFT JOIN: oa_works × pm_papers × icite_raw
        "papers_export",  # LEFT JOIN: pm_papers × icite_raw
        "icite_citations",  # 2× INNER JOIN: parsed × works_export
        "merged_citations",  # 2× INNER JOIN paper_ids + ANTI JOIN oa_work_citations
    }
    # Large GROUP BY — spill hash table to disk when it exceeds 32GB.
    _EXTERNAL_GROUP_BY_TABLES = {"_junk_hashes"}

    # Extract CREATE OR REPLACE TABLE statements
    stmts = re.split(r"(?=CREATE OR REPLACE (?:TABLE|VIEW))", content)
    for stmt in stmts:
        stmt = stmt.strip()
        if not stmt.startswith("CREATE"):
            continue
        # CH --query accepts a single statement; trim anything after
        # the final semicolon (trailing section comments from SQL file).
        if ";" in stmt:
            stmt = stmt[: stmt.rindex(";") + 1]
        # Extract table name for logging
        match = re.match(r"CREATE OR REPLACE (?:TABLE|VIEW) (\S+)", stmt)
        table_name = match.group(1) if match else ""
        label = f"[CH] CREATE {table_name}" if match else "[CH] CREATE TABLE"
        _ch_query(
            stmt,
            context,
            label=label,
            grace_hash=table_name in _GRACE_HASH_TABLES,
            external_group_by=table_name in _EXTERNAL_GROUP_BY_TABLES,
        )

    # Drop intermediate table from 2-pass boilerplate filter
    # Drop intermediate tables from 2-pass boilerplate filter
    for tmp in ("_works_raw", "_junk_hashes"):
        _ch_query(
            f"DROP TABLE IF EXISTS {tmp}",
            context,
            label=f"[CH] DROP {tmp}",
        )

    return MaterializeResult(
        metadata={"status": MetadataValue.text("ok")},
    )


# ── Parquet export ──

# CH table → (pg_table, columns) for Parquet export.
# Column list matches PG schema exactly; CH export SQL pre-formats types
# (e.g. pub_type Array → '{a,b}' PG text[] literal).
# works: tier hive partition × work_id_int % 256 bucket files.
_WORKS_CH_TABLE = "works_export"
# PG target columns (tier comes from hive partition, not parquet file).
_WORKS_COLUMNS = (
    "work_id, work_id_int, tier, pmid, doi, title, abstract, "
    "pub_year, pub_date, type, cited_by_count, host_venue, oa_status, oa_url, "
    "is_retracted, updated_date, pm_journal_abbr, pm_country, pm_medline_status, "
    "pm_pub_type, pm_created_date, pm_revised_date, pm_indexed_date, "
    "rcr, nih_percentile, apt, is_clinical, "
    "language, fwci, citation_normalized_percentile, "
    "cited_by_percentile_year_min, cited_by_percentile_year_max"
)
# Parquet export columns: tier excluded (derived from hive directory tier=t1/).
_WORKS_EXPORT_COLUMNS = (
    "work_id, work_id_int, pmid, doi, title, abstract, content_hash, "
    "pub_year, pub_date, type, cited_by_count, host_venue, oa_status, oa_url, "
    "is_retracted, updated_date, pm_journal_abbr, pm_country, pm_medline_status, "
    "pm_pub_type, pm_created_date, pm_revised_date, pm_indexed_date, "
    "rcr, nih_percentile, apt, is_clinical, "
    "language, fwci, citation_normalized_percentile, "
    "cited_by_percentile_year_min, cited_by_percentile_year_max"
)
_TIERS = ["t1", "t2", "t3", "t4"]
_BUCKETS = 256  # work_id_int % 256 → uniform file sizes per tier

_EXPORT_TABLES: list[tuple[str, str, str]] = [
    # (ch_table, pg_table, columns)
    ("merged_citations", "work_citations", "citing_id, cited_id"),
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
    ("id_crosswalk_export", "id_crosswalk", "work_id, pmid, pmc_id"),
    ("pm_mesh_tree", "mesh_tree", "descriptor_ui, descriptor_name, tree_number"),
    # mesh_lookup entry terms: CH mesh_lookup_export adds source/has_tree columns.
    # Historical descriptors (has_tree=false) added in post-load step.
    (
        "mesh_lookup_export",
        "mesh_lookup",
        "descriptor_ui, descriptor_name, term, source, has_tree",
    ),
    (
        "work_mesh_export",
        "work_mesh",
        "work_id, descriptor_ui, descriptor_name, qualifier_ui, "
        "qualifier_name, is_major_topic",
    ),
    ("cited_by_clin_export", "cited_by_clin", "pmid, citing_pmid"),
    ("oa_counts_by_year", "work_counts_by_year", "work_id, year, cited_by_count"),
]


def _ch_export_one(
    ch_table: str, columns: str, out_path: Path, *, where: str | None = None
) -> subprocess.CompletedProcess:
    """Export one CH table to Parquet file via subprocess."""
    query = f"SELECT {columns} FROM {ch_table}"
    if where:
        query += f" WHERE {where}"
    cmd = _ch_client_cmd() + ["--query", query + " FORMAT Parquet"]
    with open(out_path, "wb") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)


def _split_parquet_by_bucket(src: Path, dest_dir: Path) -> int:
    """Split single parquet into _BUCKETS files by work_id_int % _BUCKETS.

    Streams record batches → bounded memory. Per-bucket rows are buffered
    and flushed as row groups of ch_export_row_group_size so downstream scanners
    (embeddings) get batches matching their requested batch_size.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pa_ds
    import pyarrow.parquet as pq

    writers: dict[int, pq.ParquetWriter] = {}
    buffers: dict[int, list[pa.Table]] = {}
    buf_lens: dict[int, int] = {}
    total = 0
    mask = pa.scalar(_BUCKETS - 1, type=pa.uint64())

    def _flush(bval: int) -> None:
        if bval not in buffers or not buffers[bval]:
            return
        merged = pa.concat_tables(buffers[bval])
        writers[bval].write_table(merged)
        buffers[bval].clear()
        buf_lens[bval] = 0

    try:
        for batch in pa_ds.dataset(src, format="parquet").scanner().to_batches():
            tbl = pa.Table.from_batches([batch])
            total += len(tbl)
            bucket_ids = pc.bit_wise_and(tbl.column("work_id_int"), mask)
            for bval in pc.unique(bucket_ids).to_pylist():
                subset = tbl.filter(pc.equal(bucket_ids, bval))
                if bval not in writers:
                    writers[bval] = pq.ParquetWriter(
                        str(dest_dir / f"{bval:03d}.parquet"), tbl.schema
                    )
                    buffers[bval] = []
                    buf_lens[bval] = 0
                buffers[bval].append(subset)
                buf_lens[bval] += len(subset)
                if buf_lens[bval] >= settings.ch_export_row_group_size:
                    _flush(bval)
    finally:
        for bval in list(buffers):
            _flush(bval)
        for w in writers.values():
            w.close()
    return total


@asset(
    group_name="export",
    deps=[ch_transform],
    description="CH → Parquet (works: tier×256 hive-bucketed + 12 flat tables).",
    kinds={"clickhouse", "parquet"},
)
def parquet_export(context: AssetExecutionContext) -> MaterializeResult:
    out_dir = Path(settings.parquet_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    total = len(_EXPORT_TABLES) + len(_TIERS)

    # --- works: CH export per tier (1 full scan each) → PyArrow bucket split ---
    def _export_and_split_tier(tier: str) -> str:
        temp_path = out_dir / f"_tmp_works_{tier}.parquet"
        context.log.info(f"[Parquet] exporting works tier={tier} → temp")
        result = _ch_export_one(
            _WORKS_CH_TABLE, _WORKS_EXPORT_COLUMNS, temp_path, where=f"tier = '{tier}'"
        )
        if result.returncode != 0:
            context.log.error(
                f"[Parquet] works tier={tier} failed: {result.stderr.strip()}"
            )
            raise RuntimeError(
                f"Parquet export failed for works tier={tier}: {result.stderr}"
            )
        tier_dir = out_dir / "works" / f"tier={tier}"
        tier_dir.mkdir(parents=True, exist_ok=True)
        try:
            rows = _split_parquet_by_bucket(temp_path, tier_dir)
        finally:
            temp_path.unlink(missing_ok=True)
        context.log.info(
            f"[Parquet] works tier={tier}: {rows:,} rows → {_BUCKETS} buckets"
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

    # works tiers: sequential to bound memory (each tier opens 256 ParquetWriters
    # + row-group buffers; running 4 tiers in parallel consumed ~105GB RSS).
    done = 0
    for tier in _TIERS:
        try:
            _export_and_split_tier(tier)
            done += 1
            context.log.info(f"[Parquet] works/tier={tier} done [{done}/{total}]")
        except RuntimeError:
            failed.append(f"works/tier={tier}")

    # flat tables: parallel (single-file export, memory-light)
    with ThreadPoolExecutor(max_workers=settings.ch_export_max_concurrent) as pool:
        futures: dict[object, str] = {}
        for ch, pg, cols in _EXPORT_TABLES:
            futures[pool.submit(_export, ch, pg, cols)] = pg

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


def _ch_to_pg(
    ch_table: str,
    pg_table: str,
    columns: str,
    *,
    where: str | None = None,
) -> None:
    """Stream CH → CSVWithNames → psql COPY. Zero memory overhead.

    clickhouse-client writes CSV to stdout, piped directly into psql COPY.
    No DuckDB, no intermediate files, no named pipes.
    text[] columns (pm_pub_type, pub_type) are stored as '{a,b}' strings
    in CH. PG COPY parses this directly as text[] — no conversion needed.
    """
    query = f"SELECT {columns} FROM {ch_table}"
    if where:
        query += f" WHERE {where}"
    query += " FORMAT CSVWithNames"

    ch_proc = subprocess.Popen(
        _ch_client_cmd() + ["--query", query],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    pg_proc = subprocess.Popen(
        [
            "psql",
            settings.pg_conninfo,
            "-c",
            f"\\COPY {pg_table} ({columns}) FROM STDIN CSV HEADER NULL '\\N'",
        ],
        stdin=ch_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Allow ch_proc to receive SIGPIPE if pg_proc exits early.
    assert ch_proc.stdout is not None
    ch_proc.stdout.close()

    _, pg_stderr = pg_proc.communicate()
    ch_proc.wait()

    if ch_proc.returncode != 0:
        assert ch_proc.stderr is not None
        ch_err = ch_proc.stderr.read().decode()
        raise RuntimeError(f"CH export failed for {ch_table}: {ch_err}")
    if pg_proc.returncode != 0:
        raise RuntimeError(f"psql COPY failed for {pg_table}: {pg_stderr}")


_ALL_PG_TABLES = [
    "works",
    "papers",
    "work_authors",
    "work_topics",
    "work_citations",
    "work_mesh",
    "authors",
    "mesh_headings",
    "grants",
    "chemicals",
    "id_crosswalk",
    "mesh_tree",
    "cited_by_clin",
    "work_counts_by_year",
]


@asset(
    group_name="serve",
    deps=[ch_transform],
    description="CH → PG via CSVWithNames pipe (UNLOGGED + parallel, zero memory).",
    kinds={"clickhouse", "postgres"},
)
def pg_load(context: AssetExecutionContext, pg: PGResource) -> MaterializeResult:
    _psql(
        f"\\i {_SQL_DIR / 'drop_indexes.sql'}", context, label="[PG] dropping indexes"
    )

    # SET UNLOGGED — skip WAL during bulk load
    for t in _ALL_PG_TABLES:
        _psql(f"ALTER TABLE {t} SET UNLOGGED", context, label=f"[PG] UNLOGGED {t}")

    _psql(
        "TRUNCATE papers, authors, mesh_headings, grants, chemicals, "
        "cited_by_clin, works, work_authors, work_topics, work_mesh, "
        "work_citations, work_counts_by_year, id_crosswalk, mesh_tree, "
        "mesh_lookup CASCADE",
        context,
        label="[PG] truncating all tables",
    )

    failed: list[str] = []

    # works: sequential per-tier (same table → PG table lock prevents parallel COPY)
    for tier in _TIERS:
        context.log.info(f"[PG] loading works tier={tier}")
        _ch_to_pg(
            "works_export",
            "works",
            _WORKS_COLUMNS,
            where=f"tier = '{tier}'",
        )
        context.log.info(f"[PG] works tier={tier} done")

    # Remaining flat tables: parallel (independent tables, no lock contention)
    def _load_table(ch_table: str, pg_table: str, columns: str) -> str:
        _ch_to_pg(ch_table, pg_table, columns)
        return pg_table

    total = len(_EXPORT_TABLES)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures: dict[object, str] = {}
        for ch_tbl, pg_tbl, cols in _EXPORT_TABLES:
            futures[pool.submit(_load_table, ch_tbl, pg_tbl, cols)] = pg_tbl

        done = 0
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                done += 1
                context.log.info(f"[PG] {name} done [{done}/{total}]")
            except Exception as exc:
                context.log.error(f"[PG] {name} failed: {exc}")
                failed.append(name)

    if failed:
        raise RuntimeError(f"PG load failed for: {', '.join(failed)}")

    # Restore LOGGED — WAL resumes, safe for production
    for t in _ALL_PG_TABLES:
        _psql(f"ALTER TABLE {t} SET LOGGED", context, label=f"[PG] LOGGED {t}")

    # Boost parallel index build for bulk load session only (via PGOPTIONS).
    _psql(
        f"\\i {_SQL_DIR / 'schema.sql'}",
        context,
        label="[PG] recreating indexes",
        pgoptions="-c maintenance_work_mem=4GB -c max_parallel_maintenance_workers=8",
    )

    _psql("VACUUM ANALYZE works", context, label="[PG] VACUUM ANALYZE works")
    for _, t, _ in _EXPORT_TABLES:
        _psql(f"VACUUM ANALYZE {t}", context, label=f"[PG] VACUUM ANALYZE {t}")

    # mesh_lookup: unified search index for MeSH synonym lookup.
    # Combines two sources:
    #   1. Entry terms from desc*.xml (via CH pm_mesh_terms → PG COPY)
    #      — official NLM synonyms, abbreviations, cross-references
    #   2. Historical descriptors from work_mesh — descriptors whose
    #      tree_numbers were removed in MeSH revisions but still appear
    #      in paper annotations (e.g., D011499 Post-Translational, 17K papers)
    # Replaces the previous mesh_descriptors table.
    # Note: step 1 (entry terms) is loaded via _EXPORT_TABLES COPY above.
    # Step 2 (historical) requires work_mesh to be loaded first.
    _psql(
        "INSERT INTO mesh_lookup (descriptor_ui, descriptor_name, term, source, has_tree) "
        "SELECT DISTINCT wm.descriptor_ui, wm.descriptor_name, wm.descriptor_name, "
        "'historical', false FROM work_mesh wm "
        "WHERE NOT EXISTS (SELECT 1 FROM mesh_lookup ml WHERE ml.descriptor_ui = wm.descriptor_ui)",
        context,
        label="[PG] adding historical descriptors to mesh_lookup",
    )
    _psql(
        "VACUUM ANALYZE mesh_lookup",
        context,
        label="[PG] VACUUM ANALYZE mesh_lookup",
    )

    # REINDEX after bulk COPY to eliminate index bloat from batch insertions.
    # Safe without CONCURRENTLY since no other sessions during bulk load.
    _psql("REINDEX SCHEMA public", context, label="[PG] REINDEX")

    # Report row counts for key tables via PGResource
    counts = {}
    for table in ("works", "papers", "work_citations"):
        rows = pg.store.query(f"SELECT count(*) AS n FROM {table}")[0]["n"]
        counts[f"{table}_rows"] = MetadataValue.int(rows)
        context.log.info(f"[PG] {table}: {rows:,} rows")

    return MaterializeResult(metadata=counts)
