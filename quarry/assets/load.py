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

from quarry.assets.download import icite_metadata_sync, pubmed_baseline_sync
from quarry.assets.stage import (
    biorxiv_stage,
    mesh_stage,
    pubmed_updates_stage,
)
from quarry.config import settings
from quarry.etl.staging import iter_chunks, read_tables
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
        pubmed_updates_stage,
        mesh_stage,
        biorxiv_stage,
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

    # 2. PubMed updates (from Feather staging — incremental)
    n = _load_updates(conn, staging / "pubmed_updates", context)
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

    # 4. bioRxiv preprints
    tables = read_tables(staging / "biorxiv")
    if "preprints" in tables:
        t = tables["preprints"]
        conn.register("_stg_preprints", t)
        conn.execute(
            "DELETE FROM preprints WHERE doi IN (SELECT doi FROM _stg_preprints)"
        )
        conn.execute(
            "INSERT INTO preprints (doi, title, abstract, date, server, category, "
            "version, published_doi) "
            "SELECT doi, title, abstract, TRY_CAST(date AS DATE), server, category, "
            "version, published_doi FROM _stg_preprints"
        )
        conn.unregister("_stg_preprints")
        stats["preprints"] = t.num_rows
        context.log.info(f"bioRxiv: {t.num_rows:,} preprints loaded")

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


def _load_updates(conn, staging_dir: Path, context) -> int:
    """Incremental: INSERT OR REPLACE for papers, DELETE+INSERT for child tables."""
    chunks = list(iter_chunks(staging_dir))
    if not chunks:
        return 0
    n_chunks = len(chunks)
    context.log.info(f"Updates: loading {n_chunks} chunks")

    total = 0
    for i, chunk in enumerate(chunks):
        conn.execute("BEGIN TRANSACTION")
        try:
            papers = chunk.get("papers")
            if papers is not None and papers.num_rows > 0:
                conn.register("_stg_papers", papers)
                conn.execute(
                    "DELETE FROM papers WHERE pmid IN (SELECT pmid FROM _stg_papers)"
                )
                conn.execute(
                    f"INSERT OR REPLACE INTO papers ({_COLS_INSERT}) "
                    f"SELECT {_COLS_SELECT} FROM _stg_papers"
                )
                conn.unregister("_stg_papers")
                total += papers.num_rows

            for table_name, columns in _CHILD_TABLES.items():
                t = chunk.get(table_name)
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

            del_t = chunk.get("delete_pmids")
            if del_t is not None and del_t.num_rows > 0:
                conn.register("_stg_del", del_t)
                conn.execute(
                    "UPDATE papers SET is_deleted = TRUE, deleted_date = CURRENT_DATE "
                    "WHERE pmid IN (SELECT pmid FROM _stg_del)"
                )
                conn.unregister("_stg_del")

            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
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
            cited_by_clin = m.cited_by_clin,
            field_citation_rate = m.field_citation_rate
        FROM read_csv('{csv_path}',
             auto_detect=true, ignore_errors=true, parallel=true) m
        WHERE papers.pmid = m.pmid
    """)
    n = result.fetchone()[0]
    return n
