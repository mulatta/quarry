"""DuckDB bulk load asset: reads all Arrow staging → single sequential DB write.

This is the ONLY asset that writes to DuckDB, eliminating lock contention.
DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.stage import (
    biorxiv_stage,
    icite_metrics_stage,
    mesh_stage,
    pubmed_baseline_stage,
    pubmed_updates_stage,
)
from quarry.config import settings
from quarry.etl.staging import clear, iter_chunks, read_tables
from quarry.resources import DuckDBResource

# Column order matching Rust Arrow schema (runner.py)
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


@asset(
    group_name="load",
    deps=[
        pubmed_baseline_stage,
        pubmed_updates_stage,
        mesh_stage,
        biorxiv_stage,
        icite_metrics_stage,
    ],
    description="Bulk load all Arrow staging files into DuckDB (single writer).",
    kinds={"duckdb"},
)
def duckdb_load(
    context: AssetExecutionContext,
    duckdb: DuckDBResource,
) -> MaterializeResult:
    db = duckdb.store
    conn = db.conn
    staging = settings.staging_dir
    stats: dict[str, int] = {}

    # 1. PubMed baseline
    n = _load_pubmed_staging(conn, staging / "pubmed_baseline", context)
    stats["baseline_papers"] = n
    context.log.info(f"Baseline: {n} papers loaded")

    # 2. PubMed updates
    n = _load_pubmed_staging(conn, staging / "pubmed_updates", context)
    stats["update_papers"] = n
    context.log.info(f"Updates: {n} papers loaded")

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
        context.log.info(f"MeSH: {t.num_rows} entries loaded")

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
        context.log.info(f"bioRxiv: {t.num_rows} preprints loaded")

    # 5. iCite metrics
    n = _load_icite_staging(conn, staging / "icite_metrics", context)
    stats["icite_rows"] = n
    context.log.info(f"iCite: {n} rows updated")

    # Clear staging after successful load
    for sub in (
        "pubmed_baseline",
        "pubmed_updates",
        "mesh",
        "biorxiv",
        "icite_metrics",
    ):
        clear(staging / sub)

    return MaterializeResult(
        metadata={k: MetadataValue.int(v) for k, v in stats.items()}
    )


def _load_pubmed_staging(conn, staging_dir, context) -> int:
    """Load pubmed staging chunks into DuckDB."""
    total = 0
    for chunk in iter_chunks(staging_dir):
        conn.execute("BEGIN TRANSACTION")
        try:
            # Papers
            papers = chunk.get("papers")
            if papers is not None and papers.num_rows > 0:
                conn.register("_stg_papers", papers)
                conn.execute(
                    "DELETE FROM papers WHERE pmid IN (SELECT pmid FROM _stg_papers)"
                )
                cols_insert = ", ".join(_PAPERS_COLS)
                cols_select = ", ".join(
                    f"TRY_CAST({c} AS DATE)"
                    if c in _DATE_COLS
                    else f"COALESCE({c}, '')"
                    if c == "title"
                    else c
                    for c in _PAPERS_COLS
                )
                conn.execute(
                    f"INSERT INTO papers ({cols_insert}) "
                    f"SELECT {cols_select} FROM _stg_papers"
                )
                conn.unregister("_stg_papers")
                total += papers.num_rows

            # Child tables
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

            # Soft deletes
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
    return total


def _load_icite_staging(conn, staging_dir, context) -> int:
    """Load iCite metrics staging into DuckDB (bulk UPDATE)."""
    total = 0
    for chunk in iter_chunks(staging_dir):
        t = chunk.get("metrics")
        if t is None or t.num_rows == 0:
            continue
        conn.register("_stg_metrics", t)
        conn.execute("""
            UPDATE papers SET
                rcr = m.rcr, nih_percentile = m.nih_percentile, apt = m.apt,
                is_clinical = m.is_clinical, human = m.human, animal = m.animal,
                molecular_cellular = m.molecular_cellular,
                cited_by_clin = m.cited_by_clin,
                field_citation_rate = m.field_citation_rate
            FROM _stg_metrics m
            WHERE papers.pmid = m.pmid
        """)
        conn.unregister("_stg_metrics")
        total += t.num_rows
    return total
