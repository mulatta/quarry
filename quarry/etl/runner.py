"""ETL runner: parse PubMed XML (via Rust quarry_parse) → insert into DuckDB.

Uses quarry_parse for XML parsing (quick-xml + rayon parallel).
Arrow RecordBatches are registered directly with DuckDB for zero-copy bulk insert.
"""

import time
from pathlib import Path

import quarry_parse

from quarry.config import settings
from quarry.store.duckdb import DuckDBStore

# Column order matching Rust Arrow schema (arrow.rs)
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

# Date columns that come as Utf8 from Rust and need CAST to DATE for DuckDB
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


def _flush_result(db: DuckDBStore, result: dict):
    """Insert Arrow RecordBatches from Rust parser into DuckDB."""
    conn = db.conn
    conn.execute("BEGIN TRANSACTION")
    try:
        # Papers: DELETE + INSERT with date casting
        papers = result["papers"]
        if papers.num_rows > 0:
            conn.register("_tmp_papers", papers)
            conn.execute(
                "DELETE FROM papers WHERE pmid IN (SELECT pmid FROM _tmp_papers)"
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
                f"SELECT {cols_select} FROM _tmp_papers"
            )
            conn.unregister("_tmp_papers")

        # Child tables: DELETE by PMID set + INSERT
        for table_name, columns in _CHILD_TABLES.items():
            rb = result[table_name]
            if rb.num_rows > 0:
                tmp = f"_tmp_{table_name}"
                conn.register(tmp, rb)
                conn.execute(
                    f"DELETE FROM {table_name} "
                    f"WHERE pmid IN (SELECT DISTINCT pmid FROM {tmp})"
                )
                cols = ", ".join(columns)
                conn.execute(
                    f"INSERT INTO {table_name} ({cols}) SELECT {cols} FROM {tmp}"
                )
                conn.unregister(tmp)

        # Soft delete
        db.soft_delete(result["delete_pmids"])
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise


def _stats_to_counts(stats: dict) -> dict[str, int]:
    return {
        "papers": stats["num_papers"],
        "authors": stats["num_authors"],
        "mesh": stats["num_mesh"],
        "grants": stats["num_grants"],
        "chemicals": stats["num_chemicals"],
        "deletes": stats["num_deletes"],
    }


def load_file(db: DuckDBStore, path: Path) -> dict[str, int]:
    """Parse a single PubMed XML file and load into DuckDB."""
    result = quarry_parse.parse_pubmed_file(str(path))
    _flush_result(db, result)
    return _stats_to_counts(result["stats"])


def load_files(db: DuckDBStore, paths: list[Path]) -> dict[str, int]:
    """Parse multiple PubMed XML files in parallel (rayon) and load into DuckDB."""
    result = quarry_parse.parse_pubmed_files([str(p) for p in paths])
    _flush_result(db, result)
    return _stats_to_counts(result["stats"])


def load_baseline(
    db: DuckDBStore,
    baseline_dir: Path | None = None,
    single_file: str | None = None,
    chunk_size: int = 20,
):
    """Load PubMed baseline XML files into DuckDB.

    Files are processed in parallel chunks (rayon) with bounded memory.
    chunk_size controls how many files are parsed in parallel per batch.
    """
    baseline_dir = baseline_dir or settings.pubmed_baseline_dir

    if single_file:
        files = [baseline_dir / single_file]
    else:
        files = sorted(baseline_dir.glob("pubmed*.xml.gz"))

    if not files:
        print(f"No baseline files found in {baseline_dir}")
        return

    print(f"Baseline: {len(files)} files, chunk_size={chunk_size}")
    grand_total = {
        "papers": 0,
        "authors": 0,
        "mesh": 0,
        "grants": 0,
        "chemicals": 0,
        "deletes": 0,
    }
    t0 = time.time()

    for i in range(0, len(files), chunk_size):
        chunk = files[i : i + chunk_size]
        ct0 = time.time()
        counts = load_files(db, chunk)
        elapsed = time.time() - ct0

        for k in grand_total:
            grand_total[k] += counts[k]

        done = min(i + chunk_size, len(files))
        rate = counts["papers"] / elapsed if elapsed > 0 else 0
        print(
            f"  [{done}/{len(files)}] papers={counts['papers']}, "
            f"mesh={counts['mesh']}, deletes={counts['deletes']} "
            f"({rate:.0f} papers/s, {elapsed:.1f}s)"
        )

    total_elapsed = time.time() - t0
    print(f"\nBaseline done in {total_elapsed:.0f}s: {grand_total}")


def load_updates(
    db: DuckDBStore,
    update_dir: Path | None = None,
):
    """Load PubMed daily update XML files."""
    update_dir = update_dir or settings.pubmed_update_dir
    files = sorted(update_dir.glob("pubmed*.xml.gz"))

    if not files:
        print(f"No update files found in {update_dir}")
        return

    print(f"Updates: {len(files)} files")
    t0 = time.time()
    total_papers = 0
    total_deletes = 0

    for i, f in enumerate(files, 1):
        counts = load_file(db, f)
        total_papers += counts["papers"]
        total_deletes += counts["deletes"]
        print(
            f"  [{i}/{len(files)}] {f.name}: "
            f"papers={counts['papers']}, deletes={counts['deletes']}"
        )

    elapsed = time.time() - t0
    print(
        f"\nUpdates done in {elapsed:.0f}s: "
        f"papers={total_papers}, deletes={total_deletes}"
    )
    return {"papers": total_papers, "deletes": total_deletes}
