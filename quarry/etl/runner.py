"""ETL runner: parse PubMed XML → insert into DuckDB."""

import time
from pathlib import Path

from quarry.config import settings
from quarry.etl.parse import ParseResult, parse_xml_stream
from quarry.store.duckdb import DuckDBStore


def _flush_batch(db: DuckDBStore, batch: ParseResult):
    """Insert accumulated batch into DuckDB.

    Registers the PMID set once and reuses it across all child-table
    deletes (7.6x faster than per-table register/unregister).
    """
    db.conn.execute("BEGIN TRANSACTION")
    try:
        db.upsert_papers(batch.papers)

        # Register PMID set once for all child-table delete-before-insert
        child_pmids = list(
            {r["pmid"] for r in batch.authors}
            | {r["pmid"] for r in batch.mesh_headings}
            | {r["pmid"] for r in batch.grants}
            | {r["pmid"] for r in batch.chemicals}
        )
        if child_pmids:
            db.register_pmid_set(child_pmids)
            db.insert_authors(batch.authors, pmids_registered=True)
            db.insert_mesh_headings(batch.mesh_headings, pmids_registered=True)
            db.insert_grants(batch.grants, pmids_registered=True)
            db.insert_chemicals(batch.chemicals, pmids_registered=True)
            db.unregister_pmid_set()
        else:
            db.insert_authors(batch.authors)
            db.insert_mesh_headings(batch.mesh_headings)
            db.insert_grants(batch.grants)
            db.insert_chemicals(batch.chemicals)

        db.soft_delete(batch.delete_pmids)
        db.conn.execute("COMMIT")
    except Exception:
        try:
            db.conn.execute("ROLLBACK")
        except Exception:
            pass  # Already rolled back or no active transaction
        raise


def load_file(db: DuckDBStore, path: Path, batch_size: int = 10_000) -> dict[str, int]:
    """Parse a single PubMed XML file and load into DuckDB.

    Returns counts: papers, authors, mesh, grants, chemicals, deletes.
    """
    batch = ParseResult()
    total = {
        "papers": 0,
        "authors": 0,
        "mesh": 0,
        "grants": 0,
        "chemicals": 0,
        "deletes": 0,
    }
    parsed_count = 0

    for result in parse_xml_stream(path):
        batch.extend(result)
        parsed_count += len(result.papers)

        # Progress every 1000 articles
        if parsed_count % 1000 < len(result.papers):
            print(f"    parsed {parsed_count} articles...", flush=True)

        if len(batch.papers) >= batch_size:
            _flush_batch(db, batch)
            total["papers"] += len(batch.papers)
            total["authors"] += len(batch.authors)
            total["mesh"] += len(batch.mesh_headings)
            total["grants"] += len(batch.grants)
            total["chemicals"] += len(batch.chemicals)
            total["deletes"] += len(batch.delete_pmids)
            batch = ParseResult()

    # Flush remaining
    if batch.papers or batch.delete_pmids:
        _flush_batch(db, batch)
        total["papers"] += len(batch.papers)
        total["authors"] += len(batch.authors)
        total["mesh"] += len(batch.mesh_headings)
        total["grants"] += len(batch.grants)
        total["chemicals"] += len(batch.chemicals)
        total["deletes"] += len(batch.delete_pmids)

    return total


def load_baseline(
    db: DuckDBStore,
    baseline_dir: Path | None = None,
    single_file: str | None = None,
    batch_size: int = 10_000,
):
    """Load PubMed baseline XML files into DuckDB."""
    baseline_dir = baseline_dir or settings.pubmed_baseline_dir

    if single_file:
        files = [baseline_dir / single_file]
    else:
        files = sorted(baseline_dir.glob("pubmed*.xml.gz"))

    if not files:
        print(f"No baseline files found in {baseline_dir}")
        return

    print(f"Baseline: {len(files)} files, batch_size={batch_size}")
    grand_total = {
        "papers": 0,
        "authors": 0,
        "mesh": 0,
        "grants": 0,
        "chemicals": 0,
        "deletes": 0,
    }
    t0 = time.time()

    for i, f in enumerate(files, 1):
        ft0 = time.time()
        counts = load_file(db, f, batch_size)
        elapsed = time.time() - ft0

        for k in grand_total:
            grand_total[k] += counts[k]

        rate = counts["papers"] / elapsed if elapsed > 0 else 0
        print(
            f"  [{i}/{len(files)}] {f.name}: "
            f"papers={counts['papers']}, mesh={counts['mesh']}, "
            f"deletes={counts['deletes']} ({rate:.0f} papers/s, {elapsed:.1f}s)"
        )

    total_elapsed = time.time() - t0
    print(f"\nBaseline done in {total_elapsed:.0f}s: {grand_total}")


def load_updates(
    db: DuckDBStore,
    update_dir: Path | None = None,
    batch_size: int = 10_000,
):
    """Load PubMed daily update XML files.

    Processes all .xml.gz in update_dir. Caller should manage
    which files are new (e.g., by tracking last processed file).
    """
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
        counts = load_file(db, f, batch_size)
        total_papers += counts["papers"]
        total_deletes += counts["deletes"]
        print(
            f"  [{i}/{len(files)}] {f.name}: papers={counts['papers']}, deletes={counts['deletes']}"
        )

    elapsed = time.time() - t0
    print(
        f"\nUpdates done in {elapsed:.0f}s: papers={total_papers}, deletes={total_deletes}"
    )

    return {"papers": total_papers, "deletes": total_deletes}
