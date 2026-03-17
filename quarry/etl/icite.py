"""iCite Open Citation Collection → CSR mmap build + papers metrics UPDATE.

Two data files from figshare:
  1. open_citation_collection.csv (citing_pmid, cited_pmid) → CSR mmap
  2. icite_metadata.csv → papers table bulk UPDATE (rcr, apt, etc.)

Usage:
    python -m quarry.etl.icite citations   # build CSR from OCC CSV
    python -m quarry.etl.icite metrics      # update papers with iCite metrics
    python -m quarry.etl.icite all          # both
"""

import argparse
import csv
import time
from pathlib import Path

import pyarrow as pa

from quarry.config import settings
from quarry.store.duckdb import DuckDBStore


def build_csr_from_csv(
    csv_path: Path,
    csr_dir: Path | None = None,
) -> dict:
    """Build CSR mmap files from iCite open_citation_collection.csv.

    Delegates to quarry_csr Rust extension (rayon parallel, ~31s for 805M edges).
    """
    import quarry_csr

    csr_dir = csr_dir or settings.csr_dir
    return quarry_csr.build_csr_from_csv(str(csv_path), str(csr_dir))


def update_metrics(
    csv_path: Path,
    db: DuckDBStore | None = None,
    batch_size: int = 50_000,
):
    """Bulk UPDATE papers table with iCite metrics from icite_metadata.csv.

    CSV columns used: pmid, relative_citation_ratio, nih_percentile,
    apt, is_clinical, human, animal, molecular_cellular,
    cited_by_clin, field_citation_rate
    """
    close_db = False
    if db is None:
        db = DuckDBStore()
        close_db = True

    print(f"Updating papers metrics from {csv_path} ...")
    t0 = time.time()
    total = 0

    csv.field_size_limit(2**30)  # iCite metadata has very long fields (cited_by lists)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        batch: list[tuple] = []

        for row in reader:
            pmid_str = row.get("pmid", "")
            if not pmid_str or not pmid_str.isdigit():
                continue

            batch.append(
                (
                    _float_or_none(row.get("relative_citation_ratio")),
                    _float_or_none(row.get("nih_percentile")),
                    _float_or_none(row.get("apt")),
                    row.get("is_clinical", "").lower() == "true",
                    _float_or_none(row.get("human")),
                    _float_or_none(row.get("animal")),
                    _float_or_none(row.get("molecular_cellular")),
                    _int_or_none(row.get("cited_by_clin")),
                    _float_or_none(row.get("field_citation_rate")),
                    int(pmid_str),
                )
            )

            if len(batch) >= batch_size:
                _flush_metrics(db, batch)
                total += len(batch)
                batch = []
                elapsed = time.time() - t0
                print(f"  {total:,} rows updated ({total / elapsed:.0f} rows/s)")

        if batch:
            _flush_metrics(db, batch)
            total += len(batch)

    elapsed = time.time() - t0
    print(f"  Metrics update done: {total:,} rows in {elapsed:.0f}s")

    if close_db:
        db.close()


def _flush_metrics(db: DuckDBStore, batch: list[tuple]):
    """Bulk UPDATE via pyarrow table registration."""
    arrow_table = pa.table(
        {
            "rcr": pa.array([r[0] for r in batch], type=pa.float32()),
            "nih_percentile": pa.array([r[1] for r in batch], type=pa.float32()),
            "apt": pa.array([r[2] for r in batch], type=pa.float32()),
            "is_clinical": pa.array([r[3] for r in batch], type=pa.bool_()),
            "human": pa.array([r[4] for r in batch], type=pa.float32()),
            "animal": pa.array([r[5] for r in batch], type=pa.float32()),
            "molecular_cellular": pa.array([r[6] for r in batch], type=pa.float32()),
            "cited_by_clin": pa.array([r[7] for r in batch], type=pa.int32()),
            "field_citation_rate": pa.array([r[8] for r in batch], type=pa.float32()),
            "pmid": pa.array([r[9] for r in batch], type=pa.int32()),
        }
    )
    db.conn.register("_tmp_metrics", arrow_table)
    db.conn.execute("""
        UPDATE papers SET
            rcr = m.rcr, nih_percentile = m.nih_percentile, apt = m.apt,
            is_clinical = m.is_clinical, human = m.human, animal = m.animal,
            molecular_cellular = m.molecular_cellular, cited_by_clin = m.cited_by_clin,
            field_citation_rate = m.field_citation_rate
        FROM _tmp_metrics m
        WHERE papers.pmid = m.pmid
    """)
    db.conn.unregister("_tmp_metrics")


def _float_or_none(s: str | None) -> float | None:
    if not s or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _int_or_none(s: str | None) -> int | None:
    if not s or s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="iCite OCC → CSR + DuckDB metrics")
    sub = parser.add_subparsers(dest="command", required=True)

    cit = sub.add_parser("citations", help="Build CSR from OCC CSV")
    cit.add_argument(
        "--csv", type=Path, required=True, help="open_citation_collection.csv path"
    )
    cit.add_argument("--csr-dir", type=Path, default=None)

    met = sub.add_parser("metrics", help="Update papers with iCite metadata")
    met.add_argument("--csv", type=Path, required=True, help="icite_metadata.csv path")

    both = sub.add_parser("all", help="Build CSR + update metrics")
    both.add_argument(
        "--occ-csv", type=Path, required=True, help="open_citation_collection.csv"
    )
    both.add_argument("--meta-csv", type=Path, required=True, help="icite_metadata.csv")
    both.add_argument("--csr-dir", type=Path, default=None)

    args = parser.parse_args()

    if args.command == "citations":
        build_csr_from_csv(args.csv, args.csr_dir)
    elif args.command == "metrics":
        update_metrics(args.csv)
    elif args.command == "all":
        build_csr_from_csv(args.occ_csv, args.csr_dir)
        update_metrics(args.meta_csv)


if __name__ == "__main__":
    main()
