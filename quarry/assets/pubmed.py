"""PubMed baseline + daily update Dagster assets.

Uses quarry_parse (Rust) for XML parsing via quarry.etl.runner.
DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.config import settings
from quarry.etl.runner import load_file, load_files
from quarry.resources import DuckDBResource


@asset(
    group_name="pubmed",
    description="Load PubMed baseline XML files into DuckDB (one-time full load).",
    kinds={"duckdb", "rust"},
)
def pubmed_baseline(
    context: AssetExecutionContext,
    duckdb: DuckDBResource,
) -> MaterializeResult:
    db = duckdb.store
    baseline_dir = settings.pubmed_baseline_dir
    files = sorted(baseline_dir.glob("pubmed*.xml.gz"))
    context.log.info(f"Baseline: {len(files)} files in {baseline_dir}")

    grand_total = {
        "papers": 0,
        "authors": 0,
        "mesh": 0,
        "grants": 0,
        "chemicals": 0,
        "deletes": 0,
    }
    chunk_size = 20

    for i in range(0, len(files), chunk_size):
        chunk = files[i : i + chunk_size]
        counts = load_files(db, chunk)
        for k in grand_total:
            grand_total[k] += counts[k]
        done = min(i + chunk_size, len(files))
        context.log.info(
            f"[{done}/{len(files)}] papers={counts['papers']}, mesh={counts['mesh']}"
        )

    return MaterializeResult(
        metadata={
            "num_files": MetadataValue.int(len(files)),
            "papers": MetadataValue.int(grand_total["papers"]),
            "authors": MetadataValue.int(grand_total["authors"]),
            "mesh": MetadataValue.int(grand_total["mesh"]),
            "deletes": MetadataValue.int(grand_total["deletes"]),
        }
    )


@asset(
    group_name="pubmed",
    deps=[pubmed_baseline],
    description="Process PubMed daily update XML files (incremental upsert + deletes).",
    kinds={"duckdb", "rust"},
)
def pubmed_daily_update(
    context: AssetExecutionContext,
    duckdb: DuckDBResource,
) -> MaterializeResult:
    db = duckdb.store
    update_dir = settings.pubmed_update_dir
    files = sorted(update_dir.glob("pubmed*.xml.gz"))

    if not files:
        context.log.info(f"No update files in {update_dir}")
        return MaterializeResult(
            metadata={
                "num_files": MetadataValue.int(0),
                "papers": MetadataValue.int(0),
                "deletes": MetadataValue.int(0),
            }
        )

    total_papers = 0
    total_deletes = 0

    for i, f in enumerate(files, 1):
        counts = load_file(db, f)
        total_papers += counts["papers"]
        total_deletes += counts["deletes"]
        context.log.info(
            f"[{i}/{len(files)}] {f.name}: "
            f"papers={counts['papers']}, deletes={counts['deletes']}"
        )

    return MaterializeResult(
        metadata={
            "num_files": MetadataValue.int(len(files)),
            "papers": MetadataValue.int(total_papers),
            "deletes": MetadataValue.int(total_deletes),
        }
    )
