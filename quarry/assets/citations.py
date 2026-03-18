"""Citation graph (CSR mmap) + iCite metrics Dagster assets.

Wraps quarry.etl.icite logic.
DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.config import settings
from quarry.etl.icite import build_csr_from_csv, update_metrics
from quarry.resources import DuckDBResource


@asset(
    group_name="citations",
    description="Build CSR mmap citation graph from iCite OCC CSV (monthly rebuild).",
    kinds={"python", "rust"},
)
def csr_graph(
    context: AssetExecutionContext,
) -> MaterializeResult:
    occ_csv = settings.icite_dir / "open_citation_collection.csv"
    if not occ_csv.exists():
        context.log.warning(f"OCC CSV not found: {occ_csv}")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    context.log.info(f"Building CSR from {occ_csv} → {settings.csr_dir}")
    stats = build_csr_from_csv(occ_csv, settings.csr_dir)

    return MaterializeResult(
        metadata={
            "num_nodes": MetadataValue.int(stats.get("num_nodes", 0)),
            "num_edges": MetadataValue.int(stats.get("num_edges", 0)),
            "csr_dir": MetadataValue.path(str(settings.csr_dir)),
        }
    )


@asset(
    group_name="citations",
    description="Update papers table with iCite metrics (RCR, APT, human/animal scores).",
    kinds={"duckdb", "python"},
)
def icite_metrics(
    context: AssetExecutionContext,
    duckdb: DuckDBResource,
) -> MaterializeResult:
    meta_csv = settings.icite_dir / "icite_metadata.csv"
    if not meta_csv.exists():
        context.log.warning(f"iCite metadata CSV not found: {meta_csv}")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    context.log.info(f"Updating papers metrics from {meta_csv}")
    update_metrics(meta_csv, db=duckdb.store)
    return MaterializeResult(metadata={"csv_path": MetadataValue.path(str(meta_csv))})
