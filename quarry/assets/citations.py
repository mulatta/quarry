"""Citation graph (CSR mmap) Dagster asset.

CSR build does not use DuckDB — reads directly from iCite CSV.
DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.download import icite_occ_sync
from quarry.config import settings
from quarry.etl.icite import build_csr_from_csv


@asset(
    group_name="citations",
    deps=[icite_occ_sync],
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
