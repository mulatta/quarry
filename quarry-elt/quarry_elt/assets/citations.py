"""Citation graph (CSR mmap) Dagster asset.

CSR build uses Parquet → DuckDB CSV export → Rust quarry_graph.
DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from pathlib import Path

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry_elt.assets.load import parquet_export
from quarry.config import settings


@asset(
    group_name="citations",
    deps=[parquet_export],
    description="Build CSR mmap citation graph directly from Parquet (no intermediate CSV).",
    kinds={"parquet", "rust"},
)
def csr_graph(
    context: AssetExecutionContext,
) -> MaterializeResult:
    import quarry_graph

    settings.csr_dir.mkdir(parents=True, exist_ok=True)

    pq_path = Path(settings.parquet_dir) / "work_citations.parquet"
    context.log.info(f"[CSR] building CSR from {pq_path}")
    stats = quarry_graph.build_from_parquet(str(pq_path), str(settings.csr_dir))

    return MaterializeResult(
        metadata={
            "num_nodes": MetadataValue.int(stats.get("num_nodes", 0)),
            "num_edges": MetadataValue.int(stats.get("num_edges", 0)),
            "csr_dir": MetadataValue.path(str(settings.csr_dir)),
        }
    )
