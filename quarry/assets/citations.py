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

from quarry.assets.load import parquet_export
from quarry.config import settings


@asset(
    group_name="citations",
    deps=[parquet_export],
    description="Build CSR mmap citation graph from Parquet → DuckDB CSV → Rust.",
    kinds={"python", "rust"},
)
def csr_graph(
    context: AssetExecutionContext,
) -> MaterializeResult:
    import duckdb
    import quarry_graph

    csv_path = settings.csr_dir / "edges.csv"
    settings.csr_dir.mkdir(parents=True, exist_ok=True)

    # Parquet → CSV via DuckDB Python binding (no CH dependency)
    pq_path = Path(settings.parquet_dir) / "work_citations.parquet"
    context.log.info(f"[CSR] exporting edges from {pq_path} → {csv_path}")
    duckdb.sql(
        f"COPY (SELECT citing_id AS src, cited_id AS dst "
        f"FROM read_parquet('{pq_path}')) TO '{csv_path}' (HEADER true)"
    )

    context.log.info(f"[CSR] building CSR from {csv_path}")
    stats = quarry_graph.build_from_csv(str(csv_path), str(settings.csr_dir))

    return MaterializeResult(
        metadata={
            "num_nodes": MetadataValue.int(stats.get("num_nodes", 0)),
            "num_edges": MetadataValue.int(stats.get("num_edges", 0)),
            "csr_dir": MetadataValue.path(str(settings.csr_dir)),
        }
    )
