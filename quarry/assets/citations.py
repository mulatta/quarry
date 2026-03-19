"""Citation graph (CSR mmap) Dagster asset.

CSR build uses OpenAlex work_citations (via DuckDB) → CSV → Rust quarry_graph.
DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.load import oa_snapshot_load
from quarry.config import settings
from quarry.resources import DuckDBResource


@asset(
    group_name="citations",
    deps=[oa_snapshot_load],
    description="Build CSR mmap citation graph from OA work_citations (i64 node IDs).",
    kinds={"python", "rust"},
)
def csr_graph(
    context: AssetExecutionContext,
    duckdb: DuckDBResource,
) -> MaterializeResult:
    import quarry_graph

    db = duckdb.store
    conn = db.conn
    csv_path = settings.csr_dir / "edges.csv"
    settings.csr_dir.mkdir(parents=True, exist_ok=True)

    # Export edges as CSV: work_id_int (i64) pairs
    context.log.info(f"Exporting citation edges → {csv_path}")
    conn.execute(f"""
        COPY (
            SELECT citing.work_id_int AS src, cited.work_id_int AS dst
            FROM work_citations wc
            JOIN works citing ON wc.citing_work_id = citing.work_id
            JOIN works cited ON wc.cited_work_id = cited.work_id
        ) TO '{csv_path}' (HEADER)
    """)

    context.log.info(f"Building CSR from {csv_path} → {settings.csr_dir}")
    stats = quarry_graph.build_from_csv(str(csv_path), str(settings.csr_dir))

    return MaterializeResult(
        metadata={
            "num_nodes": MetadataValue.int(stats.get("num_nodes", 0)),
            "num_edges": MetadataValue.int(stats.get("num_edges", 0)),
            "csr_dir": MetadataValue.path(str(settings.csr_dir)),
        }
    )
