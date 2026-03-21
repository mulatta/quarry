"""Citation graph (CSR mmap) Dagster asset.

CSR build uses OpenAlex work_citations (via PG) → CSV → Rust quarry_graph.
DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.load import oa_pg_load
from quarry.config import settings
from quarry.resources import PGResource


@asset(
    group_name="citations",
    deps=[oa_pg_load],
    description="Build CSR mmap citation graph from OA work_citations (i64 node IDs).",
    kinds={"python", "rust"},
)
def csr_graph(
    context: AssetExecutionContext,
    pg: PGResource,
) -> MaterializeResult:
    import quarry_graph

    store = pg.store
    conn = store.conn
    csv_path = settings.csr_dir / "edges.csv"
    settings.csr_dir.mkdir(parents=True, exist_ok=True)

    # Export edges as CSV via PG COPY TO
    context.log.info(f"Exporting citation edges → {csv_path}")
    with open(csv_path, "w") as f:
        f.write("src,dst\n")
        with conn.cursor() as cur:
            with cur.copy(
                "COPY (SELECT citing_id AS src, cited_id AS dst FROM work_citations) TO STDOUT WITH (FORMAT csv)"
            ) as copy:
                for data in copy:
                    text = data if isinstance(data, str) else bytes(data).decode()
                    f.write(text)

    context.log.info(f"Building CSR from {csv_path} → {settings.csr_dir}")
    stats = quarry_graph.build_from_csv(str(csv_path), str(settings.csr_dir))

    return MaterializeResult(
        metadata={
            "num_nodes": MetadataValue.int(stats.get("num_nodes", 0)),
            "num_edges": MetadataValue.int(stats.get("num_edges", 0)),
            "csr_dir": MetadataValue.path(str(settings.csr_dir)),
        }
    )
