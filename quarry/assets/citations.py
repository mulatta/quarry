"""Citation graph (CSR mmap) Dagster asset.

CSR build uses CH work_citations → CSV pipe → Rust quarry_graph.
DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

import subprocess

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.load import ch_transform
from quarry.config import settings


@asset(
    group_name="citations",
    deps=[ch_transform],
    description="Build CSR mmap citation graph from CH work_citations → CSV → Rust.",
    kinds={"python", "rust"},
)
def csr_graph(
    context: AssetExecutionContext,
) -> MaterializeResult:
    import quarry_graph

    csv_path = settings.csr_dir / "edges.csv"
    settings.csr_dir.mkdir(parents=True, exist_ok=True)

    # CH → CSV pipe (column aliases for quarry_graph: src, dst)
    context.log.info(f"[CSR] exporting edges from CH → {csv_path}")
    ch_cmd = [
        "clickhouse-client",
        "--host",
        settings.ch_host,
        "--port",
        str(settings.ch_port),
        "--database",
        settings.ch_database,
        "--query",
        "SELECT citing_id AS src, cited_id AS dst FROM oa_work_citations",
        "--format",
        "CSVWithNames",
    ]
    with open(csv_path, "w") as f:
        proc = subprocess.run(ch_cmd, stdout=f, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"CH export failed: {proc.stderr.strip()}")

    context.log.info(f"[CSR] building CSR from {csv_path}")
    stats = quarry_graph.build_from_csv(str(csv_path), str(settings.csr_dir))

    return MaterializeResult(
        metadata={
            "num_nodes": MetadataValue.int(stats.get("num_nodes", 0)),
            "num_edges": MetadataValue.int(stats.get("num_edges", 0)),
            "csr_dir": MetadataValue.path(str(settings.csr_dir)),
        }
    )
