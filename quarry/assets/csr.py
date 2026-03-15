"""Dagster asset: build CSR mmap graph from ClickHouse edges table."""

import dagster as dg

from quarry.etl.csr import build_and_save


@dg.asset(
    deps=["ch_edges"],
    description="Build forward + reverse CSR mmap files from ClickHouse edges table.",
    kinds={"python"},
)
def csr_graph(context: dg.AssetExecutionContext) -> dg.Output[dict]:
    context.log.info("Building CSR from quarry.edges")

    meta = build_and_save()

    context.log.info(
        f"CSR built: {meta['num_nodes']:,} nodes, {meta['num_edges']:,} edges"
    )

    return dg.Output(meta, metadata=meta)
