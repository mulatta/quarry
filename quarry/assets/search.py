"""Embedding + LanceDB index Dagster assets.

Wraps quarry.etl.embeddings logic.
DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.citations import csr_graph
from quarry.assets.load import parquet_export
from quarry.config import settings
from quarry.etl.embeddings import run as run_embeddings


@asset(
    group_name="search",
    deps=[parquet_export, csr_graph],
    description="Encode works (blake3 change detection) → LanceDB vectors + FTS index.",
    kinds={"lancedb", "python", "gpu"},
)
def paper_embeddings(
    context: AssetExecutionContext,
) -> MaterializeResult:
    context.log.info("[Embed] running pipeline (blake3 cache + jina-v5)")
    run_embeddings()
    return MaterializeResult(
        metadata={
            "lancedb_uri": MetadataValue.text(settings.lancedb_uri),
        }
    )
