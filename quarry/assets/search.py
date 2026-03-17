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

from quarry.config import settings
from quarry.etl.embeddings import run as run_embeddings


@asset(
    group_name="search",
    description="Encode papers (blake3 change detection) → LanceDB vectors + FTS index.",
    kinds={"lancedb", "python", "gpu"},
)
def paper_embeddings(
    context: AssetExecutionContext,
) -> MaterializeResult:
    context.log.info("Running embedding pipeline (blake3 cache + jina-v5)")
    # run() uses settings internally for DuckDB + LanceDB paths
    run_embeddings()
    return MaterializeResult(
        metadata={
            "lancedb_uri": MetadataValue.text(settings.lancedb_uri),
        }
    )
