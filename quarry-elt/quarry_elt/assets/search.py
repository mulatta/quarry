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

from quarry_elt.assets.load import ch_transform
from quarry.config import settings
from quarry.etl.embeddings import run as run_embeddings


@asset(
    group_name="search",
    deps=[ch_transform],
    description="CH → LanceDB: two-phase incremental encode (BLAKE3 diff + GPU).",
    kinds={"clickhouse", "lancedb", "gpu"},
)
def paper_embeddings(
    context: AssetExecutionContext,
) -> MaterializeResult:
    context.log.info("[Embed] running pipeline (CH BLAKE3 diff + jina-v5)")
    run_embeddings(logger=context.log)
    return MaterializeResult(
        metadata={
            "lancedb_uri": MetadataValue.text(settings.lancedb_uri),
        }
    )
