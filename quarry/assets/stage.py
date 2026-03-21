"""Staging assets: MeSH parse → PG direct load via Rust PyO3.

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

import quarry_rs
from dagster import (
    AssetExecutionContext,
    AutomationCondition,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.download import mesh_descriptor_sync
from quarry.config import settings


@asset(
    group_name="supplementary",
    deps=[mesh_descriptor_sync],
    description="Parse MeSH descriptor XML → PG mesh_tree table via Rust PyO3.",
    kinds={"rust", "postgres"},
    automation_condition=AutomationCondition.eager(),
)
def mesh_stage(context: AssetExecutionContext) -> MaterializeResult:
    xml_files = sorted(settings.pubmed_mesh_dir.glob("desc*.xml"))
    if not xml_files:
        context.log.warning("No MeSH descriptor XML found")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    xml_path = xml_files[-1]
    context.log.info(f"Parsing MeSH from {xml_path}")

    rows = quarry_rs.mesh_stage_pg(
        pg_conninfo=settings.pg_conninfo,
        xml_path=str(xml_path),
    )

    return MaterializeResult(metadata={"tree_entries": MetadataValue.int(rows)})
