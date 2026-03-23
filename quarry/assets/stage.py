"""Staging assets: MeSH parse → PG direct load via quarry-ingest.

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from dagster import (
    AssetExecutionContext,
    AutomationCondition,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.download import mesh_descriptor_sync
from quarry.assets.load import _run_ingest
from quarry.config import settings


@asset(
    group_name="supplementary",
    deps=[mesh_descriptor_sync],
    description="Parse MeSH descriptor XML → PG mesh_tree table via quarry-ingest.",
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

    stats = _run_ingest(["load", "mesh", "--xml-path", str(xml_path)], context)
    return MaterializeResult(
        metadata={
            k: MetadataValue.int(v) if isinstance(v, int) else MetadataValue.float(v)
            for k, v in stats.items()
        },
    )
