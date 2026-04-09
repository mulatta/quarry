"""Staging assets: MeSH parse → Parquet via quarry-parse.

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
from quarry.assets.helpers import run_parse
from quarry.config import settings


@asset(
    group_name="supplementary",
    deps=[mesh_descriptor_sync],
    description="Parse MeSH descriptor XML → mesh_tree.parquet + mesh_terms.parquet via quarry-parse.",
    kinds={"rust", "parquet"},
    automation_condition=AutomationCondition.eager(),
)
def mesh_stage(context: AssetExecutionContext) -> MaterializeResult:
    xml_files = sorted(settings.pubmed_mesh_dir.glob("desc*.xml"))
    if not xml_files:
        context.log.warning("No MeSH descriptor XML found")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    xml_path = xml_files[-1]
    context.log.info(f"[MeSH] parse: {xml_path.name}")

    run_parse(
        [
            "mesh",
            "--xml-path",
            str(xml_path),
            "--output-dir",
            str(settings.mesh_parquet_dir),
        ],
        context,
    )
    return MaterializeResult(
        metadata={"status": MetadataValue.text("ok")},
    )
