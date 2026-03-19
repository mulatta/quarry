"""Staging assets: parse raw data → Arrow Feather files.

No DuckDB access. All assets here can run in parallel.
Uses AutomationCondition.eager() with data_version_changed() so that
scheduled runs skip re-processing when upstream data hasn't changed.

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

import pyarrow as pa
from dagster import (
    AssetExecutionContext,
    AutomationCondition,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.download import mesh_descriptor_sync
from quarry.config import settings
from quarry.etl.staging import clear, write_tables

_EAGER_ON_VERSION_CHANGE = AutomationCondition.eager().replace(
    "any_deps_updated",
    AutomationCondition.any_deps_updated().replace(
        "newly_updated", AutomationCondition.data_version_changed()
    ),
)


# -- MeSH --------------------------------------------------------------------


@asset(
    group_name="supplementary",
    deps=[mesh_descriptor_sync],
    description="Parse MeSH descriptor XML → Arrow Feather staging.",
    kinds={"python", "arrow"},
    automation_condition=_EAGER_ON_VERSION_CHANGE,
)
def mesh_stage(
    context: AssetExecutionContext,
) -> MaterializeResult:
    from quarry.etl.mesh import parse_mesh_descriptors

    staging = settings.staging_dir / "mesh"
    clear(staging)

    xml_files = sorted(settings.pubmed_mesh_dir.glob("desc*.xml"))
    if not xml_files:
        context.log.warning("No MeSH descriptor XML found")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    xml_path = xml_files[-1]
    context.log.info(f"Parsing MeSH from {xml_path}")
    rows = parse_mesh_descriptors(xml_path)

    table = pa.table(
        {
            "descriptor_ui": pa.array(
                [r["descriptor_ui"] for r in rows], type=pa.string()
            ),
            "descriptor_name": pa.array(
                [r["descriptor_name"] for r in rows], type=pa.string()
            ),
            "tree_number": pa.array([r["tree_number"] for r in rows], type=pa.string()),
        }
    )
    write_tables(staging, {"mesh_tree": table})

    return MaterializeResult(metadata={"tree_entries": MetadataValue.int(len(rows))})
