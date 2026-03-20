"""Staging assets: MeSH parse → PG direct load.

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

import pyarrow as pa
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
from quarry.resources import PGResource


@asset(
    group_name="supplementary",
    deps=[mesh_descriptor_sync],
    description="Parse MeSH descriptor XML → PG mesh_tree table.",
    kinds={"python", "postgres"},
    automation_condition=AutomationCondition.eager(),
)
def mesh_stage(
    context: AssetExecutionContext,
    pg: PGResource,
) -> MaterializeResult:
    xml_files = sorted(settings.pubmed_mesh_dir.glob("desc*.xml"))
    if not xml_files:
        context.log.warning("No MeSH descriptor XML found")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    xml_path = xml_files[-1]
    context.log.info(f"Parsing MeSH from {xml_path}")

    # Rust quick-xml parser → Arrow RecordBatch
    batch = quarry_rs.parse_mesh_xml(str(xml_path))
    table = pa.Table.from_batches([batch])

    # Load into PG via COPY
    store = pg.store
    conn = store.conn
    conn.execute("DELETE FROM mesh_tree")

    with conn.cursor() as cur:
        with cur.copy(
            "COPY mesh_tree (descriptor_ui, descriptor_name, tree_number) FROM STDIN"
        ) as copy:
            for i in range(table.num_rows):
                copy.write_row(
                    (
                        table.column("descriptor_ui")[i].as_py(),
                        table.column("descriptor_name")[i].as_py(),
                        table.column("tree_number")[i].as_py(),
                    )
                )

    return MaterializeResult(
        metadata={"tree_entries": MetadataValue.int(table.num_rows)}
    )
