"""Supplementary data assets: MeSH tree + bioRxiv preprints.

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.config import settings
from quarry.etl.biorxiv import load_preprints
from quarry.etl.mesh import load_mesh_tree
from quarry.resources import DuckDBResource


@asset(
    group_name="supplementary",
    description="Parse MeSH descriptor XML → DuckDB mesh_tree table (annual refresh).",
    kinds={"duckdb", "python"},
)
def mesh_tree(
    context: AssetExecutionContext,
    duckdb: DuckDBResource,
) -> MaterializeResult:
    mesh_dir = settings.pubmed_mesh_dir
    xml_files = sorted(mesh_dir.glob("desc*.xml"))

    if not xml_files:
        context.log.warning(f"No MeSH descriptor XML in {mesh_dir}")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    xml_path = xml_files[-1]  # latest descriptor file
    context.log.info(f"Loading MeSH tree from {xml_path}")

    db = duckdb.get_store()
    try:
        load_mesh_tree(xml_path, db=db)
        count = db.conn.execute("SELECT count(*) FROM mesh_tree").fetchone()[0]
        return MaterializeResult(
            metadata={
                "xml_file": MetadataValue.text(xml_path.name),
                "tree_entries": MetadataValue.int(count),
            }
        )
    finally:
        db.close()


@asset(
    group_name="supplementary",
    description="Fetch recent bioRxiv/medRxiv preprints → DuckDB preprints table.",
    kinds={"duckdb", "python"},
)
def biorxiv_preprints(
    context: AssetExecutionContext,
    duckdb: DuckDBResource,
) -> MaterializeResult:
    db = duckdb.get_store()
    try:
        total = 0
        for server in ("biorxiv", "medrxiv"):
            context.log.info(f"Fetching {server} preprints (last 7 days)")
            n = load_preprints(server=server, db=db)
            total += n
            context.log.info(f"  {server}: {n} preprints upserted")

        return MaterializeResult(metadata={"total_preprints": MetadataValue.int(total)})
    finally:
        db.close()
