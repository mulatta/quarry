"""Staging assets: parse raw data → Arrow Feather files, Rust build pipeline.

No DuckDB access. All assets here can run in parallel.
Uses AutomationCondition.eager() with data_version_changed() so that
scheduled runs skip re-processing when upstream data hasn't changed.

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

import subprocess

import pyarrow as pa
import quarry_parse
from dagster import (
    AssetExecutionContext,
    AutomationCondition,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.download import mesh_descriptor_sync, pubmed_baseline_sync
from quarry.config import settings
from quarry.etl.feather import clear, write_tables

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
    staging = settings.staging_dir / "mesh"
    clear(staging)

    xml_files = sorted(settings.pubmed_mesh_dir.glob("desc*.xml"))
    if not xml_files:
        context.log.warning("No MeSH descriptor XML found")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    xml_path = xml_files[-1]
    context.log.info(f"Parsing MeSH from {xml_path}")

    # Rust quick-xml parser → Arrow RecordBatch directly
    batch = quarry_parse.parse_mesh_xml(str(xml_path))
    table = pa.Table.from_batches([batch])
    write_tables(staging, {"mesh_tree": table})

    return MaterializeResult(
        metadata={"tree_entries": MetadataValue.int(table.num_rows)}
    )


# -- Rust build pipeline ---------------------------------------------------


@asset(
    group_name="build",
    deps=[pubmed_baseline_sync],
    description="quarry-build build-pubmed → PubMed Parquet files.",
    kinds={"rust", "parquet"},
)
def pubmed_parquet_build(context: AssetExecutionContext) -> MaterializeResult:
    out = settings.build_output_dir
    subprocess.run(
        [
            "quarry-build",
            "--output",
            str(out),
            "build-pubmed",
            "--xml-dir",
            str(settings.pubmed_baseline_dir),
        ],
        check=True,
    )
    return MaterializeResult(metadata={"output_dir": MetadataValue.path(str(out))})


@asset(
    group_name="build",
    description="quarry-build build-oa → OpenAlex Parquet files.",
    kinds={"rust", "parquet"},
)
def oa_parquet_build(context: AssetExecutionContext) -> MaterializeResult:
    out = settings.build_output_dir
    subprocess.run(
        [
            "quarry-build",
            "--output",
            str(out),
            "build-oa",
            "--s3-prefix",
            settings.oa_s3_prefix,
            "--cache-dir",
            str(settings.oa_cache_dir),
        ],
        check=True,
    )
    return MaterializeResult(metadata={"output_dir": MetadataValue.path(str(out))})


@asset(
    group_name="build",
    deps=[pubmed_parquet_build, oa_parquet_build],
    description="quarry-build enrich → enriched works + work_mesh Parquet.",
    kinds={"rust", "parquet"},
)
def enrich_build(context: AssetExecutionContext) -> MaterializeResult:
    out = settings.build_output_dir
    subprocess.run(
        ["quarry-build", "--output", str(out), "enrich"],
        check=True,
    )
    return MaterializeResult(metadata={"output_dir": MetadataValue.path(str(out))})
