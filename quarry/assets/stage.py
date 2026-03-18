"""Staging assets: parse raw data → Arrow Feather files.

No DuckDB access. All assets here can run in parallel.
Uses AutomationCondition.eager() with data_version_changed() so that
scheduled runs skip re-processing when upstream data hasn't changed.

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

import orjson
import pyarrow as pa
from dagster import (
    AssetExecutionContext,
    AutomationCondition,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.download import (
    mesh_descriptor_sync,
    pubmed_updates_sync,
)
from quarry.config import settings
from quarry.etl.staging import clear, write_tables

import quarry_parse

_EAGER_ON_VERSION_CHANGE = AutomationCondition.eager().replace(
    "any_deps_updated",
    AutomationCondition.any_deps_updated().replace(
        "newly_updated", AutomationCondition.data_version_changed()
    ),
)


# -- PubMed ------------------------------------------------------------------


@asset(
    group_name="pubmed",
    deps=[pubmed_updates_sync],
    description="Parse PubMed daily update XML → Arrow Feather staging files.",
    kinds={"rust", "arrow"},
    automation_condition=_EAGER_ON_VERSION_CHANGE,
)
def pubmed_updates_stage(
    context: AssetExecutionContext,
) -> MaterializeResult:
    staging = settings.staging_dir / "pubmed_updates"
    clear(staging)

    files = sorted(settings.pubmed_update_dir.glob("pubmed*.xml.gz"))
    if not files:
        context.log.info("No update files to stage")
        return MaterializeResult(metadata={"num_files": MetadataValue.int(0)})

    total_papers = 0
    for i, f in enumerate(files):
        result = quarry_parse.parse_pubmed_file(str(f))
        _write_pubmed_result(staging / f"chunk_{i:04d}", result)
        total_papers += result["stats"]["num_papers"]
        context.log.info(
            f"  [{i + 1}/{len(files)}] {f.name}: papers={result['stats']['num_papers']}"
        )

    return MaterializeResult(
        metadata={
            "num_files": MetadataValue.int(len(files)),
            "total_papers": MetadataValue.int(total_papers),
        }
    )


def _write_pubmed_result(chunk_dir, result):
    """Write quarry_parse output as Feather files."""
    tables = {}
    for name in ("papers", "authors", "mesh_headings", "grants", "chemicals"):
        rb = result[name]
        if rb.num_rows > 0:
            tables[name] = rb
    if result["delete_pmids"]:
        tables["delete_pmids"] = pa.table(
            {"pmid": pa.array(result["delete_pmids"], type=pa.int32())}
        )
    write_tables(chunk_dir, tables)


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


# -- bioRxiv ------------------------------------------------------------------


@asset(
    group_name="supplementary",
    description="Fetch bioRxiv/medRxiv API → JSONL cache → Arrow Feather staging.",
    kinds={"http", "arrow"},
)
def biorxiv_stage(
    context: AssetExecutionContext,
) -> MaterializeResult:
    from quarry.etl.biorxiv import fetch_preprints
    from datetime import date, timedelta

    staging = settings.staging_dir / "biorxiv"
    clear(staging)
    cache_dir = settings.biorxiv_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    to_date = date.today()
    from_date = to_date - timedelta(days=7)

    all_preprints = []
    for server in ("biorxiv", "medrxiv"):
        context.log.info(f"Fetching {server} preprints ({from_date} → {to_date})")
        preprints = fetch_preprints(server=server, from_date=from_date, to_date=to_date)
        context.log.info(f"  {server}: {len(preprints)} preprints")

        # Cache raw API response as JSONL
        cache_file = cache_dir / f"{server}_{from_date}_{to_date}.jsonl"
        with open(cache_file, "wb") as f:
            for p in preprints:
                f.write(orjson.dumps(p) + b"\n")

        all_preprints.extend(preprints)

    if not all_preprints:
        return MaterializeResult(metadata={"total": MetadataValue.int(0)})

    # Deduplicate: keep latest version per DOI
    by_doi: dict[str, dict] = {}
    for p in all_preprints:
        existing = by_doi.get(p["doi"])
        if existing is None or p["version"] > existing["version"]:
            by_doi[p["doi"]] = p
    deduped = list(by_doi.values())

    table = pa.table(
        {
            "doi": pa.array([p["doi"] for p in deduped], type=pa.string()),
            "title": pa.array([p["title"] for p in deduped], type=pa.string()),
            "abstract": pa.array([p["abstract"] for p in deduped], type=pa.string()),
            "date": pa.array(
                [p["date"] for p in deduped],
                type=pa.string(),  # cast to DATE at load time
            ),
            "server": pa.array([p["server"] for p in deduped], type=pa.string()),
            "category": pa.array([p["category"] for p in deduped], type=pa.string()),
            "version": pa.array([p["version"] for p in deduped], type=pa.int16()),
            "published_doi": pa.array(
                [p["published_doi"] for p in deduped], type=pa.string()
            ),
        }
    )
    write_tables(staging, {"preprints": table})

    return MaterializeResult(
        metadata={"total_preprints": MetadataValue.int(len(deduped))}
    )
