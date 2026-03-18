"""Staging assets: parse raw data → Arrow Feather files.

No DuckDB access. All assets here can run in parallel.
DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

import orjson
import pyarrow as pa
from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.download import (
    icite_metadata_sync,
    mesh_descriptor_sync,
    pubmed_baseline_sync,
    pubmed_updates_sync,
)
from quarry.config import settings
from quarry.etl.staging import clear, write_tables

import quarry_parse


# -- PubMed ------------------------------------------------------------------


@asset(
    group_name="pubmed",
    deps=[pubmed_baseline_sync],
    description="Parse PubMed baseline XML → Arrow Feather staging files.",
    kinds={"rust", "arrow"},
)
def pubmed_baseline_stage(
    context: AssetExecutionContext,
) -> MaterializeResult:
    staging = settings.staging_dir / "pubmed_baseline"
    clear(staging)

    files = sorted(settings.pubmed_baseline_dir.glob("pubmed*.xml.gz"))
    context.log.info(f"Staging baseline: {len(files)} files")

    total_papers = 0
    chunk_size = 20

    for i in range(0, len(files), chunk_size):
        chunk = files[i : i + chunk_size]
        result = quarry_parse.parse_pubmed_files([str(p) for p in chunk])
        chunk_id = i // chunk_size
        _write_pubmed_result(staging / f"chunk_{chunk_id:04d}", result)
        total_papers += result["stats"]["num_papers"]
        done = min(i + chunk_size, len(files))
        context.log.info(
            f"  [{done}/{len(files)}] papers={result['stats']['num_papers']}"
        )

    return MaterializeResult(
        metadata={
            "num_files": MetadataValue.int(len(files)),
            "total_papers": MetadataValue.int(total_papers),
        }
    )


@asset(
    group_name="pubmed",
    deps=[pubmed_updates_sync],
    description="Parse PubMed daily update XML → Arrow Feather staging files.",
    kinds={"rust", "arrow"},
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


# -- iCite metrics ------------------------------------------------------------


@asset(
    group_name="citations",
    deps=[icite_metadata_sync],
    description="Parse iCite metadata CSV → Arrow Feather staging.",
    kinds={"python", "arrow"},
)
def icite_metrics_stage(
    context: AssetExecutionContext,
) -> MaterializeResult:
    import csv

    staging = settings.staging_dir / "icite_metrics"
    clear(staging)

    meta_csv = settings.icite_dir / "icite_metadata.csv"
    if not meta_csv.exists():
        context.log.warning(f"iCite metadata CSV not found: {meta_csv}")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    context.log.info(f"Parsing iCite metrics from {meta_csv}")
    csv.field_size_limit(2**30)

    batch: list[dict] = []
    batch_id = 0
    batch_size = 50_000
    total = 0

    def _flush(rows, bid):
        table = pa.table(
            {
                "pmid": pa.array([r["pmid"] for r in rows], type=pa.int32()),
                "rcr": pa.array([r["rcr"] for r in rows], type=pa.float32()),
                "nih_percentile": pa.array(
                    [r["nih_percentile"] for r in rows], type=pa.float32()
                ),
                "apt": pa.array([r["apt"] for r in rows], type=pa.float32()),
                "is_clinical": pa.array(
                    [r["is_clinical"] for r in rows], type=pa.bool_()
                ),
                "human": pa.array([r["human"] for r in rows], type=pa.float32()),
                "animal": pa.array([r["animal"] for r in rows], type=pa.float32()),
                "molecular_cellular": pa.array(
                    [r["molecular_cellular"] for r in rows], type=pa.float32()
                ),
                "cited_by_clin": pa.array(
                    [r["cited_by_clin"] for r in rows], type=pa.int32()
                ),
                "field_citation_rate": pa.array(
                    [r["field_citation_rate"] for r in rows], type=pa.float32()
                ),
            }
        )
        write_tables(staging / f"batch_{bid:04d}", {"metrics": table})

    with open(meta_csv, "r") as f:
        for row in csv.DictReader(f):
            pmid_str = row.get("pmid", "")
            if not pmid_str or not pmid_str.isdigit():
                continue
            batch.append(
                {
                    "pmid": int(pmid_str),
                    "rcr": _float_or_none(row.get("relative_citation_ratio")),
                    "nih_percentile": _float_or_none(row.get("nih_percentile")),
                    "apt": _float_or_none(row.get("apt")),
                    "is_clinical": row.get("is_clinical", "").lower() == "true",
                    "human": _float_or_none(row.get("human")),
                    "animal": _float_or_none(row.get("animal")),
                    "molecular_cellular": _float_or_none(row.get("molecular_cellular")),
                    "cited_by_clin": _int_or_none(row.get("cited_by_clin")),
                    "field_citation_rate": _float_or_none(
                        row.get("field_citation_rate")
                    ),
                }
            )
            if len(batch) >= batch_size:
                _flush(batch, batch_id)
                total += len(batch)
                batch = []
                batch_id += 1
                context.log.info(f"  {total:,} rows staged")

    if batch:
        _flush(batch, batch_id)
        total += len(batch)

    return MaterializeResult(metadata={"total_rows": MetadataValue.int(total)})


def _float_or_none(s):
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _int_or_none(s):
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None
