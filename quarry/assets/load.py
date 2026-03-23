"""PostgreSQL bulk load assets.

Each data source loads via quarry-ingest CLI subprocess:
- PubMed: quarry-ingest load pubmed
- OpenAlex: quarry-ingest load oa
- iCite: quarry-ingest load icite
- Enrichment: quarry-ingest enrich

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

import json
import subprocess

from dagster import (
    AssetExecutionContext,
    AutomationCondition,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.download import (
    icite_metadata_sync,
    pubmed_baseline_sync,
    pubmed_updates_sync,
)
from quarry.config import settings


def _run_ingest(args: list[str], context: AssetExecutionContext) -> dict:
    """Run quarry-ingest subprocess, stream stderr to Dagster log, parse stdout JSON."""
    cmd = ["quarry-ingest", "--pg-conninfo", settings.pg_conninfo] + args
    context.log.info(f"Running: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    # Stream stderr lines to Dagster log in real-time.
    assert proc.stderr is not None
    for line in proc.stderr:
        line = line.rstrip("\n")
        if line:
            context.log.info(line)
    proc.wait()

    assert proc.stdout is not None
    stdout = proc.stdout.read()

    if proc.returncode != 0:
        raise RuntimeError(f"quarry-ingest failed (exit {proc.returncode})")

    return json.loads(stdout) if stdout.strip() else {}


def _metadata_from_stats(stats: dict) -> dict[str, MetadataValue]:
    """Convert stats dict to Dagster metadata values."""
    return {
        k: MetadataValue.int(v) if isinstance(v, int) else MetadataValue.float(v)
        for k, v in stats.items()
    }


@asset(
    group_name="build",
    deps=[pubmed_baseline_sync, pubmed_updates_sync],
    description="Rust PubMed XML parser → PG direct load via quarry-ingest.",
    kinds={"rust", "postgres"},
)
def pubmed_pg_load(context: AssetExecutionContext) -> MaterializeResult:
    _run_ingest(["db", "drop-indexes"], context)

    args = [
        "load",
        "pubmed",
        "--xml-dir",
        str(settings.pubmed_baseline_dir),
        "--threads",
        str(settings.pubmed_parse_threads),
        "--pg-writers",
        str(settings.pubmed_pg_writers),
        "--channel-buffer",
        str(settings.pubmed_channel_buffer),
    ]

    update_dir = settings.pubmed_update_dir
    if update_dir.exists() and any(update_dir.glob("pubmed*.xml.gz")):
        args += ["--updates-dir", str(update_dir)]

    stats = _run_ingest(args, context)
    return MaterializeResult(metadata=_metadata_from_stats(stats))


@asset(
    group_name="build",
    deps=[pubmed_pg_load],  # sequential: run after pubmed to avoid WAL contention
    description="Rust OpenAlex JSONL parser → PG direct load via quarry-ingest.",
    kinds={"rust", "postgres"},
)
def oa_pg_load(context: AssetExecutionContext) -> MaterializeResult:
    _run_ingest(["db", "drop-indexes"], context)

    stats = _run_ingest(
        [
            "load",
            "oa",
            "--s3-prefix",
            settings.oa_s3_prefix,
            "--s3-concurrency",
            str(settings.oa_s3_concurrency),
            "--prefetch-buffer",
            str(settings.oa_prefetch_buffer),
            "--pg-writers",
            str(settings.oa_pg_writers),
            "--channel-buffer",
            str(settings.oa_channel_buffer),
            "--fetch-max-retries",
            str(settings.oa_fetch_max_retries),
            "--fetch-initial-backoff-ms",
            str(settings.oa_fetch_initial_backoff_ms),
            "--fetch-max-backoff-ms",
            str(settings.oa_fetch_max_backoff_ms),
        ],
        context,
    )

    _run_ingest(["db", "create-indexes"], context)
    _run_ingest(["db", "vacuum"], context)

    return MaterializeResult(metadata=_metadata_from_stats(stats))


@asset(
    group_name="load",
    deps=[pubmed_pg_load, oa_pg_load, icite_metadata_sync],
    description="iCite metrics: quarry-ingest load icite → UPDATE papers + works.",
    kinds={"rust", "postgres"},
    automation_condition=AutomationCondition.eager(),
)
def icite_pg_load(context: AssetExecutionContext) -> MaterializeResult:
    meta_csv = settings.icite_dir / "icite_metadata.csv"
    if not meta_csv.exists():
        context.log.warning(f"iCite CSV not found: {meta_csv}")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    stats = _run_ingest(["load", "icite", "--csv-path", str(meta_csv)], context)
    return MaterializeResult(metadata=_metadata_from_stats(stats))


@asset(
    group_name="load",
    deps=[pubmed_pg_load, oa_pg_load],
    description="Enrich: UPDATE works SET pm_* FROM papers; generate work_mesh via quarry-ingest.",
    kinds={"rust", "postgres"},
)
def enrich_pg(context: AssetExecutionContext) -> MaterializeResult:
    stats = _run_ingest(["enrich"], context)
    return MaterializeResult(metadata=_metadata_from_stats(stats))
