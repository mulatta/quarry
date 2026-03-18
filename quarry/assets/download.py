"""Upstream data download assets (immutable cache layer).

These assets mirror raw data from external sources to local disk.
Transform assets (pubmed, citations, supplementary) depend on these
and read from the cached files to produce derived state.

Each asset returns a DataVersion derived from the remote file state,
enabling AutomationCondition.eager() on downstream assets to skip
re-processing when upstream data hasn't changed.

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

import hashlib

from dagster import (
    AssetExecutionContext,
    DataVersion,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.config import settings
from quarry.etl.download import (
    download_and_extract_zip,
    find_latest_ftp_file,
    resolve_figshare_files,
    sync_ftp_dir,
)


def _version_from_file_listing(
    files: dict[str, int],
    mtimes: dict[str, str] | None = None,
) -> DataVersion:
    """Stable DataVersion from {filename: size} + optional {filename: mtime}."""
    parts = sorted(files.items())
    if mtimes:
        parts = [(k, v, mtimes.get(k, "")) for k, v in parts]
    fingerprint = str(parts)
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    return DataVersion(digest)


@asset(
    group_name="pubmed",
    description="Mirror PubMed baseline XML files from NCBI FTP (~657 files, ~11GB).",
    kinds={"ftp"},
)
def pubmed_baseline_sync(
    context: AssetExecutionContext,
) -> MaterializeResult:
    context.log.info(
        f"Syncing baseline from {settings.pubmed_ftp_host}{settings.pubmed_ftp_baseline}"
    )
    stats = sync_ftp_dir(
        host=settings.pubmed_ftp_host,
        remote_dir=settings.pubmed_ftp_baseline,
        local_dir=settings.pubmed_baseline_dir,
        pattern="pubmed*.xml.gz",
        parallel=settings.ftp_parallel,
        on_progress=lambda done, total, name: context.log.info(
            f"  [{done}/{total}] {name}"
        ),
    )
    return MaterializeResult(
        data_version=_version_from_file_listing(
            stats["remote_files"], stats.get("remote_mtimes")
        ),
        metadata={
            "downloaded": MetadataValue.int(stats["downloaded"]),
            "skipped": MetadataValue.int(stats["skipped"]),
            "errors": MetadataValue.int(stats["errors"]),
            "bytes": MetadataValue.int(stats["bytes"]),
            "dir": MetadataValue.path(str(settings.pubmed_baseline_dir)),
        },
    )


@asset(
    group_name="pubmed",
    description="Mirror PubMed daily update XML files from NCBI FTP.",
    kinds={"ftp"},
)
def pubmed_updates_sync(
    context: AssetExecutionContext,
) -> MaterializeResult:
    context.log.info(
        f"Syncing updates from {settings.pubmed_ftp_host}{settings.pubmed_ftp_updates}"
    )
    stats = sync_ftp_dir(
        host=settings.pubmed_ftp_host,
        remote_dir=settings.pubmed_ftp_updates,
        local_dir=settings.pubmed_update_dir,
        pattern="pubmed*.xml.gz",
        parallel=settings.ftp_parallel,
        on_progress=lambda done, total, name: context.log.info(
            f"  [{done}/{total}] {name}"
        ),
    )
    return MaterializeResult(
        data_version=_version_from_file_listing(
            stats["remote_files"], stats.get("remote_mtimes")
        ),
        metadata={
            "downloaded": MetadataValue.int(stats["downloaded"]),
            "skipped": MetadataValue.int(stats["skipped"]),
            "errors": MetadataValue.int(stats["errors"]),
            "bytes": MetadataValue.int(stats["bytes"]),
            "dir": MetadataValue.path(str(settings.pubmed_update_dir)),
        },
    )


@asset(
    group_name="supplementary",
    description="Download latest MeSH descriptor XML from NLM FTP (annual update).",
    kinds={"ftp"},
)
def mesh_descriptor_sync(
    context: AssetExecutionContext,
) -> MaterializeResult:
    filename, remote_size = find_latest_ftp_file(
        host=settings.mesh_ftp_host,
        remote_dir=settings.mesh_ftp_dir,
        pattern="desc*.xml",
    )
    context.log.info(f"Latest MeSH descriptor: {filename} ({remote_size:,} bytes)")

    stats = sync_ftp_dir(
        host=settings.mesh_ftp_host,
        remote_dir=settings.mesh_ftp_dir,
        local_dir=settings.pubmed_mesh_dir,
        pattern=filename,
    )
    return MaterializeResult(
        data_version=_version_from_file_listing(
            stats["remote_files"], stats.get("remote_mtimes")
        ),
        metadata={
            "file": MetadataValue.text(filename),
            "downloaded": MetadataValue.int(stats["downloaded"]),
            "bytes": MetadataValue.int(stats["bytes"]),
        },
    )


@asset(
    group_name="citations",
    description="Download iCite Open Citation Collection CSV from figshare (~6GB zip).",
    kinds={"http"},
)
def icite_occ_sync(
    context: AssetExecutionContext,
) -> MaterializeResult:
    files = resolve_figshare_files(settings.icite_figshare_collection)
    url = files.get("open_citation_collection.zip")
    if not url:
        context.log.warning(
            "open_citation_collection.zip not found in figshare collection"
        )
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    context.log.info(f"Downloading iCite OCC: {url}")
    info = download_and_extract_zip(
        url=url,
        local_dir=settings.icite_dir,
        expected_file="open_citation_collection.csv",
        max_age_days=35,
    )
    # DataVersion from figshare article file listing (changes on new monthly release)
    version_str = str(sorted(files.items()))
    digest = hashlib.sha256(version_str.encode()).hexdigest()[:16]
    return MaterializeResult(
        data_version=DataVersion(digest),
        metadata={
            "status": MetadataValue.text(str(info["status"])),
            "path": MetadataValue.path(str(info["path"])),
            "bytes": MetadataValue.int(int(info["bytes"])),
        },
    )


@asset(
    group_name="citations",
    description="Download iCite metadata CSV from figshare.",
    kinds={"http"},
)
def icite_metadata_sync(
    context: AssetExecutionContext,
) -> MaterializeResult:
    files = resolve_figshare_files(settings.icite_figshare_collection)
    url = files.get("icite_metadata.zip")
    if not url:
        context.log.warning("icite_metadata.zip not found in figshare collection")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    context.log.info(f"Downloading iCite metadata: {url}")
    info = download_and_extract_zip(
        url=url,
        local_dir=settings.icite_dir,
        expected_file="icite_metadata.csv",
        max_age_days=35,
    )
    version_str = str(sorted(files.items()))
    digest = hashlib.sha256(version_str.encode()).hexdigest()[:16]
    return MaterializeResult(
        data_version=DataVersion(digest),
        metadata={
            "status": MetadataValue.text(str(info["status"])),
            "path": MetadataValue.path(str(info["path"])),
            "bytes": MetadataValue.int(int(info["bytes"])),
        },
    )
