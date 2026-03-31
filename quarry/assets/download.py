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
from pathlib import Path

from quarry.etl.fetch import (
    download_and_extract_zip,
    find_latest_ftp_file,
    resolve_figshare_files,
    sync_ftp_dir,
)
import subprocess


def _version_from_file_listing(
    files: dict,
    mtimes: dict[str, str] | None = None,
) -> DataVersion:
    """Stable DataVersion from {filename: value} + optional {filename: mtime}."""
    parts = sorted(files.items())
    if mtimes:
        parts = [(k, v, mtimes.get(k, "")) for k, v in parts]
    fingerprint = str(parts)
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    return DataVersion(digest)


def _ftp_sync_asset(
    label: str,
    host: str,
    remote_dir: str,
    local_dir: Path,
    pattern: str,
    context: AssetExecutionContext,
) -> dict:
    """Run sync_ftp_dir with unified logging."""
    stats = sync_ftp_dir(
        host=host,
        remote_dir=remote_dir,
        local_dir=local_dir,
        pattern=pattern,
        parallel=settings.ftp_parallel,
        on_listing=lambda total, new: context.log.info(
            f"[{label}] sync: {total} remote, {new} new, {total - new} cached"
        ),
        on_progress=lambda done, total, _name: context.log.info(
            f"[{label}] sync: downloading [{done}/{total}]"
        ),
    )
    byt = stats["bytes"]
    if byt > 0:
        context.log.info(f"[{label}] sync: done — {byt:,} bytes downloaded")
    else:
        context.log.info(f"[{label}] sync: done — all cached")
    return stats


@asset(
    group_name="pubmed",
    description="Mirror PubMed baseline XML files from NCBI FTP (~657 files, ~11GB).",
    kinds={"ftp"},
)
def pubmed_baseline_sync(
    context: AssetExecutionContext,
) -> MaterializeResult:
    stats = _ftp_sync_asset(
        "PubMed",
        settings.pubmed_ftp_host,
        settings.pubmed_ftp_baseline,
        settings.pubmed_baseline_dir,
        "pubmed*.xml.gz",
        context,
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
    stats = _ftp_sync_asset(
        "PubMed",
        settings.pubmed_ftp_host,
        settings.pubmed_ftp_updates,
        settings.pubmed_update_dir,
        "pubmed*.xml.gz",
        context,
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
    context.log.info(f"[MeSH] sync: found {filename} ({remote_size:,} bytes)")
    stats = _ftp_sync_asset(
        "MeSH",
        settings.mesh_ftp_host,
        settings.mesh_ftp_dir,
        settings.pubmed_mesh_dir,
        filename,
        context,
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
    description="Download iCite metadata CSV from figshare.",
    kinds={"http"},
)
def icite_metadata_sync(
    context: AssetExecutionContext,
) -> MaterializeResult:
    files = resolve_figshare_files(settings.icite_figshare_collection)
    url = files.get("icite_metadata.zip")
    if not url:
        context.log.warning("[iCite] sync: icite_metadata.zip not found")
        return MaterializeResult(metadata={"status": MetadataValue.text("skipped")})

    context.log.info("[iCite] sync: downloading icite_metadata.zip")
    info = download_and_extract_zip(
        url=url,
        local_dir=settings.icite_dir,
        expected_file="icite_metadata.csv",
        max_age_days=35,
    )
    sz = int(info["bytes"])
    context.log.info(f"[iCite] sync: done — {info['status']}, {sz:,} bytes")
    return MaterializeResult(
        data_version=_version_from_file_listing(files),
        metadata={
            "status": MetadataValue.text(str(info["status"])),
            "path": MetadataValue.path(str(info["path"])),
            "bytes": MetadataValue.int(sz),
        },
    )


@asset(
    group_name="sync",
    description="aws s3 sync for OpenAlex works.",
    kinds={"s3"},
)
def oa_sync(
    context: AssetExecutionContext,
) -> MaterializeResult:
    local_dir = settings.oa_local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    s3_prefix = settings.oa_s3_prefix
    base_cmd = [
        "aws",
        "s3",
        "sync",
        s3_prefix,
        str(local_dir),
        "--delete",
        "--no-sign-request",
        "--no-progress",
    ]

    # Dry-run to count pending operations
    dry = subprocess.run(
        base_cmd + ["--dryrun"],
        capture_output=True,
        text=True,
    )
    pending = dry.stdout.count("\n") if dry.returncode == 0 else 0

    def _local_version() -> DataVersion:
        """Hash of local .gz file listing for DataVersion."""
        files = sorted(str(p) for p in local_dir.rglob("*.gz"))
        digest = hashlib.sha256("\n".join(files).encode()).hexdigest()[:16]
        return DataVersion(digest)

    if pending == 0:
        gz_count = sum(1 for _ in local_dir.rglob("*.gz"))
        context.log.info(f"[OpenAlex] sync: all cached ({gz_count} .gz)")
        return MaterializeResult(
            data_version=_local_version(),
            metadata={
                "gz_files": MetadataValue.int(gz_count),
                "dir": MetadataValue.path(str(local_dir)),
            },
        )

    context.log.info(f"[OpenAlex] sync: {pending} files to process")
    proc = subprocess.Popen(
        base_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdout is not None
    done = 0
    for _line in proc.stdout:
        done += 1
        if done % max(pending // 20, 1) == 0 or done == pending:
            context.log.info(f"[OpenAlex] sync: [{done}/{pending}]")
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"aws s3 sync failed (exit {proc.returncode})")

    gz_count = sum(1 for _ in local_dir.rglob("*.gz"))
    context.log.info(f"[OpenAlex] sync: done — {done} processed, {gz_count} .gz")
    return MaterializeResult(
        data_version=_local_version(),
        metadata={
            "gz_files": MetadataValue.int(gz_count),
            "dir": MetadataValue.path(str(local_dir)),
        },
    )
