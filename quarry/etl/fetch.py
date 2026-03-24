"""Download upstream data sources to local cache (immutable raw files).

PubMed XML: FTP mirror from NCBI (ftplib + parallel threads)
iCite data: HTTP streaming download from NIH figshare
MeSH descriptors: FTP from NLM (auto-discover latest year)
"""

import ftplib
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable

import httpx


def sync_ftp_dir(
    host: str,
    remote_dir: str,
    local_dir: Path,
    pattern: str = "*",
    parallel: int = 4,
    max_retries: int = 3,
    on_progress: Callable[[int, int, str], None] | None = None,
    on_listing: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Mirror an FTP directory, downloading only new/changed files.

    Uses MLSD for reliable file listing, size comparison for change detection,
    atomic writes (.tmp → rename) for crash safety.
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    # List remote files via MLSD (machine-readable)
    # remote_files: {name: size} for DataVersion fingerprint
    # remote_mtimes: {name: "YYYYMMDDHHMMSS"} for change detection
    remote_files: dict[str, int] = {}
    remote_mtimes: dict[str, str] = {}
    with ftplib.FTP(host, timeout=30) as ftp:
        ftp.login()
        for name, facts in ftp.mlsd(remote_dir):
            if facts.get("type") != "file":
                continue
            if not fnmatch(name, pattern):
                continue
            remote_files[name] = int(facts.get("size", 0))
            remote_mtimes[name] = facts.get("modify", "")

    # Determine what needs downloading (mtime-first, size as fallback)
    # Note: some FTP servers (e.g. NCBI) report stale MLSD sizes, so
    # mtime is the primary change signal; size-only check is unreliable.
    to_download = []
    for name, remote_size in remote_files.items():
        local_path = local_dir / name
        if not local_path.exists():
            to_download.append(name)
            continue
        modify = remote_mtimes.get(name, "")
        if modify:
            try:
                remote_ts = datetime.strptime(modify[:14], "%Y%m%d%H%M%S")
                local_ts = datetime.fromtimestamp(local_path.stat().st_mtime)
                if remote_ts > local_ts:
                    to_download.append(name)
            except ValueError:
                pass
        elif local_path.stat().st_size != remote_size:
            # No mtime available — fall back to size comparison
            to_download.append(name)

    if on_listing:
        on_listing(len(remote_files), len(to_download))

    if not to_download:
        return {
            "downloaded": 0,
            "skipped": len(remote_files),
            "errors": 0,
            "bytes": 0,
            "remote_files": remote_files,
            "remote_mtimes": remote_mtimes,
        }

    # Parallel download: one persistent FTP connection per worker thread
    total_bytes = 0
    downloaded = 0
    errors = 0

    def _worker(files: list[str]) -> list[tuple[str, int]]:
        """Download a batch of files over a single FTP connection."""
        results = []
        conn = ftplib.FTP(host, timeout=120)
        conn.login()
        try:
            for name in files:
                local_path = local_dir / name
                tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
                for attempt in range(max_retries):
                    try:
                        with open(tmp_path, "wb") as f:
                            conn.retrbinary(f"RETR {remote_dir}/{name}", f.write)
                        tmp_path.rename(local_path)
                        results.append((name, local_path.stat().st_size))
                        break
                    except (ftplib.error_temp, OSError, EOFError):
                        if attempt == max_retries - 1:
                            tmp_path.unlink(missing_ok=True)
                            results.append((name, -1))
                        else:
                            time.sleep(2**attempt)
                            # Reconnect on transient failure
                            try:
                                conn.close()
                            except Exception:
                                pass
                            conn = ftplib.FTP(host, timeout=120)
                            conn.login()
        finally:
            try:
                conn.quit()
            except Exception:
                pass
        return results

    # Split files evenly across workers
    chunks = [to_download[i::parallel] for i in range(parallel)]

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_worker, chunk): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            for name, size in future.result():
                if size >= 0:
                    total_bytes += size
                    downloaded += 1
                else:
                    errors += 1
                if on_progress:
                    on_progress(downloaded + errors, len(to_download), name)

    return {
        "downloaded": downloaded,
        "skipped": len(remote_files) - len(to_download),
        "errors": errors,
        "bytes": total_bytes,
        "remote_files": remote_files,
        "remote_mtimes": remote_mtimes,
    }


def find_latest_ftp_file(
    host: str,
    remote_dir: str,
    pattern: str,
) -> tuple[str, int]:
    """Find the latest matching file on FTP by name (lexicographic sort).

    Returns (filename, size). Raises FileNotFoundError if no match.
    """
    with ftplib.FTP(host, timeout=30) as ftp:
        ftp.login()
        matches = []
        for name, facts in ftp.mlsd(remote_dir):
            if facts.get("type") != "file":
                continue
            if fnmatch(name, pattern):
                matches.append((name, int(facts.get("size", 0))))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} on {host}:{remote_dir}")
    matches.sort(key=lambda x: x[0])
    return matches[-1]  # latest by name (e.g. desc2026.xml > desc2025.xml)


def download_http(
    url: str,
    dest: Path,
    max_age_days: int | None = None,
) -> dict[str, str | int]:
    """Download a file via HTTPS. Skips if local copy is fresh enough."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and max_age_days is not None:
        age = datetime.now() - datetime.fromtimestamp(dest.stat().st_mtime)
        if age < timedelta(days=max_age_days):
            return {"status": "fresh", "path": str(dest), "bytes": dest.stat().st_size}

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with httpx.Client(
        timeout=httpx.Timeout(600, connect=30), follow_redirects=True
    ) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
    tmp.rename(dest)
    return {"status": "downloaded", "path": str(dest), "bytes": dest.stat().st_size}


def resolve_figshare_files(collection_id: int) -> dict[str, str]:
    """Get download URLs for files in the latest article of a figshare collection.

    Returns dict mapping filename → download_url.
    """
    api = "https://api.figshare.com/v2"
    with httpx.Client(timeout=30) as client:
        # Get latest article in collection
        resp = client.get(
            f"{api}/collections/{collection_id}/articles",
            params={
                "page_size": 1,
                "order": "published_date",
                "order_direction": "desc",
            },
        )
        resp.raise_for_status()
        articles = resp.json()
        if not articles:
            raise RuntimeError(f"No articles in figshare collection {collection_id}")
        article_id = articles[0]["id"]

        # Get files for that article
        resp = client.get(f"{api}/articles/{article_id}/files")
        resp.raise_for_status()
        return {f["name"]: f["download_url"] for f in resp.json()}


def download_and_extract_zip(
    url: str,
    local_dir: Path,
    expected_file: str,
    max_age_days: int = 35,
) -> dict[str, str | int]:
    """Download a zip, extract contents. Skips if expected_file is fresh."""
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / expected_file

    if target.exists():
        age = datetime.now() - datetime.fromtimestamp(target.stat().st_mtime)
        if age < timedelta(days=max_age_days):
            return {
                "status": "fresh",
                "path": str(target),
                "bytes": target.stat().st_size,
            }

    zip_path = local_dir / f"_{expected_file}.zip"
    download_http(url, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(local_dir)
    zip_path.unlink()

    return {
        "status": "extracted",
        "path": str(target),
        "bytes": target.stat().st_size if target.exists() else 0,
    }
