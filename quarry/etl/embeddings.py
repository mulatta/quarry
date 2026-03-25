"""Batch-encode papers from CH → LanceDB with blake3 content caching.

Reads works from CH via ArrowStream pipe (T1+T2 filter in SQL),
skips unchanged (blake3 match), encodes new/modified papers with jina-v5 nano,
upserts to LanceDB.
"""

import logging
import subprocess
import time

try:
    import blake3
    import pyarrow.ipc
except ImportError:
    raise ImportError("pip install quarry[elt]") from None

from quarry.config import settings
from quarry.embed.jina import JinaEncoder
from quarry.store.lance import LanceStore

log = logging.getLogger(__name__)

_WORKS_QUERY = (
    "SELECT work_id, title, abstract FROM works_export "
    "WHERE tier IN ('t1', 't2') "
    "AND abstract IS NOT NULL AND abstract != '' AND title != ''"
)


def content_hash(title: str, abstract: str) -> bytes:
    """blake3 hash of normalized title + abstract for change detection."""
    return blake3.blake3(f"{title}\n{abstract}".encode()).digest()


def _ch_arrow_reader() -> tuple[subprocess.Popen, pyarrow.ipc.RecordBatchStreamReader]:
    """Open CH ArrowStream pipe for works (T1+T2 with abstract)."""
    cmd = [
        "clickhouse-client",
        "--host",
        settings.ch_host,
        "--port",
        str(settings.ch_port),
        "--database",
        settings.ch_database,
        "--receive_timeout",
        "7200",
        "--send_timeout",
        "7200",
        "--query",
        _WORKS_QUERY,
        "--format",
        "ArrowStream",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    reader = pyarrow.ipc.open_stream(proc.stdout)
    return proc, reader


def run(batch_size: int = 5000, limit: int | None = None):
    lance = LanceStore(settings.lancedb_uri)

    # Ensure table exists
    try:
        lance.table
    except Exception:
        lance.create_table()

    encoder = JinaEncoder(dim=256)

    # Stream from CH via ArrowStream (memory-bounded, batch at a time)
    proc, reader = _ch_arrow_reader()

    total_encoded = 0
    total_skipped = 0
    batch_num = 0

    for batch in reader:
        if len(batch) == 0:
            continue

        works = [
            {
                "work_id": batch.column("work_id")[i].as_py(),
                "title": batch.column("title")[i].as_py(),
                "abstract": batch.column("abstract")[i].as_py(),
            }
            for i in range(len(batch))
        ]

        hashes = {w["work_id"]: content_hash(w["title"], w["abstract"]) for w in works}

        # Check existing hashes in LanceDB
        ids = list(hashes.keys())
        try:
            existing = lance.existing_hashes(ids)
        except Exception as exc:
            log.warning("existing_hashes failed (%s), re-encoding all", exc)
            existing = {}

        # Filter to only new/changed works
        to_encode = [
            w for w in works if hashes[w["work_id"]] != existing.get(w["work_id"])
        ]

        if not to_encode:
            total_skipped += len(works)
            batch_num += 1
            if limit and (batch_num * batch_size) >= limit:
                break
            continue

        # Encode
        texts = [f"{w['title']}. {w['abstract']}" for w in to_encode]
        t0 = time.time()
        vec_ret = encoder.encode_passages(texts)
        vec_clust = encoder.encode_clustering(texts)
        elapsed = time.time() - t0

        # Build rows for LanceDB upsert
        lance_rows = []
        for i, w in enumerate(to_encode):
            lance_rows.append(
                {
                    "work_id": w["work_id"],
                    "content_hash": hashes[w["work_id"]],
                    "title": w["title"],
                    "abstract": w["abstract"],
                    "vec_retrieval": vec_ret[i].tolist(),
                    "vec_cluster": vec_clust[i].tolist(),
                }
            )

        lance.upsert(lance_rows)

        total_encoded += len(to_encode)
        total_skipped += len(works) - len(to_encode)
        batch_num += 1
        throughput = len(texts) / elapsed if elapsed > 0 else 0

        log.info(
            "batch %d: encoded=%d, skipped=%d, %.0f vec/s, %.1fs",
            batch_num,
            len(to_encode),
            len(works) - len(to_encode),
            throughput,
            elapsed,
        )

        if limit and (batch_num * batch_size) >= limit:
            break

    encoder.unload()

    # Check CH process exit
    proc.wait()
    if proc.returncode != 0:
        assert proc.stderr is not None
        err = proc.stderr.read().decode().strip()
        log.error("CH query failed: %s", err)

    log.info("Done: encoded=%d, skipped=%d", total_encoded, total_skipped)

    # Build indices if we encoded anything
    if total_encoded > 0:
        log.info("Building FTS index...")
        lance.create_fts_index()
        log.info("Building vector index...")
        lance.create_vector_index("vec_retrieval")
        lance.create_vector_index("vec_cluster")
        log.info("Indices built.")
