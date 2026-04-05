"""Batch-encode papers from CH → LanceDB with content-hash caching.

Two-phase incremental update via ClickHouse:
  Phase 1: Stream (work_id, content_hash) from CH via ArrowStream,
           compare against LanceDB in batches (producer-consumer pipeline).
           Memory: O(diff_size), not O(total_works).
  Phase 2: Fetch full text from CH for diff work_ids only (index lookup),
           encode on GPU in accumulated batches, upsert to LanceDB.

content_hash is pre-computed by CH (BLAKE3) during ch_transform.
Orphan GC runs separately — not part of Phase 1 hot path.
"""

import logging
import queue
import subprocess
import threading
import time

from quarry.config import settings
from quarry.embed.jina import JinaEncoder
from quarry.store.lance import LanceStore

log = logging.getLogger(__name__)

_PREFETCH_DEPTH = 3
_OPTIMIZE_EVERY = 100

# Embedding-eligible filter (mirrors parquet tier/type/language filter).
_EMBED_WHERE = (
    "tier IN ('t1', 't2')"
    " AND type IN ({types})"
    " AND abstract IS NOT NULL AND abstract != ''"
    " AND title IS NOT NULL AND title != ''"
    " AND is_retracted = false"
    " AND (language = 'en' OR language IS NULL)"
)


def _embed_where() -> str:
    types = ", ".join(f"'{t}'" for t in settings.embed_allowed_types)
    return _EMBED_WHERE.format(types=types)


def _ch_cmd() -> list[str]:
    return [
        "clickhouse-client",
        "--host",
        settings.ch_host,
        "--port",
        str(settings.ch_port),
        "--database",
        settings.ch_database,
    ]


# ── Phase 1: streaming diff via ArrowStream ──


def _compute_diff(
    lance: LanceStore, logger, batch_size: int = 50_000
) -> tuple[set[str], int]:
    """Stream CH hashes via ArrowStream, compare against LanceDB in pipeline.

    Producer: reads ArrowStream RecordBatches from CH (C++ decoding).
    Consumer: batch-lookups LanceDB hashes, computes diff.
    Memory: O(diff_size) — no seen_ids set, no full hash dict.

    Returns (to_encode_ids, skipped_count).
    """
    import pyarrow.ipc as ipc

    where = _embed_where()
    query = f"SELECT work_id, content_hash FROM works_export WHERE {where}"

    logger.info("[Diff] Starting CH ArrowStream + LanceDB pipeline comparison...")

    # Start CH process with ArrowStream output
    proc = subprocess.Popen(
        _ch_cmd() + ["--query", query + " FORMAT ArrowStream"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None

    reader = ipc.RecordBatchStreamReader(proc.stdout)

    # Producer-consumer: CH read → queue → LanceDB lookup
    batch_queue: queue.Queue = queue.Queue(maxsize=_PREFETCH_DEPTH)
    to_encode_ids: set[str] = set()
    result_lock = threading.Lock()
    skipped = 0
    total_scanned = 0

    _consumer_error: list[BaseException] = []

    def _consumer():
        nonlocal skipped, total_scanned
        try:
            while True:
                item = batch_queue.get()
                if item is None:
                    break
                wids, ch_hashes = item

                # Batch lookup against LanceDB
                existing = lance.existing_hashes(wids)

                local_encode = []
                local_skip = 0
                for wid, ch_h in zip(wids, ch_hashes):
                    if ch_h == existing.get(wid):
                        local_skip += 1
                    else:
                        local_encode.append(wid)

                with result_lock:
                    to_encode_ids.update(local_encode)
                    skipped += local_skip
                    total_scanned += len(wids)
        except Exception as exc:
            _consumer_error.append(exc)

    consumer_thread = threading.Thread(target=_consumer, daemon=True)
    consumer_thread.start()

    # Producer: read ArrowStream batches, send to consumer
    try:
        buf_wids: list[str] = []
        buf_hashes: list[bytes] = []

        for batch in reader:
            wids = batch.column("work_id").to_pylist()
            hashes = batch.column("content_hash").to_pylist()

            for wid, h in zip(wids, hashes):
                if not wid or h is None:
                    continue
                buf_wids.append(wid)
                buf_hashes.append(bytes(h))

                if len(buf_wids) >= batch_size:
                    batch_queue.put((list(buf_wids), list(buf_hashes)))
                    buf_wids.clear()
                    buf_hashes.clear()

        if buf_wids:
            batch_queue.put((list(buf_wids), list(buf_hashes)))
    finally:
        batch_queue.put(None)
        consumer_thread.join()

    proc.wait()
    if proc.returncode != 0:
        assert proc.stderr is not None
        raise RuntimeError(f"CH query failed: {proc.stderr.read().decode()}")

    if _consumer_error:
        raise _consumer_error[0]

    logger.info(
        "[Diff] scanned=%d, to_encode=%d, skipped=%d",
        total_scanned,
        len(to_encode_ids),
        skipped,
    )
    return to_encode_ids, skipped


# ── Orphan GC (separate from Phase 1) ──


def _gc_orphans(lance: LanceStore, logger) -> int:
    """Detect and delete orphan LanceDB entries not in CH.

    Streams work_ids from CH (lightweight — no content_hash) and compares
    against LanceDB work_id set.
    """
    import pyarrow.ipc as ipc

    where = _embed_where()
    query = f"SELECT work_id FROM works_export WHERE {where}"

    logger.info("[GC] Streaming CH work_ids...")
    proc = subprocess.Popen(
        _ch_cmd() + ["--query", query + " FORMAT ArrowStream"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None

    ch_ids: set[str] = set()
    for batch in ipc.RecordBatchStreamReader(proc.stdout):
        ch_ids.update(batch.column("work_id").to_pylist())

    proc.wait()
    if proc.returncode != 0:
        assert proc.stderr is not None
        raise RuntimeError(f"CH query failed: {proc.stderr.read().decode()}")

    logger.info("[GC] CH: %d work_ids", len(ch_ids))

    logger.info("[GC] Scanning LanceDB work_ids...")
    lance_ids = lance.all_work_ids()
    logger.info("[GC] LanceDB: %d work_ids", len(lance_ids))

    orphans = lance_ids - ch_ids
    if not orphans:
        logger.info("[GC] No orphans found")
        return 0

    logger.info("[GC] Deleting %d orphans...", len(orphans))
    deleted = lance.delete_work_ids_batch(orphans)
    logger.info("[GC] Deleted %d orphans", deleted)
    return deleted


# ── Phase 2: CH fetch + encode ──


def _ch_stream_rows(query: str, columns: list[str]):
    """Stream rows from CH as dicts via TabSeparatedWithNames."""
    proc = subprocess.Popen(
        _ch_cmd() + ["--query", query + " FORMAT TabSeparatedWithNames"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None

    header = proc.stdout.readline().strip().split("\t")
    if header != columns:
        proc.kill()
        raise ValueError(f"Column mismatch: expected {columns}, got {header}")

    for line in proc.stdout:
        vals = line.rstrip("\n").split("\t")
        yield dict(zip(columns, vals))

    proc.wait()
    if proc.returncode != 0:
        assert proc.stderr is not None
        raise RuntimeError(f"CH query failed: {proc.stderr.read()}")


def _prefetch(to_encode_ids: set[str], encode_batch: int, logger):
    """Background I/O: fetch full text for diff work_ids from CH.

    Yields (works_batch, hashes_batch) tuples of size encode_batch.
    Uses CH WHERE work_id IN (...) for index lookup.
    """
    q: queue.Queue = queue.Queue(maxsize=_PREFETCH_DEPTH)

    def _producer():
        try:
            where = _embed_where()
            ids_list = list(to_encode_ids)
            buf: list[dict] = []
            buf_hashes: dict[str, bytes] = {}

            chunk_size = 50_000
            for i in range(0, len(ids_list), chunk_size):
                chunk = ids_list[i : i + chunk_size]
                id_csv = ", ".join(f"'{w}'" for w in chunk)
                query = (
                    f"SELECT work_id, title, abstract,"
                    f" hex(content_hash) AS hex_hash"
                    f" FROM works_export WHERE {where}"
                    f" AND work_id IN ({id_csv})"
                )

                for row in _ch_stream_rows(
                    query, ["work_id", "title", "abstract", "hex_hash"]
                ):
                    wid = row["work_id"]
                    title = row["title"]
                    abstract = row["abstract"]

                    if not title or not abstract or title == "\\N" or abstract == "\\N":
                        continue

                    buf.append({"work_id": wid, "title": title, "abstract": abstract})
                    buf_hashes[wid] = bytes.fromhex(row["hex_hash"])

                    if len(buf) >= encode_batch:
                        q.put((list(buf), dict(buf_hashes)))
                        buf.clear()
                        buf_hashes.clear()

            if buf:
                q.put((list(buf), dict(buf_hashes)))
        except Exception as exc:
            logger.exception("prefetch producer failed")
            q.put(exc)
        finally:
            q.put(None)

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, BaseException):
            thread.join()
            raise item
        yield item

    thread.join()


# ── Main entry point ──


def run(batch_size: int | None = None, limit: int | None = None, logger=None):
    if logger is None:
        logger = log
    lance = LanceStore(settings.lancedb_uri)

    # Ensure table exists; build work_id index for fast hash lookups
    try:
        if lance.table.count_rows() > 0:
            logger.info("Building work_id BTree index for hash lookups...")
            lance.create_scalar_index("work_id")
    except Exception:
        lance.create_table()

    # Phase 1: streaming diff (ArrowStream + producer-consumer)
    to_encode_ids, total_skipped = _compute_diff(lance, logger)

    if limit:
        to_encode_ids = set(list(to_encode_ids)[:limit])

    if not to_encode_ids:
        logger.info("Nothing to encode: skipped=%d", total_skipped)
        # Still run orphan GC
        orphan_count = _gc_orphans(lance, logger) if not limit else 0
        if orphan_count > 0:
            lance.optimize()
        return

    # Phase 2: fetch from CH + encode (only changed works)
    encoder = JinaEncoder(
        dim=256,
        batch_size=settings.embed_batch_size,
        max_tokens=settings.embed_max_tokens,
    )

    encode_batch = settings.embed_encode_batch
    total_encoded = 0
    batch_num = 0

    for to_encode, hashes in _prefetch(to_encode_ids, encode_batch, logger):
        texts = [f"{w['title']}. {w['abstract']}" for w in to_encode]
        t0 = time.time()
        vec_ret = encoder.encode_passages(texts)
        vec_clust = encoder.encode_clustering(texts)
        elapsed = time.time() - t0

        lance_rows = [
            {
                "work_id": w["work_id"],
                "content_hash": hashes[w["work_id"]],
                "title": w["title"],
                "abstract": w["abstract"],
                "vec_retrieval": vec_ret[i].tolist(),
                "vec_cluster": vec_clust[i].tolist(),
            }
            for i, w in enumerate(to_encode)
        ]

        lance.upsert(lance_rows)

        total_encoded += len(to_encode)
        batch_num += 1
        throughput = len(texts) / elapsed if elapsed > 0 else 0

        logger.info(
            "batch %d: encoded=%d, total=%d/%d, %.0f vec/s, %.1fs",
            batch_num,
            len(to_encode),
            total_encoded,
            len(to_encode_ids),
            throughput,
            elapsed,
        )

        if batch_num % _OPTIMIZE_EVERY == 0:
            logger.info("batch %d: optimizing LanceDB...", batch_num)
            lance.optimize()

    encoder.unload()

    logger.info("Done: encoded=%d, skipped=%d", total_encoded, total_skipped)

    # Orphan GC (separate phase)
    orphan_count = _gc_orphans(lance, logger) if not limit else 0

    dirty = total_encoded > 0 or orphan_count > 0

    if dirty:
        logger.info("Final LanceDB optimize...")
        lance.optimize()
        logger.info("Building FTS index...")
        lance.create_fts_index()
        logger.info("Building scalar index on work_id...")
        lance.create_scalar_index("work_id")
        logger.info("Building vector index...")
        lance.create_vector_index("vec_retrieval")
        lance.create_vector_index("vec_cluster")
        logger.info("Indices built.")
