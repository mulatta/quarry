"""Batch-encode papers from CH → LanceDB with content-hash caching.

Two-phase incremental update via ClickHouse:
  Phase 1: Stream (work_id, content_hash) from CH works_export, compare
           against LanceDB hashes in batches to find new/changed/orphan IDs.
           Memory: O(diff_size), not O(total_works).
  Phase 2: Fetch full text from CH for diff work_ids only (index lookup),
           encode on GPU, upsert to LanceDB.

content_hash is pre-computed by CH (BLAKE3) during ch_transform.
No Python hash computation needed — model config is fixed.
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
    """Build CH WHERE clause for embedding-eligible works."""
    types = ", ".join(f"'{t}'" for t in settings.embed_allowed_types)
    return _EMBED_WHERE.format(types=types)


def _ch_cmd() -> list[str]:
    """Base clickhouse-client command."""
    return [
        "clickhouse-client",
        "--host",
        settings.ch_host,
        "--port",
        str(settings.ch_port),
        "--database",
        settings.ch_database,
    ]


def _ch_stream_rows(query: str, columns: list[str]):
    """Stream rows from CH as dicts via TabSeparatedWithNames.

    Yields dicts with string values. Caller must convert types.
    Uses subprocess pipe — zero memory on Python side (streaming).
    """
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


# ── Phase 1: streaming diff computation ──


def _compute_diff(
    lance: LanceStore, logger, batch_size: int = 10_000
) -> tuple[set[str], set[str], int]:
    """Stream CH hashes, compare against LanceDB in batches.

    Memory: O(diff_size + orphan_size), not O(total_works).
    """
    where = _embed_where()
    query = (
        f"SELECT work_id, hex(content_hash) AS hex_hash FROM works_export WHERE {where}"
    )

    logger.info("[Diff] Streaming CH hashes + LanceDB batch comparison...")

    to_encode_ids: set[str] = set()
    seen_ids: set[str] = set()
    skipped = 0
    total_scanned = 0

    batch_wids: list[str] = []
    batch_hashes: dict[str, bytes] = {}

    for row in _ch_stream_rows(query, ["work_id", "hex_hash"]):
        wid = row["work_id"]
        ch_hash = bytes.fromhex(row["hex_hash"])

        batch_wids.append(wid)
        batch_hashes[wid] = ch_hash
        seen_ids.add(wid)

        if len(batch_wids) >= batch_size:
            # Batch lookup against LanceDB
            existing = lance.existing_hashes(batch_wids)
            for w in batch_wids:
                if batch_hashes[w] == existing.get(w):
                    skipped += 1
                else:
                    to_encode_ids.add(w)
            total_scanned += len(batch_wids)
            batch_wids.clear()
            batch_hashes.clear()

    # Flush remaining
    if batch_wids:
        existing = lance.existing_hashes(batch_wids)
        for w in batch_wids:
            if batch_hashes[w] == existing.get(w):
                skipped += 1
            else:
                to_encode_ids.add(w)
        total_scanned += len(batch_wids)

    # Orphan detection: LanceDB IDs not in CH
    logger.info("[Diff] Scanning LanceDB work_ids for orphan detection...")
    lance_ids = lance.all_work_ids()
    orphan_ids = lance_ids - seen_ids

    logger.info(
        "[Diff] scanned=%d, to_encode=%d, skipped=%d, orphans=%d",
        total_scanned,
        len(to_encode_ids),
        skipped,
        len(orphan_ids),
    )
    return to_encode_ids, orphan_ids, skipped


# ── Phase 2: CH fetch + encode ──


def _prefetch(to_encode_ids: set[str], encode_batch: int, logger):
    """Background I/O: fetch full text for diff work_ids from CH.

    Yields (works_batch, hashes_batch) tuples of size encode_batch.
    Uses CH WHERE work_id IN (...) for index lookup — minimal I/O.
    """
    q: queue.Queue = queue.Queue(maxsize=_PREFETCH_DEPTH)

    def _producer():
        try:
            where = _embed_where()
            ids_list = list(to_encode_ids)
            buf: list[dict] = []
            buf_hashes: dict[str, bytes] = {}

            # Process in chunks to avoid too-large IN clauses
            chunk_size = 50_000
            for i in range(0, len(ids_list), chunk_size):
                chunk = ids_list[i : i + chunk_size]
                id_csv = ", ".join(f"'{w}'" for w in chunk)
                query = (
                    f"SELECT work_id, title, abstract, hex(content_hash) AS hex_hash "
                    f"FROM works_export WHERE {where} AND work_id IN ({id_csv})"
                )

                for row in _ch_stream_rows(
                    query, ["work_id", "title", "abstract", "hex_hash"]
                ):
                    wid = row["work_id"]
                    title = row["title"]
                    abstract = row["abstract"]
                    ch = bytes.fromhex(row["hex_hash"])

                    if not title or not abstract or title == "\\N" or abstract == "\\N":
                        continue

                    buf.append({"work_id": wid, "title": title, "abstract": abstract})
                    buf_hashes[wid] = ch

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

    # Phase 1: streaming diff (CH hashes vs LanceDB, batch comparison)
    to_encode_ids, orphan_ids, total_skipped = _compute_diff(lance, logger)

    if limit:
        to_encode_ids = set(list(to_encode_ids)[:limit])

    if not to_encode_ids and not orphan_ids:
        logger.info("Nothing to do: encoded=0, skipped=%d, orphans=0", total_skipped)
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

    # Orphan GC
    if orphan_ids and not limit:
        logger.info("[GC] Deleting %d orphans...", len(orphan_ids))
        deleted = lance.delete_work_ids_batch(orphan_ids)
        logger.info("[GC] Deleted %d orphans", deleted)

    dirty = total_encoded > 0 or bool(orphan_ids)

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
