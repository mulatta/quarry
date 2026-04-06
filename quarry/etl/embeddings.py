"""Batch-encode papers from CH → LanceDB with content-hash caching.

Incremental update via ClickHouse:
  1. Export LanceDB hashes to CH temp table (~1.4GB, 2-3min)
  2. CH JOIN: works_export vs lance_hashes → diff work_ids (~3min)
  3. Fetch full text from CH for diff IDs, encode on GPU, upsert
  4. Orphan GC: lance_hashes LEFT JOIN works_export → delete missing

content_hash is pre-computed by CH (BLAKE3) during ch_transform.
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

_OPTIMIZE_EVERY = 100
_PREFETCH_DEPTH = 3

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
        "--receive_timeout",
        "7200",
        "--send_timeout",
        "7200",
    ]


def _ch_exec(query: str) -> str:
    """Execute CH query, return stdout. Raises on error."""
    proc = subprocess.run(
        _ch_cmd() + ["--query", query],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"CH query failed: {proc.stderr.strip()}")
    return proc.stdout


def _load_ids_to_ch(ids: list[str], table: str, logger) -> int:
    """Load work_id list into CH table via RowBinary pipe."""
    _ch_exec(
        f"CREATE TABLE IF NOT EXISTS {table} ("
        f"  work_id String"
        f") ENGINE = MergeTree() ORDER BY work_id"
    )
    _ch_exec(f"TRUNCATE TABLE {table}")

    proc = subprocess.Popen(
        _ch_cmd() + ["--query", f"INSERT INTO {table} FORMAT RowBinary"],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None

    for wid in ids:
        wid_bytes = wid.encode("utf-8")
        length = len(wid_bytes)
        while length >= 0x80:
            proc.stdin.write(bytes([length & 0x7F | 0x80]))
            length >>= 7
        proc.stdin.write(bytes([length]))
        proc.stdin.write(wid_bytes)

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        assert proc.stderr is not None
        raise RuntimeError(f"CH INSERT failed: {proc.stderr.read().decode()}")

    logger.info("[CH] Loaded %d IDs to %s", len(ids), table)
    return len(ids)


# ── Step 1: Load LanceDB hashes into CH temp table ──


def _load_lance_hashes_to_ch(lance: LanceStore, logger) -> int:
    """Export LanceDB (work_id, content_hash) → CH temp table via pipe."""
    logger.info("[Diff] Creating CH temp table for LanceDB hashes...")
    _ch_exec(
        "CREATE TABLE IF NOT EXISTS _tmp_lance_hashes ("
        "  work_id String,"
        "  content_hash FixedString(32)"
        ") ENGINE = MergeTree() ORDER BY work_id"
    )
    _ch_exec("TRUNCATE TABLE _tmp_lance_hashes")

    # Stream from LanceDB → CH INSERT via pipe
    logger.info("[Diff] Exporting LanceDB hashes to CH...")
    proc = subprocess.Popen(
        _ch_cmd()
        + [
            "--query",
            "INSERT INTO _tmp_lance_hashes FORMAT RowBinary",
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None

    count = 0
    for batch in (
        lance.table.to_lance()
        .scanner(columns=["work_id", "content_hash"], batch_size=100_000)
        .to_batches()
    ):
        wids = batch.column("work_id").to_pylist()
        hashes = batch.column("content_hash").to_pylist()
        for wid, h in zip(wids, hashes):
            if not wid or h is None:
                continue
            wid_bytes = wid.encode("utf-8")
            # RowBinary: String = varint_len + bytes, FixedString(32) = 32 raw bytes
            # Varint encoding for string length
            length = len(wid_bytes)
            while length >= 0x80:
                proc.stdin.write(bytes([length & 0x7F | 0x80]))
                length >>= 7
            proc.stdin.write(bytes([length]))
            proc.stdin.write(wid_bytes)
            proc.stdin.write(bytes(h))
            count += 1

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        assert proc.stderr is not None
        raise RuntimeError(f"CH INSERT failed: {proc.stderr.read().decode()}")

    logger.info("[Diff] Loaded %d LanceDB hashes to CH", count)
    return count


# ── Step 2: CH JOIN → diff ──


def _compute_diff_in_ch(logger) -> tuple[list[str], list[str], int]:
    """CH-side JOIN: works_export vs _tmp_lance_hashes → diff + orphans.

    Content-hash dedup: only one representative work_id per content_hash
    is encoded. Journal variants (same title+abstract, different work_id)
    share a content_hash — encoding any one is sufficient.

    Returns (to_encode_ids, orphan_ids, skipped_count).
    """
    where = _embed_where()

    # Pre-compute representative work_ids (one per content_hash).
    # Reused by both diff and orphan queries to avoid duplicate window function.
    logger.info("[Diff] Building representative work_id table...")
    _ch_exec(
        "CREATE TABLE IF NOT EXISTS _tmp_representative ("
        "  work_id String"
        ") ENGINE = MergeTree() ORDER BY work_id"
    )
    _ch_exec("TRUNCATE TABLE _tmp_representative")
    _ch_exec(
        f"INSERT INTO _tmp_representative "
        f"SELECT work_id FROM ("
        f"  SELECT work_id,"
        f"  row_number() OVER (PARTITION BY content_hash ORDER BY work_id) AS rn"
        f"  FROM works_export"
        f"  WHERE {where}"
        f") WHERE rn = 1"
    )
    rep_count = int(_ch_exec("SELECT count() FROM _tmp_representative").strip())
    logger.info("[Diff] %d representative work_ids (unique content_hashes)", rep_count)

    # New + changed: representative work_ids not yet in LanceDB with matching hash
    logger.info("[Diff] Computing diff via CH JOIN...")
    diff_result = _ch_exec(
        "SELECT r.work_id FROM _tmp_representative r "
        "WHERE r.work_id NOT IN ("
        "  SELECT l.work_id FROM _tmp_lance_hashes l "
        "  INNER JOIN works_export w ON l.work_id = w.work_id "
        f"  WHERE w.content_hash = l.content_hash AND {where}"
        ")"
    )
    to_encode_ids = [
        wid.strip() for wid in diff_result.strip().split("\n") if wid.strip()
    ]

    # Total = representative count; skipped = already in LanceDB with correct hash
    skipped = rep_count - len(to_encode_ids)

    # Orphans: in LanceDB but not a representative work_id.
    # This catches both: (a) works removed from works_export, and
    # (b) non-representative duplicates from before dedup was added.
    logger.info("[Diff] Computing orphans via CH JOIN...")
    orphan_result = _ch_exec(
        "SELECT l.work_id FROM _tmp_lance_hashes l "
        "WHERE l.work_id NOT IN ("
        "  SELECT work_id FROM _tmp_representative"
        ")"
    )
    orphan_ids = [
        wid.strip() for wid in orphan_result.strip().split("\n") if wid.strip()
    ]

    logger.info(
        "[Diff] to_encode=%d, skipped=%d, orphans=%d",
        len(to_encode_ids),
        skipped,
        len(orphan_ids),
    )
    return to_encode_ids, orphan_ids, skipped


# ── Step 3: CH fetch + encode ──


def _prefetch(encode_batch: int, logger):
    """Background I/O: fetch full text for diff work_ids from CH.

    Reads from _tmp_encode_ids JOIN works_export — no IN clause needed.
    All IDs already in CH temp table.
    """
    q: queue.Queue = queue.Queue(maxsize=_PREFETCH_DEPTH)

    def _producer():
        try:
            where = _embed_where()
            query = (
                f"SELECT w.work_id, w.title, w.abstract,"
                f" hex(w.content_hash) AS hex_hash"
                f" FROM works_export w"
                f" INNER JOIN _tmp_encode_ids e ON w.work_id = e.work_id"
                f" WHERE {where}"
            )

            proc = subprocess.Popen(
                _ch_cmd()
                + [
                    "--query",
                    query + " FORMAT TabSeparatedWithNames",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert proc.stdout is not None

            buf: list[dict] = []
            buf_hashes: dict[str, bytes] = {}

            proc.stdout.readline()  # skip TSV header
            for line in proc.stdout:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                wid, title, abstract, hex_hash = (
                    parts[0],
                    parts[1],
                    parts[2],
                    parts[3],
                )
                if not title or not abstract or title == "\\N" or abstract == "\\N":
                    continue

                buf.append({"work_id": wid, "title": title, "abstract": abstract})
                buf_hashes[wid] = bytes.fromhex(hex_hash)

                if len(buf) >= encode_batch:
                    q.put((list(buf), dict(buf_hashes)))
                    buf.clear()
                    buf_hashes.clear()

            proc.wait()
            if proc.returncode != 0:
                assert proc.stderr is not None
                raise RuntimeError(f"CH fetch failed: {proc.stderr.read()}")

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


# ── Cleanup ──


def _cleanup_temp(logger):
    """Drop CH temp tables."""
    for table in ["_tmp_lance_hashes", "_tmp_encode_ids", "_tmp_representative"]:
        try:
            _ch_exec(f"DROP TABLE IF EXISTS {table}")
        except Exception as exc:
            logger.warning("Failed to drop %s: %s", table, exc)


# ── Main entry point ──


def run(limit: int | None = None, logger=None):
    if logger is None:
        logger = log
    lance = LanceStore(settings.lancedb_uri)

    # Ensure table exists
    try:
        if lance.table.count_rows() > 0:
            logger.info("Building work_id BTree index for hash lookups...")
            lance.create_scalar_index("work_id")
    except Exception:
        lance.create_table()

    try:
        # Step 1: LanceDB hashes → CH temp table
        lance_count = _load_lance_hashes_to_ch(lance, logger)
        logger.info("[Step 1] Loaded %d LanceDB hashes to CH", lance_count)

        # Step 2: CH JOIN → diff
        to_encode_ids, orphan_ids, total_skipped = _compute_diff_in_ch(logger)

        if limit:
            to_encode_ids = to_encode_ids[:limit]

        if not to_encode_ids and not orphan_ids:
            logger.info(
                "Nothing to do: encoded=0, skipped=%d, orphans=0", total_skipped
            )
            return

        # Step 3: load diff IDs → CH temp table, then fetch + encode
        _load_ids_to_ch(to_encode_ids, "_tmp_encode_ids", logger)

        encoder = JinaEncoder(
            dim=256,
            batch_size=settings.embed_batch_size,
            max_tokens=settings.embed_max_tokens,
        )

        encode_batch = settings.embed_encode_batch
        total_encoded = 0
        batch_num = 0

        for to_encode, hashes in _prefetch(encode_batch, logger):
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
            deleted = lance.delete_work_ids_batch(set(orphan_ids))
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

    finally:
        _cleanup_temp(logger)
