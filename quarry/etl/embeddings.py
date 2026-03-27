"""Batch-encode papers from Parquet → LanceDB with blake3 content caching.

Reads works from Parquet via PyArrow C++ engine (column pruning + predicate
pushdown), skips unchanged (blake3 match), encodes new/modified papers with
jina-v5 nano, upserts to LanceDB.
"""

import logging
import time
from pathlib import Path

try:
    import pyarrow.dataset as ds
except ImportError:
    raise ImportError("pip install quarry[elt]") from None

from quarry.config import settings
from quarry.embed.jina import JinaEncoder
from quarry.store.lance import LanceStore

log = logging.getLogger(__name__)


def _parquet_batches(batch_size: int = 5000):
    """Yield dicts from works Parquet via streaming (no full table load).

    Uses pyarrow.dataset Scanner for constant memory (~50MB per batch)
    instead of pq.read_table which loads everything into RAM (~60-80GB).
    """
    works_dir = Path(settings.parquet_dir) / "works"
    dataset = ds.dataset(works_dir, format="parquet", partitioning="hive")
    scanner = dataset.scanner(
        columns=["work_id", "title", "abstract", "content_hash"],
        filter=(
            ds.field("tier").isin(["t1", "t2"])
            & ds.field("abstract").is_valid()
            & ds.field("title").is_valid()
        ),
        batch_size=batch_size,
    )
    for batch in scanner.to_batches():
        if len(batch) == 0:
            continue
        d = batch.to_pydict()
        yield [
            {
                "work_id": d["work_id"][i],
                "title": d["title"][i],
                "abstract": d["abstract"][i],
                "content_hash": d["content_hash"][i],
            }
            for i in range(len(d["work_id"]))
            if d["title"][i] and d["abstract"][i]
        ]


def run(batch_size: int = 5000, limit: int | None = None, logger=None):
    if logger is None:
        logger = log
    lance = LanceStore(settings.lancedb_uri)

    # Ensure table exists
    try:
        lance.table
    except Exception:
        lance.create_table()

    encoder = JinaEncoder(dim=256)

    total_encoded = 0
    total_skipped = 0
    batch_num = 0

    for works in _parquet_batches(batch_size):
        if not works:
            continue

        hashes = {w["work_id"]: w["content_hash"] for w in works}

        # Check existing hashes in LanceDB
        ids = list(hashes.keys())
        try:
            existing = lance.existing_hashes(ids)
        except Exception as exc:
            logger.warning("existing_hashes failed (%s), re-encoding all", exc)
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

        logger.info(
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

    logger.info("Done: encoded=%d, skipped=%d", total_encoded, total_skipped)

    # Build indices if we encoded anything
    if total_encoded > 0:
        logger.info("Building FTS index...")
        lance.create_fts_index()
        logger.info("Building scalar index on content_hash...")
        lance.create_scalar_index("content_hash")
        logger.info("Building vector index...")
        lance.create_vector_index("vec_retrieval")
        lance.create_vector_index("vec_cluster")
        logger.info("Indices built.")
