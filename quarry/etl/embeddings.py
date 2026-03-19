"""Batch-encode papers from DuckDB → LanceDB with blake3 content caching.

Reads papers in batches from DuckDB, skips unchanged (blake3 match), encodes
new/modified papers with jina-v5 nano, upserts to LanceDB.
"""

import re
import time

import blake3

from quarry.config import settings
from quarry.embed.jina import JinaEncoder
from quarry.store.duckdb import DuckDBStore
from quarry.store.lance import LanceStore

# Strip HTML tags while preserving math inequalities (< 0.05, > 18, etc.)
_HTML_TAG = re.compile(
    r"</?\w[\w.-]*"
    r'(?:\s+\w[\w.-]*(?:\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s>]*))?)*'
    r"\s*/?>"
)
_MULTI_SP = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Strip HTML tags, collapse whitespace."""
    text = _HTML_TAG.sub(" ", text)
    text = _MULTI_SP.sub(" ", text)
    return text.strip()


def content_hash(title: str, abstract: str) -> bytes:
    """blake3 hash of normalized title + abstract for change detection."""
    return blake3.blake3(f"{title}\n{abstract}".encode()).digest()


def run(batch_size: int = 5000, limit: int | None = None, start_work_id: str = ""):
    db = DuckDBStore()
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
    cursor = start_work_id

    while True:
        # Keyset pagination on work_id (T1+T2 only, with abstract)
        works = db.conn.execute(
            "SELECT work_id, title, abstract FROM works "
            "WHERE work_id > ? AND tier IN ('t1', 't2') "
            "AND abstract IS NOT NULL AND abstract != '' AND title != '' "
            "ORDER BY work_id LIMIT ?",
            [cursor, batch_size],
        ).fetchall()

        if not works:
            break

        # Advance cursor to last work_id in this batch
        cursor = works[-1][0]
        works = [{"work_id": r[0], "title": r[1], "abstract": r[2]} for r in works]

        # Normalize and compute blake3 hashes
        for w in works:
            w["title"] = normalize_text(w["title"])
            w["abstract"] = normalize_text(w["abstract"])

        hashes = {w["work_id"]: content_hash(w["title"], w["abstract"]) for w in works}

        # Check existing hashes in LanceDB
        ids = list(hashes.keys())
        try:
            existing = lance.existing_hashes(ids)
        except Exception:
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
        rows = []
        for i, w in enumerate(to_encode):
            rows.append(
                {
                    "work_id": w["work_id"],
                    "content_hash": hashes[w["work_id"]],
                    "title": w["title"],
                    "abstract": w["abstract"],
                    "vec_retrieval": vec_ret[i].tolist(),
                    "vec_cluster": vec_clust[i].tolist(),
                }
            )

        lance.upsert(rows)

        total_encoded += len(to_encode)
        total_skipped += len(works) - len(to_encode)
        batch_num += 1
        throughput = len(texts) / elapsed if elapsed > 0 else 0

        print(
            f"  batch {batch_num}: encoded={len(to_encode)}, "
            f"skipped={len(works) - len(to_encode)}, "
            f"{throughput:.0f} vec/s, {elapsed:.1f}s"
        )

        if limit and (batch_num * batch_size) >= limit:
            break

    encoder.unload()
    db.close()

    print(f"\nDone: encoded={total_encoded}, skipped={total_skipped}")

    # Build indices if we encoded anything
    if total_encoded > 0:
        print("Building FTS index...")
        lance.create_fts_index()
        print("Building vector index...")
        lance.create_vector_index("vec_retrieval")
        lance.create_vector_index("vec_cluster")
        print("Indices built.")
