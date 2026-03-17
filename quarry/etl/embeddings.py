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


def run(batch_size: int = 5000, limit: int | None = None, start_pmid: int = 0):
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
    cursor_pmid = start_pmid

    while True:
        # Keyset pagination: WHERE pmid > cursor avoids full-table OFFSET scan
        papers = db.conn.execute(
            "SELECT pmid, title, abstract FROM papers "
            "WHERE pmid > ? AND abstract != '' AND title != '' AND NOT is_deleted "
            "ORDER BY pmid LIMIT ?",
            [cursor_pmid, batch_size],
        ).fetchall()

        if not papers:
            break

        # Advance cursor to last pmid in this batch
        cursor_pmid = papers[-1][0]
        papers = [{"pmid": str(r[0]), "title": r[1], "abstract": r[2]} for r in papers]

        # Normalize and compute blake3 hashes
        for p in papers:
            p["title"] = normalize_text(p["title"])
            p["abstract"] = normalize_text(p["abstract"])

        hashes = {p["pmid"]: content_hash(p["title"], p["abstract"]) for p in papers}

        # Check existing hashes in LanceDB
        ids = list(hashes.keys())
        try:
            existing = lance.existing_hashes(ids)
        except Exception:
            existing = {}

        # Filter to only new/changed papers
        to_encode = [p for p in papers if hashes[p["pmid"]] != existing.get(p["pmid"])]

        if not to_encode:
            total_skipped += len(papers)
            batch_num += 1
            if limit and (batch_num * batch_size) >= limit:
                break
            continue

        # Encode
        texts = [f"{p['title']}. {p['abstract']}" for p in to_encode]
        t0 = time.time()
        vec_ret = encoder.encode_passages(texts)
        vec_clust = encoder.encode_clustering(texts)
        elapsed = time.time() - t0

        # Build rows for LanceDB upsert
        rows = []
        for i, p in enumerate(to_encode):
            rows.append(
                {
                    "pmid": p["pmid"],
                    "content_hash": hashes[p["pmid"]],
                    "title": p["title"],
                    "abstract": p["abstract"],
                    "vec_retrieval": vec_ret[i].tolist(),
                    "vec_cluster": vec_clust[i].tolist(),
                }
            )

        lance.upsert(rows)

        total_encoded += len(to_encode)
        total_skipped += len(papers) - len(to_encode)
        batch_num += 1
        throughput = len(texts) / elapsed if elapsed > 0 else 0

        print(
            f"  batch {batch_num}: encoded={len(to_encode)}, "
            f"skipped={len(papers) - len(to_encode)}, "
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
