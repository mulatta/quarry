"""LanceDB store for vector ANN + BM25 full-text + hybrid search.

Key API patterns (lancedb>=0.29):
  - vector: search(vec, vector_column_name="col").metric("cosine").limit(N)
  - fts: search("query", query_type="fts").limit(N)
  - hybrid: search(query_type="hybrid", vector_column_name="col")
            .vector(vec).text("query").rerank(RRFReranker()).limit(N)
  - upsert: merge_insert("key").when_matched_update_all()
            .when_not_matched_insert_all().execute(data)
"""

import re

try:
    import lancedb
    import numpy as np
    import pyarrow as pa
    from lancedb.rerankers import RRFReranker
except ImportError:
    raise ImportError("pip install lancedb numpy pyarrow") from None

_WORK_ID_RE = re.compile(r"^W\d+$")

# LanceDB table schema (pyarrow) — work_id-based (OpenAlex)
SCHEMA = pa.schema(
    [
        pa.field("work_id", pa.string()),
        pa.field("content_hash", pa.binary(32)),  # blake3(title + abstract)
        pa.field("title", pa.string()),
        pa.field("abstract", pa.string()),
        pa.field("vec_retrieval", pa.list_(pa.float32(), 256)),
        pa.field("vec_cluster", pa.list_(pa.float32(), 256)),
    ]
)

TABLE_NAME = "papers"


class LanceStore:
    def __init__(self, uri: str):
        self._db = lancedb.connect(uri)

    @property
    def table(self) -> lancedb.table.Table:
        return self._db.open_table(TABLE_NAME)

    def create_table(self, data: pa.Table | None = None) -> lancedb.table.Table:
        """Create papers table. Overwrites if exists."""
        if data is not None:
            return self._db.create_table(TABLE_NAME, data=data, mode="overwrite")
        return self._db.create_table(TABLE_NAME, schema=SCHEMA, mode="overwrite")

    def add(self, data: list[dict] | pa.Table):
        """Append rows to the table."""
        self.table.add(data)

    def upsert(self, data: list[dict] | pa.Table):
        """Insert or update rows by work_id."""
        self.table.merge_insert(
            "work_id"
        ).when_matched_update_all().when_not_matched_insert_all().execute(data)

    def delete_work_ids(self, work_ids: list[str]):
        """Delete rows by work_id list."""
        if not work_ids:
            return
        _validate_work_ids(work_ids)
        id_list = ", ".join(f"'{w}'" for w in work_ids)
        self.table.delete(f"work_id IN ({id_list})")

    def existing_hashes(self, ids: list[str]) -> dict[str, bytes]:
        """Get content_hash for existing IDs (for blake3 cache check)."""
        _validate_work_ids(ids)
        id_list = ", ".join(f"'{i}'" for i in ids)
        result = (
            self.table.search()
            .where(f"work_id IN ({id_list})")
            .select(["work_id", "content_hash"])
            .limit(len(ids))
            .to_list()
        )
        return {r["work_id"]: bytes(r["content_hash"]) for r in result}

    def existing_hashes_all(self, batch_size: int = 100_000) -> dict[str, bytes]:
        """Return all (work_id → content_hash) pairs from LanceDB.

        Streams in batches for bounded memory.
        """
        result: dict[str, bytes] = {}
        scanner = self.table.to_lance().scanner(
            columns=["work_id", "content_hash"], batch_size=batch_size
        )
        for batch in scanner.to_batches():
            wids = batch.column("work_id").to_pylist()
            hashes = batch.column("content_hash").to_pylist()
            for wid, h in zip(wids, hashes):
                result[wid] = bytes(h)
        return result

    def all_work_ids(self, batch_size: int = 100_000) -> set[str]:
        """Return the full set of work_ids stored in LanceDB.

        Streams in batches to avoid loading entire table into memory at once.
        """
        ids: set[str] = set()
        scanner = self.table.to_lance().scanner(
            columns=["work_id"], batch_size=batch_size
        )
        for batch in scanner.to_batches():
            ids.update(batch.column("work_id").to_pylist())
        return ids

    def delete_work_ids_batch(self, work_ids: set[str], batch_size: int = 10_000):
        """Delete orphan work_ids in batches."""
        if not work_ids:
            return 0
        items = list(work_ids)
        deleted = 0
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            _validate_work_ids(chunk)
            id_list = ", ".join(f"'{w}'" for w in chunk)
            self.table.delete(f"work_id IN ({id_list})")
            deleted += len(chunk)
        return deleted

    def optimize(self):
        """Compact fragments + prune old versions. Call periodically during long runs."""
        from datetime import timedelta

        self.table.optimize(cleanup_older_than=timedelta(seconds=0))

    def create_fts_index(self, column: str):
        """Build BM25 full-text index on a single column.

        Native FTS requires one index per column; queries can still span
        multiple columns via `fts_columns=[...]` at search time.
        """
        self.table.create_fts_index(column, replace=True)

    def create_scalar_index(self, column: str = "work_id"):
        """Build b-tree scalar index for fast equality lookups."""
        self.table.create_scalar_index(column, index_type="BTREE", replace=True)

    def create_vector_index(
        self, column: str = "vec_retrieval", accelerator: str | None = None
    ):
        """Build IVF_PQ vector index for ANN search.

        accelerator: pass "cuda" for GPU-accelerated k-means (5–10x faster).

        num_sub_vectors=16 assumes 256-dim Jina embeddings (16-dim per subvec).
        Required explicitly because the CUDA path doesn't handle None default
        (CPU path auto-computes from dim, but accelerator path raises TypeError).
        """
        kwargs: dict = {
            "metric": "cosine",
            "vector_column_name": column,
            "num_sub_vectors": 16,
        }
        if accelerator:
            kwargs["accelerator"] = accelerator
        self.table.create_index(**kwargs)

    # -- Search methods --

    def vector_search(
        self, query_vec: np.ndarray, limit: int = 20, column: str = "vec_retrieval"
    ) -> list[dict]:
        """ANN similarity search."""
        return (
            self.table.search(query_vec, vector_column_name=column)
            .metric("cosine")
            .limit(limit)
            .to_list()
        )

    def text_search(self, query: str, limit: int = 20) -> list[dict]:
        """BM25 full-text search on title + abstract."""
        return (
            self.table.search(
                query, query_type="fts", fts_columns=["title", "abstract"]
            )
            .limit(limit)
            .to_list()
        )

    def hybrid_search(
        self,
        query_vec: np.ndarray,
        query_text: str,
        limit: int = 20,
        column: str = "vec_retrieval",
    ) -> list[dict]:
        """Hybrid search: ANN + BM25 fused with RRF."""
        return (
            self.table.search(
                query_type="hybrid",
                vector_column_name=column,
                fts_columns=["title", "abstract"],
            )
            .vector(query_vec)
            .text(query_text)
            .rerank(RRFReranker())
            .limit(limit)
            .to_list()
        )


def _validate_work_ids(ids: list[str]) -> None:
    """Reject work_ids that don't match OpenAlex format (W followed by digits)."""
    for wid in ids:
        if not _WORK_ID_RE.match(wid):
            raise ValueError(f"Invalid work_id format: {wid!r}")
