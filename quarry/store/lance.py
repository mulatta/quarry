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
    raise ImportError("pip install quarry[server]") from None

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

    def optimize(self):
        """Compact fragments + prune old versions. Call periodically during long runs."""
        from datetime import timedelta

        self.table.optimize(cleanup_older_than=timedelta(seconds=0))

    def create_fts_index(self):
        """Build BM25 full-text index on title and abstract."""
        self.table.create_fts_index(["title", "abstract"], replace=True)

    def create_scalar_index(self, column: str = "work_id"):
        """Build b-tree scalar index for fast equality lookups."""
        self.table.create_scalar_index(column, index_type="BTREE", replace=True)

    def create_vector_index(self, column: str = "vec_retrieval"):
        """Build IVF_PQ vector index for ANN search."""
        self.table.create_index(metric="cosine", vector_column_name=column)

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
        return self.table.search(query, query_type="fts").limit(limit).to_list()

    def hybrid_search(
        self,
        query_vec: np.ndarray,
        query_text: str,
        limit: int = 20,
        column: str = "vec_retrieval",
    ) -> list[dict]:
        """Hybrid search: ANN + BM25 fused with RRF."""
        return (
            self.table.search(query_type="hybrid", vector_column_name=column)
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
