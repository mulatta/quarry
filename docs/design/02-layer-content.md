# Layer 2: Content (Semantic Similarity)

> Signal: meaning overlap with seed paper.

## Data Sources

- Embeddings: LanceDB (jina-v5 nano, 256-dim, T1+T2 148M papers — in progress)
- MeSH: PG mesh_headings table (T1 ~22M papers with PMID)
- OA Topics: PG work_topics table (~100% coverage, 3 topics per work)

## L0 Atomic Operations

### Embedding ANN Search

- **Input**: seed embedding vector (or seed work_id → lookup)
- **Output**: {node_id: cosine_similarity} top-k
- **Property**: semantic similarity of title+abstract
- **Coverage**: T1+T2 (148M after encoding complete)
- **Implementation**: LanceDB vector_search / hybrid_search
- **Status**: blocked on embedding completion (~22 days remaining)

### BM25 Full-Text Search

- **Input**: seed title+abstract as query text
- **Output**: {node_id: bm25_score} top-k
- **Property**: keyword matching on title+abstract
- **Coverage**: T1+T2 (same as embeddings, FTS index built at encoding end)
- **Implementation**: LanceDB text_search
- **Status**: blocked on embedding completion (FTS index built after encoding)

### MeSH Overlap

- **Input**: seed PMID → MeSH descriptors; candidate PMID → MeSH descriptors
- **Output**: {node_id: jaccard_similarity}
- **Property**: shared medical subject headings
- **Coverage**: T1 only (~22M papers with PMID + MeSH annotations)
- **Implementation**: PG query, Jaccard computation in Python
- **Status**: data available, computation not implemented
- **Variant**: weighted Jaccard using MeSH tree depth (specific descriptors > broad ones)

### OA Topic Overlap

- **Input**: seed work_id → topics; candidate work_id → topics
- **Output**: {node_id: overlap_score}
- **Property**: shared OpenAlex topic classifications
- **Coverage**: ~100% of works (3 topics per work, with is_primary flag)
- **Implementation**: PG query
- **Status**: data available, computation not implemented
- **Note**: coarser than MeSH but much higher coverage

## L1 Aggregation

TBD. Likely RRF over available signals (embedding + MeSH + topics).
BM25 may be used as fallback when embedding unavailable.

## Open Questions

- [ ] How to combine embedding similarity with MeSH/topic overlap
- [ ] MeSH is hierarchical — should descendants count as partial match?
- [ ] OA topics have score field — use as confidence weight?
