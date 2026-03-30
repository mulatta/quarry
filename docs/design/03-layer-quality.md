# Layer 3: Quality (Impact Metrics)

> Signal: paper importance / reliability, independent of seed relationship.

## Data Sources

- iCite: 40.2M PubMed papers (RCR, FCR, APT, NIH percentile)
- OA: cited_by_count (~100% coverage)
- Global PageRank: precomputable from CSR graph

## L0 Atomic Operations

### RCR (Relative Citation Ratio)

- **Input**: PMID → iCite lookup
- **Output**: RCR score (field-normalized citation impact)
- **Interpretation**: RCR=1.0 = field average; RCR=2.0 = 2x field average
- **Coverage**: 29.6M / 40.2M iCite papers = 74% of PubMed
- **Non-PubMed**: unavailable (0% of T2/T4)

### FCR (Field Citation Rate)

- **Input**: PMID → iCite lookup
- **Output**: expected citation rate for the paper's field/year
- **Use**: baseline for RCR; also useful standalone as field norm
- **Coverage**: 33.6M / 40.2M = 84% of PubMed

### APT (Approximate Potential to Translate)

- **Input**: PMID → iCite lookup
- **Output**: probability of clinical translation (0-1)
- **Use**: biomedical relevance filter
- **Coverage**: 21.9M iCite papers

### cited_by_count

- **Input**: work_id → OA works table
- **Output**: absolute citation count
- **Coverage**: ~100%
- **Caveat**: highly skewed (power law), hub-biased, not field-normalized

### Global PageRank (future)

- **Input**: precomputed from full CSR graph
- **Output**: global network centrality score
- **Use**: "importance in the overall citation network"
- **Status**: not precomputed (requires full-graph iteration, ~minutes)

## Role in Pipeline

**Quality is a reranking signal, not a primary ranker.**

Structure (Layer 1) determines which papers are related.
Quality determines which related papers are most impactful.

```
L2 fusion output: [A, B, C, D, E, ...]  (ranked by structural + content relevance)
L3 reranking:     [A, C, B, D, E, ...]  (C promoted because RCR=5.0 vs B's RCR=0.8)
```

Reranking does not add or remove candidates — only reorders.

## Open Questions

- [ ] Should Quality participate in L2 fusion (as a ranked list) or only in L3 reranking?
      Argument for fusion: quality is a legitimate relevance signal
      Argument for reranking: avoids missing data bias (see 07-missing-data.md)
- [ ] Global PageRank precomputation: worth the cost? Or cited_by_count sufficient?
