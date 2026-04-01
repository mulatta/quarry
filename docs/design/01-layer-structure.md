# Layer 1: Structure (Citation Graph Topology)

> Signal: structural proximity to seed in citation network.

## Data Source

- CSR graph: 182M nodes, 3.77B edges (mmap, quarry-graph Rust)
- Forward (outgoing references) + Reverse (incoming citations)
- OA + iCite merged (iCite added 746M edges for PubMed papers)

## L0 Atomic Operations

### APPR (Approximate Personalized PageRank)

- **Input**: seed node, α (teleport), ε (precision)
- **Output**: sparse {node_id: ppr_score} — only touched nodes
- **Property**: push-based local computation on full CSR, O(1/ε)
- **Hub handling**: high-degree nodes dilute push mass across neighbors
- **Implementation**: `algo/appr.rs` — replaces power iteration `subgraph_pagerank`
- **Default**: α=0.15, ε=1e-6
- **Verified**: PET paper → 2,366 nodes in 1.45s, all top 20 on-topic

### Bibliographic Coupling (Adamic-Adar weighted, cosine normalized)

- **Input**: seed node
- **Output**: {node_id: coupling_score}
- **Scoring**: shared references weighted by 1/log(indegree) — niche references
  count more than popular ones. Cosine-normalized to remove reference list size bias.
  ```
  raw(A,B) = Σ_{r ∈ refs(A) ∩ refs(B)} 1/log(indegree(r))
  score(A,B) = raw(A,B) / √(self_weight(A) × self_weight(B))
  ```
- **Property**: discovers related papers even without direct citation link
- **Implementation**: existing `bibcoupling.rs` (raw count) → upgrade to AA+cosine
- **Complexity**: O(d_seed) for seed-centric computation

### Co-citation (Adamic-Adar weighted, cosine normalized)

- **Input**: seed node
- **Output**: {node_id: cocitation_score}
- **Scoring**: papers co-cited with seed, weighted by 1/log(outdegree) of the
  citer — niche citers count more than review papers. Cosine-normalized.
  ```
  raw(A,B) = Σ_{c ∈ citers(A) ∩ citers(B)} 1/log(outdegree(c))
  score(A,B) = raw(A,B) / √(self_weight(A) × self_weight(B))
  ```
- **Property**: captures community perception ("these papers go together")
- **Implementation**: existing `co_citation.rs` (raw count) → upgrade to AA+cosine
- **Complexity**: O(d_seed) for seed-centric computation

### Note on Adamic-Adar

Originally planned as a separate L0 operation. Now incorporated as the
weighting scheme for coupling and co-citation in Phase 1b — not a separate
signal but a quality improvement to existing signals. This avoids redundant computation
and keeps the signal count manageable.

## L1 Aggregation: wRRF

Three signals → one Structure score via weighted RRF:

```
structure_score(paper) = w_appr/(k + rank_appr) + w_coup/(k + rank_coup) + w_cocite/(k + rank_cocite)

defaults: w_appr=0.5, w_coup=0.25, w_cocite=0.25, k=60
```

APPR gets higher weight because it captures global graph proximity
(multi-hop paths), while coupling/cocitation are local (1-2 hop derived).

This fusion happens in Rust (`expand` function), not Python.

## Current State

- [x] APPR: implemented and verified (`algo/appr.rs`)
- [x] Bib coupling: Rust code exists (raw count, `pattern/bibcoupling.rs`)
- [x] Co-citation: Rust code exists (raw count, `pattern/co_citation.rs`)
- [ ] AA weighting + cosine normalization for coupling/cocitation
- [ ] wRRF fusion in Rust `expand()` function
- [ ] Seed exclusion in `expand()` output

## Known Data Issues

- 75% of works have no reference data (zero out-degree)
- OA citation graph is all-discipline, not bio-specific → non-bio hubs
- Self-citations (1.07M) — minor but could inflate seed neighbor scores
- CSR is static snapshot — new citations not reflected until rebuild
- iCite merge added 746M edges but only for PubMed papers
