# Subgraph Mining Architecture

> Status: Phase 1 in progress. PPR + merged CSR verified.

## Problem

Given a seed paper, extract a relevant subgraph from the citation network.
Must be deterministic, non-agentic (same input → same output, no LLM in loop).

## Computation Hierarchy

```
L3: Reranking (optional post-hoc quality boost)
     │
L2: Fusion (cross-layer rank aggregation → final ranked list)
     │
L1: Layer aggregation (each layer produces its own ranked list)
     │
L0: Atomic operations (single algorithm, single data source)
```

### L0 → L1: Atomic → Layer

Each layer combines multiple L0 operations into one ranked list.
Within a layer, combination method is layer-specific (e.g., score sum, max, or mini-RRF).

### L1 → L2: Layer → Fusion

Four layers produce four independent ranked lists.
Fusion combines them via RRF (Reciprocal Rank Fusion) — rank-based, scale-invariant.

### L2 → L3: Fusion → Reranking

Optional quality-based reranking on the fused list.
Only modifies ordering, does not add/remove candidates.

## Four Layers

```
┌──────────────────────────────────────────────────────────────────┐
│                     L2: Fusion (RRF)                             │
│  Input: 4 ranked lists → Output: final ranked subgraph          │
└─────┬──────────┬──────────────┬──────────────┬───────────────────┘
      │          │              │              │
 ┌────┴────┐ ┌───┴─────┐ ┌─────┴─────┐ ┌──────┴──────┐
 │ Layer 1 │ │ Layer 2 │ │  Layer 3  │ │   Layer 4   │
 │Structure│ │ Content │ │  Quality  │ │  Temporal   │
 │         │ │         │ │           │ │             │
 │ PPR     │ │ Embed   │ │ RCR/FCR   │ │ Recency     │
 │ Coupling│ │ BM25    │ │ PageRank  │ │ Velocity    │
 │ CoCite  │ │ MeSH    │ │ APT       │ │             │
 │ Adamic  │ │ Topics  │ │           │ │             │
 └─────────┘ └─────────┘ └───────────┘ └─────────────┘
```

### Layer 1: Structure (citation graph topology)

Signal: structural proximity to seed in citation network.
Data: CSR graph (182M nodes, 3.77B edges — OA + iCite merged).
See: [01-layer-structure.md](01-layer-structure.md)

### Layer 2: Content (semantic similarity)

Signal: meaning overlap with seed paper.
Data: embeddings (LanceDB), MeSH (PG), OA topics (PG).
See: [02-layer-content.md](02-layer-content.md)

### Layer 3: Quality (impact metrics)

Signal: paper importance / reliability.
Data: iCite (RCR, FCR, APT), OA cited_by_count.
See: [03-layer-quality.md](03-layer-quality.md)

### Layer 4: Temporal (citation dynamics)

Signal: time-dependent patterns.
Data: pub_year, cited_by_count, iCite citation_count.
See: [04-layer-temporal.md](04-layer-temporal.md)

## Candidate Generation: Iterative PPR-guided Expansion

NOT fixed k-hop BFS. Score-guided iterative expansion:

```
seen = {seed}
frontier = {seed}

for i in 1..max_iter:
    new = BFS_1hop(frontier) - seen
    ppr = PPR(seen ∪ new, seed)
    worthy = {n ∈ new | ppr[n] >= threshold}
    if not worthy: break
    seen |= worthy
    frontier = worthy
    if |seen| >= max_nodes: break
```

Hop count is dynamic — expands deeper into relevant clusters,
stops early in noisy/hub-dominated directions.

See: [05-candidate-generation.md](05-candidate-generation.md)

## Fusion

See: [06-fusion.md](06-fusion.md)

## Open Questions

- [ ] Missing data fairness: how to weight signals with partial coverage
      (e.g., RCR available for 74% of PubMed papers, 0% of non-PubMed)
      See: [07-missing-data.md](07-missing-data.md)
- [ ] Optimal PPR α for citation networks
- [ ] Whether L3 (Quality) should be reranking-only or participate in fusion
- [ ] Embedding model choice for Layer 2 (jina-v5 nano currently)
- [ ] Evaluation methodology (no ground truth labels)

## Implementation Roadmap

| Phase | Layers | Fusion | Status |
|-------|--------|--------|--------|
| 1 | Structure (PPR + coupling) | RRF(2 signals) | **In progress** — PPR verified, coupling/cocite next |
| 2 | + Quality (RCR rerank) + MeSH | RRF(4 signals) + rerank | |
| 3 | + Content (embedding, BM25) | RRF(6+ signals) | After embeddings |
| 4 | + Temporal + Adamic-Adar | Full RRF + learned weights | Long-term |

Each phase is additive — previous interfaces are stable.

## Current State (Phase 1 Progress)

Implemented:
- [x] PPR in Rust (`subgraph_pagerank` with `restart_node`)
- [x] iCite citation merge (OA ∪ iCite, +746M edges)
- [x] CSR build from Parquet (no CSV, peak ~77GB, 182M nodes / 3.77B edges)
- [x] PPR verified: hub suppression (Laemmli dropped), topic relevance (IL6/glycosylation surfaced)

Verified issues to address:
- [ ] Fixed 2-hop pool (5000 cap) → iterative expansion
- [ ] Seed score dominance (0.805 vs 0.003 for #2) → α tuning or seed exclusion
- [ ] Off-topic papers (GSEA #3, PGC-1α #6) → coupling/MeSH filter
- [ ] Single signal (PPR only) → RRF(PPR + coupling + cocite)

Baseline performance (N-glycosylation test paper, 2-hop pool=5000):
- k_hop: <0.1s
- subgraph_pagerank: <0.1s
- PG enrichment: <0.01s
- total: <0.2s
