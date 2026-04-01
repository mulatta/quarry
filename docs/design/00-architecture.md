# Subgraph Mining Architecture

> Status: Phase 1a next. APPR implementation pending. CSR + merged citations verified.

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
│  Input: 4 ranked lists → Output: final ranked subgraph           │
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

## Candidate Generation: APPR (Approximate Personalized PageRank)

Push-based local computation on full CSR graph. Single pass replaces
the previous iterative BFS + subgraph PPR design.

```
APPR(graph, seed, α=0.15, ε=1e-6):
    push mass from seed through neighbors
    stop pushing when residual/degree < ε
    → sparse result: only relevant nodes have scores
```

No BFS boundary, no iteration loop. APPR handles both candidate
discovery and scoring in one operation. O(1/ε) regardless of graph size.

See: [05-candidate-generation.md](05-candidate-generation.md)

## Fusion

See: [06-fusion.md](06-fusion.md)

## expand Command

CLI entry point for subgraph mining. Single/multi-seed, direction filtering,
relation tagging, RRF fusion. See: [08-expand-command.md](08-expand-command.md)

## Evaluation

Quality assessment without ground truth labels. Reference recovery,
leave-one-out, symmetry, expert judgment. See: [09-evaluation.md](09-evaluation.md)

## Open Questions

- [ ] Missing data fairness: how to weight signals with partial coverage
      (e.g., RCR available for 74% of PubMed papers, 0% of non-PubMed)
      See: [07-missing-data.md](07-missing-data.md)
- [ ] Optimal APPR α for citation networks (start with 0.15, tune via eval)
- [ ] Whether L3 (Quality) should be reranking-only or participate in fusion
- [ ] Embedding model choice for Layer 2 (jina-v5 nano currently)
- [ ] Evaluation methodology (no ground truth labels)

## Implementation Roadmap

| Phase | Layers                        | Fusion                          | Status                                                |
| ----- | ----------------------------- | ------------------------------- | ----------------------------------------------------- |
| 1a    | Structure (APPR)              | —                               | **Next** — APPR replaces power iteration PPR          |
| 1b    | + coupling + cocite           | RRF(3 signals)                  | After APPR                                            |
| 1c    | expand CLI command            | RRF + direction filtering       | After fusion                                          |
| 2     | + Quality (RCR rerank) + HKPR | RRF(4+ signals) + rerank        | HKPR builds on APPR push framework                    |
| 3     | + Content (embedding, BM25)   | RRF(6+ signals)                 | After embeddings                                      |
| 4     | + Temporal + Adamic-Adar      | Full RRF + empirical param tune | Long-term                                             |

Each phase is additive — previous interfaces are stable.

## Current State

Implemented:

- [x] iCite citation merge (OA ∪ iCite, +746M edges)
- [x] CSR build from Parquet (no CSV, peak ~77GB, 182M nodes / 3.77B edges)
- [x] PPR prototype verified: hub suppression, topic relevance confirmed
- [x] k_hop for simple neighbor lookup

To be replaced:

- [ ] Power iteration PPR (`subgraph_pagerank`) → APPR (Phase 1a)
- [ ] Fixed 2-hop BFS pool → APPR natural boundary (Phase 1a)

Issues APPR addresses:

- Seed score dominance → APPR's local push distributes mass more naturally
- Off-topic via hub edges → APPR dilutes hub mass across many neighbors
- Single signal → RRF fusion (Phase 1b, after APPR)

Next: implement `appr()` in Rust, delete `pagerank::compute` and `pagerank::subgraph`.
