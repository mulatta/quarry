# Subgraph Mining Architecture

> Status: Phase 1b complete. Phase 2 planning.

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
Fusion combines them via wRRF (Weighted Reciprocal Rank Fusion) — rank-based, scale-invariant.

### L2 → L3: Fusion → Reranking

Optional quality-based reranking on the fused list.
Only modifies ordering, does not add/remove candidates.

## Four Layers

```
┌──────────────────────────────────────────────────────────────────┐
│                     L2: Fusion (wRRF)                            │
│  Input: 4 ranked lists → Output: final ranked subgraph           │
└─────┬──────────┬──────────────┬──────────────┬───────────────────┘
      │          │              │              │
 ┌────┴────┐ ┌───┴─────┐ ┌─────┴─────┐ ┌──────┴──────┐
 │ Layer 1 │ │ Layer 2 │ │  Layer 3  │ │   Layer 4   │
 │Structure│ │ Content │ │  Quality  │ │  Temporal   │
 │         │ │         │ │           │ │             │
 │ APPR    │ │ Embed   │ │ RCR/FCR   │ │ Recency     │
 │ Coupling│ │ BM25    │ │ APT       │ │ Velocity    │
 │ CoCite  │ │ MeSH    │ │           │ │             │
 └─────────┘ └─────────┘ └───────────┘ └─────────────┘
```

Note: Coupling and co-citation use Adamic-Adar weighting with cosine
normalization. This replaces the previously planned separate "Adamic-Adar"
atomic operation — the weighting is now built into coupling/cocitation scoring.

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

CLI entry point for subgraph mining. Core logic in Rust (`expand` function),
Python CLI is a thin wrapper. See: [08-expand-command.md](08-expand-command.md)

## Evaluation

Quality assessment without ground truth labels. Reference recovery,
leave-one-out, symmetry, expert judgment. See: [09-evaluation.md](09-evaluation.md)

## Open Questions

- [ ] Missing data fairness: how to weight signals with partial coverage
  (e.g., RCR available for 74% of PubMed papers, 0% of non-PubMed)
  See: [07-missing-data.md](07-missing-data.md)
- [ ] Optimal APPR α for citation networks (start with 0.15, tune via eval)
- [x] ~~L3 Quality: fusion vs reranking~~ → **neither** — metadata only (see 03)
- [ ] Embedding model choice for Layer 2 (jina-v5 nano currently)

## Implementation Roadmap

| Phase | Scope | Status |
| ----- | ----- | ------ |
| 1a | APPR (push-based) | **Done** |
| 1b | expand(fused/separated) + AA coupling/cocite + wRRF + CLI | **Done** |
| 2a | Bridge discovery (7 types: LCA, CD, coupling, cocite, path, PPR, Steiner) | Design confirmed — [design](11-bridge.md) |
| 2a+ | CLI `info` command + expand/bridge output enrichment (DOI, venue, cited_by) | Planned — [AD-8](10-architecture-decisions.md#ad-8-info-command-and-sql-lifecycle) |
| 2b | MCP server expand tool | Planned (deferred) |
| 3 | Content (embedding, BM25) + embedding bridge candidates + HKPR re-eval | After embeddings |
| 4 | Temporal | Long-term |

Direction-restricted APPR **rejected** — experiment showed bidirectional APPR
has higher precision; direction use cases served by relation tags + limit increase.
See [AD-4](10-architecture-decisions.md#ad-4-no-direction-restricted-appr).

Quality signals (cnp, fwci, rcr) are **metadata only** — not used in fusion.
See [03-layer-quality.md](03-layer-quality.md) for rationale.

## Current State

Implemented:

- [x] iCite citation merge (OA ∪ iCite, +746M edges)
- [x] CSR build from Parquet (no CSV, peak ~77GB, 182M nodes / 3.77B edges)
- [x] APPR in Rust (push-based, O(1/ε), bidirectional)
- [x] expand() — dual modes (fused wRRF + separated with lateral slots)
- [x] AA-weighted cosine-normalized coupling/cocitation
- [x] CLI: `quarry expand` with DOI/W-prefix input, rich table + JSON output
- [x] Regression tests: 35 tests across 3 seeds
- [x] PG index: idx_works_work_id_int (53s → 9ms lookup)
- [x] Bridge collection (pass 2, lazy) for lateral papers — SharedRef + SharedCiter with AA weight
- [x] Quality metadata enrichment (cnp, fwci, rcr, cited_by) — metadata only (AD-2)
- [x] Output schema finalized (see 08-expand-command.md)
- [x] Weighted APPR evaluation: cnp/fwci post-hoc weighting tested, rejected (AD-2 confirmed)

Verified (5 seeds: PET, base editing, PLM, glycosylase BE, aptamer):

- All top-20 on-topic across all seeds (PMC abstract verified)
- Lateral papers all relevant (0 noise across 5 seeds)
- wRRF w=0.7/0.15/0.15 preserves ref recovery while adding laterals
- cnp weighting: promotes established papers, penalizes preprints (NULL) — inappropriate
- fwci weighting: promotes bioinformatics tools over seed-specific papers — inappropriate
