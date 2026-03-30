# Score Fusion

> Combining ranked lists from multiple layers into a final ranking.

## Computation Levels

```
L0: Atomic operations (PPR, bib coupling, MeSH Jaccard, RCR, ...)
     ↓
L1: Layer aggregation (each layer → one ranked list)
     ↓
L2: Cross-layer fusion (4 ranked lists → 1 final list)
     ↓
L3: Reranking (optional quality boost on final list)
```

## L1: Within-Layer Aggregation

Each layer may use multiple L0 operations. How to combine within a layer:

### Layer 1 (Structure): PPR + coupling + co-citation
- Option A: mini-RRF(rank_ppr, rank_coupling, rank_cocite)
- Option B: PPR primary, coupling/cocite as additive bonus
- Decision: TBD

### Layer 2 (Content): embedding + BM25 + MeSH + topics
- Option: RRF over available signals (skip unavailable)
- Decision: TBD

### Layer 3 (Quality): RCR + cited_by + global PR
- Option: max(norm(RCR), norm(cited_by)) — take best available
- Decision: TBD

### Layer 4 (Temporal): recency + velocity
- Option: weighted sum (user-controlled weights)
- Decision: TBD

## L2: Cross-Layer Fusion

**RRF (Reciprocal Rank Fusion)**:

```
RRF_score(paper) = Σ_layer  1 / (k + rank_layer(paper))

k = 60 (standard constant)
```

Properties:
- Scale-invariant (uses ranks, not scores)
- No normalization needed
- Single parameter (k)
- Handles missing entries (paper not ranked by a layer → no contribution)

### Handling Missing Layers

If a layer is unavailable (e.g., embeddings not ready):
- Simply omit that layer from the sum
- Remaining layers produce valid fusion
- This is why RRF is chosen — graceful degradation

## L3: Reranking

Post-hoc quality adjustment on L2 output:

```
final_score(paper) = L2_rank_position × (1 + γ·norm(quality_signal))
```

Or equivalently: sort L2 top-k by (L2_rank, quality_signal) with quality as tiebreaker.

**Reranking is independent of L2 fusion** — it only reorders, never adds/removes.

Currently operates without embeddings. Uses RCR/cited_by_count only.

## Open Questions

- [ ] k value for RRF (60 is standard but may need tuning)
- [ ] Whether layers should have unequal weights in RRF
- [ ] L3 reranking formula — multiplicative vs tiebreaker
- [ ] Evaluation: how to measure fusion quality without ground truth
