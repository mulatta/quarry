# Score Fusion

> Combining ranked lists from multiple signals into a final ranking.

## Phase 1b: Structure Layer wRRF (Rust)

Three signals fused in Rust `expand()` function:

```
fused_score(paper) = Σ  w_i / (k + rank_i(paper))

signals: APPR, coupling (AA-cosine), cocitation (AA-cosine)
defaults: w = [0.7, 0.15, 0.15], k = 60
```

### Why wRRF over alternatives

| Method | Pros | Cons | Decision |
|--------|------|------|----------|
| **wRRF** | Scale-invariant, handles missing entries, single param k | Weight tuning needed | **Phase 1b** |
| CombMNZ | Score-aware (uses actual values) | Requires normalization, sensitive to outliers | Compare in eval |
| Condorcet | Theoretically sound | O(n²), minimal gain over RRF | Skip |
| LambdaMART | Best quality | Needs labeled data | Phase 4+ if ever |

### Weight Rationale

- APPR (0.7): dominates to preserve ref/citer recovery (~80%+ across seeds)
- Coupling (0.15): methodology similarity via shared references
- Co-citation (0.15): community signal via shared citers

Validated via 5-seed parameter sweep: w=0.7/0.15/0.15 matches APPR-only
ref/citer recovery while adding lateral papers. Lower APPR weight (0.5)
causes 10-20% ref recovery loss as coupling candidates flood the results.

### Handling Missing Signals

If a paper appears in APPR but not in coupling/cocitation:

- That signal simply contributes 0 to the sum
- RRF gracefully degrades — remaining signals still produce valid ranking

Papers appearing in coupling/cocitation but NOT in APPR:

- These are "lateral" candidates — no structural proximity but bibliometric similarity
- Included in output with only coupling/cocitation score contributing

## Future Phases

### Phase 2: Quality as Metadata (Not Fusion)

Quality signals (cnp, fwci, rcr) are exposed as metadata in output.
NOT used in fusion scoring — "field importance" ≠ "relevance to seed".
See: [03-layer-quality.md](03-layer-quality.md) for evaluation rationale.

### Phase 3: Content Layer Fusion

When embedding layer is available:

```
fused_score(paper) = wRRF(structure_score, content_score)
```

Content (embedding similarity) is the first signal that measures
**semantic relevance** independent of citation structure. This is
qualitatively different from quality signals (which measure impact).

### Phase 4: Full Multi-Layer (if justified by eval)

```
fused_score(paper) = wRRF(structure, content, temporal)
```

Temporal layer only if eval shows recency weighting improves results.

## Computation Levels (Full Architecture)

```
L0: Atomic operations (APPR, coupling, MeSH Jaccard, RCR, ...)
     ↓
L1: Layer aggregation (each layer → one ranked list)
     ↓
L2: Cross-layer fusion (4 ranked lists → 1 final list)
     ↓
L3: Reranking (optional quality boost on final list)
```

Phase 1b implements L0 + L1 for Structure layer only.
L2/L3 activate when additional layers are implemented.

## Open Questions

- [ ] k value for wRRF (60 is standard but may need tuning)
- [ ] Whether CombMNZ outperforms wRRF for this workload — A/B test in eval
- [ ] L3 reranking formula — multiplicative vs tiebreaker
- [ ] Evaluation: how to measure fusion quality without ground truth
