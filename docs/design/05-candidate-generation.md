# Candidate Generation: APPR (Approximate Personalized PageRank)

> Core algorithm for subgraph extraction. Replaces k-hop BFS + power iteration PPR.

## Why APPR over Previous Approach

Previous design: iterative BFS expansion + subgraph PPR (power iteration).

| | BFS + subgraph PPR | APPR (push-based) |
|--|--------------------|--------------------|
| Expansion | BFS decides boundary, PPR scores within | APPR does both — score-guided boundary |
| Computation | Re-runs PPR each iteration on growing subgraph | Single pass, O(1/ε) |
| Boundary | Arbitrary (BFS hop limit) — can miss relevant paths | Natural (score-based) — no missed paths |
| Direction filtering | Incomplete — BFS filters direction but PPR uses all edges in subgraph | Complete — push follows only chosen edges |

BFS imposes an arbitrary boundary that PPR cannot recover from. APPR's push
operation naturally forms the boundary based on score, which is both faster and
more accurate for our use case.

## Algorithm

Push-based local computation on full CSR graph (Andersen et al. 2006).

```
Input:  graph (CSR), seed, α=0.15, ε=1e-6

residual = {seed: 1.0}  // all mass starts at seed
estimate = {}           // accumulated PPR scores

while any node v has residual[v] / degree(v) > ε:
    pick such v (FIFO queue)
    // retain α fraction as PPR score
    estimate[v] += α * residual[v]
    // push (1-α) fraction to neighbors
    for u in neighbors(v):
        residual[u] += (1-α) * residual[v] / degree(v)
    residual[v] = 0

return estimate  // sparse: only touched nodes have entries
```

Key properties:

- **Local**: only visits nodes reachable from seed with significant score
- **Deterministic**: same seed → same result
- **Bounded**: O(1/ε) total work regardless of graph size (182M nodes irrelevant)
- **Natural boundary**: expansion stops where score drops below ε — no arbitrary hop limit
- **Hub suppression**: high-degree nodes dilute push mass across many neighbors

## Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| α | 0.15 | Teleport probability. Higher = tighter around seed |
| ε | 1e-6 | Precision threshold. Lower = more nodes discovered, slower |

- α=0.15: Andersen 2006 default, validated on citation networks
- ε=1e-6: standard default, good precision/speed tradeoff
- Both configurable via CLI for empirical tuning

## Rust API

```rust
pub fn appr(
    graph: &Graph,
    seed: i64,
    alpha: f64,
    epsilon: f64,
    top_k: Option<usize>,
) -> Vec<(i64, f64)>
```

APPR is called internally by `expand()`. Direct Python binding also available
for debugging/testing:

```python
graph.appr(seed, alpha=0.15, epsilon=1e-6, top_k=None) -> list[tuple[int, float]]
```

Direction-restricted APPR was evaluated and rejected (see AD-4).
Relation tags (foundation/follow-up/lateral) serve direction use cases.

## Verified Results

PET depolymerase paper (Seo 2025, α=0.15, ε=1e-6):

- 2,366 nodes scored in 1.45s
- Top 20: all PET enzyme papers, no off-topic
- Seed score: 44.1% of total mass
- Hub suppression working (no generic high-citation papers)

5-seed evaluation (PET, base editing, PLM evolution, glycosylase BE, aptamer):

- All seeds produce on-topic results
- APPR alone recovers 81-97% of seed references in top 200
- Lateral papers (via coupling/cocitation fusion) are all on-topic across all seeds

## Replaces

- `pagerank::compute` — full-graph PR → **deleted**
- `pagerank::subgraph` — subgraph PPR via power iteration → **deleted**
- Python `subgraph_pagerank` binding → **deleted**, replaced by `appr`

## Retained

- `k_hop` — kept for simple neighbor lookup (e.g., "show direct citations").
  NOT used in expansion logic. APPR handles expansion independently.

## vs HKPR (deferred to Phase 3)

HKPR (Heat Kernel PageRank) uses Poisson decay instead of geometric.
Lower seed dominance (~10% at t=3 vs ~47% for APPR), broader exploration
(45-82K nodes vs 1.5-2.4K). Evaluated as APPR replacement and wRRF hybrid.

**Result**: HKPR promotes field landmark/origin papers (high-citation, 2-3 hop
upstream) but displaces seed-specific niche papers. Effect is inconsistent
across seeds. Deferred to Phase 3 — embedding similarity can filter HKPR
candidates for relevance before fusion. See AD-5.

Code retained: `algo/hkpr.rs`, `graph.hkpr()` Python binding.

## Open Questions

- [ ] Optimal α for citation networks — empirical tuning via eval protocol
- [ ] Whether ε=1e-6 gives sufficient coverage or needs adjustment
- [x] ~~Direction filtering~~ → rejected, relation tags sufficient (AD-4)
