# Candidate Generation: Iterative PPR-guided Expansion

> How the candidate pool is built before scoring.

## Problem with Fixed k-hop

- k=1: too narrow (~100 papers)
- k=2: too wide (5,000+ papers, hub noise)
- k=3: explodes (100K+, mostly irrelevant)

All edges treated equally — hub papers (Laemmli, RECIST) pull in
unrelated literature.

## Solution: Score-guided Iterative Expansion

```
Input:  seed, max_nodes=500, max_iter=5, threshold=0.001, α=0.2

seen = {seed}
frontier = {seed}

for i in 1..max_iter:
    # Expand: 1-hop neighbors of frontier only
    new = neighbors(frontier) - seen
    if not new: break
    
    # Score: PPR on entire accumulated set
    pool = seen ∪ new
    ppr = PersonalizedPageRank(pool, seed, α)
    
    # Prune: only keep new nodes above threshold
    worthy = {n ∈ new | ppr[n] >= threshold}
    if not worthy: break
    
    seen |= worthy
    frontier = worthy
    if |seen| >= max_nodes: break

return seen
```

## Properties

- **Dynamic depth**: expands deeper into relevant clusters, shallow in noisy directions
- **Deterministic**: same seed → same result (PPR converges, threshold is fixed)
- **Hub suppression**: PPR restart probability + threshold prunes hub-connected noise
- **Bounded**: max_nodes and max_iter prevent runaway expansion

## Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| max_nodes | 500 | Hard cap on subgraph size |
| max_iter | 5 | Max expansion rounds |
| threshold | 0.001 | Min PPR score to keep a node |
| α | 0.2 | PPR teleport probability (lower = wider exploration) |

## Implementation Plan

- Phase 1: Python prototype using existing `quarry_graph.k_hop` + `subgraph_pagerank`
- Phase 2: Rust native `iterative_ppr_expand` for performance
- PPR on subgraph (not full 181M graph) — bounded by candidate pool size

## Open Questions

- [ ] Optimal threshold and α — empirical tuning on test papers
- [ ] Whether to use forward-only, reverse-only, or both directions per iteration
- [ ] Memory budget for pool size during iteration
