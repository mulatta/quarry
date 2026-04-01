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
    pick such v
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
/// Push-based Approximate Personalized PageRank on full CSR graph.
/// Returns (node_id, score) pairs sorted by score descending.
/// Only nodes with score > 0 are returned (sparse result).
pub fn appr(
    graph: &Graph,
    seed: i64,
    alpha: f64,
    epsilon: f64,
    top_k: Option<usize>,  // None = return all
) -> Vec<(i64, f64)>
```

APPR is called internally by `expand()`. Direct Python binding also available
for debugging/testing:
```python
graph.appr(seed, alpha=0.15, epsilon=1e-6, top_k=None) -> list[tuple[int, float]]
```

Phase 2: add `direction: Direction` parameter for directional filtering.
```rust
pub enum Direction { Forward, Reverse, Both }
```

## Replaces

- `pagerank::compute` — full-graph PR, no use case → **delete**
- `pagerank::subgraph` — subgraph PPR via power iteration → **delete**
- Python `subgraph_pagerank` binding → **delete**, replaced by `appr`

## Retained

- `k_hop` — kept for simple neighbor lookup (e.g., "show direct citations").
  NOT used in expansion logic. APPR handles expansion independently.

## vs HKPR (Phase 2)

HKPR (Heat Kernel PageRank) uses Poisson decay instead of geometric.
Structurally resolves seed score dominance (seed contribution ~0.7% vs PPR ~20%).
Planned as Phase 2 addition — same push framework, different decay function.
APPR implementation provides the foundation for HKPR.

## Expected Behavior

On the N-glycosylation test paper (seed, α=0.15, ε=1e-6):
- Result: sparse set of ~thousands of nodes with PPR scores
- Seed neighbors with high relevance: high scores
- Hub papers (Laemmli, RECIST): low scores (mass diluted across many edges)
- Off-topic but structurally close: low scores (teleport pulls back to seed)
- Time: sub-second (local computation, not proportional to graph size)

## Open Questions

- [ ] Optimal α for citation networks — empirical tuning via eval protocol
- [ ] Whether ε=1e-6 gives sufficient coverage or needs adjustment
- [ ] Direction filtering semantics for expand command (Phase 1c)
