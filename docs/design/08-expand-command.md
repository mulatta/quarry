# quarry expand — Command Design

> Status: Phase 1b implementation next.

## Definition

`quarry expand` takes one or more seed papers and returns a ranked subgraph
of related papers from the citation graph. All operations are deterministic.

## Architecture

Core logic in Rust (`expand` function). Python CLI is a thin wrapper.

```
Python CLI          Rust expand()                    Python CLI
──────────         ──────────────                   ──────────
DOI/PMID input  →  APPR(seed, α, ε)             →  PG metadata enrichment
  ↓ PG lookup      coupling_aa(seed)                  (title, year, etc.)
work_id_int     →  cocitation_aa(seed)           →  format + output
                   wRRF fusion
                   seed exclusion
                   top-k truncation
                → Vec<(work_id, fused_score)>
```

## Rust API

```rust
pub struct ExpandResult {
    pub papers: Vec<(i64, f64)>,  // (work_id, fused_score) sorted desc
    pub stats: ExpandStats,
}

pub struct ExpandStats {
    pub appr_candidates: usize,
    pub coupling_candidates: usize,
    pub cocitation_candidates: usize,
    pub elapsed_ms: u64,
}

pub fn expand(
    graph: &Graph,
    seed: i64,
    alpha: f64,           // APPR teleport (default 0.15)
    epsilon: f64,         // APPR precision (default 1e-6)
    weights: [f64; 3],    // [appr, coupling, cocitation] wRRF weights
    rrf_k: usize,         // RRF smoothing constant (default 60)
    limit: usize,         // max results (default 200)
) -> ExpandResult
```

Python binding:
```python
graph.expand(
    seed: int,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
    rrf_k: int = 60,
    limit: int = 200,
) -> tuple[list[tuple[int, float]], dict]  # (papers, stats)
```

### Internals

1. APPR(seed, α, ε) → structural scores
2. coupling_aa(seed) → AA-weighted cosine-normalized coupling scores
3. cocitation_aa(seed) → AA-weighted cosine-normalized cocitation scores
4. Union all candidates from 3 signals
5. Rank each candidate per signal
6. wRRF: `score = Σ w_i / (k + rank_i)` (missing signal → 0 contribution)
7. Exclude seed from output
8. Sort by fused score, truncate to limit

## CLI Interface

```bash
# Single seed (work_id, DOI, or PMID)
quarry expand W4406019873
quarry expand --doi 10.1126/science.adp5637
quarry expand --pmid 39251588

# Options
quarry expand <work_id> \
  --limit 200 \
  --sort fused|structural|coupling|cocitation \
  --format json|table

# Future (Phase 2+)
quarry expand <work_id> --direction both|forward|backward
quarry expand <id1> <id2> <id3>  # multi-seed
```

### Default Behavior

```bash
quarry expand W4406019873
# limit=200, sort=fused, format=table
```

## Output Structure

```json
{
  "seeds": ["W4406019873"],
  "params": { "limit": 200, "alpha": 0.15, "epsilon": 1e-6, "weights": [0.5, 0.25, 0.25] },
  "papers": [
    {
      "work_id": "W...",
      "title": "...",
      "year": 2016,
      "cited_by": 3111,
      "relation": "foundation",
      "scores": {
        "structural": 0.045,
        "coupling": 0.82,
        "cocitation": 0.63,
        "fused": 0.087
      }
    }
  ],
  "stats": {
    "appr_candidates": 2366,
    "coupling_candidates": 450,
    "cocitation_candidates": 380,
    "returned": 200,
    "elapsed_ms": 1800
  }
}
```

## Relation Tagging

Deterministic classification based on edge existence:

| seed → paper | paper → seed | Relation                                            |
| :----------: | :----------: | --------------------------------------------------- |
|     yes      |      no      | `foundation` — seed cites this paper                |
|      no      |     yes      | `follow-up` — this paper cites seed                 |
|     yes      |     yes      | `mutual`                                            |
|      no      |      no      | `lateral` — discovered via coupling/cocitation only |

## Multi-seed Behavior (Phase 3)

| Mode              | Semantics                   | Algorithm                                                 |
| ----------------- | --------------------------- | --------------------------------------------------------- |
| `union` (default) | Papers related to ANY seed  | Multi-seed APPR: `residual = {s: w/sum for s,w in seeds}` |
| `intersection`    | Papers related to ALL seeds | Individual APPR per seed → min score                      |
| `bridge`          | Papers connecting seeds     | Pairwise shortest paths → union                           |

### Seed Weights (Phase 3)

- Default: equal (`1/N` each)
- `--primary W...`: named seed gets `2×` weight
- `--weights 0.6,0.2,0.2`: explicit (must sum to ~1.0)

## Implementation Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1a | APPR only | **Done** |
| 1b | expand() in Rust + CLI + AA coupling/cocitation + wRRF | **Next** |
| 2 | + direction filtering + HKPR + Quality rerank | |
| 3 | + multi-seed + search→expand pipeline | |

## Open Questions

- [ ] wRRF weight tuning: eval protocol needed
- [ ] `technique` relation: requires MeSH qualifier (22M coverage)
- [ ] Multi-seed weight determination: user, agent, or heuristic?
- [ ] How many search results become expand seeds?
