# quarry expand — Command Design

> Status: Phase 1b implemented. Dual modes (fused + separated).

## Definition

`quarry expand` takes a seed paper and returns a ranked subgraph of related
papers from the citation graph. All operations are deterministic.

## Two Modes

| Mode | Use case | How it works |
|------|----------|-------------|
| `fused` | Focused exploration (experiment design) | wRRF merges all candidates into single ranked list |
| `separated` | Broad survey (review writing) | APPR fills structural slots + guaranteed lateral slots |

### Fused Mode (default)

All candidates (APPR + coupling + cocitation) compete in a single pool via
weighted Reciprocal Rank Fusion. Best for finding the most directly relevant
papers when you know what you're looking for.

```
score(paper) = w_appr/(k + rank_appr) + w_coup/(k + rank_coup) + w_cocite/(k + rank_cocite)
defaults: w = [0.7, 0.15, 0.15], k = 60
```

### Separated Mode

APPR top-(limit × 75%) fills structural slots (refs, citers, indirect).
Coupling/cocitation lateral top-(limit × 25%) fills guaranteed lateral slots.
Lateral = papers NOT in seed's direct references or citers AND not in APPR top.

Best for discovering related work you might not know about — lateral papers
share methodology or community context but have no direct citation path to seed.

## Architecture

Core logic in Rust. Python CLI is a thin wrapper.

```
Python CLI          Rust expand()                    Python CLI
──────────         ──────────────                   ──────────
DOI/PMID input  →  APPR(seed, α, ε)             →  PG metadata enrichment
  ↓ PG lookup      coupling_aa(seed)                  (title, year, etc.)
work_id_int     →  cocitation_aa(seed)           →  format + output
                   fused: wRRF / separated: slot fill
                   seed exclusion
                   top-k truncation
                → Vec<(work_id, score)>
```

## Rust API

```rust
pub struct ExpandParams {
    pub alpha: f64,           // APPR teleport (default 0.15)
    pub epsilon: f64,         // APPR precision (default 1e-6)
    pub mode: Mode,           // Fused or Separated
    pub weights: [f64; 3],    // [appr, coupling, cocitation] wRRF weights
    pub rrf_k: usize,         // RRF smoothing constant (default 60)
    pub lateral_pct: usize,   // % of limit for laterals in separated mode (default 25)
    pub limit: usize,         // max results (default 200)
}

pub fn compute(graph: &Graph, seed: i64, params: &ExpandParams) -> ExpandResult
```

Python binding:
```python
graph.expand(
    seed: int,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    mode: str = "fused",           # "fused" | "separated"
    weights: tuple = (0.7, 0.15, 0.15),
    rrf_k: int = 60,
    lateral_pct: int = 25,
    limit: int = 200,
) -> tuple[list[tuple[int, float]], dict]
```

## CLI Interface

```bash
# Fused mode (default — focused exploration)
quarry expand <work_id>
quarry expand <work_id> --mode fused

# Separated mode (broad survey)
quarry expand <work_id> --mode separated

# Options
quarry expand <work_id> \
  --limit 200 \
  --mode fused|separated \
  --format json|table

# Resolve by DOI or PMID
quarry expand --doi 10.1126/science.adp5637
quarry expand --pmid 39251588
```

## Signals

### APPR (structural proximity)
Push-based Approximate PPR on full CSR graph. O(1/ε), ~1-2s.

### Coupling (AA-weighted, cosine-normalized)
Bibliographic coupling: papers sharing references with seed.
Adamic-Adar weighting: niche shared references score higher.
Cosine normalization: reference list size bias removed.

### Co-citation (AA-weighted, cosine-normalized)
Papers co-cited with seed by third papers.
Adamic-Adar weighting: niche citers count more.
Cosine normalization applied.

## Relation Tagging

Deterministic classification based on edge existence:

| seed → paper | paper → seed | Relation |
| :----------: | :----------: | -------- |
| yes | no | `foundation` — seed cites this paper |
| no | yes | `follow-up` — this paper cites seed |
| yes | yes | `mutual` |
| no | no | `lateral` — discovered via coupling/cocitation |

## Evaluation Results (5 seeds)

| Metric | Fused (0.7/0.15/0.15) | Separated (75/25) | APPR-only |
|--------|----------------------|-------------------|-----------|
| avg ref recovery | 78.6% | 78.9% | 80.9% |
| avg citer recovery | 78-80% | 77.9% | 86.3% |
| avg lateral count | 60-62 | 67 | 47 |
| lateral quality | all on-topic (5/5 seeds) | all on-topic | 7/10 good laterals |

Key finding: APPR alone misses 3 high-quality lateral papers that
coupling/cocitation discovers. Both fused and separated modes recover them.
Separated mode guarantees more lateral slots.

## Future Phases

| Phase | Addition |
|-------|---------|
| 2 | Direction filtering (`--direction forward|reverse|both`) + HKPR |
| 3 | Multi-seed (`quarry expand <id1> <id2>`) |

## Open Questions

- [ ] wRRF weight tuning across more diverse seeds
- [ ] `technique` relation: requires MeSH qualifier (22M coverage)
- [ ] Multi-seed weight determination
