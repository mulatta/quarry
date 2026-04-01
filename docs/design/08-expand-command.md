# quarry expand — Command Design

> Status: Design. Implementation follows Phase 1a → 3.

## Definition

`quarry expand` takes one or more seed papers and returns a ranked subgraph
of related papers from the citation graph. All operations are deterministic.

## Pipeline

```
Input                    Algorithm              Signal              Output
─────                    ─────────              ──────              ──────

seed work_id(s)    →  APPR / HKPR push   →  structural score     ─┐
   + direction          (CSR mmap direct,      (graph proximity)  │
   + params             no subgraph extract)                      ├→ RRF fusion → ranked list
                                                                  │     + relation tag
                     bib_coupling(seed)   →  coupling score      ─┤     + metadata
                        (shared refs)         (methodology sim)   │
                                                                  │
                     co_citation(seed)    →  cocitation score    ─┘
                        (co-cited by)          (community signal)
```

## CLI Interface

```bash
# Single seed
quarry expand <work_id>

# Multiple seeds
quarry expand <work_id1> <work_id2> <work_id3>

# Options
quarry expand <work_id> \
  --direction both|forward|backward \
  --focus foundation|follow-up|technique|lateral|all \
  --limit 200 \
  --sort fused|structural|coupling|cocitation \
  --primary <work_id>        # boost weight for this seed
  --weights 0.6,0.2,0.2      # explicit seed weights
  --format json|table

# Resolve by DOI or PMID
quarry expand --doi 10.1126/science.adp5637
quarry expand --pmid 39251588
```

### Default Behavior

```bash
quarry expand W4406019873
# direction=both, focus=all, limit=200, sort=fused, format=table
```

### Seed Exclusion

Seed papers are always excluded from the output. APPR returns seed as top-1
(highest score by definition), but expand filters it out before ranking/output.
This is handled in the expand command (Python), not in Rust — the APPR API
returns the raw algorithm result including seed.

## Output Structure

```json
{
  "seeds": ["W4406019873"],
  "params": { "direction": "both", "limit": 200, "algorithm": "hkpr", "t": 5 },
  "papers": [
    {
      "work_id": "W...",
      "title": "...",
      "year": 2016,
      "cited_by": 3111,
      "type": "article",
      "relation": "foundation",
      "scores": {
        "structural": 0.045,
        "coupling": 15,
        "cocitation": 63,
        "fused": 0.087
      }
    }
  ],
  "edges": [{ "source": "W...", "target": "W..." }],
  "stats": {
    "candidates_explored": 12000,
    "returned": 200,
    "elapsed_ms": 2100,
    "by_relation": { "foundation": 45, "follow-up": 32, "lateral": 18 }
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

## Multi-seed Behavior

| Mode              | Semantics                   | Algorithm                                                 |
| ----------------- | --------------------------- | --------------------------------------------------------- |
| `union` (default) | Papers related to ANY seed  | Multi-seed APPR: `residual = {s: w/sum for s,w in seeds}` |
| `intersection`    | Papers related to ALL seeds | Individual APPR per seed → min score                      |
| `bridge`          | Papers connecting seeds     | Pairwise shortest paths → union                           |

### Seed Weights

- Default: equal (`1/N` each)
- `--primary W...`: named seed gets `2×` weight
- `--weights 0.6,0.2,0.2`: explicit (must sum to ~1.0)
- Agent decides weights based on context; quarry core accepts any distribution

## Algorithm Selection

| Phase | Structural                    | Signals                    | Fusion    |
| ----- | ----------------------------- | -------------------------- | --------- |
| 1a    | APPR (push-based, local)      | structural only            | None      |
| 1b    | APPR                          | + coupling + cocitation    | RRF       |
| 1c    | APPR                          | + relation tags            | RRF + CLI |
| 2     | HKPR (seed domination solved) | same                       | RRF       |
| 3     | HKPR                          | + multi-seed + search pipe | RRF       |

## Open Questions

- [ ] APPR vs HKPR: implement both and compare, or HKPR only?
- [ ] RRF signal weights: equal, or structural-first?
- [ ] Fused score minimum cutoff: fixed or adaptive?
- [ ] Coupling/cocitation threshold: distribution-based adaptive
- [ ] `technique` relation: requires MeSH qualifier (22M coverage)
- [ ] Multi-seed weight determination: user, agent, or heuristic?
- [ ] How many search results become expand seeds?
