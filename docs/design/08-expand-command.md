# quarry expand — Command Design

> Status: Phase 1b complete.

## Definition

`quarry expand` takes a seed paper and returns a ranked subgraph of related
papers with full metadata from the citation graph. All operations are
deterministic. quarry returns **complete data** — filtering/sorting is left
to downstream consumers (jq, SQL, agents).

## Design Principle

quarry is a **data pipeline**, not a query engine. It computes structural
signals (APPR, coupling, cocitation) in Rust, enriches with PG metadata,
and outputs complete JSON. No `--sort`, `--filter`, or `--rel` options —
use `jq`, DuckDB, or pandas downstream.

```bash
# quarry outputs data
quarry expand W4406019873 --format json > results.json

# downstream filters
jq '.papers[] | select(.relation == "lateral")' results.json
jq '.papers | sort_by(-.quality.cnp)' results.json
```

## Two Modes

| Mode | Use case | How it works |
|------|----------|-------------|
| `fused` (default) | Focused exploration | wRRF merges all candidates into single ranked list |
| `separated` | Broad survey | APPR top-75% + guaranteed lateral top-25% |

## CLI Interface

```bash
quarry expand <seed>                              # work_id_int
quarry expand W4406019873                         # W-prefix
quarry expand 10.1126/science.adp5637             # DOI
quarry expand https://doi.org/10.1126/science.adp5637  # full DOI

# Options (algorithm parameters only — no sort/filter)
quarry expand <seed> \
  --mode fused|separated \       # default: fused
  --limit 200 \                  # default: 200
  --format table|json|detail \   # default: table
  --alpha 0.15 \                 # APPR teleport
  --epsilon 1e-6                 # APPR precision
```

## Output Schema

### JSON (`--format json` or `--format detail`)

`detail` includes `abstract` field; `json` omits it.

```json
{
  "meta": {
    "quarry_version": "0.2.0",
    "graph": {
      "num_nodes": 182810383,
      "num_edges": 3781541110,
      "build_date": "2026-03-31T18:14:08Z"
    }
  },
  "seed": {
    "work_id": 4406019873,
    "title": "Landscape profiling of PET depolymerases...",
    "doi": "https://doi.org/10.1126/science.adp5637",
    "year": 2025
  },
  "params": {
    "mode": "fused",
    "alpha": 0.15,
    "epsilon": 1e-6,
    "weights": [0.7, 0.15, 0.15],
    "limit": 200,
    "_note": "weights not yet exposed in Python API — uses Rust defaults"
  },
  "papers": [
    {
      "rank": 1,
      "work_id": 4385342215,
      "doi": "https://doi.org/10.1002/...",
      "title": "Discovery and rational engineering of PET hydrolase...",
      "abstract": "...",
      "year": 2023,
      "relation": "foundation",
      "scores": {
        "fused": 0.014714,
        "appr": 0.001646
      },
      "quality": {
        "cited_by": 106,
        "cnp": 0.99,
        "fwci": 7.7,
        "rcr": 8.0
      },
      "bridges": null
    },
    {
      "rank": 42,
      "work_id": 4412048048,
      "doi": "https://doi.org/10.1021/...",
      "title": "Harnessing protein language model for PET hydrolases",
      "abstract": "...",
      "year": 2025,
      "relation": "lateral",
      "scores": {
        "fused": 0.007717,
        "appr": null
      },
      "quality": {
        "cited_by": 10,
        "cnp": 0.96,
        "fwci": 5.0,
        "rcr": null
      },
      "bridges": [
        {
          "work_id": 2294707565,
          "type": "shared_ref",
          "title": "A bacterium that degrades and assimilates PET",
          "weight": 0.12
        },
        {
          "work_id": 4224988655,
          "type": "shared_ref",
          "title": "ML-aided engineering of hydrolases for PET",
          "weight": 0.45
        }
      ]
    }
  ],
  "stats": {
    "appr_candidates": 2366,
    "coupling_candidates": 137427,
    "cocitation_candidates": 2437,
    "returned": 200,
    "elapsed_s": 1.07
  }
}
```

### Schema Field Reference

**scores**:

- `fused`: wRRF fused score (fused mode) or APPR/lateral score (separated mode)
- `appr`: raw APPR score. null if paper only found via coupling/cocitation.

**quality** (all nullable — see 03-layer-quality.md for why):

- `cited_by`: raw citation count (~100% coverage)
- `cnp`: citation_normalized_percentile, 0-1, field+year normalized (87-99%)
- `fwci`: Field-Weighted Citation Impact, unbounded (87-99%, ρ=0.99 with cnp)
- `rcr`: Relative Citation Ratio, iCite, PubMed only (55-85%)

**bridges** (lateral papers only, null for foundation/follow-up/mutual):

- All shared refs/citers returned (no top-N cutoff)
- Sorted by AA weight descending (structural rarity proxy)
- `type`: `shared_ref` (seed and paper both cite bridge) or `shared_citer` (third paper cites both)
- `weight`: Adamic-Adar weight = 1/log(indegree or outdegree)

**relation**:

- `foundation`: seed cites this paper (seed → paper edge exists)
- `follow-up`: this paper cites seed (paper → seed edge exists)
- `mutual`: both directions
- `lateral`: no direct citation edge. Discovered via coupling/cocitation.

### Quality Signals — Selection Rationale

| Signal | Included | Rationale |
|--------|----------|-----------|
| cnp | ✅ | 0-1 range, field+year normalized, 87-99% coverage. Primary quality indicator. |
| fwci | ✅ | ρ=0.99 with cnp but absolute scale. Some users prefer this. |
| cited_by | ✅ | ~100% coverage. Biased but universally available fallback. |
| rcr | ✅ | High quality for biomedical. 55-85% coverage (PubMed only). |
| apt | ❌ | 7.9% coverage. Clinical translation only — misleading for basic science. |
| nih_percentile | ❌ | 6.4% coverage. NIH-funded only — extreme selection bias. |
| fcr | ❌ | Baseline for RCR. Not useful standalone. |
| Global PageRank | ❌ | Not precomputed. cited_by serves as rough proxy. |

See [03-layer-quality.md](03-layer-quality.md) for detailed evaluation.

### NULL Handling

All quality fields are nullable. NULL means "not measured" — not "zero" or
"average". No imputation. See [07-missing-data.md](07-missing-data.md).

Common NULL patterns:

- Preprints (bioRxiv/arXiv): cnp, fwci, rcr all NULL
- Chemistry journals: rcr NULL (not in PubMed)
- Very new papers (2025+): cnp/fwci may be NULL or near-zero

## Architecture

```
Rust expand()                          Python CLI
──────────────                        ──────────
APPR(seed, α, ε)                      PG metadata enrichment:
coupling_aa(seed)          →            title, doi, year, abstract
cocitation_aa(seed)                     cnp, fwci, rcr, cited_by
fused/separated fusion                 relation tagging
seed exclusion                         bridge title enrichment
bridge collection (pass 2)            JSON serialization
→ Vec<ExpandPaper>                    → stdout
```

## Evaluation Results (5 seeds)

| Metric | Fused (0.7/0.15/0.15) | Separated (75/25) | APPR-only |
|--------|----------------------|-------------------|-----------|
| avg ref recovery | 78.6% | 78.9% | 80.9% |
| avg citer recovery | 78-80% | 77.9% | 86.3% |
| avg lateral count | 60-62 | 67 | 47 |
| lateral quality | all on-topic (5/5 seeds) | all on-topic | 7/10 |

Key findings:

- APPR alone misses 3+ high-quality lateral papers per seed
- Both modes recover them; separated guarantees more lateral slots
- Lateral papers verified on-topic via PMC abstract review (all 5 seeds)
- fwci/cnp capture landmark papers but demote seed-relevant niche work

## Future Phases

| Phase | Addition |
|-------|---------|
| 1b+ | Bridge collection (pass 2, lazy) + quality metadata enrichment |
| 2 | Direction filtering + HKPR |
| 3 | Embedding bridge reranking (**CRITICAL** — AA weight alone cannot distinguish niche-relevant from niche-coincidental) |
| 4 | Multi-seed + temporal |

## Open Questions

- [ ] Weighted APPR (CSR edge weight = fwci) vs post-hoc — requires CSR structure change
- [ ] `technique` relation: requires MeSH qualifier (22M coverage)
- [ ] Multi-seed weight determination
