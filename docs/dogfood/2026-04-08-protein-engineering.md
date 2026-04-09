---
session_id: dogfood-2026-04-08-protein-engineering
date: 2026-04-08
persona: surveyor
topic: "Protein engineering field mapping via expand → mesh-summary drill-down"
seeds: [W2767044445, W4296032638, W2018295245, W2061932208]
commands_used: [mesh, info, expand, shrink, bridge, sql]
commands_count: 16
friction_count: 5
copy_paste_count: 8
scores:
  discoverability: 4
  output_clarity: 5
  workflow_continuity: 3
  information_sufficiency: 4
  command_ergonomics: 3
  total: 19
previous_session: dogfood-2026-04-08-rna-degrader-landscape
score_delta: -5
improvements_tested:
  - "expand --mesh-summary for field mapping"
  - "multi-seed bridge for paradigm connection"
  - "shrink coverage comparison across subfields"
design_refs: []
---

# Dogfood: Protein Engineering Field Mapping

## Session Context

**Goal**: Map the entire protein engineering field using expand → mesh-summary
to discover subtopics, then drill into each subtopic to find key papers.

**Constraints**: LanceDB still building — FTS and embedding search unavailable.
Had to rely on MeSH vocabulary + SQL + citation graph exclusively.

**Persona**: Surveyor planning a review paper / grant proposal.

## Command Sequence Log

```
CMD  1: quarry mesh "protein engineering"                          → D019020 only     0.5s
CMD  2: quarry mesh D019020 --tree                                 → hierarchy         0.3s
CMD  3: quarry mesh "protein engineering" --limit 5                → same result       0.4s
       quarry mesh "mutagenesis" --limit 5                         → 5 matches         0.3s
CMD  4: quarry info W2767044445 --mesh                             → Arnold Nobel rev  0.3s
       ★ Discovery: D015202 "Protein Engineering" exists as separate MeSH descriptor
CMD  5: quarry mesh D015202 --tree                                 → no children       0.3s
       quarry mesh D015202                                         → 9426 papers       1.1s
CMD  6: quarry expand W2767044445 --mesh-summary --limit 200       → biocatalysis axis 5.7s
CMD  7: quarry mesh "de novo protein"                              → no match          0.3s
CMD  8: quarry sql (de novo design papers with D015202)            → 10 results        1.1s
CMD  9: quarry sql (ML/DL protein engineering papers >= 2020)      → ProteinMPNN top   0.5s
CMD 10: quarry expand W4296032638 --mesh-summary --limit 200       → ML design axis    6.3s
CMD 11: quarry mesh "antibodies" + "enzyme stability"              → D004795 found     0.6s
CMD 12: quarry sql (Protein Eng ∩ Antibodies cross-MeSH)           → yeast display     1.0s
CMD 13: quarry expand W2018295245 --mesh-summary --limit 200       → antibody axis     4.9s
CMD 14: quarry shrink W2767044445 --top 7                          → 55% coverage      8.0s
       quarry shrink W4296032638 --top 7                           → 94% coverage      9.0s
CMD 15: quarry bridge W2767044445 W4296032638 --limit 10           → sp=8, AF2 hub     4.1s
CMD 16: quarry expand W2061932208 --mesh-summary --limit 100       → industrial enz    5.0s
TOTAL: 16 commands, ~50s wall time
```

**Annotations**:
- **Unnecessary commands**: CMD 1-3 were spent searching for the correct MeSH
  descriptor. `quarry mesh "protein engineering"` should have found D015202
  directly — instead it returned D019020 (Directed Molecular Evolution).
  Discovery of D015202 was accidental via `info --mesh`.
- **Repeated patterns**: expand --mesh-summary used 4 times (CMD 6, 10, 13, 16).
  Each takes 5-9s — cumulative 22s just for mesh summaries.
- **Copy-paste points**: 8 total — work_ids between mesh → info → expand → shrink
  → bridge transitions.
- **Slow commands**: shrink (8-9s) and expand --mesh-summary (5-7s) are the
  bottlenecks. Both involve expand + post-processing.
- **Transitions**: mesh → sql → expand was the most common pattern for
  discovering new axis seeds. This worked but required SQL knowledge.

## Five-Dimension Rating

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Discoverability | 4 | expand --mesh-summary is excellent for field mapping. Lost 1 point because mesh search missed D015202, requiring accidental discovery via info --mesh. |
| Output clarity | 5 | mesh-summary bar chart + topic distribution is immediately actionable. expand table clearly shows foundation/follow-up/lateral classification. |
| Workflow continuity | 3 | No prescribed "field mapping" workflow. Had to improvise: mesh → seed discovery → expand → mesh-summary → pick new seed → repeat. The loop works but each iteration requires manual seed selection and copy-paste. |
| Information sufficiency | 4 | mesh-summary + shrink + bridge gave comprehensive field picture. Lost 1 point because there's no way to compare mesh-summaries across axes side-by-side. |
| Command ergonomics | 3 | 8 copy-paste points. SQL required to find cross-MeSH papers. shrink takes 8-9s. mesh search has false negatives (missed exact descriptor name). |
| **Total** | **19** | |

## Friction Log

### FRICTION 1: MeSH search misses exact descriptor name
```
FRICTION: Finding "Protein Engineering" MeSH descriptor
COMMAND: quarry mesh "protein engineering"
OUTPUT: D019020 Directed Molecular Evolution (only)
PROBLEM: D015202 "Protein Engineering" exists but wasn't returned
WANTED: Exact name match should take priority. Show all matching descriptors.
```

### FRICTION 2: work_mesh column name guessing
```
FRICTION: SQL query on work_mesh table
COMMAND: quarry sql (... AND wm1.is_major ...)
OUTPUT: UndefinedColumn error
PROBLEM: Column is "is_major_topic" not "is_major". Had to run --schema.
WANTED: Fuzzy column suggestion in error message, or always show --schema hint.
```

### FRICTION 3: SQL join produces duplicates from qualifier rows
```
FRICTION: Cross-MeSH SQL query returns duplicate papers
COMMAND: quarry sql (JOIN work_mesh twice)
OUTPUT: Same paper appears 2x due to different qualifier_ui rows
PROBLEM: work_mesh has qualifier-level granularity that inflates JOINs
WANTED: Documentation note about qualifier duplication, or a DISTINCT hint.
```

### FRICTION 4: expand --mesh-summary is slow for iterative drilling
```
FRICTION: Repeated expand --mesh-summary for multi-axis mapping
COMMAND: quarry expand W... --mesh-summary --limit 200 (4 times)
OUTPUT: 5-9 seconds each
PROBLEM: Cumulative 22s of waiting breaks surveyor flow
WANTED: Cached mesh-summary, or pre-computed topic distributions.
```

### FRICTION 5: No "field mapping" compound command
```
FRICTION: Mapping a field requires manual multi-axis iteration
COMMAND: mesh → sql → expand → mesh-summary → pick seed → repeat
OUTPUT: Works but tedious
PROBLEM: Surveyor persona's core workflow is a manual loop
WANTED: A "quarry map" command that takes a MeSH descriptor and automatically
discovers axes via top papers, runs expand --mesh-summary on each, and produces
a unified topic map.
```

## Additional Observations

### Surprises (positive)
- **mesh-summary is the killer feature for surveyors**. The bar chart
  immediately reveals field structure without reading 200 paper titles.
- **shrink coverage as field characterization**: 94% coverage for ML design
  (centralized) vs 55% for biocatalysis (distributed) is itself a
  meaningful signal about field structure.
- **bridge cocitation reveals AF2 as the universal hub** connecting directed
  evolution and ML design. This is exactly the kind of insight a surveyor needs.

### Dead ends
- `quarry mesh "de novo protein"` → no match. MeSH vocabulary doesn't have
  a "de novo protein design" descriptor. Had to fall back to SQL.
- `quarry mesh "antibody engineering"` → no match. Same issue.

### Missing commands
- **quarry map \<descriptor_ui>**: Automated field mapping as described in
  Friction 5.
- **quarry compare \<seed1> \<seed2>**: Side-by-side mesh-summary comparison.
- **quarry mesh --fuzzy**: Broader MeSH matching that includes partial name
  matches and entry terms.

### Workflow suggestions
- **mesh → expand pipeline**: `quarry mesh D015202 --expand-top 3` could
  automatically expand the top 3 papers and merge mesh-summaries.
- **cross-axis shrink**: `quarry shrink W... W... --multi-seed` that runs
  expand from multiple seeds and finds the minimum covering set across all.

## Research Findings

### Protein Engineering — 4-Axis Field Map

| Axis | Seed | Key MeSH Topics | Shrink Coverage | Character |
|------|------|-----------------|-----------------|-----------|
| Directed evolution / biocatalysis | W2767044445 (Arnold 2017) | Biocatalysis, Enzymes, Cyt P450 | 55% (distributed) | Experimental, iterative |
| ML-guided design | W4296032638 (ProteinMPNN 2022) | Deep Learning, Protein Folding, AI | 94% (centralized) | Computational, generative |
| Antibody engineering | W2018295245 (Yeast display 2006) | Saccharomyces, Peptide Library, mAb | not tested | Platform-based, therapeutic |
| Industrial enzyme stability | W2061932208 (Thermostable enz 2003) | Lipase, α-Amylase, Cellulase | not tested | Application-driven |

### Hub paper connecting all axes
- **AlphaFold2** (W3177828909, 42952 citations) — cocitation bridge between
  directed evolution and ML design. Universal reference point.

### Key papers per axis (shrink results)

**Directed evolution (7 papers, 55% coverage)**:
1. W2554528613 — Directed evolution of cytochrome c (Science 2016, 568 cited)
2. W2991758999 — Design of in vitro biocatalytic cascade (Science 2019, 616)
3. W2118265845 — Exploring protein fitness landscapes (Nat Rev Genet 2009, 1183)
4. W4281945460 — Road to fully programmable protein catalysis (Nature 2022, 333)
5. W2946012049 — Bridging the gap (Nat Commun 2019, 233)
6. W4280570348 — Directed evolution of new catalytic activity (Science 2022, 144)
7. W4391849830 — ML for functional protein design (Nat Biotechnol 2024, 214)

**ML-guided design (7 papers, 94% coverage)**:
1. W4383957026 — De novo design of protein structure and function (Nature 2023, 1692) ← RFdiffusion
2. W4286488966 — Scaffolding protein functional sites (Science 2022, 448)
3. W4391849830 — ML for functional protein design (Nat Biotechnol 2024, 214)
4. W4210861939 — Protein sequence design with deep learning (Nat Commun 2022, 167)
5. W4391849760 — Sparks of function by de novo design (Nat Biotechnol 2024, 96)
6. W4288066876 — ProtGPT2 (Nat Commun 2022, 717)
7. W4384820649 — Mega-scale experimental protein fitness (Nature 2023, 306)
