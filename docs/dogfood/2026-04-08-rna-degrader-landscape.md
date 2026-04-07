---
session_id: dogfood-2026-04-08-rna-degrader-landscape
date: 2026-04-08
persona: reader
topic: "RNA-targeting small molecule degrader landscape"
seeds: [W4377982067]
commands_used: [info, expand, mesh, sql, bridge, shrink]
commands_count: 10
friction_count: 1
copy_paste_count: 5
scores:
  discoverability: 5
  output_clarity: 5
  workflow_continuity: 5
  information_sufficiency: 4
  command_ergonomics: 5
  total: 24
previous_session: dogfood-2026-04-07-abe-ptm-crispr
score_delta: 2
improvements_tested:
  - "DOI resolve in info command"
  - "shrink --venue custom (field-specific journals)"
  - "reader persona full pipeline"
design_refs: []
---

# Dogfood: RNA-targeting small molecule degrader landscape

## Session Context

Reader persona starting from DOI 10.1038/s41586-023-06091-8
(Disney lab, Nature 2023). Goal: understand the research landscape
around this RNA degrader paper — what came before, what followed,
and how it connects to the broader targeted degradation field.

Constraint: vsearch unavailable (embedding building).

## Command Sequence Log

| # | Command | Result | Time | Notes |
|---|---------|--------|------|-------|
| 1 | `info "https://doi.org/10.1038/s41586-023-06091-8" --mesh` | W4377982067 identified | <1s | DOI resolve works |
| 2 | `expand W4377982067 --limit 25 --mesh-summary` | 25 papers + MeSH bar | 0.5s | RNA-targeting field |
| 3 | `info W... W... W...` (6 papers) | Foundation details | <1s | Disney lab lineage |
| 4 | `expand W2925099802 --limit 20 --mesh-summary` | 20 papers | 1.4s | Broader landscape |
| 5 | `mesh "proteolysis targeting"` | D000094243 PROTAC | <1s | MeSH discovery |
| 6 | `bridge W4377982067 W4205379795` | sp=1, 5 types | 1.0s | RNA↔PROTAC |
| 7 | `info W4290672950 --mesh` | Nat Rev review | <1s | — |
| 8 | `shrink --top 5` (NCS+) | 53% coverage | 0.5s | Pool too small |
| 9 | `shrink --top 5 --no-foundation` | 50% | 0.5s | Recent trends |
| 10 | `shrink --top 7 --venue custom` | **84% coverage** | 0.5s | Chemical bio journals |

### Sequence Annotations
- Copy-paste: 5 times (seed → expand, expand → info, etc.)
- Unnecessary commands: none
- Backtracking: CMD 5 "targeted degradation" → "proteolysis targeting" (1x)
- Performance: all commands <1.5s, no flow interruption

## Five-Dimension Rating

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Discoverability | 5/5 | DOI → info → expand → bridge seamless |
| Output clarity | 5/5 | venue column, MeSH summary, cited_by all present |
| Workflow continuity | 5/5 | Reader's natural flow: info → expand → bridge → shrink |
| Information sufficiency | 4/5 | -1: abstract not shown by default in bridge results |
| Command ergonomics | 5/5 | shrink --venue custom: 53% → 84% coverage |

**Total: 24/25**

## Friction Log

```
FRICTION: Did not use --format detail for bridge results
COMMAND: quarry bridge W4377982067 W4205379795 (table mode)
OUTPUT:  10 coupling bridge papers (title only)
PROBLEM: Assessed relevance from titles alone when --format detail
         (which includes full abstracts) was available. Not a tool
         limitation but a usage pattern issue.
WANTED:  Agent/skill should default to --format detail for bridge
         when assessing paper relevance, or use info for key papers.
```

## Additional Observations

**Surprises (positive)**:
- DOI resolve works perfectly — raw https://doi.org/... URL accepted
- shrink --venue custom is essential for chemical biology (NCS+ coverage
  only 53%, custom venues 84%)
- bridge sp=1 between RNA degrader and PROTAC — direct connection

**Workflow discovery**:
- Reader persona optimal path: `info --mesh → expand --mesh-summary →
  bridge(seed, related_field) → shrink --venue custom`
- NCS+ preset is insufficient for chemical biology — field-specific
  venue lists are practically required

## Research Findings

### Targeted Degradation Paradigm Tree

```
Targeted Degradation
├── Protein degradation
│   ├── PROTAC (2001, Crews) — ubiquitin-proteasome
│   ├── LYTAC (2020) — lysosome
│   ├── AUTAC (2019) — autophagy
│   └── Molecular glues
└── RNA degradation
    ├── RIBOTAC (Disney lab)
    │   ├── Inforna 2.0 (2016) — design platform
    │   ├── In vivo proof (2020, PNAS)
    │   ├── Reprogramming (2021, JACS)
    │   └── Inactive → active (2023, Nature) ← this paper
    └── Other: ASO, siRNA, small molecule binders

Bridge: RNA degrader ↔ PROTAC review
  sp=1, cocitation includes PROTAC(2001), LYTAC(2020), AUTAC(2019)
  Common concept: proximity-induced pharmacology
```

### Reading List (7 papers, 84% coverage)

| # | Year | Venue | Paper | Role |
|---|------|-------|-------|------|
| 1 | 2022 | Nat Rev Drug Disc | Targeting RNA structures | Field overview |
| 2 | 2016 | ACS Chem Biol | Inforna 2.0 | Design platform |
| 3 | 2018 | JACS | Small Molecule Targeted Recruitment | RIBOTAC mechanism |
| 4 | 2018 | Nat Rev Drug Disc | Principles for targeting RNA | RNA druggability |
| 5 | 2016 | Nat Chem Bio | Precise small-molecule recognition | Interaction principles |
| 6 | 2014 | Science | RNase L structure | Structural basis |
| 7 | 2017 | Angew Chem | Key Physicochemical Properties | Drug design |
