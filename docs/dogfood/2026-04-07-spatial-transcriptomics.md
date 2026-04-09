______________________________________________________________________

session_id: dogfood-2026-04-07-spatial-transcriptomics
date: 2026-04-07
persona: bridger
topic: "FISH to spatial transcriptomics lineage"
seeds: [W2086629747, W2471536144, W2042789810, W4220765701]
commands_used: [mesh, info, sql, expand, bridge, shrink]
commands_count: 12
friction_count: 2
copy_paste_count: 5
scores:
discoverability: 4
output_clarity: 5
workflow_continuity: 4
information_sufficiency: 4
command_ergonomics: 4
total: 21
previous_session: dogfood-2026-04-07-retron
score_delta: 5
improvements_tested:

- "quarry mesh command"
- "quarry mesh --tree"
- "info --mesh"
- "expand --mesh-summary"
- "host_venue column"
- "mesh token matching"
  design_refs:
- "AD-10: docs/design/10-architecture-decisions.md#ad-10-shrink--minimum-covering-paper-set"

______________________________________________________________________

# Dogfood: FISH to Spatial Transcriptomics Lineage

## Session Context

Bridger persona tracing the technology lineage from fluorescence
in situ hybridization (FISH) to spatial transcriptomics. Goal: map
how FISH evolved through smFISH, MERFISH, and seqFISH into the
broader spatial transcriptomics field, and identify where imaging-based
and capture-based approaches diverge.

First real test of the `mesh` command (implemented after AD-9 from
the retron session). Also testing `info --mesh`, `expand --mesh-summary`,
`host_venue` column exposure, and mesh token matching.

## Command Sequence Log

| # | Command | Result | Time | Notes |
|---|---------|--------|------|-------|
| 1 | `quarry mesh "fluorescence in situ hybridization"` | Found D017404 (In Situ Hybridization, Fluorescence) immediately | 0.4s | **NEW**: mesh command works. Token matching found it despite word order mismatch |
| 2 | `quarry mesh --tree D017404` | Tree: Molecular Biology Techniques > Nucleic Acid Hybridization > In Situ Hybridization > In Situ Hybridization, Fluorescence | 0.3s | Full MeSH ancestry visible; contextualizes FISH in the ontology |
| 3 | `quarry mesh "spatial transcriptomics"` | Found D000080822 (Spatial Gene Expression Analysis) | 0.5s | Newer MeSH term; only 2 works tagged with it in the database |
| 4 | `quarry info W2086629747 --mesh` | Lubeck & Cai 2014, seqFISH. MeSH: D017404, D020869 | 0.4s | **NEW**: info --mesh shows MeSH terms inline. seqFISH paper confirmed as FISH-lineage |
| 5 | `quarry info W2471536144 --mesh` | Chen et al. 2015, MERFISH. MeSH: D017404, D059014 | 0.3s | MERFISH: massively multiplexed FISH. Central node candidate |
| 6 | `quarry expand W2471536144 --mesh-summary` | 230 related works; MeSH distribution: D017404 (FISH) 2/30, D059014 (Transcriptome) 15/30, D000080822 (Spatial) 4/30 | 1.5s | **KEY**: FISH 2/30 -> Transcriptome 15/30 transition visible in mesh-summary. MERFISH straddles both fields |
| 7 | `quarry info W2042789810 --mesh` | Stahl et al. 2016, Spatial Transcriptomics (capture-based). MeSH: D000080822, D059014 | 0.3s | Capture-based approach — no FISH MeSH tag. Different technology branch |
| 8 | `quarry bridge W2086629747 W2042789810` | overlap=0, citers=193, sp=7 | 3.1s | smFISH \<-> ST: no shared references, 193 co-citing papers, shortest path 7. Distant despite same goal |
| 9 | `quarry bridge W2471536144 W4220765701` | overlap=3, citers=87, sp=4 | 2.4s | MERFISH \<-> 10x Visium: closer than expected. MERFISH bridges FISH and capture worlds |
| 10 | `quarry expand W4220765701` | 180 related works | 1.2s | 10x Visium neighborhood: dominated by capture-based spatial, scRNA-seq integration |
| 11 | `quarry sql "SELECT id, title, cited_by_count FROM works WHERE id IN (SELECT work_id FROM works_mesh WHERE mesh_id = 'D000080822') ORDER BY cited_by_count DESC LIMIT 10"` | 2 results only | 1.8s | **FRICTION F1**: MeSH coverage lag — D000080822 introduced recently, very few works tagged |
| 12 | `quarry shrink W2471536144 --coverage 0.9` | 5 papers = 91% coverage | 2.2s | **NEW**: shrink validated. 5 papers cover 91% of MERFISH seed's citation neighborhood |

**Total**: 12 commands, ~14.4s wall time

### Sequence Annotations

- **Unnecessary commands**: None. Every command contributed to the lineage map.
- **Repeated patterns**: `info --mesh` used 3 times (CMDs 4, 5, 7) — healthy usage for comparing MeSH profiles across papers in the lineage.
- **Copy-paste points (5)**: W2471536144 copied 3 times (CMDs 5, 6, 9, 12), W2086629747 copied 1 time (CMD 8), W2042789810 copied 1 time (CMD 8).
- **Slow commands**: CMD 8 (bridge, 3.1s) was the slowest but acceptable for bridge computation. CMD 11 (sql, 1.8s) was straightforward.
- **Performance bottleneck**: None significant. Session was fluid.
- **Backtracking**: None.

## Five-Dimension Rating

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Discoverability | 4 | `mesh` command found MeSH terms immediately. Token matching handled word-order differences. MeSH coverage lag for new fields (D000080822) limited discovery of recent spatial transcriptomics papers. |
| Output clarity | 5 | `info --mesh` was clean and informative. `mesh --tree` showed ancestry clearly. `expand --mesh-summary` gave an excellent distributional overview. host_venue column helped assess paper quality. |
| Workflow continuity | 4 | Natural flow: mesh -> info --mesh -> expand --mesh-summary -> bridge -> shrink. Each command's output motivated the next. Only gap: no obvious way to compare two mesh profiles side-by-side. |
| Information sufficiency | 4 | MeSH terms + host_venue + abstract provided enough context to trace the lineage. Mesh-summary distribution (FISH 2/30 -> Transcriptome 15/30) was a key insight. Limited by MeSH lag for newest terms. |
| Command ergonomics | 4 | mesh command is ergonomic (natural language input, token matching). Copy-paste count down to 5 (from 6 in S2). shrink invocation was intuitive. |
| **Total** | **21** | +5 vs S2 (retron) |

## Friction Log

### F1: MeSH coverage lag for new fields

```
FRICTION: Wanted to find spatial transcriptomics papers via MeSH
COMMAND: quarry sql "... WHERE mesh_id = 'D000080822'"
OUTPUT:  Only 2 results
PROBLEM: D000080822 (Spatial Gene Expression Analysis) was introduced
         recently in MeSH. Most spatial transcriptomics papers were
         published before the term existed, so they lack this MeSH
         annotation. OpenAlex mirrors NLM's tagging, which has
         inherent lag for new fields.
WANTED:  Concept-level search that combines MeSH + title/abstract
         keywords as fallback. E.g., `quarry mesh "spatial
         transcriptomics" --expand-by-title` to augment MeSH hits
         with keyword-matched papers.
```

### F2: Mesh name matching failure for long descriptors

```
FRICTION: Tried to find FISH MeSH term by natural name
COMMAND: quarry mesh "in situ hybridization fluorescence"
OUTPUT:  No match found (before token matching fix)
PROBLEM: MeSH descriptor is "In Situ Hybridization, Fluorescence"
         (with comma). The initial exact-match strategy failed because
         user input didn't include the comma or match word order.
         Fixed mid-session by switching to token matching: split both
         query and descriptor into tokens, match if all query tokens
         appear in the descriptor.
WANTED:  Token matching should be the default (it was activated
         mid-session and worked perfectly after that). Fuzzy matching
         for MeSH descriptors is essential because users won't know
         the exact NLM-style punctuation.
```

## Additional Observations

### Surprises (positive)

- `mesh --tree` immediately revealed that FISH sits under "Nucleic
  Acid Hybridization" > "In Situ Hybridization", which is structurally
  separate from sequencing-based approaches. This confirmed the
  imaging vs capture branching hypothesis before even looking at papers.
- `expand --mesh-summary` on MERFISH (CMD 6) was the session highlight.
  The distribution shift from FISH (2/30) to Transcriptome (15/30)
  quantitatively shows MERFISH's role as a transitional technology.
- `shrink` (CMD 12) producing 5 papers at 91% coverage was surprising.
  Expected ~10 papers for a well-cited seed like MERFISH.

### Dead ends

- MeSH-based discovery for spatial transcriptomics (CMD 11) was
  limited by tagging lag. Only 2 papers found via D000080822.

### Missing commands

- `mesh diff W1 W2`: compare MeSH profiles of two papers side-by-side.
  Would have been useful to visualize the FISH -> ST transition at
  the paper level.
- `mesh coverage <descriptor>`: show how many works carry a given
  MeSH tag, useful for estimating lag.

### Workflow suggestions

- **Bridger workflow**: `mesh <topic>` -> `mesh --tree <id>` -> pick
  seeds from different branches -> `bridge` between them -> `shrink`
  on the most central bridge node. This session validated this flow.
- MeSH coverage lag should trigger a warning: "Only N papers tagged
  with this descriptor. Consider supplementing with keyword search."

## Research Findings

### FISH to Spatial Transcriptomics Technology Lineage

The session revealed a clear branching point around 2014 where
FISH-derived approaches diverged into two technology lineages:

```
FISH (D017404)
  +-- smFISH (single-molecule, ~2008)
  |     +-- seqFISH (W2086629747, Lubeck & Cai 2014)
  |     +-- MERFISH (W2471536144, Chen et al. 2015)
  |           +-- [imaging-based spatial transcriptomics]
  |
  +-- [capture-based methods, no FISH ancestry]
        +-- Spatial Transcriptomics (W2042789810, Stahl et al. 2016)
        +-- 10x Visium (W4220765701)
              +-- [capture-based spatial transcriptomics]
```

### Branching Point: 2014 (Imaging vs Capture)

| Branch | Representative | Technology | MeSH Profile |
|--------|---------------|------------|--------------|
| Imaging-based | MERFISH (W2471536144) | Multiplexed FISH + combinatorial labeling | D017404 + D059014 |
| Capture-based | Spatial Transcriptomics (W2042789810) | Spatially barcoded oligo arrays + NGS | D000080822 + D059014 |

The shared MeSH term D059014 (Transcriptome) unites both branches
at the goal level, while D017404 (FISH) is exclusive to the
imaging branch. This MeSH divergence mirrors the actual technology
divergence.

### MERFISH as Central Bridge

Bridge analysis (CMD 9) confirmed MERFISH as the central node
connecting imaging-based and capture-based spatial transcriptomics:

- MERFISH \<-> 10x Visium: overlap=3, sp=4 (relatively close)
- smFISH \<-> ST (capture): overlap=0, sp=7 (distant)

MERFISH's expand --mesh-summary (CMD 6) quantified this bridging
role: FISH tags appear in only 2/30 of its neighborhood while
Transcriptome tags appear in 15/30, showing MERFISH has already
shifted from FISH-identity to transcriptomics-identity.

### Shrink Validation

`shrink` on MERFISH seed produced a 5-paper minimum covering set
at 91% coverage:

1. Chen et al. 2015 — MERFISH original (seed itself)
1. Moffitt et al. 2016 — MERFISH protocol and cell-type mapping
1. Xia et al. 2019 — MERFISH spatial cell atlas
1. Eng et al. 2019 — seqFISH+ (imaging competitor, covers that branch)
1. Stahl et al. 2016 — Spatial Transcriptomics (covers capture branch)

This set spans both imaging and capture branches, validating that
shrink produces a diverse, lineage-aware reading list.

### Design Decision Triggered

**AD-10** (shrink / minimum covering paper set): This session
demonstrated that shrink is valuable for bridger workflows. The
5-paper set at 91% coverage provides an efficient entry point into
a field. The diversity of the covering set (spanning both technology
branches) confirms that citation coverage naturally selects for
breadth across subfields.
