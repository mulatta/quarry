---
session_id: dogfood-2026-04-08-shrink-venue-comparison
date: 2026-04-08
persona: surveyor
topic: "AI × prime editing — shrink venue comparison"
seeds: [W4412993096]
commands_used: [info, expand, shrink, bridge, mesh]
commands_count: 8
friction_count: 3
copy_paste_count: 3
scores:
  discoverability: 5
  output_clarity: 5
  workflow_continuity: 4
  information_sufficiency: 5
  command_ergonomics: 4
  total: 23
previous_session: dogfood-2026-04-08-protein-evolution-regression
score_delta: -2
improvements_tested:
  - "shrink --venue NCS+ vs custom side-by-side"
  - "shrink --no-foundation for recent trends"
design_refs: []
---

# Dogfood: AI × Prime Editing — Shrink Venue Comparison

## Session Context

Surveyor persona testing scenario S2 (shrink venue comparison).
Seed: AI-generated MLH1 small binder improves prime editing efficiency
(Cell 2025). Compare NCS+ vs protein engineering custom venues.

Constraint: FTS/embedding unavailable (building).

## Command Sequence Log

| # | Command | Result | Time | Notes |
|---|---------|--------|------|-------|
| 1 | `info DOI --mesh` | W4412993096, 13 MeSH tags | <1s | |
| 2 | `expand W4412993096 --limit 25 --mesh-summary` | 25 papers | 1.0s | Gene Editing dominant |
| 3 | `shrink W4412993096 --top 7` | 86% NCS+ | 1.0s | |
| 4 | `shrink W4412993096 --top 7 --venue "Nat Biotech,..."` | 86% custom | 1.0s | Same coverage! |
| 5 | `shrink W4412993096 --top 5 --no-foundation` | 67% | 1.0s | Recent trends |
| 6 | `bridge W4412993096 W4383957026` | sp=1, overlap=5 | 0.5s | AI design × PE |
| 7 | `mesh "prime editing"` | Not found | <1s | MeSH gap |
| 8 | `mesh D000072669 --tree` | Gene Editing tree | <1s | |

### Sequence Annotations
- Copy-paste: 3 times
- Unnecessary commands: none
- Performance: all commands ≤1s

## Five-Dimension Rating

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Discoverability | 5/5 | DOI → expand → shrink → bridge natural flow |
| Output clarity | 5/5 | shrink table columns (ex#, +cov, cum%) enable comparison |
| Workflow continuity | 4/5 | -1: no built-in compare mode for two shrink runs |
| Information sufficiency | 5/5 | coverage %, pool size, uncovered count all present |
| Command ergonomics | 4/5 | -1: custom venue list is long to type, no preset save |

**Total: 23/25**

## Friction Log

```
FRICTION-1: venue comparison requires manual side-by-side
COMMAND: shrink --top 7 (run 1), shrink --top 7 --venue custom (run 2)
OUTPUT:  two separate tables
PROBLEM: comparing two tables requires scrolling and memory
WANTED:  shrink --compare NCS+ "custom,venues,..." or diff-friendly output

FRICTION-2: custom venue list input is cumbersome
COMMAND: shrink --venue "Nature Biotechnology,Nature Methods,Nucleic Acids..."
OUTPUT:  works correctly
PROBLEM: 11 venues typed manually each time, typo risk
WANTED:  venue presets (--venue-preset protein_eng) or config file

FRICTION-3: "prime editing" not in MeSH
COMMAND: mesh "prime editing"
OUTPUT:  No MeSH descriptor matching
PROBLEM: MeSH vocabulary lag — prime editing (2019) still not a descriptor
WANTED:  fuzzy suggestion: "Did you mean: Gene Editing (D000072669)?"
```

## Additional Observations

**Key finding — venue comparison**:
- NCS+ and custom venues produce identical 86% coverage for this field
- 5/7 papers overlap (71%)
- This is a NCS+-concentrated field (prime editing + AI protein design)
- Contrasts sharply with S5 (RNA degrader): NCS+ 53% vs custom 84%
- **Conclusion**: custom venue only essential for field-specific journal
  ecosystems (chemical biology, imaging, etc.)

**Surprises (positive)**:
- shrink output columns (ex#, +cov, cum%) make comparison feasible
  even without built-in compare mode
- --no-foundation shows distinct recent-trend papers

**Surprises (negative)**:
- "prime editing" absent from MeSH despite being the defining technique
  of this field since 2019

## Research Findings

### Shrink Venue Comparison

| Metric | NCS+ | Custom (11 protein eng venues) |
|--------|------|-------------------------------|
| Coverage | 86% (173/200) | 86% (171/200) |
| Pool size | 99 | 75 |
| Papers overlapping | 5/7 (71%) | 5/7 (71%) |

Unique to NCS+:
- Harnessing AI + gene editing (Nat Rev, 2025) — field overview
- ABE at rank 7 vs rank 6

Unique to custom:
- One-shot protein design (Nature, 2025) — AI design emphasis

### Reading List (NCS+, 7 papers, 86% coverage)

| # | Year | Venue | Title | Role |
|---|------|-------|-------|------|
| 1 | 2019 | Nature | Search-and-replace genome editing | PE foundation |
| 2 | 2023 | Nature | RFdiffusion de novo design | AI protein design |
| 3 | 2013 | Science | Cas9 genome editing | CRISPR foundation |
| 4 | 2007 | Mol Cell | Human MLH1 structure | Structural basis |
| 5 | 2025 | Nat Rev | AI + gene editing review | Field overview |
| 6 | 2025 | Nature | Functional genome editors | AI × editing |
| 7 | 2017 | Nature | ABE (base editing) | Base editing origin |

### Field Structure

AI protein design and prime editing converge at "AI-generated
editing tool components" — this paper (MLH1 binder) is at that
intersection. The shared foundation includes AlphaFold, RFdiffusion,
ProteinMPNN on the AI side, and PE/PEmax/ABE on the editing side.
