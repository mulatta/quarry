---
session_id: dogfood-2026-04-08-protein-eng-regression
date: 2026-04-08
persona: surveyor
topic: "Protein engineering — S9 regression test"
seeds: [W2767044445, W4296032638]
commands_used: [mesh, info, expand, shrink]
commands_count: 10
friction_count: 2
copy_paste_count: 3
scores:
  discoverability: 5
  output_clarity: 5
  workflow_continuity: 4
  information_sufficiency: 5
  command_ergonomics: 4
  total: 23
previous_session: dogfood-2026-04-08-protein-engineering
score_delta: 4
improvements_tested:
  - "mesh search for exact descriptor name"
  - "mesh-summary drill-down loop"
  - "NCS+ vs custom venue for enzyme engineering"
design_refs: []
---

# Dogfood: Protein Engineering — S9 Regression Test

## Session Context

Regression test of S9 session. Primary goals:
1. Verify mesh "protein engineering" → D015202 (S9 missed this)
2. Re-validate mesh-summary drill-down loop
3. Compare NCS+ vs custom venue for enzyme engineering subfield

Constraint: FTS/embedding unavailable.

## Command Sequence Log

| # | Command | Result | Time | Notes |
|---|---------|--------|------|-------|
| 1 | `mesh "protein engineering"` | D015202 returned | 0.5s | **S9 F-1 resolved** |
| 2 | `expand W2115734610 --mesh-summary` | fluorescent proteins | 10s | Too niche seed |
| 3 | `mesh D015202` | 9426 papers | <1s | |
| 4 | `info W2767044445 --mesh` | Arnold Nobel review | <1s | Better seed |
| 5 | `expand W2767044445 --limit 30 --mesh-summary` | enzyme eng axis | 1.6s | DE + Biocatalysis |
| 6 | `mesh "biocatalysis"` | D055162, 2833 papers | <1s | Drill-down OK |
| 7 | `mesh "machine learning"` | D000069550 | <1s | |
| 8 | `expand W4296032638 --mesh-summary` | ML design axis | — | Deep Learning 10/20 |
| 9 | `shrink W2767044445 --top 5` | 48% NCS+ | 1.4s | Low coverage |
| 10 | `shrink W2767044445 --top 5 --venue custom` | 55% custom | 1.3s | +7%p |

### Sequence Annotations
- Copy-paste: 3 times (S9: 8 — 63% reduction)
- Unnecessary commands: CMD 2 (wrong seed choice)
- Performance: expand <2s for review paper (S9 reported 5-9s)

## Five-Dimension Rating

| Dimension | Score | S9 | Justification |
|-----------|-------|-----|---------------|
| Discoverability | 5/5 | 4 | +1: mesh "protein engineering" finds D015202 |
| Output clarity | 5/5 | 5 | Same |
| Workflow continuity | 4/5 | 3 | +1: drill-down works but still manual seed selection |
| Information sufficiency | 5/5 | 4 | +1: pairwise sp + diagnostics available |
| Command ergonomics | 4/5 | 3 | +1: fewer copy-paste, faster expand |

**Total: 23/25** (S9: 19/25, **Δ: +4**)

## Friction Log

```
FRICTION-1: mesh top paper may not be field-representative
COMMAND: expand W2115734610 (top cited in D015202)
OUTPUT:  fluorescent protein engineering — too niche for field mapping
PROBLEM: most-cited paper ≠ most representative paper
WANTED:  mesh output to suggest "representative review" or let user
         specify --type review to filter

FRICTION-2: no venue suggestion for low-coverage shrink
COMMAND: shrink W2767044445 --top 5 (48% NCS+)
OUTPUT:  48% coverage, 103 uncovered
PROBLEM: user doesn't know which venues to add to improve coverage
WANTED:  shrink to suggest top venues from uncovered papers
```

## S9 Friction Resolution Status

| S9 Friction | Status | Evidence |
|-------------|--------|----------|
| F-1: mesh misses D015202 | **RESOLVED** | `mesh "protein engineering"` → D015202 |
| F-2: work_mesh column guessing | not retested | used --schema in S9 |
| F-3: SQL join duplicates | not retested | no SQL needed this session |
| F-4: expand slow (5-9s) | **IMPROVED** | 1.6s for same type of paper |
| F-5: no field mapping command | same | still manual loop |

## Research Findings

### Venue Coverage by Subfield

| Subfield | NCS+ coverage | Custom coverage | Δ |
|----------|--------------|-----------------|---|
| Prime editing + AI (S8) | 86% | 86% | 0 |
| RNA degrader (S5) | 53% | 84% | +31 |
| Enzyme engineering (this) | 48% | 55% | +7 |

**Pattern**: NCS+ coverage correlates with how "mainstream" the field is.
- High NCS+: CRISPR/gene editing, AI/ML protein design
- Low NCS+: chemical biology, enzyme engineering, RNA therapeutics
- Custom venue helps most for fields with strong specialty journals
