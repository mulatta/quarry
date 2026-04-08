---
session_id: dogfood-2026-04-08-steiner-crispr-apps
date: 2026-04-08
persona: bridger
topic: "CRISPR 3-app steiner bridge — diagnostics × therapy × ecology"
seeds: [W2973445868, W2190054909, W3119813698]
commands_used: [info, bridge, expand, shrink]
commands_count: 6
friction_count: 1
copy_paste_count: 5
scores:
  discoverability: 5
  output_clarity: 5
  workflow_continuity: 5
  information_sufficiency: 5
  command_ergonomics: 4
  total: 24
previous_session: dogfood-2026-04-08-far-field-bridge
score_delta: 5
improvements_tested:
  - "steiner with sp≥2 seeds (first non-empty steiner result)"
  - "pairwise sp matrix for seed compatibility assessment"
  - "steiner pre-check workflow from cli-reference guide"
design_refs: []
---

# Dogfood: CRISPR 3-App Steiner Bridge (B1 Revalidation)

## Session Context

B1 scenario revalidation with sp≥2 seeds. S6 used sp=1 seeds (all
directly connected) → empty steiner. This session uses seeds from
3 distinct CRISPR application domains.

Seeds:
- W2973445868 — SHERLOCK nucleic acid detection (diagnostics, 2019)
- W2190054909 — CRISPR gene drive in mosquitoes (ecology, 2015)
- W3119813698 — In vivo base editing for progeria (therapy, 2021)

Pairwise sp: 3, 8, ∞ — ideal steiner range.

## Command Sequence Log

| # | Command | Result | Time | Notes |
|---|---------|--------|------|-------|
| 1 | `info W... W... W... --mesh` | 3 seeds, minimal MeSH overlap | <1s | Pre-check: distinct subfields |
| 2 | `bridge W... W... W...` | sp=3/8/∞, steiner=4 papers | 2.4s | **First non-empty steiner!** |
| 3 | `info` (4 steiner hubs) | CRISPR reviews + diagnostics | <1s | |
| 4 | `expand W2803581084 --mesh-summary` | CRISPR landscape | 2.7s | Hub covers 3 domains |
| 5 | `bridge` (3 pairwise, --type coupling) | 3/4 steiner hubs NOT in pairwise | ~1s | **Steiner-unique papers** |
| 6 | `shrink W2803581084 --top 5` | 56% coverage | 2.5s | shrink ≠ steiner |

### Sequence Annotations
- Copy-paste: 5 times
- Pre-check workflow (info → sp assessment → steiner): effective
- Performance: all commands <3s

## Five-Dimension Rating

| Dimension | Score | S6 | Justification |
|-----------|-------|-----|---------------|
| Discoverability | 5/5 | 5 | Same |
| Output clarity | 5/5 | 4 | +1: actionable suggestions available (not needed this time) |
| Workflow continuity | 5/5 | 5 | Pre-check guide from cli-reference worked perfectly |
| Information sufficiency | 5/5 | 4 | +1: steiner returned 4 junction papers with real value |
| Command ergonomics | 4/5 | 4 | -1: manual grep to verify steiner uniqueness vs pairwise |

**Total: 24/25** (S6: 22/25, **Δ: +2**)

## Friction Log

```
FRICTION-1: steiner uniqueness verification is manual
COMMAND: 3× pairwise bridge + grep for steiner work_ids
OUTPUT:  3/4 steiner hubs not in any pairwise coupling top-100
PROBLEM: no automatic "unique to steiner" tagging
WANTED:  steiner output to indicate which papers are pairwise-unique
```

## Key Findings

### Steiner Validates — First Non-Empty Result

| Paper | Year | Cited | Role | In pairwise? |
|-------|------|-------|------|-------------|
| Applications of CRISPR (Nat Biotech) | 2016 | 1,000 | Application overview | **No** |
| Cas9 Structures & Mechanisms (Annu Rev) | 2017 | 2,009 | Structural foundation | **No** |
| CRISPR tool kit (Nat Commun) | 2018 | 1,631 | Tool landscape | Yes (1 pair) |
| CRISPR graphene FET detection (Nat BME) | 2019 | 626 | Diagnostics bridge | **No** |

**3/4 steiner papers are not in any pairwise bridge** — steiner
provides genuinely unique connectivity information.

### Steiner vs Shrink vs Pairwise

| Method | What it finds | Overlap with steiner |
|--------|--------------|---------------------|
| Steiner(A,B,C) | 4 junction papers | — |
| Pairwise union | ~300 coupling bridges | 1/4 overlap |
| Shrink(hub) | 5 coverage papers | 0/4 overlap |

The three methods answer fundamentally different questions:
- **Steiner**: "What connects all three?" → junction backbone
- **Pairwise**: "What connects each pair?" → bilateral bridges
- **Shrink**: "What covers this field?" → reading list

### Review Backbone Recommendation

For a CRISPR applications review covering diagnostics + therapy +
ecology, the steiner backbone (4 papers) provides the narrative
structure, and each seed's expand provides the domain-specific content.

```
                    Steiner backbone
                    ┌─────────────┐
                    │ Cas9 struct  │
                    │ CRISPR apps  │
                    │ CRISPR kit   │
                    │ graphene FET │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
    SHERLOCK (dx)    gene drive (eco)   in vivo BE (tx)
    expand → 200      expand → 200      expand → 200
