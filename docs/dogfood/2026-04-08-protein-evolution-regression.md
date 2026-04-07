---
session_id: dogfood-2026-04-08-protein-evolution-regression
date: 2026-04-08
persona: bridger
topic: "protein directed evolution — S6 regression test"
seeds: [W4412086768, W4404577932, W7130605770]
commands_used: [info, bridge, expand, shrink]
commands_count: 6
friction_count: 0
copy_paste_count: 4
scores:
  discoverability: 5
  output_clarity: 5
  workflow_continuity: 5
  information_sufficiency: 5
  command_ergonomics: 5
  total: 25
previous_session: dogfood-2026-04-08-protein-directed-evolution
score_delta: 3
improvements_tested:
  - "pairwise sp matrix in bridge header"
  - "empty bridge type diagnostic messages"
  - "overlap hidden when common_refs not computed"
  - "steiner unreachable pair count"
design_refs: []
---

# Dogfood: Protein Directed Evolution — S6 Regression Test

## Session Context

Regression test of S6 session using identical seeds. Purpose: verify
that the 3 friction points from S6 (steiner no explanation, overlap
miscount, empty type no reason) are resolved.

Seeds (same as S6):
- W4412086768 — Inverse folding + evolutionary constraints (Cell 2025)
- W4404577932 — EVOLVEpro (Science 2024)
- W7130605770 — PLM + epistatic interactions (Science 2026)

## Command Sequence Log

| # | Command | Result | Time | Notes |
|---|---------|--------|------|-------|
| 1 | `info DOI1 DOI2 DOI3` | 3 papers resolved | <1s | Batch info (S6: 3 separate calls) |
| 2 | `bridge W4412086768 W4404577932 W7130605770` | 5 types + sp matrix | 1.5s | **S6 F-1,F-2,F-3 resolved** |
| 3 | `bridge ... --type steiner` | sp matrix, reason msg | 0.7s | **No overlap=0 miscount** |
| 4 | `expand W4404577932 --limit 15 --mesh-summary` | 15 papers + MeSH | 0.5s | |
| 5 | `shrink W4404577932 --top 5` | 78% coverage | 0.5s | |
| 6 | `bridge W4412086768 W4404577932` | 2-seed, sp matrix | 0.4s | Consistent format |

### Sequence Annotations
- Copy-paste: 4 times (S6: 10 times — 60% reduction from batch info)
- Unnecessary commands: none
- Performance: all commands <2s

## Five-Dimension Rating

| Dimension | Score | S6 | Justification |
|-----------|-------|-----|---------------|
| Discoverability | 5/5 | 5 | Same |
| Output clarity | 5/5 | 4 | +1: pairwise sp matrix shows which pairs are connected/unreachable |
| Workflow continuity | 5/5 | 5 | Same |
| Information sufficiency | 5/5 | 4 | +1: "2/3 pairs unreachable" explains why steiner is empty |
| Command ergonomics | 5/5 | 4 | +1: no overlap miscount, clean --type filter output |

**Total: 25/25** (S6: 22/25, **Δ: +3**)

## Friction Log

No friction points. All 3 issues from S6 resolved:

| S6 Friction | Resolution | Verified |
|-------------|-----------|----------|
| F-1: steiner absent, no explanation | "no steiner tree (2/3 pairs unreachable)" | ✓ |
| F-2: --type filter shows overlap=0 | overlap hidden when common_refs not computed | ✓ |
| F-3: empty type gives no reason | diagnostic message for path/steiner | ✓ |

## Additional Observations

**New feature highlights**:
- Pairwise sp matrix in header: `sp: A↔B=1 A↔C=∞ B↔C=∞`
  - Immediately reveals which seed pairs are directionally connected
  - ∞ symbol is intuitive for "unreachable in directed graph"
  - Works for both 2-seed and 3-seed bridges consistently
- Steiner diagnostic with unreachable count is actionable:
  users/LLMs can decide "these seeds are too close/disconnected for bridge"

**Regression confirmed**: all S6 improvements working as designed.
No new friction introduced.
