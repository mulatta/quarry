---
session_id: dogfood-2026-04-08-f1-f2-regression
date: 2026-04-08
persona: reader
topic: "Regression test: F1 bridge sp fix + F2 expand --min-citations"
seeds: [W2045435533, W2336828812, W2766599608, W2981137429, W2973445868, W2190054909, W3119813698]
commands_used: [bridge, expand, shrink]
commands_count: 11
friction_count: 0
copy_paste_count: 0
scores:
  discoverability: 4
  output_clarity: 5
  workflow_continuity: 3
  information_sufficiency: 4
  command_ergonomics: 4
  total: 20
previous_session: dogfood-2026-04-08-cas9-pe-lineage
score_delta: +6
improvements_tested:
  - "F1: bridge sp bidirectional (min of both directions)"
  - "F2: expand --min-citations post-filter"
---

# Dogfood Regression: F1 sp Fix + F2 --min-citations

## Session Context

Regression test after two changes:
- **F1**: bridge sp computed as min(sp(i→j), sp(j→i)) instead of sp(i→j) only
- **F2**: expand --min-citations N post-filter for high-cited seed papers

Re-runs S14 (Cas9→PE lineage) commands plus spot-checks from S6, S11, S12, S13.

## Command Sequence Log

| # | Command | Result | Time | Regression Check |
|---|---------|--------|------|-----------------|
| 1 | `bridge W2336828812 W2766599608` | sp=1 | 1.6s | **F1 RESOLVED**: was sp=∞ in S14 |
| 2 | `bridge W2766599608 W2336828812` | sp=1 | 1.5s | No regression (was sp=1) |
| 3 | `bridge W2045435533 W2336828812 --type path` | sp=1 | 1.4s | **F1 RESOLVED**: was sp=∞ in S14 |
| 4 | `bridge W2973445868 W2190054909 W3119813698` | sp=3/8/3 | 0.9s | S12: ∞→3 (improved) |
| 5 | `bridge W2973445868 W2190054909` | sp=3 | 1.5s | S11: no regression |
| 6 | `bridge W2766599608 W2336828812 W2981137429` | sp=1/1/1 | 0.9s | S6: no regression |
| 7 | `expand W2045435533 --limit 10` | noise (same as S14) | 7.1s | Baseline: no regression |
| 8 | `expand W2045435533 --min-citations 100 --mesh-summary --limit 200` | 188 results | 7.5s | **F2 RESOLVED**: CBE #30, ABE #57, PE #63 |
| 9 | `expand W2981137429 --limit 5` | identical to S13 | 3.6s | S13: no regression |
| 10 | `expand W2981137429 --min-citations 100 --limit 200` | 175/200 | 4.1s | No harmful filtering |
| 11 | `shrink W2981137429 --top 5` | identical to S13 | 3.8s | S13 shrink: no regression |

**Total**: 11 commands, ~34s

## Five-Dimension Rating

| Dimension | Score | vs S14 | Justification |
|-----------|-------|--------|---------------|
| Discoverability | 4 | +2 | --min-citations enables milestone discovery from classic seeds |
| Output clarity | 5 | +1 | sp values now correct; "seeds directly connected" message helpful |
| Workflow continuity | 3 | +1 | Seed order no longer matters for bridge sp |
| Information sufficiency | 4 | +1 | Correct sp conveys citation relationship immediately |
| Command ergonomics | 4 | +1 | No seed-order anxiety; --min-citations 0 default preserves existing UX |

**Total: 20/25** (S14: 14/25, **Δ = +6**)

## S14 Friction Resolution

| S14 Friction | Status | Evidence |
|-------------|--------|----------|
| F-1: Classic expand noise | **MITIGATED** | --min-citations 100 surfaces CBE/ABE/PE. Default unchanged |
| F-2: Bridge path off-topic | **RESOLVED** | Cas9↔CBE now sp=1 (bidirectional picks ABE→CBE direction) |
| F-3: sp unidirectional bug | **RESOLVED** | CBE↔ABE sp=1 regardless of seed order |
| F-4: Lateral same-domain | unchanged | Deferred (needs MeSH-based cross-domain detection) |

## Cross-Session Full Regression (All Sessions with bridge sp≥7 or ∞)

| Session | Bridge Pair | Before | After | Status |
|---------|------------|--------|-------|--------|
| S3 | smFISH↔MERFISH | sp=7 | **sp=1** | ✅ MERFISH cites smFISH directly |
| S3 | MERFISH↔10x Visium | sp=4 | **sp=1** | ✅ direct citation in reverse |
| S4 | PTM↔CRISPR | sp=8 | sp=8 | ✅ genuinely distant |
| S6 | ABE+CBE+PE | sp=1/1/1 | sp=1/1/1 | ✅ no change |
| S9 | protein eng seeds | sp=8 | **sp=3** | ✅ reverse direction shorter |
| S11 | QKD↔DNA computing | sp=7 | sp=7 | ✅ genuinely distant |
| S12 | steiner 3-seed | sp=3/8/∞ | sp=3/8/**3** | ✅ ∞→3 |
| S13 | ML pred↔MMR | sp=8 | **sp=2** | ✅ reverse direction shorter |
| S13 | epegRNA↔compact PE | sp=8 | **sp=1** | ✅ direct citation |
| S13 | epegRNA↔PE4/PE5 | sp=8 | **sp=1** | ✅ direct citation |
| S14 | CBE↔ABE | sp=∞ | **sp=1** | ✅ bug fix |
| S14 | Cas9↔CBE | sp=∞ | **sp=1** | ✅ bug fix |

**12 pairs checked. 9 improved (including 2 ∞→1 bug fixes), 3 unchanged
(genuinely distant fields). Zero regressions.**

Improvements verified via `Graph.shortest_path()` API — reverse direction
has shorter path due to newer paper citing older paper (natural citation
direction). Previous single-direction computation missed these.
