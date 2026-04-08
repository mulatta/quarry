---
session_id: dogfood-2026-04-08-cas9-pe-lineage
date: 2026-04-08
persona: reader/surveyor
topic: "Cas9→CBE→ABE→PE technology lineage tracing from classic 2012 paper"
seeds: [W2045435533, W2336828812, W2766599608, W2981137429]
commands_used: [mesh, sql, info, expand, shrink, bridge]
commands_count: 14
friction_count: 4
copy_paste_count: 8
scores:
  discoverability: 2
  output_clarity: 4
  workflow_continuity: 2
  information_sufficiency: 3
  command_ergonomics: 3
  total: 14
previous_session: dogfood-2026-04-08-prime-editing-efficiency
score_delta: -6
improvements_tested:
  - "classic paper as expand seed"
  - "bridge path for technology lineage"
  - "close bridge same-lab papers"
  - "lateral → bridge transition"
---

# Dogfood: Cas9→PE Technology Lineage

## Session Context

Chain question covering 4 uncovered scenarios:
- **R4**: Old classic paper (Jinek 2012 Cas9) as expand seed
- **S3**: Chronological lineage tracing via foundation chain + bridge
- **R2**: Expand lateral discovery → bridge transition
- **B2**: Close bridge between same-field papers (CBE ↔ ABE)

Goal: trace the Cas9(2012) → CBE(2016) → ABE(2017) → PE(2019) technology
lineage using quarry's graph tools.

## Command Sequence Log

| # | Command | Result | Time | Note |
|---|---------|--------|------|------|
| 1 | `quarry mesh "CRISPR-Associated Protein 9"` | 759 papers, no Jinek 2012 in top | 0.5s | MeSH descriptor post-dates classic |
| 2 | `quarry sql "...ILIKE '%programmable%endonuclease%' AND pub_year=2012..."` | W2045435533 found | 43.7s | Slow but works |
| 3 | `quarry info W2045435533 --mesh` | Jinek 2012, 16935 cites | 0.4s | No "Gene Editing" MeSH — era-appropriate |
| 4 | `quarry expand W2045435533 --mesh-summary --limit 200` | 200 results | 7.6s | **Critical finding**: CBE/ABE/PE absent from top-200 |
| 5 | `quarry shrink W2045435533 --top 10` | 10 papers, 56% coverage | 7.1s | Pool=14 only, milestones missing |
| 6 | `quarry info W2336828812 W2766599608 W2981137429 --mesh` | 3 milestones | 0.3s | Manual seed selection required |
| 7 | `quarry bridge W2045435533 W2336828812 --type path` | sp=∞, 4 off-topic paths | 0.9s | **Failure**: no tech lineage |
| 8 | `quarry bridge W2045435533 W2336828812 --type common_refs --type coupling` | 0 common refs, 100 coupling (reviews) | 0.5s | Coupling = review papers only |
| 9 | `quarry bridge W2336828812 W2766599608` | sp=∞, 2 common refs, 2856 citers | 1.2s | **Bug found**: sp should be 1 |
| 10 | `quarry sql --schema work_citations` | column names | 0.3s | Needed for manual verification |
| 11 | `quarry sql "...work_citations WHERE citing_id=ABE AND cited_id=CBE"` | exists=1 | 0.3s | ABE cites CBE — sp should be 1 |
| 12 | `quarry expand W2336828812 --limit 50` (grep lateral) | 0 lateral | 3.3s | No lateral in CBE expand |
| 13 | `quarry expand W2981137429 --limit 200` (grep lateral) | 3 lateral | 3.7s | PE has laterals but same-domain |
| 14 | `quarry bridge W2981137429 W3012327875 --type common_citers --type coupling` | 395 citers | 0.5s | Lateral→bridge works but low differentiation |

**Total**: 14 commands, ~1m10s (excl SQL 43.7s)

### Sequence Analysis

- **Unnecessary commands**: CMD 10-11 — manual SQL to verify bridge sp bug.
  These shouldn't be needed if sp were correct.
- **Repeated patterns**: info --mesh used for milestone identification after
  expand failed to surface them
- **Backtracking**: CMD 7-8 — retried bridge with different types after path
  failed (expected, but type selection guidance would help)
- **Copy-paste points**: 8 work_id transfers
- **Performance bottleneck**: SQL ILIKE (43.7s); expand on classic paper (7.6s)

## Five-Dimension Rating

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Discoverability | 2 | Classic paper expand is dominated by noise — CBE/ABE/PE absent from 200 results. Technology milestones require external knowledge to identify (-3) |
| Output clarity | 4 | expand output, bridge types, and info are clear. MeSH summary correctly shows field composition. sp display has a bug but format is good (-1 for bug) |
| Workflow continuity | 2 | Lineage tracing is impossible with current tools — no foundation chain, bridge path is off-topic, manual seed identification required at every step (-3) |
| Information sufficiency | 3 | info --mesh shows era-appropriate MeSH evolution (interesting!). But expand doesn't surface the information needed for lineage tracing (-2) |
| Command ergonomics | 3 | Same SQL slowness and copy-paste issues as S13. Bridge --type flag for filtering is useful (+1 vs not having it) |

**Total: 14/25**

## Friction Log

```
FRICTION: Classic paper expand is noise-dominated
COMMAND: quarry expand W2045435533 --mesh-summary --limit 200
OUTPUT: 200 papers, none are CBE/ABE/PE milestones
PROBLEM: With 16,935 citations, APPR spreads too thin. The top results are
         low-citation niche papers (Revista Digital, 1 cite) rather than field
         milestones. coupling and cocitation signals are diluted by the massive
         citation fan-out.
WANTED: A "milestone" mode that prioritizes high-impact follow-ups over APPR
         score. Or at minimum, a --min-citations filter for expand.

FRICTION: Bridge path fails for technology lineage
COMMAND: quarry bridge W2045435533 W2336828812 --type path
OUTPUT: sp=∞, 4 off-topic path bridges (microscopy, Streptococcus)
PROBLEM: Directed graph path Cas9→CBE doesn't exist within depth=5.
         The technology lineage exists (Cas9 enables CBE) but isn't captured
         by direct citation chains. Conceptual lineage ≠ citation lineage.
WANTED: A "lineage" bridge type that uses semantic similarity or MeSH overlap
         to trace technology evolution, not just citation paths.

FRICTION: Bridge sp calculation is single-direction
COMMAND: quarry bridge W2336828812 W2766599608
OUTPUT: sp=∞ (CBE→ABE direction)
PROBLEM: ABE→CBE is sp=1 (ABE directly cites CBE), verified via SQL.
         But bridge reports sp=∞ because it only checks seed1→seed2 direction.
         Bidirectional BFS starts from src but max_depth=5 < actual path
         length in the reverse direction.
WANTED: sp = min(sp(A→B), sp(B→A)). For undirected interpretation, the
         shorter direction is what matters for assessing seed proximity.
BUG: shortest_path_length in bridge.rs computes src→dst only. Should take
     min of both directions, or at least report both.

FRICTION: Lateral papers are same-domain, not cross-domain
COMMAND: quarry expand W2981137429 --limit 200 | grep lateral
OUTPUT: 3 laterals, all gene-editing adjacent (PACE ABE, PAM-modified Cas9)
PROBLEM: For a highly self-contained field like genome editing, "lateral"
         (bibliographic coupling without co-citation) still stays within the
         same domain. The lateral tag doesn't guarantee cross-domain discovery.
WANTED: Clearer lateral semantics or a "cross-domain" flag based on MeSH
         divergence between seed and lateral paper.
```

## Additional Observations

### Bug: Bridge sp is unidirectional

Confirmed via direct API test:
```
sp ABE→CBE: [ABE, CBE]         # length=1 (direct citation)
sp CBE→ABE: [CBE, ..., ABE]    # length=6
```

Bridge reports `sp: W2336828812↔W2766599608=∞` — this is the CBE→ABE direction
which exceeds max_depth=5. The ABE→CBE direction (sp=1) is not checked.

**Root cause**: `bridge.rs:235` calls `shortest_path_length(graph, seed_indices[i],
seed_indices[j], ...)` with fixed i→j order. The function does bidirectional BFS
but only from src→dst direction (fwd BFS from src, rev BFS from dst).

**Fix**: compute `min(sp(i,j), sp(j,i))` or report both directions.

### Classic paper expand failure mode

With 16,935 citations, Cas9 paper's APPR walk spreads over an enormous subgraph.
Coupling/cocitation signals are diluted because the seed shares references with
thousands of papers across all of biology. The result is dominated by niche
papers that happen to have high APPR proximity but low scientific relevance.

This is a fundamental limitation of citation-graph-only expansion for highly
cited classic papers. Possible mitigations:
- **Citation threshold**: filter expand candidates by min cited_by_count
- **Time-decay**: weight recent high-impact papers higher
- **MeSH-constrained expansion**: restrict to papers sharing major MeSH with seed

### Technology lineage ≠ citation lineage

The Cas9→CBE→ABE→PE lineage is clear to any domain expert, but it doesn't
manifest as a clean citation chain because:
1. Cas9 paper (2012) is cited by ~17,000 papers — CBE is just one of them
2. CBE doesn't share many references with Cas9 (different conceptual framing)
3. Bridge path relies on citation direction, but conceptual dependency is
   asymmetric (CBE depends on Cas9, but cites many non-Cas9 papers too)

A "technology lineage" tool would need semantic understanding beyond citation
graphs — perhaps combining MeSH hierarchy, temporal ordering, and keyword
evolution (e.g., "endonuclease" → "base editing" → "prime editing").

### R2: Lateral → bridge transition

The transition from expand lateral to bridge is mechanically possible (pick
lateral work_id, run bridge with seed + lateral), but the lateral papers in
gene editing don't provide genuine cross-domain insights. The field is too
self-contained for lateral to surface truly surprising connections.

### B2: Close bridge value

CBE↔ABE bridge produced:
- 2856 common citers (overwhelming, mostly reviews/applications)
- Only 2 common refs (not the expected shared conceptual foundations)
- sp=∞ (bug — should be 1)

For same-lab/same-field papers, bridge produces quantity over quality.
Common citers are too numerous to be informative. A "close bridge" might
benefit from stricter AA thresholds or automatic deduplication of review papers.

## Research Findings

### Genome Editing Technology Lineage (manually reconstructed)

| Year | Milestone | Work ID | Citations | Key Innovation |
|------|-----------|---------|-----------|---------------|
| 2012 | Cas9 | W2045435533 | 16,935 | RNA-programmable DNA endonuclease |
| 2013 | Multiplex CRISPR | W2064815984 | 15,534 | Genome engineering in human cells |
| 2016 | CBE (BE3) | W2336828812 | 5,270 | C→T editing without DSB |
| 2017 | ABE (ABE7.10) | W2766599608 | 4,067 | A→G editing without DSB |
| 2019 | PE | W2981137429 | 4,449 | Search-and-replace without DSB or donor |

This lineage was **not discoverable** through quarry's graph tools alone.
It required prior domain knowledge to identify the milestone papers.
