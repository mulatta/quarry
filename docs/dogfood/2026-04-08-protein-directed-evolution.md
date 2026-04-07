---
session_id: dogfood-2026-04-08-protein-directed-evolution
date: 2026-04-08
persona: bridger
topic: "AI-guided protein directed evolution — 3-seed bridge"
seeds: [W4412086768, W4404577932, W7130605770]
commands_used: [info, bridge, expand, shrink, mesh]
commands_count: 14
friction_count: 3
copy_paste_count: 10
scores:
  discoverability: 5
  output_clarity: 4
  workflow_continuity: 5
  information_sufficiency: 4
  command_ergonomics: 4
  total: 22
previous_session: dogfood-2026-04-08-rna-degrader-landscape
score_delta: -2
improvements_tested:
  - "3-seed bridge (steiner scenario)"
  - "bridge --type filter"
  - "pairwise vs 3-seed comparison"
design_refs: []
---

# Dogfood: AI-Guided Protein Directed Evolution (3-seed bridge)

## Session Context

Bridger persona with 3 DOIs from the same subfield (ML-guided
protein directed evolution). Goal: find papers connecting these
three and understand the research structure.

Constraint: FTS/embedding unavailable (building).

Seeds:
- W4412086768 — Inverse folding + structural/evolutionary constraints (Cell 2025)
- W4404577932 — EVOLVEpro: PLM-based in silico directed evolution (Science 2024)
- W7130605770 — PLM + epistatic interactions for directed evolution (Science 2026)

This tests scenario B1 (3-seed steiner bridge) and partially B2
(close bridge — all three from the same subfield).

## Command Sequence Log

| # | Command | Result | Time | Notes |
|---|---------|--------|------|-------|
| 1 | `info "https://doi.org/10.1016/j.cell.2025.06.014" --mesh` | W4412086768 | <1s | DOI resolve OK |
| 2 | `info "https://doi.org/10.1126/science.adr6006" --mesh` | W4404577932 | <1s | |
| 3 | `info "https://doi.org/10.1126/science.aea1820" --mesh` | W7130605770 | <1s | No MeSH (2026) |
| 4 | `bridge W4412086768 W4404577932 W7130605770` | 5 types, no steiner | 1.7s | **steiner missing** |
| 5 | `bridge ... --type steiner` | Empty, overlap=0 | 0.6s | No error msg |
| 6 | `bridge ... --type path` | Empty, overlap=0 | 0.7s | sp=1, no path |
| 7 | `info W4400313991 W2118265845 W4296032638 W4389472984 W3179485843 --mesh` | Foundation | <1s | 5 key refs |
| 8 | `info W4400800287 W4391544240 W4408363707 --mesh` | Coupling bridges | <1s | |
| 9 | `expand W4404577932 --limit 30 --mesh-summary` | 30 papers + MeSH | 0.8s | Seeds visible |
| 10 | `shrink W4404577932 --top 7` | 82% coverage | 0.5s | NCS+ works well |
| 11 | `mesh D019020 --tree` | Hierarchy shown | <1s | |
| 12 | `mesh "protein language model"` | Not found | <1s | Too new for MeSH |
| 13 | `mesh "machine learning"` | 10 matches | <1s | |
| 14 | `bridge W4412086768 W7130605770` | 2-seed, overlap=11 | 0.4s | EVOLVEpro = common ref |

### Sequence Annotations
- Copy-paste: 10 times (work_id transfers between commands)
- Unnecessary commands: #5, #6 — investigating why steiner/path absent
- Backtracking: none
- Performance: all commands <2s

## Five-Dimension Rating

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Discoverability | 5/5 | DOI → info → bridge → expand → shrink seamless |
| Output clarity | 4/5 | -1: steiner/path missing with no explanation, --type changes overlap |
| Workflow continuity | 5/5 | bridger flow natural: info → bridge → expand(hub) → shrink |
| Information sufficiency | 4/5 | -1: empty bridge types give no explanation (why empty?) |
| Command ergonomics | 4/5 | -1: steiner requires manual --type probe to discover it's missing |

**Total: 22/25**

## Friction Log

```
FRICTION-1: steiner type not shown in 3-seed bridge output
COMMAND: quarry bridge W4412086768 W4404577932 W7130605770
OUTPUT:  5 types shown (common_refs, common_citers, coupling, cocitation, ppr)
PROBLEM: steiner is documented as "k>=3 only" but absent from output.
         No message explaining why steiner was skipped.
WANTED:  Either show steiner results, or print "Steiner: skipped (reason)"

FRICTION-2: --type filter changes overlap stats
COMMAND: quarry bridge ... --type steiner
OUTPUT:  "refs overlap=0 citers overlap=0" (vs overlap=21/8 in full run)
PROBLEM: overlap is a seed-pair property, should be constant regardless
         of bridge type filter
WANTED:  overlap stats always reflect the full seed-pair relationship

FRICTION-3: empty bridge type gives no explanation
COMMAND: quarry bridge ... --type path
OUTPUT:  header only, no results, no message
PROBLEM: user doesn't know if path failed, found nothing, or isn't
         applicable (sp=1 means direct connection, no stepping stones)
WANTED:  "Path: 0 results (seeds are directly connected, sp=1)"
```

## Additional Observations

**Surprises (positive)**:
- bridge with 3 same-field seeds still produces rich results (21 common_refs)
- EVOLVEpro appears as common_ref in pairwise bridge of the other two seeds —
  confirms it's the hub paper
- shrink 82% coverage with NCS+ — ML+protein evolution publishes in top venues
- 2026 paper (W7130605770) already connected via citation graph despite cited_by=1

**Surprises (negative)**:
- "protein language model" not in MeSH vocabulary — MeSH lags behind fast-moving
  ML subfields by years

**Dead ends**:
- steiner and path bridge types returned nothing — later confirmed as
  correct behavior: all pairs sp=1 (directly connected), so no intermediate
  nodes exist. Not a bug. The real issue was lack of diagnostic messaging,
  which was fixed post-session (empty type reason messages added).
- mesh "protein language model" — concept too new

## Research Findings

### 3-Seed Connection Structure

EVOLVEpro (Science 2024, cited=130) is the hub connecting all three:

```
                 ┌─ W4412086768 (Cell 2025)
                 │  + structural/evolutionary constraints
                 │
EVOLVEpro ───────┤  W4404577932 (Science 2024)
(hub, cited=130) │  PLM few-shot directed evolution
                 │
                 └─ W7130605770 (Science 2026)
                    + epistatic interactions
```

All three address: "How to use protein language models to accelerate
directed evolution with minimal experimental data (few-shot)."

### Shared Foundation (21 common_refs, key papers)

| Year | Paper | Cited | Role |
|------|-------|-------|------|
| 2009 | Exploring protein fitness landscapes | 1,183 | Theoretical foundation |
| 2021 | ESM-1v zero-shot mutation prediction | 641 | PLM for fitness |
| 2022 | ProteinMPNN | 1,625 | Inverse folding |
| 2023 | ESM-2 evolutionary-scale | 4,367 | PLM at scale |
| 2023 | ProteinGym benchmarks | 190 | Evaluation standard |
| 2024 | AlphaFold3 | 11,661 | Structure prediction |

### Reading List (shrink, 7 papers, 82% coverage)

| # | Year | Venue | Title | Role |
|---|------|-------|-------|------|
| 1 | 2023 | Science | ESM-2 evolutionary-scale prediction | PLM scaling |
| 2 | 2021 | Nat Methods | Low-N protein engineering | Few-shot principle |
| 3 | 2022 | Science | ProteinMPNN | Inverse folding |
| 4 | 2021 | Nat Biotech | Efficient CRISPR editing | Application |
| 5 | 2023 | Nat Biotech | Efficient antibody evolution | Application |
| 6 | 2025 | Science | ESM3 / 500M years simulation | Latest PLM |
| 7 | 2023 | Nature | RFdiffusion de novo design | De novo design |
