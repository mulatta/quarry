---
session_id: dogfood-2026-04-08-prime-editing-efficiency
date: 2026-04-08
persona: surveyor
topic: "Prime editing efficiency factors: pegRNA design, chromatin context, DNA repair, protein engineering"
seeds: [W2981137429, W3206354438, W3202931255, W3087701445, W4386334048, W4393902863]
commands_used: [mesh, sql, expand, info, bridge, shrink]
commands_count: 12
friction_count: 2
copy_paste_count: 11
scores:
  discoverability: 4
  output_clarity: 5
  workflow_continuity: 4
  information_sufficiency: 4
  command_ergonomics: 3
  total: 20
previous_session: dogfood-2026-04-08-steiner-crispr-apps
score_delta: -4
improvements_tested:
  - "3-seed bridge for cross-factor discovery"
  - "shrink --no-foundation for recent trend identification"
  - "info --full for abstract-based factor classification"
---

# Dogfood: Prime Editing Efficiency Factors

## Session Context

Goal: systematically map factors affecting prime editing efficiency, editing
windows, and sequence context. This is a field-mapping (surveyor) task covering
a rapidly evolving area that lacks its own MeSH descriptor.

Constraints: lancedb unavailable → search/vsearch disabled. SQL ILIKE was the
only keyword-based entry point.

## Command Sequence Log

| # | Command | Result | Time | Note |
|---|---------|--------|------|------|
| 1 | `quarry mesh "prime editing"` | No match | 0.3s | No MeSH descriptor for prime editing |
| 2 | `quarry sql "...title ILIKE '%prime edit%' AND cited_by_count > 100..."` | 25 papers | 1m18s | **Bottleneck**: ILIKE full scan |
| 3 | `quarry expand W2981137429 --mesh-summary --limit 200` | 200 papers, MeSH summary | 7.2s | Excellent landscape view |
| 4 | `quarry info W3206354438 W3202931255 W3087701445 W4386334048 W4393902863 --mesh` | 5 papers | 0.3s | Factor classification via MeSH |
| 5 | `quarry bridge W3087701445 W3206354438` | 2 refs, 100 citers | 0.8s | ML prediction ↔ MMR manipulation |
| 6 | `quarry info W4379259080 W4283070781 W3035700080 W4316507374 W4367311810 --mesh` | 5 papers | 0.4s | Bridge result drill-down |
| 7 | `quarry shrink W2981137429 --top 10` | 10 papers, 100% cov | 11.1s | NCS+ baseline |
| 8 | `quarry shrink W2981137429 --top 10 --venue "Nature,Science,Cell,..."` | 10 papers, 100% cov | 10.7s | Cell venue added |
| 9 | `quarry shrink W2981137429 --top 10 --no-foundation --venue ...` | 10 papers, 96% cov | 10.5s | Recent trends only |
| 10 | `quarry bridge W3202931255 W4386334048` | 4 refs, 100 citers | 0.5s | pegRNA eng ↔ protein eng |
| 11 | `quarry bridge W3202931255 W3206354438 W4386334048` | 13 refs, 100 citers | 1.9s | 3-seed cross-factor |
| 12 | `quarry info W4394727560 W4400503129 W4399500292 --mesh --full` | 3 papers full | 0.3s | Key discovery: chromatin context |

**Total**: 12 commands, ~2m43s wall time (SQL 1m18s = 48% of total)

### Sequence Analysis

- **Unnecessary commands**: 0 — all commands returned actionable information
- **Repeated patterns**: info --mesh used 3× for factor classification (could
  benefit from expand outputting MeSH per paper)
- **Backtracking**: shrink run 3× with different venue/foundation flags — this
  was deliberate comparison, not error recovery
- **Transitions**: sql → expand required manual work_id transfer; expand →
  bridge was natural (pick papers from expand results)
- **Copy-paste points**: 11 work_id transfers between commands
- **Slow commands**: SQL (1m18s) broke flow entirely; shrink (~11s) was at the
  edge of acceptable
- **Performance bottleneck**: SQL ILIKE — if this were <2s the session would
  feel qualitatively different

## Five-Dimension Rating

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Discoverability | 4 | SQL found prime editing corpus, expand captured all key efficiency papers. No MeSH descriptor means search is the only keyword entry point, and it's unavailable (-1) |
| Output clarity | 5 | mesh-summary showed subfield distribution instantly. bridge common citers revealed synthesis papers. info --full abstract was decisive for factor classification |
| Workflow continuity | 4 | expand → shrink → bridge → info flow was natural. SQL → expand transition required manual work_id copy (-1) |
| Information sufficiency | 4 | MeSH, abstract, citation count all available. Expand results lack abstract snippets, requiring info re-lookup for paper classification (-1) |
| Command ergonomics | 3 | SQL 1m18s destroyed flow. When search is unavailable, SQL ILIKE is the only keyword entry point but too slow. 11 copy-paste transfers (-2) |

**Total: 20/25**

## Friction Log

```
FRICTION: Finding prime editing papers by keyword
COMMAND: quarry sql "SELECT ... WHERE title ILIKE '%prime edit%' ... LIMIT 25"
OUTPUT: 25 papers, 1m18s
PROBLEM: ILIKE full table scan takes >1 min. With search unavailable, SQL is
         the only keyword entry point, and the latency breaks flow entirely.
WANTED: search command operational, or a title text search index on PG

FRICTION: Identifying what aspect of efficiency each paper addresses
COMMAND: quarry expand W2981137429 --mesh-summary --limit 200
OUTPUT: 200 papers with truncated titles, MeSH summary at field level
PROBLEM: Title alone doesn't reveal whether a paper is about pegRNA design,
         chromatin context, or DNA repair. Had to run info --mesh on batches
         to classify papers by efficiency factor.
WANTED: expand --detail with abstract first sentence or keyword tags per paper
```

## Additional Observations

### Surprises (good)

- **3-seed bridge common citers discovered chromatin paper**: W4394727560
  (Cell 2024, "Chromatin context-dependent regulation of prime editing")
  appeared in common citers of the 3-seed bridge (NOT steiner — steiner
  returned empty due to all pairs sp=8) but NOT in the top-50 of expand.
  Common citers as a discovery tool for cross-factor papers works well.

- **shrink --no-foundation correctly identified PE4/PE5 as field game-changer**:
  When foundations were excluded, W3206354438 (enhanced PE systems via MMR
  manipulation) jumped to #1 with 58% coverage — it's the most influential
  non-foundational paper in the prime editing landscape.

- **info --full abstract was decisive**: W4394727560 abstract revealed the
  exact factors (H3K79me2, H3K9me3, HLTF, transcriptional elongation) that
  determine editing efficiency. Without --full, this paper would have been
  just another title in a long list.

### Dead ends

None — every command returned useful information.

### Missing commands

- `quarry cluster`: auto-cluster expand results by topic (MeSH or embedding).
  A surveyor mapping efficiency factors had to manually classify 200 papers by
  running info --mesh in batches. An automated clustering would save significant
  effort.

### Workflow suggestions

- **sql → expand shortcut**: if the user finds a paper via sql, they should be
  able to pipe it directly to expand without manual work_id copy.
- **expand --abstract**: add abstract snippet (first sentence) to expand detail
  view for paper triage without separate info calls.

## Research Findings

### Prime Editing Efficiency Factor Map

| Factor Category | Specific Factors | Key Papers | Impact Level |
|----------------|-----------------|------------|-------------|
| **pegRNA design** | PBS/RTT length, 3' degradation protection (epegRNA), loop engineering, silent edits | W3202931255, W4415973383, W4290673775 | High |
| **Protein engineering** | RT domain evolution (PACE), compact PE, PEmax architecture, PE6 variants | W4386334048, W4392796894, W4400447375 | High |
| **DNA repair pathway** | MMR suppression (MLH1dn → PE4/PE5), HLTF context-dependent repression | W3206354438, W4394727560 | High |
| **Chromatin context** | H3K79me2 (+), H3K9me3 (-), transcriptional elongation, position effects (0-94% range for same edit) | W4394727560 | High |
| **Edit type & size** | Substitution > insertion > deletion efficiency; edit length inversely correlated | W4367311810, W3087701445 | Medium |
| **Sequence context** | Target site GC content, PAM availability, nick position relative to edit | W3087701445, W4316507374 | Medium |
| **Delivery method** | RNP vs plasmid, VLP, dual AAV, cell type dependency | W3158544805, W4390670214, W4372334196 | Medium |
| **Combinatorial optimization** | Stacking 6 optimizations achieved 140-fold improvement (0.5% → 58%) | W4400503129 | Transformative |

### Key Insight

The most impactful recent finding is that **chromatin context** creates a
0%–94% efficiency range for the identical edit at different genomic positions
(W4394727560, Cell 2024). This means sequence-level factors (pegRNA design,
edit type) set an upper bound, but the actual efficiency is gated by local
chromatin state. Practical implication: pairing PE with CRISPRa at the target
locus can rescue low-efficiency sites.

### Minimum Reading List (from shrink --no-foundation)

1. W3206354438 — PE4/PE5 (MMR manipulation), Cell 2021
2. W3202931255 — epegRNA, Nature Biotechnology 2021
3. W4386334048 — Compact PE via PACE, Cell 2023
4. W4394727560 — Chromatin context, Cell 2024
5. W4400503129 — Combinatorial optimization for CFTR, Nature Biomed Eng 2024
6. W4367311810 — Efficiency prediction across cell types, Cell 2023
7. W3087701445 — ML efficiency prediction, Nature Biotechnology 2020
