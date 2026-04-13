---
session_id: dogfood-2026-04-08-reverse-central-dogma
date: 2026-04-08
persona: surveyor/bridger
topic: "Protein→RNA reverse central dogma: feasibility via biological mechanisms"
seeds: [W1985387528, W2213968760, W2018756788, W2124254997, W2008954911]
commands_used: [mesh, info, expand, bridge, shrink, sql]
commands_count: 24
friction_count: 2
copy_paste_count: 6
scores:
  discoverability: 4
  output_clarity: 5
  workflow_continuity: 4
  information_sufficiency: 4
  command_ergonomics: 3
  total: 20
previous_session: dogfood-2026-04-08-f1-f2-regression
score_delta: 0
improvements_tested:
  - "bridge sp bidirectional (used throughout, all sp values correct)"
  - "creative use of MeSH cross-classification for serendipity"
---

# Dogfood: Protein → RNA (Reverse Central Dogma)

## Session Context

Hypothetical research question: can the "impossible" direction of the central
dogma (Protein → RNA) be implemented via biological reactions? Not chemical/
physical methods, but enzymatic pathways.

Approach: naive decomposition of the problem into sub-systems, then literature
search for the closest biological precedents for each sub-system, followed by
bridge-based serendipity tracking.

Constraint: lancedb unavailable → no search/vsearch.

## Problem Decomposition

### Naive Decomposition

Protein → RNA requires:
1. **Protein Reading Head**: sequentially scan amino acid sequence (N→C)
2. **Reverse Genetic Code Adapter**: map each amino acid → codon (with degeneracy)
3. **RNA Assembler**: polymerize codons into RNA strand

### Key Insight from Literature

Sub-system 2 is the fundamental barrier. No enzyme catalyzes aa→codon conversion
in nature. However, **Rqc2p/NEMF** (RQC complex) represents a partial exception:
a protein that determines tRNA selection independently of mRNA.

## Command Sequence Log

| # | Command | Result | Time | Note |
|---|---------|--------|------|------|
| 1 | `mesh "central dogma"` | No match | 0.3s | Concept absent from MeSH |
| 2 | `mesh "reverse transcription"` | D048348 | 0.4s | RNA→DNA, not protein→RNA |
| 3 | `mesh "protein biosynthesis"` | 5 matches | 0.3s | Forward direction only |
| 4 | `mesh "aminoacyl-tRNA synthetase"` | D000604, 3100 papers | 0.7s | aa recognition system |
| 5 | `mesh "RNA-directed DNA polymerase"` | D012194 | 0.8s | RT landscape |
| 6 | `info W2018756788 --mesh --full` | aaRS/genetic code review | 0.3s | |
| 7 | `mesh "prion"` | D011328 | 0.3s | Protein→protein template (conformation only) |
| 8-9 | `mesh "ribosome"`, `mesh "proteasome"` | Multiple | 0.6s | |
| 10 | `mesh D049071` (Endopeptidase Clp) | 553 papers | 0.4s | ClpXP found |
| 11 | `expand W2213968760 --mesh-summary --limit 30` | ClpXP landscape | 1.7s | AAA+ ATPase |
| 12 | `mesh D046908` (NRPS) | 131 papers | 0.4s | **Serendipity: Rqc2p found here!** |
| 13 | `info W1985387528 --mesh --full` | Rqc2p CAT-tail | 0.3s | **Key paper** |
| 14 | `expand W1985387528 --mesh-summary --limit 100` | RQC landscape | 1.5s | |
| 15 | `bridge W2213968760 W2018756788` | ClpXP↔aaRS, sp=4 | 1.8s | QC bridge |
| 16 | `bridge W1985387528 W2124254997` | RQC↔NRPS, sp=4 | 1.8s | Modular assembly |
| 17 | `bridge W1985387528 W2008954911` | RQC↔RT, sp=3 | 1.6s | **De novo priming** |
| 18 | `info W4220814074 W3124268703 W4379600803 --full` | 3 papers | 0.3s | |
| 19 | `shrink W1985387528 --top 5` | RQC essentials | 1.3s | |
| 20 | `sql "...ILIKE '%reverse translat%'..."` | Clinical only | 1m27s | **Bottleneck** |
| 21 | `sql "...ILIKE '%nanopore%protein%sequenc%'..."` | 9 papers | 1m27s | Physical methods |
| 22 | `mesh "polynucleotide adenylyltransferase"` | 466 papers | 0.4s | Template-free RNA synthesis |
| 23 | `mesh "directed molecular evolution"` | 1956 papers | 0.5s | |
| 24 | `mesh "nonribosomal peptide"` | (same as CMD 12) | 0.4s | |

**Total**: 24 commands, ~3m05s (SQL 2m54s = 94% of total)

## Five-Dimension Rating

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Discoverability | 4 | MeSH hierarchy + bridge serendipity discovered Rqc2p as key precedent. "Reverse translation" concept absent from MeSH requires workarounds (-1) |
| Output clarity | 5 | info --full abstract immediately revealed "protein determines tRNA recruitment" — the core insight. mesh-summary effective for field structure |
| Workflow continuity | 4 | mesh → info → expand → bridge → shrink chain natural for each sub-problem. Repeating this for 3 sub-systems was systematic but manual (-1) |
| Information sufficiency | 4 | Abstract + MeSH sufficient for relevance judgment. NRPS→Rqc2p serendipity came from shared MeSH classification (-1 for luck dependency) |
| Command ergonomics | 3 | SQL 2×1m27s destroyed flow. Other commands fast. 6 copy-paste transfers (-2) |

**Total: 20/25**

## Friction Log

```
FRICTION: Conceptual keyword search unavailable
COMMAND: quarry mesh "central dogma", "reverse translation"
OUTPUT: No match / clinical medicine only
PROBLEM: Core molecular biology concepts absent from MeSH descriptors.
         search command (FTS) would solve this but unavailable.
WANTED: FTS operational for title/abstract concept search

FRICTION: SQL ILIKE repeated bottleneck
COMMAND: quarry sql "...ILIKE '%reverse translat%'..." (1m27s × 2)
OUTPUT: Clinical papers only, no molecular biology results
PROBLEM: 94% of session time spent on SQL. ILIKE is wrong tool for concept search.
WANTED: FTS would eliminate this entirely
```

## Serendipity Tracking

The most important discovery — Rqc2p as a natural precedent for protein-directed
nucleotide selection — came through an unexpected path:

1. Searched `mesh "nonribosomal peptide"` for NRPS (nucleic-acid-independent
   peptide synthesis)
2. MeSH descriptor D046908 includes both NRPS papers AND Rqc2p paper
3. Rqc2p appeared as #6 because MeSH classified it under "Peptide Biosynthesis,
   Nucleic Acid-Independent" — same category as NRPS

This serendipity was enabled by MeSH's curated cross-classification. A keyword
search for "reverse translation" would NOT have found this paper. The MeSH
taxonomy connected two conceptually distant systems (NRPS and RQC) through
their shared property: nucleic-acid-independent peptide synthesis.

## Research Findings

### Proposed Architecture: "Reverse Ribosome"

```
Protein → [ClpXP unfoldase] → aa stream
                                   ↓
                            [Rqc2p variants × 20]
                            (each recognizes 1 aa, recruits specific tRNA)
                                   ↓
                            tRNA-codon stream
                                   ↓
                            [RNA ligase / polymerase]
                                   ↓
                                  mRNA
```

### Sub-system Feasibility

| Sub-system | Natural Precedent | Key Paper | Feasibility |
|-----------|------------------|-----------|-------------|
| Protein Reading Head | ClpXP processive unfoldase | W2213968760 | ★★★★ |
| aa→codon Adapter | Rqc2p (protein-directed tRNA selection) | W1985387528 | ★★☆☆ |
| RNA Assembler | Poly(A) polymerase + RNA ligase | — | ★★★☆ |
| Codon Degeneracy | Not a problem (any codon works) | — | ★★★★ |

### Critical Technical Challenges

1. **Rqc2p specificity expansion**: Current Rqc2p selects only Ala/Thr. Need
   20 variants, each specific for one amino acid. Directed evolution (PACE) or
   computational design (RFdiffusion) could address this.

2. **ClpXP→Rqc2p coupling**: Physical coupling of unfolded aa stream to the
   adapter system. May require engineered fusion or scaffold.

3. **tRNA release → RNA ligation**: Converting recruited tRNA anticodons into
   a contiguous RNA strand. Template-free polymerases exist but lack sequence
   specificity — need to be directed by the tRNA stream.

4. **Processivity**: The entire system must operate processively on a single
   protein molecule, maintaining reading frame.

### Reading List

| # | Paper | Field | Role |
|---|-------|-------|------|
| 1 | W1985387528 (Shen 2015, Science) | RQC | **Core**: mRNA-independent translation |
| 2 | W2740852495 (Shen 2017, Science) | RQC | CAT-tailing mechanism |
| 3 | W2000789682 (Brandman 2012, Cell) | RQC | RQC complex discovery |
| 4 | W2213968760 (Olivares 2011, NRM) | ClpXP | Processive protein reading |
| 5 | W2018756788 (Woese 2000, MMBR) | aaRS | Genetic code + aaRS evolution |
| 6 | W2124254997 (Fischbach 2004) | NRPS | Non-ribosomal peptide synthesis |
| 7 | W4379600803 (NAR 2023) | RT | De novo primer synthesis by RT |
| 8 | W2077686875 (Science 1999) | QC | Translation quality control mechanisms |
