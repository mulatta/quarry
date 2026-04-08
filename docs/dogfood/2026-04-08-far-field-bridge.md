---
session_id: dogfood-2026-04-08-far-field-bridge
date: 2026-04-08
persona: bridger
topic: "Quantum cryptography × DNA computing — far-field bridge"
seeds: [W2121213780, W2159876240]
commands_used: [mesh, sql, info, bridge]
commands_count: 10
friction_count: 2
copy_paste_count: 5
scores:
  discoverability: 3
  output_clarity: 5
  workflow_continuity: 3
  information_sufficiency: 5
  command_ergonomics: 3
  total: 19
previous_session: dogfood-2026-04-08-protein-eng-regression
score_delta: -4
improvements_tested:
  - "pairwise sp matrix for far-field (sp=7)"
  - "sp>=6 warning message"
  - "bridge with extreme distance"
design_refs: []
---

# Dogfood: Quantum Cryptography × DNA Computing (Far-field Bridge)

## Session Context

Bridger persona testing scenario B3 (far-field bridge, sp>10 expected).
Goal: find connections between quantum cryptography and DNA computing —
two fields with presumably zero citation overlap.

Constraint: FTS/embedding unavailable. MeSH doesn't cover CS/physics
subfields well.

## Command Sequence Log

| # | Command | Result | Time | Notes |
|---|---------|--------|------|-------|
| 1 | `mesh "quantum cryptography"` | Not found | <1s | MeSH gap |
| 2 | `mesh "quantum"` | 5 matches (QM, QDots) | <1s | Wrong subfield |
| 3 | `mesh "DNA computing"` | Not found | <1s | MeSH gap |
| 4 | `mesh D011789` (Quantum Theory) | Quantum chemistry | <1s | Wrong papers |
| 5 | `sql ... title ILIKE '%quantum key%'` | QKD papers found | <1s | SQL required |
| 6 | `sql ... title ILIKE '%DNA comput%'` | DNA computing found | <1s | |
| 7 | `info W2121213780 W1561605958 --mesh` | No MeSH on either | <1s | PubMed gap |
| 8 | `bridge W2121213780 W1561605958` | Seed not in graph | <1s | CSR graph gap |
| 9 | `bridge W2121213780 W2159876240` | sp=7, coupling=12 | 0.8s | Connection found! |
| 10 | `info W4282938990 W1984638028 W4362474674` | Bridge papers | <1s | |

### Sequence Annotations
- Copy-paste: 5 times
- Unnecessary commands: CMD 4 (Quantum Theory → quantum chemistry, wrong)
- Backtracking: CMD 8 (seed not in graph → try different seed)
- SQL required: 2 times (seed discovery impossible via MeSH)

## Five-Dimension Rating

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Discoverability | 3/5 | -2: both fields missing from MeSH, SQL-only seed discovery |
| Output clarity | 5/5 | sp=7 + warning, coupling/cocitation/path all informative |
| Workflow continuity | 3/5 | -2: mesh fail → SQL, seed not in graph → retry with new seed |
| Information sufficiency | 5/5 | coupling found real cross-domain papers |
| Command ergonomics | 3/5 | -2: SQL required, no graph membership check before bridge |

**Total: 19/25**

## Friction Log

```
FRICTION-1: MeSH doesn't cover CS/physics subfields
COMMAND: mesh "quantum cryptography", mesh "DNA computing"
OUTPUT:  No match for either
PROBLEM: MeSH = biomedical vocabulary. CS/physics topics absent.
         SQL title search is the only workaround.
WANTED:  FTS (blocked). Or mesh fallback: "No MeSH match. Try:
         quarry sql \"SELECT ... WHERE title ILIKE '%...%'\""

FRICTION-2: seed not in citation graph
COMMAND: bridge W2121213780 W1561605958
OUTPUT:  "Seed 1561605958 not found in graph"
PROBLEM: Paper exists in PG metadata but not in CSR citation graph.
         No suggestion for alternatives.
WANTED:  "Seed not in graph. Similar papers in graph:" with SQL fallback.
         Or: info output should show "in_graph: true/false".
```

## Additional Observations

**Surprises (positive)**:
- sp=7 but connection EXISTS! quantum crypto × DNA computing are not
  totally disconnected. Coupling found 12 bridge papers.
- Top coupling: "DNA origami prime factorization" (W4362474674) —
  DNA computing doing Shor's algorithm target. Genuine cross-domain insight.
- Cocitation reveals deep shared ancestry: Shannon information theory,
  Turing computability, Landauer's principle — the theoretical roots
  connecting information, computation, and physics.
- Path bridges found 3 stepping stones through exciton/FRET physics.

**Surprises (negative)**:
- Both seeds have no MeSH tags (not in PubMed). Physics/CS gap.
- DNA Computing textbook (W1561605958) not in CSR graph despite
  cited_by=588. Likely a book, not a journal article.

**B3 scenario verdict**:
Bridge handles far-field well. sp=7 is far but not ∞. The tool
correctly warns about long chains and still finds meaningful coupling.
"No connection" was not the case here — there IS a connection through
molecular computing and information theory.

## Research Findings

### Quantum Crypto × DNA Computing — Connection Map

```
Quantum Key Distribution          DNA Computing
(W2121213780, sp=7)               (W2159876240)
         │                              │
         ├─ Information theory ←────────┤  (Shannon 1948, cocitation)
         ├─ Computation theory ←────────┤  (Turing 1937, cocitation)
         │                              │
         └──── Coupling bridges ────────┘
               │
               ├─ DNA logic gates (2022) — enzyme-based logic
               ├─ Natural computing (2008) — unifying framework
               ├─ DNA origami prime factorization (2023) — Shor analog
               └─ Molecular communication (2006-2011) — nano-scale info
```

### Key Bridge Papers

| Paper | Year | Cited | Connection Type |
|-------|------|-------|----------------|
| DNA logic gates (IEEE TNB) | 2022 | 10 | Bio-logic ↔ quantum logic |
| Natural computing (CACM) | 2008 | 238 | Umbrella framework |
| DNA origami factorization (Sci Adv) | 2023 | 8 | Shor's algorithm in DNA |
| Exciton model (stepping stone) | 1965 | 4168 | Energy transfer physics |

### Cocitation Ancestry (top shared intellectual roots)

| Paper | Year | Cited | Role |
|-------|------|-------|------|
| Shannon — Information Theory | 1948 | 79,104 | Information foundation |
| Turing — Computable Numbers | 1937 | 8,024 | Computation foundation |
| Shannon — Secrecy Systems | 1949 | 9,229 | Cryptography foundation |
| Landauer — Irreversibility | 1961 | 4,710 | Physics of computation |
| Feynman — QM Computers | 1986 | 1,705 | Quantum computing origin |
