---
scenarios:
  # reader
  R1: { status: done,    session: S5, desc: "Start from DOI" }
  R2: { status: done,    session: S14, desc: "Expand → lateral → bridge" }
  R3: { status: done,    session: S5, desc: "Expand → shrink" }
  R4: { status: done,    session: S14, desc: "Old classic paper as seed" }
  # newbie
  N1: { status: blocked, blocker: FTS, priority: P1, desc: "Start from search" }
  N2: { status: blocked, blocker: embedding, desc: "Start from vsearch" }
  N3: { status: blocked, blocker: FTS, desc: "Search vs mesh comparison" }
  # bridger
  B1: { status: done,    session: "S6,S12", desc: "3-seed steiner bridge" }
  B2: { status: done,    session: "S6,S14", desc: "Close bridge (same field)" }
  B3: { status: done,    session: S11, desc: "Far-field bridge (sp>10)" }
  # surveyor
  S1: { status: done,    session: "S2,S4,S6,S9", desc: "mesh-summary → drill-down loop" }
  S2: { status: done,    session: S8, desc: "Shrink venue comparison" }
  S3: { status: done,    session: S14, desc: "Chronological lineage + bridge chain" }
---

# Dogfood Scenarios

Scenario coverage tracker for quarry CLI UX testing.
Each scenario targets a specific persona × workflow pattern.
Status summary is in the YAML frontmatter above.

## Scenario Details

### R2: Expand → lateral discovery → bridge

**Goal**: Test whether a reader who spots a "lateral" paper in expand
results can naturally transition to bridge for systematic cross-domain
exploration. Serendipity → structured investigation.

**Key questions**:
- Do lateral tags surface genuinely interesting cross-domain papers?
- Is the expand → bridge transition intuitive?
- Does bridge explain *why* the lateral paper appeared?

**Example seed**: W3141751158 (CRISPR epigenome editing, Cell 2021)

**Expected workflow**: info → expand → pick lateral → bridge(seed, lateral) → info

---

### R4: Old classic paper

**Goal**: Test expand/shrink behavior on pre-2000 classic papers where
the citation graph structure differs — few forward citations, thousands
of reverse citations (being cited).

**Key questions**:
- Does expand show "entire field history" or skew toward recent papers?
- Is foundation/follow-up ratio different from modern papers?
- Does shrink work on classic fields?

**Example seed**: W2045435533 (Cas9, Jinek et al. 2012, Science)

**Expected workflow**: info --mesh → expand --mesh-summary → shrink

---

### N1: Start from search (blocked: FTS)

**Goal**: Test the newbie first-experience when only a keyword is known.
search → seed discovery → expand pipeline.

**Key questions**:
- Are relevant papers ranked near the top?
- Can users pick a seed from title + year + cited_by alone?
- How much copy-paste friction in search → expand transition?

**Example prompt**: "aptamer selection" (user has no specific paper)

**Expected workflow**: search "aptamer selection" → info (top hits) → expand → shrink

---

### N2: Start from vsearch (blocked: embedding)

**Goal**: Test vsearch for "find similar papers" when the user already
knows one paper but wants citation-disconnected neighbors.

**Key questions**:
- How much do vsearch and expand results overlap?
- Does vsearch find citation-disconnected papers?
- Is vsearch → expand → bridge natural?

**Example seed**: W2495359280 (AptaTRACE)

**Expected workflow**: vsearch W... → info (top hits) → expand (best hit)

---

### N3: Search vs mesh comparison (blocked: FTS)

**Goal**: Compare free-text search vs controlled vocabulary (MeSH) for
the same topic. When should users prefer which tool?

**Key questions**:
- Result overlap between search "CRISPR delivery" and mesh "Drug Delivery Systems"?
- What does MeSH miss (new/niche)? What does search miss (noise)?
- Optimal combined pattern?

**Expected workflow**: search "CRISPR delivery" → mesh "drug delivery" →
compare results → expand (top hit from each)

---

### B1: 3-seed steiner bridge

**Goal**: Validate steiner bridge (k≥3) in practice. Does it provide
insights beyond pairwise bridge union?

S6 tested with sp=1 seeds (all directly connected) → empty steiner
(correct behavior). Need sp=2-4 seeds for meaningful intermediate nodes.

**Key questions**:
- Does steiner return junction papers not found in any single expand?
- Are steiner hub papers genuinely useful as narrative backbone?
- How does steiner differ from shrink union? (connectivity vs coverage)

**Example seeds** (verified sp=3/8/∞):
- W2973445868 (SHERLOCK, CRISPR diagnostics, 2019)
- W2190054909 (CRISPR gene drive, mosquito, 2015)
- W3119813698 (in vivo base editing, progeria, 2021)

**Pre-check**: info --mesh each seed, then 2-seed bridge to verify
pairwise sp ≥ 2. ABE+CBE+PE (original example) fails because sp=1.

**Expected workflow**: info --mesh (3 seeds) → bridge (2-seed sp check)
→ bridge (3-seed steiner) → info (steiner hubs) → expand (hub paper)

---

### B2: Close bridge (same field)

**Goal**: Test whether bridge is useful when two papers are already
close (same lab, same subfield, high overlap).

**Key questions**:
- Are common_refs/common_citers too numerous to be meaningful?
- Does coupling/cocitation produce only redundant information?
- Is "bridge within a known field" a valid use case?

**Example seeds**: W2766599608 (ABE) + W2336828812 (CBE) — both David Liu lab

**Expected workflow**: bridge W... W... → interpret (expect high overlap)

---

### B3: Far-field bridge (sp>10)

**Goal**: Test bridge behavior when two fields have zero citation
overlap. Does bridge fail gracefully or produce generic tool-papers?

**Key questions**:
- How does bridge communicate "no meaningful connection"?
- Do coupling/cocitation results contain only generic papers (BLAST, AlphaFold)?
- Is "no connection" itself useful information (niche opportunity signal)?

**Example**: quantum computing × CRISPR genome editing

**Expected workflow**: find seeds via mesh/sql → bridge W... W... →
interpret (expect overlap=0)

---

### S1: mesh-summary → drill-down loop

**Goal**: Test iterative field mapping: expand → mesh-summary →
drill into a sub-MeSH → expand again. Does the loop converge?

**Key questions**:
- Does drilling into a child MeSH descriptor discover new seeds?
- Is expand → mesh-summary → mesh → expand effective for field mapping?
- Can the user detect saturation (when to stop)?

**Example topic**: protein engineering (broad field)

**Expected workflow**: mesh "protein engineering" → expand (top paper)
→ --mesh-summary → mesh (child descriptor) → expand (sub-topic) → repeat

---

### S2: Shrink venue comparison

**Goal**: Quantify the quality difference between NCS+ and field-specific
venue lists for the same seed.

**Key questions**:
- Paper overlap between NCS+ and custom venue shrink?
- Are there fields where a venue preset would be better than NCS+?
- Does low NCS+ coverage indicate a field-specific journal ecosystem?

**Example seed**: W2042789810 (MERFISH)

**Expected workflow**: shrink --venue NCS+ → shrink --venue custom →
compare overlap, coverage, paper types

---

### S3: Chronological lineage + bridge chain

**Goal**: Trace technology evolution over time using expand's
foundation/follow-up chain and bridge as stepping stones.

**Key questions**:
- Do foundation papers form a chronological technology evolution path?
- Does bridge(old_tech, new_tech) path type reveal stepping stones?
- Is shrink temporally diverse (mixing decades)?

**Example**: PCR (1985) → digital PCR (2012) technology lineage

**Expected workflow**: find seeds (PCR origin + digital PCR key paper) →
bridge(old, new) → path bridges = intermediate tech → expand (intermediate)
→ shrink (full lineage)
