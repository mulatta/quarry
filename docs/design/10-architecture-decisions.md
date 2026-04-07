# Architecture Decisions

> Record of key architecture decisions with context and rationale.

## AD-1: Python CLI + Rust .so (Phase 1-2), Pure Rust CLI (Phase 2+)

**Date**: 2026-04-01
**Status**: Active

### Context

`quarry expand` has two layers:

- **Graph computation** (APPR, coupling, cocitation, wRRF, bridge): Rust, ~1s
- **Metadata enrichment** (PG lookup, JSON assembly, relation tagging): Python, ~50ms

The Python CLI is ~200 lines of thin wrapper around Rust .so (via PyO3).

### Decision

| Phase | Architecture | Rationale |
|-------|-------------|-----------|
| 1b (now) | Python CLI + Rust .so | Working, no performance issue. Avoid delay. |
| 2 | Evaluate pure Rust CLI | If MCP expand duplication or deployment burden triggers |
| 3 | Server-client (Rust server + gRPC) | When embedding service (Python) is added |

### Migration Triggers (Phase 2)

Migrate to pure Rust CLI when **any** of these occur:

- MCP server needs `expand()` → logic duplication becomes real
- Output schema has downstream consumers that need type safety
- Python dependency (pydantic, psycopg, rich) becomes deployment burden

### Phase 3 Embedding Architecture (Pre-decided)

```
Python embedding service (sentence-transformers + LanceDB)
  ↕ gRPC
Rust CLI / Rust server
  ↕ mmap
CSR graph
```

Python stays for embedding inference. Rust stays for graph computation.
gRPC bridges them. This means Phase 2 Rust CLI should include gRPC client
scaffolding for Phase 3 readiness.

### Mitigation: Shared Logic Extraction

To minimize Phase 2 migration cost, extract expand logic into
`quarry/core/expand.py` — a pure function that both CLI and MCP server
call. This function becomes the exact scope of Rust migration.

```python
# quarry/core/expand.py
def run_expand(seed, mode, limit, ...) -> dict:
    """Resolve seed → Rust expand → PG enrich → JSON-ready dict."""
```

### Alternatives Considered

| Alternative | Why not (now) |
|-------------|--------------|
| Pure Rust CLI now | Phase 1b delayed. PyO3 binding still needed for MCP. Maintenance doesn't decrease. |
| Server-client now | Over-engineering for single-user CLI. No embedding service yet. |
| Keep Python forever | maturin .venv issues recurring. Schema safety weak (dict-based). |

## AD-2: Quality Signals as Metadata Only

**Date**: 2026-04-01
**Status**: Active

### Context

Quality signals (cnp, fwci, rcr) measure "field importance" not "relevance
to seed". Evaluation showed fwci reranking promotes off-topic high-impact
papers and demotes seed-relevant niche papers.

### Decision

Quality signals are **exposed in output JSON as metadata**. Not used for
scoring, reranking, or fusion. Consumer decides how to use them.

### Rationale

- quarry is a data pipeline — returns complete data, not opinions
- fwci measures importance, not relevance (verified on 5 seeds)
- cnp and fwci are ρ=0.99 correlated — cnp preferred (0-1 range)
- NULL handling: return null, no imputation (preprints = unmeasured, not average)

See: [03-layer-quality.md](03-layer-quality.md), [07-missing-data.md](07-missing-data.md)

## AD-3: No --sort/--filter in CLI

**Date**: 2026-04-01
**Status**: Active

### Decision

quarry expand outputs complete JSON. No `--sort`, `--filter`, `--rel`
options. Filtering/sorting is done downstream with jq, DuckDB, pandas.

### Rationale

Adding sort/filter options creates a query language — each combination
needs testing and maintenance. This is a solved problem (SQL, jq).
quarry's value is in graph computation, not in reimplementing WHERE clauses.

```bash
# quarry outputs data
quarry expand W4406019873 --format json > results.json

# downstream filters (user's choice of tool)
jq '.papers[] | select(.relation == "lateral")' results.json
duckdb -c "SELECT * FROM read_json('results.json') WHERE year >= 2020"
```

## AD-4: No Direction-Restricted APPR

**Date**: 2026-04-02
**Status**: Active (rejected after experiment)

### Context

Direction filtering = restrict APPR push to forward-only or reverse-only edges.
Use case: "forward = ancestry chain", "reverse = impact chain".

### Experiment

Tested on 3 seeds (Dao-2016, Seo-2025, Zhao-2021) with directional APPR vs
bidirectional APPR (both). Compared top-200 results with title+abstract review.

### Findings

**Forward-only APPR problems:**

- Includes upstream generic tools (PHENIX, Bradford assay, Cytoscape, ColabFold)
  that seeds happen to cite — **noise, not ancestry**
- Loses all direct-topic recent research that Both APPR finds
- Node count 3-4× larger than Both → diluted precision

**Reverse-only APPR problems:**

- Includes broad application papers (cardiovascular therapy, eye disease)
  that happen to cite seed-related work — **mixed quality**
- Fails completely for high-citation seeds (Zhao: 449 citers → mass dilution)
- 5/5 target grandchildren not scored at all for Zhao

**Both APPR advantage:**

- Bidirectional push acts as natural relevance filter — generic tools get
  low scores because they lack reverse support, and vice versa
- Highest precision across all seeds

**Terminology note:** "Forward/Reverse PPR" in literature means computation
direction (from source vs toward target), NOT edge restriction. Our experiment
tested edge restriction, which is non-standard.

### Alternative: limit increase

- limit=200 → 0/10 missing 2-hop papers recovered
- limit=500 → 29/30 recovered (rank 202-492)
- Simple, consistent, no implementation cost

### Decision

Direction filtering at APPR level rejected. Instead:

1. **Relation tag** (foundation/follow-up/lateral) for post-hoc direction filtering
1. **Higher limit** (500) when deeper chain coverage needed
1. Use cases like "show only foundations" → `jq '.papers[] | select(.relation == "foundation")'`

### References

- Andersen, Chung, Lang (FOCS 2006) — APPR framework
- Lofgren et al. (KDD 2015) — forward vs reverse PPR (computation direction, not edge restriction)
- Connected Papers — uses co-citation/coupling, not directional PPR
- No major citation tool uses direction-restricted PPR

## AD-5: HKPR Deferred to Phase 3

**Date**: 2026-04-02
**Status**: Active (deferred after experiment)

### Context

HKPR (Heat Kernel PageRank, Kloster & Gleich 2014) uses Poisson decay instead
of APPR's geometric decay. Lower seed dominance (~10% vs ~47%), broader
exploration (45-82K nodes vs 1.5-2.4K). Evaluated as APPR replacement,
complement, and wRRF hybrid.

### Experiment

3 seeds × 6 weight configs × 2 t values. Abstract-level quality review.

**HKPR-only vs APPR-only (top-200 unique papers):**

| Seed | APPR precision | HKPR precision |
|------|---------------|----------------|
| Zhao-2021 | 90% | 40% (4/10 relevant, 4 upstream landmark) |
| Dao-2016 | 80% | 60% (6/10 relevant, 4 generic tool) |
| Seo-2025 | 100% | 20% (2/10 relevant, 7 generic tool) |

HKPR adds **field origin papers** (SELEX 1990, Jinek 2012) and **generic tools**
(PHENIX, Mfold, AlphaFold). On-topic but not seed-specific.

**wRRF hybrid (expand + HKPR):**

| Config | Dao | Seo | Zhao | Net |
|--------|-----|-----|------|-----|
| hkpr=0.10 | equal~improved | improved | **degraded** | mixed |
| hkpr=0.20 | slightly improved | slightly improved | **clearly degraded** | degraded |

Hybrid at hkpr=0.10: Seo gains landmark (MHETase c=455, biocatalysis c=296),
but Zhao loses core BE papers (cytidine BE c=136, CP-Cas9 c=323, precise ABE
c=175) replaced by broad application papers (cardiovascular, osteoarthritis).

### Root Cause

HKPR's broad exploration promotes **high-citation landmark papers** that are
2-3 hops upstream from seed. These displace **seed-specific niche papers** in
the top-200. The effect is inconsistent across seeds — beneficial when
landmarks are directly relevant (Seo/PET), harmful when landmarks are
generic (Zhao/CRISPR origin papers displacing specific BE techniques).

### Decision

1. **APPR remains the sole scoring algorithm** for expand
1. **HKPR code retained** (`algo/hkpr.rs`, `graph.hkpr()` binding) for future use
1. **Re-evaluate in Phase 3** when embedding layer is available — embedding
   similarity can distinguish "seed-relevant landmark" from "generic upstream
   tool", enabling selective HKPR paper promotion
1. Potential Phase 3 approach: HKPR candidates filtered by embedding
   similarity to seed before fusion

### References

- Kloster & Gleich (KDD 2014) — "Heat Kernel Based Community Detection"
- Chung (2007) — "The heat kernel as the pagerank of a graph"

## AD-6: Bridge Discovery — Per-Type Independent Algorithms

**Date**: 2026-04-02
**Status**: Active (supersedes previous multi-seed expand design)

### Context

Multi-seed use case: find papers that **connect** two or more seeds from
different subfields. Initial design used APPR-based modes (field/common/bridge).

### Experiment (Phase 2a)

Implemented and tested APPR-based multi-seed expand with 3 modes on 3 seed
pairs (close/medium/far overlap). Results:

| Mode | Approach | Result |
|------|----------|--------|
| field (shared teleport APPR) | Union of neighborhoods | = single-seed expand × 2, no new information. Biased toward higher-citation seed. |
| common (APPR intersection) | Score > 0 filter | 72% noise (144/200 papers not in either baseline). Too permissive. |
| bridge (APPR + coupling) | PPR product + cross-coupling | Far pair overlap = 0, cannot find bridges. Coupling rerank promotes reviews, demotes seed-specific papers. |

**Root cause**: APPR is designed for "neighborhood exploration around a seed",
not "connection between seeds". Different algorithms find structurally
different bridges — mixing them loses information.

### Decision

**7 independent bridge types**, each computed separately:

| Type | Algorithm | Scope | Question answered |
|------|-----------|-------|-------------------|
| common_refs | refs(A) ∩ refs(B) | local 1-hop | Shared foundation? |
| common_citers | citers(A) ∩ citers(B) | local 1-hop | Who synthesized both? |
| coupling | cross-seed bibliographic coupling | local 2-hop | Who combined both methods? |
| cocitation | cross-seed co-citation | local 2-hop | What do both communities read? |
| path | shortest path / k-shortest paths | global | How does literature connect them? |
| ppr | per-seed APPR product | global | What's central to both networks? |
| steiner | Steiner tree (k ≥ 3 only) | global | Minimal reading path through all? |

Each type is returned independently. Consumer decides which types to use.
APPR is reserved for single-seed `expand` only.

### Rationale

- Different algorithms find **different kinds** of bridges (shared foundation
  vs methodology combination vs narrative path vs structural hub)
- Mixing into one ranked list loses the "why is this a bridge?" information
- Consistent with AD-2 (metadata only) and AD-3 (no opinions — let consumer
  decide)
- Far pair (overlap < 1%): citation-only bridges are sparse. Distance
  classification (computed as byproduct) signals when Phase 3 embedding
  is needed.

### Alternatives Rejected

| Alternative | Why not |
|-------------|---------|
| APPR-based multi-seed (field/common/bridge modes) | Tested: field = union, common = noisy, bridge fails on far pairs |
| Single fused score across bridge types | Loses bridge-type information |
| Pixie random walk | Stochastic, conflicts with deterministic principle |

### Phase 3 Extension

Embedding adds a **candidate source**, not a replacement:

- Close pair: citation bridges sufficient
- Medium pair: citation bridges + embedding rerank
- Far pair: embedding candidates + citation validation (order reverses)

See: [11-bridge.md](11-bridge.md) for full design.

## AD-7: CSR Graph Cleanup and Lightweight Build Options

**Date**: 2026-04-03
**Status**: Proposed

### Context

Current CSR: 182M nodes, 3.77B edges, ~30GB. Built from OA works +
OA citations + iCite citations merge.

Bridge quality evaluation (Type 1-4) revealed data quality issues that
inflate graph size without adding signal:

### Issues Found

**1. Duplicate edges** — OA + iCite merge produced 792M duplicate edges
(20%). 93% of iCite citations already exist in OA (same PubMed source).
Fix: LEFT ANTI JOIN to insert only iCite-exclusive edges (~58M).

**2. Supplementary files as nodes** — OpenAlex registers "Additional
file 1 of ..." as separate works with type=article. These appear in
Type 3 coupling results as non-PMID noise.

**3. Non-paper work types** — dataset (21M), book-chapter (10M),
dissertation (1.7M), book (3M), other (2M) = 39M nodes (24% of CSR).
These participate in citations but are not useful for bridge discovery.

**4. Duplicate works (journal variants)** — Same paper in multiple
journal editions (e.g., Angewandte Chemie / Angewandte Chemie Int. Ed.)
registered as separate work_ids with independent citation edges.

### Decision

Build-time filters in `merged_citations` (sql/ch_transform.sql):

| Filter | Method | Effect |
|--------|--------|--------|
| Edge dedup | iCite ANTI JOIN against OA | -792M edges (20%) |
| Paper types only | INNER JOIN `paper_ids` view | -39M nodes, -700M+ edges |
| Supplementary removal | `NOT startsWith(title, 'Additional file')` | -1.7M noise nodes |

`paper_ids` VIEW: `type IN ('article','review','preprint','editorial','letter')`,
excluding supplementary files by title prefix.

Engine changed: `ReplacingMergeTree` → `MergeTree` (no duplicates to replace).

Expected result: ~143M nodes / ~2.6B edges (~20GB, from 182M / 3.8B / ~30GB).

### Lightweight Build Option (Deferred)

PubMed-only profile: `PMID > 0` filter → ~34M nodes / ~1.7B edges (~8GB).
Trades Type 3 coverage (60-70%) for 4x size reduction.

### Metadata API Migration (Deferred)

PG/CH metadata can be replaced by OA API + iCite API at runtime.
CSR is the sole irreplaceable local asset.

## AD-8: `info` Command and `sql` Lifecycle

**Date**: 2026-04-07
**Status**: Active

### Context

Dogfood session (surveyor persona: cytidine deaminase discovery for base
editors) revealed that `quarry sql` is used for two distinct purposes:

1. **Seed discovery**: finding papers by keyword before expand/bridge
1. **Metadata lookup**: getting DOI, venue, cited_by_count for papers
   found by expand/bridge

Purpose (1) is properly served by `search` when FTS + embedding are
available. Purpose (2) has no dedicated command — users must write raw
SQL with unknown schema, causing repeated friction:

- Schema guessing (`publication_year` → `UndefinedColumn` error)
- Copy-paste work_id from expand output → sql query → repeat
- 5 out of 13 commands in the session were metadata lookups via sql

MCP already has `get_paper` tool for single-paper lookup. CLI has no
equivalent.

### Decision

**Add `quarry info <work_id>` CLI command.**

```
quarry info W4382247824
quarry info W4382247824 W4413279334   # multiple
```

Output: work_id, title, pub_year, doi, host_venue, cited_by_count,
oa_status, rcr, abstract (truncated unless --full).

Implementation: thin wrapper around PG parameterized query (same as
MCP `get_paper`). No raw SQL, no injection surface.

### `sql` Lifecycle

`sql` is a development escape hatch, not a user-facing feature.

| Phase | `sql` status | Rationale |
|-------|-------------|-----------|
| 1-2 (CLI) | **Keep** | Developer debugging, ad-hoc exploration |
| 3 (Server) | **Do not expose** via API | Injection surface, no parameterization |

`sql --schema` added as convenience for development phase. Both `sql`
and `--schema` are removed when Server API replaces CLI as primary
interface. Server exposes only parameterized endpoints: `info`,
`expand`, `bridge`, `search`.

### `expand`/`bridge` Output Enrichment

Separate from `info`: add DOI, host_venue, cited_by_count to
expand/bridge table output. This eliminates the most common
`sql` usage pattern (metadata lookup after expand).

Implementation: these fields are already fetched in PG enrichment
step (`core/expand.py`, `core/bridge.py`). Change is output-only:
add columns to `--format table`, include in JSON output.

### Alternatives Considered

| Alternative | Why not |
|-------------|---------|
| Only enrich expand/bridge output | Doesn't cover standalone lookup ("what is W4382247824?") |
| Keep sql as the only lookup method | Schema guessing, injection risk, bad UX |
| Add `quarry schema` as separate command | Unnecessary if `sql --schema` exists; both are dev-only anyway |

## AD-9: MeSH CLI Exposure and Ontology Strategy

**Date**: 2026-04-07
**Status**: Active

### Context

Two dogfood sessions revealed MeSH as a complementary discovery mechanism
to citation-based tools:

1. **Surveyor session** (cytidine deaminase): sql ILIKE was the only seed
   discovery method. FTS/embedding unavailable.
1. **Newbie session** (retron): `ILIKE '%retron%'` matched "retronasal"
   noise. MeSH descriptor D018626 (Retroelements) would have provided
   precise topic filtering.

MeSH provides a **curated, hierarchical vocabulary** that finds papers
citation graph cannot reach. Verified empirically: top Retroelements
papers by MeSH vs retron expand results had **zero overlap** — entirely
different sets.

Backend infrastructure already exists:

- PG: work_mesh (379M rows), mesh_tree (65K), both indexed
- pg.py: `mesh_search_by_name()`, `mesh_descendants()`, `mesh_by_ui()`,
  `top_mesh()`
- MCP: `mesh_explore` tool
- CLI: **nothing** — this is the gap

### Decision

**Three MeSH features for CLI, in priority order:**

#### 1. `info --mesh` — Paper MeSH tags

```
quarry info W3096234081 --mesh
  ...
  mesh:
    ★ Bacteria (D001419)
    ★ Retroelements (D018626)
      Escherichia coli (D004926)
```

Query: `work_mesh WHERE work_id = ?` — 0.1ms, no schema change needed.

#### 2. `quarry mesh` — MeSH-based paper discovery + tree browsing

```
quarry mesh "retroelement"      # name search → top papers
quarry mesh D018626 --tree      # hierarchy browsal
```

Paper discovery query: `work_mesh JOIN works WHERE descriptor_ui = ? AND is_major_topic ORDER BY cited_by_count DESC LIMIT N`.

Performance by descriptor cardinality (EXPLAIN ANALYZE):

- Retroelements (5.6K papers): **36ms** — acceptable
- Bacteria (306K papers): **6.2s** — unacceptable but irrelevant

High-cardinality descriptors are too broad for seed discovery.
CLI guards with count check and suggests `--tree` drill-down:

```
⚠ Bacteria (D001419): 104,994 major-topic papers (too broad)
  Use --tree to find sub-descriptors
```

New pg.py method: `get_top_works_by_mesh(descriptor_ui, limit)`.

#### 3. `expand --mesh-summary` — Result MeSH distribution

```
quarry expand W3096234081 --mesh-summary
  MeSH summary (major topics, top 10):
    Bacteriophages (D001435)         16/25
    CRISPR-Cas Systems (D000069236)  12/25
```

Query: `work_mesh WHERE work_id = ANY(result_ids) AND is_major_topic GROUP BY descriptor_ui` — 28ms for 25 papers.

New pg.py method: `top_mesh_by_work_ids(work_ids, limit)`.

### Schema Assessment

Current schema and indexes are sufficient:

| Index | Covers |
|-------|--------|
| `idx_work_mesh_wid` btree(work_id) | info --mesh, expand mesh-summary |
| `idx_work_mesh_desc` btree(descriptor_ui) | mesh paper discovery |

Optional improvement: composite index `(descriptor_ui, is_major_topic)`
eliminates in-memory filter. Not required — current performance is
acceptable for realistic descriptor cardinalities (\<50K papers).

No denormalization needed. No schema redesign needed.

### Ontology Strategy

#### MeSH: in quarry (paper-level enrichment + filtering)

MeSH maps directly to papers via work_mesh. quarry's unit is the paper,
so MeSH is a natural fit for enrichment and discovery.

#### UMLS/GO: outside quarry

| Ontology | Why outside |
|----------|-------------|
| GO | Annotates genes/proteins, not papers. Paper→GO requires indirect mapping (UniProt mentions) |
| UMLS | 3.5M concepts, but paper mapping still goes through MeSH. Marginal gain vs complexity |
| SNOMED/OMIM | Clinical focus, not research paper discovery |

UMLS's main advantage (synonym expansion) is better served by
embedding similarity. Re-evaluate after Phase 3 embedding is live.

#### MeSH in L2 Content Layer (Phase 3)

After embedding, MeSH overlap becomes an L2 atomic operation:

```
MeSH_overlap(paper_i, seed) = |MeSH(i) ∩ MeSH(seed)| / |MeSH(seed)|
```

Weighted by IDF to suppress generic descriptors (Humans, DNA).
Fused with embedding + BM25 via wRRF. This is the structured
complement to unstructured embedding similarity.

#### Heterogeneous Graph (Phase 3+, research)

Adding MeSH edges to citation CSR is technically feasible but
architecturally premature. Trigger: repeated evidence that embedding
misses what MeSH catches (or vice versa) at the graph traversal level.

### Implementation Notes

- `info --mesh`: no new pg.py method, direct query in CLI
- `quarry mesh`: wraps existing `mesh_search_by_name()` + new
  `get_top_works_by_mesh()`
- `quarry mesh --tree`: wraps existing `mesh_by_ui()` +
  `mesh_descendants()` + new `mesh_parent()`
- `expand --mesh-summary`: new `top_mesh_by_work_ids()` (work_id
  based, current `top_mesh()` is pmid-based)
