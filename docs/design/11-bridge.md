# Bridge Discovery

> Status: Design confirmed. Implementation pending.

## Problem

Given two or more seed papers from different subfields, find papers that
**connect** them. Single-seed expand finds papers *related to* one seed;
bridge finds papers *between* seeds.

Bridge discovery is the primary multi-seed use case for niche/interdisciplinary
ideation.

## Design Principle: Separate Bridge Types

Different algorithms find structurally different bridges. Mixing them into
one ranked list loses information. Each bridge type answers a distinct question
and is computed independently.

This follows quarry's philosophy: "data pipeline — returns complete data,
not opinions" (consistent with AD-2, AD-3).

## 7 Bridge Types

### Type 1: common_refs (LCA)

**Definition**: `refs(A) ∩ refs(B)` — papers cited by all seeds.

**Meaning**: Shared intellectual foundation.

**Question answered**: "What theoretical/methodological basis do these
papers share?"

**Example**: Gaudelli 2017 (ABE) — cited by both progeria BE and sickle cell BE.

**Computation**:

```
fwd_neighbors(seed_A) ∩ fwd_neighbors(seed_B)
Each shared ref weighted by AA: 1/log(indegree)
```

**Cost**: O(deg(A) + deg(B)), < 1ms.

**k > 2**: Full intersection (all seeds cite) + pairwise intersections
with `cited_by_seeds` count.

**Phase 3**: Embedding filters "core foundation" vs "pro-forma citation".

______________________________________________________________________

### Type 2: common_citers (Common Descendant)

**Definition**: `citers(A) ∩ citers(B)` — papers that cite all seeds.

**Meaning**: Synthesis/comparison/review papers that already combine
both fields.

**Question answered**: "Who has already looked at these topics together?"

**Example**: "In vivo base editing for monogenic diseases" (review) — cites
both progeria BE and sickle cell BE.

**Computation**:

```
rev_neighbors(seed_A) ∩ rev_neighbors(seed_B)
Each shared citer weighted by AA: 1/log(outdegree)
```

**Cost**: O(deg(A) + deg(B)), < 1ms.

**k > 2**: Full intersection + pairwise, with `cites_seeds` count.

**Phase 3**: Embedding verifies whether the citer actually compares/synthesizes
vs merely lists both in a generic review.

______________________________________________________________________

### Type 3: coupling_bridges (Cross-seed Bibliographic Coupling)

**Definition**: Paper C where `refs(C) ∩ refs(A) > 0` AND
`refs(C) ∩ refs(B) > 0`.

**Meaning**: A third paper that combines methodologies/concepts from both
seeds' reference worlds.

**Question answered**: "What research actually uses methods from both fields?"

**Example**: A paper citing both in-vivo delivery methods (A's world) and
HSC therapy protocols (B's world).

**Computation**:

```
For each ref r in refs(A):
    Skip if indegree(r) > max_neighbor_degree   # prune hubs (default 10,000)
    For each citer C of r:
        overlap_A(C) += AA_weight(r)
Filter to candidates where overlap_A > 0

For each candidate C:
    overlap_B(C) = Σ AA_weight(r) for r ∈ refs(C) ∩ refs(B)

score(C) = overlap_A(C) × overlap_B(C)
Return candidates where score > 0
```

**Cost**: O(Σ indegree(ref)), ~100ms. `max_neighbor_degree` prune keeps
traversal fast by skipping high-indegree hub refs (low AA weight, high cost).

**k > 2**: `score = Π overlap_i` over all seeds.

**Phase 3**: Embedding midpoint candidates validated by coupling score.

**Origin**: Kessler (1963) bibliographic coupling + Uzzi (2013) atypical
combinations.

______________________________________________________________________

### Type 4: cocitation_bridges (Cross-seed Co-citation)

**Definition**: Paper C where `citers(A)` and `citers(B)` communities both
reference C.

**Meaning**: A paper that both seeds' downstream communities consider important.

**Question answered**: "What do researchers in both fields commonly read?"

**Computation**:

```
For each citer c of A:
    Skip if outdegree(c) > max_neighbor_degree   # prune review papers (default 10,000)
    For each paper C cited by c:
        overlap_A(C) += AA_weight(c)

For each candidate C:
    overlap_B(C) = Σ AA_weight(c) for c ∈ citers(B) that cite C

score(C) = overlap_A(C) × overlap_B(C)
```

**Cost**: O(Σ outdegree(citer)), ~100ms. `max_neighbor_degree` prune skips
review papers with excessive outdegree (low AA weight, high traversal cost).

**Distinction from coupling**: Coupling = "author of C combined both worlds".
Co-citation = "both communities pay attention to C".

**Phase 3**: Embedding distinguishes "genuinely important to both" vs
"high-citation hub coincidence".

**Origin**: Small (1973) co-citation analysis.

______________________________________________________________________

### Type 5: path_bridges (Directed Shortest Path)

**Definition**: Intermediate nodes on the shortest directed citation
chain between seeds: forward BFS from src (references), reverse BFS
from dst (citers), bridge = intersection.

**Meaning**: Stepping-stone papers — "read these in order to get from A to B".

**Question answered**: "How does the literature connect these two topics?"

**Example**: progeria BE → ABE method → CRISPR review → gene therapy →
sickle cell BE. The intermediate papers form a narrative chain.

**Architecture Decision: Why Directed (not undirected)**

Undirected BFS was evaluated first and rejected. Citation graphs are
small-world: undirected BFS always finds trivial sp=2 paths through
high-degree hubs, producing 100% overlap with Type 1-2 (common_refs/citers).
Tested across 9 seed pairs with parameter sweep (hub pruning threshold
500-50K, Dijkstra with AA edge weight) — all produced sp=2, 0% unique.
Directed BFS forces longer chains (sp=3-8) that follow actual citation
direction, producing 80-100% unique results vs Type 1-4.

**Computation**:

```
1. Forward BFS from src (fwd_neighbors, max_depth)
2. Reverse BFS from dst (rev_neighbors, max_depth)
3. Find minimum total_dist where dist_fwd[v] + dist_rev[v] is minimized
4. Collect bridge nodes where dist_fwd[v] + dist_rev[v] == total_dist
5. path_count = count_fwd[v] × count_rev[v]
```

**Cost**: O(V+E) bounded by max_depth (default 5) and max_visited (500K).
Typical: 30-400ms.

**Parameters**: `--max-path-depth N` (default 5). Determines max BFS
expansion from each side. Effective sp range = 2 × max_path_depth.

**sp quality**: sp 3-5 produces high-quality bridges. sp ≥ 6 may include
off-topic papers (long chains traverse multiple fields). CLI and JSON
output include a warning when sp ≥ 6.

**k > 2 seeds**: Pairwise directed paths, merged by path_count.

**Phase 3**: When no directed path exists (very far pair), embedding
provides an "implicit path" — semantic stepping stones.

______________________________________________________________________

### Type 6: ppr_bridges (PPR Product)

**Definition**: `score(v) = geometric_mean(APPR_1(v), ..., APPR_k(v))`.

**Meaning**: Papers that are structurally central to all seeds' citation
neighborhoods in the global graph.

**Question answered**: "What papers are important in both fields' networks?"

**Distinction from Types 1-5**: Types 1-4 are local (1-2 hop neighborhood).
Type 5 is global but deterministic (directed path). Type 6 is global and
probabilistic (random walk) — finds structurally important papers even
without direct citation relationships to seeds.

**Computation**:

```
For each seed_i:
    π_i = APPR(graph, seed_i, α, ε)

For each node v scored by ALL seeds:
    score(v) = (Π π_i(v))^(1/k)

Exclude seeds. Sort descending. Truncate to limit.
```

**Cost**: O(k/ε) per seed, ~500ms total for k=2 with default params.

**Parameters**:

- `--alpha` (default 0.15): teleport probability. Higher = more seed-local.
- `--epsilon` (default 1e-6): precision threshold. Lower = wider exploration.

Parameter sensitivity (validated across 5 seed pairs, α×ε sweep):

| ε | result count | unique vs T1-5 | time | use case |
|---|-------------|----------------|------|----------|
| 1e-5 | 0-50 | low | \<10ms | fast preview |
| **1e-6** | **15-200** | **20-50%** | **~500ms** | **default — balanced** |
| 1e-7 | 200 (cap) | 15-97% | ~1s | far pairs needing wider search |

α=0.15 is stable across all pair distances. ε has stronger impact:
for far pairs with few results at default ε, lowering to 1e-7 expands
the search at the cost of ~2× runtime.

**Best use case**: medium-distance pairs (same broad field, different
subfields). Close pairs overlap 77%+ with Type 1-4. Far pairs may need
ε=1e-7 for sufficient results. Medium pairs produce 20-35% unique
bridges at default parameters — papers like field-defining reviews,
seminal methods, and cross-subfield benchmarks that Type 1-4 misses.

**k > 2**: Natural extension — geometric mean of k scores.

**Phase 3**: Embedding midpoint ANN may complement PPR for far pairs
where even ε=1e-7 produces sparse results.

**Origin**: Haveliwala (2002) personalized PageRank; Lofgren et al. (2016)
PPR product ≈ meeting probability of independent random walks.

______________________________________________________________________

### Type 7: steiner_bridges (Steiner Tree — k ≥ 3)

**Definition**: Non-seed internal nodes of the minimum Steiner tree
connecting all seeds.

**Meaning**: The minimal set of papers needed to connect all seeds. "Read
only these to understand how all topics relate."

**Question answered**: "What is the most efficient reading path through
all these topics?"

**Distinction from path_bridges**: Pairwise paths may have redundant
overlaps. Steiner tree finds the optimal shared structure.

```
path_bridges(A,B) + path_bridges(A,C) + path_bridges(B,C)
  = 3 independent paths, may share segments redundantly

steiner_bridges(A,B,C)
  = 1 optimal tree, exploits shared segments

    A ──→ X ──→ Y ──→ B       (path A-B)
    A ──→ X ──→ Z ──→ C       (path A-C, shares A→X)

    Steiner tree: A → X → Y → B
                       └→ Z → C
    Steiner nodes: X, Y, Z
```

**Computation**: KMB heuristic (Kou-Markowsky-Berman 1981):

```
1. All-pairs shortest path among k seeds → distance matrix
2. Build complete graph on seeds, edge weight = shortest path distance
3. MST of complete graph (Prim/Kruskal)
4. Replace each MST edge with actual shortest path
5. Prune non-seed leaves
6. Internal non-seed nodes = Steiner bridges
```

**Cost**: O(k² × V) for all-pairs BFS among k seeds.

**k = 2**: Identical to shortest path (Type 5). Skipped; use path_bridges.

**k ≥ 3**: Distinct from pairwise paths. Activated automatically.

**Phase 3**: Embedding-based "semantic Steiner" — find papers that minimize
semantic distance to all seeds, even without citation paths.

**Origin**: Steiner tree problem (Karp 1972, NP-hard). KMB heuristic gives
2-approximation in polynomial time.

______________________________________________________________________

## Summary Table

| Type | Scope | Cost | k=2 | k≥3 | Question |
|------|-------|------|-----|-----|----------|
| common_refs | local 1-hop | O(d) | ✅ | ✅ | Shared foundation? |
| common_citers | local 1-hop | O(d) | ✅ | ✅ | Who synthesized both? |
| coupling | local 2-hop | O(Σd) | ✅ | ✅ | Who combined both methods? |
| cocitation | local 2-hop | O(Σd) | ✅ | ✅ | What do both communities read? |
| path | global | O(V) | ✅ | pairwise | How does literature connect them? |
| ppr | global | O(k/ε) | ✅ | ✅ | What's central to both networks? |
| steiner | global | O(k²V) | skip | ✅ | Minimal reading path through all? |

## Architecture Decision: Why Not APPR-based Multi-Seed

Evaluated and rejected (Phase 2a experiment):

| Approach | Problem |
|----------|---------|
| Shared teleport APPR (field mode) | = union of single-seed expands, no new information |
| APPR intersection (common mode) | Score > 0 filter too permissive, 72% noise in results |
| APPR intersection + coupling (bridge mode) | Far pair overlap = 0, cannot find bridges |

Root cause: APPR is designed for "neighborhood exploration around a seed",
not "connection between seeds". Citation topology bridge discovery requires
algorithms that directly examine inter-seed structure (shared refs, shared
citers, paths), not diffusion-based scoring.

APPR remains the core algorithm for single-seed `expand`. Bridge uses
dedicated algorithms per type.

## CLI Design

```bash
# All types (steiner only if k ≥ 3)
quarry bridge W1234 W5678

# Specific type
quarry bridge W1234 W5678 --type common_refs
quarry bridge W1234 W5678 --type path --k-paths 5

# 3+ seeds (enables steiner)
quarry bridge W1234 W5678 W9012

# Per-type limit
quarry bridge W1234 W5678 --limit 50

# Output
quarry bridge W1234 W5678 --format json > bridges.json
quarry bridge W1234 W5678 --format table
```

## Output Schema

```json
{
  "seeds": [1234, 5678],
  "stats": {
    "per_seed_ref_count": [45, 62],
    "per_seed_citer_count": [120, 89],
    "overlap_refs": 15,
    "overlap_citers": 8,
    "shortest_path_length": 4,
    "elapsed_ms": 230
  },
  "common_refs": [
    {"work_id": 111, "aa_weight": 0.85, "cited_by_seeds": 2}
  ],
  "common_citers": [
    {"work_id": 222, "aa_weight": 0.72, "cites_seeds": 2}
  ],
  "coupling_bridges": [
    {"work_id": 333, "per_seed_overlap": [0.6, 0.4], "score": 0.24}
  ],
  "cocitation_bridges": [
    {"work_id": 444, "per_seed_overlap": [0.5, 0.3], "score": 0.15}
  ],
  "path_bridges": [
    {"work_id": 555, "hop_from_a": 2, "hop_from_b": 2, "path_count": 1}
  ],
  "ppr_bridges": [
    {"work_id": 666, "per_seed_scores": [0.003, 0.001], "score": 0.0017}
  ],
  "steiner_bridges": []
}
```

## Phase 3 Extension Points

```
Citation candidates (Phase 2)        Semantic candidates (Phase 3)
  common_refs                          embedding midpoint ANN
  common_citers                        abstract similarity search
  coupling_bridges
  cocitation_bridges
  path_bridges
  ppr_bridges
  steiner_bridges
         │                                      │
         ▼                                      ▼
    ┌──────── Candidate Pool (union) ─────────┐
    └────────────────┬────────────────────────┘
                     ▼
    ┌──────── Scoring ────────────────────────┐
    │  citation: type-specific scores          │
    │  semantic: embedding similarity (Phase 3)│
    └────────────────┬────────────────────────┘
                     ▼
              per-type ranked bridges
```

Key insight: For far pairs (overlap < 1%), citation-only bridges are sparse.
Embedding candidates fill the gap. The candidate/scoring separation means
Phase 3 adds a new candidate source without changing existing scoring.

Distance classification (free — computed as byproduct of bridge):

- overlap > 10%: citation bridges sufficient
- overlap 1-10%: citation bridges + caution
- overlap < 1%: citation bridges sparse, recommend embedding (Phase 3)

## k-Scalability

Type 1-2 use `seed_count ≥ 2` threshold (papers shared by any pair).
This works well for k=2-3 but degrades at larger k:

```
k=2:  seed_count ≥ 2 → strict (only 1 pair possible)
k=5:  seed_count ≥ 2 → C(5,2)=10 pairs → approaches union, noisy
k=10: seed_count ≥ 2 → C(10,2)=45 pairs → essentially union
```

Type suitability by k:

| Type | k=2-3 | k=5+ | Reason |
|------|-------|------|--------|
| common_refs/citers | ✅ | ❌ | intersection → 0, pairwise → noisy |
| coupling/cocitation | ✅ | △ | product → 0 if any seed has no overlap |
| path | ✅ | ❌ | C(k,2) pairwise paths explode |
| ppr (geometric mean) | ✅ | ✅ | naturally normalizes across k |
| steiner | ✅ (k≥3) | ✅ | designed for multi-terminal |

For k ≥ 4: recommend PPR and Steiner. CLI should warn on Type 1-4
with high k.

## Implementation Plan

| Step | Scope | Status |
|------|-------|--------|
| 1 | Rust `bridge()` + common_refs/common_citers | **Done** |
| 2 | coupling_bridges + cocitation_bridges | **Done** |
| 3 | path_bridges (directed BFS) | **Done** |
| 4 | ppr_bridges (per-seed APPR product) | **Done** |
| 5 | steiner_bridges (KMB heuristic) | Planned |
| 6 | Python binding + CLI (steps 1-5) | **Done** (Type 1-6; Type 7 pending) |
| 7 | Quality evaluation (abstract review, 3 seed pairs) | **Done** (Type 1-6) |

### Step 1-2 quality evaluation

KOBLAN+SICKLE, EvolvePro+MULTI-evolve (3 seed pairs):
common_refs noise=0%, common_citers noise=0%.
Coupling noise=0% across all pairs. Cocitation finds canonical literature.
Medium pair (EvolvePro+PET): Type 1-2 returned 0 results, Type 3-4 found
713/309 candidates — 2-hop traversal unlocks bridges that 1-hop misses.
`max_neighbor_degree=10,000` prune keeps traversal fast (~0.2s).

### Step 6-7 quality evaluation (CLI end-to-end, 2026-04-07)

Three seed pairs tested via `quarry bridge` CLI:

| Pair | Distance | overlap_refs | Type 1 | Type 2 | Type 3 | Type 4 | Noise |
|------|----------|-------------|--------|--------|--------|--------|-------|
| KOBLAN + SICKLE | close | 5/66 (7.6%) | 5 | 75 | 100+ | 100+ | 0% |
| EVOLVEpro + few-shot | medium | 51/76 (67%) | 51 | 1 | 100+ | 100+ | 0% |
| EVOLVEpro + KOBLAN | far | 0/76 (0%) | 0 | 1 | 100+ | 100+ | 0% |

Key findings:

- **Close pair**: All 4 types productive. Common citers are review/application
  papers that genuinely synthesize both fields.
- **Medium pair**: High ref overlap (67%) — same subfield. Type 1 dominant.
  Type 2 sparse (1) because both seeds are recent (2024).
- **Far pair**: Type 1 = 0, Type 2 = 1 (guide RNA prediction — ML+editing
  crosspoint). Type 3-4 find bridges through CRISPR hub papers. Confirms
  design prediction: far pairs need embedding candidates (Phase 3).
- **Performance**: 12-170ms across all pairs.
- **Distance classification validated**: overlap > 10% → citation bridges
  sufficient; overlap < 1% → citation bridges sparse, embedding recommended.

## References

- Kessler (1963) — Bibliographic coupling
- Small (1973) — Co-citation analysis
- Karp (1972) — Steiner tree (NP-hardness)
- Kou, Markowsky, Berman (1981) — Steiner tree 2-approximation
- Freeman (1977) — Betweenness centrality
- Haveliwala (2002) — Personalized PageRank
- Uzzi et al. (2013) — Atypical combinations and scientific impact
- Granovetter (1973) — Strength of weak ties
- Lofgren et al. (2016) — PPR product as random walk meeting probability
- Yen (1971) — K-shortest simple paths algorithm
